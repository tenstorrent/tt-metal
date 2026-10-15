#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Reject unbounded waits in active firmware images that cannot observe STOP.

An active image leaves the L2CPU only when it returns from main. There is no
interrupt and no preemption: the host's STOP is a control word that firmware
must read. A spin loop with no `x280_hart_stop_requested` in it therefore has
exactly one exit, whichever remote agent it is waiting on, and if that agent
departs first the hart is stranded until `tt-smi -r`.

The scan is deliberately narrow. It looks only for the LIM-poll idiom this tree
actually uses -- a non-counting loop whose own body issues `fence ir, ir` --
and asks whether that body can reach a STOP poll. A lint that cries wolf gets
disabled, so recall is traded for precision on purpose.

`fence ir, ir` is the anchor because it is not optional. A LIM poll can wedge
on a stale word through a `volatile` (BlackholeA0/L2CPUTile/Caches.md), so every wait on a word another agent
writes carries the load fence; a loop computing on its own registers has no
reason to. That makes the fence a marker of "waiting on someone else" rather
than a marker of "loops", which is the distinction the check needs.

Known blind spots, all deliberate:

- Only the sources named in the Makefile's ACTIVE_IMAGE_SRCS are read. Shared
  runtime headers are not, so the RETURN_ACK wait in
  `rt/x280_idle.h` is invisible here. The idle header's
  waits are correct by construction -- both are entered only after STOP has
  been latched, so nothing is left to observe. Firmware outside the active set is out of scope too: the
  resident idle image is supposed to spin until the next handoff.
- A wait with no load fence is not seen at all. It is also already broken for
  the reason above, which is why the idiom is universal in this tree.
- A `for` with all three clauses is assumed to terminate. A bound read from a
  remote word, `for (i = 0; i < *mailbox; ++i)`, is not caught.
- Reachability is path-insensitive: a body that calls a STOP-aware helper
  passes even where the call sits under a condition. It is also per translation
  unit -- a helper that polls STOP but is defined in a header is not recognised,
  and a loop calling one would be reported. `x280_hart_wait_word_or_stop` in
  rt/x280_idle.h is such a helper, but it owns its own wait, so no call site
  sits inside a fence-bearing loop and none is misreported. A header helper that
  a caller then wraps in its own wait needs that header scanned.

Usage:
    check_stop_poll.py [--makefile PATH] [--verbose] SRC...
"""

import argparse
import os
import re
import sys

# Loop control keywords never name a call, and `sizeof`/`return` in front of a
# parenthesis are not calls either.
NOT_A_CALL = frozenset(
    """if else for while do switch case return sizeof alignof typeof
    defined static_cast reinterpret_cast const_cast dynamic_cast
    __asm__ asm volatile __attribute__ __builtin_expect
    and or not""".split()
)

# The root of the STOP-aware relation. Anything that reaches this reads the
# per-hart control word and generation-matches it.
STOP_PREDICATE = "x280_hart_stop_requested"

# The one loop shape the scan treats as a LIM poll.
FENCE_RE = re.compile(r"\bfence\s+ir\s*,\s*ir\b")

LOOP_HEAD_RE = re.compile(r"\b(for|while|do)\b")
CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
FUNC_DEF_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\([^;{}]*\)\s*(?:const\s*)?$")

# A legitimate exception carries the fact that makes it legitimate: which agent
# releases the wait, and why that agent cannot leave without doing so. Both
# fields are mandatory, the justification has a floor on its length, and the
# marker binds to the loop that immediately follows it -- so it cannot be
# parked at file scope or reduced to "lint: ok".
MARKER_TAG = "STOP-POLL-EXEMPT"
MARKER_RE = re.compile(MARKER_TAG + r"\(released-by=([A-Za-z0-9_.\-]+)\)\s*:(.*)", re.DOTALL)
MARKER_MIN_JUSTIFICATION = 80


class Source:
    """One translation unit, in three index-aligned views.

    Masking substitutes spaces in place, so an offset means the same thing in
    every view and in the original text. `struct_text` drops comments and
    literal contents and is what the brace/paren walk sees. `code_text` keeps
    literals, so the fence inside an asm string survives while a comment that
    merely mentions a fence does not.
    """

    def __init__(self, path, text):
        self.path = path
        self.text = text
        self.struct_text, self.code_text, self.comments = _mask(text)
        self._line_starts = [0] + [i + 1 for i, c in enumerate(text) if c == "\n"]

    def line_of(self, offset):
        lo, hi = 0, len(self._line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._line_starts[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1


def _mask(text):
    """Blank comments and string/char literal contents, preserving offsets."""
    struct_chars = list(text)
    code_chars = list(text)
    comments = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            j = text.find("\n", i)
            j = n if j < 0 else j
            comments.append((i, j, text[i:j]))
            for k in range(i, j):
                struct_chars[k] = code_chars[k] = " "
            i = j
        elif c == "/" and nxt == "*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            comments.append((i, j, text[i:j]))
            for k in range(i, j):
                if text[k] != "\n":
                    struct_chars[k] = code_chars[k] = " "
            i = j
        elif c == '"' or (c == "'" and not (i and (text[i - 1].isalnum() or text[i - 1] == "_"))):
            # A `'` after an alphanumeric is a C++ digit separator, not a
            # literal; misreading one swallows the rest of the file.
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            for k in range(i + 1, j - 1):
                struct_chars[k] = " "
            i = j
        else:
            i += 1
    return "".join(struct_chars), "".join(code_chars), comments


def _match_delim(text, start, open_c, close_c):
    """Offset just past the delimiter matching the one at `start`, or None."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == open_c:
            depth += 1
        elif text[i] == close_c:
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def _skip_space(text, i):
    while i < len(text) and text[i].isspace():
        i += 1
    return i


class Loop:
    def __init__(self, kind, head, control, body, own_body):
        self.kind = kind  # "for" | "while" | "do"
        self.head = head  # offset of the loop keyword
        self.control = control  # control expression, masked
        self.body = body  # (start, end) of the whole body
        self.own_body = own_body  # body spans outside any nested loop

    def owns(self, span):
        return any(lo <= span[0] and span[1] <= hi for lo, hi in self.own_body)


def _find_loops(src):
    """Every loop in the file, each carrying the body spans it owns directly.

    A fence inside a nested loop belongs to that nested loop, not to its
    parent: the poll's owner is the innermost loop, and attributing it upwards
    would flag every enclosing loop as a wait too.
    """
    text = src.struct_text
    loops = []
    for m in LOOP_HEAD_RE.finditer(text):
        kind = m.group(1)
        i = _skip_space(text, m.end())
        if kind == "do":
            if i >= len(text) or text[i] != "{":
                continue
            end = _match_delim(text, i, "{", "}")
            if end is None:
                continue
            body = (i, end)
            j = _skip_space(text, end)
            control = ""
            if text.startswith("while", j):
                k = _skip_space(text, j + 5)
                close = _match_delim(text, k, "(", ")")
                if close is not None:
                    control = text[k + 1 : close - 1]
        else:
            if i >= len(text) or text[i] != "(":
                continue
            close = _match_delim(text, i, "(", ")")
            if close is None:
                continue
            control = text[i + 1 : close - 1]
            j = _skip_space(text, close)
            if j >= len(text):
                continue
            if text[j] == "{":
                end = _match_delim(text, j, "{", "}")
                if end is None:
                    continue
                body = (j, end)
            else:
                end = text.find(";", j)
                if end < 0:
                    continue
                body = (j, end + 1)
        loops.append(Loop(kind, m.start(), control, body, [body]))

    # Subtract each loop's nested loops from the spans it owns.
    for outer in loops:
        for inner in loops:
            if inner is outer:
                continue
            if not (outer.body[0] <= inner.head and inner.body[1] <= outer.body[1]):
                continue
            carved = []
            for lo, hi in outer.own_body:
                if inner.head >= hi or inner.body[1] <= lo:
                    carved.append((lo, hi))
                    continue
                if lo < inner.head:
                    carved.append((lo, inner.head))
                if inner.body[1] < hi:
                    carved.append((inner.body[1], hi))
            outer.own_body = carved
    return loops


def _find_functions(src):
    """Name -> body span for every brace block that reads as a definition.

    Deliberately loose: a struct body or a namespace picked up by mistake adds
    a node nobody calls, whereas missing a real definition would drop an edge
    from the STOP-aware relation and flag a loop that does poll.
    """
    text = src.struct_text
    funcs = {}
    i = 0
    while True:
        i = text.find("{", i)
        if i < 0:
            return funcs
        end = _match_delim(text, i, "{", "}")
        if end is None:
            return funcs
        head_start = max(text.rfind(";", 0, i), text.rfind("}", 0, i), text.rfind("{", 0, i)) + 1
        head = text[head_start:i].strip()
        m = FUNC_DEF_RE.search(head)
        if m and m.group(1) not in NOT_A_CALL:
            funcs.setdefault(m.group(1), (i, end))
        i += 1


def _calls_in(src, spans):
    names = set()
    for lo, hi in spans:
        for m in CALL_RE.finditer(src.struct_text, lo, hi):
            name = m.group(1)
            if name not in NOT_A_CALL:
                names.add(name)
    return names


def _stop_aware(src, funcs):
    """Functions in this unit that can reach the STOP predicate.

    Path-insensitive by construction: a body that calls a STOP-aware helper
    counts even where the call sits under a condition. That is the direction
    that produces false negatives rather than false positives.
    """
    aware = {STOP_PREDICATE}
    changed = True
    while changed:
        changed = False
        for name, span in funcs.items():
            if name in aware:
                continue
            if _calls_in(src, [span]) & aware:
                aware.add(name)
                changed = True
    return aware


def _is_unbounded(loop):
    """True unless the loop is one of the two shapes that cannot wait.

    A `for` with all three clauses present terminates on its own induction
    variable, so it is bounded whatever it contains. `do { ... } while (0)` is
    the statement-wrapper macro idiom and runs exactly once. Everything else --
    `for (;;)`, any `while` -- has a condition that may name a word only
    another agent writes, and is treated as a wait.
    """
    if loop.kind == "do":
        return loop.control.strip() not in ("0", "false")
    if loop.kind != "for":
        return True
    clauses = loop.control.split(";")
    if len(clauses) != 3:
        return True
    return not all(c.strip() for c in clauses)


def _marker_before(src, offset):
    """The exemption comment bound to the construct at `offset`, if any."""
    for start, end, body in src.comments:
        if end <= offset and src.text[end:offset].strip() == "":
            if MARKER_TAG in body:
                return (start, end, body)
    return None


def _parse_marker(body):
    m = MARKER_RE.search(body)
    if not m:
        return None, "malformed: expected %s(released-by=<agent>): <why>" % MARKER_TAG
    agent = m.group(1)
    justification = re.sub(r"^\s*\*+", "", m.group(2), flags=re.MULTILINE)
    justification = justification.replace("*/", " ")
    packed = " ".join(justification.split())
    if len(packed) < MARKER_MIN_JUSTIFICATION:
        return None, (
            "justification is %d characters; at least %d are required, naming why "
            "%s cannot leave without releasing the wait" % (len(packed), MARKER_MIN_JUSTIFICATION, agent)
        )
    return (agent, packed), None


def scan(path, verbose=False):
    """Returns (failures, exemptions), each a list of formatted lines."""
    with open(path, "r", encoding="utf-8") as fh:
        src = Source(path, fh.read())

    loops = _find_loops(src)
    aware = _stop_aware(src, _find_functions(src))
    failures, exemptions = [], []
    claimed_markers = set()

    for loop in loops:
        line = src.line_of(loop.head)
        where = "%s:%d" % (path, line)
        fences = [m.span() for m in FENCE_RE.finditer(src.code_text, *loop.body) if loop.owns(m.span())]
        if not fences:
            continue
        if not _is_unbounded(loop):
            if verbose:
                print("  %s: counting %s, not a wait" % (where, loop.kind))
            continue

        marker = _marker_before(src, loop.head)
        polls = sorted(_calls_in(src, [loop.body]) & aware)

        if marker:
            claimed_markers.add(marker[0])
            parsed, err = _parse_marker(marker[2])
            if not parsed:
                failures.append("%s: %s marker %s" % (where, MARKER_TAG, err))
            elif polls:
                failures.append(
                    "%s: %s marker is redundant; this loop already polls STOP via %s"
                    % (where, MARKER_TAG, ", ".join(polls))
                )
            else:
                exemptions.append("%s: exempt, released by %s -- %s" % (where, parsed[0], parsed[1]))
            continue

        if polls:
            if verbose:
                print("  %s: polls STOP via %s" % (where, ", ".join(polls)))
            continue

        failures.append(
            "%s: unbounded `%s` wait issues `fence ir, ir` but never reaches %s; "
            "a STOP delivered here cannot end it" % (where, loop.kind, STOP_PREDICATE)
        )

    for start, end, body in src.comments:
        if MARKER_TAG in body and start not in claimed_markers:
            failures.append(
                "%s:%d: stale %s marker -- it precedes no unbounded `fence ir, ir` wait"
                % (path, src.line_of(start), MARKER_TAG)
            )
    return failures, exemptions


HARNESS_OBJS = ("entry_active", "x280_active_start")


def _make_vars(text):
    joined = re.sub(r"\\\n", " ", text)
    return dict(re.findall(r"^(\w+)\s*:?=\s*(.*)$", joined, re.MULTILINE))


def _expand(value, variables, depth=8):
    for _ in range(depth):
        expanded = re.sub(r"\$\((\w+)\)", lambda m: variables.get(m.group(1), m.group(0)), value)
        if expanded == value:
            break
        value = expanded
    return value


def check_scope(makefile, scanned):
    """Every image linking the active harness must be in the scanned set.

    This is the hole the defect went through: three images coordinate helper
    harts, two were converted by hand and the third was missed. An image added
    to the Makefile without being added to ACTIVE_IMAGE_SRCS would be exempt by
    omission, which is the failure this whole target exists to prevent.
    """
    with open(makefile, "r", encoding="utf-8") as fh:
        text = fh.read()
    variables = _make_vars(text)
    root = os.path.dirname(os.path.abspath(makefile))
    scanned_stems = {os.path.splitext(os.path.basename(p))[0] for p in scanned}
    linked_stems = set()
    problems = []
    for m in re.finditer(r"^(\S+\.elf):([^\n]*)$", text, re.MULTILINE):
        target = _expand(m.group(1), variables)
        prereqs = _expand(m.group(2), variables)
        if not all("%s.o" % obj in prereqs for obj in HARNESS_OBJS):
            continue
        for obj in re.findall(r"([\w.\-]+)\.o\b", prereqs):
            obj = os.path.basename(obj)
            if obj in HARNESS_OBJS:
                continue
            linked_stems.add(obj)
            if obj in scanned_stems:
                continue
            for ext in (".c", ".cpp"):
                if os.path.exists(os.path.join(root, "src", obj + ext)):
                    problems.append(
                        "%s: %s links the active harness but src/%s%s is not in "
                        "ACTIVE_IMAGE_SRCS" % (makefile, os.path.basename(target), obj, ext)
                    )
                    break
    for stem in sorted(scanned_stems - linked_stems):
        problems.append(
            "%s: ACTIVE_IMAGE_SRCS lists %s, which no image links through the " "active harness" % (makefile, stem)
        )
    return problems


def check_build_coverage(makefile):
    with open(makefile, "r", encoding="utf-8") as fh:
        text = fh.read()
    joined = re.sub(r"\\\n", " ", text)
    variables = _make_vars(text)

    rules = {}
    for m in re.finditer(r"^([^\s:#][^:\n]*):(?!=)([^\n]*)$", joined, re.MULTILINE):
        prereqs = _expand(m.group(2), variables).split()
        for target in _expand(m.group(1), variables).split():
            rules.setdefault(target, []).extend(prereqs)

    # legacy-tests builds the active images outside the active-image contract.
    reachable, stack = set(), rules.get("all-tests", []) + rules.get("legacy-tests", [])
    while stack:
        node = stack.pop()
        if node in reachable:
            continue
        reachable.add(node)
        stack.extend(rules.get(node, []))

    problems = []
    for m in re.finditer(r"^(\S+\.elf):([^\n]*)$", joined, re.MULTILINE):
        prereqs = _expand(m.group(2), variables)
        if not all("%s.o" % obj in prereqs for obj in HARNESS_OBJS):
            continue
        binary = _expand(m.group(1), variables)[: -len(".elf")] + ".bin"
        if binary not in reachable:
            problems.append(
                "%s: %s links the active harness but neither all-tests nor "
                "legacy-tests builds it" % (makefile, os.path.basename(binary))
            )
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--makefile", help="cross-check the scanned set against it")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    failures, exemptions = [], []
    for path in args.sources:
        if not os.path.exists(path):
            failures.append("%s: no such source" % path)
            continue
        if args.verbose:
            print("scanning %s" % path)
        f, e = scan(path, args.verbose)
        failures += f
        exemptions += e

    if args.makefile:
        failures += check_scope(args.makefile, args.sources)
        failures += check_build_coverage(args.makefile)

    # Exemptions print on every run, pass or fail: an exemption nobody sees is
    # an exemption nobody reviews.
    for line in exemptions:
        print("EXEMPT: %s" % line)

    if failures:
        for line in failures:
            sys.stderr.write("FAIL: %s\n" % line)
        sys.stderr.write("FAIL: %d STOP-poll problem(s) in the active image sources\n" % len(failures))
        return 1

    print(
        "PASS: %d active image source(s) scanned, every unbounded `fence ir, ir` "
        "wait polls STOP (%d exemption(s))" % (len(args.sources), len(exemptions))
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
