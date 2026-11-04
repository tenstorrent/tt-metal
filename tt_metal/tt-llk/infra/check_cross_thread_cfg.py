#!/usr/bin/env python3
"""Flag a Tensix config write that can corrupt a config field another thread owns.

Most Tensix backend config lives in ONE copy shared by all three threads, packed several fields to
a 32-bit word. Two consequences:

  CLOBBER    A whole-word write (WRCFG_32b, `cfg[w] = ...`, the MMIO `cfg_rmw`/`cfg_write` helpers)
             writes all 32 bits, so it destroys every other field in that word -- including fields
             belonging to another thread.
  SAME-FIELD Two threads doing a masked read-modify-write of the SAME bits still race on the value.
             (DIFFERENT bits are safe without a mutex: the config RMW is per-byte atomic.)

The per-thread register file (written by SETC16) is exempt: each thread has its own copy. Note the
two spaces numerically ALIAS -- THREAD_CFGREG_BASE_ADDR32 and ALU_CFGREG_BASE_ADDR32 are both 0 --
so classification is by the WRITING INSTRUCTION, never by the address.

Field ownership is derived from the tree rather than hand-maintained: the checker first records
which thread writes which field (from the enclosing llk_unpack_* / llk_math_* / llk_pack_* file or
function, and the math thread for SFPU kernels), then flags writes that cross those boundaries.

This is the blocking pre-commit gate: plain Python, no build and no Clang, so it runs on every
commit. The llk-audit `cfg-word-overlap` check models the same hazard more deeply (REG2FLOP,
CFGSHIFTMASK, intra-thread clobbers, Quasar) but needs the Clang extractor, so it stays an
advisory audit. Keep the two agreeing on what counts as a write when either changes.
"""
import argparse
import os
import re
import sys
from collections import defaultdict

# --- config-register definitions ----------------------------------------------------------------
DEF_ADDR = re.compile(r"^#define\s+(\w+?)_ADDR32\s+(\d+)")
DEF_SHAMT = re.compile(r"^#define\s+(\w+?)_SHAMT\s+(\d+)")
DEF_MASK = re.compile(r"^#define\s+(\w+?)_MASK\s+(0x[0-9a-fA-F]+|\d+)")


def load_defs(path):
    """field -> (word, shamt, mask). Only fields with all three are usable."""
    addr, shamt, mask = {}, {}, {}
    for line in open(path, errors="ignore"):
        for rx, d in ((DEF_ADDR, addr), (DEF_SHAMT, shamt), (DEF_MASK, mask)):
            m = rx.match(line)
            if m:
                d[m.group(1)] = int(m.group(2), 0)
    return {k: (addr[k], shamt[k], mask[k]) for k in addr if k in shamt and k in mask}


def thread_section_fields(path):
    """Fields in the `// Registers for THREAD` section, which index ThreadConfig, not Config.

    Their word numbers alias the shared Config words (both bases are 0), so they must not be
    read as other fields of a shared word.
    """
    names, in_thread = set(), False
    for line in open(path, errors="ignore"):
        if line.startswith("// Registers for "):
            in_thread = line.strip() == "// Registers for THREAD"
        elif in_thread:
            m = DEF_ADDR.match(line)
            if m:
                names.add(m.group(1))
    return names


# --- write mechanisms ---------------------------------------------------------------------------
# Whole-word: every bit of the word is overwritten.
# Addressing is frequently BASE + OFFSET. Three pitfalls, all of which silently corrupt a result:
#   cfg[REG + i]        a variable offset -- the target word is not statically known
#   WRCFG(.., REG + 1)  a literal offset the address pattern must not drop
#   WRCFG_128b          writes FOUR consecutive words, not one
# Offsets are parsed as expressions; a non-literal offset is reported UNRESOLVED, never assumed.
# The mode argument is the wr128b bit: spelled p_cfg::WRCFG_32b/WRCFG_128b at most sites and as
# the bare literal at the rest. TT_OP_WRCFG builds a MOP word, which executes a WRCFG when the MOP
# runs, so it is a write like any other -- the macro definitions in ckernel_ops.h pass the mode
# through as `wr128b` and so do not match.
WRCFG_ANY = re.compile(
    r"\b(?:TTI?|TT_OP)_WRCFG\s*\(\s*[^,]+,\s*(?:p_cfg::)?(WRCFG_32b|WRCFG_128b|0|1)\s*,\s*([^)]+)\)"
)
WRCFG_WIDE = ("WRCFG_128b", "1")
CFG_INDEX = re.compile(r"\bcfg\s*\[([^\]]+)\]\s*=")
# The WH/BH `cfg_rmw` / `cfg_rmw_gpr` / `cfg_write` helpers are RISC MMIO: a load, a modify and a
# full 32-bit store (`cfg_write` is the store alone). A write by another thread to the same word
# between the load and the store is lost, so they clobber like a whole-word write -- disjoint bits
# are NOT safe through them. (On Quasar `cfg_rmw` is a Tensix RMWCIB; that arch is refused below.)
MMIO_WRITE = re.compile(r"\bcfg_(?:rmw_gpr|rmw|write)\s*\(\s*([^,()]+)")
ADDR_EXPR = re.compile(r"^\s*([A-Za-z_0-9]+)_ADDR32\s*(?:\+\s*(.+?))?\s*$")
FULL_WORD_MASK = 0xFFFFFFFF


def resolve_addr(expr, defs):
    """(word, ok). ok=False when the offset is not a compile-time constant."""
    m = ADDR_EXPR.match(expr)
    if not m or m.group(1) not in defs:
        return None, False
    base = defs[m.group(1)][0]
    off = m.group(2)
    if off is None:
        return base, True
    off = off.strip()
    if re.fullmatch(r"\d+", off):
        return base + int(off), True
    return base, False  # variable offset: target not statically known


# Masked read-modify-write: only the masked bits change. The template arguments are parsed
# whole, since they may span lines: `<FIELD_RMW>`, or `<ADDR32 [+ k], SHAMT, MASK>` where MASK
# is a `FIELD_MASK`, a literal, a local mask constant, or an OR of those.
MASKED_START = re.compile(r"\bcfg_reg_rmw_tensix\s*<")


def _template_args(code, lt):
    """The text between the `<` at offset `lt` and its matching `>`, which may span lines."""
    depth, j = 1, lt + 1
    while j < len(code) and depth:
        depth += {"<": 1, ">": -1}.get(code[j], 0)
        j += 1
    return None if depth else code[lt + 1 : j - 1]


def _split_args(args):
    """Split on top-level commas only."""
    parts, depth, cur = [], 0, []
    for ch in args:
        depth += 1 if ch in "<([" else -1 if ch in ">)]" else 0
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur).strip())
    return parts


# A combined write names one field for the address but masks several: the reconfig helpers build
# `config_mask = A_MASK | B_MASK`. Reading only the named field's mask misses the other fields in
# the same write -- which is how INT8_math_enabled was invisible in the first version.
MASK_CONST = re.compile(r"\b(\w*mask\w*)\s*=\s*([^;]+);", re.I)
# How many layers of local mask aliases (`a_mask = b_mask | c_mask`) are followed. `seen` already
# guarantees termination; this bounds work on a pathological chain, which then reads as unresolved.
MAX_MASK_ALIAS_DEPTH = 6


def resolve_mask(name, text, defs, depth=0, seen=None):
    """Resolve a mask identifier to a bitmask, following local constants transitively.

    Masks are built in layers: `alu_mask = alu_format_mask | alu_stoch_rnd_mask`, and only the
    leaves are cfg_defines `*_MASK` names. A non-recursive resolver returns nothing here, and
    falling back to the NAMED field's mask then attributes a write to bits it never touches --
    which is how a disjoint-bit (and therefore safe) unpack write looked like a format-bit race.

    Returns None when the mask cannot be resolved, so the caller can decline to judge rather
    than guess.
    """
    if depth > MAX_MASK_ALIAS_DEPTH:
        return None
    seen = seen or set()
    if name in seen:
        return None
    seen.add(name)
    if name in defs:
        return defs[name][2]
    # `text` runs from the top of the file to the write being resolved, and a name like
    # `config_mask` is redeclared in sibling functions with different bits. The definition that
    # governs is the nearest one above the write, which is the LAST match, not the first.
    hits = [m for m in MASK_CONST.finditer(text) if m.group(1) == name]
    for m in hits[-1:]:
        bits, ok = 0, False
        for ref in re.findall(r"\b(\w+)\b", m.group(2)):
            # defs is keyed on the FIELD name; the source writes `<FIELD>_MASK`. Strip the
            # suffix before lookup or every leaf reference fails and the mask never resolves.
            leaf = ref[:-5] if ref.endswith("_MASK") else ref
            if leaf in defs:
                bits |= defs[leaf][2]
                ok = True
            elif ref.lower().endswith("mask"):
                sub = resolve_mask(ref, text, defs, depth + 1, seen)
                if sub is not None:
                    bits |= sub
                    ok = True
        return bits if ok else None
    return None


TREE_MASK_CONST = re.compile(
    r"\bconstexpr\b[^;=]*?\b(\w*mask\w*)\s*=\s*(0[xX][0-9a-fA-F]+|\d+)\s*;", re.I
)


def tree_mask_constants(paths):
    """Literal `constexpr` mask constants from the whole tree, for masks defined in another header
    (e.g. TILE_DESC_UPPER_HALFWORD_MASK in cunpack_common.h). A name given two different values is
    ambiguous and left out, so it stays unresolved rather than guessed."""
    seen = defaultdict(set)
    for p in paths:
        for m in TREE_MASK_CONST.finditer(open(p, errors="ignore").read()):
            seen[m.group(1)].add(int(m.group(2), 0))
    return {k: next(iter(v)) for k, v in seen.items() if len(v) == 1}


def resolve_mask_expr(expr, text, defs, tree_masks=None):
    """A template MASK argument -> bitmask, or None if any part of it does not resolve.

    Every OR'd term must resolve: dropping one would under-report the bits the write touches.
    A constant defined in this file above the write wins over a tree-wide one.
    """
    bits = 0
    for term in expr.split("|"):
        term = term.strip().strip("()").strip()
        if re.fullmatch(r"0[xX][0-9a-fA-F]+|\d+", term):
            bits |= int(term, 0)
            continue
        # a cfg_defines `FIELD_MASK` is a macro, so it wins over any same-named local
        leaf = term[:-5] if term.endswith("_MASK") else None
        if leaf in defs:
            bits |= defs[leaf][2]
            continue
        sub = resolve_mask(term, text, defs)
        if sub is None:
            sub = (tree_masks or {}).get(term)
        if sub is None:
            return None
        bits |= sub
    return bits


def masked_target(args, text, defs, tree_masks=None):
    """(field, word, bits, kind) for a cfg_reg_rmw_tensix<...> argument list, or None."""
    parts = _split_args(args)
    if len(parts) == 1:
        field = parts[0][:-4] if parts[0].endswith("_RMW") else None
        if field not in defs:
            return None
        word, _sh, mask = defs[field]
        return field, word, mask, "masked"
    if len(parts) != 3:
        return None
    word, ok = resolve_addr(parts[0], defs)
    if word is None:
        return None
    field = ADDR_EXPR.match(parts[0]).group(1)
    if not ok:
        return field, word, FULL_WORD_MASK, "masked-addr-unresolved"
    bits = resolve_mask_expr(parts[2], text, defs, tree_masks)
    if bits is None:
        # Never fall back to the named field's mask: that attributes the write to bits it may
        # never touch. Report it as unresolved instead.
        return field, word, 0, "mask-unresolved"
    return field, word, bits, "masked"


# Per-thread register file -- each thread has its own copy, so it cannot cross threads.
PER_THREAD = re.compile(r"\bTTI?_SETC16\b")
MUTEX_ACQ = re.compile(r"t6_mutex_acquire\s*\(\s*mutex::REG_RMW")
MUTEX_REL = re.compile(r"t6_mutex_release\s*\(\s*mutex::REG_RMW")

THREADS = ("UNPACK", "MATH", "PACK")


# --- consumer map ------------------------------------------------------------------------------
# Which THREAD's instructions consume a field is not derivable from the LLK source; it is a
# hardware fact, so it is supplied as a vendored table rather than inferred here.
CONSUMERS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "cfg_consumers.yaml"
)


def load_consumers(arch, path=None):
    """field -> set(consuming threads), from the vendored table.

    The table is vendored, so the check needs no external input at run time; it is maintained by
    hand (see the header of cfg_consumers.yaml).
    A field ABSENT from it has no reader recorded for it: that is UNKNOWN, not 'nobody consumes
    it'.
    """
    try:
        import yaml
    except ImportError:
        return None
    try:
        doc = yaml.safe_load(open(path or CONSUMERS_FILE))
    except Exception:
        return None
    table = (doc or {}).get(arch) or {}
    return {k: set(v) for k, v in table.items()} or None


def thread_of(path, text_before):
    """Which Tensix thread executes this line, from the file and enclosing function."""
    base = os.path.basename(path)
    for name, keys in zip(
        THREADS,
        (
            ("llk_unpack", "cunpack_common"),
            ("llk_math", "cmath_common"),
            ("llk_pack", "cpack_common"),
        ),
    ):
        if any(k in base for k in keys):
            return name
    m = None
    for fn in re.finditer(r"_llk_(unpack|math|pack)\w*_\s*\(", text_before):
        m = fn
    if m:
        return m.group(1).upper()
    return None


def inferred_thread(path):
    """The thread an SFPU kernel USUALLY runs on, for a write thread_of() cannot name.

    SFPU instructions are normally issued by the math thread, and the kernels live in ckernel_sfpu_*
    headers under sfpu/ -- names that carry no llk_math. But some SFPU code is issued from another
    thread, so this is a guess: a finding that depends on it is advisory and never fails a commit.
    """
    base = os.path.basename(path)
    if base.startswith("ckernel_sfpu") or f"{os.sep}sfpu{os.sep}" in path:
        return "MATH"
    return None


def writes_in(path, defs, tree_masks=None):
    """Yield (line_no, thread, kind, field, word, bitmask, guarded, inferred)."""
    lines = open(path, errors="ignore").read().split("\n")
    # comment-stripped text with offsets per line, so a template argument list can be read across lines
    code_lines = [raw.split("//")[0] for raw in lines]
    code = "\n".join(code_lines)
    line_start, pos = [], 0
    for cl in code_lines:
        line_start.append(pos)
        pos += len(cl) + 1
    joined = ""
    # Track the mutex as acquire/release STATE, not a fixed lookback window: a protected write can
    # sit far below its acquire (cunpack_common.h has 26 lines between them), and a window-based
    # check reports those as unguarded.
    held = False
    for i, raw in enumerate(lines):
        line = raw.split("//")[0]
        joined += raw + "\n"
        if not line.strip():
            continue
        if PER_THREAD.search(line):
            continue  # per-thread file; cannot cross threads
        # A write whose thread cannot be determined is REPORTED, not dropped: the file/function
        # naming conventions thread_of() keys on do not cover every spelling (a ThreadId template
        # parameter dispatches by value, shared headers carry no thread in the name). Silently
        # skipping those is the one outcome the rest of this checker is written to avoid. It is a
        # note rather than a finding -- an unclassified write is a gap in this tool's coverage, not
        # evidence of a hazard, and must not fail anyone's commit.
        th = thread_of(path, joined)
        inferred = th is None and inferred_thread(path) is not None
        th = th or inferred_thread(path)
        if MUTEX_ACQ.search(line):
            held = True
        if MUTEX_REL.search(line):
            held = False
        guarded = held

        def whole_word(expr, span=1):
            """Yield a whole-word write over `span` words; span 4 is a 128b WRCFG."""
            word, ok = resolve_addr(expr, defs)
            if word is None:
                return
            if span > 1:
                word &= ~(
                    span - 1
                )  # a 128b WRCFG writes the ALIGNED group of four words
            for k in range(span):
                yield i + 1, th, (
                    "whole-word" if ok else "addr-unresolved"
                ), expr.strip(), word + k, FULL_WORD_MASK, guarded, inferred

        hit = []
        m = WRCFG_ANY.search(line)
        if m:
            hit = list(whole_word(m.group(2), 4 if m.group(1) in WRCFG_WIDE else 1))
        if not hit:
            m = CFG_INDEX.search(line)
            if m:
                hit = list(whole_word(m.group(1)))
        if not hit:
            m = MMIO_WRITE.search(line)
            if m:
                arg = m.group(1).strip()
                # `cfg_rmw(FIELD_RMW, v)` names the field; the macro expands to its ADDR32 first
                expr = arg[:-4] + "_ADDR32" if arg.endswith("_RMW") else arg
                hit = list(whole_word(expr))
        yield from hit
        if not hit:
            for m in MASKED_START.finditer(line):
                args = _template_args(code, line_start[i] + m.end() - 1)
                target = (
                    masked_target(args, joined, defs, tree_masks)
                    if args is not None
                    else None
                )
                if target:
                    # A MASK is already positioned within the word (e.g. SrcA: SHAMT 17,
                    # MASK 0x1e0000). Shifting by SHAMT again would double-shift and compare
                    # the wrong bits, producing both false positives and false negatives.
                    field, word, bits, kind = target
                    yield i + 1, th, kind, field, word, bits, guarded, inferred


def collect(paths, defs, tree_masks=None):
    sites = []
    for p in paths:
        sites += [(p,) + w for w in writes_in(p, defs, tree_masks)]
    return sites


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("files", nargs="*")
    ap.add_argument("--defs", required=True, help="cfg_defines.h for the architecture")
    ap.add_argument(
        "--tree",
        required=True,
        help="LLK arch tree; ownership is derived from all of it",
    )
    ap.add_argument(
        "--arch", default="wormhole_b0", help="key into the vendored consumer table"
    )
    ap.add_argument("--consumers", help="override the vendored consumer table")
    ap.add_argument("--baseline")
    ap.add_argument("--write-baseline", action="store_true")
    args = ap.parse_args()

    # Quasar writes config with cfg_rmw / RMWCIB rather than cfg_reg_rmw_tensix, and cfg_rmw is a
    # DIFFERENT mechanism there (RMWCIB) than on Wormhole/Blackhole (MMIO). A parser tuned to the
    # 1xx spelling sees almost nothing on Quasar and reports a comfortable, meaningless zero.
    SUPPORTED = ("wormhole_b0", "blackhole")
    if args.arch not in SUPPORTED:
        print(
            f"REFUSING: arch '{args.arch}' is not supported (supported: {', '.join(SUPPORTED)})."
        )
        print(
            "  Quasar uses cfg_rmw/RMWCIB for config writes; this checker does not model them,"
        )
        print("  so a clean result here would mean 'unmeasured', not 'no hazards'.")
        return 2

    defs = load_defs(args.defs)
    per_thread = thread_section_fields(args.defs)
    consumers = load_consumers(args.arch, args.consumers)
    all_h = [
        os.path.join(d, f)
        for d, _, fs in os.walk(args.tree)
        for f in fs
        if f.endswith((".h", ".hpp"))  # the same suffixes the hooks trigger on
    ]
    tree_masks = tree_mask_constants(all_h)
    every = collect(all_h, defs, tree_masks)

    # Derive ownership from the tree: which threads write which bits of which word. Each owner bit
    # remembers where it came from, so a finding can say which write it conflicts with, and whether
    # that write's thread is certain or only inferred.
    owners = defaultdict(lambda: defaultdict(int))  # word -> thread -> bitmask
    owner_sites = defaultdict(list)  # word -> [(thread, bits, path, inferred)]
    for path, _, th, kind, _, word, bits, _, inferred in every:
        # th is None for an unclassified write; it must not become an owner, or every real write
        # to the word would read it as another thread and report against it.
        if (
            kind == "masked" and th is not None
        ):  # whole-word/unresolved writes tell us nothing about ownership
            owners[word][th] |= bits
            owner_sites[word].append((th, bits, path, inferred))

    # Vacuity guard: if the tree has headers but essentially no masked writes were recognised,
    # the parser is blind to this tree's write API and a zero result proves nothing.
    n_masked = sum(1 for e in every if e[3] == "masked")
    if all_h and n_masked == 0:
        print(
            f"REFUSING: parsed {len(all_h)} header(s) but recognised no masked config writes."
        )
        print(
            "  The checker does not understand this tree's config-write API; a zero result would"
        )
        print(
            "  be vacuous. Extend the write-mechanism patterns before trusting any output."
        )
        return 2

    # Findings are computed over the whole tree, then scoped to what the commit is responsible for.
    # Each carries `sources` (the conflicting owner writes) and `blocking`: a finding fails a commit
    # only when both sides are certain -- the writer's thread and at least one conflicting owner's
    # thread are named, not inferred -- and the write itself was resolved. Everything else,
    # including every UNRESOLVED write, is printed as advisory, so a guess never blocks a PR.
    findings, unclassified = [], []
    for path, ln, th, kind, field, word, bits, guarded, inferred in every:
        if th is None:
            unclassified.append((path, ln, field, word))
            continue
        others = {t: b for t, b in owners[word].items() if t != th and b}
        if not others:
            continue

        def conflict(mask=None):
            srcs = [
                (p, inf)
                for t, b, p, inf in owner_sites[word]
                if t != th and b and (mask is None or b & mask)
            ]
            return {p for p, _ in srcs}, any(not inf for _, inf in srcs)

        if kind in ("addr-unresolved", "mask-unresolved", "masked-addr-unresolved"):
            srcs, _certain = conflict()
            why = (
                "mask could not be resolved; review by hand"
                if kind == "mask-unresolved"
                else "offset is not a compile-time constant; review by hand"
            )
            # The checker cannot tell whether this write conflicts, so it never blocks a commit:
            # it is surfaced for review, and blocking stays reserved for what is certain.
            findings.append(
                (
                    path,
                    ln,
                    "UNRESOLVED",
                    field,
                    word,
                    why,
                    srcs,
                    False,
                )
            )
        elif kind == "whole-word":
            # Name who CONSUMES the destroyed bits, not merely who else writes them: a whole-word
            # write destroys every field in the word, and the damage lands on the thread whose
            # instructions read those fields.
            victims = ", ".join(
                f"{t} (bits 0x{b:08x})" for t, b in sorted(others.items())
            )
            harmed = set()
            if consumers is not None:
                for fname, (fw, _fs, _fm) in defs.items():
                    if fw == word and fname not in per_thread:
                        harmed |= {t for t in consumers.get(fname, set()) if t != th}
            extra = (
                f"; other fields in this word are consumed by {', '.join(sorted(harmed))}"
                if harmed
                else ""
            )
            srcs, certain = conflict()
            findings.append(
                (
                    path,
                    ln,
                    "CLOBBER",
                    field,
                    word,
                    f"whole-word write from {th} destroys {victims}{extra}",
                    srcs,
                    certain and not inferred,
                )
            )
        else:
            overlap = {t: b & bits for t, b in others.items() if b & bits}
            if not overlap:
                continue
            v = ", ".join(f"{t} (0x{b:08x})" for t, b in sorted(overlap.items()))
            srcs, certain = conflict(bits)
            if not guarded:
                findings.append(
                    (
                        path,
                        ln,
                        "SAME-FIELD",
                        field,
                        word,
                        f"{th} RMWs bits also written by {v}, no mutex::REG_RMW",
                        srcs,
                        certain and not inferred,
                    )
                )
                continue
            # Guarded: mutex::REG_RMW stops a multi-byte RMW from tearing, but two threads writing
            # the same field still race on its VALUE. That is sound only when the writer is a
            # thread that consumes the field (it sets what it is about to use). A writer that does
            # not consume it is reported: blocking when the consumer table records every field
            # the shared bits cover, advisory when any of them is unrecorded.
            shared = 0
            for b in overlap.values():
                shared |= b
            fields = [
                fn
                for fn, (fw, _fs, fm) in defs.items()
                if fw == word and fm & shared and fn not in per_thread
            ]
            table = consumers or {}
            foreign = [fn for fn in fields if fn in table and th not in table[fn]]
            unknown = not fields or any(fn not in table for fn in fields)
            if foreign or unknown:
                read_by = "; ".join(
                    f"{fn} read by {', '.join(sorted(table[fn]))}" for fn in foreign
                )
                how = (
                    f"does not consume them ({read_by})"
                    if foreign
                    else "may not consume them (no reader recorded for "
                    + (", ".join(fields) or "these bits")
                    + ")"
                )
                findings.append(
                    (
                        path,
                        ln,
                        "SAME-FIELD",
                        field,
                        word,
                        f"{th} RMWs bits also written by {v} under mutex::REG_RMW, but {th} {how}",
                        srcs,
                        bool(foreign) and certain and not inferred,
                    )
                )

    # Scope. Run with no files (by hand, or CI over the tree) and everything is reported. From the
    # hook, a commit is answerable for findings AT the files it touches and for findings its
    # writes CAUSE elsewhere (it added the conflicting owner) -- never for unrelated debt in the
    # tree, which would block every LLK commit until someone else fixed it. A commit that touches
    # a verdict input instead of a header (the checker, the defs, the consumer table, the
    # baseline) moves the answer everywhere, so it is checked against the whole tree.
    tree_root = os.path.abspath(args.tree) + os.sep
    touched = {os.path.abspath(f) for f in args.files}
    whole_tree = not touched or any(not t.startswith(tree_root) for t in touched)

    def in_scope(path, srcs):
        return (
            whole_tree
            or os.path.abspath(path) in touched
            or any(os.path.abspath(s) in touched for s in srcs)
        )

    # A blocking finding is also owed by the commit that CAUSED it; advisory output is not owed
    # by anyone, so it is shown only at its own file. pre-commit splits a large file list into
    # batches, and output scoped more widely would repeat once per batch.
    findings = [f for f in findings if in_scope(f[0], f[6] if f[7] else ())]
    unclassified = [u for u in unclassified if in_scope(u[0], ())]

    accepted = set()
    if args.baseline and not args.write_baseline:
        try:
            accepted = {
                l.split("#")[0].strip()
                for l in open(args.baseline)
                if l.strip() and not l.startswith("#")
            }
        except FileNotFoundError:
            pass

    keys = [f"{os.path.relpath(f[0])}:{f[2]}:{f[3]}" for f in findings]
    if args.write_baseline:
        with open(args.baseline, "w") as fh:
            fh.write("# Cross-thread config-register writes accepted for now.\n")
            for k in sorted(set(keys)):
                fh.write(k + "\n")
        print(f"baseline written: {len(set(keys))} site(s)")
        return 0

    new = [f for f, k in zip(findings, keys) if k not in accepted]
    blocking = [f for f in new if f[7]]
    advisory = [f for f in new if not f[7]]

    def report(items):
        for path, ln, kind, field, word, why, _srcs, _blk in items:
            print(f"{path}:{ln}: [{kind}] config word {word} shared across threads")
            print(f"  field: {field}")
            print(f"  {why}")
            if consumers is not None:
                cs = consumers.get(field.replace("_ADDR32", ""))
                print(
                    f"  consumed by: {', '.join(sorted(cs))}"
                    if cs
                    else "  consumed by: NOT RECORDED -- treat as unknown, not safe"
                )
            print(
                "  fix: masked cfg_reg_rmw_tensix for your own bits. A FIELD two threads write needs one\n"
                "       owning thread, or its writers ordered (e.g. by a semaphore): mutex::REG_RMW only\n"
                "       stops a multi-byte RMW from tearing, and the last writer still wins.\n"
            )

    report(blocking)
    if advisory:
        print(
            f"advisory: {len(advisory)} possible cross-thread config write(s) that do NOT fail the commit --\n"
            "  a thread here is inferred (an SFPU kernel's location), or the write's target is only\n"
            "  partly known. Review them; baseline or fix any that are real.\n"
        )
        report(advisory)
    if unclassified:
        print(
            f"note: {len(unclassified)} config write(s) whose thread could not be determined, "
            "so they were not checked:"
        )
        for path, ln, field, word in unclassified:
            print(f"  {os.path.relpath(path)}:{ln}: {field} (config word {word})")
        print(
            "  These are a coverage gap in this checker, not hazards. thread_of() keys on the\n"
            "  llk_unpack/llk_math/llk_pack file and function naming; a write dispatched by a\n"
            "  ThreadId template parameter, or sitting in a shared header, carries neither.\n"
            "  Review by hand, or teach thread_of() the spelling. Never fails the commit.\n"
        )

    if blocking:
        print(f"{len(blocking)} new cross-thread config write(s).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
