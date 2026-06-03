#!/usr/bin/env python3
"""Comment-out / restore the bulk NoC data-movement primitives in the ring-joint SDPA dataflow
kernels (NO macros) so we can measure the compute ceiling.

Drop-set mirrors noc_dm_gate.hpp exactly:
  noc_async_read(  noc_async_read_page(  noc_async_read_one_packet_set_state(
  noc_async_write( noc_async_write_page( noc_async_write_multicast(
  noc_async_read_one_packet_with_state<...>(   (templated; handled specially)
KEPT: every *_barrier / *_flushed* / *_set_trid / noc_semaphore_* / cb_* primitive.

Usage:
  python dm_toggle.py off    # back up + comment out the DM calls
  python dm_toggle.py on     # restore from backups
  python dm_toggle.py status # report
"""
import sys
import os

ROOT = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow"
FILES = [
    f"{ROOT}/ring_joint_reader.cpp",
    f"{ROOT}/ring_joint_writer.cpp",
    f"{ROOT}/dataflow_common.hpp",
    f"{ROOT}/chain_link.hpp",
]

# Bare-call targets: identifier must be immediately followed by '(' (so noc_async_read does NOT
# match noc_async_read_barrier / _page / _one_packet_set_state, etc.).
BARE = [
    "noc_async_read",
    "noc_async_read_page",
    "noc_async_read_one_packet_set_state",
    "noc_async_write",
    "noc_async_write_page",
    "noc_async_write_multicast",
]
TEMPLATED = "noc_async_read_one_packet_with_state"  # followed by '<...>('

MARK = "//DMOFF "  # sentinel prefix so we can detect/idempotency-check


def _ident_char(c):
    return c.isalnum() or c == "_"


def find_matches(text):
    """Return list of (stmt_start_idx, stmt_end_idx) byte ranges to comment out."""
    matches = []
    n = len(text)
    # Build ordered list of (name, is_templated)
    targets = [(t, False) for t in BARE] + [(TEMPLATED, True)]
    for name, templated in targets:
        start = 0
        while True:
            idx = text.find(name, start)
            if idx == -1:
                break
            start = idx + len(name)
            # word boundary before
            if idx > 0 and _ident_char(text[idx - 1]):
                continue
            after = text[idx + len(name)]
            if templated:
                if after != "<":
                    continue
            else:
                if after != "(":
                    continue
            # skip if inside a line comment (// ... ) on this line before idx
            line_begin = text.rfind("\n", 0, idx) + 1
            if "//" in text[line_begin:idx]:
                continue
            # find first '(' at/after idx (covers the '<...>(' of templated)
            paren = text.find("(", idx)
            if paren == -1:
                continue
            depth = 0
            j = paren
            while j < n:
                c = text[j]
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if j >= n:
                continue
            # find terminating ';'
            semi = text.find(";", j)
            if semi == -1:
                continue
            matches.append((idx, semi + 1))
    # dedup / sort
    matches = sorted(set(matches))
    return matches


def comment_out(path):
    with open(path) as f:
        text = f.read()
    if MARK in text:
        print(f"  [skip] {path} already DM-off")
        return 0
    matches = find_matches(text)
    if not matches:
        print(f"  [warn] {path}: no DM call sites found")
        return 0
    # Determine line spans, then comment each affected whole line.
    lines = text.split("\n")
    # offset->line index
    line_starts = [0]
    for ln in lines:
        line_starts.append(line_starts[-1] + len(ln) + 1)

    def line_of(off):
        # binary-ish linear is fine for file sizes here
        for i in range(len(line_starts) - 1):
            if line_starts[i] <= off < line_starts[i + 1]:
                return i
        return len(lines) - 1

    to_comment = set()
    for s, e in matches:
        ls, le = line_of(s), line_of(e - 1)
        for li in range(ls, le + 1):
            to_comment.add(li)
    for li in sorted(to_comment):
        if not lines[li].lstrip().startswith(MARK):
            # preserve indentation
            stripped = lines[li]
            indent_len = len(stripped) - len(stripped.lstrip())
            lines[li] = stripped[:indent_len] + MARK + stripped[indent_len:]
    new_text = "\n".join(lines)
    with open(path + ".dmbak", "w") as f:
        f.write(text)
    with open(path, "w") as f:
        f.write(new_text)
    print(f"  [off]  {path}: commented {len(to_comment)} lines across {len(matches)} call sites")
    return len(matches)


def restore(path):
    bak = path + ".dmbak"
    if not os.path.exists(bak):
        print(f"  [skip] {path}: no backup")
        return
    with open(bak) as f:
        text = f.read()
    with open(path, "w") as f:
        f.write(text)
    os.remove(bak)
    print(f"  [on]   {path}: restored from backup")


def status(path):
    with open(path) as f:
        text = f.read()
    state = "OFF" if MARK in text else "ON"
    bak = "bak" if os.path.exists(path + ".dmbak") else "----"
    print(f"  {state:3} {bak} {path}")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("on", "off", "status"):
        print(__doc__)
        sys.exit(2)
    mode = sys.argv[1]
    total = 0
    for p in FILES:
        if not os.path.exists(p):
            print(f"  [err] missing {p}")
            continue
        if mode == "off":
            total += comment_out(p)
        elif mode == "on":
            restore(p)
        else:
            status(p)
    if mode == "off":
        print(f"Total DM call sites commented: {total}")


if __name__ == "__main__":
    main()
