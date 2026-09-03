#!/usr/bin/env python3
"""Compare the pack-loop addresses across the nop alignment passes.

The nop sweep assumes that N nops before the pack loop move that loop by 4N bytes.
The text size only proves the binary grew; the compiler may re-pad and leave the
loop where it was.  This script finds the loops in each pass's disassembly and
prints their start addresses and alignment, so the assumption is measured.

Usage:  perf_nop_disasm_report.py [dir]      (default ~/nopdeep)
"""
import re
import sys
from pathlib import Path

# objdump instruction line:  "     4b0:\t1101      \taddi\tsp,sp,-32"
INSN = re.compile(r"^\s*([0-9a-f]+):\s")
# branch / jump with a resolved target: "... bne a5,a4,4b0 <run_kernel+0x8>"
TARGET = re.compile(r"\b([0-9a-f]+)\s+<[^>]*>\s*$")
LABEL = re.compile(r"^([0-9a-f]+)\s+<([^>]+)>:")


def loops(path):
    """Backward branches in the file: (loop start, loop end, enclosing symbol)."""
    found, sym = [], "?"
    for line in path.read_text(errors="ignore").splitlines():
        lab = LABEL.match(line)
        if lab:
            sym = lab.group(2)
            continue
        m = INSN.match(line)
        if not m:
            continue
        addr = int(m.group(1), 16)
        tgt = TARGET.search(line)
        if not tgt:
            continue
        dest = int(tgt.group(1), 16)
        if dest < addr:  # branches backwards -> bottom of a loop
            found.append((dest, addr, sym))
    return found


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "nopdeep")
    files = sorted(
        root.glob("nop*_pack.asm"),
        key=lambda p: int(re.search(r"nop(\d+)_", p.name).group(1)),
    )
    if not files:
        print(f"no disassembly found under {root}")
        print("The sweep saves nopN_pack.asm only when an objdump is on PATH.")
        return 1

    base = None
    for f in files:
        n = int(re.search(r"nop(\d+)_", f.name).group(1))
        ls = loops(f)
        print(f"=== nop{n}  ({len(ls)} loops, expected shift {4 * n} bytes) ===")
        if base is None:
            base = {i: s for i, (s, _, _) in enumerate(ls)}
        for i, (start, end, sym) in enumerate(ls):
            moved = start - base.get(i, start)
            flag = "" if moved == 4 * n else "   <-- NOT 4N"
            print(
                f"   loop {i}: 0x{start:06x}..0x{end:06x}  "
                f"size {end - start + 2:4d}B  in {sym[:38]:38s} "
                f"mod16={start % 16:2d} mod32={start % 32:2d} mod64={start % 64:2d} "
                f"moved {moved:+4d}{flag}"
            )
    print(
        "\n'moved' is relative to the first pass. It should equal 4N if the nops"
        "\nreally shift the loop. Any 'NOT 4N' line means the compiler re-padded,"
        "\nand the offsets assumed by the sweep report are wrong for that loop."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
