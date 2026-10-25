#!/usr/bin/env python3
"""loopstat.py ELF [FUNC_SUBSTR]: the E0 inner pass loop: header = the grouped free-slots read (lui 0xffb40 + lw ..1188),
back-edge = the branch back to it. Prints contiguous loop size, sp loads/stores in it, and the frame of the function."""
import re, subprocess, sys
OD = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump"
NM = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-nm"
def fn_range(elf, sub):
    for l in subprocess.run([NM, "-S", "-C", elf], capture_output=True, text=True).stdout.splitlines():
        p = l.split(None, 3)
        if len(p) == 4 and sub in p[3]:
            return int(p[0], 16), int(p[0], 16) + int(p[1], 16)
def dis(elf, a, b):
    out = subprocess.run([OD, "-d", "--no-show-raw-insn", f"--start-address={a:#x}", f"--stop-address={b:#x}", elf],
                         capture_output=True, text=True).stdout
    ins = []
    for l in out.splitlines():
        m = re.match(r"\s+([0-9a-f]+):\s+(\S+)\s*(.*)$", l)
        if m:
            ins.append((int(m.group(1), 16), m.group(2), re.sub(r"\s*[#<].*$", "", m.group(3))))
    return ins
def stat(elf, sub):
    a, b = fn_range(elf, sub)
    ins = dis(elf, a, b)
    frame = next((int(x[2].split(",")[-1].lstrip("-")) for x in ins if x[1] == "addi" and x[2].startswith("sp,sp,-")), 0)
    # header: first 'lui r,0xffb40' followed within 2 by 'lw ..,1188(r)'
    hdr = None
    for i, (ad, op, ar) in enumerate(ins):
        if op == "lw" and ",1188(" in ar:
            hdr = ins[i - 1][0] if ins[i - 1][1] == "lui" else ad; break
    if hdr is None:
        return dict(frame=frame)
    def tgt(ar):
        try:
            return int(ar.split(",")[-1].strip(), 16)
        except ValueError:
            return None
    cands = [(ad, tgt(ar)) for ad, op, ar in ins if op.startswith("b") and tgt(ar) is not None and hdr - 96 <= tgt(ar) <= hdr and ad > hdr]
    top = max(t for _, t in cands)
    back = max(ad for ad, t in cands if t == top)
    hdr = top
    body = [x for x in ins if hdr <= x[0] <= back]
    spl = sum(1 for x in body if x[1] in ("lw", "lhu", "lbu", "lh", "lb") and "(sp)" in x[2])
    sps = sum(1 for x in body if x[1] in ("sw", "sh", "sb") and "(sp)" in x[2])
    calls = sum(1 for x in body if x[1] in ("jal", "jalr") and not x[2].startswith("zero"))
    return dict(frame=frame, header=hex(hdr), backedge=hex(back), loop=len(body), sp_loads=spl, sp_stores=sps, calls=calls,
                fn_insns=len(ins))
if __name__ == "__main__":
    sub = sys.argv[2] if len(sys.argv) > 2 else "e0::run_main_loop"
    for e in sys.argv[1:2] if len(sys.argv) <= 3 else sys.argv[1:]:
        print(e, stat(e, sub))
