#!/usr/bin/env python3
"""pcs_stack.py CACHE GDIR ROW... : time the busiest router ERISCs spend on stack traffic (shape inventory, RESULTS §98).

From the stage-3 PC samples (pcs_sections.py's mapping): for the busiest E0 and E1 of each row (largest share of samples
in work sections), the share of kernel samples on
  - stack loads (lw/lh/lhu/lb/lbu with an sp-relative address) and stack stores (sw/sh/sb),
  - the instruction right after a stack load when it reads the loaded register (the load-use stall is charged to the
    instruction waiting for the value, ROUTER_MAP_V3 caveats),
each also as cycles per packet (share x the row's cycles/packet). Disassembly from the -g twin (same code as ran)."""
import collections
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pcs_sections as ps  # noqa: E402

OBJDUMP = ps.ROOT + "/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump"
LOADS = ("lw", "lh", "lhu", "lb", "lbu")
STORES = ("sw", "sh", "sb")


def disasm(elf):
    """{addr: (mnemonic, operands)} and the address order."""
    ins = {}
    for line in subprocess.run([OBJDUMP, "-d", "--no-show-raw-insn", elf], capture_output=True, text=True).stdout.splitlines():
        m = re.match(r"\s*([0-9a-f]+):\s+(\S+)\s*(.*)$", line)
        if m:
            ins[int(m.group(1), 16)] = (m.group(2), m.group(3).split("#")[0].strip())
    return ins, sorted(ins)


def stack_kinds(ins, order):
    """{addr: kind}: 'stack load', 'stack store', 'stack load-use' (the next instruction reads the loaded register)."""
    kinds = {}
    for i, a in enumerate(order):
        mn, ops = ins[a]
        if "(sp)" not in ops:
            continue
        if mn in LOADS:
            kinds[a] = "stack load"
            rd = ops.split(",")[0].strip()
            if i + 1 < len(order):
                nmn, nops = ins[order[i + 1]]
                srcs = [x.strip() for x in re.split(r"[,()]", nops)[1:] if x.strip()]
                if nmn.startswith(("s", "b")) and not nmn.startswith(("sl", "sr", "sub")):  # stores/branches read all
                    srcs = [x.strip() for x in re.split(r"[,()]", nops) if x.strip()]
                if rd in srcs:
                    kinds.setdefault(order[i + 1], "stack load-use")
        elif mn in STORES:
            kinds[a] = "stack store"
    return kinds


def row(cache, gdir, name):
    r = ps.row_report(cache, gdir, name)
    cpp = r["cycles_per_packet"]
    out = []
    for erisc in (0, 1):
        cands = [t for t in r["targets"] if t["target"]["erisc"] == erisc and "sections" in t]
        if not cands:
            continue
        t = max(cands, key=lambda t: t["work_share"])
        out.append((erisc, t))
    return r, out, cpp


def main():
    cache, gdir, rows = sys.argv[1], sys.argv[2], sys.argv[3:]
    dis = {}
    print(f"{'row':12} {'ERISC':5} {'cyc/pkt':>7} {'stack ld':>9} {'ld-use':>9} {'stack st':>9} {'total':>9}  (cycles/packet)")
    for name in rows:
        r, picks, cpp = row(cache, gdir, name)
        for erisc, t in picks:
            proc = "active_erisc" if erisc == 0 else "subordinate_active_erisc"
            twin = f"{gdir}/{t['build']}/{proc}/{proc}.elf"
            if twin not in dis:
                ins, order = disasm(twin)
                dis[twin] = stack_kinds(ins, order)
            kinds = dis[twin]
            addrs = ps_kernel_addrs(cache, name, t)
            c = collections.Counter(kinds.get(a, "other") for a in addrs)
            n = sum(c.values())
            sh = {k: c[k] / n for k in ("stack load", "stack load-use", "stack store")}
            tot = sum(sh.values())
            print(f"{name:12} {'E' + str(erisc):5} {cpp:7.1f} " + " ".join(f"{sh[k] * cpp:9.1f}" for k in sh)
                  + f" {tot * cpp:9.1f}  ({tot * 100:.1f}% of samples)")


def ps_kernel_addrs(cache, name, tgt):
    """The target's kernel-sample ELF addresses after the 5% trims, as pcs_sections.row_report takes them."""
    import json, struct
    meta = json.load(open(f"{ps.RES}/pcs3_{name}_samples.json"))
    i = meta["targets"].index(tgt["target"])
    smp = [(t, pc) for t, j, pc in struct.Struct("<dHI").iter_unpack(open(f"{ps.RES}/pcs3_{name}_samples.bin", "rb").read())
           if j == i]
    key = f"{tgt['target']['chip']}/{tgt['target']['chan']}"
    lay = meta.get("layouts_end", {}).get(key) or meta.get("layouts", {}).get(key, {})
    e = lay["erisc"][str(tgt["target"]["erisc"])]
    proc = "active_erisc" if tgt["target"]["erisc"] == 0 else "subordinate_active_erisc"
    import glob
    xip = glob.glob(f"{cache}/**/fabric_erisc_router_v2/{tgt['build']}/{proc}/{proc}.elf.xip.elf", recursive=True)[0]
    base, _ = ps.text_segment(xip)
    load, size = e["load"], e["text_size"]
    kern = sorted((t, pc - load + base) for t, pc in smp if load <= pc < load + size)
    if len(kern) > 20:
        k = len(kern) // 20
        kern = kern[k:len(kern) - k]
    return [a for _, a in kern]


if __name__ == "__main__":
    main()
