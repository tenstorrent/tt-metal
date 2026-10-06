#!/usr/bin/env python3
"""Write a small VCD (GTKWave, Surfer) with only the extracted signals, for a time window.

usage: make_small_vcd.py <out.vcd> <t0> <t1> <signals.txt> [more.txt]   (time = 2 x clock cycle)
"""
import string
import sys

out, t0, t1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
NAMES = {
    "clk": "clk",
    "rden": "dest_read_request_packers_3210",
    "ready": "dest_read_grant_packers_3210",
    "ready_i": "dest_read_ready_in_packers_3210",
    "dvld": "dest_read_data_valid_packers_3210",
    "pdvld": "pack_dest_data_valid_3210",
    "rdfifo_rden": "pack_dest_fifo_read_3210",
    "t2_ibuf_wren": "pack_riscv_instr_push",
    "t2_ibuf_ready": "pack_riscv_instr_queue_ready",
    "t1_l1_rden": "math_riscv_l1_read",
    "t1_l1_wren": "math_riscv_l1_write",
    "t1_l1_addr": "math_riscv_l1_addr_16B",
    "t0_l1_rden": "unpack_riscv_l1_read",
    "t2_l1_rden": "pack_riscv_l1_read",
}
seen = {}
changes = []
for fn in sys.argv[4:]:
    for line in open(fn):
        if line[0] == "#":
            continue
        t, lab, val = line.split()
        t = int(t)
        if lab in seen and seen[lab][0] != fn:
            continue
        seen.setdefault(lab, [fn, len(val)])
        seen[lab][1] = max(seen[lab][1], len(val))
        changes.append((t, lab, val))
changes.sort(key=lambda c: c[0])
labs = sorted(seen)
chars = string.ascii_letters + string.digits
ident = {l: chars[i // len(chars)] + chars[i % len(chars)] for i, l in enumerate(labs)}
cur = {}
with open(out, "w") as w:
    w.write("$timescale 1ps $end\n$scope module versim_wormhole_pack_repro $end\n")
    for l in labs:
        w.write(f"$var wire {seen[l][1]} {ident[l]} {NAMES.get(l, l)} $end\n")
    w.write("$upscope $end\n$enddefinitions $end\n")
    last = None
    for t, lab, val in changes:
        if t < t0:
            cur[lab] = val
            continue
        if t > t1:
            break
        if last is None:
            w.write(f"#{t0}\n$dumpvars\n")
            for l in labs:
                v = cur.get(l, "x")
                w.write(
                    (f"b{v} {ident[l]}" if seen[l][1] > 1 else f"{v}{ident[l]}") + "\n"
                )
            w.write("$end\n")
            last = t0
        if t != last:
            w.write(f"#{t}\n")
            last = t
        w.write(
            (f"b{val} {ident[lab]}" if seen[lab][1] > 1 else f"{val}{ident[lab]}")
            + "\n"
        )
print("signals", len(labs))
