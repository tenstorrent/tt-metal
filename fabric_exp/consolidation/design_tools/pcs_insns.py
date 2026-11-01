#!/usr/bin/env python3
"""pcs_insns.py CACHE GDIR PREFIX ROW CHIP CHAN ERISC OUT: per-instruction samples (cycles/packet) of one target, with
the disassembly and innermost source line of every instruction in the sampled range."""
import collections, csv, subprocess, sys, re
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
import pcs_sections as ps, pcs_profile as pp
cache, gdir, prefix, row, chip, chan, erisc, out = sys.argv[1:9]
vals = [float(r["avg_packets_per_s"]) for r in csv.DictReader(open(f"{ps.RES}/{prefix}_{row}_13129088_trial.csv"))]
cpp = 1.35e9 / (sum(vals) / len(vals))
for i, (tgt, h, addrs) in pp.target_samples(cache, prefix, row).items():
    if tgt["chip"] == int(chip) and tgt["chan"] == int(chan) and tgt["erisc"] == int(erisc):
        break
else:
    sys.exit("no such target")
proc = "active_erisc" if int(erisc) == 0 else "subordinate_active_erisc"
twin = f"{gdir}/{h}/{proc}/{proc}.elf"
cnt = collections.Counter(addrs); n = len(addrs)
dis = subprocess.run([ps.ROOT + "/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump", "-d", "--no-show-raw-insn", twin],
                     capture_output=True, text=True).stdout
ins = [(int(m.group(1), 16), m.group(2).strip()) for m in (re.match(r"\s*([0-9a-f]+):\s+(.*)$", l) for l in dis.splitlines()) if m]
lo, hi = min(cnt), max(cnt)
sel = [(a, t) for a, t in ins if lo - 64 <= a <= hi + 64]
loc = ps.a2l(twin, [a for a, _ in sel])
with open(out, "w") as f:
    f.write(f"# {twin} target chip {chip} chan {chan} E{erisc} {n} samples {cpp:.1f} cyc/pkt\n")
    for a, t in sel:
        fr = loc.get(a, [])
        src = f"{fr[0][1]}:{fr[0][2]}" if fr else "?"
        f.write(f"{a:08x} {cnt.get(a, 0) * cpp / n:6.2f}  {t:<48} {src}\n")
