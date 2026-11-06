#!/usr/bin/env python3
"""cisp_cmp.py MAIN_DIR PORT_DIR: per bandwidth test, profiler on/off ratio per pass for two cisp_ab.sh result dirs
(main and port), the off GB/s, and port off / main off. Bandwidth parsed as cisp_report.py does."""
import re, statistics, sys
RUNNING = re.compile(r"Running Test: (\S+)"); BW = re.compile(r"BW \(GB/s\)=([0-9.]+)")
def bandwidth(log):
    bw, running = {}, None
    for line in open(log, errors="replace"):
        if m := RUNNING.search(line): running = m.group(1)
        elif running and not running.startswith("Latency") and (m := BW.search(line)):
            bw.setdefault(running, []).append(float(m.group(1)))
    return {t: statistics.mean(v) for t, v in bw.items()}
M, P = sys.argv[1], sys.argv[2]
m = {k: bandwidth(f"{M}/sp{k}.log") for k in ("a_off","a_on","b_off","b_on")}
p = {k: bandwidth(f"{P}/sp{k}.log") for k in ("a_off","a_on","b_off","b_on")}
short = lambda t: t.replace("Unicast_size_","/").replace("_ntype_unicast_","/").replace("_ntype_","/").replace("_num_links_","/L").replace("SingleSender","SS").replace("AllDevices","")
print("| test | main off a GB/s | main on/off a | main on/off b | port off a GB/s | port on/off a | port on/off b | port off / main off |")
print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
for t in sorted(m["a_off"]):
    if t not in p["a_off"]: print("missing in port", t); continue
    print(f"| {short(t)} | {m['a_off'][t]:.3f} | {m['a_on'][t]/m['a_off'][t]:.4f} | {m['b_on'][t]/m['b_off'][t]:.4f} | {p['a_off'][t]:.3f} | {p['a_on'][t]/p['a_off'][t]:.4f} | {p['b_on'][t]/p['b_off'][t]:.4f} | {p['a_off'][t]/m['a_off'][t]:.4f} |")
for name, d in (("main", m), ("port", p)):
    for ps in "ab":
        r = [d[f"{ps}_on"][t]/d[f"{ps}_off"][t] for t in d[f"{ps}_off"]]
        print(name, ps, "median %.4f min %.4f max %.4f n %d" % (statistics.median(r), min(r), max(r), len(r)))
