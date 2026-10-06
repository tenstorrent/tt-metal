#!/usr/bin/env python3
"""Pack RISC-V (TRISC2) fetch, icache and branch-predictor numbers over the pack loop.

usage: analyze_fetch.py <run.fetch.cyc> [params.txt]
Input: to_cycles.py output of extract_fetch.py (without the 'param' lines).
"""
import sys
from collections import Counter

lines = open(sys.argv[1]).read().split("\n")
keys = lines[0].split()
rows = [dict(zip(keys, l.split())) for l in lines[1:] if l]
print("signals that never changed in the window:", sorted({"cmt_pc", "if_pc", "ex_mispredict", "ex_taken", "ex_branch_pc", "bp_taken", "bp_addr", "req_stall", "bp_stall", "ififo_empty", "imem_req", "imem_gnt", "imem_vld", "ic_l1_rden", "ic_l1_addr", "ic_l1_ready", "ic_l1_vld", "ic_mshr_miss", "ic_pf_stall", "ic_pf_vld", "ic_pf_addr", "ibuf_wren", "ibuf_ready", "bp_pred_pc", "if_mispredict"} - set(keys)))


def num(v):
    try:
        return int(v, 2)
    except ValueError:
        return None


pcs = [num(r.get("cmt_pc", "1" if "cmt_pc".endswith("ready") else "?")) for r in rows]
commits = [i for i in range(1, len(rows)) if pcs[i] != pcs[i - 1] and pcs[i] is not None]
cnt = Counter(pcs[i] for i in commits)
hot = sorted(pc for pc, n in cnt.items() if n >= 200)
head = min(hot)
at_head = [i for i in commits if pcs[i] == head]
a, b = at_head[0], at_head[-1]
print(f"hot loop: PCs {hot[0]:#x} to {hot[-1]:#x} ({len(hot)} instructions), head {head:#x}")
print(f"  committed {cnt[head]} times, cycles {a} to {b}")
for pc in hot:
    print(f"    {pc:#x}: {cnt[pc]} commits")
per = Counter(at_head[k + 1] - at_head[k] for k in range(len(at_head) - 1))
print(f"cycles per pass of the loop: {dict(sorted(per.items()))}")
print(f"  mean {(b - a) / (len(at_head) - 1):.2f}")

span = rows[a:b]


def count(cond):
    return sum(1 for r in span if cond(r))


def edges(key):
    return sum(1 for k in range(1, len(span)) if span[k][key] == "1" and span[k - 1][key] != "1")


print("over the loop:")
print(f"  branch mispredicts (execute stage): {count(lambda r: r.get('ex_mispredict', '?') == '1')} cycles, {edges('ex_mispredict')} events")
print(f"  fetch-side mispredict flag:          {count(lambda r: r.get('if_mispredict', '?') == '1')} cycles")
print(f"  predicted-taken fetches:             {count(lambda r: r.get('bp_taken', '?') == '1')} cycles")
print(f"  fetch request stalled:               {count(lambda r: r.get('req_stall', '?') == '1')} cycles")
print(f"  predictor fetch stall:               {count(lambda r: r.get('bp_stall', '?') == '1')} cycles")
print(f"  fetched-instruction queue empty:     {count(lambda r: r.get('ififo_empty', '?') == '1')} cycles")
print(f"  imem request not granted:            {count(lambda r: r.get('imem_req', '?') == '1' and r.get('imem_gnt', '?') != '1')} cycles")
print(f"  icache L1 reads requested:           {count(lambda r: r.get('ic_l1_rden', '?') == '1')} cycles, accepted {count(lambda r: r.get('ic_l1_rden', '?') == '1' and r.get('ic_l1_ready', '?') == '1')}")
print(f"  icache L1 read data returned:        {count(lambda r: r.get('ic_l1_vld', '?') == '1')} cycles")
print(f"  icache miss (MSHR) cycles:           {count(lambda r: r.get('ic_mshr_miss', '?') == '1')}")
print(f"  prefetch stall cycles:               {count(lambda r: r.get('ic_pf_stall', '?') == '1')}")
print(f"  pack instruction pushes: accepted {count(lambda r: r.get('ibuf_wren', '?') == '1' and r.get('ibuf_ready', '?') == '1')}, blocked {count(lambda r: r.get('ibuf_wren', '?') == '1' and r.get('ibuf_ready', '?') != '1')} cycles")
for k in ("ic_cnt_req", "ic_cnt_hit", "ic_cnt_stall", "ic_cnt_mshr"):
    x, y = num(span[0].get(k, "?")), num(span[-1].get(k, "?"))
    print(f"  icache counter {k[7:]}: +{y - x if None not in (x, y) else '?'}")
addrs = Counter(num(r.get("ic_l1_addr", "1" if "ic_l1_addr".endswith("ready") else "?")) for r in span if r.get("ic_l1_rden", "1" if "ic_l1_rden".endswith("ready") else "?") == "1" and r.get("ic_l1_ready", "1" if "ic_l1_ready".endswith("ready") else "?") == "1")
print("  icache L1 fetch line addresses (byte address = value * 16):")
for ad, n in sorted(addrs.items()):
    print(f"    {ad * 16:#x}: {n}")
mis = Counter(num(r.get("ex_branch_pc", "1" if "ex_branch_pc".endswith("ready") else "?")) for k, r in enumerate(span) if r.get("ex_mispredict", "1" if "ex_mispredict".endswith("ready") else "?") == "1" and (k == 0 or span[k - 1].get("ex_mispredict") != "1"))
print("  mispredicted branch PCs:", {f"{p:#x}": n for p, n in mis.most_common(8)})
mid = at_head[len(at_head) // 2]
end = at_head[len(at_head) // 2 + 2]
print(f"two passes from cycle {mid}:")
print("  cyc   cmt_pc  if_pc   l1rd l1vld miss mispr bpT stall ifqE push")
for k in range(mid, end):
    r = rows[k]
    p = num(r.get("cmt_pc", "1" if "cmt_pc".endswith("ready") else "?")); q = num(r.get("if_pc", "1" if "if_pc".endswith("ready") else "?"))
    print(f"  {k - mid:3d}  {p:#7x} {q if q is None else hex(q):>7} {r.get('ic_l1_rden', '?'):>4} {r.get('ic_l1_vld', '?'):>5} {r.get('ic_mshr_miss', '?'):>4} {r.get('ex_mispredict', '?'):>5} {r.get('bp_taken', '?'):>3} {r.get('req_stall', '?'):>5} {r.get('ififo_empty', '?'):>4} {('W' if r.get('ibuf_ready', '?') == '1' else 'w') if r.get('ibuf_wren', '?') == '1' else '.':>4}")
