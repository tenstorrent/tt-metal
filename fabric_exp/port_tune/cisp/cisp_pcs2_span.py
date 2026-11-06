#!/usr/bin/env python3
"""cisp_pcs2_span.py RESULTS_DIR: per NX 4096 B write L2 iteration, each router's first and last ERISC0 sender::send
sample; the stall-event routers' offsets against the same iteration's quiet routers, against their telemetry excess."""
import sys, os, re, collections, statistics
import numpy as np
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/port_tune/cisp")
D = sys.argv[1]; sys.argv = ["x", D]
import cisp_pcs2_report as P
import cisp_router_map as rmap
peer, c2chip = rmap.topology()
for arm in ("on", "off"):
    meta, rec, t_abs, tg_info, bucket_of, line_of = P.load(arm)
    win = P.windows(os.path.join(D, f"pcs2_{arm}.log"))
    its = rmap.iterations(os.path.join(D, f"pcs2_{arm}.log")); off_it = rmap.iterations(os.path.join(D, "pcs2_off.log"))
    tidx = {(t["chip"], t["chan"]): i for i, t in tg_info.items() if t["matched"] and t["erisc"] == 0}
    sendpc = {}
    def is_send(j):
        key = line_of.get(j)
        if not key or key[0] == "fw": return False
        r = sendpc.get(key)
        if r is None:
            r = sendpc[key] = any(f.endswith("sender::send") for f, _ in P._chains[key])
        return r
    diffs, cyc_d, span_ev, span_ot = [], [], [], []
    for test, k, a, b in win:
        if not re.search(r"size_4096_ntype_unicast_write_num_links_2$", test) or k >= len(its.get(test, [])): continue
        spans, evs = {}, {}
        for rt, cyc in its[test][k].items():
            base = statistics.median(it[rt] for it in off_it[test] if rt in it)
            (r, c), ch = rt; i = tidx.get((c2chip[(r, c)], ch))
            if i is None: continue
            sel = np.nonzero((rec["i"] == i) & (t_abs >= a) & (t_abs <= b))[0]
            ts = [t_abs[j] for j in sel.tolist() if is_send(j)]
            if len(ts) < 3: continue
            spans[rt] = (ts[0], ts[-1]); evs[rt] = (cyc - base > 300e3, cyc - base)
        quiet = [spans[r] for r in spans if not evs[r][0]]
        if not quiet: continue
        qf = statistics.mean(x[0] for x in quiet); ql = statistics.mean(x[1] for x in quiet)
        for r in spans:
            if evs[r][0]:
                diffs.append((spans[r][0] - qf, spans[r][1] - ql)); cyc_d.append(evs[r][1])
    if diffs:
        f = [d[0] for d in diffs]; l = [d[1] for d in diffs]
        sf = statistics.stdev(f) / len(f) ** 0.5; sl = statistics.stdev(l) / len(l) ** 0.5
        print(f"{arm}: {len(diffs)} event router-iterations vs the same iteration's quiet routers: first send {1e3 * statistics.mean(f):+.3f} ms (SE {1e3 * sf:.3f}), last send {1e3 * statistics.mean(l):+.3f} ms (SE {1e3 * sl:.3f}); telemetry excess {statistics.mean(cyc_d) / 1.35e6:.3f} ms")
    else:
        print(f"{arm}: no event router-iterations")
