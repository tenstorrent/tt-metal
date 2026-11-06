#!/usr/bin/env python3
"""cisp_x7_report.py RESULTS_DIR: the X7 router PC samples (cisp_pcs3.sh, profiler on). For each sampled arm
(i: stock L1/L2 alternation, ii: L1-only, gap: alternation with an idle pause after each L2 block), the routers' ERISC0
and ERISC1 samples inside NX write iterations, split by router state in that iteration:
  idle-link1   a link-1 router (one the num_links 1 rows don't use) during a num_links 1 row: in (i) / gap these follow
               an L2 block (link 1 carried traffic one row earlier), in (ii) link 1 never carries traffic
  active       a router that printed telemetry in the iteration (it carried traffic); 'low' when its cycles are +2%
               over the off arm's median for the row type and router (the tally's rule)
Per group: sample shares by bucket (cisp_pcs2_report.py's buckets: passes, end_round, ctx_switch, telemetry,
link_sync, eth_ptp, ring_ensure_room_slow, kernel_profiler, firmware) and the top source lines outside the router's
passes. The idle-link1 group of (i) is further split into the slow-state iterations (some router low) and the rest.
Observation only: PC samples split cycles within an ERISC; they don't name a cause (CLAUDE.md)."""
import collections, csv, glob, json, os, statistics, sys
import numpy as np

R = sys.argv[1]
sys.argv = ["x", R]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cisp_pcs2_report as P  # noqa: E402
import cisp_router_tally as T  # noqa: E402

CACHE = "/home/aagarwal/.tt-metal-cache-cisp-on-ci"


def load(v):
    pre = os.path.join(R, f"x7_{v}_on_samples")
    meta = json.load(open(pre + ".json"))
    rec = np.fromfile(pre + ".bin", dtype=np.dtype([("t", "<f8"), ("i", "<u2"), ("pc", "<u4")]))
    t_abs = rec["t"] + meta["t_start_epoch"]
    imgs = {p: [(e, *P.text_segment(e)) for e in sorted(glob.glob(
        f"{CACHE}/**/fabric_erisc_router*/*/{p}/{p}.elf.xip.elf", recursive=True))]
        for p in ("active_erisc", "subordinate_active_erisc")}
    bucket_of = np.full(len(rec), -1, dtype=np.int8)
    line_of = {}
    tg_ok = {}
    for idx, tg in enumerate(meta["targets"]):
        e = tg["erisc"]
        proc = "active_erisc" if e == 0 else "subordinate_active_erisc"
        lay = meta["layouts"].get(f"{tg['chip']}/{tg['chan']}", {}).get("erisc", {}).get(str(e))
        if not lay:
            continue
        head, ld, size = bytes.fromhex(lay["head"]), lay["load"], lay["text_size"]
        m = next(((x, base) for x, base, text in imgs[proc] if head and text[:len(head)] == head), None)
        sel = np.nonzero(rec["i"] == idx)[0]
        if m is None:
            continue
        tg_ok[idx] = tg
        elf, base = m[0][: -len(".xip.elf")], m[1]
        pcs = rec["pc"][sel]
        inside = (pcs >= ld) & (pcs < ld + size)
        bucket_of[sel[~inside]] = P.ORDER.index("firmware")
        link = pcs[inside].astype(np.int64) - ld + base
        uniq = sorted(set(link.tolist()))
        ch = P.chains(elf, uniq)
        bmap = {a: P.ORDER.index(P.bucket(ch[a])) for a in uniq}
        bucket_of[sel[inside]] = np.array([bmap[a] for a in link.tolist()], dtype=np.int8)
        for j, a in zip(sel[inside].tolist(), link.tolist()):
            c0 = ch[a][0] if ch[a] else ("?", "?")
            line_of[j] = f"{c0[0]} {c0[1]}"
        for j, pc in zip(sel[~inside].tolist(), pcs[~inside].tolist()):
            line_of[j] = f"firmware pc 0x{pc & ~0xff:x}xx"
    return meta, rec, t_abs, tg_ok, bucket_of, line_of


# link-1 routers: those the num_links 1 rows never use (from the off logs of every arm)
used_l1 = set()
for v in ("i", "ii", "gap"):
    for (n, tel), r in zip(T.parse(f"{R}/x7_{v}_off.log"), csv.DictReader(open(f"{R}/x7_{v}_off_bandwidth_results.csv"))):
        if T.rowkey(n, r)[2] == "1" and r["num_packets"] != "1":
            used_l1 |= set(tel)
peer_ok = sum(1 for rt in used_l1 if rt in T.peer)
print(f"link-0 routers (used by num_links 1 rows): {len(used_l1)}; all in the cluster descriptor: {peer_ok == len(used_l1)}")
groups = {}
for v in ("i", "ii", "gap"):
    if not os.path.exists(f"{R}/x7_{v}_on_samples.bin"):
        continue
    meta, rec, t_abs, tg_ok, bucket_of, line_of = load(v)
    off = T.parse(f"{R}/x7_{v}_off.log"); offr = list(csv.DictReader(open(f"{R}/x7_{v}_off_bandwidth_results.csv")))
    base = collections.defaultdict(list)
    for (n, tel), r in zip(off, offr):
        for rt, c in tel.items():
            base[(T.rowkey(n, r), rt)].append(c)
    base = {k: statistics.median(x) for k, x in base.items()}
    on = T.parse(f"{R}/x7_{v}_on.log"); onr = list(csv.DictReader(open(f"{R}/x7_{v}_on_bandwidth_results.csv")))
    win = P.windows(f"{R}/x7_{v}_on.log")
    assert len(win) == len(on) == len(onr), (v, len(win), len(on), len(onr))
    tidx = {(t["chip"], t["chan"]): [] for t in tg_ok.values()}
    for i, t in tg_ok.items():
        tidx[(t["chip"], t["chan"])].append(i)
    starts = np.array([w[2] for w in win]); ends = np.array([w[3] for w in win])
    j_of = np.searchsorted(starts, t_abs, side="right") - 1
    valid = (j_of >= 0) & (t_abs <= ends[np.clip(j_of, 0, len(ends) - 1)])
    unknown = sum(1 for k in tidx if k not in T.peer)
    print(f"\n== arm {v}: {meta['samples']} samples, {len(tg_ok)} matched targets ({unknown} router ids not in the "
          f"cluster descriptor), {len(win)} iterations")
    slowj = set()
    rstate = {}
    for j, ((n, tel), r) in enumerate(zip(on, onr)):
        k = T.rowkey(n, r)
        if k[1] != "unicast_write" or r["num_packets"] == "1":
            continue
        low = {rt for rt, c in tel.items() if c / base[(k, rt)] - 1 > 0.02}
        if low:
            slowj.add(j)
        for key in tidx:
            if key in tel:
                rstate[(j, key)] = "active-low" if key in low else "active"
            elif k[2] == "1" and key not in used_l1:
                rstate[(j, key)] = "idle-link1"
    print(f"   write iterations with a low router: {len(slowj)}")
    acc = collections.defaultdict(lambda: [collections.Counter(), collections.Counter()])
    for (j, key), st in rstate.items():
        g = st if st != "idle-link1" else ("idle-link1 (slow it)" if j in slowj else "idle-link1 (normal it)")
        for i in tidx[key]:
            e = tg_ok[i]["erisc"]
            sel = np.nonzero((rec["i"] == i) & valid & (j_of == j))[0]
            for s_ in sel.tolist():
                b = bucket_of[s_]
                if b < 0:
                    continue
                acc[(v, g, e)][0][P.ORDER[b]] += 1
                if P.ORDER[b] != "passes":
                    acc[(v, g, e)][1][line_of.get(s_, "?")] += 1
    groups.update(acc)
for key in sorted(groups):
    bc, lc = groups[key]
    n = sum(bc.values())
    if not n:
        continue
    print(f"\n-- arm {key[0]}, {key[1]}, ERISC{key[2]}: {n} samples: " +
          ", ".join(f"{b} {100 * bc[b] / n:.1f}%" for b in P.ORDER if bc[b]))
    for ln, c in lc.most_common(int(os.environ.get("TOP", "6"))):
        print(f"      {100 * c / n:5.2f}%  {ln}")
