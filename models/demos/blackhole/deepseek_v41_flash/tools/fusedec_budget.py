"""Per-op roofline budget of one decode round.
python3 -I fusedec_budget.py <run dir> [--trace M] [--walls <walls.json of the unprofiled run>] [--out BUDGET.md] [--csv ops.csv]

For EVERY op of the traced round (digest rows of the profiled replay joined with the OpRecorder rows of the capture): measured kernel time (mean / slowest / fastest chip), bytes moved
(DRAM per chip from the tile-padded shapes and dtypes of the op's tensors; fabric per chip for collectives), FLOPs (matmul / linear: 2 * M_padded * K * N), the ideal time from the measured
ceilings (DRAM 511.6 GB/s per chip, fabric ring 77 GB/s, matmul peaks LoFi 307 / HiFi2 190 / HiFi4 108 TFLOP/s per chip), the ratio measured / ideal and a cause class:
  at ceiling | DRAM-bound (below the ceiling) | compute-bound | fabric transfer | latency-bound CCL | wait for the slowest chip (collective time above its fastest chip) | tiny-op overhead (ideal < 1.5 us).
The ranked table is 'time above ideal' summed per op group; layer types and the round are aggregated; the gap between ops is wall (unprofiled replay) minus the kernel sum.
Byte rules that are not the plain sum of the DRAM tensors: moe_compute (n active experts per chip from the real routed ids x 37.6 MB), sparse_sdpa (rows x topk x 512 x pool dtype bytes), indexer score
(users x n_score x 128 x bfp8), embedding (rows read, not the table), tensors > 256 MB (pools / slabs) are excluded from the plain sum.
"""
import csv
import json
import os
import re
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fusedec_roundtab as RT  # noqa: E402

DRAM_BW = 511.6e9
FAB_BW = 77e9
PEAK = {"LoFi": 307e12, "HiFi2": 190e12, "HiFi4": 108e12}
EXPERT_BYTES = 3 * 5120 * 2304 * 1.0625
BYTES = {
    "BFLOAT16": 2,
    "FLOAT32": 4,
    "BFLOAT8_B": 1.0625,
    "BFLOAT4_B": 0.5625,
    "UINT32": 4,
    "INT32": 4,
    "UINT16": 2,
    "UINT8": 1,
    "FLOAT8_E4M3": 1,
}
TYPES = {}
for i in (0, 1):
    TYPES[i] = "SWA (L0, L1)"
for i in (2, 8, 14):
    TYPES[i] = "CSA2 owner + indexer (L2, L8, L14)"
for i in list(range(3, 8)) + list(range(9, 14)) + list(range(15, 20)):
    TYPES[i] = "CSA2 reuse (15 layers)"
TYPES[20] = "ratio-1 owner + indexer (L20)"
for i in (24, 28, 32, 36):
    TYPES[i] = "ratio-1 index-only (L24, 28, 32, 36)"
for i in list(range(21, 24)) + list(range(25, 28)) + list(range(29, 32)) + list(range(33, 36)) + list(range(37, 40)):
    TYPES[i] = "ratio-1 reader (15 layers)"


def parse(d):
    m = re.match(r"\[([0-9x]+)\]\s+(\S+)\s+(T|RM)\s+(L1|DRAM)(-sh)?", d or "")
    if not m:
        return None
    shape = [int(x) for x in m.group(1).split("x")]
    dt = m.group(2).upper()
    return dict(shape=shape, dt=dt, tile=m.group(3) == "T", mem=m.group(4), bytes=nbytes(shape, dt, m.group(3) == "T"))


def nbytes(shape, dt, tile):
    s = list(shape)
    if tile and len(s) >= 2:
        s[-1] = -(-s[-1] // 32) * 32
        s[-2] = -(-s[-2] // 32) * 32
    n = 1
    for x in s:
        n *= x
    return n * BYTES.get(dt, 2)


def routes_per_layer(path):
    """-> {layer: (mean distinct experts per chip, max, mean dest chips per token)} from the real routed ids of the round."""
    if not path or not os.path.exists(path):
        return {}
    R = json.load(open(path))
    out = {}
    items = R.items() if isinstance(R, dict) else enumerate(R)
    for L, rows in items:
        L = int(L)
        ids = [e for r in rows for e in r]
        chip = lambda e: ((e % 48) // 12) * 8 + e // 48
        cnt = defaultdict(set)
        for e in set(ids):
            cnt[chip(e)].add(e)
        n = [len(cnt.get(c, ())) for c in range(32)]
        dest = statistics.mean(len({chip(e) for e in r}) for r in rows)
        out[L] = (statistics.mean(n), max(n), dest)
    return out


def op_model(row, per, layer, routes, rows_T):
    """-> dict(dram, fab, flops, peak, note) per chip for one op"""
    op, who, kern = row["op"], row["caller"], row.get("kernels", "")
    low = op.lower()
    ins = [parse(d) for d in row.get("ins", [])]
    outs = [parse(d) for d in row.get("outs", [])]
    ins = [x for x in ins if x]
    outs = [x for x in outs if x]
    dram = sum(x["bytes"] for x in ins if x["mem"] == "DRAM" and x["bytes"] < 256e6) + sum(
        x["bytes"] for x in outs if x["mem"] == "DRAM" and x["bytes"] < 256e6
    )
    fab, flops, peak, note = 0.0, 0.0, None, ""
    if "moe_compute" in low:
        mean_n, max_n, _ = routes.get(layer, (4.0, 6, 5))
        dram = mean_n * EXPERT_BYTES
        flops = mean_n * 3 * 2 * 32 * 5120 * 2304
        peak = PEAK["HiFi2"]
        note = f"n experts/chip mean {mean_n:.2f} max {max_n}"
    elif "sparse_sdpa" in low:
        # q [1,32,T,512] rows, idx [1,1,rows,topk]
        idx = [x for x in ins if x["dt"] in ("UINT32", "INT32") and len(x["shape"]) == 4]
        topk = idx[0]["shape"][-1] if idx else 640
        rows = idx[0]["shape"][-2] if idx else rows_T
        dram = rows * topk * 512 * 1.0
        flops = 2 * 2 * rows * 32 * topk * 512
        peak = PEAK["HiFi4"]
        note = f"rows {rows} topk {topk}"
    elif "score_mask_fused" in who or "score_reader" in kern:
        outs0 = outs[0]["shape"] if outs else [1, 1, 1, 1]
        n_score = outs0[-1]
        users = max(1, rows_T // 4 // 2) if rows_T else 2
        dram = 2 * n_score * 128 * 1.0625
        note = f"n_score {n_score}"
    elif "embedding" in low:
        o = outs[0] if outs else None
        dram = (o["bytes"] if o else 0) * 2
    elif "linear" in low or "matmul" in low:
        a = ins[0] if ins else None
        b = ins[1] if len(ins) > 1 else None
        if a and b and len(b["shape"]) >= 2:
            M = 1
            for x in a["shape"][:-1]:
                M *= x
            Mp = -(-M // 32) * 32
            K, N = b["shape"][-2], b["shape"][-1]
            flops = 2 * Mp * K * N
            peak = PEAK["HiFi4"]
            note = f"M {M} K {K} N {N}"
    elif any(s in low for s in ("reduce_scatter", "all_reduce")):
        inb = ins[0]["bytes"] if ins else 0
        fab = inb * 7 / 8
        dram = 0
    elif "all_gather" in low:
        ob = outs[0]["bytes"] if outs else 0
        fab = ob * 7 / 8
        dram = 0
    elif "all_to_all_dispatch" in low:
        _, _, dest = routes.get(layer, (4.0, 6, 5.0))
        xb = ins[0]["bytes"] if ins else rows_T * 10240
        fab = xb * dest / 8
        dram = 0
    return dict(dram=dram, fab=fab, flops=flops, peak=peak, note=note)


def classify(meas_mean, meas_min, ideal, dram_t, comp_t, fab_t, is_ccl):
    if is_ccl:
        if meas_min <= 1.5 * max(ideal, 1e-9) + 6e-6:
            return "fabric transfer" if (meas_mean - meas_min) < 0.3 * meas_mean else "wait for the slowest chip"
        return "latency-bound CCL" if (meas_mean - meas_min) < 0.3 * meas_mean else "wait for the slowest chip"
    if ideal < 1.5e-6:
        return "tiny-op overhead"
    r = meas_mean / ideal
    if comp_t > dram_t:
        return "compute-bound" if r > 1.25 else "at ceiling"
    return "at ceiling" if r <= 1.25 else "DRAM-bound (below the ceiling)"


def main():
    run = sys.argv[1]
    arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    name = arg("--trace", "M")
    O, pm, walls = RT.load(run)
    tid = pm[name]
    T = RT.trace_table(run, O, name, tid)
    routes = routes_per_layer(os.path.join(run, "routes_round.json"))
    meta = O["meta"]
    rows_T = meta.get("T", 8) * 4
    wall = None
    wp = arg("--walls")
    if wp and os.path.exists(wp):
        wall = json.load(open(wp)).get("walls", {})
    elif walls.get("walls"):
        wall = walls["walls"]
    # device timeline: per chip, gap before every op = its start minus the end of the previous op on that chip (cycles at 1.35 per ns; one replay session)
    seq = defaultdict(list)
    for oid, row, ph, per in T["ops"]:
        for d, (k_, g_, s_, e_) in per.items():
            seq[d].append((oid, s_, e_))
    gap_by = defaultdict(list)
    chip_wall = {}
    for d, lst in seq.items():
        lst.sort()
        for (o0, s0, e0), (o1, s1, e1) in zip(lst, lst[1:]):
            gap_by[o1].append(max(0.0, (s1 - e0) / RT.CLK) * 1e-9)
        chip_wall[d] = (max(e for _, _, e in lst) - min(s for _, s, _ in lst)) / RT.CLK * 1e-9
    risc = T.get("risc", {})
    recs = []
    for oid, row, ph, per in T["ops"]:
        if row is None:
            continue
        m = re.match(r"layer(\d+)", ph)
        layer = int(m.group(1)) if m else None
        ks = [v[0] * 1e-9 for v in per.values()]
        mean, mx, mn = statistics.mean(ks), max(ks), min(ks)
        mod = op_model(row, per, layer, routes, rows_T)
        cat, nm = RT.category(row)
        is_ccl = "CCL" in cat or "all_gather" in row["op"] or "reduce_scatter" in row["op"] or "all_to_all" in row["op"]
        dram_t = mod["dram"] / DRAM_BW
        comp_t = (mod["flops"] / mod["peak"]) if mod["peak"] else 0.0
        fab_t = mod["fab"] / FAB_BW
        ideal = max(dram_t, comp_t, fab_t)
        if (
            ideal > mean
        ):  # an ideal above the measurement means the byte model over-counts (a tensor read only in part): clip, flagged
            mod["note"] = (mod["note"] + " ideal clipped (bytes over-counted)").strip()
            ideal = mean
        cls = classify(mean, mn, ideal, dram_t, comp_t, fab_t, is_ccl)
        wait = (mean - mn) if is_ccl else 0.0
        recs.append(
            dict(
                oid=oid,
                phase=ph,
                layer=layer,
                cat=cat,
                name=nm,
                op=row["op"],
                caller=row["caller"],
                mean=mean,
                slow=mx,
                fast=mn,
                dram=mod["dram"],
                fab=mod["fab"],
                flops=mod["flops"],
                ideal=ideal,
                cls=cls,
                wait=wait,
                note=mod["note"],
                gap=(statistics.mean(gap_by[oid]) if gap_by.get(oid) else 0.0),
                cores=risc.get(oid, {}).get("cc", 0),
                brisc=risc.get(oid, {}).get("br", 0) * 1e-9,
                ncrisc=risc.get(oid, {}).get("nc", 0) * 1e-9,
                trisc1=risc.get(oid, {}).get("t1", 0) * 1e-9,
            )
        )
    tot = sum(r["mean"] for r in recs)
    ideal_tot = sum(r["ideal"] for r in recs)
    wait_tot = sum(r["wait"] for r in recs)
    out = []
    P = out.append
    P(
        f"# Decode round budget (per-op roofline): {meta['layers'] if len(meta['layers']) < 12 else '40 layers'}, B={meta['B']} k={meta['k']} T={meta['T']}, bucket {meta['bucket']}, run {os.path.basename(run)}\n"
    )
    P(
        "Ceilings used: DRAM 511.6 GB/s per chip (with PIN_EAST_X), fabric ring 77 GB/s per chip, matmul peaks LoFi 307 / HiFi2 190 / HiFi4 108 TFLOP/s per chip. Per-op bytes / FLOPs from the recorded shapes, dtypes and the real routed ids of the round (rules in the script header).\n"
    )
    P(
        f"ops {len(recs)}; kernel sum on the mean chip {tot*1e3:.2f} ms; ideal sum {ideal_tot*1e3:.2f} ms; time above ideal {(tot-ideal_tot)*1e3:.2f} ms; of the collective time {wait_tot*1e3:.2f} ms is wait for the slowest chip (mean minus fastest chip)."
    )
    if wall:
        w = wall.get(name)
        if w:
            P(
                f"Wall of the unprofiled replay of this trace {w:.2f} ms -> gap / launch time between ops {w-tot*1e3:.2f} ms ({100*(w-tot*1e3)/w:.1f} % of the wall, {1e3*(w-tot*1e3)/len(recs):.2f} us per op)."
            )
    P("")
    # ranked above-ideal by group
    g = defaultdict(lambda: dict(n=0, mean=0.0, ideal=0.0, slow=0.0, wait=0.0, cls=defaultdict(float)))
    for r in recs:
        x = g[(r["cat"], r["name"][:70])]
        x["n"] += 1
        x["mean"] += r["mean"]
        x["ideal"] += r["ideal"]
        x["slow"] += r["slow"]
        x["wait"] += r["wait"]
        x["cls"][r["cls"]] += r["mean"] - r["ideal"]
    P("## Ranked: time above ideal (op groups, whole round)\n")
    P("| rank | op group | n | measured ms | ideal ms | above ideal ms | meas/ideal | main cause |")
    P("|---|---|---|---|---|---|---|---|")
    for i, (k, x) in enumerate(sorted(g.items(), key=lambda kv: -(kv[1]["mean"] - kv[1]["ideal"]))[:45]):
        cause = max(x["cls"].items(), key=lambda kv: kv[1])[0]
        P(
            f"| {i+1} | {k[0]}: {k[1]} | {x['n']} | {x['mean']*1e3:.3f} | {x['ideal']*1e3:.3f} | {(x['mean']-x['ideal'])*1e3:.3f} | {x['mean']/max(x['ideal'],1e-9):.1f} | {cause} |"
        )
    # by cause
    c = defaultdict(lambda: [0, 0.0, 0.0])
    for r in recs:
        c[r["cls"]][0] += 1
        c[r["cls"]][1] += r["mean"]
        c[r["cls"]][2] += r["ideal"]
    P("\n## By cause class (whole round)\n")
    P("| cause | ops | measured ms | ideal ms | above ideal ms |")
    P("|---|---|---|---|---|")
    for k, (n, m_, i_) in sorted(c.items(), key=lambda kv: -(kv[1][1] - kv[1][2])):
        P(f"| {k} | {n} | {m_*1e3:.3f} | {i_*1e3:.3f} | {(m_-i_)*1e3:.3f} |")
    # by category
    cc = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    for r in recs:
        x = cc[r["cat"]]
        x[0] += 1
        x[1] += r["mean"]
        x[2] += r["ideal"]
        x[3] += r["slow"]
    P("\n## By category (whole round)\n")
    P(
        "| category | ops | measured ms (mean chip) | slowest-chip sum ms | ideal ms | above ideal ms | % of kernel sum |"
    )
    P("|---|---|---|---|---|---|---|")
    for k, (n, m_, i_, s_) in sorted(cc.items(), key=lambda kv: -kv[1][1]):
        P(f"| {k} | {n} | {m_*1e3:.3f} | {s_*1e3:.3f} | {i_*1e3:.3f} | {(m_-i_)*1e3:.3f} | {100*m_/tot:.1f} |")
    # per layer type
    lt = defaultdict(lambda: dict(layers=set(), mean=0.0, ideal=0.0, cat=defaultdict(lambda: [0.0, 0.0])))
    for r in recs:
        if r["layer"] is None:
            continue
        t = TYPES.get(r["layer"], "other")
        x = lt[t]
        x["layers"].add(r["layer"])
        x["mean"] += r["mean"]
        x["ideal"] += r["ideal"]
        x["cat"][r["cat"]][0] += r["mean"]
        x["cat"][r["cat"]][1] += r["ideal"]
    P("\n## Per layer type (per layer: measured us / ideal us)\n")
    P(
        "| layer type | layers | measured us per layer | ideal us per layer | above ideal us | top categories (measured / ideal us per layer) |"
    )
    P("|---|---|---|---|---|---|")
    for t, x in sorted(lt.items()):
        n = len(x["layers"])
        top = sorted(x["cat"].items(), key=lambda kv: -(kv[1][0] - kv[1][1]))[:6]
        P(
            f"| {t} | {n} | {x['mean']/n*1e6:.0f} | {x['ideal']/n*1e6:.0f} | {(x['mean']-x['ideal'])/n*1e6:.0f} | "
            + "; ".join(f"{k} {v[0]/n*1e6:.0f}/{v[1]/n*1e6:.0f}" for k, v in top)
            + " |"
        )
    # non-layer phases
    nl = defaultdict(lambda: [0, 0.0, 0.0])
    for r in recs:
        if r["layer"] is None:
            x = nl[r["phase"]]
            x[0] += 1
            x[1] += r["mean"]
            x[2] += r["ideal"]
    P("\n## Non-layer phases of the round (measured ms / ideal ms)\n")
    P("| phase | ops | measured ms | ideal ms |")
    P("|---|---|---|---|")
    for k, (n, m_, i_) in sorted(nl.items(), key=lambda kv: -kv[1][1]):
        P(f"| {k} | {n} | {m_*1e3:.3f} | {i_*1e3:.3f} |")
    # top ops
    P("\n## The 40 slowest single ops (mean chip)\n")
    P(
        "| op | layer | measured us | slowest us | fastest us | DRAM MB | fabric MB | ideal us | meas/ideal | cause | note |"
    )
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(recs, key=lambda r: -r["mean"])[:40]:
        P(
            f"| {r['cat']}: {r['name'][:50]} | {r['layer']} | {r['mean']*1e6:.1f} | {r['slow']*1e6:.1f} | {r['fast']*1e6:.1f} | {r['dram']/1e6:.2f} | {r['fab']/1e6:.3f} | {r['ideal']*1e6:.1f} | {r['mean']/max(r['ideal'],1e-9):.1f} | {r['cls']} | {r['note']} |"
        )
    open(arg("--out", os.path.join(run, "BUDGET_raw.md")), "w").write("\n".join(out) + "\n")
    if "--csv" in sys.argv:
        with open(arg("--csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
            w.writeheader()
            w.writerows(recs)
    print("\n".join(out[:12]))


if __name__ == "__main__":
    main()
