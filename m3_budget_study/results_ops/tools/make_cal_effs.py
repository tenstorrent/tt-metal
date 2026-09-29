#!/usr/bin/env python3
"""CAL.effs[mesh] from our zone profiles, in sim_core zoneEff form (chip-mean zone ms, latency floor, wave factor).

  python3 make_cal_effs.py --mesh 2x4 --W 4096                 # -> JSON on stdout (reproduces cal_effs_ours_2x4_w4096.json)
  python3 make_cal_effs.py --mesh 4x2 --W 8192 --native        # (4,2) with ag_kv = ag_kv_native_est (no per-head copies)
  python3 make_cal_effs.py --mesh 4x2 --W 4096 --out FILE --pipe-out FILE

Condition: single request at h=139264 (run_id *_w{W}_h141312_{input}), moe = mean of sparse layers 3-6, dense = layer 1.
eff = roof * wave / (mean ms - floor), clamped to [0.003, 1]; roof from roofline_ops.js --segments W:139264 --idx bf8.
"""
import argparse, csv, glob, json, math, os, subprocess, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
CCL = {"norm_ag", "attn_rs", "dispatch", "combine", "ag_kv", "ag_idx", "kv_a2a", "shared", "moe_reduce", "dense_mlp"}
MOE = [
    "norm_ag",
    "qkv",
    "idx_branch",
    "o_proj",
    "attn_rs",
    "shared",
    "router",
    "dispatch",
    "experts",
    "combine",
    "moe_reduce",
    "ag_kv",
    "ag_idx",
    "indexer",
    "sparse",
]
DEN = ["norm_ag", "qkv", "o_proj", "attn_rs", "dense_mlp"]
WAVE = {"sparse", "indexer", "ring_c"}
# Pavlo's pipeline fit (calibrate(calib_data.json).pipe) and his [2,4] dense zone ring_c: pipe.ringC / zone ring_c
PAVLO_RINGC, PAVLO_RINGSCAN, PAVLO_ZONE_RINGC_2X4 = 0.30248237322064786, 0.014013203607732828, 0.3575944448870825


def lat(op):
    return (0.04 if op in CCL else 0.01) * (2 if op == "norm_ag" else 1)


def wf(n, sp, tp):
    u = max(1, math.ceil(n / sp / 32)) * (64 // tp)
    return math.ceil(u / 110) * 110 / u


def node():
    return (
        os.environ.get("NODE") or sorted(glob.glob(os.path.expanduser("~/.vscode-server/cli/servers/*/server/node")))[0]
    )


def fit(roof, mean, op, w=1.0):
    return min(1.0, max(0.003, roof * w / max(1e-3, mean - lat(op))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument(
        "--inputs", default=None, help="comma list; default prose,code for 2x4, prose for 4x2 (only input profiled)"
    )
    ap.add_argument("--native", action="store_true", help="4x2: ag_kv = ag_kv_native_est, layer minus the head copies")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--detail", action="store_true")
    a = ap.parse_args()
    sp, tp = map(int, a.mesh.split("x"))
    inputs = (a.inputs or ("prose,code" if a.mesh == "2x4" else "prose")).split(",")
    csvf = a.csv or os.path.join(R, "per_op.csv" if a.mesh == "2x4" else f"per_op_{a.mesh}.csv")
    pre = "p0a" if a.mesh == "2x4" else "p" + a.mesh
    runs = {f"{pre}_w{a.W}_h141312_{i}" for i in inputs}
    acc = {"moe": defaultdict(list), "dense": defaultdict(list)}
    with open(csvf) as f:
        for r in csv.DictReader(f):
            if r["run_id"] not in runs:
                continue
            L = int(r["layer"])
            k = "moe" if L in (3, 4, 5, 6) else "dense" if L == 1 else None
            if k:
                acc[k][r["op"]].append(float(r["mean_ms"]))
    xs = {op: sum(v) / len(v) for op, v in acc["moe"].items()}
    xd = {op: sum(v) / len(v) for op, v in acc["dense"].items()}
    if not xs or not xd:
        sys.exit(f"no rows for {sorted(runs)} in {csvf}")
    if a.native:
        # ag_kv without the harness per-head slice/concat copies; the copies are also inside layer_total
        xs["layer_total"] -= xs["ag_kv"] - xs["ag_kv_native_est"]
        xs["ag_kv"] = xs["ag_kv_native_est"]
    rf = json.loads(
        subprocess.run(
            [
                node(),
                os.path.join(HERE, "roofline_ops.js"),
                "--mesh",
                a.mesh,
                "--segments",
                f"{a.W}:139264",
                "--layer",
                "both",
                "--idx",
                "bf8",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    wr = wf(a.W, sp, tp) / wf(5120, sp, tp)
    moe, den, det = {}, {}, {}
    for op in MOE:
        moe[op] = fit(rf["sparse"][op], xs[op], op, wr if op in WAVE else 1.0)
        det["moe." + op] = (xs[op], rf["sparse"][op], moe[op])
    listed = sum(xs[op] for op in MOE)
    misc_ms = xs["layer_total"] - listed
    moe["misc"] = fit(rf["sparse"]["misc"], max(0.05, misc_ms), "misc")
    det["moe.misc"] = (misc_ms, rf["sparse"]["misc"], moe["misc"])
    moe["kv_a2a"] = moe["dispatch"]
    for op in DEN:
        den[op] = fit(rf["dense"][op], xd[op], op)
        det["dense." + op] = (xd[op], rf["dense"][op], den[op])
    dmisc = xd["layer_total"] - sum(xd[op] for op in DEN) - xd["ring"]
    den["misc"] = fit(rf["dense"]["misc"], max(0.05, dmisc), "misc")
    det["dense.misc"] = (dmisc, rf["dense"]["misc"], den["misc"])
    den["ring_c"] = fit(rf["dense"]["ring_c"], xd["ring"], "ring", wr)
    det["dense.ring_c"] = (xd["ring"], rf["dense"]["ring_c"], den["ring_c"])
    den["ring_scan"] = 0.015
    den["kv_a2a"] = moe["dispatch"]
    about = [
        f"CAL.effs['{a.mesh}'] measured on our zone profiles: one ({sp},{tp}) stage, SP={sp} TP={tp} EP=8, layers 0-6 contiguous (real routing), 1D fabric, bf4 experts, M3_MOE_W_NDSHARD=1 M3_MOE_HYBRID_THRESHOLD=128, v1 dispatch/combine, bf8 index_k cache.",
        f"Condition: W={a.W} single request at h=139264 (the 141312 request aligned down to whole chunks); inputs {'+'.join(inputs)}{' pooled' if len(inputs) > 1 else ''}. moe = mean of sparse layers 3-6, dense = layer 1. Source {os.path.basename(csvf)}.",
        "Formula = sim_core zoneEff: eff = roof / (zone ms - latency floor), clamped to [0.003, 1]. zone ms = MEAN over the 8 chips. Floor 0.04 ms for CCL ops, 0.01 ms otherwise; norm_ag = both all-gathers (roof x2, floor x2).",
        f"Roofline = sim_core roofTok/roofSeg via tools/roofline_ops.js --mesh {a.mesh} --segments {a.W}:139264 --idx bf8, imbalance 1.2 in the experts FLOPs, Tr = W.",
        f"sparse, indexer, ring_c are multiplied by waveFactor(W)/waveFactor(5120) = {wr:.4f}, so that layerMs at T=W reproduces the measured zone.",
        "misc = layer zone mean - every mapped op. kv_a2a = dispatch. ring_scan 0.015 is Pavlo's placeholder; pipelines use CAL.pipe.ringC (x effs[mesh].dense.ring_c / effs['2x4'].dense.ring_c) and CAL.pipe.ringScan.",
        "With opEff = 0 the sim uses min(eff, target): values above target have no effect. Generated by tools/make_cal_effs.py.",
    ]
    if a.native:
        about.append(
            "NATIVE variant: ag_kv = ag_kv_native_est (ag_kv minus the harness per-head slice/concat copies), layer total reduced by the same amount; estimates a native multi-head gather (kv_4x2_status.md step b)."
        )
    elif a.mesh != "2x4":
        about.append(
            "AS MEASURED: ag_kv includes the harness per-head slice/concat copies (tools/profile_4x2.py workaround)."
        )
    rnd = lambda d: {k: round(v, 6) for k, v in d.items()}
    out = {"about": about, a.mesh: {"moe": rnd(moe), "dense": rnd(den)}}
    if a.detail:
        out["detail"] = {
            k: {"mean_ms": round(m, 4), "roof_ms": round(r, 4), "eff": round(e, 4)} for k, (m, r, e) in det.items()
        }
        out["detail"]["layer_mean_sparse_ms"] = round(xs["layer_total"], 4)
        out["detail"]["layer_mean_dense_ms"] = round(xd["layer_total"], 4)
        out["detail"]["pipe_ringC_if_2x4"] = round(PAVLO_RINGC * den["ring_c"] / PAVLO_ZONE_RINGC_2X4, 6)
    s = json.dumps(out, indent=1)
    if a.out:
        open(a.out, "w").write(s + "\n")
    else:
        print(s)


if __name__ == "__main__":
    main()
