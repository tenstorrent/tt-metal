#!/usr/bin/env python3
"""Turn the tracy ops_perf_results CSV of tests/test_decode_tracy.py into op tables (REAL op names, no by-difference).

  python tracy_report.py <ops_perf_results_*.csv> --plan 1:2,8:2 [--layers 48] [--out DIR] [--ref-json earlier.json]

--plan  U:n_replays,... in the order the test replayed (TR_US with TR_WARM+TR_REPS replays each, flushed one by one): the trace replay sessions
        (METAL TRACE ID, METAL TRACE REPLAY SESSION ID) are ordered by their first GLOBAL CALL COUNT and assigned to the plan in order; the LAST replay
        of each U is analysed (the earlier ones only warm).
Structure of one decode trace per device (checked by assertion): [embedding ops] then per layer [LayerNorm(in) attention.. LayerNorm(post) ffn..], then
[final LayerNorm, lm_head matmul, max, argmax ...]. Layers are cut at the 1-core LayerNorm (rms_norm) ops (2*layers+1 of them). Layer kind from the config
hybrid_layer_pattern (0 = global) and layer 0 = dense. MoE stages are cut at the op codes MaskedBincount / Dispatch / UnifiedRoutedExpertFfn / Combine /
PostCombineReduce (anchors), so every stage is a list of real op names.
Per position the 32 devices are aggregated as mean and max; "sum of max" is sum over ops of the per-op max over devices (as in LOG E261007-1632-433).
Writes <out>/tracy_U<U>.json (all tables) and prints markdown tables."""
import argparse
import json
import os
import re
import sys
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd

K = "DEVICE KERNEL DURATION [ns]"
OO = "OP TO OP LATENCY [ns]"
BYTES = {
    "BFLOAT8_B": 1.0625,
    "BFLOAT4_B": 0.5625,
    "BFLOAT16": 2.0,
    "FLOAT32": 4.0,
    "UINT32": 4.0,
    "INT32": 4.0,
    "UINT16": 2.0,
}
DRAM_PEAK_GBS = 512.0
E_LOCAL, EXP_K, EXP_N = (
    8,
    4096,
    2048,
)  # experts per chip, hidden, intermediate (config check in LOG): w1,w2,w3 bfp8 per expert
GLOBAL_LAYERS = {
    0,
    5,
    11,
    17,
    23,
    29,
    35,
    41,
    47,
}  # config hybrid_layer_pattern == 0 (checked against /mnt/tt-data/MiMo-V2.6-Flash-RL/config.json)

USE = [
    "OP CODE", "GLOBAL CALL COUNT", "DEVICE ID", "CORE COUNT", K, OO, "METAL TRACE ID", "METAL TRACE REPLAY SESSION ID", "ATTRIBUTES",
    "INPUT_0_Y_PAD[LOGICAL]", "INPUT_0_X_PAD[LOGICAL]", "INPUT_0_DATATYPE", "INPUT_1_Y_PAD[LOGICAL]", "INPUT_1_X_PAD[LOGICAL]", "INPUT_1_DATATYPE",
    "OUTPUT_0_Y_PAD[LOGICAL]", "OUTPUT_0_X_PAD[LOGICAL]", "OUTPUT_0_DATATYPE",
]  # fmt: skip


def dim(v):
    return "" if pd.isna(v) else str(v).split("[")[0]


def load(path):
    parts = []
    for ch in pd.read_csv(path, usecols=USE, chunksize=200000, low_memory=False):
        parts.append(ch[ch["METAL TRACE ID"].notna()])
    d = pd.concat(parts, ignore_index=True)
    d["METAL TRACE ID"] = d["METAL TRACE ID"].astype(int)
    d["METAL TRACE REPLAY SESSION ID"] = d["METAL TRACE REPLAY SESSION ID"].astype(int)
    return d


def label(r):
    code = r["OP CODE"].replace("DeviceOperation", "")
    a = r["ATTRIBUTES"] if isinstance(r["ATTRIBUTES"], str) else ""
    if code == "BinaryNg":
        m = re.search(r"binary_op_type': 'BinaryOpType::(\w+)", a)
        code += "." + (m.group(1) if m else "?")
    elif code == "Unary":
        m = re.search(r"op_type=UnaryOpType::(\w+)", a)
        code += "." + (m.group(1) if m else "?")
    i0 = f"{dim(r['INPUT_0_Y_PAD[LOGICAL]'])}x{dim(r['INPUT_0_X_PAD[LOGICAL]'])}"
    i1 = f"{dim(r['INPUT_1_Y_PAD[LOGICAL]'])}x{dim(r['INPUT_1_X_PAD[LOGICAL]'])}"
    o0 = f"{dim(r['OUTPUT_0_Y_PAD[LOGICAL]'])}x{dim(r['OUTPUT_0_X_PAD[LOGICAL]'])}"
    s = f"{code} {i0}"
    if i1 != "x":
        s += f"*{i1}:{str(r['INPUT_1_DATATYPE']).replace('BFLOAT', 'bf').replace('FLOAT32', 'fp32')}"
    return s + f"->{o0}"


def sessions(d):
    g = d.groupby(["METAL TRACE ID", "METAL TRACE REPLAY SESSION ID"])["GLOBAL CALL COUNT"].min().sort_values()
    return list(g.index)


def per_op(dsel):
    """-> list of op dicts (position order, aggregated over devices) for one replay session"""
    devs = {}
    for dev, g in dsel.groupby("DEVICE ID"):
        devs[dev] = g.sort_values("GLOBAL CALL COUNT").reset_index(drop=True)
    n = {len(v) for v in devs.values()}
    assert len(n) == 1, f"devices disagree on op count: {sorted(n)}"
    n = n.pop()
    kern = np.stack([v[K].fillna(0).to_numpy(float) for v in devs.values()])
    oo = np.stack([v[OO].fillna(0).to_numpy(float) for v in devs.values()])
    ref = next(iter(devs.values()))
    ops = []
    for i in range(n):
        r = ref.iloc[i]
        ops.append(
            dict(
                i=i,
                code=r["OP CODE"].replace("DeviceOperation", ""),
                label=label(r),
                cores=int(r["CORE COUNT"]) if not pd.isna(r["CORE COUNT"]) else 0,
                mean=float(kern[:, i].mean()),
                max=float(kern[:, i].max()),
                min=float(kern[:, i].min()),
                oo=float(oo[:, i].mean()),
                in1_y=dim(r["INPUT_1_Y_PAD[LOGICAL]"]),
                in1_x=dim(r["INPUT_1_X_PAD[LOGICAL]"]),
                in1_dt=str(r["INPUT_1_DATATYPE"]),
            )
        )
    return ops, len(devs)


def split_layers(ops, nl):
    ln = [o["i"] for o in ops if o["code"] == "LayerNorm"]
    assert len(ln) == 2 * nl + 1, f"expected {2*nl+1} LayerNorm ops, got {len(ln)}"
    layers = []
    for L in range(nl):
        a, b, c = ln[2 * L], ln[2 * L + 1], ln[2 * L + 2]
        layers.append(dict(L=L, norm_in=ops[a], attn=ops[a + 1 : b], norm_post=ops[b], ffn=ops[b + 1 : c]))
    return ops[: ln[0]], layers, ops[ln[-1] :]


def moe_stages(ffn):
    """real-op-name stages of a MoE layer FFN part (ops after the post-attention norm)"""
    idx = {}
    for j, o in enumerate(ffn):
        idx.setdefault(o["code"], j)
    need = ["MaskedBincount", "Dispatch", "UnifiedRoutedExpertFfn", "Combine", "PostCombineReduce"]
    assert all(c in idx for c in need), {c: idx.get(c) for c in need}
    b = [
        0,
        idx["MaskedBincount"],
        idx["Dispatch"],
        idx["UnifiedRoutedExpertFfn"],
        idx["Combine"],
        idx["PostCombineReduce"],
        idx["PostCombineReduce"] + 1,
        len(ffn),
    ]
    names = [
        "router+spread",
        "counts (bincount+allgather+cumsum)",
        "dispatch",
        "experts (unified_routed_expert_ffn)",
        "combine",
        "post_combine_reduce",
        "moe all-reduce+residual",
    ]
    return OrderedDict((n, ffn[b[k] : b[k + 1]]) for k, n in enumerate(names))


def sm(ops, key="mean"):
    return sum(o[key] for o in ops) / 1e6


def analyse(ops, nl, ndev):
    emb, layers, head = split_layers(ops, nl)
    for l in layers:
        l["kind"] = ("dense" if l["L"] == 0 else "moe") + "_" + ("global" if l["L"] in GLOBAL_LAYERS else "swa")
    out = dict(n_ops=len(ops), n_devices=ndev, layers=nl)
    out["kernel_ms_mean"], out["kernel_ms_sumofmax"], out["op2op_ms"] = sm(ops), sm(ops, "max"), sm(ops[1:], "oo")
    # buckets (E261007-1632-433 format): kernel ms over all layers, mean [sum of max]
    B = OrderedDict()

    def add(name, ops_):
        m, x, n = B.get(name, (0.0, 0.0, 0))
        B[name] = (m + sm(ops_), x + sm(ops_, "max"), n + len(ops_))

    for l in layers:
        add("ATTENTION (qkv..o_proj..all_reduce..residual)", l["attn"])
        add("RMSNORMS (2 per layer, 1 core)", [l["norm_in"], l["norm_post"]])
        if l["kind"].startswith("dense"):
            add("dense L0 FFN", l["ffn"])
        else:
            for n, o in moe_stages(l["ffn"]).items():
                add("MoE: " + n, o)
    add("embedding", emb)
    add("head: final norm+lm_head+max+argmax", head)
    out["buckets"] = {k: dict(mean_ms=v[0], sumofmax_ms=v[1], n_ops=v[2]) for k, v in B.items()}
    moe = sum(v[0] for k, v in B.items() if k.startswith("MoE:"))
    moe_x = sum(v[1] for k, v in B.items() if k.startswith("MoE:"))
    out["moe_ffn_total"] = dict(mean_ms=moe, sumofmax_ms=moe_x)
    # per layer type
    kinds = defaultdict(list)
    for l in layers:
        kinds[l["kind"]].append(l)
    LT = {}
    for k, ls in kinds.items():
        n = len(ls)
        row = dict(n_layers=n, layers=[l["L"] for l in ls][:12])
        row["norms_ms"] = sum(sm([l["norm_in"], l["norm_post"]]) for l in ls) / n
        row["attention_ms"] = sum(sm(l["attn"]) for l in ls) / n
        row["attention_max_ms"] = sum(sm(l["attn"], "max") for l in ls) / n
        row["ffn_ms"] = sum(sm(l["ffn"]) for l in ls) / n
        row["ffn_max_ms"] = sum(sm(l["ffn"], "max") for l in ls) / n
        row["n_attn_ops"] = len(ls[0]["attn"])
        row["n_ffn_ops"] = len(ls[0]["ffn"])
        if k.startswith("moe"):
            st = [moe_stages(l["ffn"]) for l in ls]
            row["stages_ms"] = {
                s: dict(
                    mean=sum(sm(x[s]) for x in st) / n, max=sum(sm(x[s], "max") for x in st) / n, n_ops=len(st[0][s])
                )
                for s in st[0]
            }
        LT[k] = row
    out["layer_types"] = LT
    # per op-name table over all layers (and head/embedding), count, total, mean per op
    T = defaultdict(lambda: dict(n=0, mean_ns=0.0, max_ns=0.0))
    for o in ops:
        t = T[o["label"]]
        t["n"] += 1
        t["mean_ns"] += o["mean"]
        t["max_ns"] += o["max"]
    out["by_name"] = sorted(
        (
            {
                "op": k,
                "count": v["n"],
                "total_ms": v["mean_ns"] / 1e6,
                "mean_us": v["mean_ns"] / v["n"] / 1e3,
                "total_max_ms": v["max_ns"] / 1e6,
            }
            for k, v in T.items()
        ),
        key=lambda r: -r["total_ms"],
    )
    # DRAM bandwidth of matmul-type ops (weights bytes per chip / kernel time)
    bw = []

    def bwrow(name, o, wbytes):
        bw.append(
            dict(
                op=name,
                kernel_us_mean=o["mean"] / 1e3,
                kernel_us_max=o["max"] / 1e3,
                weight_MB=wbytes / 1e6,
                GBps_mean=wbytes / o["mean"],
                pct_peak=100 * wbytes / o["mean"] / DRAM_PEAK_GBS,
            )
        )

    def wb(o):
        return float(o["in1_y"] or 0) * float(o["in1_x"] or 0) * BYTES.get(o["in1_dt"], 2.0)

    seen = set()
    for l in layers:
        for part, grp in (("attn", l["attn"]), ("ffn", l["ffn"])):
            for o in grp:
                if o["code"] == "Matmul" and o["in1_y"] and wb(o) > 1e6 and (l["kind"], o["label"]) not in seen:
                    seen.add((l["kind"], o["label"]))
                    bwrow(f"{l['kind']} L{l['L']} {o['label']}", o, wb(o))
                if o["code"] == "UnifiedRoutedExpertFfn" and ("exp", l["kind"]) not in seen:
                    seen.add(("exp", l["kind"]))
                    one = 3 * EXP_K * EXP_N * BYTES["BFLOAT8_B"]  # w1+w2+w3 of ONE expert (bfp8)
                    bwrow(
                        f"{l['kind']} L{l['L']} experts (all {E_LOCAL} local experts' weights = upper bound; only experts that got tokens are read)",
                        o,
                        E_LOCAL * one,
                    )
                    bw[-1]["experts_readable_at_peak"] = (
                        o["mean"] * DRAM_PEAK_GBS / one
                    )  # how many experts' weights fit in the kernel time at 512 GB/s
    for o in head:
        if o["code"] == "Matmul":
            bwrow("lm_head " + o["label"], o, wb(o))
    out["dram_bw"] = bw
    out["per_op_first_layers"] = {
        f"L{l['L']}": [
            dict(
                i=o["i"],
                label=o["label"],
                cores=o["cores"],
                mean_us=o["mean"] / 1e3,
                max_us=o["max"] / 1e3,
                oo_us=o["oo"] / 1e3,
            )
            for o in [l["norm_in"]] + l["attn"] + [l["norm_post"]] + l["ffn"]
        ]
        for l in layers
        if l["L"] in (0, 1, 5)
    }
    out["head_ops"] = [dict(label=o["label"], mean_us=o["mean"] / 1e3, max_us=o["max"] / 1e3) for o in head]
    return out


def md(out, U):
    p = print
    p(f"\n## U={U}: {out['n_ops']} ops/device, {out['n_devices']} devices, {out['layers']} layers")
    p(
        f"kernel sum (mean over devices) {out['kernel_ms_mean']:.2f} ms; sum of per-op max over devices {out['kernel_ms_sumofmax']:.2f} ms; op-to-op gaps inside the trace {out['op2op_ms']:.2f} ms\n"
    )
    p("### Buckets (kernel ms over all layers: mean [sum of max]) ranked")
    for k, v in sorted(out["buckets"].items(), key=lambda kv: -kv[1]["mean_ms"]):
        p(f"- {k}: {v['mean_ms']:.2f} [{v['sumofmax_ms']:.2f}] ms, {v['n_ops']} ops")
    p(f"- MoE FFN total: {out['moe_ffn_total']['mean_ms']:.2f} [{out['moe_ffn_total']['sumofmax_ms']:.2f}] ms")
    p("\n### Per layer type (ms per layer, mean [max-sum])")
    for k, r in out["layer_types"].items():
        p(
            f"- {k} (n={r['n_layers']}): norms {r['norms_ms']:.3f}, attention {r['attention_ms']:.3f} [{r['attention_max_ms']:.3f}] ({r['n_attn_ops']} ops), ffn {r['ffn_ms']:.3f} [{r['ffn_max_ms']:.3f}] ({r['n_ffn_ops']} ops)"
        )
        for s, v in r.get("stages_ms", {}).items():
            p(f"    - {s}: {v['mean']*1e3:.1f} [{v['max']*1e3:.1f}] us, {v['n_ops']} ops")
    p("\n### Top ops by name (all layers; count, total ms mean [max], mean us)")
    for r in out["by_name"][:40]:
        p(
            f"- {r['op']}: n={r['count']} total {r['total_ms']:.2f} [{r['total_max_ms']:.2f}] ms mean {r['mean_us']:.1f} us"
        )
    p("\n### DRAM bandwidth of matmul-type ops (per chip, weights bytes / kernel time, peak 512 GB/s)")
    for r in out["dram_bw"]:
        p(
            f"- {r['op']}: {r['kernel_us_mean']:.1f} us (max {r['kernel_us_max']:.1f}), {r['weight_MB']:.1f} MB, {r['GBps_mean']:.0f} GB/s = {r['pct_peak']:.0f}% of peak"
            + (
                f"; kernel time = {r['experts_readable_at_peak']:.1f} experts' weights at peak BW"
                if "experts_readable_at_peak" in r
                else ""
            )
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--plan", default="1:2,8:2")
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_dir = a.out or os.path.dirname(os.path.abspath(a.csv))
    os.makedirs(out_dir, exist_ok=True)
    plan = [tuple(int(x) for x in s.split(":")) for s in a.plan.split(",")]
    d = load(a.csv)
    ss = sessions(d)
    need = sum(n for _, n in plan)
    print(f"trace replay sessions in csv: {len(ss)} (plan {plan}, need {need}): {ss}", file=sys.stderr)
    assert len(ss) == need, "session count != plan"
    k = 0
    for U, n in plan:
        grp = ss[k : k + n]
        k += n
        res = []
        for tid, sid in grp:
            sel = d[(d["METAL TRACE ID"] == tid) & (d["METAL TRACE REPLAY SESSION ID"] == sid)]
            ops, ndev = per_op(sel)
            res.append((ops, ndev))
        ops, ndev = res[-1]
        out = analyse(ops, a.layers, ndev)
        out["U"] = U
        out["replays"] = [
            dict(session=list(g), kernel_ms_mean=sm(r[0]), kernel_ms_sumofmax=sm(r[0], "max"), n_ops=len(r[0]))
            for g, r in zip(grp, res)
        ]
        json.dump(out, open(os.path.join(out_dir, f"tracy_U{U}.json"), "w"), indent=1)
        md(out, U)


if __name__ == "__main__":
    main()
