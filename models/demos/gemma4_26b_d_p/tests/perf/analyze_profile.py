# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarize a test_profile_prefill ops_perf_results CSV.

Per profiled chunk (signpost-delimited) and per chip: embed / sliding layer / full layer device-kernel time,
op-category breakdown, op-to-op gaps, and the 30-layer extrapolation (1 embed + 25 sliding + 5 full).

    python models/demos/gemma4_26b_d_p/tests/perf/analyze_profile.py <csv> [--json out.json]
"""

import argparse
import collections
import csv
import json

CATEGORY = [
    ("MoE dispatch", ("DispatchDeviceOperation",)),
    ("MoE combine", ("CombineDeviceOperation",)),
    ("MoE experts", ("UnifiedRoutedExpertFfnDeviceOperation",)),
    ("MoE reduce+routing", ("PostCombineReduce", "MaskedBincount", "OffsetCumsum", "TopK", "Softmax")),
    ("SDPA", ("RingJointSDPA", "ScaledDotProductAttention", "SDPAOperation")),
    ("matmul", ("Matmul",)),
    ("norm", ("LayerNorm",)),
    ("rope", ("RotaryEmbedding",)),
    ("heads", ("NlpCreateHeads", "NLPConcatHeads")),
    ("kv write", ("UpdatePaddedKvCache",)),
    ("CCL", ("AllGather", "AllReduce", "ReduceScatter")),
    ("layout/typecast", ("Tilize", "Untilize", "Typecast", "Reshape", "Slice", "Concat", "Pad")),
    ("eltwise", ("BinaryNg", "Unary", "Where", "Embeddings")),
]


def category(op):
    for name, keys in CATEGORY:
        if any(k in op for k in keys):
            return name
    return "other"


def load(path):
    rows = list(csv.DictReader(open(path)))
    chunks = collections.OrderedDict()
    cur = None
    for x in rows:
        if x["OP TYPE"] == "signpost":
            c = x["OP CODE"]
            if c.endswith("_start"):
                cur = c[: -len("_start")]
                chunks[cur] = collections.defaultdict(list)
            elif c.endswith("_end"):
                cur = None
            continue
        if cur is None:
            continue
        dur = float(x["DEVICE KERNEL DURATION [ns]"] or 0) / 1e3
        gap = float(x["OP TO OP LATENCY [ns]"] or 0) / 1e3
        chunks[cur][int(x["DEVICE ID"])].append((x["OP CODE"], dur, gap))
    return chunks


def split_layers(ops):
    """[embed ops], [layer ops]... using one NlpCreateHeads per layer; a layer starts 2 ops before it (norm, qkv)."""
    heads = [i for i, (op, _, _) in enumerate(ops) if op.startswith("NlpCreateHeads")]
    starts = [h - 2 for h in heads] + [len(ops)]
    return ops[: starts[0]], [ops[starts[i] : starts[i + 1]] for i in range(len(heads))]


def summarize(path, n_sliding=25, n_full=5, chunk=4096):
    out = {}
    for name, per_dev in load(path).items():
        devs = {}
        for dev, ops in sorted(per_dev.items()):
            embed, (slid, full) = split_layers(ops)
            t = lambda seq: sum(d for _, d, _ in seq)
            g = lambda seq: sum(gp for _, _, gp in seq)
            cats = {k: collections.Counter() for k in ("sliding", "full")}
            for key, seq in (("sliding", slid), ("full", full)):
                for op, d, _ in seq:
                    cats[key][category(op)] += d
            model_ms = (t(embed) + n_sliding * t(slid) + n_full * t(full)) / 1e3
            gaps_ms = (g(embed) + n_sliding * g(slid) + n_full * g(full)) / 1e3
            devs[dev] = dict(embed_us=t(embed), sliding_us=t(slid), full_us=t(full), model_kernel_ms=model_ms, model_gap_ms=gaps_ms,
                             cats={k: dict(v.most_common()) for k, v in cats.items()})
        worst = max(devs.values(), key=lambda d: d["model_kernel_ms"])
        out[name] = dict(per_device=devs, worst=worst, tok_per_s=chunk / (worst["model_kernel_ms"] / 1e3))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--json")
    a = ap.parse_args()
    res = summarize(a.csv)
    for name, r in res.items():
        w = r["worst"]
        print(f"== {name}: slowest chip -> sliding layer {w['sliding_us'] / 1e3:.2f} ms, full layer {w['full_us'] / 1e3:.2f} ms, "
              f"embed {w['embed_us'] / 1e3:.2f} ms")
        print(f"   30-layer chunk: kernels {w['model_kernel_ms']:.1f} ms (+ op-to-op gaps {w['model_gap_ms']:.1f} ms) "
              f"-> {r['tok_per_s']:.0f} tok/s on kernel time")
        for lt in ("sliding", "full"):
            tot = sum(w["cats"][lt].values())
            parts = ", ".join(f"{k} {v / 1e3:.2f} ({100 * v / tot:.0f}%)" for k, v in list(w["cats"][lt].items())[:7])
            print(f"   {lt:7s} layer: {parts}")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
