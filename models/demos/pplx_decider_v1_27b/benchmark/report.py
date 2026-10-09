# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-11 tables: accuracy vs model card, per-bucket latency, throughput, roofline.

Reads the frozen subsets manifest, the device run (``run_benchmark.py``) and the per-bucket harness
output (``tests/perf/test_model_perf.py`` -> ``model_perf.json``); writes ``results.json`` and
``predictions.csv`` to ``--doc`` and prints the Markdown tables used in ``doc/benchmark/REPORT.md``.

Roofline accounting (prefill, batch 1, per executed bucket length S; the device computes the whole
right-padded bucket):

- linear FLOPs = 2 x P_text_linear x S, P_text_linear = every projection weight of the 64 text
  layers (DeltaNet in_proj_qkv/z/a/b + out_proj, attention q/k/v/o, MLP gate/up/down), counted from
  the snapshot safetensors headers. Vision weights excluded (text rows); the embedding is a gather
  (0 FLOPs); the readout (255 x 5120) runs on the last token only (2 x 255 x 5120).
- full-attention FLOPs = 16 layers x 2 x H x d x S(S+1) (QK^T and PV over the causal triangle,
  H = 24, d = 256).
- DeltaNet FLOPs (approximate, chunked form, C = 64) = 48 layers x S x Hv x (6 dk dv + 2 C (2 dk + dv))
  (state read/update/output 6 dk dv; intra-chunk K K^T, Q K^T and A V), Hv = 48, dk = dv = 128;
  plus the causal conv1d 2 x 4 x 10240 per token per layer. Norms, gates and other elementwise ops
  are not counted.
- weight bytes: BFP8_B (1088 B per 1024-element tile) for every projection. The decoder runs
  ``prefill_chunk`` = 2048-token chunks and each chunk reads every layer's weights, so streamed
  weight bytes = ceil(S / 2048) x unique bytes (a lower bound: one DRAM read per chunk).
- peak: 359.4 TFLOP/s = 130 Tensix (13 x 10 compute grid) x 1.35 GHz x 4096 FLOP/cycle / 2 for
  HiFi2 (tech_reports/GEMM_FLOPS/GEMM_FLOPS.md; matches tt-perf-report, e.g. 260.6 TFLOP/s = 72.5 %
  in doc/fused_decoder/perf/L3_full_S2048_perf_report.txt). DRAM 512 GB/s (p150a product spec,
  inferred). Every projection runs HiFi2 (selected_precision_config.json), so HiFi2 is the peak.

Usage::

    python -m models.demos.pplx_decider_v1_27b.benchmark.report --stage11 $STAGE11 \
        --doc models/demos/pplx_decider_v1_27b/doc/benchmark
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import struct
from pathlib import Path

from models.demos.pplx_decider_v1_27b.benchmark.datasets import CARD
from models.demos.pplx_decider_v1_27b.reference.decision_prompts import DEFAULT_SNAPSHOT

PEAK_TFLOPS_HIFI2 = 130 * 1.35e9 * 4096 / 2 / 1e12
DRAM_GBPS = 512.0
PREFILL_CHUNK = 2048
BFP8_BYTES_PER_PARAM = 1088 / 1024
N_FULL, N_LINEAR = 16, 48
HEADS, HEAD_DIM = 24, 256
HV, DK, DV, C = 48, 128, 128, 64
CONV_CH, CONV_K = 10240, 4
READOUT = (255, 5120)
NAMES = {"winogrande": "WinoGrande", "financial_phrasebank": "FinancialPhraseBank", "belebele": "Belebele"}


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def text_linear_params(snapshot: Path) -> dict:
    """Projection weights of the text layers by kind, from the safetensors headers."""
    shapes = {}
    for path in sorted(glob.glob(str(snapshot / "model-*.safetensors"))):
        with open(path, "rb") as fh:
            header = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        shapes.update({k: v["shape"] for k, v in header.items() if k != "__metadata__"})
    out = {"mlp": 0, "full_attention": 0, "linear_attention": 0, "embedding": 0, "vision": 0, "other": 0}
    for name, shape in shapes.items():
        n = math.prod(shape)
        if name.startswith("visual."):
            out["vision"] += n
        elif "embed_tokens" in name:
            out["embedding"] += n
        elif re.search(r"\.mlp\.(gate|up|down)_proj\.weight$", name):
            out["mlp"] += n
        elif re.search(r"\.self_attn\.[qkvo]_proj\.weight$", name):
            out["full_attention"] += n
        elif re.search(r"\.linear_attn\.(in_proj_\w+|out_proj)\.weight$", name):
            out["linear_attention"] += n
        else:
            out["other"] += n  # norms, conv1d, A_log, dt_bias (not matmul weights)
    out["text_linear"] = out["mlp"] + out["full_attention"] + out["linear_attention"]
    return out


def prefill_flops(S: int, p_linear: int) -> dict:
    linear = 2 * p_linear * S + 2 * READOUT[0] * READOUT[1]
    attention = N_FULL * 2 * HEADS * HEAD_DIM * S * (S + 1)
    deltanet = N_LINEAR * S * (HV * (6 * DK * DV + 2 * C * (2 * DK + DV)) + 2 * CONV_K * CONV_CH)
    return {"linear": linear, "attention": attention, "deltanet": deltanet, "total": linear + attention + deltanet}


def weight_bytes(S: int, p_linear: int) -> dict:
    unique = (p_linear + READOUT[0] * READOUT[1]) * BFP8_BYTES_PER_PARAM
    return {"unique": unique, "streamed": math.ceil(S / PREFILL_CHUNK) * unique}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage11", type=Path, required=True)
    parser.add_argument("--doc", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, default=Path(DEFAULT_SNAPSHOT))
    args = parser.parse_args()
    s11 = args.stage11
    manifest = json.loads((s11 / "subsets" / "manifest.json").read_text())
    preds = read_jsonl(s11 / "run" / "predictions_accuracy.jsonl")
    run = json.loads((s11 / "run" / "run_summary.json").read_text())
    perf = json.loads((s11 / "perf_buckets" / "model_perf.json").read_text())
    params = text_linear_params(args.snapshot)
    results = {"params": params, "peak_tflops_hifi2": PEAK_TFLOPS_HIFI2, "dram_gbps": DRAM_GBPS}

    # accuracy
    print("## Accuracy\n")
    print(
        "| benchmark | subset (config / split) | n | correct | TT accuracy | 95% CI (Wilson) | card pplx-decider-v1-27b | delta vs card | card Qwen3.8-27B base | rejected > 8192 |"
    )
    print("|---|---|---:|---:|---:|---|---:|---:|---:|---:|")
    acc = {}
    for name, m in manifest["benchmarks"].items():
        rows = [r for r in preds if r["benchmark"] == name]
        k, n = sum(r["correct"] for r in rows), len(rows)
        lo, hi = wilson(k, n)
        a = 100 * k / n
        card = CARD[name]
        acc[name] = {
            "n": n,
            "correct": k,
            "accuracy_pct": a,
            "ci95_pct": [100 * lo, 100 * hi],
            "card_pct": card["pplx_decider_v1_27b"],
            "delta_vs_card_pp": a - card["pplx_decider_v1_27b"],
            "card_base_pct": card["qwen3_8_27b_base"],
            "rejected": sum(r["rejected"] for r in rows),
            "mean_p_label": sum(r["probabilities"][r["label"]] for r in rows if not r["rejected"]) / n,
        }
        print(
            f"| {NAMES[name]} | {m['config']} / {m['split']} | {n} | {k} | {a:.2f}% | {100 * lo:.1f}-{100 * hi:.1f}% | "
            f"{card['pplx_decider_v1_27b']:.2f}% | {a - card['pplx_decider_v1_27b']:+.2f} pp | {card['qwen3_8_27b_base']:.2f}% | {acc[name]['rejected']} |"
        )
    results["accuracy"] = acc

    # FPB confusion (3 classes) for the protocol discussion
    fpb = [r for r in preds if r["benchmark"] == "financial_phrasebank"]
    labels = ["negative", "neutral", "positive"]
    conf = {t: {p: sum(r["label"] == t and r["prediction"] == p for r in fpb) for p in labels} for t in labels}
    results["fpb_confusion"] = conf
    print("\nFPB confusion (rows = label, cols = TT prediction):\n")
    print("| label \\ pred | " + " | ".join(labels) + " |\n|---|" + "---:|" * 3)
    for t in labels:
        print(f"| {t} | " + " | ".join(str(conf[t][p]) for p in labels) + " |")

    # per-bucket latency (stage-6 harness)
    print("\n## Per-bucket request latency (stage-6 harness, current HEAD)\n")
    print(
        "| bucket | prompt tokens | device forward burst ms | device forward sustained ms | request burst ms | request sustained ms | decisions/s burst | decisions/s sustained |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for b, r in perf["buckets"].items():
        rb, rs = r["request_burst"]["median_ms"], r["request_sustained"]["median_ms"]
        print(
            f"| {b} | {r['seq_len']} | {r['device_forward_burst']['median_ms']:.1f} | {r['device_forward_sustained']['median_ms']:.1f} | "
            f"{rb:.1f} | {rs:.1f} | {1e3 / rb:.2f} | {1e3 / rs:.2f} |"
        )

    # image
    print("\n## Image request latency\n")
    print(
        "| golden row | prompt tokens | patches | image tokens | bucket | request burst median ms | request sustained median ms | choice (expected) |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    for rid, r in run["image"].items():
        print(
            f"| {rid} | {r['seq_len']} | {r['patches']} | {r['image_tokens']} | {r['bucket']} | {r['burst']['median_ms']:.1f} | "
            f"{r['sustained']['median_ms']:.1f} | {r['answer']['choice']} ({r['expected']}) |"
        )

    # throughput
    tp, ap = run["throughput"], run["accuracy_pass"]
    print("\n## Mixed-workload throughput (600 benchmark prompts, back to back, batch 1)\n")
    print("| pass | order | requests | wall s | decisions/s | p50 ms | p90 ms | p99 ms | mean ms |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, p, order in (("throughput", tp, "seeded shuffle"), ("accuracy", ap, "file order (WG, FPB, BB)")):
        lat = p["latency"]
        n = p.get("served", lat["n"])
        print(
            f"| {label} | {order} | {n} | {p['wall_s']:.1f} | {p['decisions_per_s']:.3f} | {lat['p50_ms']:.1f} | "
            f"{lat['p90_ms']:.1f} | {lat['p99_ms']:.1f} | {lat['mean_ms']:.1f} |"
        )
    print("\nThroughput pass by bucket:\n")
    print("| bucket | requests | p50 ms | p90 ms | p99 ms | mean ms |\n|---:|---:|---:|---:|---:|---:|")
    for b, lat in tp["latency_by_bucket"].items():
        print(
            f"| {b} | {lat['n']} | {lat['p50_ms']:.1f} | {lat['p90_ms']:.1f} | {lat['p99_ms']:.1f} | {lat['mean_ms']:.1f} |"
        )
    print(
        f"\nDeterminism: same choice {tp['same_choice_as_accuracy_pass']}/{tp['served']}, max |prob diff| "
        f"{tp['max_abs_prob_diff_vs_accuracy_pass']:.3g} vs the accuracy pass."
    )

    # roofline
    p_lin = params["text_linear"]
    print("\n## Roofline (executed bucket length; burst timing)\n")
    print(
        f"Text projection params P = {p_lin / 1e9:.3f} B (MLP {params['mlp'] / 1e9:.3f} B, DeltaNet {params['linear_attention'] / 1e9:.3f} B, attention {params['full_attention'] / 1e9:.3f} B); embedding {params['embedding'] / 1e9:.3f} B (gather), vision {params['vision'] / 1e9:.3f} B (excluded).\n"
    )
    print(
        "| bucket | GFLOP (linear / attn / DeltaNet) | total GFLOP | device fwd burst ms | TFLOP/s | % HiFi2 peak | request burst ms | TFLOP/s (request) | % peak (request) | streamed weight GB | weight GB/s | % of 512 GB/s | useful tokens | useful % of executed FLOPs |"
    )
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    roof = {}
    for b, r in perf["buckets"].items():
        S = int(b)
        f = prefill_flops(S, p_lin)
        w = weight_bytes(S, p_lin)
        fwd, req = r["device_forward_burst"]["median_ms"] / 1e3, r["request_burst"]["median_ms"] / 1e3
        useful = prefill_flops(r["seq_len"], p_lin)["total"]
        roof[b] = {
            "flops": f,
            "weight_bytes": w,
            "device_forward_burst_s": fwd,
            "request_burst_s": req,
            "tflops_device": f["total"] / fwd / 1e12,
            "tflops_request": f["total"] / req / 1e12,
            "pct_peak_device": 100 * f["total"] / fwd / 1e12 / PEAK_TFLOPS_HIFI2,
            "pct_peak_request": 100 * f["total"] / req / 1e12 / PEAK_TFLOPS_HIFI2,
            "weight_gbps_device": w["streamed"] / fwd / 1e9,
            "useful_tokens": r["seq_len"],
            "useful_flops": useful,
            "sustained": {
                "tflops_device": f["total"] / (r["device_forward_sustained"]["median_ms"] / 1e3) / 1e12,
                "pct_peak_device": 100
                * f["total"]
                / (r["device_forward_sustained"]["median_ms"] / 1e3)
                / 1e12
                / PEAK_TFLOPS_HIFI2,
            },
        }
        x = roof[b]
        print(
            f"| {S} | {f['linear'] / 1e9:.0f} / {f['attention'] / 1e9:.1f} / {f['deltanet'] / 1e9:.1f} | {f['total'] / 1e9:.0f} | "
            f"{fwd * 1e3:.1f} | {x['tflops_device']:.1f} | {x['pct_peak_device']:.1f}% | {req * 1e3:.1f} | {x['tflops_request']:.1f} | "
            f"{x['pct_peak_request']:.1f}% | {w['streamed'] / 1e9:.1f} | {x['weight_gbps_device']:.0f} | "
            f"{100 * x['weight_gbps_device'] / DRAM_GBPS:.0f}% | {r['seq_len']} | {100 * useful / f['total']:.0f}% |"
        )
    print(
        "\nSustained (back-to-back) device forward: "
        + ", ".join(
            f"{b}: {x['sustained']['tflops_device']:.1f} TFLOP/s ({x['sustained']['pct_peak_device']:.1f}%)"
            for b, x in roof.items()
        )
    )
    results["roofline"] = roof

    # throughput-run roofline: executed (bucket) and useful (real tokens) FLOPs over the wall
    served = [r for r in read_jsonl(s11 / "run" / "predictions_throughput.jsonl") if not r["rejected"]]
    executed = sum(prefill_flops(r["bucket"], p_lin)["total"] for r in served)
    useful = sum(prefill_flops(r["seq_len"], p_lin)["total"] for r in served)
    results["throughput_roofline"] = {
        "executed_flops": executed,
        "useful_flops": useful,
        "wall_s": tp["wall_s"],
        "executed_tflops": executed / tp["wall_s"] / 1e12,
        "useful_tflops": useful / tp["wall_s"] / 1e12,
    }
    t = results["throughput_roofline"]
    print(
        f"\nThroughput run: executed {executed / 1e15:.2f} PFLOP over {tp['wall_s']:.1f} s = {t['executed_tflops']:.1f} TFLOP/s "
        f"({100 * t['executed_tflops'] / PEAK_TFLOPS_HIFI2:.1f}% of peak); useful (real tokens) {t['useful_tflops']:.1f} TFLOP/s "
        f"({100 * t['useful_tflops'] / PEAK_TFLOPS_HIFI2:.1f}%)."
    )

    results["perf_buckets"] = perf
    results["run_summary"] = run
    args.doc.mkdir(parents=True, exist_ok=True)
    (args.doc / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = ["id,benchmark,seq_len,bucket,count,label,prediction,p_label,p_prediction,correct,latency_ms"]
    for r in preds:
        lines.append(
            ",".join(
                json.dumps(str(v), ensure_ascii=False) if i in (0, 1, 5, 6) else str(v)
                for i, v in enumerate(
                    (
                        r["id"],
                        r["benchmark"],
                        r["seq_len"],
                        r["bucket"],
                        r["count"],
                        r["label"],
                        r["prediction"],
                        f"{r['probabilities'][r['label']]:.6f}",
                        f"{r['probabilities'][r['prediction']]:.6f}",
                        int(r["correct"]),
                        f"{r['latency_ms']:.1f}",
                    )
                )
            )
        )
    (args.doc / "predictions.csv").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
