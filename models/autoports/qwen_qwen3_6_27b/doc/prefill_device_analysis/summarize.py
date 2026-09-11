# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Summarize this directory's measurements without opening a TT device."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).parent / "artifacts"
    names = [
        "baseline",
        "outer",
        "mac",
        "reuse12",
        "packed",
        "ccl_bfp8",
        "ccl_direct",
        "single_silu",
        "chunk128",
        "hillis",
        "kda_scan",
        "kda_scan_fp32",
        "traced_keepalive",
        "traced_keepalive_mac",
    ]
    rows = []
    for name in names:
        path = root / f"{name}_s128.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        rows.append(
            {
                "candidate": name,
                "linear_ms": result["layer_median_ms"]["0"],
                "full_attention_ms": result["layer_median_ms"]["3"],
                "reduced_generator_ms": result["median_ms"],
                "layer_output_pcc": result.get("comparison", {}).get("layer_0", {}).get("pcc"),
                "recurrent_state_pcc": result.get("comparison", {}).get("cache_0_recurrent", {}).get("pcc"),
                "logits_top1_equal": result.get("logits_top1_equal"),
            }
        )
    (root / "candidate_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        print(
            f"{row['candidate']:24s} {row['linear_ms']:8.3f} ms linear   {row['reduced_generator_ms']:8.3f} ms generator"
        )

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    axes[0].barh(
        [r["candidate"] for r in rows],
        [r["linear_ms"] for r in rows],
        color=["#287d58" if r["candidate"] == "traced_keepalive" else "#637e9b" for r in rows],
    )
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Warmed linear-layer wall time (ms)")
    axes[0].set_title("S128, batch 1, real Qwen3.8 weights\nExperimental A/Bs; accuracy varies")
    axes[0].grid(axis="x", alpha=0.2)

    kernel, gaps = [], []
    labels = ["Linear attention", "Full attention"]
    for kind in ("linear", "full"):
        with (root / f"baseline_{kind}_perf.csv").open() as handle:
            data = list(csv.DictReader(handle))
        kernel.append(sum(float(r["Device Time"] or 0) for r in data) / 1000)
        gaps.append(sum(float(r["Op-to-Op Gap"] or 0) for r in data) / 1000)
    axes[1].bar(labels, kernel, label="Device kernels", color="#287d58")
    axes[1].bar(labels, gaps, bottom=kernel, label="Inter-op gaps", color="#e7a45b")
    axes[1].set_ylabel("Profiled time (ms)")
    axes[1].set_title("Baseline kernel/gap accounting\nProfiling increases wall time")
    axes[1].legend(frameon=False)
    fig.savefig(root / "prefill_analysis.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
