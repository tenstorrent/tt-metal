# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Render paired sampled frames and summarize measured Wan results."""

import argparse
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

VARIANTS = ("stock", "D", "C", "B", "E", "F", "G")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    args = parser.parse_args()
    manifests = {v: json.loads((args.suite / v / "manifest.json").read_text()) for v in VARIANTS}
    assert all(m["status"] == "completed" and len(m["results"]) == 2 for m in manifests.values())
    scores = json.loads((args.suite / "clip-scores.json").read_text())
    captures = json.loads((args.suite / "real-qkv.json").read_text())
    assert captures["status"] == "completed"
    lines = [
        "# Wan2.2 480p attention comparison",
        "",
        "14/14 videos: two prompts, seed 42, 832x480, 81 frames, 40 steps, CFG 4/3. Eight Blackhole chips, SP4/TP2. All choices reuse the same converted weights. No attention fallback.",
        "",
        "Stock uses ring attention. Frontier choices use prepared-format KV all-gather then local-Q attention, with native masking of the eight padded tokens. Cross-attention, projections, FFNs, text encoder, scheduler and VAE are unchanged.",
        "",
        "## Generation time and CLIP",
        "",
        "Generation time is warmed untraced pipeline wall time, including expert reloads but excluding initial setup, pilot diagnostics and video encoding. CLIP is raw OpenAI ViT-B/32 cosine averaged across eight uncompressed sampled frames; it is not a temporal-quality or reference-fidelity metric. Two videos per choice are exploratory.",
        "",
        "| Choice | Butterfly seconds | Human seconds | Butterfly CLIP | Human CLIP |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant, m in manifests.items():
        results = {r["prompt_id"]: r for r in m["results"]}
        clip = {r["prompt_id"]: r["mean_clip"] for r in scores["rows"] if r["variant"] == variant}
        lines.append(
            f"| {variant} | {results[0]['pipeline_seconds']:.2f} | {results[1]['pipeline_seconds']:.2f} | {clip[0]:.5f} | {clip[1]:.5f} |"
        )
    total = sum(r["pipeline_seconds"] for m in manifests.values() for r in m["results"])
    lines += [
        "",
        f"Total measured video generation: {total / 60:.1f} minutes, excluding setup, pilots, scoring and qualification.",
        "",
        "## Full-block timings",
        "",
        "Blocking mesh trace replays, five warmups and 15 samples per block. Includes attention preprocessing and communication. Inputs come from each variant's own two-step pilot, not a shared captured input; short warmup and dynamic clocks limit small-difference interpretation. These are full-block times, not SDPA FLOP utilization.",
        "",
        "| Choice | High-noise block 0 ms | High-noise block 20 ms | Low-noise block 0 ms | Low-noise block 20 ms | Replay exact |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for variant, m in manifests.items():
        entries = [m["block_bench"][f"expert{e}.block{b}"] for e in (0, 1) for b in (0, 20)]
        times = " | ".join(f"{r['median_ms']:.3f}" for r in entries)
        lines.append(f"| {variant} | {times} | {all(r['replay_exact'] for r in entries)} |")
    lines += [
        "",
        "## Identical real-QKV accuracy",
        "",
        "Four D-pilot captures, four selected global heads and 256 selected query rows, all 32,760 valid keys. Reference is FP64 attention on the original BF16 Q/K/V. These are operator errors, not video errors. This bounded sample does not qualify all denoising timesteps or the human prompt.",
        "",
        "| Choice | Min–max L2 | Min PCC |",
        "|---|---:|---:|",
    ]
    for variant in VARIANTS[1:]:
        rows = [r for r in captures["rows"] if r["variant"] == variant]
        lines.append(
            f"| {variant} | {min(r['l2_pct'] for r in rows):.3f}%–{max(r['l2_pct'] for r in rows):.3f}% | {min(r['pcc'] for r in rows):.6f} |"
        )
    lines += ["", "## Paired outputs", ""]
    for prompt_id in (0, 1):
        fig, axes = plt.subplots(7, 3, figsize=(12, 16), constrained_layout=True)
        for row, variant in enumerate(VARIANTS):
            result = next(r for r in manifests[variant]["results"] if r["prompt_id"] == prompt_id)
            for col, sample in enumerate((0, 4, 7)):
                entry = result["sampled_frames"][sample]
                with Image.open(args.suite / variant / entry["file"]) as image:
                    axes[row, col].imshow(image)
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
                if col == 0:
                    axes[row, col].set_ylabel(variant, fontsize=16, rotation=0, labelpad=25)
                if row == 0:
                    axes[row, col].set_title(f"Frame {entry['frame']}")
        fig.suptitle("Butterfly" if prompt_id == 0 else "Human subject: face and hands", fontsize=16)
        filename = f"prompt{prompt_id}-comparison.png"
        fig.savefig(args.suite / filename, dpi=120)
        plt.close(fig)
        lines.append(f"- [Prompt {prompt_id} sampled frames]({filename})")
        for variant in VARIANTS:
            lines.append(f"  - [{variant} video]({variant}/prompt{prompt_id}-seed42.mp4)")
    lines += [
        "",
        "## Integration qualification",
        "",
        "All six recipes passed the padded SP4/TP2 poison-tail test and exact trace replay. F initially failed because its BFP8 K unpack format was incorrectly retained when reading the BF16 mask palette. An explicit mask-format reconfiguration fixed this; the native exp formula is unchanged. The fix is gated to the padded adapter, preserving the prior unpadded FLUX path.",
        "",
        "Each non-stock manifest records Q/K/V formats for all 80 self-attention blocks. E/F transport BFP8_B KV, G transports BFP4_B KV, D/C/B transport BF16 KV; Q remains BF16. F's FP32 destination is independent of its input storage formats.",
        "",
        "Converted-weight misses are forbidden by the run harness. Source provenance is in run-metadata.json; individual manifests retain frame/video hashes, timings, captures and cache-load records.",
        "",
    ]
    (args.suite / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
