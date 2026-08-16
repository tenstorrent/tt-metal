# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reduce a verified profile window to the handful of numbers the documents quote.

``rank_full_model_48.py`` prints a human ranking and ``tt-perf-report`` prints a
much larger one; neither is a shape a checker can read a single figure out of
without re-implementing the ranking. This writes the same figures to JSON, off
the *same* windowed CSV, so ``check_published_figures.py`` re-derives every
published profile number from an artifact field rather than from a table in a
markdown file.

The region split is the one ``rank_full_model_48.py`` defines and is imported
from it rather than restated, so the two cannot drift.

    python profile_summary.py /tmp/fm48_decode_window.csv --out profile_summary_decode.json
    python profile_summary.py /tmp/fm48_prefill_window.csv --out profile_summary_prefill.json \\
        --mode prefill
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import rank_full_model_48 as RANK

LAYERS = 48


def open_csv(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("decode", "prefill"), default="decode")
    args = parser.parse_args()

    with open_csv(args.csv) as handle:
        rows = list(csv.DictReader(handle))

    by_device = defaultdict(list)
    for row in rows:
        by_device[row["DEVICE ID"]].append(row)

    out: dict = {"mode": args.mode, "layers": LAYERS, "window_rows": len(rows), "devices": len(by_device)}
    out["ops_per_device"] = {d: len(r) for d, r in sorted(by_device.items())}

    device_totals = {}
    for device, drows in sorted(by_device.items()):
        device_totals[device] = sum(RANK.us(r) for r in drows)
    out["device_kernel_us"] = device_totals
    spread = max(device_totals.values()) - min(device_totals.values())
    out["device_spread_us"] = spread
    out["device_spread_percent"] = 100.0 * spread / max(device_totals.values())

    # Per-op-code totals on the reported device, and on the mesh mean.
    reported = by_device["0"]
    total = device_totals["0"]
    out["iteration_us"] = total

    per_code: dict = defaultdict(lambda: {"us": 0.0, "n": 0})
    for row in reported:
        entry = per_code[row["OP CODE"]]
        entry["us"] += RANK.us(row)
        entry["n"] += 1
    out["op_code_us"] = {k: v for k, v in sorted(per_code.items(), key=lambda kv: -kv[1]["us"])}

    if args.mode == "decode":
        pre, layers, post = RANK.regions(reported)
        pre_us = sum(RANK.us(r) for r in pre)
        layers_us = sum(RANK.us(r) for r in layers)
        post_us = sum(RANK.us(r) for r in post)
        out["regions_us"] = {"terminal_pre": pre_us, "layer_stack": layers_us, "terminal_post": post_us}
        out["regions_percent"] = {
            "terminal_pre": 100.0 * pre_us / total,
            "layer_stack": 100.0 * layers_us / total,
            "terminal_post": 100.0 * post_us / total,
        }
        out["per_layer_us"] = layers_us / LAYERS
        out["per_layer_us_all_devices"] = {
            d: sum(RANK.us(r) for r in RANK.regions(drows)[1]) / LAYERS for d, drows in sorted(by_device.items())
        }
        # The sampler is a *separate* captured trace from the model trace, so
        # its kernel time is exactly what ``token_out - model_trace`` should
        # buy. Everything in terminal-post from the LM head matmul onward.
        lm_head = next(
            i for i, r in enumerate(post) if r["OP CODE"] == "MatmulDeviceOperation" and "37984" in RANK.shape(r, 1)
        )
        out["lm_head_us"] = RANK.us(post[lm_head])
        # ...and on every device, because ``tt-perf-report`` merges the mesh and
        # reports the *slowest* device's row. Pairing device 0's kernel time with
        # that row's bandwidth utilisation mixes two devices; the stage-06 review
        # caught exactly that in the LM-head headroom arithmetic. Publishing all
        # four lets the checker pair a utilisation with its own device.
        out["lm_head_us_all_devices"] = {
            d: next(
                RANK.us(r)
                for r in RANK.regions(drows)[2]
                if r["OP CODE"] == "MatmulDeviceOperation" and "37984" in RANK.shape(r, 1)
            )
            for d, drows in sorted(by_device.items())
        }
        out["sampler_us"] = sum(RANK.us(r) for r in post[lm_head + 1 :])
        out["norm_and_lm_head_us"] = sum(RANK.us(r) for r in post[: lm_head + 1])

        # The two 4-wide composite all-gathers, delimited structurally rather
        # than by hand: ``ttnn.all_gather``'s composite path is
        # ``AllBroadcast -> UntilizeWithUnpadding x4 -> Concat -> Permute ->
        # TilizeWithValPadding``, so each gather is the run from an
        # ``AllBroadcast`` to the first ``TilizeWithValPadding`` after the next
        # ``Concat``. This is the lever named under Limitations, so it is
        # measured off the CSV and not summed by eye.
        composite = 0.0
        composite_rows = 0
        for start, row in enumerate(post):
            if row["OP CODE"] != "AllBroadcastDeviceOperation":
                continue
            concat = next(i for i in range(start, len(post)) if post[i]["OP CODE"] == "ConcatDeviceOperation")
            end = next(
                i for i in range(concat, len(post)) if post[i]["OP CODE"] == "TilizeWithValPaddingDeviceOperation"
            )
            composite += sum(RANK.us(r) for r in post[start : end + 1])
            composite_rows += end + 1 - start
        out["composite_gather_us"] = composite
        out["composite_gather_rows"] = composite_rows

        # The expert-tail layout churn: the three ReshapeViews that compact the
        # 32x-row-padded expert output, plus the tilize/untilize around the
        # router. Blocked on TTNN's Tile([1,32]) gap; the largest identified
        # blocked item in the model, so it is quoted and therefore derived.
        churn_labels = {
            "ReshapeView 1x32x32x2048",
            "ReshapeView 1x32x32x1536",
            "ReshapeView 1x1x32x768",
            "TilizeWithValPadding 1x1x1x128",
            "UntilizeWithUnpadding 1x1x32x32",
            "UntilizeWithUnpadding 1x1x32x128",
        }
        churn = sum(RANK.us(r) for r in layers if RANK.label(r) in churn_labels)
        out["layout_churn"] = {
            "us_per_layer": churn / LAYERS,
            "ms_per_iteration": churn / 1000.0,
            "percent_of_iteration": 100.0 * churn / total,
            "labels": sorted(churn_labels),
        }

    # Per-layer ranking, for the table the README publishes.
    if args.mode == "decode":
        agg = defaultdict(lambda: [0.0, 0, set()])
        for r in layers:
            entry = agg[RANK.label(r)]
            entry[0] += RANK.us(r)
            entry[1] += 1
            entry[2].add(int(r["CORE COUNT"]))
        out["per_layer_ranking"] = [
            {"op": op, "us": t, "us_per_layer": t / LAYERS, "n": n, "cores": sorted(c), "percent": 100.0 * t / total}
            for op, (t, n, c) in sorted(agg.items(), key=lambda kv: -kv[1][0])
        ]
        agg = defaultdict(lambda: [0.0, 0, set()])
        for r in post:
            entry = agg[RANK.label(r)]
            entry[0] += RANK.us(r)
            entry[1] += 1
            entry[2].add(int(r["CORE COUNT"]))
        out["terminal_post_ranking"] = [
            {"op": op, "us": t, "n": n, "cores": sorted(c), "percent": 100.0 * t / total}
            for op, (t, n, c) in sorted(agg.items(), key=lambda kv: -kv[1][0])
        ]
    else:
        agg = defaultdict(lambda: [0.0, 0, set()])
        for r in reported:
            entry = agg[RANK.label(r)]
            entry[0] += RANK.us(r)
            entry[1] += 1
            entry[2].add(int(r["CORE COUNT"]))
        out["ranking"] = [
            {"op": op, "us": t, "us_per_layer": t / LAYERS, "n": n, "cores": sorted(c), "percent": 100.0 * t / total}
            for op, (t, n, c) in sorted(agg.items(), key=lambda kv: -kv[1][0])
        ]

    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}: {total:,.1f} us over {len(reported)} ops on device 0")
    for key, value in out.get("regions_us", {}).items():
        print(f"  {key:<14} {value:9,.1f} us  {out['regions_percent'][key]:5.2f}%")
    if args.mode == "decode":
        print(f"  per layer      {out['per_layer_us']:9.3f} us")
        print(f"  lm head        {out['lm_head_us']:9.3f} us")
        print(f"  sampler        {out['sampler_us']:9.3f} us")


if __name__ == "__main__":
    main()
