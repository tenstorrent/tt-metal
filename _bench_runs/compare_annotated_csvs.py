#!/usr/bin/env python3
"""Annotate multiple pi0.5 ops_perf_results CSVs and print a side-by-side
stage-breakdown comparison.

Uses the same annotation heuristics as annotate_ops_csv.py:
  - 135 SDPA calls split into stages: siglip (1..27), vlm_prefill (28..45),
    denoise_step_1..5 (46..63 / 64..81 / 82..99 / 100..117 / 118..135)
  - Layer boundaries via "last LN between consecutive SDPAs"
  - project_output starts at 2nd LN after the last SDPA

Usage:
    python _bench_runs/compare_annotated_csvs.py csv1 csv2 [csv3 ...]
    python _bench_runs/compare_annotated_csvs.py \\
        --labels num_cams=1 num_cams=2 num_cams=3 \\
        cam1.csv cam2.csv cam3.csv
"""

import argparse
import csv
from collections import OrderedDict
from pathlib import Path


def annotate(in_csv: str):
    """Returns (stage_kernel_ms: OrderedDict[stage,ms], total_ms)."""
    with open(in_csv) as f:
        rows = list(csv.reader(f))
    header = rows[0]
    body = rows[1:]
    op_col = header.index("OP CODE")
    kd_col = header.index("DEVICE KERNEL DURATION [ns]")

    sdpa_idx = [i for i, r in enumerate(body) if r[op_col] == "SDPAOperation"]
    assert len(sdpa_idx) == 135, f"{in_csv}: expected 135 SDPAs, got {len(sdpa_idx)}"

    stages = [
        ("siglip", 1, 27, 27),
        ("vlm_prefill", 28, 45, 18),
        ("denoise_step_1", 46, 63, 18),
        ("denoise_step_2", 64, 81, 18),
        ("denoise_step_3", 82, 99, 18),
        ("denoise_step_4", 100, 117, 18),
        ("denoise_step_5", 118, 135, 18),
    ]

    def pre_attn_ln_for(sdpa_body_idx, lower_bound):
        last_ln = None
        for j in range(lower_bound, sdpa_body_idx):
            if body[j][op_col] == "LayerNormDeviceOperation":
                last_ln = j
        if last_ln is None:
            return lower_bound
        if last_ln > 0 and body[last_ln - 1][op_col] == "InterleavedToShardedDeviceOperation":
            return last_ln - 1
        return last_ln

    def project_output_start(last_sdpa_body_idx):
        ln_seen = 0
        for j in range(last_sdpa_body_idx + 1, len(body)):
            if body[j][op_col] == "LayerNormDeviceOperation":
                ln_seen += 1
                if ln_seen == 2:
                    if j > 0 and body[j - 1][op_col] == "InterleavedToShardedDeviceOperation":
                        return j - 1
                    return j
        return len(body)

    project_start = project_output_start(sdpa_idx[-1])
    stage_col = [""] * len(body)

    for stage_label, s_start, s_end, n_layers in stages:
        for layer_i in range(n_layers):
            sdpa_num = s_start + layer_i
            sdpa_body_idx = sdpa_idx[sdpa_num - 1]
            if layer_i == 0 and stage_label == "siglip":
                attn_start = pre_attn_ln_for(sdpa_body_idx, 0)
            elif layer_i == 0:
                attn_start = pre_attn_ln_for(sdpa_body_idx, sdpa_idx[s_start - 2] + 1)
            else:
                attn_start = pre_attn_ln_for(sdpa_body_idx, sdpa_idx[s_start - 1 + layer_i - 1] + 1)

            if layer_i + 1 < n_layers:
                next_sdpa_body_idx = sdpa_idx[s_start - 1 + layer_i + 1]
            elif (s_start, s_end) == (118, 135):
                mlp_end = project_start - 1
                next_sdpa_body_idx = None
            else:
                next_sdpa_body_idx = sdpa_idx[s_end]
            if next_sdpa_body_idx is not None:
                last_ln = None
                for j in range(sdpa_body_idx + 1, next_sdpa_body_idx):
                    if body[j][op_col] == "LayerNormDeviceOperation":
                        last_ln = j
                if last_ln is not None:
                    attn_start_next = (
                        last_ln - 1
                        if last_ln > 0 and body[last_ln - 1][op_col] == "InterleavedToShardedDeviceOperation"
                        else last_ln
                    )
                    mlp_end = attn_start_next - 1
                else:
                    mlp_end = next_sdpa_body_idx - 1

            for j in range(attn_start, mlp_end + 1):
                stage_col[j] = stage_label

    for i in range(len(body)):
        if stage_col[i] == "":
            stage_col[i] = "prefix_setup" if i < sdpa_idx[0] else "project_output"

    stage_kernel = OrderedDict()
    for s in [
        "prefix_setup",
        "siglip",
        "vlm_prefill",
        "denoise_step_1",
        "denoise_step_2",
        "denoise_step_3",
        "denoise_step_4",
        "denoise_step_5",
        "project_output",
    ]:
        stage_kernel[s] = 0.0
    for i, s in enumerate(stage_col):
        ns = float(body[i][kd_col] or 0)
        stage_kernel[s] += ns / 1e6
    total = sum(stage_kernel.values())
    return stage_kernel, total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="*", help="ops_perf_results CSV files (positional, in label order)")
    ap.add_argument(
        "--labels",
        help="Comma-separated column labels matching the CSV positional order, e.g. "
        "--labels=num_cams=1,num_cams=2,num_cams=3. Defaults to num_cams=N with N descending.",
    )
    args = ap.parse_args()

    csvs = args.csvs or [
        "/home/tt-admin/sdawle/pi0/tt-metal/generated/profiler/reports/2026_06_07_04_52_56/ops_perf_results_2026_06_07_04_52_56.csv",
        "/home/tt-admin/sdawle/pi0/tt-metal/generated/profiler/reports/2026_06_07_05_01_59/ops_perf_results_2026_06_07_05_01_59.csv",
        "/home/tt-admin/sdawle/pi0/tt-metal/generated/profiler/reports/2026_06_07_05_05_29/ops_perf_results_2026_06_07_05_05_29.csv",
    ]
    if args.labels:
        labels = [s.strip() for s in args.labels.split(",")]
        if len(labels) != len(csvs):
            ap.error(f"--labels has {len(labels)} entries but {len(csvs)} CSVs were given")
    else:
        labels = [f"num_cams={n}" for n in range(len(csvs), 0, -1)]

    results = []
    for label, csv_path in zip(labels, csvs):
        sk, total = annotate(csv_path)
        results.append((label, csv_path, sk, total))

    # Print side-by-side table
    print()
    print("=" * 96)
    print("  PI0.5 STAGE BREAKDOWN — side-by-side")
    print("  All runs: chunk=1024 + bf8_out + minimal_matmul + MINIMAL_CFG=4,8,8,1,8")
    print("            + SDPA_DENOISE_K_FORCE=64, 5 denoise steps")
    print("  Differing knob: PI0_NUM_CAMERAS")
    print("=" * 96)
    print()
    # Build header
    cols = [(lbl, sk, total) for lbl, _, sk, total in results]
    header = f"  {'STAGE':<18}"
    for lbl, _, _ in cols:
        header += f"  {lbl:>15}"
    # Delta column: last column minus first column.
    delta_label = f"Δ({cols[0][0]}→{cols[-1][0]})" if len(cols) >= 2 else ""
    header += f"  {delta_label:>20}"
    print(header)
    print("  " + "-" * 92)

    stages = list(results[0][2].keys())
    for stage in stages:
        line = f"  {stage:<18}"
        ms_vals = []
        for lbl, sk, _ in cols:
            ms = sk[stage]
            line += f"  {ms:>11.3f} ms"
            ms_vals.append(ms)
        if len(ms_vals) >= 2:
            delta = ms_vals[-1] - ms_vals[0]
            sign = "+" if delta >= 0 else ""
            line += f"  {sign}{delta:>7.3f} ms"
        print(line)

    print("  " + "-" * 92)
    # Total row
    line = f"  {'TOTAL':<18}"
    totals = []
    for lbl, _, total in cols:
        line += f"  {total:>11.3f} ms"
        totals.append(total)
    if len(totals) >= 2:
        delta = totals[-1] - totals[0]
        sign = "+" if delta >= 0 else ""
        line += f"  {sign}{delta:>7.3f} ms"
    print(line)
    print()

    # Percent table
    print("  " + "=" * 92)
    print(f"  {'STAGE':<18}", end="")
    for lbl, _, _ in cols:
        print(f"  {lbl:>15}", end="")
    print()
    print("  " + "-" * 92)
    for stage in stages:
        line = f"  {stage:<18}"
        for lbl, sk, total in cols:
            pct = (sk[stage] / total * 100.0) if total > 0 else 0.0
            line += f"  {pct:>14.1f}%"
        print(line)
    print()


if __name__ == "__main__":
    main()
