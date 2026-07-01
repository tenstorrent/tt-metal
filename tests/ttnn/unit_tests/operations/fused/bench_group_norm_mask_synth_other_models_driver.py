# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Tracy-driven A/B benchmark runner for the mask-synthesis change across
# representative shapes from every non-SDXL model family that calls
# ttnn.group_norm (see bench_group_norm_mask_synth_other_models.py).

import json
import sys

from models.perf.device_perf_utils import run_device_perf_detailed

BENCH_FILE = "tests/ttnn/unit_tests/operations/fused/bench_group_norm_mask_synth_other_models.py"

# Mirrors SHAPES in bench_group_norm_mask_synth_other_models.py
UNIQUE_SHAPES = [
    "sd_vae_C128_H128",
    "sd_vae_C256_H64",
    "sd_vae_C512_H64",
    "sd_vae_C512_H32",
    "sd35_vae_C128_H128",
    "sd35_vae_C256_H64",
    "sd35_vae_C512_H32",
    "oft_C256_H96",
    "oft_C256_H48",
    "oft_C256_H24",
    "retinanet_C256_H128",
    "retinanet_C256_H64",
    "retinanet_C256_H32",
    "unet3d_C128_H128",
    "unet3d_C256_H64",
]


def measure(shape_id, mode):
    case = f"test_bench_other_models_group_norm_mask[mode={mode}-{shape_id}-" f"device_params={{'l1_small_size': 0}}]"
    command = f'pytest "{BENCH_FILE}::{case}" -v'
    subdir = f"bench_gn_other_{shape_id}_{mode}"
    cols = ["DEVICE KERNEL"]
    op_name = "GroupNormDeviceOperation"
    results = run_device_perf_detailed(command=command, subdir=subdir, cols=cols, op_name=op_name)
    return results["DEVICE KERNEL"]


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--single":
        shape_id, mode = sys.argv[2], sys.argv[3]
        r = measure(shape_id, mode)
        print(json.dumps(r, indent=2))
        return

    rows = []
    for shape_id in UNIQUE_SHAPES:
        row = {"shape": shape_id}
        for mode in ("dram", "synth"):
            try:
                stats = measure(shape_id, mode)
                row[f"{mode}_avg_ns"] = stats["AVG"]
                row[f"{mode}_min_ns"] = stats["MIN"]
                row[f"{mode}_max_ns"] = stats["MAX"]
            except Exception as e:
                row[f"{mode}_error"] = str(e)
        if "dram_avg_ns" in row and "synth_avg_ns" in row:
            delta = row["dram_avg_ns"] - row["synth_avg_ns"]
            pct = 100.0 * delta / row["dram_avg_ns"] if row["dram_avg_ns"] else 0.0
            row["delta_ns"] = delta
            row["pct"] = pct
        rows.append(row)

    print()
    print("# Per-shape device kernel duration (AVG ns) — other model shapes")
    print(f"| {'shape':<22} | {'dram_avg':>9} | {'synth_avg':>9} | {'delta':>9} | {'pct':>6} |")
    print(f"| {'-'*22} | {'-'*9} | {'-'*9} | {'-'*9} | {'-'*6} |")
    for r in rows:
        d = r.get("dram_avg_ns", float("nan"))
        s = r.get("synth_avg_ns", float("nan"))
        delta = r.get("delta_ns", float("nan"))
        pct = r.get("pct", float("nan"))
        print(f"| {r['shape']:<22} | {d:>9.0f} | {s:>9.0f} | {delta:>9.0f} | {pct:>5.2f}% |")

    print()
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
