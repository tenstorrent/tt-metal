# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Spot-check: matmul (over seeds) + sum (pattern/axis). Writes matmul_spot_check.csv and sum_spot_check.csv.
"""

import argparse
import csv
from pathlib import Path

from loguru import logger
import torch
import ttnn

from generators import generate_test_patterns, generate_distributions
from constants import OperationType, ResultKeys
from runner import _run_precision_test
from utils.plot_utils import plot_tensor_distribution, tensor_to_heatmap

# Default seed range when not specified
DEFAULT_SEED_START = 0
DEFAULT_SEED_END = 9
# Default CSVs next to this script so they're always in the suite folder
_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_CSV = str(_SCRIPT_DIR / "matmul_spot_check.csv")
DEFAULT_SUM_OUTPUT_CSV = str(_SCRIPT_DIR / "sum_spot_check.csv")

# CSV column order
MATMUL_FIELDNAMES = [
    "seed",
    "pattern",
    "pcc",
    "max_abs_error",
    "mean_abs_error",
    "max_rel_error",
    "mean_rel_error",
    "ulp_max",
    "ulp_mean",
    "allclose_1e_2",
    "allclose_1e_3",
]
SUM_FIELDNAMES = [
    "pattern",
    "axis",
    "pcc",
    "max_abs_error",
    "mean_abs_error",
    "max_rel_error",
    "mean_rel_error",
    "ulp_max",
    "ulp_mean",
    "allclose_1e_2",
    "allclose_1e_3",
]


def _write_csv(path: Path, rows: list, fieldnames: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def run_spot_check(
    seed_start: int = DEFAULT_SEED_START,
    seed_end: int = DEFAULT_SEED_END,
    output_csv: str = DEFAULT_OUTPUT_CSV,
    output_sum_csv: str = DEFAULT_SUM_OUTPUT_CSV,
    col_tilize: bool = False,
    write_csv: bool = True,
    enable_heatmap: bool = False,
    enable_distribution: bool = False,
    save_path: str = None,
):
    if save_path is None:
        save_path = str(_SCRIPT_DIR)
    device = ttnn.open_device(device_id=0)
    shape = (512, 512)

    patterns = generate_test_patterns(shape)
    distributions = generate_distributions(shape)

    spot_patterns = ["column_gradient", "row_gradient"]
    dist_tensor = distributions["constant"]

    seeds = range(seed_start, seed_end + 1)
    matmul_rows = []
    sum_rows = []

    for seed in seeds:
        torch.manual_seed(seed)
        matmul_second_tensor = torch.randn(shape[1], shape[0])

        for pattern_name in spot_patterns:
            if pattern_name not in patterns:
                logger.warning(f"Pattern {pattern_name} not found, skipping")
                continue
            test_input = (patterns[pattern_name]() * dist_tensor).float()

            logger.info(f"Running matmul: seed={seed} pattern={pattern_name}")
            metrics = _run_precision_test(
                test_input,
                device,
                OperationType.MATMUL,
                optional_matmul_second_tensor=matmul_second_tensor,
                col_tilize=col_tilize,
            )
            matmul_rows.append(
                {
                    "seed": seed,
                    "pattern": pattern_name,
                    "pcc": metrics[ResultKeys.PCC],
                    "max_abs_error": metrics[ResultKeys.MAX_ABS_ERROR],
                    "mean_abs_error": metrics[ResultKeys.MEAN_ABS_ERROR],
                    "max_rel_error": metrics[ResultKeys.MAX_REL_ERROR],
                    "mean_rel_error": metrics[ResultKeys.MEAN_REL_ERROR],
                    "ulp_max": metrics[ResultKeys.ULP_MAX],
                    "ulp_mean": metrics[ResultKeys.ULP_MEAN],
                    "allclose_1e_2": metrics[ResultKeys.ALLCLOSE_1E_2],
                    "allclose_1e_3": metrics[ResultKeys.ALLCLOSE_1E_3],
                }
            )

            # Optional heatmap and distribution for output tensors
            ref_out = metrics["reference_output"]
            ttnn_out = metrics["ttnn_output"]
            prefix = f"matmul_seed{seed}_{pattern_name}"
            if enable_distribution:
                plot_tensor_distribution(ref_out, title=f"{prefix}_out_torch_distribution", save_path=save_path)
                plot_tensor_distribution(ttnn_out, title=f"{prefix}_out_ttnn_distribution", save_path=save_path)
            if enable_heatmap:
                tensor_to_heatmap(ref_out, output_path=Path(save_path) / f"{prefix}_out_torch_heatmap.png")
                tensor_to_heatmap(ttnn_out, output_path=Path(save_path) / f"{prefix}_out_ttnn_heatmap.png")

    # Sum tests: one run per (pattern, axis); no seed (input is deterministic)
    for pattern_name in spot_patterns:
        if pattern_name not in patterns:
            continue
        test_input = (patterns[pattern_name]() * dist_tensor).float()
        for axis in [0, 1]:
            logger.info(f"Running sum(axis={axis}): pattern={pattern_name}")
            metrics = _run_precision_test(test_input, device, OperationType.SUM, axis=axis, col_tilize=col_tilize)
            sum_rows.append(
                {
                    "pattern": pattern_name,
                    "axis": axis,
                    "pcc": metrics[ResultKeys.PCC],
                    "max_abs_error": metrics[ResultKeys.MAX_ABS_ERROR],
                    "mean_abs_error": metrics[ResultKeys.MEAN_ABS_ERROR],
                    "max_rel_error": metrics[ResultKeys.MAX_REL_ERROR],
                    "mean_rel_error": metrics[ResultKeys.MEAN_REL_ERROR],
                    "ulp_max": metrics[ResultKeys.ULP_MAX],
                    "ulp_mean": metrics[ResultKeys.ULP_MEAN],
                    "allclose_1e_2": metrics[ResultKeys.ALLCLOSE_1E_2],
                    "allclose_1e_3": metrics[ResultKeys.ALLCLOSE_1E_3],
                }
            )

    ttnn.close_device(device)

    if write_csv:
        matmul_path = Path(output_csv)
        _write_csv(matmul_path, matmul_rows, MATMUL_FIELDNAMES)
        logger.info(f"Wrote {len(matmul_rows)} matmul rows to {matmul_path}")

        sum_path = Path(output_sum_csv)
        _write_csv(sum_path, sum_rows, SUM_FIELDNAMES)
        logger.info(f"Wrote {len(sum_rows)} sum rows to {sum_path}")

    # Brief summary
    print("\n" + "=" * 60)
    print(f"Spot-check: seeds {seed_start}..{seed_end}")
    print("=" * 60)
    if write_csv:
        print(f"  Matmul: {len(matmul_rows)} rows → {Path(output_csv)}")
        print(f"  Sum:    {len(sum_rows)} rows → {Path(output_sum_csv)}")
    else:
        print(f"  Matmul: {len(matmul_rows)} rows (CSV disabled)")
        print(f"  Sum:    {len(sum_rows)} rows (CSV disabled)")
    print("=" * 60)

    return {"matmul": matmul_rows, "sum": sum_rows}


def main():
    parser = argparse.ArgumentParser(
        description="Run matmul (over seeds) and sum spot-checks (512×512 constant); write two CSVs."
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=DEFAULT_SEED_START,
        help=f"First seed (inclusive). Default: {DEFAULT_SEED_START}",
    )
    parser.add_argument(
        "--seed-end",
        type=int,
        default=DEFAULT_SEED_END,
        help=f"Last seed (inclusive). Default: {DEFAULT_SEED_END}",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_CSV,
        help="Matmul output CSV path (default: matmul_spot_check.csv in this script's directory)",
    )
    parser.add_argument(
        "--output-sum",
        type=str,
        default=DEFAULT_SUM_OUTPUT_CSV,
        help="Sum output CSV path (default: sum_spot_check.csv in this script's directory)",
    )
    parser.add_argument(
        "--col-tilize",
        action="store_true",
        help="Use BFP col_tilize (exponent shared along columns); requires bfloat8_b, TILE layout",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not write matmul_spot_check.csv or sum_spot_check.csv",
    )
    parser.add_argument(
        "--heatmap",
        action="store_true",
        help="Generate heatmap images for output tensors and abs/rel error",
    )
    parser.add_argument(
        "--distribution",
        action="store_true",
        help="Generate distribution plots for output tensors and abs/rel error",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Directory for heatmap/distribution outputs (default: script directory)",
    )
    args = parser.parse_args()
    if args.seed_end < args.seed_start:
        parser.error("--seed-end must be >= --seed-start")
    run_spot_check(
        seed_start=args.seed_start,
        seed_end=args.seed_end,
        output_csv=args.output,
        output_sum_csv=args.output_sum,
        col_tilize=args.col_tilize,
        write_csv=not args.no_csv,
        enable_heatmap=args.heatmap,
        enable_distribution=args.distribution,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
