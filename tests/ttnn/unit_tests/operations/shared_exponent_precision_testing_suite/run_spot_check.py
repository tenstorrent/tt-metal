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

# Default seed range when not specified
DEFAULT_SEED_START = 0
DEFAULT_SEED_END = 9
# Default CSVs next to this script so they're always in the suite folder
_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_CSV = str(_SCRIPT_DIR / "matmul_spot_check.csv")
DEFAULT_SUM_OUTPUT_CSV = str(_SCRIPT_DIR / "sum_spot_check.csv")

# Numeric metrics that get a percentage-difference column (row vs column)
_METRIC_KEYS = [
    "pcc",
    "max_abs_error",
    "mean_abs_error",
    "max_rel_error",
    "mean_rel_error",
    "ulp_max",
    "ulp_mean",
]


def _pct_diff(row_val: float, col_val: float) -> float:
    """Percentage difference (row - column) / column * 100. Returns 0.0 if col_val is 0."""
    if col_val == 0 or (isinstance(col_val, float) and abs(col_val) < 1e-12):
        return 0.0
    return 100.0 * (row_val - col_val) / col_val


def _add_pct_diff_columns(rows: list, key_fn, col_pattern: str, row_pattern: str) -> None:
    """Add *_pct_diff columns (row vs column) to each row. Mutates rows in place."""
    by_key = {}
    for r in rows:
        k = key_fn(r)
        if r["pattern"] == col_pattern:
            by_key.setdefault(k, {})["col"] = r
        elif r["pattern"] == row_pattern:
            by_key.setdefault(k, {})["row"] = r
    for group in by_key.values():
        col_row = group.get("col")
        row_row = group.get("row")
        if col_row is None or row_row is None:
            continue
        for m in _METRIC_KEYS:
            pct = _pct_diff(row_row[m], col_row[m])
            col_row[f"{m}_pct_diff"] = row_row[f"{m}_pct_diff"] = pct


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
):
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

    # Sum tests: one run per (pattern, axis); no seed (input is deterministic)
    for pattern_name in spot_patterns:
        if pattern_name not in patterns:
            continue
        test_input = (patterns[pattern_name]() * dist_tensor).float()
        for axis in [0, 1]:
            logger.info(f"Running sum(axis={axis}): pattern={pattern_name}")
            metrics = _run_precision_test(test_input, device, OperationType.SUM, axis=axis)
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

    # Add percentage-difference columns (row vs column) for each numeric metric
    _add_pct_diff_columns(
        matmul_rows, key_fn=lambda r: r["seed"], col_pattern="column_gradient", row_pattern="row_gradient"
    )
    _add_pct_diff_columns(
        sum_rows, key_fn=lambda r: r["axis"], col_pattern="column_gradient", row_pattern="row_gradient"
    )

    # Build fieldnames: each metric followed by its _pct_diff (except booleans)
    def _fieldnames_with_pct_diff(prefix: list) -> list:
        out = list(prefix)
        for m in _METRIC_KEYS:
            out.append(m)
            out.append(f"{m}_pct_diff")
        out.extend(["allclose_1e_2", "allclose_1e_3"])
        return out

    matmul_fieldnames = _fieldnames_with_pct_diff(["seed", "pattern"])
    matmul_path = Path(output_csv)
    _write_csv(matmul_path, matmul_rows, matmul_fieldnames)
    logger.info(f"Wrote {len(matmul_rows)} matmul rows to {matmul_path}")

    sum_fieldnames = _fieldnames_with_pct_diff(["pattern", "axis"])
    sum_path = Path(output_sum_csv)
    _write_csv(sum_path, sum_rows, sum_fieldnames)
    logger.info(f"Wrote {len(sum_rows)} sum rows to {sum_path}")

    # Brief summary
    print("\n" + "=" * 60)
    print(f"Spot-check: seeds {seed_start}..{seed_end}")
    print("=" * 60)
    print(f"  Matmul: {len(matmul_rows)} rows → {matmul_path}")
    print(f"  Sum:    {len(sum_rows)} rows → {sum_path}")
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
    args = parser.parse_args()
    if args.seed_end < args.seed_start:
        parser.error("--seed-end must be >= --seed-start")
    run_spot_check(
        seed_start=args.seed_start,
        seed_end=args.seed_end,
        output_csv=args.output,
        output_sum_csv=args.output_sum,
    )


if __name__ == "__main__":
    main()
