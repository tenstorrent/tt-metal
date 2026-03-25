#!/usr/bin/env python3
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


INPUT_FILENAME = "all_numeric_results_merged.csv"
OUTPUT_FILENAME = "sorted_all_numeric_results_merged.csv"

# Metrics to aggregate (max) per test_name and per dtype
BASE_METRICS = [
    "max_abs_dif",
    "max_rel_dif",
    "frobenius_value",
    "max_atol_div_k",
    "max_rtol_div_k",
    "frobenius_value_div_k",
]

# Output dtype suffix ordering
DTYPE_SUFFIX_ORDER = ["bf4", "bf8", "bf16", "fp32"]


def normalize_dtype(dtype_value: Optional[str]) -> Optional[str]:
    """
    Normalize dtype strings coming from CSV. Expected inputs include:
      - 'bfloat4', 'bfloat8', 'bfloat16', 'float32'
      - Possibly enum-like tokens already normalized upstream
    Returns one of: 'bf4', 'bf8', 'bf16', 'fp32', or None if unknown.
    """
    if not dtype_value:
        return None
    s = str(dtype_value).strip().lower()
    if s == "bfloat4":
        return "bf4"
    if s == "bfloat8":
        return "bf8"
    if s == "bfloat16":
        return "bf16"
    if s == "float32":
        return "fp32"
    return None


def parse_float(value: str) -> Optional[float]:
    """Parse a float from string safely; return None if not parseable."""
    try:
        # strip in case CSV has whitespace
        v = value.strip()
        # Accept 'N/A' or similar tokens
        if v.lower() in ("n/a", "na", ""):
            return None
        return float(v)
    except Exception:
        return None


def base_test_name(test_name: str) -> str:
    """
    Extract base test name by stripping everything after the first '['.
    e.g., 'test_addmm_square_matrices[dtype=...,...]' -> 'test_addmm_square_matrices'
    """
    if not test_name:
        return ""
    idx = test_name.find("[")
    return test_name[:idx] if idx != -1 else test_name


def build_output_header() -> List[str]:
    """
    Build the output CSV header with the required columns.
    Format:
      test_name,
      max_abs_dif_bf4, max_abs_dif_bf8, max_abs_dif_bf16, max_abs_dif_fp32,
      max_rel_dif_bf4, ...,
      frobenius_value_bf4, ...,
      max_atol_div_k_bf4, ...,
      max_rtol_div_k_bf4, ...,
      frobenius_value_div_k_bf4, ...
    """
    header = ["test_name"]
    for metric in BASE_METRICS:
        for suffix in DTYPE_SUFFIX_ORDER:
            header.append(f"{metric}_{suffix}")
    return header


def aggregate_max_per_test_dtype(
    rows: List[Dict[str, str]],
) -> Tuple[List[str], Dict[str, Dict[str, Optional[float]]]]:
    """
    Compute max for each metric per base test_name and dtype.
    Returns:
      - sorted unique base test names
      - mapping: test -> { f"{metric}_{dtype_suffix}": max_value }
    """
    tests_set = set()
    # test_name -> metric_key -> max_value
    agg: Dict[str, Dict[str, Optional[float]]] = defaultdict(dict)

    for row in rows:
        tname = base_test_name(row.get("test_name", ""))
        if not tname:
            continue
        tests_set.add(tname)

        dtype_suffix = normalize_dtype(row.get("test_dtype", ""))
        if dtype_suffix is None:
            # skip rows where dtype cannot be normalized
            continue

        for metric in BASE_METRICS:
            if metric not in row:
                continue
            val = parse_float(str(row[metric]))
            if val is None:
                continue
            key = f"{metric}_{dtype_suffix}"
            prev = agg[tname].get(key)
            if prev is None or val > prev:
                agg[tname][key] = val

    return sorted(tests_set), agg


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_csv(path: str, header: List[str], tests: List[str], data: Dict[str, Dict[str, Optional[float]]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for tname in tests:
            row = [tname]
            for metric in BASE_METRICS:
                for suffix in DTYPE_SUFFIX_ORDER:
                    key = f"{metric}_{suffix}"
                    val = data.get(tname, {}).get(key)
                    row.append(f"{val:.6e}" if isinstance(val, float) else "")
            writer.writerow(row)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    in_path = os.path.join(here, INPUT_FILENAME)
    out_path = os.path.join(here, OUTPUT_FILENAME)

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    rows = read_csv(in_path)
    tests, agg = aggregate_max_per_test_dtype(rows)
    header = build_output_header()
    write_csv(out_path, header, tests, agg)

    # Also emit a simple unique test list file for convenience
    test_list_path = os.path.join(here, "unique_test_list.txt")
    with open(test_list_path, "w", encoding="utf-8") as f:
        for t in tests:
            f.write(t + "\n")

    print(f"Wrote {out_path} with {len(tests)} tests.")
    print(f"Wrote {test_list_path}.")


if __name__ == "__main__":
    main()
