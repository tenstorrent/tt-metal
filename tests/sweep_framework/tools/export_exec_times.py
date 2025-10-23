#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
from typing import Dict, List, Optional, Tuple


def parse_first_batch_size(batch_sizes_text: str) -> Optional[int]:
    match = re.search(r"\((\d+)", batch_sizes_text or "")
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def load_curated_vectors(
    vectors_json_path: str,
    suite: str,
) -> List[Dict[str, str]]:
    with open(vectors_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if suite not in data:
        raise KeyError(f"Suite '{suite}' not found in vectors JSON.")

    curated_entries: List[Dict[str, str]] = []
    for input_hash, meta in data[suite].items():
        curated_entries.append(
            {
                "input_hash": input_hash,
                "batch_sizes": meta.get("batch_sizes", ""),
                "height": str(meta.get("height", "")),
                "width": str(meta.get("width", "")),
                "layout": meta.get("layout", ""),
                "input_dtype": meta.get("input_dtype", ""),
                "op_name": meta.get("op_name", ""),
            }
        )
    return curated_entries


def build_duration_map(
    results_json_path: str,
    metric_name: str,
) -> Dict[str, float]:
    input_hash_to_duration: Dict[str, float] = {}
    with open(results_json_path, "r", encoding="utf-8") as f:
        try:
            results = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse results JSON: {e}")

    if not isinstance(results, list):
        raise ValueError("Results JSON must be a list of entries.")

    for entry in results:
        input_hash = entry.get("input_hash")
        if not input_hash or input_hash in input_hash_to_duration:
            continue

        metrics = entry.get("metrics")
        if not isinstance(metrics, list):
            continue

        for m in metrics:
            if m.get("metric_name") == metric_name:
                try:
                    value = float(m.get("metric_value"))
                except (TypeError, ValueError):
                    continue
                input_hash_to_duration[input_hash] = value
                break

    return input_hash_to_duration


def derive_input_shape(batch_sizes_text: str, height_text: str, width_text: str) -> str:
    batch = parse_first_batch_size(batch_sizes_text)
    height = str(height_text)
    width = str(width_text)
    if batch is None:
        return f"(, {height}, {width})"
    return f"({batch}, {height}, {width})"


def write_csv(
    rows: List[Dict[str, str]],
    durations: Dict[str, float],
    output_csv_path: str,
) -> Tuple[int, int, int]:
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    headers = [
        "batch_sizes",
        "input_shape",
        "height",
        "width",
        "layout",
        "input_dtype",
        "op_name",
        "execution_time",
    ]

    total = 0
    matched = 0
    unmatched = 0

    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            total += 1
            input_hash = r.get("input_hash", "")
            execution_time_text = ""
            if input_hash in durations:
                matched += 1
                execution_time_text = str(durations[input_hash])
            else:
                unmatched += 1

            writer.writerow(
                {
                    "batch_sizes": r.get("batch_sizes", ""),
                    "input_shape": derive_input_shape(
                        r.get("batch_sizes", ""), r.get("height", ""), r.get("width", "")
                    ),
                    "height": r.get("height", ""),
                    "width": r.get("width", ""),
                    "layout": r.get("layout", ""),
                    "input_dtype": r.get("input_dtype", ""),
                    "op_name": r.get("op_name", ""),
                    "execution_time": execution_time_text,
                }
            )

    return total, matched, unmatched


def default_output_path(results_json_path: str, vectors_json_path: str) -> str:
    results_dir = os.path.dirname(os.path.abspath(results_json_path))
    vectors_stem = os.path.splitext(os.path.basename(vectors_json_path))[0]
    return os.path.join(results_dir, f"{vectors_stem}_curated_exec_times.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Join curated vectors JSON with results JSON by input_hash and export a CSV "
            "with execution_time (ns) from metric 'device_DEVICE FW DURATION [ns]'."
        )
    )
    parser.add_argument(
        "--vectors-json",
        required=True,
        help="Path to vectors JSON (e.g., data_movement.curated.curated.json)",
    )
    parser.add_argument(
        "--results-json",
        required=True,
        help="Path to results JSON (e.g., data_movement_<run>.json)",
    )
    parser.add_argument(
        "--output-csv",
        required=False,
        help="Optional output CSV path. If omitted, a default will be used.",
    )
    parser.add_argument(
        "--suite",
        default="nightly",
        help="Suite name inside vectors JSON to use (default: nightly)",
    )
    parser.add_argument(
        "--metric-name",
        default="device_DEVICE FW DURATION [ns]",
        help="Metric name to extract from results JSON (default: device_DEVICE FW DURATION [ns])",
    )

    args = parser.parse_args()

    output_csv = args.output_csv or default_output_path(args.results_json, args.vectors_json)

    curated_rows = load_curated_vectors(args.vectors_json, args.suite)
    durations = build_duration_map(args.results_json, args.metric_name)
    total, matched, unmatched = write_csv(curated_rows, durations, output_csv)

    print(
        f"Wrote CSV: {output_csv}\n"
        f"Total curated rows: {total}\n"
        f"Matched execution_time: {matched}\n"
        f"Unmatched: {unmatched}"
    )


if __name__ == "__main__":
    main()
