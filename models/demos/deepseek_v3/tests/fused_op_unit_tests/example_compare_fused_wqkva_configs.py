# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
from collections import defaultdict
from pathlib import Path

# Usage:
#   python3 compare_fused_wqkva_configs.py --verify-dir /path/to/verify_dir
# The verify directory must contain:
#   fused_decode.csv, fused_prefill_128.csv, mla_decode.csv, mla_prefill_128.csv


def normalize(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip()


def load_rows(path: Path):
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []
    return rows, fields


def extract_between_signposts(rows):
    start_idx = None
    stop_idx = None
    for idx, row in enumerate(rows):
        if row.get("OP TYPE") == "signpost" and row.get("OP CODE") == "start":
            start_idx = idx
            break
    for idx, row in enumerate(rows):
        if row.get("OP TYPE") == "signpost" and row.get("OP CODE") == "stop":
            stop_idx = idx
            break
    if start_idx is None or stop_idx is None or start_idx >= stop_idx:
        return rows
    return rows[start_idx + 1 : stop_idx]


def merge_device_rows(rows):
    block_by_device = defaultdict(list)
    for row in rows:
        if row.get("OP TYPE") != "tt_dnn_device":
            continue
        op_name = row.get("OP CODE")
        device_id = int(row.get("DEVICE ID"))
        block_by_device[device_id].append((op_name, row))

    device_ids = sorted(block_by_device.keys())
    merged = []
    while max(len(block_by_device[d]) for d in device_ids) > 0:
        blocks = []
        op_name = None
        for device_id in device_ids:
            if not block_by_device[device_id]:
                continue
            if op_name is None:
                op_name = block_by_device[device_id][0][0]
            if op_name != block_by_device[device_id][0][0]:
                continue
            blocks.append(block_by_device[device_id].pop(0))
        if not blocks:
            break

        is_collective = any(tag in op_name for tag in ("AllGather", "ReduceScatter", "AllReduce", "AllToAll"))
        if is_collective:
            device_kernel_durations = []
            for _, data in blocks:
                value = data.get("DEVICE KERNEL DURATION [ns]")
                try:
                    device_kernel_durations.append(float(value))
                except (TypeError, ValueError):
                    continue
            avg = (
                sum(device_kernel_durations) / len(device_kernel_durations) if device_kernel_durations else float("nan")
            )
            base = dict(blocks[0][1])
            base["DEVICE KERNEL DURATION [ns]"] = str(avg)
            merged.append(base)
        else:

            def duration(row):
                try:
                    return float(row.get("DEVICE KERNEL DURATION [ns]", "nan"))
                except (TypeError, ValueError):
                    return float("nan")

            max_block = max(blocks, key=lambda x: duration(x[1]))
            merged.append(dict(max_block[1]))
    return merged


def find_period(op_codes):
    if not op_codes:
        return 0
    first = op_codes[0]
    repeats = [i for i, code in enumerate(op_codes) if code == first]
    if len(repeats) < 2:
        return len(op_codes)
    return repeats[1]


def row_key(row, keys):
    return {k: normalize(row.get(k, "")) for k in keys}


def compare_rows(fused_rows, module_rows, keys):
    fused_keys = [row_key(r, keys) for r in fused_rows]
    module_keys = [row_key(r, keys) for r in module_rows]

    matches = []
    for start in range(0, len(module_keys) - len(fused_keys) + 1):
        ok = True
        for offset, fused_key in enumerate(fused_keys):
            module_key = module_keys[start + offset]
            for key, value in fused_key.items():
                if value != module_key.get(key, ""):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            matches.append(start)
    return matches


def report_case(name, fused_csv, module_csv, output_lines):
    fused_rows, fused_fields = load_rows(fused_csv)
    module_rows, _ = load_rows(module_csv)

    fused_rows = extract_between_signposts(fused_rows)
    fused_rows = merge_device_rows(fused_rows)
    module_rows = merge_device_rows([r for r in module_rows if r.get("OP TYPE") == "tt_dnn_device"])

    fused_codes = [row.get("OP CODE") for row in fused_rows]
    period = find_period(fused_codes)
    fused_one_iter = fused_rows[:period]

    key_fields = [f for f in fused_fields if f.startswith("INPUT_") or f.startswith("OUTPUT_")]
    for extra in ("OP CODE", "ATTRIBUTES"):
        if extra in fused_fields:
            key_fields.append(extra)

    output_lines.append(f"Case: {name}")
    output_lines.append(f"Fused merged rows: {len(fused_rows)}; period: {period}")
    output_lines.append(f"Module merged rows: {len(module_rows)}")
    output_lines.append(f"Fused op codes (one iter): {[row.get('OP CODE') for row in fused_one_iter]}")

    matches = compare_rows(fused_one_iter, module_rows, key_fields)
    if not matches:
        output_lines.append("Result: NO MATCHING SUBSEQUENCE")
        return False
    output_lines.append(f"Result: matched subsequence at start indices: {matches}")
    if len(matches) > 1:
        output_lines.append("Result: MULTIPLE MATCHES (ambiguous)")
        return False

    start = matches[0]
    ok = True
    for idx, fused_row in enumerate(fused_one_iter):
        module_row = module_rows[start + idx]
        for key in key_fields:
            fused_value = normalize(fused_row.get(key, ""))
            module_value = normalize(module_row.get(key, ""))
            if fused_value != module_value:
                output_lines.append(
                    f"MISMATCH row {idx} op {fused_row.get('OP CODE')} key {key}: fused={fused_value} module={module_value}"
                )
                ok = False
    output_lines.append("Result: MATCH" if ok else "Result: MISMATCH")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Compare fused op CSVs against MLA module CSVs.")
    parser.add_argument(
        "--verify-dir",
        type=Path,
        required=True,
        help="Directory containing fused/module CSVs and where the report will be written.",
    )
    args = parser.parse_args()
    verify_dir = args.verify_dir

    output_lines = []
    ok_decode = report_case(
        "decode",
        verify_dir / "fused_decode.csv",
        verify_dir / "mla_decode.csv",
        output_lines,
    )
    output_lines.append("")
    ok_prefill = report_case(
        "prefill_128",
        verify_dir / "fused_prefill_128.csv",
        verify_dir / "mla_prefill_128.csv",
        output_lines,
    )

    report_path = verify_dir / "config_compare_report.txt"
    report_path.write_text("\n".join(output_lines) + "\n")
    print(report_path)
    print("\n".join(output_lines))
    if not (ok_decode and ok_prefill):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
