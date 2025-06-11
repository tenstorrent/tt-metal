# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import json
import pandas as pd
import argparse
from tqdm import tqdm


def load_op_categories(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def load_known_ops(op_file):
    with open(op_file, "r") as f:
        return set(line.strip().split(".")[-1] for line in f if line.strip())


def load_existing_variants(json_path):
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r") as f:
        return json.load(f)


def is_duplicate(new_entry, existing_entries):
    return any(new_entry == entry for entry in existing_entries)


def parse_trace_df(df, op_col):
    traces = []
    for _, row in df.iterrows():
        entry = {"Name": row[op_col]}
        for col in df.columns:
            if col == op_col:
                continue
            val = row[col]
            entry[col] = str(val).strip() if pd.notna(val) else None
        traces.append(entry)
    return traces


def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


def main(op_file, trace_file, category_map, op_col, model_name, output_root):
    op_categories = load_op_categories(category_map)
    known_ops = load_known_ops(op_file)
    unknown_ops = set()
    uncategorized_ops = set()
    new = False

    # Load trace file
    try:
        if trace_file.endswith(".xlsx"):
            df = pd.read_excel(trace_file, sheet_name=1)
        elif trace_file.endswith(".csv"):
            df = pd.read_csv(trace_file)
        else:
            print(f"[!] Unsupported file type: {trace_file}")
            return
    except Exception as e:
        print(f"[!] Failed to read {trace_file}: {e}")
        return

    if op_col not in df.columns:
        print(f"[!] Column '{op_col}' not found in {trace_file}.")
        return

    trace_entries = parse_trace_df(df, op_col)

    for entry in tqdm(trace_entries, desc=f"Parsing {model_name}"):
        op = entry["Name"]
        base_op = op.lower()

        if base_op not in known_ops:
            unknown_ops.add(base_op)

        category_path = op_categories.get(base_op, base_op)
        if category_path == "":
            category_path = base_op
        if base_op not in op_categories and base_op not in uncategorized_ops:
            print(f"[!] WARNING: Op '{base_op}' not found in op_categories.json — placing in root folder.")
        op_folder = os.path.join(output_root, category_path)

        if base_op not in uncategorized_ops:
            uncategorized_ops.add(base_op)

        safe_mkdir(op_folder)
        json_path = os.path.join(op_folder, f"{model_name}.json")

        # Load existing
        existing = load_existing_variants(json_path)

        # If new, append and write back
        if not is_duplicate(entry, existing):
            new = True
            existing.append(entry)
            with open(json_path, "w") as f:
                json.dump(existing, f, indent=2)

    if not new:
        print("\n[!] No new traces found")

    if unknown_ops:
        print(f"\n[!] WARNING: The following ops were not in the known list:")
        for op in sorted(unknown_ops):
            print(f" - {op}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse a single trace file into categorized op JSONs.")
    parser.add_argument("--op_list", required=True, help="Path to .txt file with list of known ops (e.g., all_ops.txt)")
    parser.add_argument("--model_file", required=True, help="Path to a single model trace file (.csv or .xlsx)")
    parser.add_argument("--category_map", required=True, help="Path to op_categories.json")
    parser.add_argument("--op_col", required=True, help="Column name that contains the op name (e.g., 'Name')")
    parser.add_argument("--model_name", required=True, help="Model name to use for saving JSON")
    parser.add_argument("--output_root", default="parsed_traces", help="Output folder for trace JSONs")
    args = parser.parse_args()

    main(args.op_list, args.model_file, args.category_map, args.op_col, args.model_name, args.output_root)
