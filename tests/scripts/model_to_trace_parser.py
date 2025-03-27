import os
import json
import pandas as pd
import argparse
from collections import defaultdict


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


def parse_trace_df(df):
    return [
        {
            "Name": row["Name"],
            "Input Size": row["Input Size"],
            "Output Size": row["Output Size"],
            "Attributes": str(row["Attributes"]).strip() if pd.notna(row["Attributes"]) else None,
            "Count": row["Count"],
        }
        for _, row in df.iterrows()
    ]


def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


def main(op_file, model_dir, category_map, output_root="parsed_traces"):
    op_categories = load_op_categories(args.category_map)
    known_ops = load_known_ops(op_file)
    unknown_ops = set()
    uncategorized_ops = set()
    new = False

    for model_name in os.listdir(model_dir):
        model_path = os.path.join(model_dir, model_name)
        trace_file = os.path.join(model_path, "torchview_ops.xlsx")
        if not os.path.isfile(trace_file):
            print(f"[!] Skipping {model_name}: no torchview_ops.xlsx found")
            continue

        try:
            df = pd.read_excel(trace_file, sheet_name=1)
        except Exception as e:
            print(f"[!] Error reading {trace_file}: {e}")
            continue

        trace_entries = parse_trace_df(df)

        for entry in trace_entries:
            op = entry["Name"]
            base_op = op.lower()

            if base_op not in known_ops:
                unknown_ops.add(base_op)

            # Output path: parsed_traces/{base_op}/{model_name}.json
            category_path = op_categories.get(base_op, base_op)  # fallback to base_op if not categorized
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--op_list", required=True, help="Path to .txt file with list of known ops (e.g., all_ops.txt)")
    parser.add_argument("--model_dir", required=True, help="Path to directory containing model folders")
    parser.add_argument(
        "--category_map", required=True, help="Path to op_categories.json file mapping op names to category folders."
    )
    parser.add_argument("--output_root", default="parsed_traces", help="Directory to save parsed JSON traces")
    args = parser.parse_args()

    main(args.op_list, args.model_dir, args.category_map, args.output_root)
