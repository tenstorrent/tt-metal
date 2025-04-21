# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import json
import argparse


def get_op_name(filename):
    # delete '_pytorch2' if within filename
    if "_pytorch2" in filename:
        filename = filename.replace("_pytorch2", "")
    return os.path.splitext(filename)[0]  # remove .py


def generate_op_category_map(root_dir):
    op_category_map = {}

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                full_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(full_path, root_dir)
                category = os.path.dirname(rel_path).replace(os.path.sep, "/")
                op_name = get_op_name(filename)
                op_category_map[op_name] = category

        for dirname in dirnames:
            # Check if the directory is a leaf (has no subdirs or .py files inside)
            sub_path = os.path.join(dirpath, dirname)
            if not any(os.path.isdir(os.path.join(sub_path, x)) or x.endswith(".py") for x in os.listdir(sub_path)):
                rel_path = os.path.relpath(sub_path, root_dir)
                category = os.path.dirname(rel_path).replace(os.path.sep, "/")
                op_category_map[dirname] = category

    return op_category_map


def main(sweeps_dir, output_file):
    op_map = generate_op_category_map(sweeps_dir)

    with open(output_file, "w") as f:
        json.dump(op_map, f, indent=2)

    print(f"[+] Operation category map written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate operation -> category mapping from directory structure.")
    parser.add_argument(
        "--sweeps_dir", required=True, help="Top-level directory containing categorized ops (e.g. sweeps/)"
    )
    parser.add_argument("--output_file", default="op_category_map.json", help="Path to save the output JSON map.")
    args = parser.parse_args()

    main(args.sweeps_dir, args.output_file)
