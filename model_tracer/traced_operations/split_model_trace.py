#!/usr/bin/env python3
"""Split a model trace JSON into one file per operation.

Each output file preserves the full schema (operations + metadata) but
contains only a single operation.  Output directory layout:

    <output_dir>/<safe_op_name>/ttnn_operations_master.json

where <safe_op_name> replaces dots with underscores after the leading
"ttnn." prefix (e.g. "ttnn.experimental.all_gather_async" becomes
"ttnn_experimental_all_gather_async").
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def safe_op_dirname(op_name: str) -> str:
    """Convert 'ttnn.foo.bar' -> 'ttnn_foo_bar' for filesystem safety."""
    return op_name.replace(".", "_")


def split_model_trace(input_path: str, output_dir: str | None = None) -> None:
    input_path = Path(input_path).resolve()
    if not input_path.is_file():
        print(f"Error: {input_path} does not exist", file=sys.stderr)
        sys.exit(1)

    if output_dir is None:
        output_dir = input_path.parent / f"{input_path.stem}_split"
    else:
        output_dir = Path(output_dir).resolve()

    with open(input_path) as f:
        data = json.load(f)

    operations = data.get("operations", {})
    metadata = data.get("metadata", {})

    if not operations:
        print("No operations found in the trace file.", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for op_name, op_data in operations.items():
        configs = op_data.get("configurations", [])
        num_configs = len(configs)

        per_op = {
            "operations": {
                op_name: op_data,
            },
            "metadata": {
                "models": metadata.get("models", []),
                "unique_operations": 1,
                "total_configurations": num_configs,
                "operations_summary": {
                    op_name: num_configs,
                },
                "source_file": input_path.name,
                "last_updated": timestamp,
            },
        }

        dirname = safe_op_dirname(op_name)
        op_dir = output_dir / dirname
        op_dir.mkdir(parents=True, exist_ok=True)

        out_file = op_dir / "ttnn_operations_master.json"
        with open(out_file, "w") as f:
            json.dump(per_op, f, indent=2)

        print(f"  {op_name:<55} {num_configs:>4} configs -> {out_file}")

    print(f"\nSplit {len(operations)} operations into {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Split a model trace JSON into one file per operation.")
    parser.add_argument(
        "input",
        help="Path to the model trace JSON (ttnn_operations_master*.json)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output directory (default: <input_stem>_split/ next to input file)",
    )
    args = parser.parse_args()
    split_model_trace(args.input, args.output_dir)


if __name__ == "__main__":
    main()
