#!/usr/bin/env python3
"""
Aggregate Tracy profiler CSV reports and generate a summary of device operations.

This script:
1. Searches for ops_perf_results_*.csv files in the reports directory
2. Extracts operation names from the OP CODE column
3. Aggregates operations by model
4. Generates text, JSON, and GitHub Summary reports
"""

import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path


def extract_model_name(csv_path: Path) -> str:
    """Extract model name from CSV filename or parent directory."""
    filename = csv_path.stem  # e.g., ops_perf_results_resnet_N150
    match = re.match(r'ops_perf_results_(.+)$', filename)
    return match.group(1) if match else csv_path.parent.name


def parse_csv_operations(csv_path: Path) -> set:
    """Parse a Tracy CSV and return set of unique operation names."""
    ops = set()
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                op_name = row.get("OP CODE", row.get("op_code", ""))
                if op_name:
                    ops.add(op_name)
    except Exception as e:
        print(f"Error parsing {csv_path}: {e}")
    return ops


def generate_text_report(op_models: dict, output_path: Path) -> Path:
    """Generate a text report of operations by model."""
    report_path = output_path / "aggregated_operations.txt"
    with open(report_path, 'w') as f:
        f.write("Device Operations Report\n")
        f.write("=" * 60 + "\n\n")
        f.write("Operations by Model\n")
        f.write("-" * 60 + "\n")
        for op_name, models in sorted(op_models.items()):
            f.write(f"\n{op_name}:\n")
            for model in sorted(models):
                f.write(f"  - {model}\n")
    return report_path


def generate_json_report(op_models: dict, output_path: Path) -> Path:
    """Generate a JSON report of operations by model."""
    report_path = output_path / "aggregated_operations.json"
    with open(report_path, 'w') as f:
        json.dump({
            "by_operation": {op: sorted(list(models)) for op, models in sorted(op_models.items())}
        }, f, indent=2)
    return report_path


def generate_github_summary(op_models: dict, model_op_counts: dict):
    """Generate GitHub Actions job summary."""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY", "")
    if not summary_path:
        return

    with open(summary_path, "w") as f:
        f.write("# 📊 Device Operations Report\n\n")

        if not op_models:
            f.write("⚠️ **No operations data found.** Tracy profiling may have failed for all models.\n\n")
            return

        # Operations table
        f.write("## 🔧 Operations by Model\n\n")
        f.write("| Operation | Models | Count |\n")
        f.write("|-----------|--------|-------|\n")
        for op_name, models in sorted(op_models.items()):
            models_str = ", ".join(f"`{m}`" for m in sorted(models))
            f.write(f"| `{op_name}` | {models_str} | {len(models)} |\n")
        f.write("\n")

        # Models summary
        f.write("## 📦 Models Summary\n\n")
        f.write("| Model | Unique Operations |\n")
        f.write("|-------|------------------|\n")
        for model, count in sorted(model_op_counts.items()):
            f.write(f"| `{model}` | {count} |\n")
        f.write("\n")

        # Detailed breakdown (collapsible)
        f.write("<details>\n<summary>📋 Detailed Operations List</summary>\n\n")
        for op_name, models in sorted(op_models.items()):
            f.write(f"### `{op_name}`\n")
            f.write("Models using this operation:\n")
            for model in sorted(models):
                f.write(f"- {model}\n")
            f.write("\n")
        f.write("</details>\n")


def set_github_output(name: str, value: str):
    """Set a GitHub Actions output variable."""
    output_file = os.environ.get("GITHUB_OUTPUT", "")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"{name}={value}\n")


def main():
    # Get configuration from environment
    reports_path = Path(os.environ.get("REPORTS_PATH", "all_tracy_reports"))
    output_path = Path(os.environ.get("OUTPUT_PATH", "aggregated_ops"))

    print(f"Searching for Tracy reports in: {reports_path}")
    print(f"Output directory: {output_path}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Aggregate operations from all CSV files
    op_models = defaultdict(set)  # operation -> set of models
    model_op_counts = {}  # model -> count of unique ops

    csv_files = list(reports_path.rglob("ops_perf_results_*.csv"))
    print(f"Found {len(csv_files)} CSV files")

    for csv_path in csv_files:
        model_name = extract_model_name(csv_path)
        ops = parse_csv_operations(csv_path)

        if ops:
            model_op_counts[model_name] = len(ops)
            for op in ops:
                op_models[op].add(model_name)
        else:
            print(f"  {model_name}: No operations found")

    # Generate reports
    print("\nGenerating reports...")
    text_report = generate_text_report(op_models, output_path)
    json_report = generate_json_report(op_models, output_path)
    generate_github_summary(op_models, model_op_counts)

    # Set GitHub outputs
    set_github_output("json-report", str(json_report))
    set_github_output("text-report", str(text_report))
    set_github_output("operations-count", str(len(op_models)))
    set_github_output("models-count", str(len(model_op_counts)))

    print(f"  Text report: {text_report}")
    print(f"  JSON report: {json_report}")

    # Print text report to stdout
    print("\n" + "=" * 60)
    with open(text_report, 'r') as f:
        print(f.read())


if __name__ == "__main__":
    main()

