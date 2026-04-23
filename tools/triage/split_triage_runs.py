#!/usr/bin/env python3
"""
Split multi-invocation triage output files into per-run files.

Each triage invocation starts with 'dump_configuration.py:'.
Saves split files to triage_outputs_split/ as {job_id}_run{N}.txt
Also builds an updated index with run number metadata.
"""

import json
import re
from pathlib import Path

INPUT_DIR = Path(__file__).resolve().parents[2] / "triage_outputs"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "triage_outputs_split"


def split_file(text):
    """Split a triage output into individual runs based on dump_configuration.py: markers."""
    # Find all positions where a new triage run starts
    marker = "dump_configuration.py:"
    positions = [m.start() for m in re.finditer(r"^dump_configuration\.py:", text, re.MULTILINE)]

    if not positions:
        return [text]  # No marker found, return as-is (probably a crash/no-triage file)

    runs = []
    for i, pos in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        runs.append(text[pos:end])

    return runs


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    index = json.load(open(INPUT_DIR / "index.json"))
    split_index = {}

    total_runs = 0
    for job_id, meta in index.items():
        input_file = INPUT_DIR / f"{job_id}.txt"
        if not input_file.exists():
            continue

        text = input_file.read_text()
        if text.strip() == "[NO TRIAGE SECTION FOUND]":
            # Copy as-is
            out = OUTPUT_DIR / f"{job_id}_run1.txt"
            out.write_text(text)
            split_index[f"{job_id}_run1"] = {
                **meta,
                "run_number": 1,
                "total_runs": 0,
                "original_job_id": job_id,
            }
            continue

        runs = split_file(text)
        for i, run_text in enumerate(runs):
            run_num = i + 1
            key = f"{job_id}_run{run_num}"
            out = OUTPUT_DIR / f"{key}.txt"
            out.write_text(run_text)
            split_index[key] = {
                **meta,
                "run_number": run_num,
                "total_runs": len(runs),
                "original_job_id": job_id,
                "has_output": True,
            }
            total_runs += 1

    with open(OUTPUT_DIR / "index.json", "w") as f:
        json.dump(split_index, f, indent=2)

    # Stats
    first_runs = sum(1 for v in split_index.values() if v.get("run_number") == 1)
    nth_runs = sum(1 for v in split_index.values() if v.get("run_number", 1) > 1)
    print(f"Total split files: {total_runs}")
    print(f"First runs: {first_runs}")
    print(f"Subsequent runs: {nth_runs}")


if __name__ == "__main__":
    main()
