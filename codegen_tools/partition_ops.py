#!/usr/bin/env python3
"""
Partition device_ops_to_process.txt into batches for parallel processing.

This script reads device_ops_to_process.txt and splits it into approximately
equal batches, outputting a JSON file that can be consumed by GitHub Actions
as a matrix input.
"""

import json
import sys
from pathlib import Path


def read_ops_file(file_path: Path) -> list[str]:
    """Read the operations file and extract file paths, ignoring status markers."""
    if not file_path.exists():
        print(f"Error: {file_path} not found", file=sys.stderr)
        sys.exit(1)

    ops = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Remove status markers like [ ], [/], [x]
            if line.startswith("["):
                # Extract the file path after the status marker (e.g., "[ ] ./path/to/file.hpp")
                # Find the first space after the closing bracket
                bracket_end = line.find("]")
                if bracket_end != -1 and bracket_end + 1 < len(line):
                    file_path = line[bracket_end + 1 :].strip()
                    if file_path:
                        ops.append(file_path)
            else:
                # No status marker, use the whole line
                ops.append(line)

    return ops


def partition_ops(ops: list[str], num_batches: int = 20) -> list[list[str]]:
    """Partition operations into approximately equal batches."""
    if num_batches <= 0:
        raise ValueError("num_batches must be positive")

    if not ops:
        return []

    batch_size = len(ops) // num_batches
    remainder = len(ops) % num_batches

    batches = []
    start_idx = 0

    for i in range(num_batches):
        # Distribute remainder across first batches
        current_batch_size = batch_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_batch_size
        batches.append(ops[start_idx:end_idx])
        start_idx = end_idx

    return batches


def main():
    """Main entry point."""
    repo_root = Path(__file__).parent.parent
    ops_file = repo_root / "device_ops_to_process.txt"
    output_file = repo_root / "batch_mapping.json"

    # Read operations
    ops = read_ops_file(ops_file)
    print(f"Found {len(ops)} operations to partition", file=sys.stderr)

    # Partition into batches
    num_batches = 20
    batches = partition_ops(ops, num_batches)

    # Create matrix structure for GitHub Actions
    matrix = {"batch_id": [], "files": []}

    for i, batch in enumerate(batches, start=1):
        batch_id = f"{i:02d}"  # Zero-padded: 01, 02, ..., 20
        matrix["batch_id"].append(batch_id)
        matrix["files"].append(batch)

    # Output JSON
    with open(output_file, "w") as f:
        json.dump(matrix, f, indent=2)

    print(f"Created {len(batches)} batches", file=sys.stderr)
    print(f"Batch sizes: {[len(b) for b in batches]}", file=sys.stderr)
    print(f"Output written to {output_file}", file=sys.stderr)

    # Also print summary
    print("\nBatch Summary:", file=sys.stderr)
    for i, batch in enumerate(batches, start=1):
        print(f"  Batch {i:02d}: {len(batch)} files", file=sys.stderr)


if __name__ == "__main__":
    main()
