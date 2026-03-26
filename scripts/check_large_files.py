#!/usr/bin/env python3
"""
Pre-commit hook: flag staged files that exceed a size threshold.

Large files bloat git history permanently — every clone must download them forever.
Store binary assets, model weights, and generated data outside the repository.

Exit code 1 (fail) if any file exceeds FAIL_THRESHOLD_MB.
Exit code 0 (warn only) if any file is between WARN_THRESHOLD_MB and FAIL_THRESHOLD_MB.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

WARN_THRESHOLD_MB = 0.5
FAIL_THRESHOLD_MB = 1

# Extensions that are almost certainly not source code and should live elsewhere
BINARY_ASSET_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",  # video
    ".pth",
    ".pt",
    ".onnx",
    ".bin",
    ".pkl",  # model weights
    ".so",
    ".dylib",
    ".dll",
    ".a",  # compiled binaries
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",  # archives
    ".parquet",
    ".hdf5",
    ".h5",  # data files
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",  # images
    ".ipynb",  # notebooks (often embed large outputs)
}


def get_staged_files():
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [f for f in result.stdout.splitlines() if f]


def get_staged_file_size(filepath):
    """Get the size of the staged (index) version of a file, not the working tree."""
    result = subprocess.run(
        ["git", "cat-file", "-s", f":0:{filepath}"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return int(result.stdout.strip())
    # Fallback: working tree size (e.g. for newly added untracked files)
    try:
        return os.path.getsize(filepath)
    except OSError:
        return 0


def format_size(size_bytes):
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / 1024 / 1024:.2f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def suggestion(filepath):
    ext = Path(filepath).suffix.lower()
    if ext in BINARY_ASSET_EXTENSIONS:
        return (
            f"  → Binary asset ({ext}). Store this file outside the repository\n"
            f"    (e.g. an artifact server, object storage, or documentation hosting)\n"
            f"    and reference it by URL instead."
        )
    if "traced_operations" in filepath or filepath.endswith(".json"):
        return (
            "  → Looks like generated data. Publish it as a CI artifact instead\n"
            "    of committing it to the repository."
        )
    if ext == ".ipynb":
        return (
            "  → Jupyter notebook with large embedded outputs. Strip outputs before\n"
            "    committing: nbstripout <file> or add nbstripout to pre-commit hooks."
        )
    return (
        "  → Store large files outside the repository and reference them by URL.\n"
        "    Every byte committed here is downloaded by every future clone, forever."
    )


def main():
    parser = argparse.ArgumentParser(description="Flag large staged files")
    parser.add_argument(
        "--maxmb", type=float, default=FAIL_THRESHOLD_MB, help=f"Fail threshold in MB (default: {FAIL_THRESHOLD_MB})"
    )
    parser.add_argument(
        "--warnmb", type=float, default=WARN_THRESHOLD_MB, help=f"Warn threshold in MB (default: {WARN_THRESHOLD_MB})"
    )
    args, _ = parser.parse_known_args()

    fail_bytes = int(args.maxmb * 1024 * 1024)
    warn_bytes = int(args.warnmb * 1024 * 1024)

    staged = get_staged_files()
    warnings = []
    failures = []

    for filepath in staged:
        size = get_staged_file_size(filepath)
        if size >= fail_bytes:
            failures.append((filepath, size))
        elif size >= warn_bytes:
            warnings.append((filepath, size))

    if warnings:
        print(f"\n⚠️  Large files staged (>{args.warnmb*1024:.0f} KB) — consider alternatives:\n")
        for filepath, size in warnings:
            print(f"  {format_size(size):>10}  {filepath}")
            print(suggestion(filepath))
            print()

    if failures:
        print(f"\n❌ Files exceeding the {args.maxmb:.0f} MB limit are blocked from commit:\n")
        for filepath, size in failures:
            print(f"  {format_size(size):>10}  {filepath}")
            print(suggestion(filepath))
            print()
        print(
            "Large files bloat git history permanently — every clone must download\n"
            "them forever. Store binary assets outside the repo and reference by URL.\n"
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
