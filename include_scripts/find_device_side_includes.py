#!/usr/bin/env python3
"""
Find all device-side source files and extract their include paths.

Device-side files are identified by the presence of entry points:
- void kernel_main()  -- dataflow kernels
- void MAIN           -- compute kernels (within namespace NAMESPACE)

Output: JSON with list of device-side files and their includes.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import NamedTuple


class DeviceSideFile(NamedTuple):
    path: str
    kernel_type: str  # "dataflow" or "compute"
    includes: list[str]


def is_device_side_file(content: str) -> str | None:
    """
    Check if file content indicates a device-side kernel.
    Returns kernel type ("dataflow" or "compute") or None.
    """
    # Dataflow kernels have: void kernel_main()
    if re.search(r'\bvoid\s+kernel_main\s*\(\s*\)', content):
        return "dataflow"

    # Compute kernels have: void MAIN (macro that expands to function)
    if re.search(r'\bvoid\s+MAIN\b', content):
        return "compute"

    return None


def extract_includes(content: str) -> list[str]:
    """Extract all #include paths from file content."""
    includes = []
    for match in re.finditer(r'^\s*#include\s*[<"]([^>"]+)[>"]', content, re.MULTILINE):
        includes.append(match.group(1))
    return includes


def find_device_side_files(root_dir: Path, exclude_dirs: list[str] = None) -> list[DeviceSideFile]:
    """
    Recursively find all device-side source files under root_dir.
    """
    exclude_dirs = exclude_dirs or ["build_", ".git", "__pycache__"]
    results = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip excluded directories
        dirnames[:] = [d for d in dirnames if not any(excl in d for excl in exclude_dirs)]

        for filename in filenames:
            if not filename.endswith(('.cpp', '.cc')):
                continue

            filepath = Path(dirpath) / filename
            relative_path = str(filepath.relative_to(root_dir))

            # Skip jit_build/ directory - it's host-side build system code
            # that contains kernel entry point signatures in comments/strings
            if relative_path.startswith('tt_metal/jit_build/'):
                continue

            try:
                content = filepath.read_text(encoding='utf-8', errors='replace')
            except Exception as e:
                print(f"Warning: Could not read {filepath}: {e}", file=sys.stderr)
                continue

            kernel_type = is_device_side_file(content)
            if kernel_type:
                includes = extract_includes(content)
                results.append(DeviceSideFile(
                    path=relative_path,
                    kernel_type=kernel_type,
                    includes=includes,
                ))

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root_dir",
        nargs="?",
        default=".",
        help="Root directory to search (default: current directory)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "summary"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file (default: stdout)",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    print(f"Searching for device-side files in: {root_dir}", file=sys.stderr)

    results = find_device_side_files(root_dir)
    print(f"Found {len(results)} device-side files", file=sys.stderr)

    if args.format == "json":
        output = [
            {
                "path": r.path,
                "kernel_type": r.kernel_type,
                "includes": r.includes,
            }
            for r in results
        ]
        output_str = json.dumps(output, indent=2)
    else:
        # Summary format
        all_includes = {}
        for r in results:
            for inc in r.includes:
                if inc not in all_includes:
                    all_includes[inc] = 0
                all_includes[inc] += 1

        lines = [
            f"Device-side files: {len(results)}",
            f"  Dataflow kernels: {sum(1 for r in results if r.kernel_type == 'dataflow')}",
            f"  Compute kernels: {sum(1 for r in results if r.kernel_type == 'compute')}",
            f"",
            f"Unique includes: {len(all_includes)}",
            f"",
            f"Top includes by frequency:",
        ]
        for inc, count in sorted(all_includes.items(), key=lambda x: -x[1])[:30]:
            lines.append(f"  {count:4d}  {inc}")
        output_str = "\n".join(lines)

    if args.output:
        Path(args.output).write_text(output_str)
        print(f"Output written to: {args.output}", file=sys.stderr)
    else:
        print(output_str)


if __name__ == "__main__":
    main()
