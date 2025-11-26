#!/usr/bin/env python3
"""
Script to verify the integrity of compute_kernel_api_reference.md
Checks if all documented functions actually exist in the codebase.
"""

import re
import os
from pathlib import Path
from typing import List, Tuple, Dict

# Base directory for compute kernel API
API_DIR = Path("/proj_sw/user_dev/njokovic/tt-metal/tt_metal/include/compute_kernel_api")
INCLUDE_DIR = Path("/proj_sw/user_dev/njokovic/tt-metal/tt_metal/include")
REFERENCE_FILE = Path("/proj_sw/user_dev/njokovic/tt-metal/docs/compute_kernel_api_reference.md")


def extract_function_name(func_signature: str) -> str:
    """Extract function name from signature like 'func_name<template>(params)' or 'func_name(params)'"""
    # Remove template parameters
    func = re.sub(r"<[^>]+>", "", func_signature)
    # Extract function name (before first parenthesis)
    match = re.match(r"`?([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", func)
    if match:
        return match.group(1)
    # Handle cases without parentheses
    match = re.match(r"`?([a-zA-Z_][a-zA-Z0-9_]*)", func)
    if match:
        return match.group(1)
    return func.strip("`").split("(")[0].strip()


def find_function_in_codebase(func_name: str) -> List[Tuple[str, int]]:
    """Search for function definition in codebase"""
    results = []
    # Search in API directory and also in compute_kernel_api.h
    search_dirs = [API_DIR, INCLUDE_DIR / "compute_kernel_api.h"]

    for search_dir in search_dirs:
        if search_dir.is_file():
            # Single file
            files_to_check = [search_dir]
        else:
            # Directory
            files_to_check = list(search_dir.rglob("*.h"))

        for filepath in files_to_check:
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    lines = content.split("\n")

                    # Look for function definitions - handle multi-line templates
                    rel_path = str(filepath.relative_to(INCLUDE_DIR.parent.parent))
                    found_in_file = False

                    for i, line in enumerate(lines):
                        # Check for function name followed by opening parenthesis
                        # This handles both single-line and multi-line definitions
                        if re.search(rf"\b{re.escape(func_name)}\s*\(", line):
                            # Check if this line or previous lines contain function definition keywords
                            # Look back a few lines for template/ALWI/FORCE_INLINE keywords
                            context_start = max(0, i - 3)
                            context = "\n".join(lines[context_start : i + 1])

                            # Check various function definition patterns
                            patterns = [
                                rf"template\s+<[^>]+>\s*.*?\b{re.escape(func_name)}\s*\(",
                                rf"ALWI\s+.*?\b{re.escape(func_name)}\s*\(",
                                rf"FORCE_INLINE\s+.*?\b{re.escape(func_name)}\s*\(",
                                rf"static\s+FORCE_INLINE\s+.*?\b{re.escape(func_name)}\s*\(",
                                rf"inline\s+.*?\b{re.escape(func_name)}\s*\(",
                                rf"void\s+{re.escape(func_name)}\s*\(",
                                rf"[a-zA-Z_][a-zA-Z0-9_<>:&\s*]+\s+{re.escape(func_name)}\s*\(",
                            ]

                            for pattern in patterns:
                                if re.search(pattern, context, re.DOTALL | re.MULTILINE):
                                    results.append((rel_path, i + 1))
                                    found_in_file = True
                                    break

                            if found_in_file:
                                break  # Found in this file
            except Exception as e:
                pass  # Skip files we can't read
    return results


def parse_reference_file() -> Dict[str, List[str]]:
    """Parse the reference file and extract all function signatures"""
    with open(REFERENCE_FILE, "r") as f:
        content = f.read()

    # Extract functions from markdown tables
    # Pattern: | `function_name<...>(...)` | Description |
    pattern = r"\|\s*`([^`]+)`\s*\|\s*[^|]+\s*\|"
    matches = re.findall(pattern, content)

    all_functions = []
    seen = set()

    for match in matches:
        func_sig = match.strip()

        # Skip file names (they end with .h and are in the "Other eltwise_unary/ Functions" section)
        if func_sig.endswith(".h"):
            continue

        # Skip entries that are just file names without function signatures
        if func_sig in [
            "isinf_isnan.h",
            "identity.h",
            "negative.h",
            "logical_not_noti.h",
            "log1p.h",
            "left_shift.h",
            "right_shift.h",
            "remainder.h",
            "fmod.h",
            "prelu.h",
            "rand.h",
            "i0.h",
            "i1.h",
            "erfinv.h",
            "erf_erfc.h",
            "elu.h",
            "dropout.h",
            "clamp.h",
            "hardtanh.h",
            "hardmish.h",
            "rpow.h",
            "threshold.h",
            "selu.h",
            "cbrt.h",
            "reverseops.h",
            "sfpu_int_sum.h",
        ]:
            continue

        func_name = extract_function_name(func_sig)
        if func_name and func_name not in seen:
            all_functions.append((func_sig, func_name))
            seen.add(func_name)

    return all_functions


def main():
    print("=" * 80)
    print("Compute Kernel API Reference Integrity Check")
    print("=" * 80)
    print()

    functions = parse_reference_file()
    print(f"Found {len(functions)} function signatures in reference document")
    print()

    missing = []
    found = []
    issues = []

    for func_sig, func_name in functions:
        results = find_function_in_codebase(func_name)
        if results:
            found.append((func_sig, func_name, results))
        else:
            missing.append((func_sig, func_name))

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total functions checked: {len(functions)}")
    print(f"Found: {len(found)}")
    print(f"Missing: {len(missing)}")
    print()

    if missing:
        print("=" * 80)
        print("MISSING FUNCTIONS")
        print("=" * 80)
        for func_sig, func_name in missing:
            print(f"  ❌ {func_sig}")
            print(f"     Function name: {func_name}")
            print()

    # Check for potential issues (functions found but signature might differ)
    print("=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)

    return len(missing) == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
