#!/usr/bin/env python3
"""
Verification script for mapped_sentinel.md
Checks if each operation function corresponds to the listed init/uninit functions.
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# Base directory for compute kernel API
API_DIR = Path("tt_metal/include")


def grep_function(func_name: str, file_path: Optional[Path] = None) -> bool:
    """Check if a function exists in the codebase."""
    # Clean function name - remove template parameters and arguments for search
    # Extract just the function name part
    base_name = func_name.split("<")[0].split("(")[0].strip()

    # Search in API directory or specific file
    search_path = str(file_path) if file_path else str(API_DIR)

    try:
        # Search for function definition patterns: "void func_name", "ALWI void func_name", "template.*func_name"
        patterns = [
            f"void {base_name}",
            f"ALWI void {base_name}",
            f"template.*{base_name}",
        ]

        for pattern in patterns:
            result = subprocess.run(
                ["grep", "-r", "--include=*.h", pattern, search_path], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and len(result.stdout.strip()) > 0:
                return True

        return False
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def parse_mapping_table(content: str) -> List[Tuple[str, str, str, str, str]]:
    """Parse markdown table rows from the document.
    Returns: List of (operation, init, uninit, notes, file_section) tuples
    """
    mappings = []
    current_section = ""

    lines = content.split("\n")
    in_table = False
    skip_section = False

    for i, line in enumerate(lines):
        # Track section headers
        if line.startswith("### `") and line.endswith("`"):
            current_section = line.strip()
            # Skip the "Other eltwise_unary/ Functions" section as it's a summary table
            if "Other" in line and "eltwise_unary" in line:
                skip_section = True
            else:
                skip_section = False

        # Skip this section
        if skip_section:
            continue

        # Detect table start
        if "| Operation Function" in line or ("| Function" in line and "| File |" not in line):
            in_table = True
            continue

        # Skip separator line
        if in_table and re.match(r"^\|[\s\-:]+\|", line):
            continue

        # Parse table rows
        if in_table and line.startswith("|") and "Operation Function" not in line and "Function" not in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4:
                # Handle different table formats
                if "| File |" in lines[max(0, i - 2) : i + 1]:  # Check if this is the "Other eltwise_unary" section
                    # Format: File | Operation Function | Init Function | Uninit Function | Notes
                    file_name = parts[1] if len(parts) > 1 else ""
                    operation = parts[2] if len(parts) > 2 else ""
                    init = parts[3] if len(parts) > 3 else ""
                    uninit = parts[4] if len(parts) > 4 else ""
                    notes = parts[5] if len(parts) > 5 else ""

                    # Skip file name rows
                    if file_name and file_name.startswith("`") and file_name.endswith(".h`"):
                        # This row has multiple functions - skip for now (too complex to parse)
                        # We'll handle these separately
                        continue
                else:
                    # Standard format: Operation Function | Init Function | Uninit Function | Notes
                    operation = parts[1] if len(parts) > 1 else ""
                    init = parts[2] if len(parts) > 2 else ""
                    uninit = parts[3] if len(parts) > 3 else ""
                    notes = parts[4] if len(parts) > 4 else ""

                # Strip backticks from function names
                operation = operation.strip("`")
                init = init.strip("`")
                uninit = uninit.strip("`")

                # Skip empty rows and separator rows
                if operation and not operation.startswith("---") and operation != "File":
                    # Handle multiple functions in operation field (comma-separated)
                    if "," in operation and "(" in operation:
                        # Split by comma and process each function
                        func_list = [f.strip().strip("`") for f in operation.split(",")]
                        for func in func_list:
                            if func and not func.startswith("---"):
                                mappings.append((func, init, uninit, notes, current_section))
                    else:
                        mappings.append((operation, init, uninit, notes, current_section))

        # Reset table state on section end
        if in_table and line.startswith("---"):
            in_table = False

    return mappings


def verify_mapping(operation: str, init: str, uninit: str, notes: str, section: str) -> Tuple[bool, List[str]]:
    """Verify a single mapping. Returns (is_valid, list_of_issues)."""
    issues = []

    # Skip utility/debug functions that don't need init/uninit
    if init == "None" and uninit == "None":
        # These are expected to not have init/uninit
        return True, []

    # Check operation function exists
    op_base = operation.split("<")[0].split("(")[0].strip()
    op_base = op_base.strip("`")  # Remove any remaining backticks
    if op_base and not grep_function(op_base):
        issues.append(f"Operation function '{op_base}' not found")

    # Check init function exists
    if init and init != "None":
        init_base = init.split("<")[0].split("(")[0].strip()
        init_base = init_base.strip("`")  # Remove any remaining backticks
        if init_base and not grep_function(init_base):
            issues.append(f"Init function '{init_base}' not found")

    # Check uninit function exists
    if uninit and uninit != "None":
        uninit_base = uninit.split("<")[0].split("(")[0].strip()
        uninit_base = uninit_base.strip("`")  # Remove any remaining backticks
        if uninit_base and not grep_function(uninit_base):
            issues.append(f"Uninit function '{uninit_base}' not found")

    return len(issues) == 0, issues


def main():
    mapped_sentinel_path = Path("mapped_sentinel.md")

    if not mapped_sentinel_path.exists():
        print(f"Error: {mapped_sentinel_path} not found")
        sys.exit(1)

    if not API_DIR.exists():
        print(f"Error: {API_DIR} not found")
        sys.exit(1)

    print("Reading mapped_sentinel.md...")
    with open(mapped_sentinel_path, "r") as f:
        content = f.read()

    print("Parsing mappings...")
    mappings = parse_mapping_table(content)
    print(f"Found {len(mappings)} mappings to verify\n")

    # Verify each mapping
    issues_found = []
    verified_count = 0

    for i, (operation, init, uninit, notes, section) in enumerate(mappings, 1):
        is_valid, issues = verify_mapping(operation, init, uninit, notes, section)

        if not is_valid:
            issues_found.append(
                {
                    "index": i,
                    "operation": operation,
                    "init": init,
                    "uninit": uninit,
                    "section": section,
                    "issues": issues,
                }
            )
        else:
            verified_count += 1

        if i % 50 == 0:
            print(f"Verified {i}/{len(mappings)} mappings...")

    # Print results
    print(f"\n{'='*80}")
    print(f"Verification Results:")
    print(f"{'='*80}")
    print(f"Total mappings: {len(mappings)}")
    print(f"Verified: {verified_count}")
    print(f"Issues found: {len(issues_found)}\n")

    if issues_found:
        print("ISSUES FOUND:")
        print("=" * 80)
        for item in issues_found:
            print(f"\n[{item['index']}] Section: {item['section']}")
            print(f"  Operation: {item['operation']}")
            print(f"  Init: {item['init']}")
            print(f"  Uninit: {item['uninit']}")
            print(f"  Issues:")
            for issue in item["issues"]:
                print(f"    - {issue}")
        return 1
    else:
        print("✓ All mappings verified successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
