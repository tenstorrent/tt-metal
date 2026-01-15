#!/usr/bin/env python3
"""
Fuse multiple LLK (Low-Level Kernel) calls together to improve performance.

This tool identifies sequences of SFPU/FPU operations and fuses them by:
1. Removing intermediate synchronization calls (tile_regs_commit, tile_regs_wait, cb_pop_front, cb_push_back)
2. Chaining operations together to keep intermediate values in registers
3. Keeping only the final synchronization operations

Usage:
    python fuse_llk_calls.py <input_file> [--output <output_file>] [--dry-run]

Example:
    python fuse_llk_calls.py kernels/compute/my_kernel.cpp --output kernels/compute/my_kernel_fused.cpp
"""

import re
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class LLKOperation:
    """Represents an LLK operation call"""

    init_call: Optional[str] = None
    operation_calls: List[str] = None
    done_call: Optional[str] = None
    line_start: int = 0
    line_end: int = 0

    def __post_init__(self):
        if self.operation_calls is None:
            self.operation_calls = []


# Patterns for identifying LLK operations
SFPU_INIT_PATTERNS = [
    r"(\w+_tile_init\s*\([^)]*\);)",  # e.g., exp_tile_init(), log_tile_init()
    r"(\w+_init\s*\([^)]*\);)",  # e.g., add_binary_tile_init()
    r"(llk_math_eltwise_unary_sfpu_init[^;]+;)",
    r"(llk_math_eltwise_unary_sfpi_init[^;]+;)",
    r"(_llk_math_eltwise_unary_sfpu_init_[^;]+;)",
]

SFPU_OP_PATTERNS = [
    r"(\w+_tile\s*\([^)]*\);)",  # e.g., exp_tile(0), log_tile(0)
    r"(\w+_binary_tile\s*\([^)]*\);)",  # e.g., add_binary_tile(0, 1, 0)
    r"(llk_math_eltwise_unary_sfpu[^;]+;)",
    r"(llk_math_eltwise_unary_sfpi[^;]+;)",
    r"(_llk_math_eltwise_unary_sfpu_[^;]+;)",
    r"(_llk_math_eltwise_unary_sfpu_start_[^;]+;)",
    r"(call_sfpu_operation[^;]+;)",
]

SFPU_DONE_PATTERNS = [
    r"(_llk_math_eltwise_unary_sfpu_done\s*\([^)]*\);)",
    r"(_llk_math_eltwise_unary_sfpu_done_\s*\(\);)",
]

# Patterns for synchronization calls that can be removed when fusing
SYNC_PATTERNS = [
    r"tile_regs_commit\s*\(\);",
    r"tile_regs_wait\s*\(\);",
    r"cb_pop_front\s*\([^)]+\);",
    r"cb_push_back\s*\([^)]+\);",
    r"_llk_math_dest_section_done_[^;]+;",
]

# Patterns that should be preserved (final synchronization)
FINAL_SYNC_PATTERNS = [
    r"cb_reserve_back\s*\([^)]+\);",
    r"pack_tile\s*\([^)]+\);",
    r"cb_push_back\s*\([^)]+\);",
    r"tile_regs_release\s*\(\);",
]


def find_llk_operations(lines: List[str]) -> List[LLKOperation]:
    """Find all LLK operations in the code"""
    operations = []
    current_op = None
    in_operation = False

    for i, line in enumerate(lines):
        # Check for init calls
        for pattern in SFPU_INIT_PATTERNS:
            match = re.search(pattern, line)
            if match:
                if current_op is None:
                    current_op = LLKOperation(line_start=i)
                current_op.init_call = line.strip()
                in_operation = True
                break

        # Check for operation calls
        if in_operation:
            for pattern in SFPU_OP_PATTERNS:
                match = re.search(pattern, line)
                if match:
                    if current_op is None:
                        current_op = LLKOperation(line_start=i)
                    current_op.operation_calls.append(line.strip())
                    break

        # Check for done calls
        for pattern in SFPU_DONE_PATTERNS:
            match = re.search(pattern, line)
            if match:
                if current_op is not None:
                    current_op.done_call = line.strip()
                    current_op.line_end = i
                    operations.append(current_op)
                    current_op = None
                    in_operation = False
                break

        # Check if we hit a synchronization call that ends the operation
        if in_operation:
            for pattern in SYNC_PATTERNS:
                if re.search(pattern, line):
                    # Check if this is a final sync (should preserve) or intermediate (can remove)
                    is_final = any(re.search(p, line) for p in FINAL_SYNC_PATTERNS)
                    if not is_final and current_op is not None:
                        # End current operation
                        current_op.line_end = i
                        operations.append(current_op)
                        current_op = None
                        in_operation = False
                    break

    # Add any remaining operation
    if current_op is not None and current_op.operation_calls:
        current_op.line_end = len(lines) - 1
        operations.append(current_op)

    return operations


def find_sync_calls(lines: List[str], start_line: int, end_line: int) -> List[Tuple[int, str, bool]]:
    """Find synchronization calls in a range of lines.
    Returns list of (line_number, call, is_final) tuples.
    """
    sync_calls = []

    for i in range(start_line, min(end_line + 1, len(lines))):
        line = lines[i]

        # Check if it's a final sync call
        is_final = any(re.search(p, line) for p in FINAL_SYNC_PATTERNS)

        # Check if it's any sync call
        for pattern in SYNC_PATTERNS:
            if re.search(pattern, line):
                sync_calls.append((i, line.strip(), is_final))
                break

        # Also check final sync patterns
        if not is_final:
            for pattern in FINAL_SYNC_PATTERNS:
                if re.search(pattern, line):
                    sync_calls.append((i, line.strip(), True))
                    break

    return sync_calls


def fuse_operations(lines: List[str], operations: List[LLKOperation]) -> Tuple[List[str], List[str]]:
    """Fuse consecutive LLK operations together"""
    if len(operations) < 2:
        return lines, []

    fused_lines = lines.copy()
    changes = []

    # Group consecutive operations that can be fused
    fused_groups = []
    current_group = [operations[0]]

    for i in range(1, len(operations)):
        prev_op = operations[i - 1]
        curr_op = operations[i]

        # Check if operations are consecutive (no significant code between them)
        gap = curr_op.line_start - prev_op.line_end
        if gap <= 5:  # Allow small gap for comments/whitespace
            # Check if there are only sync calls between them
            sync_calls = find_sync_calls(lines, prev_op.line_end + 1, curr_op.line_start - 1)
            intermediate_syncs = [s for s in sync_calls if not s[2]]  # Non-final syncs

            if intermediate_syncs:
                # Can fuse - add to current group
                current_group.append(curr_op)
            else:
                # Cannot fuse - start new group
                if len(current_group) > 1:
                    fused_groups.append(current_group)
                current_group = [curr_op]
        else:
            # Gap too large - start new group
            if len(current_group) > 1:
                fused_groups.append(current_group)
            current_group = [curr_op]

    # Add last group
    if len(current_group) > 1:
        fused_groups.append(current_group)

    # Process groups in reverse order to maintain line numbers
    for group in reversed(fused_groups):
        if len(group) < 2:
            continue

        first_op = group[0]
        last_op = group[-1]

        # Find the range to replace
        start_line = first_op.line_start
        end_line = last_op.line_end

        # Collect all operations in the group
        fused_ops = []

        # Add first init (if exists)
        if first_op.init_call:
            fused_ops.append(first_op.init_call)

        # Add all operation calls
        for op in group:
            fused_ops.extend(op.operation_calls)
            if op.done_call:
                fused_ops.append(op.done_call)

        # Find intermediate sync calls to remove
        lines_to_remove = set()
        for op in group[:-1]:  # All except last
            # Find sync calls between operations
            if group.index(op) < len(group) - 1:
                next_op = group[group.index(op) + 1]
                sync_calls = find_sync_calls(lines, op.line_end + 1, next_op.line_start - 1)
                for line_num, _, is_final in sync_calls:
                    if not is_final:
                        lines_to_remove.add(line_num)

        # Build replacement code
        replacement = []
        replacement.append("    // Fused LLK operations")
        replacement.extend(fused_ops)

        # Replace the range
        # Remove lines in reverse order
        for line_num in sorted(lines_to_remove, reverse=True):
            fused_lines.pop(line_num)
            # Adjust line numbers for operations after this
            for op in operations:
                if op.line_start > line_num:
                    op.line_start -= 1
                if op.line_end > line_num:
                    op.line_end -= 1

        # Replace the operation range
        # Calculate new end after removals
        num_removed = sum(1 for ln in lines_to_remove if ln < end_line)
        new_end = end_line - num_removed

        # Replace lines
        replacement_lines = [line.rstrip() for line in replacement]
        fused_lines[start_line : new_end + 1] = replacement_lines

        changes.append(f"Fused {len(group)} operations (lines {start_line+1}-{end_line+1})")

    return fused_lines, changes


def process_file(input_file: Path, output_file: Optional[Path] = None, dry_run: bool = False) -> int:
    """Process a kernel file to fuse LLK calls"""
    if not input_file.exists():
        print(f"Error: File not found: {input_file}", file=sys.stderr)
        return 1

    # Read file
    with open(input_file, "r") as f:
        lines = f.readlines()

    # Find LLK operations
    operations = find_llk_operations(lines)

    if not operations:
        print("No LLK operations found in file.")
        return 0

    print(f"Found {len(operations)} LLK operation(s)")

    # Fuse operations
    fused_lines, changes = fuse_operations(lines, operations)

    if not changes:
        print("No fusion opportunities found.")
        return 0

    # Print changes
    print("\nFusion changes:")
    for change in changes:
        print(f"  - {change}")

    if dry_run:
        print("\n[DRY RUN] No changes written to file.")
        return 0

    # Write output
    output_path = output_file or input_file
    with open(output_path, "w") as f:
        f.writelines(fused_lines)

    print(f"\nFused code written to: {output_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Fuse multiple LLK calls together to improve performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fuse operations in-place
  python fuse_llk_calls.py kernels/compute/my_kernel.cpp

  # Fuse operations to a new file
  python fuse_llk_calls.py kernels/compute/my_kernel.cpp --output kernels/compute/my_kernel_fused.cpp

  # Dry run to see what would change
  python fuse_llk_calls.py kernels/compute/my_kernel.cpp --dry-run
        """,
    )
    parser.add_argument("input_file", type=Path, help="Input kernel file to process")
    parser.add_argument("--output", "-o", type=Path, help="Output file (default: overwrite input)")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing to file")

    args = parser.parse_args()
    return process_file(args.input_file, args.output, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
