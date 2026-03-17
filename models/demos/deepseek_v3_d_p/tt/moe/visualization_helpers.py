# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Visualization utilities for MoE dispatch/combine debugging."""

from typing import List, Set, Tuple

import torch
from loguru import logger
from tabulate import tabulate

from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import ValidationResult


def visualize_expert_dispatch_table(
    expert_dispatch_table: torch.Tensor,
    num_dispatch_groups: int,
    dispatch_group_size: int,
    num_routed_experts: int,
    title: str = "Expert Dispatch Table",
) -> str:
    """
    Visualize expert dispatch table as ASCII table.

    Shows which experts are assigned to each chip within each dispatch group.
    Columns are dispatch groups, rows are chips within dispatch groups.

    Args:
        expert_dispatch_table: Shape (num_dispatch_groups, num_routed_experts)
            Values are chip IDs (0 to dispatch_group_size-1) or -1 if not present
        num_dispatch_groups: Number of dispatch groups (EP ranks)
        dispatch_group_size: Number of chips in each dispatch group
        num_routed_experts: Total number of routed experts
        title: Title for the table

    Returns:
        Formatted ASCII table string

    Example output for 4 dispatch groups, 2 chips per group, 64 experts:
        Expert Dispatch Table:
        (Raw table shape: torch.Size([4, 64]))
        +------+---------------------------+---------------------------+---------------------------+---------------------------+
        | Chip |     Dispatch Group 0      |     Dispatch Group 1      |     Dispatch Group 2      |     Dispatch Group 3      |
        +------+---------------------------+---------------------------+---------------------------+---------------------------+
        |  0   |     [0,1,2,3,4,5,6,7]     | [16,17,18,19,20,21,22,23] | [32,33,34,35,36,37,38,39] | [48,49,50,51,52,53,54,55] |
        |  1   |  [8,9,10,11,12,13,14,15]  | [24,25,26,27,28,29,30,31] | [40,41,42,43,44,45,46,47] | [56,57,58,59,60,61,62,63] |
        +------+---------------------------+---------------------------+---------------------------+---------------------------+
    """
    # Build a mapping: (dispatch_group, chip) -> list of expert IDs
    experts_by_chip = {}
    for dispatch_group in range(num_dispatch_groups):
        for chip in range(dispatch_group_size):
            experts_by_chip[(dispatch_group, chip)] = []

    # Populate the mapping from the dispatch table
    for dispatch_group in range(num_dispatch_groups):
        for expert_id in range(num_routed_experts):
            chip_id = expert_dispatch_table[dispatch_group, expert_id].item()
            if chip_id != -1:
                experts_by_chip[(dispatch_group, chip_id)].append(expert_id)

    # Build table data: rows are chips, columns are dispatch groups
    headers = ["Chip"] + [f"Dispatch Group {g}" for g in range(num_dispatch_groups)]
    rows = []
    for chip in range(dispatch_group_size):
        row = [chip]
        for dispatch_group in range(num_dispatch_groups):
            experts = experts_by_chip[(dispatch_group, chip)]
            # Format as comma-separated list in brackets
            row.append(f"[{','.join(map(str, experts))}]" if experts else "[]")
        rows.append(row)

    # Format table with pretty_grid style
    table_str = tabulate(rows, headers=headers, tablefmt="pretty", stralign="center")

    # Add title and shape info
    output = f"{title}:\n"
    output += f"(Raw table shape: {tuple(expert_dispatch_table.shape)})\n"
    output += table_str

    return output


def log_expert_dispatch_table(
    expert_dispatch_table: torch.Tensor,
    num_dispatch_groups: int,
    dispatch_group_size: int,
    num_routed_experts: int,
    title: str = "Expert Dispatch Table",
) -> None:
    """
    Log expert dispatch table visualization using loguru.

    Args:
        expert_dispatch_table: Shape (num_dispatch_groups, num_routed_experts)
        num_dispatch_groups: Number of dispatch groups (EP ranks)
        dispatch_group_size: Number of chips in each dispatch group
        num_routed_experts: Total number of routed experts
        title: Title for the table
    """
    table_str = visualize_expert_dispatch_table(
        expert_dispatch_table=expert_dispatch_table,
        num_dispatch_groups=num_dispatch_groups,
        dispatch_group_size=dispatch_group_size,
        num_routed_experts=num_routed_experts,
        title=title,
    )
    logger.info(f"\n{table_str}")


def _extract_chip_status(result: ValidationResult) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """
    Extract sets of (dispatch_group, chip) pairs for failures and all validated cells.

    Mismatches are tuples of (dispatch_group_idx, chip_id, ...).

    Returns:
        failed: Set of (dispatch_group, chip) pairs with failures
        validated: Set of (dispatch_group, chip) pairs that were validated
    """
    failed = set()

    # Extract failures from mismatches
    for mismatch in result.mismatches:
        if len(mismatch) >= 2:
            dispatch_group_idx, chip_id = mismatch[0], mismatch[1]
            failed.add((dispatch_group_idx, chip_id))

    # Use validated_cells from result if available, otherwise fall back to failures only
    validated = result.validated_cells if result.validated_cells else failed

    return failed, validated


def visualize_validation_results(
    results: List[ValidationResult],
    num_dispatch_groups: int,
    dispatch_group_size: int,
    title: str = "Validation Results",
) -> str:
    """
    Visualize validation results as ASCII table with emoji status.

    Shows pass/fail status for each (dispatch_group, chip) combination.
    Each ValidationResult contributes one emoji per cell.

    Args:
        results: List of ValidationResult objects (e.g., [buffer_result, metadata_result])
        num_dispatch_groups: Number of dispatch groups (EP ranks)
        dispatch_group_size: Number of chips in each dispatch group
        title: Title for the table

    Returns:
        Formatted ASCII table string

    Example output with [buffer_result, metadata_result]:
        Validation Results:
        (Columns: buffer | metadata)
        +------+------------------+------------------+
        | Chip | Dispatch Group 0 | Dispatch Group 1 |
        +------+------------------+------------------+
        |  0   |       ✅✅       |       ✅❌       |
        |  1   |       ✅✅       |       ✅✅       |
        +------+------------------+------------------+
        buffer: 2/2 ✅ | metadata: 1/2 ❌
    """
    if not results:
        return f"{title}: No results to display"

    # Build failure and validated sets for each result
    status_sets = [_extract_chip_status(r) for r in results]

    # Build table data: rows are chips, columns are dispatch groups
    headers = ["Chip"] + [f"Dispatch Group {g}" for g in range(num_dispatch_groups)]
    rows = []
    for chip in range(dispatch_group_size):
        row = [chip]
        for dispatch_group in range(num_dispatch_groups):
            key = (dispatch_group, chip)
            emojis = ""
            for failed_set, validated_set in status_sets:
                if key not in validated_set:
                    emojis += "-"  # Not validated (skipped)
                elif key in failed_set:
                    emojis += "❌"  # Validated and failed
                else:
                    emojis += "✅"  # Validated and passed
            row.append(emojis)
        rows.append(row)

    # Format table
    table_str = tabulate(rows, headers=headers, tablefmt="pretty", stralign="center")

    # Build header with result names
    result_names = [r.name for r in results]
    columns_desc = " | ".join(result_names)

    # Build summary line
    summary_parts = []
    for r in results:
        status = "✅" if r.passed else "❌"
        summary_parts.append(f"{r.name}: {r.matches}/{r.total} {status}")
    summary = " | ".join(summary_parts)

    # Combine output
    output = f"{title}:\n"
    output += f"(Columns: {columns_desc})\n"
    output += table_str + "\n"
    output += summary

    return output


def log_validation_results(
    results: List[ValidationResult],
    num_dispatch_groups: int,
    dispatch_group_size: int,
    title: str = "Validation Results",
) -> None:
    """
    Log validation results visualization using loguru.

    Args:
        results: List of ValidationResult objects
        num_dispatch_groups: Number of dispatch groups (EP ranks)
        dispatch_group_size: Number of chips in each dispatch group
        title: Title for the table
    """
    table_str = visualize_validation_results(
        results=results,
        num_dispatch_groups=num_dispatch_groups,
        dispatch_group_size=dispatch_group_size,
        title=title,
    )
    logger.info(f"\n{table_str}")
