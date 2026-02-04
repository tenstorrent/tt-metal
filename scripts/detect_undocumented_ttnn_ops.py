#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
This script compares the list of registered ttnn operations from ttnn.query_registered_operations(include_experimental=False)
with the documented operations in docs/source/ttnn/ttnn/api.rst. It identifies any operations that are registered
but missing from the documentation file.
"""

import sys
from pathlib import Path

import ttnn

# Constants
# Skipped operations should not be included in the documentation check
SKIPPED_OPS = [
    "ttnn.allocate_tensor_on_device",  # Missing docs for OP causing docs build crash, need to create documentation and add to docs page, GH issue: #34681
    "ttnn.allocate_tensor_on_host",  # Missing docs for OP causing docs build crash, need to create documentation and add to docs page, GH issue: #34681
    "ttnn.composite_example",  # Example operation.
    "ttnn.composite_example_multiple_return",  # Example operation.
    "ttnn.fused_rms_minimal",  # Internal operation only.
    "ttnn.matmul_batched_weights",  # Internal operation only.
    "ttnn.moreh_abs_pow",  # Moreh operation.
    "ttnn.moreh_adam",  # Moreh operation.
    "ttnn.moreh_adamw",  # Moreh operation.
    "ttnn.moreh_arange",  # Moreh operation.
    "ttnn.moreh_bmm",  # Moreh operation.
    "ttnn.moreh_bmm_backward",  # Moreh operation.
    "ttnn.moreh_clip_grad_norm",  # Moreh operation.
    "ttnn.moreh_cumsum",  # Moreh operation.
    "ttnn.moreh_cumsum_backward",  # Moreh operation.
    "ttnn.moreh_dot",  # Moreh operation.
    "ttnn.moreh_dot_backward",  # Moreh operation.
    "ttnn.moreh_fold",  # Moreh operation.
    "ttnn.moreh_full",  # Moreh operation.
    "ttnn.moreh_full_like",  # Moreh operation.
    "ttnn.moreh_getitem",  # Moreh operation.
    "ttnn.moreh_group_norm",  # Moreh operation.
    "ttnn.moreh_group_norm_backward",  # Moreh operation.
    "ttnn.moreh_layer_norm",  # Moreh operation.
    "ttnn.moreh_layer_norm_backward",  # Moreh operation.
    "ttnn.moreh_linear",  # Moreh operation.
    "ttnn.moreh_linear_backward",  # Moreh operation.
    "ttnn.moreh_logsoftmax",  # Moreh operation.
    "ttnn.moreh_logsoftmax_backward",  # Moreh operation.
    "ttnn.moreh_matmul",  # Moreh operation.
    "ttnn.moreh_matmul_backward",  # Moreh operation.
    "ttnn.moreh_mean",  # Moreh operation.
    "ttnn.moreh_mean_backward",  # Moreh operation.
    "ttnn.moreh_nll_loss",  # Moreh operation.
    "ttnn.moreh_nll_loss_backward",  # Moreh operation.
    "ttnn.moreh_nll_loss_unreduced_backward",  # Moreh operation.
    "ttnn.moreh_norm",  # Moreh operation.
    "ttnn.moreh_norm_backward",  # Moreh operation.
    "ttnn.moreh_sgd",  # Moreh operation.
    "ttnn.moreh_softmax",  # Moreh operation.
    "ttnn.moreh_softmax_backward",  # Moreh operation.
    "ttnn.moreh_softmin",  # Moreh operation.
    "ttnn.moreh_softmin_backward",  # Moreh operation.
    "ttnn.moreh_sum",  # Moreh operation.
    "ttnn.moreh_sum_backward",  # Moreh operation.
    "ttnn.padded_slice",  # Experimental operation, but wrongly registered without ttnn.experimental.
    "ttnn.pearson_correlation_coefficient",  # Internal operation only.
    "ttnn.plus_one",  # Experimental operation, but wrongly registered without ttnn.experimental.
    "ttnn.hang_device_operation",  # Internal operation only.
    "ttnn.slice_write",  # Experimental operation, but wrongly registered without ttnn.experimental.
    "ttnn.tosa_gather",  # TOSA operation omitted for docs.
    "ttnn.tosa_scatter",  # TOSA operation omitted for docs.
]
TTNN_PREFIX = "ttnn."
DOCS_OP_LIST_PATH = "docs/source/ttnn/ttnn/api.rst"


# Helper functions
def get_registered_operations(include_experimental=False) -> set:
    """
    Get all registered operations from ttnn.query_registered_operations()

    :param include_experimental: Whether to include experimental operations.

    :return: Set of registered operation names.
    """
    # Prepare set to hold registered operations
    registered_ops = set()

    # Get registered operations from ttnn
    all_ops = ttnn.query_registered_operations(include_experimental)
    for op in all_ops:
        if op.python_fully_qualified_name in SKIPPED_OPS:
            # Skip operations that are in the SKIPPED_OPS list
            continue
        registered_ops.add(op.python_fully_qualified_name)

    return registered_ops


def extract_ops_from_docs_rst_file(api_rst_path: str | Path) -> set:
    """
    Extract all ttnn operations from docs file by checking each line

    :param api_rst_path: Path to the api.rst file.

    :return: Set of operation names found in the file.
    """
    # Prepare set to hold operations found in the file
    api_ops = set()

    with open(api_rst_path, "r") as f:
        for line in f:
            # Remove leading and trailing spaces
            stripped_line = line.strip()

            # Check if the line starts with ttnn. (after stripping)
            if stripped_line.startswith(TTNN_PREFIX):
                api_ops.add(stripped_line)

    return api_ops


# Main processing function
def main() -> None:
    # Define file path
    script_dir = Path(__file__).parent.parent  # Assume script is in scripts/ directory
    api_rst_path = script_dir / DOCS_OP_LIST_PATH

    # Check if file exists
    if not api_rst_path.exists():
        print(f"Error: Could not find docs OPs list at {api_rst_path}")
        sys.exit(1)

    # Get registered operations from ttnn
    registered_ops = get_registered_operations()

    # Extract operations from docs file
    api_ops = extract_ops_from_docs_rst_file(api_rst_path)

    # Find operations that are registered but missing from docs file
    # For each registered operation, check if it exists in api_ops
    missing_ops = set()
    for registered_op in registered_ops:
        if registered_op not in api_ops:
            missing_ops.add(registered_op)

    # Report results
    if missing_ops:
        print("The following registered ttnn operations are missing from the documentation:")
        for op in sorted(missing_ops):
            print(f" - {op}")
        print(f"\nTotal missing operations: {len(missing_ops)}")
        print(
            f"Please update the documentation file ({api_rst_path}) to include these operations or add them to the SKIPPED_OPS with reason list if they should be excluded in {__file__}."
        )
        sys.exit(1)
    else:
        print(f"All registered ttnn operations are documented in {api_rst_path}!")
        sys.exit(0)


if __name__ == "__main__":
    main()
