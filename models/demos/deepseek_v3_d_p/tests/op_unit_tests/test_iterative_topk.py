# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Unit test for ttnn.experimental.deepseek_prefill.iterative_topk().

Verifies that the iterative top-k op (repeated max with masking) produces
results matching torch.topk on the last dimension.
"""

import pytest
import torch
from loguru import logger

import ttnn

TEST_PARAMS = [
    (1, 16, 8),
    (1, 256, 8),
    (4, 256, 8),
    (1, 256, 32),
    (32, 1024, 16),
]

TEST_PARAM_IDS = [
    "small_16x8",
    "256x8",
    "batch4_256x8",
    "256x32",
    "batch32_1024x16",
]


@pytest.mark.parametrize("num_rows,width,k", TEST_PARAMS, ids=TEST_PARAM_IDS)
def test_iterative_topk(device, num_rows, width, k):
    """Verify iterative_topk matches torch.topk for values and indices."""
    torch.manual_seed(42)

    input_tensor = torch.randn(num_rows, width, dtype=torch.float32)

    ref_values, ref_indices = torch.topk(input_tensor, k, dim=-1)

    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    ttnn_values, ttnn_indices = ttnn.experimental.deepseek_prefill.iterative_topk(ttnn_input, k=k)

    tt_values = ttnn.to_torch(ttnn_values)
    tt_indices = ttnn.to_torch(ttnn_indices)

    # Values should match exactly (both are float32, same algorithm)
    values_match = torch.allclose(tt_values, ref_values, atol=1e-6)
    status = "PASS" if values_match else "FAIL"
    if not values_match:
        max_diff = (tt_values - ref_values).abs().max().item()
        logger.info(f"[{status}] Values max diff = {max_diff:.6e} for rows={num_rows}, width={width}, k={k}")
    else:
        logger.info(f"[{status}] Values match for rows={num_rows}, width={width}, k={k}")

    # Indices should select the same set of top-k elements per row.
    # torch.topk returns indices sorted by value (descending), and our iterative
    # approach also returns them in descending order, so they should match directly.
    indices_match = torch.equal(tt_indices.to(torch.int64), ref_indices)
    status = "PASS" if indices_match else "FAIL"
    if not indices_match:
        # Even if indices differ in order, verify they select the same elements
        recall = _compute_recall(tt_indices.to(torch.int64), ref_indices)
        logger.info(
            f"[{status}] Indices exact match failed, recall = {recall:.4f} for rows={num_rows}, width={width}, k={k}"
        )
    else:
        logger.info(f"[{status}] Indices match for rows={num_rows}, width={width}, k={k}")

    assert values_match, f"Values mismatch for rows={num_rows}, width={width}, k={k}"
    assert indices_match, f"Indices mismatch for rows={num_rows}, width={width}, k={k}"


def _compute_recall(predicted_indices: torch.Tensor, reference_indices: torch.Tensor) -> float:
    """Compute average recall: fraction of reference indices found in predicted per row."""
    num_rows = predicted_indices.shape[0]
    total_recall = 0.0
    for i in range(num_rows):
        pred_set = set(predicted_indices[i].tolist())
        ref_set = set(reference_indices[i].tolist())
        if len(ref_set) > 0:
            total_recall += len(pred_set & ref_set) / len(ref_set)
    return total_recall / num_rows
