# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
deepseek_moe_fast_reduce_nc_fused with more than one row tile of tokens, on one device.

Every output row t must be the weighted sum of the expert slices with the scores of row t. The test
checks that against a torch reference per 32-row slice (so a slice weighted with another slice's
scores fails on its own) and checks that the T-row result equals the concatenation of the per-32-row
results bitwise (the per-element MAC order does not depend on the input height).
"""

import random

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc

PCC_THRESHOLD = 0.999


def _run_fused_reduce(mesh_device, activation, scores, indices, mapping, hidden_size):
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    tt_activation = ttnn.from_torch(
        activation,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=replicate,
    )
    tt_scores = ttnn.from_torch(
        scores,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=replicate,
    )
    tt_indices = ttnn.from_torch(
        indices,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=replicate,
    )
    tt_mapping = ttnn.from_torch(
        mapping,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=replicate,
    )
    outputs = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
        tt_activation,
        tt_indices,
        tt_mapping,
        reduce_dim=0,
        split_size=hidden_size,
        cluster_axis=0,
        output_memory_config=ttnn.L1_MEMORY_CONFIG,
        scores_tensor=tt_scores,
    )
    assert len(outputs) == 1
    return ttnn.to_torch(outputs[0], dtype=torch.bfloat16)


@pytest.mark.parametrize("tokens", [32, 64, 96, 128])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [2048])
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_deepseek_moe_fast_reduce_nc_fused_row_tiles(mesh_device, mesh_shape, tokens, select_experts_k, hidden_size):
    torch.manual_seed(2005)
    random.seed(2005)
    experts = 2 * select_experts_k

    activation = torch.rand((select_experts_k, 1, tokens, hidden_size), dtype=torch.bfloat16) - 0.5
    # Distinct normalized scores per token: a row weighted with another row's scores is detectable.
    scores = torch.rand((tokens, 1, 1, select_experts_k), dtype=torch.bfloat16)
    scores = scores / scores.sum(dim=-1, keepdim=True)
    indices = torch.empty((tokens, 1, 1, select_experts_k), dtype=torch.int32)
    for t in range(tokens):
        indices[t, 0, 0, :] = torch.tensor(random.sample(range(experts), select_experts_k), dtype=torch.int32)
    # Every expert lives on the one device, so every (token, k) slot is on axis.
    mapping = torch.zeros((1, experts), dtype=torch.int32)

    golden = (activation.float() * scores.float().permute(3, 1, 0, 2)).sum(dim=0, keepdim=True)

    result = _run_fused_reduce(mesh_device, activation, scores, indices, mapping, hidden_size)
    assert result.shape == (1, 1, tokens, hidden_size)

    for row_tile in range(tokens // 32):
        rows = slice(32 * row_tile, 32 * (row_tile + 1))
        passed, pcc = comp_pcc(golden[:, :, rows, :], result[:, :, rows, :].float(), PCC_THRESHOLD)
        assert passed, f"rows {rows.start}..{rows.stop - 1}: pcc {pcc}"

    # The T-row reduce equals the per-32-row reduces bitwise.
    for row_tile in range(tokens // 32):
        rows = slice(32 * row_tile, 32 * (row_tile + 1))
        slice_result = _run_fused_reduce(
            mesh_device,
            activation[:, :, rows, :].contiguous(),
            scores[rows].contiguous(),
            indices[rows].contiguous(),
            mapping,
            hidden_size,
        )
        assert torch.equal(
            result[:, :, rows, :], slice_result
        ), f"rows {rows.start}..{rows.stop - 1} differ from the 32-row reduce"
