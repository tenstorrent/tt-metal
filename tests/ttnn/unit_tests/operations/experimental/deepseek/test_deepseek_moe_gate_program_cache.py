# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Cache-hit regression for ttnn.experimental.deepseek.moe.deepseek_moe_gate.

Runs the op twice with the same attributes and tensor specs, and with both allocations live, so the
second dispatch cannot reuse the first call's buffer addresses. The second scores and expert ids are
checked against the grouped golden. One program-cache entry means the hit took the override path.
"""

import pytest
import torch

import ttnn
from models.common.modules.moe.tt_moe_gate import TTMoEGate


def _allocate(device, batch_size, seed, enable_sigmoid):
    eps, scaling_factor = 1e-20, 2.5
    input_shape = (batch_size, 8, 32)
    reshaped_input_shape = (batch_size, 16, 16)
    shard = (32, 32)
    tile = ttnn.Tile(shard)
    out_shape = (batch_size, 1, 16)

    torch.manual_seed(seed)
    torch_input = (2 * torch.rand(input_shape, dtype=torch.bfloat16)) - 1
    if not enable_sigmoid:
        torch_input = torch.sigmoid(torch_input)
    torch_bias = (2 * torch.rand(input_shape, dtype=torch.bfloat16)) - 1

    grid = device.compute_with_storage_grid_size()
    core_grid = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_grid, shard, ttnn.ShardOrientation.ROW_MAJOR),
    )

    ttnn_input = ttnn.from_torch(
        torch.reshape(torch_input, reshaped_input_shape),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem,
        tile=tile,
    )
    reshaped_bias = torch.transpose(torch.reshape(torch_bias, reshaped_input_shape), -2, -1)
    ttnn_bias = ttnn.from_torch(
        reshaped_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem, tile=tile
    )
    torch_input_indices = torch.arange(reshaped_input_shape[1] * reshaped_input_shape[2], dtype=torch.int32)
    torch_input_indices = torch_input_indices.unsqueeze(0).expand(reshaped_input_shape[0], -1)
    torch_input_indices = torch_input_indices.reshape(reshaped_input_shape)
    torch_input_indices = torch.transpose(torch_input_indices, -2, -1).to(torch.uint16)
    ttnn_input_indices = ttnn.from_torch(
        torch_input_indices, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem, tile=tile
    )
    ttnn_output = ttnn.from_torch(
        torch.zeros(out_shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem,
        tile=tile,
    )
    ttnn_output_indices = ttnn.from_torch(
        torch.zeros(out_shape, dtype=torch.uint16),
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem,
        tile=tile,
    )
    return {
        "torch_input": torch_input,
        "torch_bias": torch_bias,
        "eps": eps,
        "scaling_factor": scaling_factor,
        "enable_sigmoid": enable_sigmoid,
        "batch_size": batch_size,
        "tensors": (ttnn_input, ttnn_bias, ttnn_input_indices, ttnn_output, ttnn_output_indices),
    }


def _run(device, case):
    ttnn_input, ttnn_bias, ttnn_input_indices, ttnn_output, ttnn_output_indices = case["tensors"]
    with device.cache_entries_counter.measure():
        res_scores, res_idx = ttnn.experimental.deepseek.moe.deepseek_moe_gate(
            ttnn_input,
            bias_tensor=ttnn_bias,
            input_indices_tensor=ttnn_input_indices,
            output_tensor=ttnn_output,
            output_indices_tensor=ttnn_output_indices,
            eps=case["eps"],
            scaling_factor=case["scaling_factor"],
            enable_sigmoid=case["enable_sigmoid"],
        )
    return res_scores, res_idx


def _assert_matches_grouped_golden(case, res_scores, res_idx):
    batch_size = case["batch_size"]
    eps = case["eps"]
    scaling_factor = case["scaling_factor"]
    enable_sigmoid = case["enable_sigmoid"]
    torch_input = case["torch_input"]
    torch_bias = case["torch_bias"]

    _, gold_idx = TTMoEGate.grouped_golden(
        torch_input, torch_bias, eps=eps, scaling_factor=scaling_factor, enable_sigmoid=enable_sigmoid
    )
    output_torch = ttnn.to_torch(res_scores)[:, 0, :8]
    output_indices_torch = ttnn.to_torch(res_idx)[:, 0, :8]
    sorted_idx, order = torch.sort(output_indices_torch, dim=-1)
    sorted_scores = torch.gather(output_torch, dim=-1, index=order)

    ranking = torch.sigmoid(torch_input) if enable_sigmoid else torch_input
    bias_key = (ranking + torch_bias).reshape(batch_size, -1).float()
    raw_scores = ranking.reshape(batch_size, -1).float()
    dev_idx = sorted_idx.long()
    gold_idx = torch.sort(gold_idx, dim=-1).values.long()

    assert dev_idx.min() >= 0 and dev_idx.max() < 256, f"device produced out-of-range expert id:\n{dev_idx}"
    dev_key = torch.gather(bias_key, -1, dev_idx).sort(-1).values
    gold_key = torch.gather(bias_key, -1, gold_idx).sort(-1).values
    assert torch.allclose(dev_key, gold_key, atol=1e-2), (
        f"grouped selection not consistent with the grouped golden.\n dev_idx={dev_idx}\n gold_idx={gold_idx}\n"
        f" dev_key={dev_key}\n gold_key={gold_key}"
    )
    dev_sel = torch.gather(raw_scores, -1, dev_idx)
    expected = dev_sel / (dev_sel.sum(-1, keepdim=True) + eps) * scaling_factor
    assert torch.allclose(
        sorted_scores.float(), expected, atol=1e-2, rtol=1e-4
    ), f"grouped normalized scores not consistent with device selection.\n dev={sorted_scores}\n expected={expected}"


@pytest.fixture
def isolate_program_cache(device):
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    yield
    device.disable_and_clear_program_cache()


def test_deepseek_moe_gate_cache_hit_rebinds_tensor_cbs(device, isolate_program_cache):
    """Same attributes, two live allocations. The second result must match its own inputs, and the cache stays at one entry."""
    first = _allocate(device, batch_size=1, seed=42, enable_sigmoid=True)
    # Hold the first allocation until the second dispatch so the allocator cannot hand back the same addresses.
    second = _allocate(device, batch_size=1, seed=201, enable_sigmoid=True)
    assert not torch.equal(first["torch_input"], second["torch_input"])

    _run(device, first)
    res_scores, res_idx = _run(device, second)
    _assert_matches_grouped_golden(second, res_scores, res_idx)
    assert device.cache_entries_counter.total == 1
