# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Program-cache regression tests for ttnn.experimental.deepseek.moe.generalized_moe_gate
(GeneralizedMoeGateDeviceOperation).

Pins the cache-keying granularity so it can be verified BEFORE and AFTER removing
GeneralizedMoeGateDeviceOperation::compute_program_hash. The custom hash keyed on the six
operation_attributes (eps/scaling_factor/enable_sigmoid/topk/output_softmax/grouped) and the
TensorSpec (logical) of all five tensors. The framework default hashes the whole attrs struct +
tensor_args (which hold exactly those five tensors), i.e. the same distinctions and the same
logical-shape keying (the op's own hash comment states it is "same as the framework's default hash").

Setup mirrors models/common/tests/modules/moe/test_generalized_moe_gate.py (ungrouped, 256 experts,
one token/core, HEIGHT_SHARDED L1).

- Same config, two live allocations with different data -> reuse (1 entry), and the second
  result matches that call's own inputs.
- topk / enable_sigmoid toggled (hashed attributes) -> distinct entries.
"""

import pytest
import torch

import ttnn


def _allocate(device, batch_size, topk, enable_sigmoid, output_softmax, seed):
    """Build the 5 height-sharded tensors. The caller keeps the returned tensors alive."""
    input_shape = (batch_size, 8, 32)
    reshaped_input_shape = (batch_size, 16, 16)
    input_tile = ttnn.Tile((32, 32))
    output_shape = (batch_size, 1, 16)
    output_tile = ttnn.Tile((32, 32))
    eps, scaling_factor = 1e-20, 2.5

    torch.manual_seed(seed)
    torch_input = (2 * torch.rand(input_shape, dtype=torch.bfloat16)) - 1
    if enable_sigmoid or not output_softmax:
        torch_input = torch.sigmoid(torch_input)
    torch_bias = (2 * torch.rand(input_shape, dtype=torch.bfloat16)) - 1

    grid = device.compute_with_storage_grid_size()
    core_grid = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    in_shard = ttnn.ShardSpec(core_grid, (32, 32), ttnn.ShardOrientation.ROW_MAJOR)
    out_shard = ttnn.ShardSpec(core_grid, (32, 32), ttnn.ShardOrientation.ROW_MAJOR)
    in_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in_shard)
    out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, out_shard)

    ttnn_input = ttnn.from_torch(
        torch.reshape(torch_input, reshaped_input_shape),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in_mem,
        tile=input_tile,
    )
    ttnn_bias = ttnn.from_torch(
        torch.transpose(torch.reshape(torch_bias, reshaped_input_shape), -2, -1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in_mem,
        tile=input_tile,
    )
    idx = torch.arange(16 * 16, dtype=torch.int32).unsqueeze(0).expand(batch_size, -1).reshape(reshaped_input_shape)
    idx = torch.transpose(idx, -2, -1).to(torch.uint16)
    ttnn_input_indices = ttnn.from_torch(
        idx,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in_mem,
        tile=input_tile,
    )
    ttnn_output = ttnn.from_torch(
        torch.zeros(output_shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=out_mem,
        tile=output_tile,
    )
    ttnn_output_indices = ttnn.from_torch(
        torch.zeros(output_shape, dtype=torch.uint16),
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=out_mem,
        tile=output_tile,
    )
    return {
        "torch_input": torch_input,
        "torch_bias": torch_bias,
        "eps": eps,
        "scaling_factor": scaling_factor,
        "enable_sigmoid": enable_sigmoid,
        "output_softmax": output_softmax,
        "topk": topk,
        "batch_size": batch_size,
        "tensors": (ttnn_input, ttnn_bias, ttnn_input_indices, ttnn_output, ttnn_output_indices),
    }


def _run(device, case):
    ttnn_input, ttnn_bias, ttnn_input_indices, ttnn_output, ttnn_output_indices = case["tensors"]
    with device.cache_entries_counter.measure():
        res, res_idx = ttnn.experimental.deepseek.moe.generalized_moe_gate(
            ttnn_input,
            bias_tensor=ttnn_bias,
            input_indices_tensor=ttnn_input_indices,
            output_tensor=ttnn_output,
            output_indices_tensor=ttnn_output_indices,
            eps=case["eps"],
            scaling_factor=case["scaling_factor"],
            enable_sigmoid=case["enable_sigmoid"],
            topk=case["topk"],
            output_softmax=case["output_softmax"],
        )
    return res, res_idx


def _assert_matches_own_inputs(case, res, res_idx):
    """Tie-robust check from test_generalized_moe_gate: the device selection is a valid top-k of THIS
    call's inputs, and the scores match that selection. A stale CB would score the previous allocation."""
    batch_size = case["batch_size"]
    topk = case["topk"]
    eps = case["eps"]
    scaling_factor = case["scaling_factor"]
    enable_sigmoid = case["enable_sigmoid"]
    output_softmax = case["output_softmax"]
    torch_input = case["torch_input"]
    torch_bias = case["torch_bias"]

    ranking = torch.sigmoid(torch_input) if enable_sigmoid else torch_input
    _, gold_idx = torch.topk((ranking + torch_bias).reshape(batch_size, -1), topk, dim=-1, sorted=True)
    output_torch = ttnn.to_torch(res)[:, 0, :topk]
    output_indices_torch = ttnn.to_torch(res_idx)[:, 0, :topk]
    sorted_idx, order = torch.sort(output_indices_torch, dim=-1)
    sorted_scores = torch.gather(output_torch, dim=-1, index=order)

    bias_key = (ranking + torch_bias).reshape(batch_size, -1).float()
    raw_scores = ranking.reshape(batch_size, -1).float()
    dev_idx = sorted_idx.long()
    gold_idx = torch.sort(gold_idx, dim=-1).values.long()

    assert dev_idx.min() >= 0 and dev_idx.max() < 256, f"device produced out-of-range expert id:\n{dev_idx}"
    dev_key = torch.gather(bias_key, -1, dev_idx).sort(-1).values
    gold_key = torch.gather(bias_key, -1, gold_idx).sort(-1).values
    assert torch.allclose(dev_key, gold_key, atol=1e-2), (
        f"cache-hit selection is not a valid top-{topk} of the second inputs.\n"
        f" dev_idx={dev_idx}\n gold_idx={gold_idx}\n dev_key={dev_key}\n gold_key={gold_key}"
    )
    dev_sel = torch.gather(raw_scores, -1, dev_idx)
    if output_softmax:
        weights = torch.exp(dev_sel - dev_sel.max(dim=-1, keepdim=True).values)
    else:
        weights = dev_sel
    expected = weights / (weights.sum(-1, keepdim=True) + eps) * scaling_factor
    assert torch.allclose(
        sorted_scores.float(), expected, atol=1e-2, rtol=1e-4
    ), f"cache-hit scores are not consistent with the second inputs.\n dev={sorted_scores}\n expected={expected}"


def run_gate(device, batch_size, topk, enable_sigmoid, output_softmax, seed=42):
    """Run generalized_moe_gate once and return the sorted device-selected expert indices.
    Exact top-k correctness is covered by models/common/tests/modules/moe/test_generalized_moe_gate.py."""
    _, res_idx = _run(device, _allocate(device, batch_size, topk, enable_sigmoid, output_softmax, seed))
    return torch.sort(ttnn.to_torch(res_idx)[:, 0, :topk].to(torch.int64), dim=-1).values


@pytest.fixture
def isolate_program_cache(device):
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    yield
    device.disable_and_clear_program_cache()


def test_gate_cache_reuse_same_config(device, isolate_program_cache):
    """Same attributes and two live allocations. The second result matches its own inputs, with one cache entry."""
    first = _allocate(device, batch_size=1, topk=8, enable_sigmoid=True, output_softmax=False, seed=42)
    # Hold the first allocation until the second dispatch so the allocator cannot hand back the same addresses.
    second = _allocate(device, batch_size=1, topk=8, enable_sigmoid=True, output_softmax=False, seed=201)
    assert not torch.equal(first["torch_input"], second["torch_input"])

    _run(device, first)
    res, res_idx = _run(device, second)
    _assert_matches_own_inputs(second, res, res_idx)
    assert device.cache_entries_counter.total == 1


def test_gate_cache_miss_topk(device, isolate_program_cache):
    """topk is a hashed compile-time attribute -> 2 entries."""
    run_gate(device, batch_size=1, topk=8, enable_sigmoid=True, output_softmax=False)
    run_gate(device, batch_size=1, topk=4, enable_sigmoid=True, output_softmax=False)
    assert device.cache_entries_counter.total == 2


def test_gate_cache_miss_enable_sigmoid(device, isolate_program_cache):
    """enable_sigmoid is a hashed compile-time attribute -> 2 entries."""
    run_gate(device, batch_size=1, topk=8, enable_sigmoid=True, output_softmax=False)
    run_gate(device, batch_size=1, topk=8, enable_sigmoid=False, output_softmax=False)
    assert device.cache_entries_counter.total == 2
