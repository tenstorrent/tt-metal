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

- Same config -> reuse (1 entry).
- topk / enable_sigmoid toggled (hashed attributes) -> distinct entries.
"""

import pytest
import torch

import ttnn


def run_gate(device, batch_size, topk, enable_sigmoid, output_softmax, seed=42):
    """Build the 5 height-sharded tensors, run generalized_moe_gate in the cache counter, return the
    sorted device-selected expert indices. (Exact top-k correctness is covered by the dedicated
    reference suite models/common/tests/modules/moe/test_generalized_moe_gate.py; here we assert the
    op runs, selects valid experts, and is deterministic across a cache-reused program.)"""
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

    with device.cache_entries_counter.measure():
        _, res_idx = ttnn.experimental.deepseek.moe.generalized_moe_gate(
            ttnn_input,
            bias_tensor=ttnn_bias,
            input_indices_tensor=ttnn_input_indices,
            output_tensor=ttnn_output,
            output_indices_tensor=ttnn_output_indices,
            eps=eps,
            scaling_factor=scaling_factor,
            enable_sigmoid=enable_sigmoid,
            topk=topk,
            output_softmax=output_softmax,
        )
    dev_idx = torch.sort(ttnn.to_torch(res_idx)[:, 0, :topk].to(torch.int64), dim=-1).values
    return dev_idx


@pytest.fixture
def isolate_program_cache(device):
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    yield
    device.disable_and_clear_program_cache()


def test_gate_cache_reuse_same_config(device, isolate_program_cache):
    """Same config run twice -> 1 entry; identical (deterministic) valid selection across reuse."""
    dev1 = run_gate(device, batch_size=1, topk=8, enable_sigmoid=True, output_softmax=False)
    dev2 = run_gate(device, batch_size=1, topk=8, enable_sigmoid=True, output_softmax=False)
    assert torch.equal(dev1, dev2)
    assert dev1.min().item() >= 0 and dev1.max().item() < 256
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
