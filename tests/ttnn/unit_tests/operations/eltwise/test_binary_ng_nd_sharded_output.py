# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

pytestmark = pytest.mark.use_module_device

SHAPE = torch.Size([1, 1, 128, 32])


def nd_shard_config():
    # The shard is wider than the tensor, so no legacy 2D shard spec can describe it.
    return ttnn.MemoryConfig(
        ttnn.BufferType.L1,
        ttnn.NdShardSpec(
            shard_shape=ttnn.Shape([1, 1, 64, 64]),
            grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (1, 1))}),
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )


def to_device(device, t, memory_config):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)


@pytest.mark.parametrize(
    "inputs_nd_sharded, preallocated_output",
    [(False, True), (True, True), (True, False)],
    ids=["interleaved_inputs-nd_output", "nd_inputs-nd_output", "nd_inputs-no_output"],
)
def test_binary_nd_sharded_operands(device, inputs_nd_sharded, preallocated_output):
    """An ND-sharded tensor is sharded but has no 2D shard spec for the worker grid or memory config to read."""
    torch.manual_seed(0)
    torch_a = torch.randn(SHAPE).bfloat16()
    torch_b = torch.randn(SHAPE).bfloat16()
    expected = ttnn.to_torch(
        ttnn.add(to_device(device, torch_a, ttnn.L1_MEMORY_CONFIG), to_device(device, torch_b, ttnn.L1_MEMORY_CONFIG))
    )

    input_config = nd_shard_config() if inputs_nd_sharded else ttnn.L1_MEMORY_CONFIG
    a = to_device(device, torch_a, input_config)
    b = to_device(device, torch_b, input_config)
    if preallocated_output:
        out = to_device(device, torch.zeros(SHAPE).bfloat16(), nd_shard_config())
        assert out.memory_config().shard_spec is None
        ttnn.add(a, b, output_tensor=out)
    else:
        out = ttnn.add(a, b)
    assert torch.equal(ttnn.to_torch(out), expected)
