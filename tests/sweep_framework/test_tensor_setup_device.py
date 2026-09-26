# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The host-side setup route must produce the same tensor as device-side construction: equal
TensorSpec and equal data, for the inputs the sweep helpers build. Needs one device."""

import sys
from pathlib import Path

import pytest
import torch

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

import ttnn  # noqa: E402
from sweep_utils.tensor_setup import host_side_tensor_construction  # noqa: E402


def height_sharded(shard_shape, num_cores):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )


CASES = [
    pytest.param((1, 1, 64, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT, lambda: ttnn.DRAM_MEMORY_CONFIG, id="bf16-tile-dram"),
    pytest.param((1, 1, 64, 64), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, lambda: ttnn.L1_MEMORY_CONFIG, id="bf8-tile-l1"),
    pytest.param(
        (1, 1, 64, 64), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, lambda: ttnn.DRAM_MEMORY_CONFIG, id="fp32-rm-dram"
    ),
    pytest.param(
        (1, 1, 64, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT, lambda: height_sharded((32, 64), 2), id="bf16-tile-hs-2cores"
    ),
    pytest.param(
        (1, 1, 8, 32),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        lambda: height_sharded((32, 32), 1),
        id="bf16-tile-hs-shard-taller-than-tensor",
    ),
]


def _torch_input(shape, dtype):
    return torch.randn(shape, dtype=torch.float32 if dtype == ttnn.float32 else torch.bfloat16)


def _same(a, b):
    assert a.spec == b.spec, f"spec differs:\n device: {a.spec}\n host:   {b.spec}"
    assert torch.equal(ttnn.to_torch(a), ttnn.to_torch(b))


@pytest.mark.parametrize("shape, dtype, layout, memory_config", CASES)
def test_from_torch_with_mesh_mapper_matches_host_build_plus_write(device, shape, dtype, layout, memory_config):
    mc = memory_config()
    x = _torch_input(shape, dtype)
    kwargs = dict(dtype=dtype, layout=layout, device=device, memory_config=mc)
    if hasattr(ttnn, "ReplicateTensorToMesh"):
        kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
    on_device = ttnn.from_torch(x, **kwargs)
    with host_side_tensor_construction():
        via_host = ttnn.from_torch(x, **kwargs)
    _same(on_device, via_host)


@pytest.mark.parametrize("shape, dtype, layout, memory_config", CASES[3:])
def test_reshard_from_dram_matches_host_round_trip(device, shape, dtype, layout, memory_config):
    # The mesh helpers build DRAM-interleaved first and reshard, so logical shape survives a
    # shard taller than the tensor; the host route must preserve that too.
    x = _torch_input(shape, dtype)
    dram = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    on_device = ttnn.to_memory_config(dram, memory_config())
    with host_side_tensor_construction():
        via_host = ttnn.to_memory_config(dram, memory_config())
    assert via_host.shape == dram.shape and via_host.padded_shape == on_device.padded_shape
    _same(on_device, via_host)


def test_cq_id_is_forwarded_as_queue_id(device):
    x = _torch_input((1, 1, 32, 32), ttnn.bfloat16)
    with host_side_tensor_construction():
        t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, cq_id=0)
    assert torch.equal(ttnn.to_torch(t), x)
