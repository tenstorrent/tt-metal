# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1 — integer dtype passthrough for point_to_point (uint16/int32/uint32).

point_to_point is pure data movement: the dataflow kernels are dtype-agnostic byte
copies (tt_memmove + noc_async_read/write sized in bytes). Integer dtypes therefore
transfer bit-for-bit exactly, so the oracle here is EXACT integer equality (torch.equal),
not a PCC band — an integer identity copy has no rounding error.

Both layouts (TILE, ROW_MAJOR) are exercised for every integer dtype: unlike bfloat8_b,
integers have a row-major representation, so there is no INVALID (dtype, layout) pair.

Verified on the multi-device craq-sim, e.g.:

  source python_env/bin/activate
  scripts/run_multidevice_sim_pytest.py --topology bh_8xP150_p2p -- \
      tests/ttnn/unit_tests/operations/point_to_point/test_point_to_point_integer_dtypes.py \
      -k "int32 and TILE"
"""

from math import prod

import pytest
import torch
from loguru import logger

import ttnn

from ttnn.operations.point_to_point import point_to_point

# ttnn integer dtype -> torch dtype for round-tripping. torch has no native uint16/uint32,
# so those are staged as int32 (values kept well within uint16 range so nothing wraps).
_TORCH_DTYPE = {
    ttnn.uint16: torch.int32,
    ttnn.int32: torch.int32,
    ttnn.uint32: torch.int32,
}

# Integer dtype x layout — every combination must pass (no INVALID for integers).
DTYPE_LAYOUTS = [
    (ttnn.uint16, ttnn.TILE_LAYOUT),
    (ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT),
    (ttnn.int32, ttnn.TILE_LAYOUT),
    (ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
    (ttnn.uint32, ttnn.TILE_LAYOUT),
    (ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
]

# Per-device shard shapes: single-tile, multi-tile, non-tile-aligned. Last dims are
# multiples of 8 so the row-major page size stays 16-byte aligned for every width.
SHARD_SHAPES = [
    (1, 1, 32, 32),  # single tile
    (1, 1, 64, 128),  # multi-tile
    (1, 1, 48, 64),  # non-tile-aligned (H not %32), 16B-aligned page
]

MESH_SHAPE = (2, 4)
FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}
TOPOLOGIES = [ttnn.Topology.Linear, ttnn.Topology.Ring]


def _linear_index(coord, mesh_shape):
    mesh_shape = tuple(mesh_shape)
    return coord[0] * mesh_shape[1] + coord[1]


def _make_input(mesh_device, shard_shape, dtype, layout):
    """Shard a freshly-seeded INTEGER tensor along dim 0 across the mesh."""
    num_devices = prod(tuple(mesh_device.shape))
    full_shape = (shard_shape[0] * num_devices, *shard_shape[1:])

    torch.manual_seed(42)
    # 0..999 fits every integer width (uint16 max 65535); distinct per element so a
    # byte-swap / reorder bug would show up as a mismatch, not a lucky pass.
    torch_full = torch.randint(0, 1000, full_shape, dtype=_TORCH_DTYPE[dtype])

    input_tensor = ttnn.from_torch(
        torch_full,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn.synchronize_device(mesh_device)

    input_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(input_tensor)]
    return input_tensor, input_shards


@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("dtype, layout", DTYPE_LAYOUTS)
@pytest.mark.parametrize("shard_shape", SHARD_SHAPES)
def test_point_to_point_integer(mesh_device, topology, dtype, layout, shard_shape):
    """Receiver shard == sender shard EXACTLY (integer identity byte copy); sender unchanged."""
    if prod(tuple(mesh_device.shape)) < 2:
        pytest.skip("point_to_point requires at least 2 mesh devices")

    sender_coord = ttnn.MeshCoordinate(0, 0)
    receiver_coord = ttnn.MeshCoordinate(0, 1)
    send_idx = _linear_index(sender_coord, mesh_device.shape)
    recv_idx = _linear_index(receiver_coord, mesh_device.shape)

    input_tensor, input_shards = _make_input(mesh_device, shard_shape, dtype, layout)

    output_tensor = point_to_point(input_tensor, sender_coord, receiver_coord, topology=topology)
    ttnn.synchronize_device(mesh_device)

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]

    # Exact integer equality — pure byte copy, zero rounding error.
    assert torch.equal(input_shards[send_idx], output_shards[recv_idx]), (
        f"receiver shard != sender shard for {dtype} {layout} {shard_shape} {topology}: "
        f"max abs diff {(input_shards[send_idx].to(torch.int64) - output_shards[recv_idx].to(torch.int64)).abs().max()}"
    )
    # Sender's own shard is unchanged.
    assert torch.equal(input_shards[send_idx], output_shards[send_idx]), "sender shard was mutated"
    logger.info(f"p2p integer {dtype} {layout} {shard_shape} {topology}: exact receiver==sender match")
