# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Extended (verifier-authored) coverage for point_to_point.

The immutable acceptance test (test_point_to_point.py) covers
bfloat16/float32/bfloat8_b x TILE/ROW_MAJOR x {4 tile-aligned shapes} x {Linear,Ring}.
It does NOT exercise two axis values that SUPPORTED claims:

  1. the integer dtypes (uint16, int32, uint32) — pure byte-copy paths, and
  2. non-tile-aligned shapes (H or W not a 32-multiple, but 16-byte page aligned).

This file fills exactly those two gaps with a deliberately small matrix. It is NOT a
substitute for the golden suite — it is targeted coverage for the SUPPORTED cells the
acceptance test leaves untouched.

Like the acceptance test, this op is inherently multi-device: the `mesh_device` fixture
auto-skips on machines with fewer than 2 devices. The transfer is a bit-exact copy, so
correctness is integer/bitwise equality (assert_equal), not an approximate PCC.
"""

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_equal, assert_with_pcc


def _linear(coord, mesh_shape):
    return coord[0] * mesh_shape[1] + coord[1]


# Integer / byte-copy dtypes not covered by the acceptance test. All are TILE+ROW_MAJOR
# capable (no block quantization), so the copy must be bit-exact.
INT_DTYPES = [
    pytest.param(ttnn.uint16, torch.int16, id="uint16"),
    pytest.param(ttnn.int32, torch.int32, id="int32"),
    pytest.param(ttnn.uint32, torch.int32, id="uint32"),
]

# Non-tile-aligned shapes from the golden feature_spec INPUTS (last two dims not both
# 32-multiples) that are still 16-byte page aligned for the dtypes under test.
NON_TILE_ALIGNED_SHAPES = [
    (1, 1, 47, 64),  # H not tile-aligned
    (1, 1, 32, 48),  # W a 16-multiple but not a 32-multiple
    (1, 13, 1, 32),  # multi-channel, tiny H
]


def _make_input(mesh_device, shape, ttnn_dtype, torch_dtype, layout):
    mesh_shape = tuple(mesh_device.shape)
    num_devices = mesh_shape[0] * mesh_shape[1]
    S = shape[0]
    global_shape = (S * num_devices,) + tuple(shape[1:])

    if torch_dtype in (torch.int16, torch.int32):
        torch_input = torch.randint(-(2**14), 2**14, global_shape, dtype=torch_dtype)
    else:
        torch_input = torch.randn(global_shape, dtype=torch.float32).to(torch_dtype)

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn_dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    return torch_input, input_tensor, num_devices, S


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
@pytest.mark.parametrize("ttnn_dtype, torch_dtype", INT_DTYPES)
def test_point_to_point_integer_dtypes(mesh_device, layout, ttnn_dtype, torch_dtype):
    """uint16 / int32 / uint32 are pure byte copies — the receiver shard must be bit-exact."""
    torch.manual_seed(7)
    shape = (1, 1, 64, 128)
    mesh_shape = tuple(mesh_device.shape)

    torch_input, input_tensor, num_devices, S = _make_input(mesh_device, shape, ttnn_dtype, torch_dtype, layout)

    sender, receiver = (0, 0), (0, 1)
    out = point_to_point_call(input_tensor, sender, receiver)
    out_torch = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    sl, rl = _linear(sender, mesh_shape), _linear(receiver, mesh_shape)
    expected = torch_input[sl * S : (sl + 1) * S]
    received = out_torch[rl * S : (rl + 1) * S].to(torch_dtype)
    assert_equal(expected, received)


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
@pytest.mark.parametrize("shape", NON_TILE_ALIGNED_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_point_to_point_non_tile_aligned(mesh_device, layout, shape):
    """Non-tile-aligned shapes (alignment=non_tile_aligned) must still copy correctly."""
    torch.manual_seed(11)
    ttnn_dtype, torch_dtype, pcc = ttnn.bfloat16, torch.bfloat16, 0.995
    mesh_shape = tuple(mesh_device.shape)

    torch_input, input_tensor, num_devices, S = _make_input(mesh_device, shape, ttnn_dtype, torch_dtype, layout)

    sender, receiver = (0, 0), (0, 1)
    out = point_to_point_call(input_tensor, sender, receiver)
    out_torch = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    sl, rl = _linear(sender, mesh_shape), _linear(receiver, mesh_shape)
    expected = torch_input[sl * S : (sl + 1) * S]
    received = out_torch[rl * S : (rl + 1) * S]
    assert_with_pcc(expected, received, pcc)


def point_to_point_call(input_tensor, sender, receiver, topology=ttnn.Topology.Linear):
    """Thin wrapper so the import lives next to use (keeps top-of-file import order clean)."""
    from ttnn.operations.point_to_point import point_to_point

    return point_to_point(
        input_tensor,
        ttnn.MeshCoordinate(*sender),
        ttnn.MeshCoordinate(*receiver),
        topology=topology,
    )
