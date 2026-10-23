# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# all_gather cases that route to the composite path (all_broadcast + concat): a persistent output must be written
# in place and sub_core_grid must reach the composite's ops. Every device is checked bit-exactly.

import pytest
import torch
import ttnn

from ttnn import ShardTensorToMesh
from tt_py_test_utils_common.utility_functions import skip_for_wormhole_b0

_LINE = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}
_RING = {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}

# Gathered (output) shape, dim, layout. Each routes to composite: a tile slice padded on the gather dim, or
# row-major rows padded to the memory alignment.
_SHAPES = {
    "w_padded": ([1, 1, 64, 320], 3, ttnn.TILE_LAYOUT),  # 40 columns per device
    "h_padded": ([1, 2, 64, 96], 2, ttnn.TILE_LAYOUT),  # 8 rows per device
    "rm_w": ([1, 1, 32, 192], 3, ttnn.ROW_MAJOR_LAYOUT),  # 48 B rows per device
}


def _grid(x0, y0, x1, y1):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))})


# "disjoint" leaves out core (0, 0), where the default worker selection starts. "minimal" is one core per link of a
# 2-link box.
_GRIDS = {"default": None, "disjoint": _grid(3, 4, 5, 5), "minimal": _grid(6, 2, 7, 2)}


def _tensor(mesh_device, torch_tensor, layout, mesh_mapper):
    return ttnn.from_torch(
        torch_tensor,
        device=mesh_device,
        layout=layout,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )


def _gather(mesh_device, shape_id, out, grid, subdevice_id=None):
    shape, dim, layout = _SHAPES[shape_id]
    torch.manual_seed(0)
    full = torch.randn(shape, dtype=torch.bfloat16)
    tt_in = _tensor(mesh_device, full, layout, ShardTensorToMesh(mesh_device, dim=dim))
    tt_out = ttnn.all_gather(tt_in, dim=dim, output_tensor=out, subdevice_id=subdevice_id, sub_core_grids=grid)
    ttnn.synchronize_device(mesh_device)
    return tt_out, full


def _check(tt_out, expected, label):
    for d, dev_out in enumerate(ttnn.get_device_tensors(tt_out)):
        got = ttnn.to_torch(dev_out)
        assert got.shape == expected.shape and torch.equal(got, expected), f"{label}: device {d} mismatch"


@skip_for_wormhole_b0()
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [_LINE, _RING], indirect=True, ids=["line", "ring"])
@pytest.mark.parametrize("shape_id", list(_SHAPES))
@pytest.mark.parametrize("grid_id", list(_GRIDS))
@pytest.mark.parametrize("persistent", [True, False], ids=["persistent", "fresh"])
def test_all_gather_composite(mesh_device, device_params, shape_id, grid_id, persistent):
    out = None
    if persistent:
        shape, _, layout = _SHAPES[shape_id]
        # A sentinel, so a region the gather never writes cannot pass.
        sentinel = torch.full(shape, -3.0, dtype=torch.bfloat16)
        out = _tensor(mesh_device, sentinel, layout, ttnn.ReplicateTensorToMesh(mesh_device))
    tt_out, expected = _gather(mesh_device, shape_id, out, _GRIDS[grid_id])
    _check(tt_out, expected, shape_id)
    if out is not None:
        assert tt_out.buffer_address() == out.buffer_address(), "composite did not return the persistent output"
        _check(out, expected, f"{shape_id} (persistent output)")


# sub_core_grid must reach the composite's ops: a different grid is a different program, so it adds cache entries.
# An explicit SubDeviceId takes the same path.
@skip_for_wormhole_b0()
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [_LINE], indirect=True, ids=["line"])
def test_all_gather_composite_grid_reaches_ops(mesh_device, device_params):
    subdevice_id = ttnn.SubDeviceId(0)
    tt_out, expected = _gather(mesh_device, "h_padded", None, _GRIDS["disjoint"], subdevice_id)
    _check(tt_out, expected, "disjoint")
    entries = mesh_device.num_program_cache_entries()
    tt_out, expected = _gather(mesh_device, "h_padded", None, _GRIDS["minimal"], subdevice_id)
    _check(tt_out, expected, "minimal")
    assert mesh_device.num_program_cache_entries() > entries, "sub_core_grid did not reach the composite's ops"


# ttnn::concat assumes 32x32 tiles and would scramble any other tile, so composite must refuse it.
@skip_for_wormhole_b0()
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [_LINE], indirect=True, ids=["line"])
def test_all_gather_composite_refuses_tiny_tile(mesh_device, device_params, expect_error):
    # 8 rows per device on 16x32 tiles: padded on the gather dim, so composite.
    tt_in = ttnn.from_torch(
        torch.randn([1, 1, 64, 64], dtype=torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        tile=ttnn.Tile((16, 32)),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=2),
    )
    with expect_error(RuntimeError, "32x32 tiles"):
        ttnn.all_gather(tt_in, dim=2)
        ttnn.synchronize_device(mesh_device)
