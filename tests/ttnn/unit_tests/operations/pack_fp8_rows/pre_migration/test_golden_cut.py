# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Golden coverage cut of pack_fp8_rows (ttnn/ttnn/bringup/fp8_rows_ttnn/tests/unit/test_pack_fp8_rows.py).

The golden suite's axes: the row count M in {32, 96, 320, 640} (smallest 32, largest 640), the cast's input layout
(bf16 TILE, and ROW_MAJOR for M <= 96), the whole box as one mesh (FABRIC_2D, per-chip keys), max_grid
(min(grid.x, 11), grid.y), and an allocated or caller's output. Every value appears in at least one case below (not
their product); each case is checked with the golden suite's own references and tolerances: every byte equal to the
cast's raw outputs packed on the host, the scales byte-equal to Blaze's recipe and the values within one e4m3 step of
it, pad rows and padding bytes zero. The golden suite has no SUPPORTED / TARGET / INVALID tables."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import ttnn

import pack_fp8_rows_pre_migration_helpers as H

# The route-selecting fixture lives in test_acceptance.py (driver-written); load it under a private name.
_spec = importlib.util.spec_from_file_location(
    "_pack_fp8_rows_pre_migration_routes", Path(__file__).with_name("test_acceptance.py")
)
_routes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_routes)
pack_fp8_rows_op = _routes.pack_fp8_rows_op


def _grid(mesh):
    g = mesh.compute_with_storage_grid_size()
    return ttnn.CoreCoord(min(g.x, 11), g.y)  # (a Galaxy's 110-core cap, as the golden suite)


def _dev(t, mesh, layout):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=ttnn.bfloat16,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def _bytes(t, nd, m):
    """uint8 [nd, m, 192] of a device uint8 tensor (the golden suite's readback)."""
    return torch.cat([ttnn.to_torch(x).reshape(1, m, 192) for x in ttnn.get_device_tensors(t)])[:nd]


GOLDEN_CASES = [
    # (m, cast input layout, caller's output buffer)
    pytest.param(32, ttnn.TILE_LAYOUT, False, id="m32_smallest_tile_cast"),
    pytest.param(96, ttnn.ROW_MAJOR_LAYOUT, False, id="m96_rm_cast"),
    pytest.param(320, ttnn.TILE_LAYOUT, True, id="m320_tile_cast_caller_out"),
    pytest.param(640, ttnn.TILE_LAYOUT, False, id="m640_largest_tile_cast"),
    pytest.param(640, ttnn.TILE_LAYOUT, True, id="m640_largest_caller_out"),
]


@pytest.mark.parametrize(("m", "layout", "caller_out"), GOLDEN_CASES)
def test_golden_cut(golden_mesh, pack_fp8_rows_op, m, layout, caller_out):
    mesh = golden_mesh
    nd = mesh.get_num_devices()
    keys = H.golden_keys(nd, m, seed=m)
    x = _dev(keys, mesh, layout)
    e4m3, sc = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x, round_scale_to_power_of_two=False)
    out = None
    if caller_out:
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, m, 192]), ttnn.uint8, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        )
    rows = pack_fp8_rows_op(e4m3, sc, output_tensor=out, max_grid=_grid(mesh))
    if caller_out:
        assert rows is out
    H.assert_output_spec(rows, e4m3)
    got = _bytes(rows, nd, m)
    H.assert_rows_equal(got, H.golden_op_rows(e4m3, sc, nd, m), "rows vs the cast's raw outputs")
    H.golden_within_one_step(got, H.golden_recipe(keys.reshape(nd, m, 128)))
    assert (got[:, 2] == 0).all() and (got[:, 5] == 0).all(), "pad rows not all zero"
    assert (got[..., 132:] == 0).all(), "padding bytes not zero"
    for t in (x, e4m3, sc, rows):
        ttnn.deallocate(t)
