# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Golden coverage cut of pair_pool (ttnn/ttnn/bringup/pair_pool_ttnn/tests/unit/test_pair_pool.py), on its topology.

The golden suite runs the whole box as one mesh, a different random input on every chip (ShardTensorToMesh dim 0),
max_grid (min(grid.x, 11), grid.y), over CASES (R, hd) = (640, 512), (1280, 512), (64, 32), (192, 96); reference float64
(its ``_want``), tolerance relative norm <= 1e-6, plus negative controls. The cut takes all four (every R and hd value
once; the smallest (64, 32) and the largest (1280, 512) among them), checked with that reference and tolerance
(conftest ``check``), and a fresh-buffer cache hit on the largest. Not taken: the golden's ulp comparison against the
device op chain (it is not an independent CPU reference) and its FABRIC_2D setting (the op sends nothing over fabric).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import ttnn

# The one call point of the suite (route selection, native guard) lives in test_acceptance.py; reuse its fixture.
_spec = importlib.util.spec_from_file_location(
    "pair_pool_pre_migration_acceptance", Path(__file__).with_name("test_acceptance.py")
)
_acceptance = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_acceptance)
pair_pool_op = _acceptance.pair_pool_op

GOLDEN_CASES = [(640, 512), (1280, 512), (64, 32), (192, 96)]


def _grid(mesh):
    g = mesh.compute_with_storage_grid_size()
    return ttnn.CoreCoord(min(g.x, 11), g.y)


def _dev(x, mesh):
    return ttnn.from_torch(
        x,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def _host(t, mesh):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))


@pytest.mark.parametrize("R, hd", GOLDEN_CASES, ids=[f"R{r}-hd{h}" for r, h in GOLDEN_CASES])
def test_golden_cut(mesh_device, pair_pool_op, pp, R, hd):
    nd = mesh_device.get_num_devices()
    x = pp.make_input((nd, 1, R, 2 * hd), seed=R + hd)
    xd = _dev(x, mesh_device)
    out = pair_pool_op(xd, max_grid=_grid(mesh_device))
    assert list(out.shape) == [1, 1, R // 2, hd] and out.dtype == ttnn.float32 and out.layout == ttnn.TILE_LAYOUT
    assert out.memory_config().buffer_type == ttnn.BufferType.DRAM
    got = _host(out, mesh_device)
    pp.check(got, x, label=f"mesh R{R} hd{hd}")
    assert torch.equal(_host(xd, mesh_device), x), "pair_pool modified its input"


def test_golden_cut_largest_cache_hit(mesh_device, pair_pool_op, pp):
    """The largest golden input on the mesh: a miss, then a hit on fresh buffers with new data, and a hit into a
    caller's buffer; every result from its own call's input."""
    nd = mesh_device.get_num_devices()
    R, hd = 1280, 512
    shape = (nd, 1, R, 2 * hd)
    kept = []
    for i, seed in enumerate((101, 102)):
        x = pp.make_input(shape, seed=seed)
        xd = _dev(x, mesh_device)
        before = mesh_device.num_program_cache_entries()
        out = pair_pool_op(xd, max_grid=_grid(mesh_device))
        delta = mesh_device.num_program_cache_entries() - before
        assert delta == (1 if i == 0 else 0), f"call {i}: {delta} new program(s)"
        kept.append((x, xd, out))
    x = pp.make_input(shape, seed=103)
    xd = _dev(x, mesh_device)
    buf = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, R // 2, hd]), ttnn.float32, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
    )
    before = mesh_device.num_program_cache_entries()
    r = pair_pool_op(xd, output_tensor=buf, max_grid=_grid(mesh_device))
    assert mesh_device.num_program_cache_entries() - before == 0
    assert r.buffer_address() == buf.buffer_address(), "the result must be the caller's buffer"
    kept.append((x, xd, buf))
    assert len({o.buffer_address() for _, _, o in kept}) == 3
    for x, _, out in kept:
        pp.check(_host(out, mesh_device), x, label="mesh hit")
