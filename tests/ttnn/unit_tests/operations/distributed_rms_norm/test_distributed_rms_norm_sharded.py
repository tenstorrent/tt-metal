# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 1: HEIGHT_SHARDED / WIDTH_SHARDED inputs, native zero-copy (cb_x_tiles / cb_output on the shards).

Shard specs come from the golden suite's own chooser (eval.sharding.auto_shard_config via create_ttnn_input_tensor),
so the shard grid fills the compute grid from (0, 0) and covers the row-0 port cores: the ports must move to free cores.
  HEIGHT -> R1-X over the shard grid (each shard is whole local rows; the shard is the block).
  WIDTH  -> R3-X with ONE row-group = the shard grid (every core holds all rows of its W-slice; ragged last slice on
            8192: 256 tiles over 86 cores of 3 -> the last holds 1 valid tile), rows split over R3-RS owners (each
            sums / publishes / finalizes R / owners rows and unicasts their inv to every member).
Also: a sharded input written to a different output placement (DRAM interleaved) goes through the streaming writer,
and waves over a shard-backed block (BLOCK_ROWS_CAP) with alternating inputs (a stale cross-call partial would show).

Mesh: the root conftest `mesh_device` (system mesh), FABRIC_2D_TORUS_XY, group along cluster_axis 1.
"""

import pytest
import torch
import ttnn

from eval.golden_tests.distributed_rms_norm.helpers import (
    check_output,
    create_ttnn_gamma,
    create_ttnn_input_tensor,
    pytorch_distributed_rms_norm,
    random_stacked,
)
from ttnn.operations.distributed_rms_norm import distributed_rms_norm
from ttnn.operations.distributed_rms_norm import distributed_rms_norm_program_descriptor as pd

AXIS = 1
HEIGHT = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
WIDTH = ttnn.TensorMemoryLayout.WIDTH_SHARDED
_TORCH = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}


def _router_config():
    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = (14 if "blackhole" in ttnn.get_arch_name() else 7) * 1024 + 64
    return cfg


FABRIC = [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY, "fabric_router_config": _router_config()}]


def _case(mesh, shape, layout, *, dtype=ttnn.bfloat16, gamma_dtype=ttnn.bfloat16, seed=7):
    x = random_stacked(mesh, shape, _TORCH[dtype], seed=seed)
    g = None
    if gamma_dtype is not None:
        g = random_stacked(mesh, (1, 1, 1, shape[-1]), _TORCH[gamma_dtype], seed=seed + 100)
    tx = create_ttnn_input_tensor(x, mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_layout=layout)
    tg = None if g is None else create_ttnn_gamma(g, mesh, dtype=gamma_dtype, layout=ttnn.TILE_LAYOUT)
    return tx, tg, pytorch_distributed_rms_norm(x, g, AXIS, 1e-6)


def _check(out, expected, shape, dtype=ttnn.bfloat16):
    check_output(out, expected, shape=shape, dtype=dtype, fp32_dest_acc_en=True, expected_layout=ttnn.TILE_LAYOUT)


SHAPES = [
    ((1, 1, 32, 1792), HEIGHT),
    ((1, 1, 640, 1792), HEIGHT),
    ((1, 1, 4096, 1024), HEIGHT),  # 2 tile-rows per shard
    ((2, 4, 128, 512), HEIGHT),
    ((1, 1, 32, 1792), WIDTH),
    ((1, 1, 640, 1792), WIDTH),
    ((1, 1, 32, 8192), WIDTH),  # ragged last slice, 3-tile shards
    ((1, 1, 32, 64), WIDTH),
    ((1, 1, 4096, 1024), WIDTH),  # R3-RS: 32 owners x 4 rows
]


@pytest.mark.parametrize("device_params", FABRIC, indirect=True)
@pytest.mark.parametrize("gamma", [True, False], ids=["gamma", "no_gamma"])
@pytest.mark.parametrize(
    "shape,layout", SHAPES, ids=[f"{'x'.join(map(str, s))}-{l.name.split('_')[0]}" for s, l in SHAPES]
)
def test_sharded(mesh_device, shape, layout, gamma):
    tx, tg, expected = _case(mesh_device, shape, layout, gamma_dtype=ttnn.bfloat16 if gamma else None)
    out = distributed_rms_norm(tx, cluster_axis=AXIS, gamma=tg, memory_config=tx.memory_config())
    assert out.memory_config().memory_layout == layout
    _check(out, expected, shape)


@pytest.mark.parametrize("device_params", FABRIC, indirect=True)
@pytest.mark.parametrize("layout", [HEIGHT, WIDTH], ids=["HEIGHT", "WIDTH"])
def test_sharded_fp32_ring_links(mesh_device, layout):
    shape = (1, 1, 128, 1024)
    tx, tg, expected = _case(mesh_device, shape, layout, dtype=ttnn.float32, gamma_dtype=ttnn.float32)
    out = distributed_rms_norm(tx, cluster_axis=AXIS, gamma=tg, topology=ttnn.Topology.Ring, num_links=None)
    assert out.memory_config().memory_layout == layout  # default output placement: the input's own
    _check(out, expected, shape, dtype=ttnn.float32)


@pytest.mark.parametrize("device_params", FABRIC, indirect=True)
@pytest.mark.parametrize("layout", [HEIGHT, WIDTH], ids=["HEIGHT", "WIDTH"])
def test_sharded_input_interleaved_output(mesh_device, layout):
    shape = (1, 1, 256, 2048)  # WIDTH: 64 one-tile slices; HEIGHT: one tile-row per shard, 64 tiles wide
    tx, tg, expected = _case(mesh_device, shape, layout)
    out = distributed_rms_norm(tx, cluster_axis=AXIS, gamma=tg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    _check(out, expected, shape)


@pytest.mark.parametrize("device_params", FABRIC, indirect=True)
@pytest.mark.parametrize(
    "shape,layout,cap,owners",
    [((1, 1, 4096, 1024), HEIGHT, 1, 1), ((1, 1, 256, 512), WIDTH, 1, 1), ((1, 1, 256, 512), WIDTH, 3, 3)],
    ids=["HEIGHT-2rows", "WIDTH-8rows", "WIDTH-RS-ragged"],
)
def test_sharded_waves_alternating(mesh_device, monkeypatch, shape, layout, cap, owners):
    """Several blocks over one resident shard. WIDTH-RS-ragged: R = 3 over 8 rows -> blocks 3, 3, 2 with 3 owners of
    one row each; in the last block owner 2 owns nothing (it still counts at the ports)."""
    monkeypatch.setattr(pd, "BLOCK_ROWS_CAP", cap)
    cases = [_case(mesh_device, shape, layout, seed=s) for s in (11, 12)]
    plan = pd.make_plan(mesh_device, cases[0][0], cases[0][1], AXIS, ttnn.Topology.Linear, 1)
    assert plan.x_res and plan.out_res and plan.num_waves >= 2 and plan.owners == owners, plan
    for k in range(4):
        tx, tg, expected = cases[k % 2]
        _check(distributed_rms_norm(tx, cluster_axis=AXIS, gamma=tg, num_links=1), expected, shape)


# The production topology: the box's snake ring opened as 1xN (a real wrap), Ring and Linear, 2 links where available.
def _ring_meshes():
    from eval.golden_tests.distributed_rms_norm.topology import ring_mock_meshes

    return [shape for shape, axis in ring_mock_meshes() if axis == AXIS]


@pytest.mark.skipif(not _ring_meshes(), reason="no 1xN snake ring on this cluster")
@pytest.mark.parametrize("device_params", FABRIC, indirect=True)
@pytest.mark.parametrize("mesh_device", _ring_meshes()[:1], indirect=True)
@pytest.mark.parametrize("topology", [ttnn.Topology.Ring, ttnn.Topology.Linear], ids=["Ring", "Linear"])
@pytest.mark.parametrize(
    "shape,layout",
    [((1, 1, 640, 1792), HEIGHT), ((1, 1, 640, 1792), WIDTH), ((1, 1, 32, 1792), WIDTH)],
    ids=["640x1792-HEIGHT", "640x1792-WIDTH", "32x1792-WIDTH"],
)
def test_sharded_ring(mesh_device, shape, layout, topology):
    tx, tg, expected = _case(mesh_device, shape, layout)
    if shape[-2] == 640 and layout == WIDTH:
        plan = pd.make_plan(mesh_device, tx, tg, AXIS, topology, 2)
        assert plan.owners == 20 and plan.L_act == 2, plan  # 20 rows over 20 owners, split over both links
    out = distributed_rms_norm(tx, cluster_axis=AXIS, gamma=tg, topology=topology, memory_config=tx.memory_config())
    _check(out, expected, shape)
