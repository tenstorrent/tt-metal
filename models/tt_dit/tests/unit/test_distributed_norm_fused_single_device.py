# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Single-device (1x1 mesh, ring_size == 1) fused RMSNorm / LayerNorm at Wan2.2 LoudBox per-device shapes."""

from __future__ import annotations

import pytest
from loguru import logger

import ttnn

from models.tt_dit.tests.unit import test_distributed_rmsnorm_fused as t

WAN = t.WAN
ROWS = [8192, 16384]
DIMS = [5120, 2560]


def _run(mesh_device, cfg):
    sem = t.CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear).get_ag_ping_pong_semaphore(1)
    inp = t._build(mesh_device, cfg, 1)
    pob = t._make_pob(inp, mesh_device, cfg, 1, 1)
    ref = t._torch_ref(cfg)
    out0 = t._gather(t._run_fused(inp, mesh_device, sem, cfg, ttnn.Topology.Linear, 1, pob, 1), 1)
    out1 = t._gather(t._run_fused(inp, mesh_device, sem, cfg, ttnn.Topology.Linear, 1, pob, 1), 1)
    pcc = t._pcc(out0, ref)
    logger.info(f"NORM1D {cfg.cid} pcc={pcc * 100:.4f}% maxabs={(out0 - ref).abs().max().item():.4f}")
    assert (out1 - out0).abs().max().item() == 0.0
    assert pcc >= 0.999


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("rope", [True, False], ids=["rope", "norope"])
def test_rmsnorm_qk_single_device(mesh_device, rows, dim, rope):
    cfg = t.Cfg(f"rms_N{rows}_D{dim}_rope{int(rope)}", WAN, 1, rows, dim, 128, rope, dim // 128, True)
    _run(mesh_device, cfg)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("dim", DIMS)
def test_layernorm_adaln_single_device(mesh_device, rows, dim):
    cfg = t.Cfg(f"ln_N{rows}_D{dim}", WAN, 1, rows, dim, None, False, dim // 128, True, norm="layernorm")
    _run(mesh_device, cfg)
