# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""exchange_mode: "direct", "relay" and "mcast" schedules of the Σx² exchange.

Every mode must match the torch reference and give a bit-identical output to "direct" (the fold order is fixed by
slot, so the schedule cannot change a bit). Each call runs twice back to back, so a missed re-arm of the ready /
arrival / credit counters shows up as a hang or stale rows on the repeat.

Fabrics (a 1x4 mesh, opened per test by the root mesh_device fixture with the production router payload):
  torus2d  FABRIC_2D_TORUS_XY, Ring: direct and relay; mcast is refused (a 2D line multicast over the bent ring of the
           QuietBox 2 torus mock hangs)
  ring1d   FABRIC_1D_RING, Ring: all three
  line1d   FABRIC_1D, Linear: all three

Run this file on its own (it opens its own meshes, unlike the op's other unit tests, which share the golden suite's
torus mesh):

    scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/distributed_rms_norm/test_distributed_rms_norm_exchange_modes.py
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from eval.golden_tests.distributed_rms_norm.conftest import production_router_config
from ttnn.operations.distributed_rms_norm import distributed_rms_norm
from ttnn.operations.distributed_rms_norm import distributed_rms_norm_program_descriptor as pd

MODES = ("direct", "relay", "mcast")
FABRICS = [
    pytest.param(ttnn.FabricConfig.FABRIC_2D_TORUS_XY, ttnn.Topology.Ring, id="torus2d"),
    pytest.param(ttnn.FabricConfig.FABRIC_1D_RING, ttnn.Topology.Ring, id="ring1d"),
    pytest.param(ttnn.FabricConfig.FABRIC_1D, ttnn.Topology.Linear, id="line1d"),
]


def _params(fabric):
    return {"fabric_config": fabric, "fabric_router_config": production_router_config()}


def _inputs(mesh, shape, seed):
    G = mesh.shape[1]
    B, _, M, W = shape
    torch.manual_seed(seed)
    x = torch.randn(B, G, M, W).to(torch.bfloat16)
    gamma = torch.randn(1, G, 1, W).to(torch.bfloat16)
    mapper = ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=tuple(mesh.shape))
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=mapper)
    tg = ttnn.from_torch(gamma, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=mapper)
    full = x.float().permute(0, 2, 1, 3).reshape(B, M, G * W)
    ref = full * torch.rsqrt(full.pow(2).mean(-1, keepdim=True) + 1e-6)
    ref = ref.reshape(B, M, G, W).permute(0, 2, 1, 3) * gamma.float()
    return tx, tg, ref


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def _run_modes(mesh, fabric, topology, shape, num_links, expect_error):
    """{mode: output} for every mode this fabric runs (each called twice); mcast on 2D must refuse."""
    tx, tg, ref = _inputs(mesh, shape, seed=0)
    concat = ttnn.ConcatMeshToTensor(mesh, dim=1)
    outs = {}
    for mode in MODES:
        call = lambda: distributed_rms_norm(  # noqa: E731
            tx, cluster_axis=1, gamma=tg, topology=topology, num_links=num_links, exchange_mode=mode
        )
        if mode == "mcast" and pd.is_2d_fabric(fabric):
            with expect_error(NotImplementedError, "1D fabric"):
                call()
            continue
        for _ in range(2):
            try:
                got = ttnn.to_torch(call(), mesh_composer=concat)
            except ValueError as e:
                if "num_links" in str(e):
                    pytest.skip(f"{num_links} links not available on this fabric: {e}")
                raise
            pcc = _pcc(got, ref)
            assert pcc > 0.9995, f"{mode}: pcc {pcc}"
        built = pd.LAST_EXCHANGE_LAYOUT["exchange_mode"]
        # direct may fall back to relay where the fabric routes a destination off the op's ports; the others never
        assert built == mode or (mode == "direct" and built == "relay"), f"asked {mode}, built {built}"
        outs[mode] = got
    for mode, got in outs.items():
        assert torch.equal(got, outs["direct"]), f"{mode} differs from direct"
    return outs


@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["mesh1x4"], indirect=True)
@pytest.mark.parametrize("fabric,topology", FABRICS)
@pytest.mark.parametrize("num_links", [1, 2])
@pytest.mark.parametrize(
    "shape", [(1, 1, 640, 1792), (1, 1, 32, 1792), (1, 1, 2048, 1280)], ids=["640x1792", "32x1792", "2048x1280"]
)
def test_exchange_modes(shape, num_links, fabric, topology, mesh_device, expect_error):
    _run_modes(mesh_device, fabric, topology, shape, num_links, expect_error)


@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["mesh1x4"], indirect=True)
@pytest.mark.parametrize("fabric,topology", FABRICS)
def test_exchange_modes_bounded_landing(fabric, topology, mesh_device, monkeypatch, expect_error):
    """The bounded landing (credits on the ready path) under every mode: forced ring of 2 wave slots."""
    monkeypatch.setattr(pd, "LANDING_FULL_MAX_BYTES", 0)
    monkeypatch.setattr(pd, "LANDING_RING_WAVES", 2)
    monkeypatch.setattr(pd, "MIN_BLOCKS_PER_CORE", 8)
    _run_modes(mesh_device, fabric, topology, (4, 1, 4096, 1792), 1, expect_error)
    lay = pd.LAST_EXCHANGE_LAYOUT
    assert lay["landing_waves"] < lay["num_waves"], f"bounded landing not engaged: {lay}"


@pytest.fixture
def device_params(fabric):
    """The fabric under test, as the root mesh_device fixture's device_params."""
    return _params(fabric)
