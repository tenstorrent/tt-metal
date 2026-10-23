# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Verifier regression tests for distributed_rms_norm: the two kernel bugs found in verification. DO NOT DELETE.

1. test_waves_alternating_inputs — `SEM_OWN_STATS` on a sending port counts publishes cumulatively over waves. Before the
   `SEM_OWN_COUNTED` gate, a root whose next block was empty incremented for wave w+1 immediately, standing in for a
   slower root's still-unpublished wave-w rows, so the port shipped wave w with stale rows (R1, num_waves = 2: ~40 of
   108 rows wrong per call on the p = 1 devices, invisible when the same input is repeated). The test alternates two
   different inputs on one plan so a stale partial from the previous call shows up as a wrong scale.
2. test_chunked_multi_row_store_order — compute emits output quanta chunk-outer / tile-row-inner; the writer consumed
   them tile-row-outer. With R_b >= 2 rows per core AND nch >= 2 W-chunks (Wt > W_CHUNK_TILES) row 1's chunk 0 was
   stored where row 0's chunk 1 belongs.

Mesh from the root conftest `mesh_device` fixture (2x2 system mesh, group along cluster_axis 1, Linear, 1 link).
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


def _router_config():
    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = (14 if "blackhole" in ttnn.get_arch_name() else 7) * 1024 + 64
    return cfg


FABRIC = [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY, "fabric_router_config": _router_config()}]


def _case(mesh, shape, seed, gamma):
    x = random_stacked(mesh, shape, torch.bfloat16, seed=seed)
    g = random_stacked(mesh, (1, 1, 1, shape[-1]), torch.bfloat16, seed=seed + 100) if gamma else None
    tx = create_ttnn_input_tensor(x, mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tg = create_ttnn_gamma(g, mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) if gamma else None
    return tx, tg, pytorch_distributed_rms_norm(x, g, AXIS, 1e-6)


def _check(out, expected, shape):
    check_output(
        out, expected, shape=shape, dtype=ttnn.bfloat16, fp32_dest_acc_en=True, expected_layout=ttnn.TILE_LAYOUT
    )


@pytest.mark.parametrize("device_params", FABRIC, indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
@pytest.mark.parametrize("pipe", [True, False], ids=["lookahead", "back_to_back"])
@pytest.mark.parametrize("gamma", [False, True], ids=["no_gamma", "gamma"])
@pytest.mark.parametrize("shape", [(1, 1, 4096, 1024), (1, 1, 640, 1792)], ids=["r1_many_roots", "r3_perf_focus"])
def test_waves_alternating_inputs(mesh_device, monkeypatch, shape, gamma, pipe):
    """Refinement 3: also the WAVE_PIPELINE look-ahead (block w+1's partials / own stats published while block w is
    still on the Fabric: two gather / stats regions, per-parity gather semaphores, per-direction landed semaphores)."""
    monkeypatch.setattr(pd, "BLOCK_ROWS_CAP", 1)  # num_waves >= 2 on both regimes
    monkeypatch.setattr(pd, "WAVE_PIPELINE", pipe)
    cases = [_case(mesh_device, shape, seed, gamma) for seed in (11, 12)]
    plan = pd.make_plan(mesh_device, cases[0][0], cases[0][1], AXIS, ttnn.Topology.Linear, 1)
    assert plan.num_waves >= 2, f"test needs waves, got {plan.num_waves}"
    assert plan.pipe == pipe, f"look-ahead {pipe} expected, plan has {plan.pipe}"
    for k in range(4):
        tx, tg, expected = cases[k % 2]
        out = distributed_rms_norm(tx, cluster_axis=AXIS, gamma=tg, num_links=1)
        _check(out, expected, shape)


@pytest.mark.parametrize("device_params", FABRIC, indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
@pytest.mark.parametrize(
    "shape, gamma",
    [((1, 1, 4096, 4096), False), ((1, 1, 4096, 2560), False), ((1, 1, 4096, 2560), True)],
    ids=["4096x4096-no_gamma", "4096x2560-no_gamma", "4096x2560-gamma"],
)
def test_chunked_multi_row_store_order(mesh_device, monkeypatch, shape, gamma):
    monkeypatch.setattr(pd, "BLOCK_ROWS_CAP", 2)  # R_b = 2 rows per block wherever a core owns >= 2 rows
    tx, tg, expected = _case(mesh_device, shape, 21, gamma)
    plan = pd.make_plan(mesh_device, tx, tg, AXIS, ttnn.Topology.Linear, 1)
    assert (
        plan.regime == "R1" and plan.nch >= 2 and plan.R >= 2
    ), f"test needs R1 with >= 2 W-chunks and >= 2 rows per block, got {plan.regime} nch={plan.nch} R={plan.R}"
    out = distributed_rms_norm(tx, cluster_axis=AXIS, gamma=tg, num_links=1)
    _check(out, expected, shape)


@pytest.mark.parametrize("device_params", FABRIC, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("rect_groups", [True, False], ids=["rect_groups", "run_groups"])
@pytest.mark.parametrize("direct", [True, False], ids=["direct", "relay"])
@pytest.mark.parametrize("pipe", [True, False], ids=["lookahead", "back_to_back"])
@pytest.mark.parametrize("shape", [(1, 1, 4096, 1024), (1, 1, 640, 1792)], ids=["r1_many_roots", "r3_perf_focus"])
def test_ring_transport_alternating_inputs(mesh_device, monkeypatch, shape, pipe, direct, rect_groups):
    """Refinement 5: the Ring exchange transport (G = 4, 2 links) — direct multi-hop sends (DIRECT_RING_SENDS: per-source
    arrival counters, entries landing out of order, a ready from every destination chip) or the relay, per-entry
    notify, and (Perf 1) the root's slot forward to every member over single-rectangle (RECT_GROUPS) or multi-rectangle
    row-groups — with waves, alternating inputs (a stale cross-call or cross-wave slot shows up as a wrong scale), and
    bit-identical outputs for a repeated input."""
    monkeypatch.setattr(pd, "BLOCK_ROWS_CAP", 1)  # num_waves >= 2 on both regimes
    monkeypatch.setattr(pd, "WAVE_PIPELINE", pipe)
    monkeypatch.setattr(pd, "DIRECT_RING_SENDS", direct)
    monkeypatch.setattr(pd, "RECT_GROUPS", rect_groups)
    cases = [_case(mesh_device, shape, seed, True) for seed in (21, 22)]
    plan = pd.make_plan(mesh_device, cases[0][0], cases[0][1], AXIS, ttnn.Topology.Ring, 2)
    assert plan.ring and plan.num_waves >= 2, f"test needs a ring with waves, got ring={plan.ring} {plan.num_waves}"
    outs = []
    for k in range(4):
        tx, tg, expected = cases[k % 2]
        out = distributed_rms_norm(tx, cluster_axis=AXIS, gamma=tg, topology=ttnn.Topology.Ring, num_links=2)
        _check(out, expected, shape)
        outs.append(ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)))
    assert torch.equal(outs[0], outs[2]) and torch.equal(outs[1], outs[3]), "outputs differ run to run"
