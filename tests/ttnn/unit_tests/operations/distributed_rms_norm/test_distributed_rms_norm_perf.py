# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf harness for distributed_rms_norm (measurement, not a gate): N back-to-back calls on the perf-case ring
(1x4, Ring, 2 links, bf16 x and gamma). Run under `scripts/run_safe_pytest.sh --profile`; read the GenericOp rows of
the CSV and take the steady-state calls (the first call carries the chips' launch skew).

The mesh comes from the root conftest `mesh_device` fixture (fabric config via `device_params`).
"""

import os

import pytest
import torch
import ttnn

import ttnn.operations.distributed_rms_norm.distributed_rms_norm_program_descriptor as pd
from eval.golden_tests.distributed_rms_norm.helpers import (
    check_output,
    create_ttnn_gamma,
    create_ttnn_input_tensor,
    pytorch_distributed_rms_norm,
    random_stacked,
)
from ttnn.operations.distributed_rms_norm import distributed_rms_norm

N_CALLS = 6


@pytest.fixture(autouse=True)
def _knob_overrides(monkeypatch):
    """Knob sweeps without edits: DRMS_KNOBS="BLOCK_ROWS_CAP=1;DM_NOCS=(1, 0)" overrides module knobs of the
    program descriptor for this run (values are Python literals; None allowed)."""
    for item in filter(None, os.environ.get("DRMS_KNOBS", "").split(";")):
        name, value = item.split("=")
        assert hasattr(pd, name), f"unknown knob {name}"
        monkeypatch.setattr(pd, name, eval(value))


def _router_config():
    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = (14 if "blackhole" in ttnn.get_arch_name() else 7) * 1024 + 64
    return cfg


FABRIC = [
    {
        "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
        "fabric_router_config": _router_config(),
        "trace_region_size": 1 << 22,
    }
]


@pytest.mark.parametrize("device_params", FABRIC, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "shape",
    [(1, 1, 640, 1792), (1, 1, 32, 1792), (1, 1, 2048, 1280), (1, 1, 4096, 1024), (2, 4, 128, 512), (1, 1, 1024, 2048)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_perf_ring(mesh_device, shape):
    x = random_stacked(mesh_device, shape, torch.bfloat16, seed=0)
    g = random_stacked(mesh_device, (1, 1, 1, shape[-1]), torch.bfloat16, seed=1)
    tx = create_ttnn_input_tensor(x, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tg = create_ttnn_gamma(g, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    run = lambda: distributed_rms_norm(
        tx, cluster_axis=1, gamma=tg, epsilon=1e-5, topology=ttnn.Topology.Ring, num_links=2
    )
    run()  # warm-up: compile + persistent resources, outside the trace
    ttnn.synchronize_device(mesh_device)
    # Traced replay: the chips launch together (no host dispatch skew), as in a traced model.
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for _ in range(N_CALLS):
        out = run()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    ttnn.release_trace(mesh_device, tid)
    # The traced output is a real result: check it (a perf variant that breaks correctness is not a variant).
    expected = pytorch_distributed_rms_norm(x, g, 1, 1e-5)
    check_output(
        out, expected, shape=shape, dtype=ttnn.bfloat16, fp32_dest_acc_en=True, expected_layout=ttnn.TILE_LAYOUT
    )


@pytest.mark.parametrize("device_params", FABRIC, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "layout",
    [ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TensorMemoryLayout.WIDTH_SHARDED],
    ids=["HEIGHT", "WIDTH"],
)
@pytest.mark.parametrize(
    "shape",
    [(1, 1, 640, 1792), (1, 1, 32, 1792), (1, 1, 2048, 1280), (1, 1, 4096, 1024)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_perf_ring_sharded(mesh_device, shape, layout):
    """Refinement 1: the same traced ring, with the input sharded by the golden suite's chooser (auto_shard_config)
    and the output in the input's own placement (zero-copy both ways)."""
    x = random_stacked(mesh_device, shape, torch.bfloat16, seed=0)
    g = random_stacked(mesh_device, (1, 1, 1, shape[-1]), torch.bfloat16, seed=1)
    tx = create_ttnn_input_tensor(x, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_layout=layout)
    tg = create_ttnn_gamma(g, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    run = lambda: distributed_rms_norm(
        tx, cluster_axis=1, gamma=tg, epsilon=1e-5, topology=ttnn.Topology.Ring, num_links=2
    )
    # Warm-up mirrors the captured sequence (each call made while the previous L1 output is still live): the L1 plan
    # depends on the live L1 state, so the captured calls must meet the states the warm-up compiled for.
    for _ in range(N_CALLS):
        out = run()
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for _ in range(N_CALLS):
        out = run()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    ttnn.release_trace(mesh_device, tid)
    assert out is not None
