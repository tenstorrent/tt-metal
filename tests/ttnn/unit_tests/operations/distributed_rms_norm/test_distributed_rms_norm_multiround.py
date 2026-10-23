# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1: multi-block per-core schedule (several exchange rounds per core).

* tall x wide shapes whose one-block-per-core working set exceeds L1 (the solve picks num_rounds > 1 by itself);
* small shapes with the MIN_ROUNDS knob forced up, so every round-dependent path (ragged last round, empty last
  round on some row-slots, ragged W-chunk, gamma / no-gamma round orders, back-to-back counter re-arm) runs cheaply.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from ttnn.operations.distributed_rms_norm import distributed_rms_norm_program_descriptor as pd

import importlib.util
from pathlib import Path

# reuse the acceptance test's reference + checker (this directory is not a package; pytest uses importlib mode)
_spec = importlib.util.spec_from_file_location(
    "_drn_acceptance", Path(__file__).with_name("test_distributed_rms_norm.py")
)
_acc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_acc)
LINEAR_AXES, run_case = _acc.LINEAR_AXES, _acc.run_case

pytestmark = pytest.mark.skipif(not LINEAR_AXES, reason="no mesh axis of size >= 2")


@pytest.mark.parametrize(
    "shape,dtype,use_gamma",
    [
        ((2, 1, 2048, 4096), ttnn.bfloat16, True),  # the 8 translated golden cells' per-device shape
        ((1, 1, 8192, 4096), ttnn.bfloat16, True),
        ((1, 1, 4096, 4096), ttnn.float32, True),
        ((1, 1, 8192, 4096), ttnn.bfloat16, False),
    ],
    ids=["2x2048x4096_bf16_g", "8192x4096_bf16_g", "4096x4096_fp32_g", "8192x4096_bf16_nog"],
)
def test_tall_wide_fits(shape, dtype, use_gamma, mesh_device):
    run_case(mesh_device, shape, dtype, LINEAR_AXES[0], use_gamma=use_gamma)


@pytest.mark.parametrize("min_rounds", [2, 3])
@pytest.mark.parametrize("use_gamma", [True, False], ids=["gamma", "no_gamma"])
@pytest.mark.parametrize(
    "shape", [(1, 1, 640, 1792), (1, 1, 200, 1312), (2, 4, 64, 256)], ids=lambda s: "x".join(map(str, s))
)
def test_forced_rounds(shape, use_gamma, min_rounds, mesh_device, monkeypatch):
    monkeypatch.setattr(pd, "MIN_ROUNDS", min_rounds)
    run_case(mesh_device, shape, ttnn.bfloat16, LINEAR_AXES[0], use_gamma=use_gamma)


def test_forced_rounds_back_to_back(mesh_device, monkeypatch):
    """Counters are cumulative over rounds and re-armed per call: three calls must agree bit for bit."""
    monkeypatch.setattr(pd, "MIN_ROUNDS", 3)
    outs = [
        ttnn.to_torch(
            ttnn.get_device_tensors(
                run_case(mesh_device, (1, 1, 640, 1792), ttnn.bfloat16, LINEAR_AXES[0], use_gamma=True)
            )[0]
        )
        for _ in range(3)
    ]
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[1], outs[2])


_RING_CASES = [pytest.param(None, a, id=f"system-axis{a}") for a in _acc.RING_AXES]
_RING_CASES += [pytest.param(s, a, id=f"mock{s[0]}x{s[1]}-axis{a}") for s, a in _acc.ring_mock_meshes()]


@pytest.mark.skipif(not _RING_CASES, reason="no ring this cluster can close")
@pytest.mark.parametrize("mesh_shape,cluster_axis", _RING_CASES)
@pytest.mark.parametrize("use_gamma", [True, False], ids=["gamma", "no_gamma"])
def test_forced_rounds_ring(use_gamma, mesh_shape, cluster_axis, mesh_device, monkeypatch):
    """Ring relays: arrivals are one cumulative in-order counter, relay k of round r waits expect_in * r + k."""
    monkeypatch.setattr(pd, "MIN_ROUNDS", 3)
    run_case(
        mesh_device, (1, 1, 640, 1792), ttnn.bfloat16, cluster_axis, use_gamma=use_gamma, topology=ttnn.Topology.Ring
    )
