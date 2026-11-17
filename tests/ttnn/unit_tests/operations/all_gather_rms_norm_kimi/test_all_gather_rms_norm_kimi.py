# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for all_gather_rms_norm_kimi (immutable spec — do not modify).

A multi-device CCL op: every test runs on the 4-device ring along cluster_axis 1 that the golden suite opens
(FABRIC_2D_TORUS_XY, production router payload; a QuietBox 2 opens its snake ring as 1x4, a Galaxy rings along its
TP axis). The single-device `device` fixture does not apply; the mesh comes from the golden suite's fabric mesh
cache (eval/golden_tests/all_gather_rms_norm_kimi/conftest.py), opened once for this module.

Accuracy limits are the golden suite's own: helpers.TOLERANCES[(bfloat16, fp32_dest_acc_en=True)], applied per
device AND per group-position slice by helpers.check_output.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from eval.golden_tests.all_gather_rms_norm_kimi.conftest import _FabricMeshCache, focus_mesh_or_skip
from eval.golden_tests.all_gather_rms_norm_kimi.helpers import (
    TOLERANCES,
    check_output,
    create_ttnn_gamma,
    create_ttnn_input_tensor,
    gathered_shape,
    pytorch_all_gather_rms_norm_kimi,
)
from ttnn.operations.all_gather_rms_norm_kimi import all_gather_rms_norm_kimi

CLUSTER_AXIS = 1
EPS = 1e-6


@pytest.fixture(scope="module")
def mesh_device():
    cache = _FabricMeshCache()
    try:
        yield cache.get(focus_mesh_or_skip())
    finally:
        cache.close()


def _compute_config():
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    return cfg


def _group(mesh_device):
    return tuple(mesh_device.shape)[CLUSTER_AXIS]


def _inputs(mesh_device, shape, seed=42):
    """Distinct randn slice per device (stacked (R, C, *shape)) and a full-W gamma (W,)."""
    rows, cols = tuple(mesh_device.shape)
    torch.manual_seed(seed)
    x = torch.randn((rows, cols, *shape), dtype=torch.float32).to(torch.bfloat16)
    gamma = torch.randn((_group(mesh_device) * shape[-1],), dtype=torch.float32).to(torch.bfloat16)
    return x, gamma


def _run(mesh_device, x, gamma, compute_kernel_config="default"):
    tt_x = create_ttnn_input_tensor(x, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_gamma = create_ttnn_gamma(gamma, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    cfg = _compute_config() if compute_kernel_config == "default" else compute_kernel_config
    return all_gather_rms_norm_kimi(
        tt_x,
        gamma=tt_gamma,
        cluster_axis=CLUSTER_AXIS,
        epsilon=EPS,
        topology=ttnn.Topology.Ring,
        num_links=2,
        compute_kernel_config=cfg,
    )


def _check(mesh_device, out, x, gamma, shape):
    expected = pytorch_all_gather_rms_norm_kimi(x, gamma, CLUSTER_AXIS, EPS)
    pcc, rms = TOLERANCES[(ttnn.bfloat16, True)]
    check_output(
        out,
        expected,
        shape=gathered_shape(shape, _group(mesh_device)),
        dtype=ttnn.bfloat16,
        fp32_dest_acc_en=True,
        expected_layout=ttnn.TILE_LAYOUT,
        group=_group(mesh_device),
        pcc_target=pcc,
        rms_target=rms,
    )


SHAPES = [
    pytest.param((1, 1, 32, 32), id="single_tile"),
    pytest.param((1, 1, 64, 128), id="multi_tile"),
    pytest.param((1, 1, 128, 256), id="non_square"),
    pytest.param((1, 1, 640, 384), id="kimi_mla_q_a"),  # the production / perf-focus call
    # Regime pin: G * Mt = 256 compute blocks exceed the free grid, so block_h > 1 with a ragged last row block.
    pytest.param((1, 1, 2048, 384), id="tall_block_h_gt_1"),
]


@pytest.mark.parametrize("shape", SHAPES)
def test_all_gather_rms_norm_kimi(mesh_device, shape):
    x, gamma = _inputs(mesh_device, shape)
    out = _run(mesh_device, x, gamma)
    assert out.layout == ttnn.TILE_LAYOUT
    assert out.memory_config().buffer_type == ttnn.BufferType.DRAM
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    _check(mesh_device, out, x, gamma, shape)


def test_default_compute_kernel_config(mesh_device):
    """compute_kernel_config=None resolves through default_compute_kernel_config() (fp32 DEST, HiFi4)."""
    shape = (1, 1, 640, 384)
    x, gamma = _inputs(mesh_device, shape, seed=43)
    out = _run(mesh_device, x, gamma, compute_kernel_config=None)
    _check(mesh_device, out, x, gamma, shape)


def test_repeated_calls_deterministic_and_replicated(mesh_device):
    """Back-to-back calls without host sync between them (fence + counter re-arm) are bit-identical run to run,
    and every member of a group returns the same full output (fixed-order fp32 partial sum)."""
    shape = (1, 1, 640, 384)
    x, gamma = _inputs(mesh_device, shape, seed=44)
    outs = [_run(mesh_device, x, gamma) for _ in range(4)]
    _check(mesh_device, outs[0], x, gamma, shape)
    cols = tuple(mesh_device.shape)[1]
    per_call = [[ttnn.to_torch(t) for t in ttnn.get_device_tensors(o)] for o in outs]
    for call in per_call[1:]:
        for idx, (a, b) in enumerate(zip(per_call[0], call)):
            assert torch.equal(a, b), f"device {divmod(idx, cols)}: output differs between identical calls"
    first = per_call[0]
    for idx, t in enumerate(first):
        group_start = (idx // cols) * cols  # cluster_axis 1: a group is a mesh row
        assert torch.equal(t, first[group_start]), f"device {divmod(idx, cols)}: differs from its group's member 0"


def test_gamma_wrong_format_raises(mesh_device, expect_error):
    """A single-row (1, 1, 1, W) ROW_MAJOR gamma is not the stick format (1, 1, ceil(W/32), 32): caller error whose
    message names gamma."""
    shape = (1, 1, 64, 128)
    x, gamma = _inputs(mesh_device, shape)
    tt_x = create_ttnn_input_tensor(x, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    bad_gamma = ttnn.from_torch(
        gamma.reshape(1, 1, 1, -1),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    with expect_error((ValueError, RuntimeError), "(?i)gamma"):
        all_gather_rms_norm_kimi(
            tt_x,
            gamma=bad_gamma,
            cluster_axis=CLUSTER_AXIS,
            epsilon=EPS,
            topology=ttnn.Topology.Ring,
            num_links=2,
            compute_kernel_config=_compute_config(),
        )
