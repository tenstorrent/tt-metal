# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for all_to_all_matmul (the spec; the implementer does not modify it).

A multi-device CCL op: every test runs on the golden suite's focus group, a 4-device ring along cluster_axis 1
opened with FABRIC_2D_TORUS_XY (a Galaxy keeps its 8x4 mesh; a QuietBox 2 opens its snake ring as 1x4). The
`mesh_device` fixture and its session cache come from the golden suite's conftest, imported below (the
directory conftest's use_module_device marker only affects the single-device `device` fixture, which this op
does not use). A cluster that cannot close the ring skips.

Phase 0 configuration: bfloat16 activation and weight, fp32 DEST, HiFi4, output in DRAM, Ring, 2 links.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

# Golden-suite fixtures (mesh opened once per session with the fabric; never opened manually here).
from eval.golden_tests.all_to_all_matmul.conftest import _fabric_mesh_cache, mesh_device  # noqa: F401
from eval.golden_tests.all_to_all_matmul.helpers import (
    _tolerance_for,
    check_output,
    create_ttnn_input_tensor,
    output_shape,
    pytorch_all_to_all_matmul,
    read_back,
    replicate,
)
from ttnn.operations.all_to_all_matmul import all_to_all_matmul

CLUSTER_AXIS = 1
G = 4


def _phase0_config():
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    return cfg


def _inputs(mesh_device, x_shape, w_shape, seed):
    """Distinct randn input per device (stacked (R, C, *x_shape)) and one replicated weight scaled 1/sqrt(K)."""
    torch.manual_seed(seed)
    rows, cols = tuple(mesh_device.shape)
    x = torch.randn((rows, cols, *x_shape), dtype=torch.float32).to(torch.bfloat16)
    w = (torch.randn(w_shape, dtype=torch.float32) / w_shape[-2] ** 0.5).to(torch.bfloat16)
    return x, w


def _run_and_check(mesh_device, x_shape, w_shape, seed=42):
    assert tuple(mesh_device.shape)[CLUSTER_AXIS] == G, f"focus ring has G={G}, mesh {tuple(mesh_device.shape)}"
    x, w = _inputs(mesh_device, x_shape, w_shape, seed)
    tx = create_ttnn_input_tensor(x, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tw = create_ttnn_input_tensor(replicate(w, mesh_device), mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    out = all_to_all_matmul(
        tx,
        tw,
        cluster_axis=CLUSTER_AXIS,
        topology=ttnn.Topology.Ring,
        num_links=2,
        compute_kernel_config=_phase0_config(),
    )

    expected = pytorch_all_to_all_matmul(x, w, CLUSTER_AXIS)
    check_output(
        out,
        expected,
        shape=output_shape(x_shape, w_shape, G),
        dtype=ttnn.bfloat16,
        expected_layout=ttnn.TILE_LAYOUT,
        group=G,
        tolerance=_tolerance_for(ttnn.bfloat16, True),
        fp32_dest_acc_en=True,
    )
    return out


# ((B, Hl, S, K), (1, H, K, N)) per device, H = G * Hl; every case tile-aligned (S and S/G multiples of 32).
SHAPES = [
    pytest.param((1, 4, 128, 32), (1, 16, 32, 32), id="single_tile_per_head"),  # one tile per device per head
    pytest.param((1, 8, 256, 128), (1, 32, 128, 256), id="multi_tile"),
    pytest.param((1, 4, 128, 64), (1, 16, 64, 128), id="non_square"),  # K != N, rows != K
    pytest.param((2, 8, 128, 128), (1, 32, 128, 512), id="multi_batch"),  # weight shared across the batch
    pytest.param((1, 4, 128, 128), (1, 16, 128, 4096), id="multi_n_block"),  # N split into several column blocks
    pytest.param((1, 24, 640, 128), (1, 96, 128, 512), id="kimi_k31"),  # the perf-focus call
    pytest.param((1, 32, 640, 128), (1, 128, 128, 512), id="deepseek_v32"),  # > 1 head per core
    pytest.param((1, 16, 640, 192), (1, 64, 192, 512), id="glm_53"),  # K = 192
]


@pytest.mark.parametrize("x_shape,w_shape", SHAPES)
def test_all_to_all_matmul(mesh_device, x_shape, w_shape):
    _run_and_check(mesh_device, x_shape, w_shape)


def test_all_to_all_matmul_back_to_back(mesh_device):
    """Two calls with different inputs on the cached program: no stale rows, arrival counts or fence state."""
    x_shape, w_shape = (1, 24, 640, 128), (1, 96, 128, 512)
    _run_and_check(mesh_device, x_shape, w_shape, seed=42)
    _run_and_check(mesh_device, x_shape, w_shape, seed=43)


def test_all_to_all_matmul_exchange_is_bit_exact(mesh_device):
    """Per-head identity weight: the output must equal the exchanged rows bit for bit (the rows move as stored)."""
    B, Hl, S, K = 1, 8, 256, 64
    H = G * Hl
    rows, cols = tuple(mesh_device.shape)
    torch.manual_seed(42)
    x = torch.randn((rows, cols, B, Hl, S, K), dtype=torch.float32).to(torch.bfloat16)
    w = torch.eye(K, dtype=torch.float32).expand(1, H, K, K).contiguous().to(torch.bfloat16)
    tx = create_ttnn_input_tensor(x, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tw = create_ttnn_input_tensor(replicate(w, mesh_device), mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    out = all_to_all_matmul(
        tx,
        tw,
        cluster_axis=CLUSTER_AXIS,
        topology=ttnn.Topology.Ring,
        num_links=2,
        compute_kernel_config=_phase0_config(),
    )
    got = read_back(out, (rows, cols))
    want = pytorch_all_to_all_matmul(x, w, CLUSTER_AXIS).to(torch.bfloat16)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16)), "exchanged rows were not moved bit-exactly"
