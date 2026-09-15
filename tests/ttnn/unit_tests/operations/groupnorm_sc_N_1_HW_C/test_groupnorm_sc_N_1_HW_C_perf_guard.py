# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf guard set for groupnorm_sc_N_1_HW_C (op_requirements.md -> Perf measurement).

One op dispatch per test so a `scripts/run_safe_pytest.sh --profile` run yields one Tracy CSV row per
case, in this file's execution order. Correctness is checked loosely (PCC) — the acceptance suite is the
correctness gate; this file exists to give every perf refinement the same shapes to measure.

Guard set: {TILE, RM} x {resident_2d, streaming_2d, single_core_per_image} x {no_affine, gamma_beta}, bf16,
on (1,1,1024,640) G=32, plus the (1,1,32,32) G=1 latency floor, the flagship (1,1,16384,320) G=32, the
other large SD/SDXL shapes, and the small shapes the `MIN_TILES_PER_CORE` lamp acts on.
"""

import pytest
import torch
import ttnn

import ttnn.operations.groupnorm_sc_N_1_HW_C as mod
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C


def _reference(x, num_groups, gamma=None, beta=None, eps=1e-5):
    xf = x.to(torch.float32)
    N, _, HW, C = xf.shape
    x_nchw = xf.squeeze(1).permute(0, 2, 1)
    w = gamma.to(torch.float32).reshape(C) if gamma is not None else None
    b = beta.to(torch.float32).reshape(C) if beta is not None else None
    out = torch.nn.functional.group_norm(x_nchw, num_groups, weight=w, bias=b, eps=eps)
    return out.permute(0, 2, 1).unsqueeze(1)


def _pcc(a, b):
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _run(device, shape, num_groups, layout, affine, regime):
    torch.manual_seed(0)
    C = shape[-1]
    x = torch.randn(shape, dtype=torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    g = b = tg = tb = None
    if affine == "gamma_beta":
        g = torch.randn(1, 1, 1, C, dtype=torch.bfloat16)
        b = torch.randn(1, 1, 1, C, dtype=torch.bfloat16)
        tg = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        tb = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    expected = _reference(x, num_groups, g, b)
    if regime == "streaming":
        mod.set_l1_budget_bytes_override(0)
    elif regime == "single_core":
        mod.set_max_cores_override(4)
    try:
        out = groupnorm_sc_N_1_HW_C(tx, num_groups, gamma=tg, beta=tb)
    finally:
        mod.set_l1_budget_bytes_override(None)
        mod.set_max_cores_override(None)
    got = ttnn.to_torch(out)
    assert _pcc(got, expected) > 0.99


GUARD_SHAPE = ((1, 1, 1024, 640), 32)
SINGLE_CORE_SHAPE = ((8, 1, 64, 160), 8)  # N = 8 over a 4-core cap -> single_core_per_image


@pytest.mark.parametrize("regime", ["resident", "streaming", "single_core"])
@pytest.mark.parametrize("affine", ["no_affine", "gamma_beta"])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
def test_guard_set(device, layout, affine, regime):
    shape, g = SINGLE_CORE_SHAPE if regime == "single_core" else GUARD_SHAPE
    _run(device, shape, g, layout, affine, regime)


@pytest.mark.parametrize(
    "shape,num_groups",
    [
        pytest.param((1, 1, 32, 32), 1, id="floor_32x32"),
        pytest.param((1, 1, 64, 64), 2, id="small_64x64"),
        pytest.param((1, 1, 128, 128), 4, id="small_128x128"),
        pytest.param((1, 1, 64, 320), 32, id="small_64x320"),
        pytest.param((1, 1, 256, 1280), 32, id="sd_256x1280"),
        pytest.param((1, 1, 4096, 320), 32, id="sd_4096x320"),
        pytest.param((1, 1, 4096, 640), 32, id="sdxl_4096x640"),
        pytest.param((1, 1, 1024, 1920), 32, id="sd_1024x1920"),
        pytest.param((1, 1, 16384, 320), 32, id="sdxl_16384x320"),
    ],
)
def test_shapes_tile_gamma_beta(device, shape, num_groups):
    _run(device, shape, num_groups, ttnn.TILE_LAYOUT, "gamma_beta", "resident")
