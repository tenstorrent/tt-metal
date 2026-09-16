# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Latency-floor perf set for groupnorm_sc_N_1_HW_C (Refinement 5 — cheaper per-image combine).

One op dispatch per test so a `scripts/run_safe_pytest.sh --profile` run of this file yields one Tracy CSV row
per case, in this file's execution order (`probes/compare_perf_floor_csv.py` pairs two such CSVs). Correctness
is checked loosely (PCC); the acceptance suite is the correctness gate.

Cases: the Done-when shapes of Refinement 5 — `(1,1,32,32) G=1` (1 core, local combine), `(1,1,64,320) G=32`
(20-core rectangle), the multi-image `(8,1,64,64) G=2` (8 rectangles of 4 cores) — plus the other <= 64-tile
floor shapes, the `single_core_per_image` pin (N = 8 images on a 4-core cap: images looped per core), and two
DRAM-bound sentinels to catch a regression on the large shapes.
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


def _run(device, shape, num_groups, layout=ttnn.TILE_LAYOUT, affine="gamma_beta", max_cores=None):
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
    if max_cores is not None:
        mod.set_max_cores_override(max_cores)
    try:
        out = groupnorm_sc_N_1_HW_C(tx, num_groups, gamma=tg, beta=tb)
    finally:
        mod.set_max_cores_override(None)
    got = ttnn.to_torch(out)
    assert _pcc(got, expected) > 0.99


FLOOR_CASES = [
    pytest.param((1, 1, 32, 32), 1, None, id="floor_32x32_g1"),
    pytest.param((1, 1, 64, 64), 2, None, id="small_64x64_g2"),
    pytest.param((1, 1, 128, 128), 4, None, id="small_128x128_g4"),
    pytest.param((1, 1, 64, 320), 32, None, id="small_64x320_g32"),
    pytest.param((8, 1, 64, 64), 2, None, id="multi_8x64x64_g2"),
    pytest.param((8, 1, 64, 160), 8, 4, id="single_core_8x64x160_g8_cap4"),
    pytest.param((1, 1, 1024, 640), 32, None, id="sentinel_1024x640"),
    pytest.param((1, 1, 16384, 320), 32, None, id="sentinel_16384x320"),
]


@pytest.mark.parametrize("shape,num_groups,max_cores", FLOOR_CASES)
def test_floor_tile_gamma_beta(device, shape, num_groups, max_cores):
    _run(device, shape, num_groups, max_cores=max_cores)


@pytest.mark.parametrize(
    "shape,num_groups",
    [
        pytest.param((1, 1, 32, 32), 1, id="floor_32x32_g1_no_affine"),
        pytest.param((1, 1, 64, 320), 32, id="small_64x320_g32_no_affine"),
    ],
)
def test_floor_tile_no_affine(device, shape, num_groups):
    _run(device, shape, num_groups, affine="no_affine")
