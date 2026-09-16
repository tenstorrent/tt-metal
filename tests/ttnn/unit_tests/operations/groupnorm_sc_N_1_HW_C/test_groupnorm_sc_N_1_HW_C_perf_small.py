# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf 1 small-shape guard (fast): the four Ct_core = 1 cells of the guard set + the focus shape, one op per test,
for `--profile` A/B runs (3 runs, median — these ~5-7 us shapes swing +-6 % between single runs). Same harness as
test_groupnorm_sc_N_1_HW_C_perf_tournament.py (the tests tree has no __init__.py, hence the copy)."""

import os

import pytest
import torch
import ttnn

from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C
from ttnn.operations.groupnorm_sc_N_1_HW_C.groupnorm_sc_N_1_HW_C_program_descriptor import KERNEL_DEFINES_ENV


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


def _run(device, shape, num_groups, affine="gamma_beta"):
    torch.manual_seed(0)
    C = shape[-1]
    x = torch.randn(shape, dtype=torch.bfloat16)
    tx = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    g = b = tg = tb = None
    if affine == "gamma_beta":
        g = torch.randn(1, 1, 1, C, dtype=torch.bfloat16)
        b = torch.randn(1, 1, 1, C, dtype=torch.bfloat16)
        tg = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        tb = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out = groupnorm_sc_N_1_HW_C(tx, num_groups, gamma=tg, beta=tb)
    got = ttnn.to_torch(out)
    if os.environ.get(KERNEL_DEFINES_ENV, "").strip():
        return  # ablated kernel: perf-only, output wrong by design
    assert _pcc(got, _reference(x, num_groups, g, b)) > 0.99


CASES = [
    pytest.param((1, 1, 32, 32), 1, id="floor_32x32"),
    pytest.param((1, 1, 64, 64), 2, id="small_64x64"),
    pytest.param((1, 1, 128, 128), 4, id="small_128x128"),
    pytest.param((1, 1, 64, 320), 32, id="small_64x320"),
    pytest.param((1, 1, 1024, 640), 32, id="focus_1024x640"),
]


@pytest.mark.parametrize("shape,num_groups", CASES)
def test_small_tile_gamma_beta(device, shape, num_groups):
    _run(device, shape, num_groups)
