# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf-tournament bench for groupnorm_sc_N_1_HW_C (Perf 1).

One op dispatch per test so a `scripts/run_safe_pytest.sh --profile` run of this file yields one Tracy CSV row per
case, in this file's execution order, plus the per-stage zones (MaybeDeviceZoneScope) in profile_log_device.csv —
`probes/zone_report.py <report_dir> <op_index>` turns those into the per-stage breakdown.

Cases (bf16, TILE, gamma_beta unless named otherwise):
  focus_1024x640      (1,1,1024,640)  G=32  — the tournament's focus shape (SD U-Net stage, 110 cores, ROOT combine,
                                            resident; 173 GB/s of in+out traffic against a ~330 GB/s measured DRAM
                                            envelope on this grid -> headroom is in the fixed per-core costs)
  sdxl_4096x640       (1,1,4096,640)  G=32  — larger SDXL stage (more chunks per core)
  sdxl_16384x320      (1,1,16384,320) G=32  — flagship, closest to the DRAM roofline
  sd_256x1280         (1,1,256,1280)  G=32  — wide-C, few rows
  floor_32x32         (1,1,32,32)     G=1   — 1-core latency floor (LOCAL combine)
  multi_8x64x64       (8,1,64,64)     G=2   — 8 rectangles of 4 cores (ALL_GATHER combine)

Ablation runs set GROUPNORM_SC_N_1_HW_C_KERNEL_DEFINES (the op's output is then wrong by design) — the PCC gate is
skipped whenever that variable is non-empty. Correctness of the real op is the acceptance / golden suites' job.
"""

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
    pytest.param((1, 1, 1024, 640), 32, id="focus_1024x640"),
    pytest.param((1, 1, 4096, 640), 32, id="sdxl_4096x640"),
    pytest.param((1, 1, 16384, 320), 32, id="sdxl_16384x320"),
    pytest.param((1, 1, 256, 1280), 32, id="sd_256x1280"),
    pytest.param((1, 1, 32, 32), 1, id="floor_32x32"),
    pytest.param((8, 1, 64, 64), 2, id="multi_8x64x64"),
]


@pytest.mark.parametrize("shape,num_groups", CASES)
def test_tournament_tile_gamma_beta(device, shape, num_groups):
    _run(device, shape, num_groups)
