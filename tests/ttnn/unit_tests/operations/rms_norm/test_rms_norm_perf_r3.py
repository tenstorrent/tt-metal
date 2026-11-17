# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 3 perf probe: run under `scripts/run_safe_pytest.sh --profile`; RMS_GCAP / RMS_ODEPTH override knobs."""
import os

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm, rms_norm_program_descriptor as pd

if "RMS_GCAP" in os.environ:
    pd.G_CAP = int(os.environ["RMS_GCAP"])
if "RMS_ODEPTH" in os.environ:
    pd.OUT_DEPTH = int(os.environ["RMS_ODEPTH"])


@pytest.mark.parametrize("shape", [(8192, 1024), (8192, 2304), (8192, 5120), (8192, 7168), (32, 2304), (32, 5120)])
def test_perf(device, shape):
    x = torch.randn(shape).bfloat16()
    g = torch.randn(1, shape[-1]).bfloat16()
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tg = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False)
    out = rms_norm(tx, gamma=tg, compute_kernel_config=cfg)
    ttnn.synchronize_device(device)
    ref = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * g.float()
    r = ttnn.to_torch(out).float()
    assert torch.corrcoef(torch.stack([r.flatten(), ref.flatten()]))[0, 1] > 0.9995
