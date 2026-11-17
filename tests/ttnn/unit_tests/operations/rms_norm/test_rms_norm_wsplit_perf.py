# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf probe for W-split: run under `scripts/run_safe_pytest.sh --profile`; RMS_WSPLIT=0 disables the split."""
import os

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm, rms_norm_program_descriptor as pd

if os.environ.get("RMS_WSPLIT", "1") == "0":
    pd.W_SPLIT_MIN_GAIN = 0.0


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 32, 5120), (1, 1, 32, 1024), (1, 1, 32, 7168), (1, 1, 1, 16384), (1, 1, 64, 12288), (1, 1, 32, 32768)],
)
def test_perf(device, shape):
    x = torch.randn(shape).bfloat16()
    g = torch.randn(1, 1, 1, shape[-1]).bfloat16()
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tg = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False)
    rms_norm(tx, gamma=tg, compute_kernel_config=cfg)
    ttnn.synchronize_device(device)
