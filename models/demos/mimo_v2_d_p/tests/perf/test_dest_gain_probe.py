# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Probe: output gain (norm ratio vs the quantized-weight fp32 reference) of plain ttnn.matmul, LoFi, bfp8 x @ bfp4 W,
bf16 vs fp32 DEST accumulation, at routed-expert K (the flat expert showed ~1.11 per gate projection + down)."""
import pytest
import torch
from loguru import logger

import ttnn


@pytest.mark.parametrize("K", [2048, 7168])
@pytest.mark.parametrize("fp32", [False, True])
def test_dest_gain(device, K, fp32):
    torch.manual_seed(0)
    x = torch.randn(128, K)
    w = torch.randn(K, 2048) * 0.02
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    wt = ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device)
    xq, wq = ttnn.to_torch(xt).float(), ttnn.to_torch(wt).float()
    cfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=fp32, packer_l1_acc=False
    )
    y = ttnn.to_torch(ttnn.matmul(xt, wt, compute_kernel_config=cfg, dtype=ttnn.bfloat16)).float()
    ref = xq @ wq
    logger.info(f"K {K} fp32_dest {fp32}: norm ratio {float(y.norm() / ref.norm()):.4f}")
