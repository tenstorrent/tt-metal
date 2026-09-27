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
@pytest.mark.parametrize("l1acc", [False, True])
def test_dest_gain(device, K, fp32, l1acc):
    torch.manual_seed(0)
    x = torch.randn(128, K)
    w = torch.randn(K, 2048) * 0.02
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    wt = ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device)
    xq, wq = ttnn.to_torch(xt).float(), ttnn.to_torch(wt).float()
    cfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=fp32, packer_l1_acc=l1acc
    )
    # K blocks of 8 tiles (partials through L1 when packer_l1_acc, else DEST reloads), 4 x 8 cores
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 4),
        in0_block_w=8,
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=1,
        per_core_N=8,
        transpose_mcast=False,
        fused_activation=None,
    )
    y = ttnn.matmul(xt, wt, compute_kernel_config=cfg, dtype=ttnn.bfloat16, program_config=pc)
    y = ttnn.to_torch(y).float()
    ref = xq @ wq
    logger.info(f"K {K} fp32_dest {fp32} packer_l1_acc {l1acc}: norm ratio {float(y.norm() / ref.norm()):.4f}")
