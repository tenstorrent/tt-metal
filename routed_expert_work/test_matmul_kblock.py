# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Does the on-device error of a LoFi bf16-DEST matmul depend on how many K tiles accumulate in DEST
before a bf16 L1 spill? Emulates the fused op's down matmul: h[256,2048] bfp8 @ Wd[2048,96] bfp4,
single core, in0_block_w = K tiles per DEST accumulation (64 = whole K, like the mrow path)."""
import os
import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import comp_pcc


@pytest.mark.parametrize("in0_block_w", [64, 32, 8, 4, 2, 1])
@pytest.mark.parametrize("fp32_acc", [False, True])
def test_kblock(device, in0_block_w, fp32_acc):
    torch.manual_seed(42)
    M, K, N = 256, 2048, 96
    h = torch.randn(M, K)
    w = torch.randn(K, N) * 0.02
    ref = h @ w
    a = ttnn.from_torch(h, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device)
    href = ttnn.to_torch(a).float() @ ttnn.to_torch(b).float()  # quantized-operand exact reference
    cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(1, 1),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=3,
        per_core_M=M // 32,
        per_core_N=N // 32,
        transpose_mcast=False,
        fused_activation=None,
    )
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=fp32_acc, packer_l1_acc=True
    )
    out = ttnn.matmul(a, b, program_config=cfg, compute_kernel_config=ck, dtype=ttnn.bfloat16)
    o = ttnn.to_torch(out).float()
    _, pcc = comp_pcc(ref, o)
    rr = ((o - ref).norm() / ref.norm()).item()
    rrq = ((o - href).norm() / href.norm()).item()
    print(
        f"KBLOCK in0_block_w={in0_block_w} fp32_acc={fp32_acc} pcc={pcc:.6f} rel_rms_vs_fp32={rr:.5f} rel_rms_vs_quantized_exact={rrq:.5f}",
        flush=True,
    )
