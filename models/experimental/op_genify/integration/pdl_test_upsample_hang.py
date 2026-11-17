# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolate the genify_matmul hang seen in PDL's test_tt_upsample.py with TT_GENIFY_MATMUL=1 (#65150).

test_bilinear_upsample_ttnn_matmul_vs_ttnn_upsample[DataType.BFLOAT16-1-128-256-32-4-False-False-
memory_config11-110_cores] hung (dispatch timeout) on its genify passes. That case runs the PDL
upsample shapes in bf16 with the module's bf16 compute config (HiFi2, approx, fp32_dest_acc_en=False)
and every tensor in DRAM. Each case here is one genify_matmul call with exactly those operands.

    scripts/run_safe_pytest.sh --dev models/experimental/op_genify/integration/pdl_test_upsample_hang.py -s -k <case>
"""

import pytest
import torch
import ttnn

from models.experimental.op_genify import genify_matmul_adapter as genify

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
# BilinearUpsampleMatmulTTNN's default config for a bf16 output (tt_upsample.py).
BF16_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=False
)
BFP8_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=False
)

CASES = {
    # name: (A shape, B shape, dtype, compute, A mem, B mem, out mem)
    "w_bf16_dram": ((1, 32, 128, 256), (1, 1, 256, 1024), ttnn.bfloat16, BF16_CFG, DRAM, DRAM, DRAM),
    "h_bf16_dram": ((1, 32, 512, 128), (1, 32, 128, 1024), ttnn.bfloat16, BF16_CFG, DRAM, DRAM, DRAM),
    "w_bfp8_dram": ((1, 32, 128, 256), (1, 1, 256, 1024), ttnn.bfloat8_b, BFP8_CFG, DRAM, DRAM, DRAM),
    "h_bfp8_dram": ((1, 32, 512, 128), (1, 32, 128, 1024), ttnn.bfloat8_b, BFP8_CFG, DRAM, DRAM, DRAM),
    "w_bf16_l1": ((1, 32, 128, 256), (1, 1, 256, 1024), ttnn.bfloat16, BF16_CFG, L1, DRAM, L1),
    "h_bf16_l1b": ((1, 32, 512, 128), (1, 32, 128, 1024), ttnn.bfloat16, BF16_CFG, DRAM, L1, DRAM),
}


@pytest.mark.parametrize("case", list(CASES))
def test_upsample_pass(device, case):
    a_shape, b_shape, dtype, cfg, a_mem, b_mem, out_mem = CASES[case]
    torch.manual_seed(0)
    a_t, b_t = torch.rand(a_shape), torch.rand(b_shape)
    a = ttnn.from_torch(a_t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=a_mem)
    b = ttnn.from_torch(b_t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=b_mem)
    out = genify.matmul(a, b, dtype=dtype, memory_config=out_mem, compute_kernel_config=cfg)
    ttnn.synchronize_device(device)
    got = ttnn.to_torch(out).float()
    ref = ttnn.to_torch(a).float() @ ttnn.to_torch(b).float()
    pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
    print(f"UPSAMPLE_CASE {case}: pcc={pcc:.6f}")
    assert pcc > 0.99, pcc  # bf16 DEST accumulation of 128-256 positive products: ~0.996 is expected
