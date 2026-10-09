# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Regression for fp32 partials cross-block reload (issue #58405).

Without UnpackToDest on the partials buffer (or its reload alias when bias is
present), the reload is routed through SrcA and TF32-rounded, losing ~13 bits
of mantissa per block boundary. These tests exercise the three factories
touched by the fix and compare against an fp64 golden with `packer_l1_acc=False`
+ `num_blocks > 1`.
"""

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("has_bias", [True, False])
def test_optimized_factory_fp32_partials_reload(device, has_bias):
    """`MatmulMultiCoreReuseProgramConfig` → optimized factory. Issue #58405 repro."""
    torch.manual_seed(0)

    m, k, n = 256, 512, 256
    tile = 32
    in0_block_w = 4  # 4 tiles = 128 K per block → num_blocks = k/32/4 = 4 > 1

    a_f64 = torch.randn(m, k, dtype=torch.float64)
    b_f64 = torch.randn(k, n, dtype=torch.float64)
    golden = a_f64 @ b_f64
    bias_f64 = torch.randn(1, n, dtype=torch.float64) if has_bias else None
    if has_bias:
        golden = golden + bias_f64

    a_t = ttnn.from_torch(
        a_f64.to(torch.float32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    b_t = ttnn.from_torch(
        b_f64.to(torch.float32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    bias_t = (
        ttnn.from_torch(bias_f64.to(torch.float32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        if has_bias
        else None
    )

    program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(1, 1),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=m // tile,
        per_core_N=n // tile,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    if has_bias:
        out_t = ttnn.linear(a_t, b_t, bias=bias_t, program_config=program_config, compute_kernel_config=compute_kernel_config)
    else:
        out_t = ttnn.matmul(a_t, b_t, program_config=program_config, compute_kernel_config=compute_kernel_config)

    out = ttnn.to_torch(out_t).to(torch.float64)
    assert_with_pcc(golden, out, pcc=0.9999)
