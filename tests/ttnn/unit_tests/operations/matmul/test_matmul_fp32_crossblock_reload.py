# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for fp32 accumulation precision of the matmul cross-block partial reload.

When a matmul reduces over K in more than one block with fp32_dest_acc_en=True and
packer_l1_acc=False, the intermediate K-partials (Float32) are spilled to a CB and reloaded
into DEST between blocks. Unless that CB is marked UnpackToDestFp32, the reload is routed
through SrcA and truncated to TF32 (10 mantissa bits), so fp32 accumulation silently degrades.

These tests use inputs with a large common offset and a B whose columns sum to zero: the
offset contributes nothing to the true result but makes the mid-accumulation partial sums
large, so a TF32-truncated reload catastrophically corrupts the (small) true result. With a
lossless fp32 reload the result stays accurate.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_numeric_metrics


def _offset_cancellation_inputs(m, k, n, offset, seed=0):
    """A = offset + small signal; B has zero column sums so offset*B contributes 0 to the
    true product. The true result is the small signal@B, but the partial sums during the K
    reduction reach ~offset in magnitude, which is what a lossy TF32 reload destroys."""
    torch.manual_seed(seed)
    a = (torch.randn(m, k) + offset).bfloat16()
    b = torch.randn(k, n)
    b = (b - b.mean(dim=0, keepdim=True)).bfloat16()  # each column sums to 0
    ref = (a.to(torch.float64) @ b.to(torch.float64)).float()
    return a, b, ref


def _mcast_1d_config(mt, nt, in0_block_w):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(1, 1),
        in0_block_w=in0_block_w,
        out_subblock_h=mt,
        out_subblock_w=nt,
        per_core_M=mt,
        per_core_N=nt,
        fuse_batch=True,
        mcast_in0=True,
    )


def _mcast_2d_config(mt, nt, in0_block_w):
    # 2x2 grid, one output tile per core; small in0_block_w so K is split into many blocks.
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(2, 2),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=mt // 2,
        per_core_N=nt // 2,
        transpose_mcast=False,
    )


# Each entry drives a distinct matmul program factory that performs the cross-block reload.
_FACTORY_CONFIGS = {
    "mcast_1d": _mcast_1d_config,  # MatmulMultiCoreReuseMultiCast1DProgramConfig
    "mcast_2d": _mcast_2d_config,  # MatmulMultiCoreReuseMultiCastProgramConfig
}


@pytest.mark.parametrize("factory", list(_FACTORY_CONFIGS))
def test_matmul_fp32_crossblock_reload_precision(device, factory):
    m, k, n = 64, 8192, 64
    in0_block_w = 2  # tiles per K-block; Kt=256 -> 128 blocks, so the reload path is exercised
    a, b, ref = _offset_cancellation_inputs(m, k, n, offset=1000.0)

    program_config = _FACTORY_CONFIGS[factory](m // 32, n // 32, in0_block_w)
    # fp32_dest_acc_en=True with packer_l1_acc=False is the configuration whose correctness
    # depends on a lossless fp32 reload of the K-partials.
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    a_t = ttnn.from_torch(
        a, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    b_t = ttnn.from_torch(
        b, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = ttnn.matmul(
        a_t,
        b_t,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.to_torch(out).float()

    # A lossless fp32 reload gives PCC ~0.996 / rel-Frobenius ~0.09; the TF32-truncated reload
    # collapses to PCC ~0.2 / rel-Frobenius ~8. allclose/ULP are not meaningful for a
    # deliberately ill-conditioned bf16 matmul, so only PCC and Frobenius are checked.
    assert_numeric_metrics(
        ref,
        out,
        pcc_threshold=0.99,
        frobenius_threshold=0.5,
        check_allclose=False,
        check_pcc=True,
        check_frobenius=True,
        check_ulp=False,
    )
