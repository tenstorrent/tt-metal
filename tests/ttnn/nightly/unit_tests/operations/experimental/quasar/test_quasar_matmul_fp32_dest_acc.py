# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""#61018: fp32_dest_acc_en through the Quasar copies of the Reuse, 1D (in0 and in1 multicast) and 2D factories.

The fp32 partials are reloaded into Dest between K blocks. They must stay fp32: split into 4 or 16 K blocks, with or
without packer_l1_acc, the result must match the single-K-block result. A reload through SrcA rounds the partials to
TF32 at every block boundary, which PCC does not catch. The fused bias (1D only) reads the partials through SrcA, so
those matmuls accumulate in L1 instead of reloading."""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# Every test in this module takes the function-scoped `device` fixture without
# parametrizing device_params, so the device is opened once per module rather than
# once per test.
pytestmark = pytest.mark.use_module_device

qsr = ttnn._ttnn.operations.experimental.quasar

M, K, N = 64, 512, 128
MT, NT = M // 32, N // 32


def _program_config(family, in0_block_w):
    if family == "reuse":
        return qsr.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(2, 1),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=NT,
        )
    if family in ("1d_in0", "1d_in1"):
        mcast_in0 = family == "1d_in0"
        return qsr.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(2, 1),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=2,
            per_core_M=MT if mcast_in0 else MT // 2,
            per_core_N=NT // 2 if mcast_in0 else NT,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=mcast_in0,
        )
    return qsr.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(2, 2),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=2,
        per_core_M=MT // 2,
        per_core_N=NT // 2,
        transpose_mcast=False,
        fused_activation=None,
    )


def _run(device, family, in0_block_w, packer_l1_acc, a, b, bias):
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=packer_l1_acc,
    )
    kwargs = dict(
        program_config=_program_config(family, in0_block_w),
        compute_kernel_config=compute_kernel_config,
        dtype=ttnn.float32,
    )
    if bias is None:
        output = ttnn.experimental.quasar.matmul(a, b, **kwargs)
    else:
        output = ttnn.experimental.quasar.linear(a, b, bias=bias, **kwargs)
    return ttnn.to_torch(output).float()


@pytest.mark.parametrize("family", ["reuse", "1d_in0", "1d_in1", "2d"])
@pytest.mark.parametrize("fused_bias", [False, True], ids=["no_bias", "fused_bias"])
def test_quasar_matmul_fp32_dest_acc_k_blocks(device, family, fused_bias):
    if fused_bias and not family.startswith("1d"):
        pytest.skip("only the 1D multicast factory fuses the bias on Quasar")
    torch.manual_seed(0)
    torch_a = torch.randn(1, 1, M, K).bfloat16()
    torch_b = torch.randn(1, 1, K, N).bfloat16()
    a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    expected = (torch_a.double() @ torch_b.double()).float()
    bias = None
    if fused_bias:
        torch_bias = torch.randn(1, 1, 1, N).bfloat16()
        bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        expected = expected + torch_bias.float()

    # One K block: the whole reduction accumulates in Dest, no partials are reloaded.
    single = _run(device, family, K // 32, False, a, b, bias)
    assert_with_pcc(expected, single, 0.9999)
    single_error = (single - expected).abs().mean().item()
    for in0_block_w, packer_l1_acc in [(4, False), (4, True), (1, False)]:
        split = _run(device, family, in0_block_w, packer_l1_acc, a, b, bias)
        label = f"{K // (in0_block_w * 32)} K blocks, packer_l1_acc={packer_l1_acc}"
        if fused_bias:
            # The bias add reads the partials through SrcA and rounds the sum to TF32 once in every split, so single
            # elements may land a TF32 step apart (the last fp32 bits of an L1 and a Dest accumulation differ). A TF32
            # round at every block boundary on top of that would at least double the mean error.
            split_error = (split - expected).abs().mean().item()
            assert split_error <= 1.25 * single_error, f"{label}: mean error {split_error} vs {single_error}"
        else:
            # A TF32 round of the partials at each block boundary moves the result by up to 8e-2 here at 4 blocks and
            # 3e-1 at 16; fp32 partials leave it unchanged.
            max_diff = (split - single).abs().max().item()
            assert max_diff <= 1e-3, f"{label}: max |split - single K block| = {max_diff}"
