# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Regression test for the Qwen 2.5-VL vision patch-embed Conv3D shape.

Before the L1-aware auto-shrink in `conv3d_program_factory.cpp`, the default
`Conv3dConfig` (and the values currently emitted by tt-mlir's lowering)
produced per-core circular buffers that exceeded Wormhole's L1 budget:

    Statically allocated circular buffers on core range [0-0 - 7-7]
    grow to 1738528 B which is beyond max L1 size of 1499136 B

The Qwen 2.5-VL-3B-Instruct vision encoder begins with:

    nn.Conv3d(in_channels=3, out_channels=1280,
              kernel_size=[2, 14, 14], stride=[2, 14, 14], bias=False)

which has matmul K = kT*kH*kW*C_in_block = 2*14*14*32 = 12544 — large
enough that the weight + vol2col CBs blow L1 with the default `C_in_block`.
The auto-shrink picks a smaller `(C_in_block, C_out_block)` pair that fits
(for this shape: `C_in_block=16`).

This test pins the failing config and asserts the op runs to completion
with PCC ≥ 0.99 against torch CPU reference. If a future change reverts
or breaks the auto-shrink, this test will fail — either with an OOM or
with a PCC drop.
"""

from loguru import logger
import pytest
import torch
import torch.nn as nn
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc


ALIGNMENT = 32  # L1 alignment for Wormhole and Blackhole.


def _out_size(in_size, pad, stride, k, dilation):
    effective_k = (dilation * (k - 1)) + 1
    return (in_size + 2 * pad - effective_k) // stride + 1


def _prepare_input_tensor(input_tensor, C, device, alignment=ALIGNMENT, dtype=ttnn.DataType.BFLOAT16):
    """Permute (N,C,D,H,W) -> (N,D,H,W,C) and right-pad C to `alignment`."""
    tt_input = input_tensor.permute(0, 2, 3, 4, 1)
    if C % alignment != 0:
        align_pad = alignment - C % alignment
        tt_input = torch.nn.functional.pad(tt_input, (0, align_pad))
    return ttnn.from_torch(tt_input, device=device, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)


@pytest.mark.parametrize("N", [64], ids=["N_64"])
def test_qwen_2_5_vl_patch_embed_conv3d(device, N):
    """Default `Conv3dConfig` must fit L1 for the Qwen patch-embed shape."""
    torch.manual_seed(42)

    # Qwen 2.5-VL-3B-Instruct vision-config patch-embed parameters.
    C_in = 3
    out_channels = 1280
    kernel_size = (2, 14, 14)
    stride = (2, 14, 14)
    padding = (0, 0, 0)
    dilation = (1, 1, 1)
    groups = 1
    padding_mode = "zeros"
    dtype = ttnn.DataType.BFLOAT16

    D, H, W = 2, 14, 14
    input_shape = (N, C_in, D, H, W)
    D_out = _out_size(D, padding[0], stride[0], kernel_size[0], dilation[0])
    H_out = _out_size(H, padding[1], stride[1], kernel_size[1], dilation[1])
    W_out = _out_size(W, padding[2], stride[2], kernel_size[2], dilation[2])

    input_tensor = torch.randn(*input_shape, dtype=torch.float32)
    conv3d_module = nn.Conv3d(
        C_in,
        out_channels,
        groups=groups,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=False,  # Qwen patch embed has bias=False.
        padding_mode=padding_mode,
    )
    gt_output = conv3d_module(input_tensor)

    tt_input = _prepare_input_tensor(input_tensor, C_in, device, dtype=dtype)

    # Match the compute config emitted by tt-mlir's lowering for this op.
    kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    # `C_out_block=0` and `C_in_block=ALIGNMENT` (=32) is the case that
    # used to TT_THROW with "grow to 1738528 B" before the auto-shrink.
    grid_size = device.compute_with_storage_grid_size()
    config = ttnn.Conv3dConfig(
        weights_dtype=dtype,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        T_out_block=1,
        W_out_block=1,
        H_out_block=1,
        C_out_block=0,
        C_in_block=ALIGNMENT,
        dilation=dilation,
        compute_with_storage_grid_size=grid_size,
    )

    # Pass weight as a rank-5 host tensor so conv3d.cpp prepares it AFTER the
    # L1-aware auto-shrink resolves the final C_in_block. This is the same
    # path tt-mlir's lowering uses. Pre-preparing weights via
    # `prepare_conv3d_weights` with a fixed C_in_block (as
    # `tests/ttnn/unit_tests/operations/conv/test_conv3d.py` does) bypasses
    # the auto-shrink and locks the layout to the user's C_in_block — only
    # safe when that value already fits L1.
    w_host = ttnn.from_torch(conv3d_module.weight.data, dtype=dtype, pad_value=0)

    tt_output = ttnn.experimental.conv3d(
        input_tensor=tt_input,
        weight_tensor=w_host,
        device=device,
        bias_tensor=None,
        dtype=dtype,
        output_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        groups=groups,
        padding=padding,
        dilation=dilation,
        padding_mode=padding_mode,
        config=config,
        compute_kernel_config=kernel_config,
    )

    tt_output = ttnn.to_torch(tt_output, device=device, dtype=torch.float32)
    tt_output = tt_output.reshape(N, D_out, H_out, W_out, out_channels)
    tt_output = tt_output.permute(0, 4, 1, 2, 3)

    assert tt_output.shape == gt_output.shape, f"shape mismatch: tt={tt_output.shape} gt={gt_output.shape}"

    pcc_passed, pcc_message = check_with_pcc(gt_output, tt_output, pcc=0.99)
    logger.info("conv3d torch vs ttnn: {}", pcc_message)
    assert pcc_passed, pcc_message
