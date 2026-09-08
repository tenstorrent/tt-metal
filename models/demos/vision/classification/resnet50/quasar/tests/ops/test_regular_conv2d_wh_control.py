# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
CONTROL / A-B experiment: run the MAINLINE ttnn.conv2d (NOT ttnn.experimental.quasar.conv2d) on WH with the
exact shapes + compute configs where the quasar conv2d fork hangs, to answer: "does the regular conv2d have
the same hangs?"

Background (see the quasar ops tests): the quasar fork hangs on WH in two places --
  1. the SPLIT path's Program A (conv_tilize_only) regular datacopy tilize_block MATH<->PACK DEST-sync
     deadlock (test_conv2d_split_program*, e2e_pure/_shapes/gap_wide_n), and
  2. the FUSED conv_bmm_tilize_metal2 fast_tilize pack-flush race (test_conv2d_correctness_bisect[relu_now_sfpu],
     test_conv2d.py[stem_7x7], layer conv2 fused).
Mainline conv2d has NO split / conv_tilize_only / drain_out, so (1) cannot be reproduced here -- but mainline
conv_bmm_tilize.cpp uses the IDENTICAL compute_kernel_lib::tilize + fast_tilize as the quasar fork, so (2) is a
SHARED WH LLK path. The race-guard (kRaceGuardSpin) exists only in the quasar fork; mainline has none. This
control runs the fused mainline path on the same shapes to see whether it hangs too (=> broad WH LLK bug) or
passes (=> the fork's Metal-2.0 cadence / forced SFPU-relu is what exposes the latent race).

READING IT:
  * A config here HANGS on WH  -> the fast_tilize race bites mainline too; it's a general WH LLK bug.
  * A config PASSES on WH      -> mainline's cadence avoids the race; the quasar fork exposes it (fork-specific
                                  trigger, though the underlying LLK race is still latent).
Each case has a timeout so a hang is recorded (not an infinite wait). Run WITHOUT any TT_METAL_QSR_* env
(mainline conv2d ignores the split env anyway).

Run (WH):
  pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_regular_conv2d_wh_control.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

PCC = 0.98


def _run_regular_conv2d(
    mesh_device,
    *,
    in_channels,
    out_channels,
    kernel_size,
    out_h,
    out_w,
    stride=(1, 1),
    padding=(0, 0),
    act_block_h_override=128,
    with_bias_relu=False,
    packer_l1_acc=True,
):
    """Mainline ttnn.conv2d on one shape/config, HEIGHT_SHARDED, matched to the quasar fork's compute config."""
    device = mesh_device
    torch.manual_seed(0)
    batch_size = 1
    kh, kw = kernel_size
    # in = (out - 1) * stride + kernel - 2 * pad
    input_height = (out_h - 1) * stride[0] + kh - 2 * padding[0]
    input_width = (out_w - 1) * stride[1] + kw - 2 * padding[1]

    torch_input_nchw = torch.randn((batch_size, in_channels, input_height, input_width), dtype=torch.bfloat16).float()
    torch_weight = torch.randn((out_channels, in_channels, kh, kw), dtype=torch.bfloat16).float()
    torch_bias = torch.randn((out_channels,), dtype=torch.bfloat16).float() if with_bias_relu else None
    torch_golden = torch.nn.functional.conv2d(
        torch_input_nchw, torch_weight, bias=torch_bias, stride=stride, padding=padding
    )
    if with_bias_relu:
        torch_golden = torch.relu(torch_golden)

    # Mainline conv2d input: [1, 1, N*H*W, C] row-major on DRAM; conv2d reshards to its HEIGHT_SHARDED layout.
    nhw = batch_size * input_height * input_width
    flat = torch.permute(torch_input_nchw, (0, 2, 3, 1)).reshape(1, 1, nhw, in_channels).contiguous()
    tt_input = ttnn.from_torch(flat, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16)
    tt_bias = (
        ttnn.from_torch(torch_bias.reshape(1, 1, 1, out_channels), dtype=ttnn.bfloat16) if with_bias_relu else None
    )

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        reshard_if_not_optimal=True,
        act_block_h_override=(act_block_h_override if act_block_h_override is not None else 0),
        activation=(ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU) if with_bias_relu else None),
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, packer_l1_acc=packer_l1_acc
    )

    out, [oh, ow], [tt_weight, tt_bias] = ttnn.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=(1, 1),
        groups=1,
        device=device,
        conv_config=conv_config,
        compute_config=compute_config,
        return_output_dim=True,
        return_weights_and_bias=True,
        dtype=ttnn.bfloat16,
    )

    tt_out = ttnn.to_torch(ttnn.from_device(out)).reshape(batch_size, oh, ow, -1)[:, :, :, :out_channels]
    tt_out = torch.permute(tt_out, (0, 3, 1, 2))
    assert_with_pcc(torch_golden, tt_out.float(), pcc=PCC)


# Each dict mirrors a quasar-fork WH hang. `note` names the quasar test it corresponds to.
_CASES = [
    # 4x4 stem-like (K=16), act_block_h=128 -> the quasar SPLIT Program A tilize deadlock geometry.
    # Mainline runs the FUSED fast_tilize path on the same shape (no split exists in mainline).
    dict(in_channels=32, out_channels=64, kernel_size=(4, 4), out_h=16, out_w=32, act_block_h_override=128),
    dict(
        in_channels=32,
        out_channels=64,
        kernel_size=(4, 4),
        out_h=16,
        out_w=32,
        act_block_h_override=128,
        with_bias_relu=True,
    ),
    # wide-N 4x4 + fused bias+relu -> the quasar gap_wide_n_fused_bias geometry.
    dict(
        in_channels=32,
        out_channels=256,
        kernel_size=(4, 4),
        out_h=16,
        out_w=32,
        act_block_h_override=128,
        with_bias_relu=True,
    ),
    # 3x3 + bias + relu (packer_l1_acc) -> the correctness_bisect[relu_now_sfpu] config (mainline WH uses
    # packer-relu here; the quasar test forces SFPU-relu, which is the fork-specific perturbant).
    dict(
        in_channels=64,
        out_channels=64,
        kernel_size=(3, 3),
        out_h=16,
        out_w=16,
        act_block_h_override=128,
        with_bias_relu=True,
    ),
    # logical 7x7/s2 stem (3->64, 224) -> the test_conv2d.py[stem_7x7] geometry (fused, fast_tilize).
    dict(
        in_channels=3,
        out_channels=64,
        kernel_size=(7, 7),
        out_h=112,
        out_w=112,
        stride=(2, 2),
        padding=(3, 3),
        act_block_h_override=128,
        with_bias_relu=True,
    ),
]
_IDS = ["stem4x4_pure", "stem4x4_bias_relu", "wide_n256_4x4_bias_relu", "relu_3x3_bias", "stem7x7_s2"]


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_regular_conv2d_wh_control(mesh_device, case):
    """Run mainline ttnn.conv2d on a shape where the quasar fork hangs on WH. Hang (timeout) => shared WH LLK
    fast_tilize race bites mainline too; pass => the fork's cadence/split is what exposes it."""
    _run_regular_conv2d(mesh_device, **case)
