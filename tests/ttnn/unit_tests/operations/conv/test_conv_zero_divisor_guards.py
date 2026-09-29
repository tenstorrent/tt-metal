# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


# Each of these arguments used to reach an integer division by zero on the host, which killed the
# process with SIGFPE instead of raising.


def _conv2d(device, in_channels=32, out_channels=32, stride=(1, 1), conv_config=None):
    x = ttnn.from_torch(torch.randn(1, 1, 64, in_channels).bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    w = ttnn.from_torch(torch.randn(out_channels, in_channels, 3, 3).bfloat16(), dtype=ttnn.bfloat16)
    ttnn.conv2d(
        input_tensor=x,
        weight_tensor=w,
        device=device,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=1,
        input_height=8,
        input_width=8,
        kernel_size=(3, 3),
        stride=stride,
        padding=(1, 1),
        conv_config=conv_config,
    )


def _conv2d_stride(device):
    _conv2d(device, stride=(0, 1))


def _conv2d_stride_with_kernel_folding(device):
    _conv2d(device, stride=(0, 2), conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=True))


def _prepare_conv_weights_stride_with_kernel_folding(device):
    w = ttnn.from_torch(torch.randn(32, 32, 3, 3).bfloat16(), dtype=ttnn.bfloat16)
    ttnn.prepare_conv_weights(
        weight_tensor=w,
        input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        weights_format="OIHW",
        in_channels=32,
        out_channels=32,
        batch_size=1,
        input_height=8,
        input_width=8,
        kernel_size=[3, 3],
        stride=[0, 2],
        padding=[1, 1],
        dilation=[1, 1],
        has_bias=False,
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=True),
    )


# With folding left at its default, auto-detection matches kernel_size == stride and takes % stride.
_PREP_DEFAULT_FOLDING = dict(
    input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    input_layout=ttnn.ROW_MAJOR_LAYOUT,
    in_channels=32,
    out_channels=32,
    batch_size=1,
    input_height=8,
    input_width=8,
    kernel_size=[0, 2],
    stride=[0, 2],
    padding=[0, 0],
    dilation=[1, 1],
    groups=1,
    input_dtype=ttnn.bfloat16,
)


def _prepare_conv_weights_stride_default_folding(device):
    w = ttnn.from_torch(torch.randn(32, 32, 1, 2).bfloat16(), dtype=ttnn.bfloat16)
    ttnn.prepare_conv_weights(
        weight_tensor=w, weights_format="OIHW", has_bias=False, device=device, **_PREP_DEFAULT_FOLDING
    )


def _prepare_conv_bias_stride_default_folding(device):
    b = ttnn.from_torch(torch.randn(1, 1, 1, 32).bfloat16(), dtype=ttnn.bfloat16)
    ttnn.prepare_conv_bias(
        bias_tensor=b,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16),
        device=device,
        **_PREP_DEFAULT_FOLDING,
    )


def _conv2d_act_block_w_div(device):
    config = ttnn.Conv2dConfig(act_block_w_div=0, shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED)
    _conv2d(device, in_channels=256, out_channels=64, conv_config=config)


def _conv2d_act_block_h_override(device):
    config = ttnn.Conv2dConfig(act_block_h_override=16, shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED)
    _conv2d(device, conv_config=config)


# A pre-sharded input keeps its own parallel config, so only the block-config check sees the override.
def _conv2d_act_block_h_override_presharded(device):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (32, 32), ttnn.ShardOrientation.ROW_MAJOR),
    )
    x = ttnn.from_torch(
        torch.randn(1, 1, 64, 32).bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem
    )
    w = ttnn.from_torch(torch.randn(32, 32, 3, 3).bfloat16(), dtype=ttnn.bfloat16)
    ttnn.conv2d(
        input_tensor=x,
        weight_tensor=w,
        device=device,
        in_channels=32,
        out_channels=32,
        batch_size=1,
        input_height=8,
        input_width=8,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        conv_config=ttnn.Conv2dConfig(act_block_h_override=16),
    )


def _conv_transpose2d_stride(device):
    x = ttnn.from_torch(torch.randn(1, 1, 64, 32).bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    w = ttnn.from_torch(torch.randn(32, 32, 3, 3).bfloat16(), dtype=ttnn.bfloat16)
    ttnn.conv_transpose2d(
        input_tensor=x,
        weight_tensor=w,
        device=device,
        in_channels=32,
        out_channels=32,
        batch_size=1,
        input_height=8,
        input_width=8,
        kernel_size=(3, 3),
        stride=(0, 1),
        padding=(1, 1),
        output_padding=(0, 0),
        dilation=(1, 1),
        groups=1,
    )


def _conv3d(device, stride=(1, 1, 1), groups=1):
    x = ttnn.from_torch(torch.randn(1, 4, 8, 8, 32).bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    w = ttnn.from_torch(torch.randn(32 * 27, 32).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.experimental.conv3d(
        input_tensor=x,
        weight_tensor=w,
        device=device,
        dtype=ttnn.bfloat16,
        output_channels=32,
        kernel_size=[3, 3, 3],
        stride=list(stride),
        padding=[0, 1, 1],
        groups=groups,
    )


def _conv3d_stride(device):
    _conv3d(device, stride=(0, 1, 1))


def _conv3d_groups(device):
    _conv3d(device, groups=0)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "run, message",
    [
        (_conv2d_stride, "stride must be greater than 0"),
        (_conv2d_stride_with_kernel_folding, "stride must be greater than 0"),
        (_prepare_conv_weights_stride_with_kernel_folding, "stride must be greater than 0"),
        (_prepare_conv_weights_stride_default_folding, "stride must be greater than 0"),
        (_prepare_conv_bias_stride_default_folding, "stride must be greater than 0"),
        (_conv2d_act_block_w_div, "act_block_w_div must be greater than 0"),
        (_conv2d_act_block_h_override, r"act_block_h_override \(16\) must be a multiple of 32"),
        (_conv2d_act_block_h_override_presharded, r"act_block_h_override \(16\) must be a multiple of 32"),
        (_conv_transpose2d_stride, "stride must be greater than 0"),
        (_conv3d_stride, "stride must be greater than 0"),
        (_conv3d_groups, "groups must be greater than 0"),
    ],
    ids=[
        "conv2d_stride",
        "conv2d_stride_kernel_folding",
        "prepare_conv_weights_stride_kernel_folding",
        "prepare_conv_weights_stride_default_folding",
        "prepare_conv_bias_stride_default_folding",
        "conv2d_act_block_w_div",
        "conv2d_act_block_h_override",
        "conv2d_act_block_h_override_presharded",
        "conv_transpose2d_stride",
        "conv3d_stride",
        "conv3d_groups",
    ],
)
def test_zero_divisor_raises(device, expect_error, run, message):
    with expect_error(RuntimeError, message):
        run(device)
