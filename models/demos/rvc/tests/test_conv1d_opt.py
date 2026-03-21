# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch

import ttnn
from models.demos.rvc.tt_impl.conv1d import Conv1d


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size,input_length,in_channels,out_channels,kernel_size,stride,padding,dilation,groups",
    [
        # Conv1d: batch_size=1, input_length=35600, output_length=35600, in_channels=256, out_channels=256,
        # kernel_size=11, stride=1, padding=5, dilation=1, groups=1
        # (1, 35600, 256, 256, 11, 1, 5, 1, 1),
        # Conv1d: batch_size=1, input_length=1780, output_length=1781, in_channels=768, out_channels=768,
        # kernel_size=128, stride=1, padding=64, dilation=1, groups=1
        # (1, 1780, 768, 768, 128, 1, "same", 1, 1),
        # Old: output_length=113986
        # (1, 569938, 1, 512, 10, 5, 0, 1, 1),
        # New: output_length=(113986//128)*128 = 113920
        # (1, 569605, 1, 512, 10, 5, 0, 1, 1),
        # (1, 113986, 512, 512, 3, 2, 0, 1, 1),
        # (1, 56992, 512, 512, 3, 2, 0, 1, 1),
        # (1, 28495, 512, 512, 3, 2, 0, 1, 1),
        # (1, 1708800, 1, 256, 96, 48, 24, 1, 1),
        # (1, 854400, 32, 32, 3, 1, 1, 1, 1),
        (1, 854400, 32, 32, 11, 1, 5, 1, 1),
    ],
)
def test_conv1d_opt(
    device, batch_size, input_length, in_channels, out_channels, kernel_size, stride, padding, dilation, groups
):
    torch.manual_seed(0)

    # Internal Conv2d-shaped activation for Conv1d wrapper is [N, 1, L, C].
    input_2d_shape = (batch_size, 1, input_length, in_channels)
    torch_input = torch.randn((batch_size, in_channels, input_length), dtype=torch.bfloat16).float()
    torch_input_nlc = torch_input.permute(0, 2, 1).reshape(batch_size, input_length, in_channels)
    tt_input = ttnn.from_torch(
        torch_input_nlc,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    compute_grid = device.compute_with_storage_grid_size()
    full_core_grid = ttnn.CoreGrid(y=compute_grid.y, x=compute_grid.x)
    # input_sharded_config = ttnn.create_sharded_memory_config_(
    #     shape=tt_input.shape,
    #     core_grid=full_core_grid,
    #     strategy=ttnn.ShardStrategy.BLOCK,
    #     orientation=ttnn.ShardOrientation.ROW_MAJOR,
    # )
    # tt_input = ttnn.to_memory_config(tt_input, input_sharded_config)

    torch_conv = torch.nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=True,
    ).eval()

    tt_conv = Conv1d(
        device=device,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        dtype=ttnn.bfloat16,
    )
    tt_conv.load_parameters(
        {
            "conv.weight": torch_conv.weight.detach().cpu(),
            "conv.bias": torch_conv.bias.detach().cpu(),
        },
        key="conv",
    )

    warmup_iters = 3
    measure_iters = 10

    for _ in range(warmup_iters):
        _ = tt_conv(tt_input)
    ttnn.synchronize_device(device)

    start = time.perf_counter()
    for _ in range(measure_iters):
        tt_output = tt_conv(tt_input)
    ttnn.synchronize_device(device)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / measure_iters) * 1000.0
    total_ms = elapsed * 1000.0
    print(
        "Conv1d speed:"
        f" warmup_iters={warmup_iters}, measure_iters={measure_iters},"
        f" total_ms={total_ms:.3f}, avg_ms={avg_ms:.3f}"
    )

    tt_output_torch = ttnn.to_torch(tt_output)
