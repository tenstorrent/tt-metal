# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
import ttnn

from ..tt.conv2d import TtConv2d, TtConv2dParameters
from ..tt.utils import assert_quality, from_torch_fast, to_torch


@pytest.mark.parametrize(
    ("batch_size", "in_channels", "out_channels", "kernel_size", "stride", "height", "width"),
    [
        (1, 32, 32, (2, 3), (2, 2), 64, 64),
        (1, 20, 32, (3, 3), (2, 3), 128, 256),
        # these are needed in the VAE for an image resolution of 1024x1024:
        (1, 128, 128, (3, 3), (1, 1), 1024, 1024),
        # the next case with lower slice_count https://github.com/tenstorrent/tt-metal/issues/17489#issuecomment-2886552080
        (1, 128, 3, (3, 3), (1, 1), 1024, 1024),
        (1, 16, 512, (3, 3), (1, 1), 128, 128),
        (1, 256, 128, (3, 3), (1, 1), 1024, 1024),
        (1, 256, 256, (3, 3), (1, 1), 1024, 1024),
        (1, 256, 256, (3, 3), (1, 1), 512, 512),
        (1, 512, 512, (3, 3), (1, 1), 128, 128),
        (1, 512, 512, (3, 3), (1, 1), 256, 256),
        (1, 512, 256, (3, 3), (1, 1), 512, 512),
        (1, 512, 512, (3, 3), (1, 1), 512, 512),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192 * 2}], indirect=True)
@pytest.mark.usefixtures("use_program_cache")
def test_conv2d(
    *,
    mesh_device: ttnn.Device,
    batch_size: int,
    in_channels: int,
    out_channels: int,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    height: int,
    width: int,
) -> None:
    # TODO: #23290 - Fix the underlying issue.
    skip_configs = [
        (1, 128, 128, (3, 3), (1, 1), 1024, 1024),
        (1, 128, 3, (3, 3), (1, 1), 1024, 1024),
        (1, 256, 128, (3, 3), (1, 1), 1024, 1024),
        (1, 256, 256, (3, 3), (1, 1), 1024, 1024),
        (1, 256, 256, (3, 3), (1, 1), 512, 512),
        (1, 512, 512, (3, 3), (1, 1), 256, 256),
        (1, 512, 256, (3, 3), (1, 1), 512, 512),
        (1, 512, 512, (3, 3), (1, 1), 512, 512),
    ]

    current_config = (batch_size, in_channels, out_channels, kernel_size, stride, height, width)
    if current_config in skip_configs:
        pytest.skip("Configuration expected to fail with memory config error")

    dtype = ttnn.bfloat16

    total_batch_size = batch_size * mesh_device.get_num_devices()

    torch.manual_seed(0)

    torch_model = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
    )
    torch_model.eval()

    parameters = TtConv2dParameters.from_torch(torch_model.state_dict(), dtype=dtype, device=mesh_device)
    tt_model = TtConv2d(parameters, stride=stride)

    torch_input_tensor = torch.randn((total_batch_size, in_channels, height, width))

    tt_input_tensor = from_torch_fast(
        torch_input_tensor.permute([0, 2, 3, 1]),  # BCYX -> BYXC
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        shard_dim=0,
    )

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    tt_output = tt_model(tt_input_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_output_torch = to_torch(tt_output, mesh_device=mesh_device, shard_dim=0).permute([0, 3, 1, 2])

    assert_quality(torch_output, tt_output_torch, pcc=0.95, ccc=0.949)
