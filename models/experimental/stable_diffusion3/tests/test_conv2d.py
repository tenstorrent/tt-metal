# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import pytest
import torch
import ttnn

from ..tt.patch_embedding import TtConv2d, TtConv2dParameters
from ..tt.utils import assert_quality, to_torch


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    ("batch_size", "in_channels", "out_channels", "kernel_size", "stride", "height", "width"),
    [
        (2, 16, 2432, (2, 2), (2, 2), 64, 64),
        (2, 16, 2432, (2, 2), (2, 2), 128, 128),
        (2, 16, 1536, (2, 2), (2, 2), 64, 64),
        (2, 16, 1536, (2, 2), (2, 2), 128, 128),
        # these are needed in the VAE for an image resolution of 1024x1024:
        # (1, 128, 128, (3, 3), (1, 1), 1024, 1024),
        # (1, 128, 3, (3, 3), (1, 1), 1024, 1024),
        # (1, 16, 512, (3, 3), (1, 1), 128, 128),
        # (1, 256, 128, (3, 3), (1, 1), 1024, 1024),
        # (1, 256, 256, (3, 3), (1, 1), 1024, 1024),
        # (1, 256, 256, (3, 3), (1, 1), 512, 512),
        # (1, 512, 512, (3, 3), (1, 1), 128, 128),
        # (1, 512, 512, (3, 3), (1, 1), 256, 256),
        # (1, 512, 256, (3, 3), (1, 1), 512, 512),
        # (1, 512, 512, (3, 3), (1, 1), 512, 512),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.usefixtures("use_program_cache")
def test_conv2d(
    *,
    mesh_device: ttnn.MeshDevice,
    batch_size: int,
    in_channels: int,
    out_channels: int,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    height: int,
    width: int,
) -> None:
    dtype = ttnn.bfloat16

    torch_model = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
    )
    torch_model.eval()

    parameters = TtConv2dParameters.from_torch(
        torch_model.state_dict(),
        dtype=dtype,
        out_channels=out_channels,
        device=mesh_device,
    )
    tt_model = TtConv2d(parameters, mesh_device)

    torch_input_tensor = torch.randn((batch_size, in_channels, height, width))

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    ## Pre-processing for the ttnn.fold
    torch_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))  # BCYX -> BYXC
    batch_size, img_h, img_w, img_c = torch_input_tensor.shape  # permuted input NHWC
    patch_size = 2
    torch_input_tensor = torch_input_tensor.reshape(batch_size, img_h, img_w // patch_size, patch_size, img_c)
    torch_input_tensor = torch_input_tensor.reshape(batch_size, img_h, img_w // patch_size, patch_size * img_c)
    N, H, W, C = torch_input_tensor.shape
    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(7, 7),
            ),
        }
    )
    n_cores = 64
    shard_spec = ttnn.ShardSpec(shard_grid, [N * H * W // n_cores, C], ttnn.ShardOrientation.ROW_MAJOR)

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            shard_spec,
        ),
    )

    tt_output = tt_model(tt_input_tensor)
    tt_output_torch = (
        to_torch(tt_output, mesh_device=mesh_device, dtype=dtype, shard_dim=0)
        .permute(0, 1, 3, 2)
        .reshape(batch_size, out_channels, height // kernel_size[1], width // kernel_size[0])
    )

    # print(torch_output.shape, tt_output_torch.shape, tt_output.shape)
    assert_quality(torch_output, tt_output_torch, pcc=0.999_900, shard_dim=0, num_devices=mesh_device.get_num_devices())
