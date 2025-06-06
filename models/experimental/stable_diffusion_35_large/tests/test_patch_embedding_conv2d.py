# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import pytest
import torch
import ttnn

from ..tt.patch_embedding import TtPatchEmbeddingConv2d, TtPatchEmbeddingConv2dParameters
from ..tt.utils import assert_quality, to_torch

TILE_SIZE = 32


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    ("batch_size", "in_channels", "out_channels", "kernel_size", "stride", "height", "width"),
    [
        # (2, 16, 2560, (2, 2), (2, 2), 64, 64),
        (2, 16, 2560, (2, 2), (2, 2), 128, 128),
        (2, 16, 2432, (2, 2), (2, 2), 128, 128),
        # (2, 16, 1536, (2, 2), (2, 2), 64, 64),
        # (2, 16, 1536, (2, 2), (2, 2), 128, 128),
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

    num_devices = mesh_device.get_num_devices()
    pad_embedding_dim = False
    if os.environ["MESH_DEVICE"] == "T3K" and out_channels == 2432:
        pad_embedding_dim = True
        hidden_dim_padding = (((out_channels // num_devices // TILE_SIZE) + 1) * TILE_SIZE) * num_devices - out_channels
    else:
        hidden_dim_padding = 0

    parameters = TtPatchEmbeddingConv2dParameters.from_torch(
        torch_model.state_dict(),
        dtype=dtype,
        hidden_dim_padding=hidden_dim_padding,
        out_channels=out_channels,
        device=mesh_device,
    )
    tt_model = TtPatchEmbeddingConv2d(parameters, mesh_device)

    torch_input_tensor = torch.randn((batch_size, in_channels, height, width))

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor.permute([0, 2, 3, 1]),  # BCYX -> BYXC
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
    )

    tt_output = tt_model(tt_input_tensor)
    tt_output_torch = to_torch(tt_output, mesh_device=mesh_device, shard_dim=-1)
    tt_output_torch = tt_output_torch.permute(0, 3, 1, 2)

    assert_quality(torch_output, tt_output_torch, pcc=0.999_900)
