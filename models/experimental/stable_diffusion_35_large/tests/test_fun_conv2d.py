# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import pytest
import torch
import ttnn
import math

from ..tt.fun_conv2d import TtConv2dParameters, sd_conv2d
from ..tt.utils import assert_quality, from_torch_fast_2d, initialize_sd_parallel_config

TILE_SIZE = 32


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (2, 4), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    (
        "batch_size",
        "in_channels",
        "out_channels",
        "num_heads",
        "kernel_size",
        "stride",
        "height",
        "width",
        "cfg_factor",
        "sp_factor",
        "tp_factor",
        "topology",
    ),
    [
        # (2, 16, 2560, (2, 2), (2, 2), 64, 64),
        (1, 16, 2560, 40, (2, 2), (2, 2), 128, 128, 1, 2, 4, ttnn.Topology.Linear),
        (1, 16, 2432, 38, (2, 2), (2, 2), 128, 128, 1, 2, 4, ttnn.Topology.Linear),
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
def test_conv2d(
    *,
    mesh_device: ttnn.MeshDevice,
    batch_size: int,
    in_channels: int,
    out_channels: int,
    num_heads: int,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    height: int,
    width: int,
    cfg_factor: int,
    sp_factor: int,
    tp_factor: int,
    topology: ttnn.Topology,
) -> None:
    mesh_shape = tuple(mesh_device.shape)
    dit_parallel_config = initialize_sd_parallel_config(mesh_shape, cfg_factor, sp_factor, tp_factor, topology)
    torch_dtype = torch.float32
    ttnn_dtype = ttnn.bfloat16

    torch_model = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
    )
    torch_model.eval()

    num_devices = mesh_device.get_num_devices()
    pad_embedding_dim = (bool)(num_heads) % tp_factor
    if pad_embedding_dim:
        head_size = out_channels // num_heads
        num_heads = math.ceil(num_heads / tp_factor) * tp_factor
        hidden_dim_padding = (num_heads * head_size) - out_channels
    else:
        hidden_dim_padding = 0

    parameters = TtConv2dParameters.from_torch(
        torch_model.state_dict(),
        dtype=ttnn_dtype,
        hidden_dim_padding=hidden_dim_padding,
        out_channels=out_channels,
        device=mesh_device,
        parallel_config=dit_parallel_config,
    )

    torch_input_tensor = torch.randn((batch_size, in_channels, height, width), dtype=torch_dtype)

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    seq_parallel_shard_dim = 1  # can either do 2 = width or 1 = height
    tt_input_tensor = from_torch_fast_2d(
        torch_input_tensor.permute([0, 2, 3, 1]),  # BCYX -> BYXC
        mesh_device=mesh_device,
        mesh_shape=dit_parallel_config.cfg_parallel.mesh_shape,
        dims=[seq_parallel_shard_dim, None],
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_output = sd_conv2d(tt_input_tensor, parameters, dit_parallel_config)
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device,
            mesh_shape=tuple(mesh_device.shape),
            dims=[
                dit_parallel_config.sequence_parallel.mesh_axis + seq_parallel_shard_dim,
                dit_parallel_config.tensor_parallel.mesh_axis + 2,
            ],
        ),
    )
    tt_output_torch = tt_output_torch.permute(0, 3, 1, 2)
    tt_output_torch = tt_output_torch[:, 0:out_channels, :, :]

    assert_quality(torch_output, tt_output_torch, pcc=0.999_900, shard_dim=0, num_devices=mesh_device.get_num_devices())
