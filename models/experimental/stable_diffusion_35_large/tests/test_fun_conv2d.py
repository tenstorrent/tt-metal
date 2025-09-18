# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
import ttnn
import math

from ..tt.fun_conv2d import TtConv2dParameters, sd_conv2d
from ..tt.utils import assert_quality, from_torch_fast_2d
from ..tt.parallel_config import StableDiffusionParallelManager

TILE_SIZE = 32


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
    ),
    [
        # (2, 16, 2560, (2, 2), (2, 2), 64, 64),
        (1, 16, 2560, 40, (2, 2), (2, 2), 128, 128),
        (1, 16, 2432, 38, (2, 2), (2, 2), 128, 128),
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
@pytest.mark.parametrize(
    (
        "mesh_device",
        "cfg",
        "sp",
        "tp",
        "topology",
        "num_links",
    ),
    [
        [(2, 4), (1, 0), (2, 0), (4, 1), ttnn.Topology.Linear, 1],
        [(4, 8), (2, 1), (4, 0), (4, 1), ttnn.Topology.Linear, 3],
    ],
    ids=[
        "t3k_cfg1_sp2_tp4",
        "tg_cfg2_sp4_tp4",
    ],
    indirect=["mesh_device"],
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
    cfg: tuple[int, int],
    sp: tuple[int, int],
    tp: tuple[int, int],
    topology: ttnn.Topology,
    num_links: int,
) -> None:
    cfg_factor, cfg_axis = cfg
    sp_factor, sp_axis = sp
    tp_factor, tp_axis = tp
    parallel_manager = StableDiffusionParallelManager(
        mesh_device,
        cfg_factor,
        sp_factor,
        tp_factor,
        sp_factor,
        tp_factor,
        topology,
        cfg_axis=cfg_axis,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
    )
    submesh = parallel_manager.submesh_devices[0]
    torch_dtype = torch.float32
    ttnn_dtype = ttnn.bfloat16

    torch_model = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
    )
    torch_model.eval()

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
        device=submesh,
        parallel_config=parallel_manager.dit_parallel_config,
    )

    torch_input_tensor = torch.randn((batch_size, in_channels, height, width), dtype=torch_dtype)

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    seq_parallel_shard_dim = 1  # can either do 2 = width or 1 = height
    channels_shard_dim = 3  # output channels
    dims = [None, None]
    dims[parallel_manager.dit_parallel_config.sequence_parallel.mesh_axis] = seq_parallel_shard_dim
    tt_input_tensor = from_torch_fast_2d(
        torch_input_tensor.permute([0, 2, 3, 1]),  # BCYX -> BYXC
        mesh_device=submesh,
        mesh_shape=parallel_manager.dit_parallel_config.cfg_parallel.mesh_shape,
        dims=dims,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
    )

    dims = [None, None]
    dims[parallel_manager.dit_parallel_config.sequence_parallel.mesh_axis] = seq_parallel_shard_dim
    dims[parallel_manager.dit_parallel_config.tensor_parallel.mesh_axis] = channels_shard_dim
    tt_output = sd_conv2d(tt_input_tensor, parameters, parallel_manager)
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            submesh,
            mesh_shape=tuple(submesh.shape),
            dims=dims,
        ),
    )
    tt_output_torch = tt_output_torch.permute(0, 3, 1, 2)
    tt_output_torch = tt_output_torch[:, 0:out_channels, :, :]

    assert_quality(torch_output, tt_output_torch, pcc=0.999_900, shard_dim=0, num_devices=submesh.get_num_devices())
