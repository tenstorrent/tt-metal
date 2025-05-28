# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import os
import pytest
import torch
import ttnn
import math

from ..reference import SD3Transformer2DModel
from ..tt.fun_patch_embedding import sd_patch_embed, TtPatchEmbedParameters
from ..tt.utils import assert_quality, from_torch_fast_2d, initialize_sd_parallel_config

if TYPE_CHECKING:
    from ..reference.patch_embedding import PatchEmbed

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
        "model_name",
        "batch_size",
        "in_channels",
        "height",
        "width",
        "cfg_factor",
        "sp_factor",
        "tp_factor",
        "topology",
    ),
    [
        ("large", 1, 16, 128, 128, 1, 2, 4, ttnn.Topology.Linear),
        ("large", 2, 16, 128, 128, 1, 2, 4, ttnn.Topology.Linear),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
def test_patch_embedding(
    *,
    mesh_device: ttnn.MeshDevice,
    model_name,
    batch_size: int,
    in_channels: int,
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

    parent_torch_model = SD3Transformer2DModel.from_pretrained(
        f"stabilityai/stable-diffusion-3.5-{model_name}", subfolder="transformer", torch_dtype=torch_dtype
    )
    embedding_dim = 1536 if model_name == "medium" else 2432

    torch_model: PatchEmbed = parent_torch_model.pos_embed
    torch_model.eval()

    ## heads padding
    assert not embedding_dim % parent_torch_model.transformer_blocks[0].num_heads, "Embedding_dim % num_heads != 0"
    pad_embedding_dim = (bool)(parent_torch_model.transformer_blocks[0].num_heads) % tp_factor
    if pad_embedding_dim:
        head_size = embedding_dim // parent_torch_model.transformer_blocks[0].num_heads
        num_heads = math.ceil(parent_torch_model.transformer_blocks[0].num_heads / tp_factor) * tp_factor
        hidden_dim_padding = (num_heads * head_size) - embedding_dim
    else:
        num_heads = parent_torch_model.transformer_blocks[0].num_heads

    parameters = TtPatchEmbedParameters.from_torch(
        torch_model.state_dict(),
        device=mesh_device,
        hidden_dim_padding=hidden_dim_padding,
        out_channels=embedding_dim,
        parallel_config=dit_parallel_config,
        dtype=ttnn_dtype,
        height=height,
        width=width,
    )

    torch_input_tensor = torch.randn((batch_size, in_channels, height, width), dtype=torch_dtype)

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    seq_parallel_shard_dim = 1  # 1 is height
    tt_input_tensor = from_torch_fast_2d(
        torch_input_tensor.permute([0, 2, 3, 1]),  # BCYX -> BYXC
        mesh_device=mesh_device,
        mesh_shape=dit_parallel_config.cfg_parallel.mesh_shape,
        dims=[seq_parallel_shard_dim, None],
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_output = sd_patch_embed(tt_input_tensor, parameters, parallel_config=dit_parallel_config)
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device,
            mesh_shape=tuple(mesh_device.shape),
            dims=[-2, -1],
        ),
    )
    tt_output_torch = tt_output_torch.squeeze(1)[:batch_size, :, :embedding_dim]

    assert_quality(torch_output, tt_output_torch, pcc=0.999_990, shard_dim=0, num_devices=mesh_device.get_num_devices())
