# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
from typing import TYPE_CHECKING

import pytest
import torch
import ttnn

from ..reference import SD3Transformer2DModel
from ..tt.attention import TtAttention, TtAttentionParameters
from ..tt.utils import assert_quality, from_torch

if TYPE_CHECKING:
    from ..reference.attention import Attention

TILE_SIZE = 32


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
    ("model_name", "block_index", "batch_size", "spatial_sequence_length", "prompt_sequence_length"),
    [
        # ("medium", 0, 1, 4096, 333),
        # ("medium", 23, 1, 4096, 333),
        ("large", 0, 1, 4096, 333),
        # ("large", 23, 1, 4096, 333),
    ],
)
@pytest.mark.parametrize("joint_attention", [True])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 517120}], indirect=True)
def test_attention(
    *,
    mesh_device: ttnn.MeshDevice,
    model_name: str,
    block_index: int,
    batch_size: int,
    spatial_sequence_length: int,
    prompt_sequence_length: int,
    joint_attention: bool,
) -> None:
    mesh_device.enable_async(True)
    torch_dtype = torch.float32
    ttnn_dtype = ttnn.bfloat16

    parent_torch_model = SD3Transformer2DModel.from_pretrained(
        f"stabilityai/stable-diffusion-3.5-{model_name}", subfolder="transformer", torch_dtype=torch_dtype
    )
    embedding_dim = 1536 if model_name == "medium" else 2432

    torch_model: Attention = parent_torch_model.transformer_blocks[block_index].attn
    torch_model.eval()

    parameters = TtAttentionParameters.from_torch(
        torch_model.state_dict(), num_heads=torch_model.num_heads, device=mesh_device, dtype=ttnn_dtype
    )

    ## heads padding for T3K TP
    pad_40_heads = 0
    if os.environ["FAKE_DEVICE"] == "T3K" and embedding_dim == 2432:
        pad_40_heads = 1
        embedding_dim_padding = 128
        num_heads = 40
    else:
        num_heads = torch_model.num_heads

    tt_model = TtAttention(parameters, num_heads=num_heads, device=mesh_device)

    torch.manual_seed(0)
    spatial = torch.randn((batch_size, spatial_sequence_length, embedding_dim), dtype=torch_dtype)
    prompt = (
        torch.randn((batch_size, prompt_sequence_length, embedding_dim), dtype=torch_dtype) if joint_attention else None
    )

    spatial_extra = spatial_sequence_length % TILE_SIZE
    spatial_padding = TILE_SIZE - spatial_extra if spatial_extra > 0 else 0
    spatial_padded_4d = torch.nn.functional.pad(
        spatial.unsqueeze(1), pad=(0, 0, 0, spatial_padding), mode="constant", value=0
    )
    if pad_40_heads:
        spatial_padded_4d = torch.nn.functional.pad(
            spatial_padded_4d, pad=(0, embedding_dim_padding), mode="constant", value=0
        )
    tt_spatial = from_torch(spatial_padded_4d, dtype=ttnn_dtype, mesh_device=mesh_device, layout=ttnn.TILE_LAYOUT)

    if joint_attention:
        prompt_extra = prompt_sequence_length % TILE_SIZE
        prompt_padding = TILE_SIZE - prompt_extra if prompt_extra > 0 else 0
        prompt_padded_4d = torch.nn.functional.pad(
            prompt.unsqueeze(1), pad=(0, 0, 0, prompt_padding), mode="constant", value=0
        )
        if pad_40_heads:
            prompt_padded_4d = torch.nn.functional.pad(
                prompt_padded_4d, pad=(0, embedding_dim_padding), mode="constant", value=0
            )
        tt_prompt = from_torch(prompt_padded_4d, dtype=ttnn_dtype, mesh_device=mesh_device, layout=ttnn.TILE_LAYOUT)
    else:
        tt_prompt = None

    with torch.no_grad():
        spatial_output, prompt_output = torch_model(spatial=spatial, prompt=prompt)

    # tt_spatial = allocate_tensor_on_device_like(tt_spatial_host, device=device)
    # tt_prompt = allocate_tensor_on_device_like(tt_prompt_host, device=device) if joint_attention else None

    # ttnn.copy_host_to_device_tensor(tt_spatial_host, tt_spatial)
    # if joint_attention:
    #     ttnn.copy_host_to_device_tensor(tt_prompt_host, tt_prompt)
    tt_spatial_output, tt_prompt_output = tt_model(
        spatial=tt_spatial, prompt=tt_prompt, N=spatial_sequence_length, L=prompt_sequence_length
    )

    tt_spatial_output_torch = ttnn.to_torch(
        tt_spatial_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)
    )[:, :, 0:spatial_sequence_length, :embedding_dim]
    assert_quality(
        spatial_output, tt_spatial_output_torch, pcc=0.990, shard_dim=0, num_devices=mesh_device.get_num_devices()
    )

    if joint_attention:
        tt_prompt_output_torch = ttnn.to_torch(
            tt_prompt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)
        )[:, :, 0:prompt_sequence_length, :embedding_dim]
        assert_quality(
            prompt_output, tt_prompt_output_torch, pcc=0.990, shard_dim=0, num_devices=mesh_device.get_num_devices()
        )
