# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
from typing import TYPE_CHECKING

import pytest
import torch
import ttnn

from ..reference import SD3Transformer2DModel
from ..tt.attention import TtAttention, TtAttentionParameters
from ..tt.utils import assert_quality, from_torch_fast

if TYPE_CHECKING:
    from ..reference.attention import Attention

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
    torch_dtype = torch.float32
    ttnn_dtype = ttnn.bfloat16

    parent_torch_model = SD3Transformer2DModel.from_pretrained(
        f"stabilityai/stable-diffusion-3.5-{model_name}", subfolder="transformer", torch_dtype=torch_dtype
    )
    embedding_dim = 1536 if model_name == "medium" else 2432

    torch_model: Attention = parent_torch_model.transformer_blocks[block_index].attn
    torch_model.eval()

    num_devices = mesh_device.get_num_devices()
    ## heads padding for T3K TP
    pad_embedding_dim = False
    if os.environ.get("MESH_DEVICE") == "T3K" and embedding_dim == 2432:
        pad_embedding_dim = True
        hidden_dim_padding = (
            ((embedding_dim // num_devices // TILE_SIZE) + 1) * TILE_SIZE
        ) * num_devices - embedding_dim
        num_heads = 40
    else:
        hidden_dim_padding = 0
        num_heads = torch_model.num_heads

    parameters = TtAttentionParameters.from_torch(
        torch_model.state_dict(),
        num_heads=num_heads,
        unpadded_num_heads=torch_model.num_heads,
        hidden_dim_padding=hidden_dim_padding,
        device=mesh_device,
        dtype=ttnn_dtype,
    )

    tt_model = TtAttention(parameters, num_heads=num_heads, device=mesh_device)

    torch.manual_seed(0)
    spatial = torch.randn((batch_size, spatial_sequence_length, embedding_dim), dtype=torch_dtype)
    prompt = (
        torch.randn((batch_size, prompt_sequence_length, embedding_dim), dtype=torch_dtype) if joint_attention else None
    )

    spatial_padded_4d = spatial.unsqueeze(1)
    if pad_embedding_dim:
        spatial_padded_4d = torch.nn.functional.pad(
            spatial_padded_4d, pad=(0, hidden_dim_padding), mode="constant", value=0
        )
    tt_spatial = from_torch_fast(spatial_padded_4d, dtype=ttnn_dtype, device=mesh_device, layout=ttnn.TILE_LAYOUT)

    if joint_attention:
        prompt_padded_4d = prompt.unsqueeze(1)
        if pad_embedding_dim:
            prompt_padded_4d = torch.nn.functional.pad(
                prompt_padded_4d, pad=(0, hidden_dim_padding), mode="constant", value=0
            )
        tt_prompt = from_torch_fast(prompt_padded_4d, dtype=ttnn_dtype, device=mesh_device, layout=ttnn.TILE_LAYOUT)
    else:
        tt_prompt = None

    with torch.no_grad():
        spatial_output, prompt_output = torch_model(spatial=spatial, prompt=prompt)

    # if joint_attention:
    tt_spatial_output, tt_prompt_output = tt_model(
        spatial=tt_spatial, prompt=tt_prompt, N=spatial_sequence_length, L=prompt_sequence_length
    )

    tt_spatial_output_torch = ttnn.to_torch(
        tt_spatial_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)
    )[:, :, 0:spatial_sequence_length, :embedding_dim]

    assert_quality(spatial_output, tt_spatial_output_torch, pcc=0.9995, mse=5e-6)

    if joint_attention:
        tt_prompt_output_torch = ttnn.to_torch(
            tt_prompt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)
        )[:, :, 0:prompt_sequence_length, :embedding_dim]

        assert_quality(prompt_output, tt_prompt_output_torch, pcc=0.9995, mse=7e-6)
