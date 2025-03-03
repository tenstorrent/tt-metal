# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

import os
import pytest
import torch
import ttnn

from ..reference import SD3Transformer2DModel
from ..tt.attention import TtAttention, TtAttentionParameters
from ..tt.utils import from_torch, assert_quality

if TYPE_CHECKING:
    from ..reference.attention import Attention


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
        #        ("medium", 0, 1, 4096, 333),
        #        ("medium", 23, 1, 4096, 333),
        ("large", 0, 1, 4096, 333),
        #        ("large", 23, 1, 4096, 333),
    ],
)
@pytest.mark.parametrize(
    "joint_attention",
    [
        #        False,
        True,
    ],
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 517120}], indirect=True)
def test_attention(
    *,
    mesh_device: ttnn.MeshDevice,
    model_name,
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
    if model_name == "medium":
        embedding_dim = 1536
    else:
        embedding_dim = 2432

    torch_model: Attention = parent_torch_model.transformer_blocks[block_index].attn
    torch_model.eval()

    parameters = TtAttentionParameters.from_torch(torch_model.state_dict(), device=mesh_device, dtype=ttnn_dtype)
    tt_model = TtAttention(parameters, num_heads=torch_model.num_heads)

    torch.manual_seed(0)
    spatial = torch.randn((batch_size, spatial_sequence_length, embedding_dim), dtype=torch_dtype)
    prompt = (
        torch.randn((batch_size, prompt_sequence_length, embedding_dim), dtype=torch_dtype) if joint_attention else None
    )

    tt_spatial = from_torch(spatial, dtype=ttnn_dtype, mesh_device=mesh_device)
    tt_prompt = from_torch(prompt, dtype=ttnn_dtype, mesh_device=mesh_device) if joint_attention else None

    with torch.no_grad():
        spatial_output, prompt_output = torch_model(spatial=spatial, prompt=prompt)

    # tt_spatial = allocate_tensor_on_device_like(tt_spatial_host, device=device)
    # tt_prompt = allocate_tensor_on_device_like(tt_prompt_host, device=device) if joint_attention else None

    # ttnn.copy_host_to_device_tensor(tt_spatial_host, tt_spatial)
    # if joint_attention:
    #     ttnn.copy_host_to_device_tensor(tt_prompt_host, tt_prompt)
    tt_spatial_output, tt_prompt_output = tt_model(spatial=tt_spatial, prompt=tt_prompt)

    assert_quality(spatial_output, tt_spatial_output, pcc=0.990, shard_dim=0)

    if joint_attention:
        assert_quality(prompt_output, tt_prompt_output, pcc=0.990, shard_dim=0)
