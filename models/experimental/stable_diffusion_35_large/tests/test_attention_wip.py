# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

import pytest
import torch
import tracy
import ttnn

from ..reference import SD3Transformer2DModel
from ..tt.attention import TtAttention, TtAttentionParameters
from ..tt.utils import allocate_tensor_on_device_like

if TYPE_CHECKING:
    from ..reference.attention import Attention


@pytest.mark.parametrize(
    ("block_index", "batch_size", "spatial_sequence_length", "prompt_sequence_length"),
    [
        (0, 2, 4096, 333),
    ],
)
@pytest.mark.parametrize("joint_attention", [False, True])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 517120}], indirect=True)
@pytest.mark.usefixtures("use_program_cache")
def test_attention(
    *,
    device: ttnn.Device,
    block_index: int,
    batch_size: int,
    spatial_sequence_length: int,
    prompt_sequence_length: int,
    joint_attention: bool,
) -> None:
    torch_dtype = torch.float32
    ttnn_dtype = ttnn.bfloat16

    parent_torch_model = SD3Transformer2DModel.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", subfolder="transformer", torch_dtype=torch_dtype
    )
    torch_model: Attention = parent_torch_model.transformer_blocks[block_index].attn
    torch_model.eval()

    parameters = TtAttentionParameters.from_torch(torch_model.state_dict(), device=device, dtype=ttnn.bfloat8_b)
    tt_model = TtAttention(parameters, num_heads=torch_model.num_heads)

    torch.manual_seed(0)
    spatial = torch.randn((batch_size, spatial_sequence_length, 1536), dtype=torch_dtype)
    prompt = torch.randn((batch_size, prompt_sequence_length, 1536), dtype=torch_dtype) if joint_attention else None

    tt_spatial_host = ttnn.from_torch(spatial, layout=ttnn.TILE_LAYOUT, dtype=ttnn_dtype)
    tt_prompt_host = ttnn.from_torch(prompt, layout=ttnn.TILE_LAYOUT, dtype=ttnn_dtype) if joint_attention else None

    with torch.no_grad():
        spatial_output, prompt_output = torch_model(spatial=spatial, prompt=prompt)

    tt_spatial = allocate_tensor_on_device_like(tt_spatial_host, device=device)
    tt_prompt = allocate_tensor_on_device_like(tt_prompt_host, device=device) if joint_attention else None

    # cache
    tracy.signpost("start")
    tt_model(spatial=tt_spatial, prompt=tt_prompt)
