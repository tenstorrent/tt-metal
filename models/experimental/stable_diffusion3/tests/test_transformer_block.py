# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

import pytest
import torch
import ttnn

from ..reference import SD3Transformer2DModel
from ..tt.transformer_block import TtTransformerBlock, TtTransformerBlockParameters
from ..tt.utils import allocate_tensor_on_device_like, assert_quality

if TYPE_CHECKING:
    from ..reference.transformer_block import TransformerBlock


@pytest.mark.parametrize(
    ("block_index", "batch_size", "spatial_sequence_length", "prompt_sequence_length"),
    [
        (0, 2, 4096, 333),
        (23, 2, 4096, 333),
    ],
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 716800}], indirect=True)
@pytest.mark.usefixtures("use_program_cache")
def test_transformer_block(
    *,
    device: ttnn.Device,
    block_index: int,
    batch_size: int,
    spatial_sequence_length: int,
    prompt_sequence_length: int,
) -> None:
    parent_torch_model = SD3Transformer2DModel.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", subfolder="transformer"
    )
    torch_model: TransformerBlock = parent_torch_model.transformer_blocks[block_index]
    torch_model.eval()

    parameters = TtTransformerBlockParameters.from_torch(torch_model.state_dict(), device=device, dtype=ttnn.bfloat8_b)
    tt_model = TtTransformerBlock(parameters, num_heads=torch_model.num_heads)

    embedding_dim = 1536

    torch.manual_seed(0)
    spatial = torch.randn((batch_size, spatial_sequence_length, embedding_dim))
    prompt = torch.randn((batch_size, prompt_sequence_length, embedding_dim))
    time = torch.randn((batch_size, embedding_dim))

    tt_spatial_host = ttnn.from_torch(spatial, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_prompt_host = ttnn.from_torch(prompt, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    tt_time_host = ttnn.from_torch(time.unsqueeze(1), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    with torch.no_grad():
        spatial_output, prompt_output = torch_model(spatial=spatial, prompt=prompt, time_embed=time)

    tt_spatial = allocate_tensor_on_device_like(tt_spatial_host, device=device)
    tt_prompt = allocate_tensor_on_device_like(tt_prompt_host, device=device)
    tt_time = allocate_tensor_on_device_like(tt_time_host, device=device)

    # cache
    tt_model(spatial=tt_spatial, prompt=tt_prompt, time_embed=tt_time)

    # trace
    tid = ttnn.begin_trace_capture(device)
    tt_spatial_output, tt_prompt_output = tt_model(spatial=tt_spatial, prompt=tt_prompt, time_embed=tt_time)
    ttnn.end_trace_capture(device, tid)

    # execute
    ttnn.copy_host_to_device_tensor(tt_spatial_host, tt_spatial)
    ttnn.copy_host_to_device_tensor(tt_prompt_host, tt_prompt)
    ttnn.copy_host_to_device_tensor(tt_time_host, tt_time)
    ttnn.execute_trace(device, tid)

    assert (prompt_output is None) == (tt_prompt_output is None)

    if prompt_output is not None and tt_prompt_output is not None:
        assert_quality(prompt_output, tt_prompt_output, pcc=0.995)

    assert_quality(spatial_output, tt_spatial_output, pcc=0.995)
