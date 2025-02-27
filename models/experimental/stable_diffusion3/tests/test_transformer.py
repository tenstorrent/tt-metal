# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from ..reference.transformer import SD3Transformer2DModel
from ..tt.transformer import TtSD3Transformer2DModel, TtSD3Transformer2DModelParameters
from ..tt.utils import allocate_tensor_on_device_like, assert_quality


@pytest.mark.parametrize(
    ("batch_size", "prompt_sequence_length", "height", "width"),
    [
        (2, 333, 1024, 1024),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192, "trace_region_size": 15157248}], indirect=True)
@pytest.mark.usefixtures("use_program_cache")
def test_transformer(
    *,
    device: ttnn.Device,
    batch_size: int,
    prompt_sequence_length: int,
    height: int,
    width: int,
) -> None:
    torch_model = SD3Transformer2DModel.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", subfolder="transformer"
    )
    torch_model.eval()

    parameters = TtSD3Transformer2DModelParameters.from_torch(
        torch_model.state_dict(), device=device, dtype=ttnn.bfloat8_b
    )
    tt_model = TtSD3Transformer2DModel(parameters, num_attention_heads=torch_model.config.num_attention_heads)

    torch.manual_seed(0)
    spatial = torch.randn((batch_size, 16, height // 8, width // 8))
    prompt = torch.randn((batch_size, prompt_sequence_length, 4096))
    pooled_projection = torch.randn((batch_size, 2048))
    timestep = torch.randint(1000, (batch_size,), dtype=torch.float32)

    tt_spatial_host = ttnn.from_torch(
        spatial.permute([0, 2, 3, 1]), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )  # BCYX -> BYXC
    tt_prompt_host = ttnn.from_torch(prompt, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    tt_pooled_projection_host = ttnn.from_torch(pooled_projection, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_timestep_host = ttnn.from_torch(timestep.unsqueeze(1), layout=ttnn.TILE_LAYOUT)

    with torch.no_grad():
        torch_output = torch_model(
            spatial=spatial, prompt_embed=prompt, pooled_projections=pooled_projection, timestep=timestep
        )

    tt_spatial = allocate_tensor_on_device_like(tt_spatial_host, device=device)
    tt_prompt = allocate_tensor_on_device_like(tt_prompt_host, device=device)
    tt_pooled_projection = allocate_tensor_on_device_like(tt_pooled_projection_host, device=device)
    tt_timestep = allocate_tensor_on_device_like(tt_timestep_host, device=device)

    trace = tt_model.cache_and_trace(
        spatial=tt_spatial, prompt=tt_prompt, pooled_projection=tt_pooled_projection, timestep=tt_timestep
    )
    tt_output = trace(
        spatial=tt_spatial_host,
        prompt=tt_prompt_host,
        pooled_projection=tt_pooled_projection_host,
        timestep=tt_timestep_host,
    )

    assert_quality(torch_output, tt_output, mse=0.1, pcc=0.999_500)
