# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from diffusers import AutoPipelineForText2Image
from models.demos.stable_diffusion.tt.utils import custom_preprocessor
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.demos.stable_diffusion.tt.resnetblock2d_utils import update_params
from models.demos.stable_diffusion.tt.ttnn_optimized_sdxl_turbo import stable_diffusion_xl_turbo


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_sd(device, reset_seeds):
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo")
    model = pipe.unet
    model.eval()
    config = model.config
    input_data = torch.randn(1, 4, 128, 128)
    timestep = torch.tensor([999.0])
    encoder_hidden_states = torch.randn(1, 77, 2048)
    text_embeds = torch.randn(1, 1280)
    time_ids = torch.randn(1, 6)
    added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters = update_params(parameters)

    torch_output = model(
        sample=input_data,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        added_cond_kwargs=added_cond_kwargs,
    )

    tt_input_data = ttnn.from_torch(input_data, dtype=ttnn.bfloat16, device=device)
    tt_timestep = ttnn.from_torch(timestep, dtype=ttnn.bfloat16, device=device)
    tt_encoder_hidden_states = ttnn.from_torch(encoder_hidden_states, dtype=ttnn.bfloat16, device=device)
    tt_text_embeds = ttnn.from_torch(text_embeds, dtype=ttnn.float32, device=device)
    tt_time_ids = ttnn.from_torch(time_ids, dtype=ttnn.float32, device=device)

    tt_output = stable_diffusion_xl_turbo(
        config,
        tt_input_data,
        tt_timestep,
        tt_encoder_hidden_states,
        tt_text_embeds,
        tt_time_ids,
        parameters=parameters,
        device=device,
    )

    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output[0], tt_output, 0.98)
