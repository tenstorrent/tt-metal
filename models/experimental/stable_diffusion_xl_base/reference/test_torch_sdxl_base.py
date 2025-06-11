# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from diffusers import DiffusionPipeline
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
import torch
import pytest


# Wrapper UNet2DCondition to slice it over a batch dimension
class UNet2DConditionSliceWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(
        self,
        latent,
        timestep,
        encoder_hidden_states,
        timestep_cond,
        cross_attention_kwargs,
        added_cond_kwargs,
        return_dict,
    ):
        assert latent.shape[0] == 2, "Batch size must be 2 for this wrapper"

        # predict the noise residual
        added_cond_kwargs_1 = {
            "text_embeds": added_cond_kwargs["text_embeds"][0:1,],
            "time_ids": added_cond_kwargs["time_ids"][0:1,],
        }
        added_cond_kwargs_2 = {
            "text_embeds": added_cond_kwargs["text_embeds"][1:2,],
            "time_ids": added_cond_kwargs["time_ids"][1:2,],
        }

        if "image_embeds" in added_cond_kwargs.keys():
            if added_cond_kwargs["image_embeds"] is not None:
                added_cond_kwargs_1 = added_cond_kwargs["image_embeds"][0:1,]
                added_cond_kwargs_2 = added_cond_kwargs["image_embeds"][1:2,]

        latent_1, latent_2 = latent[0:1,], latent[1:2,]
        encoder_hidden_states_1, encoder_hidden_states_2 = encoder_hidden_states[0:1], encoder_hidden_states[1:2]

        unet_output_1 = self.unet(
            latent_1,
            timestep,
            encoder_hidden_states=encoder_hidden_states_1,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs_1,
            return_dict=return_dict,
        )[0]
        unet_output_2 = self.unet(
            latent_2,
            timestep,
            encoder_hidden_states=encoder_hidden_states_2,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs_2,
            return_dict=return_dict,
        )[0]
        noise_pred = torch.cat([unet_output_1, unet_output_2], dim=0)

        if not return_dict:
            return (noise_pred,)

        return UNet2DConditionOutput(sample=noise_pred)

    @property
    def config(self):
        return self.unet.config

    @property
    def dtype(self):
        return self.unet.dtype

    @property
    def encoder_hid_proj(self):
        return self.unet.encoder_hid_proj

    @property
    def add_embedding(self):
        return self.unet.add_embedding

    @property
    def _execution_device(self):
        return self.unet._execution_device


@pytest.mark.parametrize("prompt", ["Red Apple over black background"])
@pytest.mark.parametrize("repeat", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("num_inference_steps", [15])
def test_diffusion_pipeline(is_ci_env, prompt, repeat, num_inference_steps):
    if is_ci_env:
        pytest.skip("Skipping test in CI environment")

    torch.manual_seed(0)

    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16, use_safetensors=True
    )
    pipe.unet = UNet2DConditionSliceWrapper(pipe.unet)

    images = pipe(prompt=prompt, num_inference_steps=num_inference_steps).images
    hashed_prompt = hash(prompt)
    print(f"Hashed prompt: {hashed_prompt}")
    assert images is not None
    images[0].save("sliced_test_output_image" + str(hash(prompt)) + str(repeat) + ".png")
