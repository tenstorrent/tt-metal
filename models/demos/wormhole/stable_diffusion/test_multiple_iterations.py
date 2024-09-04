# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import json
import torch
import pytest
import numpy as np
from PIL import Image
from loguru import logger
from tqdm.auto import tqdm
from datasets import load_dataset

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    LMSDiscreteScheduler,
)
from models.utility_functions import (
    comp_allclose_and_pcc,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
)
from models.utility_functions import skip_for_wormhole_b0
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.wormhole.stable_diffusion.custom_preprocessing import custom_preprocessor
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_unet_2d_condition_model import (
    UNet2DConditionModel as UNet2D,
)

from torchvision.transforms import ToTensor


def load_inputs(input_path):
    with open(input_path) as f:
        input_data = json.load(f)
        assert input_data, "Input data is empty."
        prompt = [item["prompt"] for item in input_data]
        return prompt


def constant_prop_time_embeddings(timesteps, sample, time_proj):
    timesteps = timesteps[None]
    timesteps = timesteps.expand(sample.shape[0])
    t_emb = time_proj(timesteps)
    return t_emb


def save_image_and_latents(latents, iter, vae, pre_fix="", pre_fix2=""):
    pre_fix = "" if pre_fix == "" else f"{pre_fix}_"
    pre_fix2 = "" if pre_fix2 == "" else f"{pre_fix2}_"
    _latents = 1 / 0.18215 * latents

    with torch.no_grad():
        image = vae.decode(_latents).sample
    # Image post-processing
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images][0]
    pil_images.save(f"{pre_fix}{pre_fix2}image_iter_{iter}.png")


def guide(noise_pred, guidance_scale, t):  # will return latents
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return noise_pred


def latent_expansion(latents, scheduler, t):
    latent_model_input = torch.cat([latents] * 2, dim=0)
    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
    return latent_model_input


def calculate_fid_score(imgs_path1, imgs_path2):
    fid = FrechetInceptionDistance(normalize=True)
    fid.update(imgs_path1, real=False)
    fid.update(imgs_path2, real=True)
    return fid.compute()


def preprocess_images(image_paths):
    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
        image = image.convert("RGB")
        image = image.resize((299, 299))
        image = ToTensor()(image)
        images.append(image)
    return torch.stack(images)


def run_demo_inference_diffusiondb(device, reset_seeds, input_path, num_inference_steps, image_size):
    disable_persistent_kernel_cache()

    height, width = image_size

    experiment_name = f"diffusiondb_{height}x{width}"
    input_prompt = [
        "oil painting frame of Breathtaking mountain range with a clear river running through it, surrounded by tall trees and misty clouds, serene, peaceful, mountain landscape, high detail"
    ]
    logger.info(f"input_prompts: {input_prompt}")

    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

    # 4. load the K-LMS scheduler with some fitting parameters.
    ttnn_scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    torch_device = "cpu"
    vae.to(torch_device)
    text_encoder.to(torch_device)
    unet.to(torch_device)

    guidance_scale = 7.5  # Scale for classifier-free guidance
    generator = torch.manual_seed(174)  # 10233 Seed generator to create the inital latent noise
    batch_size = len(input_prompt)

    ## First, we get the text_embeddings for the prompt. These embeddings will be used to condition the UNet model.
    # Tokenizer and Text Encoder
    text_input = tokenizer(
        input_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

    # For classifier-free guidance, we need to do two forward passes: one with the conditioned input (text_embeddings),
    # and another with the unconditional embeddings (uncond_embeddings).
    # In practice, we can concatenate both into a single batch to avoid doing two forward passes.
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    ttnn_text_embeddings = ttnn.from_torch(text_embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    # Initial random noise
    latents = torch.randn(
        (batch_size, unet.config.in_channels, height // vae_scale_factor, width // vae_scale_factor),
        generator=generator,
    )
    latents = latents.to(torch_device)

    ttnn_scheduler.set_timesteps(num_inference_steps)

    latents = latents * ttnn_scheduler.init_noise_sigma
    ttnn_latents = torch.tensor(latents)

    iter = 0
    config = unet.config

    parameters = preprocess_model_parameters(
        initialize_model=lambda: unet, custom_preprocessor=custom_preprocessor, device=device
    )
    input_height = 64
    input_width = 64
    reader_patterns_cache = {} if height == 512 and width == 512 else None

    model = UNet2D(device, parameters, 2, input_height, input_width, reader_patterns_cache)
    # # Denoising loop
    for t in tqdm(ttnn_scheduler.timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        ttnn_latent_model_input = latent_expansion(ttnn_latents, ttnn_scheduler, t)
        ttnn_latent_model_input = ttnn.from_torch(
            ttnn_latent_model_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        _t = constant_prop_time_embeddings(t, ttnn_latent_model_input, unet.time_proj)
        _t = _t.unsqueeze(0).unsqueeze(0)
        _t = ttnn.from_torch(_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # predict the noise residual
        with torch.no_grad():
            ttnn_output = model(
                ttnn_latent_model_input,  # input
                timestep=_t,
                encoder_hidden_states=ttnn_text_embeddings,
                class_labels=None,
                attention_mask=None,
                cross_attention_kwargs=None,
                return_dict=True,
                config=config,
            )
            noise_pred = ttnn.to_torch(ttnn_output)

        # perform guidance
        noise_pred = guide(noise_pred, guidance_scale, t)

        ttnn_latents = ttnn_scheduler.step(noise_pred, t, ttnn_latents).prev_sample
        save_image_and_latents(ttnn_latents, iter, vae, pre_fix=f"{experiment_name}_tt", pre_fix2="")

        iter += 1
        enable_persistent_kernel_cache()

    latents = ttnn_latents
    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample

    # Image post-processing
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images][0]
    ttnn_output_path = f"{experiment_name}_ttnn.png"
    pil_images.save(ttnn_output_path)

    ref_paths = [ref_img_path, ref_img_path]
    ttnn_paths = [ttnn_output_path, ttnn_output_path]

    ref_images = preprocess_images(ref_paths)
    ttnn_images = preprocess_images(ttnn_paths)


def test_tt2_multiple_iteration(device, reset_seeds, input_path):
    # 30 iterations, generate 512x512 image
    return run_demo_inference_diffusiondb(device, reset_seeds, input_path, 30, (512, 512))
