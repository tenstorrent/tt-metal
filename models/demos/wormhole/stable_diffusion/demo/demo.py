# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import os
import time

import numpy as np
import pytest
import torch
from datasets import load_dataset
from diffusers import AutoencoderKL, UNet2DConditionModel
from loguru import logger
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.wormhole.stable_diffusion.custom_preprocessing import custom_preprocessor
from models.demos.wormhole.stable_diffusion.sd_pndm_scheduler import TtPNDMScheduler
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_unet_2d_condition_model_new_conv import (
    UNet2DConditionModel as UNet2D,
)
from models.utility_functions import enable_persistent_kernel_cache, profiler


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


def tt_guide(noise_pred, guidance_scale):  # will return latents
    noise_pred_uncond = noise_pred[:1, :, :, :]
    noise_pred_text = ttnn.slice(
        noise_pred,
        [1, 0, 0, 0],
        [
            noise_pred.shape[0],
            noise_pred.shape[1],
            noise_pred.shape[2],
            noise_pred.shape[3],
        ],
    )
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return noise_pred


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


def run_demo_inference(device, reset_seeds, input_path, num_prompts, num_inference_steps, image_size=(256, 256)):
    enable_persistent_kernel_cache()
    device.enable_program_cache()
    profiler.clear()

    # Until di/dt issues are resolved
    os.environ["SLOW_MATMULS"] = "1"
    assert (
        num_inference_steps >= 4
    ), f"PNDMScheduler only supports num_inference_steps >= 4. Found num_inference_steps={num_inference_steps}"

    height, width = image_size

    torch_device = "cpu"
    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae.to(torch_device)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

    # 4. load the K-LMS scheduler with some fitting parameters.
    ttnn_scheduler = TtPNDMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        skip_prk_steps=True,
        steps_offset=1,
        device=device,
    )

    text_encoder.to(torch_device)
    unet.to(torch_device)

    config = unet.config
    parameters = preprocess_model_parameters(
        initialize_model=lambda: unet, custom_preprocessor=custom_preprocessor, device=device
    )
    input_height = 64
    input_width = 64
    model = UNet2D(device, parameters, 2, input_height, input_width)

    guidance_scale = 7.5  # Scale for classifier-free guidance
    generator = torch.manual_seed(174)  # 10233 Seed generator to create the inital latent noise
    batch_size = 1

    # Initial random noise
    latents = torch.randn(
        (batch_size, unet.config.in_channels, height // vae_scale_factor, width // vae_scale_factor),
        generator=generator,
    )
    latents = latents.to(torch_device)

    ttnn_scheduler.set_timesteps(num_inference_steps)

    latents = latents * ttnn_scheduler.init_noise_sigma
    rand_latents = torch.tensor(latents)
    rand_latents = ttnn.from_torch(rand_latents, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # ttnn_latents = ttnn.from_torch(ttnn_latents, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    ttnn_latent_model_input = ttnn.concat([rand_latents, rand_latents], dim=0)
    _tlist = []
    for t in ttnn_scheduler.timesteps:
        _t = constant_prop_time_embeddings(t, ttnn_latent_model_input, unet.time_proj)
        _t = _t.unsqueeze(0).unsqueeze(0)
        _t = _t.permute(2, 0, 1, 3)  # pre-permute temb
        _t = ttnn.from_torch(_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        _tlist.append(_t)

    time_step = ttnn_scheduler.timesteps.tolist()
    i = 0

    inputs = load_inputs(input_path)
    input_prompts = inputs[:num_prompts]

    while i < num_prompts:
        ttnn_scheduler.set_timesteps(num_inference_steps)
        input_prompt = [input_prompts[i]]
        i = i + 1

        experiment_name = f"input_data_{i}_{height}x{width}"
        logger.info(f"input prompt : {input_prompt}")
        batch_size = len(input_prompt)

        profiler.start(f"inference_prompt_{i}")
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
        ttnn_text_embeddings = torch.nn.functional.pad(text_embeddings, (0, 0, 0, 19))
        ttnn_text_embeddings = ttnn.from_torch(
            ttnn_text_embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        iter = 0
        ttnn_latents = rand_latents
        # # Denoising loop
        for index in tqdm(range(len(time_step))):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            ttnn_latent_model_input = ttnn.concat([ttnn_latents, ttnn_latents], dim=0)
            _t = _tlist[index]
            t = time_step[index]
            # predict the noise residual
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
            # perform guidance
            noise_pred = tt_guide(ttnn_output, guidance_scale)

            ttnn_latents = ttnn_scheduler.step(noise_pred, t, ttnn_latents).prev_sample

            iter += 1

        latents = ttnn.to_torch(ttnn_latents).to(torch.float32)

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = vae.decode(latents).sample
        profiler.end(f"inference_prompt_{i}")

        # Image post-processing
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images][0]
        ttnn_output_path = f"{experiment_name}_ttnn.png"
        pil_images.save(ttnn_output_path)

    profiler.print()

    # we calculate average time per prompt only when there is more than 1 iteration,
    # since first iteration includes compile time
    if num_prompts > 1:
        # skip first for compile
        total_time = sum([profiler.get("inference_prompt_" + str(i)) for i in range(2, num_prompts + 1)])
        avg_time = total_time / (num_prompts - 1)
        FPS = 1 / avg_time

        print(
            f"Average time per prompt: {avg_time}, FPS: {FPS}",
        )


def run_interactive_demo_inference(device, num_inference_steps, image_size=(256, 256)):
    enable_persistent_kernel_cache()
    device.enable_program_cache()

    # Until di/dt issues are resolved
    os.environ["SLOW_MATMULS"] = "1"
    assert (
        num_inference_steps >= 4
    ), f"PNDMScheduler only supports num_inference_steps >= 4. Found num_inference_steps={num_inference_steps}"

    height, width = image_size

    torch_device = "cpu"
    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae.to(torch_device)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

    # 4. load the K-LMS scheduler with some fitting parameters.
    ttnn_scheduler = TtPNDMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        skip_prk_steps=True,
        steps_offset=1,
        device=device,
    )

    text_encoder.to(torch_device)
    unet.to(torch_device)

    config = unet.config
    parameters = preprocess_model_parameters(
        initialize_model=lambda: unet, custom_preprocessor=custom_preprocessor, device=device
    )
    input_height = 64
    input_width = 64
    model = UNet2D(device, parameters, 2, input_height, input_width)

    guidance_scale = 7.5  # Scale for classifier-free guidance
    generator = torch.manual_seed(174)  # 10233 Seed generator to create the inital latent noise
    batch_size = 1

    # Initial random noise
    latents = torch.randn(
        (batch_size, unet.config.in_channels, height // vae_scale_factor, width // vae_scale_factor),
        generator=generator,
    )
    latents = latents.to(torch_device)

    ttnn_scheduler.set_timesteps(num_inference_steps)

    latents = latents * ttnn_scheduler.init_noise_sigma
    rand_latents = torch.tensor(latents)
    rand_latents = ttnn.from_torch(rand_latents, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # ttnn_latents = ttnn.from_torch(ttnn_latents, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    ttnn_latent_model_input = ttnn.concat([rand_latents, rand_latents], dim=0)
    _tlist = []
    for t in ttnn_scheduler.timesteps:
        _t = constant_prop_time_embeddings(t, ttnn_latent_model_input, unet.time_proj)
        _t = _t.unsqueeze(0).unsqueeze(0)
        _t = _t.permute(2, 0, 1, 3)  # pre-permute temb
        _t = ttnn.from_torch(_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        _tlist.append(_t)

    time_step = ttnn_scheduler.timesteps.tolist()

    while 1:
        ttnn_scheduler.set_timesteps(num_inference_steps)
        print("Enter the input promt, or q to exit:")
        new_prompt = input()
        if len(new_prompt) > 0:
            input_prompt = [new_prompt]
        if input_prompt[0] == "q":
            break

        experiment_name = f"interactive_{height}x{width}"
        logger.info(f"input prompt : {input_prompt}")
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
        ttnn_text_embeddings = torch.nn.functional.pad(text_embeddings, (0, 0, 0, 19))
        ttnn_text_embeddings = ttnn.from_torch(
            ttnn_text_embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        iter = 0
        ttnn_latents = rand_latents
        # # Denoising loop
        total_accum = 0
        for index in tqdm(range(len(time_step))):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            t0 = time.time()
            ttnn_latent_model_input = ttnn.concat([ttnn_latents, ttnn_latents], dim=0)
            _t = _tlist[index]
            t = time_step[index]
            # predict the noise residual
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
            # perform guidance
            noise_pred = tt_guide(ttnn_output, guidance_scale)

            ttnn_latents = ttnn_scheduler.step(noise_pred, t, ttnn_latents).prev_sample
            total_accum += time.time() - t0
            iter += 1
        print(f"Time taken for {iter} iterations: total: {total_accum:.3f}")

        latents = ttnn.to_torch(ttnn_latents).to(torch.float32)

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = vae.decode(latents).sample

        # Image post-processing
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images][0]
        ttnn_output_path = f"{experiment_name}_ttnn.png"
        pil_images.save(ttnn_output_path)


def run_demo_inference_diffusiondb(
    device, reset_seeds, input_path, num_prompts, num_inference_steps, image_size=(256, 256)
):
    enable_persistent_kernel_cache()
    device.enable_program_cache()

    # Until di/dt issues are resolved
    os.environ["SLOW_MATMULS"] = "1"

    assert (
        num_inference_steps >= 4
    ), f"PNDMScheduler only supports num_inference_steps >= 4. Found num_inference_steps={num_inference_steps}"
    # 0. Load a sample prompt from the dataset
    dataset = load_dataset("poloclub/diffusiondb", "2m_random_1k")
    data_1k = dataset["train"]

    height, width = image_size

    torch_device = "cpu"
    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae.to(torch_device)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

    # 4. load the K-LMS scheduler with some fitting parameters.
    ttnn_scheduler = TtPNDMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        skip_prk_steps=True,
        steps_offset=1,
        device=device,
    )

    text_encoder.to(torch_device)
    unet.to(torch_device)

    config = unet.config
    parameters = preprocess_model_parameters(
        initialize_model=lambda: unet, custom_preprocessor=custom_preprocessor, device=device
    )
    input_height = 64
    input_width = 64
    model = UNet2D(device, parameters, 2, input_height, input_width)

    guidance_scale = 7.5  # Scale for classifier-free guidance
    generator = torch.manual_seed(174)  # 10233 Seed generator to create the inital latent noise
    batch_size = 1

    # Initial random noise
    latents = torch.randn(
        (batch_size, unet.config.in_channels, height // vae_scale_factor, width // vae_scale_factor),
        generator=generator,
    )
    latents = latents.to(torch_device)

    ttnn_scheduler.set_timesteps(num_inference_steps)

    latents = latents * ttnn_scheduler.init_noise_sigma
    rand_latents = torch.tensor(latents)
    rand_latents = ttnn.from_torch(rand_latents, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # ttnn_latents = ttnn.from_torch(ttnn_latents, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    ttnn_latent_model_input = ttnn.concat([rand_latents, rand_latents], dim=0)
    _tlist = []
    for t in ttnn_scheduler.timesteps:
        _t = constant_prop_time_embeddings(t, ttnn_latent_model_input, unet.time_proj)
        _t = _t.unsqueeze(0).unsqueeze(0)
        _t = _t.permute(2, 0, 1, 3)  # pre-permute temb
        _t = ttnn.from_torch(_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        _tlist.append(_t)

    time_step = ttnn_scheduler.timesteps.tolist()

    i = 0
    while i < num_prompts:
        experiment_name = f"diffusiondb_{i}__{height}x{width}"
        ttnn_scheduler.set_timesteps(num_inference_steps)
        input_prompt = [f"{data_1k['prompt'][i]}"]

        image = np.array(data_1k["image"][i])
        ref_images = Image.fromarray(image)
        ref_img_path = f"{experiment_name}_ref.png"
        ref_images.save(ref_img_path)
        i = i + 1

        logger.info(f"input_prompts: {input_prompt}")

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
        ttnn_text_embeddings = torch.nn.functional.pad(text_embeddings, (0, 0, 0, 19))
        ttnn_text_embeddings = ttnn.from_torch(
            ttnn_text_embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        iter = 0
        ttnn_latents = rand_latents
        # # Denoising loop
        for index in tqdm(range(len(time_step))):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            ttnn_latent_model_input = ttnn.concat([ttnn_latents, ttnn_latents], dim=0)
            _t = _tlist[index]
            t = time_step[index]
            # predict the noise residual
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

            # perform guidance
            noise_pred = tt_guide(ttnn_output, guidance_scale)
            ttnn_latents = ttnn_scheduler.step(noise_pred, t, ttnn_latents).prev_sample

            iter += 1

        latents = ttnn.to_torch(ttnn_latents).to(torch.float32)
        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = vae.decode(latents).sample

        # Image post-processing
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images][0]
        ttnn_output_path = f"{experiment_name}_ttnn.png"
        pil_images.save(ttnn_output_path)

        ttnn_paths = [ttnn_output_path, ttnn_output_path]
        ttnn_images = preprocess_images(ttnn_paths)
        ref_paths = [ref_img_path, ref_img_path]
        ref_images = preprocess_images(ref_paths)

        # Calculate FID scores
        fid_score_ref_ttnn = calculate_fid_score(ref_images, ttnn_images)
        logger.info(f"FID Score (Reference vs TTNN): {fid_score_ref_ttnn}")

        # calculate Clip score
        clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")

        clip_score_ttnn = clip_score(ttnn_images[0], input_prompt)
        clip_score_ttnn = clip_score_ttnn.detach()
        logger.info(f"CLIP Score (TTNN): {clip_score_ttnn}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_path",
    (("models/demos/wormhole/stable_diffusion/demo/input_data.json"),),
    ids=["default_input"],
)
@pytest.mark.parametrize(
    "num_prompts",
    ((10),),
)
@pytest.mark.parametrize(
    "num_inference_steps",
    ((50),),
)
@pytest.mark.parametrize(
    "image_size",
    ((512, 512),),
)
def test_demo(device, reset_seeds, input_path, num_prompts, num_inference_steps, image_size):
    return run_demo_inference(device, reset_seeds, input_path, num_prompts, num_inference_steps, image_size)
