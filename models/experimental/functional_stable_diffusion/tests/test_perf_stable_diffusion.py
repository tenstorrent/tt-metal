# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import ttnn
import json
import torch
import pytest
import numpy as np
from PIL import Image
from loguru import logger
from tqdm.auto import tqdm
from datasets import load_dataset
from scipy import integrate

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
)
from models.utility_functions import (
    skip_for_grayskull,
)
from models.utility_functions import (
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
)
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.functional_stable_diffusion.sd_helper_funcs import TtLMSDiscreteScheduler
from models.experimental.functional_stable_diffusion.custom_preprocessing import custom_preprocessor
from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_unet_2d_condition_model import (
    UNet2DConditionModel as UNet2D,
)

from torchvision.transforms import ToTensor
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image.fid import FrechetInceptionDistance

from models.perf.perf_utils import prep_perf_report
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.utility_functions import profiler, enable_persistent_kernel_cache, skip_for_grayskull


def constant_prop_time_embeddings(timesteps, sample, time_proj):
    timesteps = timesteps[None]
    timesteps = timesteps.expand(sample.shape[0])
    t_emb = time_proj(timesteps)
    return t_emb


def tt_guide(noise_pred, guidance_scale):  # will return latents
    noise_pred_uncond, noise_pred_text = ttnn.split(noise_pred, noise_pred.shape[0] // 2, dim=0)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return noise_pred


def tt_latent_expansion(latents, scheduler, sigma, device):
    latent_model_input = ttnn.concat([latents, latents], dim=0)
    latent_model_input = scheduler.scale_model_input(latent_model_input, sigma, device)
    return latent_model_input


def get_lms_coefficient(order, t, current_order, sigmas):
    def lms_derivative(tau):
        prod = 1.0
        for k in range(order):
            if current_order == k:
                continue
            prod *= (tau - sigmas[t - k]) / (sigmas[t - current_order] - sigmas[t - k])
        return prod

    integrated_coeff = integrate.quad(lms_derivative, sigmas[t], sigmas[t + 1], epsrel=1e-4)[0]

    return integrated_coeff


@skip_for_grayskull()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_l1_small_size", [32768], indirect=True)
@pytest.mark.parametrize(
    "num_prompts, num_inference_steps, image_size, expected_compile_time, expected_inference_time",
    ((1, 2, (512, 512), 3600, 1.8),),
)
def test_stable_diffusion_perf(
    device, num_prompts, num_inference_steps, image_size, expected_compile_time, expected_inference_time
):
    disable_persistent_kernel_cache()

    # Clear global profiler state before starting measurements
    profiler.clear()

    # 0. Load a sample prompt from the dataset
    dataset = load_dataset("poloclub/diffusiondb", "2m_random_1k")
    data_1k = dataset["train"]
    height, width = image_size

    for i in range(num_prompts):
        experiment_name = f"diffusiondb_{i}__{height}x{width}"
        input_prompt = [f"{data_1k['prompt'][i]}"]
        logger.info(f"input_prompts: {input_prompt}")

        image = np.array(data_1k["image"][i])
        ref_images = Image.fromarray(image)
        ref_img_path = f"{experiment_name}_ref.png"
        ref_images.save(ref_img_path)

        # 1. Load the autoencoder model which will be used to decode the latents into image space.
        vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

        # 2. Load the tokenizer and text encoder to tokenize and encode the text.
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

        # 3. The UNet model for generating the latents.
        unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

        # 4. load the K-LMS scheduler with some fitting parameters.
        ttnn_scheduler = TtLMSDiscreteScheduler(
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
        ttnn_text_embeddings = torch.nn.functional.pad(text_embeddings, (0, 0, 0, 19))
        ttnn_text_embeddings = ttnn.from_torch(
            ttnn_text_embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

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
        ttnn_latents = ttnn.from_torch(ttnn_latents, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        config = unet.config
        parameters = preprocess_model_parameters(
            initialize_model=lambda: unet, custom_preprocessor=custom_preprocessor, device=device
        )
        input_height = 64
        input_width = 64
        reader_patterns_cache = {} if height == 512 and width == 512 else None

        time_step_list = []
        ttnn_sigma = []
        ttnn_step_index = []
        timesteps_bkp = ttnn.to_torch(ttnn_scheduler.timesteps)
        sigma_tensor = ttnn.to_torch(ttnn_scheduler.sigmas)[0]
        step_index = (timesteps_bkp[0] == timesteps_bkp[0][0]).nonzero().item()
        ttnn_latent_model_input = tt_latent_expansion(
            ttnn_latents, ttnn_scheduler, float(sigma_tensor[step_index]), device
        )

        for t in timesteps_bkp[0]:
            _t = constant_prop_time_embeddings(t, ttnn_latent_model_input, unet.time_proj)
            _t = _t.unsqueeze(0).unsqueeze(0)
            _t = _t.permute(2, 0, 1, 3)
            _t = ttnn.from_torch(_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            time_step_list.append(_t)
            step_index = (timesteps_bkp[0] == t).nonzero().item()
            ttnn_step_index.append(step_index)
            ttnn_sigma.append(sigma_tensor[step_index])

        orders = 4
        order_list = []
        ttnn_lms_coeff = []
        lms_coeff = []
        for step_index in ttnn_step_index:
            order = min(step_index + 1, orders)
            order_list.append(order)
            lms_coeffs = [
                get_lms_coefficient(order, step_index, curr_order, sigma_tensor) for curr_order in range(order)
            ]
            lms_coeff.append(lms_coeffs)

        for lms in lms_coeff:
            ttnn_lms_tensor = None
            for value in lms:
                lms_tensor = ttnn.full(
                    (1, 4, 64, 64), fill_value=value, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
                )
                if ttnn_lms_tensor is not None:
                    ttnn_lms_tensor = ttnn.concat([ttnn_lms_tensor, lms_tensor], dim=0)
                else:
                    ttnn_lms_tensor = lms_tensor

            ttnn_lms_coeff.append(ttnn_lms_tensor)

        model = UNet2D(device, parameters, 2, input_height, input_width, reader_patterns_cache)

        # # Denoising loop
        for i in range(len(time_step_list)):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            ttnn_latent_model_input = tt_latent_expansion(ttnn_latents, ttnn_scheduler, float(ttnn_sigma[i]), device)

            # predict the noise residual
            with torch.no_grad():
                profiler.start(f"model_run_for_inference_{i}")
                ttnn_noise_pred = model(
                    sample=ttnn_latent_model_input,
                    timestep=time_step_list[i],
                    encoder_hidden_states=ttnn_text_embeddings,
                    config=config,
                )
                profiler.end(f"model_run_for_inference_{i}")

            # perform guidance
            noise_pred = tt_guide(ttnn_noise_pred, guidance_scale)

            ttnn_latents = ttnn_scheduler.step(
                model_output=noise_pred,
                sample=ttnn_latents,
                sigma=float(ttnn_sigma[i]),
                lms_coeffs=ttnn_lms_coeff[i],
                device=device,
                order=order_list[i],
            ).prev_sample
            enable_persistent_kernel_cache()
        latents = ttnn.to_torch(ttnn_latents).to(torch.float32)

        # printout the perf
        profiler.print()
        comment = f"diffusiondb__{height}x{width}"
        ref_model_run_for_inference = profiler.get("ref_model_run_for_inference_0")
        first_iter_time = profiler.get("model_run_for_inference_0")
        second_iter_time = profiler.get(f"model_run_for_inference_{num_inference_steps-1}")

        logger.info("Call prep-perf-report")
        prep_perf_report(
            model_name=f"StableDiffusion",
            batch_size=batch_size,
            inference_and_compile_time=first_iter_time,
            inference_time=second_iter_time,
            expected_compile_time=expected_compile_time,
            expected_inference_time=expected_inference_time,
            comments=comment,
        )
        logger.info("Exit SD perf test")


@skip_for_grayskull()
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "expected_perf",
    ((8.23),),
)
def test_stable_diffusion_device_perf(expected_perf):
    subdir = "ttnn_stable_diffusion"
    margin = 0.02
    batch = 1
    iterations = 1
    command = f"pytest tests/ttnn/integration_tests/stable_diffusion/test_unet_2d_condition_model.py::test_unet_2d_condition_model_512x512[batch_size=2-in_channels=4-input_height=64-input_width=64-device_l1_small_size=32768]"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    # back-up the value of WH_ARCH_YAML if exist
    wh_arch_yaml_backup = None
    if "WH_ARCH_YAML" in os.environ:
        wh_arch_yaml_backup = os.environ["WH_ARCH_YAML"]

    os.environ["WH_ARCH_YAML"] = "wormhole_b0_80_arch_eth_dispatch.yaml"
    post_processed_results = run_device_perf(command, subdir, iterations, cols, batch, has_signposts=True)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"stable_diffusion_{batch}batch",
        batch_size=batch,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )

    # set WH_ARCH_YAML back to the original value
    if wh_arch_yaml_backup is not None:
        os.environ["WH_ARCH_YAML"] = wh_arch_yaml_backup
    else:
        del os.environ["WH_ARCH_YAML"]
