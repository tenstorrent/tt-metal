# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from torch import autocast
import torch
import time
from PIL import Image
from tqdm.auto import tqdm
from loguru import logger


from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    PNDMScheduler,
    HeunDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers import LMSDiscreteScheduler
from tqdm.auto import tqdm

from models.utility_functions import (
    torch_to_tt_tensor,
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
    comp_pcc,
    comp_allclose_and_pcc,
    Profiler,
)
from models.utility_functions import enable_persistent_kernel_cache, disable_persistent_kernel_cache

import ttnn
from models.experimental.stable_diffusion.tt.unet_2d_condition import UNet2DConditionModel as tt_unet_condition


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

    torch.save(_latents, f"{pre_fix}{pre_fix2}latents_{iter}.pt")


def guide(noise_pred_uncond, noise_pred_text, guidance_scale, t):  # will return latents
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return noise_pred


def latent_expansion(latents, scheduler, t):
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
    conditioned, unconditioned = latent_model_input.chunk(2)
    return conditioned, unconditioned


def make_tt_unet(state_dict, device):
    tt_unet = tt_unet_condition(
        sample_size=64,
        in_channels=4,
        out_channels=4,
        center_input_sample=False,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"],
        mid_block_type="UNetMidBlock2DCrossAttn",
        up_block_types=["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
        only_cross_attention=False,
        block_out_channels=[320, 640, 1280, 1280],
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        act_fn="silu",
        norm_num_groups=32,
        norm_eps=1e-05,
        cross_attention_dim=768,
        attention_head_dim=8,
        dual_cross_attention=False,
        use_linear_projection=False,
        class_embed_type=None,
        num_class_embeds=None,
        upcast_attention=False,
        resnet_time_scale_shift="default",
        state_dict=state_dict,
        device=device,
        base_address="",
    )
    return tt_unet


def demo():
    # Initialize the device
    device = ttnn.open_device(0)

    ttnn.SetDefaultDevice(device)

    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

    # 4. load the K-LMS scheduler with some fitting parameters.
    # Throughout the generation process, scheduler is internally changed, hence we need one for torch and one for TT
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
    )
    tt_scheduler = LMSDiscreteScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
    )

    disable_persistent_kernel_cache()
    torch_device = "cpu"
    vae.to(torch_device)
    text_encoder.to(torch_device)
    unet.to(torch_device)

    state_dict = unet.state_dict()
    tt_unet = make_tt_unet(state_dict, device)
    tt_unet.config = unet.config

    # experiment name is associated with saved latents and images
    experiment_name = "mountain_fallback_nolatentupdate"
    # prompt = ["a photo of an astronaut riding a horse on mars"] # guidance 7.5
    prompt = [
        "oil painting frame of Breathtaking mountain range with a clear river running through it, surrounded by tall trees and misty clouds, serene, peaceful, mountain landscape, high detail"
    ]  # guidance 7.5
    # prompt = ["Skull on Fire"] # guidance = 12

    # height and width much be divisible by 32, and can be as little as 64x64
    # 64x64 images are not coherent; but useful for a quick pcc test.

    height = 64  # default height of Stable Diffusion
    width = 64  # default width of Stable Diffusion
    num_inference_steps = 2  # Number of denoising steps
    guidance_scale = 7.5  # Scale for classifier-free guidance
    generator = torch.manual_seed(174)  # 10233 Seed generator to create the inital latent noise
    batch_size = len(prompt)

    ## First, we get the text_embeddings for the prompt. These embeddings will be used to condition the UNet model.
    # Tokenizer and Text Encoder
    text_input = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

    # For classifier-free guidance, we need to do two forward passes: one with the conditioned input (text_embeddings),
    # and another with the unconditional embeddings (uncond_embeddings).
    # In practice, we can concatenate both into a single batch to avoid doing two forward passes.
    # in this demo, each forward pass will be done independently.
    _text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # Initial random noise
    latents = torch.randn(
        (batch_size, unet.config.in_channels, height // 8, width // 8),
        generator=generator,
    )
    latents = latents.to(torch_device)

    scheduler.set_timesteps(num_inference_steps)
    tt_scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.init_noise_sigma
    tt_latents = torch.tensor(latents)
    latents_dict = {}
    pcc_res = {}
    iter = 0

    ############## Beginning of Image Generation by Torch ##############

    # torch Denoising loop
    for t in tqdm(scheduler.timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        conditioned, unconditioned = latent_expansion(latents, scheduler, t)
        # predict the noise residual
        with torch.no_grad():
            # first forward pass; conditioned on the prompt
            noise_pred_cond = unet(conditioned, t, encoder_hidden_states=text_embeddings).sample
            # second forward pass; un-conditioned
            noise_pred_uncond = unet(unconditioned, t, encoder_hidden_states=uncond_embeddings).sample
        # perform guidance
        noise_pred = guide(noise_pred_uncond, noise_pred_cond, guidance_scale, t)
        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        latents_dict[iter] = latents

        save_image_and_latents(latents, iter, vae, pre_fix=f"{experiment_name}_torch", pre_fix2="")
        iter += 1

    # saving the final image generated by torch
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    # Image post-processing
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images][0]
    pil_images.save(f"{experiment_name}_torch.png")
    ############## End of Image Generation by Torch ##############

    iter = 0
    last_latents = None
    # # Denoising loop
    for t in tqdm(tt_scheduler.timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        # tt_latent_model_input = latent_expansion(tt_latents, tt_scheduler, t)
        tt_conditioned, tt_unconditioned = latent_expansion(tt_latents, tt_scheduler, t)

        # the initial embedding step is constant propped here

        _t_cond = constant_prop_time_embeddings(t, tt_conditioned, unet.time_proj)
        _t_uncond = constant_prop_time_embeddings(t, tt_unconditioned, unet.time_proj)
        _t_cond = torch_to_tt_tensor_rm(_t_cond, device, put_on_device=False)
        _t_uncond = torch_to_tt_tensor_rm(_t_uncond, device, put_on_device=False)
        tt_conditioned = torch_to_tt_tensor_rm(tt_conditioned, device, put_on_device=False)
        tt_unconditioned = torch_to_tt_tensor_rm(tt_unconditioned, device, put_on_device=False)
        tt_text_embeddings = torch_to_tt_tensor_rm(text_embeddings, device, put_on_device=False)
        tt_uncond_embeddings = torch_to_tt_tensor_rm(uncond_embeddings, device, put_on_device=False)
        # end of constant prop

        # predict the noise residual
        with torch.no_grad():
            tt_noise_pred_cond = tt_unet(tt_conditioned, _t_cond, encoder_hidden_states=tt_text_embeddings)
            tt_noise_pred_uncond = tt_unet(tt_unconditioned, _t_uncond, encoder_hidden_states=tt_uncond_embeddings)
            noise_pred_cond = tt_to_torch_tensor(tt_noise_pred_cond)
            noise_pred_uncond = tt_to_torch_tensor(tt_noise_pred_uncond)
        # perform guidance
        noise_pred = guide(noise_pred_uncond, noise_pred_cond, guidance_scale, t)
        # compute the previous noisy sample x_t -> x_t-1
        tt_latents = tt_scheduler.step(noise_pred, t, tt_latents).prev_sample
        save_image_and_latents(tt_latents, iter, vae, pre_fix=f"{experiment_name}_tt", pre_fix2="")
        pcc_res[iter] = comp_allclose_and_pcc(latents_dict[iter], tt_latents)
        logger.info(f"{iter}, {pcc_res[iter]}")
        last_latents = tt_latents
        # save things required!
        iter += 1
        # we enable compile cache after the first iteration
        enable_persistent_kernel_cache()

    latents = last_latents
    for key, val in pcc_res.items():
        logger.info(f"{key}, {val}")
    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample

    # Image post-processing
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images][0]
    pil_images.save(f"{experiment_name}_tt.png")

    ttnn.close_device(device)


"""
@article{patil2022stable,
author = {Patil, Suraj and Cuenca, Pedro and Lambert, Nathan and von Platen, Patrick},
title = {Stable Diffusion with :firecracker: Diffusers},
journal = {Hugging Face Blog},
year = {2022},
note = {[https://huggingface.co/blog/rlhf](https://huggingface.co/blog/stable_diffusion)},
}
"""
demo()
