# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm.auto import tqdm

import ttnn


@dataclass
# Copied from diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput with DDPM->LMSDiscrete
class TtLMSDiscreteSchedulerOutput:
    prev_sample: ttnn.Tensor
    pred_original_sample: Optional[ttnn.Tensor] = None


class TtLMSDiscreteScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        prediction_type: str = "epsilon",
        device=None,
    ):
        self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        self.num_train_timesteps = num_train_timesteps
        self.alphas = 1.0 - self.betas
        self.betas = ttnn.from_torch(self.betas)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = np.concatenate([sigmas[::-1], [0.0]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = self.sigmas.max()

        # setable values
        self.num_inference_steps = None
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=float)[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps)
        self.derivatives = []
        self.is_scale_input_called = False

    def scale_model_input(self, sample, sigma, device) -> ttnn.Tensor:
        value = (sigma**2 + 1) ** 0.5
        denominator = ttnn.full(
            sample.shape, fill_value=value, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        denominator = ttnn.reciprocal(denominator)
        sample = ttnn.mul(sample, denominator)
        self.is_scale_input_called = True
        return sample

    def set_timesteps(self, num_inference_steps: int, device=None):
        self.num_inference_steps = num_inference_steps

        timesteps = np.linspace(0, self.num_train_timesteps - 1, num_inference_steps, dtype=float)[::-1].copy()
        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)
        self.sigmas = ttnn.from_torch(self.sigmas, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

        self.timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32).unsqueeze(0)
        self.timesteps = ttnn.from_torch(self.timesteps, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

        self.derivatives = []

    def step(
        self,
        model_output,
        sample,
        sigma,
        lms_coeffs,
        device,
        order: int = 4,
        return_dict: bool = True,
    ) -> Union[TtLMSDiscreteSchedulerOutput, Tuple]:
        if not self.is_scale_input_called:
            warnings.warn(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        pred_original_sample = sample - sigma * model_output
        # 2. Convert to an ODE derivative
        numerator = sample - pred_original_sample
        denominator = ttnn.full(
            numerator.shape, fill_value=sigma, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        denominator = ttnn.reciprocal(denominator)
        derivative = numerator * denominator
        self.derivatives.append(derivative)
        if len(self.derivatives) > order:
            self.derivatives.pop(0)

        if len(self.derivatives) > 1:
            derivative_tensor = ttnn.concat(self.derivatives[::-1], dim=0)
        else:
            derivative_tensor = self.derivatives[0]
        derivative_tensor = derivative_tensor * lms_coeffs
        if derivative_tensor.shape[0] > 1:
            derivative_tensor = ttnn.permute(derivative_tensor, (3, 1, 2, 0))
            derivative_tensor = ttnn.sum(derivative_tensor, dim=-1, keepdim=True)
            derivative_tensor = ttnn.permute(derivative_tensor, (3, 1, 2, 0))
        prev_sample = sample + derivative_tensor

        if not return_dict:
            return (prev_sample,)

        return TtLMSDiscreteSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)


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


def run(
    model,
    config,
    tt_vae,
    input_latents,
    input_encoder_hidden_states,
    _tlist,
    time_step,
    guidance_scale,
    ttnn_scheduler,
):
    ttnn_latents = input_latents
    # Denoising loop
    for index in tqdm(range(len(time_step))):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        ttnn_latent_model_input = ttnn.concat([ttnn_latents, ttnn_latents], dim=0)
        _t = _tlist[index]
        t = time_step[index]
        # predict the noise residual
        ttnn_output = model(
            ttnn_latent_model_input,  # input
            timestep=_t,
            encoder_hidden_states=input_encoder_hidden_states,
            class_labels=None,
            attention_mask=None,
            cross_attention_kwargs=None,
            return_dict=True,
            config=config,
        )
        # perform guidance
        noise_pred = tt_guide(ttnn_output, guidance_scale)

        ttnn_latents = ttnn_scheduler.step(noise_pred, t, ttnn_latents).prev_sample

    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * ttnn_latents

    ttnn_output = ttnn.permute(latents, [0, 2, 3, 1])
    ttnn_output = tt_vae.decode(ttnn_output)
    ttnn_output = ttnn.reshape(ttnn_output, [1, 512, 512, ttnn_output.shape[3]])
    return ttnn.permute(ttnn_output, [0, 3, 1, 2])


def compile_trace_sd(
    device, model, config, tt_vae, input_latents, _tlist, time_step, guidance_scale, ttnn_scheduler, num_inference_steps
):
    ttnn_text_embeddings_device = ttnn.allocate_tensor_on_device(
        ttnn.Shape([2, 96, 768]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    encoder_hidden_states_rand = torch.randn([2, 77, 768])
    encoder_hidden_states_rand = torch.nn.functional.pad(encoder_hidden_states_rand, (0, 0, 0, 19))
    encoder_hidden_states_rand = ttnn.from_torch(
        encoder_hidden_states_rand, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )

    # COMPILE
    ttnn_scheduler.set_timesteps(num_inference_steps)
    ttnn.copy_host_to_device_tensor(encoder_hidden_states_rand, ttnn_text_embeddings_device, cq_id=0)
    output = ttnn.from_device(
        run(
            model,
            config,
            tt_vae,
            input_latents,
            ttnn_text_embeddings_device,
            _tlist,
            time_step,
            guidance_scale,
            ttnn_scheduler,
        )
    )

    # CAPTURE
    ttnn_scheduler.set_timesteps(num_inference_steps)
    ttnn.copy_host_to_device_tensor(encoder_hidden_states_rand, ttnn_text_embeddings_device, cq_id=0)
    output.deallocate(True)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    output = run(
        model,
        config,
        tt_vae,
        input_latents,
        ttnn_text_embeddings_device,
        _tlist,
        time_step,
        guidance_scale,
        ttnn_scheduler,
    )
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)

    return ttnn_text_embeddings_device, output, tid
