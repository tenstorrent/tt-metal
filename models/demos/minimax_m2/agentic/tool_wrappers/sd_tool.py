# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
StableDiffusionTool: Text-to-image generation using Stable Diffusion on TTNN.

Uses CompVis/stable-diffusion-v1-4 UNet with TTNN acceleration.
Generates 512x512 images from text prompts.
"""

import os

import numpy as np
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from loguru import logger
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.vision.generative.stable_diffusion.wormhole.custom_preprocessing import custom_preprocessor
from models.demos.vision.generative.stable_diffusion.wormhole.sd_pndm_scheduler import TtPNDMScheduler
from models.demos.vision.generative.stable_diffusion.wormhole.tt.ttnn_functional_unet_2d_condition_model_new_conv import (
    UNet2DConditionModel as UNet2D,
)
from models.demos.vision.generative.stable_diffusion.wormhole.tt.vae.ttnn_vae import Vae

MODEL_NAME = "CompVis/stable-diffusion-v1-4"
SD_L1_SMALL_SIZE = 20928
SD_TRACE_REGION_SIZE = 789835776


def constant_prop_time_embeddings(timesteps, sample, time_proj):
    """Precompute time embeddings for all timesteps."""
    timesteps = timesteps[None]
    timesteps = timesteps.expand(sample.shape[0])
    t_emb = time_proj(timesteps)
    return t_emb


class StableDiffusionTool:
    """
    TTNN-accelerated Stable Diffusion text-to-image tool.

    Generates images from text prompts using the UNet denoising process.
    """

    def __init__(self, mesh_device, num_inference_steps: int = 20):
        self.mesh_device = mesh_device
        self.num_inference_steps = num_inference_steps
        self._init_models(mesh_device)

    def _init_models(self, mesh_device):
        """Load all SD components."""
        logger.info("Loading Stable Diffusion components...")

        # Use chip0 submesh for SD (single-device model)
        if hasattr(mesh_device, "get_num_devices") and mesh_device.get_num_devices() > 1:
            self.device = mesh_device.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0))
        else:
            self.device = mesh_device

        # Load HuggingFace models
        logger.info("Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae")
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        logger.info("Loading tokenizer and text encoder...")
        self.tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder")

        logger.info("Loading UNet...")
        unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")
        self.unet_config = unet.config
        self.time_proj = unet.time_proj

        # Preprocess for TTNN
        logger.info("Preprocessing UNet for TTNN...")
        parameters = preprocess_model_parameters(
            initialize_model=lambda: unet,
            custom_preprocessor=custom_preprocessor,
            device=self.device,
        )

        # Create TTNN UNet (64x64 latent space for 256x256 output)
        self.input_height = 64
        self.input_width = 64
        self.tt_unet = UNet2D(self.device, parameters, 2, self.input_height, self.input_width)

        # Create TTNN VAE decoder
        self.tt_vae = Vae(torch_vae=self.vae, device=self.device)

        # Create scheduler
        self.scheduler = TtPNDMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            skip_prk_steps=True,
            steps_offset=1,
            device=self.device,
        )

        logger.info("Stable Diffusion ready.")

    def generate(self, prompt: str, output_path: str, guidance_scale: float = 7.5, seed: int = 42) -> str:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text description of the image to generate.
            output_path: Path to save the generated image.
            guidance_scale: Classifier-free guidance scale.
            seed: Random seed for reproducibility.

        Returns:
            Path to the generated image.
        """
        logger.info(f"Generating image for: '{prompt}'")

        # Tokenize prompt
        text_input = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Get text embeddings
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids)[0]

        # Unconditional embeddings for classifier-free guidance
        uncond_input = self.tokenizer(
            [""],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids)[0]

        # Concatenate for classifier-free guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        # Pad sequence length from 77 to 96 as required by TTNN UNet
        text_embeddings = torch.nn.functional.pad(text_embeddings, (0, 0, 0, 19))
        tt_text_embeddings = ttnn.from_torch(
            text_embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

        # Initialize latents
        generator = torch.manual_seed(seed)
        height = self.input_height * self.vae_scale_factor  # 256
        width = self.input_width * self.vae_scale_factor  # 256
        latents = torch.randn(
            (1, self.unet_config.in_channels, height // self.vae_scale_factor, width // self.vae_scale_factor),
            generator=generator,
        )

        # Set up scheduler
        self.scheduler.set_timesteps(self.num_inference_steps)
        latents = latents * self.scheduler.init_noise_sigma

        # Convert to TTNN
        tt_latents = ttnn.from_torch(latents, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        # Precompute time embeddings
        time_embeddings = []
        for t in self.scheduler.timesteps:
            t_emb = constant_prop_time_embeddings(t, latents, self.time_proj)
            t_emb = t_emb.unsqueeze(0).unsqueeze(0).permute(2, 0, 1, 3)
            tt_t_emb = ttnn.from_torch(t_emb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
            time_embeddings.append(tt_t_emb)

        # Denoising loop
        logger.info(f"Running {self.num_inference_steps} denoising steps...")
        for i, t in enumerate(self.scheduler.timesteps):
            # Expand latents for classifier-free guidance
            latent_model_input = ttnn.concat([tt_latents, tt_latents], dim=0)

            # Predict noise (pass config as required by UNet2D)
            noise_pred = self.tt_unet(
                latent_model_input,
                timestep=time_embeddings[i],
                encoder_hidden_states=tt_text_embeddings,
                config=self.unet_config,
            )

            # Perform guidance
            noise_pred_uncond, noise_pred_text = ttnn.split(noise_pred, 1, dim=0)
            noise_pred = ttnn.add(
                noise_pred_uncond,
                ttnn.multiply(ttnn.subtract(noise_pred_text, noise_pred_uncond), guidance_scale),
            )

            # Scheduler step
            tt_latents = self.scheduler.step(noise_pred, t, tt_latents).prev_sample

        # Decode latents to image
        logger.info("Decoding latents to image...")
        latents_torch = ttnn.to_torch(tt_latents).float()  # Convert bfloat16 to float32 for VAE
        latents_torch = latents_torch / 0.18215  # VAE scaling factor

        with torch.no_grad():
            image = self.vae.decode(latents_torch).sample

        # Convert to PIL and save
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        pil_image.save(output_path)

        logger.info(f"Image saved to: {output_path}")
        return output_path

    def close(self):
        """Release resources."""
        self.tt_unet = None
        self.tt_vae = None
        logger.info("StableDiffusionTool closed.")
