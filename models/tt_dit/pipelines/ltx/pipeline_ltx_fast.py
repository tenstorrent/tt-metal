# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""LTX-2.3 Fast distilled two-stage audio-video pipeline."""

from __future__ import annotations

import gc
import time

import torch
from loguru import logger

from ...models.transformers.ltx.ltx_transformer import LTXTransformerModel
from ...utils.ltx import AudioLatentShape, VideoPixelShape
from ...utils.tensor import bf16_tensor
from .pipeline_ltx import LTXPipeline, euler_step
from .pipeline_ltx_av import LTXAVPipeline

DISTILLED_SIGMA_VALUES = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]


class LTXFastPipeline(LTXAVPipeline):
    """Distilled 2-stage AV pipeline: half-res denoise → upsample → full-res refine."""

    @staticmethod
    def create_pipeline(mesh_device: ttnn.MeshDevice, **kwargs) -> "LTXFastPipeline":
        kwargs.setdefault("mode", "av")
        kwargs["pipeline_class"] = LTXFastPipeline
        return LTXPipeline.create_pipeline(mesh_device, **kwargs)

    def _denoise_no_guidance(
        self,
        v_embeds: torch.Tensor,
        a_embeds: torch.Tensor,
        *,
        num_frames: int,
        height: int,
        width: int,
        sigma_values: list[float],
        seed: int,
        initial_video_latent: torch.Tensor | None = None,
        initial_audio_latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = 1
        latent_frames = (num_frames - 1) // 8 + 1
        latent_h, latent_w = height // 32, width // 32
        video_N = latent_frames * latent_h * latent_w

        vps = VideoPixelShape(batch=B, frames=num_frames, height=height, width=width, fps=24)
        als = AudioLatentShape.from_video_pixel_shape(vps)
        audio_N_real = als.frames
        sp_factor = self.parallel_config.sequence_parallel.factor
        audio_N = ((audio_N_real + 32 * sp_factor - 1) // (32 * sp_factor)) * (32 * sp_factor)

        v_cos, v_sin = self._prepare_rope(latent_frames, latent_h, latent_w)
        a_cos, a_sin = self._prepare_audio_rope(audio_N, audio_N_real)
        tt_attn_mask, tt_pad_mask = self._prepare_audio_masks(audio_N, audio_N_real)

        tt_vp = self._prepare_prompt(v_embeds)
        tt_ap = bf16_tensor(a_embeds.unsqueeze(0), device=self.mesh_device)

        sigmas = torch.tensor(sigma_values, dtype=torch.float32)

        if initial_video_latent is not None:
            video_lat = initial_video_latent.float()
            torch.manual_seed(seed)
            noise_v = torch.randn_like(video_lat)
            video_lat = video_lat * (1 - sigmas[0]) + noise_v * sigmas[0]
        else:
            torch.manual_seed(seed)
            video_lat = torch.randn(B, video_N, self.in_channels, dtype=torch.bfloat16).float() * sigmas[0]

        if initial_audio_latent is not None:
            audio_lat = torch.zeros(B, audio_N, self.in_channels)
            audio_lat[:, :audio_N_real, :] = initial_audio_latent[:, :audio_N_real, :].float()
            torch.manual_seed(seed + 1)
            noise_a = torch.randn_like(audio_lat)
            audio_lat = audio_lat * (1 - sigmas[0]) + noise_a * sigmas[0]
        else:
            torch.manual_seed(seed)
            _ = torch.randn(B, video_N, self.in_channels, dtype=torch.bfloat16)
            audio_lat_real = torch.randn(B, audio_N_real, self.in_channels, dtype=torch.bfloat16).float() * sigmas[0]
            audio_lat = torch.zeros(B, audio_N, self.in_channels)
            audio_lat[:, :audio_N_real, :] = audio_lat_real

        num_steps = len(sigma_values) - 1
        for step_idx in range(num_steps):
            sigma = sigmas[step_idx].item()
            sigma_next = sigmas[step_idx + 1].item()

            v_out, a_out = self.transformer.inner_step(
                video_1BNI_torch=video_lat.unsqueeze(0),
                video_prompt_1BLP=tt_vp,
                video_rope_cos=v_cos,
                video_rope_sin=v_sin,
                video_N=video_N,
                audio_1BNI_torch=audio_lat.unsqueeze(0),
                audio_prompt_1BLP=tt_ap,
                audio_rope_cos=a_cos,
                audio_rope_sin=a_sin,
                audio_N=audio_N,
                trans_mat=None,
                timestep_torch=torch.tensor([sigma]),
                audio_attn_mask=tt_attn_mask,
                audio_padding_mask=tt_pad_mask,
            )
            v_vel = LTXTransformerModel.device_to_host(v_out).squeeze(0)
            a_vel = LTXTransformerModel.device_to_host(a_out).squeeze(0)

            v_den = (video_lat.bfloat16().float() - v_vel.float() * sigma).bfloat16()
            a_den = (audio_lat.bfloat16().float() - a_vel.float() * sigma).bfloat16()

            if sigma_next == 0.0:
                video_lat = v_den.float()
                a_new = a_den.float()
            else:
                video_lat = euler_step(video_lat, v_den.float(), sigma, sigma_next).bfloat16().float()
                a_new = euler_step(audio_lat, a_den.float(), sigma, sigma_next).bfloat16().float()

            audio_lat = torch.zeros_like(audio_lat)
            audio_lat[:, :audio_N_real, :] = a_new[:, :audio_N_real, :]
            logger.info(f"  Step {step_idx + 1}/{num_steps}: σ {sigma:.4f} → {sigma_next:.4f}")

        return video_lat, audio_lat[:, :audio_N_real, :]

    @staticmethod
    def _upsample_latent_reference(
        video_latent: torch.Tensor, upsampler_path: str, checkpoint_path: str
    ) -> torch.Tensor:
        import sys

        sys.path.insert(0, "LTX-2/packages/ltx-core/src")
        sys.path.insert(0, "LTX-2/packages/ltx-pipelines/src")
        from ltx_pipelines.utils.model_ledger import ModelLedger

        ledger = ModelLedger(
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            checkpoint_path=checkpoint_path,
            spatial_upsampler_path=upsampler_path,
        )
        upsampler = ledger.spatial_upsampler()
        with torch.no_grad():
            upsampled = upsampler(video_latent.bfloat16())
        del upsampler, ledger
        gc.collect()
        return upsampled.float()

    def generate(
        self,
        prompt: str,
        *,
        output_path: str,
        checkpoint_path: str,
        upsampler_path: str,
        gemma_path: str,
        num_frames: int = 121,
        height: int = 512,
        width: int = 768,
        seed: int = 10,
        fps: int = 24,
    ) -> str:
        """Run the distilled 2-stage AV pipeline and write an MP4."""
        assert height % 64 == 0, f"Height must be divisible by 64 (got {height})"
        assert width % 64 == 0, f"Width must be divisible by 64 (got {width})"

        s1_height = height // 2
        s1_width = width // 2
        total_t0 = time.time()

        t0 = time.time()
        results = self.encode_prompts_reference([prompt], checkpoint_path, gemma_path)
        logger.info(f"Encoding: {time.time() - t0:.1f}s")

        v_embeds = results[0].video_encoding.float()
        a_embeds = results[0].audio_encoding.float()

        self.load_transformer_from_checkpoint(checkpoint_path)
        gc.collect()

        logger.info(f"Stage 1: {s1_height}x{s1_width}, {len(DISTILLED_SIGMA_VALUES) - 1} steps")
        t0 = time.time()
        s1_video, s1_audio = self._denoise_no_guidance(
            v_embeds,
            a_embeds,
            num_frames=num_frames,
            height=s1_height,
            width=s1_width,
            sigma_values=DISTILLED_SIGMA_VALUES,
            seed=seed,
        )
        logger.info(f"Stage 1: {time.time() - t0:.1f}s")

        self.transformer = None
        gc.collect()

        latent_frames = (num_frames - 1) // 8 + 1
        s1_h, s1_w = s1_height // 32, s1_width // 32
        s1_spatial = s1_video.reshape(1, latent_frames, s1_h, s1_w, 128).permute(0, 4, 1, 2, 3)
        upsampled = self._upsample_latent_reference(s1_spatial, upsampler_path, checkpoint_path)
        upsampled_flat = upsampled.permute(0, 2, 3, 4, 1).reshape(
            1, latent_frames * (height // 32) * (width // 32), 128
        )

        self.load_transformer_from_checkpoint(checkpoint_path)
        gc.collect()

        logger.info(f"Stage 2: {height}x{width}, {len(STAGE_2_DISTILLED_SIGMA_VALUES) - 1} steps")
        t0 = time.time()
        s2_video, s2_audio = self._denoise_no_guidance(
            v_embeds,
            a_embeds,
            num_frames=num_frames,
            height=height,
            width=width,
            sigma_values=STAGE_2_DISTILLED_SIGMA_VALUES,
            seed=seed,
            initial_video_latent=upsampled_flat,
            initial_audio_latent=s1_audio.unsqueeze(0) if s1_audio.dim() == 2 else s1_audio,
        )
        logger.info(f"Stage 2: {time.time() - t0:.1f}s")

        self.transformer = None
        gc.collect()

        self.load_vae_from_checkpoint()
        latent_h, latent_w = height // 32, width // 32
        video_pixels = self.decode_latents(s2_video, latent_frames, latent_h, latent_w)
        audio_obj = self.decode_audio_reference(s2_audio, checkpoint_path, num_frames, fps=fps)
        self.export_video(video_pixels, output_path, fps=fps, audio=audio_obj)

        logger.info(f"Total: {time.time() - total_t0:.1f}s | Output: {output_path}")
        return output_path
