# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""LTX-2.3 Pro audio-video pipeline (one-stage, full guidance)."""

from __future__ import annotations

import os
import sys
import time
from typing import TYPE_CHECKING

import torch
from loguru import logger

import ttnn

from .pipeline_ltx import LTXPipeline

if TYPE_CHECKING:
    pass


class LTXAVPipeline(LTXPipeline):
    """LTX-2.3 Pro AV pipeline: Gemma encode → DiT denoise → VAE decode → MP4 export."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("mode", "av")
        super().__init__(*args, **kwargs)

    @staticmethod
    def create_pipeline(mesh_device: ttnn.MeshDevice, **kwargs) -> "LTXAVPipeline":
        kwargs.setdefault("mode", "av")
        kwargs["pipeline_class"] = LTXAVPipeline
        return LTXPipeline.create_pipeline(mesh_device, **kwargs)

    def _ensure_device_encoder(self) -> None:
        """Lazily load the on-device Gemma encoder + video/audio connectors (once)."""
        if self.gemma_encoder is not None:
            return
        from safetensors import safe_open

        connector_prefixes = (
            "text_embedding_projection.video_aggregate_embed.",
            "text_embedding_projection.audio_aggregate_embed.",
            "model.diffusion_model.video_embeddings_connector.",
            "model.diffusion_model.audio_embeddings_connector.",
        )
        self.load_gemma_encoder(self.gemma_path, num_layers=48, sequence_length=1024)
        conn_state = {}
        with safe_open(self.checkpoint_name, "pt") as f:
            for k in f.keys():
                if k.startswith(connector_prefixes):
                    conn_state[k] = f.get_tensor(k)
        self.load_embeddings_connectors(conn_state)

    def warmup_buffers(
        self,
        *,
        num_frames: int,
        height: int,
        width: int,
        num_inference_steps: int = 2,
    ) -> None:
        """Compile every device program full-guidance ``call_av`` will exercise
        (4 transformer passes/step: cond/uncond/ptb/iso) plus VAE decode.
        ``ge_gamma=0`` skips the GE branch (pure host math)."""
        t0 = time.time()
        logger.info(f"warmup (AV): {num_frames}f@{height}x{width}, {num_inference_steps} steps")

        results = self.encode_prompts_reference(["warmup", "warmup"])
        v_p = results[0].video_encoding.float()
        a_p = results[0].audio_encoding.float()
        v_n = results[1].video_encoding.float()
        a_n = results[1].audio_encoding.float()

        self.call_av(
            video_prompt_embeds=v_p,
            audio_prompt_embeds=a_p,
            neg_video_prompt_embeds=v_n,
            neg_audio_prompt_embeds=a_n,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            seed=0,
            ge_gamma=0.0,
        )

        self._warmup_decode(num_frames, height, width)
        self._prepare_transformer(0)
        logger.info(f"warmup (AV) done in {time.time() - t0:.1f}s")

    def generate(
        self,
        prompt: str,
        *,
        output_path: str,
        negative_prompt: str | None = None,
        num_frames: int = 121,
        height: int = 512,
        width: int = 768,
        num_inference_steps: int = 30,
        video_cfg_scale: float = 3.0,
        audio_cfg_scale: float = 7.0,
        video_stg_scale: float = 1.0,
        audio_stg_scale: float = 1.0,
        video_modality_scale: float = 3.0,
        audio_modality_scale: float = 3.0,
        rescale_scale: float = 0.7,
        stg_block: int = 28,
        seed: int = 10,
        ge_gamma: float | None = None,
        fps: int = 24,
    ) -> str:
        """Run the full LTX-2.3 Pro AV generation pipeline and write an MP4."""
        if ge_gamma is None:
            # Official LTX gradient-estimation sampling; enable on BH by default for quality.
            ge_gamma = 2.0 if ttnn.device.is_blackhole() else 0.0
            logger.info(f"ge_gamma={ge_gamma} (arch default)")

        sys.path.insert(0, "LTX-2/packages/ltx-core/src")
        sys.path.insert(0, "LTX-2/packages/ltx-pipelines/src")
        torch.cuda.synchronize = lambda *a, **kw: None  # noqa: ARG005
        from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT

        neg = negative_prompt if negative_prompt is not None else DEFAULT_NEGATIVE_PROMPT

        def _env_float(name: str, default: float) -> float:
            val = os.environ.get(name)
            return float(val) if val is not None else default

        video_cfg_scale = _env_float("VIDEO_CFG_SCALE", video_cfg_scale)
        audio_cfg_scale = _env_float("AUDIO_CFG_SCALE", audio_cfg_scale)
        video_stg_scale = _env_float("VIDEO_STG_SCALE", video_stg_scale)
        audio_stg_scale = _env_float("AUDIO_STG_SCALE", audio_stg_scale)
        video_modality_scale = _env_float("VIDEO_MODALITY_SCALE", video_modality_scale)
        audio_modality_scale = _env_float("AUDIO_MODALITY_SCALE", audio_modality_scale)
        rescale_scale = _env_float("RESCALE_SCALE", rescale_scale)
        if os.environ.get("GE_GAMMA") is not None:
            ge_gamma = float(os.environ["GE_GAMMA"])

        total_t0 = time.time()

        t0 = time.time()
        if os.environ.get("LTX_DEVICE_ENCODE") == "1":
            # On-device Gemma encode. Under dynamic_load the encoder is registered
            # coresident-excluded with the DiT + VAE (see _register_encoder_exclusions),
            # so loading it auto-evicts the DiT and the later _prepare_transformer(0)
            # auto-evicts the encoder — no manual deallocation needed. FSDP shards the
            # encoder weights on the SP axis to further cut per-chip memory.
            self._ensure_device_encoder()
            enc = self.encode_prompts_device([prompt, neg])
            v_embeds, a_embeds = enc[0][0].float(), enc[0][1].float()
            neg_v, neg_a = enc[1][0].float(), enc[1][1].float()
            logger.info(f"Encoding (device): {time.time() - t0:.1f}s")
        else:
            results = self.encode_prompts_reference([prompt, neg])
            v_embeds = results[0].video_encoding.float()
            a_embeds = results[0].audio_encoding.float()
            neg_v = results[1].video_encoding.float()
            neg_a = results[1].audio_encoding.float()
            logger.info(f"Encoding: {time.time() - t0:.1f}s")

        self._prepare_transformer(0)

        t0 = time.time()
        video_latent, audio_latent = self.call_av(
            video_prompt_embeds=v_embeds,
            audio_prompt_embeds=a_embeds,
            neg_video_prompt_embeds=neg_v,
            neg_audio_prompt_embeds=neg_a,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            video_cfg_scale=video_cfg_scale,
            audio_cfg_scale=audio_cfg_scale,
            video_stg_scale=video_stg_scale,
            audio_stg_scale=audio_stg_scale,
            video_modality_scale=video_modality_scale,
            audio_modality_scale=audio_modality_scale,
            rescale_scale=rescale_scale,
            stg_block=stg_block,
            seed=seed,
            ge_gamma=ge_gamma,
        )
        denoise_time = time.time() - t0
        logger.info(f"Denoising: {denoise_time:.1f}s ({denoise_time / num_inference_steps:.1f}s/step)")

        t0 = time.time()
        self._prepare_vae()
        logger.info(f"VAE loaded in {time.time() - t0:.0f}s")

        latent_frames = (num_frames - 1) // 8 + 1
        latent_h, latent_w = height // 32, width // 32

        t0 = time.time()
        video_pixels = self.decode_latents(video_latent, latent_frames, latent_h, latent_w)
        logger.info(f"VAE decode: {time.time() - t0:.1f}s — {video_pixels.shape}")

        audio_obj = self.decode_audio(audio_latent, num_frames, fps=fps)
        self.export_video(video_pixels, output_path, fps=fps, audio=audio_obj)

        total_time = time.time() - total_t0
        logger.info(f"Total: {total_time:.1f}s | Output: {output_path}")
        return output_path
