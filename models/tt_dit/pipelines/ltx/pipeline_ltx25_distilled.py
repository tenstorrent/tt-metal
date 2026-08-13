# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""LTX-2.5 distilled two-stage audio-video pipeline.

Separate from the 2.3 distilled pipeline so split-checkpoint wiring and Gemma-4 stay
out of the 2.3 path. Reuses distilled schedules / ``generate`` / ``warmup_buffers``.

Known gaps (not required for distilled T2V):
- Video VAE: prefer ``*-video-vae-conv-bf16``; until HF access lands we fall back to the 2.3
  monolith conv VAE (arch-identical). DiffVAE deferred.
- Keyframes abs-pos / DFR / duration head / I2V CRF — out of scope for distilled T2V.
"""

from __future__ import annotations

import os

from loguru import logger

import ttnn

from ...encoders.gemma4.encoder_pair import Gemma4TokenizerEncoderPair
from ...models.audio_vae.audio_decoder_ltx import LTXAudioDecoderAdapter
from ...models.transformers.ltx.transformer_ltx import LTXTransformerCheckpoint
from ...models.upsampler.latent_upsampler_ltx import LTXLatentUpsampler
from ...models.vae.vae_ltx import LTXVideoVAEAdapter
from ...utils.fuse_loras import LoraSpec
from ...utils.ltx import (
    LTX25_AUDIO_VAE,
    LTX25_DISTILLED_TRANSFORMER,
    LTX25_SPATIAL_UPSAMPLER,
    LTX25_TEXT_ENCODER,
    SPATIAL_COMPRESSION,
    TEMPORAL_COMPRESSION,
    ceil_to,
    default_ltx25_path,
    default_ltx25_video_vae,
)
from .pipeline_ltx import TransformerState
from .pipeline_ltx_distilled import LTXDistilledPipeline


class LTX25DistilledPipeline(LTXDistilledPipeline):
    """Distilled 2-stage AV pipeline for LTX-2.5 split checkpoints + Gemma-4."""

    HAS_UPSAMPLER = True

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        parallel_config,
        ccl_manager,
        *,
        video_vae_path: str | None = None,
        audio_vae_path: str | None = None,
        upsampler_path: str | None = None,
        use_ancestral_sampler: bool = True,
        **kwargs,
    ):
        # Set before ``super().__init__`` — ``_make_gemma_encoder_pair`` / ``_instantiate_modules``
        # run during base construction and read these paths.
        self._ltx25_video_vae_path = video_vae_path
        self._ltx25_audio_vae_path = audio_vae_path
        self._ltx25_upsampler_path = upsampler_path
        # Stage-1 ancestral Euler for 2.5+ (upstream ``should_use_ancestral_sampler``); stage 2 stays
        # deterministic. Override False only for A/B against the plain Euler path.
        self.use_ancestral_sampler = use_ancestral_sampler
        super().__init__(mesh_device, parallel_config, ccl_manager, **kwargs)

    def _denoise_no_guidance(self, v_embeds, a_embeds, *, trace_key: str | None = None, seed: int = 10, **kwargs):
        from .pipeline_ltx_distilled import ANCESTRAL_NOISE_SEED_OFFSET

        ancestral = self.use_ancestral_sampler and trace_key == "s1"
        return super()._denoise_no_guidance(
            v_embeds,
            a_embeds,
            trace_key=trace_key,
            seed=seed,
            ancestral=ancestral,
            ancestral_noise_seed=(seed + ANCESTRAL_NOISE_SEED_OFFSET) if ancestral else None,
            **kwargs,
        )

    def _make_gemma_encoder_pair(self, gemma_path: str | None):
        assert gemma_path is not None, "LTX-2.5 requires the packed Gemma-4 text-encoder path"
        assert self.checkpoint_name is not None, "LTX-2.5 requires the distilled transformer path"
        return Gemma4TokenizerEncoderPair(
            gemma_path,
            mesh_device=self.mesh_device,
            ccl_manager=self.vae_ccl_manager,
            parallel_config=self.encoder_parallel_config,
            transformer_checkpoint=self.checkpoint_name,
            mode=self.mode,
            dynamic_load=self.dynamic_load,
        )

    def _instantiate_modules(self, extra_variants: list[tuple[str, list[LoraSpec]]]) -> None:
        """Same module set as 2.3 distilled, but each component loads from its split file."""
        video_vae = self._ltx25_video_vae_path or self.checkpoint_name
        audio_vae = self._ltx25_audio_vae_path or self.checkpoint_name
        upsampler = self._ltx25_upsampler_path

        self.vae = LTXVideoVAEAdapter(
            video_vae,
            mesh_device=self.mesh_device,
            vae_parallel_config=self.vae_parallel_config,
            vae_ccl_manager=self.vae_ccl_manager,
            dit_parallel_config=self.parallel_config,
            num_frames=self._init_num_frames,
            height=self._init_height,
            width=self._init_width,
        )
        if not self.vae.decoder_blocks:
            raise RuntimeError(
                f"Video VAE at {video_vae!r} has no conv decoder_blocks "
                "(DiffVAE / wrong file). Use ltx-2.5-video-vae-conv-bf16 or a 2.3 monolith."
            )

        self.transformer_checkpoint = LTXTransformerCheckpoint(self.checkpoint_name, inner_dim=self.inner_dim)
        self.transformer = self._build_transformer(self.transformer_checkpoint)
        self.transformer_states.append(TransformerState(model=self.transformer, checkpoint=self.transformer_checkpoint))
        for tag, lora_specs in extra_variants:
            specs = list(lora_specs)
            self.transformer_states.append(
                TransformerState(
                    model=self._build_transformer(self.transformer_checkpoint),
                    checkpoint=self.transformer_checkpoint,
                    lora_specs=specs,
                )
            )
            logger.info(f"Registered transformer variant {tag} with {len(specs)} LoRA(s)")

        if self.HAS_UPSAMPLER:
            assert (
                self._init_height > 0 and self._init_width > 0 and self._init_num_frames > 0
            ), f"{type(self).__name__} requires num_frames/height/width at create_pipeline."
            assert upsampler is not None, "LTX-2.5 distilled requires the spatial upsampler path"
            self._upsampler_path = upsampler
            upsampler_latent_frames = (self._init_num_frames - 1) // TEMPORAL_COMPRESSION + 1
            hf = self.vae_parallel_config.height_parallel.factor
            wf = self.vae_parallel_config.width_parallel.factor
            upsampler_input_hw = (
                ceil_to(self._init_height // (SPATIAL_COMPRESSION * 2), hf),
                ceil_to(self._init_width // (SPATIAL_COMPRESSION * 2), wf),
            )
            self.upsampler = LTXLatentUpsampler.from_checkpoint(
                self._upsampler_path,
                input_hw=upsampler_input_hw,
                latent_frames=upsampler_latent_frames,
                mesh_device=self.mesh_device,
                parallel_config=self.vae_parallel_config,
                ccl_manager=self.vae_ccl_manager,
                dit_parallel_config=self.parallel_config,
            )

        self._audio_adapter = LTXAudioDecoderAdapter(
            audio_vae,
            mesh_device=self.mesh_device,
            vae_ccl_manager=self.vae_ccl_manager,
            dit_parallel_config=self.parallel_config,
            traced=self._traced,
        )

    @classmethod
    def create_pipeline(
        cls,
        mesh_device: ttnn.MeshDevice,
        *,
        checkpoint_name: str | None = None,
        gemma_path: str | None = None,
        text_encoder: str | None = None,
        transformer: str | None = None,
        video_vae: str | None = None,
        audio_vae: str | None = None,
        upsampler: str | None = None,
        **kwargs,
    ) -> "LTX25DistilledPipeline":
        """Resolve the five LTX-2.5 split paths, then build via the shared mesh defaults."""
        text_encoder = text_encoder or gemma_path or default_ltx25_path(LTX25_TEXT_ENCODER)
        transformer = transformer or checkpoint_name or default_ltx25_path(LTX25_DISTILLED_TRANSFORMER)
        video_vae = video_vae or default_ltx25_video_vae()
        audio_vae = audio_vae or default_ltx25_path(LTX25_AUDIO_VAE)
        upsampler = upsampler or default_ltx25_path(LTX25_SPATIAL_UPSAMPLER)
        missing = {
            "text_encoder": text_encoder,
            "transformer": transformer,
            "video_vae": video_vae,
            "audio_vae": audio_vae,
            "upsampler": upsampler,
        }
        absent = [name for name, path in missing.items() if not path]
        if absent:
            raise FileNotFoundError(
                "LTX-2.5 split checkpoints missing: "
                + ", ".join(absent)
                + " (set LTX25_ROOT or populate ~/.cache/ltx-checkpoints/ltx-2.5; "
                "video VAE can fall back to a local 2.3 monolith)"
            )
        if "ltx-2.3" in os.path.basename(video_vae) or "LTX-2.3" in video_vae:
            logger.warning(
                f"LTX-2.5 video VAE falling back to 2.3 monolith {video_vae} "
                "(download vae/ltx-2.5-video-vae-conv-bf16.safetensors when HF access allows)"
            )
        logger.info(
            f"LTX-2.5 distilled paths: text={text_encoder}, dit={transformer}, "
            f"video_vae={video_vae}, audio_vae={audio_vae}, upsampler={upsampler}"
        )
        return super().create_pipeline(
            mesh_device,
            checkpoint_name=transformer,
            gemma_path=text_encoder,
            video_vae_path=video_vae,
            audio_vae_path=audio_vae,
            upsampler_path=upsampler,
            **kwargs,
        )
