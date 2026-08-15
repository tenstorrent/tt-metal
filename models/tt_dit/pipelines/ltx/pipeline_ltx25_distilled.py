# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""LTX-2.5 distilled two-stage audio-video pipeline.

Separate from the 2.3 distilled pipeline so split-checkpoint wiring and Gemma-4 stay
out of the 2.3 path. Reuses distilled schedules / ``generate`` / ``warmup_buffers``.

Known gaps (not required for distilled T2V):
- Video VAE: prefer ``*-video-vae-conv-bf16``; until HF access lands we fall back to the 2.3
  monolith conv VAE (arch-identical). ``LTX25_DIFFVAE=1`` routes video decode through the
  diffusion decoder the model card actually ships instead, split across the mesh.
- Keyframes abs-pos / DFR / duration head / I2V CRF — out of scope for distilled T2V.
"""

from __future__ import annotations

import os

from loguru import logger

import ttnn

from ...encoders.gemma3.encoder_pair import GemmaTokenizerEncoderPair
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
    LTX25_VIDEO_VAE_DIFF,
    SPATIAL_COMPRESSION,
    TEMPORAL_COMPRESSION,
    ceil_to,
    default_ltx25_path,
    default_ltx25_video_vae,
)
from .pipeline_ltx import TransformerState
from .pipeline_ltx_distilled import LTXDistilledPipeline


def _default_gemma3_path() -> str:
    """Local Gemma-3 snapshot for the LTX25_TEXT_STACK=gemma3 A/B, same lookup as the 2.3 tests."""
    import glob

    cands = glob.glob(
        os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-3-12b-it-qat-q4_0-unquantized/snapshots/*/")
    )
    assert cands, "no local Gemma-3 snapshot; set LTX25_GEMMA3_PATH"
    return cands[0].rstrip("/")


def _dram_allocated(mesh_device) -> int:
    """Bytes of DRAM in use on one device, or 0 where the memory view is unavailable."""
    try:
        view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
        return view.total_bytes_allocated_per_bank * view.num_banks
    except Exception:  # noqa: BLE001 - reporting only
        return 0


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
        # A/B lever for the prompt-adherence investigation: LTX25_TEXT_STACK=gemma3 runs the whole
        # 2.3 text stack — Gemma-3 plus the 2.3 monolith's projection and connectors — under an
        # otherwise untouched 2.5 pipeline, leaving the text path as the only variable. Both 2.5
        # text stages are numerically faithful to the checkpoint (see LTX2_5_PORT.md), so this is
        # comparing two correct implementations, not a correct one against a broken one.
        if os.environ.get("LTX25_TEXT_STACK") == "gemma3":
            gemma3_path = os.environ.get("LTX25_GEMMA3_PATH") or _default_gemma3_path()
            gemma3_ckpt = os.environ.get("LTX25_GEMMA3_CHECKPOINT") or os.path.expanduser(
                "~/.cache/ltx-checkpoints/ltx-2.3-22b-distilled-1.1.safetensors"
            )
            logger.warning(f"LTX25_TEXT_STACK=gemma3: text path from {gemma3_path} + {gemma3_ckpt}")
            return GemmaTokenizerEncoderPair(
                gemma3_path,
                mesh_device=self.mesh_device,
                ccl_manager=self.vae_ccl_manager,
                parallel_config=self.encoder_parallel_config,
                checkpoint_name=gemma3_ckpt,
                mode=self.mode,
                dynamic_load=self.dynamic_load,
            )

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
        self._build_diffvae()

    def _prepare_vae(self) -> None:
        """Make room for the DiffVAE decode, the way the conv decoder makes room for its own.

        This runs immediately before ``decode_latents``, which is where the conv path pages the
        active transformer out — a 22B DiT and a video VAE do not both fit on Blackhole. The
        DiffVAE needs that eviction *more*: its attention buffers are gigabytes where the conv
        decoder's are megabytes. But it gets it from nowhere, because the co-resident exclusion
        graph is only wired under ``dynamic_load`` and lists the conv decoder, not this one.

        The transformer is reloaded at the top of every generation, so paging it out here costs a
        cache read on the next clip and nothing else. Traces are the exception: a captured DiT
        trace holds pointers into those weight buffers, so freeing them would replay garbage.

        The conv decoder's own weights are never loaded at all — its only forward is the
        ``decode_latents`` this class overrides. The encoder reloads separately and is untouched,
        so i2v conditioning still works.
        """
        if getattr(self, "diffvae", None) is None:
            super()._prepare_vae()
            return

        if self._traced:
            return

        before = _dram_allocated(self.mesh_device)
        freed = []
        for idx, state in enumerate(self.transformer_states):
            if state.model is not None and state.model.is_loaded():
                state.model.deallocate_weights()
                freed.append(f"transformer[{idx}]")
        if self.upsampler is not None and self.upsampler.is_loaded():
            self.upsampler.deallocate_weights()
            freed.append("upsampler")
        # Text encoding is finished by the time anything decodes, and ``ensure_loaded`` pages it
        # back for the next prompt.
        gemma = getattr(self.gemma_encoder_pair, "gemma_encoder", None)
        if gemma is not None and gemma.is_loaded():
            gemma.deallocate_weights()
            freed.append("gemma")
        after = _dram_allocated(self.mesh_device)
        logger.info(
            f"DiffVAE decode: evicted {', '.join(freed) or 'nothing'} — "
            f"DRAM {before / 1e9:.2f} -> {after / 1e9:.2f} GB per device"
        )

    def _build_diffvae(self) -> None:
        """Optionally load the diffusion decoder in place of the conv one.

        Off by default. Only the conv *decoder* steps aside; its encoder stays, since i2v still
        conditions on real pixels and nothing about that path changes.
        """
        self.diffvae = None
        self.diffvae_shard = None
        if os.environ.get("LTX25_DIFFVAE") not in ("1", "true", "True"):
            return

        from ...layers.na3d import DEFAULT_SCORE_BUDGET
        from ...models.vae.diffvae_ltx import DiffVAEDecoder, MeshShardConfig, decoder_config

        path = os.environ.get("LTX25_DIFFVAE_PATH") or default_ltx25_path(LTX25_VIDEO_VAE_DIFF)
        assert path, f"LTX25_DIFFVAE=1 needs {LTX25_VIDEO_VAE_DIFF} under the 2.5 root, or LTX25_DIFFVAE_PATH"
        logger.info(f"LTX-2.5 DiffVAE video decode enabled: {path}")
        self.diffvae = DiffVAEDecoder(decoder_config(path), mesh_device=self.mesh_device)
        self.diffvae.load_checkpoint(path)
        self.diffvae_shard = MeshShardConfig(
            ccl=self.vae_ccl_manager,
            mesh=tuple(self.mesh_device.shape),
            # Which stage the volume stops being replicated at. Bounded by divisibility, not
            # memory: the latent grid is 34 wide at 1080p and a mesh of 4 does not divide it,
            # but everything from the first upsample on does.
            enter_stage=int(os.environ.get("LTX25_DIFFVAE_ENTER_STAGE", "1")),
            # The budget trades dispatched work against peak memory: the gathered key tensor is
            # tiles x keys-per-tile with no query factor, so smaller tiles barely shrink the key
            # span but multiply the tile count, and the biggest buffer runs from 1.4 GB at 2^26 to
            # 12 GB at 2^18. Paging the DiT out leaves ~29 GB free, which is enough that the
            # decode should be bought at the cheap end rather than the roomy one.
            budget=int(os.environ.get("LTX25_DIFFVAE_SCORE_BUDGET", DEFAULT_SCORE_BUDGET)),
        )

    def release_traces(self) -> None:
        super().release_traces()
        if getattr(self, "diffvae", None) is not None:
            self.diffvae.release_traces()

    def decode_latents(self, latent, latent_frames: int, latent_h: int, latent_w: int, *, output_type: str = "float"):
        """Route video decode through the DiffVAE when it is loaded, else the conv decoder."""
        if self.diffvae is None:
            return super().decode_latents(latent, latent_frames, latent_h, latent_w, output_type=output_type)

        assert output_type != "yuv", (
            "DiffVAE returns host pixels, so the conv decoder's on-device YUV path does not "
            "apply; set LTX_YUV_EXPORT=0"
        )
        # Follows LTX_TRACED unless overridden. The override exists because DiffVAE's capture is
        # the only traced region carrying fabric CCL, so it is the one worth turning off on its
        # own when a traced run misbehaves.
        override = os.environ.get("LTX25_DIFFVAE_TRACED")
        self.diffvae._traced = bool(self._traced) if override is None else override in ("1", "true", "True")
        batch = latent.shape[0]
        spatial = latent.reshape(batch, latent_frames, latent_h, latent_w, self.in_channels)
        spatial = spatial.permute(0, 4, 1, 2, 3)

        if dump_dir := os.environ.get("LTX_DUMP_LATENT"):
            self._dump_decode_input(spatial, dump_dir)

        # Stage 5 predicts x0 from noise, so the noise is an input. A fixed seed keeps a decode
        # reproducible across runs; upstream draws a fresh one per call.
        video = self.diffvae.decode(
            spatial, seed=int(os.environ.get("LTX25_DIFFVAE_SEED", "0")), shard=self.diffvae_shard
        )
        return video if output_type == "float" else video.numpy()

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
