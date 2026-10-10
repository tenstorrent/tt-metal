# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""LTX-2.5 distilled two-stage audio-video pipeline.

Separate from the 2.3 distilled pipeline so split-checkpoint wiring and Gemma-4 stay
out of the 2.3 path. Reuses the distilled schedules, ``generate`` and the traced resident path.

Known gaps (not required for distilled T2V):
- Video VAE: prefer ``*-video-vae-conv-bf16``; until HF access lands we fall back to the 2.3
  monolith conv VAE (arch-identical). ``diffusion_decoder`` decodes with the DiffVAE instead.
- Keyframes abs-pos / DFR / duration head / I2V CRF — out of scope for distilled T2V.
"""

from __future__ import annotations

import json
import os

from loguru import logger
from safetensors import safe_open

import ttnn

from ...encoders.gemma3.encoder_pair import GemmaTokenizerEncoderPair
from ...encoders.gemma4.encoder_pair import Gemma4TokenizerEncoderPair
from ...models.vae.diffvae_ltx import DiffVAEDecoder, DiffVAEOptions
from ...models.vae.diffvae_ltx import decoder_config as diffvae_config
from ...utils import cache as cache_module
from ...utils.ltx import (
    LTX25_AUDIO_VAE,
    LTX25_DISTILLED_TRANSFORMER,
    LTX25_SPATIAL_UPSAMPLER,
    LTX25_TEXT_ENCODER,
    LTX25_VIDEO_VAE_CONV_DEFAULT,
    default_ltx25_path,
    default_ltx25_video_vae,
)
from .pipeline_ltx_distilled import LTXDistilledPipeline


def _default_gemma3_path() -> str:
    """Local Gemma-3 snapshot for the LTX25_TEXT_STACK=gemma3 A/B, same lookup as the 2.3 tests."""
    import glob

    cands = glob.glob(
        os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-3-12b-it-qat-q4_0-unquantized/snapshots/*/")
    )
    assert cands, "no local Gemma-3 snapshot; set LTX25_GEMMA3_PATH"
    return cands[0].rstrip("/")


def _vae_header_config(path: str) -> dict:
    with open(path, "rb") as f:
        header_size = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_size))
    return json.loads(header.get("__metadata__", {}).get("config", "{}")).get("vae", {})


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
        diffvae_path: str | None = None,
        use_ancestral_sampler: bool = True,
        diffusion_decoder: bool = False,
        diffvae_options: DiffVAEOptions | None = None,
        **kwargs,
    ):
        # Set before ``super().__init__``: the config read and module construction run during base
        # construction and read these paths.
        self._ltx25_video_vae_path = video_vae_path
        self._ltx25_audio_vae_path = audio_vae_path
        self._ltx25_upsampler_path = upsampler_path
        self._ltx25_diffvae_path = diffvae_path
        self._diffusion_decoder = diffusion_decoder
        self._diffvae_options = diffvae_options
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
        # LTX25_TEXT_STACK=gemma3 runs the whole 2.3 text stack (Gemma-3 plus the 2.3 monolith's
        # projection and connectors) under an otherwise untouched 2.5 pipeline, isolating the text path.
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

    def _load_config_from_checkpoint(self) -> None:
        """Transformer config from the split transformer file; conv-VAE config from the video VAE.

        The 2.5 transformer file carries no ``vae`` block, so the base read leaves the VAE unset.
        The conv file also supplies the VAE encoder and the latent statistics the upsampler
        brackets with, whichever decoder runs.
        """
        super()._load_config_from_checkpoint()
        video_vae = self._ltx25_video_vae_path
        vae_cfg = _vae_header_config(video_vae)
        if not vae_cfg.get("decoder_blocks"):
            raise RuntimeError(
                f"Video VAE at {video_vae!r} has no conv decoder_blocks (DiffVAE / wrong file). "
                "Use ltx-2.5-video-vae-conv-bf16 or a 2.3 monolith."
            )
        self._vae_checkpoint_path = video_vae
        self._vae_decoder_blocks = vae_cfg.get("decoder_blocks", [])
        self._vae_encoder_blocks = vae_cfg.get("encoder_blocks", [])
        self._vae_causal = vae_cfg.get("causal_decoder", False)
        self._vae_base_channels = vae_cfg.get("decoder_base_channels", 128)
        self._vae_patch_size = vae_cfg.get("patch_size", 4)
        self._vae_in_channels = vae_cfg.get("in_channels", 3)
        self._vae_latent_channels = vae_cfg.get("latent_channels", 128)
        self._vae_norm_layer = vae_cfg.get("norm_layer", "pixel_norm")
        self._vae_latent_log_var = vae_cfg.get("latent_log_var", "uniform")
        self._vae_spatial_padding_mode = vae_cfg.get("spatial_padding_mode", "zeros")
        logger.info(f"LTX-2.5 conv VAE config from {video_vae}: {len(self._vae_decoder_blocks)} decoder blocks")

    def _upsampler_checkpoint_path(self) -> str:
        return self._ltx25_upsampler_path

    def _audio_checkpoint_path(self) -> str:
        return self._ltx25_audio_vae_path

    def _instantiate_modules(self, extra_variants, *, audio_only: bool = False) -> None:
        super()._instantiate_modules(extra_variants, audio_only=audio_only)
        if audio_only or not self._diffusion_decoder:
            return
        options = self._diffvae_options or DiffVAEOptions()
        self.vae_decoder = DiffVAEDecoder(
            diffvae_config(self._ltx25_diffvae_path),
            mesh_device=self.mesh_device,
            ccl_manager=self.vae_ccl_manager,
            options=options,
        )
        # The DiffVAE is not rebuilt per frame count (see ``_ensure_vae_decoder_frames``).
        self._vae_decoder_nf = None
        logger.info(f"VAE config: DiffVAE diffusion decoder {options}")

    def _ensure_vae_decoder_frames(self, num_frames: int) -> None:
        if self._diffusion_decoder:
            return
        super()._ensure_vae_decoder_frames(num_frames)

    def _prepare_vae(self) -> None:
        if not self._diffusion_decoder:
            super()._prepare_vae()
            return
        decoder = self.vae_decoder
        if decoder.requires_exclusive_residency:
            for module in self._video_decode_evictable():
                module.deallocate_weights()
        if decoder.is_loaded():
            return
        path = self._ltx25_diffvae_path
        # No conv3d blocking to key on; the parameter layout is keyed because the deterministic
        # block options change which parameters exist.
        cache_module.load_model(
            decoder,
            model_name=os.path.basename(path).removesuffix(".safetensors"),
            subfolder=f"diffvae/{decoder.parameter_layout()}",
            parallel_config=self.parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            mesh_device=self.mesh_device,
            get_torch_state_dict=lambda: decoder.torch_state_from_checkpoint(path),
        )
        logger.info("Loaded TTNN DiffVAE decoder")

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
        diffusion_decoder: bool = False,
        **kwargs,
    ) -> "LTX25DistilledPipeline":
        """Resolve the LTX-2.5 split paths, then build via the shared mesh defaults.

        ``diffusion_decoder`` selects the DiffVAE over the conv decoder; ``diffvae_options`` (a
        :class:`DiffVAEOptions`, through ``kwargs``) says how it runs.
        """
        text_encoder = text_encoder or gemma_path or default_ltx25_path(LTX25_TEXT_ENCODER)
        transformer = transformer or checkpoint_name or default_ltx25_path(LTX25_DISTILLED_TRANSFORMER)
        video_vae = video_vae or default_ltx25_video_vae(diffusion=False)
        diffvae = default_ltx25_video_vae(diffusion=True) if diffusion_decoder else None
        audio_vae = audio_vae or default_ltx25_path(LTX25_AUDIO_VAE)
        upsampler = upsampler or default_ltx25_path(LTX25_SPATIAL_UPSAMPLER)
        paths = {
            "text_encoder": text_encoder,
            "transformer": transformer,
            "video_vae": video_vae,
            "audio_vae": audio_vae,
            "upsampler": upsampler,
        }
        if diffusion_decoder:
            paths["diffvae"] = diffvae
        absent = [name for name, path in paths.items() if not path]
        if absent:
            raise FileNotFoundError(
                "LTX-2.5 split checkpoints missing: "
                + ", ".join(absent)
                + " (set LTX25_ROOT or populate ~/.cache/ltx-checkpoints/ltx-2.5; the conv video VAE "
                f"also resolves from LTX25_VIDEO_VAE or {LTX25_VIDEO_VAE_CONV_DEFAULT}, and "
                "LTX25_VAE_FALLBACK_23=1 opts into a local 2.3 monolith)"
            )
        if not os.path.isfile(video_vae):
            raise FileNotFoundError(f"LTX-2.5 video VAE {video_vae} does not exist")
        if "ltx-2.3" in os.path.basename(video_vae) or "LTX-2.3" in video_vae:
            logger.warning(f"LTX-2.5 conv video VAE falling back to 2.3 monolith {video_vae}")
        with safe_open(transformer, framework="pt") as f:
            version = (f.metadata() or {}).get("model_version", "?")
        logger.info(
            f"LTX-2.5 distilled paths (model_version={version}): text={text_encoder}, dit={transformer}, "
            f"video_vae={video_vae}, diffvae={diffvae}, audio_vae={audio_vae}, upsampler={upsampler}"
        )
        return super().create_pipeline(
            mesh_device,
            checkpoint_name=transformer,
            gemma_path=text_encoder,
            video_vae_path=video_vae,
            audio_vae_path=audio_vae,
            upsampler_path=upsampler,
            diffvae_path=diffvae,
            diffusion_decoder=diffusion_decoder,
            **kwargs,
        )
