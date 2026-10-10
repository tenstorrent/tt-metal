# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Lifecycle adapter for the LTX-2 audio decode stack (mel-VAE decoder + vocoder/BWE).

Owns config parsing, module construction (including the audio parallel-config selection), and the
disk-cache weight load — mirroring ``LTXVideoVAEAdapter`` for video. There is no WAN analog (WAN
has no audio), but the shape is identical: the pipeline holds an adapter and calls thin delegators.
"""

from __future__ import annotations

import json
import os

import torch
from loguru import logger
from safetensors import safe_open

import ttnn

from ...parallel.config import AudioTCParallelConfig, DiTParallelConfig, ParallelFactor
from ...parallel.manager import CCLManager
from ...utils import cache as cache_module
from ...utils.conv3d import conv3d_blocking_hash
from .bwe_ltx import MelSTFT, VocoderWithBWE
from .mel_decoder_ltx import MelDecoder
from .vocoder_ltx import Vocoder

# Config keys forwarded verbatim from the checkpoint's vocoder/bwe config into ``Vocoder``.
_VOCODER_CONFIG_KEYS = (
    "resblock_kernel_sizes",
    "upsample_rates",
    "upsample_kernel_sizes",
    "resblock_dilation_sizes",
    "upsample_initial_channel",
    "resblock",
    "activation",
    "use_tanh_at_final",
    "use_bias_at_final",
)

_TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})
_FALSE_TOKENS = frozenset({"0", "false", "no", "off"})


def _env_flag(name: str, *, default: bool) -> bool:
    """Boolean env var with one truthiness convention across all trace gates."""
    val = os.environ.get(name)
    if val is None:
        return default
    token = val.strip().lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    raise ValueError(f"{name}={val!r} is not a boolean (expected one of {_TRUE_TOKENS | _FALSE_TOKENS})")


class LTXAudioDecoderAdapter:
    """Owns the LTX audio decode stack: parses the checkpoint's ``audio_vae`` / ``vocoder`` config,
    builds the ``MelDecoder`` + ``VocoderWithBWE`` (two vocoders via the BWE path), selects the
    audio parallel config, and loads/reloads their weights.

    ``mel_decoder`` / ``vocoder_with_bwe`` expose the underlying modules (both ``None`` when the
    checkpoint requires an unsupported norm/causality). ``reload_weights`` is idempotent and driven
    by the pipeline; it re-injects the mel-VAE per-channel stats after load (load-bearing — those
    buffers are not carried by the binary cache).
    """

    def __init__(
        self,
        checkpoint_path: str,
        *,
        mesh_device: ttnn.MeshDevice,
        vae_ccl_manager: CCLManager,
        dit_parallel_config: DiTParallelConfig,
        traced: bool,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._mesh_device = mesh_device
        self._vae_ccl_manager = vae_ccl_manager
        self._dit_parallel_config = dit_parallel_config
        self._traced = traced

        self._mel_decoder: MelDecoder | None = None
        self._vocoder_with_bwe: VocoderWithBWE | None = None

        self._build()

    @property
    def mel_decoder(self) -> "MelDecoder | None":
        return self._mel_decoder

    @property
    def vocoder_with_bwe(self) -> "VocoderWithBWE | None":
        return self._vocoder_with_bwe

    def _build(self) -> None:
        """Construct audio decoder + vocoder module shells from checkpoint config.

        No weights are loaded — ``reload_weights`` handles that via the disk cache the same way
        the video VAE adapter does.
        """
        with safe_open(self._checkpoint_path, framework="pt") as f:
            config = json.loads(f.metadata()["config"])

        ad = config["audio_vae"]["model"]["params"]
        ddconfig = ad["ddconfig"]
        stft_cfg = config["audio_vae"].get("preprocessing", {}).get("stft", {})
        mel_cfg = config["audio_vae"].get("preprocessing", {}).get("mel", {})
        mel_bins = ddconfig.get("mel_bins") or mel_cfg.get("n_mel_channels")
        if ddconfig.get("norm_type", "pixel") != "pixel" or ddconfig.get("causality_axis", "height") != "height":
            logger.warning("Audio decoder: checkpoint requires unsupported norm/causality; skipping construction")
            return

        voc_cfg = config["vocoder"]["vocoder"]
        bwe_cfg = config["vocoder"]["bwe"]

        mesh_shape = tuple(self._mesh_device.shape)
        t_axis = 0 if mesh_shape[0] >= mesh_shape[1] else 1
        t_factor = mesh_shape[t_axis]
        c_axis = 1 - t_axis
        c_factor = mesh_shape[c_axis]
        channel_tp_on = os.environ.get("LTX_AUDIO_CHANNEL_TP", "0") == "1"
        if t_factor > 1 and c_factor > 1 and channel_tp_on:
            audio_parallel_config = AudioTCParallelConfig(
                time_parallel=ParallelFactor(factor=t_factor, mesh_axis=t_axis),
                channel_parallel=ParallelFactor(factor=c_factor, mesh_axis=c_axis),
            )
        elif t_factor > 1:
            audio_parallel_config = ParallelFactor(factor=t_factor, mesh_axis=t_axis)
        else:
            audio_parallel_config = None
        audio_ccl = self._vae_ccl_manager if audio_parallel_config is not None else None

        self._mel_decoder = MelDecoder(
            ch=ddconfig.get("ch", 128),
            out_ch=ddconfig.get("out_ch", 2),
            ch_mult=tuple(ddconfig.get("ch_mult", (1, 2, 4))),
            num_res_blocks=ddconfig.get("num_res_blocks", 2),
            attn_resolutions=tuple(ddconfig.get("attn_resolutions", ())),
            resolution=ddconfig.get("resolution", 256),
            z_channels=ddconfig.get("z_channels", 8),
            mid_block_add_attention=ddconfig.get("mid_block_add_attention", False),
            sample_rate=ad.get("sampling_rate", 16000),
            mel_hop_length=stft_cfg.get("hop_length", 160),
            is_causal=stft_cfg.get("causal", True),
            mel_bins=mel_bins,
            mesh_device=self._mesh_device,
            dtype=ttnn.bfloat16,
        )

        if isinstance(audio_parallel_config, AudioTCParallelConfig):
            override = os.environ.get("LTX_BWE_CHANNEL_TP")
            use_bwe_ctp = override == "1" if override is not None else audio_parallel_config.channel_parallel.factor > 2
            bwe_pc = audio_parallel_config if use_bwe_ctp else audio_parallel_config.time_parallel
        else:
            bwe_pc = audio_parallel_config
        main_voc = self._build_vocoder(
            voc_cfg, apply_final_activation=True, parallel_config=audio_parallel_config, ccl_manager=audio_ccl
        )
        bwe_voc = self._build_vocoder(
            bwe_cfg, apply_final_activation=False, parallel_config=bwe_pc, ccl_manager=audio_ccl
        )
        mel_stft = MelSTFT(
            filter_length=bwe_cfg["n_fft"],
            hop_length=bwe_cfg["hop_length"],
            win_length=bwe_cfg["n_fft"],
            n_mel_channels=bwe_cfg["num_mels"],
            mesh_device=self._mesh_device,
            dtype=ttnn.float32,
        )
        self._vocoder_with_bwe = VocoderWithBWE(
            vocoder=main_voc,
            bwe_generator=bwe_voc,
            mel_stft=mel_stft,
            input_sampling_rate=bwe_cfg["input_sampling_rate"],
            output_sampling_rate=bwe_cfg["output_sampling_rate"],
            hop_length=bwe_cfg["hop_length"],
            mesh_device=self._mesh_device,
            dtype=ttnn.float32,
        )
        self._vocoder_with_bwe.use_trace = self._traced and _env_flag("LTX_VOC_TRACE", default=True)
        self._vocoder_with_bwe.use_trace_bwe = self._traced
        self._mel_decoder.use_trace = self._traced and _env_flag("LTX_VAE_TRACE", default=False)
        if isinstance(audio_parallel_config, AudioTCParallelConfig):
            cfg_desc = f"T-shard={t_factor} axis{t_axis} + channel-TP={c_factor} axis{c_axis}"
        elif audio_parallel_config is not None:
            cfg_desc = f"T-shard={t_factor} axis{t_axis} (single-axis)"
        else:
            cfg_desc = "replicated"
        logger.info(f"Constructed audio decoder shells (mesh {mesh_shape}, vocoder {cfg_desc})")

    def _build_vocoder(
        self,
        cfg: dict,
        *,
        apply_final_activation: bool,
        parallel_config,
        ccl_manager,
    ) -> Vocoder:
        """Construct a ``Vocoder`` from a checkpoint vocoder/bwe config block.

        Used for both the main vocoder and the BWE generator, which differ only in their config
        block, final-activation behavior, and parallel config.
        """
        return Vocoder(
            **{k: cfg[k] for k in _VOCODER_CONFIG_KEYS if k in cfg},
            apply_final_activation=apply_final_activation,
            mesh_device=self._mesh_device,
            dtype=ttnn.float32,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )

    def _audio_decoder_state_provider(self) -> dict[str, torch.Tensor]:
        """Audio mel-VAE decoder weights from the checkpoint, prefix-stripped to the module keys."""
        logger.info("Audio decoder cache miss — loading from checkpoint")
        state: dict[str, torch.Tensor] = {}
        with safe_open(self._checkpoint_path, framework="pt") as f:
            for k in f.keys():
                if k.startswith("audio_vae.decoder."):
                    state[k.removeprefix("audio_vae.decoder.")] = f.get_tensor(k).float()
                elif k.startswith("audio_vae.per_channel_statistics."):
                    state[k.removeprefix("audio_vae.")] = f.get_tensor(k).float()
        return state

    def _vocoder_state_provider(self) -> dict[str, torch.Tensor]:
        """Vocoder (+ BWE) weights from the checkpoint, prefix-stripped to the module keys."""
        logger.info("Audio vocoder cache miss — loading from checkpoint")
        state: dict[str, torch.Tensor] = {}
        with safe_open(self._checkpoint_path, framework="pt") as f:
            for k in f.keys():
                if k.startswith("vocoder."):
                    state[k.removeprefix("vocoder.")] = f.get_tensor(k).float()
        return state

    def reload_weights(self) -> None:
        """Load audio decoder + vocoder weights via the disk cache (mirrors ``reload_decoder``).

        The per-channel denormalize stats are re-injected separately after load: they are
        non-Parameter buffers, so the binary cache path doesn't carry them and would otherwise
        leave garbage stats (and a garbage audio track).
        """
        if self._mel_decoder is None or self._vocoder_with_bwe is None:
            return
        if self._mel_decoder.is_loaded() and self._vocoder_with_bwe.is_loaded():
            return

        model_name = os.path.basename(self._checkpoint_path).removesuffix(".safetensors")
        blocking_key = conv3d_blocking_hash(self._vocoder_with_bwe)
        dec_subfolder = f"audio_dec_{blocking_key}" if blocking_key else "audio_dec"
        voc_subfolder = f"audio_voc_{blocking_key}" if blocking_key else "audio_voc"

        if not self._mel_decoder.is_loaded():
            cache_module.load_model(
                self._mel_decoder,
                model_name=model_name,
                subfolder=dec_subfolder,
                parallel_config=self._dit_parallel_config,
                mesh_shape=tuple(self._mesh_device.shape),
                mesh_device=self._mesh_device,
                get_torch_state_dict=self._audio_decoder_state_provider,
            )

        if not self._vocoder_with_bwe.is_loaded():
            cache_module.load_model(
                self._vocoder_with_bwe,
                model_name=model_name,
                subfolder=voc_subfolder,
                parallel_config=self._dit_parallel_config,
                mesh_shape=tuple(self._mesh_device.shape),
                mesh_device=self._mesh_device,
                get_torch_state_dict=self._vocoder_state_provider,
            )

        # Re-inject the mel-VAE per-channel stats from the checkpoint — see docstring.
        with safe_open(self._checkpoint_path, framework="pt") as f:
            self._mel_decoder.set_per_channel_stats(
                f.get_tensor("audio_vae.per_channel_statistics.std-of-means"),
                f.get_tensor("audio_vae.per_channel_statistics.mean-of-means"),
            )

        logger.info("Loaded TTNN audio decoder + vocoder")
