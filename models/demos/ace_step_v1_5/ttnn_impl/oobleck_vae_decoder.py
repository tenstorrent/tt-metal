# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Load Hugging Face ``AutoencoderOobleck`` VAE folders and run the decoder on TTNN."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .vae.decoder import TtOobleckDecoder


def _pick_safetensors_file(vae_dir: Path) -> Path:
    for name in (
        "model.safetensors",
        "diffusion_pytorch_model.safetensors",
        "model.fp16.safetensors",
    ):
        cand = vae_dir / name
        if cand.is_file():
            return cand
    all_st = sorted(vae_dir.glob("*.safetensors"))
    if not all_st:
        raise FileNotFoundError(f"No *.safetensors found under {vae_dir}")
    return all_st[0]


def _load_state_dict_torch(path: Path) -> dict[str, Any]:
    try:
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    except ImportError:
        import torch

        try:
            return torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(str(path), map_location="cpu")
    except Exception:
        import torch

        try:
            return torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(str(path), map_location="cpu")


class TtOobleckVaeDecoder:
    """Wraps ``TtOobleckDecoder`` built from HF ``vae/`` checkpoints."""

    def __init__(self, decoder: TtOobleckDecoder) -> None:
        self._decoder = decoder

    @classmethod
    def from_hf_vae_dir(
        cls,
        vae_dir: str,
        *,
        device,
        latent_frames: int | None = None,
        batch_size: int | None = None,
        activation_dtype=None,
        weights_dtype=None,
        decoder_prefix: str = "decoder.",
    ) -> TtOobleckVaeDecoder:
        """Build from Diffusers-style ``config.json`` + ``*.safetensors`` weights.

        ``latent_frames`` / ``batch_size`` are accepted for ABI compatibility; conv
        packing is deferred to the forward pass according to runtime shapes (same idea
        as ``TtAceStepPatchEmbed1D``), so warmup is not triggered here (avoids a full
        decode on boot for long clips).
        """
        _ = latent_frames
        _ = batch_size

        root = Path(vae_dir)
        cfg_path = root / "config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Missing HF VAE config: {cfg_path}")
        sd_path = _pick_safetensors_file(root)

        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        downs = cfg.get("downsampling_ratios")
        if not isinstance(downs, list):
            raise ValueError("downsampling_ratios missing or not a list in VAE config.json")
        ups = downs[::-1]

        decoder_channels = int(cfg["decoder_channels"])
        decoder_input_channels = int(cfg["decoder_input_channels"])
        audio_channels = int(cfg["audio_channels"])
        channel_multiples = cfg["channel_multiples"]
        if not isinstance(channel_multiples, list):
            raise ValueError("channel_multiples missing or not a list in VAE config.json")

        state_dict = _load_state_dict_torch(sd_path)

        inner = TtOobleckDecoder(
            state_dict=state_dict,
            device=device,
            decoder_prefix=decoder_prefix,
            channels=decoder_channels,
            input_channels=decoder_input_channels,
            audio_channels=audio_channels,
            upsampling_ratios=tuple(ups),
            channel_multiples=tuple(channel_multiples),
            activation_dtype=activation_dtype,
            weights_dtype=weights_dtype,
        )
        return cls(inner)

    def __call__(self, latents_btc):
        """Decode ``[B, T_latent, C_latent]`` row-major to ``[B, T_audio, audio_channels]``."""
        return self._decoder(latents_btc)

    def forward(self, latents_btc):
        return self(latents_btc)

    def decode_tiled(
        self,
        latents_btc,
        *,
        chunk_size: int = 32,
        overlap: int = 4,
    ):
        """Decode ``[B, T_latent, C]`` along time using overlap-add (same idea as AceStep ``_tiled_decode_gpu``).

        Long sequences can exceed TTNN conv L1 circular-buffer limits when decoded in one shot; tiling
        keeps each ``ttnn.conv1d`` invocation on a bounded temporal extent.
        """
        dec = self._decoder
        ttnn = dec.ttnn
        if len(latents_btc.shape) != 3:
            raise ValueError(f"TtOobleckVaeDecoder.decode_tiled expects [B,T,C], got {latents_btc.shape}")

        # ttnn.slice on TILE layout requires both slice-start and output dimensions to be multiples of
        # TILE_HEIGHT=32 (asserted in slice_device_operation.cpp).  Chunk windows like [24:60] are never
        # 32-aligned, so slicing TILE layout silently returns wrong data in Release builds.
        # Also cast float32 latents (from diffusion) to bfloat16 here so TtConv1d gets the dtype it expects
        # (weights were packed with input_dtype=activation_dtype=bfloat16).
        if latents_btc.layout != ttnn.ROW_MAJOR_LAYOUT:
            latents_btc = ttnn.to_layout(latents_btc, ttnn.ROW_MAJOR_LAYOUT)
        if latents_btc.dtype != dec.activation_dtype:
            latents_btc = ttnn.typecast(latents_btc, dec.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        batch = int(latents_btc.shape[0])
        latent_frames = int(latents_btc.shape[1])
        c_lat = int(latents_btc.shape[2])
        if batch > 1:
            parts = []
            for b in range(batch):
                slab = ttnn.slice(latents_btc, (b, 0, 0), (b + 1, latent_frames, c_lat))
                parts.append(self.decode_tiled(slab, chunk_size=chunk_size, overlap=overlap))
            return ttnn.concat(parts, dim=0) if hasattr(ttnn, "concat") else ttnn.concatenate(parts, dim=0)

        chunk_size = int(chunk_size)
        overlap = int(overlap)
        if chunk_size < 8:
            raise ValueError(f"chunk_size must be >= 8 for stable tiling, got {chunk_size}")
        min_overlap = 4
        effective_overlap = overlap
        while chunk_size - 2 * effective_overlap <= 0 and effective_overlap > min_overlap:
            effective_overlap //= 2
        if effective_overlap < min_overlap and overlap >= min_overlap:
            effective_overlap = min_overlap
        overlap = effective_overlap

        if latent_frames <= chunk_size:
            return dec(latents_btc)

        stride = chunk_size - 2 * overlap
        if stride <= 0:
            raise ValueError(f"chunk_size {chunk_size} must be > 2 * overlap {overlap}")
        num_steps = math.ceil(latent_frames / stride)

        # Device ``ttnn.slice`` on ROW_MAJOR latents is safe for arbitrary ``[win_start, win_end)``;
        # TILE slices require 32-aligned bounds (see comment at top of this method).
        cores = []
        upsample_factor = None

        for i in range(num_steps):
            core_start = i * stride
            core_end = min(core_start + stride, latent_frames)
            win_start = max(0, core_start - overlap)
            win_end = min(latent_frames, core_end + overlap)

            latent_chunk = ttnn.slice(latents_btc, (0, win_start, 0), (1, win_end, c_lat))
            wav = dec(latent_chunk)
            # Decoder ends with conv2 → often TILE; trim uses ttnn.slice which is not safe on TILE
            # for arbitrary [T_start, T_end) (same 32-tile alignment rules as latents).
            wav = ttnn.to_layout(wav, ttnn.ROW_MAJOR_LAYOUT)
            latent_t = win_end - win_start
            b_w = int(wav.shape[0])
            ta = int(wav.shape[1])
            ca = int(wav.shape[2])

            if upsample_factor is None:
                upsample_factor = float(ta) / float(latent_t)

            added_start = core_start - win_start
            trim_start_i = int(round(added_start * upsample_factor))
            added_end = win_end - core_end
            trim_end_i = int(round(added_end * upsample_factor))

            trim_start_i = max(0, min(trim_start_i, ta))
            end_i = ta - trim_end_i if trim_end_i > 0 else ta
            end_i = max(trim_start_i, min(end_i, ta))

            audio_core = ttnn.slice(wav, (0, trim_start_i, 0), (b_w, end_i, ca))
            cores.append(audio_core)

        if len(cores) == 1:
            return cores[0]
        return ttnn.concat(cores, dim=1) if hasattr(ttnn, "concat") else ttnn.concatenate(cores, dim=1)
