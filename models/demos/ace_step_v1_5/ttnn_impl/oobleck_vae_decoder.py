# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Load Hugging Face ``AutoencoderOobleck`` VAE folders and run the decoder on TTNN."""

from __future__ import annotations

import json
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
