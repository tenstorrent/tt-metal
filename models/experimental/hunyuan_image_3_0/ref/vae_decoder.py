# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
PyTorch CPU golden reference for HunyuanImage-3.0 VAE decoder.

Loads only vae.decoder weights from local HunyuanImage-3 checkpoints.
"""

import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open

# Hunyuan package lives outside tt-metal
HUNYUAN_SRC = Path(
    os.environ.get(
        "HUNYUAN_SRC",
        "/home/iguser/ign-sakthi/HunyuanImage-3.0",
    )
)
MODEL_DIR = Path(
    os.environ.get(
        "HUNYUAN_MODEL_DIR",
        "/home/iguser/ign-sakthi/HunyuanImage-3.0/HunyuanImage-3",
    )
)

sys.path.insert(0, str(HUNYUAN_SRC))
from hunyuan_image_3.autoencoder_kl_3d import AutoencoderKLConv3D  # noqa: E402

SCALING_FACTOR = 0.562679178327931
LATENT_CHANNELS = 32
LATENT_H = 64
LATENT_W = 64
OUTPUT_H = 1024
OUTPUT_W = 1024


def _load_decoder_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
    """Load only vae.decoder.* keys (does not merge full 158GB checkpoint)."""
    index_path = model_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    decoder_map = {k: shard for k, shard in weight_map.items() if k.startswith("vae.decoder.")}
    if not decoder_map:
        raise RuntimeError(f"No vae.decoder.* keys found in {index_path}")

    # group keys by shard to minimize file opens
    shard_to_keys: dict[str, list[str]] = {}
    for key, shard in decoder_map.items():
        shard_to_keys.setdefault(shard, []).append(key)

    state_dict = {}
    for shard_file, keys in shard_to_keys.items():
        shard_path = model_dir / shard_file
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard {shard_path}. Download HunyuanImage-3 weights first.")
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for full_key in keys:
                short_key = full_key.removeprefix("vae.decoder.")
                state_dict[short_key] = f.get_tensor(full_key)

    return state_dict


class VaeDecoderPT:
    """PyTorch CPU VAE decoder for golden comparison."""

    def __init__(self, model_dir: Path = MODEL_DIR, dtype=torch.float32):
        with open(model_dir / "config.json") as f:
            vae_config = json.load(f)["vae"]

        vae_config = dict(vae_config)
        vae_config["only_decoder"] = True

        self.vae = AutoencoderKLConv3D.from_config(vae_config)
        self.vae.to(dtype=dtype)
        self.vae.eval()

        decoder_sd = _load_decoder_state_dict(model_dir)
        self.vae.decoder.load_state_dict(decoder_sd, strict=True)

        self.decoder = self.vae.decoder
        self.state_dict = self.vae.decoder.state_dict()

    @torch.no_grad()
    def forward(self, raw_latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            raw_latent: [B, 32, 64, 64] float32 — diffusion output BEFORE pipeline scaling

        Returns:
            [B, 3, 1024, 1024] float32
        """
        z = raw_latent.float() / SCALING_FACTOR
        z = z.unsqueeze(2)  # [B, 32, 1, 64, 64]
        out = self.decoder(z)  # [B, 3, T, 1024, 1024], T>1 when temporal upsample runs
        if z.shape[2] == 1:
            out = out[:, :, -1:]  # match AutoencoderKLConv3D.decode()
        return out.squeeze(2).float()  # [B, 3, 1024, 1024]


def get_input() -> torch.Tensor:
    """Deterministic random latent for testing."""
    torch.manual_seed(42)
    return torch.randn(1, LATENT_CHANNELS, LATENT_H, LATENT_W, dtype=torch.float32)
