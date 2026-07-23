# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Official HuggingFace AutoencoderKLConv3D VAE decode (matches hunyuan_image_3_pipeline)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from torch import Tensor

from models.experimental.hunyuan_image_3_0.ref.model_config import VAE_SCALING_FACTOR
from models.experimental.hunyuan_image_3_0.ref.weights import load_tensors

_VAE_CACHE: dict[Path, object] = {}


def resolve_hunyuan_src() -> Path:
    """Directory containing the ``hunyuan_image_3`` Python package."""
    if override := os.environ.get("HUNYUAN_SRC"):
        root = Path(override)
        if (root / "hunyuan_image_3" / "autoencoder_kl_3d.py").is_file():
            return root
        raise FileNotFoundError(f"HUNYUAN_SRC={root} has no hunyuan_image_3/autoencoder_kl_3d.py")

    here = Path(__file__).resolve()
    for candidate in (
        here.parents[6] / "HunyuanImage-3.0",
        here.parents[5].parent / "HunyuanImage-3.0",
        Path.home() / "HunyuanImage-3.0",
    ):
        if (candidate / "hunyuan_image_3" / "autoencoder_kl_3d.py").is_file():
            return candidate

    raise FileNotFoundError(
        "HunyuanImage-3.0 source not found. Clone it next to tt-metal or set HUNYUAN_SRC "
        "to the repo root containing hunyuan_image_3/autoencoder_kl_3d.py"
    )


def _ensure_hunyuan_import():
    src = str(resolve_hunyuan_src())
    if src not in sys.path:
        sys.path.insert(0, src)


def load_hf_vae(model_dir: Path):
    """Load official HF VAE weights from safetensors (cached per model_dir)."""
    model_dir = Path(model_dir)
    if model_dir in _VAE_CACHE:
        return _VAE_CACHE[model_dir]

    _ensure_hunyuan_import()
    from hunyuan_image_3.autoencoder_kl_3d import AutoencoderKLConv3D, load_weights

    cfg = json.load(open(model_dir / "config.json"))
    vae = AutoencoderKLConv3D.from_config(cfg["vae"]).eval()

    index = json.load(open(model_dir / "model.safetensors.index.json"))["weight_map"]
    vae_keys = [k for k in index if k.startswith("vae.")]
    load_weights(vae, load_tensors(model_dir, vae_keys))

    _VAE_CACHE[model_dir] = vae
    return vae


@torch.no_grad()
def decode_latent_hf(
    latent_bchw: Tensor,
    *,
    model_dir: Path,
    scaling_factor: float = VAE_SCALING_FACTOR,
) -> Tensor:
    """Decode diffusion latent [B,C,h,w] -> RGB [B,3,H,W] in [0,1] via HF AutoencoderKLConv3D."""
    vae = load_hf_vae(model_dir)
    z = latent_bchw.float() / scaling_factor
    z = z.unsqueeze(2)  # [B, C, 1, h, w]
    out = vae.decode(z, return_dict=False)[0]
    if out.ndim == 5:
        out = out.squeeze(2)
    return (out / 2 + 0.5).clamp(0, 1)
