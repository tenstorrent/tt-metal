# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Load patch_embed / timestep modules from Hunyuan checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch

from models.experimental.hunyuan_image_3_0.ref.weights import MODEL_DIR, load_prefixed_state_dict

from models.experimental.hunyuan_image_3_0.ref.vision.siglip2 import (
    LightProjector,
    Siglip2VisionTransformer,
    load_aligner as _load_aligner,
    load_siglip2_vision as _load_siglip2_vision,
)

from .patch_embed import UNetDown
from .timestep_embedder import TimestepEmbedder


def _patch_embed_dims(state_dict: dict[str, torch.Tensor]) -> tuple[int, int, int, int, int]:
    in_channels = int(state_dict["model.0.weight"].shape[1])
    hidden_channels = int(state_dict["model.1.in_layers.2.weight"].shape[1])
    out_channels = int(state_dict["model.1.in_layers.2.weight"].shape[0])
    emb_channels = int(state_dict["model.1.emb_layers.1.weight"].shape[1])
    return 1, in_channels, emb_channels, hidden_channels, out_channels


def load_patch_embed(model_dir: Path = MODEL_DIR, *, dtype: torch.dtype = torch.float32) -> UNetDown:
    state = load_prefixed_state_dict(model_dir, "patch_embed.", dtype=dtype)
    patch_size, in_channels, emb_channels, hidden_channels, out_channels = _patch_embed_dims(state)
    module = UNetDown(
        patch_size=patch_size,
        in_channels=in_channels,
        emb_channels=emb_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
    )
    module.load_state_dict(state)
    module.to(dtype=dtype)
    module.eval()
    return module


def load_timestep_embedder(
    prefix: str,
    model_dir: Path = MODEL_DIR,
    *,
    hidden_size: int = 4096,
    dtype: torch.dtype = torch.float32,
) -> TimestepEmbedder:
    state = load_prefixed_state_dict(model_dir, f"{prefix}.", dtype=dtype)
    module = TimestepEmbedder(hidden_size=hidden_size)
    module.load_state_dict(state)
    module.to(dtype=dtype)
    module.eval()
    return module


def load_siglip2_vision(
    model_dir: Path = MODEL_DIR,
    *,
    num_layers: int | None = None,
    dtype: torch.dtype = torch.float32,
) -> Siglip2VisionTransformer:
    return _load_siglip2_vision(model_dir, num_layers=num_layers, dtype=dtype)


def load_aligner(
    model_dir: Path = MODEL_DIR,
    *,
    dtype: torch.dtype = torch.float32,
) -> LightProjector:
    return _load_aligner(model_dir, dtype=dtype)
