# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-side pieces of the Hunyuan3D-2.1 shape DiT (HunYuanDiTPlain, hunyuan3d-dit-v2-1) for the TT port.

The checkpoint `model.fp16.ckpt` holds {"model", "vae", "conditioner"}. `load_dit_state` extracts the DiT part once
and caches it as safetensors. The upstream attention concatenates q|k|v (k|v for cross-attention) and splits them per
head from the concatenation, so head h's q, k, v are interleaved slices; `fix_qkv_layout` permutes the projection
rows so a standard per-projection head split reproduces it exactly.
"""

from __future__ import annotations

import math
import os

import torch

HIDDEN = 2048
HEADS = 16
HEAD_DIM = HIDDEN // HEADS
DEPTH = 21
LATENTS = 4096
IN_CHANNELS = 64
CONTEXT_DIM = 1024
CONTEXT_LEN = 1370
NUM_MOE_LAYERS = 6
NUM_EXPERTS = 8
MOE_TOP_K = 2
NORM_EPS = 1e-6
QK_NORM_EPS = 1e-6


def load_dit_state(ckpt_path: str, cache_path: str) -> dict[str, torch.Tensor]:
    """DiT state dict in fp32 (cached as fp16 safetensors next to the run, built once from the 7.4 GB ckpt)."""
    from safetensors.torch import load_file, save_file

    if not os.path.exists(cache_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        save_file({k: v.contiguous() for k, v in ckpt["model"].items()}, cache_path)
        del ckpt
    return {k: v.float() for k, v in load_file(cache_path).items()}


def fix_qkv_layout(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rewrite attn1.to_q/k/v and attn2.to_k/v rows of every block into the standard per-head layout."""
    d = HEAD_DIM
    out = dict(state)
    for i in range(DEPTH):
        p = f"blocks.{i}.attn1."
        w = torch.cat([state[p + "to_q.weight"], state[p + "to_k.weight"], state[p + "to_v.weight"]], 0)
        w = w.reshape(HEADS, 3, d, -1)  # head h owns rows [h*3d, (h+1)*3d) of the concatenation
        for j, name in enumerate(("to_q", "to_k", "to_v")):
            out[p + name + ".weight"] = w[:, j].reshape(HIDDEN, -1).contiguous()
        p = f"blocks.{i}.attn2."
        w = torch.cat([state[p + "to_k.weight"], state[p + "to_v.weight"]], 0).reshape(HEADS, 2, d, -1)
        for j, name in enumerate(("to_k", "to_v")):
            out[p + name + ".weight"] = w[:, j].reshape(HIDDEN, -1).contiguous()
    return out


def timestep_vector(state: dict[str, torch.Tensor], t: torch.Tensor) -> torch.Tensor:
    """t [B] in [0, 1] -> c [B, HIDDEN]: sin/cos of HIDDEN channels (max period 1e4), Linear -> GELU -> Linear."""
    half = HIDDEN // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
    emb = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], -1)
    h = torch.nn.functional.linear(emb, state["t_embedder.mlp.0.weight"], state["t_embedder.mlp.0.bias"])
    h = torch.nn.functional.gelu(h)
    return torch.nn.functional.linear(h, state["t_embedder.mlp.2.weight"], state["t_embedder.mlp.2.bias"])


def is_moe_layer(i: int) -> bool:
    return DEPTH - i <= NUM_MOE_LAYERS


def has_skip(i: int) -> bool:
    return i > DEPTH // 2
