# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Lazy safetensors access for the three sub-models plus host-side weight preprocessing."""
from __future__ import annotations

import json
import os
from typing import Dict, Iterable

import torch
from safetensors import safe_open

from .config import snapshot_dir


class LazyCheckpoint:
    """Opens the safetensors shards of one sub-folder lazily and returns tensors by key."""

    def __init__(self, subfolder: str, root: str | None = None):
        self.root = os.path.join(root or snapshot_dir(), subfolder)
        idx_files = [f for f in os.listdir(self.root) if f.endswith(".safetensors.index.json")]
        if idx_files:
            if len(idx_files) != 1:
                raise ValueError(f"expected one safetensors index in {self.root}; found {sorted(idx_files)}")
            with open(os.path.join(self.root, idx_files[0])) as index:
                idx = json.load(index)
            self.key_to_file = {k: os.path.join(self.root, v) for k, v in idx["weight_map"].items()}
        else:
            single = [f for f in os.listdir(self.root) if f.endswith(".safetensors")]
            assert len(single) == 1, single
            path = os.path.join(self.root, single[0])
            with safe_open(path, framework="pt") as f:
                self.key_to_file = {k: path for k in f.keys()}
        self._handles: Dict[str, object] = {}

    def keys(self) -> Iterable[str]:
        return self.key_to_file.keys()

    def __contains__(self, key: str) -> bool:
        return key in self.key_to_file

    def _handle(self, path: str):
        h = self._handles.get(path)
        if h is None:
            h = safe_open(path, framework="pt", device="cpu")
            self._handles[path] = h
        return h

    def get(self, key: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Always returns a tensor the caller owns: safetensors hands out memory-mapped views, and an
        in-place edit of one (e.g. folding an attention scale into a weight) would otherwise leak into every
        later reader of the same key in this process."""
        t = self._handle(self.key_to_file[key]).get_tensor(key)
        if dtype is not None and t.dtype != dtype:
            return t.to(dtype)
        return t.clone()

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.get(key)

    def get_rows(self, key: str, rows: torch.Tensor) -> torch.Tensor:
        """Gather rows of a 2-D tensor without materializing the whole tensor (embedding lookup)."""
        sl = self._handle(self.key_to_file[key]).get_slice(key)
        idx = rows.reshape(-1).tolist()
        return torch.stack([sl[i : i + 1][0] for i in idx], dim=0)

    def close(self):
        self._handles.clear()


def transformer_ckpt(root: str | None = None) -> LazyCheckpoint:
    return LazyCheckpoint("transformer", root)


def text_encoder_ckpt(root: str | None = None) -> LazyCheckpoint:
    return LazyCheckpoint("text_encoder", root)


def vae_ckpt(root: str | None = None) -> LazyCheckpoint:
    return LazyCheckpoint("vae", root)


# ----------------------------------------------------------------------------- preprocessing
def linear_to_mm(w: torch.Tensor) -> torch.Tensor:
    """torch nn.Linear weight [out, in] -> matmul weight [in, out]."""
    return w.t().contiguous()


def interleave_pairs_permutation(head_dim: int) -> torch.Tensor:
    """Permutation p with new[j] = old[p[j]] turning llama-style rotate_half pairs (i, i+D/2)
    into adjacent pairs (2i, 2i+1) so the interleaved-pair RoPE kernel can be used."""
    half = head_dim // 2
    p = torch.empty(head_dim, dtype=torch.long)
    p[0::2] = torch.arange(half)
    p[1::2] = torch.arange(half) + half
    return p


def permute_heads_rows(w_out_in: torch.Tensor, n_heads: int, head_dim: int, perm: torch.Tensor) -> torch.Tensor:
    """Apply `perm` inside every head along the output axis of a [n_heads*head_dim, in] Linear weight."""
    out, inp = w_out_in.shape
    assert out == n_heads * head_dim
    w = w_out_in.view(n_heads, head_dim, inp)[:, perm, :]
    return w.reshape(out, inp).contiguous()


def swiglu_interleave(gate_out_in: torch.Tensor, up_out_in: torch.Tensor, tile: int = 32) -> torch.Tensor:
    """Build the tile-pair-interleaved [K, 2N] matmul weight expected by minimal_matmul(fuse_swiglu=True):
    column tile 2p = gate tile p, column tile 2p+1 = up tile p."""
    g = linear_to_mm(gate_out_in)  # [K, N]
    u = linear_to_mm(up_out_in)
    K, N = g.shape
    assert N % tile == 0
    g = g.view(K, N // tile, tile)
    u = u.view(K, N // tile, tile)
    return torch.stack([g, u], dim=2).reshape(K, 2 * N).contiguous()


def fuse_qkv(wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor) -> torch.Tensor:
    """[in, Nq+Nk+Nv] fused matmul weight from three [out, in] Linear weights."""
    return torch.cat([linear_to_mm(wq), linear_to_mm(wk), linear_to_mm(wv)], dim=1).contiguous()
