# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ERNIE-4.5-21B-A3B chunked prefill on a 1x4 Blackhole mesh: decoder block + full model."""

from __future__ import annotations

import gc
import time
from typing import Callable

import torch

import ttnn
from models.demos.ernie45_d_p.reference.ernie_ref import ErnieConfig, LayerWeights, WeightLoader, load_layer
from models.demos.ernie45_d_p.tt.attention import TtAttention, TtKVCache
from models.demos.ernie45_d_p.tt.common import COMPUTE_HIFI2, cache_name, shard
from models.demos.ernie45_d_p.tt.embedding import TtEmbedding
from models.demos.ernie45_d_p.tt.moe import TtMoE
from models.demos.ernie45_d_p.tt.ops import TtRMSNorm, TtRope, TtSwiGLU, all_reduce


class TtDecoderLayer:
    def __init__(self, mesh, cfg: ErnieConfig, i: int, w: LayerWeights, rope: TtRope):
        self.i = i
        self.attn_norm = TtRMSNorm(mesh, w.attn_norm, cfg.rms_norm_eps, f"L{i}.attn_norm")
        self.ffn_norm = TtRMSNorm(mesh, w.ffn_norm, cfg.rms_norm_eps, f"L{i}.ffn_norm")
        self.attn = TtAttention(mesh, cfg, i, w, rope)
        self.is_moe = w.is_moe
        self.mlp = TtMoE(mesh, cfg, i, w) if w.is_moe else TtSwiGLU(mesh, w.w_gate, w.w_up, w.w_down, name=f"L{i}/mlp")

    def __call__(self, h, start: int, cache: TtKVCache):
        x = self.attn_norm(h)
        a = self.attn(x, start, cache)
        ttnn.deallocate(x)
        h2 = ttnn.add(h, a)
        ttnn.deallocate(a)
        x = self.ffn_norm(h2)
        if self.is_moe:
            m = self.mlp(x)
        else:
            p = self.mlp(x)
            m = all_reduce(p)
            ttnn.deallocate(p)
        ttnn.deallocate(x)
        out = ttnn.add(h2, m)
        ttnn.deallocate(h2)
        ttnn.deallocate(m)
        return out


class TtErnieModel:
    """Embedding -> 28 decoder layers -> final norm -> (vocab-sharded) LM head for selected rows."""

    def __init__(self, mesh, loader: WeightLoader, cfg: ErnieConfig, layers: list[int] | None = None, lm_head=True):
        self.mesh, self.cfg = mesh, cfg
        self.layer_ids = list(range(cfg.num_hidden_layers)) if layers is None else layers
        t0 = time.time()
        embed = loader.get("model.embed_tokens.weight")
        self.embed = TtEmbedding(mesh, embed)
        self.rope = TtRope(mesh, cfg.head_dim, cfg.rope_theta)
        self.layers = []
        for i in self.layer_ids:
            w = load_layer(loader, cfg, i, dtype=torch.bfloat16)
            self.layers.append(TtDecoderLayer(mesh, cfg, i, w, self.rope))
            del w
            gc.collect()
        self.final_norm = TtRMSNorm(mesh, loader.get("model.norm.weight").float(), cfg.rms_norm_eps, "final_norm")
        self.lm_head = None
        if lm_head:  # tied: logits = h @ embed^T ; vocab-sharded across chips
            self.lm_head = shard(mesh, embed.T.contiguous()[None, None], dim=-1, cache=cache_name("lm_head"))
        self.load_seconds = time.time() - t0

    def new_cache(self, max_seq: int, dtype=ttnn.bfloat16) -> TtKVCache:
        return TtKVCache(self.mesh, self.cfg, max_seq, self.layer_ids, dtype=dtype)

    def prefill_chunk(
        self,
        tokens: torch.Tensor,
        start: int,
        cache: TtKVCache,
        on_layer: Callable[[int, ttnn.Tensor], None] | None = None,
        hidden_in: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """Run one chunk at absolute positions [start, start+len). Returns final-normed hidden [1,1,S,H]."""
        h = self.embed(tokens) if hidden_in is None else hidden_in
        for layer in self.layers:
            h2 = layer(h, start, cache)
            ttnn.deallocate(h)
            h = h2
            if on_layer is not None:
                on_layer(layer.i, h)
        out = self.final_norm(h)
        ttnn.deallocate(h)
        return out

    def logits(self, hidden: ttnn.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """Logits [len(rows), V] (host fp32) for selected rows of the chunk's final hidden state."""
        host = ttnn.to_torch(ttnn.get_device_tensors(hidden)[0])[0, 0]
        sel = host[rows]
        n = sel.shape[0]
        pad = (-n) % 32
        sel = torch.cat([sel, torch.zeros(pad, sel.shape[-1], dtype=sel.dtype)]) if pad else sel
        x = ttnn.from_torch(
            sel[None, None],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        lg = ttnn.linear(x, self.lm_head, compute_kernel_config=COMPUTE_HIFI2)
        out = torch.cat([ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(lg)], dim=-1)[0, 0, :n]
        ttnn.deallocate(x)
        ttnn.deallocate(lg)
        return out
