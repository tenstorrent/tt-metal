# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Whole-model prefill on the 8x4 mesh (outline pattern: gpt_oss_d_p/tt/model.py).

    tokens -> TtEmbedding -> N x TtDecoderLayer (hybrid dispatch by config.layer_types)
           -> final TtRMSNorm (-> TtLMHead only when logits are requested)

Per-layer weights come from a *weight source* (random state dict for tests, the real checkpoint for
P1) as HF-named dicts relative to ``layers.{i}.``; each layer's tilized weights are cached on disk
(``tt/common.weight_cache_dir``) and a per-layer marker file says the layer is complete, so a warm
start reads no safetensors at all.
"""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.qwen_3_8_27b.config import PrefillSpec, Qwen38Config, ttnn_dtype
from models.demos.qwen_3_8_27b.tt.common import residual_dtype
from models.demos.qwen_3_8_27b.tt.context import PrefillCtx
from models.demos.qwen_3_8_27b.tt.embedding import TtEmbedding, TtLMHead
from models.demos.qwen_3_8_27b.tt.gdn import read_gdn_state
from models.demos.qwen_3_8_27b.tt.kv_cache import allocate_caches, read_attn_kv
from models.demos.qwen_3_8_27b.tt.layer import TtDecoderLayer
from models.demos.qwen_3_8_27b.tt.rms_norm import TtRMSNorm
from models.demos.qwen_3_8_27b.tt.rope import TtRope
from models.demos.qwen_3_8_27b.tt.sdpa import cache_attn_mode, cache_read_mask


class StateDictWeights:
    """Weight source over an in-memory HF-named text-model state dict (random-weight tests)."""

    def __init__(self, sd: dict, tag: str):
        self.sd, self.tag = sd, tag

    def embed(self):
        return self.sd["embed_tokens.weight"]

    def norm(self):
        return self.sd["norm.weight"]

    def lm_head(self):
        return self.sd.get("lm_head.weight")

    def layer(self, i):
        p = f"layers.{i}."
        return {k[len(p) :]: v for k, v in self.sd.items() if k.startswith(p)}


class CheckpointWeights:
    """Weight source over the real safetensors checkpoint (P1)."""

    tag = "real"

    def __init__(self, reader):
        self.r = reader

    def embed(self):
        return self.r.text("embed_tokens.weight")

    def norm(self):
        return self.r.text("norm.weight")

    def lm_head(self):
        return self.r.lm_head()

    def layer(self, i):
        return self.r.layer(i)


class TtQwen38Model:
    def __init__(
        self,
        mesh_config,
        ccl,
        cfg: Qwen38Config,
        spec: PrefillSpec,
        weights,
        *,
        num_layers: int | None = None,
        cache: Path | None = None,
        with_lm_head: bool = False,
    ):
        self.mc, self.ccl, self.cfg, self.spec = mesh_config, ccl, cfg, spec
        self.num_layers = num_layers or cfg.num_hidden_layers
        # tensor cache is per weight source (random vs real must never share files)
        # versioned: v2 = fp32 norm gains (v1 files held bf16 gains under other names)
        self.cache = None if cache is None else Path(cache) / f"{weights.tag}_v2"
        if self.cache is not None:
            self.cache.mkdir(parents=True, exist_ok=True)

        def done(name):
            return self.cache is not None and (self.cache / f"{name}.done").exists()

        def mark(name):
            if self.cache is not None:
                (self.cache / f"{name}.done").touch()

        self.embedding = TtEmbedding(mesh_config, None if done("embed") else weights.embed(), cache=self.cache)
        mark("embed")
        self.layers = []
        for i in range(self.num_layers):
            sd = None if done(f"layers.{i}") else weights.layer(i)
            self.layers.append(TtDecoderLayer(mesh_config, ccl, cfg, sd, i, spec, cache=self.cache))
            mark(f"layers.{i}")
            if sd is not None:
                logger.info(f"layer {i}/{self.num_layers} ({cfg.layer_types[i]}) loaded")
            del sd
        self.norm = TtRMSNorm(
            mesh_config, None if done("norm") else weights.norm(), cfg.rms_norm_eps, cache=self.cache, name="final_norm"
        )
        mark("norm")
        self.lm_head = None
        if with_lm_head:
            self.lm_head = TtLMHead(
                mesh_config,
                None if done("lm_head") else weights.lm_head(),
                weight_dtype=ttnn_dtype(spec.weight_dtype_default),
                cache=self.cache,
            )
            mark("lm_head")
        self.rope = TtRope(mesh_config, cfg)

    # ---- caches ----
    @property
    def attn_layers(self):
        return [i for i in range(self.num_layers) if self.cfg.is_full_attention(i)]

    @property
    def gdn_layers(self):
        return [i for i in range(self.num_layers) if not self.cfg.is_full_attention(i)]

    def allocate_caches(self, capacity: int, num_users: int = 1):
        return allocate_caches(
            self.mc.mesh_device,
            num_attn_layers=len(self.attn_layers),
            gdn_layers=self.gdn_layers,
            max_seq_len=capacity,
            head_dim=self.cfg.head_dim,
            num_users=num_users,
            cache_dtype=ttnn_dtype(self.spec.kv_cache_dtype),
        )

    # ---- forward ----
    def forward(self, tokens, ctx: PrefillCtx, on_layer=None):
        """tokens: device ``[1, s_local]`` uint32 per chip (``TtEmbedding.make_tokens``). Returns the
        final-normed hidden ``[1, 1, s_local, H]`` (SP-sharded, TP-replicated)."""
        s_local = ctx.tokens // self.mc.sp
        ctx.cos, ctx.sin = self.rope.tables(ctx.start, s_local)
        if ctx.start > 0 and self.attn_layers and cache_attn_mode() == "masked" and ctx.cache_mask is None:
            ctx.cache_mask = build_cache_mask(self.mc, ctx.start, ctx.tokens)
        x = self.embedding(tokens)
        if residual_dtype() != x.dtype:
            x32 = ttnn.typecast(x, residual_dtype())
            ttnn.deallocate(x)
            x = x32
        for layer in self.layers:
            x = layer(x, ctx, self.rope)
            if on_layer is not None:  # diagnostics: per-layer residual stream
                on_layer(layer.layer_idx, x)
        out = self.norm(x)
        ttnn.deallocate(x)
        ttnn.deallocate(ctx.cos)
        ttnn.deallocate(ctx.sin)
        if ctx.cache_mask is not None:
            ttnn.deallocate(ctx.cache_mask)
            ctx.cache_mask = None
        return out

    def logits(self, x):
        assert self.lm_head is not None, "built without an LM head"
        return self.lm_head(x)

    # ---- read-back in the golden trace's layout ----
    def read_layer_state(self, caches, layer_idx: int, *, n_tokens: int, period: int, user_id: int = 0):
        """-> (k, v) for attention layers ``[1, n_kv, n_tokens, D]``; (recurrent_state, conv_state) for GDN."""
        if self.cfg.is_full_attention(layer_idx):
            return read_attn_kv(
                caches,
                self.mc.mesh_device,
                user_id=user_id,
                attn_ordinal=self.attn_layers.index(layer_idx),
                n_tokens=n_tokens,
                period=period,
            )
        g = self.layers[layer_idx].mixer
        return read_gdn_state(caches.gdn_state(user_id, layer_idx), self.mc, g.nv, g.qd, g.vd)


def build_cache_mask(mesh_config, start: int, tokens: int):
    m = cache_read_mask(mesh_config.sp, start, tokens)
    return ttnn.from_torch(
        m,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_config.mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_config.shard(0, None),
    )


def gather_hidden(x, mesh_config) -> torch.Tensor:
    """SP-sharded, TP-replicated ``[1,1,s_local,H]`` -> host ``[1, T, H]`` (TP column 0)."""
    full = ttnn.to_torch(x, mesh_composer=mesh_config.compose(2, 3)).float()
    return full[..., : full.shape[-1] // mesh_config.tp][0]
