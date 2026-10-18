# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 26B-A4B all-device prefill model on the 8x4 Galaxy mesh.

The hidden state is a replicated [1, 1, S, H] bf16 device tensor from the embedding to the final norm. Every block
runs through ``run_block`` over the reference's BLOCK_GRAPH with one device fn per step: the same modules the component
and swap gates validated (tt/rms_norm, attention, mlp, router, experts, residual). Boundaries are freed after their last
reader (a schedule built once at load); the block input is never freed here (the caller owns it).

Load-time constants: the embedding table with sqrt(H) folded in (bf16, ROW_MAJOR, replicated), RoPE tables and the
global layers' identity page table (built per cache size in ``new_caches``, outside the forward).
"""

import os
import time

import torch

import ttnn
from models.demos.gemma4_a4b_glx.reference._bringup_interface import Ctx, run_block
from models.demos.gemma4_a4b_glx.reference.gemma4_ref import (
    BLOCK_GRAPH,
    PREFIX,
    Gemma4TextConfig,
    WeightLoader,
    rope_inv_freq,
)

from .attention import TP_AXIS, TtGlobalAttention, TtSlidingAttention
from .experts import TtExperts
from .mlp import TtMLP
from .residual import TtResidualAdd, TtScaledResidualAdd
from .rms_norm import TtRMSNorm
from .router import TtRouter

NORM_WEIGHTS = {
    "attn_norm": "input_layernorm.weight",
    "post_attn_norm": "post_attention_layernorm.weight",
    "ffn_norm": "pre_feedforward_layernorm.weight",
    "post_mlp_norm": "post_feedforward_layernorm_1.weight",
    "moe_norm": "pre_feedforward_layernorm_2.weight",
    "post_moe_norm": "post_feedforward_layernorm_2.weight",
    "post_ffn_norm": "post_feedforward_layernorm.weight",
}


def build_attention(mesh, cfg, loader, layer):
    """Sliding or global attention of `layer` (loads only that layer's self_attn tensors)."""
    sliding = cfg.is_sliding(layer)
    p = f"{PREFIX}layers.{layer}.self_attn."
    inv_freq, _ = rope_inv_freq(cfg, sliding=sliding)
    common = dict(rope_inv_freq=inv_freq, eps=cfg.rms_norm_eps, scale=1.0)
    w = {n: loader.get(p + f"{n}.weight") for n in ("q_proj", "k_proj", "o_proj", "q_norm", "k_norm")}
    if not sliding:
        assert cfg.attention_k_eq_v and not loader.has(p + "v_proj.weight"), "global layer expected without v_proj"
        hkv, d = cfg.attn_dims(layer)
        return TtGlobalAttention(
            mesh,
            w["q_proj"],
            w["k_proj"],
            w["o_proj"],
            w["q_norm"],
            w["k_norm"],
            num_heads=cfg.num_attention_heads,
            num_kv_heads=hkv,
            head_dim=d,
            **common,
        )
    return TtSlidingAttention(
        mesh,
        w["q_proj"],
        w["k_proj"],
        loader.get(p + "v_proj.weight"),
        w["o_proj"],
        w["q_norm"],
        w["k_norm"],
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        window=cfg.sliding_window,
        **common,
    )


def build_router(mesh, cfg, loader, layer):
    p = f"{PREFIX}layers.{layer}.router."
    return TtRouter(
        mesh,
        loader.get(p + "proj.weight"),
        loader.get(p + "scale"),
        loader.get(p + "per_expert_scale"),
        top_k=cfg.top_k_experts,
        eps=cfg.rms_norm_eps,
    )


def build_experts(mesh, loader, layer):
    p = f"{PREFIX}layers.{layer}.experts."
    return TtExperts(mesh, loader.get(p + "gate_up_proj"), loader.get(p + "down_proj"))


def build_mlp(mesh, loader, layer):
    p = f"{PREFIX}layers.{layer}.mlp."
    return TtMLP(mesh, *(loader.get(p + f"{n}_proj.weight") for n in ("gate", "up", "down")))


def build_norm(mesh, cfg, loader, layer, step):
    return TtRMSNorm(mesh, loader.get(f"{PREFIX}layers.{layer}.{NORM_WEIGHTS[step]}"), eps=cfg.rms_norm_eps)


def layer_scalar(loader, layer) -> float:
    return loader.get(f"{PREFIX}layers.{layer}.layer_scalar").float().reshape(-1)[0].item()


def _free(t):
    if isinstance(t, ttnn.Tensor) and t.is_allocated():
        ttnn.deallocate(t)


def _free_schedule(steps):
    """After step k, the boundaries whose last reader is step k (never "in": the caller owns the block input)."""
    last = {}
    for k, s in enumerate(steps):
        for name in s.inputs:
            last[name] = k
    out = [[] for _ in steps]
    for name, k in last.items():
        if name != "in":
            out[k].append(name)
    return out


class TtGemma4Block:
    """One decoder layer: run_block(BLOCK_GRAPH) with device step fns. ctx.state is the layer's TtKVCache."""

    def __init__(self, mesh, cfg, loader, layer):
        self.i = layer
        self.sliding = cfg.is_sliding(layer)
        self.attention = build_attention(mesh, cfg, loader, layer)
        self.norms = {name: build_norm(mesh, cfg, loader, layer, name) for name in NORM_WEIGHTS}
        self.mlp = build_mlp(mesh, loader, layer)
        self.router = build_router(mesh, cfg, loader, layer)
        self.experts = build_experts(mesh, loader, layer)
        self.add = TtResidualAdd(mesh)
        self.scaled_add = TtScaledResidualAdd(mesh, layer_scalar(loader, layer))
        self.steps = list(BLOCK_GRAPH)
        self.frees = _free_schedule(self.steps)
        self.fns = self._step_fns()

    def _router(self, ctx, x):
        dense, ids = self.router(x)
        ttnn.deallocate(ids)
        return dense

    def _step_fns(self):
        fns = {name: (lambda ctx, x, m=m: m(x)) for name, m in self.norms.items()}
        fns.update(
            attention=lambda ctx, x: self.attention(x, ctx.state, ctx.start, ctx.extra.get("end")),
            attn_residual=lambda ctx, a, b: self.add(a, b),
            mlp=lambda ctx, x: self.mlp(x),
            router=self._router,
            experts=lambda ctx, x, r: self.experts(x, r),
            ffn_combine=lambda ctx, a, b: self.add(a, b),
            ffn_residual=lambda ctx, a, b: self.scaled_add(a, b),
        )
        return fns

    def __call__(self, h, start, cache, end=None):
        """h: [1, 1, S, H] bf16 replicated at positions [start, start+S); rows at and past `end` (actual_end) are pad.
        Returns the block output (h is kept)."""
        env = {}
        k = {"step": 0}

        def rec(name, t):  # called after every boundary write: free what the step just finished reading
            if name == "in":
                env["in"] = t
                return
            env[name] = t
            for dead in self.frees[k["step"]]:
                _free(env.pop(dead, None))
            k["step"] += 1

        ctx = Ctx(self.i, start, h.shape[-2], cache, extra={"end": end})
        return run_block(self.steps, lambda name: self.fns[name], ctx, h, rec)


class TtGemma4Model:
    """Embedding, the requested blocks and the final norm, all on the device. ``layers`` may be a subset."""

    def __init__(self, mesh, model_path, layers=None):
        t0 = time.time()
        self.mesh = mesh
        self.cfg = Gemma4TextConfig.from_json(os.path.join(model_path, "config.json"))
        self.loader = WeightLoader(model_path)
        layers = list(range(self.cfg.num_hidden_layers)) if layers is None else list(layers)
        table = self.loader.get(PREFIX + "embed_tokens.weight").float()
        scale = torch.tensor(self.cfg.hidden_size**0.5, dtype=torch.float32)
        self.embed_table = ttnn.from_torch(
            (table * scale).reshape(1, 1, *table.shape),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        vocab = table.shape[0]
        assert vocab & (vocab - 1) == 0, f"vocab {vocab} not a power of two (the pad id mask needs one)"
        self.id_mask = vocab - 1
        del table
        self.norm = TtRMSNorm(mesh, self.loader.get(PREFIX + "norm.weight"), eps=self.cfg.rms_norm_eps)
        self.blocks = [TtGemma4Block(mesh, self.cfg, self.loader, i) for i in layers]
        self.load_seconds = time.time() - t0

    def new_caches(self, max_seq: int) -> dict:
        """Empty per-layer device KV caches for max_seq tokens; the RoPE / page tables for max_seq are built here."""
        caches = {}
        for b in self.blocks:
            b.attention.ensure_rope(max_seq)
            caches[b.i] = b.attention.make_cache(max_seq)
        return caches

    def embed(self, ids):
        """ids: [1, 1, 1, S] uint32 ROW_MAJOR replicated -> [1, 1, S, H] bf16 TILE replicated (sqrt(H) folded in).

        The server pads a chunk past actual_end with 0xFFFFFFFF; ids & (vocab - 1) keeps every real id and maps the pad
        id to a valid row (vocab - 1), so the lookup never reads past the table. Pad rows only feed pad rows (causal)
        and attention writes zeros to their cache rows."""
        safe = ttnn.bitwise_and(ids, self.id_mask)
        e = ttnn.embedding(
            safe, self.embed_table, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(safe)
        return ttnn.reshape(e, (1, 1, ids.shape[-1], e.shape[-1]))

    def final_norm(self, h):
        return self.norm(h)


def cache_to_torch(cache, kv_repeat: int, length: int) -> dict:
    """Device TtKVCache -> host {key, value} position-major [length, Hkv * D] (mesh row 0; each KV head once), the
    reference's state_tensors layout."""
    cols = cache.mesh.shape[TP_AXIS]

    def read(t):
        parts = [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)[:cols]]  # row 0, columns 0..cols-1
        full = torch.cat(parts, dim=1)[0, :, :length].float()
        full = full[::kv_repeat]  # [Hkv, length, D]
        return full.transpose(0, 1).reshape(length, -1)

    return {"key": read(cache.k), "value": read(cache.v)}
