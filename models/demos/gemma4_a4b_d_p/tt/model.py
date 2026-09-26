# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 26B-A4B text decoder on a 1x4 mesh, fully on the device (serving path: embedding -> 30 blocks).

Built from the per-step device modules the component/swap gates validated (rms_norm, attention, mlp, router, experts,
residual). The hidden state stays on the device as a replicated [1, 1, S, H] bf16 TILE tensor between steps; the
only host input per chunk is the engine's uint32 token tensor.

Block (as reference/gemma4_ref.py BLOCK_GRAPH):
    attn_norm -> attention -> post_attn_norm -> h_mid = in + .
    ffn_norm -> mlp -> post_mlp_norm                      (dense branch)
    router(h_mid) ; moe_norm(h_mid) -> experts -> post_moe_norm   (MoE branch)
    ffn_out = post_ffn_norm(mlp_post + moe_post) ; out = (h_mid + ffn_out) * layer_scalar

Load-time constants (no per-chunk host work): RoPE cos/sin for max_seq per RoPE type (device-sliced per chunk), the
embedding table with sqrt(H) folded in, the global caches' identity page tables (device-sliced per chunk).
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Callable

import torch

import ttnn
from models.demos.gemma4_a4b_d_p.tt.attention import TtGlobalAttention, TtSlidingAttention, rope_tables
from models.demos.gemma4_a4b_d_p.tt.experts import CACHE_ROOT, TtExperts
from models.demos.gemma4_a4b_d_p.tt.mlp import TtDenseMLP
from models.demos.gemma4_a4b_d_p.tt.residual import TtResidualAdd
from models.demos.gemma4_a4b_d_p.tt.rms_norm import TtRMSNorm
from models.demos.gemma4_a4b_d_p.tt.router import TtRouter

_NORMS = {
    "attn_norm": "input_layernorm.weight",
    "post_attn_norm": "post_attention_layernorm.weight",
    "ffn_norm": "pre_feedforward_layernorm.weight",
    "post_mlp_norm": "post_feedforward_layernorm_1.weight",
    "moe_norm": "pre_feedforward_layernorm_2.weight",
    "post_moe_norm": "post_feedforward_layernorm_2.weight",
    "post_ffn_norm": "post_feedforward_layernorm.weight",
}


class TtEmbedding:
    """Replicated [V, H] bf16 table with the Gemma sqrt(H) scale folded in (1.48 GB/chip), ROW_MAJOR.

    Input: the engine's uint32 ROW_MAJOR [1, 1, S] tensor. Pad ids (0xFFFFFFFF past actual_end) are masked with
    `& (V - 1)` (V = 2^18), which leaves real ids unchanged and keeps the embedding read in bounds; pad rows come
    after every real row, so causal attention never lets them reach a real position."""

    def __init__(self, mesh, weight: torch.Tensor, scale: float, cache: bool = True):
        self.mesh = mesh
        self.vocab, self.hidden = weight.shape
        assert self.vocab & (self.vocab - 1) == 0, "pad masking assumes a power-of-two vocab"
        self.weight = ttnn.as_tensor(
            (weight.float() * scale).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            cache_file_name=str(CACHE_ROOT / "embed_scaled_bf16") if cache else None,
        )

    def __call__(self, ids: ttnn.Tensor) -> ttnn.Tensor:
        seq = ids.shape[-1]
        it = ttnn.to_layout(ids, ttnn.TILE_LAYOUT)
        masked = ttnn.bitwise_and(it, self.vocab - 1)
        ttnn.deallocate(it)
        rm = ttnn.to_layout(masked, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(masked)
        rm4 = ttnn.reshape(rm, (1, 1, 1, seq))
        x = ttnn.embedding(rm4, self.weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        ttnn.deallocate(rm)
        return ttnn.reshape(x, (1, 1, seq, self.hidden))


class TtGemma4Block:
    def __init__(self, mesh, cfg, loader, layer: int, max_chunk: int, rope: dict):
        from models.demos.gemma4_a4b_d_p.reference.gemma4_ref import PREFIX, rope_inv_freq

        self.i, self.mesh = layer, mesh
        self.sliding = cfg.is_sliding(layer)
        p = f"{PREFIX}layers.{layer}."
        eps = cfg.rms_norm_eps
        self.norms = {k: TtRMSNorm(mesh, loader.get(p + n), eps=eps) for k, n in _NORMS.items()}
        a = p + "self_attn."
        names = {"wq": "q_proj", "wk": "k_proj", "wo": "o_proj", "q_norm": "q_norm", "k_norm": "k_norm"}
        if self.sliding:
            names["wv"] = "v_proj"
        w = SimpleNamespace(**{k: loader.get(a + n + ".weight").float() for k, n in names.items()})
        inv_freq, _ = rope_inv_freq(cfg, self.sliding)
        if self.sliding:
            self.attn = TtSlidingAttention(mesh, cfg, w, inv_freq, cfg.sliding_window, eps=eps)
        else:
            assert cfg.attention_k_eq_v and not loader.has(a + "v_proj.weight")
            self.attn = TtGlobalAttention(mesh, cfg, w, inv_freq, eps=eps)
        self.attn.set_rope_tables(*rope[self.sliding])
        m = p + "mlp."
        self.mlp = TtDenseMLP(mesh, *(loader.get(m + n + ".weight") for n in ("gate_proj", "up_proj", "down_proj")))
        r = p + "router."
        self.router = TtRouter(
            mesh, loader.get(r + "proj.weight"), loader.get(r + "scale"), loader.get(r + "per_expert_scale")
        )
        e = p + "experts."
        self.experts = TtExperts(
            mesh, layer, loader.get(e + "gate_up_proj"), loader.get(e + "down_proj"), max_seq_len=max_chunk
        )
        self.add = TtResidualAdd(mesh)
        scalar = float(loader.get(p + "layer_scalar").float().reshape(-1)[0].item())
        self.out_add = TtResidualAdd(mesh, scale=scalar)

        self._build_steps()

    def _build_steps(self):
        """Device fn(ctx, *inputs) per block-graph step (reference/gemma4_ref.py BLOCK_GRAPH), and after each step the
        boundaries it was the last reader of (freed there). ctx.state is the layer's attention cache; ctx.extra may
        carry kv_sink. The router boundary is (idx, wts): the experts take them directly (no dense -> topk)."""
        from models.demos.gemma4_a4b_d_p.reference.gemma4_ref import BLOCK_GRAPH

        n = self.norms

        def norm(k):
            return lambda ctx, x: n[k](x)

        def router(ctx, x):
            dense, idx, wts = self.router(x)
            ttnn.deallocate(dense)
            return (idx, wts)

        self.steps = {k: norm(k) for k in n}
        self.steps.update(
            attention=lambda ctx, x: self.attn(x, ctx.start, ctx.state, kv_sink=ctx.extra.get("kv_sink")),
            attn_residual=lambda ctx, a, b: self.add(a, b),
            mlp=lambda ctx, x: self.mlp(x),
            router=router,
            experts=lambda ctx, x, r: self.experts(x, idx=r[0], wts=r[1]),
            ffn_combine=lambda ctx, a, b: self.add(a, b),
            ffn_residual=lambda ctx, a, b: self.out_add(a, b),
        )
        self.graph = list(BLOCK_GRAPH)
        last_use = {}
        for k, st in enumerate(self.graph):
            for name in st.inputs:
                last_use[name] = k
        assert set(self.steps) == {st.name for st in self.graph}
        self.overrides = {}
        for k, st in enumerate(self.graph):
            dead = tuple(dict.fromkeys(nm for nm in st.inputs if nm != "in" and last_use[nm] == k))
            self.overrides[st.name] = self._freeing(self.steps[st.name], st.inputs, dead)

    @staticmethod
    def _freeing(fn, inputs, dead):
        if not dead:
            return fn
        pos = [inputs.index(nm) for nm in dead]

        def wrapped(ctx, *args):
            y = fn(ctx, *args)
            for p in pos:
                for t in args[p] if isinstance(args[p], tuple) else (args[p],):
                    ttnn.deallocate(t)
            return y

        return wrapped

    def ctx(self, start: int, seq: int, cache, kv_sink=None):
        from models.demos.common.bringup.reference.interface import Ctx

        return Ctx(self.i, start, seq, cache, {"kv_sink": kv_sink} if kv_sink is not None else {})

    def __call__(self, x: ttnn.Tensor, start: int, cache, kv_sink=None) -> ttnn.Tensor:
        """x: replicated [1, 1, S, H] block input (not freed). Runs the block graph through run_block (one profiler
        section per step); every intermediate is freed after its last reader."""
        from models.demos.common.bringup.reference.interface import run_block

        def missing(name):
            raise KeyError(f"no device step {name}")

        return run_block(self.graph, missing, self.ctx(start, x.shape[-2], cache, kv_sink), x, overrides=self.overrides)


class TtGemma4Model:
    """Embedding + all decoder blocks on the device. `prefill_chunk` runs one chunk against per-layer attention caches
    (tt.attention.TtKVCacheSliding / TtKVCacheGlobal) and returns the last block's hidden state [1, 1, S, H]."""

    def __init__(self, mesh, model_path: str, max_seq: int, max_chunk: int, layers: list[int] | None = None):
        from models.demos.gemma4_a4b_d_p.reference.gemma4_ref import (
            PREFIX,
            Gemma4TextConfig,
            WeightLoader,
            rope_inv_freq,
        )

        self.mesh = mesh
        self.cfg = cfg = Gemma4TextConfig.from_json(os.path.join(model_path, "config.json"))
        loader = WeightLoader(model_path)
        self.layer_ids = list(range(cfg.num_hidden_layers)) if layers is None else list(layers)
        self.max_seq = -(-max_seq // 64) * 64
        self.embed = TtEmbedding(mesh, loader.get(PREFIX + "embed_tokens.weight"), cfg.hidden_size**0.5)
        self.rope = {}
        for sliding in (True, False):
            inv_freq, _ = rope_inv_freq(cfg, sliding)
            cos, sin = rope_tables(inv_freq, 0, self.max_seq)
            self.rope[sliding] = tuple(self._replicate(t[None, None]) for t in (cos, sin))
        self.blocks = [TtGemma4Block(mesh, cfg, loader, i, max_chunk, self.rope) for i in self.layer_ids]
        self.final_norm = TtRMSNorm(mesh, loader.get(PREFIX + "norm.weight"), eps=cfg.rms_norm_eps)

    def _replicate(self, t):
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def prefill_chunk(
        self,
        ids: ttnn.Tensor,
        start: int,
        caches: dict,
        on_layer: Callable[[int], None] | None = None,
        kv_sink_of: Callable[[int], Callable] | None = None,
    ) -> ttnn.Tensor:
        """ids: device uint32 [1, 1, S] (engine layout). Positions [start, start+S). on_layer(i) is called after block i
        is enqueued; kv_sink_of(i) returns the block's kv_sink(k, v) (or None)."""
        h = self.embed(ids)
        for blk in self.blocks:
            h2 = blk(h, start, caches[blk.i], kv_sink=kv_sink_of(blk.i) if kv_sink_of else None)
            ttnn.deallocate(h)
            h = h2
            if on_layer is not None:
                on_layer(blk.i)
        return h


def new_attention_caches(mesh, cfg, layers, max_seq: int) -> dict:
    """Per-layer device attention caches for one user (sliding: 8 x 256 by head; global: 2 x 512, each on 2 chips;
    global page-table slices are cut on the device)."""
    from models.demos.gemma4_a4b_d_p.tt.attention import TtKVCacheGlobal, TtKVCacheSliding

    seq = -(-max_seq // 64) * 64
    caches = {}
    for i in layers:
        hkv, d = cfg.attn_dims(i)
        if cfg.is_sliding(i):
            caches[i] = TtKVCacheSliding(mesh, hkv, d, seq)
        else:
            caches[i] = TtKVCacheGlobal(mesh, hkv, d, seq)
            caches[i].device_page_slices = True
    return caches
