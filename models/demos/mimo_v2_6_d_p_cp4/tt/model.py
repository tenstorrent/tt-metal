# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2.6-Flash-RL text decoder on the 1x4 mesh with context parallelism CP=4, fully on the device
(embedding -> blocks -> final norm).

Assembled from the per-step device modules the component/swap gates validated (tt/rms_norm, attention, mlp, router,
experts, residual), built exactly as the hooks' component path builds them. The hidden state stays on the device as
chip c's contiguous CP slice [1, 1, S/4, H] bf16 TILE (rows [c*S/4, (c+1)*S/4) of the chunk); the only host input per
chunk is the token ids (sharded the same way).

Block graphs (reference/mimo_ref.py DENSE_GRAPH / MOE_GRAPH), run through run_block (one profiler section per step):
    attn_norm -> attention -> attn_residual (h_mid = in + .) -> ffn_norm
    dense: mlp -> mlp_residual (out = h_mid + .)
    MoE:   router(ffn_norm) -> experts(ffn_norm, router) -> ffn_residual (out = h_mid + .)
The router boundary is (idx, wts): the experts take them directly (no dense -> topk round).

Load-time constants (no per-chunk host work): RoPE cos/sin tables per attention module for every chunk size, in the
chunk-major row order, sliced on the device per chunk; the sliding layers' chunk-0 masks (shared, RingCCL.constant);
the ring-gather buffers (RingCCL); the router's zero / bias tables for max_chunk / 4 rows; the experts' dispatch
tables, and the dispatch / combine modules for every chunk size (prebuilt here); bfp8 expert weights (device cache
under generated/mimo_v2_6_d_p_cp4/tt_cache).
"""

from __future__ import annotations

import os
from typing import Callable

import torch

import ttnn

from .experts import CACHE_ROOT

# Norm steps -> checkpoint weight name (under model.layers.<i>.).
NORM_WEIGHTS = {
    "attn_norm": "input_layernorm.weight",
    "ffn_norm": "post_attention_layernorm.weight",
}


# --------------------------------------------------------------------------------------
# Module builders (the same constructions as bringup/hooks.py's component path)
# --------------------------------------------------------------------------------------


def build_norm(mesh, loader, layer: int, step: str, eps: float = 1e-6):
    from .rms_norm import TtRMSNorm

    return TtRMSNorm(mesh, loader.get(f"model.layers.{layer}.{NORM_WEIGHTS[step]}"), eps=eps)


def build_attention(mesh, ccl, loader, cfg, layer: int, max_seq: int, chunk_sizes):
    """TtFullAttention / TtSlidingAttention (CP=4, TP=1) for one layer: fused qkv dequantized per stored TP-rank slab
    and reassembled in global order, bf16 o_proj, bf16 sink."""
    from models.demos.mimo_v2_6_d_p.reference.mimo_ref import rope_inv_freq
    from models.demos.mimo_v2_6_d_p.reference.weights import qkv_weight

    from .attention import TtFullAttention, TtSlidingAttention

    sliding = cfg.is_sliding(layer)
    hq, hkv, d, dv = cfg.attn_dims(layer)
    p = f"model.layers.{layer}.self_attn."
    wqkv = qkv_weight(loader, p, (hq * d, hkv * d, hkv * dv), torch.float32)
    wo = loader.get(p + "o_proj.weight").float()
    inv_freq = rope_inv_freq(cfg.swa_rope_theta if sliding else cfg.rope_theta, cfg.rope_dim(layer))
    common = (mesh, ccl, wqkv, wo, (hq, hkv, d, dv), inv_freq, max_seq, cfg.attention_value_scale)
    if sliding:
        sink = loader.get(p + "attention_sink_bias").float() if cfg.has_sink(layer) else None
        return TtSlidingAttention(*common, chunk_sizes, cfg.sliding_window, sink)
    assert not cfg.has_sink(layer)
    return TtFullAttention(*common, chunk_sizes)


def build_router(mesh, loader, cfg, layer: int, max_chunk: int):
    """TtRouter (replicated weight, each chip routes its own S/4 rows; tables for max_chunk / 4 rows).
    MIMO_ROUTER_MODE=fused selects moe_grouped_topk for comparison."""
    from .router import TtRouter

    assert cfg.n_group == 1 and cfg.scoring_func == "sigmoid" and cfg.norm_topk_prob
    p = f"model.layers.{layer}.mlp.gate."
    w = loader.get(p + "weight").float()
    b = loader.get(p + "e_score_correction_bias").float()
    rs = cfg.routed_scaling_factor if cfg.routed_scaling_factor is not None else 1.0
    rows = -(-int(max_chunk) // mesh.get_num_devices())
    mode = os.environ.get("MIMO_ROUTER_MODE", "fp32")
    return TtRouter(mesh, w, b, rows, top_k=cfg.num_experts_per_tok, route_scale=rs, mode=mode)


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------


class TtEmbedding:
    """Replicated [V, H] bf16 table (152576 x 4096, 1.25 GB/chip), ROW_MAJOR, cached as a tensorbin.

    Input: uint32 ROW_MAJOR ids [1, 1, S/4] per chip (chip c: ids of its CP slice). Output: chip c's CP slice of the
    hidden state [1, 1, S/4, H] bf16 TILE."""

    def __init__(self, mesh, weight: torch.Tensor, cache: bool = True):
        self.mesh = mesh
        self.vocab, self.hidden = weight.shape
        if cache:
            CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        self.weight = ttnn.as_tensor(
            weight.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            cache_file_name=str(CACHE_ROOT / "embed_bf16") if cache else None,
        )

    def __call__(self, ids: ttnn.Tensor) -> ttnn.Tensor:
        seq = ids.shape[-1]
        ids4 = ttnn.reshape(ids, (1, 1, 1, seq))
        x = ttnn.embedding(ids4, self.weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        return ttnn.reshape(x, (1, 1, seq, self.hidden))


class TtMiMoBlock:
    """One decoder block: the validated step modules, keyed by the reference block graph, run through run_block.
    ctx.state is the layer's TtKVCacheRing."""

    def __init__(self, mesh, ccl, cfg, loader, layer: int, max_seq: int, max_chunk: int, chunk_sizes):
        from models.demos.mimo_v2_6_d_p.reference.mimo_ref import DENSE_GRAPH, MOE_GRAPH

        from .experts import build_experts
        from .mlp import build_mlp
        from .residual import TtResidualAdd

        self.i, self.mesh = layer, mesh
        self.moe = cfg.is_moe(layer)
        eps = cfg.layernorm_epsilon
        self.norms = {k: build_norm(mesh, loader, layer, k, eps) for k in NORM_WEIGHTS}
        self.attn = build_attention(mesh, ccl, loader, cfg, layer, max_seq, chunk_sizes)
        self.add = TtResidualAdd(mesh)
        if self.moe:
            self.router = build_router(mesh, loader, cfg, layer, max_chunk)
            self.experts = build_experts(mesh, loader, cfg, layer, max_chunk)
            for c in chunk_sizes:  # dispatch / combine modules per chunk size, at load
                self.experts._seq_modules(c // mesh.get_num_devices())
        else:
            self.mlp = build_mlp(mesh, loader, layer)
        self.graph = list(MOE_GRAPH if self.moe else DENSE_GRAPH)
        # Residual add + the next RMSNorm in one call (MIMO_FUSE_RESIDUAL_NORM=1, set up by TtMiMoModel): the residual
        # step returns the sum and parks the norm in `stash`; the norm step that follows (ffn_norm, or the next
        # block's attn_norm) returns it instead of recomputing. The graph's boundaries are unchanged.
        self.stash = None
        self.next_norm = None
        self._build_steps()

    def new_cache(self, max_seq: int, chunk: int | None = None):
        return self.attn.new_cache(max_seq, chunk)

    def _build_steps(self):
        """Device fn(ctx, *inputs) per graph step, and after each step the boundaries it was the last reader of (freed
        there; the block input "in" belongs to the caller)."""
        n = self.norms

        def norm(k):
            def fn(ctx, x):
                hit = self.stash.pop(id(x), None) if self.stash is not None else None
                if hit is not None and hit[0] is x:
                    return hit[1]
                return n[k](x)

            return fn

        def residual(next_norm):
            def fn(ctx, a, b):
                nn = next_norm()
                if self.stash is None or nn is None:
                    return self.add(a, b)
                y, t = nn.fused_add(a, b)
                self.stash[id(t)] = (t, y)
                return t

            return fn

        steps = {k: norm(k) for k in n}
        steps.update(
            attention=lambda ctx, x: self.attn(x, ctx.start, ctx.state),
            attn_residual=residual(lambda: n["ffn_norm"]),
        )
        if self.moe:

            def router(ctx, x):
                dense, idx, wts = self.router(x)
                ttnn.deallocate(dense)
                return (idx, wts)

            steps.update(
                router=router,
                experts=lambda ctx, x, r: self.experts(x, idx=r[0], wts=r[1]),
                ffn_residual=residual(lambda: self.next_norm),
            )
        else:
            steps.update(mlp=lambda ctx, x: self.mlp(x), mlp_residual=residual(lambda: self.next_norm))
        assert set(steps) == {st.name for st in self.graph}, (sorted(steps), [st.name for st in self.graph])
        last_use = {}
        for k, st in enumerate(self.graph):
            for name in st.inputs:
                last_use[name] = k
        self.overrides = {}
        for k, st in enumerate(self.graph):
            dead = tuple(dict.fromkeys(nm for nm in st.inputs if nm != "in" and last_use[nm] == k))
            self.overrides[st.name] = self._freeing(steps[st.name], st.inputs, dead)

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

    def __call__(self, x: ttnn.Tensor, start: int, cache) -> ttnn.Tensor:
        """x: chip c's CP slice [1, 1, S/4, H] (not freed); positions [start, start + S). Returns the same layout."""
        from models.demos.common.bringup.reference.interface import Ctx, run_block

        def missing(name):
            raise KeyError(f"no device step {name}")

        ctx = Ctx(self.i, start, x.shape[-2] * self.mesh.get_num_devices(), cache, {})
        return run_block(self.graph, missing, ctx, x, overrides=self.overrides)


class TtMiMoModel:
    """Embedding + decoder blocks + final norm on the device (CP=4). `prefill_chunk` runs one chunk against per-layer
    ring caches (new_caches) and returns the last block's hidden state (CP slices)."""

    def __init__(
        self,
        mesh,
        model_path: str,
        max_seq: int,
        chunk_sizes,
        layers: list[int] | None = None,
    ):
        from models.demos.mimo_v2_6_d_p.reference.mimo_ref import MiMoConfig
        from models.demos.mimo_v2_6_d_p.reference.weights import WeightLoader

        from .ccl import RingCCL
        from .rms_norm import TtRMSNorm

        self.mesh = mesh
        self.cfg = cfg = MiMoConfig.from_json(os.path.join(model_path, "config.json"))
        loader = WeightLoader(model_path)
        self.layer_ids = list(range(cfg.num_hidden_layers)) if layers is None else list(layers)
        self.chunk_sizes = sorted({int(c) for c in chunk_sizes})
        self.max_seq = int(max_seq)
        self.max_chunk = max(self.chunk_sizes)
        self.ccl = RingCCL(mesh)
        self.embed = TtEmbedding(mesh, loader.get("model.embed_tokens.weight"))
        self.blocks = [
            TtMiMoBlock(mesh, self.ccl, cfg, loader, i, self.max_seq, self.max_chunk, self.chunk_sizes)
            for i in self.layer_ids
        ]
        if os.environ.get("MIMO_FUSE_RESIDUAL_NORM", "1") != "0":
            stash = {}
            for k, blk in enumerate(self.blocks):
                blk.stash = stash
                nxt = self.blocks[k + 1] if k + 1 < len(self.blocks) else None
                blk.next_norm = nxt.norms["attn_norm"] if nxt is not None and nxt.i == blk.i + 1 else None
        self.final_norm = TtRMSNorm(mesh, loader.get("model.norm.weight"), eps=cfg.layernorm_epsilon)

    def new_caches(self, max_seq: int, chunk: int | None = None) -> dict:
        """Per-layer ring KV caches (chunk-major layout bound to the first chunk run, or to `chunk`)."""
        return {b.i: b.new_cache(max_seq, chunk) for b in self.blocks}

    def ids_to_device(self, tokens: torch.Tensor) -> ttnn.Tensor:
        """Host ids [S] -> uint32 [1, 1, S/4] per chip (chip c: its CP slice). Harness / per-chunk input only."""
        n = self.mesh.get_num_devices()
        assert tokens.numel() % (n * ttnn.TILE_SIZE) == 0, f"chunk {tokens.numel()} must split into {n} tile slices"
        return ttnn.from_torch(
            tokens.reshape(1, 1, -1).to(torch.int64).to(torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=2),
        )

    def prefill_chunk(
        self,
        ids: ttnn.Tensor,
        start: int,
        caches: dict,
        on_layer: Callable[[int], None] | None = None,
    ) -> ttnn.Tensor:
        """ids: device uint32 [1, 1, S/4] per chip (CP slices). Positions [start, start+S). on_layer(i) is called
        after block i is enqueued."""
        h = self.embed(ids)
        for blk in self.blocks:
            h2 = blk(h, start, caches[blk.i])
            ttnn.deallocate(h)
            h = h2
            if on_layer is not None:
                on_layer(blk.i)
        return h
