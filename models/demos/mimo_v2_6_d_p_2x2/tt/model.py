# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2.6-Flash-RL text decoder on the 2x2 mesh: module builders shared by the component hooks and (later) the
all-device model (TtMiMoModel, below), so both paths construct identical modules."""

from __future__ import annotations

# Norm steps -> checkpoint weight name (under model.layers.<i>.).
NORM_WEIGHTS = {
    "attn_norm": "input_layernorm.weight",
    "ffn_norm": "post_attention_layernorm.weight",
}


def build_norm(mesh, loader, layer: int, step: str, eps: float = 1e-6):
    """TtRMSNorm (replicated on all 4 chips, HiFi4 + fp32 acc, plain w) for one layer's norm step."""
    from models.demos.mimo_v2_6_d_p_2x2.tt.rms_norm import TtRMSNorm

    return TtRMSNorm(mesh, loader.get(f"model.layers.{layer}.{NORM_WEIGHTS[step]}"), eps=eps)


def build_attention(mesh, loader, cfg, layer: int, max_seq: int):
    """TtFullAttention / TtSlidingAttention (TP=4 over the 2x2 mesh) for one layer: fused qkv dequantized per TP rank,
    bf16 o_proj; sliding layers add the window and the per-head sink (chip d holds sink values 16d..16d+15)."""
    import torch

    from models.demos.mimo_v2_6_d_p.reference.mimo_ref import rope_inv_freq
    from models.demos.mimo_v2_6_d_p.reference.weights import qkv_weight
    from models.demos.mimo_v2_6_d_p_2x2.tt.attention import TtFullAttention, TtSlidingAttention

    hq, hkv, d, dv = cfg.attn_dims(layer)
    p = f"model.layers.{layer}.self_attn."
    wqkv = qkv_weight(loader, p, (hq * d, hkv * d, hkv * dv), torch.float32)
    wo = loader.get(p + "o_proj.weight").float()
    sliding = cfg.is_sliding(layer)
    inv_freq = rope_inv_freq(cfg.swa_rope_theta if sliding else cfg.rope_theta, cfg.rope_dim(layer))
    dims, vs = (hq, hkv, d, dv), cfg.attention_value_scale
    if sliding:
        sink = loader.get(p + "attention_sink_bias").float() if cfg.has_sink(layer) else None
        return TtSlidingAttention(mesh, wqkv, wo, dims, inv_freq, max_seq, vs, cfg.sliding_window, sink)
    assert not cfg.has_sink(layer)
    return TtFullAttention(mesh, wqkv, wo, dims, inv_freq, max_seq, vs)


def new_kv_cache(mesh, cfg, layer: int, max_seq: int):
    """Empty device KV cache for one layer (K 192 wide, V 128): full layers 4 KV heads (head d on chip d = 2*row + col,
    paged-shaped, identity page table resident), sliding layers 8 KV heads (heads 2d, 2d+1 on chip d, contiguous)."""
    from models.demos.mimo_v2_6_d_p_2x2.tt.attention import TtKVCacheFull, TtKVCacheSliding

    _, hkv, d, dv = cfg.attn_dims(layer)
    if cfg.is_sliding(layer):
        return TtKVCacheSliding(mesh, hkv, d, dv, max_seq)
    return TtKVCacheFull(mesh, hkv, d, dv, max_seq)


def build_mlp(mesh, loader, layer: int):
    """TtDenseMLP (TP=4 SwiGLU over the 2x2 mesh, all_reduce over both axes); fp8 + 128x128 block scale dequantized
    to bf16 at load."""
    import torch

    from models.demos.mimo_v2_6_d_p.reference.weights import fp8_weight
    from models.demos.mimo_v2_6_d_p_2x2.tt.mlp import TtDenseMLP

    p = f"model.layers.{layer}.mlp."
    wg, wu, wd = (fp8_weight(loader, p + f"{n}.weight", torch.float32) for n in ("gate_proj", "up_proj", "down_proj"))
    return TtDenseMLP(mesh, wg, wu, wd)


def build_router(mesh, loader, cfg, layer: int, max_chunk: int):
    """TtRouter (replicated on the 2x2 mesh, fp32 logits HiFi4 + fp32 acc, fp32 sigmoid + bias choice, ttnn.topk, no
    CCL). MIMO_ROUTER_MODE=fused selects moe_grouped_topk (TF32 keys) for comparison."""
    import os

    from models.demos.mimo_v2_6_d_p_2x2.tt.router import TtRouter

    assert cfg.n_group == 1 and cfg.scoring_func == "sigmoid" and cfg.norm_topk_prob
    p = f"model.layers.{layer}.mlp.gate."
    w = loader.get(p + "weight").float()
    b = loader.get(p + "e_score_correction_bias").float()
    rs = cfg.routed_scaling_factor if cfg.routed_scaling_factor is not None else 1.0
    mode = os.environ.get("MIMO_ROUTER_MODE", "fp32")
    return TtRouter(mesh, w, b, max_chunk, top_k=cfg.num_experts_per_tok, route_scale=rs, mode=mode)


def build_experts(mesh, loader, cfg, layer: int, max_chunk: int):
    """TtExperts (2x2: dispatch axis 0 with 2 chips per group, 2 groups = columns, 64 experts per chip, bfp8).
    Default mode 'unified' (ttnn.bringup.unified_routed_expert_moe, high_precision, HiFi4 + fp32 dest);
    MIMO_EXPERTS_MODE=loop selects the per-expert ttnn.linear path, unified_lofi the op without high_precision."""
    import os

    from models.demos.mimo_v2_6_d_p_2x2.tt.experts import LazyExpertWeights, TtExperts

    weights = LazyExpertWeights(loader, f"model.layers.{layer}.mlp.experts.", cfg.n_routed_experts)
    return TtExperts(
        mesh,
        layer,
        weights,
        emb_dim=cfg.hidden_size,
        hidden_dim=cfg.moe_intermediate_size,
        top_k=cfg.num_experts_per_tok,
        max_seq_len=max_chunk,
        mode=os.environ.get("MIMO_EXPERTS_MODE", "unified"),
    )


def new_attention_caches(mesh, cfg, layers, max_seq: int) -> dict:
    return {i: new_kv_cache(mesh, cfg, i, max_seq) for i in layers}


# --------------------------------------------------------------------------------------
# All-device model (assemble step): port of models/demos/mimo_v2_6_d_p/tt/model.py (1x4) onto the 2x2 modules.
#
# The hidden state stays on the device as a replicated [1, 1, S, H] bf16 TILE tensor from the embedding to the final
# norm; the only host input per chunk is the token ids. Block graphs (reference/mimo_ref.py DENSE_GRAPH / MOE_GRAPH):
#     attn_norm -> attention -> attn_residual (h_mid = in + .) -> ffn_norm
#     dense: mlp -> mlp_residual (out = h_mid + .)
#     MoE:   router(ffn_norm) -> experts(ffn_norm, router) -> ffn_residual (out = h_mid + .)
# Load-time constants: RoPE cos/sin for max_seq per attention module (sliced on the device), the full layers' identity
# page table, the router's zero / bias tables for max_chunk, the experts' dispatch tables and bfp8 weights. The router
# boundary is (idx, wts): the experts take them directly (no dense -> topk round).
# --------------------------------------------------------------------------------------


class TtEmbedding:
    """Replicated [V, H] bf16 table (152576 x 4096, 1.25 GB/chip), ROW_MAJOR, cached as a tensorbin.

    Input: uint32 ROW_MAJOR ids [1, 1, S] on the device. The vocab is not a power of two, so pad ids must be masked by
    the contract step before this lookup."""

    def __init__(self, mesh, weight, cache: bool = True):
        import torch

        import ttnn
        from models.demos.mimo_v2_6_d_p_2x2.tt.experts import CACHE_ROOT

        self.mesh = mesh
        self.vocab, self.hidden = weight.shape
        self.weight = ttnn.as_tensor(
            weight.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            cache_file_name=str(CACHE_ROOT / "embed_bf16") if cache else None,
        )

    def __call__(self, ids):
        import ttnn

        seq = ids.shape[-1]
        ids4 = ttnn.reshape(ids, (1, 1, 1, seq))
        x = ttnn.embedding(ids4, self.weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        return ttnn.reshape(x, (1, 1, seq, self.hidden))


class TtMiMoBlock:
    """One decoder block: the validated step modules, keyed by the reference block graph, run through run_block."""

    def __init__(self, mesh, cfg, loader, layer: int, max_seq: int, max_chunk: int):
        from models.demos.mimo_v2_6_d_p.reference.mimo_ref import DENSE_GRAPH, MOE_GRAPH
        from models.demos.mimo_v2_6_d_p_2x2.tt.residual import TtResidualAdd

        self.i, self.mesh = layer, mesh
        self.moe = cfg.is_moe(layer)
        eps = cfg.layernorm_epsilon
        self.norms = {k: build_norm(mesh, loader, layer, k, eps) for k in NORM_WEIGHTS}
        self.attn = build_attention(mesh, loader, cfg, layer, max_seq)
        self.add = TtResidualAdd(mesh)
        if self.moe:
            self.router = build_router(mesh, loader, cfg, layer, max_chunk)
            self.experts = build_experts(mesh, loader, cfg, layer, max_chunk)
        else:
            self.mlp = build_mlp(mesh, loader, layer)
        self.graph = list(MOE_GRAPH if self.moe else DENSE_GRAPH)
        # Residual add + the next RMSNorm in one call (MIMO_FUSE_RESIDUAL_NORM, set by TtMiMoModel): the residual step
        # returns the sum and parks the norm in `stash`, which the norm step that follows (ffn_norm, or the next
        # block's attn_norm) returns instead of recomputing it. A block built on its own has no stash and runs unfused.
        self.stash = None
        self.next_norm = None  # the next block's attn_norm (TtMiMoModel)
        self._build_steps()

    def _build_steps(self):
        """Device fn(ctx, *inputs) per graph step, and after each step the boundaries it was the last reader of (freed
        there; the block input "in" belongs to the caller). ctx.state is the layer's KV cache."""
        import ttnn

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
            attention=lambda ctx, x: self.attn(x, ctx.start, ctx.state, kv_sink=ctx.extra.get("kv_sink")),
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
        import ttnn

        pos = [inputs.index(nm) for nm in dead]

        def wrapped(ctx, *args):
            y = fn(ctx, *args)
            for p in pos:
                for t in args[p] if isinstance(args[p], tuple) else (args[p],):
                    ttnn.deallocate(t)
            return y

        return wrapped

    def __call__(self, x, start: int, cache, kv_sink=None):
        """x: replicated [1, 1, S, H] block input (not freed). One profiler section per step; every intermediate is
        freed after its last reader."""
        from models.demos.common.bringup.reference.interface import Ctx, run_block

        def missing(name):
            raise KeyError(f"no device step {name}")

        ctx = Ctx(self.i, start, x.shape[-2], cache, {"kv_sink": kv_sink} if kv_sink is not None else {})
        return run_block(self.graph, missing, ctx, x, overrides=self.overrides)


class TtMiMoModel:
    """Embedding + decoder blocks + final norm on the 2x2 mesh. `prefill_chunk` runs one chunk against per-layer KV
    caches (new_attention_caches) and returns the last block's hidden state [1, 1, S, H]."""

    def __init__(self, mesh, model_path: str, max_seq: int, max_chunk: int, layers=None):
        import os

        from models.demos.mimo_v2_6_d_p.reference.mimo_ref import MiMoConfig
        from models.demos.mimo_v2_6_d_p.reference.weights import WeightLoader
        from models.demos.mimo_v2_6_d_p_2x2.tt.rms_norm import TtRMSNorm

        self.mesh = mesh
        self.cfg = cfg = MiMoConfig.from_json(os.path.join(model_path, "config.json"))
        loader = WeightLoader(model_path)
        self.layer_ids = list(range(cfg.num_hidden_layers)) if layers is None else list(layers)
        self.max_seq = -(-int(max_seq) // 64) * 64
        self.max_chunk = int(max_chunk)
        self.embed = TtEmbedding(mesh, loader.get("model.embed_tokens.weight"))
        self.blocks = [TtMiMoBlock(mesh, cfg, loader, i, self.max_seq, self.max_chunk) for i in self.layer_ids]
        if os.environ.get("MIMO_FUSE_RESIDUAL_NORM", "1") != "0":
            stash = {}
            for k, blk in enumerate(self.blocks):
                blk.stash = stash
                nxt = self.blocks[k + 1] if k + 1 < len(self.blocks) else None
                blk.next_norm = nxt.norms["attn_norm"] if nxt is not None and nxt.i == blk.i + 1 else None
        self.final_norm = TtRMSNorm(mesh, loader.get("model.norm.weight"), eps=cfg.layernorm_epsilon)

    def prefill_chunk(self, ids, start: int, caches: dict, on_layer=None, kv_sink_of=None):
        """ids: device uint32 [1, 1, S]. Positions [start, start+S). on_layer(i) is called after block i is enqueued;
        kv_sink_of(i) returns the block's kv_sink(k, v) (or None)."""
        import ttnn

        h = self.embed(ids)
        for blk in self.blocks:
            h2 = blk(h, start, caches[blk.i], kv_sink=kv_sink_of(blk.i) if kv_sink_of else None)
            ttnn.deallocate(h)
            h = h2
            if on_layer is not None:
                on_layer(blk.i)
        return h
