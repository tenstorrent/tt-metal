# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Hy4 Preview all-device model on the 2x2 mesh (assemble step): embedding -> blocks -> hc_head + final norm.

The hidden state (the 4 iHC streams) stays on the device from the embedding to the final norm, as
[1, 1, S/2, 4 x 3072] fp32 TILE per chip (tt/layout.py: rows split over axis 0, hidden columns over axis 1,
stream-major). Every block runs through ``run_block`` over the reference block graph (reference/hy4_ref.py
``Hy4Reference.block_graph``), with the validated step modules as overrides (the same builders the component / swap
/ hybrid hooks use, bringup/hooks.py), so the profiler sees one section per step. No step is CPU or OPGEN
(components.yaml), so there is no CpuBridge.

Device state per layer (TtHy4DeviceState): the MLA latent cache (TtHy4Attention) and, on full layers, the index-key
cache (TtHy4Indexer). Both are built once per (chunk, max_seq) geometry by the modules' ``setup`` (RoPE tables for
max_seq, caches, scratch), and sliced on the device per chunk (the ops take the chunk start). A shared layer attends
over the latest full layer's device top-k of the same chunk (TtTopkShared, an identity): the indexer output is that
layer's persistent gather buffer, so it is never freed here.

Only the token ids go host -> device per chunk (``TtHy4Model.embed``); state load / read-back and the trail
read-back are harness boundaries outside ``layer``.
"""

from __future__ import annotations

import types

import torch

import ttnn

from .layout import HC

# Boundaries a block step must not free after its last reader: the block input belongs to the caller, and the
# indexer's top-k is its persistent output buffer (shared layers read the latest full layer's).
_KEEP = {"in", "topk"}


class TtHy4Embedding:
    """embed_tokens [V, H] bf16, hidden split over the mesh columns ([V, H/2] per chip, replicated over the rows),
    ROW_MAJOR, cached as a tensorbin. Input: uint32 ROW_MAJOR ids [1, 1, S/2] per chip (the row split over axis 0).
    Output: the 4 identical iHC streams [1, 1, S/2, 4 x H/2] fp32 TILE (HF: embedding repeated hc_mult times; bf16
    -> fp32 is exact)."""

    def __init__(self, mesh, weight: torch.Tensor, cache_file: str | None = None):
        self.mesh = mesh
        self.vocab, self.hidden = weight.shape
        self.weight = ttnn.as_tensor(
            weight.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(None, 1)),
            cache_file_name=cache_file,
        )

    def __call__(self, ids: ttnn.Tensor) -> ttnn.Tensor:
        dram = ttnn.DRAM_MEMORY_CONFIG
        s2 = ids.shape[-1]
        e = ttnn.embedding(
            ttnn.reshape(ids, (1, 1, s2)), self.weight, layout=ttnn.TILE_LAYOUT, memory_config=dram
        )  # [1, 1, S/2, H/2] bf16
        e = ttnn.reshape(e, (1, 1, s2, self.weight.shape[-1]))
        ef = ttnn.typecast(e, ttnn.float32, memory_config=dram)
        ttnn.deallocate(e)
        out = ttnn.concat([ef] * HC, dim=-1, memory_config=dram)  # stream j = local columns [j*H/2, (j+1)*H/2)
        ttnn.deallocate(ef)
        return out


class TtHy4FinalNorm:
    """hc_head (HF HYV4HyperHead: pre = sigmoid(mixes * scale + base) + eps, x = sum_j pre_j stream_j, fp32) and the
    final RMSNorm (model.norm, eps 1e-5). Built from the validated iHC / norm modules: TtHcGates with hc_head_fn
    zero-padded to 8 rows (columns 4-7 unused), TtHcPre on gate columns 0-3, then TtGatheredRmsNorm (all_gather over
    axis 1, ttnn.bringup.rms_norm, fp32 out). Output [1, 1, S/2, H] fp32 per chip, rows over axis 0, replicated over
    axis 1."""

    def __init__(self, mesh, cfg, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, norm_w: torch.Tensor):
        from .ihc import TtHcGates, TtHcPre
        from .norm import TtGatheredRmsNorm

        h = cfg.hidden_size
        fn8 = torch.zeros(2 * HC, HC * h)
        fn8[:HC] = fn.float()
        base8 = torch.zeros(2 * HC)
        base8[:HC] = base.float()
        sc = torch.tensor([float(scale.flatten()[0]), 0.0])
        self.gates = TtHcGates(
            mesh, fn8, base8, sc, h, norm_eps=cfg.rms_norm_eps, hc_eps=cfg.hc_eps, magnitude=cfg.hc_magnitude
        )
        self.pre = TtHcPre(mesh, h)
        self.norm = TtGatheredRmsNorm(mesh, norm_w.float(), cfg.rms_norm_eps, cluster_axis=1, dtype=ttnn.float32)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        g = self.gates(x)
        y = self.pre(x, g)
        ttnn.deallocate(g)
        out = self.norm(y)
        ttnn.deallocate(y)
        return out


class TtHy4DeviceState:
    """Per-layer device state of one sequence: each layer's TtHy4Attention (kv_latent) and, on full layers,
    TtHy4Indexer (index_key) caches for (chunk, max_seq), plus the latest full layers' device top-k of the current
    chunk. The caches live in the modules' geometries; ``activate(chunk)`` selects (and on first use builds) them and
    applies the pending prefix (zeros or a golden prefix), a harness boundary run before the first chunk's layers."""

    def __init__(self, model: "TtHy4Model", max_seq: int, chunk: int | None = None):
        self.model, self.max_seq = model, int(max_seq)
        self.chunk = None
        self.pending = {i: {} for i in model.blocks}  # layer -> {state name: natural-order prefix}
        self.topk = {}  # full layer -> its device top-k of the current chunk
        if chunk is not None:
            self.activate(chunk)

    def activate(self, chunk: int) -> None:
        if self.chunk == chunk:
            return
        assert self.chunk is None, f"state built for chunk {self.chunk}, called with {chunk}"
        self.chunk = chunk
        for i, blk in self.model.blocks.items():
            blk.setup(chunk, self.max_seq)
            blk.attention.load_state(self.pending[i].get("kv_latent"))
            if blk.indexer is not None:
                blk.indexer.load_state(self.pending[i].get("index_key"))
        self.pending = {i: {} for i in self.model.blocks}

    def load_prefix(self, layer: int, tensors: dict, length: int) -> None:
        blk = self.model.blocks[layer]
        kv = tensors["kv_latent"][:length]
        ik = tensors["index_key"][:length] if blk.indexer is not None else None
        if self.chunk is None:
            self.pending[layer] = {"kv_latent": kv, "index_key": ik}
            return
        blk.attention.load_state(kv)
        if blk.indexer is not None:
            blk.indexer.load_state(ik)

    def to_torch(self, layer: int, length: int) -> dict:
        blk = self.model.blocks[layer]
        kv = blk.attention.read_state(length)
        ik = (
            blk.indexer.read_state(length) if blk.indexer is not None else torch.zeros(0, self.model.cfg.index_head_dim)
        )
        return {"kv_latent": kv, "index_key": ik}


class TtHy4Block:
    """One decoder block: the validated step modules keyed by the reference block graph, run through run_block.
    Every intermediate boundary is freed after its last reader (except the block input and the top-k)."""

    def __init__(self, mesh, spec, cfg, loader, layer: int):
        from models.demos.hy4_preview_d_p.bringup import hooks as H
        from models.demos.hy4_preview_d_p.reference.hy4_ref import Hy4Reference

        from .ihc import TtHcPost, TtHcPre
        from .mlp import TtMoeCombine

        self.i, self.mesh, self.cfg = layer, mesh, cfg
        self.moe, self.full = cfg.is_moe(layer), cfg.is_full(layer)
        self.src = cfg.topk_source(layer)
        # The reference's own graph (its block_graph reads only cfg).
        self.graph = list(Hy4Reference.block_graph(types.SimpleNamespace(cfg=cfg), layer))

        hid = cfg.hidden_size
        self.hc = {s: H._hc_module(mesh, spec, layer, s, loader, cfg) for s in ("attn_hc", "ffn_hc")}
        self.hc_pre = TtHcPre(mesh, hid)
        self.hc_post = TtHcPost(mesh, hid)
        self.attn_norm = H._norm_module(mesh, spec, layer, "attn_norm", loader, cfg)
        self.ffn_norm = H._gathered_norm_module(mesh, spec, layer, "ffn_norm", loader, cfg)
        self.q_a = H._qa_module(mesh, spec, layer, loader, cfg)
        self.indexer = H._indexer_module(mesh, spec, layer, loader, cfg) if self.full else None
        self.topk_shared = None if self.full else H._topk_shared_module(mesh, spec, layer, loader, cfg)
        self.attention = H._attention_module(mesh, spec, layer, loader, cfg)
        if self.moe:
            self.router = H._router_module(mesh, spec, layer, loader, cfg)
            self.experts = H._experts_module(mesh, spec, layer, loader, cfg)
            self.shared = H._mlp_module(mesh, spec, layer, loader, cfg, prefix=H._SHARED_STEPS["shared_expert"])
            self.combine = TtMoeCombine(mesh)
        else:
            self.mlp = H._mlp_module(mesh, spec, layer, loader, cfg)
        self._build_steps()

    def setup(self, chunk: int, max_seq: int) -> None:
        """Load-time geometry for (chunk, max_seq): RoPE tables, caches, scratch, dispatch / combine sizes."""
        self.attention.setup(chunk, max_seq)
        if self.indexer is not None:
            self.indexer.setup(chunk, max_seq)
        if self.moe:
            self.experts.setup(chunk)

    def _topk(self, ctx):
        st = ctx.state
        if self.full:
            raise AssertionError("full layers run the indexer")
        if self.src not in st.topk:
            raise KeyError(f"layer {self.i} reuses layer {self.src}'s top-k, which has not run on this chunk")
        return self.topk_shared(st.topk[self.src])

    def _indexer(self, ctx, x, qr):
        tk = self.indexer(x, qr, ctx.start)
        ctx.state.topk[self.i] = tk
        return tk

    def _build_steps(self):
        steps = {
            "attn_hc": lambda ctx, s: self.hc["attn_hc"](s),
            "attn_hc_pre": lambda ctx, s, g: self.hc_pre(s, g),
            "attn_norm": lambda ctx, x: self.attn_norm(x),
            "q_a": lambda ctx, x: self.q_a(x),
            "attention": lambda ctx, x, qr, tk: self.attention(x, qr, tk, ctx.start),
            "attn_residual": lambda ctx, s, g, y: self.hc_post(s, g, y),
            "ffn_hc": lambda ctx, s: self.hc["ffn_hc"](s),
            "ffn_hc_pre": lambda ctx, s, g: self.hc_pre(s, g),
            "ffn_norm": lambda ctx, x: self.ffn_norm(x),
            "ffn_residual": lambda ctx, s, g, y: self.hc_post(s, g, y),
        }
        if self.full:
            steps["indexer"] = self._indexer
        else:
            steps["topk_shared"] = lambda ctx, x: self._topk(ctx)
        if self.moe:
            steps.update(
                router=lambda ctx, x: tuple(self.router(x)),  # (idx, wts) [1, 1, S/2, 8]: the experts take them
                experts=lambda ctx, x, r: self.experts(x, r[0], r[1]),
                shared_expert=lambda ctx, x: self.shared(x),
                moe_combine=lambda ctx, a, b: self.combine(a, b),
            )
        else:
            steps["mlp"] = lambda ctx, x: self.mlp(x)
        names = [st.name for st in self.graph]
        assert set(steps) == set(names), (sorted(steps), names)
        last_use = {}
        for k, st in enumerate(self.graph):
            for nm in st.inputs:
                last_use[nm] = k
        self.overrides = {}
        for k, st in enumerate(self.graph):
            dead = tuple(dict.fromkeys(nm for nm in st.inputs if nm not in _KEEP and last_use[nm] == k))
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
                    if t.is_allocated():
                        ttnn.deallocate(t)
            return y

        return wrapped

    def __call__(self, x: ttnn.Tensor, start: int, state: TtHy4DeviceState) -> ttnn.Tensor:
        from models.demos.common.bringup.reference.interface import Ctx, run_block

        def missing(name):
            raise KeyError(f"layer {self.i}: no device step {name}")

        ctx = Ctx(self.i, start, x.shape[-2] * self.mesh.shape[0], state)
        return run_block(self.graph, missing, ctx, x, overrides=self.overrides)


class TtHy4Model:
    """Embedding + blocks + hc_head / final norm on the 2x2 mesh."""

    def __init__(self, mesh, spec, layers, final_norm: bool = True):
        from models.demos.hy4_preview_d_p.bringup import hooks as H

        from .experts import CACHE_ROOT

        self.mesh, self.spec = mesh, spec
        loader = H._loader(spec)
        self.cfg = cfg = H._cfg(loader)
        self.layer_ids = list(layers)
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        self.embedding = TtHy4Embedding(
            mesh, loader.get("model.embed_tokens.weight"), cache_file=str(CACHE_ROOT / "embed_tokens_bf16_cols")
        )
        self.blocks = {i: TtHy4Block(mesh, spec, cfg, loader, i) for i in self.layer_ids}
        self.final = None
        if final_norm:
            fn, base, scale = (loader.get(f"model.hc_head.hc_head_{n}").float() for n in ("fn", "base", "scale"))
            self.final = TtHy4FinalNorm(mesh, cfg, fn, base, scale, loader.get("model.norm.weight").float())

    def embed_ids(self, ids: ttnn.Tensor) -> ttnn.Tensor:
        """ids: uint32 ROW_MAJOR [1, 1, S/2] per chip (row split over axis 0, replicated over axis 1)."""
        return self.embedding(ids)
