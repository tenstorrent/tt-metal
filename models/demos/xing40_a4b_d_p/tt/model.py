# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Xing4.0-29B-A4B text decoder on the 4x2 mesh, fully on the device (embedding -> 40 blocks -> final norm).

Built from the per-step device modules the component / swap gates validated (the same build_* functions the hooks'
component path uses), keyed by the reference block graph (reference/xing_ref.py HC_ATTN / HC_FFN / DENSE_FFN /
MOE_FFN / FFN_RESIDUAL) and run through run_block, so the profiler sees one section per step. The pattern is
glm53_flash_d_p/tt/model.py:TtGlmBlock (last-use free schedule).

Residual: the 4 mHC streams as [1, 1, S/4, 4 x 1792] fp32 TILE per chip (tt/layout.py): chip (r, c) holds chunk rows
[r S/4, (r+1) S/4) (SP over axis 0) and hidden columns [1792 c, 1792 (c+1)) of each stream (TP over axis 1),
stream-major. Every boundary between steps is the one the hybrid harness used, minus the host:

    attn_hc / ffn_hc      streams -> [S/4, 24] fp32, replicated over axis 1 (all_reduce axis 1 inside)
    *_collapse            -> [S/4, 1792] fp32 column split
    attn_norm             -> [S/4, 1792] bf16 column split
    q_a                   -> [S/4, 768] bf16, replicated over axis 1
    attention             -> [S/4, 1792] fp32 column split (ring_mla over axis 0; latent cache in the module)
    *_residual            -> streams
    ffn_norm              -> [S/4, 3584] bf16, replicated over axis 1 (all_gather axis 1 inside)
    mlp / shared_expert   -> [S/4, 1792] fp32 column split (reduce_scatter axis 1 inside)
    router                -> (idx uint16, wts fp32) [S/4, 4], replicated over axis 1 (the dense [S/4, 64] matrix
                             the hybrid read back is freed: the experts take the top-k directly)
    experts               -> [S/4, 1792] fp32 column split (dispatch / combine axis 0, reduce_scatter axis 1)
    moe_add               -> [S/4, 1792] fp32 column split

Embedding: the bf16 table split by hidden columns over axis 1 ([V, 1792] per chip, ROW_MAJOR, tensorbin cache), ids
[1, 1, 1, S/4] uint32 per chip (row split); the chip's columns typecast to fp32 and repeated for the 4 streams.
Final norm: fp32 mean of the 4 streams, then TtDistributedRmsNorm (model.norm.weight) -> [S/4, 1792] fp32 column split.

Per-shape / per-position constants (RoPE tables, latent cache, gather scratch, SDPA config, dispatch / combine
modules) are built by ``setup(chunk, max_seq)``, called at state creation (outside the forward); ``__call__`` does
no host work.
"""

from __future__ import annotations

import os

import torch

import ttnn

CACHE_EMBED = "embed_bf16_colsplit"


def block_graph(cfg, layer: int):
    """The reference block graph (xing_ref.XingReference.block_graph) without loading the reference."""
    from models.demos.xing40_a4b_d_p.reference import xing_ref as R

    ffn = R.MOE_FFN if cfg.is_moe(layer) else R.DENSE_FFN
    return list(R.HC_ATTN + R.HC_FFN + ffn + R.FFN_RESIDUAL)


def _free(t):
    if isinstance(t, (tuple, list)):
        for x in t:
            _free(x)
    elif isinstance(t, ttnn.Tensor) and t.is_allocated():
        ttnn.deallocate(t)


def _freeing(fn, inputs, dead):
    """fn with the boundaries in ``dead`` (their last reader is this step) freed after it runs."""
    if not dead:
        return fn
    pos = [inputs.index(nm) for nm in dead]

    def wrapped(ctx, *args):
        y = fn(ctx, *args)
        for p in pos:
            _free(args[p])
        return y

    return wrapped


class TtEmbedding:
    """[V, H] bf16 table split by hidden columns over axis 1 (chip column c holds [V, 1792 c .. + 1792]), replicated
    over axis 0, ROW_MAJOR in DRAM (470 MB per chip), cached as a tensorbin. ids: [1, 1, 1, S/4] uint32 ROW_MAJOR per
    chip (row split over axis 0, replicated over axis 1) -> streams [1, 1, S/4, 4 x 1792] fp32."""

    def __init__(self, mesh, weight: torch.Tensor, n: int, cache: bool = True):
        from models.demos.xing40_a4b_d_p.tt.experts import CACHE_ROOT

        self.mesh, self.n = mesh, n
        self.vocab, self.hidden = weight.shape
        self.local = self.hidden // mesh.shape[1]
        if cache:
            CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        self.weight = ttnn.as_tensor(
            weight.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(None, 1)),
            cache_file_name=str(CACHE_ROOT / CACHE_EMBED) if cache else None,
        )

    def ids_to_device(self, tokens: torch.Tensor) -> ttnn.Tensor:
        """Host token ids [S] -> device [1, 1, 1, S/4] uint32 per chip (the one host input per chunk)."""
        return ttnn.from_torch(
            tokens.reshape(1, 1, 1, -1).to(torch.int64).to(torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(3, None)),
        )

    def __call__(self, ids: ttnn.Tensor) -> ttnn.Tensor:
        mc = ttnn.DRAM_MEMORY_CONFIG
        s4 = ids.shape[-1]
        e = ttnn.embedding(ids, self.weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=mc)
        e = ttnn.reshape(e, (1, 1, s4, self.local))
        e32 = ttnn.typecast(e, ttnn.float32, memory_config=mc)
        ttnn.deallocate(e)
        out = ttnn.concat([e32] * self.n, dim=-1, memory_config=mc)
        ttnn.deallocate(e32)
        return out


class TtFinalNorm:
    """final_norm = rms_norm(mean_n stream_n) * w: the 4 stream slices summed in fp32, x 1/4, then the distributed
    RMSNorm (stats all_gather over axis 1, HiFi4 + fp32 dest) -> [1, 1, S/4, 1792] fp32 column split."""

    def __init__(self, mesh, weight: torch.Tensor, n: int, eps: float):
        from models.demos.xing40_a4b_d_p.tt.norm import TtDistributedRmsNorm

        self.n = n
        self.norm = TtDistributedRmsNorm(mesh, weight.float(), eps, cluster_axis=1, dtype=ttnn.float32)

    def __call__(self, h: ttnn.Tensor) -> ttnn.Tensor:
        mc = ttnn.DRAM_MEMORY_CONFIG
        s4, w = h.shape[-2], h.shape[-1] // self.n
        acc = None
        for j in range(self.n):
            s = ttnn.slice(h, [0, 0, 0, j * w], [1, 1, s4, (j + 1) * w], memory_config=mc)
            if acc is None:
                acc = s
            else:
                nxt = ttnn.add(acc, s, dtype=ttnn.float32, memory_config=mc)
                ttnn.deallocate(acc)
                ttnn.deallocate(s)
                acc = nxt
        mean = ttnn.multiply(acc, 1.0 / self.n, memory_config=mc)
        ttnn.deallocate(acc)
        y = self.norm(mean)
        ttnn.deallocate(mean)
        return y


class TtXingBlock:
    """One decoder block: the validated step modules keyed by the reference block graph, run through run_block."""

    def __init__(self, mesh, cfg, loader, layer: int, max_rows: int, max_chunk: int):
        from models.demos.xing40_a4b_d_p.tt.attention import build_attention
        from models.demos.xing40_a4b_d_p.tt.collapse import build_collapse
        from models.demos.xing40_a4b_d_p.tt.mhc import build_hc
        from models.demos.xing40_a4b_d_p.tt.norm import build_norm
        from models.demos.xing40_a4b_d_p.tt.q_a import build_q_a
        from models.demos.xing40_a4b_d_p.tt.residual import build_residual

        self.i, self.mesh, self.cfg = layer, mesh, cfg
        self.sp = mesh.shape[0]
        self.moe = cfg.is_moe(layer)
        self.graph = block_graph(cfg, layer)
        collapse, residual = build_collapse(cfg), build_residual(cfg, mesh)
        hc = {s: build_hc(mesh, loader, cfg, layer, s) for s in ("attn_hc", "ffn_hc")}
        norm = {s: build_norm(mesh, loader, cfg, layer, s) for s in ("attn_norm", "ffn_norm")}
        self.q_a = build_q_a(mesh, loader, cfg, layer)
        self.attn = build_attention(mesh, loader, cfg, layer)
        steps = {
            "attn_hc": lambda ctx, x: hc["attn_hc"](x),
            "attn_collapse": lambda ctx, x, h: collapse(x, h),
            "attn_norm": lambda ctx, x: norm["attn_norm"](x),
            "q_a": lambda ctx, x: self.q_a(x),
            "attention": lambda ctx, x, qr: self.attn(x, qr, ctx.start),
            "attn_residual": lambda ctx, x, h, y: residual(x, h, y),
            "ffn_hc": lambda ctx, x: hc["ffn_hc"](x),
            "ffn_collapse": lambda ctx, x, h: collapse(x, h),
            "ffn_norm": lambda ctx, x: norm["ffn_norm"](x),
            "ffn_residual": lambda ctx, x, h, y: residual(x, h, y),
        }
        self.experts = None
        if self.moe:
            from models.demos.xing40_a4b_d_p.tt.experts import build_experts
            from models.demos.xing40_a4b_d_p.tt.mlp import build_mlp
            from models.demos.xing40_a4b_d_p.tt.moe_add import build_moe_add
            from models.demos.xing40_a4b_d_p.tt.router import build_router

            self.router = build_router(mesh, loader, cfg, layer, max_rows)
            self.experts = build_experts(mesh, loader, cfg, layer, max_chunk)
            self.shared = build_mlp(mesh, loader, cfg, layer, prefix="mlp.shared_experts.")
            add = build_moe_add(cfg)

            def router(ctx, x):
                dense, idx, wts = self.router(x)
                ttnn.deallocate(dense)
                return (idx, wts)

            steps.update(
                router=router,
                experts=lambda ctx, x, r: self.experts(x, r[0], r[1]),
                shared_expert=lambda ctx, x: self.shared(x),
                moe_add=lambda ctx, a, b: add(a, b),
            )
        else:
            from models.demos.xing40_a4b_d_p.tt.mlp import build_mlp

            self.mlp = build_mlp(mesh, loader, cfg, layer)
            steps["mlp"] = lambda ctx, x: self.mlp(x)
        assert set(steps) == {st.name for st in self.graph}, (sorted(steps), [st.name for st in self.graph])
        last_use = {}
        for k, st in enumerate(self.graph):
            for nm in st.inputs:
                last_use[nm] = k
        self.overrides = {}
        for k, st in enumerate(self.graph):
            dead = tuple(dict.fromkeys(nm for nm in st.inputs if nm != "in" and last_use[nm] == k))
            self.overrides[st.name] = _freeing(steps[st.name], st.inputs, dead)

    # ---- load time / harness boundary (never inside the forward)
    def setup(self, chunk: int, max_seq: int) -> None:
        """Build (once per geometry) the attention's RoPE tables / latent cache / scratch and the experts' dispatch
        and combine modules for this chunk length."""
        self.attn.setup(chunk, max_seq)
        if self.experts is not None:
            self.experts.setup(chunk)

    def load_state(self, kv_latent: torch.Tensor | None) -> None:
        self.attn.load_state(kv_latent)

    def state_torch(self, length: int) -> dict:
        return {"kv_latent": self.attn.read_state(length)}

    # ---- forward
    def __call__(self, x: ttnn.Tensor, start: int) -> ttnn.Tensor:
        """x: the chip's streams [1, 1, S/4, 4 x 1792] fp32 (not freed) -> block output, same layout. The geometry
        (chunk, max_seq) must be set up."""
        from models.demos.common.bringup.reference.interface import Ctx, run_block

        def missing(name):
            raise KeyError(f"no device step {name}")

        g = self.attn.geom
        chunk = x.shape[-2] * self.sp
        assert g is not None and g.chunk == chunk, f"layer {self.i}: setup({chunk}, max_seq) first"
        ctx = Ctx(self.i, start, chunk, None)
        return run_block(self.graph, missing, ctx, x, overrides=self.overrides)


class TtXingModel:
    """Embedding + decoder blocks + final norm on the device."""

    def __init__(self, mesh, model_path: str, max_rows: int, max_chunk: int, layers: list[int] | None = None):
        from models.demos.xing40_a4b_d_p.reference.weights import WeightLoader
        from models.demos.xing40_a4b_d_p.reference.xing_ref import XingConfig

        self.mesh = mesh
        self.cfg = cfg = XingConfig.from_json(os.path.join(model_path, "config.json"))
        loader = WeightLoader(model_path)
        self.layer_ids = list(range(cfg.num_hidden_layers)) if layers is None else list(layers)
        self.embed = TtEmbedding(mesh, loader.get("model.embed_tokens.weight"), cfg.hc_mult)
        self.blocks = [TtXingBlock(mesh, cfg, loader, i, max_rows, max_chunk) for i in self.layer_ids]
        self.final_norm = TtFinalNorm(mesh, loader.get("model.norm.weight"), cfg.hc_mult, cfg.rms_norm_eps)

    def setup(self, chunk: int, max_seq: int) -> None:
        for b in self.blocks:
            b.setup(chunk, max_seq)
