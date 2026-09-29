# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash text decoder on the 2x2 mesh, fully on the device (embedding -> blocks -> final norm).

Built from the per-step device modules the component / swap gates validated (the same build_* functions the hooks'
component path uses). The residual stays on the device as a replicated [1, 1, S, 4 H] bf16 TILE tensor (the 4 mHC
streams packed along the last dim: the same memory as the reference's token-major [S * 4, H]); the only host input per
chunk is the token ids.

Block graphs (reference/glm_ref.py HC_ATTN / KDA_ATTN / DSA_ATTN / HC_FFN / DENSE_FFN / MOE_FFN / FFN_RESIDUAL),
each step one device module, run through run_block (one profiler section per step), every intermediate freed after
its last reader. Boundaries that differ from the hybrid harness: the indexer's per-chip uint32 index rows go straight
into the MLA (no host compaction: the device format already has the sentinel tail); the router's dense [S, 288]
routing goes into the experts as in the swap tests.

Per-layer state lives in the modules (KDA carries in address-stable buffers, the indexer's pooled-key cache, the MLA
latent cache); a chunk at start 0 starts from a zero KDA state. Every shape / position constant (KDA start table,
indexer masks and tail tables per chunk size, router zeros, dispatch tables) is built at load.
"""

from __future__ import annotations

import os
from typing import Callable

import torch

import ttnn
from models.demos.glm53_flash_d_p.reference.weights import PREFIX

CACHE_ROOT_EMBED = "embed_bf16"


def block_graph(cfg, layer: int):
    """The reference block graph (glm_ref.GlmReference.block_graph) without loading the reference."""
    from models.demos.glm53_flash_d_p.reference import glm_ref as R

    attn = R.KDA_ATTN if cfg.is_kda(layer) else R.DSA_ATTN
    ffn = R.MOE_FFN if cfg.is_moe(layer) else R.DENSE_FFN
    return list(R.HC_ATTN + attn + R.HC_FFN + ffn + R.FFN_RESIDUAL)


class TtEmbedding:
    """Replicated [V, H] bf16 table (154880 x 4096, 1.27 GB per chip), ROW_MAJOR, cached as a tensorbin. Input uint32
    ROW_MAJOR ids [1, 1, S] on the device; output the 4 mHC streams [1, 1, S, 4 H] (the embedding broadcast)."""

    def __init__(self, mesh, weight: torch.Tensor, n: int, cache: bool = True):
        from models.demos.glm53_flash_d_p.tt.experts import CACHE_ROOT

        self.mesh, self.n = mesh, n
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
            cache_file_name=str(CACHE_ROOT / CACHE_ROOT_EMBED) if cache else None,
        )

    def __call__(self, ids: ttnn.Tensor) -> ttnn.Tensor:
        seq = ids.shape[-1]
        ids4 = ttnn.reshape(ids, (1, 1, 1, seq))
        e = ttnn.embedding(ids4, self.weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        e = ttnn.reshape(e, (1, 1, seq, self.hidden))
        out = ttnn.concat([e] * self.n, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(e)
        return out


class TtFinalNorm:
    """final_norm = w * rms(mean_n stream_n): streams summed in fp32, x 1/n, then TtRMSNorm (HiFi4, fp32 dest)."""

    def __init__(self, mesh, weight: torch.Tensor, n: int, eps: float):
        from models.demos.glm53_flash_d_p.tt.rms_norm import TtRMSNorm

        self.n = n
        self.norm = TtRMSNorm(mesh, weight, eps)

    def __call__(self, h: ttnn.Tensor) -> ttnn.Tensor:
        from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import _streams

        mc = ttnn.DRAM_MEMORY_CONFIG
        streams = _streams(h, self.n)
        acc = ttnn.typecast(streams[0], ttnn.float32, memory_config=mc)
        for s in streams[1:]:
            s32 = ttnn.typecast(s, ttnn.float32, memory_config=mc)
            nxt = ttnn.add(acc, s32, memory_config=mc)
            ttnn.deallocate(acc)
            ttnn.deallocate(s32)
            acc = nxt
        for s in streams:
            ttnn.deallocate(s)
        mean = ttnn.multiply(acc, 1.0 / self.n, memory_config=mc)
        ttnn.deallocate(acc)
        y = self.norm(mean)
        ttnn.deallocate(mean)
        out = ttnn.typecast(y, ttnn.bfloat16, memory_config=mc)
        ttnn.deallocate(y)
        return out


class TtGlmBlock:
    """One decoder block: the validated step modules keyed by the reference block graph, run through run_block."""

    def __init__(self, mesh, cfg, loader, layer: int, max_seq: int, chunks):
        from models.demos.glm53_flash_d_p.tt.collapse import build_collapse
        from models.demos.glm53_flash_d_p.tt.mhc import build_hc
        from models.demos.glm53_flash_d_p.tt.moe_add import build_moe_add
        from models.demos.glm53_flash_d_p.tt.residual import build_residual
        from models.demos.glm53_flash_d_p.tt.rms_norm import build_norm

        self.i, self.mesh, self.cfg = layer, mesh, cfg
        self._end = None  # the chunk's valid end while a call runs (KDA carries stop there)
        self.kda, self.moe = cfg.is_kda(layer), cfg.is_moe(layer)
        chunks = sorted(set(chunks))
        self.graph = block_graph(cfg, layer)
        n = cfg.hc_mult
        collapse, residual = build_collapse(cfg), build_residual(cfg, mesh)
        hc = {w: build_hc(mesh, loader, cfg, layer, w) for w in ("attn", "ffn")}
        norm = {k: build_norm(mesh, loader, cfg, layer, w) for k, w in NORMS.items()}
        steps = {
            "attn_hc": lambda ctx, x: hc["attn"](x),
            "attn_collapse": lambda ctx, x, h: collapse(x, h),
            "attn_norm": lambda ctx, x: norm["attn_norm"](x),
            "attn_residual": lambda ctx, x, h, y: residual(x, h, y),
            "ffn_hc": lambda ctx, x: hc["ffn"](x),
            "ffn_collapse": lambda ctx, x, h: collapse(x, h),
            "ffn_norm": lambda ctx, x: norm["ffn_norm"](x),
            "ffn_residual": lambda ctx, x, h, y: residual(x, h, y),
        }
        self.stateful = []  # modules holding this layer's state (KDA, or indexer + MLA)
        if self.kda:
            from models.demos.glm53_flash_d_p.tt.kda_attention import build_kda_attention

            self.attn = build_kda_attention(mesh, loader, cfg, layer, max_seq)
            self.stateful.append(self.attn)
            steps["attention"] = lambda ctx, x: self.attn(x, ctx.start, self._end)
        else:
            from models.demos.glm53_flash_d_p.tt.indexer import build_indexer
            from models.demos.glm53_flash_d_p.tt.mla_attention import build_mla
            from models.demos.glm53_flash_d_p.tt.q_a import build_q_a

            self.q_a = build_q_a(mesh, loader, cfg, layer)
            self.indexer = build_indexer(mesh, loader, cfg, layer, max_seq, chunks)
            self.attn = build_mla(mesh, loader, cfg, layer, max_seq)
            self.stateful += [self.indexer, self.attn]
            steps.update(
                q_a=lambda ctx, x: self.q_a(x),
                indexer=lambda ctx, x, qr: self.indexer(x, qr, ctx.start),
                attention=lambda ctx, x, qr, idx: self.attn(x, qr, idx, ctx.start),
            )
        if self.moe:
            from models.demos.glm53_flash_d_p.tt.experts import build_experts
            from models.demos.glm53_flash_d_p.tt.mlp import build_mlp
            from models.demos.glm53_flash_d_p.tt.router import build_router

            self.router = build_router(mesh, loader, cfg, layer, max(chunks))
            self.experts = build_experts(mesh, loader, cfg, layer, max(chunks))
            self.shared = build_mlp(mesh, loader, cfg, layer, name="mlp.shared_experts")
            add = build_moe_add(cfg)

            def router(ctx, x):
                dense, idx, wts = self.router(x)
                ttnn.deallocate(idx)
                ttnn.deallocate(wts)
                return dense

            steps.update(
                router=router,
                experts=lambda ctx, x, r: self.experts(x, dense=r),
                shared_expert=lambda ctx, x: self.shared(x),
                moe_add=lambda ctx, a, b: add(a, b),
            )
        else:
            from models.demos.glm53_flash_d_p.tt.mlp import build_mlp

            self.mlp = build_mlp(mesh, loader, cfg, layer)
            steps["mlp"] = lambda ctx, x: self.mlp(x)
        assert set(steps) == {st.name for st in self.graph}, (sorted(steps), [st.name for st in self.graph])
        self.n = n
        # free each boundary after its last reader ("in" belongs to the caller)
        last_use = {}
        for k, st in enumerate(self.graph):
            for name in st.inputs:
                last_use[name] = k
        self.overrides = {}
        for k, st in enumerate(self.graph):
            dead = tuple(dict.fromkeys(nm for nm in st.inputs if nm != "in" and last_use[nm] == k))
            self.overrides[st.name] = _freeing(steps[st.name], st.inputs, dead)

    def __call__(self, x: ttnn.Tensor, start: int, end: int | None = None) -> ttnn.Tensor:
        """x: replicated [1, 1, S, 4 H] block input (not freed) -> block output, same layout. end (optional): the
        exclusive valid end of a padded chunk (rows past it are pad; only the KDA carries need it)."""
        from models.demos.common.bringup.reference.interface import Ctx, run_block

        def missing(name):
            raise KeyError(f"no device step {name}")

        ctx = Ctx(self.i, start, x.shape[-2], None)
        self._end = end
        try:
            return run_block(self.graph, missing, ctx, x, overrides=self.overrides)
        finally:
            self._end = None

    def bind_state(self, state: dict) -> None:
        """Point the stateful modules at a serving slot's buffers (tt/runners/adapter.py new_block_state)."""
        if self.kda:
            self.attn.bind_state(state["kda"])
        else:
            self.indexer.bind_cache(state["index_key"])
            self.attn.bind_cache(state["kv_latent"])

    # ---- state at the harness boundary (prefix load / read-back; never inside the forward)
    def load_state(self, tensors: dict, length: int) -> None:
        if self.kda:
            self.attn.load_state(tensors)
        else:
            self.indexer.load_state(tensors, length)
            self.attn.load_state(tensors, length)

    def state_torch(self, length: int) -> dict:
        out = {}
        for m in self.stateful:
            out.update(m.state_torch())
        if "index_key" in out:
            out["index_key"] = out["index_key"][: length // self.cfg.index_kpool]
        if "kv_latent" in out:
            out["kv_latent"] = out["kv_latent"][:length]
        return out


NORMS = {"attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm"}


def _freeing(fn, inputs, dead):
    if not dead:
        return fn
    pos = [inputs.index(nm) for nm in dead]

    def wrapped(ctx, *args):
        y = fn(ctx, *args)
        for p in pos:
            if args[p].is_allocated():
                ttnn.deallocate(args[p])
        return y

    return wrapped


class TtGlmModel:
    """Embedding + decoder blocks + final norm on the device. ``prefill_chunk`` runs one chunk (state in the blocks)
    and returns the last block's residual [1, 1, S, 4 H]."""

    def __init__(self, mesh, model_path: str, max_seq: int, chunks, layers: list[int] | None = None):
        from models.demos.glm53_flash_d_p.reference.glm_ref import GlmConfig
        from models.demos.glm53_flash_d_p.reference.weights import WeightLoader

        self.mesh = mesh
        self.cfg = cfg = GlmConfig.from_json(os.path.join(model_path, "config.json"))
        loader = WeightLoader(model_path)
        self.layer_ids = list(range(cfg.num_hidden_layers)) if layers is None else list(layers)
        self.max_seq, self.chunks = int(max_seq), sorted(set(chunks))
        self.embed = TtEmbedding(mesh, loader.get(PREFIX + "embed_tokens.weight"), cfg.hc_mult)
        self.blocks = [TtGlmBlock(mesh, cfg, loader, i, self.max_seq, self.chunks) for i in self.layer_ids]
        self.final_norm = TtFinalNorm(mesh, loader.get(PREFIX + "norm.weight"), cfg.hc_mult, cfg.rms_norm_eps)

    def prefill_chunk(
        self, ids: ttnn.Tensor, start: int, on_layer: Callable[[int], None] | None = None, end: int | None = None
    ):
        """ids: device uint32 [1, 1, S] at positions [start, start + S). on_layer(i) after block i is enqueued.
        end (optional): the exclusive valid end when the chunk's tail is pad."""
        h = self.embed(ids)
        for blk in self.blocks:
            h2 = blk(h, start, end)
            ttnn.deallocate(h)
            h = h2
            if on_layer is not None:
                on_layer(blk.i)
        return h
