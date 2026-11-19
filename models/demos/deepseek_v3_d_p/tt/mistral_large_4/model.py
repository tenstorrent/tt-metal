# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The all-device model (task M.1): hidden state on the device from the embedding to the final norm.

Each block runs through ``run_block`` over the reference's block graph, every step a device module (tt/*.py) on the
residual layout of tt/layout.py ([1, 1, S/8, H/4] per chip, rows over SP, columns over TP). Nothing goes to the host
between the embedding and the final norm.

The served KV cache is the model's state (serving_contract.md "KV cache"): two configs, both bfloat8_b TILE,
ND-sharded [1, 1, 32, width] ROUND_ROBIN_1D over the DRAM banks (init_kvpe_cache), SP-sharded and TP-replicated, 2
slots. ``kvpe`` (576 wide) has one region per layer that owns its KV (NoPE layers keep the pe columns zero), ``index``
(128 wide) one per full-indexer layer. Every module of a layer writes the region it owns; a shared-KV layer reads its
source layer's region and a shared-index layer reuses its source layer's top-k of the same chunk.

Cross-layer values of the latest chunk stay on the device: a full layer's top-k (``_topk``; freed when that layer
runs the next chunk, never after its own attention: the shared layers after it read it) and a KV source layer's bf16
kvpe rows of the chunk (``TtAttention.chunk_kv``), which the shared-KV layer attends as the source does its own.

Serving entry points: ``embed(tokens, start=)`` takes one chunk in natural order (PAD_ID past the real tokens) and
lays it out as the engine's H2D stream does (ring_sdpa_reshuffle by ``start``): mesh row r holds the chunk positions
g with (g // (S/8)) % 8 == r. ``layer(i, h, start, state, end=)`` writes rows [start, end) of the caches the layer owns
and zeroes the pad rows of the last record. Modules are built on a layer's first call (weights read then).

Trace-safe forward (``meta``, tt/chunk_meta.py): with ``TtModel.meta`` set, every per-chunk value (slot, start, end)
is read on the device from the ChunkMeta tensors instead of being baked into the ops, and every chunk runs one code
path, so ``forward_meta`` captured once as a ttnn trace replays for any chunk after ``meta.write``.
"""

from __future__ import annotations

import time

import torch

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.dram_zero_fill.op import DRAMZeroFill
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache

from models.demos.deepseek_v3_d_p.reference.mistral_large_4._bringup_interface import Ctx, run_block
from models.demos.deepseek_v3_d_p.tt.mistral_large_4.attention import _stamp_sp_topology, kv_cache_storage
from models.demos.deepseek_v3_d_p.tt.mistral_large_4.layout import SP_AXIS, TP_AXIS, residual_to_device, residual_to_host
from models.demos.deepseek_v3_d_p.tt.mistral_large_4.settings import settings

NUM_USERS = 2  # serving_answers slots


def alloc_caches(mesh, cfg, max_seq: int, kv_regions: int, index_regions: int, num_users: int):
    """The served caches (module docstring): kvpe [num_users * kv_regions, 1, max_seq / SP, 576] and index
    [num_users * index_regions, 1, max_seq / SP, 128] per chip, bfloat8_b, one region per layer that owns the cache."""
    shape = tuple(mesh.shape)
    dtype, layout = kv_cache_storage()
    kvpe = init_kvpe_cache(
        cfg.kv_lora + cfg.qk_rope,
        mesh,
        max_seq,
        shape,
        SP_AXIS,
        max(1, kv_regions),
        num_users=num_users,
        dtype=dtype,
        layout=layout,
    )
    index = init_kvpe_cache(cfg.index_dim, mesh, max_seq, shape, SP_AXIS, max(1, index_regions), num_users=num_users)
    return kvpe, index


class State:
    """The served caches (TtModel.kvpe / .index) as one sequence's state (slot 0), and their host view. A layer's region
    of each cache it owns is TtModel.kv_region / index_region; a shared-KV layer reads its source's kvpe region."""

    def __init__(self, model: "TtModel", max_seq: int):
        assert max_seq <= model.max_seq and max_seq % model.chunk == 0, (max_seq, model.max_seq, model.chunk)
        self.model, self.slot = model, 0
        for t in (model.kvpe, model.index):
            DRAMZeroFill.op(t)  # a new sequence: the caches start at zero (pad rows and unwritten rows read 0)
            _stamp_sp_topology(t, model.mesh)

    def formats(self) -> dict:
        return {
            "kv_latent": self.model.kvpe.dtype,
            "k_rope": self.model.kvpe.dtype,
            "index_key": self.model.index.dtype,
        }

    def kv_caches(self) -> dict:
        return {"kvpe": self.model.kvpe, "index": self.model.index}

    # ---- host side (harness and contract tests; never inside a layer)
    def _write(self, cache, n_regions: int, region: int, rows: torch.Tensor, length: int) -> None:
        """Host rows [>= length, W] of positions [0, length) (any 32-aligned length) -> the cache, a chunk at a time
        through update_padded_kv_cache, where the model writes them; rows from ``length`` on stay as they are."""
        m = self.model
        dims = [None, None]
        dims[SP_AXIS] = 2
        for a in range(0, length, m.chunk):
            blk = torch.zeros(m.chunk, rows.shape[-1])
            n = min(m.chunk, length - a)
            blk[:n] = rows[a : a + n].float()
            t = ttnn.from_torch(
                blk.reshape(1, 1, m.chunk, -1),
                device=m.mesh,
                dtype=cache.dtype,
                layout=cache.layout,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(m.mesh, mesh_shape=tuple(m.mesh.shape), dims=dims),
            )
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                cache,
                t,
                slot_idx=self.slot,
                layer_idx=region,
                num_layers=n_regions,
                kv_actual_global=a,
                cluster_axis=SP_AXIS,
                valid_global=a + n,
            )
            ttnn.deallocate(t)

    def load_prefix(self, layer: int, tensors: dict, length: int) -> None:
        """The state of positions [0, length) (a golden prefix) -> the caches this layer owns."""
        m = self.model
        if not length:
            return
        if layer in m.kv_region and tensors["kv_latent"].shape[-1]:  # a shared-KV layer's golden holds no KV
            lat, rope = tensors["kv_latent"][:length].float(), tensors["k_rope"][:length].float()
            rows = torch.nn.functional.pad(
                torch.cat([lat, rope], dim=-1), (0, m.kvpe_width - lat.shape[1] - rope.shape[1])
            )
            self._write(m.kvpe, len(m.kv_region), m.kv_region[layer], rows, length)
        if layer in m.index_region:
            self._write(m.index, len(m.index_region), m.index_region[layer], tensors["index_key"][:length], length)

    def _read(self, cache, n_regions: int, region: int, length: int) -> torch.Tensor:
        """[length, W] host rows of one region in natural order: TP column 0's copy of each SP row's shard (rank
        major), then the block-cyclic blocks of S/8 positions put back in order (rank r holds blocks r, r + 8, ...)."""
        m = self.model
        b = self.slot * n_regions + region
        shape = list(cache.shape)
        # a slice of the ND-sharded cache into DRAM interleaved (read_slot_kv: the sharded host read is wrong)
        part = ttnn.slice(cache, [b, 0, 0, 0], [b + 1, 1, shape[2], shape[3]], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        devs = ttnn.get_device_tensors(part)
        sp, tp, w = m.sp, m.tp, m.rows
        coord = (lambda r: r * tp) if SP_AXIS == 0 else (lambda r: r)  # chip (r, 0) in row-major mesh order
        host = torch.cat([ttnn.to_torch(devs[coord(r)]).float().reshape(-1, shape[3]) for r in range(sp)])
        ttnn.deallocate(part)
        nat = host.reshape(sp, -1, w, shape[3]).permute(1, 0, 2, 3).reshape(-1, shape[3])
        return nat[:length]

    def to_torch(self, layer: int, length: int) -> dict:
        """kv_latent, k_rope, index_key of positions [0, length); a cache the layer does not own is [length, 0]
        (the reference's state_tensors)."""
        m = self.model
        none = torch.zeros(length, 0)
        out = {"kv_latent": none, "k_rope": none.clone(), "index_key": none.clone()}
        if layer in m.kv_region and m.cfg.owns_kv(layer):  # a shared-KV layer's region is a copy of its source's
            kv = self._read(m.kvpe, len(m.kv_region), m.kv_region[layer], length)
            out["kv_latent"] = kv[:, : m.cfg.kv_lora]
            out["k_rope"] = kv[:, m.cfg.kv_lora : m.cfg.kv_lora + m.cfg.rope_dim(layer)]
        if layer in m.index_region:
            out["index_key"] = self._read(m.index, len(m.index_region), m.index_region[layer], length)
        return out


class TtModel:
    def __init__(
        self,
        mesh,
        cfg,
        layers: list[int],
        *,
        max_seq: int,
        chunk: int,
        build,
        embed_weight,
        norm_weight,
        lm_head_weight=None,
        block_graph=None,
        num_users: int = NUM_USERS,
        caches: dict | None = None,
    ):
        """cfg: the reference's Config. build(layer, step, **cache) -> the device module of one step (hooks.py: its
        weights read then); ``cache`` names the served cache and the region the module owns (attention: cache,
        cache_layer, cache_layers; indexer the same for the index cache). embed_weight [V, H], norm_weight [H],
        lm_head_weight [V, H] or None (host, any float dtype). block_graph(layer) -> list[Step]. caches: the served
        caches {"kvpe", "index"} when the serving engine owns them (tt/runners/adapter.py allocate_kv_cache, the same
        layout for num_users slots); None allocates them here."""
        t0 = time.time()
        self.mesh, self.cfg, self.layers = mesh, cfg, list(layers)
        self.max_seq, self.chunk = max_seq, chunk
        self.sp, self.tp = mesh.shape[SP_AXIS], mesh.shape[TP_AXIS]
        self.rows = chunk // self.sp
        self._build, self._graph = build, block_graph
        # regions of the served caches: one per layer that owns the cache, in layer order
        owners = [i for i in self.layers if cfg.owns_kv(i) or settings.get("KV_MIRRORS")]
        self.kv_region = {i: k for k, i in enumerate(owners)}
        for i in self.layers:
            if not cfg.owns_kv(i):
                assert (
                    cfg.kv_source(i) in self.kv_region
                ), f"layer {i} reads layer {cfg.kv_source(i)}'s KV, not in {layers}"
        self.index_region = {i: k for k, i in enumerate(j for j in self.layers if cfg.is_full(j))}
        self.kvpe_width = cfg.kv_lora + cfg.qk_rope  # NoPE layers keep the 576-wide record (pe zero)
        shape = tuple(mesh.shape)
        self.num_users = num_users
        self._own_caches = caches is None
        if caches is not None:
            self.kvpe, self.index = caches["kvpe"], caches["index"]
        else:
            self.kvpe, self.index = alloc_caches(
                mesh, cfg, max_seq, len(self.kv_region), len(self.index_region), num_users
            )
        for t in (self.kvpe, self.index):
            if t is not None:  # None: the serving engine binds its caches later (tt/runners/runtime.py)
                _stamp_sp_topology(t, mesh)

        from models.demos.deepseek_v3_d_p.tt.mistral_large_4.norm import TtDistributedRmsNorm

        self.embedding = ttnn.from_torch(
            embed_weight.to(torch.bfloat16),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=shape, dims=(None, 1) if TP_AXIS == 1 else (1, None)),
        )
        self.vocab = embed_weight.shape[0]
        self.norm = TtDistributedRmsNorm(mesh, norm_weight, cfg.eps)
        self.lm_head = None
        if lm_head_weight is not None:
            dims = [None, None]
            dims[TP_AXIS] = 2
            self.lm_head = ttnn.from_torch(
                lm_head_weight.float()[None, None],
                device=mesh,
                dtype=ttnn.bfloat8_b,  # rule 7: weights bfp8; [V/4, H] per chip, read with transpose_b
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=shape, dims=dims),
            )
        self.mods = {}  # layer -> {step: module}
        self._topk = {}  # full layer -> (start, its top-k of that chunk)
        self.meta = None  # a ChunkMeta: the layers read slot / start / end on the device (trace-safe path)
        self.load_seconds = time.time() - t0

    # ---- modules
    def _modules(self, i: int) -> dict:
        if i not in self.mods:
            t0 = time.time()
            cfg = self.cfg
            m = {"attn_norm": self._build(i, "attn_norm"), "ffn_norm": self._build(i, "ffn_norm")}
            for k in ("attn_norm", "ffn_norm"):  # their readers all gather them: NORM_GATHER=input hands them on full
                m[k].gather_input = True
            src = cfg.kv_source(i)
            m["attention"] = self._build(
                i, "attention", cache=self.kvpe, cache_layer=self.kv_region[src], cache_layers=len(self.kv_region)
            )
            # a shared-KV layer with its own region (KV_MIRRORS) copies its source's rows there
            m["attention"].mirror_layer = self.kv_region[i] if src != i and i in self.kv_region else None
            # a KV source layer keeps its chunk's bf16 kvpe rows for the shared-KV layers that read them
            m["attention"].keep_chunk_kv = any(cfg.kv_source(j) == i and j != i for j in self.layers)
            if cfg.is_full(i):
                m["indexer"] = self._build(
                    i,
                    "indexer",
                    cache=self.index,
                    cache_layer=self.index_region[i],
                    cache_layers=len(self.index_region),
                )
            else:
                m["topk_shared"] = self._build(i, "topk_shared")
            for s in ("router", "experts", "shared_expert") if cfg.is_moe(i) else ("mlp",):
                m[s] = self._build(i, s)
            m["residual"] = self._build(i, "attn_residual")
            self.mods[i] = m
            self.load_seconds += time.time() - t0
        return self.mods[i]

    def _steps(self, i: int, ctx: Ctx) -> dict:
        """Device fns of layer i's steps, fn(ctx, *device inputs) -> device tensor."""
        m, cfg = self._modules(i), self.cfg
        end, slot = ctx.extra["end"], ctx.state.slot

        meta = self.meta

        def indexer(c, x):
            tk = m["indexer"](x, c.start, slot, end, meta=meta)
            old = self._topk.pop(i, None)
            if old is not None:
                ttnn.deallocate(old[1])
            self._topk[i] = (c.start, tk)
            return tk

        def topk_shared(c, x):
            src = cfg.topk_source(i)
            got = self._topk.get(src)
            assert got is not None and got[0] == c.start, f"layer {i} reuses layer {src}'s top-k, not run on this chunk"
            if settings.get("TOPK_SHARED") == "alias":
                # perf 13: the source's top-k itself; layer() never frees a _topk tensor, and the source replaces it only
                # when it runs the next chunk, after every layer of this one
                return got[1]
            return m["topk_shared"](got[1])

        def attention(c, x, tk):
            mod = m["attention"]
            if mod.owns_kv:
                return mod(x, tk, c.start, slot, end, meta=meta)
            srcmod = self._modules(cfg.kv_source(i))["attention"]
            assert (
                srcmod.chunk_kv_start == c.start
            ), f"layer {i} reads layer {cfg.kv_source(i)}'s KV, not run on this chunk"
            if mod.mirror_layer is not None:
                mod.write_rows(srcmod.chunk_kv, mod.mirror_layer, c.start, slot, end, meta=meta)
            return mod(x, tk, c.start, slot, end, srcmod.chunk_kv, meta=meta)

        add = m["residual"]
        fns = {
            "attn_norm": lambda c, x: m["attn_norm"](x),
            "attention": attention,
            "attn_residual": lambda c, a, b: add(a, b),
            "ffn_norm": lambda c, x: m["ffn_norm"](x),
            "mlp_residual": lambda c, a, b: add(a, b),
        }
        fns["indexer" if cfg.is_full(i) else "topk_shared"] = indexer if cfg.is_full(i) else topk_shared
        def _router_ids():  # ROUTER_OUT=topk: the router's expert ids of this chunk (None: dense routing)
            return m["router"].take_indices() if cfg.is_moe(i) else None

        overlap_on = settings.get("MOE_OVERLAP") != "off"
        if cfg.is_moe(i) and settings.get("MOE_REDUCE") == "fused":
            # perf 12: the experts' group partial is added into the shared expert's fp32 partial and one
            # reduce_scatter sums both; experts_out is that partial, shared_out the block's MoE output
            def shared_fused(c, x, ex):
                sh = m["shared_expert"]
                part = stash.pop("sh", None)
                if part is None:
                    part = sh.partial(x)
                tot = ttnn.add(part, ex, dtype=part.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(part)
                return sh.reduce(tot, m["experts"].tp_topology)

            stash = {}

            def experts_fused(c, x, r):
                # MOE_OVERLAP: the shared expert's partial is computed while dispatch runs (tt/sub_device.py)
                ov = (lambda sub: stash.__setitem__("sh", m["shared_expert"].partial(x, sub))) if overlap_on else None
                stash["ex"] = m["experts"](x, r, reduce=False, indices=_router_ids(), overlap=ov)
                return stash["ex"]

            fns.update(
                router=lambda c, x: m["router"](x),
                experts=experts_fused,
                shared_expert=lambda c, x: shared_fused(c, x, stash.pop("ex")),
                moe_combine=lambda c, a, b: b,
            )
        elif cfg.is_moe(i):
            fns.update(
                router=lambda c, x: m["router"](x),
                experts=lambda c, x, r: m["experts"](x, r, indices=_router_ids()),
                shared_expert=lambda c, x: m["shared_expert"](x),
                moe_combine=lambda c, a, b: add(a, b),
            )
        else:
            fns["mlp"] = lambda c, x: m["mlp"](x)
        return fns

    def resize(self, max_seq: int) -> None:
        """Long ISL: every max_seq-sized device tensor (the served caches, the attention / indexer gather scratches,
        the RoPE, softmax-scaling and position tables, the split tables of the metadata path) freed and allocated
        again for ``max_seq``, without reading the weights again. The caches must be the model's own; they come back
        zeroed by the next new_state, so a new sequence follows. A layer not built yet builds later at the size its
        builder is given (the caller's spec). Frees everything first, so a size that does not fit fails in an
        allocation here (resize again, smaller). Growing in place can fail on fragmentation (the freed per-layer
        tables leave holes between the weights; long ISL 1): build at the largest size and resize down."""
        assert max_seq % self.chunk == 0, (max_seq, self.chunk)  # whole chunks: block-cyclic blocks split over SP
        assert self._own_caches, "the serving engine owns the caches"
        from models.demos.deepseek_v3_d_p.tt.mistral_large_4 import rope, seqbuf

        for _, tk in self._topk.values():
            ttnn.deallocate(tk)
        self._topk = {}
        if self.meta is not None:
            self.meta.free_derived()
        for m in self.mods.values():
            m["attention"].release_seq()
            if "indexer" in m:
                m["indexer"].release_seq()
        seqbuf.release(self.mesh)
        rope.release_split(self.mesh)
        for t in (self.kvpe, self.index):
            if t is not None and t.is_allocated():
                ttnn.deallocate(t)
        self.kvpe = self.index = None
        self.max_seq = max_seq
        self.kvpe, self.index = alloc_caches(
            self.mesh, self.cfg, max_seq, len(self.kv_region), len(self.index_region), self.num_users
        )
        for t in (self.kvpe, self.index):
            _stamp_sp_topology(t, self.mesh)
        for m in self.mods.values():
            m["attention"].build_seq(max_seq, self.kvpe)
            if "indexer" in m:
                m["indexer"].build_seq(max_seq, self.index)

    # ---- the model interface (testing/harness.py, serving_contract.md)
    def new_state(self, max_seq: int) -> State:
        """A new sequence: the caches zeroed, no cross-layer value of an earlier chunk kept."""
        for _, tk in self._topk.values():
            ttnn.deallocate(tk)
        self._topk = {}
        for m in self.mods.values():
            m["attention"].chunk_kv_start = None
        return State(self, max_seq)

    def embed(self, tokens, start: int = 0) -> ttnn.Tensor:
        """tokens [S] natural order (torch, PAD_ID past the real ones) -> [1, 1, S/8, H/4] bf16 per chip, mesh row r
        holding the positions g of the chunk at ``start`` with (g // (S/8)) % 8 == r, ascending (the H2D layout)."""
        t = torch.as_tensor(tokens).to(torch.int64).reshape(-1)
        assert t.numel() == self.chunk, f"{t.numel()} tokens, a chunk is {self.chunk}"
        g = torch.arange(start, start + self.chunk)
        order = sorted(range(self.chunk), key=lambda k: ((int(g[k]) // self.rows) % self.sp, k))
        # PAD_ID does not fit the int32 host tensor ttnn converts from: it goes up as V - 1, which the device clamp
        # (embed_device, the engine's path) also makes of it
        ids = t[order].clamp(max=self.vocab - 1).to(torch.int32).reshape(self.sp, 1, self.rows)
        dims = [None, None]
        dims[SP_AXIS] = 0
        dev = ttnn.from_torch(
            ids,
            device=self.mesh,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=dims),
        )
        return self.embed_device(dev)

    def embed_device(self, ids: ttnn.Tensor) -> ttnn.Tensor:
        """The engine's uint32 ROW_MAJOR ids [1, 1, S/8] per chip -> the residual. PAD_ID is clamped to V - 1 on the
        device (uint32 minimum is exact); pad rows are masked later by actual_end."""
        tile = ttnn.to_layout(ids, ttnn.TILE_LAYOUT)  # known_issues: uint32 minimum on TILE, exact
        safe = ttnn.minimum(tile, self.vocab - 1)
        ttnn.deallocate(tile)
        safe_rm = ttnn.to_layout(safe, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(safe)
        h = ttnn.embedding(safe_rm, self.embedding, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(safe_rm)
        return ttnn.reshape(h, [1, 1, self.rows, h.shape[-1]])

    def from_host(self, h: torch.Tensor) -> ttnn.Tensor:
        return residual_to_device(h.float(), self.mesh)

    def to_host(self, h: ttnn.Tensor) -> torch.Tensor:
        """[S, H] in SP-row order (natural order for a chunk-aligned start)."""
        return residual_to_host(h, self.mesh)

    def layer(self, i: int, h: ttnn.Tensor, start: int, state: State, end: int | None = None) -> ttnn.Tensor:
        """Block i on the chunk at ``start`` (rows at or past ``end`` are pad). Returns its output; h is the caller's."""
        end = start + self.chunk if end is None else end
        assert start % ttnn.TILE_SIZE == 0 and start < end <= start + self.chunk <= self.max_seq, (start, end)
        ctx = Ctx(i, start, self.chunk, state, extra={"end": end})
        fns = self._steps(i, ctx)
        env = {}
        out = run_block(self._graph(i), lambda n: fns[n], ctx, h, rec=lambda n, t: env.__setitem__(n, t))
        # free every boundary but the block input (the caller's), the output and a full layer's top-k (_topk)
        keep = {id(h), id(out)} | {id(tk) for _, tk in self._topk.values()}
        done = set()
        for t in env.values():
            if id(t) not in keep and id(t) not in done:
                done.add(id(t))
                ttnn.deallocate(t)
        return out

    def final_norm(self, h: ttnn.Tensor) -> ttnn.Tensor:
        return self.norm(h)

    # ---- the trace-safe forward
    def enable_meta(self):
        """Allocate the ChunkMeta the layers read from now on (tt/chunk_meta.py). Returns it."""
        from models.demos.deepseek_v3_d_p.tt.mistral_large_4.chunk_meta import ChunkMeta

        if self.meta is None:
            self.meta = ChunkMeta(self.mesh, self.chunk)
        return self.meta

    def forward_meta(self, h_in: ttnn.Tensor, state: State, layers, final: bool = True, after_layer=None):
        """One chunk through ``layers`` with every per-chunk value read from self.meta (written before with
        meta.write): unpack the record, the layers, then the final norm (``final``). The same ops for every chunk, so it
        can be captured once and replayed (h_in a persistent input). h_in stays the caller's. after_layer(k, layer):
        called after each layer's ops (the serving runtime's acks). Returns the output (final norm or last hidden)."""
        meta = self.meta
        assert meta is not None, "enable_meta() first"
        slot, start, end = meta.host  # host view only: the ops read the device record
        meta.unpack()
        state.slot = slot
        h = h_in
        for k, i in enumerate(layers):
            out = self.layer(i, h, start, state, end=end)
            if h is not h_in:
                self.free(h)
            h = out
            if after_layer is not None:
                after_layer(k, i)
        if not final:
            return h
        out = self.final_norm(h)
        if h is not h_in:
            self.free(h)
        return out

    def logits(self, hidden: ttnn.Tensor, rows) -> torch.Tensor:
        """Softcapped logits [len(rows), V] of chunk rows ``rows`` (SP-row order): the rows go through the host
        (outside the layers), the LM head and the softcap run on the device."""
        assert self.lm_head is not None, "built without the LM head (lm_head=False)"
        rows = torch.as_tensor(rows).long()
        x = self.to_host(hidden)[rows]
        n = -(-len(rows) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        xp = torch.zeros(n, x.shape[1])
        xp[: len(rows)] = x
        xd = ttnn.from_torch(
            xp[None, None],
            device=self.mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        cfg = ttnn.init_device_compute_kernel_config(
            self.mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
        )
        lg = ttnn.matmul(xd, self.lm_head, transpose_b=True, dtype=ttnn.float32, compute_kernel_config=cfg)
        ttnn.deallocate(xd)
        cap = float(self.cfg.softcap)
        sc = ttnn.multiply(ttnn.tanh(ttnn.multiply(lg, 1.0 / cap)), cap)
        ttnn.deallocate(lg)
        dims = [0, 0]
        dims[TP_AXIS] = 3
        out = ttnn.to_torch(
            sc, mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=tuple(dims))
        )
        ttnn.deallocate(sc)
        return out[0, 0, : len(rows)].float()

    def free(self, h) -> None:
        if h is not None and h.is_allocated():
            ttnn.deallocate(h)

    def sync(self) -> None:
        ttnn.synchronize_device(self.mesh)
