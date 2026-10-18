# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash, the all-device model (assemble step M.1): the hidden state stays on the device from the
embedding to the final norm, every block runs ``run_block`` over the reference's block graph
(reference/ds41_ref.py:block_steps) with the validated device modules (one profiler section per step).

Resting layout between steps (components.yaml header, tt/layout.py): per chip (r, c) of the SP x TP mesh

    block boundary (in / x / out)  ``DevH``: streams [1, 1, S/SP, 4 x H/TP] fp32 (stream-major) + the pending pre
                                   [1, 1, S/SP, 4] fp32 (row-split, replicated over TP; DS41-3)
    h_mid                          streams [1, 1, S/SP, 4 x H/TP] fp32
    mHC mixes                      [1, 1, S/SP, 24] fp32, replicated over TP
    attn_in / attn_norm / ffn_in   [1, 1, S/SP, H/TP] fp32 (hidden split)
    q_resid / kv                   [1, 1, S/SP, 1280] fp32 / [1, 1, S/SP, 512] bf16 (row-split, TP-replicated)
    ffn_norm                       [1, 1, S/SP, H] fp32 (whole rows, replicated over TP)
    experts_out / shared_out       the fp32 partials [1, 1, S/SP, H] before the one MoE exchange (moe_add sums them,
                                   then one exact all_gather + SFPU add over TP: components.yaml moe_add notes)

Rows are the chunk's natural blocks: mesh row r holds positions [start + r S/SP, start + (r + 1) S/SP).

What the assembled model shares between steps (no host work per chunk; agent rule 5):
  * q_a and kv: one fused [wq_a | wkv] projection (tt/q_a.TtQa.with_proj -> tt/kv.TtWindowKv.from_proj): one matmul,
    one exchange.
  * RoPE tables: one per rope type (ratio 0 / compressed), the compressed-row tables per ratio, the window tables and
    the sparse index permutations: built once at load, sliced on the device per chunk.
  * Per-layer state (DS41DeviceState): the served caches (0_window_kv per layer; compress / index per KV source),
    each index source's device top-k per chunk start (the readers of the chunk take it; the ladder's state check
    reads them back), the candidate source's device pool per chunk start, the engram tail (host state per slot).
  * Engram rows: the model's host input (components.yaml engram_rows, CPU): hashed, looked up and sent to the device
    in ``begin_chunk(tokens, start)``, before the first layer; the engram_rows step only hands that tensor on.
"""

from __future__ import annotations

import time

import torch

import ttnn
from models.demos.common.bringup.reference.interface import Ctx, run_block

from . import ccl
from .attention import TtWindowAttention, TtWindowCache, WindowTables, attention_weights
from .collapse import TtHcCollapse
from .common import DRAM, SP_AXIS, TP_AXIS
from .compressor import CompressRope, TtCompressCaches, TtCompressor, compressor_weights
from .engram import TtEngramRows
from .engram_add import build_engram_add
from .engram_kv import build_engram_kv
from .experts import build_experts
from .indexer import TtCandidateIndexer, TtReindexer, build_indexer
from .kv import TtWindowKv, attn_in_weights, rope_freqs
from .layout import HC, hidden_split_to_host, row_split_to_device, row_split_to_host, streams_to_device, streams_to_host
from .mhc import build_hc
from .norm import TtDistributedRmsNorm, build_norm
from .q_a import TtAttnInProj, TtQa
from .residual import build_residual
from .rope import RopeTable
from .router import build_router
from .shared_expert import build_shared_expert
from .sparse_attention import SparseIndexTables, TtSparseAttention

NONE_ID = 0xFFFFFFFF  # the device's "no index" (top-k / candidate pads)


class DevH:
    """A block boundary on the device: the 4 mHC streams and the pending pre (DS41-3)."""

    __slots__ = ("streams", "pre")

    def __init__(self, streams: ttnn.Tensor, pre: ttnn.Tensor):
        self.streams, self.pre = streams, pre

    def tensors(self):
        return [self.streams, self.pre]


class Keep:
    """A step output the per-layer state owns (a served-cache marker, a top-k kept for this chunk's readers): the
    block's freeing never touches it."""

    __slots__ = ("value",)

    def __init__(self, value=None):
        self.value = value


def _tensors(v) -> list:
    if isinstance(v, ttnn.Tensor):
        return [v]
    if isinstance(v, DevH):
        return v.tensors()
    if isinstance(v, (tuple, list)):
        return [t for x in v for t in _tensors(x)]
    return []  # Keep, None, markers


def _free(v, protect: set[int] = frozenset()) -> None:
    for t in _tensors(v):
        if id(t) not in protect and t.is_allocated():
            ttnn.deallocate(t)


def _ids_to_host(t: ttnn.Tensor, mesh) -> torch.Tensor:
    """A row-split, TP-replicated uint32 index tensor [1, 1, W, k] -> host [S, k] int32, 0xFFFFFFFF -> -1 (harness
    boundary: the ladder's state read-back)."""
    idx = row_split_to_host(mesh, t).to(torch.int64) & 0xFFFFFFFF
    return torch.where(idx >= NONE_ID, torch.full_like(idx, -1), idx).to(torch.int32)


def _cands_to_host(t: ttnn.Tensor) -> torch.Tensor:
    """The candidate pool [1, 1, Sq, n] uint32 per chip (chip (r, c) = query rows (r TP + c) Sq, device order) ->
    host [S, n] int32 in the reference's format (ascending block ids, -1 = none, last)."""
    rows = torch.cat([ttnn.to_torch(x) for x in ttnn.get_device_tensors(t)], dim=-2)
    n = rows.shape[-1]
    ids = rows.reshape(-1, n).to(torch.int64) & 0xFFFFFFFF
    none = ids >= NONE_ID
    ids = torch.sort(torch.where(none, torch.full_like(ids, 1 << 40), ids), dim=-1).values
    return torch.where(ids >= (1 << 40), -1, ids).to(torch.int32)


# ============================================================================================= state
class DS41DeviceState:
    """Per-layer prefill state of one slot, on the device (the engram tail on the host: it depends only on the ids).

    window[i]    TtWindowCache (served 0_window_kv) of every layer
    compress[i]  TtCompressCaches (served compressed KV + index keys) of every KV-source layer
    topk[i]      {chunk start: device top-k [1, 1, S/SP, K] uint32} of every index-source layer (this chunk's readers
                 take the entry of their start; earlier entries stay for the state read-back)
    cands[i]     {chunk start: device candidate pool [1, 1, Sq, 2048] uint32 per chip} of the candidate source
    tail[i]      engram tail [3] int64 (host) of every engram layer
    The ``*_prefix`` dicts hold a golden prefix's integer state loaded on the host (ladder rung ``last``).

    ``slabs`` {layer: {config name: tt/served.CacheSlab}}: this slot's slabs of the serving engine's cache tensors
    (tt/runners/kv_contract.py); None = caches of its own (the ladder). ``keep_history`` False (serving): a store keeps
    only the latest chunk's top-k / pool (its readers run in the same chunk), so nothing grows with the requests."""

    def __init__(self, model: "TtDS41Model", max_seq: int, slabs: dict | None = None, keep_history: bool = True):
        cfg, mesh, tables = model.cfg, model.mesh, model.tables
        self.model, self.cfg, self.mesh, self.max_seq = model, cfg, mesh, max_seq
        self.keep_history = keep_history
        slabs = slabs or {}
        self.window = {
            i: TtWindowCache(mesh, tables, max_seq, slab=slabs.get(i, {}).get("0_window_kv")) for i in model.layers
        }
        self.compress = {
            i: TtCompressCaches(mesh, tables, max_seq, cfg.ratio(i), cfg.head_dim, cfg.index_dim, slabs=slabs.get(i))
            for i in model.layers
            if cfg.is_kv_source(i)
        }
        for b in model.blocks.values():  # the ring's gathered-key buffer per cache capacity (load time)
            if b.indexer is not None:
                b.indexer.setup(self.compress[b.kv_src].index_k)
        self.topk = {i: {} for i in model.layers if cfg.is_index_source(i)}
        self.cands = {i: {} for i in model.layers if cfg.is_cand_source(i)}
        self.topk_prefix, self.cands_prefix = {}, {}
        self.tail = {i: model.engram[i].new_tail() for i in model.engram}
        self.tokens = None
        self.history = None  # this slot's real token ids by position (host; the engram tail of any start)
        model.state = self  # begin_chunk reads (and advances) the active slot's engram tail

    # ------------------------------------------------------------------ per chunk (device)
    def put(self, store: dict, layer: int, start: int, t: ttnn.Tensor) -> None:
        stale = [start] if self.keep_history else list(store[layer])
        for k in stale:
            old = store[layer].pop(k, None)
            if old is not None:
                _free(old)
        store[layer][start] = t

    def note_tokens(self, tokens: torch.Tensor, start: int, end: int) -> None:
        """Host, before ``begin_chunk``: the chunk's real ids [start, end) into this slot's id history, and every
        engram layer's tail = the compressed ids of the 3 positions before ``start`` (a follow-up turn starts inside
        the slot's earlier prompt, so the tail its last chunk left is not this chunk's)."""
        if self.history is None:
            self.history = torch.zeros(self.max_seq, dtype=torch.int64)
        self.history[start:end] = tokens.reshape(-1)[: end - start].long()
        for i, rows in self.model.engram.items():
            t = rows.new_tail()
            last = self.history[max(0, start - t.shape[0]) : start]
            if last.numel():
                t[-last.numel() :] = rows.hasher.token_map[last]
            self.tail[i] = t

    def get(self, store: dict, layer: int, start: int, what: str) -> ttnn.Tensor:
        t = store.get(layer, {}).get(start)
        assert t is not None, f"no device {what} of layer {layer} for the chunk at {start} (its source runs first)"
        return t

    # ------------------------------------------------------------------ harness boundary (host)
    def load_prefix(self, layer: int, tensors: dict, length: int) -> None:
        cfg = self.cfg
        self.window[layer].load_ring(tensors["window_kv"], length)
        if layer in self.compress and length:
            c = self.compress[layer]
            c.ckv.load_host(tensors["compress_kv"][: length // c.ratio], 0)
            c.index_k.load_host(tensors["index_k"][: length // c.ratio], 0)
        if layer in self.topk:
            self.topk_prefix[layer] = tensors["topk_idx"][:length].to(torch.int32).clone()
        if layer in self.cands:
            self.cands_prefix[layer] = tensors["candidates"][:length].to(torch.int32).clone()
        if layer in self.tail:
            self.tail[layer] = tensors["engram_tail"].long().clone()
        assert cfg.ratio(layer) < 2 or not length % cfg.ratio(layer), (layer, length)

    def _index_rows(self, layer: int, length: int, store: dict, prefix: dict, width: int, per_chip: bool):
        out = torch.full((length, width), -1, dtype=torch.int32)
        if layer in prefix:
            p = prefix[layer][:length]
            out[: p.shape[0]] = p
        for start, t in sorted(store[layer].items()):
            if start >= length:
                continue
            rows = _cands_to_host(t) if per_chip else _ids_to_host(t, self.mesh)
            n = min(rows.shape[0], length - start)
            out[start : start + n] = rows[:n]
        return out

    def to_torch(self, layer: int, length: int) -> dict:
        cfg, out = self.cfg, {}
        for n in cfg.state_names(layer):
            if n == "window_kv":
                out[n] = self.window[layer].ring_host(length)
            elif n == "compress_kv":
                c = self.compress[layer]
                out[n] = c.ckv.to_host(length)[: length // c.ratio].float()
            elif n == "index_k":
                c = self.compress[layer]
                out[n] = c.index_k.to_host(length)[: length // c.ratio].float()
            elif n == "compress_partial":
                # every chunk start and length is a multiple of 32 (and of the ratio): no group is ever pending
                # (serving_contract.md "Compressor state"; reference _compressor leaves zeros)
                r = cfg.ratio(layer)
                out[n] = torch.zeros(2, r, cfg.head_dim)
            elif n == "topk_idx":
                out[n] = self._index_rows(layer, length, self.topk, self.topk_prefix, cfg.index_topk, False)
            elif n == "candidates":
                out[n] = self._index_rows(layer, length, self.cands, self.cands_prefix, cfg.cand_topk_blocks, True)
            elif n == "engram_tail":
                out[n] = self.tail[layer].clone()
        return out

    def served(self, layer: int, length: int) -> dict:
        """{config name: natural-order rows of positions [0, ceil32(length))} (tests/bringup/contract/parts.py)."""
        out = {"0_window_kv": self.window[layer].to_host(length)}
        if layer in self.compress:
            out.update({n: t.to_host(length) for n, t in self.compress[layer].by_config().items()})
        return out

    def load_served(self, layer: int, tensors: dict, length: int) -> None:
        self.window[layer].load_host(tensors["0_window_kv"][:length], 0)
        if layer in self.compress:
            for n, t in self.compress[layer].by_config().items():
                if n in tensors:
                    t.load_host(tensors[n][: length // t.ratio], 0)

    def load_tokens(self, ids: torch.Tensor) -> None:
        """A migrated prefix's ids: the engram tail of every engram layer = the compressed ids of the last 3."""
        self.tokens = ids.reshape(-1).long()
        if self.history is None:
            self.history = torch.zeros(self.max_seq, dtype=torch.int64)
        self.history[: self.tokens.numel()] = self.tokens
        for i, rows in self.model.engram.items():
            t = rows.new_tail()
            last = self.tokens[-t.shape[0] :]
            if last.numel():
                t[-last.numel() :] = rows.hasher.token_map[last]
            self.tail[i] = t

    def formats(self) -> dict:
        out = {"0_window_kv": ttnn.bfloat16}
        for c in self.compress.values():
            out.update({n: ttnn.bfloat16 for n in c.names})
        return out

    def free(self) -> None:
        for c in [*self.window.values(), *self.compress.values()]:
            c.free()
        for store in (self.topk, self.cands):
            for d in store.values():
                for t in d.values():
                    _free(t)
        self.window, self.compress, self.topk, self.cands = {}, {}, {}, {}
        if self.model.state is self:
            self.model.state = None


# ============================================================================================= block
class TtDS41Block:
    """One decoder layer on the device: run_block over the reference's graph, every step a validated module."""

    def __init__(self, model: "TtDS41Model", layer: int):
        from models.demos.deepseek_v41_flash_d_p.reference.ds41_ref import block_steps

        cfg, mesh, ck, i = model.cfg, model.mesh, model.ck, layer
        self.model, self.cfg, self.mesh, self.layer = model, cfg, mesh, i
        self.steps = block_steps(cfg, i)
        r = cfg.ratio(i)
        self.kv_src, self.idx_src = cfg.kv_source_of(i), cfg.index_source_of(i)
        rope = model.rope(r)
        p = f"layers.{i}."
        # mHC, norms, residuals
        self.attn_hc = build_hc(mesh, ck, cfg, i, "attn_hc")
        self.ffn_hc = build_hc(mesh, ck, cfg, i, "ffn_hc")
        self.collapse = TtHcCollapse(n=cfg.hc)
        self.attn_norm = build_norm(mesh, ck, cfg, i, "attn_norm")
        self.ffn_norm = build_norm(mesh, ck, cfg, i, "ffn_norm")
        self.attn_res = build_residual(cfg)
        self.ffn_res = build_residual(cfg, step="ffn_residual")
        # attention input: one fused [wq_a | wkv] projection for q_a and kv
        proj = TtAttnInProj(mesh, attn_in_weights(ck, cfg, i))
        self.q_a = TtQa(mesh, proj, ck.weight(p + "attn.q_norm.weight", torch.float32), cfg.eps)
        kv_norm = ck.weight(p + "attn.kv_norm.weight", torch.float32)
        self.kv = TtWindowKv(mesh, proj, kv_norm, cfg.eps, rope, cfg.rope_dim)
        self.compressor = None
        if cfg.is_kv_source(i):
            self.compressor = TtCompressor(
                mesh, compressor_weights(ck, i), cfg, r, model.compress_rope(r), model.tables
            )
        self.indexer = None
        if cfg.is_index_source(i):
            self.indexer = build_indexer(mesh, ck, cfg, i, model.max_seq, model.chunk, rope=rope)
        w = attention_weights(ck, i)
        if r:
            self.attention = TtSparseAttention(mesh, cfg, w, rope, model.tables, model.sparse_tables())
        else:
            self.attention = TtWindowAttention(mesh, cfg, w, rope, model.tables)
        # engram
        self.engram_kv = self.engram_add = None
        if cfg.is_engram(i):
            self.engram_kv = build_engram_kv(mesh, ck, cfg, i)
            self.engram_add = build_engram_add(mesh, ck, cfg, i)
        # MoE
        self.router = build_router(mesh, ck, cfg, i, model.chunk)
        self.experts = build_experts(mesh, ck, cfg, i, model.chunk)
        self.shared = build_shared_expert(mesh, ck, cfg, i)
        self.fns = self._fns()
        # boundaries freed after their last reader (never "in": the caller owns it; never "out")
        last = {}
        for s in self.steps:
            for name in s.inputs:
                last[name] = s.name
        self.dead = {
            s.name: [n for n in dict.fromkeys(s.inputs) if last[n] == s.name and n != "in"] for s in self.steps
        }

    # ------------------------------------------------------------------ steps
    def _fns(self) -> dict:
        cfg, i = self.cfg, self.layer

        def engram_rows(ctx):
            return self.model.take_engram_rows(i)

        def engram_kv(ctx, rows):
            return self.engram_kv(rows)

        def engram_add(ctx, x, kv):
            s, pre = self.engram_add(x.streams, kv, x.pre)
            return DevH(s, pre)

        def attn_hc(ctx, x):
            return self.attn_hc(x.streams)

        def attn_collapse(ctx, x):
            return self.collapse(x.streams, x.pre)

        def attn_norm(ctx, x):
            return self.attn_norm(x)

        def q_a(ctx, x):
            y, p = self.q_a.with_proj(x)
            ctx.extra["attn_proj"] = p
            return y

        def kv(ctx, x):
            p = ctx.extra.pop("attn_proj")
            out = self.kv.from_proj(p, ctx.start)
            ttnn.deallocate(p)
            return out

        def compressor(ctx, x):
            ckv, k = self.compressor(x, ctx.start)
            self.compressor.write(ckv, k, ctx.start, ctx.extra["end"], ctx.state.compress[i])
            ttnn.deallocate(ckv)
            ttnn.deallocate(k)
            return Keep()  # the chunk's rows live in the served caches the indexer and the attention read

        def indexer(ctx, x, qr, *compressed):
            st, start = ctx.state, ctx.start
            cache = st.compress[self.kv_src].index_k
            if isinstance(self.indexer, TtReindexer):
                cands = st.get(st.cands, cfg.cand_source, start, "candidate pool")
                top = self.indexer(x, qr, start, cache, cands)
            elif isinstance(self.indexer, TtCandidateIndexer):
                top, cands = self.indexer.forward(x, qr, start, cache)
                st.put(st.cands, i, start, cands)
            else:
                top = self.indexer(x, qr, start, cache)
            st.put(st.topk, i, start, top)
            return Keep(top)

        def attention(ctx, qr, kvt, *rest):
            st, start, end = ctx.state, ctx.start, ctx.extra["end"]
            if not cfg.ratio(i):
                return self.attention(qr, kvt, start, end, st.window[i])
            if self.idx_src == i:
                top = rest[-1].value
            else:
                top = st.get(st.topk, self.idx_src, start, "top-k")
            ckv = st.compress[self.kv_src].ckv
            return self.attention(qr, kvt, start, end, st.window[i], ckv, top)

        def attn_residual(ctx, x, mix, y):
            return self.attn_res(x.streams, mix, y)

        def ffn_hc(ctx, h):
            return self.ffn_hc(h)

        def ffn_collapse(ctx, h, mix):
            return self.collapse(h, mix)

        def ffn_norm(ctx, x):
            return self.ffn_norm(x)

        def router(ctx, x):
            return self.router(x)  # (dense [S/SP, E] fp32 RM, top-k ids, weights)

        def experts(ctx, x, routing):
            dense, idx, _ = routing
            return self.experts.partial(x, dense, idx)  # this dispatch group's fp32 partial [S/SP, H]

        def shared_expert(ctx, x):
            return self.shared.partial(x)  # this chip's intermediate share, fp32 partial [S/SP, H]

        def moe_add(ctx, a, b):
            # the two partials summed before the one exchange over TP (components.yaml moe_add notes)
            # CCL_IMPL=fabric_bf16 (P.6): the sum written in bf16 for the bf16 reduce-scatter (tt/ccl.py:partial_dtype)
            sdt = ccl.partial_dtype(self.mesh) if self.experts.sum_mode == "gather_add" else ttnn.float32
            s = ttnn.add(a, b, dtype=sdt, memory_config=DRAM)
            out = self.experts.reduce(s)
            ttnn.deallocate(s)
            return out

        def ffn_residual(ctx, h, mix, y):
            o, pre = self.ffn_res(h, mix, y)
            return DevH(o, pre)

        table = dict(
            engram_rows=engram_rows,
            engram_kv=engram_kv,
            engram_add=engram_add,
            attn_hc=attn_hc,
            attn_collapse=attn_collapse,
            attn_norm=attn_norm,
            q_a=q_a,
            kv=kv,
            compressor=compressor,
            indexer=indexer,
            attention=attention,
            attn_residual=attn_residual,
            ffn_hc=ffn_hc,
            ffn_collapse=ffn_collapse,
            ffn_norm=ffn_norm,
            router=router,
            experts=experts,
            shared_expert=shared_expert,
            moe_add=moe_add,
            ffn_residual=ffn_residual,
        )
        missing = [s.name for s in self.steps if s.name not in table]
        assert not missing, f"layer {i}: no device step for {missing}"
        return {s.name: table[s.name] for s in self.steps}

    def __call__(self, h: DevH, start: int, state: DS41DeviceState, end: int | None = None) -> DevH:
        rows = h.streams.shape[-2] * self.mesh.shape[SP_AXIS]
        assert rows == self.model.chunk, f"chunk of {rows} rows; the model was built for {self.model.chunk}"
        end = start + rows if end is None else int(end)
        ctx = Ctx(self.layer, start, rows, state, extra={"end": end})
        protect = {id(t) for t in h.tensors()}  # the caller's input (an engram layer's x shares its pre)

        def wrap(step, fn):
            dead = self.dead[step.name]

            def run(c, *args):
                out = fn(c, *args)
                keep = protect | {id(t) for t in _tensors(out)}
                inputs = dict(zip(step.inputs, args))
                for n in dead:
                    _free(inputs[n], keep)
                return out

            return run

        overrides = {s.name: wrap(s, self.fns[s.name]) for s in self.steps}

        def no_cpu(name):
            raise AssertionError(f"layer {self.layer}: step {name} has no device module (all-device model)")

        return run_block(self.steps, no_cpu, ctx, h, overrides=overrides)


# ============================================================================================= model
class TtDS41Model:
    """The device modules of the requested layers plus the embedding and the final norm, and the load-time tables
    they share. ``layers``: sorted layer ids (a reader layer's sources must be among them)."""

    def __init__(self, mesh, model_path: str, layers, max_seq: int, chunk: int, lm_head: bool = False, hasher=None):
        from models.demos.deepseek_v41_flash_d_p.reference.ds41_ref import Config
        from models.demos.deepseek_v41_flash_d_p.reference.weights import Checkpoint, EngramTable

        from .engram import engram_hasher

        t0 = time.time()
        self.mesh, self.path = mesh, str(model_path)
        self.cfg = cfg = Config.from_dir(self.path)
        self.ck = ck = Checkpoint(self.path)
        self.layers = sorted(layers)
        self.max_seq, self.chunk = int(max_seq), int(chunk)
        self.sp, self.tp = mesh.shape[SP_AXIS], mesh.shape[TP_AXIS]
        assert self.chunk % (32 * self.sp * self.tp) == 0, (self.chunk, tuple(mesh.shape))
        for i in self.layers:
            missing = [j for j in cfg.sources(i) if j not in self.layers]
            assert not missing, f"layer {i} reads the state of layers {missing}, which are not loaded"
        ccl.setup(mesh)  # CCL_IMPL=fabric: the fabric collectives' semaphores and plans, once at load (agent rule 5)
        self.tables = WindowTables(mesh, self.chunk)
        self._rope, self._crope, self._sparse = {}, {}, None
        self.state = None
        # engram (host input): hasher + memory-mapped tables, rows sent in begin_chunk
        self.engram = {}
        if any(cfg.is_engram(i) for i in self.layers):
            hasher = hasher or engram_hasher(self.path, cfg)
            self.engram = {
                i: TtEngramRows(mesh, hasher, EngramTable(ck, i), i) for i in self.layers if cfg.is_engram(i)
            }
        self._engram_dev = {}
        # embedding: [V, H/TP] bf16 ROW_MAJOR per chip (the chip's hidden columns), replicated over SP
        emb = ck.get("embed.weight")
        assert emb.dtype == torch.bfloat16, emb.dtype  # the device copy is then exact
        self.vocab = emb.shape[0]
        self.embed_w = ttnn.from_torch(
            emb.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=DRAM,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(None, 1)),
        )
        del emb
        # layer 0's pending pre: one-hot on stream 0 (reference first_block_input), [S/SP, 4] fp32, cloned per chunk
        rows = self.chunk // self.sp
        pre0 = torch.zeros(1, 1, self.chunk, HC)
        pre0[..., 0] = 1.0
        self.pre0 = row_split_to_device(mesh, pre0.reshape(self.chunk, HC))
        assert self.pre0.shape[-2] == rows, (self.pre0.shape, rows)
        self.final_collapse = TtHcCollapse(n=cfg.hc)
        self.final_norm_mod = TtDistributedRmsNorm(mesh, ck.weight("norm.weight", torch.float32), cfg.eps)
        self.head_w = ck.weight("head.weight", torch.float32) if lm_head else None
        self.blocks = {}
        for i in self.layers:
            self.blocks[i] = TtDS41Block(self, i)
        self.load_seconds = time.time() - t0

    # ------------------------------------------------------------------ load-time shared tables
    def rope(self, ratio: int) -> RopeTable:
        key = bool(ratio)
        if key not in self._rope:
            probe = next(i for i in range(self.cfg.n_layers) if bool(self.cfg.ratio(i)) == key)
            self._rope[key] = RopeTable(self.mesh, rope_freqs(self.cfg, probe), self.max_seq)
        return self._rope[key]

    def compress_rope(self, ratio: int) -> CompressRope:
        if ratio not in self._crope:
            probe = next(i for i in range(self.cfg.n_layers) if self.cfg.ratio(i) == ratio)
            self._crope[ratio] = CompressRope(self.mesh, rope_freqs(self.cfg, probe), self.max_seq, ratio)
        return self._crope[ratio]

    def sparse_tables(self) -> SparseIndexTables:
        if self._sparse is None:
            self._sparse = SparseIndexTables(self.mesh, self.tables, self.cfg.index_topk)
        return self._sparse

    # ------------------------------------------------------------------ per chunk
    def new_state(self, max_seq: int, slabs: dict | None = None, keep_history: bool = True) -> DS41DeviceState:
        return DS41DeviceState(self, max_seq, slabs=slabs, keep_history=keep_history)

    def begin_chunk(self, tokens: torch.Tensor, start: int) -> None:
        """The host input of every engram layer: hash the chunk's ids (pad past actual_end skipped, DS41-26) with the
        active slot's tail, look the rows up (memory-mapped) and send them to the device once."""
        if not self.engram:
            return
        st = self.state
        for i, rows in self.engram.items():
            if start == 0 or st is None:
                tail = rows.new_tail()
            else:
                tail = st.tail[i]
            dev, new_tail = rows(tokens.reshape(-1), tail.long())
            if st is not None:
                st.tail[i] = new_tail
            _free(self._engram_dev.pop(i, None))
            self._engram_dev[i] = dev

    def take_engram_rows(self, layer: int) -> ttnn.Tensor:
        t = self._engram_dev.pop(layer, None)
        assert t is not None, f"layer {layer}: no engram rows for this chunk (begin_chunk(tokens, start) first)"
        return t

    def embed(self, tokens: torch.Tensor, start: int = 0) -> DevH:
        """Natural-order ids of one chunk (PAD_ID past actual_end clamped into the vocabulary, K-72: those rows are pad
        and never reach a cache) -> layer 0's input: the embedding on all 4 streams, the pending pre one-hot."""
        ids = tokens.reshape(-1).long().clamp(0, self.vocab - 1).to(torch.int32)
        assert ids.numel() == self.chunk, (ids.numel(), self.chunk)
        dev_ids = ttnn.from_torch(
            ids.reshape(self.sp, self.chunk // self.sp).contiguous(),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=DRAM,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(0, None)),
        )
        e = ttnn.embedding(dev_ids, self.embed_w, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=DRAM)
        ttnn.deallocate(dev_ids)
        rows, w = self.chunk // self.sp, self.cfg.dim // self.tp
        e4 = ttnn.reshape(e, [1, 1, rows, w])
        ef = ttnn.typecast(e4, ttnn.float32, memory_config=DRAM)
        ttnn.deallocate(e4)  # a view of e's buffer (K-1): then e is freed with it
        if e.is_allocated():
            ttnn.deallocate(e)
        streams = ttnn.concat([ef] * HC, dim=-1, memory_config=DRAM)
        ttnn.deallocate(ef)
        return DevH(streams, ttnn.clone(self.pre0, memory_config=DRAM))

    def layer(self, i: int, h: DevH, start: int, state: DS41DeviceState, end: int | None = None) -> DevH:
        return self.blocks[i](h, start, state, end)

    def final_norm(self, h: DevH) -> ttnn.Tensor:
        """rms(sum_n pre_n stream_n) w -> [1, 1, S/SP, H/TP] fp32 (hidden split)."""
        x = self.final_collapse(h.streams, h.pre)
        out = self.final_norm_mod(x)
        ttnn.deallocate(x)
        return out

    # ------------------------------------------------------------------ harness boundary (host)
    def from_host(self, h: torch.Tensor, start: int = 0) -> DevH:
        """Packed block input rows [S, 4H + 4] (natural order) -> DevH."""
        hidden = self.cfg.dim
        assert h.shape[-1] == HC * hidden + HC, h.shape
        return DevH(streams_to_device(self.mesh, h, hidden), row_split_to_device(self.mesh, h[:, HC * hidden :]))

    def to_host(self, h) -> torch.Tensor:
        """DevH -> the reference's packed [S, 4H + 4]; a hidden-split tensor (the final norm) -> [S, H]."""
        if isinstance(h, DevH):
            s = streams_to_host(self.mesh, h.streams, self.cfg.dim).float()
            return torch.cat([s, row_split_to_host(self.mesh, h.pre, HC).float()], dim=-1)
        return hidden_split_to_host(self.mesh, h).float()

    def logits(self, hidden, rows) -> torch.Tensor:
        assert self.head_w is not None, "built without the LM head"
        return torch.nn.functional.linear(self.to_host(hidden)[rows], self.head_w)

    def free(self, h) -> None:
        _free(h)

    def sync(self) -> None:
        ttnn.synchronize_device(self.mesh)
