# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash, the all-device model (assemble step M.1): the hidden state stays on the device from the
embedding to the final norm, every block runs ``run_block`` over the reference's block graph
(reference/ds41_ref.py:block_steps) with the validated device modules (one profiler section per step).

Resting layout between steps (components.yaml header, tt/layout.py): per chip (r, c) of the SP x TP mesh

    block boundary (in / x / out)  ``DevH``: streams [1, 1, S/SP, 4 x H/TP] (stream-major; bf16 under
                                   BF16_ACTIVATIONS=on (P.20), else fp32) + the pending pre [1, 1, S/SP, 4] fp32
                                   (row-split, replicated over TP; DS41-3)
    h_mid                          streams [1, 1, S/SP, 4 x H/TP] (the streams' dtype)
    mHC mixes                      [1, 1, S/SP, 24] fp32, replicated over TP
    attn_in / attn_norm / ffn_in   [1, 1, S/SP, H/TP] (hidden split; bf16 under BF16_ACTIVATIONS=on, else fp32)
    q_resid / kv                   [1, 1, S/SP, 1280] (bf16 under BF16_ACTIVATIONS=on, else fp32) / [1, 1, S/SP, 512]
                                   bf16 (row-split, TP-replicated)
    ffn_norm                       [1, 1, S/SP, H] fp32 (whole rows, replicated over TP; MOE_HEAD=fused: [S/SP, H/TP])
    experts_out / shared_out       the fp32 partials [1, 1, S/SP, H] before the one MoE exchange (moe_add sums them,
                                   then one exact all_gather + SFPU add over TP: components.yaml moe_add notes)

Rows are the chunk's natural blocks: mesh row r holds positions [start + r S/SP, start + (r + 1) S/SP).

ATTN_SP32=on (P.10, the default): attn_norm, q_resid and kv are the chip's own rows with whole hidden rows instead,
[1, 1, S/(SP TP), H | 1280 | 512] (mesh column c: rows [c S/(SP TP), (c+1) S/(SP TP)) of its row block, the rows
sparse_sdpa reads): one all_to_all of attn_in over TP before a local attn_norm (tt/common.cols_to_rows); the
attention's exit all_to_all (ATTN_SP32_PROJ, P.9) returns to the hidden split.

OWN_ROWS=on (P.23, the default; tt/common.own_rows): the whole residual rests on those rows. Per chip (r, c), rows
[(r TP + c) S/(SP TP), (r TP + c + 1) S/(SP TP)) with every column (tt/layout.own_rows_to_device):

    block boundary / h_mid         streams [1, 1, S/(SP TP), 4 x H] (stream-major) + the pending pre [.., 24] fp32
    mHC mixes, attn_in / ffn_in    [1, 1, S/(SP TP), 24] / [.., H]: hc_fn whole on every chip, no TP exchange
    attn_norm / ffn_norm           local whole-row norms; attn_out the attention's own rows (no exit all_to_all)
    engram_kv                      [1, 1, S/(SP TP), 5 x H]: one all_to_all of the projection (tt/engram_kv.py)
    MoE                            MoEOptions residual "owners": x and the output the chip's own rows
    final norm                     [1, 1, S/(SP TP), H]
The embedding's [S/SP, H/TP] rows go through one all_to_all over TP a chunk.

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

from . import ccl, host_alloc
from .attention import TtWindowAttention, TtWindowCache, WindowTables, attention_weights
from .ccl import dealloc
from .collapse import TtHcCollapse
from .common import (
    DRAM,
    SP_AXIS,
    TP_AXIS,
    act_dtype,
    attn_sp32,
    bf16_activations,
    cols_to_rows,
    own_rows,
    set_load_layer,
    stream_dtype,
)
from .compressor import CompressRope, TtCompressCaches, TtCompressor, compressor_weights
from .cores import from_torch, init_caps, max_grid, sub_cores, sub_cores_tiled
from .engram import TtEngramRows
from .engram_add import build_engram_add
from .engram_kv import build_engram_kv
from .experts import build_experts
from .indexer import TtCandidateIndexer, TtReindexer, build_indexer, topk_local
from .kv import TtWindowKv, attn_in_weights, rope_freqs
from .layout import (
    HC,
    hidden_split_to_host,
    own_rows_to_device,
    own_rows_to_host,
    row_split_to_device,
    row_split_to_host,
    streams_to_device,
    streams_to_host,
)
from .mhc import NG, build_hc
from .norm import TtDistributedRmsNorm, build_norm
from .q_a import TtAttnInProj, TtQa
from .residual import build_residual
from .rope import RopeTable, indexer_glue
from .router import build_router
from .settings import settings
from .shared_expert import build_shared_expert
from .sparse_attention import ChunkMemo, SparseIndexTables, TtSparseAttention

NONE_ID = 0xFFFFFFFF  # the device's "no index" (top-k / candidate pads)
# DECODER_SKIP=replay128 (P.31): L1 a core kept at the top for the decoder tail's MoE block's persistent words
# (moe_block_adapter.reserve_l1; the block checks its words fit, and logs what they take: 41216 B on 8x4, P.31)
TAIL_L1_RESERVE = 48 * 1024


class DevH:
    """A block boundary on the device: the 4 mHC streams and the pending pre (DS41-3). MHC_FUSED=on: ``pre`` is the
    whole [S/SP, 24] mixes of the previous ffn_hc (pre in columns 0..3; layer 0: one-hot), and ``stats`` may carry
    (layer, row): this chip's partial hc row [S/SP, 32] of the streams for that layer's attn_hc, emitted by the
    previous residual's mhc_post pass (used only by that layer, freed with the boundary)."""

    __slots__ = ("streams", "pre", "stats")

    def __init__(self, streams: ttnn.Tensor, pre: ttnn.Tensor, stats=None):
        self.streams, self.pre, self.stats = streams, pre, stats

    def tensors(self):
        return [self.streams, self.pre] + ([self.stats[1]] if self.stats is not None else [])

    def take_stats(self, layer: int):
        """The partial row for ``layer``'s attn_hc, or None (none emitted, or for another layer)."""
        if self.stats is None or self.stats[0] != layer:
            return None
        row, self.stats = self.stats[1], None
        return row


class Skipped:
    """DEEP_LAYERS=tail (FIX.17): what the last full layer (20) returns on a normal chunk, and what the skipped deep
    layers hand on: no tensors (its output went to DS41DeviceState.deep_in; ``free`` of it is a no-op). DECODER_SKIP=
    exact (P.30): what layer 20's KV projection and the skipped layers 21-39 return on a chunk that is not the last
    (``final_norm`` hands it on)."""

    __slots__ = ()


SKIPPED = Skipped()


class TailH(DevH):
    """DECODER_SKIP=replay128 (P.31): a block boundary of the decoder tail, in the tail model's geometry (the tail
    chunk [dec_start, dec_start + tail_chunk), its real rows [dec_start, end) on the first chips; TtDS41Model._tail_at).
    """

    __slots__ = ()


class TailHidden:
    """DECODER_SKIP=replay128 (P.31): the final norm of the decoder tail: ``t`` [1, 1, tail_chunk / (SP TP), H] (the tail
    model's own rows), its rows the positions [start, start + tail_chunk), real up to ``end``."""

    __slots__ = ("t", "start", "end")

    def __init__(self, t, start: int, end: int):
        self.t, self.start, self.end = t, int(start), int(end)


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
    if isinstance(v, TailHidden):
        return [v.t]
    if isinstance(v, (tuple, list)):
        return [t for x in v for t in _tensors(x)]
    return []  # Keep, None, markers


def _free(v, protect: set[int] = frozenset()) -> None:
    for t in _tensors(v):
        if id(t) not in protect and t.is_allocated():
            dealloc(t)  # GLUE_FOLD=on (P.19): the binding itself (tt/ccl.py:dealloc)


def _ids_to_host(t: ttnn.Tensor, mesh, per_chip: bool = False) -> torch.Tensor:
    """A row-split, TP-replicated uint32 index tensor [1, 1, W, k] -> host [S, k] int32, 0xFFFFFFFF -> -1 (harness
    boundary: the ladder's state read-back). ``per_chip`` (INDEXER_TOPK_LOCAL=on, P.8): [1, 1, Sq, k] of the chip's
    own query rows (chip (r, c) = rows (r TP + c) Sq, device order), concatenated in chip order; the ids as they are."""
    if per_chip:
        idx = torch.cat([ttnn.to_torch(x) for x in ttnn.get_device_tensors(t)], dim=-2)
        idx = idx.reshape(-1, idx.shape[-1]).to(torch.int64) & 0xFFFFFFFF
    else:
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
                 take the entry of their start; earlier entries stay for the state read-back); INDEXER_TOPK_LOCAL=on
                 (P.8): [1, 1, Sq, K], each chip's own query rows
    cands[i]     {chunk start: device candidate pool [1, 1, Sq, 2048] uint32 per chip} of the candidate source
    tail[i]      engram tail [3] int64 (host) of every engram layer
    The ``*_prefix`` dicts hold a golden prefix's integer state loaded on the host (ladder rung ``last``).

    ``slabs`` {layer: {config name: tt/served.CacheSlab}}: this slot's slabs of the serving engine's cache tensors
    (tt/runners/kv_contract.py); None = caches of its own (the ladder). ``keep_history`` False (serving): a store keeps
    only the latest chunk's top-k / pool (its readers run in the same chunk), so nothing grows with the requests."""

    def __init__(
        self,
        model: "TtDS41Model",
        max_seq: int,
        slabs: dict | None = None,
        keep_history: bool = True,
        compress: dict | None = None,
    ):
        """``compress`` (DECODER_SKIP=replay128, P.31): {layer: TtCompressCaches} of another state, shared (the decoder
        tail's state reads the main state's layer-20 caches; never freed here)."""
        cfg, mesh, tables = model.cfg, model.mesh, model.tables
        self.model, self.cfg, self.mesh, self.max_seq = model, cfg, mesh, max_seq
        self.keep_history = keep_history
        slabs = slabs or {}
        self.window = {
            i: TtWindowCache(mesh, tables, max_seq, slab=slabs.get(i, {}).get("0_window_kv")) for i in model.layers
        }
        shared = compress or {}
        self.shared_compress = frozenset(shared)
        self.compress = {
            i: shared[i]
            if i in shared
            else TtCompressCaches(mesh, tables, max_seq, cfg.ratio(i), cfg.head_dim, cfg.index_dim, slabs=slabs.get(i))
            for i in model.layers
            if cfg.is_kv_source(i)
        }
        for b in model.blocks.values():  # the ring's gathered-key buffer per cache capacity (load time)
            if b.indexer is not None:
                b.indexer.setup(self.compress[b.kv_src].index_k)
        self.topk = {i: {} for i in model.layers if cfg.is_index_source(i)}
        self.cands = {i: {} for i in model.layers if cfg.is_cand_source(i)}
        self.topk_prefix, self.cands_prefix = {}, {}
        self.memo = ChunkMemo() if model.memo_last is not None else None  # INDEXER_GLUE=on (P.16)
        self.tail = {i: model.engram[i].new_tail() for i in model.engram}
        self.tokens = None
        self.history = None  # this slot's real token ids by position (host; the engram tail of any start)
        # DEEP_LAYERS=tail (FIX.17): {chunk start: (layer 20's output DevH, the chunk's end)} of the last two chunks,
        # the deep layers' input for TtDS41Model.tail_pass
        self.deep_in = {}
        # DECODER_SKIP=replay128 (P.31): the decoder tail's own state (TtDS41Model.tail's window caches, top-k and
        # candidate stores, ChunkMemo; this state's layer-20 compressed caches shared), made at the first tail chunk
        self.tail_state = None
        self.cut = None  # (P.31: the start every layer's window is truncated at, TtWindowCache.truncate)
        model.state = self  # begin_chunk reads (and advances) the active slot's engram tail

    # ------------------------------------------------------------------ per chunk (device)
    def put(self, store: dict, layer: int, start: int, t: ttnn.Tensor) -> None:
        if store is self.topk and self.memo is not None:  # a new top-k: the index list made from the old one goes
            self.memo.drop(("index", layer))
        stale = [start] if self.keep_history else list(store[layer])
        if self.model.deep and not self.keep_history:  # DEEP_LAYERS=tail: the tail pass may also read the previous
            stale = [k for k in stale if k != start - self.model.chunk]  # chunk's top-k / pool
        for k in stale:
            old = store[layer].pop(k, None)
            if old is not None:
                _free(old)
        store[layer][start] = t

    def keep_deep_input(self, start: int, end: int, h: DevH) -> None:
        """DEEP_LAYERS=tail: takes ``h`` (the last full layer's output of the chunk [start, end)); keeps it and the
        previous chunk's, frees any other."""
        old = self.deep_in.pop(start, None)
        if old is not None:
            _free(old[0])
        for k in [k for k in self.deep_in if k != start - self.model.chunk]:
            _free(self.deep_in.pop(k)[0])
        self.deep_in[start] = (h, int(end))

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
            if per_chip:
                rows = _cands_to_host(t)
            else:  # INDEXER_TOPK_LOCAL=on (P.8): the chip's own rows (not gathered over TP)
                local = self.mesh.shape[TP_AXIS] > 1 and t.shape[-2] * self.mesh.shape[SP_AXIS] != self.model.chunk
                rows = _ids_to_host(t, self.mesh, per_chip=local)
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
        if self.memo is not None:
            self.memo.clear()
        if self.tail_state is not None:  # (DECODER_SKIP=replay128: the tail's state, its shared caches left alone)
            self.tail_state.free()
            self.tail_state = None
        mine = [c for i, c in self.compress.items() if i not in self.shared_compress]
        for c in [*self.window.values(), *mine]:
            c.free()
        for store in (self.topk, self.cands):
            for d in store.values():
                for t in d.values():
                    _free(t)
        for h, _ in self.deep_in.values():
            _free(h)
        self.deep_in = {}
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
        set_load_layer(i)  # (DENSE_WEIGHTS=bfp8: the per-matrix log names the layer)
        self.steps = block_steps(cfg, i)
        # DECODER_SKIP=replay128 (P.31): a block of the decoder tail (TtDS41Model.tail, the tail chunk's geometry): its
        # KV source's compressed caches are the main model's (written over the whole chunk by its kv_only), so it has
        # no compressor; its MoE shares the main block's routed experts (MoELayer.at); its indexer scores keys the main
        # chunk's geometry laid out (key_rows)
        tail_of = getattr(model, "tail_of", None)
        r = cfg.ratio(i)
        self.kv_src, self.idx_src = cfg.kv_source_of(i), cfg.index_source_of(i)
        rope = model.rope(r)
        p = f"layers.{i}."
        # OWN_ROWS=on (P.23): the residual / mHC streams on the chip's own S/(SP TP) rows, whole hidden
        self.own = model.own
        # mHC, norms, residuals
        self.attn_hc = build_hc(mesh, ck, cfg, i, "attn_hc", rows=self.own)
        self.ffn_hc = build_hc(mesh, ck, cfg, i, "ffn_hc", rows=self.own)
        self.collapse = TtHcCollapse(n=cfg.hc)
        self.fused = model.fused  # MHC_FUSED=on (P.14): hc + collapse in one program, residuals emit the next hc row
        self.fold = ccl.glue_fold()  # GLUE_FOLD=on (P.19)
        # ATTN_SP32=on (P.10): attn_norm, q_a, kv, compressor and the indexer gates on the chip's own S/(SP TP) rows
        # with whole hidden rows (one all_to_all of attn_in over TP, then local); the attention projections already
        # run there (ATTN_SP32_PROJ=on, P.9)
        self.sp32 = attn_sp32()
        # NORM_GEMM_DTYPE=on (P.21): q_norm writes wq_b's bfloat8_b input (one rounding from fp32 DEST, fork option
        # pack_precise, the softmax scale in its gamma) when the attention alone reads q_resid; with an indexer, q_norm
        # writes bf16 + the indexer's bfp8 q_resid (bfp8_twin). The chip-rows path only (ATTN_SP32). attn_norm stays
        # bf16 + q_a's typecast: a bfp8 attn_norm takes the fork's split program, whose LOCAL mode runs one core a row
        # tile (5 for 160 rows): +2.3 ms of device a chunk vs the default program + typecast (P.21 measure, p51200).
        readers = {}
        for st in self.steps:
            for n in st.inputs:
                readers.setdefault(n, set()).add(st.name)
        self.norm_gemm = (
            settings.get("NORM_GEMM_DTYPE") == "on"
            and self.sp32
            and settings.get("GEMM_INPUTS") == "bfp8"
            and settings.get("NORM_IMPL") == "on"
        )
        qr_bfp8 = self.norm_gemm and readers.get("q_resid", set()) <= {"attention"}
        qr_twin = self.norm_gemm and readers.get("q_resid", set()) == {"attention", "indexer"}
        self.attn_norm = build_norm(mesh, ck, cfg, i, "attn_norm", rows=self.sp32)
        # (MOE_IMPL=block, MOE_HEAD=fused: the fused head gathers x over TP itself - ffn_norm keeps the column split)
        fused_head = settings.get("MOE_IMPL") == "block" and settings.get("MOE_HEAD") == "fused"
        if self.own:  # OWN_ROWS=on (P.23): a local norm of the chip's own whole rows (the MoE's owner rows)
            self.ffn_norm = build_norm(mesh, ck, cfg, i, "ffn_norm", gather=False, rows=True)
        else:
            self.ffn_norm = build_norm(mesh, ck, cfg, i, "ffn_norm", gather=False if fused_head else None)
        self.attn_res = build_residual(cfg)
        self.ffn_res = build_residual(cfg, step="ffn_residual")
        # attention input: one fused [wq_a | wkv] projection for q_a and kv
        proj = TtAttnInProj(mesh, attn_in_weights(ck, cfg, i), full_rows=self.sp32)
        q_scale = float(cfg.head_dim) ** -0.5  # (the attention's softmax scale, TtWindowAttention.scale)
        self.q_a = TtQa(
            mesh,
            proj,
            ck.weight(p + "attn.q_norm.weight", torch.float32),
            cfg.eps,
            gamma_scale=q_scale if qr_bfp8 else 1.0,
            dtype=ttnn.bfloat8_b if qr_bfp8 else None,
        )
        self.q_a.twin = qr_twin and self.q_a.norm.out_dtype == ttnn.bfloat16
        kv_norm = ck.weight(p + "attn.kv_norm.weight", torch.float32)
        self.kv = TtWindowKv(mesh, proj, kv_norm, cfg.eps, rope, cfg.rope_dim)
        self.compressor = None
        if cfg.is_kv_source(i) and tail_of is None:
            self.compressor = TtCompressor(
                mesh, compressor_weights(ck, i), cfg, r, model.compress_rope(r), model.tables, full_rows=self.sp32
            )
        self.indexer = None
        if cfg.is_index_source(i):
            self.indexer = build_indexer(mesh, ck, cfg, i, model.max_seq, model.chunk, rope=rope, full_rows=self.sp32)
            self.indexer.local = topk_local()  # INDEXER_TOPK_LOCAL=on (P.8): no TP gather of the top-k
            if tail_of is not None:  # (P.31: the served index keys are in the main chunk's block-cyclic layout)
                self.indexer.key_rows = tail_of.chunk // tail_of.sp
        w = attention_weights(ck, i)
        if r:
            self.attention = TtSparseAttention(mesh, cfg, w, rope, model.tables, model.sparse_tables())
        else:
            self.attention = TtWindowAttention(mesh, cfg, w, rope, model.tables)
        if self.own:  # OWN_ROWS=on (P.23): attn_out stays on the chip's rows (no exit all_to_all)
            assert self.attention.sp32, "OWN_ROWS needs ATTN_SP32_PROJ=on"
            self.attention.keep_rows = True
        if qr_bfp8:
            assert self.attention.sp32 and self.attention.stock, "NORM_GEMM_DTYPE: q_resid prescaled needs _q_stock"
            assert self.attention.scale == q_scale, (self.attention.scale, q_scale)
            self.attention.q_prescaled = True
        # engram
        self.engram_kv = self.engram_add = None
        if cfg.is_engram(i):
            self.engram_kv = build_engram_kv(mesh, ck, cfg, i, own_rows=self.own)
            self.engram_add = build_engram_add(mesh, ck, cfg, i, rows=model.rows, own_rows=self.own)
        # MoE
        self.moe_block = None
        if tail_of is not None:  # (P.31: the main block's MoE layer at the tail chunk, its weights shared)
            assert settings.get("MOE_IMPL") == "block", "DECODER_SKIP=replay128 needs MOE_IMPL=block"
            self.moe_block = tail_of.blocks[i].moe_block.at_chunk(model.chunk)
            self.router = self.experts = self.shared = None
        elif settings.get("MOE_IMPL") == "block":  # (tt/MOE_BLOCK.md: one MoEBlock call in the experts step)
            from .moe_block_adapter import build_moe_block_layer

            self.moe_block = build_moe_block_layer(
                mesh, ck, cfg, i, model.chunk, cache_dir=model.weight_cache, own_rows=self.own
            )
            self.router = self.experts = self.shared = None
        else:
            self.router = build_router(mesh, ck, cfg, i, model.chunk)
            self.experts = build_experts(mesh, ck, cfg, i, model.chunk, cache_dir=model.weight_cache)
            self.shared = build_shared_expert(mesh, ck, cfg, i, chunk=model.chunk)
            # P.12: the shared expert reads its x rows from the routed experts' gathered row-major q(x) (experts runs
            # first in the block graph)
            self.experts.keep_rows = bool(self.shared.reads_rows)
        # NORM_INTO_MOE=on (P.21): ffn_norm (the column split, bf16, the fork) writes the MoE block's own x and its
        # bfloat8_b xq (bfp8_twin) in one program; the fused head reads both in place
        self.norm_into_moe = (
            settings.get("NORM_INTO_MOE") == "on"
            and not self.own  # (OWN_ROWS: the head's TP all_to_all writes its x; ffn_norm writes its load-time site)
            and self.moe_block is not None
            and fused_head
            and self.ffn_norm.impl == "on"
            and self.ffn_norm.out_dtype == ttnn.bfloat16
        )
        self._moe_xq_ready = False
        # INDEXER_GLUE=on (P.16): the experts take the router's own weights (no dense scatter, no gather)
        self.router_wts = indexer_glue()
        if self.router is not None:
            self.router.dense = not self.router_wts
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
            # MHC_FUSED=on: the add's mhc_post pass also emits this layer's attn_hc partial row (P.14)
            s, pre, row = self.engram_add.add(x.streams, kv, x.pre, hc=self.attn_hc if self.fused else None)
            return DevH(s, pre, stats=(i, row) if row is not None else None)

        def attn_hc(ctx, x):
            return self.attn_hc(x.streams)

        def attn_collapse(ctx, x):
            return self.collapse(x.streams, x.pre)

        def attn_norm(ctx, x):
            if self.sp32 and not self.own:  # ATTN_SP32=on (P.10): [S/SP, H/TP] -> the chip's rows, a local norm
                xr = cols_to_rows(self.mesh, x)
                y = self.attn_norm(xr)
                if xr is not x:
                    dealloc(xr)
                return y
            return self.attn_norm(x)

        def q_a(ctx, x):
            y, p = self.q_a.with_proj(x)
            ctx.extra["attn_proj"] = p
            return y

        def kv(ctx, x):
            p = ctx.extra.pop("attn_proj")
            out = self.kv.from_proj(p, ctx.start)
            dealloc(p)
            return out

        def compressor(ctx, x):
            if self.compressor is None:  # (P.31, the decoder tail: the main model's kv_only wrote the chunk's groups)
                return Keep()
            ckv, k = self.compressor(x, ctx.start)
            self.compressor.write(ckv, k, ctx.start, ctx.extra["end"], ctx.state.compress[i])
            dealloc(ckv)
            dealloc(k)
            return Keep()  # the chunk's rows live in the served caches the indexer and the attention read

        def indexer(ctx, x, qr, *compressed):
            st, start = ctx.state, ctx.start
            # NORM_GEMM_DTYPE=on (P.21): q_norm's bfloat8_b q_resid, wq_b's input as is
            self.indexer.qr_twin = self.q_a.twin_out if self.q_a.twin else None
            cache = st.compress[self.kv_src].index_k
            self.indexer.share(st.memo, self.kv_src)  # INDEXER_GLUE=on: the tiled keys once per (source, chunk)
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
            if st.memo is not None:  # INDEXER_GLUE=on (P.16): the source's history / index list once per chunk
                return self.attention(
                    qr, kvt, start, end, st.window[i], ckv, top, memo=st.memo, sources=(self.kv_src, self.idx_src)
                )
            return self.attention(qr, kvt, start, end, st.window[i], ckv, top)

        def attn_residual(ctx, x, mix, y):
            return self.attn_res(x.streams, mix, y)

        def ffn_hc(ctx, h):
            return self.ffn_hc(h)

        def ffn_collapse(ctx, h, mix):
            return self.collapse(h, mix)

        def ffn_norm(ctx, x):
            if self.norm_into_moe:  # (P.21: into the block's buffers; pinned, the block frees nothing of them)
                xb, xq = self.moe_block.input_buffers()
                if xb is not None and tuple(xb.shape) == tuple(x.shape) and xb.dtype == ttnn.bfloat16:
                    self._moe_xq_ready = xq is not None
                    return self.ffn_norm(x, out=xb, twin=xq)
            self._moe_xq_ready = False
            return self.ffn_norm(x)

        def router(ctx, x):
            return self.router(x)  # (dense [S/SP, E] fp32 RM, top-k ids, weights)

        def experts(ctx, x, routing):
            dense, idx, wts = routing
            if self.router_wts:  # INDEXER_GLUE=on (P.16)
                return self.experts.partial(x, dense, idx, wts=wts)
            return self.experts.partial(x, dense, idx)  # this dispatch group's partial [S/SP, H]

        def shared_expert(ctx, x):
            # this chip's intermediate share, partial [S/SP, H] (bf16 under CCL_IMPL=fabric_bf16)
            return self.shared.partial(x, rows=self.experts.take_rows())

        def moe_add(ctx, a, b):
            # the two partials summed before the one exchange over TP (components.yaml moe_add notes)
            # CCL_IMPL=fabric_bf16 (P.6): the sum written in bf16 for the bf16 reduce-scatter (tt/ccl.py:partial_dtype)
            sdt = ccl.partial_dtype(self.mesh) if self.experts.sum_mode == "gather_add" else ttnn.float32
            s = ttnn.add(a, b, dtype=sdt, memory_config=DRAM, sub_core_grids=sub_cores(a))
            # GLUE_FOLD=on (P.19): the bf16 sum as is (ffn_residual's mhc_post takes bf16)
            out = self.experts.reduce(s, fp32=not self.fold)
            dealloc(s)
            return out

        def ffn_residual(ctx, h, mix, y):
            o, pre = self.ffn_res(h, mix, y)
            return DevH(o, pre)

        if self.moe_block is not None:
            # MOE_IMPL=block (tt/MOE_BLOCK.md): the step names stay; the experts step runs the whole MoE layer (router,
            # routed + shared experts, the TP exchange) and its output is moe_add's: [S/SP, H/TP], ffn_residual's y.
            # router / shared_expert do no work, moe_add hands the block's output on.
            def router(ctx, x):  # noqa: F811
                return Keep()

            def experts(ctx, x, routing):  # noqa: F811
                ready, self._moe_xq_ready = self._moe_xq_ready, False
                return self.moe_block(x, xq_ready=ready)

            def shared_expert(ctx, x):  # noqa: F811
                return Keep()

            def moe_add(ctx, a, b):  # noqa: F811
                return a

        if self.fused:
            # MHC_FUSED=on (P.14): the hc steps compute their coefficients and collapse the streams with the given pre
            # in one program (tt/mhc.py:TtHcMixes.fused); the collapse steps hand that output on. attn_hc collapses x
            # with the pending pre (the previous ffn_hc's whole mixes), ffn_hc h_mid with attn_hc's mixes (DS41-3).
            def attn_hc(ctx, x):  # noqa: F811
                hc, y = self.attn_hc.fused(x.streams, x.pre, row=x.take_stats(i))
                ctx.extra["attn_in"] = y
                ctx.extra["attn_mix"] = hc  # ffn_hc's collapse pre (freed by the block after ffn_collapse)
                return hc

            def attn_collapse(ctx, x):  # noqa: F811
                return ctx.extra.pop("attn_in")

            def attn_residual(ctx, x, mix, y):  # noqa: F811
                out, row = self.attn_res.fused(x.streams, mix, y, self.ffn_hc)
                if row is not None:
                    ctx.extra["ffn_row"] = row
                return out

            def ffn_hc(ctx, h):  # noqa: F811
                hc, y = self.ffn_hc.fused(h, ctx.extra["attn_mix"], row=ctx.extra.pop("ffn_row", None))
                ctx.extra["ffn_in"] = y
                return hc

            def ffn_collapse(ctx, h, mix):  # noqa: F811
                ctx.extra.pop("attn_mix", None)
                return ctx.extra.pop("ffn_in")

            def ffn_residual(ctx, h, mix, y):  # noqa: F811
                nxt = self.model.next_hc(i)
                out, row = self.ffn_res.fused(h, mix, y, nxt.attn_hc if nxt is not None else None)
                return DevH(out, mix, stats=(nxt.layer, row) if row is not None else None)

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

    KV_STEPS = ("attn_hc", "attn_collapse", "attn_norm", "compressor")

    def kv_only(self, h: DevH, start: int, state: DS41DeviceState, end: int | None = None) -> Skipped:
        """DECODER_SKIP=exact (P.30), a chunk that is not the request's last: only the layer's KV projection, the
        steps its compressor needs (attn_hc, attn_collapse, attn_norm, compressor: the compressed KV and index keys
        the later layers read, written into the served caches as in the full block), the same modules and the same
        ctx; nothing else of the layer runs (no window KV, indexer, attention, MoE). Does not consume ``h``."""
        assert self.compressor is not None and not self.cfg.is_engram(self.layer), self.layer
        rows = h.streams.shape[-2] * self.mesh.shape[SP_AXIS] * (self.mesh.shape[TP_AXIS] if self.own else 1)
        assert rows == self.model.chunk, f"chunk of {rows} rows; the model was built for {self.model.chunk}"
        end = start + rows if end is None else int(end)
        ctx = Ctx(self.layer, start, rows, state, extra={"end": end})
        held = h.tensors()  # (as __call__: the wrappers alive until the call returns, FIX.5)
        protect = {id(t) for t in held}
        steps = [s for s in self.steps if s.name in self.KV_STEPS]
        assert [s.name for s in steps] == list(self.KV_STEPS), [s.name for s in steps]
        last = {}
        for s in steps:
            for n in s.inputs:
                last[n] = s.name
        from models.demos.common.bringup.testing import profiler

        env = {"in": h}
        for s in steps:
            profiler.signpost(s.name)  # (run_block's sections)
            env[s.output] = self.fns[s.name](ctx, *[env[n] for n in s.inputs])
            keep = protect | {id(t) for t in _tensors(env[s.output])}
            for n in dict.fromkeys(s.inputs):
                if last[n] == s.name and n != "in":
                    _free(env.pop(n), keep)
        for n, v in env.items():  # outputs no kept step reads (attn_hc's mixes): freed here
            if n != "in":
                _free(v, protect)
        ctx.extra.clear()  # (MHC_FUSED: attn_hc's mixes, freed above with its output)
        del held
        return SKIPPED

    def __call__(self, h: DevH, start: int, state: DS41DeviceState, end: int | None = None) -> DevH:
        rows = h.streams.shape[-2] * self.mesh.shape[SP_AXIS] * (self.mesh.shape[TP_AXIS] if self.own else 1)
        assert rows == self.model.chunk, f"chunk of {rows} rows; the model was built for {self.model.chunk}"
        end = start + rows if end is None else int(end)
        ctx = Ctx(self.layer, start, rows, state, extra={"end": end})
        # the caller's input (an engram layer's x shares its pre). ``held`` keeps those wrappers alive until the call
        # returns: attn_hc takes the stats row out of h (take_stats) and frees it, and once its wrapper died CPython
        # could hand its id to a new tensor, which _free would then skip (a deallocate fewer, at random: FIX.5)
        held = h.tensors()
        protect = {id(t) for t in held}

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

        out = run_block(self.steps, no_cpu, ctx, h, overrides=overrides)
        del held  # protect's ids were valid for the whole block
        if state.memo is not None:  # INDEXER_GLUE=on (P.16): slots whose last reader of the chunk was this layer
            for slot in self.model.memo_last.get(self.layer, ()):
                state.memo.drop(slot)
        return out


# ============================================================================================= model
class TtDS41Model:
    """The device modules of the requested layers plus the embedding and the final norm, and the load-time tables
    they share. ``layers``: sorted layer ids (a reader layer's sources must be among them)."""

    def __init__(
        self,
        mesh,
        model_path: str,
        layers,
        max_seq: int,
        chunk: int,
        lm_head: bool = False,
        hasher=None,
        weight_cache=None,
        tail_of: "TtDS41Model | None" = None,
    ):
        """``tail_of`` (DECODER_SKIP=replay128, P.31): this model is the decoder tail of that model (its decoder layers
        at the tail chunk; ``build_tail``): no embedding, no decoder skip of its own."""
        from models.demos.deepseek_v41_flash_d_p.reference.ds41_ref import Config
        from models.demos.deepseek_v41_flash_d_p.reference.weights import Checkpoint, EngramTable

        from .engram import engram_hasher

        t0 = time.time()
        # HOST_GLUE=on (P.21): every switch read once here, settings.get serves these values (no environment read
        # per call); off: thawed, every get reads the environment (a new model's load always re-reads them)
        settings.thaw()
        if settings.get("HOST_GLUE") == "on":
            settings.snapshot()
        self.mesh, self.path = mesh, str(model_path)
        init_caps(mesh)  # FIX.15 (MAX_WORKER_CORES): the mesh's core cap for every op of the model
        self.weight_cache = weight_cache  # the routed experts' laid-out weight cache (tt/experts.py:expert_cache_dir)
        self.cfg = cfg = Config.from_dir(self.path)
        self.ck = ck = Checkpoint(self.path)
        self.layers = sorted(layers)
        self.tail_of = tail_of
        self.max_seq, self.chunk = int(max_seq), int(chunk)
        self.sp, self.tp = mesh.shape[SP_AXIS], mesh.shape[TP_AXIS]
        assert self.chunk % (32 * self.sp * self.tp) == 0, (self.chunk, tuple(mesh.shape))
        self.fused = settings.get("MHC_FUSED") == "on"  # P.14
        self.own = own_rows()  # OWN_ROWS (P.23): the streams on each chip's own S/(SP TP) rows, whole hidden
        self.rows = self.chunk // (self.sp * self.tp if self.own else self.sp)  # the streams' rows a chip
        if bf16_activations():  # P.20: the bf16 paths exist on the fused / folded / stock-rope / fork-norm path only
            off = [k for k in ("MHC_FUSED", "GLUE_FOLD", "ROPE_IMPL", "NORM_IMPL") if settings.get(k) != "on"]
            assert not off, f"BF16_ACTIVATIONS=on needs {off} on"
        ccl.configure()  # GLUE_FOLD=on (P.19): tt/ccl.py:dealloc through the binding
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
        # ENGRAM_ASYNC=on (P.6): one worker thread fills the engram rows begin_chunk allocated, layer by layer, while
        # the main thread dispatches; take_engram_rows waits for its layer's job. engram_times: per layer the last
        # chunk's host ms (hash, lookup / pack, wrap, upload on the worker; wait = what the main thread still waited)
        self._engram_jobs = {}
        self._engram_pool = None
        if self.engram and settings.get("ENGRAM_ASYNC") == "on":
            from concurrent.futures import ThreadPoolExecutor

            self._engram_pool = ThreadPoolExecutor(1, thread_name_prefix="engram-async")
        self.engram_times = {}
        # embedding: [V, H/TP] bf16 ROW_MAJOR per chip (the chip's hidden columns), replicated over SP
        if tail_of is not None:  # (P.31: the decoder tail takes its input from the main model's layers)
            self.vocab, self.embed_w = tail_of.vocab, None
        else:
            emb = ck.get("embed.weight")
            assert emb.dtype == torch.bfloat16, emb.dtype  # the device copy is then exact
            self.vocab = emb.shape[0]
            self.embed_w = from_torch(
                emb.contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=DRAM,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(None, 1)),
            )
            del emb
        # layer 0's pending pre: one-hot on stream 0 (reference first_block_input), [S/SP, 4] fp32, cloned per chunk
        # (MHC_FUSED=on: [S/SP, 24], the width of the mixes the boundary carries; only columns 0..3 are read)
        rows = self.rows
        self.pre_w = NG if self.fused else HC
        pre0 = torch.zeros(1, 1, self.chunk, self.pre_w)
        pre0[..., 0] = 1.0
        self.pre0 = self._rows_to_device(pre0.reshape(self.chunk, self.pre_w))
        assert self.pre0.shape[-2] == rows, (self.pre0.shape, rows)
        self.final_collapse = TtHcCollapse(n=cfg.hc)  # MHC_FUSED=on: mhc_pre_xing(coefficients_given=True)
        # BF16_ACTIVATIONS=on (P.20): the final norm writes bf16 (the GPU code's RMSNorm .to(dtype))
        self.final_norm_mod = TtDistributedRmsNorm(
            mesh, ck.weight("norm.weight", torch.float32), cfg.eps, split=not self.own, dtype=act_dtype()
        )  # (OWN_ROWS: the chip's own whole rows, a local norm)
        self.head_w = ck.weight("head.weight", torch.float32) if lm_head else None
        self.blocks = {}
        for i in self.layers:
            self.blocks[i] = TtDS41Block(self, i)
        # DEEP_LAYERS=tail (FIX.17): the layers above the last KV / candidate source (21-39: window attention over
        # the sources' state, no state of their own another layer reads) run only in tail_pass, on the rows the
        # prompt's last token needs: (n_deep + 1) x (window - 1) + 1 positions of the last full layer's output
        self.deep, self.deep_src, self.tail_rows = [], None, 0
        if settings.get("DEEP_LAYERS") == "tail":
            first = max(list(cfg.kv_sources) + [cfg.cand_source]) + 1
            self.deep = [i for i in self.layers if i >= first]
            if self.deep:
                assert (
                    self.deep == list(range(first, self.layers[-1] + 1)) and first - 1 in self.layers
                ), f"DEEP_LAYERS=tail needs layers {first - 1}..{self.layers[-1]} loaded, consecutive: {self.layers}"
                assert not any(cfg.is_engram(i) or cfg.is_kv_source(i) for i in self.deep), self.deep
                self.deep_src = first - 1
                self.tail_rows = (len(self.deep) + 1) * (cfg.window - 1) + 1
        self._deep_set = frozenset(self.deep)
        # DECODER_SKIP=exact (P.30): the decoder = the loaded layers from the candidate source (20) on; on a chunk the
        # caller marks not the last (begin_chunk(.., last=False)) its KV sources (20) run only their KV projection
        # (TtDS41Block.kv_only) and the rest is skipped; every decoder layer's window cache notes the skipped positions
        self.decoder, self.kv_only = [], frozenset()
        self._skip = False  # this chunk skips the decoder (set per chunk by begin_chunk)
        mode = settings.get("DECODER_SKIP")
        if mode in ("exact", "replay128") and tail_of is None:
            assert not self.deep, "DECODER_SKIP and DEEP_LAYERS=tail are two different skips: one at a time"
            self.decoder = [i for i in self.layers if cfg.cand_source >= 0 and i >= cfg.cand_source]
            self.kv_only = frozenset(i for i in self.decoder if cfg.is_kv_source(i))
            bad = [i for i in self.decoder if cfg.is_engram(i) or (cfg.is_kv_source(i) and i != cfg.cand_source)]
            assert not bad, f"DECODER_SKIP=exact: decoder layers {bad} are engram / extra KV-source layers"
            enc = [i for i in self.layers if i not in self.decoder]
            assert all(j < cfg.cand_source or j in self.kv_only for i in enc for j in cfg.sources(i)), enc
        self._decoder_set = frozenset(self.decoder)
        # DECODER_SKIP=replay128 (P.31, DeepSeek-V4.1-Flash's SWA Bounded Replay, n_win 128): one code path with exact
        # but the tail length. The caller's two numbers: replay_rows = how far a request's first chunk on a cached
        # prefix starts before its new tokens (exact 0: the prefix's caches are complete, its last chunk ran every
        # layer; replay128 the window, 128, replayed with every window truncated there: ``begin_chunk(.., replay=True)``);
        # last_rows = the rows a request's last chunk must hold (exact 20 x 127 + 128 -> 2688: the decoder's receptive
        # field over skipped chunks, P.30; replay128 0: no pull-back). The last chunk (``last=True``) under replay128
        # runs the decoder only on its last <= 128 rows, as a chunk of its own: the tail model (``build_tail``: the
        # decoder layers at tail_chunk = 32 SP TP rows, 1024 on 8x4, so the rows sit on mesh row 0 and nothing pads
        # them to 32 rows on 32 chips; at least the indexer's top-k keys), its windows truncated to the tail
        self.replay = mode == "replay128" and bool(self.decoder)
        self.replay_rows = self.last_rows = 0
        if self.decoder:
            w = cfg.window
            self.replay_rows = w if self.replay else 0
            self.last_rows = 0 if self.replay else -(-(len(self.decoder) * (w - 1) + w) // 32) * 32
        self.tail, self.tail_chunk, self._tail_mode, self._tail_at = None, 0, False, None
        self.tail_load_seconds = 0.0
        if self.replay:
            assert self.own, "DECODER_SKIP=replay128: the tail's input moves on the chips' own rows (OWN_ROWS=on)"
            unit = 32 * self.sp * self.tp  # one tile row a chip
            keys = max(
                [
                    cfg.index_topk * max(1, cfg.ratio(cfg.kv_source_of(i)))
                    for i in self.decoder
                    if cfg.is_index_source(i)
                ]
                + [unit]
            )  # (TtIndexer: a chunk scores at least top-k keys)
            self.tail_chunk = -(-keys // unit) * unit
            assert cfg.window <= self.tail_chunk <= self.chunk, (cfg.window, self.tail_chunk, self.chunk)
        # INDEXER_GLUE=on (P.16): the shared tensors of a chunk (tt/sparse_attention.ChunkMemo) and the last loaded
        # layer that reads each: {layer: [slots dropped after it]}; None = off
        self.memo_last = None
        if indexer_glue():
            last = {}
            for i in self.layers:
                if cfg.ratio(i):
                    last[("comp", cfg.kv_source_of(i))] = i
                    last[("index", cfg.index_source_of(i))] = i
                if cfg.is_index_source(i):
                    last[("keys", cfg.kv_source_of(i))] = i
            self.memo_last = {}
            for slot, i in last.items():
                self.memo_last.setdefault(i, []).append(slot)
        host_alloc.reserve_sites(self)  # HOST_ALLOC=on (P.15): the in-step gathers' outputs, at load time (DS41-146)
        host_alloc.reserve_forks(self)  # FORK_OUT_BUFFERS=on (P.16): the forks' in-layer outputs, at load time
        self.load_seconds = time.time() - t0

    def _rows_to_device(self, x: torch.Tensor) -> ttnn.Tensor:
        """Host [S, W] -> the boundary's row layout: own rows (OWN_ROWS) or the row split, fp32 (harness / load)."""
        return own_rows_to_device(self.mesh, x) if self.own else row_split_to_device(self.mesh, x)

    def next_hc(self, i: int):
        """MHC_FUSED=on: the block whose attn_hc reads layer i's output streams as they are (the next layer, loaded
        and consecutive, without an engram add first), else None (then nothing is emitted for it)."""
        b = self.blocks.get(i + 1)
        if not self.fused or b is None or self.cfg.is_engram(i + 1):
            return None
        return b

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

    # ------------------------------------------------------------------ DECODER_SKIP=replay128 (P.31): the tail
    def build_tail(self) -> "TtDS41Model | None":
        """The decoder tail (load time; made at the first tail chunk when not before): the decoder layers as a model of
        their own at ``tail_chunk`` rows (its window tables, rope rows, attention / indexer / norm weights; the MoE's
        routed experts shared with this model's, MoELayer.at), the load-time input buffers its first layer reads (the
        chips' own rows, zero but where the tail's rows land), and the fabric sends that fill them (tt/tail_send.py)."""
        if self.tail is not None or not self.replay:
            return self.tail
        from .tail_send import TailSend

        t0 = time.time()
        assert settings.get("CKV_FORMAT") == "bfp4_rows", "DECODER_SKIP=replay128: the shared table is history only"
        from .moe_block_adapter import reserve_l1

        # (before this model's MoE block is built: the tail block's persistent words go above the main block's)
        reserve_l1(self.mesh, TAIL_L1_RESERVE)
        tail = TtDS41Model(
            self.mesh,
            self.path,
            self.decoder,
            self.max_seq + self.tail_chunk,  # (its rope rows: a tail chunk may run past the prompt's end)
            self.tail_chunk,
            lm_head=False,
            weight_cache=self.weight_cache,
            tail_of=self,
        )
        width = HC * self.cfg.dim
        sdt = stream_dtype()
        bufs, sends = [], []
        for w, dt in ((width, sdt), (self.pre_w, ttnn.float32)):
            t = from_torch(
                torch.zeros(1, 1, tail.rows, w),
                dtype=dt,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=DRAM,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
            ccl.pin(self.mesh, t)  # (the block never frees its input; dealloc skips it)
            bufs.append(t)
            sends.append(TailSend(self.mesh, self.rows, tail.rows, w, dt))
        self._tail_in, self._tail_send = tuple(bufs), tuple(sends)
        self.tail = tail
        self.tail_load_seconds = time.time() - t0
        return tail

    def tail_state(self, state: DS41DeviceState) -> DS41DeviceState:
        """The decoder tail's state of ``state``'s slot (made once): its own window caches, top-k / candidate stores and
        ChunkMemo; ``state``'s compressed caches of the tail's KV sources shared (written by kv_only over the chunk)."""
        if state.tail_state is None:
            tail = self.build_tail()
            shared = {i: state.compress[i] for i in tail.layers if i in state.compress}
            state.tail_state = DS41DeviceState(tail, state.max_seq, compress=shared, keep_history=state.keep_history)
            self.state = state  # (the tail state's constructor made itself the tail model's active state only)
        return state.tail_state

    def _to_tail(self, h: DevH, a: int, n: int) -> TailH:
        """Rows [a, a + n) of this chunk's block boundary (the chips' own rows) -> the tail's input (its load-time
        buffers: rows [0, n) of the tail chunk, chip q its rows [q Rt, (q + 1) Rt)); the stats row is not carried (the
        tail's attn_hc computes its own)."""
        for src, dst, send in zip((h.streams, h.pre), self._tail_in, self._tail_send):
            send(src, dst, a, n)
        return TailH(self._tail_in[0], self._tail_in[1])

    # ------------------------------------------------------------------ per chunk
    def new_state(self, max_seq: int, slabs: dict | None = None, keep_history: bool = True) -> DS41DeviceState:
        return DS41DeviceState(self, max_seq, slabs=slabs, keep_history=keep_history)

    def begin_chunk(self, tokens: torch.Tensor, start: int, last: bool | None = None, replay: bool = False) -> None:
        """The host input of every engram layer: hash the chunk's ids (pad past actual_end skipped, DS41-26) with the
        active slot's tail, look the rows up (memory-mapped) and send them to the device once. Also empties the
        active slot's ChunkMemo (INDEXER_GLUE=on, P.16; FIX.5): every run of a chunk builds its shared tensors
        itself, even when a slot's last reader layer is not loaded or not run (a re-run of the same start would hit
        it otherwise, and the unread slot would hold its device memory).
        ``last`` (P.30): whether this chunk is the request's last (the caller knows; the model never infers it).
        None (the default: not told, the framework harness / the engine) and, under exact, True run every layer;
        False under DECODER_SKIP=exact / replay128 skips the decoder (``layer``); True under replay128 (P.31) runs the
        decoder only on the chunk's last <= 128 rows (the tail model, its windows truncated to them). Ignored when
        DECODER_SKIP=off.
        ``replay`` (P.31): the chunk starts a request on a cached prefix, ``replay_rows`` before its new tokens (the
        caller pulls back). Under replay128 every layer's window is truncated at ``start`` (the carry masked: start
        class 0); under exact nothing changes (the pull-back recomputes exactly from the complete caches)."""
        self._skip = bool(self.decoder) and last is False
        self._tail_mode = self.replay and last is True
        self._tail_at = None
        st = self.state
        if self._tail_mode:
            self.build_tail()
        if self.replay and st is not None and (replay or st.cut is not None):  # (host only)
            st.cut = start if replay else None  # every layer's carry rows before start are not read
            for c in st.window.values():
                c.truncate(st.cut)
        if st is not None and st.memo is not None:
            st.memo.clear()
        if not self.engram:
            return
        self.wait_engram()  # a job of the last chunk whose layer did not run: done before its tensor is freed
        if self._engram_pool is not None:
            return self._begin_engram_async(tokens, start)
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

    def _begin_engram_async(self, tokens: torch.Tensor, start: int) -> None:
        """ENGRAM_ASYNC=on: on this thread only the new tails (3 ids a layer) and the device rows' allocation (the
        order the off path allocates in); the hash, lookup, pack and upload go to the worker, layer order."""
        st = self.state
        ids = tokens.reshape(-1).clone()  # (the caller may reuse its tensor while the worker reads it)
        for i, rows in self.engram.items():
            tail = rows.new_tail() if start == 0 or st is None else st.tail[i]
            tail = tail.long().clone()
            new_tail = rows.next_tail(ids, tail)
            if st is not None:
                st.tail[i] = new_tail
            _free(self._engram_dev.pop(i, None))
            dev = rows.allocate(ids.numel())
            self._engram_dev[i] = dev
            self._engram_jobs[i] = self._engram_pool.submit(rows.fill, dev, ids, tail, None, new_tail)

    def wait_engram(self) -> None:
        """ENGRAM_ASYNC=on: every engram job of this chunk done (its rows enqueued on CQ 0)."""
        for i in list(self._engram_jobs):
            self._wait_job(i)

    def _wait_job(self, layer: int) -> None:
        job = self._engram_jobs.pop(layer, None)
        if job is None:
            return
        t0 = time.perf_counter()
        times = job.result()
        times["wait"] = (time.perf_counter() - t0) * 1e3
        self.engram_times[layer] = times

    def take_engram_rows(self, layer: int) -> ttnn.Tensor:
        self._wait_job(layer)  # ENGRAM_ASYNC=on: layer 1 waits for its own job only, layer 14 for its own
        t = self._engram_dev.pop(layer, None)
        assert t is not None, f"layer {layer}: no engram rows for this chunk (begin_chunk(tokens, start) first)"
        return t

    def embed(self, tokens: torch.Tensor, start: int = 0) -> DevH:
        """Natural-order ids of one chunk (PAD_ID past actual_end clamped into the vocabulary, K-72: those rows are pad
        and never reach a cache) -> layer 0's input: the embedding on all 4 streams, the pending pre one-hot."""
        ids = tokens.reshape(-1).long().clamp(0, self.vocab - 1).to(torch.int32)
        assert ids.numel() == self.chunk, (ids.numel(), self.chunk)
        dev_ids = from_torch(
            ids.reshape(self.sp, self.chunk // self.sp).contiguous(),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=DRAM,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(0, None)),
        )
        e = ttnn.bringup.embedding(
            dev_ids, self.embed_w, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=DRAM, max_grid=max_grid()
        )
        dealloc(dev_ids)
        rows, w = self.chunk // self.sp, self.cfg.dim // self.tp
        e4 = ttnn.reshape(e, [1, 1, rows, w], sub_core_grids=sub_cores(e))
        if self.own:  # OWN_ROWS=on (P.23): the chip's own rows, whole hidden (one all_to_all over TP a chunk)
            er = cols_to_rows(self.mesh, e4)
            if er is not e4:
                dealloc(e4)  # (a view of e's buffer, K-1: e goes with it)
                if e.is_allocated():
                    dealloc(e)
                e = e4 = er
        if stream_dtype() == ttnn.bfloat16:  # BF16_ACTIVATIONS=on (P.20): the bf16 embedding rows as they are
            streams = ttnn.concat([e4] * HC, dim=-1, memory_config=DRAM, sub_core_grids=sub_cores())
            dealloc(e4)
            if e.is_allocated():
                dealloc(e)
            return DevH(streams, ttnn.clone(self.pre0, memory_config=DRAM))
        ef = ttnn.typecast(e4, ttnn.float32, memory_config=DRAM, sub_core_grids=sub_cores_tiled(e4))
        dealloc(e4)  # a view of e's buffer (K-1): then e is freed with it
        if e.is_allocated():
            dealloc(e)
        streams = ttnn.concat([ef] * HC, dim=-1, memory_config=DRAM, sub_core_grids=sub_cores())
        dealloc(ef)
        return DevH(streams, ttnn.clone(self.pre0, memory_config=DRAM))

    def layer(self, i: int, h: DevH, start: int, state: DS41DeviceState, end: int | None = None) -> DevH:
        if i in self._decoder_set:  # DECODER_SKIP=exact (P.30) / replay128 (P.31)
            if self._skip:  # not the request's last chunk: layer 20's KV projection only, the rest skipped
                state.window[i].skip(start, start + self.chunk)
                if i in self.kv_only:
                    return self.blocks[i].kv_only(h, start, state, end)
                assert isinstance(h, Skipped), f"layer {i}: DECODER_SKIP skips it on this chunk"
                return h
            if self._tail_mode:  # replay128, the request's last chunk: the decoder on its last <= 128 rows only
                return self._tail_layer(i, h, start, state, end)
            out = self.blocks[i](h, start, state, end)
            state.window[i].computed(start, start + self.chunk)
            return out
        if i in self._deep_set:  # DEEP_LAYERS=tail: not on a normal chunk (tail_pass runs it)
            assert isinstance(h, Skipped), f"layer {i}: DEEP_LAYERS=tail runs it only in tail_pass"
            return h
        out = self.blocks[i](h, start, state, end)
        if i == self.deep_src:  # DEEP_LAYERS=tail: the deep layers' input, kept for the tail pass
            state.keep_deep_input(start, start + self.chunk if end is None else end, out)
            return SKIPPED
        return out

    def _tail_layer(self, i: int, h, start: int, state: DS41DeviceState, end: int | None) -> TailH:
        """DECODER_SKIP=replay128 (P.31), the request's last chunk [start, end). The first decoder layer (20): its KV
        projection over the whole chunk (kv_only: the compressed KV and index keys every decoder layer reads), then
        the tail [dec_start, end), dec_start = the 32-aligned start of the last <= 128 rows, moved to the tail model's
        layout (``_to_tail``) and run through its block; every later decoder layer runs the tail model's block. Every
        tail layer's window is truncated at dec_start (no carry: start class 0, the rows before it are not read).
        Returns the tail's boundary (TailH); ``final_norm`` takes it."""
        tail, ts = self.tail, self.tail_state(state)
        if i == self.decoder[0]:
            end = start + self.chunk if end is None else int(end)
            assert isinstance(h, DevH) and not isinstance(h, TailH), f"layer {i}: the tail starts here"
            for j in self.kv_only:  # (the decoder's KV sources: layer 20)
                assert j == i, (j, i)
                self.blocks[i].kv_only(h, start, state, end)
            dec = max(start, -(-(end - self.cfg.window) // 32) * 32)
            self._tail_at = (dec, end)
            if ts.memo is not None:
                ts.memo.clear()
            for c in ts.window.values():
                c.truncate(dec)
            x = self._to_tail(h, dec - start, end - dec)
            out = tail.blocks[i](x, dec, ts, end)
        else:
            assert (
                isinstance(h, TailH) and self._tail_at is not None
            ), f"layer {i}: the tail runs from {self.decoder[0]}"
            dec, end = self._tail_at
            out = tail.blocks[i](h, dec, ts, end)
        return TailH(out.streams, out.pre, out.stats)

    def tail_pass(self, state: DS41DeviceState, end: int | None = None, on_layer=None) -> DevH:
        """DEEP_LAYERS=tail, once the prompt's last chunk ran (the caller signals it; the prompt's length is not
        needed before): the deep layers over the kept chunks that hold positions [end - tail_rows, end) (the last
        chunk, and the one before when the last holds fewer), each chunk at its own start through the normal block
        (window carry from the layer's cache: zeros, or the previous tail chunk's). Returns the last layer's output
        of the last chunk (rows past ``end - tail_rows + (layer - 20) x 127`` exact; the last token's always).
        ``end``: the prompt's end (default, and it must be, the last chunk's). ``on_layer(i, start, h)``: a test's read-back.
        """
        assert self.deep, "tail_pass needs DEEP_LAYERS=tail and the deep layers loaded"
        assert state.deep_in, "tail_pass: no chunk ran since the last tail pass"
        last = max(state.deep_in)
        end = state.deep_in[last][1] if end is None else int(end)
        assert end == state.deep_in[last][1], f"tail_pass: the last chunk ran to {state.deep_in[last][1]}, not {end}"
        first = max(0, end - self.tail_rows)
        starts = sorted(s for s in state.deep_in if s + self.chunk > first)
        assert starts[0] <= first and starts == list(range(starts[0], last + 1, self.chunk)), (
            f"tail_pass: positions [{first}, {end}) need the chunks from {first // self.chunk * self.chunk}; kept "
            f"{sorted(state.deep_in)}"
        )
        out = None
        for s in starts:
            h, e = state.deep_in.pop(s)
            if state.memo is not None:  # (INDEXER_GLUE: the memo's tensors were made for the last normal chunk)
                state.memo.clear()
            for i in self.deep:
                h2 = self.blocks[i](h, s, state, e)
                if on_layer is not None:
                    on_layer(i, s, h2)
                _free(h)
                h = h2
            if s == last:
                out = h
            else:
                _free(h)
        for k in list(state.deep_in):
            _free(state.deep_in.pop(k)[0])
        return out

    def final_norm(self, h: DevH) -> ttnn.Tensor:
        """rms(sum_n pre_n stream_n) w -> [1, 1, S/SP, H/TP] (hidden split; bf16 under BF16_ACTIVATIONS=on; OWN_ROWS:
        [1, 1, S/(SP TP), H], the chip's own rows). DECODER_SKIP=exact (P.30): a skipped chunk's Skipped marker is
        handed on (no final norm: no layer computed the chunk's last rows)."""
        if isinstance(h, Skipped):
            return h
        if isinstance(h, TailH):  # DECODER_SKIP=replay128 (P.31): the tail's rows, in the tail's layout
            return TailHidden(self.tail.final_norm(DevH(h.streams, h.pre)), *self._tail_at)
        x = self.final_collapse(h.streams, h.pre)
        out = self.final_norm_mod(x)
        dealloc(x)
        return out

    # ------------------------------------------------------------------ harness boundary (host)
    def from_host(self, h: torch.Tensor, start: int = 0) -> DevH:
        """Packed block input rows [S, 4H + 4] (natural order) -> DevH."""
        hidden = self.cfg.dim
        assert h.shape[-1] == HC * hidden + HC, h.shape
        pre = h[:, HC * hidden :].float()
        if self.pre_w > HC:  # MHC_FUSED=on: the boundary's pre is [S, 24] wide (columns 4.. unread)
            pre = torch.nn.functional.pad(pre, (0, self.pre_w - HC))
        if self.own:  # OWN_ROWS=on (P.23): the chip's own rows, every column
            return DevH(
                own_rows_to_device(self.mesh, h[:, : HC * hidden], stream_dtype()), own_rows_to_device(self.mesh, pre)
            )
        return DevH(streams_to_device(self.mesh, h, hidden), row_split_to_device(self.mesh, pre))

    def to_host(self, h) -> torch.Tensor:
        """DevH -> the reference's packed [S, 4H + 4]; a hidden-split tensor (the final norm) -> [S, H]. DECODER_SKIP=
        replay128 (P.31): a TailH / TailHidden -> the tail chunk's rows [tail_chunk, ..] (positions from its ``start``;
        real up to ``end``)."""
        if isinstance(h, TailHidden):
            return self.tail.to_host(h.t)
        if isinstance(h, TailH):
            return self.tail.to_host(DevH(h.streams, h.pre))
        if self.own:  # OWN_ROWS=on (P.23): every tensor is the chips' own rows, concatenated in chip order
            if isinstance(h, DevH):
                s = own_rows_to_host(self.mesh, h.streams).float()
                return torch.cat([s, own_rows_to_host(self.mesh, h.pre, HC).float()], dim=-1)
            return own_rows_to_host(self.mesh, h).float()
        if isinstance(h, DevH):
            s = streams_to_host(self.mesh, h.streams, self.cfg.dim).float()
            return torch.cat([s, row_split_to_host(self.mesh, h.pre, HC).float()], dim=-1)
        return hidden_split_to_host(self.mesh, h).float()

    def logits(self, hidden, rows) -> torch.Tensor:
        assert self.head_w is not None, "built without the LM head"
        return torch.nn.functional.linear(self.to_host(hidden)[rows], self.head_w)

    def steady_state(self, start: int | None) -> None:
        """Framework F236 (FIX.16, owner 2026-10-31), called before each TIMED run of the profile: the next run at
        ``start`` takes the path of a chunk that follows a whole previous chunk (every layer's window carry from the
        kept tail, no carry rebuild from the served cache: TtWindowCache.steady). Values may differ from a real
        previous chunk (timing only). None, before an accuracy run: the normal path. Host only, no device work."""
        st = self.state
        if st is None:
            return
        if start is not None and settings.get("STEADY_STATE") == "off":  # (comparison: the rebuild in every run)
            start = None
        for c in st.window.values():
            c.steady(start)

    def free(self, h) -> None:
        _free(h)

    def sync(self) -> None:
        ttnn.synchronize_device(self.mesh)
