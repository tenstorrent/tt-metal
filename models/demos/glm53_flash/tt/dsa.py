# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DSA (DeepSeek sparse attention, MLA with NoPE) layer for GLM-5.3-Flash on one device.

HF ``Glm5NextTextAttention`` (qk_rope_head_dim = 0, so no rotary anywhere):
    q = q_b(rmsnorm(q_a(x)))                      [T, 64, 256]
    c = rmsnorm(kv_a(x))                           [T, 512]   (the latent, cached per token)
    k_h = c @ Wkb1[h]^T, v_h = c @ Wkb2[h]^T       (kv_b = [64 x (256 k | 256 v), 512])
    o = softmax(q k^T / 16) v -> concat heads -> o_proj
The indexer selects the top 512 complete key pools (2048 tokens) plus the incomplete tail, so a
query at position a sees every visible token while floor((a+1)/4) <= 512, i.e. a <= 2050: there
the layer is exact causal attention (dense path). Beyond, the pooled-key indexer picks the keys
(sparse path, ``TtDSAIndexer`` + ``sparse_sdpa``).

Device design (section 5.3 of the design doc, single chip = Phase 1):
  * prefill: the latent is written to the paged cache with ``paged_fill_cache`` and the pooled-key
    index is maintained. Query rows [0, 2048) run the dense causal SDPA on materialized K/V; rows
    from 2048 on (they always see >= 2048 pooled tokens, so the top-k row is a full valid prefix)
    run the indexer selection + ``sparse_sdpa`` on the absorbed q. A prompt may arrive in
    consecutive chunks (``start`` = the chunk's first absolute position, a block multiple): the
    chunk's latent fills the pages from block ``start // 32``, the index continues from ``start``
    (the ring holds the previous chunk's tail) and the chunk rows attend over the full prefix, the
    dense rows through the windowed causal SDPA (K/V from the cached latent of [0, start) + the
    chunk's, Q offset ``start``) and the sparse rows exactly as in the single-shot path.
  * decode: absorbed MLA. q_lat[h] = q[h] @ Wkb1[h] (256 -> 512), then
    ``paged_flash_multi_latent_attention_decode`` over the latent cache for every user slot
    (exact through position 2050), then o[h] = o_lat[h] @ Wkb2[h]^T (512 -> 256), concat heads,
    o_proj. Users at position >= 2051 get their row replaced by the per-user indexer +
    ``sparse_sdpa`` result (host chunk geometry, not traced). The latent of the new token is
    appended with ``paged_update_cache`` first and the index is maintained for every user.
  * ``sparse_sdpa`` reads an index row up to its FIRST sentinel, so every row handed to it must be
    a contiguous valid prefix: that holds exactly because only queries with >= 2048 visible pooled
    tokens reach it.

Cache: ``[max_blocks, 1, block_size, 512]`` bf16, paged; page_table ``[B, max_blocks_per_seq]`` int32.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.glm53_flash.custom_ops.dsa_decode_state.op import DsaDecodeState
from models.demos.glm53_flash.custom_ops.index_score_decode.op import IndexScoreDecode
from models.demos.glm53_flash.custom_ops.kda_decode.op import selector_tiles
from models.demos.glm53_flash.custom_ops.paged_pool_gather.op import PagedPoolGather
from models.demos.glm53_flash.custom_ops.sparse_bound import live_bound_pools
from models.demos.glm53_flash.custom_ops.topk_rows.op import TopkRows
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.indexer import TtDSAIndexer, TtIndexCache, height_shard_rows
from models.demos.glm53_flash.tt.matmul_configs import PREFILL_ROW_BLOCK, DecodeLinear, prefill_linear
from models.demos.glm53_flash.tt.memory_plan import default_prefill_qblock
from models.demos.glm53_flash.tt.mesh import TP_AXIS, GLMMesh, single_chip
from models.demos.glm53_flash.tt.vllm_config import PREFIX_BUCKET_ENV

try:  # the allocation discipline under live traces (lazily built constants as chunk transients, GLMMesh.lazy)
    from models.demos.glm53_flash.tt.mesh import lazy_const
except ImportError:  # a tree without it: plain caching

    def lazy_const(mesh, cache: dict, key, make):
        t = cache.get(key)
        if t is None:
            t = cache[key] = make()
        return t


from models.demos.glm53_flash.tt.packed_inputs import sparse_host_values

TILE = 32
CORES_PER_USER_DECODE = 4
# heads2users prefill (user-partitioned caches without a scratch slot): spare latent blocks at the end of the paged
# cache that the non-owner columns of a prefill write their (discarded) chunk latent into
DUMMY_BLOCKS = 64
# how a user-partitioned cache serves a prefill: "scratch" (every column fills a full-length scratch slot and
# attends over it with its 16 heads) | "heads2users" (no scratch slot: the owner column attends with all 64 heads
# gathered over the columns, the other columns write dummies; doc/mesh/sparse_1m.md section 7)
PREFILL_PARTITION_MODES = ("scratch", "heads2users")
# decode block: "fused" (GLM53_DSA_FUSED=1: stem matmul -> ONE dsa_decode_state program -> folded q_b.W1 matmul ->
# flash-MLA decode -> folded W2.o_proj matmul, 7 ops) | "composite" (default)
DSA_FUSED_DEFAULT = "fused" if os.environ.get("GLM53_DSA_FUSED", "0") == "1" else "composite"
# which absorb matrices the fused block folds into its projections: "both" (default) | "q" | "o" | "none"
# (an unfolded side runs the composite's per-head bmm + transposes; a precision / perf attribution knob)
DSA_FOLD = os.environ.get("GLM53_DSA_FOLD", "both")
DSA_FOLD_MODES = ("both", "q", "o", "none")
# GLM53_DSA_DECODE_STATE: the traced multi-user sparse decode (``_decode_unified``, the mesh default) with the fused
# ``dsa_decode_state`` program for the state side (q_a norm, latent append, index ring + pool maintenance: 29 ops ->
# 1), the GLM53_DSA_FOLD absorb folds (w_qb1 / w_o2: 4 + 4 ops -> 1 + 1) and the free head-major query views:
#   "0" (default) = today's composite ops; "1" = the fused state block (``_decode_state_unified``);
#   "latent" = the fallback that keeps the index maintenance on today's ops: the program still norms q_a and appends
#   the latent, its ring / pool writes are then overwritten by the composite ``maintain_decode`` (bit-identical
#   index state by construction, 19 ops back). Read at construction (``decode_state_default``), so two layers of one
#   process can A/B it (``TtDSA(decode_state=...)``).
DSA_DECODE_STATE_MODES = ("0", "1", "latent")
# GLM53_POOL_GATHER (roofline B8a, 2026-10-26): how the traced multi-user decode (``_decode_unified``) attends when
# EVERY user slot of the step is in the dense range (position <= ``dense_exact_max_pos`` = 2050, where the dense
# attention over the paged cache is HF-exact and the compact gathered cache holds the same rows in the same order):
#   "gather" (default) = today's graph: index query + batched scorer + top-k + ``paged_pool_gather`` (2052 rows per user
#     copied out of the paged cache, 136 us at 4 users / ~1.3 ms at 32) + flash-MLA over the compact copy;
#   "dense_skip" = the flash-MLA decode over the REAL paged cache (``cache.cache`` / ``cache.page_table``, ``cur_pos`` =
#     the position: the same values ``SparseDecodeMeta.cur_pos`` carries for dense users) with no index query, scoring,
#     top-k or gather (-4 ops; expected bit-identical: same kernel, same logical rows). The index maintenance (ring /
#     pools) is untouched, so a later sparse step finds its index complete.
# The regime is a host property of the step's positions (``SparseDecodeMeta.all_dense`` / ``regime``), fixed at graph
# build time: a captured trace replays the regime it was captured in. ``SparseDecodeMeta.update`` refuses to move a
# meta whose last graph was built in the dense-skip regime out of it (the replay would attend the whole prefix densely
# beyond 2050 = not HF semantics): the model keys its decode trace by ``regime`` and re-captures (or runs eagerly) when
# it changes; ``clear_graph_regime`` is the eager caller's reset. Read at construction (``pool_gather_mode``).
POOL_GATHER_ENV = "GLM53_POOL_GATHER"
POOL_GATHER_MODES = ("gather", "dense_skip")
# GLM53_SPARSE_DECODE_BOUND (2026-10-27): "1" = the traced multi-user decode's scorer and segmented top-k loop to the
# step's LIVE pool bound (``SparseDecodeMeta.live_bound``, a per-chip [1, 1] uint32 refreshed with the other per-step
# tensors: the largest per-user top-k width, rounded to ``custom_ops/sparse_bound.GRANULE``) instead of the pool
# capacity the model was built with: a served model is built at 1 M positions (262 400 pools per user) and every decode
# step scored, masked and ranked all of them for every user (one DSA layer at position 1024 on the Galaxy: scorer 731 us
# + segmented top-k 249 us against 11 + 5 us when built at 1088 positions). The ranked ids and the attention are the
# unbounded path's (``tests/test_sparse_decode_bound_mesh.py``); "0" (default until that block test passes) = today's
# capacity-wide graph. Read at construction (``sparse_decode_bound_enabled``); the compact flash-MLA decode is untouched:
# its page table is the [B, 65]-block identity of the compact cache already (73 us at either capacity). The segmented
# top-k's partial row replays the unbounded row's trailing chunks as phantoms (``GLM53_TOPK_PHANTOM``, see
# ``custom_ops/sparse_bound.py``): the classic body's ranked ids depend on the chunk count.
SPARSE_BOUND_ENV = "GLM53_SPARSE_DECODE_BOUND"
# prefill stems (q_a, kv_a, q_b, the indexer's q): "fixed" (default: M-independent prefill_linear) | "default"
PREFILL_STEMS = os.environ.get("GLM53_PREFILL_STEMS", "fixed")
# The whole-prefix ROW_MAJOR latent copy of the sparse prefill (the sparse_sdpa kv contract), per DSA layer and
# chunk: "0" (default) = rebuilt from the paged cache every chunk (``TtLatentCache.user_rows_rm``, 8 ms at 1 M per
# chip, transient); "1" = kept per layer in a persistent arena of the slot's full length and APPENDED with each
# chunk's rows (``TtLatentCache.prefix_rows_rm``: ~0.5 ms per chunk whatever the prefix; the arena is one
# ``rows_per_user x 512`` bf16 tensor per DSA layer resident while the prompt is in flight: 11 x 1 GiB at 1 M, which
# is why the default stays 0, doc: _agents/prefill_ttft/STATUS.md); "auto" = "1" when the arenas of every DSA layer
# of the config fit ``GLM53_PREFIX_INCREMENTAL_BUDGET_MB`` (default 1024).
PREFIX_INCREMENTAL = os.environ.get("GLM53_PREFIX_INCREMENTAL", "0")
PREFIX_INCREMENTAL_BUDGET_MB = int(os.environ.get("GLM53_PREFIX_INCREMENTAL_BUDGET_MB", "1024"))


def prefill_blocks(max_seq_len: int | None = None) -> tuple[int, int]:
    """(score sub-block rows, attention block rows) of the sparse prefill loop (``TtDSA._sparse_rows``).
    ``GLM53_PREFILL_QBLOCK`` (default 2048; 512 single-user / 256 at 32 users at 1 M for the score block's memory)
    bounds the scorer's [rows, pools] bf16 block; ``GLM53_PREFILL_ABLOCK`` (default = QBLOCK, the validated per-sub-block form; 2048 batches a slab) the rows of one id
    expansion + absorbed-query + sparse_sdpa pass (a whole 2048-row slab of a 16 K chunk on 8 rows), i.e. how many
    score sub-blocks are selected before their attention runs batched. ``ABLOCK <= QBLOCK`` is the per-sub-block
    form (one attention call per score sub-block, the previous loop)."""
    # unset: the capacity-aware default (tt/memory_plan.py default_prefill_qblock: 2048 up to 256 K, 512 at 1 M)
    qb_env = os.environ.get("GLM53_PREFILL_QBLOCK")
    qb_default = default_prefill_qblock(max_seq_len)
    qb = max(TILE, (int(qb_env) if qb_env else qb_default) // TILE * TILE)
    ab = (
        int(os.environ.get("GLM53_PREFILL_ABLOCK", "0")) or qb
    )  # default = the validated per-sub-block form; 2048 batches a slab
    return qb, ab


def sparse_block_plan(r0: int, r1: int, qblock: int, ablock: int) -> list[tuple[int, int, list[tuple[int, int]]]]:
    """The sparse rows [r0, r1) of a chunk (or slab) as attention blocks of up to ``ablock`` rows, each the union
    of consecutive score sub-blocks of up to ``qblock`` rows: ``[(a0, a1, [(q0, q1), ...]), ...]``. ``ablock`` is
    rounded down to a multiple of ``qblock`` (at least one sub-block), so a sub-block never straddles two attention
    blocks; the ragged last block / sub-block takes what is left. Host ints only (``tests/test_prefill_select_batched.py``).
    """
    if r1 <= r0:
        return []
    qb = max(TILE, (qblock // TILE) * TILE)
    ab = max(qb, (ablock // qb) * qb)
    plan = []
    for a0 in range(r0, r1, ab):
        a1 = min(r1, a0 + ab)
        plan.append((a0, a1, [(q0, min(a1, q0 + qb)) for q0 in range(a0, a1, qb)]))
    return plan


def decode_state_default() -> str:
    """The ``GLM53_DSA_DECODE_STATE`` knob (``DSA_DECODE_STATE_MODES``; default "0" = the composite path)."""
    mode = os.environ.get("GLM53_DSA_DECODE_STATE", "0")
    if mode not in DSA_DECODE_STATE_MODES:
        raise ValueError(f"GLM53_DSA_DECODE_STATE={mode!r}: expected one of {DSA_DECODE_STATE_MODES}")
    return mode


def pool_gather_mode(env=None) -> str:
    """The ``GLM53_POOL_GATHER`` knob (``POOL_GATHER_MODES``; default "gather" = today's graph)."""
    env = os.environ if env is None else env
    mode = env.get(POOL_GATHER_ENV, "gather")
    if mode not in POOL_GATHER_MODES:
        raise ValueError(f"{POOL_GATHER_ENV}={mode!r}: expected one of {POOL_GATHER_MODES}")
    return mode


def sparse_decode_bound_enabled(env=None) -> bool:
    """The ``GLM53_SPARSE_DECODE_BOUND`` knob (default off)."""
    env = os.environ if env is None else env
    mode = env.get(SPARSE_BOUND_ENV, "0")
    if mode not in ("0", "1"):
        raise ValueError(f"{SPARSE_BOUND_ENV}={mode!r}: expected 0 or 1")
    return mode == "1"


def dense_exact_max_pos(cfg: GLM53FlashConfig) -> int:
    """The last position the dense attention is HF-exact at: floor((a+1)/4) <= 512 <=> a <= index_topk + kpool - 2."""
    return cfg.index_topk + cfg.index_kpool - 2


def dense_regime(positions, cfg: GLM53FlashConfig) -> bool:
    """True when every user slot of the step (``positions``, the mesh-wide list) is in the dense range: the dense-skip
    graph (``GLM53_POOL_GATHER=dense_skip``) is exact for the whole step. A pure function of the host positions."""
    bound = dense_exact_max_pos(cfg)
    return all(int(a) <= bound for a in positions)


def dsa_folds(mode: str | None = None, *, strict: bool = True) -> tuple[bool, bool]:
    """(fold the q absorb into q_b, fold the o absorb into o_proj) of a ``GLM53_DSA_FOLD`` value (``None`` = the
    environment, default ``both``). ``strict``: an unknown value raises (else it folds nothing, as the fused dense
    block treats it)."""
    mode = os.environ.get("GLM53_DSA_FOLD", "both") if mode is None else mode
    if strict and mode not in DSA_FOLD_MODES:
        raise ValueError(f"GLM53_DSA_FOLD={mode!r}: expected one of {DSA_FOLD_MODES}")
    return mode in ("both", "q"), mode in ("both", "o")


def fold_absorb_weights(
    w_qb_t: torch.Tensor, wkb1: torch.Tensor, wkb2: torch.Tensor, w_o_t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The absorbed-MLA decode projections with the per-head absorb matrices folded in (fp32 host math):
    ``w_qb_t`` [Lq, H*Dq] (q_b^T, head-major), ``wkb1`` [H, Dq, L], ``wkb2`` [H, Dv, L] (the kv_b halves),
    ``w_o_t`` [H*Dv, hidden] (o_proj^T, head-major) -> (``w_qb1`` [Lq, H*L], ``w_o2`` [H*L, hidden]) with
    ``W_qb1[h] = W_qb[h] W_kb1[h]`` (q_lat[h] = q_resid @ W_qb1[h] in one matmul, packed head-major) and
    ``W_o2[h] = W_kb2[h]^T W_o[h]`` stacked over the heads (y = o_lat_packed @ W_o2: the head sum inside K).
    Equal to the composite per-head bmms (``_absorb_q`` / ``_o_lat_to_heads`` + ``o_proj``) up to fp32 rounding
    (``tests/test_dsa_decode_state_host.py``)."""
    H, Dq, L = wkb1.shape
    Dv = wkb2.shape[1]
    Lq = w_qb_t.shape[0]
    hidden = w_o_t.shape[-1]
    wqb = w_qb_t.float().reshape(Lq, H, Dq)  # [Lq, H, Dq]
    w_qb1 = torch.einsum("lhd,hdm->lhm", wqb, wkb1.float()).reshape(Lq, H * L)  # [Lq, H*L]
    wo = w_o_t.float().reshape(H, Dv, hidden)  # [H, Dv, hidden]
    w_o2 = torch.einsum("hdm,hmn->hdn", wkb2.transpose(1, 2).float(), wo).reshape(H * L, hidden)
    return w_qb1, w_o2


def sparse_decode_enabled() -> bool:
    """The traced multi-user sparse decode (default). ``GLM53_SPARSE_DECODE=eager`` keeps the dense
    flash-MLA over the real cache + per-user eager sparse rows (host geometry) as the A/B reference."""
    return os.environ.get("GLM53_SPARSE_DECODE", "traced") != "eager"


class StageTimer:
    """Wall time per stage of the eager decode step (test instrumentation). ``mark(name)`` synchronizes
    the device and charges the time since the previous mark to ``name``; the allocator's DRAM bytes at
    every boundary are recorded in ``dram``. Never enable it in a traced or served path: the
    synchronizations serialize dispatch. Install with ``TtDSA.set_stage_timer``."""

    def __init__(self, device):
        self.device = device
        self.times: dict[str, float] = {}
        self.dram: dict[str, int] = {}
        self.order: list[str] = []
        self._t: float | None = None

    def reset(self):
        self.times, self.dram, self.order, self._t = {}, {}, [], None

    def start(self):
        ttnn.synchronize_device(self.device)
        self._t = time.perf_counter()

    def mark(self, name: str):
        ttnn.synchronize_device(self.device)
        now = time.perf_counter()
        if self._t is not None:
            self.times[name] = self.times.get(name, 0.0) + (now - self._t)
        if name not in self.order:
            self.order.append(name)
        try:
            mv = ttnn.device.get_memory_view(self.device, ttnn.BufferType.DRAM)
            self.dram[name] = mv.num_banks * mv.total_bytes_allocated_per_bank
        except Exception:  # pragma: no cover - informational
            pass
        self._t = time.perf_counter()

    @property
    def total(self) -> float:
        return sum(self.times.values())

    def report(self) -> str:
        gib = 2**30
        return "  ".join(
            f"{k} {self.times.get(k, 0.0) * 1e3:.2f}ms" + (f" [{self.dram[k] / gib:.2f}GiB]" if k in self.dram else "")
            for k in self.order
        )


def _compute(device, fidelity=ttnn.MathFidelity.HiFi2, fp32_acc=True, packer_l1_acc=True):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=packer_l1_acc,
    )


def _upload(device, t: torch.Tensor, dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, device=device, memory_config=memory_config)


def decode_position_tensors(
    device, host_positions: list[int], kpool: int, mesh: GLMMesh | None = None, partition: bool = False
):
    """(positions, positions // kpool, positions % kpool) as [B] int32 ROW_MAJOR device tensors
    (row-sharded over a multi-row ``mesh``: ``host_positions`` then lists ``rows * B`` entries, row-major;
    ``partition`` = the user-partitioned cache: the row's users are split over the columns too, chip (r, c)
    gets ``B_row / cols`` entries)."""

    def up(vals):
        t = torch.tensor(vals, dtype=torch.int32)
        return _meta_upload(device, t, ttnn.int32, mesh, partition)

    return up(host_positions), up([p // kpool for p in host_positions]), up([p % kpool for p in host_positions])


def _meta_upload(device, t: torch.Tensor, dtype, mesh: GLMMesh | None, partition: bool = False):
    """Host -> device ROW_MAJOR tensor; on a multi-row mesh dim 0 (the user list, row-major) is sharded over
    the rows, and with ``partition`` (user-partitioned cache) over the columns as well (dim 0 on both mesh
    axes: row r's slice is split again into ``cols`` pieces, chip (r, c) gets piece c)."""
    if mesh is not None and (mesh.rows > 1 or (partition and mesh.cols > 1)):
        return mesh.upload(
            t, dtype, rows=0, cols=0 if partition and mesh.cols > 1 else None, layout=ttnn.ROW_MAJOR_LAYOUT
        )
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def _meta_mapper(mesh: GLMMesh | None, partition: bool):
    """The mesh mapper of ``_meta_upload`` for in-place refreshes (None on one chip)."""
    if mesh is None or not (mesh.rows > 1 or (partition and mesh.cols > 1)):
        return None
    return mesh.shard_2d(0, 0 if partition and mesh.cols > 1 else None)


@dataclass
class SparseDecodeMeta:
    """Per-step metadata of the traced multi-user sparse decode: one entry per user slot, derived on the
    host from the positions exactly like ``pool_ids`` / ``ring_slots`` and refreshed in place (``update``),
    so the captured decode graph reads the live values at every replay. With ``P`` = index_kpool, ``K`` =
    index_select_pools (512), ``n = (a + 1) // P`` complete pools and ``t = (a + 1) % P`` tail tokens of a user
    at position ``a`` (the token being decoded sees [0, a]):

    * ``valid_len[u]`` [1] uint32: the top-k's search width in pools, ``max(n, K)``: a sparse user ranks its
      ``n >= K`` complete pools only (the incomplete pool ``n`` and the stale rows beyond are never looked
      at); a dense user (``n < K``) gets a throw-away top-k over ``K`` columns that the gather ignores (it lists the
      user's pools in order).
    * ``tail_pool`` [B,1,1,1] uint32: ``n``, the user's incomplete pool, gathered into compact slot ``K``.
    * (the compact pool list itself is built by the gather kernel from the ranked ids and ``tail_pool``: the
      ranked pools for a sparse user, the pools ``0 .. K-1`` in order for a dense user (``n < K``, whose tail
      pool ``n`` then sits at slot ``n``), the tail pool at slot ``K``)
    * ``cur_pos`` [B] int32: the last attended row of the compact cache for the flash-MLA decode: ``a`` for a
      dense user (rows 0..a = tokens 0..a, exact dense attention) and ``P*K - 1 + t`` for a sparse user (the
      2048 selected tokens then the ``t`` tail tokens; the tail slot's other rows and the padding are hidden).
    A dense user is one with ``n < K``, i.e. ``a < P*K - 1`` (2047); from there on every top-K pool is a real
    complete pool and the two forms coincide (HF-exact through 2050 either way)."""

    valid_len: list[ttnn.Tensor]
    tail_pool: ttnn.Tensor
    cur_pos: ttnn.Tensor
    host: list[int]
    users_per_chip: int
    # [B, 1] uint32: the same per-user top-k bounds as ``valid_len`` in one tensor (one 4-byte page per user) for
    # the batched ``topk_large_indices`` call over all users' score rows (``GLM53_TOPK_BATCHED``, default on)
    valid_len_rows: ttnn.Tensor | None = None
    # [1, 1] uint32 per chip: the chip's live pool bound (``live_bound_values``: the largest of its users' ``valid``,
    # rounded up to ``sparse_bound.GRANULE``, not clamped to the capacity) that the scorer / segmented top-k loop to
    # under ``GLM53_SPARSE_DECODE_BOUND=1``; always built and refreshed (one 4-byte page), used only with the knob
    live_bound: ttnn.Tensor | None = None
    # host regime of the step (``dense_regime``): every slot at a position <= 2050 -> the dense-skip graph
    # (``GLM53_POOL_GATHER=dense_skip``) is exact; refreshed by ``update``
    all_dense: bool = False
    # the regime the last ``_decode_unified`` graph built on this meta used ("dense" = the dense-skip graph, "sparse" =
    # the gather graph, None = none yet); ``update`` refuses to leave the dense regime while a dense-skip graph is the
    # last one built (a captured trace would replay it): see ``POOL_GATHER_ENV``
    graph_regime: str | None = None

    @property
    def regime(self) -> str:
        """ "dense" when every slot is in the dense range, else "sparse": the key a decode trace is captured under
        when ``GLM53_POOL_GATHER=dense_skip`` (the graph differs per regime)."""
        return "dense" if self.all_dense else "sparse"

    def clear_graph_regime(self):
        """Eager callers (no captured trace to protect) reset the regime guard before crossing the dense bound."""
        self.graph_regime = None

    @staticmethod
    def host_values(positions: list[int], cfg: GLM53FlashConfig):
        """(valid [B], tail [B,1,1,1], cur_pos [B]) as host tensors (the per-user formulas live in
        ``packed_inputs.sparse_host_values``, shared with the packed decode inputs)."""
        B = len(positions)
        valid, tail, cur = sparse_host_values(positions, cfg.index_kpool, cfg.index_select_pools)
        return (
            torch.tensor(valid, dtype=torch.int64),
            torch.tensor(tail, dtype=torch.int64).reshape(B, 1, 1, 1),
            torch.tensor(cur, dtype=torch.int32),
        )

    @staticmethod
    def live_bound_values(valid: torch.Tensor, R: int, C: int, B: int) -> torch.Tensor:
        """[R*C, 1] int64: the live pool bound of every chip group (chip (r, c) = row r*C + c) for the per-user top-k
        widths ``valid`` [R*C*B] (``host_values``): ``sparse_bound.live_bound_pools`` of the group's B users."""
        vl = valid.reshape(R * C, B).tolist()
        return torch.tensor([live_bound_pools(v) for v in vl], dtype=torch.int64).reshape(R * C, 1)

    @staticmethod
    def _chips(mesh: GLMMesh | None, partition: bool) -> tuple[int, int]:
        """(row groups, column groups) the user list is split over."""
        if mesh is None:
            return 1, 1
        return mesh.rows, (mesh.cols if partition else 1)

    @classmethod
    def from_positions(
        cls, device, positions: list[int], cfg: GLM53FlashConfig, mesh: GLMMesh | None = None, partition: bool = False
    ) -> "SparseDecodeMeta":
        """``positions`` lists every user slot of the mesh (``rows * B`` on a multi-row mesh, row-major; with the
        user-partitioned cache (``partition``) chip (r, c) owns users ``[c*B/cols, (c+1)*B/cols)`` of row r)."""
        R, C = cls._chips(mesh, partition)
        B = len(positions) // (R * C)
        valid, tail, cur = cls.host_values(positions, cfg)
        # per local user u: a [R, C] tensor (chip (r, c)'s user u), sharded to one element per chip
        vl = valid.reshape(R, C, B)
        valid_t = [
            mesh.upload(
                vl[:, :, u].contiguous(), ttnn.uint32, rows=0, cols=1 if C > 1 else None, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            if mesh is not None and (R > 1 or C > 1)
            else ttnn.from_torch(
                vl[0, 0, u : u + 1].contiguous(), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
            )
            for u in range(B)
        ]
        return cls(
            valid_len=valid_t,
            tail_pool=_meta_upload(device, tail, ttnn.uint32, mesh, partition),
            cur_pos=_meta_upload(device, cur, ttnn.int32, mesh, partition),
            host=list(positions),
            users_per_chip=B,
            valid_len_rows=_meta_upload(device, valid.reshape(-1, 1).contiguous(), ttnn.uint32, mesh, partition),
            live_bound=_meta_upload(device, cls.live_bound_values(valid, R, C, B), ttnn.uint32, mesh, partition),
            all_dense=dense_regime(positions, cfg),
        )

    def update(
        self,
        positions: list[int],
        cfg: GLM53FlashConfig,
        mesh: GLMMesh | None = None,
        partition: bool = False,
        *,
        upload: bool = True,
    ):
        """Refresh every tensor in place for the new positions (trace-replay safe). Refuses to leave the dense
        regime while the last graph built on this meta is the dense-skip one (``graph_regime == "dense"``): the
        caller must re-capture under the new ``regime`` (or ``clear_graph_regime`` for an eager flow) first.
        ``upload=False`` keeps the host bookkeeping (the regime guard, ``host``, ``all_dense``) and skips the device
        copies: the packed decode inputs (``GLM53_DECODE_PACKED_INPUTS``, ``tt/packed_inputs.py``) carry the values."""
        R, C = self._chips(mesh, partition)
        B = self.users_per_chip
        assert len(positions) == R * C * B, (len(positions), R, C, B)
        all_dense = dense_regime(positions, cfg)
        if self.graph_regime == "dense" and not all_dense:
            raise RuntimeError(
                f"{POOL_GATHER_ENV}=dense_skip: the last decode graph built on this SparseDecodeMeta attends the paged "
                f"cache directly (every slot <= {dense_exact_max_pos(cfg)}); the new positions "
                f"{[a for a in positions if a > dense_exact_max_pos(cfg)]} leave the dense regime, which a replay of "
                "that graph would attend densely (not HF semantics). Re-capture the decode trace under "
                "SparseDecodeMeta.regime == 'sparse' (or clear_graph_regime() for an eager flow) before updating."
            )
        if upload:
            self._upload(positions, cfg, mesh, partition)
        self.host = list(positions)
        self.all_dense = all_dense

    def _upload(self, positions: list[int], cfg: GLM53FlashConfig, mesh: GLMMesh | None, partition: bool):
        """The device copies of ``update``: ``valid_len[u]`` (one per user), ``tail_pool``, ``valid_len_rows``,
        ``cur_pos`` (``3 + users_per_chip`` ``copy_host_to_device_tensor`` calls)."""
        R, C = self._chips(mesh, partition)
        B = self.users_per_chip
        valid, tail, cur = self.host_values(positions, cfg)
        mapper = _meta_mapper(mesh, partition)
        vmapper = mesh.shard_2d(0, 1 if C > 1 else None) if mesh is not None and (R > 1 or C > 1) else None

        def put(t: torch.Tensor, dtype, dst: ttnn.Tensor, m=mapper):
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=m), dst
            )

        vl = valid.reshape(R, C, B)
        for u in range(B):
            put(
                vl[:, :, u].contiguous() if vmapper is not None else vl[0, 0, u : u + 1].contiguous(),
                ttnn.uint32,
                self.valid_len[u],
                vmapper,
            )
        put(tail, ttnn.uint32, self.tail_pool)
        if self.valid_len_rows is not None:
            put(valid.reshape(-1, 1).contiguous(), ttnn.uint32, self.valid_len_rows)
        if self.live_bound is not None:
            put(self.live_bound_values(valid, R, C, B), ttnn.uint32, self.live_bound)
        put(cur, ttnn.int32, self.cur_pos)


def fill_rows(t_pad: int, first_block: int, blocks_per_seq: int, block_size: int) -> int:
    """Rows of a ``t_pad``-row prefill chunk that the paged fill writes from block ``first_block`` of a slot of
    ``blocks_per_seq`` blocks: the chunk's rows past the slot end are bucket padding (``mesh_model.eager_pad_length``
    raises a prompt's last chunk to a warmed length), never positions (the real tokens end inside the slot by
    construction). The 2026-10-27 1 M harness run: slot 32748 blocks (prompt + gen + 256), last 16 K chunk from
    block 32256 -> 492 blocks = 15744 rows written, the 640 padding rows dropped."""
    avail = (int(blocks_per_seq) - int(first_block)) * int(block_size)
    if avail <= 0:
        raise ValueError(f"fill from block {first_block} of a {blocks_per_seq}-block slot: nothing to write")
    return min(int(t_pad), avail)


class TtLatentCache:
    """Paged latent cache for one DSA layer (and its page table) plus the pooled-key index, device-owned."""

    def __init__(
        self,
        device,
        cfg: GLM53FlashConfig,
        *,
        max_batch: int,
        max_seq_len: int,
        block_size: int = 32,
        dtype=ttnn.bfloat16,
        mesh: GLMMesh | None = None,
        partition_users: bool = False,
        prefill_partition: str | None = None,
    ):
        """``max_batch`` user slots (per mesh row). ``partition_users`` on a multi-column mesh = the user-partitioned
        layout (doc/mesh/sparse_1m.md section 3): column chip c owns users ``[c*B/cols, (c+1)*B/cols)`` of its
        row. ``prefill_partition`` (default: the ``GLM53_PREFILL_PARTITION`` knob, ``scratch``) says how a
        prefill is served: ``scratch`` allocates ``users_per_chip + 1`` slots (the last one the prefill scratch
        slot every column fills and attends over with its 16 heads); ``heads2users`` allocates the
        ``users_per_chip`` slots only (plus ``DUMMY_BLOCKS`` spare blocks): the owner column attends the chunk
        with all 64 heads (gathered over the columns, ``TtDSA._sparse_attention(owner=...)``) over its own slot,
        the other columns write the chunk into the dummies (section 7). Decode addresses the local slots either
        way (one prompt in flight per chip, chunks back to back)."""
        self.device = device
        self.mesh = mesh
        self.block_size = block_size
        self.max_batch = max_batch
        self.partition = bool(partition_users) and mesh is not None and mesh.cols > 1
        mode = prefill_partition or os.environ.get("GLM53_PREFILL_PARTITION", "scratch")
        if mode not in PREFILL_PARTITION_MODES:
            raise ValueError(f"GLM53_PREFILL_PARTITION={mode!r}: expected one of {PREFILL_PARTITION_MODES}")
        self.prefill_mode: str | None = mode if self.partition else None
        if self.partition:
            assert max_batch % mesh.cols == 0, (max_batch, mesh.cols)
            self.users_per_chip = max_batch // mesh.cols
            if mode == "scratch":
                self.scratch: int | None = self.users_per_chip
                self.slots = self.users_per_chip + 1
            else:
                self.scratch = None
                self.slots = self.users_per_chip
        else:
            self.users_per_chip, self.scratch, self.slots = max_batch, None, max_batch
        self.blocks_per_seq = -(-max_seq_len // block_size)
        self.data_blocks = self.blocks_per_seq * self.slots
        self.dummy_blocks = DUMMY_BLOCKS if self.prefill_mode == "heads2users" else 0
        self.max_blocks = self.data_blocks + self.dummy_blocks
        self.dtype = dtype
        self.latent_dim = cfg.kv_lora_rank
        self.cache = ttnn.zeros(
            (self.max_blocks, 1, block_size, self.latent_dim),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # identity page table [slots, blocks_per_seq]: slot b owns blocks [b*bps, (b+1)*bps)
        pt = torch.arange(self.data_blocks, dtype=torch.int32).reshape(self.slots, self.blocks_per_seq)
        self.set_page_table(pt)
        self.rows_per_user = self.blocks_per_seq * block_size
        # the sparse prefill's prefix read rounds its row count up to this (GLM53_PREFIX_BUCKET, 0 = exact): one kv
        # shape per prefix bucket instead of one per chunk start (vllm_config.PREFIX_BUCKET_ENV)
        self.prefix_bucket = int(os.environ.get(PREFIX_BUCKET_ENV, "0"))
        # pooled-key index (its own identity table; a slot's pools are contiguous regardless of the latent table)
        self.index = TtIndexCache(
            device,
            cfg,
            max_batch=self.slots,
            max_seq_len=self.rows_per_user,
            mesh=mesh,
            users_per_chip=self.users_per_chip if self.partition else None,
            scratch=self.scratch,
            dummy=self.dummy_blocks > 0,
        )
        # the incremental whole-prefix ROW_MAJOR copy of the sparse prefill (``prefix_rows_rm``): one arena per
        # layer for the prompt in flight, released by ``release_prefix`` / the next user's prefill
        self._prefix: _PrefixArena | None = None
        self.n_dsa_layers = sum(1 for t in cfg.layer_types if t != "linear_attention") if cfg.layer_types else 11

    def set_page_table(self, page_table: torch.Tensor):
        """Any [slots, blocks_per_seq] mapping (vLLM-style); every block id must be unique. ``page_table`` is
        the decode table (the first ``users_per_chip`` rows when partitioned: the paged ops pair batch entry
        b with row b) and ``page_table_full`` the whole table (plus, with dummy blocks, one extra row
        ``users_per_chip`` that maps every block onto the dummies: the row a non-owner column's traced prefill
        fill names)."""
        assert tuple(page_table.shape) == (self.slots, self.blocks_per_seq), page_table.shape
        assert page_table.unique().numel() == page_table.numel(), "page table blocks must be unique"
        assert int(page_table.max()) < self.data_blocks, "page table blocks must lie below the dummy region"
        host = page_table.to(torch.int32).contiguous()
        if self.dummy_blocks:
            dummy = self.data_blocks + torch.arange(self.blocks_per_seq, dtype=torch.int32) % self.dummy_blocks
            host = torch.cat([host, dummy.reshape(1, -1)])
        self.page_table_host = host
        self.page_table_full = ttnn.from_torch(host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        self.page_table = (
            ttnn.slice(self.page_table_full, [0, 0], [self.users_per_chip, self.blocks_per_seq])
            if self.partition
            else self.page_table_full
        )
        self._prefill_views: dict = {}

    @property
    def dummy_row(self) -> int | None:
        """The page-table row of the dummy blocks (heads2users), else None."""
        return self.slots if self.dummy_blocks else None

    def prefill_slot(self, user: int) -> int:
        """The slot every chip reads during the prefill of ``user``: the scratch slot (scratch layout), the user's
        local slot (heads2users: real on the owner column, the column's own data elsewhere), or the user."""
        if not self.partition:
            return user
        return self.scratch if self.scratch is not None else user % self.users_per_chip

    def owner_of(self, user: int) -> int | None:
        """The column that attends the prefill of ``user`` with all heads over its own slot (heads2users), else
        None (every column attends its own heads over a replicated / scratch cache)."""
        return user // self.users_per_chip if self.prefill_mode == "heads2users" else None

    def _view_rows(self, user: int) -> list[torch.Tensor]:
        """The host page-table rows a prefill of ``user`` fills (batch_idx 0), one ``[n, blocks_per_seq]`` tensor per
        view: not partitioned = the user's row; partitioned = the per-chip owner row (the user's slot on its owner
        column, the scratch slot or the dummy row elsewhere) and, with a scratch slot, the replicated scratch row."""
        if not self.partition:
            return [self.page_table_host[user : user + 1]]
        owner, slot = user // self.users_per_chip, user % self.users_per_chip
        other = self.scratch if self.scratch is not None else self.dummy_row
        rows = [torch.stack([self.page_table_host[slot if c == owner else other] for c in range(self.mesh.cols)])]
        if self.scratch is not None:
            rows.append(self.page_table_host[self.scratch : self.scratch + 1])
        return rows

    def _upload_view(self, rows: torch.Tensor, per_chip: bool) -> ttnn.Tensor:
        if per_chip:
            return self.mesh.upload(rows, ttnn.int32, cols=0, layout=ttnn.ROW_MAJOR_LAYOUT)
        if getattr(self, "mesh", None) is not None:
            return self.mesh.upload(rows, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.from_torch(rows.contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)

    def _prefill_views_for(self, user: int, b0: int = 0) -> list[ttnn.Tensor]:
        """``[1, blocks_per_seq]`` page-table views a prefill of ``user`` fills from block ``b0`` on (batch_idx 0;
        ``_view_rows``). ``b0 == 0``: the user's views, built once (under live traces a chunk transient, GLMMesh.lazy).
        ``b0 > 0``: the rows shifted by ``b0`` on the host and padded back to the full width with the row's last block
        (never written: ``fill`` clips its rows to the slot end), uploaded for this call (the caller frees them). The
        width never changes, so the fill runs ONE program for every chunk start and prefix-cache hit position; the
        earlier device slice ``[0, b0] .. [1, blocks_per_seq]`` was a program per start block, compiled at request
        time under the sealed cache (the served APC hit at 1536, 2026-10-28 8x4)."""
        per_chip = self.partition
        if b0 == 0:
            key = ("fill", user)
            make = lambda: [self._upload_view(r, per_chip and i == 0) for i, r in enumerate(self._view_rows(user))]
            return lazy_const(getattr(self, "mesh", None), self._prefill_views, key, make)
        if not 0 < b0 < self.blocks_per_seq:
            raise ValueError(f"fill start block {b0} outside [1, {self.blocks_per_seq})")
        out = []
        for i, r in enumerate(self._view_rows(user)):
            shifted = torch.cat([r[:, b0:], r[:, -1:].expand(-1, b0)], dim=1).contiguous()
            out.append(self._upload_view(shifted, per_chip and i == 0))
        return out

    def fill(self, latent: ttnn.Tensor, user: int, start: int = 0, slot=None):
        """latent [1, 1, T_pad, 512] (T_pad % block_size == 0) -> pages of ``user`` from position
        ``start`` (a block multiple): page-table view from block ``start // block_size`` on.
        ``slot`` (a ``trace_prefill.PrefillSlot``, start == 0 only): the slot index comes from its
        device tensors (``batch_idx_tensor``: the same op, the user chosen on device), so the fill is
        trace-replayable for any user; partitioned caches fill the scratch slot as well."""
        if start % self.block_size:
            raise ValueError(f"fill start {start} must be a multiple of the block size {self.block_size}")
        if slot is not None and start:
            raise ValueError("a device-slot fill starts a new prompt (start == 0)")
        src = latent if latent.dtype == self.dtype else ttnn.typecast(latent, self.dtype)
        b0 = start // self.block_size
        rows = fill_rows(src.shape[-2], b0, self.blocks_per_seq, self.block_size)
        clipped = None
        if rows < src.shape[-2]:  # bucket padding past the slot end: not written (see fill_rows)
            clipped = ttnn.slice(src, [0, 0, 0, 0], [1, 1, rows, src.shape[-1]])
            src = clipped
        try:
            if slot is not None:
                pt = self.page_table_full
                ttnn.experimental.paged_fill_cache(self.cache, src, pt, batch_idx_tensor=slot.cache_idx)
                if self.scratch is not None:
                    ttnn.experimental.paged_fill_cache(self.cache, src, pt, batch_idx_tensor=slot.scratch_idx)
                return
            if start == 0 and not self.partition:
                ttnn.experimental.paged_fill_cache(self.cache, src, self.page_table, batch_idx=user)
                return
            for view in self._prefill_views_for(user, b0):  # fixed-width views: one fill program for every start
                ttnn.experimental.paged_fill_cache(self.cache, src, view, batch_idx=0)
                if b0:
                    ttnn.deallocate(view)
        finally:
            if clipped is not None:
                ttnn.deallocate(clipped)

    def update(self, latent_rows: ttnn.Tensor, positions: ttnn.Tensor):
        """latent_rows [1, B, 1, 512] (one token per user), positions [B] int32 row-major."""
        sh = height_shard_rows(self.device, latent_rows)
        ttnn.experimental.paged_update_cache(self.cache, sh, update_idxs_tensor=positions, page_table=self.page_table)
        ttnn.deallocate(sh)

    def user_rows(self, user: int, n_rows: int) -> ttnn.Tensor:
        """The first ``n_rows`` (rounded up to whole blocks) latent rows of ``user`` in page-table
        order as a fresh TILE [1, 1, rows, 512] tensor."""
        n_blocks = min(self.blocks_per_seq, -(-n_rows // self.block_size))
        ids = self.page_table_host[self.prefill_slot(user), :n_blocks].tolist()
        if ids == list(range(ids[0], ids[0] + n_blocks)):  # contiguous (identity-like) fast path
            blocks = ttnn.slice(self.cache, [ids[0], 0, 0, 0], [ids[0] + n_blocks, 1, self.block_size, self.latent_dim])
        else:
            parts = [ttnn.slice(self.cache, [b, 0, 0, 0], [b + 1, 1, self.block_size, self.latent_dim]) for b in ids]
            blocks = ttnn.concat(parts, dim=0)
            for p in parts:
                ttnn.deallocate(p)
        return ttnn.reshape(blocks, [1, 1, n_blocks * self.block_size, self.latent_dim])

    # The whole-prefix copy below is bandwidth-bound (~90 GB/s: 1.5 ms at 131 K, 13 ms at 1 M positions on one chip,
    # tests/test_prefix_rows.py), so it is a negligible part of a long-context prefill chunk; the direct untilize of the
    # cache pages (no slice) is 1.4-1.6x cheaper from ~512 K positions on but has a ~2.7 ms floor, hence the threshold.
    DIRECT_UNTILIZE_MIN_ROWS = 524288

    def user_rows_rm(self, user: int, n_rows: int, direct: bool = True) -> ttnn.Tensor:
        """``user_rows`` as a fresh ROW_MAJOR [1, 1, rows, 512] tensor (the sparse_sdpa kv contract). ``direct``:
        when the user's pages are the cache's first blocks (slot 0 of an identity table, the single-slot long-context
        state) and the prefix is long, untilize them straight out of the paged cache (``untilize_with_unpadding``
        of the leading block range, no TILE slice copy); bit-identical to the slice + untilize path."""
        if self.prefix_bucket:  # the next prefix bucket: rows past the prompt are never selectable (no valid pools)
            n_rows = min(self.rows_per_user, -(-int(n_rows) // self.prefix_bucket) * self.prefix_bucket)
        n_blocks = min(self.blocks_per_seq, -(-n_rows // self.block_size))
        ids = self.page_table_host[self.prefill_slot(user), :n_blocks].tolist()
        if direct and n_blocks * self.block_size >= self.DIRECT_UNTILIZE_MIN_ROWS and ids == list(range(n_blocks)):
            rm = ttnn.untilize_with_unpadding(
                self.cache, [n_blocks - 1, 0, self.block_size - 1, self.latent_dim - 1]
            )  # ROW_MAJOR [n_blocks, 1, 32, L]
            return ttnn.reshape(rm, [1, 1, n_blocks * self.block_size, self.latent_dim])  # a row-major view
        rows = self.user_rows(user, n_rows)
        rm = ttnn.to_layout(rows, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(rows)
        return rm

    # ------------------------------------------------- incremental whole-prefix copy (GLM53_PREFIX_INCREMENTAL)
    @property
    def prefix_arena_bytes(self) -> int:
        """Bytes of one layer's arena: the slot's full length of ROW_MAJOR latent rows (``rows_per_user x L`` bf16)."""
        return self.rows_per_user * self.latent_dim * 2

    @property
    def prefix_incremental(self) -> bool:
        """Whether ``prefix_rows_rm`` keeps the copy across chunks (the ``GLM53_PREFIX_INCREMENTAL`` knob, read at
        first use: 0 / 1 / auto = every DSA layer's arena fits the budget). The per-chunk form is a 1-slot scratch
        layout's or any layout's exact equivalent: the arena holds what ``user_rows_rm`` would copy."""
        if not hasattr(self, "_prefix_incremental"):
            mode = os.environ.get("GLM53_PREFIX_INCREMENTAL", PREFIX_INCREMENTAL)
            if mode == "auto":
                budget = int(os.environ.get("GLM53_PREFIX_INCREMENTAL_BUDGET_MB", PREFIX_INCREMENTAL_BUDGET_MB))
                self._prefix_incremental = self.n_dsa_layers * self.prefix_arena_bytes <= budget * 2**20
            elif mode in ("0", "1"):
                self._prefix_incremental = mode == "1"
            else:
                raise ValueError(f"GLM53_PREFIX_INCREMENTAL={mode!r}: expected 0, 1 or auto")
        return self._prefix_incremental

    def release_prefix(self, user: int | None = None):
        """Drop the layer's prefix arena (the end of the prompt in flight; ``user`` given = only if it is that
        user's). The next sparse chunk rebuilds it from the paged cache."""
        a = self._prefix
        if a is None or (user is not None and a.user != user):
            return
        self._prefix = None
        if a.rows.is_allocated():
            ttnn.deallocate(a.rows)

    def prefix_sync(self, user: int, start: int, chunk_c: ttnn.Tensor | None):
        """Keep the arena consistent with a cache write of ``user``'s rows from ``start`` (``fill``) that bypasses
        ``prefix_rows_rm`` (``TtDSA.prefill_state``, the traced slot fill): append the chunk's latent ``chunk_c``
        [1,1,T,L] TILE when it continues the arena's valid prefix, else forget the arena's contents (the next
        sparse chunk rebuilds them). ``chunk_c=None`` (a device-slot fill: the user is not a host value) releases."""
        a = self._prefix
        if a is None:
            return
        if chunk_c is None or a.user != user or a.n_valid != start:
            if chunk_c is None or a.user != user:
                self.release_prefix()
            else:
                a.n_valid = 0  # the rows are stale below ``start``: rebuilt on the next use
            return
        self._append_rows(a, chunk_c, start)

    def prefix_rows_rm(self, user: int, start: int, n_chunk: int, chunk_c: ttnn.Tensor | None = None):
        """The sparse_sdpa kv of a prefill chunk of ``user`` at ``start`` (``n_chunk`` padded rows): the user's
        ROW_MAJOR latent rows [0, start + n_chunk) as ``(kv, owned)``. ``owned`` = a fresh ``user_rows_rm`` copy the
        caller deallocates (the default, ``GLM53_PREFIX_INCREMENTAL=0``). Incremental: the layer's persistent arena
        ``[1,1,rows_per_user,L]`` (not to be deallocated), whose rows [0, start) were appended chunk by chunk and
        which now receives rows [start, start + n_chunk) from the chunk's own latent ``chunk_c`` [1,1,n_chunk,L]
        TILE (the tensor ``fill`` wrote into the pages: ``paged_fill_cache`` is a raw tile copy, so its untilize is
        bit-identical to the pages' untilize) through ``ttnn.experimental.slice_write`` (a ROW_MAJOR region write in
        place). A chunk that does not continue the arena (another user, a new prompt at 0, a seeded prefix the arena
        did not see, ``chunk_c`` missing) rebuilds the whole prefix once from the paged cache (``user_rows_rm`` +
        one region write), after which the chunks append again. The arena's rows past ``start + n_chunk`` are
        stale and never indexed (every index row is causal, so sparse_sdpa reads below the prefix end only); the
        kv it hands out is the slot's full length (sparse_sdpa's T is a runtime value).

        With the heads2users layout the owner column appends the user's rows; a non-owner column appends the same
        chunk latent into its own arena (SPMD: identical ops and shapes), which is as discardable as the slot rows
        it copied before (its attention result is dropped by ``TtDSA._sparse_attention(owner=...)``)."""
        n_rows = start + n_chunk
        if not self.prefix_incremental:
            return self.user_rows_rm(user, n_rows), True
        a = self._prefix
        if a is None or a.user != user:
            self.release_prefix()
            rows = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, self.rows_per_user, self.latent_dim]),
                self.dtype,
                ttnn.ROW_MAJOR_LAYOUT,
                self.device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
            a = self._prefix = _PrefixArena(user=user, rows=rows, n_valid=0)
        if start == 0:
            a.n_valid = 0  # a new prompt in this slot: the arena's old rows are dead
        if a.n_valid == start and chunk_c is not None:
            self._append_rows(a, chunk_c, start)
        else:
            whole = self.user_rows_rm(user, n_rows)  # the rebuild: one whole copy out of the paged cache ...
            self._write_rows(a, whole, 0)  # ... placed at row 0 of the arena
            ttnn.deallocate(whole)
            a.n_valid = n_rows
        return a.rows, False

    def _append_rows(self, a: "_PrefixArena", chunk_c: ttnn.Tensor, start: int):
        """Rows [start, start + T) of the arena := the chunk latent ``chunk_c`` [1,1,T,L] TILE (untilized here)."""
        src = chunk_c if chunk_c.dtype == self.dtype else ttnn.typecast(chunk_c, self.dtype)
        rm = ttnn.to_layout(src, ttnn.ROW_MAJOR_LAYOUT)
        if src is not chunk_c:
            ttnn.deallocate(src)
        self._write_rows(a, rm, start)
        ttnn.deallocate(rm)
        a.n_valid = start + chunk_c.shape[-2]

    def _write_rows(self, a: "_PrefixArena", rm: ttnn.Tensor, row0: int):
        """ROW_MAJOR rows ``rm`` [1,1,n,L] -> arena rows [row0, row0 + n) in place (``slice_write``: rank 4, bf16,
        ROW_MAJOR, interleaved output, unit steps, the full last dim; end indices exclusive)."""
        n = rm.shape[-2]
        if row0 + n > self.rows_per_user:
            raise ValueError(f"prefix arena: rows [{row0}, {row0 + n}) exceed the slot length {self.rows_per_user}")
        ttnn.experimental.slice_write(rm, a.rows, [0, 0, row0, 0], [1, 1, row0 + n, self.latent_dim], [1, 1, 1, 1])


@dataclass
class _PrefixArena:
    """One DSA layer's persistent ROW_MAJOR latent copy of the prompt in flight (``TtLatentCache.prefix_rows_rm``):
    ``rows`` [1,1,rows_per_user,L] bf16 holds ``user``'s positions [0, n_valid) in page order."""

    user: int
    rows: ttnn.Tensor
    n_valid: int


class TtDSA(LightweightModule):
    def __init__(
        self,
        device,
        cfg: GLM53FlashConfig,
        attn_state_dict: dict[str, torch.Tensor],
        *,
        layer_idx: int,
        weight_dtype=ttnn.bfloat8_b,
        mesh: GLMMesh | None = None,
        fused_decode: str | bool | None = None,
        decode_state: str | bool | None = None,
    ):
        """``attn_state_dict``: raw checkpoint tensors of ``self_attn.`` (prefix stripped, fp8 dequantized).
        ``fused_decode``: "fused" | "composite" (default: the ``GLM53_DSA_FUSED`` knob); ``True`` = "fused".
        ``decode_state``: "0" | "1" | "latent" (default: the ``GLM53_DSA_DECODE_STATE`` knob, read now); ``True`` =
        "1": the traced sparse decode (``sparse_meta`` given) runs ``_decode_state_unified`` (the fused state program
        + the GLM53_DSA_FOLD folds) instead of the composite ops.

        On a mesh the heads are split over the columns (``mesh.tp``, ``self.H`` = local heads): the
        latent stems (q_a, kv_a, their norms) and the indexer are replicated, q_b / kv_b / o_proj are
        head-sharded, the latent cache and the pooled-key index are replicated across the columns
        (Phase-2 first step; the user-partitioned cache comes next), and the output projection is
        all-reduced over the columns."""
        self.device = device
        self.mesh = mesh if mesh is not None else single_chip(device)
        tp = self.mesh.tp
        self.cfg = cfg
        self.layer_idx = layer_idx
        H, Dq, Dv, L = cfg.num_attention_heads, cfg.qk_head_dim, cfg.v_head_dim, cfg.kv_lora_rank
        assert H % tp == 0, (H, tp)
        self.H_total = H
        self.H, self.Dq, self.Dv, self.L = H // tp, Dq, Dv, L
        self.scale = cfg.mla_softmax_scale
        sd = attn_state_dict
        up = self.mesh.upload
        self.w_qa = up(sd["q_a_proj.weight"].t(), weight_dtype)  # [4096,1536]
        self.qa_norm = up(sd["q_a_layernorm.weight"].float().reshape(1, 1, 1, -1), ttnn.bfloat16)
        self.w_qb = up(sd["q_b_proj.weight"].t(), weight_dtype, cols=-1)  # [1536, H*256] head-major
        self.w_kva = up(sd["kv_a_proj_with_mqa.weight"].t(), weight_dtype)  # [4096, 512]
        self.kva_norm = up(sd["kv_a_layernorm.weight"].float().reshape(1, 1, 1, -1), ttnn.bfloat16)
        wkb = sd["kv_b_proj.weight"].reshape(H, Dq + Dv, L)  # [H, 512, L]
        wkb1 = wkb[:, :Dq, :]  # k_nope[h] = c @ wkb1[h]^T   -> [H, 256, L]
        wkb2 = wkb[:, Dq:, :]  # v[h]      = c @ wkb2[h]^T   -> [H, 256, L]
        # prefill: K = c @ Wk^T for all heads at once: [T, L] @ [L, H*256]; same for V
        self.w_k_all = up(
            wkb1.transpose(0, 1).reshape(Dq, H, L).permute(2, 1, 0).reshape(L, H * Dq), weight_dtype, cols=-1
        )
        self.w_v_all = up(wkb2.permute(2, 0, 1).reshape(L, H * Dv), weight_dtype, cols=-1)
        # decode (absorbed): per-head [H, 256, L] for q -> latent, and [H, L, 256] for latent -> v space
        self.w_absorb_q = up(wkb1.reshape(1, H, Dq, L), weight_dtype, cols=1)
        self.w_absorb_o = up(wkb2.transpose(1, 2).reshape(1, H, L, Dv), weight_dtype, cols=1)
        self.w_o = up(sd["o_proj.weight"].t(), weight_dtype, cols=-2)  # [H*256, 4096] row-parallel
        self.indexer = TtDSAIndexer(
            device, cfg, {k[len("indexer.") :]: v for k, v in sd.items() if k.startswith("indexer.")}, mesh=self.mesh
        )
        # Decode stem: the five narrow projections of x (q_a 1536 | kv_a 512 | indexer k 128 | pool gate 128 |
        # indexer head weights Hi) as ONE [4096 x 2336] matmul (74 column blocks on 74 cores; the separate
        # matmuls sat on 4-48 cores for ~170 us) followed by tile-aligned column slices. bf16 weights keep
        # the indexer's numerics (its weights are bf16; GLM53_DSA_STEM_DTYPE=bfloat8_b halves the read).
        Lq, Hi, Di = cfg.q_lora_rank, self.indexer.Hi, self.indexer.D
        stem = torch.cat(
            [
                sd["q_a_proj.weight"].t().float(),
                sd["kv_a_proj_with_mqa.weight"].t().float(),
                sd["indexer.wk.weight"].t().float(),
                sd["indexer.index_kpool_compress_gate"].t().float(),
                sd["indexer.weights_proj.weight"].float().t() * Hi**-0.5,
            ],
            dim=-1,
        )
        stem_dtype = {"bfloat16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b}[
            os.environ.get("GLM53_DSA_STEM_DTYPE", "bfloat16")
        ]
        self.w_stem = up(stem, stem_dtype)  # [4096, Lq + L + Di + Di + Hi], replicated
        b = [0, Lq, Lq + L, Lq + L + Di, Lq + L + 2 * Di, Lq + L + 2 * Di + Hi]
        assert all(v % 32 == 0 for v in b), b
        self.stem_bounds = {
            "qa": (b[0], b[1]),
            "kva": (b[1], b[2]),
            "k": (b[2], b[3]),
            "g": (b[3], b[4]),
            "w": (b[4], b[5]),
        }
        # prefill rows from here on use the sparse path (they see >= 2048 pooled tokens: full index rows)
        self.sparse_row_start = cfg.index_topk  # 2048
        # decode: dense attention is HF-exact while floor((a+1)/4) <= index_topk/4, i.e. a <= 2050
        self.dense_exact_max_pos = cfg.index_topk + cfg.index_kpool - 2  # 2050
        self.hifi2 = _compute(device)
        self.hifi4 = _compute(device, ttnn.MathFidelity.HiFi4, packer_l1_acc=False)
        # decode-shaped (M <= 32) projections use the 1D mcast configs of tt/matmul_configs.py
        self.lin_qa = DecodeLinear(device, self.w_qa)
        self.lin_stem = DecodeLinear(device, self.w_stem)
        self.lin_qb = DecodeLinear(device, self.w_qb)
        self.lin_kva = DecodeLinear(device, self.w_kva)
        self.lin_o = DecodeLinear(device, self.w_o)
        grid = device.compute_with_storage_grid_size()
        self.sdpa_prefill_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=128, k_chunk_size=128, exp_approx_mode=False
        )
        self.full_grid = grid
        self._decode_cfgs: dict[int, ttnn.SDPAProgramConfig] = {}
        self._pad_cache: dict[tuple, ttnn.Tensor] = {}
        self._cu_windows: dict[int, ttnn.Tensor] = {}  # chunked-prefill SDPA window bounds per key length
        self._slab_consts: dict[tuple, ttnn.Tensor] = {}  # slab-local prefill: per-row Q offsets / select masks
        self.sdpa_compute = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        self.stage_timer: StageTimer | None = None
        # test hook: ``on_indices(query_start, indices)`` sees every sparse-prefill index block [1,1,rows,W]
        # (uint32 ROW_MAJOR, read-only, deallocated after the call) with the absolute position of its first row
        self.on_indices = None
        if fused_decode is None:
            self.fused_decode = DSA_FUSED_DEFAULT
        elif isinstance(fused_decode, bool):
            self.fused_decode = "fused" if fused_decode else "composite"
        else:
            self.fused_decode = str(fused_decode)
        if decode_state is None:
            self.decode_state = decode_state_default()
        elif isinstance(decode_state, bool):
            self.decode_state = "1" if decode_state else "0"
        else:
            self.decode_state = str(decode_state)
        if self.decode_state not in DSA_DECODE_STATE_MODES:
            raise ValueError(f"decode_state={self.decode_state!r}: expected one of {DSA_DECODE_STATE_MODES}")
        # the folds of the state block: validated only when that block is on (the dense fused block reads DSA_FOLD)
        self.folds = dsa_folds(strict=self.decode_state != "0")
        self.state_op: DsaDecodeState | None = None
        if self.fused_decode == "fused" or self.decode_state != "0":
            self._init_decode_state(sd, wkb1, wkb2, weight_dtype)
        # the sparse-decode latent gather (model-local generic op) and its persistent compact output per batch size
        self.pool_gather = PagedPoolGather(device, pool_size=cfg.index_kpool)
        self._compact: dict[int, ttnn.Tensor] = {}
        # GLM53_POOL_GATHER: "gather" (default) | "dense_skip" (B8a: no gather while every slot is <= 2050)
        self.pool_gather_mode = pool_gather_mode()
        # the per-row-bounded top-k over all users' score rows (model-local generic op) and its persistent output
        self.topk_rows = TopkRows(device, k=cfg.index_select_pools)
        self._top_buf: dict[int, ttnn.Tensor] = {}
        # the batched one-query-per-user indexer scorer (model-local generic op) and its persistent score rows
        self.index_scorer = IndexScoreDecode(device)
        self._score_buf: dict[int, ttnn.Tensor] = {}
        # GLM53_SPARSE_DECODE_BOUND: the scorer / segmented top-k loop to the step's live pool bound (``SPARSE_BOUND_ENV``).
        # An instance attribute so a test can A/B it on one model (flipping it changes the graph: re-capture the trace)
        self.sparse_bound = sparse_decode_bound_enabled()

    def _init_decode_state(self, sd: dict[str, torch.Tensor], wkb1: torch.Tensor, wkb2: torch.Tensor, weight_dtype):
        """The ``dsa_decode_state`` program's constants (fp32 norm weights -- bf16 checkpoint values, so exactly the
        composite's bf16 weights --, the ape block, the selectors) and the folded absorb weights
        (``fold_absorb_weights``): q_lat = q_resid @ (Wqb[h] Wkb1[h]) per head -> packed [1,1,B,H*L] (the flash-MLA
        decode layout after one free view + transpose), y = o_lat_packed @ stack_h(Wkb2[h]^T Wo[h]) (one matmul).
        Shared by the dense fused block (``GLM53_DSA_FUSED=1``) and the state block of the traced sparse decode
        (``GLM53_DSA_DECODE_STATE``)."""
        cfg, idx, up = self.cfg, self.indexer, self.mesh.upload
        Lq, L = cfg.q_lora_rank, self.L
        f32 = lambda t: up(t.float().reshape(1, 1, 1, -1), ttnn.float32)
        self.qa_norm_w32 = f32(sd["q_a_layernorm.weight"])
        self.kva_norm_w32 = f32(sd["kv_a_layernorm.weight"])
        self.knorm_w32 = f32(sd["indexer.k_norm.weight"])
        self.knorm_b32 = f32(sd["indexer.k_norm.bias"])
        ape = torch.full((1, 1, TILE, idx.D), -1e9)
        ape[0, 0, : idx.P] = sd["indexer.index_kpool_compress_ape"].float().reshape(idx.P, idx.D)
        self.ape_block = up(ape, ttnn.float32)
        self.selectors = up(selector_tiles(), ttnn.bfloat16)
        b = self.stem_bounds
        self.state_op = DsaDecodeState(
            q_lora_rank=Lq,
            latent_dim=L,
            index_dim=idx.D,
            pool_size=idx.P,
            pool_blocks=1,  # set from the cache at call time
            qa_offset=b["qa"][0],
            kva_offset=b["kva"][0],
            k_offset=b["k"][0],
            g_offset=b["g"][0],
            eps_rms=cfg.rms_norm_eps,
            eps_ln=1e-6,
        )
        w_qb1, w_o2 = fold_absorb_weights(
            sd["q_b_proj.weight"].t(), wkb1, wkb2, sd["o_proj.weight"].t()
        )  # [Lq, H*L], [H*L, hidden] (H = all heads; head-major, sharded over the columns below)
        self.w_qb1 = up(w_qb1, weight_dtype, cols=-1)
        self.w_o2 = up(w_o2, weight_dtype, cols=-2)
        self.lin_qb1 = DecodeLinear(self.device, self.w_qb1)
        self.lin_o2 = DecodeLinear(self.device, self.w_o2)

    def set_stage_timer(self, timer: StageTimer | None):
        """Install (or remove, ``None``) the per-stage timer on this layer and its indexer."""
        self.stage_timer = timer
        self.indexer.stage_timer = timer

    def _mark(self, name: str):
        if self.stage_timer is not None:
            self.stage_timer.mark(name)

    def _decode_sdpa_config(self, batch: int) -> ttnn.SDPAProgramConfig:
        """Flash MLA decode splits each user's heads over cores and tree-reduces the partial results;
        its c_19 staging CB is sized (cores_per_user - 1) * (out + 2) fp32 tiles: 36 tiles * 4 KB * 11
        = 1.6 MB at 12 cores/user with the 512-wide latent. 4 cores/user keeps the op at ~1.2 MB."""
        if batch not in self._decode_cfgs:
            want = min(self.full_grid.x * self.full_grid.y, CORES_PER_USER_DECODE * batch)
            x = min(self.full_grid.x, want)
            y = max(1, min(self.full_grid.y, want // x))
            self._decode_cfgs[batch] = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(x, y),
                q_chunk_size=0,
                k_chunk_size=32,
                exp_approx_mode=False,
            )
        return self._decode_cfgs[batch]

    # ---------------------------------------------------------------- shared stems
    def _stem(self, lin: DecodeLinear, x: ttnn.Tensor, row_block: int = PREFILL_ROW_BLOCK) -> ttnn.Tensor:
        """One projection of the prefill / decode stems: decode-shaped inputs (M <= 32) take the 1D mcast config,
        prefill rows the M-independent ``prefill_linear`` (fixed K blocking in row blocks: a row's result is then
        bit-identical whatever the chunk / slab length it is computed in -- the default dispatch picks its K
        blocking per M and gives one-ulp differences between a 4096- and an 8192-row call, which flip near-tie
        top-k selections; GLM53_PREFILL_STEMS=default keeps the default dispatch for an A/B)."""
        if x.shape[-2] <= TILE or PREFILL_STEMS != "fixed":
            return lin(x, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)
        return prefill_linear(
            x, lin.w, self.full_grid, row_block=row_block, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2
        )

    def _q_resid(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x [1,1,T,4096] -> q_resid [1,1,T,1536] bf16 (q_a -> rmsnorm)."""
        qa = self._stem(self.lin_qa, x)
        return ttnn.rms_norm(qa, epsilon=self.cfg.rms_norm_eps, weight=self.qa_norm, compute_kernel_config=self.hifi4)

    def _q(self, q_resid: ttnn.Tensor) -> ttnn.Tensor:
        """q_resid [1,1,T,1536] -> q [1,1,T,H*256] bf16 (q_b)."""
        return self._stem(self.lin_qb, q_resid, row_block=PREFILL_ROW_BLOCK // 2)  # 4096 wide: 16 tiles per core

    def _latent(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x [1,1,T,4096] -> c [1,1,T,512] bf16 (kv_a -> rmsnorm)."""
        c = self._stem(self.lin_kva, x)
        return ttnn.rms_norm(c, epsilon=self.cfg.rms_norm_eps, weight=self.kva_norm, compute_kernel_config=self.hifi4)

    def _absorb_bmm_config(self, N: int, in0_block_w: int):
        """Decode-shaped per-head batched matmul: one core per head (H <= 24 blocks on a 12x2 grid), the
        whole [32 x N] output block per core. tests/test_bmm_sweep.py (chip 0): absorb_q 65.8 -> 24.1 us,
        absorb_o 106.4 -> 23.2 us (default dispatch puts them on 16 / 8 cores)."""
        return ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(12, 2),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=N // 32,
        )

    def _absorb_q(self, q: ttnn.Tensor) -> ttnn.Tensor:
        """q [1,1,T,H*Dq] -> q_lat [1,H,T,L] bf16 (per-head q @ Wkb1)."""
        T = q.shape[-2]
        qh = ttnn.transpose(ttnn.reshape(q, [1, T, self.H, self.Dq]), 1, 2)  # [1,H,T,Dq]
        kw = {"program_config": self._absorb_bmm_config(self.L, 1)} if T <= 32 and self.H <= 24 else {}
        q_lat = ttnn.matmul(qh, self.w_absorb_q, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2, **kw)
        ttnn.deallocate(qh)
        return q_lat

    def _o_lat_to_heads(self, o_lat: ttnn.Tensor) -> ttnn.Tensor:
        """o_lat [1,H,T,L] -> concat-heads v-space [1,1,T,H*Dv] (per-head @ Wkb2^T)."""
        T = o_lat.shape[-2]
        kw = {"program_config": self._absorb_bmm_config(self.Dv, 4)} if T <= 32 and self.H <= 24 else {}
        o = ttnn.matmul(
            o_lat, self.w_absorb_o, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2, **kw
        )  # [1,H,T,Dv]
        return ttnn.reshape(ttnn.transpose(o, 1, 2), [1, 1, T, self.H * self.Dv])

    def _o_proj(self, o_heads: ttnn.Tensor) -> ttnn.Tensor:
        y = self.lin_o(o_heads, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)
        return self.mesh.all_reduce_cols(y)  # TP partial -> replicated over the row's columns

    def _cu_window(self, n_keys: int) -> ttnn.Tensor:
        """``cu_window_seqlens`` = [0, n_keys] for the windowed (prefix-aware) causal SDPA: one window
        over the whole prefix. Chunk geometry, not data: built once per key length and cached."""
        if n_keys not in self._cu_windows:
            self._cu_windows[n_keys] = ttnn.from_torch(
                torch.tensor([0, n_keys], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
            )
        return self._cu_windows[n_keys]

    def _dense_attention(
        self,
        q: ttnn.Tensor,
        c: ttnn.Tensor,
        prefix: ttnn.Tensor | None = None,
        start: int = 0,
        *,
        q_offset: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """Causal attention of q [1,1,T,H*Dq] (rows at absolute positions ``start + i``) over the latent
        of the full prefix: ``prefix`` [1,1,start,L] (the user's cached rows before this chunk, None
        when start == 0) followed by this chunk's c [1,1,T_c,L] (T_c >= T: the chunk's latent rows up to
        the last key the queries need); K/V are materialized from the latent. With a prefix the SDPA
        runs in windowed mode (one window over all keys, Q offset ``start``), which builds the causal
        mask on device at the global positions and is bit-identical to the plain causal kernel's rows
        [start, start + T). ``q_offset``: the per-device form of the Q offset (a 1-element int32
        ROW_MAJOR device tensor, row-sharded over a sequence-parallel mesh: the slab-local prefill), which
        overrides ``start`` for the mask and forces the windowed mode. Returns concat-heads v-space
        [1,1,T,H*Dv]."""
        T = q.shape[-2]
        H, Dq, Dv = self.H, self.Dq, self.Dv
        if prefix is not None:
            c = ttnn.concat([prefix, c], dim=-2)  # [1,1,start+T_c,L]
        S = c.shape[-2]
        k = ttnn.linear(c, self.w_k_all, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,S,H*Dq]
        v = ttnn.linear(c, self.w_v_all, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,S,H*Dv]
        if prefix is not None:
            ttnn.deallocate(c)
        qh = ttnn.transpose(ttnn.reshape(q, [1, T, H, Dq]), 1, 2)
        kh = ttnn.transpose(ttnn.reshape(k, [1, S, H, Dq]), 1, 2)
        vh = ttnn.transpose(ttnn.reshape(v, [1, S, H, Dv]), 1, 2)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        windowed = {}
        if start or q_offset is not None:
            windowed = {"cu_window_seqlens": self._cu_window(S), "windowed_q_token_offset": start}
            if q_offset is not None:
                windowed["windowed_q_token_offset_tensor"] = q_offset
        o = ttnn.transformer.scaled_dot_product_attention(
            qh,
            kh,
            vh,
            is_causal=True,
            scale=self.scale,
            program_config=self.sdpa_prefill_cfg,
            compute_kernel_config=self.sdpa_compute,
            **windowed,
        )  # [1,H,T,Dv]
        ttnn.deallocate(qh)
        ttnn.deallocate(kh)
        ttnn.deallocate(vh)
        return ttnn.reshape(ttnn.transpose(o, 1, 2), [1, 1, T, H * Dv])

    # --------------------------------------------------------------------- prefill
    def prefill_state(self, x: ttnn.Tensor, cache: TtLatentCache, user: int, seq_len: int, start: int = 0):
        """The state half of ``prefill_forward`` without the attention output: the chunk's latent goes into
        the user's pages [start, start + T_pad) and the pooled-key index is maintained for positions
        [start, start + seq_len) (same contract as ``prefill_forward``: ``start % block_size == 0``,
        chunks in order). Used to seed long contexts synthetically (the 1 M-position cache test) and
        for prefix-state construction where the chunk's own attention output is not needed."""
        if start % cache.block_size:
            raise ValueError(f"start {start} must be a multiple of the cache block size {cache.block_size}")
        c = self._latent(x)
        cache.fill(c, user, start=start)
        cache.prefix_sync(user, start, c)  # the incremental prefix copy follows the pages (or forgets them)
        ttnn.deallocate(c)
        self.indexer.maintain_prefill(x, cache.index, user, seq_len, start=start)

    def prefill_forward(
        self,
        x: ttnn.Tensor,
        cache: TtLatentCache,
        user: int,
        seq_len: int,
        start: int = 0,
        slot=None,
        ring_sel: ttnn.Tensor | None = None,
        *,
        slab: bool = False,
    ) -> ttnn.Tensor:
        """One user, one chunk: x row i is the token at absolute position ``start + i``
        (``start % block_size == 0``; ``start == 0`` = a new prompt). x [1,1,T_pad,4096] with
        T_pad % block_size == 0 and the logical ``seq_len`` <= T_pad; returns [1,1,T_pad,4096] (rows
        past seq_len are garbage). Fills the user's latent pages [start, start + T_pad) and maintains
        the pooled-key index for positions [start, start + seq_len) (the ring must hold the previous
        chunk's tail rows, i.e. chunks of a prompt are prefilled in order). Rows at absolute positions
        < 2048 run the dense causal attention over the full prefix (K/V from the cached latent rows +
        this chunk's); rows from 2048 on use the indexer + sparse_sdpa, exactly as the single-shot path.
        ``slot`` + ``ring_sel`` (the traced prefill, tt/trace_prefill.py): the user slot and the ring
        rows come from device tensors and the whole padded chunk is indexed (dense range only).
        ``slab`` (multi-row mesh): ``x`` is THIS mesh row's slab [1,1,T_pad/rows,4096] of the chunk (row r
        holds chunk rows [r*T_pad/rows, (r+1)*T_pad/rows)) and the result is the slab's output only
        (``_prefill_slab``: the sequence-parallel form, no gathered hidden, no reduce-scatter)."""
        if slot is not None:
            cache.release_prefix()  # the device-slot fill names its user on device: no arena can follow it
        if slab and self.mesh.rows > 1:
            return self._prefill_slab(x, cache, user, seq_len, start, slot=slot, ring_sel=ring_sel)
        T = x.shape[-2]
        if start % cache.block_size:
            raise ValueError(f"start {start} must be a multiple of the cache block size {cache.block_size}")
        end = start + seq_len
        split = min(T, max(0, self.sparse_row_start - start))  # chunk rows [0, split) are dense
        if slot is not None and (start or split != T):
            raise ValueError(
                f"the traced prefill covers the dense range only (start == 0, T <= {self.sparse_row_start})"
            )
        if self.stage_timer is not None:
            self.stage_timer.start()
        q_resid = self._q_resid(x)
        q = self._q(q_resid)  # [1,1,T,H*Dq]
        c = self._latent(x)  # [1,1,T,L]
        self._mark("prefill_stems")
        cache.fill(c, user, start=start, slot=slot)
        self._mark("cache_fill")
        self.indexer.maintain_prefill(x, cache.index, user, seq_len, start=start, slot=slot, ring_sel=ring_sel)
        self._mark("index_maintain")
        prefix = self._prefix_rows(cache, user, start) if start and split else None  # [1,1,start,L]
        if split == T:
            ttnn.deallocate(q_resid)
            o = self._dense_attention(q, c, prefix, start)
            ttnn.deallocate(q)
            ttnn.deallocate(c)
            if prefix is not None:
                ttnn.deallocate(prefix)
            self._mark("dense_attention")
            y = self._o_proj(o)
            self._mark("out_proj")
            return y
        # dense rows [0, split) (absolute positions [start, 2048))
        o_dense = None
        if split:
            o_dense = self._dense_attention(
                ttnn.slice(q, [0, 0, 0, 0], [1, 1, split, self.H * self.Dq]),
                ttnn.slice(c, [0, 0, 0, 0], [1, 1, split, self.L]),
                prefix,
                start,
            )
            if prefix is not None:
                ttnn.deallocate(prefix)
            self._mark("dense_attention")
        # sparse rows [split, T) (absolute positions from max(start, 2048)) in score sub-blocks of
        # GLM53_PREFILL_QBLOCK rows and attention blocks of GLM53_PREFILL_ABLOCK rows (``_sparse_rows``); one
        # whole-prefix ROW_MAJOR kv copy (the sparse_sdpa contract) serves every block of the chunk: rebuilt
        # from the pages, or the layer's arena appended with this chunk's latent (GLM53_PREFIX_INCREMENTAL)
        kv_rm, kv_owned = cache.prefix_rows_rm(user, start, T, chunk_c=c)
        ttnn.deallocate(c)
        self._mark("kv_copy")
        o_sparse = self._sparse_rows(x, q_resid, q, cache, user, kv_rm, start=start, end=end, r0=split, r1=T)
        ttnn.deallocate(q_resid)
        ttnn.deallocate(q)
        if kv_owned:
            ttnn.deallocate(kv_rm)
        if seq_len % cache.block_size:  # a ragged chunk is the prompt's last: the arena has nothing more to serve
            cache.release_prefix(user)
        if o_dense is None:  # the whole chunk lies in the sparse region
            y = self._o_proj(o_sparse)
            self._mark("out_proj")
            return y
        o = ttnn.concat([o_dense, o_sparse], dim=-2)  # [1,1,T,H*Dv]
        ttnn.deallocate(o_dense)
        ttnn.deallocate(o_sparse)
        y = self._o_proj(o)
        self._mark("out_proj")
        return y

    # ------------------------------------------------------- slab-local sequence-parallel prefill
    def _prefill_slab(
        self, x: ttnn.Tensor, cache: TtLatentCache, user: int, seq_len: int, start: int, slot=None, ring_sel=None
    ) -> ttnn.Tensor:
        """Sequence-parallel prefill of one chunk over the mesh rows, attention computed per slab: the chunk's
        T rows are ``rows`` consecutive slabs, mesh row r holds slab r as ``x`` [1,1,T_loc,4096] (T_loc = T/rows)
        and returns the attention output of ITS slab rows only [1,1,T_loc,4096] -- the rows ``prefill_forward``
        would return for them from the gathered chunk, with no reduce-scatter and no 1/rows scaling. Every
        row still receives the whole chunk's latent and index stems (all-gathered over the rows: [1,1,T,512]
        + 2 x [1,1,T,128] bf16 instead of the hidden [1,1,T,4096]), so the replicated latent cache and
        pooled-key index are filled exactly as in the single-row path. SPMD: every row runs the same ops
        with the same shapes; what differs per row (the dense SDPA's Q offset, the scorer's causal bounds,
        the tail ids, the dense/sparse row select) lives in row-sharded constant tensors built from the
        chunk geometry (host ints, cached).

        Row classes are by absolute position (dense below ``sparse_row_start``). A chunk entirely on one side
        runs only that path per slab (exact, 1/rows of the v1 attention work). A mixed chunk (start < 2048 <
        start + T) runs a uniform superset on every row: the dense SDPA over the first n_d = min(T_loc, split)
        slab rows (an all-sparse row gets the last dense block's offset, so its throw-away rows cost one dense
        block, never the whole prefix) and the sparse path over all T_loc slab rows (dense-position rows are
        clamped to 2048 so they select a valid full pool prefix); a per-row 0/1 mask then keeps the dense
        result for the rows below 2048. ``on_indices(row_starts, indices)`` receives the tuple of the rows'
        query starts (true positions) instead of one start."""
        mesh, rows = self.mesh, self.mesh.rows
        T_loc = x.shape[-2]
        T = T_loc * rows
        if start % cache.block_size:
            raise ValueError(f"start {start} must be a multiple of the cache block size {cache.block_size}")
        end = start + seq_len
        split = min(T, max(0, self.sparse_row_start - start))  # chunk rows [0, split) are dense
        dense = [min(T_loc, max(0, split - r * T_loc)) for r in range(rows)]  # dense rows of slab r
        n_d = max(dense)  # dense SDPA rows per slab (uniform over the rows) = min(T_loc, split)
        if self.stage_timer is not None:
            self.stage_timer.start()
        q_resid = self._q_resid(x)
        q = self._q(q_resid)  # [1,1,T_loc,H*Dq]
        c_loc = self._latent(x)  # [1,1,T_loc,L]
        k_loc, g_loc = self.indexer.keys_and_gates(x)  # [1,1,T_loc,D]
        self._mark("prefill_stems")
        c = mesh.all_gather_rows(c_loc, dim=2)  # [1,1,T,L]: the chunk's latent on every row (slabs in row order)
        k_idx = mesh.all_gather_rows(k_loc, dim=2)
        g_idx = mesh.all_gather_rows(g_loc, dim=2)
        for t in (c_loc, k_loc, g_loc):
            ttnn.deallocate(t)
        self._mark("slab_gather")
        cache.fill(c, user, start=start, slot=slot)
        self._mark("cache_fill")
        self.indexer.maintain_prefill(
            None, cache.index, user, seq_len, start=start, slot=slot, ring_sel=ring_sel, kg=(k_idx, g_idx)
        )
        self._mark("index_maintain")
        o_dense = None
        if n_d:
            # keys: the cached prefix [0, start) ++ the chunk's latent up to the last key any slab's dense (or
            # superset) rows need; the per-row Q offset tensor locates the slab's rows among them. An all-sparse
            # row of a mixed chunk is pointed at the last dense block (its rows are discarded by the select).
            S_c = min(T, split + n_d)
            prefix = self._prefix_rows(cache, user, start) if start else None
            offsets = tuple(start + (r * T_loc if dense[r] else split - n_d) for r in range(rows))
            q_d = q if n_d == T_loc else ttnn.slice(q, [0, 0, 0, 0], [1, 1, n_d, self.H * self.Dq])
            c_k = c if S_c == T else ttnn.slice(c, [0, 0, 0, 0], [1, 1, S_c, self.L])
            o_dense = self._dense_attention(q_d, c_k, prefix, start, q_offset=self._row_offsets(offsets))
            for t in (prefix, None if q_d is q else q_d, None if c_k is c else c_k):
                if t is not None:
                    ttnn.deallocate(t)
            self._mark("dense_attention")
        if split >= T:  # every slab row is dense
            for t in (q_resid, q, c):
                ttnn.deallocate(t)
            y = self._o_proj(o_dense)
            self._mark("out_proj")
            return y
        # sparse rows: all T_loc slab rows in score sub-blocks / attention blocks (``_sparse_rows``); on a mixed
        # chunk the rows below 2048 are scored at the clamped position 2048 (a valid full prefix) and dropped by
        # the select. The whole-prefix kv: rebuilt from the pages, or the layer's arena appended with the chunk's
        # gathered latent ``c`` (every row holds the whole chunk; GLM53_PREFIX_INCREMENTAL)
        kv_rm, kv_owned = cache.prefix_rows_rm(user, start, T, chunk_c=c)
        ttnn.deallocate(c)
        self._mark("kv_copy")
        o_sparse = self._sparse_rows(
            x,
            q_resid,
            q,
            cache,
            user,
            kv_rm,
            start=start,
            end=end,
            r0=0,
            r1=T_loc,
            row_starts_of=lambda q0: tuple(start + r * T_loc + q0 for r in range(rows)),
            pos_floor=self.sparse_row_start,
        )
        ttnn.deallocate(q_resid)
        ttnn.deallocate(q)
        if kv_owned:
            ttnn.deallocate(kv_rm)
        if seq_len % cache.block_size:  # a ragged chunk is the prompt's last: the arena has nothing more to serve
            cache.release_prefix(user)
        if o_dense is None:  # the whole chunk lies in the sparse region
            y = self._o_proj(o_sparse)
            self._mark("out_proj")
            return y
        o = self._select_rows(o_dense, o_sparse, dense)
        ttnn.deallocate(o_dense)
        ttnn.deallocate(o_sparse)
        y = self._o_proj(o)
        ttnn.deallocate(o)
        self._mark("out_proj")
        return y

    def _prefix_rows(self, cache: TtLatentCache, user: int, start: int) -> ttnn.Tensor:
        """The cached latent rows [0, start) of ``user`` as a fresh TILE [1,1,start,L] tensor on every column (the
        dense rows' K/V prefix). With the heads2users layout only the owner column holds them: its rows are
        all-gathered over the columns on dim 0 ([cols,1,start,L]) and the owner's entry sliced out (start < 2048
        rows: a chunk with dense rows starts below the sparse boundary)."""
        rows = cache.user_rows(user, start)
        owner = cache.owner_of(user)
        if owner is None:
            return rows
        g = self.mesh.all_gather_cols(rows, dim=0)  # [cols,1,start,L]: entry c = column c's slot rows
        ttnn.deallocate(rows)
        prefix = ttnn.slice(g, [owner, 0, 0, 0], [owner + 1, 1, start, self.L])
        ttnn.deallocate(g)
        return prefix

    def _sparse_rows(
        self,
        x: ttnn.Tensor,
        q_resid: ttnn.Tensor,
        q: ttnn.Tensor,
        cache: TtLatentCache,
        user: int,
        kv_rm: ttnn.Tensor,
        *,
        start: int,
        end: int,
        r0: int,
        r1: int,
        row_starts_of=None,
        pos_floor: int = 0,
    ) -> ttnn.Tensor:
        """The sparse-path attention output [1,1,r1-r0,H*Dv] of chunk (or slab) rows [r0, r1) of ``x`` /
        ``q_resid`` / ``q`` (all [1,1,T,*] of the chunk; ``x`` is the normalized input, ``q`` the per-head queries)
        over the whole-prefix ROW_MAJOR latent copy ``kv_rm``, in the two block sizes of ``prefill_blocks``:

        * the index queries ``q_i`` / ``w_i`` of all the rows come from ONE pair of stems (``prefill_linear``'s
          fixed K blocking makes a row's value independent of the call's row count, the same invariant the
          chunked-prefill equality rests on), then every score sub-block of ``GLM53_PREFILL_QBLOCK`` rows slices
          them and runs the scorer + the bounded top-k only (``TtDSAIndexer.select_pools``): the [rows, pools] bf16
          score block is the memory bound of the loop and lives inside that call;
        * every attention block of ``GLM53_PREFILL_ABLOCK`` rows (= several sub-blocks) then expands the
          concatenated pool ids once (``expand_pools``: 6 small launches per block instead of per sub-block, the
          tail ids / sentinels of the block's rows), absorbs its queries once and runs one ``_sparse_attention``
          (one column exchange each way, one sparse_sdpa) and one output projection of the latent: the attention
          work is linear in the rows, so the batching removes launches and fixed costs only, not FLOPs.

        ``row_starts_of(q0)`` (the slab-local form): the per-mesh-row query starts of chunk row ``q0``, handed to
        the selection and the ``on_indices`` hook (``(row_starts, indices)`` per attention block; the single-row
        form hands ``(start + a0, indices)``); ``pos_floor`` clamps the dense-position rows of a mixed chunk. The
        custom scorer's query layout, when a sub-block needs it (``prefill_rows_form`` / GLM53_PREFILL_SCORER=custom),
        is a transpose of the sub-block's slice (the same values).

        Memory: the attention block's transient peak is ~4 x rows x H_total x L bf16 (the gathered 64-head queries
        in TILE and ROW_MAJOR, the sparse_sdpa output in both: 274 MB at 2048 rows), against the score sub-block's
        [rows, pools] block in TILE and ROW_MAJOR (268 MB at 256 rows x 262 K pools, 536 MB at 512): at the 32-user
        1 M point (QBLOCK 256) the default ABLOCK 2048 adds ~6 MB to the chunk's peak; ABLOCK 1024 halves it."""
        idx = self.indexer
        qb, ab = prefill_blocks(cache.rows_per_user)
        n = r1 - r0
        T = x.shape[-2]
        whole = (r0, r1) == (0, T)
        x_s = x if whole else ttnn.slice(x, [0, 0, r0, 0], [1, 1, r1, x.shape[-1]])
        qr_s = q_resid if whole else ttnn.slice(q_resid, [0, 0, r0, 0], [1, 1, r1, q_resid.shape[-1]])
        q_i, w_i = idx.queries_and_weights(x_s, qr_s)  # [1,Hi,n,D], [1,1,n,Hi]: once for every sub-block
        if not whole:
            ttnn.deallocate(x_s)
            ttnn.deallocate(qr_s)
        self._mark("index_queries")
        slot = cache.index.prefill_slot(user)
        n_pools_valid = end // idx.P
        o_parts = []
        for a0, a1, subs in sparse_block_plan(r0, r1, qb, ab):
            pools = []
            for q0, q1 in subs:
                one = (q0, q1) == (r0, r1)
                q_b = q_i if one else ttnn.slice(q_i, [0, 0, q0 - r0, 0], [1, idx.Hi, q1 - r0, idx.D])
                w_b = w_i if one else ttnn.slice(w_i, [0, 0, q0 - r0, 0], [1, 1, q1 - r0, idx.Hi])
                row_starts = None if row_starts_of is None else row_starts_of(q0)
                if row_starts is None:
                    # the scorer form of this sub-block: the custom scorer's query layout (the knob, or a multi-row
                    # superset that would start past the pooled-key allocation)
                    rows_form = idx.prefill_rows_form(
                        cache.index, query_start=start + q0, T=q1 - q0, num_pools_valid=n_pools_valid
                    )
                else:
                    # the shared ``indexer_score_dsa`` with the exact per-row bounded top-k (``row_starts``), or the
                    # custom scorer's layout under GLM53_PREFILL_SCORER=custom (per-row bounds inside the scorer)
                    rows_form = idx.prefill_scorer_custom
                if rows_form:
                    q_r = ttnn.transpose(q_b, 1, 2)  # [1,T,Hi,D]: query t's heads as the rows of its tile row
                    if q_b is not q_i:
                        ttnn.deallocate(q_b)
                    q_b = q_r
                pools.append(
                    idx.select_pools(
                        q_b,
                        w_b,
                        cache.index,
                        slot,
                        query_start=start + q0,
                        num_pools_valid=n_pools_valid,
                        q_rows_form=rows_form,
                        row_starts=row_starts,
                        pos_floor=pos_floor,
                    )
                )  # [1,1,q1-q0,512] ranked pool ids
                if q_b is not q_i:
                    ttnn.deallocate(q_b)
                if w_b is not w_i:
                    ttnn.deallocate(w_b)
            if len(pools) == 1:
                top_p = pools[0]
            else:
                top_p = ttnn.concat(pools, dim=-2)  # [1,1,a1-a0,512]: the block's rows in order
                for p in pools:
                    ttnn.deallocate(p)
            row_starts = None if row_starts_of is None else row_starts_of(a0)
            indices = idx.expand_pools(top_p, query_start=start + a0, row_starts=row_starts, pos_floor=pos_floor)
            if self.on_indices is not None:
                self.on_indices(start + a0 if row_starts is None else row_starts, indices)
            q_a = q if (a0, a1) == (0, T) else ttnn.slice(q, [0, 0, a0, 0], [1, 1, a1, self.H * self.Dq])
            q_lat = self._absorb_q(q_a)
            if q_a is not q:
                ttnn.deallocate(q_a)
            self._mark("absorb_q")
            o_lat = self._sparse_attention(
                q_lat, kv_rm, indices, owner=cache.owner_of(user)
            )  # [1,H,a1-a0,L] TILE (consumes q_lat, indices)
            o_parts.append(self._o_lat_to_heads(o_lat))  # [1,1,a1-a0,H*Dv]
            ttnn.deallocate(o_lat)
            self._mark("sparse_out")
        ttnn.deallocate(q_i)
        ttnn.deallocate(w_i)
        if len(o_parts) == 1:
            return o_parts[0]
        o_sparse = ttnn.concat(o_parts, dim=-2)  # [1,1,n,H*Dv]
        for part in o_parts:
            ttnn.deallocate(part)
        return o_sparse

    def _sparse_attention(
        self, q_lat: ttnn.Tensor, kv_rm: ttnn.Tensor, indices: ttnn.Tensor, *, owner: int | None = None
    ) -> ttnn.Tensor:
        """``sparse_sdpa`` of the absorbed queries ``q_lat`` [1,H,n,L] TILE (H = this chip's heads) over the
        ROW_MAJOR prefix copy ``kv_rm`` with the index rows ``indices`` [1,1,n,W] (replicated over the columns):
        returns o_lat [1,H,n,L] TILE; ``q_lat`` and ``indices`` are consumed. The op wants H % 32 == 0; a
        multi-column mesh whose head shard is thinner (16 heads at TP4) used to pad the heads to 32 (2x the
        attention work). As ``models/demos/deepseek_v3_d_p`` ``_sparse_mla`` does, one all-to-all over the
        columns instead hands column c EVERY head of sequence quarter c ([1,H_total,n/cols,L]), the index rows
        are partitioned alike (a per-device slice, no traffic), the attention runs on full heads over a quarter
        of the rows, and the inverse exchange restores the head shard for the per-head output projection.
        Needs n % (32 * cols) == 0 (tile-aligned quarters); other sub-blocks, single columns and tile-aligned
        head shards keep the direct form. ``GLM53_SPARSE_A2A=0`` keeps the padded form everywhere (A/B).

        ``owner`` (the heads2users prefill of a user-partitioned cache, doc/mesh/sparse_1m.md section 7): only
        column ``owner`` holds the user's cache (``kv_rm`` and ``indices`` are that column's own slot's; the other
        columns pass their own slot's, same shapes). The heads are all-gathered over the columns ([1,H_total,n,L]
        on every column), every column attends all 64 heads over its own rows (the owner's result is the real
        one), and the inverse all-to-all splits the heads back by column and stacks the sources on dim 0
        ([cols,H,n,L]); entry ``owner`` is the owner's output for this column's head shard."""
        H, n = q_lat.shape[1], q_lat.shape[2]
        cols = self.mesh.cols
        if owner is not None and cols > 1:
            q_full = self.mesh.all_gather_cols(q_lat, dim=1)  # [1,H_total,n,L]
            ttnn.deallocate(q_lat)
            q_rm = ttnn.to_layout(q_full, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(q_full)
            self._mark("gather_heads")
            o_rm = ttnn.transformer.sparse_sdpa(
                q_rm,
                kv_rm,
                indices,
                v_dim=self.L,
                kv_format=ttnn.transformer.SparseKVFormat.BF16,
                scale=self.scale,
                k_chunk_size=128,
            )  # [1,H_total,n,L] row-major
            self._mark("sparse_sdpa")
            ttnn.deallocate(q_rm)
            ttnn.deallocate(indices)
            o = ttnn.to_layout(o_rm, ttnn.TILE_LAYOUT)
            ttnn.deallocate(o_rm)
            o_x = self.mesh.all_to_all_cols(o, in_dim=1, out_dim=0)  # [cols,H,n,L]: entry s = column s's result
            ttnn.deallocate(o)
            o_lat = ttnn.slice(o_x, [owner, 0, 0, 0], [owner + 1, H, n, self.L])  # the owner's, my head shard
            ttnn.deallocate(o_x)
            self._mark("a2a_owner_to_heads")
            return o_lat
        a2a = cols > 1 and H % TILE != 0 and n % (TILE * cols) == 0 and os.environ.get("GLM53_SPARSE_A2A", "1") == "1"
        if a2a:
            q_x = self.mesh.all_to_all_cols(q_lat, in_dim=2, out_dim=1)  # [1,H*cols,n/cols,L]
            ttnn.deallocate(q_lat)
            q_rm = ttnn.to_layout(q_x, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(q_x)
            idx = ttnn.mesh_partition(indices, 2, cluster_axis=TP_AXIS)  # [1,1,n/cols,W]: this column's quarter
            ttnn.deallocate(indices)
            self._mark("a2a_heads_to_seq")
        else:
            q_rm0 = ttnn.to_layout(q_lat, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(q_lat)
            q_rm = self._pad_heads_rm(q_rm0)
            if q_rm is not q_rm0:
                ttnn.deallocate(q_rm0)
            idx = indices
        o_rm = ttnn.transformer.sparse_sdpa(
            q_rm,
            kv_rm,
            idx,
            v_dim=self.L,
            kv_format=ttnn.transformer.SparseKVFormat.BF16,
            scale=self.scale,
            k_chunk_size=128,
        )  # [1,H',n',L] row-major
        self._mark("sparse_sdpa")
        ttnn.deallocate(q_rm)
        ttnn.deallocate(idx)
        o = ttnn.to_layout(o_rm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(o_rm)
        if a2a:
            o_lat = self.mesh.all_to_all_cols(o, in_dim=1, out_dim=2)  # [1,H,n,L]: the head shard back
            ttnn.deallocate(o)
            self._mark("a2a_seq_to_heads")
            return o_lat
        o_lat = self._unpad_heads(o)
        if o_lat is not o:
            ttnn.deallocate(o)
        return o_lat

    def _row_offsets(self, offsets: tuple[int, ...]) -> ttnn.Tensor:
        """The windowed SDPA's per-device Q offset: mesh row r reads ``offsets[r]`` (a [1,1] int32 ROW_MAJOR
        tensor per chip, row-sharded, replicated over the columns; cached per geometry)."""
        key = ("qoff", offsets)
        if key not in self._slab_consts:
            t = torch.tensor(offsets, dtype=torch.int32).reshape(len(offsets), 1)
            self._slab_consts[key] = self.mesh.upload(t, ttnn.int32, rows=0, layout=ttnn.ROW_MAJOR_LAYOUT)
        return self._slab_consts[key]

    def _select_rows(self, o_dense: ttnn.Tensor, o_sparse: ttnn.Tensor, dense: list[int]) -> ttnn.Tensor:
        """Mixed chunk: slab rows [0, dense[r]) of mesh row r come from ``o_dense`` [1,1,n_d,N], the rest from
        ``o_sparse`` [1,1,T_loc,N] (n_d <= T_loc). The per-row 0/1 column mask [1,1,n_d,1] is a row-sharded
        constant (cached per geometry); rows at and beyond n_d are sparse on every row."""
        n_d, N = o_dense.shape[-2], o_dense.shape[-1]
        T_loc = o_sparse.shape[-2]
        key = ("select", n_d, tuple(dense))
        if key not in self._slab_consts:
            m = torch.zeros(len(dense), 1, n_d, 1)
            for r, d in enumerate(dense):
                m[r, 0, :d] = 1.0
            self._slab_consts[key] = self.mesh.upload(m, ttnn.bfloat16, rows=0)
        mask = self._slab_consts[key]
        head_sparse = o_sparse if n_d == T_loc else ttnn.slice(o_sparse, [0, 0, 0, 0], [1, 1, n_d, N])
        head = ttnn.where(mask, o_dense, head_sparse)
        if n_d == T_loc:
            return head
        ttnn.deallocate(head_sparse)
        tail = ttnn.slice(o_sparse, [0, 0, n_d, 0], [1, 1, T_loc, N])
        out = ttnn.concat([head, tail], dim=-2)
        ttnn.deallocate(head)
        ttnn.deallocate(tail)
        return out

    # ---------------------------------------------------------------------- decode
    def decode_forward(
        self,
        x: ttnn.Tensor,
        cache: TtLatentCache,
        positions: ttnn.Tensor,
        pool_ids: ttnn.Tensor,
        ring_slots: ttnn.Tensor,
        host_positions: list[int] | None = None,
        dense_positions: ttnn.Tensor | None = None,
        sparse_meta: SparseDecodeMeta | None = None,
    ) -> ttnn.Tensor:
        """x [1,1,B,4096] (one token per user at position a = positions[b]); ``pool_ids`` = a // 4 and
        ``ring_slots`` = a % 4 ([B] int32 device tensors, host-derived like ``positions``). Appends the
        latent, maintains the pooled-key index, then attends over [0, a] with the dense flash-MLA decode
        for every slot (HF-exact through position 2050, trace-safe). With ``host_positions`` (the same
        values), users at a >= 2051 get their row replaced by the per-user indexer + sparse_sdpa result
        (host chunk geometry, not traced). Without ``host_positions`` a user beyond 2050 is attended
        densely, which is NOT HF semantics: FunctionalDecoder enforces the routing.
        ``dense_positions`` (optional [B] int32, host-derived like ``pool_ids``): the positions the dense
        flash-MLA decode attends up to instead of ``positions``. The dense row of a sparse user is
        discarded, so ``min(a, dense_exact_max_pos)`` per user bounds that op's cost to 2051 keys at
        any context length (it reads the whole prefix otherwise: 1 GiB per layer at 1 M positions) and
        is the identity for every user still on the dense path.
        ``sparse_meta`` (a ``SparseDecodeMeta`` of this step's positions) selects the traced multi-user
        path instead: every user, dense or sparse, is attended over a compact cache gathered from its
        selected pools (``_decode_unified``), with no host geometry and no whole-prefix copy; the dense
        flash-MLA over the real cache and the per-user eager rows are then not run at all. With
        ``decode_state`` ("1" / "latent", the ``GLM53_DSA_DECODE_STATE`` knob) that path runs through
        ``_decode_state_unified`` (the fused state program + the absorb folds); it takes precedence over the
        dense fused block (``GLM53_DSA_FUSED=1``), which ignores ``sparse_meta``."""
        B = x.shape[-2]
        if self.stage_timer is not None:
            self.stage_timer.start()
        if sparse_meta is not None and self.decode_state != "0":
            return self._decode_state_unified(x, cache, positions, pool_ids, ring_slots, sparse_meta)
        if self.fused_decode == "fused":
            return self._fused_decode(x, cache, positions, host_positions, dense_positions)
        stem = self.lin_stem(x, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,B,2336]
        parts = {k: ttnn.slice(stem, [0, 0, 0, a], [1, 1, B, b]) for k, (a, b) in self.stem_bounds.items()}
        ttnn.deallocate(stem)
        q_resid = ttnn.rms_norm(
            parts["qa"], epsilon=self.cfg.rms_norm_eps, weight=self.qa_norm, compute_kernel_config=self.hifi4
        )
        ttnn.deallocate(parts["qa"])
        q = self._q(q_resid)  # [1,1,B,H*Dq]
        c = ttnn.rms_norm(
            parts["kva"], epsilon=self.cfg.rms_norm_eps, weight=self.kva_norm, compute_kernel_config=self.hifi4
        )
        ttnn.deallocate(parts["kva"])
        part = cache.partition  # user-partitioned cache: this chip appends / indexes / attends its own users only
        if part:
            if sparse_meta is None:
                raise NotImplementedError("the user-partitioned cache decodes through the traced sparse path only")
            loc = lambda t: self.mesh.partition_cols(t, 2)  # [1,1,B,*] replicated -> [1,1,B/cols,*] (my users)
            c_loc, k_loc, g_loc = loc(c), loc(parts["k"]), loc(parts["g"])
            for t in (c, parts["k"], parts["g"]):
                ttnn.deallocate(t)
            B_loc = c_loc.shape[-2]
            cache.update(ttnn.reshape(c_loc, [1, B_loc, 1, self.L]), positions)
            ttnn.deallocate(c_loc)
            self.indexer.maintain_decode(x, cache.index, pool_ids, ring_slots, kg=(k_loc, g_loc))
            w_idx = loc(parts["w"])
            ttnn.deallocate(parts["w"])
            q_resid_loc = loc(q_resid)
        else:
            cache.update(ttnn.reshape(c, [1, B, 1, self.L]), positions)
            ttnn.deallocate(c)
            self.indexer.maintain_decode(x, cache.index, pool_ids, ring_slots, kg=(parts["k"], parts["g"]))
            w_idx = parts["w"]  # [1,1,B,Hi] indexer head weights (sparse rows only)
            q_resid_loc = q_resid
        self._mark("stems+append+index_maintain")
        q_lat = self._absorb_q(q)  # [1,H,B,L]
        ttnn.deallocate(q)
        if sparse_meta is not None:
            if part:
                # heads -> users: every column holds its 16 heads of all B users; the owner needs all 64 heads of
                # its B/cols users (the inverse exchange after attention)
                q_t = ttnn.transpose(q_lat, 1, 2)  # [1,B,H_loc,L]
                q_x = self.mesh.all_to_all_cols(q_t, in_dim=1, out_dim=2)  # [1,B/cols,H,L]
                ttnn.deallocate(q_t)
                q_full = ttnn.transpose(q_x, 1, 2)  # [1,H,B/cols,L]
                ttnn.deallocate(q_x)
                self._mark("a2a_heads_to_users")
                oh_full = self._decode_unified(q_resid_loc, q_full, cache, sparse_meta, w_idx)
                ttnn.deallocate(q_full)
                ttnn.deallocate(q_resid_loc)
                o_t = ttnn.transpose(oh_full, 1, 2)  # [1,B/cols,H,L]
                ttnn.deallocate(oh_full)
                o_x = self.mesh.all_to_all_cols(o_t, in_dim=2, out_dim=1)  # [1,B,H_loc,L]
                ttnn.deallocate(o_t)
                oh = ttnn.transpose(o_x, 1, 2)  # [1,H_loc,B,L]
                ttnn.deallocate(o_x)
                self._mark("a2a_users_to_heads")
            else:
                oh = self._decode_unified(q_resid, q_lat, cache, sparse_meta, w_idx)
            ttnn.deallocate(q_resid)
            ttnn.deallocate(q_lat)
            ttnn.deallocate(w_idx)
            y = self._o_proj(self._o_lat_to_heads(oh))
            self._mark("out_proj")
            return y
        q_dec = ttnn.transpose(q_lat, 1, 2)  # [1,B,H,L] (decode layout: batch, padded heads, dim)
        self._mark("absorb_q")
        o_lat = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            q_dec,
            cache.cache,
            head_dim_v=self.L,
            page_table_tensor=cache.page_table,
            cur_pos_tensor=positions if dense_positions is None else dense_positions,
            scale=self.scale,
            program_config=self._decode_sdpa_config(B),
            compute_kernel_config=self.sdpa_compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1,B,H,L]
        ttnn.deallocate(q_dec)
        oh = ttnn.transpose(o_lat, 1, 2)  # [1,H,B,L]
        ttnn.deallocate(o_lat)
        self._mark("dense_mla_decode")
        sparse_users = (
            [u for u, a in enumerate(host_positions) if a > self.dense_exact_max_pos] if host_positions else []
        )
        if sparse_users:
            sparse_rows = self._decode_sparse_rows(x, q_resid, q_lat, cache, host_positions, sparse_users, w=w_idx)
            rows = [
                sparse_rows[u] if u in sparse_rows else ttnn.slice(oh, [0, 0, u, 0], [1, self.H, u + 1, self.L])
                for u in range(B)
            ]
            oh_new = rows[0] if B == 1 else ttnn.concat(rows, dim=-2)
            ttnn.deallocate(oh)
            oh = oh_new
        ttnn.deallocate(q_resid)
        ttnn.deallocate(q_lat)
        ttnn.deallocate(w_idx)
        y = self._o_proj(self._o_lat_to_heads(oh))
        self._mark("out_proj")
        return y

    def _fused_decode(self, x, cache: TtLatentCache, positions, host_positions, dense_positions):
        """The 7-op dense decode block: stem matmul -> ``dsa_decode_state`` (q_a norm, latent append, index ring +
        pool maintenance, all in place) -> folded q_b.Wkb1 matmul -> transpose to the decode head layout (the
        packed rows are a free view of the head-major tiles) -> flash-MLA decode -> transpose -> folded
        Wkb2.o_proj matmul (+ the TP all-reduce). Sparse users (positions > 2050) get
        their head block replaced by the per-user indexer + sparse_sdpa result as in the composite path."""
        B, H, L = x.shape[-2], self.H, self.L
        idx = cache.index
        b = self.stem_bounds
        stem = self.lin_stem(x, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,B,Ws]
        q_resid = self._run_state_op(stem, cache, positions)  # [1,1,B,Lq]
        self._mark("stem+state")
        if DSA_FOLD in ("both", "q"):
            q_lat = self.lin_qb1(q_resid, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,B,H*L]
            # B <= 32: the packed tiled rows ARE the head-major [1,H,B,L] tile sequence (a free view); the decode
            # layout is then one multicore transpose. (A logical reshape would shuffle rows at ~5 GB/s.)
            q_lat_h = ttnn.experimental.view(q_lat, [1, H, B, L]) if B <= TILE else self._heads_major(q_lat, B)
        else:
            q_lat_h = self._absorb_q(self._q(q_resid))  # [1,H,B,L] (composite absorb)
            q_lat = None
        q_dec = ttnn.transpose(q_lat_h, 1, 2)  # [1,B,H,L]
        self._mark("absorb_q")
        o_lat = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            q_dec,
            cache.cache,
            head_dim_v=L,
            page_table_tensor=cache.page_table,
            cur_pos_tensor=positions if dense_positions is None else dense_positions,
            scale=self.scale,
            program_config=self._decode_sdpa_config(B),
            compute_kernel_config=self.sdpa_compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1,B,H,L]
        self._mark("dense_mla_decode")
        sparse_users = (
            [u for u, a in enumerate(host_positions) if a > self.dense_exact_max_pos] if host_positions else []
        )
        if sparse_users:
            w_idx = ttnn.slice(stem, [0, 0, 0, b["w"][0]], [1, 1, B, b["w"][1]])
            sparse_rows = self._decode_sparse_rows(x, q_resid, q_lat_h, cache, host_positions, sparse_users, w=w_idx)
            blocks = [
                ttnn.transpose(sparse_rows[u], 1, 2)
                if u in sparse_rows
                else ttnn.slice(o_lat, [0, u, 0, 0], [1, u + 1, H, L])
                for u in range(B)
            ]  # [1,1,H,L] per user
            o_new = blocks[0] if B == 1 else ttnn.concat(blocks, dim=1)
            ttnn.deallocate(o_lat)
            o_lat = o_new
            ttnn.deallocate(w_idx)
        ttnn.deallocate(q_dec)
        ttnn.deallocate(q_lat if q_lat is not None else q_lat_h)
        ttnn.deallocate(q_resid)
        ttnn.deallocate(stem)
        oh = ttnn.transpose(o_lat, 1, 2)  # [1,H,B,L]
        ttnn.deallocate(o_lat)
        if DSA_FOLD in ("both", "o"):
            # the head-major tiles read as the packed [1,1,B,H*L] rows (free view, B <= 32)
            o_p = ttnn.experimental.view(oh, [1, 1, B, H * L]) if B <= TILE else self._heads_packed(oh, B)
            y = self.lin_o2(o_p, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,B,hidden] (TP partial)
            ttnn.deallocate(oh)
            y = self.mesh.all_reduce_cols(y)
        else:
            y = self._o_proj(self._o_lat_to_heads(oh))  # composite absorb_o + o_proj
        self._mark("out_proj")
        return y

    def _run_state_op(self, stem: ttnn.Tensor, cache: TtLatentCache, positions: ttnn.Tensor) -> ttnn.Tensor:
        """ONE ``dsa_decode_state`` program on ``stem`` [1,1,B,Ws] (B = the users this chip's caches hold, their
        ``positions`` [B]): the q_a RMSNorm of every row into the pre-allocated ``q_resid`` [1,1,B,Lq] (returned),
        the RMSNorm'd latent of row b into row ``a % 32`` of page ``page_table[b, a // 32]``, the LayerNorm'd
        indexer k row and the raw gate row into ring slot ``a % P`` of ring block b, the pooled key of pool
        ``a // P`` into pool block ``b * pool_blocks + (a // P) // 32`` -- the four caches updated in place (the
        decode page-table / ring / pool views pair batch entry b with slot b: the identity the program assumes).
        Trace-safe: ``stem`` and ``q_resid`` are allocated inside the step (in-trace allocations replay at the same
        addresses), everything else is persistent; no host work."""
        idx = cache.index
        if self.state_op.pool_blocks != idx.pool_blocks:
            self.state_op.pool_blocks = idx.pool_blocks  # the pool cache geometry of this cache (a host int)
        B = stem.shape[-2]
        q_resid = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, B, self.cfg.q_lora_rank]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.device,
            ttnn.DRAM_MEMORY_CONFIG,
        )  # the program's pre-allocated output
        self.state_op(
            stem,
            cache.cache,
            cache.page_table,
            positions,
            idx.ring_k,
            idx.ring_g,
            idx.pool_cache,
            self.qa_norm_w32,
            self.kva_norm_w32,
            self.knorm_w32,
            self.knorm_b32,
            self.ape_block,
            self.selectors,
            q_resid,
        )
        return q_resid

    def _require_state_caches(self, cache: TtLatentCache, positions: ttnn.Tensor, B_loc: int):
        """The fused state program read-modify-writes bf16 tiles: a bfp8 pool cache (``GLM53_POOL_DTYPE``) or a
        non-bf16 latent cache cannot take it (explicit, never a silent fallback)."""
        if cache.dtype != ttnn.bfloat16 or cache.index.pool_dtype != ttnn.bfloat16:
            raise ValueError(
                "GLM53_DSA_DECODE_STATE needs bf16 latent and pool caches (dsa_decode_state updates bf16 tiles in "
                f"place); got latent {cache.dtype}, pools {cache.index.pool_dtype}"
            )
        if positions.shape[0] != B_loc:
            raise ValueError(
                f"GLM53_DSA_DECODE_STATE: {positions.shape[0]} positions for the {B_loc} users of this chip's caches"
            )

    def _decode_state_unified(self, x, cache: TtLatentCache, positions, pool_ids, ring_slots, meta: SparseDecodeMeta):
        """``decode_forward`` through the traced multi-user sparse path (``_decode_unified``) with the fused state
        block (``GLM53_DSA_DECODE_STATE=1`` / ``latent``): stem matmul -> ONE ``dsa_decode_state`` program (q_a norm,
        latent append, ring + pool maintenance, in place) -> the indexer head-weight slice -> ``q_lat = q_resid W_qb1``
        (``folds[0]``: the absorb folded into the projection; the packed rows read as the head-major [1,H,B,L] tiles
        for B <= 32, a free view) -> ``_decode_unified`` (its index query also through the free view) -> ``y =
        o_lat_packed W_o2`` (``folds[1]``) + the TP all-reduce. Per DSA layer at B <= 32 users per chip: 15 ops
        where the composite runs 50 (both folds; the ``w`` slice stays: the program has no fifth output).

        User-partitioned cache (``cache.partition``: column c holds users ``[c*B/cols, (c+1)*B/cols)`` of the row,
        ``positions`` [B/cols]): the program runs on the chip's own user rows (``partition_cols`` of the stem: one
        device slice, through ROW_MAJOR when B/cols is not a tile multiple) and returns their q_resid for the
        indexer; the head-sharded q projection needs EVERY user of the row, which the program cannot norm without
        owning their caches (one role-B core per stem row), so that q_resid comes from the composite slice + rms_norm
        (2 ops; the user-offset extension of the program's reader would remove them). ``latent`` mode: after the
        program the composite ``maintain_decode`` rewrites the ring slot and the pool from its own k / gate rows
        (the program's ring / pool writes are overwritten; its other-row updates are exact ``old * 1 + 0``), so the
        index state is bit-identical to the composite path by construction."""
        B, H, L = x.shape[-2], self.H, self.L
        idx = self.indexer
        b = self.stem_bounds
        fold_q, fold_o = self.folds
        part = cache.partition
        stem = self.lin_stem(x, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,B,Ws]
        stem_loc = self.mesh.partition_cols(stem, 2) if part else stem  # [1,1,B_loc,Ws]: this chip's users
        B_loc = stem_loc.shape[-2]
        self._require_state_caches(cache, positions, B_loc)
        q_resid_loc = self._run_state_op(stem_loc, cache, positions)  # [1,1,B_loc,Lq]; caches updated in place
        w_idx = ttnn.slice(stem_loc, [0, 0, 0, b["w"][0]], [1, 1, B_loc, b["w"][1]])  # [1,1,B_loc,Hi]
        if self.decode_state == "latent":
            k_loc = ttnn.slice(stem_loc, [0, 0, 0, b["k"][0]], [1, 1, B_loc, b["k"][1]])
            g_loc = ttnn.slice(stem_loc, [0, 0, 0, b["g"][0]], [1, 1, B_loc, b["g"][1]])
            idx.maintain_decode(x, cache.index, pool_ids, ring_slots, kg=(k_loc, g_loc))  # consumes k_loc, g_loc
        if part:
            qa = ttnn.slice(stem, [0, 0, 0, b["qa"][0]], [1, 1, B, b["qa"][1]])
            q_resid = ttnn.rms_norm(
                qa, epsilon=self.cfg.rms_norm_eps, weight=self.qa_norm, compute_kernel_config=self.hifi4
            )  # [1,1,B,Lq]: every user of the row, for the head-sharded q projection
            ttnn.deallocate(qa)
            ttnn.deallocate(stem_loc)
        else:
            q_resid = q_resid_loc
        ttnn.deallocate(stem)
        self._mark("stem+state")
        if fold_q:
            q_packed = self.lin_qb1(q_resid, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,B,H*L]
            # B <= 32: the packed tiled rows ARE the head-major [1,H,B,L] tile sequence (a free view)
            q_lat = ttnn.experimental.view(q_packed, [1, H, B, L]) if B <= TILE else self._heads_major(q_packed, B)
            q_owner = q_packed if B <= TILE else q_lat
            if B > TILE:
                ttnn.deallocate(q_packed)
        else:
            q = self._q(q_resid)  # [1,1,B,H*Dq]
            q_lat = self._absorb_q(q)  # [1,H,B,L] (composite absorb)
            ttnn.deallocate(q)
            q_owner = q_lat
        self._mark("absorb_q")
        if part:
            # heads -> users: every column holds its 16 heads of all B users; the owner needs all 64 heads of
            # its B/cols users (the inverse exchange after attention), exactly as the composite path
            q_t = ttnn.transpose(q_lat, 1, 2)  # [1,B,H_loc,L]
            ttnn.deallocate(q_owner)
            q_x = self.mesh.all_to_all_cols(q_t, in_dim=1, out_dim=2)  # [1,B/cols,H,L]
            ttnn.deallocate(q_t)
            q_full = ttnn.transpose(q_x, 1, 2)  # [1,H,B/cols,L]
            ttnn.deallocate(q_x)
            self._mark("a2a_heads_to_users")
            oh_full = self._decode_unified(q_resid_loc, q_full, cache, meta, w_idx, free_q_view=True)
            ttnn.deallocate(q_full)
            ttnn.deallocate(q_resid_loc)
            ttnn.deallocate(q_resid)
            o_t = ttnn.transpose(oh_full, 1, 2)  # [1,B/cols,H,L]
            ttnn.deallocate(oh_full)
            o_x = self.mesh.all_to_all_cols(o_t, in_dim=2, out_dim=1)  # [1,B,H_loc,L]
            ttnn.deallocate(o_t)
            oh = ttnn.transpose(o_x, 1, 2)  # [1,H_loc,B,L]
            ttnn.deallocate(o_x)
            self._mark("a2a_users_to_heads")
        else:
            oh = self._decode_unified(q_resid, q_lat, cache, meta, w_idx, free_q_view=True)  # [1,H,B,L]
            ttnn.deallocate(q_owner)
            ttnn.deallocate(q_resid)
        ttnn.deallocate(w_idx)
        if fold_o:
            # the head-major tiles read as the packed [1,1,B,H*L] rows (free view, B <= 32)
            o_p = ttnn.experimental.view(oh, [1, 1, B, H * L]) if B <= TILE else self._heads_packed(oh, B)
            y = self.lin_o2(o_p, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,B,hidden] (TP partial)
            if B > TILE:
                ttnn.deallocate(o_p)
            ttnn.deallocate(oh)
            y = self.mesh.all_reduce_cols(y)
        else:
            o_heads = self._o_lat_to_heads(oh)  # composite absorb_o
            ttnn.deallocate(oh)
            y = self._o_proj(o_heads)
            ttnn.deallocate(o_heads)
        self._mark("out_proj")
        return y

    def _heads_major(self, q_lat: ttnn.Tensor, B: int) -> ttnn.Tensor:
        """[1,1,B,H*L] -> [1,H,B,L] for B > 32 (several tile rows: no free view), through the row-major layout."""
        rm = ttnn.to_layout(q_lat, ttnn.ROW_MAJOR_LAYOUT)
        t = ttnn.to_layout(ttnn.reshape(rm, [1, B, self.H, self.L]), ttnn.TILE_LAYOUT)
        ttnn.deallocate(rm)
        out = ttnn.transpose(t, 1, 2)
        ttnn.deallocate(t)
        return out

    def _heads_packed(self, oh: ttnn.Tensor, B: int) -> ttnn.Tensor:
        """[1,H,B,L] -> [1,1,B,H*L] for B > 32, through the row-major layout."""
        t = ttnn.transpose(oh, 1, 2)  # [1,B,H,L]
        rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(t)
        out = ttnn.to_layout(ttnn.reshape(rm, [1, 1, B, self.H * self.L]), ttnn.TILE_LAYOUT)
        ttnn.deallocate(rm)
        return out

    # ------------------------------------------------ traced multi-user sparse decode
    def _compact_page_table(self, B: int, out_blocks: int) -> ttnn.Tensor:
        """Identity page table [B, out_blocks] of the compact gathered cache (a constant per batch size)."""
        key = ("compact_pt", B, out_blocks)
        if key not in self._pad_cache:
            pt = torch.arange(B * out_blocks, dtype=torch.int32).reshape(B, out_blocks)
            self._pad_cache[key] = ttnn.from_torch(
                pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
            )
        return self._pad_cache[key]

    def _zero_rows(self, shape: tuple) -> ttnn.Tensor:
        key = ("zeros",) + tuple(shape)
        if key not in self._pad_cache:
            self._pad_cache[key] = ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        return self._pad_cache[key]

    def _decode_unified(
        self, q_resid, q_lat, cache: TtLatentCache, meta: SparseDecodeMeta, w, *, free_q_view: bool = False
    ) -> ttnn.Tensor:
        """All users of the chip in one trace-safe graph: per user the indexer scores its pooled keys
        (``score_decode_row``) and the top-512 pools are selected under the user's own causal bound
        (``select_decode_pools`` with ``meta.valid_len[u]``); then, batched, the pool lists are completed
        with the tail pool and the dense users' in-order lists (``meta``), the selected pools are gathered
        out of the paged latent cache into a compact cache (``paged_pool_gather``: 2052 rows per user, one
        KB per selected token) and ONE ``paged_flash_multi_latent_attention_decode`` attends every user
        over its compact rows ``[0, cur_pos]``. Returns [1,H,B,L] bf16 (the ``oh`` layout). ``free_q_view``
        (the state block): the index query's head-major layout as a free view of the projection's packed rows
        (B <= 32) instead of the reshape copy + transpose."""
        idx = self.indexer
        B = q_lat.shape[-2]
        Hi, D, P = idx.Hi, idx.D, idx.P
        # GLM53_SCORER_BATCHED=1 (default): ONE index_score_decode program scores every user's query (its 32 heads
        # as the rows of one query tile row) against its pooled keys -> [B,1,1,W] score rows, no per-user calls,
        # no 32-row score blocks (needs pools_per_user % 256 == 0: TtIndexCache rounds to 256); 0 = one shared
        # indexer_score_dsa call per user
        scorer_batched = os.environ.get("GLM53_SCORER_BATCHED", "1") == "1"
        batched = meta.valid_len_rows is not None and os.environ.get("GLM53_TOPK_BATCHED", "1") == "1"
        # the persistent generic-op buffers of the gather graph exist from the first call on whatever regime runs it:
        # a persistent tensor allocated after a trace capture may alias that trace's transient buffers (mesh_model),
        # and the dense-skip regime (below) must be able to hand over to a later sparse re-capture
        self._ensure_sparse_buffers(B, cache, scorer_batched=scorer_batched, batched=batched)
        if self.pool_gather_mode == "dense_skip" and meta.all_dense:
            # B8a: every slot <= 2050: the dense flash-MLA over the real paged cache is HF-exact and reads the same
            # rows the gather would copy (dense users' compact slots = their pools in order), so the index query, the
            # scorer, the top-k and the gather are skipped (-4 ops, -136 us at 4 users / -1.3 ms at 32 per layer)
            meta.graph_regime = "dense"
            return self._attend_paged(q_lat, cache, meta.cur_pos, B)
        meta.graph_regime = "sparse"
        q_i, w_i = idx.queries_and_weights(q_resid, q_resid, w=w, free_view=free_q_view)  # [1,Hi,B,D], [1,1,B,Hi]
        if scorer_batched:
            assert batched, "GLM53_SCORER_BATCHED needs the batched top-k (GLM53_TOPK_BATCHED=1)"
            q4 = ttnn.transpose(q_i, 1, 2)  # [1,B,Hi,D]
            ttnn.deallocate(q_i)
            W = cache.index.pools_per_user
            # GLM53_SPARSE_DECODE_BOUND=1: the scorer scores / writes the first ``live_bound`` columns only (its bands
            # split the live chunks) and the segmented top-k's stage 1 ranks each user's live columns per segment; the
            # single-stage top-k ranks [0, valid[u]) either way. Bit-identical ids (custom_ops/sparse_bound.py).
            bound = meta.live_bound if self.sparse_bound else None
            # GLM53_TOPK_SEGMENTS=1 (default; wide pool caches only, segments_for): the scorer masks the columns beyond
            # each user's bound (-inf) and writes S segment rows per user; the top-k runs in two stages (S cores per
            # user, then the S*512 candidates) instead of one serial row per core (1.26 ms at 262 K pools)
            S = self._score_segments(W)
            if S > 1:
                seg_buf, ids1, vals1 = self._score_buffers(B, W, S)
                self.index_scorer(
                    q4,
                    w_i,
                    cache.index.pools_batched(),
                    seg_buf,
                    valid_rows=meta.valid_len_rows,
                    segments=S,
                    bound=bound,
                )
                ttnn.deallocate(q4)
                ttnn.deallocate(w_i)
                self._mark("index_score")
                top = self.topk_rows.segmented(
                    seg_buf,
                    S,
                    ids1,
                    vals1,
                    self._top_buffer(B),
                    valid_rows=meta.valid_len_rows if bound is not None else None,
                )  # [B,1,1,K] uint32
                self._mark("topk")
                return self._gather_and_attend(top, q_lat, cache, meta, B, P, out_persistent=True)
            # the bounded scorer runs the masked (v2) writer: the same scores in [0, valid[u]), the columns the single-stage
            # top-k reads; -inf beyond, up to the bound (the unbounded form writes the raw scores over the whole width)
            rows = self._score_buffers(B, W, 1)  # [B,1,1,W], the persistent score rows (written in place)
            self.index_scorer(
                q4,
                w_i,
                cache.index.pools_batched(),
                rows,
                valid_rows=meta.valid_len_rows if bound is not None else None,
                bound=bound,
            )
            ttnn.deallocate(q4)
            ttnn.deallocate(w_i)
            self._mark("index_score")
            top = self.topk_rows(rows, meta.valid_len_rows, self._top_buffer(B))  # [B,1,1,K] uint32, per-row bounds
            self._mark("topk")
            return self._gather_and_attend(top, q_lat, cache, meta, B, P, out_persistent=True)
        # per-user 32-row query blocks with the user's query in row 0: the B rows followed by 31 zero
        # rows, sliced at [u, u + 32) (rows past the user's are other users' queries or zeros: scored and
        # dropped by the row-0 slice inside score_decode_row)
        q_big = ttnn.concat([q_i, self._zero_rows((1, Hi, TILE - 1, D))], dim=2)  # [1,Hi,B+31,D]
        w_big = ttnn.concat([w_i, self._zero_rows((1, 1, TILE - 1, Hi))], dim=2)  # [1,1,B+31,Hi]
        ttnn.deallocate(q_i)
        ttnn.deallocate(w_i)
        self._mark("index_queries")
        # GLM53_TOPK_BATCHED=1 (default): the top-k runs ONCE over all users' score rows with per-row bounds
        # (``valid_len_rows``) through the model-local ``topk_rows`` op: one row costs the same as B rows in one call
        # (1.14 ms at 262 K pools); 0 = one shared topk_large_indices call per user (its single bound)
        rows_or_tops = []
        for u in range(B):
            qb = ttnn.slice(q_big, [0, 0, u, 0], [1, Hi, u + TILE, D])
            wb = ttnn.slice(w_big, [0, 0, u, 0], [1, 1, u + TILE, Hi])
            row = idx.score_decode_row(qb, wb, cache.index, u)  # [1,1,1,W]
            ttnn.deallocate(qb)
            ttnn.deallocate(wb)
            self._mark("index_score")
            if batched:
                rows_or_tops.append(row)
            else:
                rows_or_tops.append(idx.select_decode_pools(row, meta.valid_len[u]))  # [1,1,1,K] uint32
                ttnn.deallocate(row)
                self._mark("topk")
        ttnn.deallocate(q_big)
        ttnn.deallocate(w_big)
        if batched:
            rows = rows_or_tops[0] if B == 1 else ttnn.concat(rows_or_tops, dim=0)  # [B,1,1,W] bf16 row-major
            if B > 1:
                for t in rows_or_tops:
                    ttnn.deallocate(t)
            top = self.topk_rows(rows, meta.valid_len_rows, self._top_buffer(B))  # [B,1,1,K] uint32, per-row bounds
            ttnn.deallocate(rows)
            self._mark("topk")
        else:
            top = rows_or_tops[0] if B == 1 else ttnn.concat(rows_or_tops, dim=0)  # [B,1,1,K]
            if B > 1:
                for t in rows_or_tops:
                    ttnn.deallocate(t)
        return self._gather_and_attend(top, q_lat, cache, meta, B, P, out_persistent=batched)

    def _gather_and_attend(
        self, top, q_lat, cache: TtLatentCache, meta: SparseDecodeMeta, B: int, P: int, *, out_persistent: bool
    ):
        """The batched tail of ``_decode_unified``: complete the ranked pool lists with the tail pool and the dense
        users' in-order lists, gather the pools into the compact cache, attend. ``top`` [B,1,1,K] uint32
        (``out_persistent``: a persistent generic-op output, not deallocated)."""
        # the slot -> pool mapping (ranked pools, dense users' pools in order, the tail pool) is done by the gather;
        # its output is persistent (the generic op's io tensors keep their addresses under trace)
        compact = self.pool_gather(cache.cache, cache.page_table, top, meta.tail_pool, self._compact_buffer(B))
        if not out_persistent:
            ttnn.deallocate(top)
        out_blocks = compact.shape[0] // B
        self._mark("pool_gather")
        q_dec = ttnn.transpose(q_lat, 1, 2)  # [1,B,H,L]
        o_lat = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            q_dec,
            compact,
            head_dim_v=self.L,
            page_table_tensor=self._compact_page_table(B, out_blocks),
            cur_pos_tensor=meta.cur_pos,
            scale=self.scale,
            program_config=self._decode_sdpa_config(B),
            compute_kernel_config=self.sdpa_compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1,B,H,L]
        ttnn.deallocate(q_dec)
        oh = ttnn.transpose(o_lat, 1, 2)  # [1,H,B,L]
        ttnn.deallocate(o_lat)
        self._mark("compact_mla_decode")
        return oh

    # ------------------------------------------------ persistent buffers of the gather graph + the dense-skip graph
    def _top_buffer(self, B: int) -> ttnn.Tensor:
        """[B,1,1,K] uint32 ROW_MAJOR: the persistent output of the batched top-k (one per batch size)."""
        if B not in self._top_buf:
            K = self.cfg.index_select_pools
            self._top_buf[B] = ttnn.zeros(
                (B, 1, 1, K), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
            )
        return self._top_buf[B]

    def _score_segments(self, W: int) -> int:
        """Segments per user of the batched scorer (``GLM53_TOPK_SEGMENTS``, default on; 1 = one row per user)."""
        return self.topk_rows.segments_for(W) if os.environ.get("GLM53_TOPK_SEGMENTS", "1") == "1" else 1

    def _score_buffers(self, B: int, W: int, S: int):
        """The persistent score buffers of the batched scorer: ``S > 1`` -> (segment rows [B*S,1,1,W/S] bf16, the
        stage-1 ids [B,1,1,S*K] uint32, their values fp32) keyed by (B, S); ``S == 1`` -> the [B,1,1,W] bf16 rows."""
        K = self.cfg.index_select_pools
        if S > 1:
            key = (B, S)
            if key not in self._score_buf:
                self._score_buf[key] = (
                    ttnn.zeros(
                        (B * S, 1, 1, W // S), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
                    ),
                    ttnn.zeros((B, 1, 1, S * K), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device),
                    ttnn.zeros((B, 1, 1, S * K), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device),
                )
            return self._score_buf[key]
        if B not in self._score_buf:
            self._score_buf[B] = ttnn.zeros(
                (B, 1, 1, W), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
            )
        return self._score_buf[B]

    def _compact_buffer(self, B: int) -> ttnn.Tensor:
        """The persistent compact cache the gather writes: [B * OB, 1, 32, L] bf16 TILE, OB = ceil((K+1) P / 32)."""
        if B not in self._compact:
            K = self.cfg.index_select_pools
            self._compact[B] = ttnn.zeros(
                [B * self.pool_gather.output_blocks(K, self.pool_gather.P), 1, TILE, self.L],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
        return self._compact[B]

    def _ensure_sparse_buffers(self, B: int, cache: TtLatentCache, *, scorer_batched: bool, batched: bool):
        """Allocate (once) every persistent buffer the gather graph of batch size B uses, whatever regime the call
        takes, so that no persistent allocation happens after a trace capture (``_decode_unified``)."""
        self._compact_buffer(B)
        if batched:
            self._top_buffer(B)
        if scorer_batched:
            W = cache.index.pools_per_user
            self._score_buffers(B, W, self._score_segments(W))

    def _attend_paged(self, q_lat: ttnn.Tensor, cache: TtLatentCache, cur_pos: ttnn.Tensor, B: int) -> ttnn.Tensor:
        """The flash-MLA decode of every slot over the REAL paged latent cache up to ``cur_pos`` (the dense-skip
        graph, ``GLM53_POOL_GATHER=dense_skip``; the dense path of ``decode_forward`` runs the same op on the same
        tensors): q_lat [1,H,B,L] -> oh [1,H,B,L]. Same program as the compact attention (``_gather_and_attend``): the
        page table and the physical pages differ, the logical rows are identical for a dense user, so the result is
        expected bit-identical."""
        q_dec = ttnn.transpose(q_lat, 1, 2)  # [1,B,H,L]
        o_lat = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            q_dec,
            cache.cache,
            head_dim_v=self.L,
            page_table_tensor=cache.page_table,
            cur_pos_tensor=cur_pos,
            scale=self.scale,
            program_config=self._decode_sdpa_config(B),
            compute_kernel_config=self.sdpa_compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1,B,H,L]
        ttnn.deallocate(q_dec)
        oh = ttnn.transpose(o_lat, 1, 2)  # [1,H,B,L]
        ttnn.deallocate(o_lat)
        self._mark("paged_mla_decode")
        return oh

    def _decode_sparse_rows(
        self, x, q_resid, q_lat, cache: TtLatentCache, positions: list[int], users: list[int], w=None
    ):
        """Per-user indexer selection + sparse_sdpa rows {u: [1,H,1,L]} for the listed users (positions
        > dense_exact_max_pos, so every index row is a full valid prefix). ``w``: the indexer head
        weights [1,1,B,Hi] when the caller already has them (the fused decode stem)."""
        idx = self.indexer
        D = idx.D
        q_i, w_i = idx.queries_and_weights(x, q_resid, w=w)  # [1,Hi,B,D], [1,1,B,Hi]
        out = {}
        for u in users:
            a = positions[u]
            # indexer_score_dsa wants a tile-aligned query block and chunk start: score the 32 positions
            # [a - a%32, a - a%32 + 32) with this user's query placed at row r = a % 32 (zero rows elsewhere,
            # scored but discarded); the selection keeps row r only (one row through the expansion + top-k)
            start, r = a - a % TILE, a % TILE
            qu = self._place_row(ttnn.slice(q_i, [0, 0, u, 0], [1, idx.Hi, u + 1, D]), r)
            wu = self._place_row(ttnn.slice(w_i, [0, 0, u, 0], [1, 1, u + 1, idx.Hi]), r)
            self._mark("index_queries")
            indices = idx.select(qu, wu, cache.index, u, query_start=start, num_pools_valid=(a + 1) // idx.P, row=r)
            q_rm = ttnn.to_layout(ttnn.slice(q_lat, [0, 0, u, 0], [1, self.H, u + 1, self.L]), ttnn.ROW_MAJOR_LAYOUT)
            self._mark("q_rm")
            kv_tile = cache.user_rows(u, a + 1)  # the user's whole prefix: a fresh TILE copy ...
            self._mark("kv_slice")
            kv_rm = ttnn.to_layout(kv_tile, ttnn.ROW_MAJOR_LAYOUT)  # ... untilized for the sparse_sdpa kv contract
            ttnn.deallocate(kv_tile)
            self._mark("kv_untilize")
            o_rm = ttnn.transformer.sparse_sdpa(
                self._pad_heads_rm(q_rm),
                kv_rm,
                indices,
                v_dim=self.L,
                kv_format=ttnn.transformer.SparseKVFormat.BF16,
                scale=self.scale,
                k_chunk_size=128,
            )  # [1,H,1,L]
            self._mark("sparse_sdpa")
            ttnn.deallocate(kv_rm)
            ttnn.deallocate(q_rm)
            ttnn.deallocate(indices)
            out[u] = self._unpad_heads(ttnn.to_layout(o_rm, ttnn.TILE_LAYOUT))
            self._mark("sparse_out")
        ttnn.deallocate(q_i)
        ttnn.deallocate(w_i)
        return out

    def selected_pools(self, x: ttnn.Tensor, cache: TtLatentCache, user: int, position: int) -> list[int]:
        """Test hook: the pool ids the TRACED path selects (``score_decode_row`` + ``select_decode_pools``
        under the user's own bound) for one query ``x`` [1,1,1,4096] of ``user`` at ``position`` on the
        current index cache (read only); descending score, ``index_select_pools`` entries."""
        idx = self.indexer
        Hi, D, P, K = idx.Hi, idx.D, idx.P, self.cfg.index_select_pools
        q_resid = self._q_resid(x)
        q_i, w_i = idx.queries_and_weights(x, q_resid)
        qb = ttnn.concat([q_i, self._zero_rows((1, Hi, TILE - 1, D))], dim=2)
        wb = ttnn.concat([w_i, self._zero_rows((1, 1, TILE - 1, Hi))], dim=2)
        row = idx.score_decode_row(qb, wb, cache.index, user)
        valid = ttnn.from_torch(
            torch.tensor([max((position + 1) // P, K)], dtype=torch.int64),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
        top = idx.select_decode_pools(row, valid)
        pools = [int(v) for v in ttnn.to_torch(top).flatten().tolist()]
        for t in (q_resid, q_i, w_i, qb, wb, row, valid, top):
            ttnn.deallocate(t)
        return pools

    def selected_ids(self, x: ttnn.Tensor, cache: TtLatentCache, user: int, position: int) -> list[int]:
        """Test hook: the raw index row (token ids, 0xFFFFFFFF sentinels) the indexer selects for one
        query ``x`` [1,1,1,4096] (the normalized attention input) of ``user`` at ``position`` on the
        CURRENT index cache (read only)."""
        idx = self.indexer
        q_resid = self._q_resid(x)
        q_i, w_i = idx.queries_and_weights(x, q_resid)
        start, r = position - position % TILE, position % TILE
        qu = self._place_row(q_i, r)
        wu = self._place_row(w_i, r)
        ids_t = idx.select(qu, wu, cache.index, user, query_start=start, num_pools_valid=(position + 1) // idx.P, row=r)
        ids = ttnn.to_torch(ids_t).flatten().tolist()
        for t in (q_resid, q_i, w_i, ids_t):
            ttnn.deallocate(t)
        return [int(i) for i in ids]

    # ------------------------------------------------------------- sparse_sdpa head padding
    def _pad_heads_rm(self, q_rm: ttnn.Tensor) -> ttnn.Tensor:
        """sparse_sdpa wants H % 32 == 0: pad a row-major [1, H, T, L] query with zero heads (their
        outputs are discarded by ``_unpad_heads``); a no-op when the local head count is tile-aligned."""
        H = q_rm.shape[1]
        if H % TILE == 0:
            return q_rm
        pad = TILE - H % TILE
        key = ("qpad", pad, q_rm.shape[-2], q_rm.shape[-1])
        if key not in self._pad_cache:
            self._pad_cache[key] = ttnn.zeros(
                (1, pad, q_rm.shape[-2], q_rm.shape[-1]),
                dtype=q_rm.dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
            )
        return ttnn.concat([q_rm, self._pad_cache[key]], dim=1)

    def _unpad_heads(self, o: ttnn.Tensor) -> ttnn.Tensor:
        if o.shape[1] == self.H:
            return o
        return ttnn.slice(o, [0, 0, 0, 0], [1, self.H, o.shape[-2], o.shape[-1]])

    def _place_row(self, row: ttnn.Tensor, r: int) -> ttnn.Tensor:
        """[.., 1, W] -> [.., 32, W] with ``row`` at row r and zeros elsewhere: a broadcast multiply by the
        one-hot column ``e_r`` [1,1,32,1] (cached per r), so every r runs the same program (a zeros +
        concat build changes geometry with r and recompiled at every new ``position % 32``)."""
        key = ("row_mask", r)
        if key not in self._pad_cache:
            m = torch.zeros(1, 1, TILE, 1)
            m[0, 0, r, 0] = 1.0
            self._pad_cache[key] = ttnn.from_torch(m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        return ttnn.multiply(row, self._pad_cache[key])
