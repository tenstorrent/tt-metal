# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 window attention (reference/ds41_ref.py:_attention for a ratio-0 layer; inference Attention) on the
SP x TP mesh, and the served window-KV cache it reads and writes (serving_contract.md "KV cache", "Attention and
cache writes"; plan.md "Attention and cache writes").

    attn_out [S, H] = fp8_linear(grouped_wo_a(fp8_qdq(unrope(softmax_sink(q . window) window))), wo_b)
    q = rope(fp8_linear(q_resid, wq_b)) [S, 64 heads x 512]; window of position p = KV rows [p - 127, p]
    (the released code's fp8 act quant-dequant of the matmul inputs: GEMM_INPUTS (FIX.7, owner 2026-10-28) feeds
    wq_b, wo_a and wo_b bfloat8_b, the device's own 8-bit format; bf16 = the path before FIX.7)

Layout per chip (r, c) (mesh axis 0 = SP rows, axis 1 = TP columns, both read from mesh.shape): the inputs are
row-split, TP-replicated (q_resid [1, 1, S/SP, 1280] fp32, kv [1, 1, S/SP, 512] bf16; row r = positions
[start + r S/SP, ...)); the output is the hidden split [1, 1, S/SP, H/TP] fp32 (tt/layout.py).

Chain (one path on every box; nothing on the host per chunk except picking load-time tables by integers):
    window KV      all_gather over axis 0 -> [S, 512] bf16 (every chip, natural order); rows at or past actual_end
                   zeroed (iota < end - start, load-time iota); ROW_MAJOR
    cache write    each SP row's rows in the server's order (ttnn.embedding with the load-time index row of
                   (start mod S) / 32), then ttnn.bringup.update_padded_kv_cache(kv_actual_global=
                   start, valid_global=end, tp_axis=1): positions [start, end) land at the owner chip, TP-dedup, the
                   last record's pad rows written as zeros
    window carry   positions [start - 128, start) read back from the served cache: the same local rows sliced on every
                   chip, gathered over the mesh, the owner's block kept (one or two pieces; zeros before position 0)
    table          [carry 128 | chunk S] rows x 512, ROW_MAJOR, replicated; the RoPE tail de-interleaved ([even |
                   odd], as q's). WINDOW_KV_FORMAT=bfp8_blocks (DS41-150, the default): the same rows (as stored) as
                   bfloat8_b blocks of 32 rows, rounded to nearest even (ttnn.bringup.ckv_bfp4 blocks=True; the chunk
                   from its tiles, the carry tiled), one buffer every layer writes in turn (WindowTables.window_blocks),
                   read by the sdpa fork's kv_format ROW_BFP8; the served cache stays bf16
    q              bf16 cast -> ttnn.linear wq_b (the chip's heads, each head's rope rows de-interleaved,
                   ATTN_FIDELITY + fp32 acc, fp32 out) -> heads [1, H/TP, S/SP, 512] -> RoPE (fp32 SFPU,
                   tt/rope.rope_heads) -> x 512^-0.5 (the softmax scale folded into q: the op's scale is then 1, exact;
                   K-27) -> bf16 -> all_to_all head -> seq over axis 1 (sparse_sdpa needs H % 32 == 0: 16 heads per
                   chip on 4x4; done whenever TP > 1, one path everywhere) -> [1, 64, S/(SP TP), 512]
    attention      stock ttnn.transformer.sparse_sdpa(attention_sink = sink / 1, scale 1) at its normal settings
                   (bf16 running state; fp32 DEST per FP32_ACC; P.13, F137): per query the 128 window rows
                   (load-time index list: row d + 1 + j of the table, a sentinel tail for positions < 0), bf16 output
                   -> all_to_all back
    output         inverse RoPE + re-interleave -> bf16 -> heads concatenated per group ->
                   grouped wo_a (the chip's whole groups, one batched matmul) -> bf16 -> wo_b (K-split: the chip's
                   groups) -> the sum over TP (tt/ccl.py:sum_cols, CCL_IMPL) -> the chip's hidden slice
    ATTN_GLUE=on   (P.15) q: nlp_create_q_heads_split -> RoPE on the 64-column tail -> x scale writing bf16 -> tail
                   interleaved by a PairPerm matmul (tt/rope.py): window layers too read the table as stored
                   (INTERLEAVED); output: bf16 nope kept, the tail de-interleaved by PairPerm (fp32 out), inverse RoPE
                   writing bf16 [re | ro] (wo_a's input rows permuted at load), nlp_concat_heads after wo_a,
                   the bf16 reduce-scatter result returned as is (tt/residual.py casts where it needs fp32)
    ROPE_IMPL=on   (P.17, default) q: x 512^-0.5 folded into the input's bf16 cast -> wq_b (checkpoint order) writing
                   bf16 -> nlp_create_q_heads_split -> ONE stock rotary_embedding_indexed with the nope prepended;
                   output: ONE stock rotary op with the transposed trans-mat on the bf16 tail in place; wo_a in
                   checkpoint order (tt/rope.rope_stock)
    ATTN_SP32_PROJ=on (P.9, default; with ROPE_IMPL=on) every projection on the chip's own S/(SP TP) query rows (the
                   rows sparse_sdpa reads, query_offsets) with full, replicated weights: q_resid mesh_partition over
                   axis 1 -> wq_b (all heads) -> [1, H, S/(SP TP), 512] directly (no head -> seq all_to_all); RoPE /
                   inverse RoPE with the chip's cos / sin rows (RopeTable.rows(tp_split=True)); wo_a all groups, wo_b
                   full -> [S/(SP TP), H] bf16 -> ONE all_to_all over axis 1 (split H, concat rows) -> [S/SP, H/TP]
                   (no seq -> head all_to_all, no TP reduce-scatter of a wo_b partial)
    ATTN_SP32=on   (P.10, default) q_resid / kv arrive as the chip's own S/(SP TP) rows (tt/model.py): no q_resid
                   mesh_partition; the window KV is gathered over TP (the SP row's block in order), then over SP

The served cache (TtWindowCache): position g on mesh row (g / W) % SP, column ((g mod W) / (W / TP)), local row
(g / S) (W / TP) + g mod (W / TP) (W = S / SP; update_padded_kv_cache's sp x tp linearized block-cyclic order),
bf16 ROW_MAJOR, one ND shard [1, 1, 32, 512] per 32-token record round-robin over the DRAM banks (tt/served.py, as
init_kvpe_cache); a (slot, layer) slab of the serving engine's 0_window_kv tensor when served (tt/runners).
"""

from __future__ import annotations

import torch

import ttnn

from . import ccl, host_alloc
from . import mm_configs as mm
from .ccl import dealloc
from .common import (
    DRAM,
    SP_AXIS,
    TP_AXIS,
    act_dtype,
    attn_compute_config,
    bf16_activations,
    cast,
    dense_dtype,
    gemm_dtype,
    gemm_input,
    sum_cols,
)
from .cores import from_torch, max_grid, sub_cores, sub_cores_tiled, to_rm
from .rope import RopeTable, deinterleave_perm, deinterleave_tail, glue, rope_chain, rope_chains, rope_heads, rope_stock
from .served import CacheSlab, DeviceScalars, alloc_served
from .settings import settings

WINDOW = 128  # sliding window (cfg.window); the table's carry region
# WINDOW_KV_FORMAT=bfp8_blocks: one page per 32 consecutive window rows (ttnn.bringup.ckv_bfp4 blocks=True, 512 wide):
# [32 x 32 shared exponents | a copy 32 B on, at 1056 | 32 x 512 sign-magnitude bytes at 2112]
WINDOW_BLOCK_BYTES = 18496


def lowbit_host() -> bool:
    """LOWBIT_HOST=on (P.12): the window blocks' carry and chunk in one compact ckv_bfp4 call (``_put_blocks``)."""
    return settings.get("LOWBIT_HOST") == "on"


def pack_before_gather() -> bool:
    """CKV_PACK_BEFORE_GATHER=on (P.11): each chip packs its own rows before the all_gather over SP (``WindowTables``
    ``packed`` buffers; tt/compressor.TtCompressor._write_packed, TtWindowAttention._window_packed)."""
    return settings.get("CKV_PACK_BEFORE_GATHER") == "on"


ROW_BFP4_BYTES = 320  # a bfloat4_b row record (ttnn.bringup.ckv_bfp4; tt/compressor.BFP4_ROW_BYTES)


def window_bfp8() -> bool:
    """WINDOW_KV_FORMAT=bfp8_blocks (DS41-150): the window region as bfloat8_b blocks of 32 rows (``WindowTables``)."""
    return settings.get("WINDOW_KV_FORMAT") == "bfp8_blocks"


def window_region() -> bool:
    """WINDOW_KV_FORMAT=bfp8_tiles (P.18): each chip's own window region [carry 128 | its q_rows rows] as bfloat8_b
    tile rows (``TtWindowAttention.region``), the carry by one neighbour send (tt/window_send.py)."""
    return settings.get("WINDOW_KV_FORMAT") == "bfp8_tiles"


SENTINEL = -1  # uploaded as int32 -1, read through uint32: 0xFFFFFFFF, sparse_sdpa's "no key"
SDPA_K_CHUNK = 128


def _mapper(mesh, dims):
    return ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims)


def _upload(mesh, t, dtype, layout, dims=None):
    return from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=DRAM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh) if dims is None else _mapper(mesh, dims),
    )


class WindowGeometry:
    """Chunk and mesh numbers the window path depends on (from mesh.shape and the serving chunk)."""

    def __init__(self, mesh, chunk: int):
        self.sp, self.tp = mesh.shape[SP_AXIS], mesh.shape[TP_AXIS]
        self.n = self.sp * self.tp
        self.chunk = chunk
        assert chunk % (32 * self.n) == 0, f"chunk {chunk} does not split into tiles over {self.n} chips"
        self.w = chunk // self.sp  # tokens per SP row per chunk (the server's W)
        self.stripe = self.w // self.tp  # tokens per chip per W block (the cache's TP-dedup stripe)
        self.q_rows = self.stripe  # query rows per chip after the head -> seq reshard (== W when TP == 1)

    def cache_rows(self, max_seq: int) -> int:
        """Local rows per chip for positions [0, max_seq) (whole chunk periods)."""
        return -(-max_seq // self.chunk) * self.stripe

    def server_rows(self, start: int) -> torch.Tensor:
        """[SP, W] natural chunk offsets d (position start + d) of each SP row's rows in the server's order
        (serving_contract.md "Input": row (g / W) % SP, positions ascending)."""
        s, w, sp = self.chunk, self.w, self.sp
        intra, c_start = start % w, (start // w) % sp
        j = torch.arange(w)
        out = []
        for r in range(sp):
            k = (r - c_start) % sp
            out.append(torch.where(j < w - intra, j, s - w + j) if k == 0 else k * w - intra + j)
        return torch.stack(out)

    def query_offsets(self) -> torch.Tensor:
        """[SP, TP, q_rows] natural chunk offsets of the queries each chip holds after the reshard."""
        d = torch.arange(self.chunk).view(self.sp, self.w)
        return d.view(self.sp, self.tp, self.q_rows)

    @staticmethod
    def start_class(start: int) -> int:
        """The index table a chunk uses: the carry is whole from position 128 on."""
        return min(start, WINDOW)


class WindowTables:
    """Load-time tables shared by every window layer of a model (they depend only on the chunk and the mesh):
    the server-order row index per (start mod S) / 32, the attention index lists per start class (0, 32, 64, 96,
    >= 128), the chunk-row iota for the pad mask, and a zero carry."""

    def __init__(self, mesh, chunk: int):
        g = self.geo = WindowGeometry(mesh, chunk)
        self.mesh = mesh
        # server order: [SP, 1, S / 32, W] -> chip row r holds its [S / 32, W] table (replicated over TP)
        srv = torch.stack([g.server_rows(32 * k) for k in range(chunk // 32)], dim=1)  # [SP, S/32, W]
        self.srv = _upload(mesh, srv.unsqueeze(1).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, (0, None))
        # attention index lists: query d reads table rows d + 1 + j (positions start + d - 127 + j); rows of
        # positions < 0 (start < 128) dropped, the valid ones first, a sentinel tail
        self.index = self.index_lists(0)
        self.iota = _upload(
            mesh, torch.arange(chunk, dtype=torch.float32).view(1, 1, chunk, 1), ttnn.float32, ttnn.TILE_LAYOUT
        )
        self.zero_carry = _upload(mesh, torch.zeros(1, 1, WINDOW, 512), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        self._scalars = {}
        # GLUE_FOLD=on (P.19): the server-order index row of the latest start, sliced once for every cache write of
        # the chunk (window caches, compressed caches); the device scalars memoized likewise (tt/served.DeviceScalars)
        self.fold = settings.get("GLUE_FOLD") == "on"
        self._srv_row = None  # (k, [1, 1, 1, W] uint32)
        self._blocks = None  # WINDOW_KV_FORMAT=bfp8_blocks: the window region every layer writes in turn
        # CKV_PACK_BEFORE_GATHER=on (P.11), load time (agent rule 5; allocated here, not mid-forward, DS41-146): per SP
        # row block size n (W: window rows, compressed rows at ratio 1; W / 2: ratio 2) the chunk-relative row index
        # of the SP row's own rows (the pad mask) and the packed buffers, {(kind, n): (local, gathered)}: "rows" =
        # bfloat4_b row records (compressed KV), "blocks" = bfloat8_b window blocks of 32 rows
        self._sp_iota, self._packed, self._carry = {}, {}, None
        # WINDOW_KV_FORMAT=bfp8_tiles (P.18), load time: the region index lists, the chip's own rows' pad-mask iota,
        # the SP row's iota for the served write's mask (``mask_local``) and the neighbour send
        self.region_index, self._chip_iota, self.send = None, None, None
        if window_region():
            assert (
                g.q_rows >= WINDOW and g.q_rows % 32 == 0
            ), f"WINDOW_KV_FORMAT=bfp8_tiles: {g.q_rows} query rows a chip; the carry must be the previous chip's"
            self.region_index = self.region_lists()
            d = g.query_offsets().unsqueeze(-1).to(torch.float32)  # [SP, TP, q_rows, 1]
            self._chip_iota = _upload(mesh, d, ttnn.float32, ttnn.TILE_LAYOUT, (0, 1))
            if g.sp > 1 and g.w % 32 == 0:
                ids = torch.arange(g.sp * g.w, dtype=torch.float32).view(g.sp, 1, g.w, 1)
                self._sp_iota[g.w] = _upload(mesh, ids, ttnn.float32, ttnn.TILE_LAYOUT, (0, None))
            if g.n > 1:
                from .window_send import WindowSend

                self.send = WindowSend(mesh, self.region_pages(), WINDOW_BLOCK_BYTES)
        if pack_before_gather() and g.sp > 1:
            ns = sorted({n for n in (g.w, g.w // 2) if n % 32 == 0 and g.w % n == 0})
            for n in ns:
                ids = torch.arange(g.sp * n, dtype=torch.float32).view(g.sp, 1, n, 1)
                self._sp_iota[n] = _upload(mesh, ids, ttnn.float32, ttnn.TILE_LAYOUT, (0, None))
            if settings.get("CKV_FORMAT") == "bfp4_rows":
                for n in ns:
                    self._packed[("rows", n)] = self._pair(n, ROW_BFP4_BYTES)
            if window_bfp8():
                self._packed[("blocks", g.w)] = self._pair(g.w // 32, WINDOW_BLOCK_BYTES)
                # the carry read from the served cache (TtWindowCache.carry_owner): each chip's 4 carry blocks, over
                # TP, over the mesh (chip-linear r TP + c)
                c = WINDOW // 32
                self._carry = [
                    ttnn.zeros(
                        [1, 1, k * c, WINDOW_BLOCK_BYTES],
                        dtype=ttnn.uint8,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=mesh,
                        memory_config=DRAM,
                    )
                    for k in (1, g.tp, g.n)
                ]

    def _pair(self, pages: int, width: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """(local [1, 1, pages, width], gathered [1, 1, SP pages, width]) uint8 ROW_MAJOR (load time)."""
        return tuple(
            ttnn.zeros(
                [1, 1, k * pages, width],
                dtype=ttnn.uint8,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                memory_config=DRAM,
            )
            for k in (1, self.geo.sp)
        )

    def packs(self, kind: str, n: int, start: int) -> bool:
        """CKV_PACK_BEFORE_GATHER=on (P.11) applies: the buffers exist and the start is chunk-aligned (the server's
        order of each SP row is then its own block: the served write is local)."""
        return start % self.geo.chunk == 0 and (kind, n) in self._packed and pack_before_gather()

    def mask_local(self, t: ttnn.Tensor, valid: int) -> ttnn.Tensor:
        """t [1, 1, n, w] bf16 TILE, the SP row's own block of chunk rows (row r n + i on SP row r): rows at or past
        ``valid`` zeroed (exact), as ``iota``'s mask does on the gathered rows. Returns ``t`` when every row is valid
        (the caller frees what it got only when it is not ``t``)."""
        n = t.shape[-2]
        if valid >= n * self.geo.sp:
            return t
        keep = ttnn.lt(self._sp_iota[n], float(valid), dtype=ttnn.bfloat16, memory_config=DRAM)
        out = ttnn.multiply(t, keep, dtype=ttnn.bfloat16, memory_config=DRAM, sub_core_grids=sub_cores(t))
        dealloc(keep)
        return out

    def packed(self, kind: str, n: int) -> ttnn.Tensor:
        """The local packed buffer of (kind, n) (the caller packs its rows into it at row 0)."""
        return self._packed[(kind, n)][0]

    def gather_packed(self, kind: str, n: int) -> ttnn.Tensor:
        """The local packed buffer gathered over SP (natural order) into the load-time output; the caller frees the
        result only when it is not that output (``is_kept``)."""
        local, out = self._packed[(kind, n)]
        res = ccl.all_gather(local, 2, SP_AXIS, out=out)
        # (the fabric gather returns a new wrapper of ``out``'s buffer: hand back ``out`` itself, never freed)
        return out if res.buffer_address() == out.buffer_address() else res

    def is_kept(self, t: ttnn.Tensor) -> bool:
        return any(t is pair[1] for pair in self._packed.values())

    def carry_local(self) -> ttnn.Tensor | None:
        """The local carry block buffer [1, 1, 4, WINDOW_BLOCK_BYTES] (None: not built)."""
        return self._carry[0] if self._carry is not None else None

    def gather_carry(self) -> ttnn.Tensor:
        """Every chip's packed carry blocks, chip-linear [1, 1, 4 N, WINDOW_BLOCK_BYTES] (load-time outputs, never
        freed: the caller slices its owner's 4 blocks)."""
        local, tp_out, all_out = self._carry
        t = local
        if self.geo.tp > 1:
            t = ccl.all_gather(t, 2, TP_AXIS, out=tp_out)
        res = ccl.all_gather(t, 2, SP_AXIS, out=all_out)
        if t is not local and t.buffer_address() != tp_out.buffer_address():
            dealloc(t)
        assert res.buffer_address() == all_out.buffer_address(), "gather_carry: the fabric gather (CCL_IMPL) is off"
        return all_out

    def window_blocks(self) -> ttnn.Tensor:
        """WINDOW_KV_FORMAT=bfp8_blocks: the window region [carry 128 | chunk S] as bfloat8_b blocks of 32 rows,
        [1, 1, (128 + S) / 32, WINDOW_BLOCK_BYTES] uint8 ROW_MAJOR, replicated; allocated once, each layer writes
        its rows and runs its attention before the next (one queue)."""
        if self._blocks is None:
            self._blocks = ttnn.zeros(
                [1, 1, (WINDOW + self.geo.chunk) // 32, WINDOW_BLOCK_BYTES],
                dtype=ttnn.uint8,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                memory_config=DRAM,
            )
        return self._blocks

    def region_pages(self) -> int:
        """WINDOW_KV_FORMAT=bfp8_tiles: pages (32-row blocks) of a chip's window region [carry 128 | q_rows]."""
        return (WINDOW + self.geo.q_rows) // 32

    def region_lists(self) -> dict:
        """WINDOW_KV_FORMAT=bfp8_tiles: ``index_lists`` in each chip's own region (load time): region row 0 is
        position start + d0 - 128 (d0 the chip's first query offset), so query d reads region rows d - d0 + 1 + j;
        positions < 0 dropped (chip 0 at starts < 128), the valid ids first, a sentinel tail. {class: [SP, TP, q_rows,
        128] uint32 ROW_MAJOR, chip (r, c) its own}."""
        d = self.geo.query_offsets().unsqueeze(-1)  # [SP, TP, q, 1]
        d0 = d[:, :, :1]
        j = torch.arange(WINDOW)
        out = {}
        for cls in range(0, WINDOW + 1, 32):
            lo = torch.clamp(d + 1, min=WINDOW - cls)
            n = d + WINDOW - lo + 1
            ids = torch.where(j < n, lo - d0 + j, torch.full_like(lo + j, SENTINEL))
            assert int(ids.max()) < self.region_pages() * 32, int(ids.max())
            out[cls] = _upload(self.mesh, ids.to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, (0, 1))
        return out

    def mask_chip(self, t: ttnn.Tensor, valid: int) -> ttnn.Tensor:
        """WINDOW_KV_FORMAT=bfp8_tiles: t [1, 1, q_rows, w] bf16 TILE, the chip's own rows: rows at or past chunk
        offset ``valid`` zeroed (exact). Returns ``t`` when every row is valid."""
        if valid >= self.geo.chunk:
            return t
        keep = ttnn.lt(self._chip_iota, float(valid), dtype=ttnn.bfloat16, memory_config=DRAM)
        out = ttnn.multiply(t, keep, dtype=ttnn.bfloat16, memory_config=DRAM, sub_core_grids=sub_cores(t))
        dealloc(keep)
        return out

    def index_lists(self, base: int) -> dict:
        """The window index lists per start class with the window region at table row ``base`` (load time): query d
        reads rows base + d + 1 + j; positions < 0 dropped, the valid ids first, a sentinel tail. {class: [SP, TP,
        q_rows, 128] uint32 ROW_MAJOR, chip (r, c) its own query rows}."""
        d = self.geo.query_offsets().unsqueeze(-1)  # [SP, TP, q, 1]
        j = torch.arange(WINDOW)
        out = {}
        for cls in range(0, WINDOW + 1, 32):
            lo = torch.clamp(d + 1, min=WINDOW - cls)
            n = d + WINDOW - lo + 1
            ids = torch.where(j < n, base + lo + j, torch.full_like(lo + j, SENTINEL))
            out[cls] = _upload(self.mesh, ids.to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, (0, 1))
        return out

    def scalars(self, step: int, n: int) -> DeviceScalars:
        """Load-time device values 0, step, .., (n - 1) step (tt/served.DeviceScalars), shared by every cache."""
        if (step, n) not in self._scalars:
            self._scalars[(step, n)] = DeviceScalars(self.mesh, [step * k for k in range(n)], memo=self.fold)
        return self._scalars[(step, n)]

    def srv_row(self, k: int) -> ttnn.Tensor:
        """Row k of the server-order index table, [1, 1, 1, W] uint32 ROW_MAJOR (positions (32 k) mod S on). The
        caller releases it with ``release_srv`` (GLUE_FOLD=on keeps the latest row for the chunk's other writes)."""
        hit = self._srv_row
        if hit is not None and hit[0] == k:
            return hit[1]
        idx = ttnn.slice(
            self.srv, [0, 0, k, 0], [1, 1, k + 1, self.geo.w], memory_config=DRAM, sub_core_grids=sub_cores(self.srv)
        )
        if self.fold:
            if hit is not None:
                dealloc(hit[1])
            self._srv_row = (k, idx)
        return idx

    def release_srv(self, idx: ttnn.Tensor) -> None:
        if self._srv_row is None or idx is not self._srv_row[1]:
            dealloc(idx)


class TtWindowCache:
    """The served 0_window_kv cache of one layer and slot: a slab [1, 1, rows, 512] bf16 ROW_MAJOR per chip of a
    served-cache tensor (tt/served.py: ND-sharded one 32-token record per shard, round-robin over the DRAM banks),
    TP-dedup (update_padded_kv_cache's sp x tp order, module docstring). ``slab``: a (slot, layer) of the serving
    engine's cache (tt/runners/kv_contract.py); None = a tensor of its own (the ladder). ``write`` and ``carry`` run
    per chunk on the device; the ``*_host`` methods are the harness / serving boundary."""

    WIDTH = 512

    def __init__(self, mesh, tables: WindowTables, max_seq: int, slab: CacheSlab | None = None):
        self.mesh, self.tables, self.geo = mesh, tables, tables.geo
        g = self.geo
        self.rows = g.cache_rows(max_seq)
        self.max_seq = self.rows * g.n
        self.tp_axis = TP_AXIS if g.tp > 1 else None
        if slab is None:
            slab = CacheSlab(alloc_served(mesh, self.WIDTH, self.rows, 32, 1, dedup=True), owned=True)
        assert (slab.rows, slab.width) == (self.rows, self.WIDTH), (slab.rows, slab.width, self.rows)
        self.slab = slab
        # GLUE_FOLD=on (P.19): the previous chunk's kv rows (next start, [1, 1, S, 512] bf16 ROW_MAJOR) when it was
        # whole: the next chunk's carry is their last 128 rows (one slice instead of the cache read). Dropped by any
        # other write (another start, a host load) and by free().
        self._tail = None
        # CKV_PACK_BEFORE_GATHER=on (P.11): the previous chunk's last 128 rows as its 4 bfloat8_b window blocks (next
        # start, [1, 1, 4, WINDOW_BLOCK_BYTES] uint8) when it was whole: the next chunk's carry blocks as they are
        self._tail_blocks = None
        # (allocated here, at load time, and written in place: a buffer allocated mid-forward and kept shifts the
        # later steps' addresses, DS41-146)
        self._tail_buf = None
        # WINDOW_KV_FORMAT=bfp8_tiles (P.18): the kept tail (load time; written by the last chip only): the carry of
        # chip 0's region, [1, 1, 4, WINDOW_BLOCK_BYTES] uint8, and the start it is the carry of (None: not valid)
        self.rtail, self.rtail_end = None, None
        if window_region():
            self.rtail = ttnn.zeros(
                [1, 1, WINDOW // 32, WINDOW_BLOCK_BYTES],
                dtype=ttnn.uint8,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=DRAM,
            )
        if tables.carry_local() is not None:
            self._tail_buf = ttnn.zeros(
                [1, 1, WINDOW // 32, WINDOW_BLOCK_BYTES],
                dtype=ttnn.uint8,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=DRAM,
            )

    def keep_blocks(self, gathered: ttnn.Tensor, start: int, end: int) -> None:
        """``gathered``: the chunk's window blocks [1, 1, S / 32, WINDOW_BLOCK_BYTES]; when the chunk is whole, its last
        4 blocks (positions [end - 128, end)) are copied into the load-time tail buffer as the next chunk's carry."""
        self._drop_tail()
        if end == start + self.geo.chunk and self._tail_buf is not None:
            nb, c = gathered.shape[-2], WINDOW // 32
            ttnn.slice(
                gathered,
                [0, 0, nb - c, 0],
                [1, 1, nb, WINDOW_BLOCK_BYTES],
                memory_config=DRAM,
                output_tensor=self._tail_buf,
                sub_core_grids=sub_cores(gathered),
            )
            self._tail_blocks = (end, self._tail_buf)

    def carry_blocks(self, start: int) -> ttnn.Tensor | None:
        """The kept carry blocks of positions [start - 128, start), or None (not kept: read the rows, ``carry``)."""
        tail = self._tail_blocks
        if tail is not None and tail[0] == start and start >= WINDOW:
            return tail[1]
        return None

    def carry_owner(self, start: int) -> tuple[int, int] | None:
        """(owner chip q (chip-linear), its local row) when the carry rows [start - 128, start) are 128 rows of ONE
        owner stripe (``carry``'s single-piece case), else None."""
        g = self.geo
        if start < WINDOW:
            return None
        s_idx, u = divmod(start - WINDOW, g.stripe)
        if g.stripe - u < WINDOW:
            return None
        return s_idx % g.n, (s_idx // g.n) * g.stripe + u

    def write_local(self, rows_rm: ttnn.Tensor, start: int, end: int) -> None:
        """CKV_PACK_BEFORE_GATHER=on (P.11): rows_rm [1, 1, W, 512] bf16 ROW_MAJOR = the SP row's own block of the
        chunk, rows at or past end zero; start % S == 0, so the server's order of SP row r is its own block (the
        index row srv_row(0) is the identity): ``write`` without the gathered rows and the embedding."""
        g = self.geo
        assert start % g.chunk == 0 and start < end <= start + g.chunk and end <= self.max_seq, (start, end)
        assert rows_rm.shape[-2] == g.w, (rows_rm.shape, g.w)
        if self._tail_blocks is not None and self._tail_blocks[0] != start:
            self._drop_tail()
        if self._tail is not None:  # (a bf16 tail is never this path's carry)
            self._drop_tail()
        ttnn.bringup.update_padded_kv_cache(
            self.slab.tensor,
            rows_rm,
            slot_idx=self.slab.slot,
            layer_idx=self.slab.row,
            num_layers=self.slab.layers,
            kv_actual_global=start,
            cluster_axis=SP_AXIS,
            valid_global=end,
            tp_axis=self.tp_axis,
            max_grid=max_grid(),
        )

    def keep_rows(self, kv_rm: ttnn.Tensor, start: int, end: int) -> None:
        """Takes ``kv_rm`` (the chunk's rows just written, natural order): kept as the next chunk's carry source
        when the chunk is whole, else freed."""
        self._drop_tail()
        if end == start + self.geo.chunk:
            self._tail = (end, kv_rm)
        else:
            dealloc(kv_rm)

    def steady(self, start: int | None) -> None:
        """FIX.16 (owner 2026-10-31, framework F236 ``steady_state``): timing only. ``start``: the next run at
        ``start`` takes the path of a chunk that follows a whole previous chunk ending at ``start``: the kept carry
        is marked as that chunk's tail, in the load-time buffers it would be in (bfp8_tiles: ``rtail``; P.11 pack
        path: ``_tail_buf``; GLUE_FOLD bf16 rows: the kept rows, when present). The carry's VALUES are whatever
        those buffers hold (the last run's tail, not positions [start - 128, start)). None: the normal path (every
        kept tail dropped, the carry read from the served cache)."""
        if start is None or start < WINDOW:
            self._drop_tail()
            return
        if self.rtail is not None:  # WINDOW_KV_FORMAT=bfp8_tiles: the last chip's send writes it every chunk
            self.rtail_end = start
        if self._tail_buf is not None and window_bfp8():  # (keep_blocks' buffer: written by every whole chunk)
            self._tail_blocks = (start, self._tail_buf)
        if self._tail is not None:  # (GLUE_FOLD, bf16 table: rows of the forward's own; re-keyed, never allocated)
            self._tail = (start, self._tail[1])

    def _drop_tail(self) -> None:
        self._tail_blocks = None  # (its buffer is the cache's own, kept)
        self.rtail_end = None  # (likewise)
        if self._tail is not None:
            if self._tail[1].is_allocated():
                dealloc(self._tail[1])
            self._tail = None

    # ------------------------------------------------------------------ per chunk (device)
    def write(self, kv_rm: ttnn.Tensor, start: int, end: int) -> None:
        """kv_rm [1, 1, S, 512] bf16 ROW_MAJOR, natural chunk order, replicated, rows at or past end zero: positions
        [start, end) into the cache (the last record's pad rows as zeros)."""
        g = self.geo
        assert start % 32 == 0 and start < end <= start + g.chunk and end <= self.max_seq, (start, end, self.max_seq)
        k = (start % g.chunk) // 32
        if self._tail is not None and self._tail[0] != start:  # another start may overwrite the kept rows' positions
            self._drop_tail()
        if self._tail_blocks is not None:  # (P.11: kept blocks are never this path's carry)
            self._drop_tail()
        idx = self.tables.srv_row(k)
        rows = ttnn.bringup.embedding(
            ttnn.reshape(idx, [1, g.w], sub_core_grids=sub_cores(idx)),
            ttnn.reshape(kv_rm, [g.chunk, self.WIDTH], sub_core_grids=sub_cores(kv_rm)),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=DRAM,
            max_grid=max_grid(),
        )
        self.tables.release_srv(idx)
        slab = ttnn.reshape(rows, [1, 1, g.w, self.WIDTH], sub_core_grids=sub_cores(rows))
        ttnn.bringup.update_padded_kv_cache(
            self.slab.tensor,
            slab,
            slot_idx=self.slab.slot,
            layer_idx=self.slab.row,
            num_layers=self.slab.layers,
            kv_actual_global=start,
            cluster_axis=SP_AXIS,
            valid_global=end,
            tp_axis=self.tp_axis,
            max_grid=max_grid(),
        )
        dealloc(rows)

    def _gather_all(self, t: ttnn.Tensor) -> ttnn.Tensor:
        """[1, 1, n, 512] TILE per chip -> [1, 1, N n, 512] on every chip, chip-linear order r TP + c."""
        out = t
        if self.geo.tp > 1:
            out = ccl.all_gather(out, 2, TP_AXIS)
        if self.geo.sp > 1:
            nxt = ccl.all_gather(out, 2, SP_AXIS)
            if out is not t:
                dealloc(out)
            out = nxt
        return out

    def carry(self, start: int) -> ttnn.Tensor:
        """The window rows of positions [start - 128, start) ([1, 1, 128, 512] bf16 ROW_MAJOR, replicated; rows of
        positions < 0 are zero) read from the cache: per owner stripe (one or two when the stripe is >= 128 tokens,
        more on a narrower one), the same local rows sliced on every chip, gathered over the mesh, the owner's block
        kept."""
        g = self.geo
        n = min(WINDOW, start)
        if n == 0:
            return self.tables.zero_carry
        tail = self._tail
        if tail is not None and tail[0] == start and n == WINDOW:  # GLUE_FOLD=on: the previous chunk's last rows
            s = tail[1].shape[-2]
            return ttnn.slice(
                tail[1], [0, 0, s - WINDOW, 0], [1, 1, s, self.WIDTH], memory_config=DRAM, sub_core_grids=sub_cores()
            )
        pieces = (
            []
            if n == WINDOW
            else [
                ttnn.slice(
                    self.tables.zero_carry,
                    [0, 0, 0, 0],
                    [1, 1, WINDOW - n, 512],
                    sub_core_grids=sub_cores(self.tables.zero_carry),
                )
            ]
        )
        p = start - n
        while p < start:
            s_idx, u = divmod(p, g.stripe)
            take = min(g.stripe - u, start - p)
            q, b = s_idx % g.n, s_idx // g.n
            loc = b * g.stripe + u
            sl = self.slab.slice_rows(loc, loc + take)
            tl = ttnn.to_layout(sl, ttnn.TILE_LAYOUT, memory_config=DRAM, sub_core_grids=sub_cores(sl))
            dealloc(sl)
            gat = self._gather_all(tl)
            if gat is not tl:
                dealloc(tl)
            mine = ttnn.slice(
                gat,
                [0, 0, q * take, 0],
                [1, 1, (q + 1) * take, self.WIDTH],
                memory_config=DRAM,
                sub_core_grids=sub_cores(gat),
            )
            dealloc(gat)
            pieces.append(to_rm(mine, memory_config=DRAM))
            dealloc(mine)
            p += take
        if len(pieces) == 1:
            return pieces[0]
        out = ttnn.concat(pieces, dim=2, memory_config=DRAM, sub_core_grids=sub_cores(pieces))
        for t in pieces:
            if t is not self.tables.zero_carry:
                dealloc(t)
        return out

    # ------------------------------------------------------------------ harness / serving boundary (host)
    def load_host(self, rows: torch.Tensor, p0: int) -> None:
        """Write host rows [n, 512] at positions [p0, p0 + n) (p0 % 32 == 0) through the device write path."""
        g = self.geo
        assert p0 % 32 == 0, p0
        self._drop_tail()
        n = rows.shape[0]
        for a in range(0, n, g.chunk):
            m = min(g.chunk, n - a)
            nat = torch.zeros(g.chunk, self.WIDTH)
            nat[:m] = rows[a : a + m].float()
            srv = nat[g.server_rows(p0 + a)].reshape(g.sp, 1, g.w, self.WIDTH)
            slab = _upload(self.mesh, srv, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, (0, None))
            ttnn.bringup.update_padded_kv_cache(
                self.slab.tensor,
                slab,
                slot_idx=self.slab.slot,
                layer_idx=self.slab.row,
                num_layers=self.slab.layers,
                kv_actual_global=p0 + a,
                cluster_axis=SP_AXIS,
                valid_global=p0 + a + m,
                tp_axis=self.tp_axis,
                max_grid=max_grid(),
            )
            dealloc(slab)

    def to_host(self, length: int) -> torch.Tensor:
        """Natural-order rows of positions [0, ceil32(length)) as host bf16 [ceil32(length), 512]."""
        g = self.geo
        shards = self.slab.host_rows()
        blocks = self.rows // g.stripe
        stacked = torch.stack([s.view(blocks, g.stripe, self.WIDTH) for s in shards], dim=1)  # [blocks, N, stripe]
        nat = stacked.reshape(blocks * g.n * g.stripe, self.WIDTH)
        return nat[: -(-length // 32) * 32].to(torch.bfloat16)

    def ring_host(self, length: int, window: int = WINDOW) -> torch.Tensor:
        """The reference's window_kv state: the ring [window, 512], slot p % window = position p of the last window
        positions before ``length`` (zeros elsewhere)."""
        ring = torch.zeros(window, self.WIDTH)
        if length:
            nat = self.to_host(length).float()
            pos = torch.arange(max(0, length - window), length)
            ring[pos % window] = nat[pos]
        return ring

    def load_ring(self, ring: torch.Tensor, length: int, window: int = WINDOW) -> None:
        """Load the reference's ring view of positions [length - window, length) into the cache."""
        n = min(window, length)
        if n == 0:
            return
        p0 = length - n
        assert p0 % 32 == 0, f"prefix length {length}: the ring's first position {p0} is not 32-aligned"
        pos = torch.arange(p0, length)
        self.load_host(ring.float()[pos % window], p0)

    def free(self) -> None:
        """Frees the tensor only when it is the cache's own (never the engine's)."""
        self._drop_tail()
        if self._tail_buf is not None and self._tail_buf.is_allocated():
            dealloc(self._tail_buf)
        self._tail_buf = None
        if self.rtail is not None and self.rtail.is_allocated():
            dealloc(self.rtail)
        self.rtail = None
        self.slab.free()


class TtWindowAttention:
    """attention of one window-only layer. ``__call__(q_resid, kv, start, end, cache)``: q_resid [1, 1, S/SP, 1280]
    fp32 and kv [1, 1, S/SP, 512] bf16 (row-split, TP-replicated, natural blocks of positions [start, start + S)),
    end = actual_end -> attn_out [1, 1, S/SP, H/TP] fp32 (hidden split); writes the chunk's window rows into ``cache``.
    Does not consume its inputs."""

    rope_sites = (None, None)  # FORK_OUT_BUFFERS=on (P.16): the q / output rope's load-time outputs (host_alloc)

    INTERLEAVED = False  # q / table rope tail order in the sparse_sdpa (False: [even | odd], the table transformed)

    def __init__(self, mesh, cfg, w: dict, rope: RopeTable, tables: WindowTables):
        self.mesh, self.cfg, self.rope, self.tables, self.geo = mesh, cfg, rope, tables, tables.geo
        g = self.geo
        self.heads, self.d, self.rd = cfg.n_heads, cfg.head_dim, cfg.rope_dim
        self.groups, self.rank = cfg.o_groups, cfg.o_lora_rank
        assert cfg.window == WINDOW and self.d == TtWindowCache.WIDTH, (cfg.window, self.d)
        assert self.heads % g.tp == 0 and self.groups % g.tp == 0, (self.heads, self.groups, g.tp)
        self.h_l, self.g_l = self.heads // g.tp, self.groups // g.tp
        # ATTN_SP32_PROJ=on (P.9): the projections on the chip's own query rows with every head / group (module
        # docstring); built on ROPE_IMPL=on's stock rope path (off there: the path before P.9)
        self.sp32 = settings.get("ATTN_SP32_PROJ") == "on" and rope.stock
        if self.sp32:
            self.h_l, self.g_l = self.heads, self.groups
        # OWN_ROWS=on (P.23, tt/model.py sets it): attn_out stays the chip's own rows [1, 1, S/(SP TP), H] (the
        # residual's layout then): no exit all_to_all
        self.keep_rows = False

        def wdim(d):  # sp32: every weight replicated
            return None if self.sp32 else d

        self.hpg = self.heads // self.groups  # heads per wo_a group
        self.ckc = attn_compute_config(mesh)  # wq_b, sparse_sdpa, wo_a, wo_b: ATTN_FIDELITY (P.7)
        self.scale = self.d**-0.5
        # NORM_GEMM_DTYPE=on (P.21, tt/model.py): q_resid arrives as wq_b's bfloat8_b input with the scale folded in
        self.q_prescaled = False
        assert self.heads % 32 == 0, f"sparse_sdpa needs a multiple of 32 heads per chip ({self.heads} after reshard)"
        self.o_sum = settings.get("ATTN_O_SUM")
        self.fold = settings.get("GLUE_FOLD") == "on"  # P.19: window-path glue (_table)
        # ATTN_GLUE=on (P.15): q through nlp_create_q_heads_split, its tail interleaved by a PairPerm matmul after the
        # bf16 cast, so the window layers too read the table in its stored (interleaved) order; the output
        # de-interleaved by PairPerm on the bf16 sdpa output, kept bf16 (the inverse rope writes bf16, wo_a's input
        # rows permuted at load to [nope | re | ro]: no re-interleave), nlp_concat_heads after wo_a
        self.glue = glue()
        # ROPE_IMPL=on (P.17): q / output through one stock rotary op each (tt/rope.rope_stock), every weight in
        # checkpoint order, the table read as stored (interleaved)
        self.stock = rope.stock
        if self.glue:
            assert self.stock or rope.pairs is not None, "ATTN_GLUE=on: the RopeTable was built without its PairPerm"
            self.pairs = rope.pairs
            self.INTERLEAVED = True  # the table as stored (no per-chunk transform of the window rows)
        # WINDOW_KV_FORMAT=bfp8_blocks (DS41-150): the window region as bfloat8_b blocks, rows as stored
        self.win8 = window_bfp8()
        assert not self.win8 or self.INTERLEAVED, "WINDOW_KV_FORMAT=bfp8_blocks reads the rows as stored (ATTN_GLUE=on)"
        # WINDOW_KV_FORMAT=bfp8_tiles (P.18): this layer's own window region on every chip (load time, per layer: a
        # sender writes a receiver's region of the same layer only, tt/window_send.py), rows as stored
        self.region = None
        if window_region():
            assert self.INTERLEAVED, "WINDOW_KV_FORMAT=bfp8_tiles reads the rows as stored (ATTN_GLUE=on)"
            assert tables.send is not None, "WINDOW_KV_FORMAT=bfp8_tiles needs a mesh of more than one chip"
            self.region = ttnn.zeros(
                [1, 1, tables.region_pages(), WINDOW_BLOCK_BYTES],
                dtype=ttnn.uint8,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=DRAM,
            )
        perm = torch.arange(self.d) if self.stock else deinterleave_perm(self.d, self.rd)
        # wq_b [heads * 512, q_lora] -> each head's rope rows de-interleaved -> [q_lora, heads * 512], columns split
        # over TP (chip c: heads [c H/TP, (c+1) H/TP)); bf16 exact (e4m3 x 2^k)
        wq_b = w["wq_b"].float().view(self.heads, self.d, -1)[:, perm].reshape(self.heads * self.d, -1)
        assert torch.equal(wq_b.bfloat16().float(), wq_b), "wq_b must be bf16-exact"
        wq_b_t = wq_b.t().reshape(1, 1, -1, self.heads * self.d).contiguous()
        # DENSE_WEIGHTS (P.13): bf16 (exact fp8 values) or bfloat8_b (the checkpoint's 8-bit width)
        self.wq_b = _upload(mesh, wq_b_t, dense_dtype(wq_b_t, "attn.wq_b"), ttnn.TILE_LAYOUT, wdim((None, 3)))
        # wo_a [groups * rank, hpg * 512] -> [1, groups, hpg * 512, rank], groups split over TP; bf16-exact
        wo_a = w["wo_a"].float().view(self.groups, self.rank, -1)
        assert torch.equal(wo_a.bfloat16().float(), wo_a), "wo_a must be bf16-exact"
        if (
            self.glue and not self.stock
        ):  # each head's input columns in the order the bf16 output writes: [nope | re | ro]
            wo_a = wo_a.view(self.groups, self.rank, self.hpg, self.d)[..., perm].reshape(self.groups, self.rank, -1)
        wo_a_t = wo_a.transpose(1, 2).unsqueeze(0).contiguous()
        self.wo_a = _upload(mesh, wo_a_t, dense_dtype(wo_a_t, "attn.wo_a"), ttnn.TILE_LAYOUT, wdim((None, 1)))
        # wo_b [H, groups * rank] -> [groups * rank, H], K (the chip's groups) split over TP
        wo_b = w["wo_b"].float()
        assert torch.equal(wo_b.bfloat16().float(), wo_b), "wo_b must be bf16-exact"
        wo_b_t = wo_b.t().reshape(1, 1, -1, wo_b.shape[0]).contiguous()
        self.wo_b = _upload(mesh, wo_b_t, dense_dtype(wo_b_t, "attn.wo_b"), ttnn.TILE_LAYOUT, wdim((None, 2)))
        self.hidden = wo_b.shape[0]
        # the sink in sparse_sdpa's units (it multiplies by the op's scale, 1 here): every head on every chip
        sink = w["attn_sink"].float().reshape(1, 1, 1, self.heads)
        self.sink = _upload(mesh, sink, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)

    # ------------------------------------------------------------------ parts
    def _reshard(self, t, in_dim, out_dim):
        if self.geo.tp == 1:
            return t
        return ttnn.experimental.all_to_all_async_generic(
            t, in_dim=in_dim, out_dim=out_dim, num_links=2, memory_config=DRAM, cluster_axis=TP_AXIS
        )

    def _rope_rows(self, start: int, length: int):
        """cos / sin of the chunk's positions as the projections read them: the chip row's S/SP rows, or the chip's
        own S/(SP TP) query rows under ATTN_SP32_PROJ=on (P.9). Release with ``self.rope.free``."""
        return self.rope.rows(start, length, tp_split=self.sp32 and self.geo.tp > 1)

    def _window_rows(self, kv: ttnn.Tensor, start: int, end: int, cache: TtWindowCache, blocks: bool = False):
        """Writes the chunk's rows into the cache; returns (carry [1, 1, 128, 512], kv_rm [1, 1, S, 512]) bf16
        ROW_MAJOR, replicated, natural order (rope tail as stored). Release both with ``_release_window``.
        ``blocks`` (WINDOW_KV_FORMAT=bfp8_blocks): also writes [carry | chunk] into the window blocks."""
        g = self.geo
        full = kv
        if g.tp > 1 and kv.shape[-2] == g.q_rows:
            # ATTN_SP32=on (P.10): kv of the chip's own rows: over TP first (the SP row's block in order), then SP
            full = ccl.all_gather(kv, 2, TP_AXIS)  # [1, 1, S/SP, 512] bf16
        if g.sp > 1:
            nxt = ccl.all_gather(full, 2, SP_AXIS)  # [1, 1, S, 512] bf16
            if full is not kv:
                dealloc(full)
            full = nxt
        if self.fold and end - start == g.chunk:  # GLUE_FOLD=on (P.19): no pad rows, the mask would multiply by 1
            masked = full
        else:
            keep = ttnn.lt(self.tables.iota, float(end - start), dtype=ttnn.bfloat16, memory_config=DRAM)
            masked = ttnn.multiply(
                full, keep, dtype=ttnn.bfloat16, memory_config=DRAM, sub_core_grids=sub_cores(full)
            )  # pad rows -> 0 (exact)
            dealloc(keep)
            if full is not kv:
                dealloc(full)
        kv_rm = to_rm(masked, memory_config=DRAM)
        one = blocks and lowbit_host()  # LOWBIT_HOST=on (P.12): [carry | chunk] in one call below
        if blocks and not one:
            self._put_blocks(masked, WINDOW)
        if masked is not kv and not one:
            dealloc(masked)
        cache.write(kv_rm, start, end)
        carry = cache.carry(start)
        if blocks:
            ct = ttnn.to_layout(carry, ttnn.TILE_LAYOUT, memory_config=DRAM, sub_core_grids=sub_cores(carry))
            if one:
                self._put_blocks(masked, 0, prefix=ct)
                if masked is not kv:
                    dealloc(masked)
            else:
                self._put_blocks(ct, 0)
            dealloc(ct)
        return carry, kv_rm

    def _packs(self, start: int) -> bool:
        """CKV_PACK_BEFORE_GATHER=on (P.11) for this chunk's window blocks (WINDOW_KV_FORMAT=bfp8_blocks)."""
        return self.win8 and self.tables.packs("blocks", self.geo.w, start)

    def _window_packed(self, kv: ttnn.Tensor, start: int, end: int, cache: TtWindowCache) -> None:
        """CKV_PACK_BEFORE_GATHER=on (P.11; WINDOW_KV_FORMAT=bfp8_blocks, start % S == 0): ``_window_rows(blocks=True)``
        with each SP row packing its own rows: kv gathered over TP (bf16, the SP row's block [W, 512]) -> pad rows
        zeroed -> the served cache written locally (bf16 rows, ``write_local``) -> the block's W / 32 bfloat8_b blocks
        (ckv_bfp4 compact into the load-time local buffer) -> all_gather of the blocks over SP -> slice_write into the
        window blocks' chunk region; the carry region from the previous chunk's kept last 4 blocks, else from the
        carry rows packed (``carry``). The same records as ``_window_rows`` (gap bytes aside, DS41-144). Does not
        consume ``kv``."""
        g, tables = self.geo, self.tables
        prepared = settings.get("CKV_BFP4_PREPARED") == "on"
        full = kv
        if g.tp > 1 and kv.shape[-2] == g.q_rows:  # ATTN_SP32=on (P.10): the chip's own rows -> the SP row's block
            full = ccl.all_gather(kv, 2, TP_AXIS, site="attn.kv_tp")  # [1, 1, W, 512] bf16
        masked = tables.mask_local(full, end - start)
        if masked is not full and full is not kv:
            dealloc(full)
        rm = to_rm(masked, memory_config=DRAM)
        cache.write_local(rm, start, end)
        dealloc(rm)
        ttnn.bringup.ckv_bfp4(
            masked,
            tables.packed("blocks", g.w),
            0,
            dtype=ttnn.bfloat8_b,
            blocks=True,
            memory_config=DRAM,
            prepared=prepared,
            compact=True,
            max_grid=max_grid(),
        )
        if masked is not kv:
            dealloc(masked)
        gat = tables.gather_packed("blocks", g.w)  # [1, 1, S / 32, WINDOW_BLOCK_BYTES]
        blocks = tables.window_blocks()
        c, nb = WINDOW // 32, gat.shape[-2]
        ttnn.bringup.slice_write(
            gat, blocks, [0, 0, c, 0], [1, 1, c + nb, WINDOW_BLOCK_BYTES], [1, 1, 1, 1], max_grid=max_grid()
        )
        tail = cache.carry_blocks(start)
        owner = cache.carry_owner(start) if tail is None and tables.carry_local() is not None else None
        if tail is not None:
            ttnn.bringup.slice_write(
                tail, blocks, [0, 0, 0, 0], [1, 1, c, WINDOW_BLOCK_BYTES], [1, 1, 1, 1], max_grid=max_grid()
            )
        elif owner is not None and ccl.fabric():
            # the carry from the served cache, packed before its gather: every chip packs the same local rows (the
            # owner's are the carry), the 4 blocks gathered over the mesh, the owner's kept (``carry``'s bf16 path)
            q, loc = owner
            rm16 = ttnn.ROW_MAJOR_LAYOUT
            sl = cache.slab.slice_rows(  # (HOST_ALLOC=on, P.15: into a load-time output)
                loc,
                loc + WINDOW,
                host_alloc.out(self.mesh, "attn.carry_rows", [1, 1, WINDOW, 512], ttnn.bfloat16, rm16),
            )
            tl = ttnn.to_layout(sl, ttnn.TILE_LAYOUT, memory_config=DRAM, sub_core_grids=sub_cores(sl))
            dealloc(sl)
            ttnn.bringup.ckv_bfp4(
                tl,
                tables.carry_local(),
                0,
                dtype=ttnn.bfloat8_b,
                blocks=True,
                memory_config=DRAM,
                prepared=prepared,
                compact=True,
                max_grid=max_grid(),
            )
            dealloc(tl)
            every = tables.gather_carry()
            mo = host_alloc.out(self.mesh, "attn.carry_mine", [1, 1, c, WINDOW_BLOCK_BYTES], ttnn.uint8, rm16)
            lo, hi = [0, 0, q * c, 0], [1, 1, (q + 1) * c, WINDOW_BLOCK_BYTES]
            if mo is None:
                mine = ttnn.slice(every, lo, hi, memory_config=DRAM, sub_core_grids=sub_cores(every))
            else:  # (HOST_ALLOC=on, P.15: the owner's 4 blocks into a load-time output)
                ttnn.slice(every, lo, hi, memory_config=DRAM, output_tensor=mo, sub_core_grids=sub_cores(every))
                mine = mo
            ttnn.bringup.slice_write(
                mine, blocks, [0, 0, 0, 0], [1, 1, c, WINDOW_BLOCK_BYTES], [1, 1, 1, 1], max_grid=max_grid()
            )
            dealloc(mine)
        else:
            carry = cache.carry(start)
            ct = ttnn.to_layout(carry, ttnn.TILE_LAYOUT, memory_config=DRAM, sub_core_grids=sub_cores(carry))
            if carry is not tables.zero_carry:
                dealloc(carry)
            ttnn.bringup.ckv_bfp4(
                ct,
                blocks,
                0,
                dtype=ttnn.bfloat8_b,
                blocks=True,
                memory_config=DRAM,
                prepared=prepared,
                compact=True,
                max_grid=max_grid(),
            )
            dealloc(ct)
        cache.keep_blocks(gat, start, end)  # the next chunk's carry: this chunk's last 4 blocks (whole chunks)
        if not tables.is_kept(gat):
            dealloc(gat)

    def _window_region(self, kv: ttnn.Tensor, start: int, end: int, cache: TtWindowCache) -> None:
        """WINDOW_KV_FORMAT=bfp8_tiles (P.18): the chunk's window rows into the served cache (unchanged: kv gathered
        over TP, the SP row's block written locally at chunk-aligned starts; else gathered over the mesh and written
        in the server's order) and into this layer's per-chip region ``region`` [carry 128 | own q_rows] (bfloat8_b
        blocks of 32 rows, nearest even, as bfp8_blocks): the chip's own rows rounded at row 128 (ckv_bfp4), the
        carry by ONE neighbour send (tt/window_send.py): chip q's last 128 rows -> chip q + 1, the last chip's kept
        tail (``cache.rtail``: positions [start - 128, start)) -> chip 0. The kept tail is this layer's previous
        chunk when it ended at ``start`` (whole chunks); else the carry rows are read from the served cache into it:
        on the owner chip alone when the owner is the last chip (every chunk-aligned start: the previous chunk's last
        stripe), else gathered over the mesh (``carry``). Does not consume ``kv``."""
        g, tables = self.geo, self.tables
        prepared = settings.get("CKV_BFP4_PREPARED") == "on"
        own = kv
        if g.tp > 1 and kv.shape[-2] != g.q_rows:  # (ATTN_SP32=off: the SP row's block, TP-replicated)
            own = ttnn.mesh_partition(kv, dim=2, cluster_axis=TP_AXIS, memory_config=DRAM)
        # 1. the served cache, as before (its TP gather and write)
        if start % g.chunk == 0 and g.w in tables._sp_iota:
            full = kv
            if g.tp > 1 and kv.shape[-2] == g.q_rows:
                full = ccl.all_gather(kv, 2, TP_AXIS, site="attn.kv_tp")  # [1, 1, W, 512] bf16
            masked = tables.mask_local(full, end - start)
            if masked is not full and full is not kv:
                dealloc(full)
            rm = to_rm(masked, memory_config=DRAM)
            if masked is not kv:
                dealloc(masked)
            cache.write_local(rm, start, end)
            dealloc(rm)
        else:  # (prefix hits at other starts: every chip needs other rows of the chunk)
            full = kv
            if g.tp > 1 and kv.shape[-2] == g.q_rows:
                full = ccl.all_gather(kv, 2, TP_AXIS)
            if g.sp > 1:
                nxt = ccl.all_gather(full, 2, SP_AXIS)
                if full is not kv:
                    dealloc(full)
                full = nxt
            keep = ttnn.lt(tables.iota, float(end - start), dtype=ttnn.bfloat16, memory_config=DRAM)
            masked = ttnn.multiply(full, keep, dtype=ttnn.bfloat16, memory_config=DRAM, sub_core_grids=sub_cores(full))
            dealloc(keep)
            if full is not kv:
                dealloc(full)
            kv_rm = to_rm(masked, memory_config=DRAM)
            dealloc(masked)
            cache.write(kv_rm, start, end)
            dealloc(kv_rm)
        # 2. the chip's own rows -> region rows [128, 128 + q_rows) (pad rows zero, as the served rows)
        om = tables.mask_chip(own, end - start)
        ttnn.bringup.ckv_bfp4(
            om,
            self.region,
            WINDOW,
            dtype=ttnn.bfloat8_b,
            blocks=True,
            memory_config=DRAM,
            prepared=prepared,
            compact=True,
            max_grid=max_grid(),
        )
        if om is not own:
            dealloc(om)
        if own is not kv:
            dealloc(own)
        # 3. chip 0's carry: the kept tail, or the served rows packed into it (start 0: no carry is read)
        if cache.rtail_end != start and start > 0:
            owner = cache.carry_owner(start)
            if owner is not None and owner[0] == g.n - 1:
                loc = owner[1]
                rm16 = ttnn.ROW_MAJOR_LAYOUT
                sl = cache.slab.slice_rows(  # (HOST_ALLOC=on, P.15: into a load-time output)
                    loc,
                    loc + WINDOW,
                    host_alloc.out(self.mesh, "attn.carry_rows", [1, 1, WINDOW, 512], ttnn.bfloat16, rm16),
                )
                ct = ttnn.to_layout(sl, ttnn.TILE_LAYOUT, memory_config=DRAM, sub_core_grids=sub_cores(sl))
                dealloc(sl)
            else:
                carry = cache.carry(start)
                ct = ttnn.to_layout(carry, ttnn.TILE_LAYOUT, memory_config=DRAM, sub_core_grids=sub_cores(carry))
                if carry is not tables.zero_carry:
                    dealloc(carry)
            ttnn.bringup.ckv_bfp4(
                ct,
                cache.rtail,
                0,
                dtype=ttnn.bfloat8_b,
                blocks=True,
                memory_config=DRAM,
                prepared=prepared,
                compact=True,
                max_grid=max_grid(),
            )
            dealloc(ct)
        # 4. the neighbour send; the last chip keeps its own last rows: the next chunk's carry when this one is whole
        tables.send(self.region, cache.rtail)
        cache.rtail_end = end if end == start + g.chunk else None

    def _put_blocks(self, rows_tile: ttnn.Tensor, row0: int, prefix: ttnn.Tensor | None = None) -> None:
        """WINDOW_KV_FORMAT=bfp8_blocks: bf16 TILE rows (replicated, not consumed) at row0 of the window blocks,
        rounded to nearest even onto the bfloat8_b grid. ``prefix`` (LOWBIT_HOST=on, P.12): the carry's TILE rows,
        written first in the same call ([carry | chunk] at row0 0), the fork's compact program (no per-core runtime
        args; row0 is always 0 here, so one prepared program)."""
        prepared = settings.get("CKV_BFP4_PREPARED") == "on"  # P.7: the program built once, row0 patched
        extra = dict(prefix=prefix, compact=True) if prefix is not None else {}
        ttnn.bringup.ckv_bfp4(
            rows_tile,
            self.tables.window_blocks(),
            row0,
            dtype=ttnn.bfloat8_b,
            blocks=True,
            memory_config=DRAM,
            prepared=prepared,
            **extra,
            max_grid=max_grid(),
        )

    def _release_window(self, carry, kv_rm, cache: TtWindowCache, start: int, end: int) -> None:
        if carry is not self.tables.zero_carry:
            dealloc(carry)
        if self.fold:  # GLUE_FOLD=on (P.19): the rows stay as the next chunk's carry source
            cache.keep_rows(kv_rm, start, end)
        else:
            dealloc(kv_rm)

    def _table(self, kv: ttnn.Tensor, start: int, end: int, cache: TtWindowCache) -> ttnn.Tensor:
        """Writes the chunk's rows into the cache; returns the sparse table [1, 1, 128 + S, 512] bf16 ROW_MAJOR
        (carry | chunk), rope tail de-interleaved (as stored, the reference's interleaved order, when
        ``INTERLEAVED``: q is then interleaved too)."""
        carry, kv_rm = self._window_rows(kv, start, end, cache)
        table = ttnn.concat([carry, kv_rm], dim=2, memory_config=DRAM, sub_core_grids=sub_cores())
        self._release_window(carry, kv_rm, cache, start, end)
        if self.INTERLEAVED:
            return table
        # the rope tail [interleaved pairs] -> [even | odd] (q's order): ROW_MAJOR transpose, exact
        rows, half, r0 = table.shape[2], self.rd // 2, self.d - self.rd
        tail = ttnn.slice(
            table, [0, 0, 0, r0], [1, 1, rows, self.d], memory_config=DRAM, sub_core_grids=sub_cores(table)
        )
        tt = ttnn.transpose(
            ttnn.reshape(tail, [1, rows, half, 2], sub_core_grids=sub_cores(tail)), -2, -1, memory_config=DRAM
        )
        dealloc(tail)
        nope = ttnn.slice(table, [0, 0, 0, 0], [1, 1, rows, r0], memory_config=DRAM, sub_core_grids=sub_cores(table))
        dealloc(table)
        out = ttnn.concat(
            [nope, ttnn.reshape(tt, [1, 1, rows, self.rd], sub_core_grids=sub_cores(tt))],
            dim=-1,
            memory_config=DRAM,
            sub_core_grids=sub_cores(),
        )
        dealloc(nope)
        dealloc(tt)
        return out

    def _q_heads(self, q_resid: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """q_resid -> scaled, roped q of the chip's heads [1, H/TP, S/SP, 512] fp32 (rope tail [even | odd])."""
        rows = q_resid.shape[-2]
        xq = gemm_input(q_resid)  # GEMM_INPUTS (FIX.7)
        qf = mm.linear(
            xq, self.wq_b, site="attn.wq_b", dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=DRAM
        )
        if xq is not q_resid:
            dealloc(xq)
        q4 = ttnn.transpose(
            ttnn.reshape(qf, [1, rows, self.h_l, self.d], sub_core_grids=sub_cores(qf)), 1, 2, memory_config=DRAM
        )
        dealloc(qf)
        qr = rope_heads(q4, cos, sin, interleave=self.INTERLEAVED)  # the table's order
        dealloc(q4)
        qs = ttnn.multiply(qr, self.scale, dtype=ttnn.float32, memory_config=DRAM, sub_core_grids=sub_cores(qr))
        dealloc(qr)
        return qs

    def _q_glue(self, q_resid: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """ATTN_GLUE=on: q_resid -> scaled, roped q of the chip's heads [1, H/TP, S/SP, 512] bf16, the rope tail
        interleaved (the table's stored order). The values of ``_q_heads`` + the bf16 cast, bit for bit: the heads
        split by nlp_create_q_heads_split (a data move), RoPE on the [.., 64] tail only, the scale's multiply writing
        bf16 (= ttnn.typecast's RNE), the interleave a PairPerm matmul on the bf16 tail."""
        xq = gemm_input(q_resid)  # GEMM_INPUTS (FIX.7)
        qf = mm.linear(
            xq, self.wq_b, site="attn.wq_b", dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=DRAM
        )
        if xq is not q_resid:
            dealloc(xq)
        qn, qr = ttnn.bringup.nlp_create_q_heads_split(
            qf, num_heads=self.h_l, split_head_dim=self.d - self.rd, memory_config=DRAM, max_grid=max_grid()
        )
        dealloc(qf)
        rot = rope_heads(qr, cos, sin)  # [re | ro] fp32
        dealloc(qr)
        tb = ttnn.multiply(rot, self.scale, dtype=ttnn.bfloat16, memory_config=DRAM, sub_core_grids=sub_cores(rot))
        dealloc(rot)
        nb = ttnn.multiply(qn, self.scale, dtype=ttnn.bfloat16, memory_config=DRAM, sub_core_grids=sub_cores(qn))
        dealloc(qn)
        ti = self.pairs(tb, self.pairs.to_inter)
        dealloc(tb)
        q = ttnn.concat([nb, ti], dim=-1, memory_config=DRAM, sub_core_grids=sub_cores())
        dealloc(nb)
        dealloc(ti)
        return q

    def _q_proj(self, q_resid: ttnn.Tensor) -> ttnn.Tensor:
        """ROPE_IMPL=on (P.17): q_resid -> wq_b's bf16 output [1, 1, rows, H/TP 512] (every head of a chip's rows under
        ATTN_SP32_PROJ): the softmax scale 512^-0.5 folded into the bf16 cast of the input (one rounding, as the cast
        was), wq_b writing bf16."""
        # GEMM_INPUTS (FIX.7): the scale's multiply writes wq_b's input in bfloat8_b (the reference's fp8 act quant)
        # HOST_ALLOC=on (P.15): xq and qb written into load-time outputs (tt/host_alloc.py), when their spec has one
        if self.q_prescaled and q_resid.dtype == gemm_dtype():
            # NORM_GEMM_DTYPE=on (P.21): q_norm wrote q_resid x 512^-0.5 as wq_b's bfloat8_b input (one rounding)
            xo = xq = q_resid
        else:
            xo = host_alloc.out(self.mesh, "attn.xq", q_resid.shape, gemm_dtype())
        if xo is None:
            xq = ttnn.multiply(
                q_resid, self.scale, dtype=gemm_dtype(), memory_config=DRAM, sub_core_grids=sub_cores(q_resid)
            )
        elif xo is not q_resid:  # (the object itself: the op hands back a new wrapper of its buffer, DS41-149)
            ttnn.multiply(
                q_resid,
                self.scale,
                dtype=gemm_dtype(),
                memory_config=DRAM,
                output_tensor=xo,
                sub_core_grids=sub_cores(q_resid),
            )
            xq = xo
        qb = mm.linear(
            xq,
            self.wq_b,
            site="attn.wq_b",
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.ckc,
            memory_config=DRAM,
            out=host_alloc.mm_out(self.mesh, "attn.qb", xq, self.wq_b, ttnn.bfloat16),
        )
        if xq is not q_resid:  # (the step's input: the block frees it)
            dealloc(xq)
        return qb

    def _q_stock(self, q_resid: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """ROPE_IMPL=on (P.17): q_resid -> scaled, roped q of the chip's heads [1, H/TP, S/SP, 512] bf16, checkpoint
        (interleaved) order: wq_b (``_q_proj``), the heads split by nlp_create_q_heads_split, ONE stock rotary op on
        the [.., 64] tails with the nope prepended (tt/rope.rope_stock)."""
        qb = self._q_proj(q_resid)
        qn, qr = ttnn.bringup.nlp_create_q_heads_split(
            qb, num_heads=self.h_l, split_head_dim=self.d - self.rd, memory_config=DRAM, max_grid=max_grid()
        )
        dealloc(qb)
        q = rope_stock(
            qr, cos, sin, self.rope.trans, prefix=qn, site=self.rope_sites[0], mesh=self.mesh
        )  # (FORK_OUT_BUFFERS)
        dealloc(qr)
        dealloc(qn)
        return q

    def _q(self, q_resid: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """q_resid -> scaled, roped q heads [1, H, S/(SP TP), 512] bf16 ROW_MAJOR (after the reshard)."""
        if self.sp32:  # ATTN_SP32_PROJ=on (P.9): the chip's own rows, every head: no reshard
            xr = q_resid
            # (ATTN_SP32=on, P.10: q_resid is already the chip's own rows)
            if self.geo.tp > 1 and q_resid.shape[-2] != self.geo.q_rows:
                xr = ttnn.mesh_partition(q_resid, dim=2, cluster_axis=TP_AXIS, memory_config=DRAM)
            if rope_chains():
                # ROPE_CHAINS=on (P.22): ONE fork call reads each head straight from wq_b's output (tile columns
                # 16 h .. 16 h + 15), ropes its tail and writes sparse_sdpa's ROW_MAJOR q (no split, no untilize)
                qb = self._q_proj(xr)
                if xr is not q_resid:
                    dealloc(xr)
                q_rm = rope_chain(
                    qb,
                    cos,
                    sin,
                    self.rope.trans,
                    self.d - self.rd,
                    input_heads=self.h_l,
                    output_layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                dealloc(qb)
                return q_rm
            qb = self._q_stock(xr, cos, sin)
            if xr is not q_resid:
                dealloc(xr)
            q_rm = to_rm(qb, memory_config=DRAM)
            dealloc(qb)
            return q_rm
        if self.glue:
            qb = self._q_stock(q_resid, cos, sin) if self.stock else self._q_glue(q_resid, cos, sin)
            qa = self._reshard(qb, in_dim=1, out_dim=2)
            if qa is not qb:
                dealloc(qb)
            q_rm = to_rm(qa, memory_config=DRAM)
            dealloc(qa)
            return q_rm
        qs = self._q_heads(q_resid, cos, sin)
        qb = ttnn.typecast(qs, ttnn.bfloat16, memory_config=DRAM, sub_core_grids=sub_cores_tiled(qs))
        dealloc(qs)
        qa = self._reshard(qb, in_dim=1, out_dim=2)
        if qa is not qb:
            dealloc(qb)
        q_rm = to_rm(qa, memory_config=DRAM)
        dealloc(qa)
        return q_rm

    def _sdpa(self, q_rm, table, index, window=None):
        if window is None and table.dtype == ttnn.uint8:  # WINDOW_KV_FORMAT=bfp8_blocks, a window-only layer
            return ttnn.bringup.sparse_sdpa(
                q_rm,
                table,
                index,
                self.d,
                kv_format=ttnn.bringup.SparseKVFormat.ROW_BFP8,
                scale=1.0,
                k_chunk_size=SDPA_K_CHUNK,
                compute_kernel_config=self.ckc,
                attention_sink=self.sink,
                max_grid=max_grid(),
            )
        if window is not None:  # CKV_FORMAT=bfp4_rows: bfloat4_b row records + the window, bf16 rows or bfloat8_b
            # blocks (WINDOW_KV_FORMAT) (sdpa fork ROW_BFP4)
            return ttnn.bringup.sparse_sdpa(
                q_rm,
                table,
                index,
                self.d,
                kv_format=ttnn.bringup.SparseKVFormat.ROW_BFP4,
                scale=1.0,
                k_chunk_size=SDPA_K_CHUNK,
                compute_kernel_config=self.ckc,
                attention_sink=self.sink,
                kv_window=window,
                max_grid=max_grid(),
            )
        return ttnn.transformer.sparse_sdpa(
            q_rm,
            table,
            index,
            self.d,
            kv_format=ttnn.transformer.SparseKVFormat.BF16,
            scale=1.0,
            k_chunk_size=SDPA_K_CHUNK,
            compute_kernel_config=self.ckc,
            attention_sink=self.sink,
        )

    def _out(self, o_rm: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """sparse_sdpa output [1, H, S/(SP TP), 512] bf16 ROW_MAJOR -> attn_out [1, 1, S/SP, H/TP] (fp32; bf16 under
        ATTN_GLUE=on with CCL_IMPL=fabric_bf16)."""
        if self.sp32 and rope_chains():
            # ROPE_CHAINS=on (P.22): ONE fork call reads sparse_sdpa's ROW_MAJOR output, applies the inverse rope and
            # writes wo_a's input: the heads of a wo_a group side by side, in its GEMM dtype (bfloat8_b packed as
            # ttnn.typecast packs it; no to_layout, nlp_concat_heads or typecast)
            rows = o_rm.shape[2]
            odt = gemm_dtype()
            out = host_alloc.out(self.mesh, "attn.oa", [1, self.g_l, rows, self.hpg * self.d], odt)  # (HOST_ALLOC)
            oa = rope_chain(
                o_rm,
                cos,
                sin,
                self.rope.trans_inv,
                self.d - self.rd,
                out=out,
                output_dtype=odt,
                group_heads=self.hpg,
            )
            return self._out_wo(oa, rows)
        ot = ttnn.to_layout(o_rm, ttnn.TILE_LAYOUT, memory_config=DRAM, sub_core_grids=sub_cores(o_rm))
        if self.sp32:  # ATTN_SP32_PROJ=on (P.9): every head of the chip's rows stays here
            oq = rope_stock(
                ot, cos, sin, self.rope.trans_inv, site=self.rope_sites[1], mesh=self.mesh
            )  # (FORK_OUT_BUFFERS)
            dealloc(ot)
            return self._out_proj(oq, oq.shape[2])
        oh = self._reshard(ot, in_dim=2, out_dim=1)  # [1, H/TP, S/SP, 512]
        if oh is not ot:
            dealloc(ot)
        if self.stock:  # ROPE_IMPL=on (P.17): the inverse rope on the bf16 tail in place, one stock op
            oq = rope_stock(oh, cos, sin, self.rope.trans_inv)
            dealloc(oh)
            return self._out_proj(oq, oq.shape[2])
        if self.glue:
            return self._out_glue(oh, cos, sin)
        of = ttnn.typecast(oh, ttnn.float32, memory_config=DRAM, sub_core_grids=sub_cores_tiled(oh))
        dealloc(oh)
        return self._out_heads(of, cos, sin)

    def _out_glue(self, oh: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """ATTN_GLUE=on: the bf16 attention output of the chip's heads [1, H/TP, S/SP, 512] (consumed) -> attn_out.
        Only the [.., 64] rope tail leaves bf16: de-interleaved by a PairPerm matmul writing fp32 (exact), inverse
        RoPE in fp32 whose last ops write bf16 in [re | ro] order, which wo_a's permuted rows read."""
        b, h, rows, width = oh.shape
        r0 = width - self.rd
        nope = ttnn.slice(oh, [0, 0, 0, 0], [b, h, rows, r0], memory_config=DRAM, sub_core_grids=sub_cores(oh))
        tail = ttnn.slice(oh, [0, 0, 0, r0], [b, h, rows, width], memory_config=DRAM, sub_core_grids=sub_cores(oh))
        dealloc(oh)
        if self.INTERLEAVED:
            tf = self.pairs(tail, self.pairs.to_de, dtype=ttnn.float32)
        else:
            tf = ttnn.typecast(tail, ttnn.float32, memory_config=DRAM, sub_core_grids=sub_cores_tiled(tail))
        dealloc(tail)
        tr = rope_heads(tf, cos, sin, inverse=True, dtype=ttnn.bfloat16)
        dealloc(tf)
        oq = ttnn.concat([nope, tr], dim=-1, memory_config=DRAM, sub_core_grids=sub_cores())
        dealloc(nope)
        dealloc(tr)
        return self._out_proj(oq, rows)

    def _out_heads(self, of: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """attention output of the chip's heads [1, H/TP, S/SP, 512] fp32 (rope tail [even | odd]; consumed) ->
        attn_out [1, 1, S/SP, H/TP] fp32."""
        rows = of.shape[2]
        if self.INTERLEAVED:  # the table's (and so the output's) tail is interleaved: [even | odd] for the rope
            od = deinterleave_tail(of, self.rd // 2)
            dealloc(of)
            of = od
        ou = rope_heads(of, cos, sin, inverse=True, interleave=True)  # the reference's order again
        dealloc(of)
        oq = cast(ou, ttnn.bfloat16)
        dealloc(ou)
        return self._out_proj(oq, rows)

    def _out_proj(self, oq: ttnn.Tensor, rows: int) -> ttnn.Tensor:
        """The bf16 output heads [1, H/TP, S/SP, 512] (consumed) -> grouped wo_a -> bf16 -> wo_b -> the
        sum over TP: attn_out [1, 1, S/SP, H/TP] (fp32; bf16 under ATTN_GLUE=on with CCL_IMPL=fabric_bf16)."""
        # heads of one wo_a group side by side: [G/TP, hpg, S/SP, 512] -> [1, G/TP, S/SP, hpg * 512]
        og = ttnn.bringup.nlp_concat_heads(
            ttnn.reshape(oq, [self.g_l, self.hpg, rows, self.d], sub_core_grids=sub_cores(oq)),
            memory_config=DRAM,
            max_grid=max_grid(),
        )
        dealloc(oq)
        og = ttnn.reshape(og, [1, self.g_l, rows, self.hpg * self.d], sub_core_grids=sub_cores(og))
        # GEMM_INPUTS (FIX.7): wo_a reads bfloat8_b, the reference's fp8 q-dq of the attention output before wo_a
        oa = gemm_input(og, host_alloc.out(self.mesh, "attn.oa", og.shape, gemm_dtype()))  # (HOST_ALLOC, P.15)
        if oa is not og:
            dealloc(og)
        return self._out_wo(oa, rows)

    def _out_wo(self, oa: ttnn.Tensor, rows: int) -> ttnn.Tensor:
        """wo_a's input [1, G/TP, S/SP, hpg 512] in its GEMM dtype (consumed) -> wo_a -> wo_b -> attn_out (as
        ``_out_proj``)."""
        # BF16_ACTIVATIONS=on (P.20): wo_a writes bf16 (the GPU code's bf16 einsum), no cast after it; GEMM_INPUTS=bfp8
        # (FIX.7): it writes wo_b's input as bfloat8_b (the reference's fp8 act quant of wo_b's input) directly
        ydt = gemm_dtype() if (self.glue and bf16_activations()) else ttnn.float32
        y = mm.matmul(
            oa,
            self.wo_a,
            site="attn.wo_a",
            dtype=ydt,
            compute_kernel_config=self.ckc,
            memory_config=DRAM,
            out=host_alloc.mm_out(self.mesh, "attn.wo_a", oa, self.wo_a, ydt),
        )
        dealloc(oa)
        # [1, G/TP, S/SP, rank] -> [1, 1, S/SP, G/TP * rank] (the chip's K range of wo_b, group-major)
        if self.glue:
            # ATTN_GLUE=on: nlp_concat_heads (a data move) in place of G/TP slices + concat, on the bf16 cast
            yb = gemm_input(y)
            if yb is not y:
                dealloc(y)
            yq = ttnn.bringup.nlp_concat_heads(yb, memory_config=DRAM, max_grid=max_grid())
            dealloc(yb)
        else:
            parts = [
                ttnn.slice(
                    y, [0, k, 0, 0], [1, k + 1, rows, self.rank], memory_config=DRAM, sub_core_grids=sub_cores(y)
                )
                for k in range(self.g_l)
            ]
            dealloc(y)
            yt = (
                ttnn.concat(parts, dim=-1, memory_config=DRAM, sub_core_grids=sub_cores(parts))
                if len(parts) > 1
                else parts[0]
            )
            if len(parts) > 1:
                for t in parts:
                    dealloc(t)
            yq = gemm_input(yt)
            if yq is not yt:
                dealloc(yt)
        if self.sp32:
            # ATTN_SP32_PROJ=on (P.9): the whole sum over K here: wo_b writes the GPU code's bf16 output of the chip's
            # rows [1, 1, S/(SP TP), H]; ONE all_to_all over TP (split H, concat rows) -> [1, 1, S/SP, H/TP]
            z = mm.linear(
                yq, self.wo_b, site="attn.wo_b", dtype=act_dtype(), compute_kernel_config=self.ckc, memory_config=DRAM
            )
            dealloc(yq)
            if self.keep_rows:  # OWN_ROWS=on (P.23): the residual rests on these rows
                return z
            out = self._reshard(z, in_dim=2, out_dim=3)  # concat rows (in_dim), split H (out_dim)
            if out is not z:
                dealloc(z)
            return out
        # CCL_IMPL=fabric_bf16 (P.6): the partial written in bf16 for the bf16 reduce-scatter (tt/ccl.py:partial_dtype)
        zdt = ccl.partial_dtype(self.mesh) if self.o_sum == "gather_add" else ttnn.float32
        z = mm.linear(yq, self.wo_b, site="attn.wo_b", dtype=zdt, compute_kernel_config=self.ckc, memory_config=DRAM)
        dealloc(yq)
        if self.geo.tp == 1:
            return z
        if self.o_sum == "reduce_scatter":
            out = ttnn.reduce_scatter(z, dim=3, cluster_axis=TP_AXIS, memory_config=DRAM)
        else:
            # ATTN_GLUE=on: the bf16 reduce-scatter's result as is (tt/residual.py casts only where it needs fp32)
            out = sum_cols(self.mesh, z, fp32=not self.glue)
        dealloc(z)
        return out

    def chunk_of(self, rows: int) -> int:
        """The chunk's token count from the rows of q_resid a chip holds: S/SP (row-split, TP-replicated) or the
        chip's own S/(SP TP) (ATTN_SP32=on, P.10)."""
        g = self.geo
        return rows * g.sp * (g.tp if rows == g.q_rows else 1)

    def __call__(self, q_resid, kv, start: int, end: int | None, cache: TtWindowCache) -> ttnn.Tensor:
        g = self.geo
        s = self.chunk_of(q_resid.shape[-2])
        assert s == g.chunk, f"chunk of {s} rows; the window tables were built for {g.chunk}"
        end = start + s if end is None else int(end)
        cos, sin = self._rope_rows(start, s)
        index = self.tables.index[g.start_class(start)]
        if self.region is not None:  # WINDOW_KV_FORMAT=bfp8_tiles (P.18): the chip's own region, ROW_BFP8
            self._window_region(kv, start, end, cache)
            table, index = self.region, self.tables.region_index[g.start_class(start)]
        elif self.win8:  # WINDOW_KV_FORMAT=bfp8_blocks: the window blocks, read as kv_format ROW_BFP8
            if self._packs(start):  # CKV_PACK_BEFORE_GATHER=on (P.11)
                self._window_packed(kv, start, end, cache)
            else:
                carry, kv_rm = self._window_rows(kv, start, end, cache, blocks=True)
                self._release_window(carry, kv_rm, cache, start, end)
            table = self.tables.window_blocks()
        else:
            table = self._table(kv, start, end, cache)
        q_rm = self._q(q_resid, cos, sin)
        o = self._sdpa(q_rm, table, index)
        dealloc(q_rm)
        if not self.win8 and table is not self.region:
            dealloc(table)
        out = self._out(o, cos, sin)
        dealloc(o)
        self.rope.free(cos, sin)  # INDEXER_GLUE=on: the chunk's memoized rows stay
        return out


def attention_weights(ck, layer: int) -> dict:
    p = f"layers.{layer}.attn."
    return {
        "wq_b": ck.weight(p + "wq_b.weight", torch.float32),
        "wo_a": ck.weight(p + "wo_a.weight", torch.float32),
        "wo_b": ck.weight(p + "wo_b.weight", torch.float32),
        "attn_sink": ck.weight(p + "attn_sink", torch.float32),
    }


def build_attention(mesh, ck, cfg, layer: int, max_seq: int, chunk: int, rope=None, tables=None):
    """The window attention of a ratio-0 layer (its own rope table and window tables unless shared ones are given)."""
    from .kv import rope_freqs

    assert not cfg.ratio(layer), f"layer {layer} has compress ratio {cfg.ratio(layer)}: not a window-only layer"
    rope = rope or RopeTable(mesh, rope_freqs(cfg, layer), max_seq)
    tables = tables or WindowTables(mesh, chunk)
    return TtWindowAttention(mesh, cfg, attention_weights(ck, layer), rope, tables)
