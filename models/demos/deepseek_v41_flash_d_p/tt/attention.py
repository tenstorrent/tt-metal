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

from . import ccl
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
from .rope import RopeTable, deinterleave_perm, deinterleave_tail, glue, rope_heads, rope_stock
from .served import CacheSlab, DeviceScalars, alloc_served
from .settings import settings

WINDOW = 128  # sliding window (cfg.window); the table's carry region
# WINDOW_KV_FORMAT=bfp8_blocks: one page per 32 consecutive window rows (ttnn.bringup.ckv_bfp4 blocks=True, 512 wide):
# [32 x 32 shared exponents | a copy 32 B on, at 1056 | 32 x 512 sign-magnitude bytes at 2112]
WINDOW_BLOCK_BYTES = 18496


def lowbit_host() -> bool:
    """LOWBIT_HOST=on (P.12): the window blocks' carry and chunk in one compact ckv_bfp4 call (``_put_blocks``)."""
    return settings.get("LOWBIT_HOST") == "on"


def window_bfp8() -> bool:
    """WINDOW_KV_FORMAT=bfp8_blocks (DS41-150): the window region as bfloat8_b blocks of 32 rows (``WindowTables``)."""
    return settings.get("WINDOW_KV_FORMAT") == "bfp8_blocks"


SENTINEL = -1  # uploaded as int32 -1, read through uint32: 0xFFFFFFFF, sparse_sdpa's "no key"
SDPA_K_CHUNK = 128


def _mapper(mesh, dims):
    return ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims)


def _upload(mesh, t, dtype, layout, dims=None):
    return ttnn.from_torch(
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
        idx = ttnn.slice(self.srv, [0, 0, k, 0], [1, 1, k + 1, self.geo.w], memory_config=DRAM)
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

    def keep_rows(self, kv_rm: ttnn.Tensor, start: int, end: int) -> None:
        """Takes ``kv_rm`` (the chunk's rows just written, natural order): kept as the next chunk's carry source
        when the chunk is whole, else freed."""
        self._drop_tail()
        if end == start + self.geo.chunk:
            self._tail = (end, kv_rm)
        else:
            dealloc(kv_rm)

    def _drop_tail(self) -> None:
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
        idx = self.tables.srv_row(k)
        rows = ttnn.embedding(
            ttnn.reshape(idx, [1, g.w]),
            ttnn.reshape(kv_rm, [g.chunk, self.WIDTH]),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=DRAM,
        )
        self.tables.release_srv(idx)
        slab = ttnn.reshape(rows, [1, 1, g.w, self.WIDTH])
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
            return ttnn.slice(tail[1], [0, 0, s - WINDOW, 0], [1, 1, s, self.WIDTH], memory_config=DRAM)
        pieces = [] if n == WINDOW else [ttnn.slice(self.tables.zero_carry, [0, 0, 0, 0], [1, 1, WINDOW - n, 512])]
        p = start - n
        while p < start:
            s_idx, u = divmod(p, g.stripe)
            take = min(g.stripe - u, start - p)
            q, b = s_idx % g.n, s_idx // g.n
            loc = b * g.stripe + u
            sl = self.slab.slice_rows(loc, loc + take)
            tl = ttnn.to_layout(sl, ttnn.TILE_LAYOUT, memory_config=DRAM)
            dealloc(sl)
            gat = self._gather_all(tl)
            if gat is not tl:
                dealloc(tl)
            mine = ttnn.slice(gat, [0, 0, q * take, 0], [1, 1, (q + 1) * take, self.WIDTH], memory_config=DRAM)
            dealloc(gat)
            pieces.append(ttnn.to_layout(mine, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM))
            dealloc(mine)
            p += take
        if len(pieces) == 1:
            return pieces[0]
        out = ttnn.concat(pieces, dim=2, memory_config=DRAM)
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
        self.slab.free()


class TtWindowAttention:
    """attention of one window-only layer. ``__call__(q_resid, kv, start, end, cache)``: q_resid [1, 1, S/SP, 1280]
    fp32 and kv [1, 1, S/SP, 512] bf16 (row-split, TP-replicated, natural blocks of positions [start, start + S)),
    end = actual_end -> attn_out [1, 1, S/SP, H/TP] fp32 (hidden split); writes the chunk's window rows into ``cache``.
    Does not consume its inputs."""

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

        def wdim(d):  # sp32: every weight replicated
            return None if self.sp32 else d

        self.hpg = self.heads // self.groups  # heads per wo_a group
        self.ckc = attn_compute_config(mesh)  # wq_b, sparse_sdpa, wo_a, wo_b: ATTN_FIDELITY (P.7)
        self.scale = self.d**-0.5
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
            masked = ttnn.multiply(full, keep, dtype=ttnn.bfloat16, memory_config=DRAM)  # pad rows -> 0 (exact)
            dealloc(keep)
            if full is not kv:
                dealloc(full)
        kv_rm = ttnn.to_layout(masked, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        one = blocks and lowbit_host()  # LOWBIT_HOST=on (P.12): [carry | chunk] in one call below
        if blocks and not one:
            self._put_blocks(masked, WINDOW)
        if masked is not kv and not one:
            dealloc(masked)
        cache.write(kv_rm, start, end)
        carry = cache.carry(start)
        if blocks:
            ct = ttnn.to_layout(carry, ttnn.TILE_LAYOUT, memory_config=DRAM)
            if one:
                self._put_blocks(masked, 0, prefix=ct)
                if masked is not kv:
                    dealloc(masked)
            else:
                self._put_blocks(ct, 0)
            dealloc(ct)
        return carry, kv_rm

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
        table = ttnn.concat([carry, kv_rm], dim=2, memory_config=DRAM)
        self._release_window(carry, kv_rm, cache, start, end)
        if self.INTERLEAVED:
            return table
        # the rope tail [interleaved pairs] -> [even | odd] (q's order): ROW_MAJOR transpose, exact
        rows, half, r0 = table.shape[2], self.rd // 2, self.d - self.rd
        tail = ttnn.slice(table, [0, 0, 0, r0], [1, 1, rows, self.d], memory_config=DRAM)
        tt = ttnn.transpose(ttnn.reshape(tail, [1, rows, half, 2]), -2, -1, memory_config=DRAM)
        dealloc(tail)
        nope = ttnn.slice(table, [0, 0, 0, 0], [1, 1, rows, r0], memory_config=DRAM)
        dealloc(table)
        out = ttnn.concat([nope, ttnn.reshape(tt, [1, 1, rows, self.rd])], dim=-1, memory_config=DRAM)
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
        q4 = ttnn.transpose(ttnn.reshape(qf, [1, rows, self.h_l, self.d]), 1, 2, memory_config=DRAM)
        dealloc(qf)
        qr = rope_heads(q4, cos, sin, interleave=self.INTERLEAVED)  # the table's order
        dealloc(q4)
        qs = ttnn.multiply(qr, self.scale, dtype=ttnn.float32, memory_config=DRAM)
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
        qn, qr = ttnn.experimental.nlp_create_q_heads_split(
            qf, num_heads=self.h_l, split_head_dim=self.d - self.rd, memory_config=DRAM
        )
        dealloc(qf)
        rot = rope_heads(qr, cos, sin)  # [re | ro] fp32
        dealloc(qr)
        tb = ttnn.multiply(rot, self.scale, dtype=ttnn.bfloat16, memory_config=DRAM)
        dealloc(rot)
        nb = ttnn.multiply(qn, self.scale, dtype=ttnn.bfloat16, memory_config=DRAM)
        dealloc(qn)
        ti = self.pairs(tb, self.pairs.to_inter)
        dealloc(tb)
        q = ttnn.concat([nb, ti], dim=-1, memory_config=DRAM)
        dealloc(nb)
        dealloc(ti)
        return q

    def _q_stock(self, q_resid: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """ROPE_IMPL=on (P.17): q_resid -> scaled, roped q of the chip's heads [1, H/TP, S/SP, 512] bf16, checkpoint
        (interleaved) order: the softmax scale 512^-0.5 folded into the bf16 cast of the input (one rounding, as the
        cast was), wq_b writing bf16, the heads split by nlp_create_q_heads_split, ONE stock rotary op on the [.., 64]
        tails with the nope prepended (tt/rope.rope_stock)."""
        # GEMM_INPUTS (FIX.7): the scale's multiply writes wq_b's input in bfloat8_b (the reference's fp8 act quant)
        xq = ttnn.multiply(q_resid, self.scale, dtype=gemm_dtype(), memory_config=DRAM)
        qb = mm.linear(
            xq, self.wq_b, site="attn.wq_b", dtype=ttnn.bfloat16, compute_kernel_config=self.ckc, memory_config=DRAM
        )
        dealloc(xq)
        qn, qr = ttnn.experimental.nlp_create_q_heads_split(
            qb, num_heads=self.h_l, split_head_dim=self.d - self.rd, memory_config=DRAM
        )
        dealloc(qb)
        q = rope_stock(qr, cos, sin, self.rope.trans, prefix=qn)
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
            qb = self._q_stock(xr, cos, sin)
            if xr is not q_resid:
                dealloc(xr)
            q_rm = ttnn.to_layout(qb, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
            dealloc(qb)
            return q_rm
        if self.glue:
            qb = self._q_stock(q_resid, cos, sin) if self.stock else self._q_glue(q_resid, cos, sin)
            qa = self._reshard(qb, in_dim=1, out_dim=2)
            if qa is not qb:
                dealloc(qb)
            q_rm = ttnn.to_layout(qa, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
            dealloc(qa)
            return q_rm
        qs = self._q_heads(q_resid, cos, sin)
        qb = ttnn.typecast(qs, ttnn.bfloat16, memory_config=DRAM)
        dealloc(qs)
        qa = self._reshard(qb, in_dim=1, out_dim=2)
        if qa is not qb:
            dealloc(qb)
        q_rm = ttnn.to_layout(qa, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
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
        ot = ttnn.to_layout(o_rm, ttnn.TILE_LAYOUT, memory_config=DRAM)
        if self.sp32:  # ATTN_SP32_PROJ=on (P.9): every head of the chip's rows stays here
            oq = rope_stock(ot, cos, sin, self.rope.trans_inv)
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
        of = ttnn.typecast(oh, ttnn.float32, memory_config=DRAM)
        dealloc(oh)
        return self._out_heads(of, cos, sin)

    def _out_glue(self, oh: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """ATTN_GLUE=on: the bf16 attention output of the chip's heads [1, H/TP, S/SP, 512] (consumed) -> attn_out.
        Only the [.., 64] rope tail leaves bf16: de-interleaved by a PairPerm matmul writing fp32 (exact), inverse
        RoPE in fp32 whose last ops write bf16 in [re | ro] order, which wo_a's permuted rows read."""
        b, h, rows, width = oh.shape
        r0 = width - self.rd
        nope = ttnn.slice(oh, [0, 0, 0, 0], [b, h, rows, r0], memory_config=DRAM)
        tail = ttnn.slice(oh, [0, 0, 0, r0], [b, h, rows, width], memory_config=DRAM)
        dealloc(oh)
        if self.INTERLEAVED:
            tf = self.pairs(tail, self.pairs.to_de, dtype=ttnn.float32)
        else:
            tf = ttnn.typecast(tail, ttnn.float32, memory_config=DRAM)
        dealloc(tail)
        tr = rope_heads(tf, cos, sin, inverse=True, dtype=ttnn.bfloat16)
        dealloc(tf)
        oq = ttnn.concat([nope, tr], dim=-1, memory_config=DRAM)
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
        og = ttnn.experimental.nlp_concat_heads(
            ttnn.reshape(oq, [self.g_l, self.hpg, rows, self.d]), memory_config=DRAM
        )
        dealloc(oq)
        og = ttnn.reshape(og, [1, self.g_l, rows, self.hpg * self.d])
        # GEMM_INPUTS (FIX.7): wo_a reads bfloat8_b, the reference's fp8 q-dq of the attention output before wo_a
        oa = gemm_input(og)
        if oa is not og:
            dealloc(og)
        # BF16_ACTIVATIONS=on (P.20): wo_a writes bf16 (the GPU code's bf16 einsum), no cast after it; GEMM_INPUTS=bfp8
        # (FIX.7): it writes wo_b's input as bfloat8_b (the reference's fp8 act quant of wo_b's input) directly
        ydt = gemm_dtype() if (self.glue and bf16_activations()) else ttnn.float32
        y = mm.matmul(oa, self.wo_a, site="attn.wo_a", dtype=ydt, compute_kernel_config=self.ckc, memory_config=DRAM)
        dealloc(oa)
        # [1, G/TP, S/SP, rank] -> [1, 1, S/SP, G/TP * rank] (the chip's K range of wo_b, group-major)
        if self.glue:
            # ATTN_GLUE=on: nlp_concat_heads (a data move) in place of G/TP slices + concat, on the bf16 cast
            yb = gemm_input(y)
            if yb is not y:
                dealloc(y)
            yq = ttnn.experimental.nlp_concat_heads(yb, memory_config=DRAM)
            dealloc(yb)
        else:
            parts = [
                ttnn.slice(y, [0, k, 0, 0], [1, k + 1, rows, self.rank], memory_config=DRAM) for k in range(self.g_l)
            ]
            dealloc(y)
            yt = ttnn.concat(parts, dim=-1, memory_config=DRAM) if len(parts) > 1 else parts[0]
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
        if self.win8:  # WINDOW_KV_FORMAT=bfp8_blocks: the window blocks, read as kv_format ROW_BFP8
            carry, kv_rm = self._window_rows(kv, start, end, cache, blocks=True)
            self._release_window(carry, kv_rm, cache, start, end)
            table = self.tables.window_blocks()
        else:
            table = self._table(kv, start, end, cache)
        q_rm = self._q(q_resid, cos, sin)
        o = self._sdpa(q_rm, table, self.tables.index[g.start_class(start)])
        dealloc(q_rm)
        if not self.win8:
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
