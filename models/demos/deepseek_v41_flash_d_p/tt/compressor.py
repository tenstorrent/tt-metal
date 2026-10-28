# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 KV compressor of a KV-source layer (reference/ds41_ref.py:_compressor; inference Compressor +
Indexer key path) on the SP x TP mesh, and the served caches it writes (serving_contract.md "KV cache":
1_compress_kv_r<r>, 2_index_k_r<r> / 4_index_k_r1, ...).

    kv = x wkv^T, gate = x wgate^T [S, 512] fp32 (x = attn_norm, fp32, no act quant; bf16 weights as stored)
    ratio 2: groups of 2 tokens from position 0; latent_j = sum_t kv_t softmax_t(gate)_t per channel; ratio 1: kv
    latent = compressor.norm(latent) (RMSNorm, fp32)
    ckv = fp4_qdq(rope(latent, pos 2 j), 16, e4m3 scale)                 the compressed KV, a group at its 1st token
    k   = fp4_qdq(rope(k_norm(latent wk^T), pos 2 j), 32, ue8m0 scale)  the index key
    out = [ckv | k] [S / r, 512 + 128]
(the released code's fp4 quant-dequant is not emulated: the device stores ckv / k in bf16 and casts to it, agent
rule 17; FIX.8)

Per chip (r, c) the input is attn_norm's column split [1, 1, S/SP, H/TP] fp32 TILE, natural row blocks (mesh row
r = positions [start + r S/SP, ...)); every start is a multiple of 32 and S/SP is even, so groups never straddle
a chip row and no group is pending at a chunk start (serving_contract.md "Compressor state").

    x [S/SP, H/TP] fp32 @ the chip's K rows of [wkv | wgate]^T (bf16, HiFi4, fp32 acc; ``ttnn.linear``)
                        -> tt/common.tp_sum (all_gather + SFPU add, exact fp32, K-30) -> [S/SP, 1024] replicated
    ratio 2: ttnn.reshape [S/SP / 2, 2048] = [kv_0 | g_0 | kv_1 | g_1] per pair -> slices; the pair softmax per
                        channel as torch computes it (max, exp(g - max), sum, divide; fp32 SFPU) -> kv_0 w_0 +
                        kv_1 w_1 (fp32 SFPU)
    tt/norm.TtDistributedRmsNorm(split=False) compressor.norm (fp32 SFPU)
    (a) tt/rope.rope_deinterleaved at the groups' positions (cos / sin from the load-time compressed-row table,
        positions r j) -> ckv bf16
    (b) ttnn.linear wk^T [512 -> 128] (replicated, bf16 as stored, HiFi4, fp32 acc) -> k_norm (fp32) -> RoPE ->
        k bf16
ROPE_IMPL=on (P.17, the default): (a) / (b) through ONE stock rotary_embedding_indexed each (tt/rope.rope_stock) on
bf16 (the latent cast once; k_norm writes bf16), every weight in checkpoint order; the paragraph below is the off
path.
The rope dims of every producing weight (wkv / wgate rows, compressor.norm, wk's input columns, wk's rows and
k_norm for the key) are permuted at load to [even | odd] (tt/rope.deinterleave_perm), so RoPE is the exact fp32
SFPU composition and returns the reference's interleaved order (DS41-15). The latent stays fp32 up to the bf16
cast (ATTN_GLUE=on: the rope's last ops write bf16, re-interleaved by a PairPerm matmul).

The served caches (``TtServedRows``): bf16 ROW_MAJOR [.., rows / r, width], one 32-token record (32 / r compressed
rows of the full width, the contract's shard [1, 1, 32 / r, width]) per ND shard round-robin over the DRAM banks
(tt/served.py). A compressed row j holds tokens [r j, r j + r). Writes go through the window cache's path
(server-order rows by ttnn.embedding from the load-time index row, then update_padded_kv_cache) in compressed rows;
a ratio-2 start (a multiple of 16 rows) takes the op's metadata path (DS41-54). Compressed KV TP-dedup (W / TP token
stripes), index keys TP-replicated (the plan). Rows of groups not complete by actual_end ((j + 1) r > end) are zero;
the last record ships whole with them zeroed.

No host work in ``__call__`` / ``write``: the per-chunk inputs are integers (start, end) picking load-time tables.
"""

from __future__ import annotations

import torch

import ttnn

from . import ccl
from . import mm_configs as mm
from .ccl import dealloc
from .common import DRAM, SP_AXIS, TP_AXIS, attn_compute_config, bf16_activations, cast, tp_sum
from .norm import TtDistributedRmsNorm
from .rope import (
    PairPerm,
    RowsMemo,
    deinterleave_perm,
    glue,
    indexer_glue,
    pair_table,
    rope_deinterleaved,
    rope_deinterleaved_bf16,
    rope_stock,
    rot_mats,
    stock,
)
from .served import CacheSlab, alloc_served
from .settings import settings


def _upload(mesh, t, dtype, layout, dims=None):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=DRAM,
        mesh_mapper=(
            ttnn.ReplicateTensorToMesh(mesh)
            if dims is None
            else ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims)
        ),
    )


class CompressRope:
    """cos / sin of the compressed rows' positions r j for j in [0, max_seq / r): [1, 1, J, R/2] fp32 ROW_MAJOR
    (ROPE_IMPL=on: [1, 1, J, R] bf16 in the Meta pair order, tt/rope.pair_table, + the trans-mat), replicated, built once at load (the reference's own fp32 table: torch.polar of fp32(pos x freq)). ``rows(g0,
    n)`` slices compressed rows [g0, g0 + n) on the device (any offset: ROW_MAJOR), tiles them and splits them over
    the SP axis, so mesh row r gets the groups of its natural token block."""

    def __init__(self, mesh, freqs: torch.Tensor, max_seq: int, ratio: int):
        from models.demos.deepseek_v41_flash_d_p.reference.ds41_ref import rope_table

        j = -(-max_seq // (32 * ratio)) * 32
        rot = rope_table(freqs.float(), torch.arange(j) * ratio)  # complex64 [J, R/2]
        self.mesh, self.rows_max, self.half, self.ratio = mesh, j, freqs.numel(), ratio
        assert self.half % 32 == 0, f"rope half width {self.half} is not a tile multiple"
        self.stock = stock()  # ROPE_IMPL=on (P.17): [J, R] bf16 pair tables for tt/rope.rope_stock
        if self.stock:
            c2, s2 = pair_table(rot)
            self.width = 2 * self.half
            self.cos = _upload(mesh, c2.reshape(1, 1, j, self.width), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
            self.sin = _upload(mesh, s2.reshape(1, 1, j, self.width), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
            self.trans, _ = rot_mats(mesh)
            self.pairs = None
        else:
            self.width = self.half
            self.cos = _upload(mesh, rot.real.float().reshape(1, 1, j, self.half), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)
            self.sin = _upload(mesh, rot.imag.float().reshape(1, 1, j, self.half), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)
            self.pairs = PairPerm(mesh, self.half) if glue() else None  # ATTN_GLUE=on (P.15)
        self.memo = RowsMemo(indexer_glue())  # INDEXER_GLUE=on (P.16): once per (g0, n) for every source layer

    def free(self, *ts) -> None:
        """Release rows from ``rows``: freed unless memoized (INDEXER_GLUE=on)."""
        self.memo.free(*ts)

    def rows(self, g0: int, n: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """cos, sin of compressed rows [g0, g0 + n), the mesh row's natural block. Release them with ``free``."""
        return self.memo.get((g0, n), lambda: self._rows(g0, n))

    def _rows(self, g0: int, n: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        sp = self.mesh.shape[SP_AXIS]
        assert n % (32 * sp) == 0 and g0 + n <= self.rows_max, (g0, n, sp, self.rows_max)
        out = []
        for t in (self.cos, self.sin):
            sl = ttnn.slice(t, [0, 0, g0, 0], [1, 1, g0 + n, self.width], memory_config=DRAM)
            tl = ttnn.to_layout(sl, ttnn.TILE_LAYOUT, memory_config=DRAM)
            dealloc(sl)
            if sp > 1:
                part = ttnn.mesh_partition(tl, dim=2, cluster_axis=SP_AXIS, memory_config=DRAM)
                dealloc(tl)
                tl = part
            out.append(tl)
        return out[0], out[1]


WINDOW = 128  # the sparse table's window region: 128 carry rows (tt/attention.WINDOW)


def static_tables() -> bool:
    """STATIC_TABLES=on (P.21): one persistent sparse-attention table per KV source (``AttnTable``)."""
    return settings.get("STATIC_TABLES") == "on"


def ckv_bfp4() -> bool:
    """CKV_FORMAT=bfp4_rows (FIX.8): the attention table's history as 288 B bfloat4_b row records (``AttnTable``)."""
    return settings.get("CKV_FORMAT") == "bfp4_rows"


BFP4_ROW_BYTES = 288  # 512 values: 32 shared exponents + 256 bytes of nibbles (tt/ckv_codec.py format B)


class AttnTable:
    """STATIC_TABLES=on (P.21, fusion report item 3): the sparse-attention table of one KV source and slot, allocated
    once with the source's served compressed cache and kept:

        rows [0, hist)                 the compressed history in natural order: group j at row j (hist = the cache's
                                       token capacity / r), the same bf16 rows as the served cache
        rows [base, base + 128 + S)    the window region (base = hist): [carry 128 | chunk S] of the layer running
                                       its attention (the source's compressed layers run in order and reuse it)

    [1, 1, hist + 128 + S, 512] bf16 ROW_MAJOR, replicated (every chip reads any row). Kept in step with the served
    cache by ``TtServedRows``: its device ``write`` appends the chunk's rows (the replicated natural-order rows it was
    given) with one slice_write, its host ``load_host`` writes the loaded rows the same way. A top-k id j is then the
    table row j as it is, and the window index lists are load-time lists offset by ``base`` (``index``). The shape
    never changes between chunks.

    CKV_FORMAT=bfp4_rows (FIX.8): ``buf`` is the history alone, [1, 1, hist, 288] uint8 ROW_MAJOR bfloat4_b row records
    written by ttnn.bringup.ckv_bfp4 (``append_tiles``: the compressor's gathered bf16 tiles, rounded to nearest even;
    it returns the rounded bf16 rows for the served cache), and ``win`` the window region [1, 1, 128 + S, 512] bf16
    (base 0): sparse_sdpa's ROW_BFP4 reads the two, the index list [window ids | top-k ids] as built, not compacted."""

    def __init__(self, mesh, tables, hist: int, width: int):
        g = tables.geo
        self.mesh, self.width = mesh, width
        self.hist = hist
        self.bfp4 = ckv_bfp4()
        self.win = None
        if self.bfp4:
            self.base, self.rows = 0, WINDOW + g.chunk
            self.buf = ttnn.zeros(
                [1, 1, hist, BFP4_ROW_BYTES],
                dtype=ttnn.uint8,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=DRAM,
            )
            self.win = ttnn.zeros(
                [1, 1, self.rows, width],
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=DRAM,
            )
        else:
            self.base, self.rows = hist, hist + WINDOW + g.chunk
            self.buf = ttnn.zeros(
                [1, 1, self.rows, width],
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=DRAM,
            )
        assert self.rows < (1 << 31), self.rows
        self.index = tables.index_lists(self.base)  # {start class: window ids + base}, chip (r, c) its query rows

    def _put(self, rows_rm: ttnn.Tensor, row0: int) -> None:
        """bf16 rows at row0 of the window region (bfp4_rows) or of the whole table (bf16)."""
        dst = self.win if self.bfp4 else self.buf
        n = rows_rm.shape[-2]
        assert rows_rm.shape[-1] == self.width and 0 <= row0 and row0 + n <= self.rows, (rows_rm.shape, row0, self.rows)
        ttnn.experimental.slice_write(rows_rm, dst, [0, 0, row0, 0], [1, 1, row0 + n, self.width], [1, 1, 1, 1])

    def append(self, rows_rm: ttnn.Tensor, g0: int) -> None:
        """History: compressed rows [g0, g0 + n) = rows_rm [1, 1, n, 512] bf16 ROW_MAJOR, replicated (device)."""
        assert not self.bfp4, "CKV_FORMAT=bfp4_rows: the history is written by append_tiles"
        assert g0 + rows_rm.shape[-2] <= self.hist, (g0, rows_rm.shape, self.hist)
        self._put(rows_rm, g0)

    def append_tiles(self, rows_tile: ttnn.Tensor, g0: int, rounded: bool = True):
        """CKV_FORMAT=bfp4_rows: history rows [g0, g0 + n) from rows_tile [1, 1, n, 512] bf16 TILE (n % 32 == 0,
        replicated; not consumed) as row records; ``rounded``: also return their bf16 values (TILE, the caller frees).
        """
        assert self.bfp4 and g0 + rows_tile.shape[-2] <= self.hist, (self.bfp4, g0, rows_tile.shape, self.hist)
        return ttnn.bringup.ckv_bfp4(rows_tile, self.buf, g0, rounded=rounded, memory_config=DRAM)

    def put_window(self, carry: ttnn.Tensor, kv_rm: ttnn.Tensor) -> None:
        """The window region = [carry 128 | chunk S] (device; both bf16 ROW_MAJOR, replicated, not consumed)."""
        assert carry.shape[-2] == WINDOW and kv_rm.shape[-2] == self.rows - self.base - WINDOW, (
            carry.shape,
            kv_rm.shape,
        )
        self._put(carry, self.base)
        self._put(kv_rm, self.base + WINDOW)

    def load_host(self, rows: torch.Tensor, g0: int) -> None:
        """Host boundary: compressed rows [n, 512] of groups [g0, g0 + n) (a loaded prefix). CKV_FORMAT=bfp4_rows:
        encoded on the device like a chunk's rows (the served rows hold bfp4-rounded values: the same records), n
        padded to whole tiles with zero rows (groups past the prefix, written again by their own chunk)."""
        n = rows.shape[0]
        if not n:
            return
        if self.bfp4:
            pad = -(-n // 32) * 32
            assert g0 + pad <= self.hist, (g0, n, self.hist)
            x = torch.zeros(pad, self.width)
            x[:n] = rows.float().reshape(n, self.width)
            t = _upload(self.mesh, x.reshape(1, 1, pad, self.width).to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
            self.append_tiles(t, g0, rounded=False)
            dealloc(t)
            return
        t = _upload(
            self.mesh, rows.reshape(1, 1, -1, self.width).to(torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT
        )
        self.append(t, g0)
        dealloc(t)

    def free(self) -> None:
        for t in [self.buf, *([self.win] if self.win is not None else []), *self.index.values()]:
            if t.is_allocated():
                dealloc(t)
        self.index = {}


class TtServedRows:
    """One served compressed-row cache of one layer and slot (``width`` values per compressed row, ``ratio`` tokens
    per row): a slab [1, 1, rows / r, width] bf16 ROW_MAJOR per chip of a served-cache tensor (tt/served.py), ND-sharded
    one 32-token record (32 / r compressed rows) per shard, the contract's record. ``dedup``: TP-dedup
    (update_padded_kv_cache's sp x tp order: token g on mesh row (g / W) % SP, column (g mod W) / (W / TP), compressed
    row j at the local row of its first token / r); else SP-sharded, TP-replicated (row (g / W) % SP, local row
    ((g / S) W + g mod W) / r). ``slab``: a (slot, layer) of the serving engine's cache tensor; None = a tensor of its
    own (the ladder). ``rows`` counts local TOKEN rows (rows / r compressed rows per chip). ``write`` runs per chunk on
    the device; ``*_host`` is the harness / serving boundary.

    Writes: the chunk's rows in the server's order (the token-order index row of tables.srv, applied to token-width
    pieces, then viewed as compressed rows: every SP run of the server order starts at a multiple of 32), then
    update_padded_kv_cache in compressed rows. A ratio-2 start is a multiple of 16 rows, which the op's scalar path
    refuses; its metadata path (start / valid / slot as 1-element device tensors sliced from load-time tables,
    tt/served.DeviceScalars) works in page rows (DS41-54)."""

    def __init__(
        self,
        mesh,
        tables,
        max_seq: int,
        width: int,
        ratio: int,
        dedup: bool,
        slab: CacheSlab | None = None,
        attn_table: bool = False,
    ):
        assert width % ratio == 0 and 32 % ratio == 0, (width, ratio)
        self.mesh, self.tables, self.geo = mesh, tables, tables.geo
        g = self.geo
        self.width, self.ratio, self.tok_width = width, ratio, width // ratio
        self.tp_axis = TP_AXIS if (dedup and g.tp > 1) else None
        self.stripe = g.stripe if self.tp_axis is not None else g.w  # tokens per chip per W block
        self.owners = g.n if self.tp_axis is not None else g.sp  # chips holding distinct stripes
        self.rows = -(-max_seq // g.chunk) * self.stripe  # local token rows per chip
        self.crows = self.rows // ratio  # local compressed rows per chip
        self.max_seq = self.rows * self.owners
        if slab is None:
            slab = CacheSlab(alloc_served(mesh, width, self.crows, 32 // ratio, 1, dedup=dedup), owned=True)
        assert (slab.rows, slab.width) == (self.crows, width), (slab.rows, slab.width, self.crows, width)
        self.slab = slab
        self.version = 0  # bumped by every write (INDEXER_GLUE=on: tt/sparse_attention.ChunkMemo keys on it)
        # STATIC_TABLES=on (P.21): a KV source's compressed KV keeps the sparse-attention table of its readers in step
        self.table = AttnTable(mesh, tables, self.max_seq // ratio, width) if attn_table and static_tables() else None
        assert not (attn_table and ckv_bfp4() and self.table is None), "CKV_FORMAT=bfp4_rows needs STATIC_TABLES=on"
        if ratio > 1:  # metadata-path values (load time): starts / valid ends in compressed rows, slot ids
            self._pos = tables.scalars(32 // ratio, max_seq // 32 + 1)
            self._slot = tables.scalars(1, slab.slots)

    # ------------------------------------------------------------------ per chunk (device)
    def _put(self, slab: ttnn.Tensor, start: int, end: int) -> None:
        """slab [1, 1, W / r, width] bf16 ROW_MAJOR per SP row (the server's order) -> token positions [start, end) of
        the cache, the last record whole (its rows past the complete groups are zero in ``slab``)."""
        self.version += 1
        args = dict(layer_idx=self.slab.row, num_layers=self.slab.layers, cluster_axis=SP_AXIS, tp_axis=self.tp_axis)
        if self.ratio == 1:
            ttnn.bringup.update_padded_kv_cache(
                self.slab.tensor, slab, slot_idx=self.slab.slot, kv_actual_global=start, valid_global=end, **args
            )
            return
        slot = self._slot.at(self.slab.slot)
        kv = self._pos.at(start // 32)  # start / r
        valid = self._pos.at(-(-end // 32))  # ceil32(end) / r: the last record's rows (zero past the groups)
        ttnn.bringup.update_padded_kv_cache(self.slab.tensor, slab, slot, kv, valid_global=valid, **args)
        for t in (slot, kv, valid):
            self._slot.release(t) if t is slot else self._pos.release(t)

    def write(self, rows_rm: ttnn.Tensor, start: int, end: int, table: bool = True) -> None:
        """rows_rm [1, 1, S / r, width] bf16 ROW_MAJOR, the chunk's compressed rows in natural order, replicated, rows
        of incomplete groups zero: token positions [start, end) into the cache (the last record whole). ``table``
        False: the attention table was written already (CKV_FORMAT=bfp4_rows: AttnTable.append_tiles)."""
        g = self.geo
        assert start % 32 == 0 and start < end <= start + g.chunk and end <= self.max_seq, (start, end, self.max_seq)
        k = (start % g.chunk) // 32
        idx = self.tables.srv_row(k)  # GLUE_FOLD=on (P.19): sliced once per start for the chunk's writes
        rows = ttnn.embedding(
            ttnn.reshape(idx, [1, g.w]),
            ttnn.reshape(rows_rm, [g.chunk, self.tok_width]),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=DRAM,
        )
        self.tables.release_srv(idx)
        slab = ttnn.reshape(rows, [1, 1, g.w // self.ratio, self.width])
        self._put(slab, start, end)
        if self.table is not None and table:  # STATIC_TABLES=on (P.21): the same rows into the table, natural order
            self.table.append(rows_rm, start // self.ratio)
        view = slab.buffer_address() == rows.buffer_address()  # a reshape may return a view (K-1, DS41-52)
        dealloc(rows)
        if not view:
            dealloc(slab)

    # ------------------------------------------------------------------ harness / serving boundary (host)
    def _owner_shards(self) -> list[torch.Tensor]:
        shards = self.slab.host_rows()  # [crows, width] per chip, chip-linear r TP + c
        if self.tp_axis is None:  # replicated over TP: mesh column 0's copy of every SP row
            shards = shards[:: self.geo.tp]
        return shards

    def to_host(self, length: int) -> torch.Tensor:
        """Natural-order compressed rows of token positions [0, ceil32(length)): host bf16 [ceil32(length) / r,
        width]."""
        hs = self.stripe // self.ratio  # compressed rows per owner stripe
        blocks = self.crows // hs
        stacked = torch.stack([s.view(blocks, hs, self.width) for s in self._owner_shards()], dim=1)
        nat = stacked.reshape(blocks * self.owners * hs, self.width)
        return nat[: -(-length // 32) * 32 // self.ratio].to(torch.bfloat16)

    def load_host(self, rows: torch.Tensor, p0: int) -> None:
        """Write host compressed rows [n, width] of token positions [p0, p0 + n r) (p0 % 32 == 0) through the device
        write path (a migrated / golden prefix)."""
        g, r = self.geo, self.ratio
        assert p0 % 32 == 0, p0
        rows = rows.float().reshape(-1, self.width)
        n, per = rows.shape[0], g.chunk // r
        for a in range(0, n, per):
            m = min(per, n - a)
            nat = torch.zeros(per, self.width)
            nat[:m] = rows[a : a + m]
            t0 = p0 + a * r
            srv = g.server_rows(t0)[:, ::r] // r  # compressed rows in the server's order
            slab = _upload(
                self.mesh,
                nat[srv].reshape(g.sp, 1, g.w // r, self.width),
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
                (0, None),
            )
            self._put(slab, t0, t0 + m * r)
            dealloc(slab)
        if self.table is not None:  # STATIC_TABLES=on (P.21): the loaded rows into the attention table too
            self.table.load_host(rows, p0 // r)

    def free(self) -> None:
        """Frees the tensor only when it is the cache's own (never the engine's); the attention table always."""
        if self.table is not None:
            self.table.free()
            self.table = None
        self.slab.free()


def config_names(ratio: int) -> tuple[str, str]:
    """The served config names of a KV source's compressed KV and index keys (serving_contract.md table order)."""
    return {2: ("1_compress_kv_r2", "2_index_k_r2"), 1: ("3_compress_kv_r1", "4_index_k_r1")}[ratio]


class TtCompressCaches:
    """A KV-source layer's two served caches (one slot): compressed KV (TP-dedup) and index keys (TP-replicated).
    ``slabs``: {config name: CacheSlab} of the serving engine's tensors; None = tensors of their own."""

    def __init__(self, mesh, tables, max_seq: int, ratio: int, head_dim: int, index_dim: int, slabs=None):
        self.ratio = ratio
        self.names = config_names(ratio)
        slabs = slabs or {}
        self.ckv = TtServedRows(
            mesh, tables, max_seq, head_dim, ratio, dedup=True, slab=slabs.get(self.names[0]), attn_table=True
        )
        self.index_k = TtServedRows(mesh, tables, max_seq, index_dim, ratio, dedup=False, slab=slabs.get(self.names[1]))

    def by_config(self) -> dict:
        return {self.names[0]: self.ckv, self.names[1]: self.index_k}

    def free(self) -> None:
        self.ckv.free()
        self.index_k.free()


class TtCompressor:
    """compressor step of one KV-source layer. ``__call__(x, start)``: attn_norm [1, 1, S/SP, H/TP] fp32 ->
    (ckv [1, 1, S/SP/r, head_dim], k [1, 1, S/SP/r, index_dim]) bf16 TILE, row-split, TP-replicated.
    ``write(ckv, k, start, end, caches)``: the served-cache write of the groups complete by ``end``. Does not consume
    its input."""

    def __init__(self, mesh, w: dict[str, torch.Tensor], cfg, ratio: int, rope: CompressRope, tables):
        self.mesh, self.ratio, self.rope, self.tables = mesh, ratio, rope, tables
        self.hd, self.idx, rd = cfg.head_dim, cfg.index_dim, cfg.rope_dim
        assert ratio in (1, 2), ratio
        # ROPE_IMPL=on (P.17): every weight in checkpoint order (the stock rope reads interleaved pairs)
        self.stock = rope.stock
        if self.stock:
            perm, permk = torch.arange(self.hd), torch.arange(self.idx)
        else:
            perm, permk = deinterleave_perm(self.hd, rd), deinterleave_perm(self.idx, rd)
        tp = mesh.shape[TP_AXIS]
        self.ckc = attn_compute_config(mesh)  # [wkv | wgate] and wk: ATTN_FIDELITY (P.7)
        blocks = [w["wkv"].float()[perm]] + ([w["wgate"].float()[perm]] if ratio > 1 else [])
        w_t = torch.cat(blocks, dim=0).t().contiguous()  # [H, 512 | 1024]
        hidden = w_t.shape[0]
        assert hidden % (32 * tp) == 0, f"hidden {hidden} does not split into tiles over TP {tp}"
        assert torch.equal(w_t.bfloat16().float(), w_t), "compressor weights must be bf16-exact (as stored)"
        dims = (None, 2) if TP_AXIS == 1 else (2, None)
        self.proj_w = _upload(mesh, w_t.reshape(1, 1, hidden, -1), ttnn.bfloat16, ttnn.TILE_LAYOUT, dims)
        # BF16_ACTIVATIONS=on (P.20, ROPE_IMPL=on): the normed latent written bf16 (the GPU code's norm(kv.to(dtype))),
        # read as is by the rope (ckv) and by wk, which writes bf16 too (the GPU code's bf16 wk)
        self.bf16 = bf16_activations() and self.stock
        self.norm = TtDistributedRmsNorm(
            mesh, w["norm"].float()[perm], cfg.eps, split=False, dtype=ttnn.bfloat16 if self.bf16 else None
        )
        wk = w["wk"].float()[permk][:, perm].t().contiguous()  # [512 (latent, de-interleaved), 128]
        assert torch.equal(wk.bfloat16().float(), wk), "wk must be bf16-exact (as stored)"
        self.wk = _upload(mesh, wk.reshape(1, 1, self.hd, self.idx), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        # ROPE_IMPL=on: k_norm writes the bf16 the stock rope reads (its last multiply's packer rounds once)
        kdt = ttnn.bfloat16 if self.stock else None
        self.k_norm = TtDistributedRmsNorm(mesh, w["k_norm"].float()[permk], cfg.eps, split=False, dtype=kdt)
        self.glue_partial = indexer_glue() and settings.get("HC_TP_SUM") == "gather_add"  # P.16 (tt/ccl.tp_sum)

    def _mm(self, a: ttnn.Tensor, b: ttnn.Tensor, dtype=ttnn.float32) -> ttnn.Tensor:
        return mm.linear(a, b, site="compressor", dtype=dtype, compute_kernel_config=self.ckc, memory_config=DRAM)

    def _pool(self, p: ttnn.Tensor) -> ttnn.Tensor:
        """[1, 1, R, 2 hd] = [kv | gate] per token -> the pair-pooled latent [1, 1, R / 2, hd] fp32."""
        rows, hd = p.shape[-2], self.hd
        pr = ttnn.reshape(p, [1, 1, rows // 2, 4 * hd])  # [kv_0 | g_0 | kv_1 | g_1]

        def col(i):
            return ttnn.slice(pr, [0, 0, 0, i * hd], [1, 1, rows // 2, (i + 1) * hd], memory_config=DRAM)

        kv0, g0, kv1, g1 = col(0), col(1), col(2), col(3)
        m = ttnn.maximum(g0, g1, memory_config=DRAM)
        e = []
        for g in (g0, g1):
            d = ttnn.subtract(g, m, dtype=ttnn.float32, memory_config=DRAM)
            e.append(ttnn.exp(d, memory_config=DRAM))
            dealloc(d)
            dealloc(g)
        dealloc(m)
        den = ttnn.add(e[0], e[1], dtype=ttnn.float32, memory_config=DRAM)
        acc = None
        for kv, ei in zip((kv0, kv1), e):
            wt = ttnn.divide(ei, den, memory_config=DRAM)
            t = ttnn.multiply(kv, wt, dtype=ttnn.float32, memory_config=DRAM)
            for u in (wt, ei, kv):
                dealloc(u)
            if acc is None:
                acc = t
            else:
                nxt = ttnn.add(acc, t, dtype=ttnn.float32, memory_config=DRAM)
                dealloc(acc)
                dealloc(t)
                acc = nxt
        dealloc(den)
        return acc

    def __call__(self, x: ttnn.Tensor, start: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        r = self.ratio
        rows = x.shape[-2]
        assert start % 32 == 0 and rows % r == 0, (start, rows, r)
        # INDEXER_GLUE=on (P.16): the TP partial written in the CCL's partial dtype (bf16 under fabric_bf16: the RS's
        # input, q_a's choice, P.6), not fp32 + a typecast in tt/ccl.py
        pdt = ccl.partial_dtype(self.mesh) if self.glue_partial else ttnn.float32
        part = self._mm(x, self.proj_w, dtype=pdt)
        p = tp_sum(self.mesh, part)
        dealloc(part)
        if r > 1:
            pooled = self._pool(p)
            dealloc(p)
        else:
            pooled = p
        latent = self.norm(pooled)
        dealloc(pooled)
        n = rows // r
        cos, sin = self.rope.rows(start // r, n * self.mesh.shape[SP_AXIS])
        if self.stock:
            return self._rope_stock(latent, cos, sin)
        # ATTN_GLUE=on (P.15): the rope's last ops write bf16 and the re-interleave is a PairPerm matmul
        fold = self.rope.pairs is not None
        if fold:
            ckv = rope_deinterleaved_bf16(latent, cos, sin, self.rope.pairs)
        else:
            lr = rope_deinterleaved(latent, cos, sin)
            ckv = cast(lr, ttnn.bfloat16)
            if ckv is not lr:
                dealloc(lr)
        kp = self._mm(latent, self.wk)
        dealloc(latent)
        kn = self.k_norm(kp)
        dealloc(kp)
        if fold:
            k = rope_deinterleaved_bf16(kn, cos, sin, self.rope.pairs)
            dealloc(kn)
            self.rope.free(cos, sin)
            return ckv, k
        kr = rope_deinterleaved(kn, cos, sin)
        dealloc(kn)
        self.rope.free(cos, sin)
        k = cast(kr, ttnn.bfloat16)
        if k is not kr:
            dealloc(kr)
        return ckv, k

    def _rope_stock(self, latent: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """ROPE_IMPL=on (P.17): the fp32 latent (consumed) -> (ckv, k) bf16 through tt/rope.rope_stock: ckv = rope of
        the latent cast to bf16; k = rope of k_norm(latent wk^T) written bf16 (wk reads the fp32 latent, as before)."""
        lb = cast(latent, ttnn.bfloat16)  # BF16_ACTIVATIONS=on: the latent itself (already bf16)
        ckv = rope_stock(lb, cos, sin, self.rope.trans)
        if lb is not latent:
            dealloc(lb)
        kp = self._mm(latent, self.wk, dtype=ttnn.bfloat16 if self.bf16 else ttnn.float32)
        dealloc(latent)
        kn = self.k_norm(kp)
        dealloc(kp)
        k = rope_stock(kn, cos, sin, self.rope.trans)
        dealloc(kn)
        self.rope.free(cos, sin)  # INDEXER_GLUE=on: the chunk's memoized rows stay
        return ckv, k

    def _gathered(self, t: ttnn.Tensor, valid: int) -> ttnn.Tensor:
        """[1, 1, n, w] bf16 TILE row-split -> [1, 1, SP n, w] bf16 ROW_MAJOR on every chip (natural order), rows at
        or past ``valid`` zero."""
        g = self._gathered_tile(t, valid)
        rm = ttnn.to_layout(g, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        if g is not t and rm is not g:
            dealloc(g)
        return rm

    def _gathered_tile(self, t: ttnn.Tensor, valid: int) -> ttnn.Tensor:
        """``_gathered`` before its ROW_MAJOR: [1, 1, SP n, w] bf16 TILE (t itself on one SP row with every row valid;
        the caller frees it when it is not t)."""
        g = t
        if self.mesh.shape[SP_AXIS] > 1:
            g = ccl.all_gather(t, 2, SP_AXIS)
        total = g.shape[-2]
        if valid < total:
            # ratio 1: total == the iota's rows, and a full-range slice returns a wrapper over the load-time iota's
            # buffer (DS41-52, K-1): use the iota itself then, never free it
            full = total == self.tables.iota.shape[-2]
            iota = (
                self.tables.iota
                if full
                else ttnn.slice(self.tables.iota, [0, 0, 0, 0], [1, 1, total, 1], memory_config=DRAM)
            )
            keep = ttnn.lt(iota, float(valid), dtype=ttnn.bfloat16, memory_config=DRAM)
            if not full:
                dealloc(iota)
            masked = ttnn.multiply(g, keep, dtype=ttnn.bfloat16, memory_config=DRAM)
            dealloc(keep)
            if g is not t:
                dealloc(g)
            g = masked
        return g

    def write(self, ckv: ttnn.Tensor, k: ttnn.Tensor, start: int, end: int, caches: TtCompressCaches) -> None:
        """The chunk's groups complete by actual_end ((j + 1) r <= end) into the served caches; the other rows of the
        chunk (an incomplete last group, pad) written as zeros. Does not consume its inputs."""
        valid = end // self.ratio - start // self.ratio  # complete groups of the chunk
        for t, cache in ((ckv, caches.ckv), (k, caches.index_k)):
            table = cache.table
            if table is not None and table.bfp4:
                # CKV_FORMAT=bfp4_rows: the table's records from the gathered tiles; the served cache gets their bf16
                # values (the rounded rows), so serving and a later host load see what attention reads
                g = self._gathered_tile(t, valid)
                rounded = table.append_tiles(g, start // self.ratio)
                if g is not t:
                    dealloc(g)
                rm = ttnn.to_layout(rounded, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
                if rm is not rounded:
                    dealloc(rounded)
                cache.write(rm, start, end, table=False)
            else:
                rm = self._gathered(t, valid)
                cache.write(rm, start, end)
            dealloc(rm)


def compressor_weights(ck, layer: int) -> dict[str, torch.Tensor]:
    p = f"layers.{layer}.attn."
    names = {
        "wkv": "compressor.wkv.weight",
        "wgate": "compressor.wgate.weight",
        "norm": "compressor.norm.weight",
        "wk": "indexer.wk.weight",
        "k_norm": "indexer.k_norm.weight",
    }
    return {k: ck.weight(p + v, torch.float32) for k, v in names.items() if (p + v) in ck.weight_map}


def build_compressor(mesh, ck, cfg, layer: int, max_seq: int, tables, rope: CompressRope | None = None):
    from .kv import rope_freqs

    r = cfg.ratio(layer)
    rope = rope or CompressRope(mesh, rope_freqs(cfg, layer), max_seq, r)
    return TtCompressor(mesh, compressor_weights(ck, layer), cfg, r, rope, tables)
