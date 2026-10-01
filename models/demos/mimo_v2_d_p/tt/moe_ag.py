# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""All-gather MoE block (``MiMoRuntimeOptions.moe_ag``, default): the routed-expert data movement without dispatch / combine.

    x, topk (indices, weights)  --high_bw_all_gather over the dispatch axis (mesh rows)-->  every chip of a mesh column
    holds the column's chunk_size tokens (gathered row g = src_row * chunk_size_per_chip + token)
    -> RoutePlan (on device): counts / regions (the flat expert's rows), token_index (flat row -> gathered row),
       y_slot (per (token, k) the flat row of its expert output on this chip, or none)
    -> flat routed expert in indexed mode (reads gathered x rows through token_index) -> y [rows, H] bfp8 TILE
    -> LocalReduce: partial[g] = sum over this chip's experts of w[g, k] * y[y_slot[g, k]] (bf16 rows)
    -> send-back over the dispatch axis (2 rows: exchange through high_bw_all_gather + add; else reduce_scatter)
    -> all-reduce over the mesh columns (high_bw_all_gather + add) -> [1, 1, chunk_size_per_chip, H] replicated.

Device programs: the ttnn.experimental.deepseek_prefill.moe_ag_* ops (C++, ttnn/cpp/ttnn/operations/experimental/
deepseek_prefill/moe_ag; one program for the whole mesh, per-device data through small per-device tensors: the local-slot
map and the chip's row).
"""

import torch

import ttnn
from models.demos.mimo_v2_d_p.tt.options import MiMoRuntimeOptions

NONE = 0xFFFFFFFF
_ops = ttnn.experimental.deepseek_prefill


def _dram(mesh_device, shape, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dtype, layout, mesh_device, ttnn.DRAM_MEMORY_CONFIG)


def _per_device(mesh_device, per_dev, dtype):
    """per_dev: torch [rows, cols, ...] -> each device its [1, 1, ...] slice (row major, DRAM)."""
    rows, cols = tuple(mesh_device.shape)
    return ttnn.from_torch(
        per_dev.reshape(rows, cols, 1, -1),
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(0, 1)),
    )


def _up(v, a):
    return -(-v // a) * a


def flat_rows(tokens, k, experts_per_chip):
    """Rows of the flat expert space that hold any routing (no pair is ever dropped): a token puts at most
    min(k, experts_per_chip) pairs on a chip, and each active local expert pads its region to 32 rows:
    sum_e ceil(c_e / 32) <= ceil(P / 32) + n - 1 for P pairs over n <= experts_per_chip active experts. The
    moe_ag_route_plan op requires at least this many."""
    pairs = tokens * min(k, experts_per_chip)
    return _up(pairs, 32) + 32 * (min(pairs, experts_per_chip) - 1)


def local_map(mesh_device, gids, n_global):
    """gids[d] (row-major device order) -> per device [1, 1, n_global] uint32: the global id's local slot, or NONE."""
    mr, mc = tuple(mesh_device.shape)
    lmap = torch.full((mr * mc, n_global), NONE, dtype=torch.int64)
    for d, gl in enumerate(gids):
        for l, g in enumerate(gl):
            lmap[d, g] = l
    return _per_device(mesh_device, lmap, ttnn.uint32)


class RoutePlan:
    """Gathered top-k indices [T, K] uint16 -> counts / regions [1, NG], token_index [1, rows], y_slot [1, T * K]
    (uint32, DRAM, persistent). gids[d]: device d's (row-major mesh order) local experts' global ids, local order."""

    def __init__(self, mesh_device, *, tokens, k, n_global, gids, rows):
        self.T, self.K, self.NG, self.rows = tokens, k, n_global, rows
        self.EPC = len(gids[0])
        mr, mc = tuple(mesh_device.shape)
        self.lmap = local_map(mesh_device, gids, n_global)
        self.counts = _dram(mesh_device, [1, n_global], ttnn.uint32)
        self.regions = _dram(mesh_device, [1, n_global], ttnn.uint32)
        # zero-initialized once: the plan writes only the used regions, and a consumer that reads the whole capacity
        # (the ttnn.embedding A/B path) must only ever see valid token indices (0 or a previous plan's)
        self.token_index = ttnn.from_torch(
            torch.zeros(1, rows, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self.y_slot = _dram(mesh_device, [1, tokens * k], ttnn.uint32)

    def __call__(self, idx, lmap=None):
        """idx: gathered top-k [.., T, K] uint16 row major DRAM; ``lmap``: a layer's own local-slot map (``local_map``,
        a per-layer expert placement) instead of the block's. Returns (counts, regions, token_index, y_slot)."""
        _ops.moe_ag_route_plan(
            idx,
            self.lmap if lmap is None else lmap,
            self.EPC,
            self.rows,
            outputs=[self.counts, self.regions, self.token_index, self.y_slot],
        )
        return self.counts, self.regions, self.token_index, self.y_slot

    @staticmethod
    def reference(idx, lmap_row, epc, rows):
        """Host reference for one device: idx [T, K] int, lmap_row [NG] (local slot or NONE)."""
        T, K = idx.shape
        NG = lmap_row.shape[0]
        lists = [[] for _ in range(epc)]
        for g in range(T):
            for k in range(K):
                l = int(lmap_row[int(idx[g, k])])
                if l < epc:
                    lists[l].append((g, k))
        counts = torch.zeros(NG, dtype=torch.int64)
        regions = torch.zeros(NG, dtype=torch.int64)
        tidx = torch.zeros(rows, dtype=torch.int64)
        yslot = torch.full((T, K), NONE, dtype=torch.int64)
        region = 0
        for l in range(epc):
            gid = int((lmap_row == l).nonzero()[0])
            counts[gid], regions[gid] = len(lists[l]), region
            for i, (g, k) in enumerate(lists[l]):
                tidx[region + i] = g
                yslot[g, k] = region + i
            region += _up(len(lists[l]), 32)
        return counts, regions, tidx, yslot, region


def chip_info(mesh_device, chunk_size_per_chip):
    """Per device [1, 16] uint32: word 0 its mesh row, word 1 the other row's block start in a 2-row gather
    ((1 - row) * S: where the peer's partials for this chip's tokens land), word 2 its own block start (row * S)."""
    rows, cols = tuple(mesh_device.shape)
    t = torch.zeros(rows, cols, 16, dtype=torch.int64)
    for r in range(rows):
        t[r, :, 0] = r
        t[r, :, 1] = (rows - 1 - r) * chunk_size_per_chip if rows == 2 else 0
        t[r, :, 2] = r * chunk_size_per_chip
    return _per_device(mesh_device, t, ttnn.uint32)


class LocalReduce:
    """partial[g] = sum over this chip's local experts of w[g, k] * y[y_slot[g, k]] for the column's T tokens.
    y: row-major bf16 [rows, H]. split (2 mesh rows): own [S, H] (this chip's row) and other [S, H] (the peer's);
    tiled (not split): the [T, H] partials as bf16 tiles (the > 2-row reduce-scatter input). Persistent outputs."""

    def __init__(
        self, mesh_device, *, tokens, k, hidden, chunk_size_per_chip, split, info, tiled=False, own_tiled=False
    ):
        """``own_tiled`` (split, fused send-back): phase 2 writes ``own`` as tiles (the TP reduce-scatter's input, no
        tilize pass)."""
        self.T, self.K, self.H, self.S, self.split, self.info = tokens, k, hidden, chunk_size_per_chip, split, info
        self.tiled = tiled and not split
        self.own_tiled = own_tiled and split
        rows_out = chunk_size_per_chip if split else tokens
        self.own = _dram(
            mesh_device,
            [1, 1, rows_out, hidden],
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT if self.tiled or self.own_tiled else ttnn.ROW_MAJOR_LAYOUT,
        )
        self.other = _dram(mesh_device, [1, 1, rows_out, hidden], ttnn.bfloat16) if split else None

    def phase(self, y, y_slot, w, phase, peer=None):
        """Two mesh rows, fused send-back: phase 1 reduces the other row's tokens into ``other``; phase 2 this row's
        tokens plus the peer's gathered phase-1 partial (``peer`` [2 S, H]) into ``own``."""
        assert self.split
        out = self.other if phase == 1 else self.own
        tiled = phase == 2 and self.own_tiled
        _ops.moe_ag_local_reduce(y, y_slot, w, self.info, self.S, phase=phase, peer=peer, tiled=tiled, outputs=[out])
        return out

    def __call__(self, y, y_slot, w):
        """y row-major bf16 [.., rows, H], y_slot [1, T K] uint32, w gathered weights [.., T, K] bf16 row major."""
        outs = [self.own, self.other] if self.split else [self.own]
        _ops.moe_ag_local_reduce(y, y_slot, w, self.info, self.S, split=self.split, tiled=self.tiled, outputs=outs)
        return (self.own, self.other) if self.split else self.own


class AddRows:
    """out[i] = a[a_off + i] + b[b_off + i] for i < n rows (row-major bf16, width H) into a persistent row-major
    [1, 1, n, H]; b_off per device from the chip info (word 1) when info_offset, else a constant."""

    def __init__(self, mesh_device, *, n_rows, hidden, info):
        self.n, self.info = n_rows, info
        self.out = _dram(mesh_device, [1, 1, n_rows, hidden], ttnn.bfloat16)

    def __call__(self, a, b, *, a_off=0, b_off=0, info_offset=False):
        return _ops.moe_ag_add_rows(
            a, b, self.info, self.n, a_offset=a_off, b_offset=b_off, info_offset=info_offset, output=self.out
        )


_BLOCKS = {}


class MoeAgBlock:
    """The all-gather MoE data movement for one mesh / shape; persistent buffers, shared by every MoE layer (the layers
    run one after another). Use ``MoeAgBlock.get(...)``; ``gather`` -> ``plan`` -> (experts) -> ``reduce``."""

    @classmethod
    def get(cls, mesh_device, **kw):
        key = (id(mesh_device),) + tuple(sorted((k, v if not isinstance(v, list) else str(v)) for k, v in kw.items()))
        if key not in _BLOCKS:
            _BLOCKS[key] = cls(mesh_device, **kw)
        return _BLOCKS[key]

    def __init__(self, mesh_device, *, chunk_size_per_chip, hidden, k, n_global, gids, buf_rows, options=None):
        self.dev = mesh_device
        self.options = options = options or MiMoRuntimeOptions()
        rows, cols = tuple(mesh_device.shape)
        self.rows, self.cols = rows, cols
        S, H = chunk_size_per_chip, hidden
        self.S, self.H, self.K = S, H, k
        self.T = T = rows * S  # chunk_size: the tokens of a mesh column
        self.links = options.hbw_links  # None: every usable link on the axis (QuietBox 4, Galaxy 2)
        from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology

        self.sp_topo, self.tp_topo = per_axis_topology()
        self.rs_links = options.moe_ag_rs_links
        # x_pages_per_row = 4: gathered x in 2 KB pages ([T * 4, 1024]: a token row over 4 DRAM banks; the indexed
        # expert reads it with x_pages_per_row = 4). Default 1: one 8 KB page per token row.
        self.xppr = options.moe_ag_x_pages_per_row
        assert self.xppr in (1, H // 1024), self.xppr
        if rows > 1:
            self.gx = _dram(mesh_device, [1, 1, T * self.xppr, H // self.xppr])
            self.gidx = _dram(mesh_device, [1, 1, T, k], ttnn.uint16)
            self.gw = _dram(mesh_device, [1, 1, T, k])
        # top-k idx / w gathered as tiles, untilized after the gather: high_bw_all_gather costs per page, and S rows of
        # K = 8 values are S pages (~80 us at 2 links) where S / 32 tiles move in ~14 us (+ ~10 us more untilize)
        self.tile_topk = options.moe_ag_tile_topk
        if rows > 1 and self.tile_topk:
            self.gidx_t = _dram(mesh_device, [1, 1, T, k], ttnn.uint16, ttnn.TILE_LAYOUT)
            self.gw_t = _dram(mesh_device, [1, 1, T, k], ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self.plan_op = RoutePlan(mesh_device, tokens=T, k=k, n_global=n_global, gids=gids, rows=buf_rows)
        self.info = chip_info(mesh_device, S)
        split = rows == 2
        self.split = split
        # TP all-reduce over the mesh columns (options.moe_ag_tp): "hbw" high_bw_all_gather + one add / tilize pass (the
        # default for 2 columns), "rsag" ttnn.reduce_scatter + ttnn.all_gather on tiles (default for > 2 columns:
        # 162 / 284 us vs 220 / 404 on 1x4 at 640 / 1280 tokens per chip, it moves 2 x 3/4 of the rows instead of 3 x)
        self.tp_mode = options.moe_ag_tp or ("rsag" if cols > 2 else "hbw")
        self.lreduce = LocalReduce(
            mesh_device,
            tokens=T,
            k=k,
            hidden=H,
            chunk_size_per_chip=S,
            split=split,
            info=self.info,
            tiled=rows > 2 or (rows == 1 and self.tp_mode == "rsag"),
            # the column partial goes straight into a tiled reduce-scatter (sequence-parallel residual, or rsag)
            own_tiled=cols > 1 and options.moe_ag_fused_send_back and (options.sp_residual or self.tp_mode == "rsag"),
        )
        if split:
            self.g_sp = _dram(mesh_device, [1, 1, 2 * S, H])
            self.ex = AddRows(mesh_device, n_rows=S, hidden=H, info=self.info)
        if cols > 1 and self.tp_mode == "hbw":
            self.g_tp = _dram(mesh_device, [1, 1, cols * S, H])
        # y_row_major (default): the flat expert writes y as row-major bf16 itself (pack-untilized on its down cores),
        # so no untilize pass and no [rows, H] untilized copy; False: bfp8 tiles + UntilizeActive
        self.y_rm = options.moe_ag_y_row_major
        self.untilize = None  # built on the first tiled y (y_rm off, or an expert without row-major output)
        self._untilize_args = dict(
            rows=buf_rows,
            hidden=H,
            n_global=n_global,
            epc=len(gids[0]),
            lmap=self.plan_op.lmap,
            W=options.untilize_width,
        )

    def tp(self, g_tp):
        """The gathered column partials [cols S, H] -> their sum as a fresh bf16 TILE [1, 1, S, H]."""
        if self.cols == 2:
            return add_rows_tiled(g_tp, n_rows=self.S, b_off=self.S)
        return sum_blocks_tiled(g_tp, n_rows=self.S, n_blocks=self.cols)

    def _ag(self, x, out, axis):
        return ttnn.experimental.high_bw_all_gather(
            x, dim=2, output_tensor=out, cluster_axis=axis, num_links=self.links
        )

    def to_rm(self, x):
        """x [1, 1, S, H] bf16 TILE -> the row-major layout the gather takes ([1, 1, S xppr, H / xppr])."""
        return untilize_x(x) if self.xppr > 1 else ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

    def gather(self, x_rm, idx, w):
        """x_rm (``to_rm``), idx [1, 1, S, K] uint16, w [1, 1, S, K] bf16 (RM, or TILE with ``tile_topk``) -> the
        column's T tokens (idx / w row major)."""
        rm = lambda t: ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT) if t.layout == ttnn.TILE_LAYOUT else t
        if self.rows == 1:  # no dispatch axis: the chip's own tokens are the column's
            self.gx, self.gidx, self.gw = x_rm, rm(idx), rm(w)
            return self.gx, self.gidx, self.gw
        self._ag(x_rm, self.gx, 0)
        if idx.layout == ttnn.TILE_LAYOUT:
            self._ag(idx, self.gidx_t, 0)
            self._ag(w, self.gw_t, 0)
            self.gidx, self.gw = rm(self.gidx_t), rm(self.gw_t)
        else:
            self._ag(idx, self.gidx, 0)
            self._ag(w, self.gw, 0)
        return self.gx, self.gidx, self.gw

    def plan(self, lmap=None):
        return self.plan_op(self.gidx, lmap)

    def buffers_mb(self):
        """Persistent DRAM per chip (MB) held by this block (gather outputs, plan, reduce buffers, untilized y)."""
        names = ("gx", "gidx", "gw", "g_sp", "g_tp") if self.rows > 1 else ("g_sp", "g_tp")
        ts = [getattr(self, n, None) for n in names] + [
            self.plan_op.counts,
            self.plan_op.regions,
            self.plan_op.token_index,
            self.plan_op.y_slot,
            self.plan_op.lmap,
            self.info,
            self.lreduce.own,
            self.lreduce.other,
            self.untilize.out if self.untilize is not None else None,
            getattr(getattr(self, "ex", None), "out", None),
        ]
        esize = {ttnn.bfloat16: 2, ttnn.uint16: 2, ttnn.uint32: 4}
        tot = 0
        for t in ts:
            if t is not None:
                n = 1
                for v in t.shape:
                    n *= v
                tot += n * esize[t.dtype]
        return tot / 1e6

    def reduce(self, y, scatter=False):
        """y [rows, H] (the experts' outputs at their flat rows: row-major bf16, or bfp8 TILE without y_rm) -> a fresh
        [1, 1, S, H] bf16 TILE tensor, summed over every chip's experts, replicated over the mesh columns."""
        if y.layout == ttnn.ROW_MAJOR_LAYOUT:
            y_rm = y
        else:
            if self.untilize is None:
                self.untilize = UntilizeActive(self.dev, **self._untilize_args)
            y_rm = self.untilize(y, self.plan_op.counts, self.plan_op.regions)
        if self.split and self.options.moe_ag_fused_send_back:
            other = self.lreduce.phase(y_rm, self.plan_op.y_slot, self.gw, 1)
            self._ag(other, self.g_sp, 0)
            col = self.lreduce.phase(y_rm, self.plan_op.y_slot, self.gw, 2, peer=self.g_sp)
        elif self.split:
            own, other = self.lreduce(y_rm, self.plan_op.y_slot, self.gw)
            self._ag(other, self.g_sp, 0)
            col = self.ex(own, self.g_sp, info_offset=True)
        elif self.rows == 1:
            col = self.lreduce(y_rm, self.plan_op.y_slot, self.gw)
        else:  # > 2 rows: reduce-scatter of the [T, H] partials over the rows (tiles)
            part = self.lreduce(y_rm, self.plan_op.y_slot, self.gw)  # bf16 tiles (LocalReduce tiled)
            col = ttnn.reduce_scatter(part, dim=2, cluster_axis=0, topology=self.sp_topo, num_links=self.rs_links)
        return self._tp_allreduce(col, scatter)

    def _tp_allreduce(self, col, scatter=False):
        """[1, 1, S, H] column partial (row major, or tiles after a reduce-scatter) -> fresh TILE, summed over cols."""
        if self.cols == 1:
            return ttnn.to_layout(col, ttnn.TILE_LAYOUT) if col.layout != ttnn.TILE_LAYOUT else col
        if scatter:  # sequence-parallel residual: this col's rows of the sum only
            t = ttnn.to_layout(col, ttnn.TILE_LAYOUT) if col.layout != ttnn.TILE_LAYOUT else col
            rs = ttnn.reduce_scatter(t, dim=2, cluster_axis=1, topology=self.tp_topo, num_links=self.rs_links)
            if t is not col:
                ttnn.deallocate(t)
            return rs
        if self.tp_mode == "rsag":
            t = ttnn.to_layout(col, ttnn.TILE_LAYOUT) if col.layout != ttnn.TILE_LAYOUT else col
            rs = ttnn.reduce_scatter(t, dim=3, cluster_axis=1, topology=self.tp_topo, num_links=self.rs_links)
            if t is not col:
                ttnn.deallocate(t)
            out = ttnn.all_gather(rs, dim=3, cluster_axis=1, topology=self.tp_topo, num_links=self.rs_links)
            ttnn.deallocate(rs)
            return out
        if col.layout != ttnn.ROW_MAJOR_LAYOUT:
            rm = ttnn.to_layout(col, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(col)
            col = rm
        self._ag(col, self.g_tp, 1)
        return self.tp(self.g_tp)


class UntilizeActive:
    """y bfp8 TILE [rows, H] -> y_rm bf16 RM [rows, H] (persistent), only the tile rows that hold tokens (each local
    expert's ceil(count / 32) tile rows at its region; counts / regions read on device)."""

    def __init__(self, mesh_device, *, rows, hidden, n_global, epc, lmap, W=32):
        self.lmap, self.EPC, self.W = lmap, epc, W
        self.out = _dram(mesh_device, [rows, hidden])

    def __call__(self, y, counts, regions):
        return _ops.moe_ag_untilize_active(
            y, counts, regions, self.lmap, self.EPC, tiles_per_block=self.W, output=self.out
        )


def add_rows_tiled(src, *, n_rows, b_off):
    """out[r] = src[r] + src[b_off + r] for r < n_rows -> a fresh bf16 TILE [1, 1, n_rows, H] (add + tilize)."""
    return _ops.moe_ag_sum_rows_tiled(src, n_rows, 2, b_off)


def sum_blocks_tiled(src, *, n_rows, n_blocks):
    """out[r] = sum_{i < n_blocks} src[i n_rows + r] (e.g. a gather over n_blocks chips) -> a fresh bf16 TILE."""
    return _ops.moe_ag_sum_rows_tiled(src, n_rows, n_blocks, n_rows)


def untilize_x(x):
    """x bf16 TILE [1, 1, S, H] -> row-major [1, 1, S * H / 1024, 1024] (2 KB pages: token row g = pages NCH g ..
    NCH g + NCH - 1), the gathered-x page layout (spreads a token's reads over NCH DRAM banks)."""
    return _ops.moe_ag_untilize_x(x)
