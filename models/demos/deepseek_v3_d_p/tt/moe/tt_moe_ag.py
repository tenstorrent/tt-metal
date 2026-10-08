# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""All-gather MoE block (TtMoe ``moe_block="all_gather"``): the routed experts' data movement without dispatch / combine.

Ported from MiMo-V2 d_p (models/demos/mimo_v2_d_p/tt/moe_ag.py), trimmed to the contract TtMoe's dispatch / combine path
has: routed_x [.., S, Hr] replicated over the TP axis (mesh columns) in, the top-k weighted sum reduce-scattered over the
columns on the hidden dim out ([1, 1, S, Hr / TP] TILE, what TtReduceModule returns).

    routed_x, top-k (indices, scores)  --fabric_all_gather over the dispatch axis (mesh rows)-->  every chip of a
    mesh column holds the column's T = rows * S tokens (gathered row g = src_row * S + token)
    -> moe_ag_route_plan (on device): counts / regions (the flat expert's rows), token_index (flat row -> gathered row),
       y_slot (per (token, k) the flat row of its expert output on this chip, or none)
    -> flat_routed_expert in indexed mode (reads gathered x rows through token_index) -> y [rows, Hr] bf16 row major
    -> moe_ag_local_reduce: partial[g] = sum over this chip's local experts of w[g, k] * y[y_slot[g, k]]
    -> back over the rows: 1 row nothing; 2 rows the peer's partial is exchanged (fabric_all_gather) and added inside
       the reduce (fused send-back); > 2 rows a reduce_scatter of the [T, Hr] partials
    -> reduce_scatter over the columns on the hidden dim -> [1, 1, S, Hr / TP] bf16 TILE.

Against dispatch / combine it moves every token of the column to every chip of the column (rows x S x Hr per chip in,
the same back) instead of each token to its K experts' chips, and needs no metadata, no dispatch buffer capacity and no
combine. That wins on few rows (LoudBox 2 x 4: one exchange each way) and costs more on many (Galaxy 8 x 4); see
tests/perf/test_moe_block_perf.py.
"""

import torch
from loguru import logger

import ttnn

NONE = 0xFFFFFFFF
_ops = ttnn.experimental.deepseek_prefill


def _dram(mesh_device, shape, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dtype, layout, mesh_device, ttnn.DRAM_MEMORY_CONFIG)


def _per_device(mesh_device, per_dev, dtype):
    """per_dev: torch [rows * cols, ...] in row-major device order -> each device its [1, 1, ...] slice (RM, DRAM)."""
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
    rows, cols = tuple(mesh_device.shape)
    lmap = torch.full((rows * cols, n_global), NONE, dtype=torch.int64)
    for d, gl in enumerate(gids):
        for l, g in enumerate(gl):
            lmap[d, g] = l
    return _per_device(mesh_device, lmap, ttnn.uint32)


def chip_info(mesh_device, chunk_size_per_chip):
    """Per device [1, 16] uint32: word 0 its mesh row, word 1 the other row's block start in a 2-row gather
    ((1 - row) * S: where the peer's partials for this chip's tokens land), word 2 its own block start (row * S)."""
    rows, cols = tuple(mesh_device.shape)
    t = torch.zeros(rows, cols, 16, dtype=torch.int64)
    for r in range(rows):
        t[r, :, 0] = r
        t[r, :, 1] = (rows - 1 - r) * chunk_size_per_chip if rows == 2 else 0
        t[r, :, 2] = r * chunk_size_per_chip
    return _per_device(mesh_device, t.reshape(rows * cols, 16), ttnn.uint32)


class RoutePlan:
    """Gathered top-k indices [T, K] uint16 -> counts / regions [1, NG], token_index [1, rows], y_slot [1, T * K]
    (uint32, DRAM, persistent). gids[d]: device d's (row-major mesh order) local experts' global ids, local order.
    Expert ids >= NG (the gate's sentinel for padded tokens) map to no local expert: those pairs are dropped."""

    def __init__(self, mesh_device, *, tokens, k, n_global, gids, rows):
        self.T, self.K, self.NG, self.rows = tokens, k, n_global, rows
        self.EPC = len(gids[0])
        self.lmap = local_map(mesh_device, gids, n_global)
        self.counts = _dram(mesh_device, [1, n_global], ttnn.uint32)
        self.regions = _dram(mesh_device, [1, n_global], ttnn.uint32)
        # zero-initialized once: the plan writes only the used regions, the rest must be valid token indices
        self.token_index = ttnn.from_torch(
            torch.zeros(1, rows, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self.y_slot = _dram(mesh_device, [1, tokens * k], ttnn.uint32)

    def __call__(self, idx):
        """idx: gathered top-k [.., T, K] uint16 row major DRAM. Returns (counts, regions, token_index, y_slot)."""
        _ops.moe_ag_route_plan(
            idx, self.lmap, self.EPC, self.rows, outputs=[self.counts, self.regions, self.token_index, self.y_slot]
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
                gid = int(idx[g, k])
                l = int(lmap_row[gid]) if gid < NG else NONE
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


_BLOCKS = {}


class TtMoeAgRouted:
    """The all-gather routed-expert data movement for one mesh / shape. Persistent buffers, shared by every MoE layer of
    that shape (the layers run one after another): use ``TtMoeAgRouted.get(...)``. Call it with the layer's flat
    expert; it returns what TtMoe's dispatch / combine path returns from its TtReduceModule."""

    @classmethod
    def get(cls, mesh_device, **kw):
        key = (id(mesh_device),) + tuple(sorted((k, str(v)) for k, v in kw.items()))
        if key not in _BLOCKS:
            _BLOCKS[key] = cls(mesh_device, **kw)
        return _BLOCKS[key]

    def __init__(
        self,
        mesh_device,
        *,
        seq_len_per_chip,
        hidden,
        k,
        n_global,
        gids,
        row_num_links=None,
        col_num_links=None,
        row_topology=ttnn.Topology.Linear,
        col_topology=ttnn.Topology.Linear,
    ):
        self.dev = mesh_device
        self._fag_sems = None  # fabric_all_gather's ready / data-valid semaphores, created on the first gather
        rows, cols = tuple(mesh_device.shape)
        self.rows, self.cols = rows, cols
        S, H = seq_len_per_chip, hidden
        self.S, self.H, self.K = S, H, k
        self.T = T = rows * S  # the tokens of a mesh column
        self.EPC = len(gids[0])
        self.buf_rows = flat_rows(T, k, self.EPC)
        self.row_links, self.col_links = row_num_links, col_num_links
        self.row_topology, self.col_topology = row_topology, col_topology
        if rows > 1:
            self.gx = _dram(mesh_device, [1, 1, T, H])
            # top-k gathered as tiles and untilized after: the gather costs per page, and S rows of K values are
            # S pages where S / 32 tiles move several times faster
            self.gidx_t = _dram(mesh_device, [1, 1, T, k], ttnn.uint16, ttnn.TILE_LAYOUT)
            self.gw_t = _dram(mesh_device, [1, 1, T, k], ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self.plan_op = RoutePlan(mesh_device, tokens=T, k=k, n_global=n_global, gids=gids, rows=self.buf_rows)
        self.info = chip_info(mesh_device, S)
        self.split = rows == 2
        # the column partial goes straight into the TP reduce-scatter: written as tiles by the reduce itself
        own_tiled = self.split or rows == 1
        self.own = _dram(
            mesh_device,
            [1, 1, S if (self.split or rows == 1) else T, H],
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT if (own_tiled or rows > 2) else ttnn.ROW_MAJOR_LAYOUT,
        )
        if self.split:
            self.other = _dram(mesh_device, [1, 1, S, H])
            self.g_sp = _dram(mesh_device, [1, 1, 2 * S, H])
        logger.info(
            f"TtMoeAgRouted: mesh {rows}x{cols}, S={S} T={T} H={H} K={k} EPC={self.EPC} flat rows={self.buf_rows}, "
            f"{self.buffers_mb():.1f} MB persistent DRAM per chip"
        )

    def _ag(self, x, out, axis):
        """Every gather of the block: ttnn.experimental.fabric_all_gather (high_bw_all_gather's contract, a fabric-chunk
        program: one link worker per ring direction and link, forwarding shard by shard). Caller-owned semaphores,
        zero and left at zero by every call, so the op allocates and synchronizes nothing per launch."""
        if self._fag_sems is None:
            g = self.dev.compute_with_storage_grid_size()
            crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))})
            # The op requires them in L1_SMALL whenever the device has that pool (where its own would go), else L1.
            # Two for the whole block (128 B / bank), instead of a pair per gather program.
            l1_small = ttnn.get_memory_view(self.dev, ttnn.BufferType.L1_SMALL).total_bytes_per_bank > 0
            bt = ttnn.BufferType.L1_SMALL if l1_small else ttnn.BufferType.L1
            self._fag_sems = [ttnn.create_global_semaphore(self.dev, crs, 0, bt) for _ in range(2)]
        return ttnn.experimental.fabric_all_gather(
            x,
            dim=2,
            output_tensor=out,
            cluster_axis=axis,
            num_links=self.row_links if axis == 0 else self.col_links,
            ready_semaphore=self._fag_sems[0],
            data_valid_semaphore=self._fag_sems[1],
        )

    def buffers_mb(self):
        names = ("gx", "gidx_t", "gw_t", "own", "other", "g_sp")
        ts = [getattr(self, n, None) for n in names] + [
            self.plan_op.counts,
            self.plan_op.regions,
            self.plan_op.token_index,
            self.plan_op.y_slot,
            self.plan_op.lmap,
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

    def gather(self, x, indices, scores):
        """x [.., S, H] bf16 row major; indices / scores [.., S, K] (TILE or row major) -> the column's T tokens:
        (gx [1, 1, T, H] RM, gidx [1, 1, T, K] uint16 RM, gw [1, 1, T, K] bf16 RM)."""
        S, H, K = self.S, self.H, self.K
        x = ttnn.reshape(x, (1, 1, S, H))
        idx = ttnn.reshape(indices, (1, 1, S, K))
        w = ttnn.reshape(scores, (1, 1, S, K))
        if w.dtype != ttnn.bfloat16:
            w = ttnn.typecast(w, ttnn.bfloat16)
        if self.rows == 1:  # no dispatch axis: the chip's own tokens are the column's
            rm = lambda t: ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT) if t.layout != ttnn.ROW_MAJOR_LAYOUT else t
            return x, rm(idx), rm(w)
        # fabric_all_gather reads DRAM (the gate's top-k can come out in L1)
        dram = (
            lambda t: t
            if t.memory_config() == ttnn.DRAM_MEMORY_CONFIG
            else ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
        )
        tile = lambda t: dram(ttnn.to_layout(t, ttnn.TILE_LAYOUT) if t.layout != ttnn.TILE_LAYOUT else t)
        self._ag(dram(x), self.gx, 0)
        self._ag(tile(idx), self.gidx_t, 0)
        self._ag(tile(w), self.gw_t, 0)
        return (
            self.gx,
            ttnn.to_layout(self.gidx_t, ttnn.ROW_MAJOR_LAYOUT),
            ttnn.to_layout(self.gw_t, ttnn.ROW_MAJOR_LAYOUT),
        )

    def reduce(self, y, y_slot, gw):
        """y [rows, H] bf16 row major (the experts' outputs at their flat rows) -> this column's [1, 1, S, H] partial
        (bf16 TILE), summed over every row's experts."""
        if self.rows == 1:
            return _ops.moe_ag_local_reduce(y, y_slot, gw, self.info, self.S, tiled=True, outputs=[self.own])[0]
        if self.split:  # fused send-back: the other row's tokens first, exchanged, then this row's plus the peer's
            _ops.moe_ag_local_reduce(y, y_slot, gw, self.info, self.S, phase=1, outputs=[self.other])
            self._ag(self.other, self.g_sp, 0)
            return _ops.moe_ag_local_reduce(
                y, y_slot, gw, self.info, self.S, phase=2, peer=self.g_sp, tiled=True, outputs=[self.own]
            )[0]
        part = _ops.moe_ag_local_reduce(y, y_slot, gw, self.info, self.S, tiled=True, outputs=[self.own])[0]
        return ttnn.reduce_scatter(
            part, dim=2, cluster_axis=0, num_links=self.row_links, topology=self.row_topology
        )  # [1, 1, S, H]: this chip's row block of the column's tokens

    def __call__(self, routed_x, indices, scores, flat_expert):
        """routed_x [.., S, H] bf16 row major (replicated over the columns), indices [.., S, K] uint16 and scores
        [.., S, K] from the gate -> [1, 1, S, H / cols] bf16 TILE: sum_k w[k] expert_k(x) over every chip's experts,
        reduce-scattered over the columns on the hidden dim."""
        gx, gidx, gw = self.gather(routed_x, indices, scores)
        counts, regions, token_index, y_slot = self.plan_op(gidx)
        y = flat_expert(ttnn.reshape(gx, (self.T, self.H)), counts, regions, token_index=token_index, y_row_major=True)
        col = self.reduce(y, y_slot, gw)
        ttnn.deallocate(y)
        if self.rows > 1:
            ttnn.deallocate(gidx)
            ttnn.deallocate(gw)
        if self.cols == 1:
            return col
        return ttnn.reduce_scatter(col, dim=3, cluster_axis=1, num_links=self.col_links, topology=self.col_topology)
