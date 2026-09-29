# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""All-gather MoE block (``MIMO_MOE_AG=1``): the routed-expert data movement without dispatch / combine.

    x, topk (indices, weights)  --high_bw_all_gather over the dispatch axis (mesh rows)-->  every chip of a mesh column
    holds the column's chunk_size tokens (gathered row g = src_row * chunk_size_per_chip + token)
    -> RoutePlan (on device): counts / regions (the flat expert's rows), token_index (flat row -> gathered row),
       y_slot (per (token, k) the flat row of its expert output on this chip, or none)
    -> flat routed expert in indexed mode (reads gathered x rows through token_index) -> y [rows, H] bfp8 TILE
    -> LocalReduce: partial[g] = sum over this chip's experts of w[g, k] * y[y_slot[g, k]] (bf16 rows)
    -> send-back over the dispatch axis (2 rows: exchange through high_bw_all_gather + add; else reduce_scatter)
    -> all-reduce over the mesh columns (high_bw_all_gather + add) -> [1, 1, chunk_size_per_chip, H] replicated.

Kernels: tt/kernels/moe_ag/ (generic_op; one program for the whole mesh, per-device data through small per-device
tensors: the local-slot map and the chip's row).
"""

import os

import torch

import ttnn

KDIR = "models/demos/mimo_v2_d_p/tt/kernels/moe_ag"
NONE = 0xFFFFFFFF


def _crs(cores):
    from models.demos.mimo_v2_d_p.tt.flat_expert import _crs as crs

    return crs(cores)


def grid_cores(mesh_device, n):
    """The first n logical worker cores, row major (y outer)."""
    g = mesh_device.compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(x, y) for y in range(g.y) for x in range(g.x)]
    assert n <= len(cores), (n, len(cores))
    return cores[:n]


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


def _cb(i, size, crs, page=None):
    return ttnn.CBDescriptor(
        total_size=size,
        core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.uint32, page_size=page or size)],
    )


def _up(v, a):
    return -(-v // a) * a


class RoutePlan:
    """Gathered top-k indices [T, K] uint16 -> counts / regions [1, NG], token_index [1, rows], y_slot [1, T * K]
    (uint32, DRAM, persistent). gids[d]: device d's (row-major mesh order) local experts' global ids, local order."""

    def __init__(self, mesh_device, *, tokens, k, n_global, gids, rows, num_cores=64):
        self.dev = mesh_device
        self.T, self.K, self.NG, self.rows = tokens, k, n_global, rows
        self.EPC = len(gids[0])
        self.R = num_cores  # an 8 x 8 rectangle: the histogram table is multicast over it
        assert self.EPC <= self.R == 64
        mr, mc = tuple(mesh_device.shape)
        lmap = torch.full((mr * mc, n_global), NONE, dtype=torch.int64)
        for d, gl in enumerate(gids):
            for l, g in enumerate(gl):
                lmap[d, g] = l
        self.lmap = _per_device(mesh_device, lmap, ttnn.uint32)
        self.counts = _dram(mesh_device, [1, n_global], ttnn.uint32)
        self.regions = _dram(mesh_device, [1, n_global], ttnn.uint32)
        self.token_index = _dram(mesh_device, [1, rows], ttnn.uint32)
        self.y_slot = _dram(mesh_device, [1, tokens * k], ttnn.uint32)
        self._prog = {}

    def _program(self, idx):
        R, EPC, NG, K, T = self.R, self.EPC, self.NG, self.K, self.T
        cores = [ttnn.CoreCoord(x, y) for y in range(8) for x in range(8)]
        crs = _crs(cores)
        npr = _up(-(-T // R), 2)  # tokens per range (even: 64 B aligned y_slot blocks)
        IDX_STRIDE = 64
        phys = [self.dev.worker_core_from_logical_core(c) for c in cores]
        xy = [(p.x << 16) | p.y for p in phys]
        rt = ttnn.RuntimeArgs()
        addrs = [
            idx.buffer_address(),
            self.lmap.buffer_address(),
            self.counts.buffer_address(),
            self.regions.buffer_address(),
            self.token_index.buffer_address(),
            self.y_slot.buffer_address(),
        ]
        for r, c in enumerate(cores):
            g0 = min(r * npr, T)
            n = max(0, min(npr, T - g0))
            rt[c.x][c.y] = addrs + [r, g0, n] + xy
        cbs = [
            _cb(0, max(64, npr * IDX_STRIDE), crs),
            _cb(1, NG * 4, crs),
            _cb(2, _up(R * 4, 64), crs),
            _cb(3, _up(3 * EPC * 4, 64), crs),
            _cb(4, max(64, _up(npr * K * 4, 64)), crs),
            _cb(5, _up(T, 32) * 4, crs),
            _cb(6, 2 * NG * 4, crs),
            _cb(7, _up(EPC * 4 + 16, 64), crs),
        ]
        kernel = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/route_plan.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs,
            compile_time_args=[R, EPC, NG, K, IDX_STRIDE, phys[0].x, phys[0].y, phys[-1].x, phys[-1].y],
            runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
        )
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=crs, initial_value=0) for i in range(6)]
        return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=sems, cbs=cbs)

    def __call__(self, idx):
        """idx: gathered top-k [.., T, K] uint16 row major DRAM. Returns (counts, regions, token_index, y_slot)."""
        key = idx.buffer_address()
        if key not in self._prog:
            self._prog[key] = self._program(idx)
        ttnn.generic_op([idx, self.lmap, self.counts, self.regions, self.token_index, self.y_slot], self._prog[key])
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


def _dm(proc, noc):
    return ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_0 if proc == 0 else ttnn.DataMovementProcessor.RISCV_1,
        noc=ttnn.NOC.NOC_0 if noc == 0 else ttnn.NOC.NOC_1,
    )


def _kd(src, crs, ct, rt, config, defines=()):
    return ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/{src}",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=crs,
        compile_time_args=ct,
        runtime_args=rt,
        defines=list(defines),
        config=config,
    )


def _tcb(i, tiles, crs, fmt=ttnn.bfloat16, page=2048):
    return ttnn.CBDescriptor(
        total_size=tiles * page,
        core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt, page_size=page)],
    )


def _ranges(total, parts, align=2):
    per = _up(-(-total // parts), align)
    return [(min(i * per, total), max(0, min(per, total - i * per))) for i in range(parts)]


class LocalReduce:
    """partial[g] = sum over this chip's local experts of w[g, k] * y[y_slot[g, k]] for the column's T tokens.
    y: row-major bf16 [rows, H]. split (2 mesh rows): own [S, H] (this chip's row) and other [S, H] (the peer's)."""

    def __init__(self, mesh_device, *, tokens, k, hidden, chunk_size_per_chip, split, info, pairs_depth=4, tiled=False):
        self.dev, self.T, self.K, self.H, self.S, self.split, self.info = (
            mesh_device,
            tokens,
            k,
            hidden,
            chunk_size_per_chip,
            split,
            info,
        )
        self.tiled = tiled and not split  # the [T, H] partials as bf16 tiles (the > 2-row reduce-scatter input)
        self.D = pairs_depth
        g = mesh_device.compute_with_storage_grid_size()
        self.cores = [ttnn.CoreCoord(x, y) for y in range(g.y) for x in range(g.x)]
        rows_out = chunk_size_per_chip if split else tokens
        self.own = _dram(
            mesh_device,
            [1, 1, rows_out, hidden],
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT if self.tiled else ttnn.ROW_MAJOR_LAYOUT,
        )
        self.other = _dram(mesh_device, [1, 1, rows_out, hidden], ttnn.bfloat16) if split else None
        self._prog = {}

    def _program(self, y, y_slot, w):
        T, K, H = self.T, self.K, self.H
        RB, TILES = H * 2, H // 1024
        crs = _crs(self.cores)
        rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        rngs = _ranges(T, len(self.cores), 32 if self.tiled else 2)
        npr = max(n for _, n in rngs)
        for c, (g0, n) in zip(self.cores, rngs):
            rrt[c.x][c.y] = [y.buffer_address(), y_slot.buffer_address(), w.buffer_address(), g0, n]
            crt[c.x][c.y] = [n]
            wrt[c.x][c.y] = (
                [self.own.buffer_address(), g0, n]
                if self.tiled
                else [
                    self.own.buffer_address(),
                    self.other.buffer_address() if self.split else 0,
                    self.info.buffer_address(),
                    g0,
                    n,
                ]
            )
        cbs = [
            _tcb(0, self.D * TILES, crs),
            _tcb(1, self.D, crs),
            _cb(2, 128, crs, page=64),
            _cb(4, _up(npr * K * 4, 64), crs),
            _cb(5, _up(npr * 64, 64), crs),
            _cb(6, RB, crs),
            _cb(7, 64, crs),
            _tcb(16, 32 * TILES if self.tiled else 2 * TILES, crs),
        ] + ([_tcb(24, 32 * TILES, crs)] if self.tiled else [])
        cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
        kernels = [
            _kd("reduce_reader.cpp", crs, [K, RB, TILES, 64], rrt, _dm(1, 1)),
            _kd("reduce_compute_t.cpp" if self.tiled else "reduce_compute.cpp", crs, [TILES], crt, cc),
            (
                _kd("reduce_writer_t.cpp", crs, [H // 32], wrt, _dm(0, 0))
                if self.tiled
                else _kd("reduce_writer.cpp", crs, [RB, TILES, self.S, int(self.split)], wrt, _dm(0, 0))
            ),
        ]
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)

    def _program2(self, y, y_slot, w, phase, peer):
        K, H, S = self.K, self.H, self.S
        RB, TILES = H * 2, H // 1024
        crs = _crs(self.cores)
        rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        rngs = _ranges(S, len(self.cores))
        npr = max(n for _, n in rngs)
        out = self.other if phase == 1 else self.own
        for c, (g0, n) in zip(self.cores, rngs):
            rrt[c.x][c.y] = [
                y.buffer_address(),
                y_slot.buffer_address(),
                w.buffer_address(),
                g0,
                n,
                self.info.buffer_address(),
                peer.buffer_address() if peer is not None else 0,
            ]
            crt[c.x][c.y] = [n]
            wrt[c.x][c.y] = [out.buffer_address(), 0, 0, g0, n]
        cbs = [
            _tcb(0, self.D * TILES, crs),
            _tcb(1, self.D, crs),
            _cb(2, 128, crs, page=64),
            _cb(4, _up(npr * K * 4, 64), crs),
            _cb(5, _up(npr * 64, 64), crs),
            _cb(6, RB, crs),
            _cb(7, 64, crs),
            _tcb(16, 2 * TILES, crs),
        ]
        kernels = [
            _kd("reduce2_reader.cpp", crs, [K, RB, TILES, 64, phase], rrt, _dm(1, 1)),
            _kd(
                "reduce_compute.cpp",
                crs,
                [TILES],
                crt,
                ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
            ),
            _kd("reduce_writer.cpp", crs, [RB, TILES, 1, 0], wrt, _dm(0, 0)),
        ]
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)

    def phase(self, y, y_slot, w, phase, peer=None):
        """Two mesh rows, fused send-back: phase 1 reduces the other row's tokens into ``other``; phase 2 this row's
        tokens plus the peer's gathered phase-1 partial (``peer`` [2 S, H]) into ``own``."""
        assert self.split
        key = (
            phase,
            y.buffer_address(),
            y_slot.buffer_address(),
            w.buffer_address(),
            peer.buffer_address() if peer is not None else 0,
        )
        if key not in self._prog:
            if len(self._prog) > 16:
                self._prog.clear()
            self._prog[key] = self._program2(y, y_slot, w, phase, peer)
        io = [y, y_slot, w, self.info] + ([peer] if peer is not None else []) + [self.other if phase == 1 else self.own]
        ttnn.generic_op(io, self._prog[key])
        return self.other if phase == 1 else self.own

    def __call__(self, y, y_slot, w):
        """y row-major bf16 [.., rows, H], y_slot [1, T K] uint32, w gathered weights [.., T, K] bf16 row major."""
        key = (y.buffer_address(), y_slot.buffer_address(), w.buffer_address())
        if key not in self._prog:
            if len(self._prog) > 16:
                self._prog.clear()
            self._prog[key] = self._program(y, y_slot, w)
        ttnn.generic_op([y, y_slot, w, self.info, self.own] + ([self.other] if self.split else []), self._prog[key])
        return (self.own, self.other) if self.split else self.own


class AddRows:
    """out[i] = a[a_off + i] + b[b_off + i] for i < n rows (row-major bf16, width H); b_off per device from the chip
    info (word 1) when info_offset, else a constant."""

    def __init__(self, mesh_device, *, n_rows, hidden, info, batch=2):
        self.dev, self.n, self.H, self.info, self.B = mesh_device, n_rows, hidden, info, batch
        g = mesh_device.compute_with_storage_grid_size()
        self.cores = [ttnn.CoreCoord(x, y) for y in range(g.y) for x in range(g.x)]
        self.out = _dram(mesh_device, [1, 1, n_rows, hidden], ttnn.bfloat16)
        self._prog = {}

    def _program(self, a, b, a_off, b_off, info_offset):
        RB, TILES = self.H * 2, self.H // 1024
        crs = _crs(self.cores)
        rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for c, (r0, n) in zip(self.cores, _ranges(self.n, len(self.cores), 1)):
            rrt[c.x][c.y] = [a.buffer_address(), b.buffer_address(), self.info.buffer_address(), r0, n, b_off, a_off]
            crt[c.x][c.y] = [n]
            wrt[c.x][c.y] = [self.out.buffer_address(), 0, 0, r0, n]
        cbs = [
            _tcb(0, 2 * self.B * TILES, crs),
            _tcb(1, 2 * self.B * TILES, crs),
            _cb(7, 64, crs),
            _tcb(16, 2 * TILES, crs),
        ]
        kernels = [
            _kd("add_reader.cpp", crs, [RB, TILES, int(info_offset), self.B], rrt, _dm(1, 1)),
            _kd(
                "add_compute.cpp",
                crs,
                [TILES],
                crt,
                ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
            ),
            _kd("reduce_writer.cpp", crs, [RB, TILES, 1, 0], wrt, _dm(0, 0)),
        ]
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)

    def __call__(self, a, b, *, a_off=0, b_off=0, info_offset=False):
        key = (a.buffer_address(), b.buffer_address(), a_off, b_off, info_offset)
        if key not in self._prog:
            if len(self._prog) > 16:
                self._prog.clear()
            self._prog[key] = self._program(a, b, a_off, b_off, info_offset)
        ttnn.generic_op([a, b, self.info, self.out], self._prog[key])
        return self.out


def hbw_links():
    """high_bw_all_gather links (``MIMO_HBW_LINKS``; the QuietBox has 4 per axis, Galaxy 2)."""
    return int(os.environ["MIMO_HBW_LINKS"]) if os.environ.get("MIMO_HBW_LINKS") else None


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

    def __init__(self, mesh_device, *, chunk_size_per_chip, hidden, k, n_global, gids, buf_rows):
        self.dev = mesh_device
        rows, cols = tuple(mesh_device.shape)
        self.rows, self.cols = rows, cols
        S, H = chunk_size_per_chip, hidden
        self.S, self.H, self.K = S, H, k
        self.T = T = rows * S  # chunk_size: the tokens of a mesh column
        self.links = hbw_links()
        from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology

        self.sp_topo, self.tp_topo = per_axis_topology()
        self.rs_links = int(os.environ["MIMO_MOE_AG_RS_LINKS"]) if os.environ.get("MIMO_MOE_AG_RS_LINKS") else None
        # MIMO_MOE_AG_XPPR = 4: gathered x in 2 KB pages ([T * 4, 1024]: a token row over 4 DRAM banks; the indexed
        # expert reads it with x_pages_per_row = 4). Default 1: one 8 KB page per token row.
        self.xppr = int(os.environ.get("MIMO_MOE_AG_XPPR", "1"))
        assert self.xppr in (1, H // 1024), self.xppr
        self.untilize_x = UntilizeX(mesh_device, rows=S, hidden=H) if self.xppr > 1 else None
        if rows > 1:
            self.gx = _dram(mesh_device, [1, 1, T * self.xppr, H // self.xppr])
            self.gidx = _dram(mesh_device, [1, 1, T, k], ttnn.uint16)
            self.gw = _dram(mesh_device, [1, 1, T, k])
        self.plan_op = RoutePlan(mesh_device, tokens=T, k=k, n_global=n_global, gids=gids, rows=buf_rows)
        self.info = chip_info(mesh_device, S)
        split = rows == 2
        self.split = split
        self.lreduce = LocalReduce(
            mesh_device,
            tokens=T,
            k=k,
            hidden=H,
            chunk_size_per_chip=S,
            split=split,
            info=self.info,
            tiled=rows > 2 or (rows == 1 and os.environ.get("MIMO_MOE_AG_TP", "rsag" if cols > 2 else "hbw") == "rsag"),
        )
        if split:
            self.g_sp = _dram(mesh_device, [1, 1, 2 * S, H])
            self.ex = AddRows(mesh_device, n_rows=S, hidden=H, info=self.info)
        # TP all-reduce over the mesh columns (MIMO_MOE_AG_TP): "hbw" high_bw_all_gather + one add / tilize pass (the
        # default for 2 columns), "rsag" ttnn.reduce_scatter + ttnn.all_gather on tiles (default for > 2 columns:
        # 162 / 284 us vs 220 / 404 on 1x4 at 640 / 1280 tokens per chip, it moves 2 x 3/4 of the rows instead of 3 x)
        self.tp_mode = os.environ.get("MIMO_MOE_AG_TP", "rsag" if cols > 2 else "hbw")
        if cols > 1 and self.tp_mode == "hbw":
            self.g_tp = _dram(mesh_device, [1, 1, cols * S, H])
            self.tp = (
                AddRowsTiled(mesh_device, n_rows=S, hidden=H)
                if cols == 2
                else SumBlocksTiled(mesh_device, n_rows=S, hidden=H, n_blocks=cols)
            )
        # MIMO_MOE_AG_YRM (default 1): the flat expert writes y as row-major bf16 itself (pack-untilized on its down
        # cores), so no untilize pass and no [rows, H] untilized copy; 0: bfp8 tiles + UntilizeActive
        self.y_rm = os.environ.get("MIMO_MOE_AG_YRM", "1") == "1"
        self.untilize = None  # built on the first tiled y (y_rm off, or an expert without row-major output)
        self._untilize_args = dict(rows=buf_rows, hidden=H, n_global=n_global, epc=len(gids[0]), lmap=self.plan_op.lmap)

    def _ag(self, x, out, axis):
        return ttnn.experimental.high_bw_all_gather(
            x, dim=2, output_tensor=out, cluster_axis=axis, num_links=self.links
        )

    def to_rm(self, x):
        """x [1, 1, S, H] bf16 TILE -> the row-major layout the gather takes ([1, 1, S xppr, H / xppr])."""
        return self.untilize_x(x) if self.untilize_x is not None else ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

    def gather(self, x_rm, idx, w):
        """x_rm (``to_rm``), idx [1, 1, S, K] uint16 RM, w [1, 1, S, K] bf16 RM -> the column's T tokens."""
        if self.rows == 1:  # no dispatch axis: the chip's own tokens are the column's
            self.gx, self.gidx, self.gw = x_rm, idx, w
            return x_rm, idx, w
        self._ag(x_rm, self.gx, 0)
        self._ag(idx, self.gidx, 0)
        self._ag(w, self.gw, 0)
        return self.gx, self.gidx, self.gw

    def plan(self):
        return self.plan_op(self.gidx)

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

    def reduce(self, y):
        """y [rows, H] (the experts' outputs at their flat rows: row-major bf16, or bfp8 TILE without y_rm) -> a fresh
        [1, 1, S, H] bf16 TILE tensor, summed over every chip's experts, replicated over the mesh columns."""
        if y.layout == ttnn.ROW_MAJOR_LAYOUT:
            y_rm = y
        else:
            if self.untilize is None:
                self.untilize = UntilizeActive(self.dev, **self._untilize_args)
            y_rm = self.untilize(y, self.plan_op.counts, self.plan_op.regions)
        if self.split and os.environ.get("MIMO_MOE_AG_FUSED_SB", "1") == "1":
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
        return self._tp_allreduce(col)

    def _tp_allreduce(self, col):
        """[1, 1, S, H] column partial (row major, or tiles after a reduce-scatter) -> fresh TILE, summed over cols."""
        if self.cols == 1:
            return ttnn.to_layout(col, ttnn.TILE_LAYOUT) if col.layout != ttnn.TILE_LAYOUT else col
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
        return self.tp(self.g_tp, self.g_tp, b_off=self.S) if self.cols == 2 else self.tp(self.g_tp)


class UntilizeActive:
    """y bfp8 TILE [rows, H] -> y_rm bf16 RM [rows, H], only the tile rows that hold tokens (each local expert's
    ceil(count / 32) tile rows at its region; counts / regions read on device)."""

    def __init__(self, mesh_device, *, rows, hidden, n_global, epc, lmap, W=None):
        W = W or int(os.environ.get("MIMO_UA_W", "32"))
        self.dev, self.rows, self.H, self.NG, self.EPC, self.lmap, self.W = (
            mesh_device,
            rows,
            hidden,
            n_global,
            epc,
            lmap,
            W,
        )
        g = mesh_device.compute_with_storage_grid_size()
        self.cores = [ttnn.CoreCoord(x, y) for y in range(g.y) for x in range(g.x)]
        self.out = _dram(mesh_device, [rows, hidden])
        self._prog = {}

    def _program(self, y, counts, regions):
        P, W, NCH = len(self.cores), self.W, self.H // (32 * self.W)
        crs = _crs(self.cores)
        rrt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for me, c in enumerate(self.cores):
            common = [counts.buffer_address(), regions.buffer_address(), self.lmap.buffer_address(), me]
            rrt[c.x][c.y] = [y.buffer_address()] + common
            wrt[c.x][c.y] = [self.out.buffer_address()] + common
        cbs = [
            _tcb(0, 2 * W, crs, ttnn.bfloat8_b, 1088),
            _cb(2, 64, crs),
            _cb(4, 3 * self.NG * 4, crs),
            _cb(5, 3 * self.NG * 4, crs),
            _tcb(16, 2 * W, crs),
        ]
        kernels = [
            _kd("untilize_reader.cpp", crs, [self.NG, self.EPC, 1088, W, NCH, P], rrt, _dm(1, 1)),
            _kd("untilize_compute.cpp", crs, [W], ttnn.RuntimeArgs(), ttnn.ComputeConfigDescriptor()),
            _kd("untilize_writer.cpp", crs, [self.NG, self.EPC, W, NCH, P], wrt, _dm(0, 0)),
        ]
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)

    def __call__(self, y, counts, regions):
        key = (y.buffer_address(), counts.buffer_address(), regions.buffer_address())
        if key not in self._prog:
            if len(self._prog) > 16:
                self._prog.clear()
            self._prog[key] = self._program(y, counts, regions)
        ttnn.generic_op([y, counts, regions, self.lmap, self.out], self._prog[key])
        return self.out


class AddRowsTiled:
    """out[r] = a[a_off + r] + b[b_off + r] for r < n rows (row-major bf16 inputs, width H) -> a fresh bf16 TILE
    [1, 1, n, H] tensor (the add and the tilize in one pass)."""

    def __init__(self, mesh_device, *, n_rows, hidden):
        assert n_rows % 32 == 0 and hidden % 1024 == 0
        self.dev, self.n, self.H = mesh_device, n_rows, hidden
        g = mesh_device.compute_with_storage_grid_size()
        self.cores = [ttnn.CoreCoord(x, y) for y in range(g.y) for x in range(g.x)]
        self._prog = {}

    def _program(self, a, b, out, a_off, b_off):
        P, NCH = len(self.cores), self.H // 1024
        blocks = self.n // 32 * NCH
        crs = _crs(self.cores)
        rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for me, c in enumerate(self.cores):
            rrt[c.x][c.y] = [a.buffer_address(), b.buffer_address(), a_off, b_off, blocks, me]
            crt[c.x][c.y] = [len(range(me, blocks, P))]
            wrt[c.x][c.y] = [out.buffer_address(), blocks, me]
        cbs = [_tcb(0, 64, crs), _tcb(1, 64, crs), _tcb(24, 32, crs), _tcb(16, 64, crs)]
        kernels = [
            _kd("addt_reader.cpp", crs, [self.H * 2, NCH, P], rrt, _dm(1, 1)),
            _kd(
                "addt_compute.cpp",
                crs,
                [],
                crt,
                ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
            ),
            _kd("addt_writer.cpp", crs, [NCH, P, self.H // 32], wrt, _dm(0, 0)),
        ]
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)

    def __call__(self, a, b, *, a_off=0, b_off=0):
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, self.n, self.H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.dev, ttnn.DRAM_MEMORY_CONFIG
        )
        key = (a.buffer_address(), b.buffer_address(), out.buffer_address(), a_off, b_off)
        if key not in self._prog:
            if len(self._prog) > 32:
                self._prog.clear()
            self._prog[key] = self._program(a, b, out, a_off, b_off)
        ttnn.generic_op([a, b, out], self._prog[key])
        return out


class UntilizeX:
    """x bf16 TILE [1, 1, S, H] -> row-major [1, 1, S * H / 1024, 1024] (2 KB pages: token row g = pages
    NCH g .. NCH g + NCH - 1), the gathered-x page layout (spreads a token's reads over NCH DRAM banks)."""

    def __init__(self, mesh_device, *, rows, hidden):
        self.dev, self.rows, self.H = mesh_device, rows, hidden
        self.NCH = hidden // 1024
        g = mesh_device.compute_with_storage_grid_size()
        self.cores = [ttnn.CoreCoord(x, y) for y in range(g.y) for x in range(g.x)]
        self._prog = {}

    def _program(self, x, out):
        P, NCH = len(self.cores), self.NCH
        blocks = self.rows // 32 * NCH
        crs = _crs(self.cores)
        rrt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for me, c in enumerate(self.cores):
            rrt[c.x][c.y] = [x.buffer_address(), blocks, me]
            wrt[c.x][c.y] = [out.buffer_address(), blocks, me]
        cbs = [_tcb(0, 64, crs), _cb(2, 64, crs), _tcb(16, 64, crs)]
        kernels = [
            _kd("untilize_x_reader.cpp", crs, [2048, NCH, P], rrt, _dm(1, 1)),
            _kd("untilize_compute.cpp", crs, [32], ttnn.RuntimeArgs(), ttnn.ComputeConfigDescriptor()),
            _kd("untilize_x_writer.cpp", crs, [NCH, P], wrt, _dm(0, 0)),
        ]
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)

    def __call__(self, x):
        assert x.layout == ttnn.TILE_LAYOUT and x.dtype == ttnn.bfloat16 and x.shape[-2] == self.rows, x
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, self.rows * self.NCH, 1024]),
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
            self.dev,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        key = (x.buffer_address(), out.buffer_address())
        if key not in self._prog:
            if len(self._prog) > 32:
                self._prog.clear()
            self._prog[key] = self._program(x, out)
        ttnn.generic_op([x, out], self._prog[key])
        return out


class SumBlocksTiled:
    """out[r] = sum_{i < N} src[i * S + r] for r < S (row-major bf16 src [N * S, H], e.g. a gather over N chips) ->
    a fresh bf16 TILE [1, 1, S, H] tensor (the sum and the tilize in one pass)."""

    def __init__(self, mesh_device, *, n_rows, hidden, n_blocks):
        assert n_rows % 32 == 0 and hidden % 1024 == 0
        self.dev, self.n, self.H, self.N = mesh_device, n_rows, hidden, n_blocks
        g = mesh_device.compute_with_storage_grid_size()
        self.cores = [ttnn.CoreCoord(x, y) for y in range(g.y) for x in range(g.x)]
        self._prog = {}

    def _program(self, src, out):
        P, NCH = len(self.cores), self.H // 1024
        blocks = self.n // 32 * NCH
        crs = _crs(self.cores)
        rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for me, c in enumerate(self.cores):
            rrt[c.x][c.y] = [src.buffer_address(), self.n, blocks, me]
            crt[c.x][c.y] = [len(range(me, blocks, P))]
            wrt[c.x][c.y] = [out.buffer_address(), blocks, me]
        cbs = [_tcb(0, 64, crs), _tcb(24, 32, crs), _tcb(16, 64, crs)]
        kernels = [
            _kd("addn_reader.cpp", crs, [self.H * 2, NCH, P, self.N], rrt, _dm(1, 1)),
            _kd(
                "addn_compute.cpp",
                crs,
                [self.N],
                crt,
                ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
            ),
            _kd("addt_writer.cpp", crs, [NCH, P, self.H // 32], wrt, _dm(0, 0)),
        ]
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)

    def __call__(self, src):
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, self.n, self.H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.dev, ttnn.DRAM_MEMORY_CONFIG
        )
        key = (src.buffer_address(), out.buffer_address())
        if key not in self._prog:
            if len(self._prog) > 32:
                self._prog.clear()
            self._prog[key] = self._program(src, out)
        ttnn.generic_op([src, out], self._prog[key])
        return out
