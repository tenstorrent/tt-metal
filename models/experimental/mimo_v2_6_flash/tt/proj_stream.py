# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Decode projections of one MiMo layer on a TP=8 row: qkv and o (BFP8, HiFi2) and the router (bf16, HiFi4, fp32
out), for one tile row of decode users.

qkv and o stream their DRAM-interleaved weights bank-locally (BankStreamMatmul, a ``ttnn.generic_op`` program with
kernels in tt/kernels/proj/). The router reads the same x as qkv, so it runs inside the qkv program on cores qkv
leaves free (BankStreamRouter); ``router`` falls back to the DRAM-sharded matmul for any other input. When x is in
DRAM, only the first core of each bank group reads it; the group's other cores and that bank's router core copy each
block from the first core's L1 once a semaphore says it has landed.
"""

import math

import ttnn

from .common import TILE, DramShardedMatmul, compute_config
from .moe_stream import bank_vcs, dram_bank_cores

KERNEL_DIR = "models/experimental/mimo_v2_6_flash/tt/kernels/proj"


def _ranges(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _cb(index, dtype, pages, ranges):
    tile = ttnn.Tile([TILE, TILE])
    page = tile.get_tile_size(dtype)
    fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page, tile=ttnn.TileDescriptor(tile))
    return ttnn.CBDescriptor(total_size=pages * page, core_ranges=ranges, format_descriptors=[fmt])


def _semaphore(ranges):
    """Semaphore 0 (initial value 0) of a program on ``ranges``, or None if the bindings take other arguments."""
    attempts = [dict(id=0, core_ranges=ranges, initial_value=0), dict(core_ranges=ranges, initial_value=0)]
    core_type = getattr(getattr(ttnn, "CoreType", None), "Worker", None)
    if core_type is not None:
        attempts.append(dict(core_type=core_type, core_ranges=ranges, initial_value=0))
    for kwargs in attempts:
        try:
            return ttnn.SemaphoreDescriptor(**kwargs)
        except (TypeError, AttributeError):
            continue
    return None


def _noc_coords(device, cores):
    """Virtual NoC coordinates of logical worker ``cores`` (the same on every chip), or None without the lookup."""
    try:
        return [device.worker_core_from_logical_core(c) for c in cores]
    except (AttributeError, TypeError, RuntimeError):
        return None


def _kernel(name, cores, ct_args, rt_args, config):
    """KernelDescriptor for ``tt/kernels/proj/name`` with ``rt_args[i]`` on ``cores[i]``."""
    kwargs = dict(
        kernel_source=f"{KERNEL_DIR}/{name}", core_ranges=_ranges(cores), compile_time_args=ct_args, config=config
    )
    try:
        return ttnn.KernelDescriptor(runtime_args=list(zip(cores, rt_args)), **kwargs)
    except TypeError:  # bindings that index runtime args as [x][y]
        grid = [[[] for _ in range(max(c.y for c in cores) + 1)] for _ in range(max(c.x for c in cores) + 1)]
        for c, a in zip(cores, rt_args):
            grid[c.x][c.y] = a
        return ttnn.KernelDescriptor(runtime_args=grid, **kwargs)


def _is_interleaved(t):
    return t.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


class BankStreamMatmul:
    """Decode matmul x [1, 1, 32, K] bf16 @ w [K, N] (DRAM-interleaved tiles) -> [1, 1, 32, N] bf16 in L1.

    Tile column n of an interleaved weight with 8 * cpb tile columns lives in bank n % 8, so bank b holds kt rows of
    cpb tiles (columns b, b + 8, ...), contiguous per row. GROUP cores in the row of bank b's reader core split those
    columns; each streams its slice of every row from bank b and accumulates the whole K in Dest, so no core waits on
    another for weights. N that is not a multiple of 8 tiles is padded once by ``prepare``. Only rows 0-3 (the users)
    of x are used; rows 4-31 of the output are zero. With x in DRAM, the group's first core (the loader) reads x and
    its peers copy it from the loader's L1, so x leaves DRAM once per bank group. ``fused`` (kernels, cbs, tensors,
    x cores, x receivers) adds an independent section on other cores to the same launch; its x cores share cb_x and
    its receivers (bank, NoC coordinate) are signalled by that bank's loader."""

    GROUP = 4
    BLOCK_BYTES = 18432
    NUM_BUFFERS = 5
    IN_FLIGHT = 3
    MAX_PACKET = 16384
    MAX_DEST = 4  # fp32 tiles in half-sync Dest
    CB_X, CB_W, CB_OUT, CB_ZERO = 0, 1, 2, 3
    SEM_X = 0  # tiles of x the bank's loader has landed

    def __init__(self, plan, k, n, compute_kernel_config):
        device = plan.mesh_device
        self.plan, self.k, self.n = plan, k, n
        self.banks = device.dram_grid_size().x
        self.kt = k // TILE
        self.cpb = math.ceil(n / (TILE * self.banks))
        self.n_pad = self.cpb * TILE * self.banks
        assert k % TILE == 0 and n % TILE == 0 and self.cpb % self.GROUP == 0, (k, n)
        self.cpc = self.cpb // self.GROUP
        assert self.cpc <= self.MAX_DEST, self.cpc
        grid = device.compute_with_storage_grid_size()
        taken, self.cores, self.slots = set(), [], []
        for b, c in enumerate(dram_bank_cores(device)):
            xs = [x for x in list(range(c.x, grid.x)) + list(range(c.x - 1, -1, -1)) if (x, c.y) not in taken]
            assert len(xs) >= self.GROUP, (b, c)
            for q, x in enumerate(xs[: self.GROUP]):
                taken.add((x, c.y))
                self.cores.append(ttnn.CoreCoord(x, c.y))
                self.slots.append((b, q))
        self.vcs = bank_vcs(self.cores)
        self.noc_cores = _noc_coords(device, self.cores)
        self.can_share = self.noc_cores is not None and _semaphore(_ranges(self.cores)) is not None
        cfg = compute_kernel_config
        self.compute_config = ttnn.ComputeConfigDescriptor(
            math_fidelity=cfg.math_fidelity,
            math_approx_mode=cfg.math_approx_mode,
            fp32_dest_acc_en=cfg.fp32_dest_acc_en,
            dst_full_sync_en=False,
        )

    def _sizes(self, w_dtype):
        w_tile = ttnn.Tile([TILE, TILE]).get_tile_size(w_dtype)
        row_bytes = self.cpc * w_tile
        assert row_bytes <= self.MAX_PACKET
        block_rows = max(1, min(self.kt, self.BLOCK_BYTES // row_bytes))
        while self.kt % block_rows:
            block_rows -= 1
        return w_tile, row_bytes, block_rows

    def x_loaders(self, x):
        """Per bank, the NoC coordinate of the core that reads x for its group, or None when every core reads x
        itself (x in L1, or bindings without the coordinate / semaphore lookups)."""
        if not self.can_share or x.memory_config().buffer_type != ttnn.BufferType.DRAM:
            return None
        loaders = [None] * self.banks
        for (b, q), noc in zip(self.slots, self.noc_cores):
            if q == 0:
                loaders[b] = noc
        return loaders

    def prepare(self, w):
        """The weight this matmul reads: ``w`` itself, or once padded on device to n_pad columns (zeros)."""
        if self.n_pad == self.n:
            return w
        out = ttnn.empty(
            [self.k, self.n_pad],
            dtype=w.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.plan.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        core = [ttnn.CoreCoord(0, 0)]
        nt_src = self.n // TILE
        w_tile = ttnn.Tile([TILE, TILE]).get_tile_size(w.dtype)
        kernel = _kernel(
            "proj_pad_reader.cpp",
            core,
            [0, self.kt, nt_src, self.n_pad // TILE, w_tile],
            [[w.buffer_address(), out.buffer_address()]],
            ttnn.ReaderConfigDescriptor(),
        )
        program = ttnn.ProgramDescriptor(
            kernels=[kernel], semaphores=[], cbs=[_cb(0, w.dtype, nt_src + 1, _ranges(core))]
        )
        ttnn.generic_op([w, out], program)
        return out

    def __call__(self, x, w, fused=None):
        assert x.dtype == ttnn.bfloat16 and x.layout == ttnn.TILE_LAYOUT
        assert _is_interleaved(x)
        out = ttnn.empty(
            [1, 1, TILE, self.n],
            dtype=x.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.plan.mesh_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        w_tile, row_bytes, block_rows = self._sizes(w.dtype)
        num_blocks = self.kt // block_rows
        block_tiles = block_rows * self.cpc
        x_dram = 1 if x.memory_config().buffer_type == ttnn.BufferType.DRAM else 0
        w_addr, x_addr, out_addr = w.buffer_address(), x.buffer_address(), out.buffer_address()
        nt = self.n // TILE
        f_kernels, f_cbs, f_tensors, x_cores, x_receivers = fused if fused is not None else ([], [], [], [], [])
        loaders = self.x_loaders(x)
        assert loaders is not None or not x_receivers
        peers = [[] for _ in range(self.banks)]
        if loaders is not None:
            for (b, q), noc in zip(self.slots, self.noc_cores):
                if q:
                    peers[b].append(noc)
            for b, noc in x_receivers:
                peers[b].append(noc)
        share_len = 4 + 2 * max(len(p) for p in peers)
        reader_rt, writer_rt = [], []
        for (b, q), vc in zip(self.slots, self.vcs):
            first = b + self.banks * q * self.cpc
            valid = sum(1 for j in range(self.cpc) if first + self.banks * j < nt)
            reader_rt.append([b, vc, w_addr, q * row_bytes])
            if loaders is None:
                share = [0, 0, 0, 0]
            elif q == 0:
                share = [1, 0, 0, len(peers[b])] + [v for c in peers[b] for v in (c.x, c.y)]
            else:
                share = [2, loaders[b].x, loaders[b].y, 0]
            writer_rt.append([x_addr, out_addr, first, valid] + share + [0] * (share_len - len(share)))

        ranges = _ranges(self.cores)
        x_ranges = _ranges(self.cores + list(x_cores))
        cbs = [
            _cb(self.CB_X, x.dtype, self.kt, x_ranges),
            _cb(self.CB_W, w.dtype, self.NUM_BUFFERS * block_tiles, ranges),
            _cb(self.CB_OUT, x.dtype, self.cpc, ranges),
            _cb(self.CB_ZERO, x.dtype, 1, ranges),
        ]
        reader = _kernel(
            "proj_reader.cpp",
            self.cores,
            [
                self.CB_W,
                num_blocks,
                block_rows,
                block_tiles,
                row_bytes,
                self.cpb * w_tile,
                self.NUM_BUFFERS,
                self.IN_FLIGHT,
            ],
            reader_rt,
            ttnn.ReaderConfigDescriptor(),
        )
        writer = _kernel(
            "proj_writer.cpp",
            self.cores,
            [self.CB_X, self.CB_OUT, self.CB_ZERO, x_dram, num_blocks, block_rows, self.cpc, self.banks, self.SEM_X],
            writer_rt,
            ttnn.WriterConfigDescriptor(),
        )
        compute = _kernel(
            "proj_compute.cpp",
            self.cores,
            [self.CB_X, self.CB_W, self.CB_OUT, num_blocks, block_rows, self.cpc],
            [[] for _ in self.cores],
            self.compute_config,
        )
        kernels = [reader, writer, compute] + list(f_kernels)
        cbs += list(f_cbs)
        tensors = [x, w] + list(f_tensors)
        semaphores = [_semaphore(x_ranges)] if loaders is not None else []
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)
        ttnn.generic_op(tensors + [out], program)
        return out


class BankStreamRouter:
    """Router x [1, 1, 32, K] bf16 @ w [K, 8 * 32] bf16 (DRAM width-sharded, bank b holds tile column b as kt
    contiguous tiles) -> [1, 1, 32, 8 * 32] fp32 DRAM-interleaved, as a section of a BankStreamMatmul launch whose
    cb_x it shares.

    One core per bank, the first core in its bank core's row that ``busy`` leaves free, streams its shard in
    BLOCK_TILES-tile packets and accumulates the whole K of its output tile in Dest (the router's fidelity and fp32
    accumulation). Rows 0-3 of the output are the users, rows 4-31 zero."""

    BLOCK_TILES = 8  # 8 x 2 KB bf16 tiles: one 16 KB packet
    NUM_BUFFERS = 4
    IN_FLIGHT = 2
    MAX_PACKET = 16384
    CB_W, CB_OUT, CB_ZERO = 5, 6, 7  # disjoint from the host section's CBs 0-3; x is the host's cb_x

    def __init__(self, plan, k, n, compute_kernel_config, busy):
        device = plan.mesh_device
        self.plan, self.k, self.n = plan, k, n
        self.banks = device.dram_grid_size().x
        self.kt = k // TILE
        assert n == TILE * self.banks and k % (TILE * self.BLOCK_TILES) == 0, (k, n)
        grid = device.compute_with_storage_grid_size()
        busy = list(busy)
        taken = {(c.x, c.y) for c in busy}
        self.cores = []
        for c in dram_bank_cores(device):
            row = list(range(c.x, grid.x)) + list(range(c.x - 1, -1, -1))
            free = [(x, c.y) for x in row if (x, c.y) not in taken]
            if not free:
                free = sorted(
                    ((x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in taken),
                    key=lambda p: (abs(p[1] - c.y), abs(p[0] - c.x)),
                )
            assert free, c
            taken.add(free[0])
            self.cores.append(ttnn.CoreCoord(*free[0]))
        self.vcs = bank_vcs(busy + self.cores)[len(busy) :]
        self.noc_cores = _noc_coords(device, self.cores)
        cfg = compute_kernel_config
        self.compute_config = ttnn.ComputeConfigDescriptor(
            math_fidelity=cfg.math_fidelity,
            math_approx_mode=cfg.math_approx_mode,
            fp32_dest_acc_en=cfg.fp32_dest_acc_en,
            dst_full_sync_en=False,
        )

    def accepts(self, x, w):
        if x.dtype != ttnn.bfloat16 or x.layout != ttnn.TILE_LAYOUT or not _is_interleaved(x):
            return False
        if tuple(x.shape)[-2:] != (TILE, self.k):
            return False
        if w.dtype != ttnn.bfloat16 or w.layout != ttnn.TILE_LAYOUT:
            return False
        mc = w.memory_config()
        return (
            mc.buffer_type == ttnn.BufferType.DRAM
            and mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
            and mc.shard_spec is not None
            and tuple(mc.shard_spec.shape) == (self.k, TILE)
        )

    def output(self):
        return ttnn.empty(
            [1, 1, TILE, self.n],
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.plan.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def section(self, x, w, out, loaders=None):
        """(kernels, cbs, tensors, x cores, x receivers) computing ``out`` = x @ w on this router's cores. x lives in
        the host section's cb_x; with ``loaders`` (per bank, the NoC coordinate of the core that reads x), each router
        core copies x from its bank's loader instead of reading DRAM."""
        share = loaders is not None and self.noc_cores is not None
        packet = self.BLOCK_TILES * ttnn.Tile([TILE, TILE]).get_tile_size(w.dtype)
        assert packet <= self.MAX_PACKET
        num_blocks = self.kt // self.BLOCK_TILES
        x_dram = 1 if x.memory_config().buffer_type == ttnn.BufferType.DRAM else 0
        w_addr, x_addr, out_addr = w.buffer_address(), x.buffer_address(), out.buffer_address()
        ranges = _ranges(self.cores)
        cb_x = BankStreamMatmul.CB_X
        cbs = [
            _cb(self.CB_W, w.dtype, self.NUM_BUFFERS * self.BLOCK_TILES, ranges),
            _cb(self.CB_OUT, out.dtype, 1, ranges),
            _cb(self.CB_ZERO, out.dtype, 1, ranges),
        ]
        reader = _kernel(
            "proj_reader.cpp",
            self.cores,
            [self.CB_W, num_blocks, 1, self.BLOCK_TILES, packet, packet, self.NUM_BUFFERS, self.IN_FLIGHT],
            [[b, vc, w_addr, 0] for b, vc in enumerate(self.vcs)],
            ttnn.ReaderConfigDescriptor(),
        )
        writer_rt = []
        for b in range(len(self.cores)):
            x_args = [2, loaders[b].x, loaders[b].y, 0] if share else [0, 0, 0, 0]
            writer_rt.append([x_addr, out_addr, b] + x_args)
        writer = _kernel(
            "router_writer.cpp",
            self.cores,
            [cb_x, self.CB_OUT, self.CB_ZERO, x_dram, num_blocks, self.BLOCK_TILES, BankStreamMatmul.SEM_X],
            writer_rt,
            ttnn.WriterConfigDescriptor(),
        )
        compute = _kernel(
            "proj_compute.cpp",
            self.cores,
            [cb_x, self.CB_W, self.CB_OUT, num_blocks, self.BLOCK_TILES, 1],
            [[] for _ in self.cores],
            self.compute_config,
        )
        receivers = list(enumerate(self.noc_cores)) if share else []
        return [reader, writer, compute], cbs, [w, out], list(self.cores), receivers


class DecodeProjections:
    """``qkv(x)``, ``o(x)`` and ``router(x)`` for x [1, 1, 32, K] bf16 (users in the leading rows)."""

    def __init__(self, plan, hidden, qkv_width, o_in, num_experts):
        self.plan = plan
        self.hidden, self.qkv_width, self.o_in, self.num_experts = hidden, qkv_width, o_in, num_experts
        self.mm_config = compute_config(ttnn.MathFidelity.HiFi2)
        self.router_config = compute_config(ttnn.MathFidelity.HiFi4, fp32_acc=True)
        self.qkv_mm = BankStreamMatmul(plan, hidden, qkv_width, self.mm_config)
        self.o_mm = BankStreamMatmul(plan, o_in, hidden, self.mm_config)
        self.router_mm = DramShardedMatmul(
            plan, hidden, num_experts, out_dtype=ttnn.float32, compute_kernel_config=self.router_config
        )
        banks = plan.mesh_device.dram_grid_size().x
        self.router_stream = None
        if num_experts == TILE * banks and hidden % (TILE * BankStreamRouter.BLOCK_TILES) == 0:
            # shares the qkv launch's cb_x, sized for the same K
            self.router_stream = BankStreamRouter(plan, hidden, num_experts, self.router_config, self.qkv_mm.cores)
        self._w_router = None
        # (x, w_router, out) of the router computed in the last qkv launch; handed out once by ``router``.
        self._fused_router = None

    def prepare_weights(self, w_qkv, w_o, w_router):
        """Device weights from the model's (DRAM interleaved BFP8 qkv [K, qkv_width] and o [o_in, hidden] per chip;
        the router [hidden, E] bf16 in the DRAM-sharded layout). qkv is padded to a multiple of 8 tile columns so
        every DRAM bank holds whole columns. Returns the tensors this class reads."""
        self._w_router = w_router
        return self.qkv_mm.prepare(w_qkv), self.o_mm.prepare(w_o), w_router

    def qkv(self, x, w):
        """qkv(x); with the prepared router weight, the same launch also computes router(x) on idle cores."""
        self._fused_router = None
        w_router = self._w_router
        if self.router_stream is None or w_router is None or not self.router_stream.accepts(x, w_router):
            return self.qkv_mm(x, w)
        r_out = self.router_stream.output()
        section = self.router_stream.section(x, w_router, r_out, self.qkv_mm.x_loaders(x))
        out = self.qkv_mm(x, w, fused=section)
        self._fused_router = (x, w_router, r_out)
        return out

    def o(self, x, w):
        return self.o_mm(x, w)

    def router(self, x, w):
        fused, self._fused_router = self._fused_router, None
        if fused is not None and fused[0] is x and fused[1] is w:
            return fused[2]
        return self.router_mm(x, w)
