# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bank-local streaming decode projections on a TP=8 row, for one tile row of decode users (rows 0-3).

BankStreamMatmul streams a DRAM-interleaved weight bank-locally (qkv and o, BFP8, HiFi2) and RouterScoreStream
computes the router scores (bf16, HiFi4, fp32 out) with the sigmoid and the bias add; both are ``ttnn.generic_op``
programs with kernels in tt/kernels/proj/. When x is in DRAM, only the first core of each bank group reads it; the
group's other cores copy each block from the first core's L1 once a semaphore says it has landed.
"""

import math

import torch

import ttnn

from .common import TILE
from .moe_stream import bank_vcs, dram_bank_cores
from .moe_stream import noc_coords as _noc_coords
from .moe_stream import semaphore_descriptor as _semaphore

KERNEL_DIR = "models/experimental/mimo_v2_6_flash/tt/kernels/proj"


def _ranges(cores):
    """``cores`` as few core ranges as possible (row runs merged into rectangles): dispatch writes each kernel binary
    once per range, so 32 single-core ranges cost a short program several microseconds per launch."""
    rows = {}
    for c in cores:
        rows.setdefault(c.y, set()).add(c.x)
    runs = []
    for y, xs in rows.items():
        xs = sorted(xs)
        start = prev = xs[0]
        for x in xs[1:]:
            if x != prev + 1:
                runs.append((start, prev, y))
                start = x
            prev = x
        runs.append((start, prev, y))
    rects = []
    for x0, x1, y in sorted(runs):
        if rects and rects[-1][:2] == [x0, x1] and rects[-1][3] == y - 1:
            rects[-1][3] = y
        else:
            rects.append([x0, x1, y, y])
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for x0, x1, y0, y1 in rects]
    )


def _cb(index, dtype, pages, ranges, rows=TILE):
    tile = ttnn.Tile([rows, TILE])
    page = tile.get_tile_size(dtype)
    fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page, tile=ttnn.TileDescriptor(tile))
    return ttnn.CBDescriptor(total_size=pages * page, core_ranges=ranges, format_descriptors=[fmt])


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
    its peers copy it from the loader's L1, so x leaves DRAM once per bank group."""

    GROUP = 4
    BLOCK_BYTES = 18432
    NUM_BUFFERS = 5
    IN_FLIGHT = 3
    MAX_PACKET = 16384
    MAX_DEST = 4  # fp32 tiles in half-sync Dest
    CB_X, CB_W, CB_OUT, CB_ZERO, CB_PART, CB_GATHER = 0, 1, 2, 3, 4, 5
    SEM_X = 0  # tiles of x the bank's loader has landed
    SEM_PART = 3  # k_split: partials landed in this core's cb_gather

    def __init__(
        self,
        plan,
        k,
        n,
        compute_kernel_config,
        skip_bank_core=False,
        group=None,
        avoid=(),
        x_rows=TILE,
        own_rows=False,
        k_split=False,
    ):
        """``skip_bank_core``: use the ``group`` (default GROUP) cores after each bank core in its row instead of
        starting at it. A group larger than GROUP takes GROUP cores per row, the bank core's row first and then the
        nearest rows (the same columns, close to the bank); ``avoid``: logical (x, y) cores to leave free (another
        section of the same launch). ``x_rows``: the height of the x and output tiles in L1 (8, 16 or 32): the users
        sit in rows 0-3, and a shorter x tile cuts the matmul per K tile (4 of a full tile's 16 MVMULs at 8 rows).
        ``own_rows`` (group <= GROUP): give every bank group a core row no other group uses (its bank core's row when
        that is free, else the nearest such row, south first, rows with ``avoid`` cores last), so no two groups' DRAM
        streams share a row's NoC links; the greedy order above when no such row is left. ``k_split`` (``section``
        only): the group's cores split K instead of the columns (each streams every column slot of its K slice and
        accumulates them in full-sync Dest), then each core sums the group's fp32 partials of its own columns."""
        assert x_rows in (8, 16, TILE), x_rows
        self.x_rows = x_rows
        device = plan.mesh_device
        self.plan, self.k, self.n = plan, k, n
        self.group = group or self.GROUP
        self.banks = device.dram_grid_size().x
        self.kt = k // TILE
        self.cpb = math.ceil(n / (TILE * self.banks))
        self.n_pad = self.cpb * TILE * self.banks
        assert k % TILE == 0 and n % TILE == 0 and self.cpb % self.group == 0, (k, n)
        self.cpc = self.cpb // self.group
        assert self.cpc <= self.MAX_DEST, self.cpc
        self.k_split = self.group if k_split else 0
        if self.k_split:
            kq = self.kt // self.k_split
            assert self.group <= self.GROUP and kq * self.k_split == self.kt and kq % self.banks == 0, (k, n)
            assert self.cpb <= 2 * self.MAX_DEST and x_rows in (16, TILE), (self.cpb, x_rows)
            assert compute_kernel_config.fp32_dest_acc_en
        grid = device.compute_with_storage_grid_size()
        bank_cores = list(dram_bank_cores(device))
        taken, self.cores, self.slots = set(avoid), [], []
        if skip_bank_core:
            taken |= {(c.x, c.y) for c in bank_cores}
        avoid_rows = {y for _, y in avoid}
        group_rows = set()
        for b, c in enumerate(bank_cores):
            order = list(range(c.x, grid.x)) + list(range(c.x - 1, -1, -1))
            picked = None
            if own_rows and self.group <= self.GROUP:
                near = sorted((y for y in range(grid.y) if y != c.y), key=lambda y: (abs(y - c.y), y < c.y))
                rows = [c.y] + [y for y in near if y not in avoid_rows] + [y for y in near if y in avoid_rows]
                for y in rows:
                    free = [(x, y) for x in order if (x, y) not in taken]
                    if y not in group_rows and len(free) >= self.group:
                        picked = free[: self.group]
                        break
            if picked is None:
                picked = []
                for y in sorted(range(grid.y), key=lambda y: (abs(y - c.y), y)):
                    row = [(x, y) for x in order]
                    picked += [p for p in row if p not in taken][: min(self.GROUP, self.group - len(picked))]
                    taken.update(picked)
                    if len(picked) == self.group:
                        break
            assert len(picked) == self.group, (b, c)
            taken.update(picked)
            group_rows.update(y for _, y in picked)
            for q, (x, y) in enumerate(picked):
                self.cores.append(ttnn.CoreCoord(x, y))
                self.slots.append((b, q))
        self.vcs = bank_vcs(self.cores)
        self.noc_cores = _noc_coords(device, self.cores)
        self.can_share = self.noc_cores is not None and _semaphore(_ranges(self.cores)) is not None
        cfg = compute_kernel_config
        self.compute_config = ttnn.ComputeConfigDescriptor(
            math_fidelity=cfg.math_fidelity,
            math_approx_mode=cfg.math_approx_mode,
            fp32_dest_acc_en=cfg.fp32_dest_acc_en,
            dst_full_sync_en=bool(self.k_split),
        )
        if self.k_split:
            modes = [ttnn.UnpackToDestMode.Default] * 64
            modes[self.CB_GATHER] = ttnn.UnpackToDestMode.UnpackToDestFp32
            self.compute_config.unpack_to_dest_mode = modes

    def _sizes(self, w_dtype, k_split=0):
        """(weight tile bytes, bytes per K row a core reads, K rows per block, K tiles per core, matmul columns)."""
        w_tile = ttnn.Tile([TILE, TILE]).get_tile_size(w_dtype)
        cols = self.cpb if k_split else self.cpc
        kt = self.kt // k_split if k_split else self.kt
        row_bytes = cols * w_tile
        assert row_bytes <= self.MAX_PACKET
        block_rows = max(1, min(kt, self.BLOCK_BYTES // row_bytes))
        while kt % block_rows:
            block_rows -= 1
        return w_tile, row_bytes, block_rows, kt, cols

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

    # Prefetch (``section``): semaphores of the consumer cores, raised by producers in the same launch.
    SEM_START = 1  # the producers are done with DRAM
    SEM_X_READY = 2  # x is written
    PREFETCH_IN_FLIGHT = 6
    PREFETCH_GROUP = 4

    @classmethod
    def prefetching(cls, plan, k, n, compute_kernel_config, avoid=()):
        """The matmul ``section`` runs as the next layer's qkv inside the previous layer's reduce launch (``avoid``:
        that launch's other cores), or None when its layout constraints do not hold."""
        try:
            return cls(
                plan,
                k,
                n,
                compute_kernel_config,
                skip_bank_core=True,
                group=cls.PREFETCH_GROUP,
                avoid=avoid,
                x_rows=16,
                own_rows=True,
                k_split=True,
            )
        except AssertionError:
            return None

    def __call__(self, x, w, compact_out=None):
        """``compact_out``: write rows 0-3 straight into this compact rows tensor ([1, 1, 32, n / 8] bf16 DRAM tiles:
        rows 4c..4c+3 of compact tile t % (n / 256) hold rows 0-3 of tile column t, c = t / (n / 256)) and return it."""
        out = compact_out if compact_out is not None else self.output()
        kernels, cbs, semaphores = self._parts(x, w, out, compact=compact_out is not None)
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)
        ttnn.generic_op([x, w, out], program)
        return out

    def output(self):
        return ttnn.empty(
            [1, 1, TILE, self.n],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.plan.mesh_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    def section(self, x, w, start_count, x_ready_count):
        """This matmul as a section of another launch (its cores must be clear of that launch's): every core
        (``stream_cores``) streams its whole weight slice into L1 once semaphore SEM_START reaches ``start_count``
        (the producers no longer read DRAM), and each bank's x loader (``loaders``) reads x once semaphore
        SEM_X_READY reaches ``x_ready_count`` (the producers write x, then raise it). Returns (section dict, the
        [1, 1, 32, n] L1 output). x must be in DRAM (the loaders share it with their groups)."""
        assert x.memory_config().buffer_type == ttnn.BufferType.DRAM and self.x_loaders(x) is not None
        out = self.output()
        prefetch = (self.SEM_START, start_count, self.SEM_X_READY, x_ready_count)
        kernels, cbs, semaphores = self._parts(x, w, out, prefetch=prefetch)
        return dict(kernels=kernels, cbs=cbs, semaphores=semaphores, tensors=[w]), out

    def stream_cores(self):
        """NoC coordinates of every core."""
        return list(self.noc_cores)

    def loaders(self):
        """Per bank, the NoC coordinate of the core that reads x from DRAM for its group."""
        return [noc for (b, q), noc in zip(self.slots, self.noc_cores) if q == 0]

    def _parts(self, x, w, out, compact=False, prefetch=None):
        """Kernels, CBs and semaphores. ``prefetch`` = (start sem, start count, x sem, x count): stream the whole
        weight slice into L1 once the start semaphore is raised, read x once the x semaphore is (see ``section``)."""
        assert x.dtype == ttnn.bfloat16 and x.layout == ttnn.TILE_LAYOUT
        assert _is_interleaved(x)
        compact_tiles = 0
        if compact:
            compact_tiles = self.n // TILE // (TILE // 4)
            assert out.shape[-1] == compact_tiles * TILE and out.memory_config().buffer_type == ttnn.BufferType.DRAM
        ks = self.k_split if prefetch is not None else 0
        w_tile, row_bytes, block_rows, kt_core, mm_cols = self._sizes(w.dtype, ks)
        num_blocks = kt_core // block_rows
        block_tiles = block_rows * mm_cols
        # k_split: every core reads its x slice in one batch; the output is a full tile (its partials are)
        x_blocks, x_block_rows = (1, kt_core) if ks else (num_blocks, block_rows)
        out_rows = TILE if ks else self.x_rows
        num_buffers, in_flight = self.NUM_BUFFERS, self.IN_FLIGHT
        start_sem = start_count = x_sem = x_count = 0
        if prefetch is not None:
            start_sem, start_count, x_sem, x_count = prefetch
            num_buffers, in_flight = num_blocks, min(self.PREFETCH_IN_FLIGHT, num_blocks - 1)
        x_dram = 1 if x.memory_config().buffer_type == ttnn.BufferType.DRAM else 0
        w_addr, x_addr, out_addr = w.buffer_address(), x.buffer_address(), out.buffer_address()
        nt = self.n // TILE
        loaders = self.x_loaders(x)
        peers = [[] for _ in range(self.banks)]
        if loaders is not None:
            for (b, q), noc in zip(self.slots, self.noc_cores):
                if q:
                    peers[b].append(noc)
        share_len = 4 + 2 * max(len(p) for p in peers)
        owners = [[] for _ in range(self.banks)]
        for (b, q), noc in sorted(zip(self.slots, self.noc_cores), key=lambda s: s[0]):
            owners[b].append(noc)
        x_tile = ttnn.Tile([TILE, TILE]).get_tile_size(x.dtype)
        reader_rt, writer_rt = [], []
        for (b, q), vc in zip(self.slots, self.vcs):
            first = b + self.banks * q * self.cpc
            valid = sum(1 for j in range(self.cpc) if first + self.banks * j < nt)
            if ks:
                # K rows [q * kt_core, (q + 1) * kt_core), every slot with a real column
                bank_valid = sum(1 for j in range(self.cpb) if b + self.banks * j < nt)
                reader_rt.append([b, vc, w_addr, q * kt_core * self.cpb * w_tile, max(1, bank_valid) * w_tile])
            else:
                # only the slots with real columns (the valid ones come first); the rest are prepare's zero padding
                reader_rt.append([b, vc, w_addr, q * row_bytes, max(1, valid) * w_tile])
            if loaders is None:
                share = [0, 0, 0, 0]
            elif q == 0:
                role = 4 if ks else 1
                share = [role, 0, 0, len(peers[b])] + [v for c in peers[b] for v in (c.x, c.y)]
            elif ks:
                share = [3, 0, 0, 0]
            else:
                share = [2, loaders[b].x, loaders[b].y, 0]
            # k_split: x tile k of this core is tile q * kt_core + k (kt_core is a multiple of the bank count)
            xa = x_addr + (q * kt_core // self.banks) * x_tile if ks else x_addr
            args = [xa, out_addr, first, valid] + share + [0] * (share_len - len(share))
            if ks:
                args += [q] + [v for c in owners[b] for v in (c.x, c.y)]
            writer_rt.append(args)

        ranges = _ranges(self.cores)
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
                num_buffers,
                in_flight,
                start_sem,
                start_count,
            ],
            reader_rt,
            ttnn.ReaderConfigDescriptor(),
        )
        cbs = [
            _cb(self.CB_X, x.dtype, kt_core, ranges, rows=self.x_rows),
            _cb(self.CB_W, w.dtype, num_buffers * block_tiles, ranges),
            _cb(self.CB_OUT, x.dtype, self.cpc, ranges, rows=out_rows),
            _cb(self.CB_ZERO, x.dtype, 1, ranges),
        ]
        if ks:
            cbs += [
                _cb(self.CB_PART, ttnn.float32, self.cpb, ranges),
                _cb(self.CB_GATHER, ttnn.float32, ks * self.cpc, ranges),
            ]
        writer = _kernel(
            "proj_writer.cpp",
            self.cores,
            [
                self.CB_X,
                self.CB_OUT,
                self.CB_ZERO,
                x_dram,
                x_blocks,
                x_block_rows,
                self.cpc,
                self.banks,
                self.SEM_X,
                compact_tiles,
                x_sem,
                x_count,
                self.x_rows,
                out_rows,
                ks,
                self.SEM_PART if ks else 0,
                self.CB_PART,
                self.CB_GATHER,
                4 + share_len,
            ],
            writer_rt,
            ttnn.WriterConfigDescriptor(),
        )
        compute = _kernel(
            "proj_compute.cpp",
            self.cores,
            [
                self.CB_X,
                self.CB_W,
                self.CB_OUT,
                num_blocks,
                block_rows,
                mm_cols,
                ks,
                self.CB_PART,
                self.CB_GATHER,
                self.cpc,
            ],
            [[] for _ in self.cores],
            self.compute_config,
        )
        semaphores = [_semaphore(ranges)] if loaders is not None else []
        if prefetch is not None:
            semaphores += [_semaphore(ranges, start_sem), _semaphore(ranges, x_sem)]
        if ks:
            semaphores.append(_semaphore(ranges, self.SEM_PART))
        return [reader, writer, compute], cbs, semaphores


class RouterScoreStream:
    """Decode routing scores in one launch: x [1, 1, 32, K] bf16 (interleaved tiles) @ w [K, 8 * 32] bf16 (the
    DRAM width-sharded router weight: bank b holds tile column b as kt contiguous tiles) -> scores = sigmoid(logits)
    and biased = scores + bias, rows 0-3 of two persistent [1, 1, 32, E] fp32 DRAM tiles, in place of the sharded
    router matmul, its reshards, sigmoid and the bias add.

    GROUP cores per bank (the bank core and its row neighbours) split K: each prefetches its kt / GROUP weight tiles
    into L1 at launch and accumulates its slice in fp32 Dest at HiFi4; the bank core (leader) adds the others' partials
    in core order, then applies sigmoid and the bias in Dest (partials and bias unpacked to Dest in fp32)."""

    GROUP = 4
    BLOCK_TILES = 8  # 8 x 2 KB bf16 tiles: one 16 KB packet
    CB_X, CB_W, CB_OUT, CB_BIAS, CB_PART, CB_GATHER = 0, 1, 2, 3, 4, 5
    SEM = 0

    def __init__(self, plan, k, n, bias):
        """``bias``: the experts' score-correction bias, torch [n]."""
        device = plan.mesh_device
        self.plan, self.k, self.n = plan, k, n
        self.banks = device.dram_grid_size().x
        self.kt = k // TILE
        self.kq = self.kt // self.GROUP
        assert n == TILE * self.banks and self.kq * self.GROUP == self.kt and self.kq % self.BLOCK_TILES == 0, (k, n)
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
        assert self.noc_cores is not None
        self.leaders = {b: noc for (b, q), noc in zip(self.slots, self.noc_cores) if q == 0}
        self.compute_config = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            dst_full_sync_en=False,
        )
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[self.CB_BIAS] = ttnn.UnpackToDestMode.UnpackToDestFp32
        modes[self.CB_GATHER] = ttnn.UnpackToDestMode.UnpackToDestFp32
        self.compute_config.unpack_to_dest_mode = modes
        rep = plan.replicate()

        def fp32_tiles(t):
            return ttnn.from_torch(
                t.contiguous(),
                device=device,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=rep,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self.bias = fp32_tiles(bias.float().reshape(1, 1, 1, n).expand(1, 1, TILE, n))
        self.scores = fp32_tiles(torch.zeros(1, 1, TILE, n))
        self.biased = fp32_tiles(torch.zeros(1, 1, TILE, n))

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

    def __call__(self, x, w):
        """Returns the persistent (scores, biased); every call rewrites their rows 0-3."""
        w_tile = ttnn.Tile([TILE, TILE]).get_tile_size(w.dtype)
        packet = self.BLOCK_TILES * w_tile
        assert packet <= BankStreamMatmul.MAX_PACKET
        num_blocks = self.kq // self.BLOCK_TILES  # every block of the slice in flight at once
        x_dram = 1 if x.memory_config().buffer_type == ttnn.BufferType.DRAM else 0
        ranges = _ranges(self.cores)
        cbs = [
            _cb(self.CB_X, ttnn.bfloat16, self.kq, ranges),
            _cb(self.CB_W, w.dtype, num_blocks * self.BLOCK_TILES, ranges),
            _cb(self.CB_OUT, ttnn.float32, 2, ranges),
            _cb(self.CB_BIAS, ttnn.float32, 1, ranges),
            _cb(self.CB_PART, ttnn.float32, 1, ranges),
            _cb(self.CB_GATHER, ttnn.float32, max(1, self.GROUP - 1), ranges),
        ]
        w_addr = w.buffer_address()
        reader = _kernel(
            "proj_reader.cpp",
            self.cores,
            [self.CB_W, num_blocks, 1, self.BLOCK_TILES, packet, packet, num_blocks + 1, num_blocks, 0, 0],
            [[b, vc, w_addr, q * self.kq * w_tile, packet] for (b, q), vc in zip(self.slots, self.vcs)],
            ttnn.ReaderConfigDescriptor(),
        )
        out_args = [x.buffer_address(), self.scores.buffer_address(), self.biased.buffer_address()]
        out_args.append(self.bias.buffer_address())
        writer_rt = []
        for b, q in self.slots:
            lead = self.leaders[b]
            writer_rt.append(out_args + [b, q, q * self.kq, lead.x, lead.y])
        writer = _kernel(
            "router_score_writer.cpp",
            self.cores,
            [
                self.CB_X,
                self.CB_OUT,
                self.CB_BIAS,
                x_dram,
                self.kq,
                self.CB_PART,
                self.CB_GATHER,
                self.GROUP,
                self.SEM,
            ],
            writer_rt,
            ttnn.WriterConfigDescriptor(),
        )
        compute = _kernel(
            "router_score_compute.cpp",
            self.cores,
            [
                self.CB_X,
                self.CB_W,
                self.CB_OUT,
                self.CB_BIAS,
                num_blocks,
                self.BLOCK_TILES,
                self.CB_PART,
                self.CB_GATHER,
                self.GROUP,
            ],
            [[1 if q == 0 else 0] for _, q in self.slots],
            self.compute_config,
        )
        program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[_semaphore(ranges)], cbs=cbs)
        ttnn.generic_op([x, w, self.bias, self.scores, self.biased], program)
        return self.scores, self.biased
