# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Bank-streamed decode linear, bit-equal to tt-metal's multi-core DRAM-sharded matmul.

``y = x @ W`` for one token. The reference is ``ttnn.linear`` with a width-sharded DRAM weight (``Nbt`` tiles per
bank) and ``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(in0_block_w=KB, num_workers_per_dram_bank=C,
use_multicore_pipeline=True)``, fp32 Dest. That op gives each core of bank b a column group cn (CN = min(C, Nbt)
groups per bank), splits the group's columns into P passes of <= 8 tiles, keeps each pass's whole K in fp32 Dest and
walks the K blocks starting at block rot = grp * nblk / NGRP (grp = b * CN + cn, NGRP = banks * CN), wrapping
(bmm_dram_sharded_multicore.cpp). Each output tile depends only on that order, so this op replays it exactly:

* layout: per bank, the tiles of every pass stored in the order compute eats them ([K block in walk order][row r
  of the block][column of the pass]), the passes of one column group back to back, so a reader core streams one
  contiguous byte range of its own bank;
* compute: per pass, matmul_block(x[j * KB + r], w, ..., ct_dim = ncq) over the same walk, fp32 Dest, same fidelity;
* placement: one core per (bank, column group) on the bank's optimal NOC0 reader row (moe_bank.row_placement).

Several reference ops with the same shape and config (e.g. the 14 unembed column chunks) become one program: each
core runs its column group of every chunk. Output: row 0 of every output tile into a dense fp32 row (value n at byte
4 n), the layout tt/fused/head.py's FusedLogitsTail reads.
"""

import math

import torch

import ttnn
from models.demos.inkling2_mini.tt.fused import common as F
from models.demos.inkling2_mini.tt.fused.moe_bank import row_placement

TILE = 32
DST_TILES = 8  # fp32 Dest tiles per pass (full sync), as the reference


def reference_schedule(Kt, Nt, Nbt, C, kb):
    """The reference op's work split: [(bank, cn, rot, [(first column, ncq) per pass])], KB, nblk."""
    CN = min(C, Nbt)
    assert C <= Nbt, "K row groups (C > Nbt) are not replayed"
    KB = min(kb, Kt)
    assert Kt % KB == 0, (Kt, KB)
    nblk = Kt // KB
    P = math.ceil(math.ceil(Nbt / CN) / DST_TILES)
    banks = math.ceil(Nt / Nbt)
    out = []
    for b in range(banks):
        for cn in range(CN):
            n0, n1 = Nbt * cn // CN, Nbt * (cn + 1) // CN
            nc = n1 - n0
            rot = (b * CN + cn) * nblk // (banks * CN)
            passes = [(b * Nbt + n0 + nc * q // P, nc * (q + 1) // P - nc * q // P) for q in range(P)]
            out.append((b, cn, rot, passes))
    return out, KB, nblk


class BankLinear:
    def __init__(
        self, device, w, dtype, chunk, kb, cores_per_bank, compute_kernel_config, out_cols=None, out_dtype=ttnn.float32,
        split=1, pass_cols=None,
    ):  # fmt: skip
        """w: torch [K, N]; ``chunk`` columns per reference op (a multiple of banks * 32; N is zero-padded to whole
        chunks, as tt/dram_sharded.DramShardedLinear); kb / cores_per_bank: the reference in0_block_w /
        num_workers_per_dram_bank. out_cols: columns written (default N). out_dtype: the reference's output dtype
        (fp32 or bf16; the tile is packed from fp32 Dest as the reference packs it). split: cores per reference
        column group (its passes are divided between them; every pass is its own Dest chain, so the output is the
        same). pass_cols: cut passes to at most this many columns (fewer columns per matmul_block call)."""
        nb = device.dram_grid_size().x
        K, N = w.shape
        n_chunks = math.ceil(N / chunk)
        w = torch.nn.functional.pad(w.bfloat16(), (0, n_chunks * chunk - N))  # BFP values are exact in bf16
        self.K, self.N, self.Kt = K, N, K // TILE
        self.chunk, self.n_chunks = chunk, n_chunks
        self.out_cols = out_cols or N
        Nbt = chunk // TILE // nb
        sched, self.KB, self.nblk = reference_schedule(self.Kt, chunk // TILE, Nbt, cores_per_bank, kb)
        tiles = w.reshape(self.Kt, TILE, -1, TILE).permute(0, 2, 1, 3)  # [Kt, Nt, 32, 32]
        # core work items: (bank, rot, [(ncq, first output column tile)]) - one reference group split over `split` cores
        self.items = []
        for b, cn, rot, passes in sched:
            todo = []
            for c in range(n_chunks):
                for first, ncq in passes:
                    g0 = c * (chunk // TILE) + first
                    ncq = min(ncq, max(0, -(-self.out_cols // TILE) - g0))  # columns past out_cols: not computed
                    if ncq:
                        step = pass_cols or ncq
                        todo += [(min(step, ncq - i), g0 + i) for i in range(0, ncq, step)]
            for s_ in range(split):
                part = todo[len(todo) * s_ // split : len(todo) * (s_ + 1) // split]
                if part:
                    self.items.append((b, rot, part))
        per_bank = [[] for _ in range(nb)]
        self.offsets = []
        for b, rot, part in self.items:
            self.offsets.append(sum(t.shape[0] for t in per_bank[b]))  # first tile slot of this item in bank b
            for ncq, g0 in part:
                for p in range(self.nblk):
                    j = (p + rot) % self.nblk
                    for r in range(self.KB):
                        per_bank[b].append(tiles[j * self.KB + r, g0 : g0 + ncq])
        stacks = [torch.cat([t.reshape(-1, TILE, TILE) for t in tb]) if tb else None for tb in per_bank]
        rows = max(len(s_) for s_ in stacks if s_ is not None)
        pages = torch.zeros(rows, nb, TILE, TILE, dtype=torch.bfloat16)
        for b, s_ in enumerate(stacks):
            if s_ is not None:
                pages[: len(s_), b] = s_
        del stacks, per_bank, tiles, w
        # page p = slot * nb + bank (DRAM interleaved)
        self.weight = ttnn.from_torch(pages.reshape(1, 1, -1, TILE), dtype=dtype, layout=ttnn.TILE_LAYOUT,
                                      device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        self.device, self.nb, self.dtype = device, nb, dtype
        counts = [sum(1 for it in self.items if it[0] == b) for b in range(nb)]
        place = row_placement(device, counts)
        seen = [0] * nb
        self.cores = []
        for i, (b, rot, part) in enumerate(self.items):
            self.cores.append((place[b][seen[b]], i))
            seen[b] += 1
        self.grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c, _ in self.cores])
        self.max_ncq = max(ncq for _, _, part in self.items for ncq, _ in part)
        self.out_dtype = out_dtype
        cfg = compute_kernel_config
        self.compute = ttnn.ComputeConfigDescriptor(
            math_fidelity=cfg.math_fidelity, fp32_dest_acc_en=True, dst_full_sync_en=True,
            math_approx_mode=cfg.math_approx_mode,
        )  # fmt: skip
        assert cfg.fp32_dest_acc_en, "the replayed schedule is the fp32 Dest one"

    def __call__(self, x, out=None):
        """x [1,1,M,K] bf16 TILE, M <= 32 (any memory). out=None: returns [1,1,M,out_cols] TILE DRAM interleaved
        in out_dtype (the reference's output after to_memory_config(DRAM) + slice: every row of each tile). out given:
        a dense fp32 ROW_MAJOR [1,1,1,W] row (W >= out_cols) receiving row 0 only (tt/fused/head.py's raw row)."""
        tb = ttnn.Tile([TILE, TILE]).get_tile_size(self.dtype)
        tiles_out = out is None
        if tiles_out:
            shape = ttnn.Shape([1, 1, x.shape[-2], self.out_cols])
            out = ttnn.allocate_tensor_on_device(shape, self.out_dtype, ttnn.TILE_LAYOUT, self.device,
                                                 ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        else:
            assert self.out_dtype == ttnn.float32 and x.shape[-2] == 1
        otb = ttnn.Tile([TILE, TILE]).get_tile_size(self.out_dtype)
        named = dict(kt=self.Kt, kb=self.KB, nblk=self.nblk, n_banks=self.nb, out_cols=self.out_cols,
                     out_page=otb if tiles_out else out.shape[-1] * 4, max_ncq=self.max_ncq, tiles_out=int(tiles_out),
                     out_tile=otb)  # fmt: skip
        r_args, c_args, w_args = {}, {}, {}
        for c, i in self.cores:
            b, rot, part = self.items[i]
            core = ttnn.CoreCoord(*c)
            r_args[core] = [self.weight.buffer_address(), b, self.offsets[i], len(part)] + [n for n, _ in part]
            c_args[core] = [rot, len(part)] + [n for n, _ in part]
            w_args[core] = [x.buffer_address(), rot, out.buffer_address(), len(part)] + [v for p_ in part for v in p_]
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        blk = self.KB * self.max_ncq
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=f"{F.KDIR}/bank_linear_reader.cpp", core_ranges=self.grid,
                compile_time_args=acc(self.weight), named_compile_time_args=list(named.items()),
                runtime_args=[(k, v) for k, v in r_args.items()],
                config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"{F.KDIR}/bank_linear_compute.cpp", core_ranges=self.grid,
                named_compile_time_args=list(named.items()), runtime_args=[(k, v) for k, v in c_args.items()],
                config=self.compute,
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"{F.KDIR}/bank_linear_writer.cpp", core_ranges=self.grid,
                compile_time_args=acc(x) + acc(out), named_compile_time_args=list(named.items()),
                runtime_args=[(k, v) for k, v in w_args.items()],
                config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
            ),
        ]  # fmt: skip
        t32 = ttnn.TileDescriptor(ttnn.Tile([TILE, TILE]))

        def cb(i, fmt, page, n):
            return ttnn.CBDescriptor(total_size=page * n, core_ranges=self.grid,
                                     format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt, page_size=page, tile=t32)])  # fmt: skip

        cbs = [
            cb(0, ttnn.bfloat16, 2048, self.Kt),  # x, whole K resident
            cb(1, self.dtype, tb, 4 * blk),  # weight blocks: 4 fixed slots (reader keeps 3 in flight)
            cb(16, self.out_dtype, otb, 2 * DST_TILES),  # output tiles of a pass
        ]
        prog = ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[])
        return ttnn.generic_op([x, self.weight, out], prog)
