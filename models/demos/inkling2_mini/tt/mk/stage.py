# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Cross-op weight staging for bank-streamed linears: fill the DRAM-idle time of DRAM-light programs.

A ``BankLinear`` (tt/fused/bank_linear.py) core streams one contiguous byte range of its own DRAM bank. Staging
copies the first ``staged`` bytes of every core's range into a persistent L1 shard on that same core, using an extra
data-movement kernel (kernels/stage_fill.cpp) that rides inside an earlier, DRAM-light program on cores that
program does not use. The linear's reader (kernels/stage_linear_reader.cpp) then takes those bytes from local L1
and only the rest from DRAM. Compute and the bytes in every CB slot are unchanged, so the output is bit-equal.

    st = StagedBankLinear(bank_linear, staged_bytes)   # owns one L1 staging tensor (shard per core)
    prog = st.attach(prog)                             # an earlier program: + fill kernels (asserts disjoint cores)
    ttnn.generic_op(io + st.fill_io, prog)
    y = st(x)                                          # the linear, prefix from L1

The program the fill rides in must run between the previous use of the staging shard and the linear, and must
not use the linear's cores (asserted). The staging tensor stays allocated: its L1 is lost to every other program
on those cores.
"""

import copy
import os

import ttnn
from models.demos.inkling2_mini.tt.fused import common as F
from models.demos.inkling2_mini.tt.fused.bank_linear import DST_TILES, TILE, half_tiles
from models.demos.inkling2_mini.tt.fused.moe_bank import row_placement

MKDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")  # not cwd-relative
# fill read size / reads in flight per RISC (StagedBankLinear(fill=...) for A/B runs). Measured in the model:
# 2 KB / 1 best (-1.51 ms @4K); 16 KB / 4 gained nothing (the hosts wait on small DRAM reads of their own)
FILL = (2048, 1)


def cores_of(crs):
    """CoreRangeSet -> set of (x, y)."""
    out = set()
    for r in crs.ranges():
        for x in range(r.start.x, r.end.x + 1):
            for y in range(r.start.y, r.end.y + 1):
                out.add((x, y))
    return out


def placement(device, per_bank, exclude=()):
    """per_bank[b] logical cores for bank b: its optimal NOC0 reader row first (nearest that worker), then the rows
    nearest it, skipping ``exclude`` (cores the hosting programs use). Bit-equality does not depend on placement."""
    opt = [(int(c.x), int(c.y)) for c in device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)]
    grid = device.compute_with_storage_grid_size()
    used = {tuple(c) for c in exclude}
    out = {b: [] for b in range(len(opt))}
    for d in range(grid.y):  # every bank's own row before any bank takes a neighbour row
        for b, (wx, wy) in enumerate(opt):
            for y in sorted({wy - d, wy + d}):
                if not 0 <= y < grid.y:
                    continue
                for x in sorted(range(grid.x), key=lambda x: (abs(x - wx), x)):
                    if len(out[b]) < per_bank[b] and (x, y) not in used:
                        used.add((x, y))
                        out[b].append((x, y))
    assert all(len(out[b]) == per_bank[b] for b in out), out
    return out


def coalesce(cores):
    """(x, y) list -> CoreRangeSet of row runs (dispatch writes kernels / args per range: few ranges launch faster)."""
    runs = []
    for x, y in sorted(set(cores), key=lambda c: (c[1], c[0])):
        if runs and runs[-1][1] == y and runs[-1][2] == x - 1:
            runs[-1][2] = x
        else:
            runs.append([x, y, x])
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(a, y), ttnn.CoreCoord(b, y)) for a, y, b in runs])


def resplit(bl, split, exclude=()):
    """A copy of BankLinear ``bl`` whose every work item is cut into ``split`` cores (contiguous runs of its passes),
    over the SAME weight tensor: an item's tiles are contiguous in its bank in pass order, so a sub-item's range is
    a sub-range (offset = the passes before it). Each pass keeps its own Dest chain: same output."""
    assert not bl.spreads(), "a spread layout (BankLinear spread=True) has no contiguous item ranges"
    out = copy.copy(bl)
    tiles = lambda part: sum(ncq for ncq, _ in part) * bl.nblk * bl.KB  # noqa: E731
    out.items, out.offsets = [], []
    for (b, rot, part), off in zip(bl.items, bl.offsets):
        for s_ in range(split):
            sub = part[len(part) * s_ // split : len(part) * (s_ + 1) // split]
            if sub:
                out.items.append((b, rot, sub))
                out.offsets.append(off + tiles(part[: len(part) * s_ // split]))
    counts = [sum(1 for it in out.items if it[0] == b) for b in range(bl.nb)]
    place = placement(bl.device, counts, exclude) if exclude else row_placement(bl.device, counts)
    seen = [0] * bl.nb
    out.cores = []
    for i, (b, rot, part) in enumerate(out.items):
        out.cores.append((place[b][seen[b]], i))
        seen[b] += 1
    out.grid = coalesce([c for c, _ in out.cores])
    return out


def range_bytes(bl):
    """Largest per-core weight range of BankLinear ``bl`` in bytes."""
    tb = ttnn.Tile([TILE, TILE]).get_tile_size(bl.dtype)
    return max(sum(ncq for ncq, _ in part) * bl.nblk * bl.KB * tb for _, _, part in bl.items)


def program_cores(prog):
    out = set()
    for k in prog.kernels:
        out |= cores_of(k.core_ranges)
    return out


class StagedBankLinear:
    def __init__(self, bl, staged_bytes=None, stage=None, fill=FILL):
        """bl: a BankLinear. staged_bytes: bytes staged per core (default: each core's whole range). stage: an existing
        staging tensor to share (linears with the same cores that never hold staged data at the same time).
        fill: (read size in bytes, reads in flight) of the fill kernel."""
        self.bl, dev = bl, bl.device
        self.fill = fill
        self.signal = None  # tensor prefetcher op signal address raised by the first core's reader (expert_overlap)
        self.tb = ttnn.Tile([TILE, TILE]).get_tile_size(bl.dtype)
        # per core: (core, bank, first slot, range bytes)
        self.ranges = []
        for c, i in bl.cores:
            b, rot, part = bl.items[i]
            tiles = sum(ncq for ncq, _ in part) * bl.nblk * bl.KB
            self.ranges.append((c, b, bl.offsets[i], tiles * self.tb))
        top = max(r[3] for r in self.ranges)
        self.staged = top if staged_bytes is None else min(staged_bytes, top)
        self.staged = self.staged // 32 * 32
        self.cores = {r[0] for r in self.ranges}
        if stage is None:  # one shard of `staged` bytes per core (uint32 ROW_MAJOR, height sharded over the cores)
            words = max(self.staged // 4, 8)
            spec = ttnn.ShardSpec(bl.grid, (1, words), ttnn.ShardOrientation.ROW_MAJOR)
            mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)
            stage = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, len(self.ranges), words]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, dev, mem
            )
        else:
            assert self.cores <= cores_of(stage.memory_config().shard_spec.grid), "shared staging: other cores"
            assert stage.shape[-1] * 4 >= self.staged, (stage.shape, self.staged)
        self.stage = stage
        self.fill_io = [bl.weight, self.stage]

    def fill_kernels(self, part=(0.0, 1.0), delay=0):
        """Data-movement kernels that stage bytes [part[0], part[1]) x staged of every core's range (both RISCs, each
        half). Several programs can each carry a part."""
        acc = list(ttnn.TensorAccessorArgs(self.bl.weight).get_compile_time_args())
        lo, hi = int(self.staged * part[0]) // 32 * 32, int(self.staged * part[1]) // 32 * 32
        mid = (lo + hi) // 2 // 32 * 32
        kernels = []
        halves = (((lo, mid), ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
                  ((mid, hi), ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1))  # fmt: skip
        if len(FILL) > 2 and FILL[2] == 1:  # one RISC per core
            halves = (((lo, hi), ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),)
        for (a, b), proc, noc in halves:
            rt = []
            for c, bank, slot, n in self.ranges:
                n_core = max(0, min(b, n, self.staged) - a)
                rt.append((ttnn.CoreCoord(*c), [self.bl.weight.buffer_address(), bank, slot,
                                                self.stage.buffer_address(), a, n_core, delay]))  # fmt: skip
            kernels.append(ttnn.KernelDescriptor(
                kernel_source=f"{MKDIR}/stage_fill.cpp", core_ranges=self.bl.grid, compile_time_args=acc,
                named_compile_time_args=[("n_banks", self.bl.nb), ("piece", self.fill[0]), ("in_flight", self.fill[1])], runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc),
            ))  # fmt: skip
        return kernels

    def attach(self, prog, part=(0.0, 1.0)):
        """prog + the fill kernels; the program must not use any of the linear's cores."""
        clash = program_cores(prog) & self.cores
        assert not clash, f"staging cores also used by the host program: {sorted(clash)[:8]}"
        return ttnn.ProgramDescriptor(
            kernels=list(prog.kernels) + self.fill_kernels(part), semaphores=list(prog.semaphores), cbs=list(prog.cbs)
        )

    def fill_alone(self):
        """The fill as its own program (tests)."""
        prog = ttnn.ProgramDescriptor(kernels=self.fill_kernels(), semaphores=[], cbs=[])
        return ttnn.generic_op(self.fill_io, prog)

    def part_bytes(self, frac):
        """Staged prefix (bytes per core) once the parts up to fraction `frac` are filled (as fill_kernels cuts)."""
        return int(self.staged * frac) // 32 * 32

    def _signal_args(self):
        if self.signal is None:
            return []
        p = self.bl.device.worker_core_from_logical_core(ttnn.CoreCoord(*self.bl.cores[0][0]))
        return [("pf_signal", self.signal), ("sig_x", p.x), ("sig_y", p.y)]

    def __call__(self, x, out=None, staged=None):
        """BankLinear.__call__ with the staged reader (same args otherwise). staged: prefix bytes that are filled
        (default all; 0 = everything from DRAM)."""
        bl = self.bl
        staged = self.staged if staged is None else staged
        tb = self.tb
        tiles_out = out is None
        if tiles_out and getattr(bl, "out_buf", None) is not None:
            out = bl.out_buf
        elif tiles_out:
            shape = ttnn.Shape([1, 1, x.shape[-2], bl.out_cols])
            out = ttnn.allocate_tensor_on_device(
                shape, bl.out_dtype, ttnn.TILE_LAYOUT, bl.device, getattr(bl, "out_mem", ttnn.DRAM_MEMORY_CONFIG)
            )
        else:
            assert bl.out_dtype == ttnn.float32 and x.shape[-2] == 1
        otb = ttnn.Tile([TILE, TILE]).get_tile_size(bl.out_dtype)
        x_rows = getattr(bl, "x_rows", 0)
        half, th, xtb, htb = half_tiles(x_rows, tiles_out, bl.out_dtype)
        named = dict(kt=bl.Kt, kb=bl.KB, nblk=bl.nblk, n_banks=bl.nb, out_cols=bl.out_cols,
                     out_page=otb if tiles_out else out.shape[-1] * 4, max_ncq=bl.max_ncq, tiles_out=int(tiles_out),
                     out_tile=htb, x_rows=x_rows, raw_rows=1, x_half=int(half))  # fmt: skip
        r_args, c_args, w_args = [], [], []
        for c, i in bl.cores:
            b, rot, part = bl.items[i]
            core = ttnn.CoreCoord(*c)
            r_args.append((core, [bl.weight.buffer_address(), b, bl.offsets[i], self.stage.buffer_address(),
                                  staged, len(part)] + [n for n, _ in part]))  # fmt: skip
            c_args.append((core, [rot, len(part)] + [n for n, _ in part]))
            w_args.append(
                (core, [x.buffer_address(), rot, out.buffer_address(), len(part)] + [v for p_ in part for v in p_])
            )
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        blk = bl.KB * bl.max_ncq
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=f"{MKDIR}/stage_linear_reader.cpp", core_ranges=bl.grid, compile_time_args=acc(bl.weight),
                named_compile_time_args=list(named.items()) + self._signal_args(), runtime_args=r_args,
                defines=[("PF_SIGNAL", "1")] * (self.signal is not None),
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"{F.KDIR}/bank_linear_compute.cpp", core_ranges=bl.grid,
                named_compile_time_args=list(named.items()), runtime_args=c_args, config=bl.compute,
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"{F.KDIR}/bank_linear_writer.cpp", core_ranges=bl.grid, compile_time_args=acc(x) + acc(out),
                named_compile_time_args=list(named.items()), runtime_args=w_args,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
            ),
        ]  # fmt: skip
        t32 = ttnn.TileDescriptor(ttnn.Tile([TILE, TILE]))

        def cb(i, fmt, page, n, tile=t32):
            return ttnn.CBDescriptor(total_size=page * n, core_ranges=bl.grid,
                                     format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt, page_size=page, tile=tile)])  # fmt: skip

        cbs = [
            cb(0, ttnn.bfloat16, xtb, bl.Kt, th),
            cb(1, bl.dtype, tb, 4 * blk),
            cb(16, bl.out_dtype, htb, 2 * DST_TILES, th),
        ]
        ride_k, ride_io = F.take_ride(c for c, _ in bl.cores) if tiles_out else ([], [])
        prog = ttnn.ProgramDescriptor(kernels=kernels + ride_k, cbs=cbs, semaphores=[])
        return ttnn.generic_op([x, bl.weight, self.stage, *ride_io, out], prog)
