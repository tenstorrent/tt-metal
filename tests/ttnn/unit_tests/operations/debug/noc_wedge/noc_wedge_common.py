# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the Blackhole NoC wedge reproducers: scratch buffers, a traffic plan that keeps every RISC on its
own NoC (dedicated-NoC mode only: a core's two RISCs never share a NoC, so their command buffers and counters never
mix), and the launch loop.

Scratch layout (one L1 buffer at the same address on every worker core, never read for its value):
    SRC   [0, 64 KB)       source of every write
    MC    [64, 96 KB)      multicast landing
    UC    [96, 192 KB)     unicast landing
    FAB   [192, 256 KB)    fabric landing
    RD    [256, 384 KB)    own DRAM read landing
    WORDS [384, 388 KB)    atomic / semaphore-set words
"""

import dataclasses
import hashlib
import os
import time

from loguru import logger

import ttnn

KERNEL_DIR = "tests/ttnn/unit_tests/operations/debug/noc_wedge/kernels"
TRAFFIC_KERNEL = f"{KERNEL_DIR}/noc_traffic.cpp"
FABRIC_KERNEL = f"{KERNEL_DIR}/fabric_traffic.cpp"

# roles of noc_traffic.cpp (compile-time arg 0)
ROLE_DRAM_READ, ROLE_DRAM_WRITE, ROLE_UNICAST, ROLE_MCAST = 0, 1, 2, 3
# DRAM write modes
W_ACKED, W_POSTED, W_POSTED_FENCE = 0, 1, 2
WRITE_MODES = {"acked": W_ACKED, "posted": W_POSTED, "posted_fence": W_POSTED_FENCE}
# mid-chain traffic of ROLE_MCAST
MID = {"none": 0, "dram_read": 1, "atomic": 2, "read_after": 3}

SRC, MC, UC, FAB, RD, WORDS = 0, 64 << 10, 96 << 10, 192 << 10, 256 << 10, 384 << 10
L1_SCRATCH_BYTES = 388 << 10
DRAM_PER_BANK_BYTES = 4 << 20  # DRAM scratch per bank: reads [0, 2 MB), writes [2, 3 MB), fabric [3, 3.5 MB), fences
DRAM_RD, DRAM_RD_REGION = 0, 2 << 20
DRAM_WR, DRAM_WR_REGION = 2 << 20, 1 << 20
DRAM_FAB, DRAM_FAB_REGION = 3 << 20, 512 << 10
DRAM_FENCE = (3 << 20) + (512 << 10)  # + 64 B per writer core

NOCS = {0: ttnn.NOC.NOC_0, 1: ttnn.NOC.NOC_1}


def launches_option(request, default):
    v = request.config.getoption("--noc-wedge-launches", default=None)
    return int(v) if v else default


def exit_on_hang_option(request):
    return bool(request.config.getoption("--noc-wedge-exit-on-hang", default=False))


def sync_option(request, default=50):
    v = request.config.getoption("--noc-wedge-sync", default=None)
    return int(v) if v else default


@dataclasses.dataclass
class Scratch:
    l1: object
    dram: object
    l1_base: int
    dram_base: int
    nbanks: int

    @property
    def tensors(self):
        return [self.l1, self.dram]


def alloc_scratch(mesh):
    """One L1 buffer at the same address on every worker core (height-sharded over the whole grid) and a DRAM buffer
    with DRAM_PER_BANK_BYTES contiguous bytes in every bank (interleaved, 1 KB pages)."""
    grid = mesh.compute_with_storage_grid_size()
    ncores = grid.x * grid.y
    rows = L1_SCRATCH_BYTES // 128
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    l1_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(crs, [rows, 32], ttnn.ShardOrientation.ROW_MAJOR),
    )
    l1 = ttnn.allocate_tensor_on_device(
        ttnn.Shape([ncores * rows, 32]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, l1_cfg
    )
    nbanks = mesh.dram_grid_size().x
    pages = DRAM_PER_BANK_BYTES // 1024
    dram = ttnn.allocate_tensor_on_device(
        ttnn.Shape([nbanks * pages, 256]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    return Scratch(l1, dram, int(l1.buffer_address()), int(dram.buffer_address()), nbanks)


def packed_xy(mesh, core):
    """Virtual (translated) NoC coordinates of a logical worker core, packed x << 16 | y. With coordinate translation
    the same coordinates address the core on NOC0 and NOC1."""
    v = mesh.worker_core_from_logical_core(core)
    return (v.x << 16) | v.y


def mcast_corners(mesh, lo, hi, noc):
    """(start, end) packed coordinates of the rectangle lo..hi (logical, inclusive) for a multicast on `noc`: NOC1
    names the rectangle from its far corner."""
    a, b = packed_xy(mesh, ttnn.CoreCoord(*lo)), packed_xy(mesh, ttnn.CoreCoord(*hi))
    return (b, a) if noc == 1 else (a, b)


class TrafficPlan:
    """Collects (core, NoC) -> traffic role entries. A core runs at most one role per NoC, NOC0 on RISCV_0 (BRISC) and
    NOC1 on RISCV_1 (NCRISC), so no core ever has two RISCs on one NoC (no dynamic-NoC mode anywhere). A second role
    for a (core, NoC) that is already taken is dropped and counted."""

    def __init__(self, grid):
        self.grid = grid
        self.entries = {}  # (x, y, noc) -> (kernel, ct, args, what)
        self.dropped = {}

    def in_grid(self, c):
        return 0 <= c[0] < self.grid.x and 0 <= c[1] < self.grid.y

    def add(self, what, kernel, ct, per_core, noc):
        for c, args in per_core:
            c = tuple(c)
            key = (c[0], c[1], noc)
            if not self.in_grid(c) or key in self.entries:
                self.dropped[what] = self.dropped.get(what, 0) + 1
                continue
            self.entries[key] = (kernel, tuple(ct), list(args), what)

    def taken(self, c, noc):
        return (c[0], c[1], noc) in self.entries

    def summary(self):
        roles = {}
        for (x, y, noc), (_, _, _, what) in self.entries.items():
            roles.setdefault(f"{what}@NOC{noc}", 0)
            roles[f"{what}@NOC{noc}"] += 1
        return roles, dict(self.dropped)

    def program(self, extra_kernels=(), program=None):
        """A ProgramDescriptor: one KernelDescriptor per (kernel, CT args, NoC); runtime args filled before each
        descriptor is built (KernelDescriptor copies them). custom_program_hash covers every core's arg count."""
        groups = {}
        for (x, y, noc), (kernel, ct, args, _) in sorted(self.entries.items()):
            groups.setdefault((kernel, ct, noc), []).append((ttnn.CoreCoord(x, y), args))
        kernels, h = [], hashlib.sha256()
        for (kernel, ct, noc), per_core in sorted(groups.items()):
            rt = ttnn.RuntimeArgs()
            for c, args in per_core:
                rt[c.x][c.y] = args
                h.update(f"{kernel}|{ct}|{noc}|{c.x},{c.y}|{len(args)};".encode())
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=kernel,
                    core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _ in per_core]),
                    compile_time_args=list(ct),
                    runtime_args=rt,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0
                        if noc == 0
                        else ttnn.DataMovementProcessor.RISCV_1,
                        noc=NOCS[noc],
                        noc_mode=ttnn.NOC_MODE.DM_DEDICATED_NOC,
                    ),
                )
            )
        for k, sig in extra_kernels:
            kernels.append(k)
            h.update(sig.encode())
        program = program if program is not None else ttnn.ProgramDescriptor()
        program.kernels = kernels
        program.custom_program_hash = int.from_bytes(h.digest()[:8], "little") & ((1 << 63) - 1)
        return program


def replicated(mesh, program):
    """The same program on every chip of the mesh."""
    rows, cols = tuple(mesh.shape)
    desc = ttnn.MeshProgramDescriptor()
    desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(rows - 1, cols - 1))] = program
    return desc


def log_config(tag, cfg, plan=None):
    fields = ", ".join(f"{f.name}={getattr(cfg, f.name)}" for f in dataclasses.fields(cfg))
    logger.info(f"NOC_WEDGE {tag}: {fields}")
    if plan is not None:
        roles, dropped = plan.summary()
        logger.info(f"NOC_WEDGE {tag}: roles (cores) {roles}; dropped (core/NoC taken or off-grid) {dropped}")


def run_launches(mesh, tensors, desc, launches, sync, tag, exit_on_hang=False):
    """Launch the program `launches` times, synchronizing every `sync` launches. A hang surfaces as the dispatch
    timeout (TT_METAL_OPERATION_TIMEOUT_SECONDS, set by scripts/run_safe_pytest.sh) raising out of a synchronize;
    `exit_on_hang` then leaves the process at once, without closing (touching) the hung mesh."""
    chips = mesh.get_num_devices()
    ttnn.generic_op(tensors, desc)  # (compile + first launch)
    ttnn.synchronize_device(mesh)
    t0 = t_last = time.perf_counter()
    done, synced = 1, 1
    try:
        for i in range(launches - 1):
            ttnn.generic_op(tensors, desc)
            done = i + 2
            if done % sync == 0:
                ttnn.synchronize_device(mesh)
                now = time.perf_counter()
                logger.info(
                    f"NOC_WEDGE {tag}: launch {done}/{launches} x {chips} chips, "
                    f"{(now - t_last) / (done - synced) * 1e3:.3f} ms/launch, elapsed {now - t0:.0f} s"
                )
                t_last, synced = now, done
        ttnn.synchronize_device(mesh)
    except Exception:
        logger.error(f"NOC_WEDGE {tag}: HANG/ERROR in launches {synced + 1}..{done} of {launches} (all before: clean)")
        if exit_on_hang:
            logger.complete()
            os._exit(2)
        raise
    elapsed = time.perf_counter() - t0
    logger.info(f"NOC_WEDGE {tag}: done, {launches} launches x {chips} chips, no hang ({elapsed:.0f} s)")
    return elapsed / max(1, launches - 1)


# ---- per-role (compile-time args, runtime args) builders for noc_traffic.cpp ----


def dram_read(s, iters, nbytes, first_bank, in_flight=8, pinned=False):
    """Stream `iters` DRAM reads of `nbytes` over the banks from `first_bank` (or always that bank if pinned),
    `in_flight` before each read barrier."""
    assert in_flight * nbytes <= WORDS - RD
    ct = [ROLE_DRAM_READ, in_flight, 0, 0]
    rt = [
        iters,
        s.dram_base + DRAM_RD,
        DRAM_RD_REGION,
        nbytes,
        first_bank % s.nbanks,
        s.nbanks,
        s.l1_base + RD,
        int(pinned),
    ]
    return ct, rt


def dram_write(
    s, units, writes, nbytes, first_bank, mode="acked", barrier_each_unit=False, own_reads=0, read_bytes=8192, index=0
):
    """`units` units of `writes` DRAM writes of `nbytes` over the banks; `own_reads` DRAM reads of `read_bytes` in
    flight while each unit's writes go; mode acked / posted / posted_fence (one non-posted 4 B write per bank, then a
    write barrier, after every unit). `index` separates the cores' fence words."""
    assert writes and nbytes <= MC - SRC and own_reads * read_bytes <= WORDS - RD
    ct = [ROLE_DRAM_WRITE, WRITE_MODES[mode], int(barrier_each_unit), 0]
    rt = [
        units,
        s.dram_base + DRAM_WR,
        DRAM_WR_REGION,
        writes,
        nbytes,
        first_bank % s.nbanks,
        s.nbanks,
        s.l1_base + SRC,
        own_reads,
        read_bytes,
        s.dram_base + DRAM_RD,
        DRAM_RD_REGION,
        s.l1_base + RD,
        s.l1_base + WORDS,
        s.dram_base + DRAM_FENCE + 64 * index,
    ]
    return ct, rt


def unicast(s, iters, targets, nbytes, land=UC, land_slots=8, atomic_every=0, atomic_xy=0, pace=0, vc=1):
    """`iters` unicast writes of `nbytes` round-robin over `targets` (packed xy), an atomic increment every
    `atomic_every` writes (to `atomic_xy`, or the write's target if 0), riscv_wait(`pace`) cycles after each."""
    assert targets and (nbytes == 0 or land + land_slots * nbytes <= (FAB if land < FAB else RD))
    ct = [ROLE_UNICAST, vc, 0, 0]
    rt = [
        iters,
        s.l1_base + SRC,
        nbytes,
        s.l1_base + land,
        land_slots,
        atomic_every,
        s.l1_base + WORDS,
        atomic_xy,
        pace,
    ]
    rt += [len(targets)] + list(targets)
    return ct, rt


def mcast(
    s,
    iters,
    corners,
    ndests,
    piece,
    chain,
    linked=True,
    sem_mcast=True,
    mid="none",
    mid_at=0,
    rd_bytes=2048,
    atomic_xy=0,
    pace=0,
):
    """`iters` chains of `chain` multicast pieces of `piece` bytes into the rectangle `corners` (from mcast_corners),
    linked within a chain if `linked`, each chain followed by a 4 B semaphore-set multicast if `sem_mcast`; `mid`
    issues a DRAM read / atomic from another command buffer after piece `mid_at` (inside the chain)."""
    assert 3 * piece <= UC - MC and chain >= 1
    ct = [ROLE_MCAST, int(linked), int(sem_mcast), MID[mid]]
    rt = [
        iters,
        s.l1_base + SRC,
        corners[0],
        corners[1],
        ndests,
        piece,
        chain,
        s.l1_base + MC,
        s.l1_base + WORDS,
        mid_at,
        s.dram_base + DRAM_RD,
        s.nbanks,
        rd_bytes,
        s.l1_base + RD,
        atomic_xy,
        pace,
    ]
    return ct, rt
