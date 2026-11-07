# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 SDPA on Wormhole with the release's core loop (tt-lab-private h3-wh, src/wh_sdpa_trisc.h), run through
ttnn.generic_op on every device of the mesh.

Kernels: ttnn/cpp/ttnn/operations/experimental/minimax_h3_wh_sdpa/kernels/ (contract in wh_sdpa_op_common.h).
The cores are split into two feed groups of four whole grid columns (logical 0-3 and 4-7; neither straddles the
physical DRAM column). Each group runs rounds of (head, first query block); its cores take consecutive 64-query
blocks and share one K/V stream of 64-key blocks of K and V^T, block g read from DRAM by reader g % readers and
multicast to the group by its I/O kernel. Four readers (default): (left, row 0) and (left, row 1) cast on NoC 0 and read
on NoC 1, (right, last row) and (right, row before) cast on NoC 1 and read on NoC 0; the second caster of each NoC uses
the buddy multicast VC, so each (NoC, VC) carries one source a group (the release's rule: two unreserved casters on
one (NoC, VC) deadlocked the NoC). The corner casters' casts do not wrap; the second ones wrap in y inside the group's
own columns. Each reader reads ~10 B a cycle, so two readers (readers=2: the corners only) left the core loop waiting.

Inputs (DRAM interleaved, bf16, tile): Q [1, H, Nq, 128], K [1, H, Nk, 128] and V^T [1, H, 128, Nk] (V transposed
whole); keys at and past the value of `logical_n` (read on device) are ignored. Output O [1, H, Nq, 128].
Fixed: head dim 128, scale 1/sqrt(128), no mask. Measured on one chip at H 14, Nq 13824, Nk 110592 (108618 valid):
123.4 ms on 8x9 with 4 readers, 140.2 with 2 (wh_sdpa_model_bench.py, host call included; ttnn ring SDPA ~175 ms).
"""

from __future__ import annotations

import os
import random

import ttnn

KDIR = "ttnn/cpp/ttnn/operations/experimental/minimax_h3_wh_sdpa/kernels/"
IDLE = 0xFFFF
MAX_READERS = 4
A_READER0_X = 19
SCRATCH_FIXED = 4096 + 7 * 8 * 2048 + 2 * 16384  # sync lines + save areas, unpacker 0 region, O staging
BLOCK = 32768  # one 64-key K/V block in L1
GATHER_BYTES = 2 * 4 * 2 * 2048 + (2 * 4 + 1) * 256  # WHS_GATHER_BYTES: the ring gather's staging and headers
RING_LINES = 32  # landing counter lines (16 B) a counter buffer: heads, then the barrier at line 31

FLAG_NO_PATH_RESERVE = 16
FLAG_CAST_BY_IO = 128
FLAG_VT_FULL = 256
FLAG_CAST_BUDDY_VC = 512

# The launch token (GO) must differ from every launch before it that used the same L1 scratch: the kernels wait for the
# GO line to equal it, and a stale equal value there lets them start before the sync lines are reset. One counter for
# every instance (each attention layer has its own WhSdpa; per-instance counters gave consecutive layers equal tokens
# and corrupted the video).
# Seeded per process so a new process does not replay the tokens an earlier one left in L1. (TODO before tracing:
# a program semaphore, which dispatch resets at every launch, instead of a runtime-arg token.)
_GO = [random.SystemRandom().getrandbits(28)]


class _RingCounters:
    """The ring gather's landing counters (wh_sdpa_gather.cpp): one L1 buffer on the counter core of every chip,
    never reset; `launches` counts the ring launches that used it (the same on every chip)."""

    def __init__(self, mesh_device: ttnn.MeshDevice, core: tuple[int, int]) -> None:
        import torch

        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*core), ttnn.CoreCoord(*core))])
        spec = ttnn.ShardSpec(cores, [1, RING_LINES * 4], ttnn.ShardOrientation.ROW_MAJOR)
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)
        self.tensor = ttnn.from_torch(
            torch.zeros(1, 1, 1, RING_LINES * 4, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=mc,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self.address = self.tensor.buffer_address()
        self.launches = 0


_RING_COUNTERS: dict[int, _RingCounters] = {}
_RING_DEBUG = int(os.environ.get("MINIMAX_H3_RING_DEBUG", "0"))


class WhSdpa:
    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        *,
        grid: tuple[int, int] | None = None,
        bufs: int = 16,
        readers: int = 4,
    ) -> None:
        self.mesh_device = mesh_device
        g = mesh_device.compute_with_storage_grid_size()
        self.gx, self.gy = grid or (g.x, g.y)
        assert self.gx == 8, "two groups of 4 columns"
        assert readers in (2, 4)
        self.bufs = bufs
        self.readers = readers
        self.groups = [(0, 3), (4, 7)]
        self.phys = {
            (x, y): mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            for x in range(self.gx)
            for y in range(self.gy)
        }
        self.grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.gx - 1, self.gy - 1))])
        self.rows = (SCRATCH_FIXED + bufs * BLOCK + GATHER_BYTES + 2047) // 2048
        spec = ttnn.ShardSpec(self.grid, [self.rows, 1024], ttnn.ShardOrientation.ROW_MAJOR)
        self.scratch_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)
        self.core_group = {}
        for gi, (ga, gb) in enumerate(self.groups):
            idx = 0
            for yy in range(self.gy):
                for xx in range(ga, gb + 1):
                    self.core_group[(xx, yy)] = (gi, idx)
                    idx += 1
        self.all_cores = [(x, y) for x in range(self.gx) for y in range(self.gy)]
        # Ring gather senders (wh_sdpa_gather.cpp) on non-reader cores: forward links 0-3, then backward links 0-3.
        ya, yb = 2, self.gy - 3
        self.senders = [(1, ya), (2, ya), (5, ya), (6, ya), (1, yb), (2, yb), (5, yb), (6, yb)]
        assert all(self._reader_index(c) == IDLE for c in self.senders)

    def _readers(self, gi: int) -> list[tuple[int, int]]:
        ga, gb = self.groups[gi]
        rc = [(ga, 0), (gb, self.gy - 1)]
        if self.readers == 4:
            # Second casters on each NoC, on the buddy VC: their casts wrap in y only, inside the group's columns.
            rc += [(ga, 1), (gb, self.gy - 2)]
        return rc

    def _reader_index(self, c: tuple[int, int]) -> int:
        rc = self._readers(self.core_group[c][0])
        return rc.index(c) if c in rc else IDLE

    @staticmethod
    def supports(q_shape, k_local_shape) -> bool:
        """Head dim 128 and whole 64-row query and key blocks (keys: the local shard, gathered over the ring)."""
        return q_shape[3] == 128 and q_shape[2] % 64 == 0 and k_local_shape[2] % 64 == 0

    def __call__(self, q: ttnn.Tensor, k: ttnn.Tensor, vt: ttnn.Tensor, logical_n: ttnn.Tensor) -> ttnn.Tensor:
        """logical_n: [1, 1, 1, 1] uint32 DRAM tensor holding the valid key count (read on device each launch)."""
        return self._run(q, k, vt, logical_n)

    def ring(
        self,
        q: ttnn.Tensor,
        k_local: ttnn.Tensor,
        vt_local: ttnn.Tensor,
        logical_n: ttnn.Tensor,
        *,
        axis: int,
        links: int,
    ) -> ttnn.Tensor:
        """As __call__ with K [1, H, n, 128] and V^T [1, H, 128, n] each chip's shard of the keys along mesh axis
        `axis` (a ring): the op gathers them over the fabric while it runs (wh_sdpa_gather.cpp), each head's key
        blocks read once that head has landed. Same result as gathering first: the keys keep their order."""
        mesh = self.mesh_device
        ring = mesh.shape[axis]
        _, heads, local_keys, dh = k_local.shape
        assert dh == 128 and local_keys % 64 == 0 and tuple(vt_local.shape) == (1, heads, 128, local_keys)
        keys = local_keys * ring
        dram = ttnn.DRAM_MEMORY_CONFIG
        k_full = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, heads, keys, 128]), k_local.dtype, ttnn.TILE_LAYOUT, mesh, dram
        )
        vt_full = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, heads, 128, keys]), vt_local.dtype, ttnn.TILE_LAYOUT, mesh, dram
        )
        counters = _RING_COUNTERS.get(id(mesh))
        if counters is None:
            counters = _RingCounters(mesh, self.senders[0])
            _RING_COUNTERS[id(mesh)] = counters
        counters.launches += 1
        ctx = dict(
            axis=axis,
            ring=ring,
            links=links,
            k_local=k_local,
            vt_local=vt_local,
            counters=counters,
            launch=counters.launches & 0xFFFFFFFF,
        )
        out = self._run(q, k_full, vt_full, logical_n, ring_ctx=ctx)
        ttnn.deallocate(k_full)
        ttnn.deallocate(vt_full)
        return out

    def _run(self, q, k, vt, logical_n, ring_ctx=None) -> ttnn.Tensor:
        _, heads, q_rows, dh = q.shape
        keys = k.shape[2]
        assert dh == 128 and q_rows % 64 == 0 and keys % 64 == 0 and (vt.shape[2], vt.shape[3]) == (128, keys)
        assert logical_n.memory_config().buffer_type == ttnn.BufferType.DRAM and logical_n.dtype == ttnn.uint32
        qb = q_rows // 64
        size = 4 * self.gy
        group_rounds: list[list[tuple[int, int]]] = [[], []]
        i = 0
        for h in range(heads):
            for base in range(0, qb, size):
                group_rounds[i % 2].append((h, base))
                i += 1

        scratch = ttnn.allocate_tensor_on_device(
            ttnn.Shape([self.rows * self.gx * self.gy, 1024]),
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
            self.mesh_device,
            self.scratch_memory_config,
        )
        out = ttnn.allocate_tensor_on_device(q.shape, q.dtype, q.layout, self.mesh_device, q.memory_config())
        _GO[0] = _GO[0] % 0x7FFFFFFF + 1
        go = _GO[0] | 0x40000000  # a value unlikely to be left in L1 by other ops
        if ring_ctx is not None:
            cp = self.phys[self.senders[0]]
            ring_target = (ring_ctx["ring"] * ring_ctx["links"] * ring_ctx["launch"]) & 0xFFFFFFFF
            ring_args = [ring_ctx["counters"].address, cp.x, cp.y, ring_target]
        else:
            ring_args = [0, 0, 0, 0]

        def args(x: int, y: int) -> list[int]:
            gi, idx = self.core_group[(x, y)]
            ga, gb = self.groups[gi]
            rounds = [(h << 16) | (base + idx if base + idx < qb else IDLE) for h, base in group_rounds[gi]]
            rc = self._readers(gi)
            readers_xy = []
            for c in rc:
                p = self.phys[c]
                readers_xy += [p.x, p.y]
            readers_xy += [0] * (2 * MAX_READERS - len(readers_xy))
            p0, p1 = self.phys[(ga, 0)], self.phys[(gb, self.gy - 1)]
            ri = rc.index((x, y)) if (x, y) in rc else IDLE
            flags = FLAG_VT_FULL | (FLAG_CAST_BY_IO | FLAG_NO_PATH_RESERVE if ri != IDLE else 0)
            if ri in (2, 3):
                flags |= FLAG_CAST_BUDDY_VC
            base = [
                scratch.buffer_address(),
                go,
                q.buffer_address(),
                k.buffer_address(),
                vt.buffer_address(),
                out.buffer_address(),
                heads,
                q_rows // 32,
                keys // 32,
                keys,  # bound; the launch reads logical_n
                self.bufs,
                len(rc),
                ri,
                size,
                idx,
                p0.x,
                p0.y,
                p1.x,
                p1.y,
            ]
            lead = self.phys[rc[0]]
            return (
                base
                + readers_xy
                + [lead.x, lead.y, 0, flags, logical_n.buffer_address()]
                + ring_args
                + [len(rounds)]
                + rounds
            )

        def kd(src: str, config, cores=None) -> ttnn.KernelDescriptor:
            cores = cores or self.all_cores
            ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in cores])
            rt = ttnn.RuntimeArgs()
            for x, y in cores:
                rt[x][y] = args(x, y)
            return ttnn.KernelDescriptor(
                kernel_source=KDIR + src,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=ranges,
                compile_time_args=[],
                defines=[("WHS_FEED_DEPTH", "2u")] if "feed" in src else [],
                runtime_args=rt,
                config=config,
            )

        def dm(proc, noc):
            c = ttnn.DataMovementConfigDescriptor()
            c.processor, c.noc = proc, noc
            return c

        # Readers 0 and 2 (top rows): feed (reads) on NoC 1, I/O (casts) on NoC 0. Everyone else: feed NoC 0, I/O NoC 1.
        # With the ring gather the senders' NCRISC runs the gather instead of the (idle) feed.
        swap = [c for c in self.all_cores if self._reader_index(c) in (0, 2)]
        rest = [c for c in self.all_cores if c not in swap]
        senders = self.senders if ring_ctx is not None else []
        P, N = ttnn.DataMovementProcessor, ttnn.NOC
        kernels = [
            kd("wh_sdpa_io.cpp", dm(P.RISCV_0, N.NOC_1), rest),
            kd("wh_sdpa_feed.cpp", dm(P.RISCV_1, N.NOC_0), [c for c in rest if c not in senders]),
            kd("wh_sdpa_io.cpp", dm(P.RISCV_0, N.NOC_0), swap),
            kd("wh_sdpa_feed.cpp", dm(P.RISCV_1, N.NOC_1), swap),
            kd("wh_sdpa_op.cpp", ttnn.ComputeConfigDescriptor()),
        ]
        sem = ttnn.SemaphoreDescriptor(id=0, core_ranges=self.grid, initial_value=0)
        if ring_ctx is None:
            pd = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[sem], cbs=[])
            ttnn.generic_op([q, k, vt, logical_n, scratch, out], pd)
        else:
            mesh_pd = self._ring_programs(kernels, sem, ring_ctx, k, vt, scratch, dm(P.RISCV_1, N.NOC_0))
            ttnn.generic_op([q, ring_ctx["k_local"], ring_ctx["vt_local"], k, vt, logical_n, scratch, out], mesh_pd)
        ttnn.deallocate(scratch)
        return out

    def _ring_programs(self, kernels, sem, ctx, k_full, vt_full, scratch, config) -> ttnn.MeshProgramDescriptor:
        """One program a chip: the SDPA kernels (the same everywhere) and the ring gather's senders, whose fabric
        connections and slice of the keys depend on the chip."""
        mesh = self.mesh_device
        axis, ring, links = ctx["axis"], ctx["ring"], ctx["links"]
        assert len(self.senders) == 2 * links
        k_local, vt_local = ctx["k_local"], ctx["vt_local"]
        heads, local_keys = k_local.shape[1], k_local.shape[2]
        cp = self.phys[self.senders[0]]
        stage = scratch.buffer_address() + SCRATCH_FIXED + self.bufs * BLOCK
        sender_ranges = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in self.senders]
        )
        mesh_pd = ttnn.MeshProgramDescriptor()
        rows, cols = mesh.shape[0], mesh.shape[1]
        for r in range(rows):
            for c in range(cols):
                coord = ttnn.MeshCoordinate(r, c)
                pos = (r, c)[axis]

                def neighbor(step):
                    nc = [r, c]
                    nc[axis] = (nc[axis] + step) % ring
                    return mesh.get_fabric_node_id(ttnn.MeshCoordinate(*nc))

                src = mesh.get_fabric_node_id(coord)
                dsts = [neighbor(1), neighbor(-1)]
                rt = ttnn.RuntimeArgs()
                for i, (x, y) in enumerate(self.senders):
                    backward, link = divmod(i, links)
                    rt[x][y] = [
                        k_local.buffer_address(),
                        vt_local.buffer_address(),
                        k_full.buffer_address(),
                        vt_full.buffer_address(),
                        heads,
                        local_keys // 32,
                        pos,
                        ring,
                        link,
                        links,
                        backward,
                        cp.x,
                        cp.y,
                        ctx["counters"].address,
                        ctx["launch"],
                        stage,
                        _RING_DEBUG,
                    ]
                gather = ttnn.KernelDescriptor(
                    kernel_source=KDIR + "wh_sdpa_gather.cpp",
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=sender_ranges,
                    compile_time_args=[],
                    defines=[],
                    runtime_args=rt,
                    config=config,
                )
                pd = ttnn.ProgramDescriptor(kernels=list(kernels) + [gather], semaphores=[sem], cbs=[])
                gi = len(kernels)
                for i, (x, y) in enumerate(self.senders):
                    backward, link = divmod(i, links)
                    fabric_args = ttnn.setup_fabric_connection(
                        mesh, src, dsts[backward], link, pd, ttnn.CoreCoord(x, y)
                    )
                    pd.kernels[gi].runtime_args[x][y].extend(fabric_args)
                mesh_pd[ttnn.MeshCoordinateRange(coord, coord)] = pd
        return mesh_pd
