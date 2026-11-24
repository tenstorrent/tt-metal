# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a prefill-megakernel collectives (doc/ttnn/07a-fused-streaming-layer/prefill_mk_collectives.md): FabricStream,
the in-program fabric collectives at PREFILL payloads (the decode FabricPhase, tt/fabric_ar.py, is a <= 32-row latency
design), along one mesh axis (a line of n chips; the pair at TP2):

  FabricStreamAG   sequence-parallel all-gather: x [1, 1, R_loc, H] bf16 TILE (this chip's rows, DRAM) -> y [1, 1, n R_loc,
                   H] (DRAM, ONE buffer address on every chip; chip p's rows at [p R_loc, (p + 1) R_loc)).
  FabricStreamRS2  the pair's reduce-scatter exchange: the partial P [1, 1, 2 R_loc, H] -> the PEER's block of P lands in
                   r [1, 1, R_loc, H] on the peer; the reduced rows are P[own block] + r (the consumer adds: mhc_post's
                   ADD mode reads y + y2). A line of > 2 chips needs adds in transit (next).

Both are one block send per chip (kernels/fabric_stream/): lane h of every chip owns ethernet link h in both line
directions (one worker connection per router). Each lane streams its share of the block's 32-row chunks: its NCRISC
reads DRAM pages into an L1 staging CB (fs_read.cpp, NB-page batches), its BRISC (fs_send.cpp) writes each page PAIR into
the local destination (all-gather only) and line-multicasts it as ONE fabric scatter packet (2 x 2 KiB) to the same
destination pages of every chip in range; a chunk's last pair is a fused scatter write + atomic increment of the
receiver's counter ARR + p LANES + h (lane-0 core): one counter per source lane, whose chunks land IN ORDER, so a
consumer of chunk k of source q waits for its lane's counter. Lane 0's BRISC is the standalone completion. Counters are
monotonic across launches (zeroed once).
Measured (d0 pair, AG, 8 MiB local): v0 (one page per packet, reads + sends on BRISC) 8.6 GB/s per lane; v1 (these
kernels) 15.8 GB/s per lane, 31.3 GB/s with 2 lanes (ttnn.all_gather 38.2).

MERGED (14b stack45 x the 1M lineage, 2026-10-23): the AG / RS2 base below is the DSA shard tree's port (wait= / cons= /
axis= / link_base / after= / y= / sync= / FabricStreamScatter / probes / the header budget NHDR), the PMK collectives
(C1 / C1X / RSR / RSL / RS8, from CB_XS on) are the stack45 line's. The 1M docstring:

Stage 7a FabricStream (doc/ttnn/07a-fused-streaming-layer/prefill_mk_collectives.md), the in-program fabric
collectives along ONE mesh axis per instance (the pair at TP2, or one axis of the 2x4 mesh), ported into the DSA shard
tree (branch aperezvicente/glm53-pmk-int 2f3f137661, tt/fabric_stream.py: the AG / RS2 classes only) with two hooks for
the decode split-K phase (tt/tp_dsa_shard.py):
  wait=(semaphore id, count): the lanes' NCRISC (fs_read.cpp FS_WAIT) reads x only once the producers' semaphore on the
      lane core reached `count` (the block is complete);
  cons=(semaphore id, [(x, y) ...]): lane 0's BRISC (fs_send.cpp FS_CONS, the waiter) increments the consumers'
      semaphore once every peer's chunks and every local lane landed.
axis= (TP8, shard_tp8_design.md section 4): the stream runs along mesh axis `axis` -- every row (axis 1) / column
(axis 0) of the mesh is one independent line, chip (r, c) sits at position c (axis 1) / r (axis 0) of line r / c.

  FabricStreamAG       all-gather: x [1, 1, R_loc, H] bf16 TILE (this chip's rows, DRAM) -> y [1, 1, N R_loc, H] (DRAM,
                       ONE buffer address on every chip): block (line n + p) holds position p's rows of line `line`
                       (local_copy: the own block too). y=: land in an existing tensor; bundle=True: x IS that y and
                       position p's block is y's block p (the 2x4 round 2: a column's 2-block bundle multicast along
                       the row).
  FabricStreamRS2      the pair's reduce-scatter exchange (an axis of 2): P [1, 1, 2 R_loc, H] -> the PEER's block of P
                       lands in r [1, 1, R_loc, H] on the peer (the consumer merges / adds).
  FabricStreamScatter  per-destination unicast along a line of n (the 2x4 RS round 2): x [n R_loc, H] block q -> peer
                       q's y [n R_loc, H] block p (no local copy; one counter per source on the receiver).
All: one block send per chip; lane h owns ethernet link link_base + h in both line directions; the lane's NCRISC streams
its share of the block's 32-row chunks (PPC = H / 32 pages each) into a staging CB, its BRISC writes every page pair
locally (AG) and line-multicasts / unicasts it as one scatter packet; a chunk's last pair also increments the receiver's
counter ARR + p LANES + h on its lane-0 core. Counters are monotonic across launches (zeroed once)."""


from __future__ import annotations

import os

import torch

import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "fabric_stream")
TB = 2048
WORD = 16


def _header_consts() -> dict:
    """the constexpr word indices of kernels/fabric_stream/fs_common.hpp: the host's buffer sizes come FROM the header
    (a stale Python SYNC_WORDS = 64 against the header's 96 put the reducers' GEN_C word past the sync shard: watcher NOC
    "Local L1 address overflow", rc 134, d0 2026-10-22 10:09Z); the pattern takes a line-wrapped definition too"""
    import re

    with open(os.path.join(KDIR, "fs_common.hpp")) as f:
        return {m.group(1): int(m.group(2)) for m in re.finditer(r"constexpr uint32_t (\w+)\s*=\s*(\d+);", f.read())}


_FS = _header_consts()
_HW = _FS  # (the 1M line's name)
SYNC_WORDS = 96  # kernels/fabric_stream/fs_common.hpp SYNC_WORDS (the host mirror, asserted against the header)
assert _FS["SYNC_WORDS"] == SYNC_WORDS, ("fs_common.hpp SYNC_WORDS", _FS["SYNC_WORDS"], SYNC_WORDS)
assert max(v for k, v in _FS.items() if k not in ("WORD", "SYNC_WORDS")) < SYNC_WORDS, _FS
assert _FS["WORD"] == WORD and _FS["ARR"] == 0 and _FS["DONE"] == 16, _FS
# the lane staging CB: CB_STAGE = 0 for the PMK collectives (C1 / C1X / RSR / RS8 parts, tt/pf_ag.py; the family's lane CB
# 0, tt/pmk_assembly.py), CB_STAGE_FS = 16 the default of the AG / RS2 / Scatter instances (the 1M DSA shard phase's map)
CB_STAGE = 0
CB_STAGE_FS = 16
CB_HDR_FS = 5  # the standalone 2x4 C1X unit's private packet-header CB on the lanes (= the composite's CB_HDR)
NHDR = 4  # fs_send.cpp NHDR: data headers per direction (the packet-header budget, tt/shard_plan.py)
with open(os.path.join(KDIR, "fs_send.cpp")) as _f:
    assert f"constexpr uint32_t NHDR = {NHDR};" in _f.read(), "fs_send.cpp NHDR"


# headers fs_send.cpp includes by path (not inlined): their content hash is part of the kernel text (rev 60, review 2
# item 5: a probe.hpp change must not reuse a binary JIT-cached by the old text)
_DEPS = ("mk/probe.hpp",)


def kernel_text(name: str) -> str:
    """kernel source with fs_common.hpp inlined (SOURCE_CODE kernels are hashed by their text) and a trailing comment
    with the sha1 of every header it includes by path (_DEPS)"""
    import hashlib

    with open(os.path.join(KDIR, name)) as f:
        src = f.read()
    with open(os.path.join(KDIR, "fs_common.hpp")) as f:
        common = f.read().replace("#pragma once", "")
    h = hashlib.sha1()
    for dep in _DEPS:
        with open(os.path.join(os.path.dirname(KDIR), dep), "rb") as f:
            h.update(f.read())
    return src.replace('#include "fs_common.hpp"', common) + f"\n// deps {','.join(_DEPS)} sha1 {h.hexdigest()[:12]}\n"


def _crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])


# the fabric connections' worker words on RESERVED semaphore ids (coordinator 2026-10-23 17:59:48Z, after three TP8
# hangs in the first op fabric CCL after a PMK sequence): a WorkerToFabricEdmSender keeps two words in its core's
# semaphore space -- the TEARDOWN ack close_finish waits on, and the BUFFER-INDEX / producer cursor open_finish adopts
# (close_start persists it into the EDM for the next opener). setup_fabric_connection allocates them with
# find_available_semaphore_id = the LOWEST ids not yet declared on the core: 0..5 on a TP8 lane (3 connections), 0..1
# on the pair, i.e. the ids the PMK hand-offs use (0..3 mhc, 4 HN, ...). fabric_connect declares every free id below
# FAB_SEM_BASE on the core first (placeholders: no kernel reads or writes them), so the words land on FAB_SEM_BASE..15
# at every TP (6 ids = 3 connections per core and program; a 4th fails the build in setup_fabric_connection).
FAB_SEM_BASE = 10
# CONNECT_LOG: a list while a caller audits one program's EDM sender channels (pmk_assembly._edm_channel_audit): every
# fabric_connect appends (core, link, peer); None = off
CONNECT_LOG = None


def _sem_cores(s_) -> set:
    return {
        (x, y)
        for r_ in s_.core_ranges.ranges()
        for x in range(r_.start.x, r_.end.x + 1)
        for y in range(r_.start.y, r_.end.y + 1)
    }


def fabric_connect(me, peer, link: int, prog, core) -> list:
    """ttnn.setup_fabric_connection(me, peer, link, prog, core) with the connection's two worker words on semaphore ids
    >= FAB_SEM_BASE of `core` (see FAB_SEM_BASE): the runtime args to append to the sender's list"""
    core = core if isinstance(core, ttnn.CoreCoord) else ttnn.CoreCoord(*core)
    xy = (int(core.x), int(core.y))
    sems = list(prog.semaphores)
    have = {int(s_.id) for s_ in sems if xy in _sem_cores(s_)}
    pad = [i for i in range(FAB_SEM_BASE) if i not in have]
    if pad:
        prog.semaphores = sems + [ttnn.SemaphoreDescriptor(id=i, core_ranges=_crs([xy]), initial_value=0) for i in pad]
    n0 = len(list(prog.semaphores))
    fab = ttnn.setup_fabric_connection(me, peer, link, prog, core)
    if CONNECT_LOG is not None:
        CONNECT_LOG.append((xy, int(link), str(peer)))
    new = [int(s_.id) for s_ in list(prog.semaphores)[n0:]]
    assert len(new) == 2 and min(new) >= FAB_SEM_BASE, (f"fabric connection words on core {xy} at semaphore ids {new}, "
                                                          f"expected 2 ids >= FAB_SEM_BASE {FAB_SEM_BASE}")  # fmt: skip
    return fab


class _FabricStream:
    """One block send per chip along mesh axis `axis` (default: the mesh's one non-trivial axis -- the pair / a line),
    every line of that axis independent; `len(lanes)` lane cores per chip (lane h on link link_base + h); the block =
    `chunks` 32-row chunks of [., H] bf16 tiles. nb: pages per staging batch; slots: batches the staging CB holds."""

    scatter = False  # FabricStreamScatter: per-destination unicast (fs_send.cpp / fs_read.cpp FS_SCATTER)

    def __init__(self, mesh, chunks: int, hidden: int, dst_rows: int, *, lanes=None, nb: int = 16, slots: int = 4,
                 wait: tuple | None = None, cons: tuple | None = None, link_base: int = 0,
                 after: int | None = None, axis: int | None = None, y=None, sync=None):  # fmt: skip
        """link_base: lane h sends on ethernet link link_base + h. after=(semaphore id): the lane opens its
        connections only around its sends (fs_send.cpp FS_LATE_OPEN) and only once that semaphore on the lane core
        was incremented by the link's other holder (FS_AFTER_SEM); it closes before DONE / the consumers' signal.
        axis: the mesh axis the lines run along (None: the one non-trivial axis of a 1 x n / n x 1 mesh). y: land in
        this existing DRAM tensor (same address on every chip) instead of allocating one. sync: an existing stream's
        sync tensor (the same lane-0 core and per-launch counts; its counters are monotonic across launches, so
        streams launched one after another in the same order on every chip can share it: tt/tp_dsa_shard.py
        GLM53_DSA_SHARD_SHARE_SYNC) instead of allocating one."""
        self.mesh = mesh
        self.R, self.C = int(mesh.shape[0]), int(mesh.shape[1])
        if axis is None:
            assert self.R == 1 or self.C == 1, "a 2-D mesh: pass axis="
            axis = 1 if self.C > 1 else 0
        assert axis in (0, 1), axis
        self.axis = int(axis)
        self.n = self.R if self.axis == 0 else self.C  # chips per line
        self.lines = self.C if self.axis == 0 else self.R
        assert self.n > 1, ("no chips along axis", axis, mesh.shape)
        assert hidden % 32 == 0, hidden
        self.hidden, self.ppc, self.chunks = hidden, hidden // 32, chunks
        self.lanes = list(lanes or [(0, 2), (1, 2)])
        assert len(self.lanes) * self.n <= 16, "ARR counters: 16 words"
        assert chunks % len(self.lanes) == 0, (chunks, len(self.lanes))
        nb = min(nb, self.ppc)
        self.nb, self.slots = nb, slots
        assert self.ppc % nb == 0 and nb % 2 == 0, (self.ppc, nb)
        self.wait, self.cons = wait, cons
        self.link_base, self.after = int(link_base), after
        rep = ttnn.ReplicateTensorToMesh(mesh)
        words = SYNC_WORDS * WORD // 4
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                               ttnn.ShardSpec(_crs(self.lanes[:1]), [1, words], ttnn.ShardOrientation.ROW_MAJOR))  # fmt: skip
        if sync is not None:  # shared: the same lane-0 core and the same word count
            assert tuple(sync.shape)[-1] == words and sync.memory_config() == mc, (tuple(sync.shape), words)
            self.sync = sync
        else:
            self.sync = ttnn.from_torch(torch.zeros(1, words, dtype=torch.int32), dtype=ttnn.uint32,
                                        layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=mc, mesh_mapper=rep)  # fmt: skip
        self.own_y = y is None
        if y is None:
            y = ttnn.from_torch(torch.zeros(1, 1, dst_rows, hidden, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                mesh_mapper=rep)  # fmt: skip
        assert tuple(y.shape)[-2:] == (dst_rows, hidden), (tuple(y.shape), dst_rows, hidden)
        self.y = y
        self._srcs = {n: kernel_text(n) for n in ("fs_send.cpp", "fs_read.cpp")}
        l0 = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*self.lanes[0]))
        self.l0_noc = (int(l0.x), int(l0.y))

    def _pos(self, r: int, c: int) -> tuple:
        """(position along the axis, line index) of mesh coordinate (r, c)"""
        return (r, c) if self.axis == 0 else (c, r)

    def _block(self, p: int, line: int = 0):
        """(source chunk base in x, destination row-tile base in y, local write, up range, down range) of the chip at
        position p of line `line`"""
        raise NotImplementedError

    def _dsts(self, p: int) -> list:
        """FabricStreamScatter: [(position q, source chunk base of block q, direction 0 up / 1 down, hops)]"""
        return []

    def parts(self, x, r: int, c: int, *, cb_stage: int = CB_STAGE_FS, probe: tuple | None = None):
        """The phase pieces for mesh coordinate (r, c), for a program to merge: (kernels [fs_send, fs_read] on the lane
        cores, cbs [the staging CB at index cb_stage], io, counters). Then call connect(prog, r, c, i) on the merged
        ProgramDescriptor, i = the index of fs_send in its kernel list. With wait= the lane cores also need a
        SemaphoreDescriptor of that id (semaphores(): the caller adds them). probe=(probe tensor address, row): the lane
        BRISC's marks (fs_send.cpp MK_PROBE_ARG; one lane only: its flush scratch is the lane-0 sync tensor)."""
        p, line = self._pos(r, c)
        src_base, row_base, local, up, dn = self._block(p, line)
        per = self.chunks // len(self.lanes)
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        ct_send = [self.ppc, self.nb, cb_stage] + acc(x) + acc(self.y)
        ct_read = [self.ppc, self.nb, cb_stage] + acc(x)
        rt_send, rt_read = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        sync = self.sync.buffer_address()
        cons_rt = []
        if self.cons is not None:
            sem, cores = self.cons
            cons_rt = [len(cores), int(sem)] + [int(v) for xy in cores for v in xy]
        dsts = self._dsts(p)
        if self.scatter:  # count-prefixed destination lists (program lesson: never loop a fixed count over padding)
            assert dsts and all(0 < hops < self.n for _, _, _, hops in dsts), dsts
        sc_send = ([len(dsts)] + [v for _, _, d, hops in dsts for v in (d, hops)]) if self.scatter else []
        sc_read = ([len(dsts)] + [base for _, base, _, _ in dsts]) if self.scatter else []
        pr = []
        if probe is not None:
            assert len(self.lanes) == 1, "the stream probe: one lane (its scratch is the lane-0 sync tensor)"
            pr = [int(probe[0]), int(probe[1])]
        for h, (lx, ly) in enumerate(self.lanes):
            args = [sync, self.l0_noc[0], self.l0_noc[1], p, up, dn, x.buffer_address(), self.y.buffer_address(),
                    h * per, per, row_base, int(h == 0), self.n, len(self.lanes), h, int(local), src_base,
                    0, self.n, 0, per, per] + cons_rt + sc_send  # fmt: skip
            n_base = len(args)  # (the probe pair's index: before the fabric connections connect() appends)
            rt_send[lx][ly] = args + pr
            rt_read[lx][ly] = ([x.buffer_address(), h * per, per, src_base] + ([int(self.wait[1])] if self.wait else [])
                               + sc_read)  # fmt: skip
        sdefs = [("FS_CONS", "1")] if self.cons is not None else []
        if self.after is not None:
            sdefs += [("FS_LATE_OPEN", "1"), ("FS_AFTER_SEM", str(int(self.after)))]
        rdefs = [("FS_WAIT", str(int(self.wait[0])))] if self.wait is not None else []
        if self.scatter:
            sdefs.append(("FS_SCATTER", "1"))
            rdefs.append(("FS_SCATTER", "1"))
        if self.n > 2 and not row_barrier_defines():  # THE PER-HOP DRAIN: a multi-hop line -> the waiter's row barrier
            sdefs.append(("FS_ROW_BARRIER", "1"))
        if pr:
            sdefs.append(("MK_PROBE_ARG", str(n_base)))
        kd = lambda name, ct, rt, cfg, defs: ttnn.KernelDescriptor(
            kernel_source=self._srcs[name], source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=_crs(self.lanes), compile_time_args=ct, runtime_args=rt, config=cfg, defines=defs)  # fmt: skip
        kernels = [kd("fs_send.cpp", ct_send, rt_send, ttnn.WriterConfigDescriptor(), sdefs),
                   kd("fs_read.cpp", ct_read, rt_read, ttnn.ReaderConfigDescriptor(), rdefs)]  # fmt: skip
        cbs = [ttnn.CBDescriptor(total_size=self.slots * self.nb * TB, core_ranges=_crs(self.lanes), format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=cb_stage, data_format=ttnn.bfloat16, page_size=TB)])]  # fmt: skip
        nl = len(self.lanes)
        counters = dict(core=self.lanes[0], noc=self.l0_noc, sync=sync, word=lambda q, h: q * nl + h,
                        chunks_per_lane=per, lanes=nl)  # fmt: skip
        return kernels, cbs, [x, self.y, self.sync], counters

    def semaphores(self) -> list:
        """the lane cores' wait semaphore (wait=), initial 0"""
        out = []
        if self.wait is not None:
            out.append(ttnn.SemaphoreDescriptor(id=int(self.wait[0]), core_ranges=_crs(self.lanes), initial_value=0))
        if self.after is not None:
            out.append(ttnn.SemaphoreDescriptor(id=int(self.after), core_ranges=_crs(self.lanes), initial_value=0))
        return out

    def lane_noc(self) -> list:
        return [(int(w.x), int(w.y)) for w in (self.mesh.worker_core_from_logical_core(ttnn.CoreCoord(*ln))
                                                for ln in self.lanes)]  # fmt: skip

    def directions(self, r: int, c: int) -> list:
        """("a<axis>+" / "a<axis>-") the directions chip (r, c) holds a connection in (tt/shard_plan.py names)"""
        p, line = self._pos(r, c)
        _, _, _, up, dn = self._block(p, line)
        return ([f"a{self.axis}+"] if up else []) + ([f"a{self.axis}-"] if dn else [])

    def headers(self, r: int, c: int) -> int:
        """fabric packet headers fs_send allocates on the lane BRISC of chip (r, c) (NHDR per direction)"""
        return NHDR * len(self.directions(r, c))

    def program(self, x, r: int, c: int):
        kernels, cbs, _, _ = self.parts(x, r, c)
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=self.semaphores(), cbs=cbs)
        self.connect(prog, r, c, 0)
        return prog

    def connect(self, prog, r: int, c: int, send_kernel_index: int):
        p, line = self._pos(r, c)
        _, _, _, up, dn = self._block(p, line)
        self._connect(prog, r, c, up, dn, send_kernel_index)

    def _connect(self, prog, r: int, c: int, up: int, dn: int, ki: int = 0):
        """lane h: link link_base + h towards the next chip "up" (position + 1) and "down" (position - 1) along the
        axis, in the kernel's order"""
        mesh = self.mesh
        me = mesh.get_fabric_node_id(ttnn.MeshCoordinate(r, c))
        step = (lambda d: (r, c + d)) if self.axis == 1 else (lambda d: (r + d, c))  # noqa: E731
        for h, lane in enumerate(self.lanes):
            core = ttnn.CoreCoord(*lane)
            for d, rng in ((1, up), (-1, dn)):
                if rng:
                    peer = mesh.get_fabric_node_id(ttnn.MeshCoordinate(*step(d)))
                    fab = fabric_connect(me, peer, self.link_base + h, prog, core)
                    prog.kernels[ki].runtime_args[lane[0]][lane[1]].extend(fab)

    def run(self, x):
        mpd = ttnn.MeshProgramDescriptor()
        for r in range(self.R):
            for c in range(self.C):
                m = ttnn.MeshCoordinate(r, c)
                mpd[ttnn.MeshCoordinateRange(m, m)] = self.program(x, r, c)
        ttnn.generic_op([x, self.y, self.sync], mpd)
        return self.y

    def release(self):
        for t in (self.y, self.sync) if self.own_y else (self.sync,):
            ttnn.deallocate(t)


class FabricStreamAG(_FabricStream):
    """All-gather of x [1, 1, R_loc, H] bf16 rows (chip p's) -> self.y [1, 1, N R_loc, H] on every chip of the line:
    block (line n + p) = position p's rows of line `line` (N = blocks = n x lines unless y= is given). bundle=True (x IS
    y, local_copy off): position p's block is y's block p, already in place, multicast to the line's other chips."""

    def __init__(self, mesh, rows_local: int, hidden: int = 4096, *, local_copy: bool = True, bundle: bool = False,
                 **kw):  # fmt: skip
        assert rows_local % 32 == 0, rows_local
        ax = kw.get("axis")
        R, C = int(mesh.shape[0]), int(mesh.shape[1])
        n = (R if ax == 0 else C) if ax is not None else R * C
        lines = (R * C) // n
        blocks = n if bundle else n * lines
        super().__init__(mesh, rows_local // 32, hidden, blocks * rows_local, **kw)
        self.rows_local, self.local_copy, self.bundle = rows_local, bool(local_copy) and not bundle, bundle

    def _block(self, p: int, line: int = 0):
        if self.bundle:
            return p * self.chunks, p * self.chunks, False, self.n - 1 - p, p
        return 0, (line * self.n + p) * self.chunks, self.local_copy, self.n - 1 - p, p

    def __call__(self, x):
        assert tuple(x.shape)[-2:] == (self.rows_local, self.hidden) and x.dtype == ttnn.bfloat16, (x.shape, x.dtype)
        return self.run(x)


class FabricStreamRS2(_FabricStream):
    """The pair's reduce-scatter exchange: the partial P [1, 1, 2 R_loc, H] bf16 -> self.y [1, 1, R_loc, H] on chip p = the
    PEER's P rows [p R_loc, (p + 1) R_loc) (the consumer merges them with its own block). axis=: the pair axis of a 2-D
    mesh (every column of the 2x4 mesh is a pair)."""

    def __init__(self, mesh, rows_local: int, hidden: int = 4096, **kw):
        assert rows_local % 32 == 0, rows_local
        super().__init__(mesh, rows_local // 32, hidden, rows_local, **kw)
        assert self.n == 2, "the pair (a line of > 2 needs adds in transit)"
        self.rows_local = rows_local

    def _block(self, p: int, line: int = 0):
        q = 1 - p  # the peer: its block of my partial goes to its y rows [0, R_loc)
        return q * self.chunks, 0, False, 1 - p, p

    def __call__(self, x):
        assert tuple(x.shape)[-2:] == (2 * self.rows_local, self.hidden) and x.dtype == ttnn.bfloat16, (
            x.shape,
            x.dtype,
        )
        return self.run(x)


class FabricStreamScatter(_FabricStream):
    """Per-destination unicast scatter along a line of n (TP8 RS round 2, shard_tp8_design.md section 3): x [1, 1, n
    R_loc, H] bf16, block q = this chip's data for line position q -> peer q's self.y [1, 1, n R_loc, H] block p (p = this
    chip's position; block p of x and of y stay local / unwritten). fs_send.cpp FS_SCATTER: one connection per direction,
    every packet routed to its destination's hop count (to_chip_unicast), the chunk-end packet a fused write + atomic
    increment of the receiver's counter for this source (ARR + p LANES + h: one counter per source, in order)."""

    scatter = True

    def __init__(self, mesh, rows_local: int, hidden: int, **kw):
        assert rows_local % 32 == 0, rows_local
        ax = kw.get("axis")
        R, C = int(mesh.shape[0]), int(mesh.shape[1])
        n = (R if ax == 0 else C) if ax is not None else R * C
        super().__init__(mesh, rows_local // 32, hidden, n * rows_local, **kw)
        self.rows_local = rows_local

    def _block(self, p: int, line: int = 0):
        return 0, p * self.chunks, False, self.n - 1 - p, p

    def _dsts(self, p: int) -> list:
        return [(q, q * self.chunks, 0 if q > p else 1, abs(q - p)) for q in range(self.n) if q != p]

    def __call__(self, x):
        assert tuple(x.shape)[-2:] == (self.n * self.rows_local, self.hidden) and x.dtype == ttnn.bfloat16, x.shape
        return self.run(x)


CB_XS = 35  # prefill_megakernel.md §4b: the MM cores' x̂ store (256 KiB = one 32-row block of 4,096 bf16 columns)
CB_OUT = 42  # §4b: the MM cores' matmul output (C2: the o_proj / FFN partial block)
SEM_FS_REDUCED = 14  # §4b-ids: the owner's MM core j: the peer's partial of block j landed (C2)
SEM_FS_ARRIVE = 15  # §4b-ids: MM core j waits FS_ARRIVE >= SLICES for its block (local or remote)


def row_barrier_defines() -> list:
    """THE PER-HOP DRAIN (kernels/fabric_stream/fs_common.hpp BAR_L / BAR_R / LANDED): the axis-1 lanes' end-of-launch
    row barrier of 1-hop tokens is on by default; GLM53_FS_ROW_BARRIER=0 compiles it out (FS_NO_ROW_BARRIER, the A/B)"""
    return [] if os.environ.get("GLM53_FS_ROW_BARRIER", "1") == "1" else [("FS_NO_ROW_BARRIER", "1")]


# THE RING-ADDRESS RULE (fs_common.hpp): the lanes / reducers +1 these ids at their OWN get_semaphore(id)
assert (_FS["SEM_ARRIVE_ID"], _FS["SEM_REDUCED_ID"]) == (
    SEM_FS_ARRIVE,
    SEM_FS_REDUCED,
), "fs_common.hpp SEM_*_ID vs the host"
CB_POST = (
    21  # §4b H6 / the family's post_layout: member (b, j)'s POST = one owner block's 32 x 512 fp32 y slice (64 KB)
)
W_XSADDR, W_OUTADDR = _FS["XSADDR"], _FS["OUTADDR"]  # kernels/fabric_stream/fs_common.hpp


class FabricStreamC1:
    """C1 / C3 of the prefill megakernel (prefill_megakernel.md §4b H1 / H2, §5): the sequence-parallel all-gather of x̂
    straight into the MM cores' XS. Block j (32 rows) lives on MM core j of EVERY chip (`mm_cores[j]`, the same XS L1
    address on every core and chip). Chip p produces its local blocks j in [p B, (p + 1) B) (B = blocks per chip); lane h
    pulls its local blocks (b % lanes == h) from their MM cores' XS once their SLICES producer slices landed (LREADY + k)
    and writes them into the same XS of every peer, each slice ending with a fused +1 of the peer MM core's FS_ARRIVE
    (semaphore 15). Every launch starts with a credit exchange between the lanes (CREDIT + q, monotonic): no remote packet
    lands before the receiving chip's program runs (its XS / semaphores exist). One mesh axis (the pair) for now.
    parts(): the lane kernels / CBs / io for a layer program; unit(): + stub producer / consumer kernels on the MM cores
    (DRAM rows -> XS -> lanes -> peer XS -> DRAM y), the standalone d0 unit."""

    def __init__(self, mesh, blocks_local: int, *, mm_cores, lanes, hidden: int = 4096, slice_t: int = 16,
                 nb: int = 16, slots: int = 4, mode: str = "c1"):  # fmt: skip
        """mode "c1": the AG above. "c2" (C2 / C4, the pair's RS, TP2 DRAM form): every chip's MM core j holds its partial
        block j in OUT (CB 42); lane h sends the blocks the PEER owns (b % lanes == h of the peer's range) from OUT into
        the owner's landing buffer r (DRAM [B 32, H], row = the block's index in the owner's range), a fused +1 of the
        owner MM core j's FS_REDUCED (semaphore 14) per slice; the owner's consumer adds own OUT + landed (fixed chip
        order, fp32). The TP8 L1 reduce ring is next."""
        assert mode in ("c1", "c2"), mode
        self.mode = mode
        self.mesh = mesh
        self.R, self.C = int(mesh.shape[0]), int(mesh.shape[1])
        assert self.R == 1 or self.C == 1, "one mesh axis (the pair); the 2x4 two-stage version is next"
        self.n = self.R * self.C
        self.bl, self.hidden = blocks_local, hidden
        self.tpb = hidden // 32  # tiles per 32-row block
        self.mm_cores = [tuple(c) for c in mm_cores]
        assert len(self.mm_cores) == self.n * blocks_local, (len(self.mm_cores), self.n, blocks_local)
        self.lanes = [tuple(c) for c in lanes]
        assert not set(self.lanes) & set(self.mm_cores)
        assert (
            blocks_local % len(self.lanes) == 0 and blocks_local // len(self.lanes) <= 16
        ), "LREADY: 16 words per lane"
        assert self.n <= 8, "CREDIT words 44..51"
        self.slice_t, self.nb, self.slots = slice_t, nb, slots
        assert self.tpb % slice_t == 0 and self.tpb % nb == 0 and nb % 2 == 0 and slice_t % 2 == 0
        words = SYNC_WORDS * WORD // 4
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                               ttnn.ShardSpec(_crs(self.lanes), [1, words], ttnn.ShardOrientation.ROW_MAJOR))  # fmt: skip
        self.sync = ttnn.from_torch(torch.zeros(len(self.lanes), words, dtype=torch.int32), dtype=ttnn.uint32,
                                    layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=mc,
                                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))  # fmt: skip
        self._srcs = {n: kernel_text(n) for n in ("c1_send.cpp", "c1_read.cpp", "c1_stub_prod.cpp", "c1_stub_cons.cpp",
                                                  "c2_stub_prod.cpp", "c2_stub_cons.cpp")}  # fmt: skip
        self.r = None
        if mode == "c2":
            assert self.n == 2, "C2 TP2 form: the pair"
            self.r = ttnn.from_torch(torch.zeros(1, 1, blocks_local * 32, hidden, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                                     layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                     mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))  # fmt: skip
        self._noc = lambda xy: (lambda w: (int(w.x), int(w.y)))(mesh.worker_core_from_logical_core(ttnn.CoreCoord(*xy)))
        self.y = None

    def _pos(self, r, c):
        return c if self.C > 1 else r

    def lane_blocks(self, p: int, h: int) -> list:
        """global block ids lane h of chip p sends, in order (c1: its own range; c2: the peer's range)"""
        o = p if self.mode == "c1" else 1 - p
        return [o * self.bl + b for b in range(self.bl) if b % len(self.lanes) == h]

    def parts(self, r: int, c: int, *, cb_stage: int = CB_STAGE, xsfree_n: int = 0, fab_at: int | None = None):
        """(kernels [c1_send, c1_read] on the lanes, cbs [staging], io [sync]) of chip (r, c); then connect().
        fab_at: the absolute runtime-arg index of the fabric args (None: right after this phase's args; a composite of
        several C1 phases passes the total so every phase opens the same connection). self._rt_last = the lanes'
        (send, read) runtime lists of this call (for composing).
        xsfree_n (PMK H0, C1 / C3 in the layer program): member XS-FREE credits per use per lane (32 at 64 members: member
        G +1s word XSFREE of lane G % 2 after its own FN use); the lane waits for them before crediting its peer"""
        p = self._pos(r, c)
        up, dn = self.n - 1 - p, p
        sync = self.sync.buffer_address()
        rt_s, rt_r = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        self._rt_last = {}
        for h, (lx, ly) in enumerate(self.lanes):
            js = self.lane_blocks(p, h)
            blk = [v for j in js for v in self._noc(self.mm_cores[j])]
            me = self._noc((lx, ly))
            args = [sync, me[0], me[1], p, self.n, up, dn, len(js)] + blk
            if self.mode == "c2":  # the owner's landing rows (a RuntimeArgs entry is a VectorUInt32 once assigned)
                args += [self.r.buffer_address()] + [j % self.bl for j in js]
            args += [xsfree_n]  # PMK H0: member XS-FREE credits per use on this lane (0: the launch credit only)
            args += [len(args) + 1 if fab_at is None else fab_at]  # where connect() puts the fabric args (absolute)
            rt_s[lx][ly] = args
            self._rt_last[(lx, ly)] = (list(args), [sync, len(js)] + blk)
            rt_r[lx][ly] = [sync, len(js)] + blk
        c2 = self.mode == "c2"
        kd = lambda name, ct, rt, cfg, defs=(): ttnn.KernelDescriptor(
            kernel_source=self._srcs[name], source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=_crs(self.lanes), compile_time_args=ct, runtime_args=rt, config=cfg, defines=list(defs))  # fmt: skip
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        kernels = [kd("c1_send.cpp", [self.tpb, self.nb, cb_stage, self.slice_t] + (acc(self.r) if c2 else []), rt_s,
                      ttnn.WriterConfigDescriptor(), [("FS_C2", "1")] if c2 else []),
                   kd("c1_read.cpp", [self.tpb, self.nb, cb_stage, self.tpb // self.slice_t, W_OUTADDR if c2 else W_XSADDR],
                      rt_r, ttnn.ReaderConfigDescriptor())]  # fmt: skip
        cbs = [ttnn.CBDescriptor(total_size=self.slots * self.nb * TB, core_ranges=_crs(self.lanes), format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=cb_stage, data_format=ttnn.bfloat16, page_size=TB)])]  # fmt: skip
        return kernels, cbs, [self.sync] + ([self.r] if c2 else [])

    def connect(self, prog, r: int, c: int, send_kernel_index: int):
        p = self._pos(r, c)
        up, dn = self.n - 1 - p, p
        me = self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(r, c))
        step = (lambda d: (r, c + d)) if self.C > 1 else (lambda d: (r + d, c))  # noqa: E731
        for h, lane in enumerate(self.lanes):
            for d, rng in ((1, up), (-1, dn)):
                if rng:
                    peer = self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(*step(d)))
                    fab = fabric_connect(me, peer, h, prog, ttnn.CoreCoord(*lane))
                    prog.kernels[send_kernel_index].runtime_args[lane[0]][lane[1]].extend(fab)

    def unit(self, x):
        if self.mode == "c2":
            return self.unit_c2(x)
        return self.unit_c1(x)

    def unit_c2(self, pt):
        """the standalone C2 unit: pt [1, 1, n B 32, H] bf16 = this chip's partial (DRAM) -> stub producers fill OUT on
        every MM core -> the lanes send the peer-owned blocks -> self.r on chip p = the peer's partial rows of p's range
        """
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        mm = _crs(self.mm_cores)
        sync = self.sync.buffer_address()
        slices = self.tpb // self.slice_t
        mpd = ttnn.MeshProgramDescriptor()
        for r in range(self.R):
            for c in range(self.C):
                p = self._pos(r, c)
                kernels, cbs, _ = self.parts(r, c)
                rt_p, rt_c = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
                for j, (mx, my) in enumerate(self.mm_cores):
                    o = j // self.bl  # the owner of block j
                    if o != p:
                        b = j - o * self.bl
                        lane = self._noc(self.lanes[b % len(self.lanes)])
                        rt_p[mx][my] = [pt.buffer_address(), j, 1, lane[0], lane[1], sync, b // len(self.lanes)]
                    else:
                        rt_p[mx][my] = [pt.buffer_address(), j, 0, 0, 0, 0, 0]
                    rt_c[mx][my] = [int(o == p)]
                kd = lambda name, ct, rt, cfg: ttnn.KernelDescriptor(
                    kernel_source=self._srcs[name], source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=mm, compile_time_args=ct, runtime_args=rt, config=cfg)  # fmt: skip
                kernels = kernels + [
                    kd("c2_stub_prod.cpp", [self.tpb, self.slice_t, CB_OUT, SEM_FS_REDUCED] + acc(pt), rt_p,
                       ttnn.ReaderConfigDescriptor()),
                    kd("c2_stub_cons.cpp", [slices, SEM_FS_REDUCED], rt_c, ttnn.WriterConfigDescriptor()),
                ]  # fmt: skip
                cbs = cbs + [ttnn.CBDescriptor(total_size=self.tpb * TB, core_ranges=mm, format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=CB_OUT, data_format=ttnn.bfloat16, page_size=TB)])]  # fmt: skip
                sems = [ttnn.SemaphoreDescriptor(id=SEM_FS_REDUCED, core_ranges=mm, initial_value=0)]
                prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
                self.connect(prog, r, c, 0)
                m = ttnn.MeshCoordinate(r, c)
                mpd[ttnn.MeshCoordinateRange(m, m)] = prog
        ttnn.generic_op([pt, self.r, self.sync], mpd)
        return self.r

    def unit_c1(self, x):
        """the standalone unit: x [1, 1, B 32, H] bf16 (this chip's rows, DRAM) -> self.y [1, 1, n B 32, H] (DRAM) via the
        stub producers (local blocks into XS), the lanes, the stub consumers (every block's XS -> y)"""
        if self.y is None:
            self.y = ttnn.from_torch(torch.zeros(1, 1, self.n * self.bl * 32, self.hidden, dtype=torch.bfloat16),
                                     dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                                     memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                     mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))  # fmt: skip
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        mm = _crs(self.mm_cores)
        sync = self.sync.buffer_address()
        mpd = ttnn.MeshProgramDescriptor()
        for r in range(self.R):
            for c in range(self.C):
                p = self._pos(r, c)
                # the 2x4 (C1X): the data headers in a private lane CB (the 12-header pool overflows on the middle chips)
                kernels, cbs, _ = self.parts(r, c, hdr_cb=CB_HDR_FS) if self.n > 2 else self.parts(r, c)
                rt_p, rt_c = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
                for j, (mx, my) in enumerate(self.mm_cores):
                    b = j - p * self.bl
                    me = self._noc((mx, my))
                    if 0 <= b < self.bl:
                        h = b % len(self.lanes)
                        lane = self._noc(self.lanes[h])
                        rt_p[mx][my] = [1, x.buffer_address(), b, lane[0], lane[1], sync, b // len(self.lanes), me[0], me[1],
                                        0, 0, 0]  # fmt: skip
                    else:
                        rt_p[mx][my] = [0] * 12
                    rt_c[mx][my] = [self.y.buffer_address(), j]
                kd = lambda name, ct, rt, cfg: ttnn.KernelDescriptor(
                    kernel_source=self._srcs[name], source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=mm, compile_time_args=ct, runtime_args=rt, config=cfg)  # fmt: skip
                slices = self.tpb // self.slice_t
                kernels = kernels + [
                    kd("c1_stub_prod.cpp", [self.tpb, self.slice_t, CB_XS, SEM_FS_ARRIVE] + acc(x), rt_p,
                       ttnn.ReaderConfigDescriptor()),
                    kd("c1_stub_cons.cpp", [self.tpb, slices, CB_XS, SEM_FS_ARRIVE] + acc(self.y), rt_c,
                       ttnn.WriterConfigDescriptor()),
                ]  # fmt: skip
                cbs = cbs + [ttnn.CBDescriptor(total_size=self.tpb * TB, core_ranges=mm, format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=CB_XS, data_format=ttnn.bfloat16, page_size=TB)])]  # fmt: skip
                sems = [ttnn.SemaphoreDescriptor(id=SEM_FS_ARRIVE, core_ranges=mm, initial_value=0)]
                prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
                self.connect(prog, r, c, 0)
                m = ttnn.MeshCoordinate(r, c)
                mpd[ttnn.MeshCoordinateRange(m, m)] = prog
        ttnn.generic_op([x, self.y, self.sync], mpd)
        return self.y

    def release(self):
        for t in (self.y, self.sync, self.r):
            if t is not None:
                ttnn.deallocate(t)


class FabricStreamRSR:
    """C2 / C4 with the OWNER-LANE REDUCE (prefill_megakernel.md §4b H5 -> H6, the pair): every chip's partial P [1, 1,
    2 B 32, H] bf16 (DRAM; chip o owns rows [o B 32, (o + 1) B 32)) -> y [1, 1, B 32, H] fp32 on every chip = own + peer
    partial of its rows. Lane h (blocks b % lanes == h) interleaves per step i: SEND the peer-owned block i (DRAM -> the
    peer's landing buffer r, 2-page scatter packets, a fused in-order +1 of the peer lane h's ARR word per block) and
    REDUCE own block i once the peer's copy landed (NCRISC streams own + landed tiles, TRISC adds in fp32, BRISC writes y
    fp32). The launch credit guards the landing buffer across launches. Readiness words (LREADY / OWNRDY: the producers'
    per-block +1) are honoured with ready=True; the unit runs ready=False (P is an input). y in the layer program goes to
    the members' CB_Y 21 + FS_REDUCED (next step); here to DRAM."""

    def __init__(self, mesh, blocks_local: int, *, lanes, hidden: int = 4096, nb: int = 16, slots: int = 4,
                 reducers=None, members=None, axis: int | None = None, out_dtype=None, block_order=None):  # fmt: skip
        """reducers: None = the lanes reduce (interleaved with their sends); a core list = the lanes ONLY send and the
        reducers (red_read / rs_reduce / red_write) take the owned blocks round-robin, polling the receiving lane's
        in-order landed counter over the NoC (d0: 2 lanes doing both move 32 MiB per chip: 623 us at 8 MiB).
        members: the PMK H6 landing (the family's member_map / post_layout): 8 groups x 8 members (member G = (G // 8,
        G % 8)); reducer k serves group k (owned blocks b % 8 == k, in order) and writes block b's column slice j (16
        fp32 tiles) into member (k, j)'s POST 21 + FS_REDUCED (14) +1, under the group's POSTFREE credits. The unit's
        stub member copies POST to y (DRAM) and frees the slot."""
        self.mesh = mesh
        self.R, self.C = int(mesh.shape[0]), int(mesh.shape[1])
        # the pair: the whole mesh (1 x 2 / 2 x 1), or axis 0 of an R = 2 mesh (TP8's stage 1: the board pairs (0, c) -
        # (1, c) exchange; chip (r, c) owns the rows of its mesh ROW: blocks [r B, (r + 1) B) of P)
        self.axis = axis if axis is not None else (1 if self.C > 1 else 0)
        assert (self.R * self.C == 2) or (self.axis == 0 and self.R == 2), (self.R, self.C, axis)
        self.bl, self.hidden, self.tpb = blocks_local, hidden, hidden // 32
        # block_order(c) = the local blocks in send / reduce order on mesh column c (TP8 stage 1 under
        # GLM53_PMK_RS8_ORDER: the stage-2 destination order); None = ascending. Lane h sends, and reducer k reduces,
        # its blocks in this order; every index (LREADY / OWNRDY / the landed-counter position) follows the lists.
        self.block_order = block_order
        self.lanes = [tuple(c) for c in lanes]
        self.reducers = [tuple(c) for c in reducers] if reducers else []
        assert not set(self.reducers) & set(self.lanes)
        self.members = [tuple(c) for c in members] if members else []
        if self.members:
            mpg = self.tpb // nb  # members per group = column slices of nb tiles
            assert len(self.reducers) == mpg and len(self.members) == mpg * mpg, (len(self.reducers), len(self.members))
            assert not set(self.members) & (set(self.reducers) | set(self.lanes))
        nl = len(self.lanes)
        assert blocks_local % nl == 0 and blocks_local // nl <= 16, (blocks_local, nl)
        assert self.tpb % nb == 0 and nb % 4 == 0, (self.tpb, nb)
        self.nb, self.slots = nb, slots
        rep = ttnn.ReplicateTensorToMesh(mesh)
        words = SYNC_WORDS * WORD // 4
        sc = self.lanes + self.reducers  # one shard per core: the same address on every lane / reducer
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                               ttnn.ShardSpec(_crs(sc), [1, words], ttnn.ShardOrientation.ROW_MAJOR))  # fmt: skip
        self.sync = ttnn.from_torch(torch.zeros(len(sc), words, dtype=torch.int32), dtype=ttnn.uint32,
                                    layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=mc, mesh_mapper=rep)  # fmt: skip
        rows = blocks_local * 32
        self.r = ttnn.from_torch(torch.zeros(1, 1, rows, hidden, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                 mesh_mapper=rep)  # fmt: skip
        # out_dtype bf16 (TP8 stage 1 under GLM53_PMK_RS8_S2_BF16): the reducers pack the exact fp32 pair sum to bf16
        # (red_write FS_Y_BF16), halving stage 2's bytes; only the reducers' DRAM path (no lanes reduce, no H6 members)
        self.y_dtype = out_dtype if out_dtype is not None else ttnn.float32
        assert self.y_dtype == ttnn.float32 or (self.reducers and not self.members), "bf16 y: the reducers' DRAM path"
        y_t = torch.bfloat16 if self.y_dtype == ttnn.bfloat16 else torch.float32
        self.y = ttnn.from_torch(torch.zeros(1, 1, rows, hidden, dtype=y_t), dtype=self.y_dtype,
                                 layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                 mesh_mapper=rep)  # fmt: skip
        self._srcs = {n: kernel_text(n) for n in ("rs_read.cpp", "rs_reduce.cpp", "rs_send.cpp", "red_read.cpp",
                                                  "red_write.cpp", "rsr_stub_member.cpp")}  # fmt: skip
        self._noc = lambda xy: (lambda w: (int(w.x), int(w.y)))(mesh.worker_core_from_logical_core(ttnn.CoreCoord(*xy)))

    def program(self, pt, r: int, c: int, ready: bool = False):
        kernels, cbs, sems, _ = self._build(pt, r, c, ready)
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        self.connect(prog, r, c, 0)
        return prog

    def order(self, c: int) -> list:
        o = list(self.block_order(c)) if self.block_order is not None else list(range(self.bl))
        assert sorted(o) == list(range(self.bl)), o
        return o

    def lane_list(self, c: int, h: int) -> list:
        """the blocks lane h sends (and, lanes reducing, reduces) on mesh column c, in order: LREADY + i = item i"""
        return [b for b in self.order(c) if b % len(self.lanes) == h]

    def red_list(self, c: int, k: int) -> list:
        """the blocks reducer k reduces on mesh column c, in order: OWNRDY + i = item i"""
        return [b for b in self.order(c) if b % len(self.reducers) == k]

    def _build(self, pt, r: int, c: int, ready: bool = False, stub_kernel: bool = True, red_signals: dict | None = None,
               fab_at: int | None = None):  # fmt: skip
        """(kernels, cbs, semaphores, stub) of the RS on chip (r, c); kernels[0] = rs_send (the lanes' fabric
        connections: connect()). stub_kernel=False: the members' H6 stub is NOT a kernel but returned as parts
        (source, ct, {member: rt}) for a composite member BRISC (tt/pf_rs.py). red_signals = {reducer core: [(noc x,
        noc y, L1 address) per block of its list]}: +1 there after the block's y rows are written (TP8 stage 1 -> 2)"""
        p = c if self.axis == 1 else r
        q = 1 - p
        up, dn = (1, 0) if p == 0 else (0, 1)
        nl, bl, tpb, nb = len(self.lanes), self.bl, self.tpb, self.nb
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        sync = self.sync.buffer_address()
        rt_r, rt_s = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        lanes_reduce = not self.reducers
        n_red = bl // nl if lanes_reduce else 0
        for h, (lx, ly) in enumerate(self.lanes):
            bs = self.lane_list(c, h)
            rb = bs if lanes_reduce else []
            send_j = [q * bl + b for b in bs]  # peer-owned blocks: P row tiles; the peer's landing rows = b
            red = [v for b in rb for v in (p * bl + b, b)]  # own block: P row tile, landed row in r
            rt_r[lx][ly] = [sync, q, nl, h, pt.buffer_address(), self.r.buffer_address(), int(ready), len(bs), len(rb)] \
                + send_j + red  # fmt: skip
            me = self._noc((lx, ly))
            args = [sync, me[0], me[1], p, q, nl, h, up, dn, self.r.buffer_address(), self.y.buffer_address(),
                    len(bs), len(rb)] + bs + rb  # fmt: skip
            rt_s[lx][ly] = args + [len(args) + 1 if fab_at is None else fab_at]  # fab_at: where connect() appends
        kd = lambda name, ct, rt, cfg: ttnn.KernelDescriptor(
            kernel_source=self._srcs[name], source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=_crs(self.lanes), compile_time_args=ct, runtime_args=rt, config=cfg)  # fmt: skip
        cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                          fp32_dest_acc_en=True)  # fmt: skip
        CB_SEND_, CB_A, CB_B, CB_Y = 0, 1, 2, 3
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[CB_A] = modes[CB_B] = ttnn.UnpackToDestMode.UnpackToDestFp32  # exact bf16 -> fp32 DST (rs_reduce.cpp)
        cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        kernels = [
            kd(
                "rs_send.cpp",
                [tpb, nb, CB_SEND_, CB_Y] + acc(self.r) + acc(self.y),
                rt_s,
                ttnn.WriterConfigDescriptor(),
            ),
            kd(
                "rs_read.cpp",
                [tpb, nb, CB_SEND_, CB_A, CB_B] + acc(pt) + acc(self.r),
                rt_r,
                ttnn.ReaderConfigDescriptor(),
            ),
        ]
        cb = lambda i, dt, page, pages, cores: ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=_crs(cores),
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)],
        )  # noqa: E731  # fmt: skip
        red_cores = self.lanes if lanes_reduce else self.reducers
        cbs = [cb(CB_SEND_, ttnn.bfloat16, TB, self.slots * nb, self.lanes), cb(CB_A, ttnn.bfloat16, TB, 2 * nb, red_cores),
               cb(CB_B, ttnn.bfloat16, TB, 2 * nb, red_cores),
               cb(CB_Y, self.y_dtype, TB if self.y_dtype == ttnn.bfloat16 else 4096, 2 * nb, red_cores)]  # fmt: skip
        if lanes_reduce:
            kernels.append(kd("rs_reduce.cpp", [n_red * tpb, nb, CB_A, CB_B, CB_Y], ttnn.RuntimeArgs(), cc))
        else:  # the reducers: owned blocks round-robin; block b landed on MY lane b % nl, position b // nl in its list
            nr = len(self.reducers)
            per = [self.red_list(c, k) for k in range(nr)]
            assert len({len(x) for x in per}) == 1, "equal reduce work per reducer (compile-time tile count)"
            rt_rr, rt_rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for k, (rx, ry) in enumerate(self.reducers):
                args = [sync, sync, bl // nl, pt.buffer_address(), self.r.buffer_address(), int(ready), len(per[k])]
                for b in per[k]:
                    ln = self._noc(self.lanes[b % nl])
                    # ARR + q LANES + h on lane h; the block's position in the PEER lane's list (same column: same order)
                    args += [p * bl + b, b, ln[0], ln[1], q * nl + b % nl, self.lane_list(c, b % nl).index(b)]
                rt_rr[rx][ry] = args
                if self.members:  # H6: reducer k -> group k's members (k, j), j = the column slice
                    mpg = tpb // nb
                    rt_rw[rx][ry] = [sync, len(per[k]), mpg] + [v for j in range(mpg)
                                                                for v in self._noc(self.members[k * mpg + j])]  # fmt: skip
                else:
                    sig = [v for t in (red_signals or {}).get((rx, ry), []) for v in t]
                    assert not red_signals or len(sig) == 3 * len(
                        per[k]
                    ), "one signal per block (no zero-padded targets)"
                    rt_rw[rx][ry] = [self.y.buffer_address(), len(per[k])] + per[k] + sig
            kr = lambda name, ct, rt, cfg, cores=self.reducers, defs=(): ttnn.KernelDescriptor(
                kernel_source=self._srcs[name], source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=_crs(cores), compile_time_args=ct, runtime_args=rt, config=cfg, defines=list(defs))  # fmt: skip
            kernels += [
                kr("red_read.cpp", [tpb, nb, CB_A, CB_B] + acc(pt) + acc(self.r), rt_rr, ttnn.ReaderConfigDescriptor()),
                kr("rs_reduce.cpp", [len(per[0]) * tpb, nb, CB_A, CB_B, CB_Y], ttnn.RuntimeArgs(), cc),
                kr(
                    "red_write.cpp",
                    [tpb, nb, CB_Y] + acc(self.y),
                    rt_rw,
                    ttnn.WriterConfigDescriptor(),
                    defs=[("FS_POST", "1")]
                    if self.members
                    else (
                        ([("FS_RED_SIG", "1")] if red_signals else [])
                        + ([("FS_Y_BF16", "1")] if self.y_dtype == ttnn.bfloat16 else [])
                    ),
                ),
            ]
        sems, stub = [], None
        if self.members:  # the unit's stub members (the PMK member's H6 consumer)
            mpg = tpb // nb
            rt_m, rt_md = ttnn.RuntimeArgs(), {}
            for G, (mx, my) in enumerate(self.members):
                k, j = divmod(G, mpg)
                red = self._noc(self.reducers[k])
                blocks = [b for b in range(bl) if b % len(self.reducers) == k]
                rt_md[(mx, my)] = [sync, red[0], red[1], self.y.buffer_address(), j, len(blocks)] + blocks
                rt_m[mx][my] = rt_md[(mx, my)]
            st_ct = [tpb, nb, CB_POST, SEM_FS_REDUCED] + acc(self.y)
            if stub_kernel:
                kernels.append(
                    kr("rsr_stub_member.cpp", st_ct, rt_m, ttnn.WriterConfigDescriptor(), cores=self.members)
                )
            else:
                stub = dict(src=self._srcs["rsr_stub_member.cpp"], ct=st_ct,
                            rt=rt_md)  # fmt: skip
            cbs.append(cb(CB_POST, ttnn.float32, 4096, nb, self.members))
            sems = [ttnn.SemaphoreDescriptor(id=SEM_FS_REDUCED, core_ranges=_crs(self.members), initial_value=0)]
        return kernels, cbs, sems, stub

    def connect(self, prog, r: int, c: int, send_kernel_index: int = 0):
        """the lanes' fabric connections (rs_send's runtime args tail)"""
        p = c if self.axis == 1 else r
        up, dn = (1, 0) if p == 0 else (0, 1)
        me = self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(r, c))
        step = (lambda d: (r, c + d)) if self.axis == 1 else (lambda d: (r + d, c))  # noqa: E731
        for h, lane in enumerate(self.lanes):
            for d, rng in ((1, up), (-1, dn)):
                if rng:
                    peer = self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(*step(d)))
                    fab = fabric_connect(me, peer, h, prog, ttnn.CoreCoord(*lane))
                    prog.kernels[send_kernel_index].runtime_args[lane[0]][lane[1]].extend(fab)

    def __call__(self, pt):
        assert tuple(pt.shape)[-2:] == (2 * self.bl * 32, self.hidden) and pt.dtype == ttnn.bfloat16, (
            pt.shape,
            pt.dtype,
        )
        mpd = ttnn.MeshProgramDescriptor()
        for r in range(self.R):
            for c in range(self.C):
                m = ttnn.MeshCoordinate(r, c)
                mpd[ttnn.MeshCoordinateRange(m, m)] = self.program(pt, r, c)
        ttnn.generic_op([pt, self.r, self.y, self.sync], mpd)
        return self.y

    def release(self):
        for t in (self.r, self.y, self.sync):
            ttnn.deallocate(t)


class FabricStreamC1X(FabricStreamC1):
    """C1 / C3 on a 2 x C mesh (TP8: the 2x4; prefill_megakernel.md §5 C1 "axis 0 (the board pair), then axis 1 line
    multicast"). Chip rank i = r C + c owns global blocks [i B, (i + 1) B) (block G on holder mm_cores[G] of every chip).
    Lane h (local blocks b % lanes == h): its OWN blocks go to the partner (axis 0) AND along the row (axis 1); then its
    PARTNER's blocks, as they land here over axis 0 (holder semaphore polled), go along the row. Every chip ends with all
    2 C B blocks. Credits: each lane credits the partner + its row peers and waits for theirs (+ PMK XS-FREE)."""

    def __init__(self, mesh, blocks_local: int, *, mm_cores, lanes, **kw):
        R, C = int(mesh.shape[0]), int(mesh.shape[1])
        assert R == 2, "the 2 x C mesh (axis 0 = the board pair)"
        self.mesh, self.R, self.C, self.n = mesh, R, C, R * C
        self.mode = "c1"
        # the base class asserts one axis: build its state by hand
        hidden = kw.pop("hidden", 4096)
        self.slice_t, self.nb, self.slots = kw.pop("slice_t", 16), kw.pop("nb", 16), kw.pop("slots", 4)
        assert not kw, kw
        self.bl, self.hidden, self.tpb = blocks_local, hidden, hidden // 32
        self.mm_cores = [tuple(c) for c in mm_cores]
        assert len(self.mm_cores) == self.n * blocks_local
        self.lanes = [tuple(c) for c in lanes]
        assert blocks_local % len(self.lanes) == 0 and blocks_local // len(self.lanes) <= 16
        assert self.tpb % self.slice_t == 0 and self.tpb % self.nb == 0
        words = SYNC_WORDS * WORD // 4
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                               ttnn.ShardSpec(_crs(self.lanes), [1, words], ttnn.ShardOrientation.ROW_MAJOR))  # fmt: skip
        self.sync = ttnn.from_torch(torch.zeros(len(self.lanes), words, dtype=torch.int32), dtype=ttnn.uint32,
                                    layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=mc,
                                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))  # fmt: skip
        self._srcs = {
            n: kernel_text(n) for n in ("c1x_send.cpp", "c1x_read.cpp", "c1_stub_prod.cpp", "c1_stub_cons.cpp")
        }
        self._noc = lambda xy: (lambda w: (int(w.x), int(w.y)))(mesh.worker_core_from_logical_core(ttnn.CoreCoord(*xy)))
        self.y, self.r = None, None

    def _pos(self, r, c):
        return r * self.C + c  # the chip's rank (row-major mesh order = the SP block order)

    def lane_blocks(self, p: int, h: int) -> list:
        return [p * self.bl + b for b in range(self.bl) if b % len(self.lanes) == h]

    def _ranges(self, r, c):
        return [self.R - 1 - r, r, self.C - 1 - c, c]

    def parts(
        self,
        r: int,
        c: int,
        *,
        cb_stage: int = CB_STAGE,
        xsfree_n: int = 0,
        fab_at: int | None = None,
        hdr_cb: int | None = None,
    ):
        """hdr_cb (the STANDALONE 2x4 unit, coordinator 2026-10-24 00:25:11Z): the send's DATA packet headers in a private
        L1 CB of the lane (FS_HDR_CB, as the PMK composite, c4eb61ea7bf): the middle chips' 3 directions x (4 data + 1
        credit) = 15 headers overflow the 12-header PacketHeaderPool, whose allocate_header then spins in its while(1)
        (homelab-1 00:18Z: all 8 chips hung). The composite passes None (it adds FS_HDR_CB and its own CB 5)."""
        rank = self._pos(r, c)
        partner = (1 - r) * self.C + c
        writers = [partner] + [r * self.C + cc for cc in range(self.C) if cc != c]
        sync = self.sync.buffer_address()
        rt_s, rt_r = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for h, (lx, ly) in enumerate(self.lanes):
            own, par = self.lane_blocks(rank, h), self.lane_blocks(partner, h)
            blk = [v for j in own + par for v in self._noc(self.mm_cores[j])]
            me = self._noc((lx, ly))
            args = [sync, me[0], me[1], rank] + self._ranges(r, c) + [xsfree_n, len(writers)] + writers \
                + [len(own), len(par)] + blk  # fmt: skip
            rt_s[lx][ly] = args + [len(args) + 1 if fab_at is None else fab_at]  # fab_at: where connect() appends
            # THE PER-HOP DRAIN: the holders of this lane's ROW-PEER blocks (their own + their partner's, which reach
            # this chip by axis-1 line multicast) = what this lane NCRISC waits for before LANDED (c1x_read.cpp)
            land = [j for cc in range(self.C) if cc != c for j in self.lane_blocks(r * self.C + cc, h)
                    + self.lane_blocks((1 - r) * self.C + cc, h)]  # fmt: skip
            rt_r[lx][ly] = (
                [sync, len(own), len(par)] + blk + [len(land)] + [v for j in land for v in self._noc(self.mm_cores[j])]
            )
        defs = row_barrier_defines()
        kd = lambda name, ct, rt, cfg, extra=(): ttnn.KernelDescriptor(
            kernel_source=self._srcs[name], source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=_crs(self.lanes), compile_time_args=ct, runtime_args=rt, config=cfg, defines=defs + list(extra))  # fmt: skip
        hdr = [("FS_HDR_CB", str(hdr_cb))] if hdr_cb is not None else []
        kernels = [kd("c1x_send.cpp", [self.tpb, self.nb, cb_stage, self.slice_t], rt_s, ttnn.WriterConfigDescriptor(), hdr),
                   kd("c1x_read.cpp", [self.tpb, self.nb, cb_stage, self.tpb // self.slice_t], rt_r,
                      ttnn.ReaderConfigDescriptor())]  # fmt: skip
        cbs = [ttnn.CBDescriptor(total_size=self.slots * self.nb * TB, core_ranges=_crs(self.lanes), format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=cb_stage, data_format=ttnn.bfloat16, page_size=TB)])]  # fmt: skip
        if hdr_cb is not None:  # one 4 KB page: up to 28 headers (NHDR 4 x 4 directions x 144 B = 2,304 B)
            assert hdr_cb != cb_stage, (hdr_cb, cb_stage)
            cbs.append(ttnn.CBDescriptor(total_size=4096, core_ranges=_crs(self.lanes), format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=hdr_cb, data_format=ttnn.uint32, page_size=4096)]))  # fmt: skip
        return kernels, cbs, [self.sync]

    def connect(self, prog, r: int, c: int, send_kernel_index: int):
        me = self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(r, c))
        steps = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
        for h, lane in enumerate(self.lanes):
            for (nr, nc), rng in zip(steps, self._ranges(r, c)):
                if rng:
                    peer = self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(nr, nc))
                    fab = fabric_connect(me, peer, h, prog, ttnn.CoreCoord(*lane))
                    prog.kernels[send_kernel_index].runtime_args[lane[0]][lane[1]].extend(fab)


class FabricStreamRSL:
    """Line RS of fp32 partials along axis 1 of an R x NC mesh (TP8 stage 2: the 4-chip row; prefill_megakernel.md §5
    C4 "then the row"): every chip's r1 [NC B2 32, H] fp32 (the pair-summed partials of its mesh row's blocks, column
    position c' owns rows [c' B2 32, (c' + 1) B2 32)) -> y [B2 32, H] fp32 on chip (r, c) = ((t0 + t1) + (t2 + t3)),
    t_c' = chip (r, c')'s r1 rows of c's blocks: the same association on every chip (chip / batch invariant).
    Lane h (blocks b mod 2 == h) sends every peer c' its blocks (fp32 tiles, one per 4-KiB packet, |c' - c| hops) into
    the peer's landing buffer r2 (row = this source's slot there x B2 + b) with a fused in-order +1 of the peer lane h's
    ARR + slot x 2 + h per block; 8 reducer cores (one owned block each at B2 = 8) poll those counters and add in fp32
    on the SFPU (kernels rsl_*.cpp, red_write.cpp). ready=True: the lanes wait R1RDY + i (the stage-1 reducers' +1)."""

    def __init__(self, mesh, b2: int, *, lanes, reducers, hidden: int = 4096, nb: int = 8, r1_dtype=None,
                 dst_order=None):  # fmt: skip
        self.mesh = mesh
        self.R, self.C = int(mesh.shape[0]), int(mesh.shape[1])
        self.nc = self.C
        # r1_dtype bf16 (GLM53_PMK_RS8_S2_BF16): r1 / the r2 landing / the SEND ring / the source slots in bf16 tiles
        # (FS_R1_BF16: 2 KB pages); the reducers unpack bf16 -> fp32 DST exactly and add in fp32, y stays fp32
        self.r1_dtype = r1_dtype if r1_dtype is not None else ttnn.float32
        # dst_order(c) = the destination positions in lane send order on position c (GLM53_PMK_RS8_ORDER); None =
        # ascending. A destination's blocks stay contiguous and in b order (the owners' landed counters count them)
        self.dst_order = dst_order
        assert self.nc == 4 and b2 % 2 == 0, (self.nc, b2)
        self.b2, self.hidden, self.tpb, self.nb = b2, hidden, hidden // 32, nb
        self.lanes = [tuple(c) for c in lanes]
        self.reducers = [tuple(c) for c in reducers]
        assert len(self.lanes) == 2 and len(self.reducers) == b2 and not set(self.lanes) & set(self.reducers)
        assert 3 * (b2 // 2) <= 12, "R1RDY words per lane"
        rep = ttnn.ReplicateTensorToMesh(mesh)
        words = SYNC_WORDS * WORD // 4
        sc = self.lanes + self.reducers
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                               ttnn.ShardSpec(_crs(sc), [1, words], ttnn.ShardOrientation.ROW_MAJOR))  # fmt: skip
        self.sync = ttnn.from_torch(torch.zeros(len(sc), words, dtype=torch.int32), dtype=ttnn.uint32,
                                    layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=mc, mesh_mapper=rep)  # fmt: skip
        z = lambda rows, dt=ttnn.float32: ttnn.from_torch(
            torch.zeros(1, 1, rows, hidden, dtype=torch.bfloat16 if dt == ttnn.bfloat16 else torch.float32),
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )  # noqa: E731  # fmt: skip
        self.r2 = z((self.nc - 1) * b2 * 32, self.r1_dtype)
        self.y = z(b2 * 32)
        self._srcs = {n: kernel_text(n) for n in ("rsl_send.cpp", "rsl_read.cpp", "rsl_red_read.cpp", "rsl_reduce.cpp",
                                                  "red_write.cpp")}  # fmt: skip
        self._noc = lambda xy: (lambda w: (int(w.x), int(w.y)))(mesh.worker_core_from_logical_core(ttnn.CoreCoord(*xy)))

    def _build(self, r1, r: int, c: int, ready: bool = False, fab_at: int | None = None):
        me, nc, b2, tpb, nb = c, self.nc, self.b2, self.tpb, self.nb
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        sync = self.sync.buffer_address()
        rt_s, rt_r = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for h, (lx, ly) in enumerate(self.lanes):
            sends, rows = [], []
            for dst in self.dsts(me):
                slot = me if me < dst else me - 1  # this source's slot at the destination
                for b in range(b2):
                    if b % 2 == h:
                        sends += [dst, slot * b2 + b, _FS["ARR"] + slot * 2 + h]
                        rows.append(dst * b2 + b)
            nl = self._noc((lx, ly))
            args = [sync, nl[0], nl[1], me, nc, self.r2.buffer_address(), len(rows)] + sends
            rt_s[lx][ly] = args + [len(args) + 1 if fab_at is None else fab_at]  # fab_at: where connect() appends
            rt_r[lx][ly] = (
                [sync, int(ready), r1.buffer_address(), len(rows)] + rows + [nc - 1, h]
            )  # + the per-hop drain
        rt_rr, rt_rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for b, (rx, ry) in enumerate(self.reducers):
            lane = self._noc(self.lanes[b % 2])
            rt_rr[rx][ry] = [sync, sync, b2 // 2, me, r1.buffer_address(), self.r2.buffer_address(), b2, int(ready), 1, b,
                             lane[0], lane[1]]  # fmt: skip
            rt_rw[rx][ry] = [self.y.buffer_address(), 1, b]
        CB_SEND, CB_T, CB_Y = 0, 1, 5
        cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                          fp32_dest_acc_en=True)  # fmt: skip
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for i in range(4):
            modes[CB_T + i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        defs = row_barrier_defines()
        bf = self.r1_dtype == ttnn.bfloat16
        r1d = [("FS_R1_BF16", "1")] if bf else []
        kd = lambda name, cores, ct, rt, cfg: ttnn.KernelDescriptor(
            kernel_source=self._srcs[name], source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=_crs(cores), compile_time_args=ct, runtime_args=rt, config=cfg,
            defines=(list(defs) if name in ("rsl_send.cpp", "rsl_read.cpp") else [])
            + (r1d if name in ("rsl_send.cpp", "rsl_read.cpp", "rsl_red_read.cpp") else []))  # fmt: skip
        kernels = [
            kd("rsl_send.cpp", self.lanes, [tpb, nb, CB_SEND] + acc(self.r2), rt_s, ttnn.WriterConfigDescriptor()),
            kd("rsl_read.cpp", self.lanes, [tpb, nb, CB_SEND] + acc(r1), rt_r, ttnn.ReaderConfigDescriptor()),
            kd("rsl_red_read.cpp", self.reducers, [tpb, nb, CB_T, nc] + acc(r1) + acc(self.r2), rt_rr,
               ttnn.ReaderConfigDescriptor()),
            kd("rsl_reduce.cpp", self.reducers, [tpb, nb, CB_T, CB_Y], ttnn.RuntimeArgs(), cc),
            kd("red_write.cpp", self.reducers, [tpb, nb, CB_Y] + acc(self.y), rt_rw, ttnn.WriterConfigDescriptor()),
        ]  # fmt: skip
        cb = lambda i, cores, dt=ttnn.float32: ttnn.CBDescriptor(
            total_size=(TB if dt == ttnn.bfloat16 else 4096) * 2 * nb,
            core_ranges=_crs(cores),
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=TB if dt == ttnn.bfloat16 else 4096)
            ],
        )  # noqa: E731  # fmt: skip
        cbs = ([cb(CB_SEND, self.lanes, self.r1_dtype)] + [cb(CB_T + i, self.reducers, self.r1_dtype) for i in range(4)]
               + [cb(CB_Y, self.reducers)])  # fmt: skip
        return kernels, cbs, []

    def dsts(self, me: int) -> list:
        """the other positions in this lane's send order (each destination's b2 / 2 blocks contiguous)"""
        o = list(self.dst_order(me)) if self.dst_order is not None else list(range(self.nc))
        assert sorted(o) == list(range(self.nc)), o
        return [d for d in o if d != me]

    def connect(self, prog, r: int, c: int, send_kernel_index: int = 0):
        me_id = self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(r, c))
        for h, lane in enumerate(self.lanes):
            for d, rng in ((-1, c), (1, self.nc - 1 - c)):  # dir 0 = toward lower positions, dir 1 = higher
                if rng:
                    peer = self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(r, c + d))
                    fab = fabric_connect(me_id, peer, h, prog, ttnn.CoreCoord(*lane))
                    prog.kernels[send_kernel_index].runtime_args[lane[0]][lane[1]].extend(fab)

    def __call__(self, r1):
        assert tuple(r1.shape)[-2:] == (self.nc * self.b2 * 32, self.hidden) and r1.dtype == self.r1_dtype, (
            r1.shape,
            r1.dtype,
        )
        mpd = ttnn.MeshProgramDescriptor()
        for r in range(self.R):
            for c in range(self.C):
                kernels, cbs, sems = self._build(r1, r, c)
                prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
                self.connect(prog, r, c, 0)
                m = ttnn.MeshCoordinate(r, c)
                mpd[ttnn.MeshCoordinateRange(m, m)] = prog
        ttnn.generic_op([r1, self.r2, self.y, self.sync], mpd)
        return self.y

    def release(self):
        for t in (self.r2, self.y, self.sync):
            ttnn.deallocate(t)


# GLM53_PMK_RS8_ORDER tables: mesh column c -> the destination columns in order, own last. 'alt' (rs8tl 2026-10-25,
# a fluid model of the row's links: 496 vs 809 us fp32, 248 vs 440 bf16 for the ascending order; the middle-link
# bound): the middle columns alternate direction, the ends start at the far end; mirror-symmetric (c <-> 3 - c)
RS8_ORDERS = {"": None, "alt": {0: (3, 2, 1, 0), 1: (3, 0, 2, 1), 2: (0, 3, 1, 2), 3: (0, 1, 2, 3)}}


class FabricStreamRS8:
    """The TP8 reduce-scatter on the 2x4 in ONE program (prefill_megakernel.md §5 C2 / C4): every chip's partial P
    [2048, 4096] bf16 -> y [256, 4096] fp32 on chip (r, c) = the sum over all 8 chips of its 8 blocks.
    Stage 1 (axis 0, FabricStreamRSR axis=0): the board pairs exchange the other row's 32 blocks; r1 = P_(0,c) + P_(1,c)
    of the row's blocks, exact fp32 (8 reducers). Stage 2 (axis 1, FabricStreamRSL): the row RS of the fp32 r1, owner
    sums ((t0 + t1) + (t2 + t3)) by column position (chip-invariant). Hand-off: each stage-1 reducer, after writing r1
    block (c', b), +1 on its stage-2 consumer: the stage-2 lane b mod 2 that sends it to c' (R1RDY + its send index) or,
    for c' = me, the stage-2 reducer b (R1RDY + 0). Cores: lanes1, reducers1 (8), lanes2, reducers2 (8)."""

    def __init__(self, mesh, *, lanes1, reducers1, lanes2, reducers2, hidden: int = 4096, nb1: int = 16, nb2: int = 8,
                 s2_bf16: bool = False, order: str = ""):  # fmt: skip
        R, C = int(mesh.shape[0]), int(mesh.shape[1])
        assert (R, C) == (2, 4), (R, C)
        self.mesh, self.R, self.C = mesh, R, C
        self.b2 = 8
        # s2_bf16 (GLM53_PMK_RS8_S2_BF16; rs8tl 2026-10-25: stage 2 is the RS tail, the row's middle link carries 8 MiB
        # fp32 per lane per direction): r1 = the exact fp32 pair sum packed to bf16, stage 2 moves half the bytes
        dt = ttnn.bfloat16 if s2_bf16 else ttnn.float32
        self.s2_bf16 = bool(s2_bf16)
        # order (GLM53_PMK_RS8_ORDER; rs8tl: stage 2 ends in column order because every chip reduces / sends column
        # 0's blocks first): RS8_ORDERS[order][c] = the destination columns in production + send order on column c,
        # own last. Bitwise: only the order of independent block reductions / sends moves (the owner sums keep
        # ((t0 + t1) + (t2 + t3)))
        self.order_name = order or ""
        tab = RS8_ORDERS[self.order_name]
        self.dst_order = (lambda c_: list(tab[c_])) if tab is not None else (lambda c_: list(range(C)))
        b2 = self.b2
        blk = (
            (lambda c_: sorted(range(C * b2), key=lambda j: (self.dst_order(c_).index(j // b2), j % b2)))
            if tab
            else None
        )
        self.rs1 = FabricStreamRSR(mesh, C * self.b2, lanes=lanes1, hidden=hidden, nb=nb1, reducers=reducers1, axis=0,
                                   out_dtype=dt, block_order=blk)  # fmt: skip
        self.rs2 = FabricStreamRSL(mesh, self.b2, lanes=lanes2, reducers=reducers2, hidden=hidden, nb=nb2, r1_dtype=dt,
                                   dst_order=self.dst_order if tab else None)  # fmt: skip
        self.y = self.rs2.y
        self._noc = self.rs2._noc

    def signals(self, c: int) -> dict:
        """stage-1 reducer k (blocks j = k, k + 8, .. of the row's 32: column position c' = j // 8, b = j % 8 = k) ->
        its stage-2 consumers in its list order"""
        sync2 = self.rs2.sync.buffer_address()
        nr1 = len(self.rs1.reducers)
        out = {}
        for k, core in enumerate(self.rs1.reducers):
            sig = []
            for j in self.rs1.red_list(c, k):  # the reducer's list order (red_write signals block by block)
                cp, b = divmod(j, self.b2)
                if cp == c:
                    x, y = self._noc(self.rs2.reducers[b])
                    sig.append((x, y, sync2 + _FS["R1RDY"] * WORD))
                else:
                    dpos = self.rs2.dsts(c).index(cp)
                    i = dpos * (self.b2 // 2) + b // 2  # the send index in stage-2 lane (b mod 2)'s list
                    x, y = self._noc(self.rs2.lanes[b % 2])
                    sig.append((x, y, sync2 + (_FS["R1RDY"] + i) * WORD))
            out[core] = sig
        return out

    def parts(self, pt, r: int, c: int, ready: bool = False):
        """(kernels, cbs, semaphores, n1) of chip (r, c); kernels[0] = stage 1's rs_send, kernels[n1] = stage 2's
        rsl_send (connect(prog, r, c, offset) on the FINAL program). ready: stage 1 waits the producers' per-block
        words (H5: lane LREADY / reducer OWNRDY, tt/pf_rs8.PfRs8)"""
        k1, cb1, s1, _ = self.rs1._build(pt, r, c, ready=ready, red_signals=self.signals(c))
        k2, cb2, s2 = self.rs2._build(self.rs1.y, r, c, ready=True)
        return list(k1) + list(k2), list(cb1) + list(cb2), list(s1) + list(s2), len(k1)

    def connect(self, prog, r: int, c: int, n1: int, offset: int = 0):
        self.rs1.connect(prog, r, c, offset)  # stage 1 rs_send (axis 0)
        self.rs2.connect(prog, r, c, offset + n1)  # stage 2 rsl_send (axis 1)

    def program(self, pt, r: int, c: int, ready: bool = False) -> ttnn.ProgramDescriptor:
        kernels, cbs, sems, n1 = self.parts(pt, r, c, ready)
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        self.connect(prog, r, c, n1)
        return prog

    def __call__(self, pt):
        assert tuple(pt.shape)[-2:] == (2 * self.C * self.b2 * 32, self.rs1.hidden) and pt.dtype == ttnn.bfloat16
        mpd = ttnn.MeshProgramDescriptor()
        for r in range(self.R):
            for c in range(self.C):
                m = ttnn.MeshCoordinate(r, c)
                mpd[ttnn.MeshCoordinateRange(m, m)] = self.program(pt, r, c)
        ttnn.generic_op([pt, self.rs1.r, self.rs1.y, self.rs1.sync, self.rs2.r2, self.rs2.y, self.rs2.sync], mpd)
        return self.y

    def release(self):
        self.rs1.release()
        self.rs2.release()
