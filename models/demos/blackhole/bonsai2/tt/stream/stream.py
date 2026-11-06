# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode weight stream.

Sender cores (PER_BANK per DRAM bank, next to it; their own sub-device) read their part of the step's weights front to
back and hand it, SLOT bytes at a time, to their receivers' persistent L1 rings; the decode ops run on the receivers
(the other sub-device) and take their weights from the ring instead of DRAM. The senders run for the whole step, so
the weight reads continue through every other op and every op-to-op gap.

- DRAM: one interleaved tensor with SLOT-byte pages; page b + 8 i is bank b's slot i, so a bank's slots are one
  contiguous region of the bank; sender k of bank b reads the k-th of PER_BANK equal parts of it, in order.
- Order: per linear (Plan.add), each sender hands its receivers their pages round-robin over the receivers that still
  have slots in that linear (the schedule, one receiver index per slot). Every receiver's pages for a linear are padded
  to whole slots.
- Flow control: counters only grow, across ops and steps. Receiver: filled (bumped by its sender after each slot) and
  consumed (its own); a slot index is count % RING_SLOTS. Sender: freed[r] (bumped by receiver r per slot taken) and
  sent[r].
- Receiver layout: kernels find their place from a per-core info record (receiver index, sender NOC x / y, index in
  the sender's group) at a fixed L1 address (define RX_INFO, added to every Bonsai kernel by install()); the ops place
  their cores with core_ranges' layout (receiver i -> rx_order[i]), whose first n receivers are a few rectangles.
"""

import os

import numpy as np
import torch

import ttnn

KERNELS = "models/demos/blackhole/bonsai2/tt/stream/kernels/"
SLOT = 16384
PAGE = 4096
PPS = SLOT // PAGE  # 4-KB pages per slot
RING_SLOTS = 16
STAGE = 12
BATCH = 4
NBANKS = 8
PER_BANK = 2
SCHED_CHUNK = 256
SEG_SLOTS = int(
    os.environ.get("STREAM_SEG_SLOTS", "256")
)  # slots per sender per segment (8 MB per bank: DRAM is fragmented)
PAIR = os.environ.get("STREAM_PAIR", "horizontal")  # schedule entries the writer reads at a time
_DEFS = [(d, "1") for d in os.environ.get("STREAM_DEFINES", "").split(",") if d]


def _crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _dm(reader: bool):
    if os.environ.get("STREAM_SWAP") == "1":  # DRAM reads on BRISC / NOC 0, ring writes on NCRISC / NOC 1
        reader = not reader
    if os.environ.get("STREAM_STOCK_CFG", "1") == "1" and ("SEND_2NOC", "1") not in _DEFS:
        return ttnn.ReaderConfigDescriptor() if reader else ttnn.WriterConfigDescriptor()
    d = ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_1 if reader else ttnn.DataMovementProcessor.RISCV_0,
        noc=ttnn.NOC.NOC_1 if reader else ttnn.NOC.NOC_0,
    )
    if ("SEND_2NOC", "1") in _DEFS:
        d.noc_mode = ttnn.NOC_MODE.DM_DYNAMIC_NOC
    return d


class Stream:
    def __init__(self, device, slot=SLOT, ring_slots=RING_SLOTS):
        self.device, self.slot, self.S = device, slot, ring_slots
        grid = device.compute_with_storage_grid_size()
        opt = ttnn.get_optimal_dram_bank_to_logical_worker_assignment(device, ttnn.NOC.NOC_0)
        self.senders, self.bank_of, self.part_of = [], [], []
        taken = {(c.x, c.y) for c in opt}
        for b, c in enumerate(opt):
            for k in range(PER_BANK):
                if k == 0:
                    xy = (c.x, c.y)
                elif PAIR == "vertical":  # the bank core and a free core above / below it: their writes use other rows
                    xy = next(
                        (c.x, y)
                        for y in (c.y + 1, c.y - 1, c.y + 2, c.y - 2)
                        if 0 <= y < grid.y and (c.x, y) not in taken
                    )
                    taken.add(xy)
                elif PAIR == "diag":  # the next column, an adjacent row: the senders spread over more rows
                    x = c.x + 1
                    xy = next(
                        (x, y) for y in (c.y + 1, c.y - 1, c.y + 2, c.y - 2) if 0 <= y < grid.y and (x, y) not in taken
                    )
                    taken.add(xy)
                else:
                    xy = (c.x + k if c.x + k < grid.x else c.x - k, c.y)
                self.senders.append(ttnn.CoreCoord(*xy))
                self.bank_of.append(b)
                self.part_of.append(k)
        NS = len(self.senders)
        sset = {(c.x, c.y) for c in self.senders}
        scols = {c.x for c in self.senders}
        # receiver order: whole free columns first (blocks of adjacent columns, row-major in each), then the cells
        # left in the sender columns; the first n receivers are then a few rectangles
        blocks, cur = [], []
        for x in range(grid.x):
            if x in scols:
                if cur:
                    blocks.append(cur)
                cur = []
            else:
                cur.append(x)
        if cur:
            blocks.append(cur)
        order = []
        for blk in blocks:
            order += [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in blk]
        left = [ttnn.CoreCoord(x, y) for x in sorted(scols) for y in range(grid.y) if (x, y) not in sset]
        order += sorted(left, key=lambda c: (c.x // 2, c.y, c.x))
        per = len(order) // NS  # receivers per sender (equal: the busiest sender sets the pace)
        self.rx_order = order[: per * NS]
        self.idle = order[per * NS :]
        # receiver i to sender i % NS: every prefix of the receivers (a linear uses its first n) spreads evenly over the
        # senders (nearest-sender groups left some senders with no active receiver in a linear: 217 vs 395 GB/s)
        if os.environ.get("STREAM_GROUPS", "near") == "rr":
            self.groups = [self.rx_order[b::NS] for b in range(NS)]
        else:  # nearest sender with room, receivers taken in row-major order (compact clusters around each sender)
            self.groups = [[] for _ in range(NS)]
            for c in sorted(self.rx_order, key=lambda c: (c.y, c.x)):
                near = sorted(range(NS), key=lambda b: abs(self.senders[b].x - c.x) + abs(self.senders[b].y - c.y))
                b = next(b for b in near if len(self.groups[b]) < per)
                self.groups[b].append(c)
        self.per = per
        self.sender_crs = _crs(self.senders)
        self.receiver_crs = _crs(self.rx_order + self.idle)
        self.manager = device.create_sub_device_manager(
            [ttnn.SubDevice([self.sender_crs]), ttnn.SubDevice([self.receiver_crs])], 0
        )
        device.load_sub_device_manager(self.manager)
        device.set_sub_device_stall_group([ttnn.SubDeviceId(1)])

        def sharded(crs_cores, rows):
            crs = _crs(crs_cores)
            n = len(crs_cores)
            init = torch.zeros(n, rows.shape[1] if hasattr(rows, "shape") else rows, dtype=torch.int32)
            if hasattr(rows, "shape"):
                init = torch.from_numpy(rows.astype(np.int32))
            mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(crs, [1, init.shape[1]], ttnn.ShardOrientation.ROW_MAJOR),
            )
            return ttnn.from_torch(
                init, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc
            )

        cores = self.rx_order + self.idle
        # shards follow the crs' core order (row-major over its ranges): build the per-core rows in that order
        shard_order = ttnn.corerange_to_cores(self.receiver_crs, row_wise=True)
        self.rx_index = {(c.x, c.y): i for i, c in enumerate(self.rx_order)}
        where = {}
        for b, g in enumerate(self.groups):
            sv = device.worker_core_from_logical_core(self.senders[b])
            for r, c in enumerate(g):
                where[(c.x, c.y)] = (sv.x, sv.y, r)
        self.info = sharded(shard_order, 16)
        # each core writes its own record (not via the shard order of a host tensor)
        rt = ttnn.RuntimeArgs()
        for c in shard_order:
            k = (c.x, c.y)
            sx, sy, r = where.get(k, (0, 0, 0))
            rt[c.x][c.y] = [self.info.buffer_address(), self.rx_index.get(k, 0xFFFF), sx, sy, r]
        ttnn.generic_op(
            [self.info, self.info],
            ttnn.ProgramDescriptor(
                kernels=[
                    ttnn.KernelDescriptor(
                        kernel_source=KERNELS + "rx_init.cpp",
                        core_ranges=self.receiver_crs,
                        compile_time_args=[],
                        runtime_args=rt,
                        config=ttnn.ReaderConfigDescriptor(),
                    )
                ],
                semaphores=[],
                cbs=[],
            ),
        )
        self.ring = sharded(shard_order, ring_slots * slot // 4)
        self.rcount = sharded(shard_order, 16)  # filled at +0, consumed at +32
        self.scount = sharded(self.senders, 2 * 4 * per)  # freed[r] at 16 r, sent[r] at 16 (per + r)

    def deactivate(self):
        """Back to the default sub-device manager (e.g. for prefill, whose ops use the whole grid)."""
        self.device.reset_sub_device_stall_group()
        self.device.clear_loaded_sub_device_manager()

    def activate(self):
        self.device.load_sub_device_manager(self.manager)
        self.device.set_sub_device_stall_group([ttnn.SubDeviceId(1)])

    def close(self):
        self.device.reset_sub_device_stall_group()
        self.device.clear_loaded_sub_device_manager()
        self.device.remove_sub_device_manager(self.manager)

    # ------------------------------------------------------------------ consumer side
    def consumer_args(self):
        """Common args a consumer kernel needs (ring, filled, consumed, freed array base; the rest is in RX_INFO)."""
        return [
            self.ring.buffer_address(),
            self.rcount.buffer_address(),
            self.rcount.buffer_address() + 32,
            self.scount.buffer_address(),
        ]

    def defines(self):
        return [
            ("RX_INFO", str(self.info.buffer_address())),
            ("RX_RING_SLOTS", str(self.S)),
            ("RX_SLOT", str(self.slot)),
        ]

    # ------------------------------------------------------------------ sender side
    def launch(self, plan, start=0):
        """Senders stream `plan` (Plan.build: its DRAM tensor and schedules); returns at once."""
        dev = self.device
        kernels = []
        for R in sorted({len(g) for g in self.groups}):
            idx = [b for b in range(len(self.senders)) if len(self.groups[b]) == R]
            crs = _crs([self.senders[b] for b in idx])
            rd, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for b in idx:
                s = self.senders[b]
                n = plan.slots[b]
                off = self.part_of[b] * plan.seg_slots * self.slot
                rd[s.x][s.y] = [self.bank_of[b], off, n, plan.seg_slots] + [w.buffer_address() for w in plan.w]
                xy = []
                for c in self.groups[b]:
                    v = dev.worker_core_from_logical_core(c)
                    xy += [v.x, v.y]
                wr[s.x][s.y] = [
                    n,
                    self.ring.buffer_address(),
                    self.rcount.buffer_address(),
                    self.scount.buffer_address(),
                    self.scount.buffer_address() + 16 * self.per,
                    plan.sched.buffer_address(),
                    b * plan.sched_stride,
                ] + xy
            kernels += [
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "send_read.cpp",
                    core_ranges=crs,
                    compile_time_args=[self.slot, STAGE, STAGE - BATCH],
                    runtime_args=rd,
                    defines=_DEFS,
                    config=_dm(True),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "send_write.cpp",
                    core_ranges=crs,
                    compile_time_args=[self.slot, self.S, R, BATCH, SCHED_CHUNK, STAGE]
                    + ttnn.TensorAccessorArgs(plan.sched).get_compile_time_args(),
                    runtime_args=wr,
                    defines=_DEFS,
                    config=_dm(False),
                ),
            ]
        cbs = [
            ttnn.CBDescriptor(
                total_size=STAGE * self.slot,
                core_ranges=self.sender_crs,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=self.slot)
                ],
            ),
            ttnn.CBDescriptor(
                total_size=SCHED_CHUNK * 4,
                core_ranges=self.sender_crs,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=1, data_format=ttnn.uint32, page_size=SCHED_CHUNK * 4)
                ],
            ),
        ]
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        return ttnn.generic_op(list(plan.w) + [plan.sched, self.ring, self.rcount, self.scount], prog)


def drain(st, taken):
    """Test consumer: every receiver takes its slots in `taken` (Plan.add's result) and gives them back."""
    counts = [taken.get(i, 0) for i in range(len(st.rx_order))]
    crs = _crs(st.rx_order)
    prog = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=KERNELS + "drain.cpp",
                core_ranges=crs,
                compile_time_args=[],
                common_runtime_args=st.consumer_args() + counts,
                defines=st.defines(),
                config=ttnn.ReaderConfigDescriptor(),
            )
        ],
        semaphores=[],
        cbs=[],
    )
    return ttnn.generic_op([st.ring, st.rcount, st.scount], prog)


class Plan:
    """The step's weight stream, built linear by linear: add(pages) takes each receiver's 4-KB pages for one linear
    (dict receiver index -> [n, 1024] uint32) and returns the slots each receiver takes."""

    def __init__(self, stream: Stream):
        self.st = stream
        ns = len(stream.senders)
        self.data = [[] for _ in range(ns)]  # per sender: list of [PPS, 1024] slot arrays
        self.who = [[] for _ in range(ns)]  # per sender: receiver index in its group, per slot

    def add(self, pages):
        st = self.st
        taken = {}
        for b, g in enumerate(st.groups):
            q = []
            for c in g:
                i = st.rx_index[(c.x, c.y)]
                p = pages.get(i)
                n = 0 if p is None else p.shape[0]
                ns = -(-n // PPS)
                if ns:
                    full = np.zeros((ns * PPS, 1024), np.uint32)
                    full[:n] = p
                    q.append([full[k * PPS : (k + 1) * PPS] for k in range(ns)])
                else:
                    q.append([])
                taken[i] = ns
            # round-robin over the receivers with slots left
            k = 0
            while any(q):
                for r in range(len(g)):
                    if k < len(q[r]):
                        self.data[b].append(q[r][k])
                        self.who[b].append(r)
                k += 1
                q = [x if k < len(x) else [] for x in q]
        return taken

    def build(self):
        st, dev = self.st, self.st.device
        ns = len(st.senders)
        self.slots = [len(d) for d in self.data]
        self.region = max(self.slots)
        # segments of seg_slots slots per sender: bank b's segment holds sender k's slots at pages b + 8 (k seg_slots + j)
        self.seg_slots = min(SEG_SLOTS, self.region)
        nseg = -(-self.region // self.seg_slots)
        self.w = []
        for g in range(nseg):
            host = np.zeros((NBANKS * PER_BANK * self.seg_slots, SLOT // 4), np.uint32)
            for b in range(ns):
                bank, part = st.bank_of[b], st.part_of[b]
                for j, a in enumerate(self.data[b][g * self.seg_slots : (g + 1) * self.seg_slots]):
                    host[bank + NBANKS * (part * self.seg_slots + j)] = a.reshape(-1)
            self.w.append(
                ttnn.from_torch(
                    torch.from_numpy(host.view(np.int32)),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=dev,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
            del host
        stride = -(-self.region // SCHED_CHUNK) * SCHED_CHUNK
        sched = np.zeros((ns * stride // SCHED_CHUNK, SCHED_CHUNK), np.int32)
        flat = sched.reshape(-1)
        for b in range(ns):
            flat[b * stride : b * stride + len(self.who[b])] = self.who[b]
        self.sched = ttnn.from_torch(
            torch.from_numpy(sched),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.sched_stride = stride // SCHED_CHUNK  # schedule pages per sender
        return self
