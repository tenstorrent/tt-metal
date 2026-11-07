# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.generic_op program builders for the Tensix <-> L2CPU link (kernels in ./kernels, layout in
kernels/l2cpu_link.h, mirrored below).

All runtime arguments are constant per program instance, so the programs are trace-safe: per-step state lives in
the channel (req_seq / done_seq / landed), never in runtime args (a trace freezes them).

generic_op's program cache hashes the descriptor, including runtime-argument COUNTS but not VALUES. Two instances
of the same kernel with different argument values would therefore share one cached program and the second would
run with the first one's arguments. Every builder here adds a define L2CPU_ARGS_ID = hash(kernel, cores, args):
distinct argument sets get distinct programs (one JIT build each); identical ones share the cache.
"""
from __future__ import annotations

import hashlib
import os
import struct

import ttnn

from l2cpu.hw import L2CPU_TILES

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")

# kernels/l2cpu_link.h
LINK_OFF_REQ_SEQ = 0x000
LINK_OFF_DONE_SEQ = 0x040
LINK_OFF_WAIT_STATUS = 0x080
LINK_OFF_LANDED = 0x0C0
LINK_OFF_DIAG = 0x100
LINK_OFF_PUSH_SRC = 0x180
LINK_OFF_REPLY = 0x300
LINK_OFF_STOP = 0x340
LINK_OFF_RESP_STATUS = 0x380
LINK_SIZE = 0x1000
LINK_RESP_MAGIC = 0x4C4E4B31
LINK_IMAGE_OFFSET = 0x10000  # test responder image: channel base + this offset

WAIT_TIMEOUT_US = 50_000  # default bound of a wait (serving: ~2 decode steps of an 8B model)
WAIT_STATUS_TIMEOUT = 0xDEAD0000

L2CPU_XY = L2CPU_TILES[0]  # CPUs 0-3; identical in raw and translated coordinates on Blackhole
UNCACHED_DELTA = 0x4000_0000_0000  # Memory Port alias - this = System Port alias (uncached, not coherent)


def _grid(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])


def program(kernel, cores, args_per_core, cb_bytes):
    """One data-movement kernel (RISCV_0 reader config) on `cores`, CB 0 of cb_bytes as L1 scratch."""
    grid = _grid(cores)
    rt = ttnn.RuntimeArgs()
    for (x, y), args in zip(cores, args_per_core):
        rt[x][y] = [int(a) & 0xFFFFFFFF for a in args]
    ident = hashlib.sha1(repr((kernel, cores, args_per_core, cb_bytes)).encode()).hexdigest()[:12]
    k = ttnn.KernelDescriptor(
        kernel_source=os.path.join(KDIR, kernel),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[],
        defines=[("L2CPU_ARGS_ID", "0x" + ident)],
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    cb = ttnn.CBDescriptor(
        total_size=cb_bytes,
        core_ranges=grid,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=cb_bytes)],
    )
    return ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb])


def _hl(v):
    return [v >> 32, v & 0xFFFFFFFF]


class Link:
    """A link channel at `base_pa` (x280 physical address, coherent Memory Port alias; 64 KiB aligned is advised) of
    the L2CPU tile at NoC `xy` (default tile 0, (8,3); L2CPU_TILES[t] for tile t; base_pa then lies in that tile's
    local DRAM). `channel_tensor`: the ttnn buffer that owns the memory (kept alive; generic_op's first io tensor)."""

    def __init__(self, device, channel_tensor, base_pa, core=(0, 0), xy=L2CPU_XY):
        self.dev, self.tensor, self.base, self.core, self.xy = device, channel_tensor, base_pa, core, tuple(xy)
        self.banks = ttnn.cluster.get_dram_bank_table(0)
        self._scratch = None

    def scratch(self):
        """Preallocated dummy output (generic_op needs >= 2 io tensors; the last one is the output)."""
        if self._scratch is None:
            import torch

            self._scratch = ttnn.from_torch(
                torch.zeros(1, 1, 1, 32, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return self._scratch

    def run(self, prog, out=None):
        return ttnn.generic_op([self.tensor, out if out is not None else self.scratch()], prog)

    def page0(self, tensor):
        b = self.banks[0]
        return b["noc_x"], b["noc_y"], tensor.buffer_address()

    def notify_program(self, diag=False):
        return program("l2cpu_notify.cpp", [self.core], [[*self.xy, *_hl(self.base), int(diag)]], 2048)

    def wait_program(self, out=None, n_words=0, timeout_us=WAIT_TIMEOUT_US, diag=False):
        """timeout_us: wall-clock bound of the wait (<= 3 s); at the bound the kernel writes
        WAIT_STATUS_TIMEOUT | (req & 0xFFFF) to LINK_OFF_WAIT_STATUS and returns (read_wait_status())."""
        assert 0 < timeout_us <= 3_000_000
        ox, oy, oa = self.page0(out) if out is not None else (0, 0, 0)
        args = [*self.xy, *_hl(self.base), timeout_us, n_words, ox, oy, oa, int(diag)]
        return program("l2cpu_wait.cpp", [self.core], [args], 2048)

    def stress_program(self, rounds, timeout_us=WAIT_TIMEOUT_US):
        return program("l2cpu_link_stress.cpp", [self.core], [[*self.xy, *_hl(self.base), timeout_us, rounds]], 2048)

    def push_source_table(self, tensor, row_bytes):
        """Bytes to write at base + LINK_OFF_PUSH_SRC (host, outside any trace capture) for push_program()."""
        stride = (row_bytes + 63) & ~63
        tab = struct.pack("<4I", row_bytes, stride, 8, 0)
        for b in range(8):
            e = self.banks[b]
            tab += struct.pack("<IIQ", e["noc_x"], e["noc_y"], e["base_addr"] + tensor.buffer_address())
        return tab

    def push_program(
        self, n_rows, row_bytes, dst_off, uncached=False, ncores=1, streamed=False, groups=1, group_rows=0, diag=False
    ):
        """Rows 0..n_rows-1 (<= 64) -> channel base + dst_off (+ b * row_bytes).
        uncached: write the System Port alias (the zone must then only ever be read uncached and never written
        through the coherent alias). streamed: doorbell first, then rows in the order
        (p % groups) * group_rows + p / groups with a `landed` count after each (1 core)."""
        dst = self.base + dst_off - (UNCACHED_DELTA if uncached else 0)
        ncores = 1 if streamed else ncores
        cores = [(i % 8, i // 8) for i in range(ncores)]
        args = [
            [*self.xy, *_hl(self.base), *_hl(dst), n_rows, j, ncores, int(streamed), groups, group_rows, int(diag), 0]
            for j in range(ncores)
        ]
        return program("l2cpu_push.cpp", cores, args, 2 * ((row_bytes + 63) & ~63) + 2048)

    def push_args(
        self, n_rows, dst_off, uncached=False, streamed=False, groups=1, group_rows=0, diag=False, src_row0=0
    ):
        """Runtime args of one push core writing rows src_row0 .. src_row0 + n_rows - 1 of the source to this channel
        (local rows 0 .. n_rows - 1 at base + dst_off); see push_program and push_split_program."""
        dst = self.base + dst_off - (UNCACHED_DELTA if uncached else 0)
        return [
            *self.xy,
            *_hl(self.base),
            *_hl(dst),
            n_rows,
            0,
            1,
            int(streamed),
            groups,
            group_rows,
            int(diag),
            src_row0,
        ]


# ---- several channels (one per L2CPU tile: a batch split over tiles) -------------------------------------------------
def _cores(n, first=(0, 0)):
    return [((first[0] + i) % 8, first[1] + (first[0] + i) // 8) for i in range(n)]


def push_split_program(
    links, blocks, row_bytes, dst_off, uncached=False, streamed=False, groups=1, group_rows=None, diag=False
):
    """One program, one core per link: core i pushes source rows blocks[i] = (src_row0, n_rows) into links[i]
    (local rows 0 .. n_rows - 1 at base + dst_off). The source table must be written into EVERY link
    (push_source_table). streamed: each core rings its own tile first (req_seq + doorbell), then lands its rows in
    the order (p % groups) * group_rows[i] + p / groups with a `landed` count each. The cores run concurrently: the
    tiles' inbound ports work in parallel (measured: 32 rows to one tile 0.50 ms, 2 x 16 rows 0.32 ms, 4 x 8 rows
    0.40 ms, where the DRAM / NoC source side limits)."""
    assert len(links) == len(blocks) and 1 <= len(links) <= 8
    group_rows = group_rows or [0] * len(links)
    cores = _cores(len(links))
    args = [
        L.push_args(
            n, dst_off, uncached=uncached, streamed=streamed, groups=groups, group_rows=gr, diag=diag, src_row0=r0
        )
        for L, (r0, n), gr in zip(links, blocks, group_rows)
    ]
    return program("l2cpu_push.cpp", cores, args, 2 * ((row_bytes + 63) & ~63) + 2048)


def notify_all_program(links, diag=False):
    """One program, one core per link: each core publishes req_seq + 1 and rings its own tile's doorbell."""
    return program("l2cpu_notify.cpp", _cores(len(links)), [[*L.xy, *_hl(L.base), int(diag)] for L in links], 2048)


def wait_all_program(links, timeout_us=WAIT_TIMEOUT_US, diag=False, core=(0, 0)):
    """One core polls every link's done_seq under one bound (kernels/l2cpu_wait_all.cpp); at the bound each link
    that is not done gets its own wait status word (read_wait_status per link)."""
    assert 0 < timeout_us <= 3_000_000 and 1 <= len(links) <= 8
    args = [len(links), timeout_us, int(diag)]
    for L in links:
        args += [*L.xy, *_hl(L.base)]
    return program("l2cpu_wait_all.cpp", [core], [args], 2048)
