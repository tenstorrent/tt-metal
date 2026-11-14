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
    """A link channel at `base_pa` (x280 physical address, coherent Memory Port alias; 64 KiB aligned is advised).
    `channel_tensor`: the ttnn buffer that owns the memory (kept alive; also used as generic_op's first io tensor)."""

    def __init__(self, device, channel_tensor, base_pa, core=(0, 0)):
        self.dev, self.tensor, self.base, self.core = device, channel_tensor, base_pa, core
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
        return program("l2cpu_notify.cpp", [self.core], [[*L2CPU_XY, *_hl(self.base), int(diag)]], 2048)

    def wait_program(self, out=None, n_words=0, timeout_us=WAIT_TIMEOUT_US, diag=False):
        """timeout_us: wall-clock bound of the wait (<= 3 s); at the bound the kernel writes
        WAIT_STATUS_TIMEOUT | (req & 0xFFFF) to LINK_OFF_WAIT_STATUS and returns (read_wait_status())."""
        assert 0 < timeout_us <= 3_000_000
        ox, oy, oa = self.page0(out) if out is not None else (0, 0, 0)
        args = [*L2CPU_XY, *_hl(self.base), timeout_us, n_words, ox, oy, oa, int(diag)]
        return program("l2cpu_wait.cpp", [self.core], [args], 2048)

    def stress_program(self, rounds, timeout_us=WAIT_TIMEOUT_US):
        return program("l2cpu_link_stress.cpp", [self.core], [[*L2CPU_XY, *_hl(self.base), timeout_us, rounds]], 2048)

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
            [*L2CPU_XY, *_hl(self.base), *_hl(dst), n_rows, j, ncores, int(streamed), groups, group_rows, int(diag)]
            for j in range(ncores)
        ]
        return program("l2cpu_push.cpp", cores, args, 2 * ((row_bytes + 63) & ~63) + 2048)
