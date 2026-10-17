# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
#
# SPDX-License-Identifier: Apache-2.0
"""The device loop's ops on the sampling firmware's arena, built on the Tensix link component (tools/l2cpu/tensix:
kernels + ops.Link). The arena embeds the link block (l2cpu_link.h: req_seq, done_seq, wait_status, landed, diag,
push source table) at a fixed offset; every link program gets `arena + that offset` as its channel base, and the
logits zones are passed as offsets from it.

    ops = L2cpuOps(device, arena_tensor)
    ops.set_push_source(logits_rm)                         # host, once (after READY, outside a capture)
    ops.run(ops.push_program(batch, uncached=B > 1))        # rows -> logits zone, acked
    ops.run(ops.notify_program()); ops.run(ops.wait_program(tokens, 1, batch), out=tokens)
    ops.run(ops.push_stream_program(batch, uncached=True))  # streamed: doorbell first, landed count per row
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import deps  # noqa: E402,F401  (bindings to the other l2cpu components, see deps.py)

from tensix import ops as link_ops  # noqa: E402  (Tensix <-> L2CPU link component)
from l2cpu.bringup import region_base_pa  # noqa: E402
from l2cpu.sampling import layout as A  # noqa: E402

LINK_BLOCK_OFF = A.L2S_OFF_LINK  # the sampling region embeds the link block (l2cpu_link.h) here
LOGITS_OFF = A.L2S_OFF_LOGITS  # coherent zone (batch 1)
LOGITS_UC_OFF = A.L2S_OFF_LOGITS_UC  # uncached zone (batch > 1): written and read uncached only
ROW = 303_872
# The firmware's consumer order for streamed rows: 4 harts own 8 rows each (users u = 8h .. 8h+7).
STREAM_GROUPS, STREAM_GROUP_ROWS = 4, 8


def arena_base_pa(buffer_address: int) -> int:
    """x280 PA of the firmware region (arena) for the ttnn buffer: the bring-up component's region base."""
    return region_base_pa(buffer_address)


class L2cpuOps:
    def __init__(self, device, arena_tensor, core=(0, 0)):
        self.dev = device
        self.arena = arena_tensor
        self.arena_pa = arena_base_pa(arena_tensor.buffer_address())
        self.link = link_ops.Link(device, arena_tensor, self.arena_pa + LINK_BLOCK_OFF, core)
        self.hw = None  # host access (set by the caller) for set_push_source

    def _zone(self, uncached):
        return (LOGITS_UC_OFF if uncached else LOGITS_OFF) - LINK_BLOCK_OFF

    def scratch(self):
        return self.link.scratch()

    def run(self, prog, out=None):
        return self.link.run(prog, out=out)

    def set_push_source(self, logits, row_bytes=ROW, hw=None):
        (hw or self.hw).pa_write(
            self.link.base + link_ops.LINK_OFF_PUSH_SRC, self.link.push_source_table(logits, row_bytes)
        )

    def push_program(self, n_rows, ncores=1, row_bytes=ROW, uncached=False):
        return self.link.push_program(n_rows, row_bytes, self._zone(uncached), uncached=uncached, ncores=ncores)

    def push_stream_program(self, n_rows, uncached=False, row_bytes=ROW, diag=False):
        return self.link.push_program(
            n_rows,
            row_bytes,
            self._zone(uncached),
            uncached=uncached,
            streamed=True,
            groups=STREAM_GROUPS,
            group_rows=STREAM_GROUP_ROWS,
            diag=diag,
        )

    def notify_program(self, diag=False):
        return self.link.notify_program(diag=diag)

    def wait_program(self, tokens, variant=1, batch=1, timeout_us=link_ops.WAIT_TIMEOUT_US, diag=False):
        """variant 1: the x280 writes the token tensor itself (FLAG_TOKENS_REMOTE); the wait only waits."""
        assert variant == 1, "only the in-place token variant is used by this example"
        return self.link.wait_program(timeout_us=timeout_us, diag=diag)

    # offsets of link words, for the host loop (absolute PAs)
    def pa(self, link_off):
        return self.link.base + link_off
