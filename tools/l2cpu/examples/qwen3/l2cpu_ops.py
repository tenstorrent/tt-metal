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

    mops = L2cpuMultiOps(device, fws, regions, batch)        # a batch split over several L2CPU tiles
    mops.run(mops.push_program(uncached=True, streamed=True)); mops.run(mops.wait_program())
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
# The firmware's consumer order for streamed rows: 4 harts own uph = ceil(batch / 4) rows each (users uph*h ..
# uph*h + uph - 1; batch 32: 8).
STREAM_GROUPS = 4


def stream_group_rows(batch):
    return (batch + STREAM_GROUPS - 1) // STREAM_GROUPS


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
            group_rows=stream_group_rows(n_rows),
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


class L2cpuMultiOps:
    """The device loop's ops for a batch split over several L2CPU tiles: tile i (fws[i], its own region and link
    block) serves the contiguous users blocks[i] = (user_base, n) (l2cpu.sampling.split_users). One push program
    with one core per tile (rows user_base .. user_base + n - 1 -> that tile's logits zone, local rows 0 .. n - 1;
    streamed: each core rings its tile first), one notify program with one core per tile (non-streamed), and a wait
    for every tile: one kernel polling all done_seq words under one bound (wait="all"), or one wait program per tile
    (wait="each")."""

    def __init__(self, device, fws, regions, blocks, core=(0, 0)):
        assert len(fws) == len(blocks) and all(n > 0 for _, n in blocks)
        self.dev, self.fws, self.blocks, self.core = device, fws, blocks, core
        self.links = []
        for fw in fws:
            reg = next(r for r in regions if arena_base_pa(r.buffer_address()) == fw.base)
            self.links.append(
                link_ops.Link(device, reg, fw.base + LINK_BLOCK_OFF, core, xy=link_ops.L2CPU_TILES[fw.hw.tile])
            )
        self.link = self.links[0]

    def _zone(self, uncached):
        return (LOGITS_UC_OFF if uncached else LOGITS_OFF) - LINK_BLOCK_OFF

    def scratch(self):
        return self.link.scratch()

    def run(self, prog, out=None):
        return self.link.run(prog, out=out)

    def set_push_source(self, logits, row_bytes=ROW):
        tab = self.link.push_source_table(logits, row_bytes)  # same table (the source tensor) in every link block
        for fw, L in zip(self.fws, self.links):
            fw.hw.pa_write(L.base + link_ops.LINK_OFF_PUSH_SRC, tab)

    def push_program(self, uncached=True, streamed=False, row_bytes=ROW, diag=False):
        return link_ops.push_split_program(
            self.links,
            self.blocks,
            row_bytes,
            self._zone(uncached),
            uncached=uncached,
            streamed=streamed,
            groups=STREAM_GROUPS,
            group_rows=[stream_group_rows(n) for _, n in self.blocks],
            diag=diag,
        )

    def notify_program(self, diag=False):
        return link_ops.notify_all_program(self.links, diag=diag)

    def wait_programs(self, timeout_us=link_ops.WAIT_TIMEOUT_US, diag=False, mode="all"):
        if mode == "all":
            return [link_ops.wait_all_program(self.links, timeout_us=timeout_us, diag=diag, core=self.core)]
        return [L.wait_program(timeout_us=timeout_us, diag=diag) for L in self.links]
