# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Explicit, trace-stable controls for the second decode experiment."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectionTuning:
    reader: str = "original"
    wide_subblocks: bool = False
    bounded_barrier: bool = False
    buffer_count: int = 2
    hoist_pack_config: bool = False
    bank_vc: bool = False
    prefetch_gu_blocks: int = 0
    prefetch_down_blocks: int = 0

    def __post_init__(self):
        if any(n not in (0, 2, 4, 6) for n in (self.prefetch_gu_blocks, self.prefetch_down_blocks)):
            raise ValueError("Prefetch depth must be zero, two, four or six blocks")
        if self.buffer_count not in (2, 3):
            raise ValueError("Projection buffer_count must be two or three")
        if self.reader not in ("original", "coalesced", "pipelined", "pipelined_rows"):
            raise ValueError(f"Unknown projection reader: {self.reader}")

    @property
    def buffers(self):
        return self.buffer_count

    @property
    def defines(self):
        return [
            ("PROJECTION_READER", str(("original", "coalesced", "pipelined", "pipelined_rows").index(self.reader))),
            ("PROJECTION_HOIST_PACK", str(int(self.hoist_pack_config))),
            ("PROJECTION_BANK_VC", str(int(self.bank_vc))),
            ("GU_PREFETCH_BLOCKS", str(self.prefetch_gu_blocks)),
            ("DOWN_PREFETCH_BLOCKS", str(self.prefetch_down_blocks)),
            ("PROJECTION_BUFFERS", str(self.buffer_count)),
            ("PROJECTION_WIDE", str(int(self.wide_subblocks))),
        ]
