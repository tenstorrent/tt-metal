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

    def __post_init__(self):
        if self.buffer_count not in (2, 3):
            raise ValueError("Projection buffer_count must be two or three")
        if self.reader not in ("original", "coalesced", "pipelined"):
            raise ValueError(f"Unknown projection reader: {self.reader}")

    @property
    def buffers(self):
        return self.buffer_count

    @property
    def defines(self):
        return [
            ("PROJECTION_READER", str(("original", "coalesced", "pipelined").index(self.reader))),
            ("PROJECTION_BUFFERS", str(self.buffer_count)),
            ("PROJECTION_WIDE", str(int(self.wide_subblocks))),
        ]
