# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compact RoPE position tables of the decode step states.

The full-width tables of ``DSV41PagedStepState`` are [max_pos, 512] bf16 (C = 1 / S = 0 off the 64 rotated dims) plus, for the indexer layer kinds, 128-wide copies: ~7 KiB per position per chip over the
three layer kinds, replicated on every chip (7 GiB at a 1M context, as much as the DRAM left after the model is built). Only the last 64 columns carry information, and the indexer rows repeat the same 64
values, so one cos and one sin table of [max_pos, 64] per layer kind is enough (0.25 KiB per position and kind, 0.75 KiB over the three kinds): the full-width rows are rebuilt on the device for the T gathered
positions of a step with one ``ttnn.concat`` against a constant row block. The rebuilt rows are bit-identical to the full tables (same bf16 values), so nothing downstream changes.

DSV41_ROPE_COMPACT=0 restores the full tables (read when a step state is built).
"""

import os

import torch

import ttnn

HEAD_DIM = 512
ROPE_DIM = 64
INDEX_DIM = 128


def compact_enabled():
    return os.environ.get("DSV41_ROPE_COMPACT", "1") == "1"


class CompactRope:
    """cos / sin of the rotated dims for every position, replicated: ``gather`` returns the rows in any of the full-width layouts of the old tables."""

    def __init__(self, md, c, s):
        """c, s: [P, 512] float tables from ``attn._rope_inputs(arange(P))`` (ones / zeros off the last 64 columns)."""
        assert bool((c[:, : HEAD_DIM - ROPE_DIM] == 1).all()) and bool(
            (s[:, : HEAD_DIM - ROPE_DIM] == 0).all()
        ), "rope tables are not C = 1 / S = 0 off the rotated dims"
        self.md = md
        self.max_pos = int(c.shape[0])
        self.t = {"C": self._up(c[:, HEAD_DIM - ROPE_DIM :]), "S": self._up(s[:, HEAD_DIM - ROPE_DIM :])}
        self._pad = {}

    def _up(self, t):
        return ttnn.from_torch(
            t.to(torch.bfloat16).contiguous(),
            device=self.md,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.md),
        )

    def _const(self, value, T, width):
        """[T, 1, width] constant rows (1 for cos, 0 for sin), created on first use (the eager compile pass runs before any trace capture)."""
        key = (value, T, width)
        if key not in self._pad:
            self._pad[key] = self._up(torch.full((T, 1, width), value))
        return self._pad[key]

    def gather(self, name, idx, shape):
        """name: 'C' / 'S' (512 wide: ones / zeros then the 64 rotated columns) or 'iC' / 'iS' (128 wide, the indexer head dim: 64 ones / zeros then the same 64 values);
        idx uint32 [T, 1] positions; -> tile tensor reshaped to ``shape`` (as ``DSV41PagedStepState._gather`` did)."""
        cos = name in ("C", "iC")
        width = HEAD_DIM if name in ("C", "S") else INDEX_DIM
        T = int(idx.shape[0])
        e = ttnn.embedding(idx, self.t["C" if cos else "S"], layout=ttnn.ROW_MAJOR_LAYOUT)  # [T, 1, 64]
        full = ttnn.concat([self._const(1.0 if cos else 0.0, T, width - ROPE_DIM), e], dim=-1)
        return ttnn.to_layout(ttnn.reshape(full, shape), ttnn.TILE_LAYOUT)
