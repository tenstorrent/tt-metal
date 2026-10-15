# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compact RoPE position tables of the decode step states.

The full-width tables of ``DSV41PagedStepState`` are [max_pos, 512] bf16 (C = 1 / S = 0 off the 64 rotated dims) plus, for the indexer layer kinds, 128-wide copies: ~7 KiB per position per chip over the
three layer kinds, replicated on every chip (7 GiB at a 1M context, as much as the DRAM left after the model is built). Only the last 64 columns carry information, and the indexer rows repeat the same 64
values, so one cos and one sin table of [max_pos, 64] per layer kind is enough (0.25 KiB per position and kind, 0.75 KiB over the three kinds): the full-width rows are rebuilt on the device for the T gathered
positions of a step with one ``ttnn.concat`` against a constant row block. The rebuilt rows are bit-identical to the full tables (same bf16 values), so nothing downstream changes.

DSV41_ROPE_COMPACT=0 restores the full tables (read when a step state is built).

DSV41_ROPE_SHARD (auto = from 524288 positions on; 1 / 0 force) additionally shards the [P, 128] (cos | sin) table by position over the 8 mesh columns (each chip keeps P / 8 rows, 0.09 KiB per position and kind): a step looks its positions
up in the local block (indices outside it are masked to zero), one all-gather over the columns brings the 8 partial rows together and a sum over them (7 exact zeros + the value) restores the row,
bit-identical to the replicated table. It costs ~25 small ops and one all-gather per layer kind per step (measured in tests/test_rope_compact.py), so the default `auto` only shards where the replicated
0.75 KiB per position starts to matter (>= 524288 positions).
"""

import os

import torch

import ttnn

HEAD_DIM = 512
ROPE_DIM = 64
INDEX_DIM = 128


def compact_enabled():
    return os.environ.get("DSV41_ROPE_COMPACT", "1") == "1"


SHARD_AUTO_POSITIONS = 524288  # DSV41_ROPE_SHARD=auto shards the tables from this many positions on (0.75 -> 0.09 GiB at 1M for ~0.4 ms per decode step)


def shard_enabled(max_pos=0):
    """DSV41_ROPE_SHARD: 1 = always, 0 = never, auto (default) = from SHARD_AUTO_POSITIONS positions on."""
    v = os.environ.get("DSV41_ROPE_SHARD", "auto")
    return v == "1" or (v == "auto" and max_pos >= SHARD_AUTO_POSITIONS)


class CompactRope:
    """cos / sin of the rotated dims for every position (replicated, or sharded by position over the mesh columns): ``lookup`` returns the 64-wide rows of a list of position sets,
    ``full`` / ``gather`` rebuild the full-width rows of the old tables."""

    def __init__(self, md, c, s, mesh_config=None, ccl=None):
        """c, s: [P, 512] float tables from ``attn._rope_inputs(arange(P))`` (ones / zeros off the last 64 columns). ``mesh_config`` / ``ccl``: the model's (needed for DSV41_ROPE_SHARD=1)."""
        assert bool((c[:, : HEAD_DIM - ROPE_DIM] == 1).all()) and bool(
            (s[:, : HEAD_DIM - ROPE_DIM] == 0).all()
        ), "rope tables are not C = 1 / S = 0 off the rotated dims"
        self.md = md
        self.max_pos = int(c.shape[0])
        self._pad = {}
        self.sharded = shard_enabled(self.max_pos) and mesh_config is not None and ccl is not None
        if not self.sharded:
            self.t = {"C": self._up(c[:, HEAD_DIM - ROPE_DIM :]), "S": self._up(s[:, HEAD_DIM - ROPE_DIM :])}
            return
        self.mesh_config, self.ccl = mesh_config, ccl
        rows, cols = tuple(md.shape)
        self.cols = cols
        self.Pc = -(-self.max_pos // cols)  # positions per column
        tab = torch.zeros(self.Pc * cols, 2 * ROPE_DIM)
        tab[: self.max_pos, :ROPE_DIM] = c[:, HEAD_DIM - ROPE_DIM :]
        tab[: self.max_pos, ROPE_DIM:] = s[:, HEAD_DIM - ROPE_DIM :]
        self.tab = ttnn.from_torch(
            tab.to(torch.bfloat16).contiguous(),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(None, 0), mesh_shape=(rows, cols)),
        )
        base = (torch.arange(cols, dtype=torch.float32) * self.Pc).reshape(1, 1, 1, cols)
        self.base = ttnn.from_torch(
            base,
            device=md,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(None, 3), mesh_shape=(rows, cols)),
        )

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

    def lookup(self, idxs, need=("C", "S")):
        """idxs: list of uint32 [T, 1] position tensors -> list of (cos, sin) 64-wide rows, each [T, 1, 64] row-major bf16 (None for a table not in ``need``)."""
        if self.sharded:
            return self._lookup_sharded(idxs)
        return [
            tuple(
                ttnn.embedding(idx, self.t[n], layout=ttnn.ROW_MAJOR_LAYOUT) if n in need else None for n in ("C", "S")
            )
            for idx in idxs
        ]

    def _lookup_sharded(self, idxs):
        T, k = int(idxs[0].shape[0]), len(idxs)
        kT = k * T
        cat = idxs[0] if k == 1 else ttnn.concat(idxs, dim=0)  # [kT, 1] uint32
        f = ttnn.typecast(ttnn.to_layout(ttnn.reshape(cat, [1, 1, 1, kT]), ttnn.TILE_LAYOUT), ttnn.float32)
        loc = ttnn.subtract(f, self.base)  # position inside this column's block
        inr = ttnn.multiply(ttnn.ge(loc, 0.0), ttnn.lt(loc, float(self.Pc)))  # 1.0 where this column owns the position
        li = ttnn.reshape(
            ttnn.to_layout(
                ttnn.typecast(ttnn.clamp(loc, min=0.0, max=float(self.Pc - 1)), ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT
            ),
            [kT, 1],
        )
        e = ttnn.to_layout(
            ttnn.reshape(ttnn.embedding(li, self.tab, layout=ttnn.ROW_MAJOR_LAYOUT), [1, 1, kT, 2 * ROPE_DIM]),
            ttnn.TILE_LAYOUT,
        )
        e = ttnn.multiply(
            e, ttnn.permute(ttnn.typecast(inr, ttnn.bfloat16), (0, 1, 3, 2))
        )  # zero rows of the other columns
        g = self.mesh_config.allgather(e, self.ccl, axis=1, dim=0)  # [cols, 1, kT, 128]
        r = ttnn.sum(g, dim=0, keepdim=True)  # one non-zero partial + zeros: exact
        out = []
        for j in range(k):
            piece = lambda c0: ttnn.reshape(
                ttnn.to_layout(
                    ttnn.slice(r, [0, 0, j * T, c0], [1, 1, (j + 1) * T, c0 + ROPE_DIM]), ttnn.ROW_MAJOR_LAYOUT
                ),
                [T, 1, ROPE_DIM],
            )
            out.append((piece(0), piece(ROPE_DIM)))
        return out

    def full(self, name, row64, shape):
        """name: 'C' / 'S' (512 wide: ones / zeros then the 64 rotated columns) or 'iC' / 'iS' (128 wide, the indexer head dim: 64 ones / zeros then the same 64 values);
        row64 [T, 1, 64] from ``lookup`` -> tile tensor reshaped to ``shape``."""
        cos = name in ("C", "iC")
        width = HEAD_DIM if name in ("C", "S") else INDEX_DIM
        T = int(row64.shape[0])
        full = ttnn.concat([self._const(1.0 if cos else 0.0, T, width - ROPE_DIM), row64], dim=-1)
        return ttnn.to_layout(ttnn.reshape(full, shape), ttnn.TILE_LAYOUT)

    def gather(self, name, idx, shape):
        """one table of one position set (``DSV41PagedStepState._gather`` of the old full tables)."""
        cos = name in ("C", "iC")
        row = self.lookup([idx], need=("C",) if cos else ("S",))[0][0 if cos else 1]
        return self.full(name, row, shape)
