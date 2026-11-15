# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Host side of ``ttnn.experimental.deepseek.decode_step_context``: its RoPE tables and integer terms.

The op computes one decode step's index tensors and RoPE rows from the step's INT32 packet in a single program
(the per-step position math otherwise runs as ~70 one-tile device ops).
"""

from __future__ import annotations

import math
from typing import Optional

import torch

import ttnn

ROPE_BLOCK = 1024
ROPE_LO_ROWS = ROPE_BLOCK + 1  # offsets -1 .. 1023
Q_BITS = 28
INT32_MAX = 2**31 - 1
_MODE = {None: 0, "div": 1, "mod": 2}


def rope_table(inv_freq: torch.Tensor, scaling: float, max_pos: int) -> torch.Tensor:
    """``[1, 1, H + 1025, 2 * Rd]`` INT32 Q28 table: ``[cos | sin]`` of ``h * 1024 * w`` for ``h < H``, then
    ``scaling * [cos | sin]`` of ``l * w`` for ``l = -1 .. 1023`` (``w`` = ``inv_freq``'s ``Rd`` columns), so
    position ``p`` is ``(cos, sin)(p * w) * scaling`` by the angle-sum of its block row and its offset row."""
    w = inv_freq.reshape(-1).to(torch.float64)
    hi_rows = max_pos // ROPE_BLOCK + 2
    hi = torch.arange(hi_rows, dtype=torch.float64)[:, None] * ROPE_BLOCK * w[None]
    lo = torch.arange(-1, ROPE_BLOCK, dtype=torch.float64)[:, None] * w[None]
    rows = torch.cat(
        [
            torch.cat([torch.cos(hi), torch.sin(hi)], dim=-1),
            torch.cat([torch.cos(lo), torch.sin(lo)], dim=-1) * float(scaling),
        ],
        dim=0,
    )
    assert abs(float(scaling)) < 4.0, scaling
    return torch.round(rows * (1 << Q_BITS)).to(torch.int32).reshape(1, 1, rows.shape[0], rows.shape[1])


def term(
    src: int = -1,
    pre_add: int = 0,
    mode: Optional[str] = None,
    div: int = 1,
    mul: int = 1,
    post_add: int = 0,
    clamp_max: int = INT32_MAX,
    add_field: int = -1,
    add_mul: int = 1,
) -> list[int]:
    """One element: ``min(op(pkt[src] + pre_add) * mul + post_add, clamp_max) + pkt[add_field] * add_mul``, with
    ``op`` a floored ``// div`` (``"div"``), ``% div`` (``"mod"``) or nothing; a field of -1 reads 0."""
    w0 = ((src + 1) & 0xFF) | (_MODE[mode] << 8) | (((add_field + 1) & 0xFF) << 16)
    return [w0, pre_add, div, mul, post_add, clamp_max, add_mul]


def eval_term(t: list[int], pkt: list[int]) -> int:
    """The host reference of one :func:`term`."""
    w0, pre_add, div, mul, post_add, clamp_max, add_mul = t
    src, mode, add_field = (w0 & 0xFF) - 1, (w0 >> 8) & 0xF, ((w0 >> 16) & 0xFF) - 1
    v = (pkt[src] if src >= 0 else 0) + pre_add
    if mode == 1:
        v = v // div
    elif mode == 2:
        v = v % div
    v = min(v * mul + post_add, clamp_max)
    return v + (pkt[add_field] if add_field >= 0 else 0) * add_mul


class StepContextBuilder:
    """Collects one step's integer outputs and RoPE triples, runs the op once, and hands each back by key."""

    def __init__(self, pkt: ttnn.Tensor, tables: dict[str, ttnn.Tensor]):
        self.pkt = pkt
        self.table_names = list(tables)
        self.tables = [tables[n] for n in self.table_names]
        self.int_keys, self.int_shapes, self.int_dtypes, self.int_terms = [], [], [], []
        self.rope_keys, self.rope_tables, self.rope_fields, self.rope_offsets = [], [], [], []

    def add_int(self, key, shape: list[int], terms: list[list[int]], dtype=ttnn.int32) -> None:
        assert math.prod(shape) == len(terms), (shape, len(terms))
        self.int_keys.append(key)
        self.int_shapes.append(list(shape))
        self.int_dtypes.append(dtype)
        self.int_terms.append([w for t in terms for w in t])

    def add_rope(self, key, family: str, field: int, offset: int = 0) -> None:
        self.rope_keys.append(key)
        self.rope_tables.append(self.table_names.index(family))
        self.rope_fields.append(field)
        self.rope_offsets.append(offset)

    def run(self) -> dict:
        outs = ttnn.experimental.deepseek.decode_step_context(
            self.pkt,
            self.tables,
            self.int_shapes,
            self.int_dtypes,
            self.int_terms,
            self.rope_tables,
            self.rope_fields,
            self.rope_offsets,
        )
        n = len(self.int_keys)
        result = dict(zip(self.int_keys, outs[:n]))
        for j, key in enumerate(self.rope_keys):
            result[key] = tuple(outs[n + 3 * j : n + 3 * j + 3])
        return result
