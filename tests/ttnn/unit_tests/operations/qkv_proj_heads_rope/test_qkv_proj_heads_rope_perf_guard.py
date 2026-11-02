# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf guard set: one representative per kernel path x layout x placement. Run with
scripts/run_safe_pytest.sh --profile; read DEVICE KERNEL DURATION per op from the profiler CSV (row order =
GUARD_ROWS order)."""

import pytest
import ttnn

from ttnn.operations.qkv_proj_heads_rope import qkv_proj_heads_rope

from .test_qkv_proj_heads_rope import check_outputs, make_inputs, reference, to_device

L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG

# (B, S, x, tables, outputs): path covered
GUARD_ROWS = [
    (8, 512, L1, L1, L1),  # focus: 12x9, R2 + R3, one block
    (2, 512, L1, L1, L1),  # 12x9, W-multicast bound, weight depth 3
    (7, 511, L1, L1, L1),  # h_non_aligned, pad-row zeroing
    (13, 300, L1, L1, L1),  # h_non_aligned, short S
    (8, 512, DRAM, L1, L1),  # x in DRAM
    (8, 512, L1, DRAM, DRAM),  # tables and outputs in DRAM
    (1, 32, L1, L1, L1),  # one tile-row: R1 W unicast
    (1, 128, L1, L1, L1),  # small grid
    (2, 37, L1, L1, L1),  # tiny, non-aligned
    (16, 8, L1, L1, L1),  # S < 32
    (4096, 1, DRAM, DRAM, DRAM),  # multi-block per core
]


def _id(row):
    b, s, x, t, o = row
    name = lambda m: "L1" if m == L1 else "DRAM"
    return f"{b}x{s}_x{name(x)}_t{name(t)}_o{name(o)}"


@pytest.mark.parametrize("batch, seq, x_mc, table_mc, out_mc", GUARD_ROWS, ids=[_id(r) for r in GUARD_ROWS])
def test_perf_guard(device, batch, seq, x_mc, table_mc, out_mc):
    x, w, b, cos, sin = make_inputs(batch, seq, seq)
    tensors = (
        to_device(x, device, x_mc),
        to_device(w, device, DRAM),
        to_device(b, device, DRAM),
        to_device(cos, device, table_mc),
        to_device(sin, device, table_mc),
    )
    outputs = qkv_proj_heads_rope(*tensors, memory_config=out_mc)
    check_outputs(outputs, reference(x, w, b, cos, sin), buffer_type=out_mc.buffer_type)
