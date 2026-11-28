# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2 (overlap reads and writes): BLOCK_BYTES x STREAM_DEPTH_BLOCKS sweep over the perf
shape and the config-spanning guard set (cells of more than 32 tiles).

One op call per test id; run under scripts/run_safe_pytest.sh --profile --no-py-zones and read the
g6_copy rows of the CSV in collection order. Every call is graded by helpers.TOLERANCES, so a
variant that wins by being wrong cannot pass. Without --profile it is a correctness sweep of the
streamed plan on every guard path.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_copy import describe_plan, g6_copy
from ttnn.operations.g6_copy import g6_copy_program_descriptor as plan_knobs
from eval.golden_tests.g6_copy import helpers as H

_F32, _BF16, _BF8B = ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b
_TILE, _RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT

GUARD_ROWS = [
    ((16, 12, 128, 128), _F32, _TILE),  # LOOSE_CASES[0], the perf target
    ((8, 12, 128, 128), _F32, _TILE),  # LOOSE_CASES[1]
    ((2, 12, 128, 128), _F32, _TILE),
    ((1, 12, 128, 128), _F32, _TILE),
    ((1, 1, 32, 62080), _BF16, _TILE),
    ((1, 1, 8448, 128), _BF16, _TILE),
    ((5, 1, 32, 256), _BF8B, _TILE),
    ((16, 3, 2560), _BF16, _RM),
]

# (BLOCK_BYTES, STREAM_DEPTH_BLOCKS)
VARIANTS = {
    "one_block": (None, 2),
    "b8k_d2": (8192, 2),
    "b16k_d2": (16384, 2),
    "b16k_d3": (16384, 3),
    "b24k_d2": (24576, 2),
    "b32k_d2": (32768, 2),
}


def row_id(row):
    shape, dtype, layout = row
    return f"{'x'.join(map(str, shape))}-{str(dtype).split('.')[-1]}-{'TILE' if layout == _TILE else 'RM'}"


@pytest.mark.parametrize("variant", list(VARIANTS), ids=list(VARIANTS))
@pytest.mark.parametrize("row", GUARD_ROWS, ids=row_id)
def test_g6_copy_overlap_sweep(device, monkeypatch, row, variant):
    block_bytes, depth = VARIANTS[variant]
    monkeypatch.setattr(plan_knobs, "BLOCK_BYTES", block_bytes)
    monkeypatch.setattr(plan_knobs, "STREAM_DEPTH_BLOCKS", depth)
    shape, dtype, layout = row
    gen = torch.Generator().manual_seed(7)
    src = H.make_source(dtype, shape, generator=gen)
    dst = H.make_source(dtype, shape, generator=gen)
    plan = describe_plan(H.create_ttnn_input_tensor(src, device, dtype=dtype, layout=layout))
    print(
        f"OVERLAP {row_id(row)} {variant}: plan={plan['plan']} block_pages={plan['block_pages']} "
        f"depth={plan['cb_depth_blocks']} core_pages_max={max(plan['core_pages'])}"
    )
    H.copy_and_check(g6_copy, src, dst, dtype=dtype, layout=layout, device=device, gate=H.default_gate(dtype))


REPEATS = 5


@pytest.mark.parametrize("repeat", range(REPEATS))
@pytest.mark.parametrize("variant", ["one_block", "b16k_d2", "b16k_d3", "b32k_d2"])
@pytest.mark.parametrize("row", GUARD_ROWS[:2], ids=row_id)
def test_g6_copy_overlap_repeats(device, monkeypatch, row, variant, repeat):
    """The Done-when samples: >= 5 fresh runs per variant on the flagged shapes."""
    test_g6_copy_overlap_sweep(device, monkeypatch, row, variant)
