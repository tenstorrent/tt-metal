# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_copy, Refinement 1: bfloat16 / bfloat8_b / int32 / uint32 and ROW_MAJOR.

Every dtype and layout runs the same page copy as Phase 0 (page_bytes =
buffer_aligned_page_size(), CB format never interpreted). Cases are the
non-float32 production rows plus the cases the contract's rules require
(BB-22): adversarial bits per dtype, bf16 poisoned padding, shortest RM row.
Graded by helpers.TOLERANCES: bit-exact, or the bfp8 one-step gate.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_copy import describe_plan, g6_copy
from eval.golden_tests.g6_copy import helpers as H

L1_BYTES_LEFT_BY_MODEL = 1347776

_BF16, _BF8B, _I32, _U32 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.int32, ttnn.uint32
_TILE, _RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT

# (shape, dtype, layout): the non-float32 production rows.
PRODUCTION_ROWS = [
    ((1, 1, 32, 62080), _BF16, _TILE),
    ((1, 2048, 128), _BF16, _TILE),
    ((1, 1, 2304, 128), _BF16, _TILE),
    ((1, 1, 8448, 128), _BF16, _TILE),
    ((1, 256, 3072), _BF16, _TILE),
    ((1, 1, 1, 256), _BF16, _TILE),
    ((1, 1), _BF16, _TILE),
    ((5, 1, 32, 256), _BF8B, _TILE),
    ((1, 3, 2560), _BF16, _RM),
    ((16, 3, 2560), _BF16, _RM),
    ((1,), _U32, _RM),  # shortest RM row: 4 B in a 64 B page
    ((32,), _U32, _RM),
    ((7, 1, 1, 32), _U32, _RM),
    ((1,), _I32, _RM),  # shortest RM row
    ((16,), _I32, _RM),
]

ADVERSARIAL_ROWS = [
    ((1, 256, 3072), _BF16, _TILE),
    ((5, 1, 32, 256), _BF8B, _TILE),
    ((16,), _I32, _RM),
    ((32,), _U32, _RM),
]


def _expected_bank_row_cores(tensor_pages, grid, num_banks=8):
    """Cores of the bank-row plan (Perf 1): bank b's q + (b < r) pages over min(k, n_b) cores,
    k = min(ceil(pages / banks), grid.x)."""
    k = min(-(-tensor_pages // num_banks), grid.x)
    q, r = divmod(tensor_pages, num_banks)
    return sum(min(k, q + (1 if b < r else 0)) for b in range(num_banks))


def row_id(row):
    shape, dtype, layout = row
    return f"{'x'.join(map(str, shape))}-{str(dtype).split('.')[-1]}-{'TILE' if layout == _TILE else 'RM'}"


def run(device, shape, dtype, layout, *, adversarial=False, poison_padding=None):
    gen = torch.Generator().manual_seed(1234)
    if adversarial:
        src = H.make_adversarial_source(dtype, shape, generator=gen)
    else:
        src = H.make_source(dtype, shape, generator=gen)
    dst = H.make_source(dtype, shape, generator=gen)
    H.copy_and_check(
        g6_copy,
        src,
        dst,
        dtype=dtype,
        layout=layout,
        device=device,
        gate=H.default_gate(dtype),
        poison_padding=poison_padding,
    )


@pytest.mark.parametrize("row", PRODUCTION_ROWS, ids=row_id)
def test_g6_copy_values_dtype_layout(device, row):
    run(device, *row)


@pytest.mark.parametrize("row", ADVERSARIAL_ROWS, ids=row_id)
def test_g6_copy_adversarial_dtype(device, row):
    run(device, *row, adversarial=True)


def test_g6_copy_poisoned_padding_bf16(device):
    run(device, (1, 1, 1, 256), _BF16, _TILE, poison_padding=1000.0)


@pytest.mark.parametrize("row", PRODUCTION_ROWS, ids=row_id)
def test_g6_copy_plan_dtype_layout(device, row):
    shape, dtype, layout = row
    x = H.create_ttnn_input_tensor(H.make_source(dtype, shape), device, dtype=dtype, layout=layout)
    plan = describe_plan(x)
    grid = device.compute_with_storage_grid_size()
    assert plan == describe_plan(x), "the plan must be deterministic (BB-08)"
    assert plan["page_bytes"] == x.buffer_aligned_page_size()
    assert plan["tensor_pages"] == x.buffer_num_pages()
    # Bank-row placement (Perf 1): each bank's pages on cores in that bank's NoC0 endpoint row.
    assert plan["num_cores"] == _expected_bank_row_cores(x.buffer_num_pages(), grid)
    assert sum(plan["core_pages"]) == plan["tensor_pages"]
    assert plan["l1_bytes_per_core"] <= L1_BYTES_LEFT_BY_MODEL
