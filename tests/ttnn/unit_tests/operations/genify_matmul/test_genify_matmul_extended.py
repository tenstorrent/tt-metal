# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Verifier-added regression tests for genify_matmul (Phase 0 cell, beyond the immutable acceptance test).

- Output `dtype` override (the loose cases write bf16 from fp32 A, bf8b from bf16 A): the output CB
  format follows `dtype`, independent of A's dtype. Covers the one-K-block path and the K-walk
  (Float32 partials packed into a narrower output).
- Fragmented L1: an L1 tensor freed above a live one leaves a hole, so the CB region must end at the
  lowest allocated L1 block, not at `bank - allocated bytes` (was a CB/L1 clash TT_THROW).
- Weight in L1 interleaved (model cases keep B in L1).
"""

import math

import pytest
import torch

import ttnn

from eval.golden_tests.genify_matmul.helpers import (
    TOLERANCES,
    check_output,
    create_ttnn_input_tensor,
    pytorch_genify_matmul,
    quantize,
)
from ttnn.operations.genify_matmul import genify_matmul

SHAPES = [
    ((1, 384, 1536), (1536, 1024)),  # one resident K-block
    ((1, 64, 16384), (16384, 512)),  # K-walk with Float32 partials
]
IDS = [f"A{'x'.join(map(str, a))}_B{'x'.join(map(str, b))}" for a, b in SHAPES]


def _operands(device, a_shape, b_shape, b_mc=None):
    torch.manual_seed(7)
    A = torch.randn(a_shape)
    B = torch.randn(b_shape) / math.sqrt(b_shape[-2])
    ta = create_ttnn_input_tensor(A, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tb = create_ttnn_input_tensor(B, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=b_mc)
    expected = pytorch_genify_matmul(quantize(A, ttnn.bfloat16), quantize(B, ttnn.bfloat16))
    return ta, tb, expected


@pytest.mark.parametrize("out_dtype", [ttnn.float32, ttnn.bfloat8_b], ids=["fp32_out", "bf8b_out"])
@pytest.mark.parametrize("a_shape, b_shape", SHAPES, ids=IDS)
def test_output_dtype_override(a_shape, b_shape, out_dtype, device):
    ta, tb, expected = _operands(device, a_shape, b_shape)
    out = genify_matmul(ta, tb, dtype=out_dtype)
    check_output(
        out,
        expected,
        shape=list(a_shape[:-1]) + [b_shape[-1]],
        dtype=out_dtype,
        expected_layout=ttnn.TILE_LAYOUT,
        tolerance=TOLERANCES[(out_dtype, True)],
    )


def test_fragmented_l1(device):
    """L1 is allocated top-down: `hole` takes the top ~300 KB/bank, `low` lands below it; freeing
    `hole` leaves `low` live with a free gap above it. allocated-bytes accounting then over-states
    the CB room by the gap; the deep-K walk grows block_k into the budget, so the old formula's CB
    region overlapped `low` (program.cpp "circular buffers ... clash with L1 buffers")."""
    hole = ttnn.from_torch(
        torch.zeros(32768, 512),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    low = ttnn.from_torch(
        torch.zeros(2048, 512),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    hole.deallocate()
    a_shape, b_shape = SHAPES[1]
    ta, tb, expected = _operands(device, a_shape, b_shape)
    out = genify_matmul(ta, tb)
    check_output(
        out,
        expected,
        shape=list(a_shape[:-1]) + [b_shape[-1]],
        dtype=ttnn.bfloat16,
        expected_layout=ttnn.TILE_LAYOUT,
        tolerance=TOLERANCES[(ttnn.bfloat16, True)],
    )
    low.deallocate()


@pytest.mark.parametrize("a_shape, b_shape", SHAPES, ids=IDS)
def test_l1_weight(a_shape, b_shape, device):
    ta, tb, expected = _operands(device, a_shape, b_shape, b_mc=ttnn.L1_MEMORY_CONFIG)
    out = genify_matmul(ta, tb)
    check_output(
        out,
        expected,
        shape=list(a_shape[:-1]) + [b_shape[-1]],
        dtype=ttnn.bfloat16,
        expected_layout=ttnn.TILE_LAYOUT,
        tolerance=TOLERANCES[(ttnn.bfloat16, True)],
    )
