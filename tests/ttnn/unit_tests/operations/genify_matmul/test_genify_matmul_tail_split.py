# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 12: the tail write split (tail_split.hpp) on multi-batch blocks.

The reader serves its share of the core's LAST block's output writes only after its K-walk. When that block
carries several batches (block_batches > 1, batched B: regime R1d), the writer used to post the block's first
batch and hold its pop for the reader while the reader still had the block's later batches to read: the output
CB filled, compute stalled, the reader blocked on an input reserve -> deadlock (bf16 PDL upsample H pass, R1a
with block_batches = 32). The split now covers only the tail block's LAST batch, whose output exists only once
the reader has pushed its final K-block, so the handshake has no cycle for any block shape.

Pinned: the PDL H pass in bf16 and bfp8, B in DRAM and in L1 (HiFi2, 16-bit DEST: the golden loose case),
plus multi-batch tail blocks on R1a (two K-blocks) and plain R1 under none / bias / addcmul.
Accuracy gate: helpers.TOLERANCES via run_genify_matmul.
"""

import pytest
import torch
import ttnn

from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor, run_genify_matmul
from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

_BF16, _BFP8 = ttnn.bfloat16, ttnn.bfloat8_b
_HIFI2 = ttnn.MathFidelity.HiFi2
_L1, _DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG

_HPASS = ((1, 32, 512, 128), (1, 32, 128, 1024))


def _plan(device, a, b, dtype, acc, b_mc, epilogue="none"):
    ta = create_ttnn_input_tensor(torch.randn(*a), device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=_DRAM)
    tb = create_ttnn_input_tensor(torch.randn(*b), device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=b_mc)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(a[:-1]) + [b[-1]]), dtype, ttnn.TILE_LAYOUT, device, _DRAM)
    geom, grid, blk, _ = pd.plan(ta, tb, out, acc, device, epilogue=epilogue, math_fidelity=_HIFI2)
    mode = pd.operand_mode(geom, grid, blk)
    for t in (ta, tb, out):
        ttnn.deallocate(t)
    return grid, blk, mode


def _run(device, a, b, dtype, acc, b_buffer, epilogue="none"):
    run_genify_matmul(
        (a, b),
        dtype=dtype,
        weight_dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        fp32_dest_acc_en=acc,
        placement="interleaved",
        epilogue=epilogue,
        device=device,
        extras={"math_fidelity": _HIFI2, "weight_buffer": b_buffer},
    )


@pytest.mark.parametrize("b_buffer", ["dram", "l1"])
@pytest.mark.parametrize("dtype", [_BF16, _BFP8], ids=["bf16", "bfp8"])
def test_pdl_hpass_batched_weight(device, dtype, b_buffer):
    """The golden loose / feature cases: bf16 plans a 32-batch block (the hang), bfp8 a 1-batch one."""
    b_mc = _L1 if b_buffer == "l1" else _DRAM
    grid, blk, _ = _plan(device, *_HPASS, dtype, False, b_mc)
    if dtype == _BF16:
        assert blk.block_batches > 1, (grid, blk)  # the shape that hung: a multi-batch tail block
    _run(device, *_HPASS, dtype, False, b_buffer)


# Multi-batch tail blocks on each walk (planned shapes, HiFi2, 16-bit DEST):
#   r1a_multi_k: R1a line multicast, block_batches = 8, two K-blocks (the partials path);
#   r1_plain:    plain R1 (every core reads both operands), block_batches = 2.
MULTI_BATCH = {
    "r1a_multi_k": (((1, 8, 512, 512), (1, 8, 512, 512)), 1),
    "r1_plain": (((1, 128, 64, 128), (1, 128, 128, 64)), 0),
}


@pytest.mark.parametrize("epilogue", ["none", "bias", "addcmul"])
@pytest.mark.parametrize("case", MULTI_BATCH.keys())
def test_multi_batch_tail(device, case, epilogue):
    """Multi-batch tail blocks under every row / residual epilogue family (the residual loop serves the tail
    mid-walk); bias / addcmul keep the plan's block_batches > 1 like none."""
    (a, b), mode = MULTI_BATCH[case]
    grid, blk, got_mode = _plan(device, a, b, _BF16, False, _DRAM)
    assert blk.block_batches > 1 and got_mode == mode, (grid, blk, got_mode)
    _run(device, a, b, _BF16, False, "dram", epilogue=epilogue)
