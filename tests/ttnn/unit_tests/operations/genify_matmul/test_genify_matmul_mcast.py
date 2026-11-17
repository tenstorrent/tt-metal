# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 3: regime R1a `interleaved_2d_mcast` (row / column operand multicast) and the
single-walk K split.

Pins what the golden suite does not see:
  - both physical orientations of the split rectangle ("rect": act lines are grid rows; "rect_t":
    act lines are grid columns);
  - the grid-uniform block walk: edge cores whose padded tail holds whole EMPTY blocks still take
    part in every line multicast (CB pointers in lockstep) and write nothing for them;
  - staggered vs first-core injectors give bit-identical results (same K order, only the source of
    the operand bytes changes);
  - a per-core walk that would be one K-block is split in SINGLE_WALK_K_BLOCKS (read overlaps compute).
Accuracy gate: helpers.TOLERANCES via run_genify_matmul (the golden-suite runner).
"""

import math

import pytest
import torch
import ttnn

from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor, run_genify_matmul
from ttnn.operations.genify_matmul import genify_matmul
from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd


def _plan(device, a_shape, b_shape, epilogue="none"):
    n_out = b_shape[-1] // 2 if epilogue == "swiglu" else b_shape[-1]
    a = ttnn.allocate_tensor_on_device(ttnn.Shape(list(a_shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    b = ttnn.allocate_tensor_on_device(ttnn.Shape(list(b_shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    o = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(a_shape[:-1]) + [n_out]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device
    )
    geom, grid, blk, _ = pd.plan(a, b, o, True, device, epilogue=epilogue)
    for t in (a, b, o):
        ttnn.deallocate(t)
    return geom, grid, blk


def _has_empty_tail_block(geom, grid, blk):
    nb_m = -(-grid.core_m_tiles // blk.block_m_tiles)
    nb_n = -(-grid.core_n_tiles // blk.block_n_tiles)
    last_rows = geom.tensor_m_tiles - (grid.grid_m - 1) * grid.core_m_tiles
    last_cols = geom.tensor_n_tiles - (grid.grid_n - 1) * grid.core_n_tiles
    return last_rows <= (nb_m - 1) * blk.block_m_tiles or last_cols <= (nb_n - 1) * blk.block_n_tiles


def _run(inputs, epilogue, device):
    run_genify_matmul(
        inputs,
        dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        fp32_dest_acc_en=True,
        placement="interleaved",
        epilogue=epilogue,
        device=device,
    )


@pytest.mark.parametrize(
    "a_shape, b_shape, epilogue, placement",
    [
        ((1, 704, 512), (512, 640), "none", "rect_t"),  # 11 row regions > grid height: transposed rectangle
        ((1, 384, 1536), (1536, 1024), "addcmul", "rect"),
        ((1, 6000, 1024), (1024, 6016), "bias", "rect"),  # empty N-tail blocks on the last column region
        ((1, 6000, 1024), (1024, 6016), "swiglu", "rect"),
        ((1, 6000, 1024), (1024, 4992), "swiglu", "rect_t"),  # empty M-tail block on the last row region
    ],
)
def test_r1a_orientation_and_padded_tail(a_shape, b_shape, epilogue, placement, device):
    geom, grid, blk = _plan(device, a_shape, b_shape, epilogue)
    assert pd.uses_operand_mcast(geom, grid, blk), f"expected regime R1a, got {grid}"
    assert grid.placement == placement, f"expected a {placement} split, got {grid}"
    if a_shape[-2] == 6000:
        assert _has_empty_tail_block(geom, grid, blk), f"expected an empty padded-tail block: {grid} {blk}"
    _run((a_shape, b_shape), epilogue, device)


def test_r1a_injector_placement_bit_identical(device, monkeypatch):
    """Staggered vs first-core injectors: only where the operand bytes come from changes."""
    torch.manual_seed(7)
    A = torch.randn(1, 1024, 2048)
    B = torch.randn(2048, 1536) / math.sqrt(2048)
    ta = create_ttnn_input_tensor(A, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tb = create_ttnn_input_tensor(B, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    outs = []
    for stagger in (False, True):
        monkeypatch.setattr(pd, "STAGGER_INJECTORS", stagger)
        outs.append(ttnn.to_torch(genify_matmul(ta, tb)))
    assert torch.equal(outs[0], outs[1])


def test_single_walk_k_split(device):
    """One block per core and K fits one K-block: the walk is split so the read overlaps compute."""
    _, grid, blk = _plan(device, (1, 384, 1536), (1536, 1024))
    assert blk.num_k_blocks == pd.SINGLE_WALK_K_BLOCKS, f"{grid} {blk}"
    _run(((1, 384, 1536), (1536, 1024)), "bias", device)
