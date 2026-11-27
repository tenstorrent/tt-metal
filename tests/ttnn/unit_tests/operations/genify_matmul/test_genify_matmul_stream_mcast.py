# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 9: regime R1b `interleaved_stream_mcast_a` (1-D N split, every core streams its own B
columns, one injector broadcasts each A K-block from the writer RISC).

Pins what the golden suite does not see:
  - the regime is actually selected (not plain R1) on the M = 1 / small-M DiT shapes, both for a flat
    split (an irregular two-rectangle receiver set) and a one-row rectangle;
  - every epilogue family rides the A stream (the reader keeps the bias / gate / residual reads);
  - the row-aware N quantum (Q_N_ROW_AWARE) and the R1b knob off give the same numbers (only the
    source of A's bytes and the column split change);
  - A in L1, fp32 A, ragged M / N, a multi-N-block walk and A residency across N-blocks.
Accuracy gate: helpers.TOLERANCES via run_genify_matmul (the golden-suite runner).
"""

import pytest
import ttnn

from eval.golden_tests.genify_matmul.helpers import run_genify_matmul
from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd


def _plan(device, a_shape, b_shape, epilogue="none", dtype=ttnn.bfloat16, mc=ttnn.DRAM_MEMORY_CONFIG):
    n_out = b_shape[-1] // 2 if epilogue == "swiglu" else b_shape[-1]
    a = ttnn.allocate_tensor_on_device(ttnn.Shape(list(a_shape)), dtype, ttnn.TILE_LAYOUT, device, mc)
    b = ttnn.allocate_tensor_on_device(ttnn.Shape(list(b_shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    o = ttnn.allocate_tensor_on_device(ttnn.Shape(list(a_shape[:-1]) + [n_out]), dtype, ttnn.TILE_LAYOUT, device)
    geom, grid, blk, _ = pd.plan(a, b, o, True, device, epilogue=epilogue)
    for t in (a, b, o):
        ttnn.deallocate(t)
    return geom, grid, blk


def _run(inputs, epilogue, device, dtype=ttnn.bfloat16, extras=None):
    run_genify_matmul(
        inputs,
        dtype=dtype,
        weight_dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        fp32_dest_acc_en=True,
        placement="interleaved",
        epilogue=epilogue,
        device=device,
        extras=extras,
    )


HIFI2 = {"math_fidelity": ttnn.MathFidelity.HiFi2}


@pytest.mark.parametrize(
    "a_shape, b_shape, epilogue",
    [
        ((1, 256, 6144), (6144, 18432), "swiglu"),  # PERF FOCUS 4
        ((1, 6144), (6144, 6144), "none"),  # FLUX guidance linear_2 (M = 1)
        ((1, 256, 2048), (2048, 3072), "addcmul"),
        ((1, 256, 2048), (2048, 3072), "bias"),
        ((1, 200, 2048), (2048, 3008), "bias_relu"),  # ragged M and N
        ((1, 64, 4096), (4096, 1408), "none"),  # one-row rectangle split
    ],
)
def test_r1b_selected_and_correct(a_shape, b_shape, epilogue, device):
    geom, grid, blk = _plan(device, a_shape, b_shape, epilogue)
    assert pd.operand_mode(geom, grid, blk) == 2, f"expected regime R1b, got {grid} {blk}"
    _run((a_shape, b_shape), epilogue, device, extras=HIFI2)


def test_r1b_fp32_a(device):
    a_shape, b_shape = (1, 6144), (6144, 6144)  # FLUX timestep linear_2: fp32 A / out
    geom, grid, blk = _plan(device, a_shape, b_shape, dtype=ttnn.float32)
    assert pd.operand_mode(geom, grid, blk) == 2
    _run((a_shape, b_shape), "none", device, dtype=ttnn.float32, extras=HIFI2)


def test_r1b_a_in_l1(device):
    a_shape, b_shape = (1, 128, 4096), (4096, 4096)
    geom, grid, blk = _plan(device, a_shape, b_shape, mc=ttnn.L1_MEMORY_CONFIG)
    assert pd.operand_mode(geom, grid, blk) == 2
    _run((a_shape, b_shape), "swiglu", device, extras={**HIFI2, "act_buffer": "l1"})


@pytest.mark.parametrize("epilogue", ["none", "swiglu"])
def test_r1b_multi_block_walk(epilogue, device, monkeypatch):
    # A small CB budget forces the S3 N-block split, so A is re-streamed once per N-block (R1a off: at
    # this budget the ranking would otherwise prefer the 2-D multicast).
    a_shape, b_shape = (1, 256, 4096), (4096, 32768)
    monkeypatch.setattr(pd, "L1_RESERVE_BYTES", 700 * 1024)
    monkeypatch.setattr(pd, "OPERAND_MCAST", False)
    geom, grid, blk = _plan(device, a_shape, b_shape, epilogue)
    assert pd.operand_mode(geom, grid, blk) == 2
    assert -(-grid.core_n_tiles // blk.block_n_tiles) > 1, f"expected several N-blocks: {grid} {blk}"
    _run((a_shape, b_shape), epilogue, device, extras=HIFI2)


def test_r1b_act_resident(device, monkeypatch):
    # A one-K-block walk split into N-blocks with A held fronted (Refinement 8) under the A stream. The flat split
    # pinned: the bank-local split (Refinement 15) takes this shape by default and never holds A resident.
    monkeypatch.setattr(pd, "ACT_STREAM_BANK_LOCAL", False)
    a_shape, b_shape = (1, 128, 384), (384, 16384)
    geom, grid, blk = _plan(device, a_shape, b_shape)
    assert pd.operand_mode(geom, grid, blk) == 2
    assert blk.act_resident, f"expected A residency: {grid} {blk}"
    _run((a_shape, b_shape), "none", device)


@pytest.mark.parametrize("knob", ["ACT_STREAM_MCAST", "Q_N_ROW_AWARE"])
def test_r1b_knobs_off(knob, device, monkeypatch):
    monkeypatch.setattr(pd, knob, False)
    _run(((1, 256, 2048), (2048, 3072)), "swiglu", device, extras=HIFI2)
