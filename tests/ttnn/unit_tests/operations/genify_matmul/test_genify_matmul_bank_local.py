# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 15: the R1b bank-local B stream (`bank` split) and the rotating A injectors.

Pins what the golden suite does not see:
  - the `bank` split is actually selected on small-M weight-streaming shapes (Qwen3.8 M = 128 bfp4, FLUX M = 1),
    including a row length that is not a multiple of the DRAM banks (a class's units then rotate banks per K row)
    and a ragged N;
  - every epilogue family rides it (the bias / gate / residual reads and the output writes follow the bank stride);
  - the placements ("row", "near", "spread"), the rotating injector counts and swiglu under BANK_LOCAL_SWIGLU all
    give correct results (they only move bytes / cores, never the math).
Accuracy gate: helpers.TOLERANCES via run_genify_matmul (the golden-suite runner).
"""

import pytest
import ttnn

from eval.golden_tests.genify_matmul.helpers import run_genify_matmul
from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

LOFI = {"math_fidelity": ttnn.MathFidelity.LoFi}


def _plan(device, a_shape, b_shape, epilogue="none", dtype=ttnn.bfloat16, w_dtype=ttnn.bfloat16):
    n_out = b_shape[-1] // 2 if epilogue == "swiglu" else b_shape[-1]
    a = ttnn.allocate_tensor_on_device(ttnn.Shape(list(a_shape)), dtype, ttnn.TILE_LAYOUT, device)
    b = ttnn.allocate_tensor_on_device(ttnn.Shape(list(b_shape)), w_dtype, ttnn.TILE_LAYOUT, device)
    o = ttnn.allocate_tensor_on_device(ttnn.Shape(list(a_shape[:-1]) + [n_out]), dtype, ttnn.TILE_LAYOUT, device)
    geom, grid, blk, _ = pd.plan(a, b, o, True, device, epilogue=epilogue)
    for t in (a, b, o):
        ttnn.deallocate(t)
    return geom, grid, blk


def _run(inputs, epilogue, device, dtype=ttnn.bfloat16, w_dtype=ttnn.bfloat16, extras=None):
    run_genify_matmul(
        inputs,
        dtype=dtype,
        weight_dtype=w_dtype,
        layout=ttnn.TILE_LAYOUT,
        fp32_dest_acc_en=True,
        placement="interleaved",
        epilogue=epilogue,
        device=device,
        extras=extras,
    )


@pytest.mark.parametrize(
    "a_shape, b_shape, epilogue, w_dtype",
    [
        ((1, 128, 1536), (1536, 5120), "none", ttnn.bfloat4_b),  # Qwen3.8 out_proj, M = 128
        ((1, 128, 5120), (5120, 4160), "none", ttnn.bfloat4_b),  # Qwen3.8 in_proj: 130 columns (not a bank multiple)
        ((1, 6144), (6144, 6144), "none", ttnn.bfloat16),  # FLUX linear_2, M = 1
        ((1, 256, 2048), (2048, 3072), "addcmul", ttnn.bfloat16),
        ((1, 256, 2048), (2048, 3072), "bias", ttnn.bfloat16),
    ],
)
def test_bank_selected_and_correct(a_shape, b_shape, epilogue, w_dtype, device):
    geom, grid, blk = _plan(device, a_shape, b_shape, epilogue, w_dtype=w_dtype)
    assert grid.placement == "bank" and pd.operand_mode(geom, grid, blk) == 2, f"{grid} {blk}"
    _run((a_shape, b_shape), epilogue, device, w_dtype=w_dtype, extras=LOFI)


def test_bank_ragged(device, monkeypatch):
    # Ragged M and N (94 columns: classes of 12 / 11 units), the bank split priced in (the cost model ranks the
    # flat split first on this shape).
    monkeypatch.setattr(pd, "BANK_LOCAL_DRAM_BYTES_PER_CYCLE", 10**6)
    a_shape, b_shape = (1, 200, 2048), (2048, 3008)
    geom, grid, blk = _plan(device, a_shape, b_shape, "bias_relu")
    assert grid.placement == "bank", f"{grid} {blk}"
    _run((a_shape, b_shape), "bias_relu", device)


def test_bank_fp32_a(device):
    _run(((1, 6144), (6144, 6144)), "none", device, dtype=ttnn.float32)


def test_bank_a_in_l1(device):
    # Qwen3.8 GDN layers keep A in L1 (the injectors read it from the L1 banks).
    _run(((1, 128, 4352), (4352, 5120)), "none", device, w_dtype=ttnn.bfloat4_b, extras={**LOFI, "act_buffer": "l1"})


@pytest.mark.parametrize("placement", ["near", "spread"])
def test_bank_placements(placement, device, monkeypatch):
    monkeypatch.setattr(pd, "BANK_LOCAL_PLACEMENT", placement)
    _run(((1, 128, 4352), (4352, 5120)), "bias", device, w_dtype=ttnn.bfloat8_b, extras=LOFI)


@pytest.mark.parametrize("injectors", [1, 2])
def test_bank_injectors(injectors, device, monkeypatch):
    monkeypatch.setattr(pd, "ACT_STREAM_INJECTORS", injectors)
    _run(((1, 256, 6144), (6144, 3072)), "addcmul", device)


def test_bank_row_fill_off(device, monkeypatch):
    # Home rows only (the spare cores idle with no columns: zero-width regions through the whole pipeline).
    monkeypatch.setattr(pd, "BANK_LOCAL_ROW_FILL", False)
    _run(((1, 128, 1536), (1536, 5120)), "none", device, w_dtype=ttnn.bfloat4_b, extras=LOFI)


def test_bank_swiglu(device, monkeypatch):
    # swiglu is off the bank split by default (measured slower); kept correct behind its knob.
    monkeypatch.setattr(pd, "BANK_LOCAL_SWIGLU", True)
    geom, grid, blk = _plan(device, (1, 128, 2048), (2048, 4096), "swiglu")
    assert grid.placement == "bank", f"{grid} {blk}"
    _run(((1, 128, 2048), (2048, 4096)), "swiglu", device)
