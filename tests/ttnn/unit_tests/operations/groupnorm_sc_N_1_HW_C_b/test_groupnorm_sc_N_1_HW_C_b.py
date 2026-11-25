# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Acceptance test for groupnorm_sc_N_1_HW_C_b. This file is the spec; do not modify it.

Phase 0 scope: bf16, TILE and ROW_MAJOR input, tile-aligned shapes, DRAM interleaved input.
The output is TILE, DRAM interleaved, with the input's dtype. gamma/beta are (1,1,1,C) bf16
ROW_MAJOR tensors in DRAM, in the three affine modes.

The design is op_design.md. What each test covers:
  * test_groupnorm_shapes: single tile, multi-tile, non-square, multi-batch, and group widths that
    are not a multiple of 32 (Cg = 4, 10, 20, 24, 56). These are where the on-device group map is
    exercised.
  * test_groupnorm_affine_modes: gamma_only and no_affine.
  * test_groupnorm_large: a combine fan-in above 32 cores (gather tile rows R >= 2),
    plus refiner- and VAE-like sizes.
  * test_groupnorm_residency_regimes: the regime pins, through TUNING_OVERRIDES:
      - pure streaming (max_resident_chunks=0);
      - partial tail residency (1, 5);
      - full residency (default);
    each with a 2-core cross-core combine.
  * test_groupnorm_multi_unit_per_core: U >= P (cpi = 1, several blocks per core).
  * test_groupnorm_validation_errors: the structural ValueErrors.
"""

import importlib

import pytest
import torch
import ttnn

from ttnn.operations.groupnorm_sc_N_1_HW_C_b import groupnorm_sc_N_1_HW_C_b
from eval.metrics import check_output
from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import (
    TOLERANCES,
    create_ttnn_input_tensor,
    pytorch_groupnorm_sc_N_1_HW_C,
)

_OP_PKG = importlib.import_module("ttnn.operations.groupnorm_sc_N_1_HW_C_b")

LAYOUTS = [
    pytest.param(ttnn.TILE_LAYOUT, id="TILE"),
    pytest.param(ttnn.ROW_MAJOR_LAYOUT, id="RM"),
]


def _make_inputs(shape, affine):
    torch.manual_seed(42)
    C = shape[-1]
    torch_input = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    gamma = beta = None
    if affine in ("gamma_beta", "gamma_only"):
        gamma = torch.randn((1, 1, 1, C), dtype=torch.float32).to(torch.bfloat16)
    if affine == "gamma_beta":
        beta = torch.randn((1, 1, 1, C), dtype=torch.float32).to(torch.bfloat16)
    return torch_input, gamma, beta


def _to_device_affine(t, device):
    if t is None:
        return None
    return create_ttnn_input_tensor(t, device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)


def _run_and_check(device, shape, num_groups, *, layout, affine="gamma_beta", eps=1e-5):
    torch_input, gamma, beta = _make_inputs(shape, affine)
    expected = pytorch_groupnorm_sc_N_1_HW_C(torch_input, num_groups, gamma=gamma, beta=beta, eps=eps)

    tt_input = create_ttnn_input_tensor(torch_input, device, dtype=ttnn.bfloat16, layout=layout)
    tt_gamma = _to_device_affine(gamma, device)
    tt_beta = _to_device_affine(beta, device)

    tt_output = groupnorm_sc_N_1_HW_C_b(tt_input, num_groups, gamma=tt_gamma, beta=tt_beta, eps=eps)

    # Output contract for an interleaved input: TILE, DRAM interleaved, same dtype and shape.
    out_cfg = tt_output.memory_config()
    assert out_cfg.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert out_cfg.buffer_type == ttnn.BufferType.DRAM
    check_output(
        tt_output,
        expected,
        shape=list(shape),
        dtype=ttnn.bfloat16,
        expected_layout=ttnn.TILE_LAYOUT,
        tolerance=TOLERANCES[ttnn.bfloat16],
    )


# (shape, num_groups). Cg = C / G is noted where it is not a multiple of 32.
SHAPES = [
    pytest.param((1, 1, 32, 32), 1, id="single_tile"),
    pytest.param((1, 1, 64, 128), 4, id="multi_tile_cg32"),
    pytest.param((1, 1, 256, 320), 32, id="cg10_two_spans"),
    pytest.param((1, 1, 64, 160), 8, id="cg20_one_span"),
    pytest.param((2, 1, 128, 192), 8, id="multi_batch_cg24"),
    pytest.param((1, 1, 128, 448), 8, id="nonsquare_cg56"),
    pytest.param((1, 1, 1024, 32), 8, id="tall_cg4"),
    pytest.param((4, 1, 256, 256), 8, id="multi_batch_cg32"),
    pytest.param((1, 1, 2048, 640), 32, id="mid_cg20"),
    pytest.param((8, 1, 64, 1024), 32, id="many_units"),
]


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("shape, num_groups", SHAPES)
def test_groupnorm_shapes(device, shape, num_groups, layout):
    _run_and_check(device, shape, num_groups, layout=layout, affine="gamma_beta")


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("affine", ["gamma_only", "no_affine"])
@pytest.mark.parametrize(
    "shape, num_groups",
    [
        pytest.param((1, 1, 256, 320), 32, id="cg10"),
        pytest.param((2, 1, 128, 192), 8, id="multi_batch_cg24"),
    ],
)
def test_groupnorm_affine_modes(device, shape, num_groups, affine, layout):
    _run_and_check(device, shape, num_groups, layout=layout, affine=affine)


@pytest.mark.parametrize(
    "shape, num_groups, eps",
    [
        pytest.param((1, 1, 8192, 32), 1, 1e-5, id="fan_in_gt_32"),
        pytest.param((1, 1, 16384, 1152), 32, 1e-5, id="refiner_cg36"),
        pytest.param((1, 1, 65536, 128), 32, 1e-6, id="vae_cg4"),
    ],
)
def test_groupnorm_large(device, shape, num_groups, eps):
    _run_and_check(device, shape, num_groups, layout=ttnn.TILE_LAYOUT, affine="gamma_beta", eps=eps)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    "max_resident_chunks",
    [
        pytest.param(0, id="stream_only"),
        pytest.param(1, id="partial_resident_1"),
        pytest.param(5, id="partial_resident_5"),
        pytest.param(None, id="fully_resident"),
    ],
)
def test_groupnorm_residency_regimes(device, monkeypatch, max_resident_chunks, layout):
    # With these overrides: 2 cores, one unit (G=1 gives one 2-tile span), cpi=2 (cross-core
    # combine), 16 rows per core in 1-row chunks of T=2 tiles, so B=16 chunks per block.
    monkeypatch.setitem(_OP_PKG.TUNING_OVERRIDES, "max_cores", 2)
    monkeypatch.setitem(_OP_PKG.TUNING_OVERRIDES, "chunk_tiles_target", 2)
    monkeypatch.setitem(_OP_PKG.TUNING_OVERRIDES, "max_resident_chunks", max_resident_chunks)
    _run_and_check(device, (1, 1, 1024, 64), 1, layout=layout, affine="gamma_beta")


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("max_resident_chunks", [pytest.param(None, id="resident"), pytest.param(1, id="partial")])
def test_groupnorm_multi_unit_per_core(device, monkeypatch, max_resident_chunks, layout):
    # 2 cores, U = N*gx >= 4 units, so cpi = 1 and every core runs several blocks back to back.
    # This exercises the per-block pointer cycles of the alias groups and the gather semaphore reset.
    monkeypatch.setitem(_OP_PKG.TUNING_OVERRIDES, "max_cores", 2)
    monkeypatch.setitem(_OP_PKG.TUNING_OVERRIDES, "chunk_tiles_target", 4)
    monkeypatch.setitem(_OP_PKG.TUNING_OVERRIDES, "max_resident_chunks", max_resident_chunks)
    _run_and_check(device, (4, 1, 256, 128), 4, layout=layout, affine="gamma_beta")


def test_groupnorm_validation_errors(device, expect_error):
    # Each message must contain the keyword in op_design.md "Parameters", matched case-insensitively.
    torch.manual_seed(42)

    def dev(t, layout=ttnn.TILE_LAYOUT):
        return create_ttnn_input_tensor(t.to(torch.bfloat16), device, dtype=ttnn.bfloat16, layout=layout)

    x = dev(torch.randn((1, 1, 32, 64)))

    # in_place is only valid for a sharded input.
    with expect_error(ValueError, r"(?i)in_place"):
        groupnorm_sc_N_1_HW_C_b(x, 2, in_place=True)

    # C % G != 0.
    with expect_error(ValueError, r"(?i)num_groups"):
        groupnorm_sc_N_1_HW_C_b(x, 3)

    # Rank != 4.
    with expect_error(ValueError, r"(?i)rank"):
        groupnorm_sc_N_1_HW_C_b(dev(torch.randn((1, 32, 64))), 2)

    # shape[1] != 1.
    with expect_error(ValueError, r"(?i)shape\[1\]"):
        groupnorm_sc_N_1_HW_C_b(dev(torch.randn((1, 2, 32, 64))), 2)

    # gamma / beta must be (1, 1, 1, C).
    bad_affine = _to_device_affine(torch.randn((1, 1, 1, 32)).to(torch.bfloat16), device)
    with expect_error(ValueError, r"(?i)gamma"):
        groupnorm_sc_N_1_HW_C_b(x, 2, gamma=bad_affine)
    with expect_error(ValueError, r"(?i)beta"):
        groupnorm_sc_N_1_HW_C_b(x, 2, beta=bad_affine)
