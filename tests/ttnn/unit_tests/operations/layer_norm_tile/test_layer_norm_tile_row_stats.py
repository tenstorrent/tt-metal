# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 3 row-statistics paths: streamed windows, one-pass variance, per-window normalize.

One-pass variance (E[z^2] - mean^2) is the risk on large-mean rows, so those run under both DEST modes on
every split class the host gates on: balanced (shifted one-pass + per-window normalize), unbalanced L1 (streamed
one-pass without residual, else resident two-pass) and unbalanced DRAM (resident two-pass). Odd widths cover
the odd-chunk window sizes.
"""

import importlib

import pytest
import torch
import ttnn

from models.common.metrics import compute_pcc
from ttnn.operations.layer_norm_tile import layer_norm
from tests.ttnn.unit_tests.operations.layer_norm_tile.test_layer_norm_tile import (
    DRAM,
    L1,
    compute_config,
    make_affine,
    reference_layer_norm,
    to_device,
)

descriptor = importlib.import_module("ttnn.operations.layer_norm_tile.layer_norm_tile_program_descriptor")

TOLERANCE = {True: (0.999, 0.02), False: (0.998, 0.05)}  # (PCC floor, rel-RMS ceiling) by fp32_dest_acc_en


def run(device, x, res, *, buf, fp32_dest_acc_en=True, math_fidelity=ttnn.MathFidelity.HiFi4):
    W = x.shape[-1]
    gamma = torch.randn(W).to(torch.bfloat16)
    beta = torch.randn(W).to(torch.bfloat16)
    out = layer_norm(
        to_device(x, device, buf),
        make_affine(gamma, device, ttnn.ROW_MAJOR_LAYOUT, L1),
        make_affine(beta, device, ttnn.ROW_MAJOR_LAYOUT, L1),
        to_device(res, device, buf) if res is not None else None,
        compute_kernel_config=compute_config(fp32_dest_acc_en, math_fidelity),
    )
    expected = reference_layer_norm(x, gamma, beta, res)
    actual = ttnn.to_torch(out).float()
    assert torch.isfinite(actual).all()
    return expected, actual, beta


def check(expected, actual, fp32_dest_acc_en):
    pcc_floor, rms_ceil = TOLERANCE[fp32_dest_acc_en]
    rel_rms = (torch.sqrt(torch.mean((actual - expected) ** 2)) / expected.std()).item()
    pcc = compute_pcc(actual, expected)
    assert pcc >= pcc_floor and rel_rms <= rms_ceil, f"pcc {pcc} rel_rms {rel_rms}"


def precision(expected, actual):
    rel_rms = (torch.sqrt(torch.mean((actual - expected) ** 2)) / expected.std()).item()
    return compute_pcc(actual, expected), rel_rms


@pytest.mark.parametrize("fp32_dest_acc_en", [True, False], ids=["fp32acc", "bf16acc"])
@pytest.mark.parametrize("with_residual", [True, False], ids=["residual", "no_residual"])
@pytest.mark.parametrize(
    "shape,buf",
    [
        pytest.param((8, 1, 512, 768), L1, id="unbalanced_8x512_L1"),
        pytest.param((11, 1, 320, 768), DRAM, id="balanced_11x320_DRAM"),
        pytest.param((8, 1, 512, 768), DRAM, id="unbalanced_8x512_DRAM"),
    ],
)
def test_large_mean(device, monkeypatch, shape, buf, with_residual, fp32_dest_acc_en):
    """|mean| = 100, std 1: no path may be less precise than the resident two-pass statistics.

    The two-pass path itself misses the randn gate here (rel-RMS ~0.02 fp32, ~0.08 bf16: the centering reads
    z and the mean through SrcA / SrcB at reduced precision), so it is the bar. The shifted one-pass path
    (residual, balanced split, fp32 DEST) centers small values and must also meet the randn gate.
    """
    torch.manual_seed(0)
    x = (100.0 + torch.randn(shape)).to(torch.bfloat16)
    res = torch.randn(shape).to(torch.bfloat16) if with_residual else None
    torch.manual_seed(1)
    expected, actual, _ = run(device, x, res, buf=buf, fp32_dest_acc_en=fp32_dest_acc_en)
    with monkeypatch.context() as patch:
        patch.setattr(descriptor, "STREAM_ROW_STATS", False)
        torch.manual_seed(1)
        base_expected, base_actual, _ = run(device, x, res, buf=buf, fp32_dest_acc_en=fp32_dest_acc_en)
    pcc, rel_rms = precision(expected, actual)
    base_pcc, base_rel_rms = precision(base_expected, base_actual)
    assert pcc >= base_pcc - 1e-5 and rel_rms <= 1.05 * base_rel_rms + 1e-4, (pcc, rel_rms, base_pcc, base_rel_rms)
    if with_residual and fp32_dest_acc_en and buf == DRAM and shape[0] == 11:
        check(expected, actual, fp32_dest_acc_en)


@pytest.mark.parametrize(
    "shape,buf", [((8, 1, 512, 768), L1), ((11, 1, 320, 768), DRAM)], ids=["streamed", "per_window"]
)
def test_constant_rows(device, shape, buf):
    """var = 0: the one-pass variance must clamp its rounding residue, so the output is beta, not NaN."""
    x = torch.full(shape, 3.0).to(torch.bfloat16)
    _, actual, beta = run(device, x, None, buf=buf)
    torch.testing.assert_close(actual, beta.float().expand_as(actual), atol=0.05, rtol=0.02)


@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.LoFi], ids=["HiFi4", "LoFi"])
@pytest.mark.parametrize(
    "width",
    [
        pytest.param(32, id="Wt1_one_window"),
        pytest.param(96, id="Wt3_odd_whole_row_window"),
        pytest.param(640, id="Wt20_C5_two_chunk_windows"),
    ],
)
def test_widths(device, width, math_fidelity):
    torch.manual_seed(0)
    shape = (4, 1, 256, width)
    x = torch.randn(shape).to(torch.bfloat16)
    res = torch.randn(shape).to(torch.bfloat16)
    # DRAM residual on a balanced split (32 rows, 32 cores): the streamed path under HiFi4.
    expected, actual, _ = run(device, x, res, buf=DRAM, math_fidelity=math_fidelity)
    assert compute_pcc(actual, expected) >= 0.995
