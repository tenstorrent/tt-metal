# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for nomic_moe_route_dispatch.

The only floating-point output is combine_weights (fp32 probabilities of the op's own selection). It is
measured against the fp64 softmax at the SAME experts (PCC, max / mean abs error, relative RMS, fp32 ULP,
got / true ratio spread). The selection is measured against the fp64 top-2 (rerouted fraction). Everything
else (dispatched rows, combine_index, expert_counts) is exact and asserted by helpers.check_output.
"""

import pytest
import torch

import ttnn
from eval.golden_tests.nomic_moe_route_dispatch.helpers import (
    case_seed,
    check_output,
    make_weight,
    make_x,
    reference_selection,
    to_device,
)
from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.nomic_moe_route_dispatch import nomic_moe_route_dispatch

TILE = 32
TOP_K = 2
PCC_FLOOR = 0.9999


def _ulp_error(got32, true64):
    """|got - round_fp32(true)| in units of the fp32 spacing at the reference value."""
    true32 = true64.to(torch.float32)
    spacing = (torch.nextafter(true32, torch.full_like(true32, float("inf"))) - true32).double()
    return (got32.double() - true32.double()).abs() / spacing


@pytest.mark.parametrize(
    "batch,seq,profile",
    [(2, 512, "layer0"), (4, 512, "layer0"), (8, 384, "layer3"), (8, 512, "layer0"), (8, 512, "layer5")],
    ids=["2x512-layer0", "4x512-layer0", "8x384-layer3", "8x512-layer0", "8x512-layer5"],
)
def test_precision_baseline(device, batch, seq, profile):
    name = f"{batch}x{seq}"
    w = make_weight(profile)
    x = make_x((batch, 1, seq, 768), w, profile, seed=case_seed(name))
    x_dev, w_dev = to_device(x, w, device, in_placement="nd_norm")
    outputs = nomic_moe_route_dispatch(x_dev, w_dev)
    stats = check_output(outputs, x, w, device=device)  # exact parts + the contract's accuracy bars

    tokens = batch * seq
    t_pad = TILE * ((tokens + TILE - 1) // TILE)
    weights = ttnn.to_torch(outputs[2]).to(torch.float32).reshape(TOP_K, t_pad)[:, :tokens].T  # (T, 2)
    counts = ttnn.to_torch(outputs[3]).to(torch.int64).reshape(-1)
    index = ttnn.to_torch(outputs[1]).to(torch.int64).reshape(TOP_K, t_pad)[:, :tokens].T
    padded = TILE * ((counts + TILE - 1) // TILE)
    starts = torch.cumsum(padded, 0) - padded
    sel = torch.searchsorted(starts, index, right=True) - 1

    p64, _ = reference_selection(x, w)
    expected = torch.gather(p64, 1, sel)  # fp64 probabilities at the op's own selection

    assert_with_pcc(expected, weights.double(), PCC_FLOOR)
    _, allclose_msg = comp_allclose(expected, weights.double())
    err = (weights.double() - expected).abs()
    rel_rms = float(torch.sqrt((err**2).mean()) / torch.sqrt((expected**2).mean()))
    ulp = _ulp_error(weights, expected)
    nonzero = expected != 0
    ratio = weights.double()[nonzero] / expected[nonzero]
    q = torch.quantile(ratio, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))
    print(
        f"\nPRECISION {name}-{profile}: max_abs {err.max().item():.3e} mean_abs {err.mean().item():.3e} "
        f"rel_rms {rel_rms:.3e} ulp_max {ulp.max().item():.0f} ulp_median {ulp.median().item():.0f} "
        f"ratio p5/p50/p95 {q[0].item():.6f}/{q[1].item():.6f}/{q[2].item():.6f} "
        f"rerouted {stats['rerouted']}/{tokens} ({100 * stats['rerouted_fraction']:.3f}%) "
        f"worst_gap {stats['worst_reroute_gap']:.3e} | {allclose_msg}"
    )
