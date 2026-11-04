# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for nomic_moe_combine (verifier, Phase 0).

Measures, against the float64 reference over the window's real tokens: PCC, max / mean absolute error, relative RMS
error, the bf16 ULP distance to the correctly rounded result (0 = one of the two roundings of the exact value, the
contract's bar), and the got/true ratio spread (the scale-bug detector: a tight cluster away from 1.0 would be a
structural/scale bug, not rounding). Prints one line per shape so the verification report can quote it.

Cell: bfloat16, expert_rows in L1, the output in the norm's ND-sharded layout, real_text routing (the perf cases').
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from eval.golden_tests.nomic_moe_combine.helpers import (
    TOLERANCES,
    bias_tensor,
    index_tensor,
    make_bias,
    make_expert_rows,
    output_config,
    prepare_combine_bias,
    reference,
    release,
    route,
    rows_for_tokens,
    rows_tensor,
    weights_tensor,
)
from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.nomic_moe_combine import nomic_moe_combine

HIDDEN = 768

SHAPES = [
    (1, 1000),  # smallest window, ragged (G = 2, r = 1)
    (2, 512),  # 1k perf guard (G = 2, r = 1)
    (4, 512),  # 2k perf guard (G = 3, r = 2)
    (8, 512),  # 4k perf focus (G = 3, r = 4)
]


def _bf16_ulp_distance(got_bf16: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    """|got - round(exact)| in bf16 ULPs, taking the nearer of the two acceptable roundings (ties even / away)."""
    exact32 = exact.float()
    even = exact32.to(torch.bfloat16)
    bits = exact32.contiguous().view(torch.int32)
    away = ((bits + 0x8000) & ~0xFFFF).view(torch.float32).to(torch.bfloat16)

    def ordinal(x):  # monotone integer ordering of bf16 bit patterns
        i = x.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -(i & 0x7FFF), i)

    g = ordinal(got_bf16)
    return torch.minimum((g - ordinal(even)).abs(), (g - ordinal(away)).abs())


@pytest.mark.parametrize("batch,seqlen", SHAPES, ids=[f"{b}x{s}" for b, s in SHAPES])
def test_precision_baseline(batch, seqlen, device):
    tokens = batch * seqlen
    shape = (batch, 1, seqlen, HIDDEN)
    plan = route(tokens, "real_text", 0)
    rows = rows_for_tokens(tokens)
    values = make_expert_rows(rows, plan["slot_row"], 0)
    bias = make_bias(0)
    config = output_config("norm_layout", shape, device)

    tt_rows = rows_tensor(values, device, "l1")
    tt_index = index_tensor(plan["combine_index"], device)
    tt_weights = weights_tensor(plan["combine_weights"], device)
    tt_bias = bias_tensor(bias, device)
    prepared = prepare_combine_bias(tt_bias, device)
    output = nomic_moe_combine(
        tt_rows, tt_index, tt_weights, prepared, num_tokens=tokens, batch=batch, memory_config=config
    )

    got_bf16 = ttnn.to_torch(output).reshape(-1, HIDDEN)[:tokens]
    expected = reference(values, bias, plan["combine_index"], plan["combine_weights"], tokens)  # float64
    got = got_bf16.double()

    err = got - expected
    max_abs = float(err.abs().max())
    mean_abs = float(err.abs().mean())
    rel_rms = float(err.pow(2).mean().sqrt() / expected.pow(2).mean().sqrt())
    ulp = _bf16_ulp_distance(got_bf16, expected)
    # ULPs are only meaningful away from cancellation: the TF32 weights' rounding (allowed by the contract) is relative
    # to the products, so a result that cancels to ~0 is many of its own ULPs off. Report the distribution and the
    # share of values within 1 ULP of the correctly rounded exact value; gate on PCC / relative RMS only.
    ulp_max, ulp_mean = int(ulp.max()), float(ulp.double().mean())
    ulp_p99 = float(torch.quantile(ulp.flatten()[:1_000_000].double(), 0.99))
    ulp0_frac = float((ulp == 0).double().mean())
    ulp1_frac = float((ulp <= 1).double().mean())

    nonzero = expected.abs() > 1e-3
    ratio = (got[nonzero] / expected[nonzero]).float()
    q = torch.quantile(
        ratio[torch.randperm(ratio.numel(), generator=torch.Generator().manual_seed(0))[:1_000_000]],
        torch.tensor([0.05, 0.5, 0.95]),
    )
    allclose_ok, allclose_msg = comp_allclose(expected.float(), got.float(), rtol=1.6e-2, atol=1e-2)

    pcc_bar, rms_bar = TOLERANCES[ttnn.bfloat16]
    passed, pcc_msg = assert_with_pcc(expected.float(), got.float(), pcc_bar)
    print(
        f"\nPRECISION {batch}x{seqlen}: pcc={pcc_msg} max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} "
        f"rel_rms={rel_rms:.5f} ulp_max={ulp_max} ulp_mean={ulp_mean:.4f} ulp_p99={ulp_p99:.0f} ulp0={ulp0_frac:.5f} "
        f"ulp<=1={ulp1_frac:.5f} "
        f"ratio_p5/p50/p95={q[0]:.5f}/{q[1]:.5f}/{q[2]:.5f} allclose(rtol=1.6e-2, atol=1e-2)={allclose_ok} "
        f"[{allclose_msg}]"
    )
    assert passed
    assert rel_rms <= rms_bar, f"relative RMS {rel_rms} over {rms_bar}"
    assert abs(float(q[1]) - 1.0) < 1e-3, "median got/true ratio is off 1.0: a scale bug, not rounding"
    # The one rounding: the bulk of values sit within 1 bf16 ULP of the correctly rounded exact value.
    assert ulp1_frac >= 0.98, f"only {ulp1_frac:.4f} of values within 1 ULP of the correctly rounded value"

    for tensor in (tt_rows, tt_index, tt_weights, tt_bias, output):
        if tensor.is_allocated():
            ttnn.deallocate(tensor)
    release(prepared)
