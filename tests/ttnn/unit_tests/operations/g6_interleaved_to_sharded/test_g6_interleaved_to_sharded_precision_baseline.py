# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for g6_interleaved_to_sharded.

The op is graded bit-exact (helpers.TOLERANCES[bfloat16] == ("exact", None)), so the baseline
expects PCC 1, zero absolute and relative-RMS error, 0 ULP, and a got/true ratio of exactly 1.
The metrics are still measured, not assumed, so a future path through a cast or an engine
shows up as a number here before a golden cell trips on it.

Cases are the SUPPORTED placements the acceptance test already pins (BB-22): production p1 in
both orientations, the 8-core (32, 640) placement, and the (33, 5120) regime pin that runs both
read spans in one block.
"""

from __future__ import annotations

import copy

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose, comp_pcc

from eval.golden_tests.g6_interleaved_to_sharded.feature_spec import FLUX_58
from eval.golden_tests.g6_interleaved_to_sharded.helpers import (
    build_memory_config,
    create_ttnn_input_tensor,
    make_torch_input,
)
from ttnn.operations.g6_interleaved_to_sharded import g6_interleaved_to_sharded


def _variant(shape=None, **out_memory):
    scenario = copy.deepcopy(FLUX_58)
    if shape is not None:
        scenario["shape"] = list(shape)
    scenario["out_memory"].update(out_memory)
    return scenario


CASES = {
    "p1_1x5120_8x4_row_major": _variant(),
    "p1_1x5120_8x4_col_major": _variant(orientation="col_major"),
    "p1_1x5120_8cores_640": _variant(grid=[[0, 0], [7, 0]], shard_shape=[32, 640]),
    "regime_pin_33x5120_8x4": _variant(shape=(33, 5120), shard_shape=[64, 160]),
}


def _bf16_ulp(actual, expected):
    """Distance in representable bfloat16 steps, via the sign-magnitude -> ordered-int map."""

    def ordered(t):
        bits = t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)

    return (ordered(actual) - ordered(expected)).abs()


def _metrics(actual, expected):
    _, pcc = comp_pcc(expected, actual, pcc=1.0)
    _, allclose = comp_allclose(expected, actual)
    a, e = actual.to(torch.float64), expected.to(torch.float64)
    diff = a - e
    rel_rms = float(diff.pow(2).mean().sqrt() / e.std())
    ulp = _bf16_ulp(actual, expected)
    nonzero = e != 0
    ratio = (a[nonzero] / e[nonzero]).flatten()
    return {
        "pcc": float(pcc),
        "max_abs": float(diff.abs().max()),
        "mean_abs": float(diff.abs().mean()),
        "rel_rms": rel_rms,
        "max_ulp": int(ulp.max()),
        "ratio_median": float(ratio.median()),
        "ratio_p5": float(ratio.quantile(0.05)),
        "ratio_p95": float(ratio.quantile(0.95)),
        "allclose": allclose,
    }


@pytest.mark.parametrize("case", list(CASES))
def test_precision_baseline(case, device):
    scenario = CASES[case]
    torch.manual_seed(0)
    torch_input = make_torch_input(scenario["shape"], scenario["dtype"])
    ttnn_input = create_ttnn_input_tensor(
        torch_input,
        device,
        dtype=scenario["dtype"],
        layout=scenario["layout"],
        memory_config=build_memory_config(scenario["in_memory"], device),
    )
    output_config = build_memory_config(scenario["out_memory"], device)
    actual = ttnn.to_torch(g6_interleaved_to_sharded(ttnn_input, output_config))

    m = _metrics(actual, torch_input)
    print(
        f"\nPRECISION {case} shape={scenario['shape']}: PCC={m['pcc']:.6f} max_abs={m['max_abs']:.3e} "
        f"mean_abs={m['mean_abs']:.3e} rel_rms={m['rel_rms']:.3e} max_ulp={m['max_ulp']} "
        f"ratio median={m['ratio_median']:.6f} p5={m['ratio_p5']:.6f} p95={m['ratio_p95']:.6f} "
        f"[{m['allclose']}]"
    )
    # Exact copy: anything above zero error is a value-changing path, not rounding.
    assert m["max_ulp"] == 0
    assert m["max_abs"] == 0.0 and m["rel_rms"] == 0.0
    assert m["pcc"] == pytest.approx(1.0, abs=1e-12)
    assert m["ratio_p5"] == 1.0 and m["ratio_p95"] == 1.0
