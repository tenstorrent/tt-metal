# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of g6_to_memory_config: one production case per built regime (BB-22).

The op moves whole tile pages, so the contract is bit equality (helpers.TOLERANCES is "exact" for every
dtype). The baseline still reports the usual metrics (PCC, max / mean abs error, relative RMS, max ULP
distance and the got/true ratio spread), so a later refinement that introduces any value change shows
up as a number, not only as a failed equality.
"""

import pytest
import torch
import ttnn

from eval.golden_tests.g6_to_memory_config.feature_spec import CASES
from eval.golden_tests.g6_to_memory_config.helpers import build_memory_config, make_torch_input
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.g6_to_memory_config import g6_to_memory_config, plan_of

# DRAM interleaved -> DRAM interleaved has no bfloat16 production case; the shape is Qwen 2077's.
_BOUNCE_CELL = {
    "shape": [16, 128, 1536],
    "dtype": ttnn.bfloat16,
    "in_memory": {"buffer": "dram", "scheme": "interleaved"},
    "out_memory": {"buffer": "dram", "scheme": "interleaved"},
}

BASELINE = {
    "Qwen 2058 pull_shard": (CASES[("Qwen", 2058)][0], "pull_shard"),
    "Qwen 2105 pull_bank": (CASES[("Qwen", 2105)][0], "pull_bank"),
    "Qwen 2071 push_shard": (CASES[("Qwen", 2071)][0], "push_shard"),
    "Qwen 2077-shape bounce": (_BOUNCE_CELL, "bounce"),
    "Panoptic 11 bf8b push_shard": (CASES[("Panoptic", 11)][0], "push_shard"),
    "Qwen 2178 bf4b bounce": (CASES[("Qwen", 2178)][0], "bounce"),
}


def _bf16_ulp_distance(actual, expected):
    """Max distance in bfloat16 units in the last place, on the monotonic integer image of the bits."""
    a = actual.to(torch.bfloat16).view(torch.int16).to(torch.int32)
    e = expected.to(torch.bfloat16).view(torch.int16).to(torch.int32)
    a = torch.where(a < 0, -32768 - a, a)
    e = torch.where(e < 0, -32768 - e, e)
    return int((a - e).abs().max().item())


def _metrics(actual, expected):
    _, pcc = comp_pcc(expected, actual)
    _, allclose_detail = comp_allclose(actual, expected)
    a = actual.to(torch.float32)
    e = expected.to(torch.float32)
    err = a - e
    std = e.std().item()
    rel_rms = (err.pow(2).mean().sqrt().item() / std) if std > 0 else float("nan")
    mask = torch.isfinite(e) & (e != 0)
    ratio = a[mask] / e[mask]
    return {
        "pcc": float(pcc),
        "allclose": allclose_detail,
        "max_abs": err.abs().max().item(),
        "mean_abs": err.abs().mean().item(),
        "rel_rms": rel_rms,
        "max_ulp": _bf16_ulp_distance(actual, expected),
        "ratio_median": ratio.median().item(),
        "ratio_p5": ratio.quantile(0.05).item(),
        "ratio_p95": ratio.quantile(0.95).item(),
    }


@pytest.mark.parametrize("case_id", list(BASELINE))
def test_precision_baseline(device, case_id):
    scenario, regime = BASELINE[case_id]
    torch.manual_seed(0)
    tt_input = ttnn.from_torch(
        make_torch_input(scenario["shape"], scenario["dtype"]),
        dtype=scenario["dtype"],
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=build_memory_config(scenario["in_memory"]),
    )
    out_config = build_memory_config(scenario["out_memory"])
    assert plan_of(tt_input, out_config)["regime"] == regime

    expected = ttnn.to_torch(tt_input)
    actual = ttnn.to_torch(g6_to_memory_config(tt_input, out_config))
    m = _metrics(actual, expected)
    print(
        f"PRECISION {case_id} shape={scenario['shape']} pcc={m['pcc']:.6f} max_abs={m['max_abs']:.3g} "
        f"mean_abs={m['mean_abs']:.3g} rel_rms={m['rel_rms']:.3g} max_ulp={m['max_ulp']} "
        f"ratio median={m['ratio_median']:.6f} p5={m['ratio_p5']:.6f} p95={m['ratio_p95']:.6f} [{m['allclose']}]"
    )

    assert torch.equal(actual, expected), f"{(actual != expected).sum().item()} values differ"
    assert m["pcc"] == 1.0 and m["max_abs"] == 0.0 and m["rel_rms"] == 0.0 and m["max_ulp"] == 0
