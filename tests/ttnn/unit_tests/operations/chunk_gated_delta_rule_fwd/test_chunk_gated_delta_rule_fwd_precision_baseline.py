# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for chunk_gated_delta_rule_fwd (verifier, Phase 0).

Measures, per output (o, final_state, h, v_new, g_cumsum, A), against the float64 oracle run on the
dtype-rounded inputs:

  * PCC                         (assert_with_pcc, tests/ttnn/utils_for_testing.py)
  * max / mean abs error        (comp_allclose, models/common/utility_functions.py, + mean)
  * relative RMS error          (RMS error / reference stddev — the golden suite's `rms`)
  * ULP p99 at the output dtype (eval.metrics — the golden suite's `ulp_p99` diagnostic)
  * got/true ratio spread       (median and p5..p95 of actual/expected over |expected| > 1e-3·max):
                                 a tight cluster around a non-1.0 constant is a scale/structural
                                 bug; a broad spread centred on 1.0 is ordinary rounding noise.

The gate is the golden suite's per-dtype band (helpers.TOLERANCES); the numbers are printed (run
with -s) and recorded in verification_report.md.  Set CGDR_PRECISION_JSON=<path> to append the
rows as JSON lines.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from eval.metrics import compute_metrics_torch
from ttnn.operations.chunk_gated_delta_rule_fwd import chunk_gated_delta_rule_fwd

from eval.golden_tests.chunk_gated_delta_rule_fwd.helpers import (
    TOLERANCES,
    make_reference_inputs,
    pytorch_chunk_gated_delta_rule_fwd,
    quantize,
)

SEED = 0
OUTPUT_NAMES = ("o", "final_state", "h", "v_new", "g_cumsum", "A")

# ((B, T, H, K, V), chunk_size) — small / ragged / wide_v largest-state / long ragged LLM head.
SHAPES = [
    ((1, 64, 1, 64, 64), 64),
    ((1, 100, 2, 64, 64), 64),
    ((1, 256, 4, 128, 256), 64),
    ((1, 1000, 4, 128, 128), 64),
]

_TORCH = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}


def _ratio_spread(got, exp):
    mask = exp.abs() > 1e-3 * exp.abs().max().clamp_min(1e-30)
    if mask.sum() < 16:
        return float("nan"), float("nan"), float("nan")
    r = (got[mask] / exp[mask]).float()
    q = torch.quantile(r[: 1 << 22], torch.tensor([0.05, 0.5, 0.95]))
    return q[1].item(), q[0].item(), q[2].item()


@pytest.mark.parametrize("state_mode", ["no_h0", "with_h0"])
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("case", SHAPES, ids=["x".join(map(str, s)) + f"_c{c}" for s, c in SHAPES])
def test_precision_baseline(case, dtype, state_mode, device):
    shape, chunk = case
    ref = quantize(make_reference_inputs(shape, state_mode, seed=SEED), dtype)
    expected = pytorch_chunk_gated_delta_rule_fwd(
        ref["q"], ref["k"], ref["v"], ref["g"], ref["beta"], initial_state=ref["initial_state"], chunk_size=chunk
    )
    dev = {
        n: (
            None
            if t is None
            else ttnn.from_torch(
                t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        )
        for n, t in ref.items()
    }
    outputs = chunk_gated_delta_rule_fwd(
        dev["q"], dev["k"], dev["v"], dev["g"], dev["beta"], initial_state=dev["initial_state"], chunk_size=chunk
    )

    pcc_t, rms_t = TOLERANCES[dtype]
    rows = []
    for idx, name in enumerate(OUTPUT_NAMES):
        got = ttnn.to_torch(outputs[idx]).double()
        exp = expected[idx].double()
        assert torch.isfinite(got).all(), f"{name}: non-finite output"
        m = compute_metrics_torch(got, exp, readback_dtype=_TORCH[dtype])
        _, allclose_msg = comp_allclose(exp, got)
        mean_abs = (got - exp).abs().mean().item()
        r_med, r_p5, r_p95 = _ratio_spread(got, exp)
        rows.append(
            {
                "shape": "x".join(map(str, shape)) + f"_c{chunk}",
                "dtype": "fp32" if dtype == ttnn.float32 else "bf16",
                "state_mode": state_mode,
                "output": name,
                "pcc": m.pcc,
                "max_abs": m.max_abs_diff,
                "mean_abs": mean_abs,
                "rel_rms": m.rms,
                "ulp_p99": m.ulp_p99,
                "ratio_median": r_med,
                "ratio_p5": r_p5,
                "ratio_p95": r_p95,
            }
        )
        print(
            f"[precision] {rows[-1]['shape']:22s} {rows[-1]['dtype']} {state_mode:7s} {name:12s} "
            f"pcc={m.pcc:.7f} max_abs={m.max_abs_diff:.3e} mean_abs={mean_abs:.3e} rel_rms={m.rms:.3e} "
            f"ulp_p99={m.ulp_p99:.0f} ratio med={r_med:.5f} [p5 {r_p5:.4f}, p95 {r_p95:.4f}] ({allclose_msg})"
        )
        assert_with_pcc(exp, got, pcc=pcc_t)
        assert m.rms <= rms_t, f"{name}: rel_rms={m.rms:.4g} > {rms_t}"

    path = os.environ.get("CGDR_PRECISION_JSON")
    if path:
        with open(path, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
