# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for gated_delta_net_backward (verifier artifact).

Measures, per shape x dtype x gradient, against the float64 autograd oracle in
`eval/golden_tests/gated_delta_net_backward/helpers.py`:

  * PCC                 (via `assert_with_pcc` / `comp_pcc`)
  * max abs error, mean abs error   (via `comp_allclose`)
  * relative RMS error
  * the got/true RATIO SPREAD — median and p5/p95 of `actual / expected` over
    the finite, non-negligible-reference elements.  This is the scale-bug
    detector: a tight cluster of the ratio around a NON-1.0 constant is a
    uniform scale / structural bug (fix the kernel), whereas a broad spread
    centred on 1.0 is ordinary rounding noise.  It is printed whenever PCC is
    high but relative RMS is not, which is exactly the signature that would
    otherwise be misfiled as a precision issue.

CONTRACT: `q` and `k` are L2-normalized along their last dim.  Without it the
UT transform is not contractive and the FORWARD this op differentiates diverges
(measured |o|max 2.9e18) — see the op prompt's Rules section.  Every input here
comes from the golden suite's `make_reference_inputs`, which normalizes.

The assertions are the loose "this op is not broken" band (the golden suite's
own per-dtype tolerances are the tight gate, in `helpers.py::TOLERANCES`); the
point of this file is the recorded TABLE, printed with `-s`.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from ttnn.operations.gated_delta_net_backward import gated_delta_net_backward

from eval.golden_tests.gated_delta_net_backward.helpers import (
    make_reference_inputs,
    pytorch_gated_delta_net_backward,
)
from models.common.utility_functions import comp_allclose, comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc

GRAD_NAMES = ("dq", "dk", "dv", "dg", "dbeta", "dh0")

# Same bands as the golden suite. Do not tighten here — this file reports.
PCC_FLOOR = {ttnn.float32: 0.999, ttnn.bfloat16: 0.99}

TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}

# (B, T, H, K, V), chunk_size — small / medium / medium-wide / larger.
SHAPES = [
    ((1, 32, 1, 32, 32), 32),  # smallest: one chunk, one head, one core
    ((1, 128, 2, 64, 64), 32),  # medium: 4 chunks, square heads
    ((1, 128, 2, 64, 128), 64),  # medium, wide_v, chunk 64
    ((1, 256, 4, 128, 256), 64),  # larger: largest state, num_v_blocks > 1
]
SHAPE_IDS = [f"B{s[0]}_T{s[1]}_H{s[2]}_K{s[3]}_V{s[4]}_c{c}" for s, c in SHAPES]


def _to_device(tensor, device, dtype):
    if tensor is None:
        return None
    return ttnn.from_torch(
        tensor.to(TORCH_DTYPE[dtype]),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _rel_rms(got: torch.Tensor, exp: torch.Tensor) -> float:
    denom = exp.double().pow(2).mean().sqrt()
    if denom == 0:
        return float(got.double().pow(2).mean().sqrt())
    return float((got.double() - exp.double()).pow(2).mean().sqrt() / denom)


def _ratio_spread(got: torch.Tensor, exp: torch.Tensor):
    """(median, p5, p95) of got/exp over elements whose reference is not noise.

    A tight cluster around a non-1.0 constant == uniform scale bug.
    A broad spread centred on 1.0 == rounding noise.
    """
    e = exp.double().flatten()
    a = got.double().flatten()
    keep = torch.isfinite(e) & torch.isfinite(a) & (e.abs() > 1e-3 * e.abs().max().clamp_min(1e-30))
    if int(keep.sum()) < 8:
        return float("nan"), float("nan"), float("nan")
    r = a[keep] / e[keep]
    q = torch.quantile(r, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))
    return float(q[1]), float(q[0]), float(q[2])


@pytest.mark.parametrize("shape,chunk_size", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16], ids=["fp32", "bf16"])
def test_precision_baseline(shape, chunk_size, dtype, device):
    B, T, H, K, V = shape
    # g_scale 0.02: the dht path needs a weak decay or |dh0| ~ exp(sum g)*|dht|
    # annihilates the state gradient and the measurement is on noise.
    ref = make_reference_inputs(shape, state_mode="with_h0_and_dht", seed=0, g_scale=0.02)

    expected = pytorch_gated_delta_net_backward(
        ref["q"],
        ref["k"],
        ref["v"],
        ref["g"],
        ref["beta"],
        ref["do"],
        dht=ref["dht"],
        initial_state=ref["h0"],
        chunk_size=chunk_size,
    )
    got = gated_delta_net_backward(
        _to_device(ref["q"], device, dtype),
        _to_device(ref["k"], device, dtype),
        _to_device(ref["v"], device, dtype),
        _to_device(ref["g"], device, dtype),
        _to_device(ref["beta"], device, dtype),
        _to_device(ref["do"], device, dtype),
        dht=_to_device(ref["dht"], device, dtype),
        initial_state=_to_device(ref["h0"], device, dtype),
        chunk_size=chunk_size,
    )

    tag = f"{SHAPE_IDS[SHAPES.index((shape, chunk_size))]}/{'fp32' if dtype == ttnn.float32 else 'bf16'}"
    print(f"\n=== PRECISION BASELINE {tag} ===")
    print(
        f"{'grad':>6} {'PCC':>12} {'max_abs':>12} {'mean_abs':>12} {'rel_RMS':>10} "
        f"{'ratio_med':>10} {'ratio_p5':>10} {'ratio_p95':>10}"
    )

    for name, dev_t, exp in zip(GRAD_NAMES, got, expected):
        assert exp is not None and dev_t is not None, f"{name}: missing gradient"
        host = ttnn.to_torch(dev_t).to(torch.float64)
        exp = exp.to(torch.float64)

        # comp_allclose carries the max/mean abs error text; parse the numbers
        # off the tensors directly for the table.
        _, allclose_msg = comp_allclose(exp, host, rtol=1e-2, atol=1e-2)
        max_abs = float((host - exp).abs().max())
        mean_abs = float((host - exp).abs().mean())
        rms = _rel_rms(host, exp)
        med, p5, p95 = _ratio_spread(host, exp)

        # PCC via the shared helpers, so the number is the harness's number.
        assert_with_pcc(exp, host, PCC_FLOOR[dtype])
        _, pcc = comp_pcc(exp, host, PCC_FLOOR[dtype])
        pcc = float(pcc)

        print(
            f"{name:>6} {pcc:12.8f} {max_abs:12.3e} {mean_abs:12.3e} {rms:10.3e} " f"{med:10.6f} {p5:10.6f} {p95:10.6f}"
        )
        if pcc > 0.999 and rms > 0.05:
            print(
                f"       ^^ {name}: PCC high but rel-RMS {rms:.3g} — check the ratio "
                f"spread above: a tight cluster off 1.0 is a SCALE bug, not precision. "
                f"({allclose_msg})"
            )
