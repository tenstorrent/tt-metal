# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for `tilize` (verifier artifact).

`tilize` is a **bijection on byte positions** — no arithmetic — so at Phase 0
(bf16 in, bf16 out, no cast) the expected result is not "high PCC" but
**bit-identity**. This file measures the numbers anyway, in the shape the
verification report wants, so that:

  * the Phase-0 row of `changelog.md` carries measured values rather than an
    assertion that "a layout op must be exact", and
  * later refinements that DO introduce real numerics (the `dtype=` cast to
    bf8b/bf16, `float32`) have a zero-error baseline to regress against.

It also records the **got/true ratio spread** (`r = actual / expected` over the
finite, non-zero-reference elements). That is the scale-bug detector: a tight
cluster of `r` around a non-1.0 constant is a uniform scale/structural bug (a
broadcast/stride/CB mistake), which for this op would show up as a shuffled or
strided tile — exactly the `uint8` "every other row zero" signature the op
design calls out (§8.5). For a correct tilize, `r ≡ 1.0` for every element.

Only the Phase-0 SUPPORTED rectangle is measured here (bf16, rank 4, DRAM->DRAM,
interleaved, single-core, tile-aligned, 32x32 tiles); the golden suite owns
coverage, this file owns the numbers.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.tilize import tilize


# small / medium / medium-wide / larger — all inside Phase 0's SUPPORTED cell.
SHAPES = [
    (1, 1, 32, 32),  # single tile
    (1, 1, 64, 128),  # multi-tile
    (1, 1, 32, 512),  # wide-short (nt_h == 1: the width-block path)
    (1, 1, 512, 512),  # larger, multi-block
]

_ROWS = []


def _metrics(expected: torch.Tensor, actual: torch.Tensor) -> dict:
    """PCC-adjacent error metrics + the got/true ratio spread."""
    exp = expected.to(torch.float32).flatten()
    act = actual.to(torch.float32).flatten()

    abs_err = (act - exp).abs()
    denom = exp.pow(2).mean().sqrt()
    rel_rms = float((abs_err.pow(2).mean().sqrt() / denom)) if float(denom) > 0 else 0.0

    # got/true ratio over finite, non-zero-reference elements.
    mask = torch.isfinite(act) & torch.isfinite(exp) & (exp.abs() > 0)
    ratio = (act[mask] / exp[mask]) if int(mask.sum()) else torch.ones(1)
    r_med = float(ratio.median())
    r_p5, r_p95 = (float(torch.quantile(ratio, 0.05)), float(torch.quantile(ratio, 0.95)))

    return {
        "max_abs": float(abs_err.max()),
        "mean_abs": float(abs_err.mean()),
        "rel_rms": rel_rms,
        "ratio_median": r_med,
        "ratio_p5": r_p5,
        "ratio_p95": r_p95,
        "ratio_spread": r_p95 - r_p5,
        "exact": bool(torch.equal(act, exp)),
    }


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(str(d) for d in s))
def test_tilize_precision_baseline(device, shape):
    torch.manual_seed(42)
    torch_input = torch.randn(shape).bfloat16()

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # Phase-0 SUPPORTED accepts only the single-core value of the parameter.
    tt_output = tilize(tt_input, use_multicore=False)
    result = ttnn.to_torch(tt_output)

    pcc_msg = assert_with_pcc(torch_input.float(), result.float(), pcc=0.9999)
    allclose_pass, allclose_msg = comp_allclose(torch_input.float(), result.float(), rtol=0.0, atol=0.0)

    m = _metrics(torch_input, result)
    m["shape"] = "x".join(str(d) for d in shape)
    _ROWS.append(m)

    print(
        f"\n[precision] {m['shape']}  {pcc_msg}  {allclose_msg}\n"
        f"            max_abs={m['max_abs']:.3e} mean_abs={m['mean_abs']:.3e} "
        f"rel_rms={m['rel_rms']:.3e}\n"
        f"            got/true ratio: median={m['ratio_median']:.6f} "
        f"p5={m['ratio_p5']:.6f} p95={m['ratio_p95']:.6f} "
        f"spread={m['ratio_spread']:.3e}  exact={m['exact']}"
    )

    # tilize is a bijection: at bf16->bf16 the output must be BIT-identical, and
    # the ratio must be exactly 1.0 everywhere. A tight cluster at a non-1.0
    # constant would be a scale/structural bug, NOT precision noise.
    assert allclose_pass, f"tilize is a byte bijection but is not exact: {allclose_msg}"
    assert m["exact"], "tilize output is not bit-identical to the input"
    assert m["ratio_median"] == pytest.approx(1.0, abs=0.0)
    assert m["ratio_spread"] == pytest.approx(0.0, abs=0.0)


def test_print_precision_table():
    """Emit the markdown table the verification report / changelog carry."""
    if not _ROWS:
        pytest.skip("run the parametrized baseline first (same session)")
    print("\n| Shape | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | got/true median | got/true spread |")
    print("|---|---|---|---|---|---|---|")
    for r in _ROWS:
        print(
            f"| {r['shape']} | 1.000000 | {r['max_abs']:.3e} | {r['mean_abs']:.3e} | "
            f"{r['rel_rms']:.3e} | {r['ratio_median']:.6f} | {r['ratio_spread']:.3e} |"
        )
