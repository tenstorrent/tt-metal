# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm — the authoritative precision characterization (Refinement 1).

Phase 0 shipped `test_rms_norm_precision_baseline.py`, which pins the ONE
precision corner Phase 0 supported (fp32_dest_acc_en=True, HiFi4).  Refinement 1
grows the precision surface to the whole TARGET rectangle:

    dtype             x {bfloat16, float32, bfloat8_b}
    fp32_dest_acc_en  x {True, False}          ({float32, False} is EXCLUDED)
    math_fidelity     x {HiFi4, HiFi3, HiFi2, LoFi}   (ungated — never in SUPPORTED)

so this file is where that rectangle is measured.  Every metric is PRINTED for
every case; only PCC is asserted (per /numeric-formats-metal §11).

Split into two parametrizations on purpose — the full 4-way cross product would
be ~200 distinct kernel builds, and `math_fidelity` is an ungated axis whose
effect is shape-independent:

  * `test_rms_norm_precision_matrix`   — the gated axes (dtype x fp32_dest_acc_en)
    swept over 8 shapes x 2 input distributions, at the default HiFi4.
  * `test_rms_norm_precision_fidelity` — the ungated `math_fidelity` axis swept
    over all 4 values on 2 representative shapes.

Skips are minimal and mirror the op's own refusal surface exactly:
  * {float32, fp32_dest_acc_en=False} — EXCLUSIONS in rms_norm.py.
  * bfloat8_b on a non-tile-aligned shape — INVALID in the golden
    feature_spec.py (block quantization + a masked/padded reduce).

Run:
    scripts/run_safe_pytest.sh --run-all \
        tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_precision_matrix.py
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import check_with_pcc
from ttnn.operations.rms_norm import rms_norm

# bfloat8_b has no torch counterpart — build it from bf16 and let ttnn quantize.
TORCH_DTYPE = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
    ttnn.bfloat8_b: torch.bfloat16,
}

# Same gates the golden suite uses (eval/golden_tests/rms_norm/helpers.py
# TOLERANCES) — one source of truth for "what counts as correct at this dtype".
PCC_GATE = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995, ttnn.bfloat8_b: 0.99}
RMS_GATE = {ttnn.float32: 0.02, ttnn.bfloat16: 0.04, ttnn.bfloat8_b: 0.10}

TILE = 32

SHAPES = [
    pytest.param((32, 32), id="32x32_small"),
    pytest.param((32, 64), id="32x64"),
    pytest.param((2, 64, 256), id="2x64x256"),
    pytest.param((1, 1, 512, 1024), id="512x1024"),
    pytest.param((1, 1, 32, 8192), id="wide_32x8192_crosscore"),
    pytest.param((32, 50), id="32x50_W_non_aligned"),
    pytest.param((48, 64), id="48x64_H_non_aligned"),
    pytest.param((47, 100), id="47x100_both_non_aligned"),
]

# The two shapes the fidelity sweep runs on: one core-local, one cross-core.
FIDELITY_SHAPES = [
    pytest.param((1, 1, 512, 1024), id="512x1024"),
    pytest.param((1, 1, 32, 8192), id="wide_32x8192_crosscore"),
]


def _is_tile_aligned(shape) -> bool:
    return shape[-1] % TILE == 0 and shape[-2] % TILE == 0


def _skip_if_refused(shape, dtype, fp32_acc):
    """Skip exactly the cells the op refuses — nothing wider."""
    if dtype == ttnn.float32 and not fp32_acc:
        pytest.skip("EXCLUSIONS: {float32, fp32_dest_acc_en=False} is a permanent refusal")
    if dtype == ttnn.bfloat8_b and not _is_tile_aligned(shape):
        pytest.skip("INVALID (feature_spec.py): bfloat8_b on a non-tile-aligned shape")


def _reference(x: torch.Tensor, gamma, epsilon: float) -> torch.Tensor:
    xf = x.to(torch.float32)
    out = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + epsilon)
    if gamma is not None:
        out = out * gamma.to(torch.float32).reshape(-1)
    return out


def _metrics(got: torch.Tensor, true: torch.Tensor) -> dict:
    g, t = got.to(torch.float32), true.to(torch.float32)
    abs_err = (g - t).abs()
    denom = t.pow(2).mean().sqrt().clamp(min=1e-10)
    return {
        "max_abs": abs_err.max().item(),
        "mean_abs": abs_err.mean().item(),
        "median_abs": abs_err.median().item(),
        "p99_abs": torch.quantile(abs_err.flatten().float(), 0.99).item(),
        "rel_rms": (abs_err.pow(2).mean().sqrt() / denom).item(),
    }


def _run_cell(device, shape, dtype, fp32_acc, math_fidelity, distribution, epsilon=1e-6, pcc_gate=None):
    """Build tensors, dispatch, print every metric, assert PCC only."""
    torch.manual_seed(0)
    torch_dtype = TORCH_DTYPE[dtype]
    gen = torch.rand if distribution == "rand" else torch.randn
    torch_input = gen(*shape).to(torch_dtype)
    torch_gamma = gen(shape[-1]).to(torch_dtype)

    expected = _reference(torch_input, torch_gamma, epsilon)

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_gamma = ttnn.from_torch(
        torch_gamma.reshape(1, 1, 1, shape[-1]), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
    )

    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = math_fidelity
    cfg.fp32_dest_acc_en = fp32_acc
    cfg.math_approx_mode = False

    got = ttnn.to_torch(rms_norm(tt_input, gamma=tt_gamma, epsilon=epsilon, compute_kernel_config=cfg))

    m = _metrics(got, expected)
    gate = PCC_GATE[dtype] if pcc_gate is None else pcc_gate
    passed, pcc_msg = check_with_pcc(expected, got.to(expected.dtype), pcc=gate)
    _, allclose_msg = comp_allclose(expected, got.to(expected.dtype))
    print(
        f"\n[precision] shape={tuple(shape)} dtype={dtype} fp32_acc={fp32_acc} "
        f"fidelity={math_fidelity} dist={distribution}\n"
        f"    {pcc_msg}\n"
        f"    {allclose_msg}\n"
        f"    max_abs={m['max_abs']:.4e} mean_abs={m['mean_abs']:.4e} "
        f"median_abs={m['median_abs']:.4e} p99_abs={m['p99_abs']:.4e} "
        f"rel_rms={m['rel_rms']:.4e} (gate {RMS_GATE[dtype]})"
    )
    assert passed, pcc_msg
    return m


@pytest.mark.parametrize("distribution", [pytest.param("rand", id="uniform"), pytest.param("randn", id="normal")])
@pytest.mark.parametrize("fp32_acc", [pytest.param(True, id="fp32_acc"), pytest.param(False, id="bf16_acc")])
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(ttnn.bfloat16, id="bf16"),
        pytest.param(ttnn.float32, id="fp32"),
        pytest.param(ttnn.bfloat8_b, id="bfp8"),
    ],
)
@pytest.mark.parametrize("shape", SHAPES)
def test_rms_norm_precision_matrix(device, shape, dtype, fp32_acc, distribution):
    """dtype x fp32_dest_acc_en over 8 shapes x 2 distributions, at HiFi4."""
    _skip_if_refused(shape, dtype, fp32_acc)
    _run_cell(device, shape, dtype, fp32_acc, ttnn.MathFidelity.HiFi4, distribution)


@pytest.mark.parametrize(
    "math_fidelity",
    [
        pytest.param(ttnn.MathFidelity.HiFi4, id="HiFi4"),
        pytest.param(ttnn.MathFidelity.HiFi3, id="HiFi3"),
        pytest.param(ttnn.MathFidelity.HiFi2, id="HiFi2"),
        pytest.param(ttnn.MathFidelity.LoFi, id="LoFi"),
    ],
)
@pytest.mark.parametrize("fp32_acc", [pytest.param(True, id="fp32_acc"), pytest.param(False, id="bf16_acc")])
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(ttnn.bfloat16, id="bf16"),
        pytest.param(ttnn.float32, id="fp32"),
        pytest.param(ttnn.bfloat8_b, id="bfp8"),
    ],
)
@pytest.mark.parametrize("shape", FIDELITY_SHAPES)
def test_rms_norm_precision_fidelity(device, shape, dtype, fp32_acc, math_fidelity):
    """The ungated `math_fidelity` axis, all 4 values, on 2 representative shapes.

    LoFi is expected to be materially worse than HiFi4 — that is hardware
    behavior, not a bug.  The dtype PCC gate still has to hold.
    """
    _skip_if_refused(shape, dtype, fp32_acc)
    _run_cell(device, shape, dtype, fp32_acc, math_fidelity, "randn")


# ---------------------------------------------------------------------------
# The exact perf-target config (Refinement 3's decode profile) as a NAMED cell.
# Refinement 1 exists to make this config dispatchable at all; pinning it here
# means a later phase cannot silently substitute an fp32_dest_acc_en=True proxy.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("W", [1024, 2304, 5120, 7168])
def test_rms_norm_perf_target_config(device, W):
    """bf16 / HiFi2 / fp32_dest_acc_en=False / TILE / bf16 TILE gamma.

    Soft precision gate for the perf group is pcc >= 0.9995 (feature_spec.py
    `_perf_case` extras), which is TIGHTER than the bf16 dtype gate, so assert
    that number here.
    """
    m = _run_cell(
        device,
        (1, 1, 32, W),
        ttnn.bfloat16,
        fp32_acc=False,
        math_fidelity=ttnn.MathFidelity.HiFi2,
        distribution="randn",
        pcc_gate=0.9995,
    )
    assert m["rel_rms"] < RMS_GATE[ttnn.bfloat16]
