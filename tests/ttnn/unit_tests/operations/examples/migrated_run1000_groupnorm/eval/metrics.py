# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared accuracy measurement + check_output for golden tests.

Replaces the near-duplicate `check_output` / `CheckOutputError` /
`_classify_severity` blocks that previously lived in each
`eval/golden_tests/<op>/helpers.py`. One place to edit when a metric is
added, a threshold is tuned, or the severity policy changes.

Two layers:

- `compute_metrics(ttnn_output, expected)` — pure measurement, returns a
  dict of scalars: `pcc`, `rms`, `max_abs_diff`, `median_abs_diff`,
  `ulp_p99`, `has_inf`, `has_nan`. No thresholds, no exceptions. This is
  what the future pytest plugin will call on every test (pass and fail)
  to populate JUnit `<properties>` for DB ingest.

- `check_output(ttnn_output, expected, *, shape, dtype, expected_layout,
                tolerance=None)` — threshold-aware assertion. Validates
  shape/dtype/layout, calls `compute_metrics`, compares to `tolerance`,
  raises `CheckOutputError` on miss with severity tagged in the message.

Tolerance: a single `(pcc_target, rms_target)` tuple. Ops with their own
profile pass `tolerance=TOLERANCES[dtype]` from a per-op `TOLERANCES`
dict (per-op tolerances are the expected predominate path). When
omitted, falls back to `DEFAULT_TOLERANCES[dtype]`.

`ttnn.to_torch` dequantizes bfloat8_b to bfloat16 on readback, so ULP is
computed on the bf16-granularity readback the test actually sees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import ttnn

# Pytest plugin is optional: when imported outside a pytest context (e.g.
# from a probe script) the recorder simply no-ops. Wrapped in try so an
# environment without pytest installed still gets a working
# `check_output`.
try:
    from eval import metrics_plugin as _metrics_plugin
except ImportError:
    _metrics_plugin = None


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Fallback per-dtype thresholds used when `check_output` is called without
# an explicit `tolerance`. Ops that need their own profile should define a
# local `TOLERANCES = {...}` and pass `tolerance=TOLERANCES[dtype]` at the
# call site — the predominate path going forward.
DEFAULT_TOLERANCES = {
    ttnn.float32: (0.999, 0.01),
    ttnn.bfloat16: (0.995, 0.04),
    ttnn.bfloat8_b: (0.99, 0.10),
}

# Catastrophe floor. severity=bug when pcc <= floor or has_inf/has_nan.
# Otherwise (pcc above floor but still missing its target, or rms over
# target) severity=precision. PCC is the sole numeric bug discriminator
# — RMS quantifies how-bad-is-the-miss but does not promote to bug.
BUG_PCC_FLOOR = 0.95


# ---------------------------------------------------------------------------
# Pure measurement
# ---------------------------------------------------------------------------


def _ulp_p99(actual_f64: torch.Tensor, expected_f64: torch.Tensor, readback_dtype: torch.dtype) -> float:
    """P99 of per-element ULP distance at the readback dtype's granularity.

    Both tensors are quantized to the readback dtype before measurement.
    Returns NaN for unsupported dtypes (float16, integer types, etc.).
    """
    if readback_dtype == torch.float32:
        a_bits = actual_f64.to(torch.float32).view(torch.int32).to(torch.int64)
        e_bits = expected_f64.to(torch.float32).view(torch.int32).to(torch.int64)
        sign_offset = 1 << 31
    elif readback_dtype == torch.bfloat16:
        a_bits = actual_f64.to(torch.bfloat16).view(torch.int16).to(torch.int64)
        e_bits = expected_f64.to(torch.bfloat16).view(torch.int16).to(torch.int64)
        sign_offset = 1 << 15
    else:
        return float("nan")

    # Map IEEE sign-magnitude bits to a monotonically ordered integer so
    # |a-b| is the ULP distance. For non-negative bits, the int view is
    # already monotonic; for negative bits (sign bit set, i.e. int < 0
    # after view), the magnitude grows as the bit pattern grows, so we
    # mirror about zero.
    def _ordered(bits: torch.Tensor) -> torch.Tensor:
        sign = bits < 0
        magnitude = torch.where(sign, bits + sign_offset, bits)
        return torch.where(sign, -magnitude, magnitude)

    ulp = (_ordered(a_bits) - _ordered(e_bits)).abs().float()
    # Nearest-rank p99 via kthvalue (a selection, no full sort). torch.quantile sorts the entire
    # tensor and is ~10-15x slower at golden sizes (~110ms vs ~7ms at 1M elements) — and it's the
    # single largest cost of the per-test output comparison. kthvalue was already the >2**24
    # fallback; use it universally. The result differs from quantile's linear interpolation by
    # << 1 ULP at p99, and ulp_p99 is a recorded diagnostic (not a pass/fail gate), so the
    # definitional change is immaterial.
    n = ulp.numel()
    k = min(n, max(1, math.ceil(0.99 * n)))
    return float(ulp.flatten().kthvalue(k).values.item())


@dataclass
class Metrics:
    """All per-test accuracy values produced by `compute_metrics`.

    Internal API surface. Serialized to dict at wire boundaries
    (`metrics_plugin.record`, `classify_failures.parse_junit_xml`,
    SQL columns) via `dataclasses.asdict`.
    """

    pcc: float
    rms: float
    max_abs_diff: float
    median_abs_diff: float
    ulp_p99: float
    has_inf: bool
    has_nan: bool


def compute_metrics_torch(
    actual_torch: torch.Tensor, expected_torch: torch.Tensor, *, readback_dtype: torch.dtype
) -> Metrics:
    """Pure-torch counterpart to `compute_metrics`. Takes a torch readback
    + the dtype the readback is at (drives ULP granularity), no ttnn
    dependency.

    All metrics are computed via stdlib torch ops where one exists. The
    only custom math is `_ulp_p99` (no stdlib equivalent).
    """
    # float32 working set with float64 *accumulation* on the reductions
    # (matches models/common comp_pcc): no full float64 copy of either tensor
    # is materialized, so neither tensor pays a whole-tensor .to(float64) cast,
    # while matching a full-float64 correlation to
    # |Δ|<1e-8 across the high-PCC (>=0.999) range (precision loss only in the
    # ~10th decimal, far below any gate threshold).
    actual = actual_torch.to(torch.float32)
    expected_f = expected_torch.to(torch.float32)

    # inf/nan sanity on the op's output. Common path is all-finite, so probe
    # with a single isfinite().all() scan and only pay the two distinguishing
    # scans when something is actually wrong (matches models/common comp_pcc's
    # short-circuit; identical results, ~half the cost on the hot path).
    if bool(torch.isfinite(actual).all()):
        has_inf = has_nan = False
    else:
        has_inf = bool(torch.isinf(actual).any())
        has_nan = bool(torch.isnan(actual).any())

    # Pearson r via centered covariance with float64 accumulation. (torch.corrcoef
    # exposes no accumulation dtype and would force a float64 copy, so compute it
    # directly with dtype= on the sums.) NaN when either input has zero variance
    # (constant tensor) — collapse to 1.0, the lenient reading: an all-constant
    # reference with a matching constant output is perfectly correlated.
    a_flat = actual.flatten()
    e_flat = expected_f.flatten()
    n = a_flat.numel()
    a_c = a_flat - (a_flat.sum(dtype=torch.float64) / n).to(a_flat.dtype)
    e_c = e_flat - (e_flat.sum(dtype=torch.float64) / n).to(e_flat.dtype)
    cov = (a_c * e_c).sum(dtype=torch.float64)
    denom = torch.sqrt(a_c.pow(2).sum(dtype=torch.float64) * e_c.pow(2).sum(dtype=torch.float64))
    pcc = (cov / denom).item()
    if math.isnan(pcc):
        pcc = 1.0

    # Absolute RMS = sqrt(mean squared error), float64 accumulation over the
    # float32 diff. Relative-RMS divides by the reference's stddev; falls back
    # to the absolute value when stddev is tiny (constant reference).
    diff = actual - expected_f
    abs_rms = math.sqrt((diff.pow(2).sum(dtype=torch.float64) / n).item())
    scale = expected_f.std().item()
    rms = abs_rms / scale if scale > 1e-12 else abs_rms

    abs_diff = diff.abs()
    max_abs_diff = abs_diff.max().item()
    median_abs_diff = abs_diff.median().item()

    if has_inf or has_nan:
        ulp_p99 = float("nan")
    else:
        ulp_p99 = _ulp_p99(actual, expected_f, readback_dtype)

    return Metrics(
        pcc=pcc,
        rms=rms,
        max_abs_diff=max_abs_diff,
        median_abs_diff=median_abs_diff,
        ulp_p99=ulp_p99,
        has_inf=has_inf,
        has_nan=has_nan,
    )


def compute_metrics(ttnn_output, expected) -> Metrics:
    """Measure accuracy of `ttnn_output` against the torch `expected`.

    No thresholds, no exceptions — always returns a Metrics dataclass.
    The pytest plugin calls this (via `check_output`) on every test
    (pass and fail) to record per-test metrics into JUnit XML / the DB.
    """
    actual_raw = ttnn.to_torch(ttnn_output)
    return compute_metrics_torch(
        actual_raw,
        expected,
        readback_dtype=actual_raw.dtype,
    )


# ---------------------------------------------------------------------------
# Threshold-aware assertion
# ---------------------------------------------------------------------------


@dataclass
class Tolerance:
    """Per-metric thresholds for `check_output`.

    Every field is optional — only the ones that are set are gated.
    Comparison direction is metric-dependent:

      pcc:             actual_pcc           >= pcc            (higher = better)
      rms:             actual_rms           <= rms
      max_abs_diff:    actual_max_abs       <= max_abs_diff
      median_abs_diff: actual_median_abs    <= median_abs_diff
      ulp_p99:         actual_ulp_p99       <= ulp_p99
      allclose_rtol,
      allclose_atol:   torch.allclose(actual, expected, rtol, atol)
                       (per-element check; set either or both — the
                       unset one falls back to torch's default)

    Inf/NaN are always gated (separate from this dataclass — they're
    universal sanity checks). PCC is also always evaluated for the
    BUG_PCC_FLOOR severity gate, even when `pcc` is None here.

    Backwards compat: `(pcc, rms)` tuples and `None` are coerced to
    `Tolerance` by `check_output`, so the 9 op TOLERANCES dicts that
    pass tuples continue to work unchanged.
    """

    pcc: float | None = None
    rms: float | None = None
    max_abs_diff: float | None = None
    median_abs_diff: float | None = None
    ulp_p99: float | None = None
    allclose_rtol: float | None = None
    allclose_atol: float | None = None


# Metric direction: True = higher is better (gate via `actual >= target`),
# False = lower is better (gate via `actual <= target`).
_METRIC_HIGHER_IS_BETTER = {"pcc": True}
_METRIC_FIELDS = ("pcc", "rms", "max_abs_diff", "median_abs_diff", "ulp_p99")


_CONTRACT_METRICS = Metrics(
    pcc=0.0,
    rms=float("inf"),
    max_abs_diff=float("inf"),
    median_abs_diff=float("inf"),
    ulp_p99=float("inf"),
    has_inf=False,
    has_nan=False,
)


def _coerce_tolerance(tolerance, dtype) -> Tolerance:
    """Normalize the caller-facing tolerance forms to a `Tolerance`:

    None          → `Tolerance(pcc=…, rms=…)` from DEFAULT_TOLERANCES
    (pcc, rms)    → `Tolerance(pcc=…, rms=…)`
    Tolerance     → passthrough
    """
    if tolerance is None:
        pcc, rms = DEFAULT_TOLERANCES[dtype]
        return Tolerance(pcc=pcc, rms=rms)
    if isinstance(tolerance, Tolerance):
        return tolerance
    if isinstance(tolerance, tuple):
        if len(tolerance) != 2:
            raise ValueError(f"Tuple tolerance must be (pcc, rms); got {tolerance!r}")
        pcc, rms = tolerance
        return Tolerance(pcc=pcc, rms=rms)
    raise TypeError(f"tolerance must be None, (pcc, rms) tuple, or Tolerance; " f"got {type(tolerance).__name__}")


class CheckOutputError(AssertionError):
    """Raised when output misses tolerances or violates shape/dtype/layout
    contract. Stamps `severity=<precision|bug>` and all measured metrics
    into the assertion message — `eval/classify_failures.py` greps for
    `severity=...` to categorize, and the ingest plugin parses
    individual metrics from JUnit `<properties>` (not the message).

    Attributes:
        severity:  "precision" or "bug"
        metrics:   `Metrics` dataclass (zeroed for contract fails)
        tolerance: `Tolerance` dataclass used for the comparison
    """

    def __init__(self, *, severity: str, metrics: Metrics, tolerance: Tolerance, extra: str = ""):
        self.severity = severity
        self.metrics = metrics
        self.tolerance = tolerance
        m = metrics
        msg = (
            f"CheckOutputError severity={severity} "
            f"pcc={m.pcc:.6f} "
            f"rms={m.rms:.6f} "
            f"max_abs={m.max_abs_diff:.6g} "
            f"median_abs={m.median_abs_diff:.6g} "
            f"ulp_p99={m.ulp_p99:.3g} "
            f"inf={m.has_inf} nan={m.has_nan}"
        )
        if extra:
            msg += f" {extra}"
        super().__init__(msg)


def _classify_severity(metrics: Metrics) -> str:
    """PCC alone is the numeric bug discriminator. Inf/NaN also bug.

    Tolerance is not consulted — BUG_PCC_FLOOR is universal: a run where
    PCC falls below the floor is "this implementation is wrong" no
    matter what gates the caller asked for.
    """
    if metrics.has_inf or metrics.has_nan:
        return "bug"
    if metrics.pcc <= BUG_PCC_FLOOR:
        return "bug"
    return "precision"


def _gate_metrics(metrics: Metrics, tolerance: Tolerance, actual_torch, expected_torch) -> list[str]:
    """Run every gate that's set on `tolerance` against `metrics`. Returns
    a list of human-readable miss descriptions (empty list = all gates
    pass). Inf/NaN are always-on gates handled by the caller.
    """
    misses: list[str] = []
    for name in _METRIC_FIELDS:
        target = getattr(tolerance, name)
        if target is None:
            continue
        actual = getattr(metrics, name)
        if _METRIC_HIGHER_IS_BETTER.get(name, False):
            if actual < target:
                misses.append(f"{name}={actual:.6g} < target {target:.6g}")
        else:
            if actual > target:
                misses.append(f"{name}={actual:.6g} > target {target:.6g}")
    if tolerance.allclose_rtol is not None or tolerance.allclose_atol is not None:
        # torch defaults: rtol=1e-5, atol=1e-8. Match here when caller
        # only specified one half of the pair.
        rtol = tolerance.allclose_rtol if tolerance.allclose_rtol is not None else 1e-5
        atol = tolerance.allclose_atol if tolerance.allclose_atol is not None else 1e-8
        a = actual_torch.float()
        e = expected_torch.float()
        if not torch.allclose(a, e, rtol=rtol, atol=atol):
            misses.append(f"allclose(rtol={rtol}, atol={atol}) failed")
    return misses


def check_output(ttnn_output, expected, *, shape, dtype, expected_layout, tolerance=None):
    """Validate shape/dtype/layout + numerical accuracy.

    Args:
        ttnn_output: the device tensor to check.
        expected: torch reference, any float dtype.
        shape: expected output shape (list/tuple).
        dtype: expected ttnn output dtype.
        expected_layout: expected ttnn output layout.
        tolerance: `Tolerance` dataclass, `(pcc, rms)` tuple, or `None`.
            Tuple and `None` are coerced — see `_coerce_tolerance`. Pass
            a `Tolerance` directly to gate additional metrics (ULP P99,
            max-abs, allclose, ...) per-call.

    Returns:
        The `Metrics` dataclass on success (useful for tests that want
        the measured values even on pass).

    Raises:
        CheckOutputError on shape/dtype/layout mismatch (severity=bug)
        or numerical miss (severity=bug when pcc <= BUG_PCC_FLOOR or
        Inf/NaN present, otherwise severity=precision).
    """
    tolerance = _coerce_tolerance(tolerance, dtype)

    if list(ttnn_output.shape) != list(shape):
        raise CheckOutputError(
            severity="bug",
            metrics=_CONTRACT_METRICS,
            tolerance=tolerance,
            extra=f"shape_mismatch: got {list(ttnn_output.shape)} expected {list(shape)}",
        )
    if ttnn_output.dtype != dtype:
        raise CheckOutputError(
            severity="bug",
            metrics=_CONTRACT_METRICS,
            tolerance=tolerance,
            extra=f"dtype_mismatch: got {ttnn_output.dtype} expected {dtype}",
        )
    if ttnn_output.layout != expected_layout:
        raise CheckOutputError(
            severity="bug",
            metrics=_CONTRACT_METRICS,
            tolerance=tolerance,
            extra=f"layout_mismatch: got {ttnn_output.layout} expected {expected_layout}",
        )

    actual_torch = ttnn.to_torch(ttnn_output)
    metrics = compute_metrics_torch(
        actual_torch,
        expected,
        readback_dtype=actual_torch.dtype,
    )
    if _metrics_plugin is not None:
        # Tag with the input shape so the dashboard SHAPE column shows just the
        # shape (e.g. "1x1x32x64") rather than the full parametrize id. Golden
        # case ids embed the axes by construction; the axes now live in the
        # per-row chips (axes_json), so repeating them in SHAPE is noise. This
        # mirrors the translated suites, which already record a shape tag.
        _metrics_plugin.record(metrics, tag="x".join(str(d) for d in shape))

    misses = _gate_metrics(metrics, tolerance, actual_torch, expected)
    if metrics.has_inf:
        misses.append("Inf in output")
    if metrics.has_nan:
        misses.append("NaN in output")

    if not misses:
        return metrics

    severity = _classify_severity(metrics)
    raise CheckOutputError(
        severity=severity,
        metrics=metrics,
        tolerance=tolerance,
        extra="; ".join(misses),
    )
