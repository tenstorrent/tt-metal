# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Integer ULP distance: the metric an SFPU accuracy assertion can be written against.

Host-only — no device, no kernel, no ``ttnn``. ``passed_test(max_ulp=...)`` in
``utils.py`` is the gate built on this, and the per-op budget registry sits on that; each
layer stays separable so each is pinned by its own host tests.

Integer rather than the fractional ``|err| / ulp(golden)`` ``accuracy_metrics`` reports:
``ulp(golden)`` differs either side of a power of two, so a fractional "1 ULP" budget
admits two steps below a boundary and one above. That stays the better diagnostic; this
is the form a pass/fail line can use.

Each bit pattern maps to its rank in the format's ordered list of representable values,
so a distance of N means "N representable values apart". Four properties of that index:

* ``flush_subnormals`` collapses the subnormal band onto zero *and out of the ranking*
  (DAZ+FTZ), so the smallest normal is one step from zero. Defaults per dtype, not
  to ``True``.
* ``+0`` and ``-0`` coincide: the index is a signed rank of the magnitude.
* ``Inf`` is one step past the largest finite, but :func:`within_ulp` rejects
  finite-against-``Inf`` positionally, so no budget buys an overflow.
* ``NaN`` has no rank: the lane is :data:`UNMEASURABLE`, judged by
  :func:`nonfinite_mismatches`; ``sfpu_domains`` owns whose NaN sign may be asserted.

Ported from the ttnn helpers rather than imported (the test venv has neither ``ttnn``
nor ``models.common``), and corrected: the ttnn DAZ index mis-bases its negative half.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from .format_config import DataFormat
from .llk_params import format_dict
from .logger import logger

#: The formats with a *native* per-element ULP. Ask :func:`has_ulp_gate` rather than
#: testing membership here -- this is only the native half, and misses the proxy table
#: below. The coarser block floats and the MX formats share a block exponent across 16
#: elements, so a per-element count is not a property of the element and they keep the
#: block-aware lattice compares in utils.py; integers want bit equality.
ULP_FORMATS: Tuple[DataFormat, ...] = (
    DataFormat.Float16_b,
    DataFormat.Float16,
    DataFormat.Float32,
)

# Formats with no float dtype of their own that are still gated in a *proxy* format's ULP
# space -- ttnn's choice, and free here because `passed_test` has already cast a Bfp8_b
# tensor to bfloat16 before comparing. Three things to know before enrolling a budget on
# one:
#
# * **A Bfp8_b budget is denominated in bf16 steps, and one Bfp8_b step is two of them.**
#   Bfp8_b's 7 magnitude bits include an explicit leading 1, so it has 6 fractional bits
#   against bfloat16's 7. `max_ulp=N` therefore buys N/2 Bfp8_b steps, and an odd budget
#   buys the same as the even one below it.
# * **It does not forgive block quantization.** A small element in a wide block is
#   quantized by the shared exponent far more coarsely than bf16 would quantize it --
#   measured, a lane at 0.06 in a block with amax 2.77 is 69 bf16 steps -- and the budget
#   charges all of it. `near_zero_atol` cannot absorb that either: its band is a fraction
#   of the *tensor* maximum, and such a lane is only small relative to its own block. So a
#   Bfp8_b budget is usable only where the block quantization is exact; elsewhere the op
#   belongs on the tolerance arm. Charging for it is deliberate -- ORing the lattice
#   verdict in would mean `max_ulp` was not the enforced maximum for this format.
# * **The flush models disagree, latent until an op carries a Bfp8_b budget.** bf16 proxy
#   space collapses below 2**-126 while a Bfp8_b golden is flushed at
#   `golden_generators._FTZ_THRESHOLD`'s 1e-37 and the unpack model does not flush at all,
#   so a lane in [1.18e-38, 1e-37) would be charged steps against a golden the harness
#   calls zero. No current stimulus domain reaches it.
#
# Bfp4_b (2 fractional bits) and Bfp2_b (0) are deliberately absent: a bf16 step count
# would read every legal quantization as a 32- or 128-step error.
_ULP_PROXY_DTYPES: Dict[DataFormat, torch.dtype] = {
    DataFormat.Bfp8_b: torch.bfloat16,
}

#: Lanes whose golden is below this fraction of the tensor's dynamic range are the ones a
#: ``near_zero_atol`` floor may rescue. ttnn's ``measure_ulp_with_near_zero_atol`` uses the
#: same 1%: scale-relative and deliberately not tuned per op.
NEAR_ZERO_FRACTION = 1e-2

#: Returned for any lane where either side is NaN. Never compare it against a budget
#: directly -- ``-1 <= max_ulp`` holds for every budget. Use :func:`within_ulp`.
UNMEASURABLE = -1


@dataclass(frozen=True)
class _UlpDtype:
    """What the bit arithmetic needs to know about one torch float dtype."""

    bits_dtype: torch.dtype  # same-width signed int, for .view()
    wide_dtype: torch.dtype  # wider signed int, room for the signed rank
    sign_mask: int
    all_mask: int
    mantissa_bits: int


_ULP_DTYPES: Dict[torch.dtype, _UlpDtype] = {
    torch.bfloat16: _UlpDtype(
        bits_dtype=torch.int16,
        wide_dtype=torch.int32,
        sign_mask=0x8000,
        all_mask=0xFFFF,
        mantissa_bits=7,
    ),
    torch.float16: _UlpDtype(
        bits_dtype=torch.int16,
        wide_dtype=torch.int32,
        sign_mask=0x8000,
        all_mask=0xFFFF,
        mantissa_bits=10,
    ),
    torch.float32: _UlpDtype(
        bits_dtype=torch.int32,
        wide_dtype=torch.int64,
        sign_mask=0x80000000,
        all_mask=0xFFFFFFFF,
        mantissa_bits=23,
    ),
}

#: Whether the harness's datapath model flushes a format's subnormal band: bf16 and fp32
#: do, **fp16 does not**, and collapsing fp16's band would report up to 1023 steps as
#: zero. The real question is whether the producing *Dest* flushed, which the metric
#: cannot see, so the default is the answer that cannot hide error.
_FLUSHES_SUBNORMALS: Dict[torch.dtype, bool] = {
    torch.bfloat16: True,
    torch.float32: True,
    torch.float16: False,
}

# ttnn's sanity ceiling: 2**mantissa_bits is exactly one binade, so a budget past it says
# the two values are more than a factor of two apart in the normal range -- at which point
# ULP has stopped being the right metric and the op belongs on the tolerance one.
MAX_MEANINGFUL_ULP: Dict[torch.dtype, int] = {
    dtype: 1 << spec.mantissa_bits for dtype, spec in _ULP_DTYPES.items()
}

#: Mantissa width per measurement dtype, published so a caller can convert between a step
#: count and a relative error without re-deriving the table.
MANTISSA_BITS_FOR_ULP: Dict[torch.dtype, int] = {
    dtype: spec.mantissa_bits for dtype, spec in _ULP_DTYPES.items()
}

# Below these counts a quantile is not a percentile, so p95/p99 fall back to the max.
_MIN_LANES_FOR_P95 = 20
_MIN_LANES_FOR_P99 = 100


def _unsupported_dtype_error(
    caller: str, dtype: torch.dtype, supported: Any
) -> ValueError:
    """One spelling of the refusal, for the three entry points that make it."""
    return ValueError(
        f"{caller}: unsupported dtype {dtype}; supported: "
        f"{', '.join(str(d) for d in supported)}"
    )


def ulp_dtype(fmt: DataFormat) -> torch.dtype:
    """The torch dtype a ULP distance for *fmt* is measured in. Raises for a format with
    no per-element ULP, so a caller asking for a gate it cannot have finds out here."""
    if fmt in ULP_FORMATS:
        return format_dict[fmt]
    if fmt in _ULP_PROXY_DTYPES:
        return _ULP_PROXY_DTYPES[fmt]
    raise ValueError(
        f"{fmt.name} has no per-element ULP. ULP is defined for "
        f"{', '.join(f.name for f in ULP_FORMATS)}, and for "
        f"{', '.join(f.name for f in _ULP_PROXY_DTYPES)} in a proxy format's space; the "
        "coarser block floats and the MX formats share a block exponent and keep their "
        "lattice compare in utils.py, and the integer formats want bit equality."
    )


def has_ulp_gate(fmt: DataFormat) -> bool:
    """Whether *fmt* can be gated on a per-element step count at all.

    What :func:`ulp_dtype` answers by raising, for callers that fall back rather than
    fail -- and the question to ask instead of testing :data:`ULP_FORMATS` membership,
    which misses the proxy formats.
    """
    try:
        ulp_dtype(fmt)
    except ValueError:
        return False
    return True


def flushes_subnormals(dtype: torch.dtype) -> bool:
    """Whether collapsing *dtype*'s subnormal band is the right default; see the table.
    Raises rather than defaulting, since every supported dtype is an explicit row and a
    default could only fire for an unsupported one, on the error-hiding polarity."""
    try:
        return _FLUSHES_SUBNORMALS[dtype]
    except KeyError:
        raise _unsupported_dtype_error(
            "flushes_subnormals", dtype, _FLUSHES_SUBNORMALS
        ) from None


def _value_order_index(
    t: torch.Tensor, spec: _UlpDtype, *, flush_subnormals: bool
) -> torch.Tensor:
    """Signed rank of each element in *spec*'s ordered list of representable values.

    Magnitude rank 0 is zero, so the two signed zeros coincide; with *flush_subnormals*
    the whole ``exp == 0`` band ranks 0 and is removed from the ranking above it.
    Unflushed it is the same total order the ULP *sweep* walks
    (``strategies.structured._to_key``) and must not drift from it.
    """
    bits = t.contiguous().view(spec.bits_dtype).to(spec.wide_dtype) & spec.all_mask
    magnitude = bits & (spec.all_mask ^ spec.sign_mask)

    if flush_subnormals:
        # Every pattern with a zero exponent field has magnitude <= this, so the clamp
        # collapses the band and the subtraction closes the gap it leaves behind.
        subnormal_count = (1 << spec.mantissa_bits) - 1
        rank = torch.clamp(magnitude - subnormal_count, min=0)
    else:
        rank = magnitude

    negative = (bits & spec.sign_mask) != 0
    return torch.where(negative, -rank, rank).to(torch.int64)


def ulp_distance(
    golden: torch.Tensor,
    result: torch.Tensor,
    *,
    flush_subnormals: Optional[bool] = None,
) -> torch.Tensor:
    """Integer count of representable steps between *golden* and *result*, per element.

    Both tensors must already be in the output format's dtype; lanes where either side is
    NaN come back as :data:`UNMEASURABLE`. *flush_subnormals* defaults per dtype, and
    ``True`` forces the collapse for a caller that knows the producing Dest flushed.
    """
    if golden.dtype != result.dtype:
        raise ValueError(
            f"ulp_distance: dtype mismatch {golden.dtype} vs {result.dtype}; cast both to "
            "the output format's dtype first"
        )
    spec = _ULP_DTYPES.get(golden.dtype)
    if spec is None:
        raise _unsupported_dtype_error("ulp_distance", golden.dtype, _ULP_DTYPES)
    if golden.shape != result.shape:
        raise ValueError(
            f"ulp_distance: shape mismatch {tuple(golden.shape)} vs {tuple(result.shape)}"
        )

    if flush_subnormals is None:
        flush_subnormals = flushes_subnormals(golden.dtype)
    golden_rank = _value_order_index(golden, spec, flush_subnormals=flush_subnormals)
    result_rank = _value_order_index(result, spec, flush_subnormals=flush_subnormals)
    distance = (golden_rank - result_rank).abs()

    unmeasurable = torch.isnan(golden) | torch.isnan(result)
    return torch.where(unmeasurable, torch.full_like(distance, UNMEASURABLE), distance)


def nonfinite_mismatches(golden: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
    """Bool mask of lanes where the two sides disagree about being non-finite.

    *Positional* agreement only: exactly one side NaN, exactly one side infinite, or both
    infinite with opposite signs. NaN sign and payload are not judged; ``sfpu_domains``
    holds that rule.
    """
    golden_nan, result_nan = torch.isnan(golden), torch.isnan(result)
    golden_inf, result_inf = torch.isinf(golden), torch.isinf(result)
    sign_differs = torch.signbit(golden) != torch.signbit(result)
    return (
        (golden_nan ^ result_nan)
        | (golden_inf ^ result_inf)
        | (golden_inf & result_inf & sign_differs)
    )


def ulp_elementwise_valid(
    golden: torch.Tensor,
    result: torch.Tensor,
    max_ulp: int,
    *,
    near_zero_atol: Optional[float] = None,
    near_zero_fraction: float = NEAR_ZERO_FRACTION,
    flush_subnormals: Optional[bool] = None,
    selected: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-element ULP verdict, shaped like the mask ``passed_test`` already prints.

    A lane is valid when the two sides agree about being non-finite **and** either they
    are within *max_ulp* steps of each other, or the lane is near zero and its absolute
    error is within *near_zero_atol*. Both-NaN lanes are valid.

    "Near zero" is bounded two ways, and a lane has to satisfy both:

    * ``|golden| < near_zero_fraction * max|finite golden|``, so the band follows the
      stimulus;
    * ``|golden| <= near_zero_atol / near_zero_fraction``, so one large golden cannot
      widen the band across the tile. The relative bound alone is unbounded in absolute
      terms -- over a golden spanning [0, 1000] the cut lands at 10.0, and a lane at 1.0
      is rescued 7 steps over a 1-step budget. Tightening the atol does not close it: the
      step count it admits grows as ``1/|golden|``. The absolute cut is the magnitude at
      which the forgiven error is exactly ``near_zero_fraction`` of the reference.

    The floor is deliberately not applied at every magnitude -- an absolute tolerance at
    large magnitude is the format- and magnitude-blind gate a step count replaces. It is
    for the lanes where the reference crosses zero (``log`` near 1, ``sin`` near pi),
    where ``ulp(golden)`` collapses and any absolute error explodes into a step count
    that says nothing about the kernel.

    Returns ``(is_valid, distance, rescued)``. A reporting caller has to exclude
    *rescued*: those lanes hold the largest step counts by construction, so ranking every
    lane names one that passed and never mentions the one that failed.
    """
    distance = ulp_distance(golden, result, flush_subnormals=flush_subnormals)
    both_nan = torch.isnan(golden) & torch.isnan(result)
    in_budget = (distance >= 0) & (distance <= max_ulp)
    valid = in_budget | both_nan
    rescued = torch.zeros_like(valid)

    if near_zero_atol is not None:
        absolute_cut = near_zero_atol / near_zero_fraction
        # In float32, like the error compare below: both cuts are Python floats, so a
        # 16-bit `golden.abs()` would promote them onto the tensor's own lattice and round
        # each edge -- narrowing the relative `<` one and widening the absolute `<=` one.
        magnitude = golden.abs().to(torch.float32)
        # Scoped to the lanes under judgement: `dynamic_range` is the only verdict input
        # that is not elementwise, so an unscoped max let a large golden in a lane the
        # caller masked *out* widen the band applied to the lanes it masked *in*.
        in_scope = torch.isfinite(golden)
        if selected is not None:
            in_scope = in_scope & selected
        finite_golden = magnitude[in_scope]
        if finite_golden.numel() == 0:
            near_zero = torch.zeros_like(valid)
        else:
            dynamic_range = float(finite_golden.max())
            near_zero = (
                torch.ones_like(valid)
                if dynamic_range == 0.0
                else magnitude < near_zero_fraction * dynamic_range
            )
            near_zero = near_zero & (magnitude <= absolute_cut)
        absolute_error = (result.to(torch.float32) - golden.to(torch.float32)).abs()
        rescued = near_zero & (absolute_error <= near_zero_atol) & ~in_budget
        valid = valid | rescued

    ok = valid & ~nonfinite_mismatches(golden, result)
    return ok, distance, rescued & ok


def ulp_stats(
    distance: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> Dict[str, Any]:
    """Distribution of a ULP distance over the lanes *mask* selects.

    The aggregates cover the measurable lanes only; the unmeasurable ones are counted
    separately so they cannot quietly shrink a mean. *mask* is checked against
    *distance*'s shape, not broadcast: ``(1,)`` would judge every lane or none.
    """
    if mask is not None and mask.shape != distance.shape:
        raise ValueError(
            f"ulp_stats: mask shape {tuple(mask.shape)} does not match distance "
            f"{tuple(distance.shape)}"
        )
    flat = distance.reshape(-1).to(torch.int64)
    selected = (
        torch.ones_like(flat, dtype=torch.bool) if mask is None else mask.reshape(-1)
    )

    unmeasurable = int((selected & (flat < 0)).sum())
    measurable = selected & (flat >= 0)
    lanes = int(measurable.sum())
    if lanes == 0:
        return {
            "lanes": 0,
            "unmeasurable": unmeasurable,
            "max": 0,
            "mean": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "exact_frac": float("nan"),
            "worst_index": None,
        }

    positions = torch.nonzero(measurable, as_tuple=False).reshape(-1)
    values = flat[positions]
    worst = int(values.max())
    as_float = values.to(torch.float64)
    return {
        "lanes": lanes,
        "unmeasurable": unmeasurable,
        "max": worst,
        "mean": float(as_float.mean()),
        "p95": (
            float(torch.quantile(as_float, 0.95))
            if lanes >= _MIN_LANES_FOR_P95
            else float(worst)
        ),
        "p99": (
            float(torch.quantile(as_float, 0.99))
            if lanes >= _MIN_LANES_FOR_P99
            else float(worst)
        ),
        "exact_frac": float((values == 0).sum()) / lanes,
        "worst_index": int(positions[int(torch.argmax(values))]),
    }


def local_step(
    value: float,
    dtype: torch.dtype,
    *,
    toward: Optional[float] = None,
    flush_subnormals: Optional[bool] = None,
) -> float:
    """Size of one *counted* step of *dtype* at ``|value|``, as the metric counts it.

    Not ``nextafter``'s raw gap, because this figure is printed next to an integer step
    count and has to agree with it:

    * inside a collapsed band the step out of zero is to the smallest *normal*, which the
      raw gap understates by ``2**mantissa_bits``;
    * at the largest finite the gap upward is to ``Inf``, so the raw difference reads
      "1 ULP = inf"; the binade is the same downward;
    * the gap is not symmetric at a power of two, so *toward* names the value the step is
      counted to and a one-step bf16 result just below ``1.0`` reports the ``3.90625e-3``
      it crossed, not the ``7.8125e-3`` above. Direction comes from the *signed* delta --
      magnitudes lose it across zero -- and is not consulted at ``value == 0``, which has
      no downward neighbour of its own.
    """
    if not math.isfinite(value):
        return float("nan")
    if dtype not in _ULP_DTYPES:
        # Its own guard: an explicit flush_subnormals= skips flushes_subnormals()'s.
        raise _unsupported_dtype_error("local_step", dtype, _ULP_DTYPES)
    if flush_subnormals is None:
        flush_subnormals = flushes_subnormals(dtype)

    info = torch.finfo(dtype)
    magnitude = abs(value)
    # Zero is excluded before the sign is consulted: `copysign(1.0, 0.0)` is `+1.0`, so a
    # negative `toward` reads as downward, and `nextafter(+0.0, 0.0)` is `+0.0` -- which
    # printed `1 ULP = 0.000000e+00` with the band kept (fp16's default).
    heads_toward_zero = (
        magnitude != 0.0
        and toward is not None
        and math.isfinite(toward)
        and math.copysign(1.0, value) * (toward - value) < 0.0
    )

    if flush_subnormals and (
        magnitude < info.tiny or (magnitude == info.tiny and heads_toward_zero)
    ):
        # With the band compacted out, one counted step is `tiny` in both directions --
        # hence the boundary case: at exactly `tiny` the raw downward gap is the smallest
        # *subnormal*, understating the step by 2**mantissa_bits.
        return float(info.tiny)

    scalar = torch.tensor(magnitude, dtype=dtype)
    step_is_downward = float(scalar) == info.max or heads_toward_zero
    if step_is_downward:
        downward = torch.nextafter(scalar, torch.tensor(0.0, dtype=dtype))
        return float((scalar - downward).to(torch.float32))
    upward = torch.nextafter(scalar, torch.tensor(float("inf"), dtype=dtype))
    return float((upward - scalar).to(torch.float32))


def nonfinite_disagreement_summary(
    golden: torch.Tensor,
    result: torch.Tensor,
    fmt: Optional[DataFormat] = None,
    *,
    mask: Optional[torch.Tensor] = None,
) -> Optional[str]:
    """The first lane where the two sides disagree about being non-finite, or ``None``.

    The step count cannot describe this failure: a NaN lane is :data:`UNMEASURABLE` and
    drops out of the statistics, so a verdict that failed only on a missing NaN reports
    "max 0 ULP (budget 0)" -- true, and useless.
    """
    mismatched = nonfinite_mismatches(golden, result)
    if mask is not None:
        mismatched = mismatched & mask
    if not bool(mismatched.any()):
        return None
    index = int(torch.nonzero(mismatched.reshape(-1), as_tuple=False)[0])
    label = fmt.name if fmt is not None else str(golden.dtype)
    return (
        f"non-finite disagreement @ [{index}]: result "
        f"{float(result.reshape(-1)[index])!r} vs golden "
        f"{float(golden.reshape(-1)[index])!r} ({label}); "
        f"{int(mismatched.sum())} such lane(s)"
    )


def ulp_verdict_message(
    golden: torch.Tensor,
    result: torch.Tensor,
    distance: torch.Tensor,
    fmt: Optional[DataFormat] = None,
    *,
    mask: Optional[torch.Tensor] = None,
    max_ulp: Optional[int] = None,
    stats: Optional[Dict[str, Any]] = None,
    flush_subnormals: Optional[bool] = None,
    rescued: Optional[torch.Tensor] = None,
) -> str:
    """What a gate logs: the non-finite disagreement first, then the steps.

    One builder for both callers, so the gate and :func:`within_ulp` cannot describe the
    same verdict differently.
    """
    steps = ulp_failure_message(
        golden,
        result,
        distance,
        fmt,
        mask=mask,
        max_ulp=max_ulp,
        stats=stats,
        flush_subnormals=flush_subnormals,
        rescued=rescued,
    )
    disagreement = nonfinite_disagreement_summary(golden, result, fmt, mask=mask)
    return steps if disagreement is None else f"{disagreement}\n  {steps}"


def ulp_failure_message(
    golden: torch.Tensor,
    result: torch.Tensor,
    distance: torch.Tensor,
    fmt: Optional[DataFormat] = None,
    *,
    mask: Optional[torch.Tensor] = None,
    max_ulp: Optional[int] = None,
    stats: Optional[Dict[str, Any]] = None,
    flush_subnormals: Optional[bool] = None,
    rescued: Optional[torch.Tensor] = None,
) -> str:
    """One actionable line for the worst lane, plus the distribution behind it.

    *stats* lets a caller that already has them skip a second pass, and supplying it
    *asserts they came from the same* ``mask``. *flush_subnormals* carries the same
    obligation against *distance*, which fixed the count printed beside the step size.
    *rescued* is the lane mask the near-zero floor accepted, which a ranking caller has
    already excluded from *mask*; reporting it is what lets a DEBUG export tell a
    budget-carried pass from a floor-carried one.
    """
    if stats is None:
        stats = ulp_stats(distance, mask)
    label = fmt.name if fmt is not None else str(golden.dtype)
    floor = "" if rescued is None else f", {int(rescued.sum())} held by the floor"
    if stats["worst_index"] is None:
        if stats["unmeasurable"] == 0:
            # Not "nothing was comparable": every lane was measurable and the mask took
            # them all out, which is what a floor that rescued the whole tile looks like.
            return (
                f"no lane under judgement (the mask selected none of "
                f"{distance.numel()} lanes{floor}, {label})"
            )
        return (
            f"no measurable lane ({stats['unmeasurable']} unmeasurable{floor}, {label})"
        )

    index = stats["worst_index"]
    golden_value = float(golden.reshape(-1)[index])
    result_value = float(result.reshape(-1)[index])
    step = local_step(
        golden_value,
        golden.dtype,
        toward=result_value,
        flush_subnormals=flush_subnormals,
    )
    budget = "" if max_ulp is None else f" (budget {max_ulp})"
    return (
        f"max {stats['max']} ULP @ [{index}]{budget}: result {result_value!r} vs golden "
        f"{golden_value!r} (1 ULP = {step:.6e}, {label})\n"
        f"  over {stats['lanes']} lanes: mean {stats['mean']:.3f}, p95 {stats['p95']:.1f}, "
        f"p99 {stats['p99']:.1f}, {100.0 * stats['exact_frac']:.1f}% exact, "
        f"{stats['unmeasurable']} unmeasurable{floor}"
    )


def warn_if_threshold_unmeaningful(max_ulp: float, dtype: torch.dtype) -> bool:
    """Warn when a budget is past the point where ULP still means anything. Returns
    whether it warned, so a test can assert on the guard rather than on log text."""
    ceiling = MAX_MEANINGFUL_ULP.get(dtype)
    if ceiling is None or max_ulp <= ceiling:
        return False
    logger.warning(
        f"max_ulp={max_ulp} exceeds the largest meaningful ULP budget for {dtype} "
        f"({ceiling} = 2**mantissa_bits, one binade). Past it the two values are more than "
        "a factor of two apart in the normal range and the op belongs on the tolerance "
        "metric, not this one."
    )
    return True


def within_ulp(
    golden: torch.Tensor,
    result: torch.Tensor,
    *,
    max_ulp: int,
    fmt: Optional[DataFormat] = None,
    flush_subnormals: Optional[bool] = None,
    mask: Optional[torch.Tensor] = None,
    near_zero_atol: Optional[float] = None,
    near_zero_fraction: float = NEAR_ZERO_FRACTION,
) -> Tuple[bool, str]:
    """The whole verdict: non-finite positions agree, and every finite lane is in budget.

    The scalar form of the gate's verdict, for callers outside ``passed_test``. It routes
    through the same :func:`ulp_elementwise_valid` and forwards every knob that changes
    that verdict, so the two cannot drift into disagreeing about one tensor -- a
    passthrough rather than a shorter signature on purpose, since without
    *near_zero_atol* there are gate verdicts this could not reproduce at any argument.
    *max_ulp* is keyword-only because it is the one real magic number in the signature;
    *mask* selects the lanes under judgement.

    **Omitting** *fmt* skips the format allowlist as well as the label, and asserts the
    caller has already put the tensors on the lattice they mean: ``format_dict`` collapses
    the block floats onto ``torch.bfloat16`` and ``Tf32`` onto ``torch.float32``, so an
    unnamed ``Tf32`` tensor is measured in float32 steps and a one-step error reads ~8192.

    Returns ``(ok, message)``, worth logging on a pass too: it turns a functional test
    into an accuracy datapoint without changing its verdict.
    """
    if golden.shape != result.shape:
        return False, (
            f"shape mismatch: golden {tuple(golden.shape)} vs result "
            f"{tuple(result.shape)}"
        )
    if fmt is not None:
        # The label and the lattice have to be the same claim -- see the docstring.
        expected = ulp_dtype(fmt)
        if golden.dtype != expected:
            raise ValueError(
                f"within_ulp: {fmt.name} is measured in {expected}, but the tensors are "
                f"{golden.dtype}; cast both to the output format's dtype first"
            )
    warn_if_threshold_unmeaningful(max_ulp, golden.dtype)

    if mask is not None and mask.shape != golden.shape:
        # Checked, not broadcast -- see ulp_stats.
        raise ValueError(
            f"within_ulp: mask shape {tuple(mask.shape)} does not match golden "
            f"{tuple(golden.shape)}"
        )
    selected = torch.ones_like(golden, dtype=torch.bool) if mask is None else mask
    is_valid, distance, rescued = ulp_elementwise_valid(
        golden,
        result,
        max_ulp,
        near_zero_atol=near_zero_atol,
        near_zero_fraction=near_zero_fraction,
        flush_subnormals=flush_subnormals,
        selected=selected,
    )
    # Rank only the lanes under judgement, less any the floor accepted -- those hold the
    # biggest step counts by construction.
    ranked = selected & ~rescued
    message = ulp_verdict_message(
        golden,
        result,
        distance,
        fmt,
        mask=ranked,
        max_ulp=max_ulp,
        stats=ulp_stats(distance, ranked),
        flush_subnormals=flush_subnormals,
        # Only with a floor configured: otherwise every verdict in the suite would grow a
        # "0 held by the floor" that says nothing.
        rescued=None if near_zero_atol is None else rescued & selected,
    )
    return bool(torch.all(is_valid | ~selected)), message
