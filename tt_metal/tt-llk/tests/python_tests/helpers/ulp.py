# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Integer ULP distance: the metric an SFPU accuracy assertion can be written against.

Host-only — no device, no kernel, no ``ttnn``. Nothing here is wired to a verdict yet:
this is the metric on its own, so that a ``passed_test(max_ulp=...)`` gate and a per-op
budget can be added on top of something already pinned by host tests.

Why an *integer* distance and not the fractional ``|err| / ulp(golden)`` that
``accuracy_metrics.compute_pointwise_metrics`` already reports: ``ulp(golden)`` is not the
same size above and below a power of two, so a single representable step across such a
boundary reads as 0.5 or 2.0 ULP and a budget of "1 ULP" stops being a well-defined
predicate. The fractional form remains the better *diagnostic* and keeps its place in the
sweeps and the CSVs; this is the form a pass/fail line can use.

The distance is measured over a **value-order index**: each bit pattern is mapped to its
rank in the format's ordered list of representable values, so a distance of N means
"N representable values apart". Three properties of that mapping are what the harness
needs:

* ``flush_subnormals`` (the default) collapses the subnormal band onto zero *and compacts
  it out of the ranking*, so the smallest normal is one step from zero. That is the SFPU's
  own number system (DAZ+FTZ), and ``UnarySFPUGolden`` already models the FP16 flush, so
  golden and metric agree. Without it a legitimately flushed result sitting next to a
  subnormal golden reads as a ``2**mantissa_bits`` error that is really a 0-step agreement.
* ``+0`` and ``-0`` are 0 steps apart, by construction rather than by a fixup: the index is
  a signed rank of the magnitude, and both zeros have magnitude rank 0. ``-0.0`` does not
  survive unpack in this harness and the pack path canonicalises it again, so the one step
  a raw bit ordering would report is an artefact of the encoding, not an error in a kernel.
* ``Inf`` is an ordinary bit pattern, one step past the largest finite. Reporting "1 step"
  for an overflow at the very top of the range is the honest answer, and a spurious ``Inf``
  anywhere else still lands a huge distance.

``NaN`` has no rank, so any lane involving one is reported as ``UNMEASURABLE`` and must be
judged separately: ``nonfinite_mismatches()`` is the positional check, and
``sfpu_domains.nan_sign_is_unspecified()`` owns the question of whose sign may be asserted
at all. This module deliberately does not encode any NaN-sign policy.

Ported from the ttnn helpers ``tests/ttnn/utils_for_testing.ulp_distance`` and
``tests/ttnn/unit_tests/operations/eltwise/eltwise_test_utils.ulp_distance_bf16_daz``
rather than imported, because the test venv has neither ``ttnn`` nor ``models.common``
(see ``tests/requirements.txt``). Generalised over the three
float formats, vectorised, and corrected on one point: the ttnn DAZ index bases its
negative half at ``0x7F7F`` while the positive half is based at ``0x7F80 - 0x7F``, which
puts every negative value ``2**mantissa_bits - 1`` steps too far from zero and inflates
every sign-crossing distance. ``test_ulp.py::test_flush_makes_smallest_normal_one_step_from_zero``
pins the crossing in both directions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from .format_config import DataFormat
from .llk_params import format_dict
from .logger import logger

# The float formats with a per-element ULP worth counting. The block floats are absent on
# purpose: their spacing is set by an exponent shared across 16 elements, so a per-element
# step count against the bfloat16 view of the block is not a property of the element. They
# keep the lattice compares already in ``utils.py``, which are the stronger, block-aware
# criterion. Integer formats are absent because "correct" there is bit equality.
ULP_FORMATS: Tuple[DataFormat, ...] = (
    DataFormat.Float16_b,
    DataFormat.Float16,
    DataFormat.Float32,
)

#: Returned for any lane where either side is NaN — NaN has no place in the value order.
#: A caller must never compare this against a budget directly: ``-1 <= max_ulp`` is true
#: for every budget. Use :func:`within_ulp`, or gate on ``dist >= 0`` yourself.
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
    torch.bfloat16: _UlpDtype(torch.int16, torch.int32, 0x8000, 0xFFFF, 7),
    torch.float16: _UlpDtype(torch.int16, torch.int32, 0x8000, 0xFFFF, 10),
    torch.float32: _UlpDtype(torch.int32, torch.int64, 0x80000000, 0xFFFFFFFF, 23),
}

# ttnn's sanity ceiling (``utils_for_testing.assert_with_ulp``): 2**mantissa_bits. A budget
# above this says the two values differ by more than an order of magnitude, at which point
# ULP has stopped being the right metric and the op belongs on the tolerance one.
MAX_MEANINGFUL_ULP: Dict[torch.dtype, int] = {
    dtype: 1 << spec.mantissa_bits for dtype, spec in _ULP_DTYPES.items()
}

# ttnn's guards in ``measure_ulp_with_near_zero_atol``: below these counts a quantile is
# not a percentile, so p95/p99 fall back to the max rather than inventing precision.
_MIN_LANES_FOR_P95 = 20
_MIN_LANES_FOR_P99 = 100


def ulp_dtype(fmt: DataFormat) -> torch.dtype:
    """The torch dtype a ULP distance for *fmt* is measured in.

    Raises ``ValueError`` for every format that has no per-element ULP, rather than
    silently measuring something else — a caller that asks for a gate it cannot have
    should find out at the call, not by reading a green result.
    """
    if fmt in ULP_FORMATS:
        return format_dict[fmt]
    raise ValueError(
        f"{fmt.name} has no per-element ULP. ULP is defined for "
        f"{', '.join(f.name for f in ULP_FORMATS)}; the block floats (Bfp*, Mx*) share a "
        "block exponent and keep their lattice compare in utils.py, and the integer "
        "formats want bit equality."
    )


def _value_order_index(
    t: torch.Tensor, spec: _UlpDtype, *, flush_subnormals: bool
) -> torch.Tensor:
    """Signed rank of each element in *spec*'s ordered list of representable values.

    Magnitude rank 0 is zero, so the two signed zeros coincide. With *flush_subnormals*
    the whole ``exp == 0`` band (zero and every subnormal) ranks 0 and the band is removed
    from the ranking above it, which makes the smallest normal rank 1.
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
    flush_subnormals: bool = True,
) -> torch.Tensor:
    """Integer count of representable steps between *golden* and *result*, per element.

    Both tensors must already be in the output format's dtype — ``passed_test`` casts
    them, so at the gate they are bit-comparable without further work here. Lanes where
    either side is NaN come back as :data:`UNMEASURABLE`.

    Returns an ``int64`` tensor shaped like the inputs.
    """
    if golden.dtype != result.dtype:
        raise ValueError(
            f"ulp_distance: dtype mismatch {golden.dtype} vs {result.dtype}; cast both to "
            "the output format's dtype first"
        )
    spec = _ULP_DTYPES.get(golden.dtype)
    if spec is None:
        raise ValueError(
            f"ulp_distance: unsupported dtype {golden.dtype}; supported: "
            f"{', '.join(str(d) for d in _ULP_DTYPES)}"
        )
    if golden.shape != result.shape:
        raise ValueError(
            f"ulp_distance: shape mismatch {tuple(golden.shape)} vs {tuple(result.shape)}"
        )

    golden_rank = _value_order_index(golden, spec, flush_subnormals=flush_subnormals)
    result_rank = _value_order_index(result, spec, flush_subnormals=flush_subnormals)
    distance = (golden_rank - result_rank).abs()

    unmeasurable = torch.isnan(golden) | torch.isnan(result)
    return torch.where(unmeasurable, torch.full_like(distance, UNMEASURABLE), distance)


def nonfinite_mismatches(golden: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
    """Bool mask of lanes where the two sides disagree about being non-finite.

    *Positional* agreement only. A lane is a mismatch when exactly one side is NaN, when
    exactly one side is infinite, or when both are infinite with opposite signs. NaN sign
    and payload are not judged: ``-NaN`` folds to the other operand on Wormhole and
    ``sfpu_domains`` holds the rule for when a NaN's sign may be asserted, so encoding a
    second opinion here would make four known-good ops fail their edge sweeps.
    """
    golden_nan, result_nan = torch.isnan(golden), torch.isnan(result)
    golden_inf, result_inf = torch.isinf(golden), torch.isinf(result)
    sign_differs = torch.signbit(golden) != torch.signbit(result)
    return (
        (golden_nan ^ result_nan)
        | (golden_inf ^ result_inf)
        | (golden_inf & result_inf & sign_differs)
    )


def ulp_stats(
    distance: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> Dict[str, Any]:
    """Distribution of a ULP distance over the lanes *mask* selects.

    ``max``/``mean``/``p95``/``p99``/``exact_frac`` are taken over the measurable lanes
    only; the unmeasurable ones are counted separately so they cannot quietly shrink a
    mean. ``worst_index`` is a flat index into the original tensor, which is what makes a
    failure reproducible.
    """
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


def local_step(value: float, dtype: torch.dtype) -> float:
    """Size of one representable step of *dtype* at ``|value|``.

    The same ``nextafter`` definition ``accuracy_metrics.local_ulp`` uses, for one scalar,
    so a failure message can say what a step is worth at the point that failed.
    """
    if not math.isfinite(value):
        return float("nan")
    magnitude = torch.tensor(abs(value), dtype=dtype)
    upward = torch.nextafter(magnitude, torch.tensor(float("inf"), dtype=dtype))
    return float((upward - magnitude).to(torch.float32))


def ulp_failure_message(
    golden: torch.Tensor,
    result: torch.Tensor,
    distance: torch.Tensor,
    fmt: Optional[DataFormat] = None,
    *,
    mask: Optional[torch.Tensor] = None,
    max_ulp: Optional[int] = None,
) -> str:
    """One actionable line for the worst lane, plus the distribution behind it.

    A ULP verdict is only useful if it names the point: what the hardware produced, what
    the reference was, and what one step is worth there.
    """
    stats = ulp_stats(distance, mask)
    label = fmt.name if fmt is not None else str(golden.dtype)
    if stats["worst_index"] is None:
        return f"no measurable lane ({stats['unmeasurable']} unmeasurable, {label})"

    index = stats["worst_index"]
    golden_value = float(golden.reshape(-1)[index])
    result_value = float(result.reshape(-1)[index])
    step = local_step(golden_value, golden.dtype)
    budget = "" if max_ulp is None else f" (budget {max_ulp})"
    return (
        f"max {stats['max']} ULP @ [{index}]{budget}: result {result_value!r} vs golden "
        f"{golden_value!r} (1 ULP = {step:.6e}, {label})\n"
        f"  over {stats['lanes']} lanes: mean {stats['mean']:.3f}, p95 {stats['p95']:.1f}, "
        f"p99 {stats['p99']:.1f}, {100.0 * stats['exact_frac']:.1f}% exact, "
        f"{stats['unmeasurable']} unmeasurable"
    )


def warn_if_threshold_unmeaningful(max_ulp: float, dtype: torch.dtype) -> bool:
    """Warn when a budget is past the point where ULP still means anything.

    Returns whether it warned, so a test can assert on the guard instead of on log text.
    """
    ceiling = MAX_MEANINGFUL_ULP.get(dtype)
    if ceiling is None or max_ulp <= ceiling:
        return False
    logger.warning(
        f"max_ulp={max_ulp} exceeds the largest meaningful ULP budget for {dtype} "
        f"({ceiling} = 2**mantissa_bits). Past it the two values differ by more than an "
        "order of magnitude and the op belongs on the tolerance metric, not this one."
    )
    return True


def within_ulp(
    golden: torch.Tensor,
    result: torch.Tensor,
    max_ulp: int,
    *,
    fmt: Optional[DataFormat] = None,
    flush_subnormals: bool = True,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[bool, str]:
    """The whole verdict: non-finite positions agree, and every finite lane is in budget.

    This is what a ``passed_test(max_ulp=...)`` gate is meant to call, so that the
    :data:`UNMEASURABLE` sentinel is never compared against a budget by hand. *mask*
    selects the lanes under judgement — pass one to exclude lanes an op's own edge rules
    have already settled.

    Returns ``(ok, message)``; the message is worth logging on a pass too, which turns a
    functional test into an accuracy datapoint without changing its verdict.
    """
    if golden.shape != result.shape:
        return False, (
            f"shape mismatch: golden {tuple(golden.shape)} vs result "
            f"{tuple(result.shape)}"
        )
    warn_if_threshold_unmeaningful(max_ulp, golden.dtype)

    selected = torch.ones_like(golden, dtype=torch.bool) if mask is None else mask
    mismatched = nonfinite_mismatches(golden, result) & selected
    if bool(mismatched.any()):
        index = int(torch.nonzero(mismatched.reshape(-1), as_tuple=False)[0])
        label = fmt.name if fmt is not None else str(golden.dtype)
        return False, (
            f"non-finite disagreement @ [{index}]: result "
            f"{float(result.reshape(-1)[index])!r} vs golden "
            f"{float(golden.reshape(-1)[index])!r} ({label})"
        )

    distance = ulp_distance(golden, result, flush_subnormals=flush_subnormals)
    stats = ulp_stats(distance, selected)
    ok = stats["worst_index"] is None or stats["max"] <= max_ulp
    return ok, ulp_failure_message(
        golden, result, distance, fmt, mask=selected, max_ulp=max_ulp
    )
