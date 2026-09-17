# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Integer ULP distance: the metric an SFPU accuracy assertion can be written against.

Host-only — no device, no kernel, no ``ttnn``. Nothing here is wired to a verdict yet:
this is the metric on its own, so that a ``passed_test(max_ulp=...)`` gate and a per-op
budget can be added on top of something already pinned by host tests.

Why an *integer* distance and not the fractional ``|err| / ulp(golden)`` that
``accuracy_metrics.compute_pointwise_metrics`` already reports: ``ulp(golden)`` is not the
same size above and below a power of two. ``local_ulp`` divides by the *upward* gap, so a
single representable step below a boundary reads as 0.5 ULP and one above it reads as 1.0:
a "1 ULP" fractional budget admits two representable steps below a power of two and only
one above it, which is not a well-defined predicate. The fractional form remains the better *diagnostic* and keeps its place in the
sweeps and the CSVs; this is the form a pass/fail line can use.

The distance is measured over a **value-order index**: each bit pattern is mapped to its
rank in the format's ordered list of representable values, so a distance of N means
"N representable values apart". Three properties of that mapping are what the harness
needs:

* ``flush_subnormals`` collapses the subnormal band onto zero *and compacts it out of the
  ranking*, so the smallest normal is one step from zero. That is the SFPU's own number
  system (DAZ+FTZ) for bf16 and fp32, where it is also a no-op because the golden models
  the same flush. Without it, a legitimately flushed result next to a subnormal golden
  would read as a ``2**mantissa_bits - 1`` error that is really a 0-step agreement.
  It defaults **per dtype** rather than to ``True`` -- see ``_FLUSHES_SUBNORMALS``, and
  note that fp16 keeps its subnormals in this harness, so collapsing them there would
  hide up to 1023 steps.
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
negative half at ``0x7F7F`` while the positive half is based at ``0x7F80 - 0x7F``. It is
missing the ``2**mantissa_bits - 1`` subnormal compaction *and* based one lower than the
mirror, so every negative value sits ``2**mantissa_bits`` steps too far from zero and
every sign-crossing distance is inflated by that much: measured,
``ulp_distance_bf16_daz(0.0, -tiny)`` returns 129 where this module returns 1. ``test_ulp.py::test_flush_makes_smallest_normal_one_step_from_zero``
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

#: Whether this harness's datapath model flushes a format's subnormal band, and so
#: whether collapsing it is the right default for that dtype.
#:
#: bf16 and fp32 flush below their smallest *normal*
#: (``golden_generators._FTZ_THRESHOLD``), so a golden in those formats never carries a
#: subnormal and the collapse is a no-op that stays as a safety net. **fp16 does not**:
#: its threshold there is ``2**-24``, the smallest fp16 *subnormal*, so an fp16 golden
#: legitimately carries the whole band. Collapsing it would report up to
#: ``2**10 - 1 = 1023`` representable steps as zero -- almost the whole
#: ``MAX_MEANINGFUL_ULP[float16]`` range -- and a 1-step fp16 gate would be blind to it.
#:
#: The underlying question is whether the *producing Dest path* flushed, not what the
#: output dtype is: an fp16 output with ``dest_acc=No`` lands in a Float16 Dest and is
#: flushed, while ``dest_acc=Yes`` gives it an fp32 Dest and keeps the band. The metric
#: cannot see the Dest, so it defaults to the answer that cannot hide error and a caller
#: that knows the Dest flushed may pass ``flush_subnormals=True``.
_FLUSHES_SUBNORMALS: Dict[torch.dtype, bool] = {
    torch.bfloat16: True,
    torch.float32: True,
    torch.float16: False,
}

# ttnn's sanity ceiling (``utils_for_testing.assert_with_ulp``): 2**mantissa_bits. Adding
# 2**mantissa_bits to an IEEE magnitude pattern bumps the exponent field by one and leaves
# the mantissa alone, so that many steps is exactly one binade: bf16 ``1.0 = 0x3F80``,
# ``+0x80 = 0x4000 = 2.0``. A budget above the ceiling therefore says the two values are
# more than a factor of two apart in the normal range -- not an order of magnitude, which
# would need ``ceil(log2(10) * 2**mantissa_bits)`` steps -- at which point ULP has stopped
# being the right metric and the op belongs on the tolerance one. Inside a subnormal band
# the steps are absolute rather than proportional, so the same count spans the band.
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


def flushes_subnormals(dtype: torch.dtype) -> bool:
    """Whether collapsing *dtype*'s subnormal band is the right default. See the table."""
    return _FLUSHES_SUBNORMALS.get(dtype, True)


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
    flush_subnormals: Optional[bool] = None,
) -> torch.Tensor:
    """Integer count of representable steps between *golden* and *result*, per element.

    Both tensors must already be in the output format's dtype — ``passed_test`` casts
    them, so at the gate they are bit-comparable without further work here. Lanes where
    either side is NaN come back as :data:`UNMEASURABLE`.

    *flush_subnormals* defaults per dtype via :func:`flushes_subnormals`: on for bf16 and
    fp32, off for fp16, which keeps its subnormals in this harness. ``True`` forces the
    collapse for a caller that knows the producing Dest flushed.

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

    if flush_subnormals is None:
        flush_subnormals = flushes_subnormals(golden.dtype)
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

    *mask* must be shaped like *distance*. It is checked rather than broadcast because a
    per-op budget computes it dynamically: a ``(1,)`` mask would broadcast all the way
    through and silently judge either every lane or none, and a wrong-length one would
    surface as a bare ``RuntimeError`` from the ``&`` below.
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

    Not simply ``accuracy_metrics.local_ulp``'s ``nextafter`` gap, because the message
    that prints this sits next to an integer step count and has to agree with it:

    * Inside the subnormal band with the band collapsed, the step out of zero is to the
      smallest *normal*, not to the smallest subnormal. Reporting the raw gap there
      understates what one counted step is worth by ``2**mantissa_bits`` -- 128x for bf16
      -- and contradicts ``test_flush_makes_smallest_normal_one_step_from_zero``.
    * At the largest finite value the gap upward is to ``Inf``, so the raw difference is
      infinite and a failure at the top of the range reads "1 ULP = inf". The binade is
      the same downward, so measure it that way. This is the ``finfo.max`` fixup ttnn's
      ``ulp()`` carries for the same reason.
    * The gap is not symmetric at a power of two: below a boundary it is half the size it
      is above. *toward* names the value the step is being counted to -- the result, at
      the gate -- and when that lies below ``|value|`` the downward gap is the one a
      counted step actually crossed. Without it a one-step bf16 result immediately below
      ``1.0`` would be reported as ``1 ULP = 7.8125e-3`` when the step it took was
      ``3.90625e-3``: the very asymmetry this integer metric exists to avoid.
    """
    if not math.isfinite(value):
        return float("nan")
    if flush_subnormals is None:
        flush_subnormals = flushes_subnormals(dtype)

    info = torch.finfo(dtype)
    magnitude = abs(value)
    if flush_subnormals and magnitude < info.tiny:
        # The band is compacted out of the ranking, so one step from here is the jump to
        # the smallest normal.
        return float(info.tiny)

    scalar = torch.tensor(magnitude, dtype=dtype)
    step_is_downward = float(scalar) == info.max or (
        toward is not None and math.isfinite(toward) and abs(toward) < magnitude
    )
    if step_is_downward:
        downward = torch.nextafter(scalar, torch.tensor(0.0, dtype=dtype))
        return float((scalar - downward).to(torch.float32))
    upward = torch.nextafter(scalar, torch.tensor(float("inf"), dtype=dtype))
    return float((upward - scalar).to(torch.float32))


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
) -> str:
    """One actionable line for the worst lane, plus the distribution behind it.

    A ULP verdict is only useful if it names the point: what the hardware produced, what
    the reference was, and what one step is worth there.

    *stats* accepts an already-computed :func:`ulp_stats` dict, so a caller that needed the
    verdict first does not pay for a second pass over the same distances — this message is
    built on passes too, so the duplicate ran on every call. Supplying it *asserts that the
    stats were computed over the same* ``mask``: *mask* is read only on the branch that
    recomputes them, so a caller that passes unmasked stats alongside a mask gets a
    ``worst_index`` in a lane the mask excluded and lane counts from the wrong population.
    :func:`within_ulp` passes the two together and keeps them consistent.
    """
    if stats is None:
        stats = ulp_stats(distance, mask)
    label = fmt.name if fmt is not None else str(golden.dtype)
    if stats["worst_index"] is None:
        return f"no measurable lane ({stats['unmeasurable']} unmeasurable, {label})"

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
        f"({ceiling} = 2**mantissa_bits, one binade). Past it the two values are more than "
        "a factor of two apart in the normal range and the op belongs on the tolerance "
        "metric, not this one."
    )
    return True


def within_ulp(
    golden: torch.Tensor,
    result: torch.Tensor,
    max_ulp: int,
    *,
    fmt: Optional[DataFormat] = None,
    flush_subnormals: Optional[bool] = None,
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
    if fmt is not None:
        # Not just a display label. format_dict collapses Bfp8_b/Bfp4_b/Bfp2_b, every Mx*
        # and Fp8_e4m3 onto torch.bfloat16 and Tf32 onto torch.float32, so the dtype check
        # in ulp_distance cannot tell them apart: without this, a verdict labelled
        # "Bfp8_b" would come back measured in bfloat16 steps, which is exactly the
        # measurement ulp_dtype exists to refuse.
        expected = ulp_dtype(fmt)
        if golden.dtype != expected:
            # Allowlisting the format is not enough on its own: two float32 tensors
            # labelled Float16_b would be measured in float32 steps under a Float16_b
            # verdict, applying the wrong lattice to the gate. The label and the lattice
            # have to be the same claim.
            raise ValueError(
                f"within_ulp: {fmt.name} is measured in {expected}, but the tensors are "
                f"{golden.dtype}; cast both to the output format's dtype first"
            )
    warn_if_threshold_unmeaningful(max_ulp, golden.dtype)

    if mask is not None and mask.shape != golden.shape:
        # Checked, not broadcast: a (1,) mask would broadcast through every operation
        # below and silently judge either all lanes or none, and a wrong-length one would
        # only surface as a bare RuntimeError from ulp_stats.
        raise ValueError(
            f"within_ulp: mask shape {tuple(mask.shape)} does not match golden "
            f"{tuple(golden.shape)}"
        )
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
        golden,
        result,
        distance,
        fmt,
        mask=selected,
        max_ulp=max_ulp,
        stats=stats,
        flush_subnormals=flush_subnormals,
    )
