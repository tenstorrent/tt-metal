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
* ``Inf`` is an ordinary bit pattern, one step past the largest finite, so
  :func:`ulp_distance` reports "1 step" for an overflow at the very top of the range and a
  huge distance for a spurious ``Inf`` anywhere else. The composite verdict is stricter
  than the distance here: :func:`within_ulp` rejects every finite-against-``Inf`` lane
  *positionally*, through :func:`nonfinite_mismatches`, before any step is counted -- so
  ``finfo.max`` against ``+Inf`` is a hard fail even at ``max_ulp=1``. That is the line
  ``utils.py::_bfp_block_aware_compare`` already takes, and it is deliberate: an overflow
  is a different kind of answer from an inexact one, and no budget should buy it. The
  1-step reading stays available to a caller measuring the distance directly.

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

# Formats with no float dtype of their own that are still gated in a *proxy* format's ULP
# space. Bfp8_b carries 7 magnitude bits per element, the same mantissa width as bfloat16,
# so one step of the bf16 lattice is one step of the Bfp8_b lattice wherever the shared
# block exponent is the one bf16 would have used. This is ttnn's choice — its
# ``assert_with_ulp`` maps ``bfloat8_b`` to ``bfloat16`` — and it costs nothing here
# because ``passed_test`` has already cast a Bfp8_b tensor to bfloat16 before it compares.
#
# The caveat to budget for: where a block spans a wide magnitude range, its small elements
# are quantized by the block exponent far more coarsely than bf16 would quantize them, and
# a bf16 step count reads that legal quantization as a multi-step error. Those are the
# small-magnitude lanes, so ``near_zero_atol`` is what absorbs them; a Bfp8_b budget
# without that floor has to be loose enough to cover the widest block in the stimulus.
#
# Bfp4_b (3 magnitude bits) and Bfp2_b (1) are deliberately absent: their lattices are so
# much coarser than bf16's that a bf16 step count would read every legal quantization as a
# 16- or 64-step error. They keep ``_bfp_block_aware_compare``.
_ULP_PROXY_DTYPES: Dict[DataFormat, torch.dtype] = {
    DataFormat.Bfp8_b: torch.bfloat16,
}

#: Lanes whose golden is below this fraction of the tensor's dynamic range are the ones a
#: ``near_zero_atol`` floor may rescue. ttnn's ``measure_ulp_with_near_zero_atol`` uses the
#: same 1%: deliberately simple and scale-relative rather than tuned per op.
NEAR_ZERO_FRACTION = 1e-2

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
# the mantissa alone, so that many steps is exactly one binade -- from *any* mantissa:
# bf16 ``1.0 = 0x3F80``, ``+0x80 = 0x4000 = 2.0``. A budget above the ceiling therefore
# says the two values are more than a factor of two apart in the normal range, and not an
# order of magnitude, which is a bit over three binades' worth of steps -- not a fixed
# count, because rank is affine in value *within* a binade, so a fixed ratio costs a
# variable number of ranks depending on where in the binade the pair starts (bf16
# ``1.0 -> 10.0`` is 416 steps, ``1.5 -> 15.0`` is 432). Past the ceiling ULP has stopped
# being the right metric and the op belongs on the tolerance one. Inside a subnormal band
# the steps are absolute rather than proportional, so the same count spans the band.
MAX_MEANINGFUL_ULP: Dict[torch.dtype, int] = {
    dtype: 1 << spec.mantissa_bits for dtype, spec in _ULP_DTYPES.items()
}

# ttnn's guards in ``measure_ulp_with_near_zero_atol``: below these counts a quantile is
# not a percentile, so p95/p99 fall back to the max rather than inventing precision.
_MIN_LANES_FOR_P95 = 20
_MIN_LANES_FOR_P99 = 100


def _unsupported_dtype_error(
    caller: str, dtype: torch.dtype, supported: Any
) -> ValueError:
    """The one spelling of "this metric does not measure that dtype".

    Three entry points refuse the same thing against two different tables
    (:func:`flushes_subnormals`, :func:`ulp_distance`, :func:`local_step`), and the per-op
    budget will want a fourth. Built in one place so the fourth copy is not written.
    """
    return ValueError(
        f"{caller}: unsupported dtype {dtype}; supported: "
        f"{', '.join(str(d) for d in supported)}"
    )


def ulp_dtype(fmt: DataFormat) -> torch.dtype:
    """The torch dtype a ULP distance for *fmt* is measured in.

    Raises ``ValueError`` for every format that has no per-element ULP, rather than
    silently measuring something else — a caller that asks for a gate it cannot have
    should find out at the call, not by reading a green result.
    """
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


def flushes_subnormals(dtype: torch.dtype) -> bool:
    """Whether collapsing *dtype*'s subnormal band is the right default. See the table.

    Raises for a dtype the metric does not measure, rather than defaulting. Every
    supported dtype is an explicit row in ``_FLUSHES_SUBNORMALS``, so a default could only
    ever fire for an unsupported one -- and ``True`` is the polarity the table's own
    comment calls error-hiding, which is why the module docstring says the default is per
    dtype *rather than* ``True``. Without this, ``local_step(1.0, torch.float64)`` handed
    back ``2**-52`` where :func:`ulp_distance` raises for the same dtype.
    """
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

    Magnitude rank 0 is zero, so the two signed zeros coincide. With *flush_subnormals*
    the whole ``exp == 0`` band (zero and every subnormal) ranks 0 and the band is removed
    from the ranking above it, which makes the smallest normal rank 1.

    With *flush_subnormals* off this is the same total order the ULP *sweep* already
    enumerates over: ``stimuli_generator.strategies.structured._to_key`` (and its inline
    copy in ``ulp_sweep_value_count``) maps ``bits`` to ``INT_MIN - bits`` for the negative
    half, which is this sign-and-magnitude rank down to the ``+0``/``-0`` collapse. That is
    not an accident and must not drift: the stimuli that say "one representable step" and
    the metric that counts them have to mean the same thing.
    ``test_ulp.py::test_the_value_order_agrees_with_the_sweep_enumerators_key`` pins the
    two against each other. The flush is this module's own addition -- the sweep enumerates
    the subnormal band, the metric may compact it -- so the agreement is claimed for
    ``flush_subnormals=False`` only.
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


def ulp_elementwise_valid(
    golden: torch.Tensor,
    result: torch.Tensor,
    max_ulp: int,
    *,
    near_zero_atol: Optional[float] = None,
    near_zero_fraction: float = NEAR_ZERO_FRACTION,
    flush_subnormals: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-element ULP verdict, shaped like the mask ``passed_test`` already prints.

    A lane is valid when the two sides agree about being non-finite **and** either

    * they are within *max_ulp* representable steps of each other, or
    * the lane is near zero and its absolute error is within *near_zero_atol*.

    The non-finite rule is stricter than the metric on purpose. :func:`ulp_distance` gives
    ``Inf`` a rank, one step past the largest finite, but an overflow to ``Inf`` where the
    reference is finite is a different kind of wrong from being one step out, and the
    tolerance gate this replaces already rejects it. Both-NaN lanes are valid, which is
    the rule this harness has always used.

    "Near zero" means ``|golden| < near_zero_fraction * max|finite golden|`` (every lane,
    if the golden is all zeros). The floor is deliberately *not* applied at every
    magnitude: an absolute tolerance at large magnitude is precisely the format- and
    magnitude-blind gate a step count is meant to replace. It is here for the lanes where
    the reference crosses zero — ``log`` near 1, ``expm1`` near 0, ``tanhshrink``, ``sin``
    near π — where ``ulp(golden)`` collapses and any absolute error explodes into a step
    count that says nothing about the kernel. Raising *max_ulp* instead would reopen the
    hole across the whole domain.

    Returns ``(is_valid, distance)``; the distance is handed back so a caller can report
    the distribution without measuring twice.
    """
    distance = ulp_distance(golden, result, flush_subnormals=flush_subnormals)
    both_nan = torch.isnan(golden) & torch.isnan(result)
    in_budget = (distance >= 0) & (distance <= max_ulp)
    valid = in_budget | both_nan

    if near_zero_atol is not None:
        finite_golden = golden[torch.isfinite(golden)]
        if finite_golden.numel() == 0:
            near_zero = torch.zeros_like(valid)
        else:
            dynamic_range = float(finite_golden.abs().max())
            near_zero = (
                torch.ones_like(valid)
                if dynamic_range == 0.0
                else golden.abs() < near_zero_fraction * dynamic_range
            )
        # In float32 so a bf16 comparison does not round the error into or out of budget.
        absolute_error = (result.to(torch.float32) - golden.to(torch.float32)).abs()
        valid = valid | (near_zero & (absolute_error <= near_zero_atol))

    return valid & ~nonfinite_mismatches(golden, result), distance


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
      the gate -- and when the path to it leaves ``value`` heading toward zero, the
      downward gap is the one a counted step actually crossed. Without it a one-step bf16
      result immediately below ``1.0`` would be reported as ``1 ULP = 7.8125e-3`` when the
      step it took was ``3.90625e-3``: the very asymmetry this integer metric exists to
      avoid.

      Direction is taken from the *signed* delta, not from ``abs(toward) < magnitude``.
      Comparing magnitudes loses the direction whenever the pair straddles zero:
      ``1.0 -> -2.0`` has the larger magnitude on the far side, but the first step out of
      ``1.0`` is downward, and reporting the upward gap there names the wrong side of the
      boundary. At ``value == 0`` there is no downward direction to take -- ranks ``-1``
      and ``+1`` both sit at the first value away from zero -- so the signed delta is not
      consulted and the step out of zero is reported whichever way *toward* points.
    """
    if not math.isfinite(value):
        return float("nan")
    if dtype not in _ULP_DTYPES:
        # The one public entry point with no other dtype check. Refused rather than
        # answered, so it cannot disagree with ulp_distance about what is measurable --
        # and an explicit flush_subnormals= would otherwise skip flushes_subnormals()'s
        # own guard.
        raise _unsupported_dtype_error("local_step", dtype, _ULP_DTYPES)
    if flush_subnormals is None:
        flush_subnormals = flushes_subnormals(dtype)

    info = torch.finfo(dtype)
    magnitude = abs(value)
    # Does the first step out of `value` head toward zero? From the signed delta, so a
    # sign-crossing pair is judged on where the path starts rather than where it ends.
    heads_toward_zero = (
        # Zero is excluded before the sign is consulted: `copysign(1.0, 0.0)` is `+1.0`, so
        # any negative `toward` would read as downward, and with the band kept (fp16's
        # default) the early return below does not catch it -- `nextafter(+0.0, 0.0)` is
        # `+0.0` and the message prints `1 ULP = 0.000000e+00` on the zero-crossing lane it
        # exists to explain. Rank 0 has no downward neighbour distinct from its upward one.
        magnitude != 0.0
        and toward is not None
        and math.isfinite(toward)
        and math.copysign(1.0, value) * (toward - value) < 0.0
    )

    if flush_subnormals and (
        magnitude < info.tiny or (magnitude == info.tiny and heads_toward_zero)
    ):
        # With the band compacted out of the ranking, zero and every subnormal share rank
        # 0 and the smallest normal is rank 1 -- so one counted step is `tiny` both on the
        # way out of the band and on the way into it. The boundary case matters: at
        # exactly `tiny` the raw downward gap is the smallest *subnormal*, which
        # understates the step by 2**mantissa_bits.
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

    Needed because the step count cannot describe this failure: a NaN lane is
    :data:`UNMEASURABLE` and drops out of the statistics, so a verdict that failed only
    on a missing NaN reports "max 0 ULP (budget 0)" — true, and useless.
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
) -> str:
    """What a gate should log: the non-finite disagreement first, then the steps.

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

    *flush_subnormals* carries the same obligation against *distance*: it feeds only the
    :func:`local_step` call, while the step *count* beside it was already decided when
    *distance* was computed. A caller that took :func:`ulp_distance`'s documented
    ``flush_subnormals=True`` escape hatch and then omitted it here prints a "1 ULP = X"
    that is off by up to ``2**mantissa_bits`` from the count next to it. Pass the same
    value to both, as :func:`within_ulp` does.
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
    *,
    max_ulp: int,
    fmt: Optional[DataFormat] = None,
    flush_subnormals: Optional[bool] = None,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[bool, str]:
    """The whole verdict: non-finite positions agree, and every finite lane is in budget.

    This is what a ``passed_test(max_ulp=...)`` gate is meant to call, so that the
    :data:`UNMEASURABLE` sentinel is never compared against a budget by hand. *max_ulp* is
    keyword-only for the same reason the gate spells it out: it is the one real magic
    number in the signature, and ``within_ulp(golden, result, 0)`` does not say which
    number. *mask*
    selects the lanes under judgement — pass one to exclude lanes an op's own edge rules
    have already settled.

    **Omitting** *fmt* skips the format allowlist as well as the label, and asserts the
    caller has already put the tensors on the lattice they mean. ``format_dict`` collapses
    ``Bfp8_b``/``Bfp4_b``/every ``Mx*``/``Fp8_e4m3`` onto ``torch.bfloat16`` and ``Tf32``
    onto ``torch.float32``, so an unnamed float32 tensor holding ``Tf32`` values is
    measured on the float32 lattice and a one-step Tf32 error reads as ~8192 steps --
    below ``MAX_MEANINGFUL_ULP[float32]``, so :func:`warn_if_threshold_unmeaningful` does
    not catch it either. ``passed_test`` always threads its ``output_data_format`` through,
    so the gate is never in that position; a direct caller over raw tensors is.

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
    distance = ulp_distance(golden, result, flush_subnormals=flush_subnormals)
    stats = ulp_stats(distance, selected)
    message = ulp_verdict_message(
        golden,
        result,
        distance,
        fmt,
        mask=selected,
        max_ulp=max_ulp,
        stats=stats,
        flush_subnormals=flush_subnormals,
    )

    # Positional agreement is decisive, and no budget buys past it: a finite golden
    # against an `Inf` result is one step apart in the value order (the module docstring's
    # third bullet), but it is an overflow, not an inexact answer, so it is refused rather
    # than measured -- the same line `utils.py::_bfp_block_aware_compare` takes. The only
    # `Inf` lane that reaches the distance is same-sign `Inf` against `Inf`, which is 0.
    if nonfinite_disagreement_summary(golden, result, fmt, mask=selected) is not None:
        return False, message

    ok = stats["worst_index"] is None or stats["max"] <= max_ulp
    return ok, message
