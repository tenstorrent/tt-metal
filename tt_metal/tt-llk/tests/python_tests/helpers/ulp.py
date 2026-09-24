# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Integer ULP distance: the metric an SFPU accuracy assertion can be written against.

Host-only — no device, no kernel, no ``ttnn``.

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

# The block floats are absent on purpose: their spacing comes from an exponent shared
# across the block, so a per-element count is not a property of the element. They keep
# the block-aware lattice compares in utils.py; integers want bit equality.
ULP_FORMATS: Tuple[DataFormat, ...] = (
    DataFormat.Float16_b,
    DataFormat.Float16,
    DataFormat.Float32,
)

#: Returned for any lane where either side is NaN. Never compare it against a budget
#: directly -- ``-1 <= max_ulp`` holds for every budget. Use :func:`within_ulp`.
UNMEASURABLE = -1


@dataclass(frozen=True)
class DataTypeDescriptor:
    """The bit layout of one torch float dtype: what any bit-level arithmetic on it needs.

    Nothing here is specific to ULP. It lives in this module because the ULP metric is
    its only user so far.
    """

    bits_dtype: torch.dtype  # same-width signed int, for .view()
    wide_dtype: torch.dtype  # wider signed int, room for the signed rank
    sign_mask: int
    all_mask: int
    mantissa_bits: int


_ULP_DTYPES: Dict[torch.dtype, DataTypeDescriptor] = {
    torch.bfloat16: DataTypeDescriptor(
        bits_dtype=torch.int16,
        wide_dtype=torch.int32,
        sign_mask=0x8000,
        all_mask=0xFFFF,
        mantissa_bits=7,
    ),
    torch.float16: DataTypeDescriptor(
        bits_dtype=torch.int16,
        wide_dtype=torch.int32,
        sign_mask=0x8000,
        all_mask=0xFFFF,
        mantissa_bits=10,
    ),
    torch.float32: DataTypeDescriptor(
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


def _require_same_dtype(
    caller: str, golden: torch.Tensor, result: torch.Tensor
) -> None:
    """Raise unless both sides are on one lattice: a ULP distance is a rank in one dtype's
    ordered value set, so there is nothing to measure across two."""
    if golden.dtype != result.dtype:
        raise ValueError(
            f"{caller}: dtype mismatch, golden {golden.dtype} vs result {result.dtype}; a "
            "ULP distance is a rank on one lattice, so cast both to the output format's "
            "dtype first"
        )


def _shape_mismatch(
    reference: torch.Tensor, other: torch.Tensor, what: str = "shape"
) -> Optional[str]:
    """Why *other* is not lane-for-lane with *reference*, or ``None`` if it is.

    Checked, never broadcast: a ``(1,)`` tensor would judge every lane or none. Returned
    rather than raised, because :func:`within_ulp` reports it as a failed verdict.
    """
    if other.shape == reference.shape:
        return None
    return f"{what} mismatch {tuple(reference.shape)} vs {tuple(other.shape)}"


def ulp_dtype(fmt: DataFormat) -> torch.dtype:
    """The torch dtype a ULP distance for *fmt* is measured in. Raises for a format with
    no per-element ULP, so a caller asking for a gate it cannot have finds out here."""
    if fmt in ULP_FORMATS:
        return format_dict[fmt]
    raise ValueError(
        f"{fmt.name} has no per-element ULP. ULP is defined for "
        f"{', '.join(f.name for f in ULP_FORMATS)}; the block floats (Bfp*, Mx*) share a "
        "block exponent and keep their lattice compare in utils.py, the integer formats "
        "want bit equality, and Tf32 collapses onto torch.float32 in format_dict, so it "
        "would be counted on a lattice that is not its own."
    )


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
    t: torch.Tensor, spec: DataTypeDescriptor, *, flush_subnormals: bool
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
    _require_same_dtype("ulp_distance", golden, result)
    spec = _ULP_DTYPES.get(golden.dtype)
    if spec is None:
        raise _unsupported_dtype_error("ulp_distance", golden.dtype, _ULP_DTYPES)
    mismatch = _shape_mismatch(golden, result)
    if mismatch:
        raise ValueError(f"ulp_distance: {mismatch}")

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


def _selection(
    mask: Optional[torch.Tensor], reference: torch.Tensor, caller: str
) -> torch.Tensor:
    """*mask* as a boolean selection over *reference*'s lanes, or all of them.

    Shape is checked, not broadcast -- see :func:`_shape_mismatch`. dtype is checked
    because the selection is combined with ``&`` -- on an integer mask that is
    bitwise arithmetic, where a truthy ``2`` becomes ``2 & 1 == 0`` and silently drops
    the lane it was meant to select, from the failure scan and from every statistic.
    """
    if mask is None:
        return torch.ones_like(reference, dtype=torch.bool)
    mismatch = _shape_mismatch(reference, mask, "mask shape")
    if mismatch:
        raise ValueError(f"{caller}: {mismatch}")
    if mask.dtype is not torch.bool:
        raise ValueError(f"{caller}: mask must be bool, got {mask.dtype}")
    return mask


def ulp_stats(
    distance: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> Dict[str, Any]:
    """Distribution of a ULP distance over the lanes *mask* selects.

    The aggregates cover the measurable lanes only; the unmeasurable ones are counted
    separately so they cannot quietly shrink a mean. *mask* is checked against
    *distance*'s shape, not broadcast: ``(1,)`` would judge every lane or none.
    """
    flat = distance.reshape(-1).to(torch.int64)
    selected = _selection(mask, distance, "ulp_stats").reshape(-1)

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
            float(torch.quantile(as_float, q=0.95))
            if lanes >= _MIN_LANES_FOR_P95
            else float(worst)
        ),
        "p99": (
            float(torch.quantile(as_float, q=0.99))
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

    *stats* lets a caller that already has them skip a second pass, and supplying it
    *asserts they came from the same* ``mask``. *flush_subnormals* carries the same
    obligation against *distance*, which fixed the count printed beside the step size.
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
    # A non-finite golden has no step, and `local_step` says so with NaN; printing it
    # would read "1 ULP = nan" on a lane both sides agree on.
    sized = f"1 ULP = {step:.6e}" if math.isfinite(step) else "no step at a non-finite"
    return (
        f"max {stats['max']} ULP @ [{index}]{budget}: result {result_value!r} vs golden "
        f"{golden_value!r} ({sized}, {label})\n"
        f"  over {stats['lanes']} lanes: mean {stats['mean']:.3f}, p95 {stats['p95']:.1f}, "
        f"p99 {stats['p99']:.1f}, {100.0 * stats['exact_frac']:.1f}% exact, "
        f"{stats['unmeasurable']} unmeasurable"
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
) -> Tuple[bool, str]:
    """The whole verdict: non-finite positions agree, and every finite lane is in budget.

    What a ``passed_test(max_ulp=...)`` gate calls, so the :data:`UNMEASURABLE` sentinel is
    never compared against a budget by hand. *max_ulp* is keyword-only because it is the
    one real magic number in the signature; *mask* selects the lanes under judgement.

    **Omitting** *fmt* skips the format allowlist as well as the label, and asserts the
    caller has already put the tensors on the lattice they mean: ``format_dict`` collapses
    the block floats onto ``torch.bfloat16`` and ``Tf32`` onto ``torch.float32``, so an
    unnamed ``Tf32`` tensor is measured in float32 steps and a one-step error reads ~8192.

    Returns ``(ok, message)``, worth logging on a pass too: it turns a functional test
    into an accuracy datapoint without changing its verdict.
    """
    mismatch = _shape_mismatch(golden, result)
    if mismatch:
        return False, mismatch
    if fmt is not None:
        # The label and the lattice have to be the same claim -- see the docstring.
        expected = ulp_dtype(fmt)
        if golden.dtype != expected:
            raise ValueError(
                f"within_ulp: {fmt.name} is measured in {expected}, but the tensors are "
                f"{golden.dtype}; cast both to the output format's dtype first"
            )
    # Before the non-finite short-circuit: an uncast golden reads there as a kernel
    # overflow, and labels it with the wrong dtype.
    _require_same_dtype("within_ulp", golden, result)
    warn_if_threshold_unmeaningful(max_ulp, golden.dtype)

    selected = _selection(mask, golden, "within_ulp")
    # Positional agreement first, and no budget buys past it: finite against `Inf` is one
    # step apart in the value order, but it is an overflow, not an inexact answer. The only
    # `Inf` lane reaching the distance is same-sign, which is 0.
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
