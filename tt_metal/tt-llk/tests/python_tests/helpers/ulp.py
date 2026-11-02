# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Integer ULP distance: the metric the SFPU accuracy gates are written against.

Each bit pattern maps to its rank in the dtype's ordered list of representable values, so
a distance of N means "N representable values apart". ``flush_subnormals`` collapses the
subnormal band onto zero and out of the ranking, so the smallest normal is one step from
zero; ``+0`` and ``-0`` coincide; ``Inf`` is one step past the largest finite, but
:func:`nonfinite_mismatches` rejects finite-against-Inf positionally; ``NaN`` has no rank
and reads as :data:`UNMEASURABLE`.
"""

import math
from typing import Any, NamedTuple

import torch

from .format_config import DataFormat
from .llk_params import format_dict

# The formats with a native per-element ULP; `has_ulp_gate` adds the proxy formats.
ULP_FORMATS: tuple[DataFormat, ...] = (
    DataFormat.Float16_b,
    DataFormat.Float16,
    DataFormat.Float32,
)

# Bfp8_b is gated in bfloat16's step space (`passed_test` has already cast it): one Bfp8_b
# step is two bf16 steps, and block quantization is charged in full, so such a budget is
# usable only where the quantization is exact. Bfp4_b and Bfp2_b have too few bits.
_ULP_PROXY_DTYPES: dict[DataFormat, torch.dtype] = {
    DataFormat.Bfp8_b: torch.bfloat16,
}

# Lanes whose golden is below this fraction of the tensor's dynamic range are the ones a
# near_zero_atol floor may rescue (ttnn uses the same 1%).
NEAR_ZERO_FRACTION = 1e-2

# The distance of a lane where either side is NaN. Never compare it against a budget:
# -1 <= max_ulp holds for every budget.
UNMEASURABLE = -1


class _Spec(NamedTuple):
    bits_dtype: torch.dtype  # same-width signed int, for .view()
    wide_dtype: torch.dtype  # wider signed int, room for the signed rank
    sign_mask: int
    all_mask: int
    mantissa_bits: int


def _from_finfo(dtype: torch.dtype) -> _Spec:
    info = torch.finfo(dtype)
    bits_dtype, wide_dtype = {
        16: (torch.int16, torch.int32),
        32: (torch.int32, torch.int64),
    }[info.bits]
    return _Spec(
        bits_dtype,
        wide_dtype,
        1 << (info.bits - 1),
        (1 << info.bits) - 1,
        round(-math.log2(info.eps)),
    )


_SPECS = {
    dtype: _from_finfo(dtype)
    for dtype in (torch.bfloat16, torch.float16, torch.float32)
}

# Whether the harness's datapath model flushes the dtype's subnormal band: bf16 and fp32
# do, fp16 does not (collapsing its band would hide up to 1023 steps of error).
_FLUSHES_SUBNORMALS = {
    torch.bfloat16: True,
    torch.float32: True,
    torch.float16: False,
}

# Subnormal outputs are ranked flushed by every step-budget gate, fp16 included: the
# golden keeps IEEE subnormals the pack path does not reproduce.
FLUSH_SUBNORMAL_OUTPUTS = True

# Below these lane counts a quantile is not a percentile: p95/p99 fall back to the max.
_MIN_LANES_FOR_P95 = 20
_MIN_LANES_FOR_P99 = 100


def _spec(dtype: torch.dtype) -> _Spec:
    try:
        return _SPECS[dtype]
    except KeyError:
        raise ValueError(
            f"unsupported dtype {dtype}; ULP is measured in "
            f"{', '.join(str(d) for d in _SPECS)}"
        ) from None


def mantissa_bits(dtype: torch.dtype) -> int:
    return _spec(dtype).mantissa_bits


def max_meaningful_ulp(dtype: torch.dtype) -> int:
    """``2**mantissa_bits`` is one binade: past it the two values are more than a factor
    of two apart and the op belongs on the tolerance metric."""
    return 1 << mantissa_bits(dtype)


def ulp_dtype(fmt: DataFormat) -> torch.dtype:
    """The torch dtype a ULP distance for *fmt* is measured in; raises for a format with
    no per-element ULP (the block floats below Bfp8_b, the MX formats, integers, Tf32).
    """
    if fmt in ULP_FORMATS:
        return format_dict[fmt]
    if fmt in _ULP_PROXY_DTYPES:
        return _ULP_PROXY_DTYPES[fmt]
    raise ValueError(f"{fmt.name} has no per-element ULP")


def has_ulp_gate(fmt: DataFormat) -> bool:
    return fmt in ULP_FORMATS or fmt in _ULP_PROXY_DTYPES


def flushes_subnormals(dtype: torch.dtype) -> bool:
    """The per-dtype flush default; raises for an unsupported dtype."""
    _spec(dtype)
    return _FLUSHES_SUBNORMALS[dtype]


def _value_order_index(
    t: torch.Tensor, spec: _Spec, *, flush_subnormals: bool
) -> torch.Tensor:
    """Signed rank of each element among the representable values: magnitude rank 0 is
    zero (so the signed zeros coincide); flushed, the whole ``exp == 0`` band ranks 0 and
    leaves the ranking. Unflushed it is the sweep's own total order."""
    bits = t.contiguous().view(spec.bits_dtype).to(spec.wide_dtype) & spec.all_mask
    magnitude = bits & (spec.all_mask ^ spec.sign_mask)
    if flush_subnormals:
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
    flush_subnormals: bool | None = None,
) -> torch.Tensor:
    """Representable steps between *golden* and *result* per element, both already in
    the output dtype; a NaN lane reads :data:`UNMEASURABLE`. *flush_subnormals* defaults
    per dtype."""
    if golden.dtype != result.dtype:
        raise ValueError(
            f"ulp_distance: dtype mismatch, golden {golden.dtype} vs result "
            f"{result.dtype}; cast both to the output format's dtype first"
        )
    spec = _spec(golden.dtype)
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
    """Lanes where the two sides disagree about being non-finite: exactly one NaN,
    exactly one infinite, or infinities of opposite sign. NaN sign is not judged."""
    golden_nan, result_nan = torch.isnan(golden), torch.isnan(result)
    golden_inf, result_inf = torch.isinf(golden), torch.isinf(result)
    sign_differs = torch.signbit(golden) != torch.signbit(result)
    return (
        (golden_nan ^ result_nan)
        | (golden_inf ^ result_inf)
        | (golden_inf & result_inf & sign_differs)
    )


def selected_lanes(mask: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
    """*mask* as a boolean selection over *reference*'s lanes, or all of them. The shape
    is checked, not broadcast, and an integer mask is refused: combined with ``&`` it
    would be bitwise, and a truthy 2 would drop the lane it meant to select."""
    if mask is None:
        return torch.ones_like(reference, dtype=torch.bool)
    if mask.shape != reference.shape:
        raise ValueError(
            f"mask shape mismatch {tuple(reference.shape)} vs {tuple(mask.shape)}"
        )
    if mask.dtype is not torch.bool:
        raise ValueError(f"mask must be bool, got {mask.dtype}")
    return mask


def near_zero_band(
    golden: torch.Tensor,
    atol: float,
    *,
    selected: torch.Tensor | None = None,
    fraction: float = NEAR_ZERO_FRACTION,
) -> torch.Tensor:
    """The lanes a floor of *atol* judges: ``|golden|`` under *fraction* of the largest
    finite golden among *selected*, so the band follows the stimulus, and at most
    ``atol / fraction``, so one large golden cannot widen it across the tile. Compared
    in float32, so a 16-bit golden does not round either cut."""
    if atol < 0:
        raise ValueError(
            f"near_zero_atol must not be negative, got {atol}; 0.0 is the no-floor "
            "value, and None is the default"
        )
    if not fraction > 0.0:
        raise ValueError(f"near_zero_fraction must be positive, got {fraction}")
    magnitude = golden.abs().to(torch.float32)
    in_scope = torch.isfinite(golden)
    if selected is not None:
        in_scope = in_scope & selected
    finite = magnitude[in_scope]
    if finite.numel() == 0:
        return torch.zeros_like(in_scope)
    dynamic_range = float(finite.max())
    band = (
        torch.ones_like(in_scope)
        if dynamic_range == 0.0
        else magnitude < fraction * dynamic_range
    )
    return band & (magnitude <= atol / fraction)


def ulp_elementwise_valid(
    golden: torch.Tensor,
    result: torch.Tensor,
    max_ulp: int,
    *,
    near_zero_atol: float | None = None,
    near_zero_fraction: float = NEAR_ZERO_FRACTION,
    flush_subnormals: bool | None = None,
    selected: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-element verdict: a lane is valid when the two sides agree about being
    non-finite and are within *max_ulp* steps, or both NaN, or it lies in the near-zero
    band with an absolute error within *near_zero_atol*. Returns ``(valid, distance,
    covered)``: *covered* is every lane the floor rescues -- in the band and within
    *near_zero_atol* -- in step budget or not, which a reporting caller keeps out of its
    ranking."""
    distance = ulp_distance(golden, result, flush_subnormals=flush_subnormals)
    both_nan = torch.isnan(golden) & torch.isnan(result)
    valid = ((distance >= 0) & (distance <= max_ulp)) | both_nan
    rescued = torch.zeros_like(valid)
    if near_zero_atol is not None:
        band = near_zero_band(
            golden, near_zero_atol, selected=selected, fraction=near_zero_fraction
        )
        absolute_error = (result.to(torch.float32) - golden.to(torch.float32)).abs()
        rescued = band & (absolute_error <= near_zero_atol)
        valid = valid | rescued
    ok = valid & ~nonfinite_mismatches(golden, result)
    return ok, distance, rescued & ok


def ulp_stats(
    distance: torch.Tensor, mask: torch.Tensor | None = None
) -> dict[str, Any]:
    """The distribution of *distance* over the lanes *mask* selects; the unmeasurable
    lanes are counted apart so they cannot shrink a mean."""
    flat = distance.reshape(-1).to(torch.int64)
    selected = selected_lanes(mask, distance).reshape(-1)
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
    toward: float | None = None,
    flush_subnormals: bool | None = None,
) -> float:
    """The size of one *counted* step of *dtype* at ``|value|``, as the metric counts
    it: across a flushed band the step out of zero is to the smallest normal; at the
    largest finite the step is the downward one, not the gap to Inf; at a power of two
    the gap is asymmetric, so *toward* names the side the step was counted on."""
    if not math.isfinite(value):
        return float("nan")
    _spec(dtype)
    if flush_subnormals is None:
        flush_subnormals = flushes_subnormals(dtype)
    info = torch.finfo(dtype)
    magnitude = abs(value)
    # Zero has no downward neighbour of its own: `copysign(1.0, 0.0)` is +1.0.
    heads_toward_zero = (
        magnitude != 0.0
        and toward is not None
        and math.isfinite(toward)
        and math.copysign(1.0, value) * (toward - value) < 0.0
    )
    if flush_subnormals and (
        magnitude < info.tiny or (magnitude == info.tiny and heads_toward_zero)
    ):
        return float(info.tiny)
    scalar = torch.tensor(magnitude, dtype=dtype)
    if float(scalar) == info.max or heads_toward_zero:
        downward = torch.nextafter(scalar, torch.tensor(0.0, dtype=dtype))
        return float((scalar - downward).to(torch.float32))
    upward = torch.nextafter(scalar, torch.tensor(float("inf"), dtype=dtype))
    return float((upward - scalar).to(torch.float32))


def ulp_verdict_message(
    golden: torch.Tensor,
    result: torch.Tensor,
    distance: torch.Tensor,
    fmt: DataFormat | None = None,
    *,
    mask: torch.Tensor | None = None,
    max_ulp: int | None = None,
    flush_subnormals: bool | None = None,
    rescued: torch.Tensor | None = None,
) -> str:
    """What a gate logs, pass, fail or report alike: a non-finite disagreement first (the
    step count cannot describe one), then the worst lane with the size of one step
    there and the distribution over *mask*. Of *rescued* (the floor's lanes) only those
    past *max_ulp* count as held."""
    stats = ulp_stats(distance, mask)
    label = fmt.name if fmt is not None else str(golden.dtype)
    if rescued is not None and max_ulp is not None:
        rescued = rescued & ((distance < 0) | (distance > max_ulp))
    floor = "" if rescued is None else f", {int(rescued.sum())} held by the floor"
    index = stats["worst_index"]
    if index is None and stats["unmeasurable"] == 0:
        steps = (
            f"no lane under judgement (the mask selected none of {distance.numel()} "
            f"lanes{floor}, {label})"
        )
    elif index is None:
        steps = (
            f"no measurable lane ({stats['unmeasurable']} unmeasurable{floor}, {label})"
        )
    else:
        golden_value = float(golden.reshape(-1)[index])
        result_value = float(result.reshape(-1)[index])
        step = local_step(
            golden_value,
            golden.dtype,
            toward=result_value,
            flush_subnormals=flush_subnormals,
        )
        budget = "" if max_ulp is None else f" (budget {max_ulp})"
        sized = (
            f"1 ULP = {step:.6e}" if math.isfinite(step) else "no step at a non-finite"
        )
        steps = (
            f"max {stats['max']} ULP @ [{index}]{budget}: result {result_value!r} vs "
            f"golden {golden_value!r} ({sized}, {label})\n"
            f"  over {stats['lanes']} lanes: mean {stats['mean']:.3f}, "
            f"p95 {stats['p95']:.1f}, p99 {stats['p99']:.1f}, "
            f"{100.0 * stats['exact_frac']:.1f}% exact, "
            f"{stats['unmeasurable']} unmeasurable{floor}"
        )
    mismatched = nonfinite_mismatches(golden, result)
    if mask is not None:
        mismatched = mismatched & mask
    if not bool(mismatched.any()):
        return steps
    where = int(torch.nonzero(mismatched.reshape(-1), as_tuple=False)[0])
    return (
        f"non-finite disagreement @ [{where}]: result "
        f"{float(result.reshape(-1)[where])!r} vs golden "
        f"{float(golden.reshape(-1)[where])!r} ({label}); "
        f"{int(mismatched.sum())} such lane(s)\n  {steps}"
    )
