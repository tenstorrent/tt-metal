# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Canonical BF16 raw scoring implementation shared by CLI and package tests.

Sidecar use never invokes fitting or compilation. The optional compiler callback
exists only for the legacy CSV-facing CLI adapter. Numerical and class-policy
implementation is owned here once; no independent upstream oracle is introduced.
"""
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ttpoly.groundtruth import reference_value
from ttpoly.groundtruth.valid_domain import (
    real_domain_mask,
    real_domain_mask_from_declaration,
    real_domain_singularity_results,
)
from ttpoly.spec.domain_action import parse_domain_actions
from ttpoly.spec.program import TargetPolicy, ProgramIOContract
from ttpoly.spec.scoring_semantics import (
    BF16_EXHAUSTIVE_EVIDENCE_SCHEMA,
    real_domain_declaration_from_spec,
    validate_scoring_semantics,
)
from ttpoly.precision.bf16 import bh_bf16_ingress_inputs
from ttpoly.spec.target_class import (
    RAW_INPUT_CLASSES,
    compile_raw_class_terminal_plan,
    compile_target_class_semantics,
    target_class_semantics_from_manifest,
)
from ttpoly.spec import units as _units

_IEEE_CLASS_NAMES = (
    "positive_zero",
    "negative_zero",
    "positive_subnormal",
    "negative_subnormal",
    "positive_normal",
    "negative_normal",
    "positive_infinity",
    "negative_infinity",
    "positive_qnan",
    "negative_qnan",
    "positive_snan",
    "negative_snan",
)

_BF16_SPACE_SIZE = 1 << 16

_FULL_BF16_CLASS_COUNTS = {
    "positive_zero": 1,
    "negative_zero": 1,
    "positive_subnormal": (1 << 7) - 1,
    "negative_subnormal": (1 << 7) - 1,
    "positive_normal": 254 * (1 << 7),
    "negative_normal": 254 * (1 << 7),
    "positive_infinity": 1,
    "negative_infinity": 1,
    "positive_qnan": 1 << 6,
    "negative_qnan": 1 << 6,
    "positive_snan": (1 << 6) - 1,
    "negative_snan": (1 << 6) - 1,
}

_FULL_BF16_INPUT_SUM64 = 0x7FFF8000

_FULL_BF16_INPUT_XOR32 = 0


def _compute_single_precision_metrics(
    y_true,
    y_pred,
    include_percentiles,
    precision,
    inputs=None,
    flush_order="pre_round",
):
    """
    Compute error metrics for a single precision.

    Internal function called by compute_error_metrics().

    Args:
        y_true: Ground truth output values
        y_pred: Predicted output values
        include_percentiles: Include median, p95, p99 metrics
        precision: Precision for ULP calculation ('fp32', 'bf16', 'fp16')
        inputs: Optional input values for filtering (ULP calculation will filter on both inputs and outputs)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    invalid_output_count = int(np.count_nonzero(np.isfinite(y_true) & ~np.isfinite(y_pred)))
    if invalid_output_count:
        metrics = {
            "mae": float("inf"),
            "rmse": float("inf"),
            "max_error": float("inf"),
            "num_points": len(y_true),
            "invalid_output_count": invalid_output_count,
            "mean_relative_error": float("inf"),
            "max_ulp_pure": float("inf"),
            "mean_ulp_pure": float("inf"),
            "ml_pass_rate": 0.0,
            "max_ulp_error": float("inf"),
            "mean_ulp_error": float("inf"),
            "median_ulp_error": float("inf"),
            "p99_ulp_error": float("inf"),
        }
        if include_percentiles:
            metrics.update(
                median_error=float("inf"),
                p95_error=float("inf"),
                p99_error=float("inf"),
            )
        return metrics

    errors = np.abs(y_true - y_pred)

    # Filter out inf/nan values that can occur from division by zero in rational functions
    # or other numerical issues in high-degree polynomials with FP32 arithmetic
    finite_mask = np.isfinite(errors)
    finite_errors = errors[finite_mask]
    finite_diff = (y_true - y_pred)[finite_mask]

    # If all values are inf/nan, return inf for all metrics
    if len(finite_errors) == 0:
        metrics = {
            "mae": float("inf"),
            "rmse": float("inf"),
            "max_error": float("inf"),
            "num_points": len(y_true),
            "invalid_output_count": 0,
            "mean_relative_error": float("inf"),
            "max_ulp_pure": float("inf"),
            "mean_ulp_pure": float("inf"),
            "ml_pass_rate": 0.0,
            "max_ulp_error": float("inf"),
            "mean_ulp_error": float("inf"),
            "median_ulp_error": float("inf"),
            "p99_ulp_error": float("inf"),
        }
        if include_percentiles:
            metrics["median_error"] = float("inf")
            metrics["p95_error"] = float("inf")
            metrics["p99_error"] = float("inf")
        return metrics

    metrics = {
        "mae": np.mean(finite_errors),
        "rmse": np.sqrt(np.mean(finite_diff**2)),
        "max_error": np.max(finite_errors),
        "num_points": len(y_true),
        "invalid_output_count": 0,
    }

    # Mean relative error (where expected is significant)
    mask = (np.abs(y_true) > 1e-6) & finite_mask
    if np.any(mask):
        metrics["mean_relative_error"] = np.mean(errors[mask] / np.abs(y_true[mask]))
    else:
        metrics["mean_relative_error"] = 0.0

    # ULP error computation — delegated to the canonical Goldberg ULP definition
    # in ttpoly.spec.units (the SINGLE owner). units.ulp_error applies FTZ BEFORE
    # spacing and masks subnormal golden/inputs, so the device-sweep path reports
    # exactly the same robust metric the fitter reports (no divergent reimpl here).
    # Returns per-point ULP error (NaN where masked); aggregate with nan* below.
    # NOTE: the max_ulp_error / bf16_maxulp slot reports the canonical Goldberg /
    # PURE ULP-of-the-result (units.ulp_error_pure): high-precision golden in the
    # numerator, spacing at the rounded golden. This is Goldberg (1991)'s definition
    # and matches TT's ttnn-eltwise-op-tester. It is NOT the retired beta-floored
    # units.ulp_error, and NOT units.ulp_error_unfloored (which rounds both operands
    # -- a ulp-DISTANCE between two floats that understates the tester by up to 0.5).
    ulp_per_point = _units.ulp_error_pure(
        y_true,
        y_pred,
        precision=precision,
        inputs=inputs,
        flush_to_zero=True,
        flush_order=flush_order,
    )
    finite_ulp_mask = np.isfinite(ulp_per_point)

    # STANDARD HEADLINE metric 1: PURE ULP (TT ttnn-eltwise-op-tester / libm
    # convention; directly comparable to native TTNN). Reported alongside the
    # beta-floored helper below; not floored, so it legitimately reports the known
    # near-zero inflation (shared with native TTNN's own metric).
    pure_per_point = _units.ulp_error_pure(
        y_true,
        y_pred,
        precision=precision,
        inputs=inputs,
        flush_to_zero=True,
        flush_order=flush_order,
    )
    pure_max, pure_mean, pure_count = _units._summarize_float32_metric(pure_per_point)
    if pure_count:
        metrics["max_ulp_pure"] = pure_max
        metrics["mean_ulp_pure"] = pure_mean
    else:
        metrics["max_ulp_pure"] = float("nan")
        metrics["mean_ulp_pure"] = float("nan")

    # STANDARD HEADLINE metric 2: ML-tolerance pass-rate (% within
    # atol+rtol*|true|, atol=rtol=1e-3 = bf16 pytorch/numpy default).
    metrics["ml_pass_rate"] = _units.ml_pass_rate(y_true, y_pred, precision=precision)

    if np.any(finite_ulp_mask):
        filtered_ulp = ulp_per_point[finite_ulp_mask]
        metrics["max_ulp_error"] = float(np.nanmax(filtered_ulp))
        metrics["mean_ulp_error"] = float(np.nanmean(filtered_ulp))
        metrics["median_ulp_error"] = float(np.nanmedian(filtered_ulp))
        metrics["p99_ulp_error"] = float(np.nanpercentile(filtered_ulp, 99))
    else:
        # No normal points to measure. If the golden has normal values but the
        # prediction collapsed, that is a real massive error -> inf, not 0 (so it
        # never falsely "wins" a best-ULP comparison). If both are subnormal/zero,
        # there is genuinely nothing to measure -> 0.
        if np.any(np.isfinite(y_true) & (np.abs(y_true) >= 2**-126)):
            metrics["max_ulp_error"] = float("inf")
            metrics["mean_ulp_error"] = float("inf")
            metrics["median_ulp_error"] = float("inf")
            metrics["p99_ulp_error"] = float("inf")
        else:
            metrics["max_ulp_error"] = 0.0
            metrics["mean_ulp_error"] = 0.0
            metrics["median_ulp_error"] = 0.0
            metrics["p99_ulp_error"] = 0.0

    if include_percentiles:
        metrics["median_error"] = np.median(finite_errors)
        metrics["p95_error"] = np.percentile(finite_errors, 95)
        metrics["p99_error"] = np.percentile(finite_errors, 99)

    return metrics


def _classify_ieee_bits(bits, *, exponent_bits, fraction_bits, storage_dtype):
    """Count every IEEE binary class without converting the bit patterns.

    In particular, this runs before any numerical operation can quiet a signaling
    NaN or flush a subnormal.  Format geometry, not an operation name, owns all
    distinctions, so the same proof code covers binary32 and bfloat16.
    """
    dtype = np.dtype(storage_dtype)
    bits = np.asarray(bits, dtype=dtype)
    scalar = dtype.type
    sign_shift = exponent_bits + fraction_bits
    exponent_mask = (1 << exponent_bits) - 1
    fraction_mask = (1 << fraction_bits) - 1
    quiet_mask = 1 << (fraction_bits - 1)
    sign = (bits >> scalar(sign_shift)) != 0
    exponent = (bits >> scalar(fraction_bits)) & scalar(exponent_mask)
    fraction = bits & scalar(fraction_mask)
    exponent_zero = exponent == 0
    exponent_ones = exponent == exponent_mask
    fraction_zero = fraction == 0
    quiet = (fraction & scalar(quiet_mask)) != 0

    masks = {
        "positive_zero": ~sign & exponent_zero & fraction_zero,
        "negative_zero": sign & exponent_zero & fraction_zero,
        "positive_subnormal": ~sign & exponent_zero & ~fraction_zero,
        "negative_subnormal": sign & exponent_zero & ~fraction_zero,
        "positive_normal": ~sign & ~exponent_zero & ~exponent_ones,
        "negative_normal": sign & ~exponent_zero & ~exponent_ones,
        "positive_infinity": ~sign & exponent_ones & fraction_zero,
        "negative_infinity": sign & exponent_ones & fraction_zero,
        "positive_qnan": ~sign & exponent_ones & ~fraction_zero & quiet,
        "negative_qnan": sign & exponent_ones & ~fraction_zero & quiet,
        "positive_snan": ~sign & exponent_ones & ~fraction_zero & ~quiet,
        "negative_snan": sign & exponent_ones & ~fraction_zero & ~quiet,
    }
    return {name: int(np.count_nonzero(masks[name])) for name in _IEEE_CLASS_NAMES}


def _classify_bf16_bits(bits):
    return _classify_ieee_bits(bits, exponent_bits=8, fraction_bits=7, storage_dtype=np.uint16)


def _policy_classes(bits, *, precision):
    """Return the closed result classes used by typed domain actions."""
    if precision == "bf16":
        dtype = np.uint16
        sign_mask, exponent_mask, fraction_mask = 0x8000, 0x7F80, 0x007F
    elif precision == "fp32":
        dtype = np.uint32
        sign_mask, exponent_mask, fraction_mask = 0x80000000, 0x7F800000, 0x007FFFFF
    else:  # pragma: no cover - internal callers pass a closed precision set.
        raise ValueError(f"unsupported policy precision {precision!r}")
    bits = np.asarray(bits, dtype=dtype)
    scalar = np.dtype(dtype).type
    sign = (bits & scalar(sign_mask)) != 0
    exponent = bits & scalar(exponent_mask)
    fraction = bits & scalar(fraction_mask)
    result = np.full(bits.shape, "finite_other", dtype="<U12")
    result[(exponent == 0) & (fraction == 0) & ~sign] = "pos_zero"
    result[(exponent == 0) & (fraction == 0) & sign] = "neg_zero"
    result[(exponent == scalar(exponent_mask)) & (fraction == 0) & ~sign] = "pos_inf"
    result[(exponent == scalar(exponent_mask)) & (fraction == 0) & sign] = "neg_inf"
    result[(exponent == scalar(exponent_mask)) & (fraction != 0)] = "nan"
    return result


def _reference_policy_classes(values, *, precision, flush_to_zero=True):
    """Classify a mathematical reference after target rounding and FTZ."""
    values = np.asarray(values, dtype=np.float64)
    if precision == "fp32":
        rounded = _units.round_to_target(values, "fp32", flush_to_zero=flush_to_zero)
        return _policy_classes(rounded.astype(np.float32).view(np.uint32), precision="fp32")
    if precision == "bf16":
        rounded = _units.round_to_target(values, "bf16", flush_to_zero=flush_to_zero)
        bits = rounded.astype(np.float32).view(np.uint32) >> np.uint32(16)
        return _policy_classes(bits.astype(np.uint16), precision="bf16")
    raise ValueError(f"unsupported reference precision {precision!r}")


def _target_egress_policy_classes(classes, semantics):
    """Compose mathematical result classes through one bound target egress."""
    result = np.asarray(classes, dtype="<U12").copy()
    if semantics is None:
        return result
    source = result.copy()
    for input_class, output_class in semantics.result_to_egress.items():
        result[source == input_class] = output_class
    return result


def _materialize_declared_real_singularities(
    activation, inputs, reference, *, singularity_provider=real_domain_singularity_results
):
    """Apply declarative real-domain pole classes to a reference array."""

    singularities = singularity_provider(activation, inputs)
    supported = (
        (singularities == "") | (singularities == "nan") | (singularities == "neg_inf") | (singularities == "pos_inf")
    )
    if not np.all(supported):
        first = int(np.flatnonzero(~supported)[0])
        raise ValueError(
            f"{activation}: unsupported declarative singularity result "
            f"{singularities.flat[first]!r} at input index {first}"
        )
    reference[singularities == "neg_inf"] = -np.inf
    reference[singularities == "pos_inf"] = np.inf
    return singularities


def _bf16_policy_classes(bits):
    """Compatibility wrapper for the original BF16 proof API."""
    return _policy_classes(bits, precision="bf16")


def _compiled_target(compiled):
    """Return the target bound to either compiled artifact shape."""
    target = getattr(compiled, "target", None)
    if target is None:
        target = getattr(compiled.approx, "target", None)
    return None if target is None else TargetPolicy.from_value(target)


def _bf16_ingress_policy(target, semantics=None):
    """Describe the target-visible BF16 classes before Program dispatch.

    Blackhole tile-copy and common SFPU evaluation expose different NaN-sign
    quotients. The compiled semantics owns that structural distinction. The
    raw producer ledger remains unchanged; this describes only the value
    visible at mathematical dispatch.
    """
    if semantics is not None:
        ingress = semantics.raw_to_ingress
        return {
            "schema": "bf16_ingress_transport_v3",
            "kind": semantics.ingress_kind,
            "positive_nan": ingress["pos_nan"],
            "negative_nan": ingress["neg_nan"],
            "negative_zero": ingress["neg_zero"],
            "positive_subnormal": ingress["pos_subnormal"],
            "negative_subnormal": ingress["neg_subnormal"],
        }
    if target is not None and (target.precision == "bf16" and target.architecture == "bh"):
        return {
            "schema": "bf16_ingress_transport_v2",
            "kind": "bh_float16_b_srca_datacopy",
            "positive_nan": "pos_inf",
            "negative_nan": "neg_inf",
            "negative_zero": "pos_zero",
            "positive_subnormal": "pos_zero",
            "negative_subnormal": "pos_zero",
        }
    return {
        "schema": "bf16_ingress_transport_v2",
        "kind": "target_numeric_policy",
        "positive_nan": "nan",
        "negative_nan": "nan",
        "negative_zero": "preserve",
        "positive_subnormal": "pos_zero" if target and target.input_daz else "preserve",
        "negative_subnormal": "neg_zero" if target and target.input_daz else "preserve",
    }


def _bf16_effective_program_inputs(raw_inputs, target, semantics=None):
    """Apply target ingress transport without changing the raw-bit ledger."""
    values = np.asarray(raw_inputs, dtype=np.float32).copy()
    if target is None or target.precision != "bf16":
        return values

    bits32 = values.view(np.uint32)
    exponent = bits32 & np.uint32(0x7F800000)
    mantissa = bits32 & np.uint32(0x007FFFFF)
    sign = bits32 & np.uint32(0x80000000)
    subnormal = (exponent == 0) & (mantissa != 0)

    if target.architecture == "bh" and semantics is not None and (semantics.ingress_kind == "bh_float16_b_sfpu_dst"):
        raw_bf16 = (bits32 >> np.uint32(16)).astype(np.uint16)
        raw_exp = raw_bf16 & np.uint16(0x7F80)
        raw_frac = raw_bf16 & np.uint16(0x007F)
        exponent_zero = raw_exp == 0
        nan = (raw_exp == np.uint16(0x7F80)) & (raw_frac != 0)
        values[exponent_zero] = np.float32(0.0)
        values[nan] = np.float32(np.inf)
        return values
    if target.architecture == "bh":
        raw_bf16 = (bits32 >> np.uint32(16)).astype(np.uint16)
        return bh_bf16_ingress_inputs(raw_bf16)

    if target.architecture == "wh" and semantics is not None:
        # Apply the declared transport before domain-action predicates, not
        # only in the separately reported output-class quotient. In particular
        # decoded signed NaNs become infinities and DAZ signs may collapse.
        raw_bf16 = (bits32 >> np.uint32(16)).astype(np.uint16)
        masks = _bf16_raw_class_masks(raw_bf16)
        class_values = {"pos_zero": 0.0, "neg_zero": -0.0, "pos_inf": math.inf, "neg_inf": -math.inf, "nan": math.nan}
        for raw_class, ingress_class in semantics.raw_to_ingress.items():
            if ingress_class == "finite_other" or ingress_class == raw_class:
                continue
            if ingress_class not in class_values:
                raise ValueError("unsupported declared BF16 ingress class: " + ingress_class)
            values[masks[raw_class]] = np.float32(class_values[ingress_class])
        return values

    if target.input_daz:
        bits32[subnormal] = sign[subnormal]
    return bits32.view(np.float32)


def _bound_closed_structural_form(compiled):
    """Resolve one artifact-bound fixed form without activation dispatch."""
    if compiled is None or (type(compiled) is SimpleNamespace and hasattr(compiled, "semantic_closed_form_terminals")):
        return None
    from ttpoly.spec.artifact_compiler import CompiledCSVApprox
    from ttpoly.spec.closed_form import closed_form_frontier_candidates

    if not isinstance(compiled, CompiledCSVApprox):
        return None
    metadata = compiled.metadata
    block_id = str(metadata.get("closed_form_correction_block", "") or metadata.get("program_block_id", "")).strip()
    candidate_identity = str(metadata.get("closed_form_candidate_sha256", "")).strip()
    candidates = tuple(
        candidate
        for candidate in closed_form_frontier_candidates(compiled.activation_spec, compiled.target)
        if candidate.form.correction_block == block_id
        and (not candidate_identity or candidate.sha256 == candidate_identity)
    )
    if len(candidates) > 1:
        raise ValueError(f"multiple closed structural forms consume fit block {block_id!r}")
    if not candidates:
        return None
    form = candidates[0].form
    declared_kind = metadata.get("closed_form_kind")
    if declared_kind and declared_kind != form.kind:
        raise ValueError("closed structural form kind metadata drift")
    return form


def _closed_structural_form_class_policy(raw_inputs, compiled, semantics):
    """Resolve typed class terminals carried by a fixed-form certificate."""
    expected = np.full(np.asarray(raw_inputs).shape, "undeclared", dtype="<U12")
    claimed = np.zeros(expected.shape, dtype=bool)
    form = _bound_closed_structural_form(compiled)
    terminals = getattr(compiled, "semantic_closed_form_terminals", None)
    if terminals is None and form is not None:
        terminals = getattr(form, "terminals", ())
    if terminals is None:
        return expected, claimed
    if not terminals:
        return expected, claimed
    values = _bf16_effective_program_inputs(raw_inputs, _compiled_target(compiled), semantics)
    for terminal in terminals:
        if terminal.coordinate != "effective_input":
            raise ValueError("unsupported closed structural-form terminal")
        with np.errstate(invalid="ignore"):
            finite = np.isfinite(values)
            if terminal.predicate == "finite_zero":
                selected = finite & (values == np.float32(0.0))
            elif terminal.predicate == "finite_negative_integer":
                selected = finite & (values < np.float32(0.0))
                selected &= values == np.rint(values)
            elif terminal.predicate == "finite_nonpositive_integer":
                selected = finite & (values <= np.float32(0.0))
                selected &= values == np.rint(values)
            else:
                raise ValueError("unsupported closed structural-form terminal")
        expected[selected] = semantics.result_to_egress[terminal.result_class]
        claimed[selected] = True
    return expected, claimed


def _declared_class_policy(raw_inputs, compiled, semantics=None):
    """Resolve first-match typed return-class actions for effective inputs.

    Every raw-coordinate action participates in first-match ownership, even
    when its result is finite.  Otherwise a later return-class action could be
    credited for lanes which the deployed program had already assigned to a
    constant/identity/affine action.  Only return-class owners create a class
    claim; finite owners merely shadow later records.
    """
    expected = np.full(np.asarray(raw_inputs).shape, "undeclared", dtype="<U12")
    owned = np.zeros(expected.shape, dtype=bool)
    class_claimed = np.zeros(expected.shape, dtype=bool)

    semantics = semantics or _target_class_contract(compiled)
    coordinate = _bf16_effective_program_inputs(raw_inputs, _compiled_target(compiled), semantics)
    approx = getattr(compiled, "approx", None)
    for record in getattr(approx, "domain_actions", ()):
        if record.coordinate != "raw":
            raise ValueError("BF16 device class proof only supports raw-coordinate actions")
        bound = np.float32(record.bound)
        if record.direction == "below":
            selected = coordinate <= bound if record.inclusive else coordinate < bound
        else:
            selected = coordinate >= bound if record.inclusive else coordinate > bound
        selected &= ~owned
        owned[selected] = True
        if record.action.kind == "return_class":
            expected[selected] = semantics.result_to_egress[record.action.return_class]
            class_claimed[selected] = True
        elif record.action.kind == "signed_inf":
            expected[selected] = np.where(np.signbit(coordinate[selected]), "neg_inf", "pos_inf")
            class_claimed[selected] = True
    if approx is None and semantics is not None:
        raw = np.asarray(raw_inputs, dtype=np.float32)
        if semantics.target.precision == "bf16":
            bits = (raw.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
            for raw_class, selected in _bf16_raw_class_masks(bits).items():
                expected[selected] = semantics.raw_to_output[raw_class]
                class_claimed[selected] = True
    terminal_expected, terminal_claimed = _closed_structural_form_class_policy(raw_inputs, compiled, semantics)
    overlap = class_claimed & terminal_claimed
    if np.any(overlap & (expected != terminal_expected)):
        raise ValueError("closed-form terminal conflicts with domain action")
    expected[terminal_claimed] = terminal_expected[terminal_claimed]
    class_claimed |= terminal_claimed
    return _apply_post_action_raw_class_override(raw_inputs, semantics, expected, class_claimed)


def _apply_post_action_raw_class_override(raw_inputs, semantics, expected, class_claimed):
    """Apply a producer-attested late raw-class finalizer.

    The ordinary Blackhole SFPU quotient still owns Program routing: both raw
    NaN signs arrive at actions as +Inf.  A raw-DST discriminator executes
    after those actions and can therefore override their result using the
    mathematical NaN policy.  Keeping this as an explicit schedule phase
    avoids pretending the core saw a NaN coordinate that hardware erased.
    """
    overrides = getattr(semantics, "post_action_raw_override", {})
    if not overrides:
        return expected, class_claimed
    raw = np.asarray(raw_inputs, dtype=np.float32)
    bits = (raw.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    masks = _bf16_raw_class_masks(bits)
    for raw_class, policy_class in overrides.items():
        selected = masks[raw_class]
        expected[selected] = semantics.result_to_egress[semantics.mathematical_policy[policy_class]]
        class_claimed[selected] = True
    return expected, class_claimed


def _declared_bf16_class_policy(raw_inputs, compiled, semantics=None):
    """Compatibility wrapper for callers of the original BF16 proof API."""
    return _declared_class_policy(raw_inputs, compiled, semantics)


def _declared_bf16_finite_action_values(raw_inputs, compiled, semantics=None):
    """Evaluate compiler-declared finite constants over their typed regions.

    The ordered DomainAction program, not an operation name or observed output,
    owns these target values.  Other action kinds remain class-only here: their
    numeric value depends on a live coordinate/result and is already evaluated
    by the selected Program body.
    """
    raw = np.asarray(raw_inputs, dtype=np.float32)
    expected = np.zeros(raw.shape, dtype=np.uint16)
    claimed = np.zeros(raw.shape, dtype=bool)
    resolved = np.zeros(raw.shape, dtype=bool)
    approx = getattr(compiled, "approx", None)
    actions = getattr(approx, "domain_actions", ())
    if not actions:
        return expected, claimed
    coordinate = _bf16_effective_program_inputs(raw, _compiled_target(compiled), semantics)
    finite_coordinate = np.isfinite(coordinate)
    for record in actions:
        if record.coordinate != "raw":
            continue
        if record.direction == "below":
            selected = coordinate <= record.bound if record.inclusive else coordinate < record.bound
        else:
            selected = coordinate >= record.bound if record.inclusive else coordinate > record.bound
        # Domain actions are ordered.  An earlier class action owns its lanes
        # even though this helper only returns numeric constant claims; a later
        # inclusive endpoint constant must not reclaim the exterior already
        # handled by an earlier invalid-domain action.
        selected &= finite_coordinate & ~resolved
        if not np.any(selected):
            continue
        resolved |= selected
        if record.action.kind != "constant":
            continue
        rounded = _units.round_to_target(
            np.array([record.action.value], dtype=np.float64),
            "bf16",
            flush_to_zero=_compiled_target(compiled).output_ftz,
        ).astype(np.float32)
        expected[selected] = np.uint16(rounded.view(np.uint32)[0] >> np.uint32(16))
        claimed[selected] = True
    return expected, claimed


def _valid_expected_nonfinite_mismatch(valid_expected_nonfinite, observed, expected):
    """Class-check natural target overflows without inventing action ownership."""
    return np.asarray(valid_expected_nonfinite, dtype=bool) & (np.asarray(observed) != np.asarray(expected))


def _load_compiled_semantics(path, *, activation, precision):
    """Load compiler semantics without artifact identity or recompilation."""
    if path is None:
        return None
    value = json.loads(Path(path).read_text())
    required = {
        "schema",
        "activation",
        "target",
        "domain_actions",
        "special_value_policy",
        "execution_kind",
        "io_contract",
        "closed_form_terminals",
        "semantic_profile",
        "real_domain",
        "raw_class_actions",
        "target_composite_class_overlay",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("compiled semantic sidecar has invalid fields")
    forbidden = ("sha", "hash", "path", "receipt", "provenance", "identity")

    def check_keys(node):
        if isinstance(node, dict):
            for key, child in node.items():
                if any(word in str(key).lower() for word in forbidden):
                    raise ValueError("compiled semantic sidecar contains identity fields")
                check_keys(child)
        elif isinstance(node, list):
            for child in node:
                check_keys(child)

    check_keys(value)
    value = validate_scoring_semantics(value)
    if value["activation"] != activation:
        raise ValueError("compiled semantic sidecar activation mismatch")
    target = TargetPolicy.from_value(value["target"])
    if target.precision != precision:
        raise ValueError("compiled semantic sidecar precision mismatch")
    actions = parse_domain_actions(value["domain_actions"])
    policy = value["special_value_policy"]
    semantics = compile_target_class_semantics(target, policy, execution_kind=value["execution_kind"])
    io_contract = ProgramIOContract.from_value(value["io_contract"])
    terminals = value["closed_form_terminals"]
    if not isinstance(terminals, list):
        raise ValueError("compiled semantic sidecar has invalid terminal semantics")
    return SimpleNamespace(
        approx=SimpleNamespace(domain_actions=actions, target=target),
        target=target,
        target_manifest={"special_value_policy": dict(policy)},
        target_class_semantics=semantics.to_dict(),
        semantic_io_contract=io_contract,
        semantic_profile=value["semantic_profile"],
        semantic_closed_form_terminals=tuple(SimpleNamespace(**row) for row in terminals),
        semantic_real_domain=dict(value["real_domain"]),
        semantic_raw_class_actions=dict(value["raw_class_actions"]),
        semantic_target_composite_class_overlay=value["target_composite_class_overlay"],
    )


def _compiled_real_domain_mask(compiled, activation, values):
    """Use compiler-carried total-domain ownership when it is available."""
    declaration = getattr(compiled, "semantic_real_domain", None)
    if declaration is None and compiled is not None:
        from ttpoly.spec.artifact_compiler import CompiledCSVApprox

        if isinstance(compiled, CompiledCSVApprox):
            declaration = real_domain_declaration_from_spec(compiled.activation_spec)
    if declaration is not None:
        return real_domain_mask_from_declaration(declaration, values)
    return real_domain_mask(activation, values)


def _special_value_contract(compiled):
    """Return a closed input-special -> output-class policy, or ``None``.

    ``evaluator_defined`` is intentionally not evidence: it names an owner but
    does not state a checkable result class.
    """
    if compiled is None:
        return None
    manifest = getattr(compiled, "target_manifest", None)
    if manifest is None:
        approx = getattr(compiled, "approx", None)
        manifest = getattr(approx, "target_manifest", None)
    if manifest is None:
        blocks = getattr(compiled, "block_compilations", None)
        if isinstance(blocks, dict) and blocks:
            manifest = next(iter(blocks.values())).target_manifest
    contract = manifest.get("special_value_policy") if isinstance(manifest, dict) else None
    allowed = {"nan", "pos_inf", "neg_inf", "pos_zero", "neg_zero", "finite_other"}
    input_classes = ("nan", "pos_inf", "neg_inf", "pos_zero", "neg_zero")
    if not isinstance(contract, dict) or not all(name in contract for name in input_classes):
        return None
    return contract if all(str(contract[name]) in allowed for name in input_classes) else None


def _target_class_contract(compiled):
    """Return the artifact-bound raw/ingress/policy/egress composition."""
    if compiled is None:
        return None
    target = _compiled_target(compiled)
    special_contract = _special_value_contract(compiled)
    if target is None or special_contract is None:
        return None
    semantics = getattr(compiled, "target_class_semantics", None)
    if isinstance(semantics, dict):
        return target_class_semantics_from_manifest(semantics, expected_target=target)
    return compile_target_class_semantics(target, special_contract, execution_kind="sfpu")


def _raw_class_masks(bits, *, precision):
    """Return sign-aware raw classes before floating operations alter them."""
    if precision == "bf16":
        dtype = np.uint16
        sign_mask, exponent_mask, fraction_mask = 0x8000, 0x7F80, 0x007F
    elif precision == "fp32":
        dtype = np.uint32
        sign_mask, exponent_mask, fraction_mask = 0x80000000, 0x7F800000, 0x007FFFFF
    else:  # pragma: no cover - internal callers pass a closed precision set.
        raise ValueError(f"unsupported raw-class precision {precision!r}")
    raw = np.asarray(bits, dtype=dtype)
    scalar = np.dtype(dtype).type
    sign = (raw & scalar(sign_mask)) != 0
    exponent = raw & scalar(exponent_mask)
    fraction = raw & scalar(fraction_mask)
    exponent_zero = exponent == 0
    exponent_ones = exponent == scalar(exponent_mask)
    fraction_zero = fraction == 0
    masks = {
        "pos_nan": ~sign & exponent_ones & ~fraction_zero,
        "neg_nan": sign & exponent_ones & ~fraction_zero,
        "pos_inf": ~sign & exponent_ones & fraction_zero,
        "neg_inf": sign & exponent_ones & fraction_zero,
        "pos_zero": ~sign & exponent_zero & fraction_zero,
        "neg_zero": sign & exponent_zero & fraction_zero,
        "pos_subnormal": ~sign & exponent_zero & ~fraction_zero,
        "neg_subnormal": sign & exponent_zero & ~fraction_zero,
    }
    masks["finite_other"] = ~np.logical_or.reduce(tuple(masks.values()))
    return masks


def _bf16_raw_class_masks(bits):
    """Compatibility wrapper for the original BF16 proof API."""
    return _raw_class_masks(bits, precision="bf16")


def _declared_raw_class_actions(bits, compiled, *, precision, output_ftz):
    """Resolve typed late raw-class actions to output classes and exact values."""
    shape = np.asarray(bits).shape
    expected = np.full(shape, "undeclared", dtype="<U12")
    claimed = np.zeros(shape, dtype=bool)
    values = np.full(shape, np.nan, dtype=np.float64)
    value_claimed = np.zeros(shape, dtype=bool)
    if compiled is None:
        return expected, claimed, values, value_claimed
    actions = getattr(compiled, "semantic_raw_class_actions", None)
    if actions is None:
        activation_spec = getattr(compiled, "activation_spec", None)
        actions = activation_spec.get("raw_class_actions", {}) if isinstance(activation_spec, dict) else {}
    actions = actions or {}
    if not actions:
        return expected, claimed, values, value_claimed
    semantics = _target_class_contract(compiled)
    plan = compile_raw_class_terminal_plan(semantics.raw_to_output, actions)
    masks = _raw_class_masks(bits, precision=precision)
    exact_class_values = {
        "pos_zero": 0.0,
        "neg_zero": -0.0,
        "pos_inf": math.inf,
        "neg_inf": -math.inf,
        "nan": math.nan,
    }
    for raw_class, action in actions.items():
        selected = masks[raw_class]
        claimed[selected] = True
        output_class = plan.output_classes[raw_class]
        if output_class == "constant":
            value = plan.constants[raw_class]
            rounded = _units.round_to_target(
                np.array([value], dtype=np.float64),
                precision,
                flush_to_zero=output_ftz,
            )
            if precision == "bf16":
                output_bits = (rounded.astype(np.float32).view(np.uint32) >> np.uint32(16)).astype(np.uint16)
            else:
                output_bits = rounded.astype(np.float32).view(np.uint32)
            expected[selected] = _policy_classes(output_bits, precision=precision)[0]
            values[selected] = value
            value_claimed[selected] = True
        else:
            expected[selected] = output_class
            if output_class in exact_class_values:
                values[selected] = exact_class_values[output_class]
                value_claimed[selected] = True
    return expected, claimed, values, value_claimed


def _bf16_target_expected_classes(bits, compiled, semantics=None):
    """Compose target transport with compiler-declared raw terminal actions.

    ``semantics.raw_to_output`` is the base ingress/execution/egress quotient.
    A selected evaluator may then declare a late raw-class action which S55/S60
    lower after that quotient.  The exhaustive class oracle must apply the same
    ordered composition; otherwise it scores the base quotient against an
    output that the compiler explicitly changed.
    """
    semantics = semantics or _target_class_contract(compiled)
    if semantics is None:
        return None
    masks = _bf16_raw_class_masks(bits)
    expected = np.empty(np.asarray(bits).shape, dtype="<U12")
    for raw_class in RAW_INPUT_CLASSES:
        expected[masks[raw_class]] = semantics.raw_to_output[raw_class]
    action_expected, action_claimed, _, _ = _declared_raw_class_actions(
        bits,
        compiled,
        precision="bf16",
        output_ftz=semantics.target.output_ftz,
    )
    expected[action_claimed] = action_expected[action_claimed]
    return expected


def _bf16_target_composite_class_override(bits, compiled):
    """Return a typed target-composite-owned BF16 class partition."""
    overlay = getattr(compiled, "semantic_target_composite_class_overlay", None)
    if overlay is None:
        return np.zeros(np.asarray(bits).shape, dtype=bool), None
    if overlay["kind"] == "raw_special_class_parity_scope":
        if overlay["population_scope"] != "declared_invalid_nonfinite_or_raw_special_only":
            raise ValueError("unsupported raw-special class-parity population scope")
        return np.zeros(np.asarray(bits).shape, dtype=bool), None
    if overlay["kind"] == "finite_sum_power_tail_class_repair":
        raw = np.asarray(bits, dtype=np.uint16)
        values = (raw.astype(np.uint32) << np.uint32(16)).view(np.float32)
        with np.errstate(invalid="ignore"):
            selected = np.isfinite(values) & (values < np.float32(0.0)) & (values == np.floor(values))
        expected = np.empty(raw.shape, dtype="<U12")
        expected[:] = "undeclared"
        transitions = overlay["negative_integer_transitions"]
        for index, transition in enumerate(transitions):
            first = np.uint16(transition["first_raw"])
            last = (
                np.uint16(transitions[index + 1]["first_raw"] - 1)
                if index + 1 < len(transitions)
                else np.uint16(0xFFFF)
            )
            owned = selected & (raw >= first) & (raw <= last)
            output_class = transition["output_class"]
            expected[owned] = "finite_other" if output_class == "finite" else output_class
        # A target-composite class relation is a special-value parity rule,
        # not authority to replace a finite mathematical-reference lane.
        # In particular, the target graph happens to round the large finite
        # positive tail to +0, but those inputs remain governed by the ordinary
        # finite validity and ULP gates.
        if overlay["population_scope"] != "declared_invalid_nonfinite_or_raw_special_only":
            raise ValueError("unsupported finite-sum class-repair population scope")
        return selected, expected
    if overlay["kind"] == "exponent_bucket_target_class_repair":
        raw = np.asarray(bits, dtype=np.uint16)
        expected = np.full(raw.shape, "undeclared", dtype="<U12")
        exponent = raw & np.uint16(0x7F80)
        mantissa = raw & np.uint16(0x007F)
        positive_nan = ((raw & np.uint16(0x8000)) == 0) & (exponent == np.uint16(0x7F80)) & (mantissa != 0)
        zero_subnormal = exponent == 0
        selected = positive_nan | zero_subnormal
        expected[positive_nan] = overlay["positive_nan_output_class"]
        expected[zero_subnormal] = overlay["zero_subnormal_output_class"]
        values = (raw.astype(np.uint32) << np.uint32(16)).view(np.float32)
        with np.errstate(invalid="ignore"):
            negative_integer = np.isfinite(values) & (values < np.float32(0.0)) & (values == np.floor(values))
        transitions = overlay["negative_integer_transitions"]
        for index, transition in enumerate(transitions):
            first = np.uint16(transition["first_raw"])
            last = (
                np.uint16(transitions[index + 1]["first_raw"] - 1)
                if index + 1 < len(transitions)
                else np.uint16(0xFFFF)
            )
            owned = negative_integer & (raw >= first) & (raw <= last)
            selected |= owned
            kind = transition["output_class"]
            expected[owned] = "finite_other" if kind == "finite" else kind
        return selected, expected
    if overlay["kind"] != "native_even_polyval_exterior":
        raise ValueError("unsupported target composite class overlay")
    if overlay["different_class_result"] not in {"target_value", "target_class_representative"}:
        raise ValueError("unsupported native-even target class relation")
    from ttpoly.precision.target_composite import (
        NativeEvenPolyvalExteriorComposite,
        native_even_polyval_exterior_finite_mask,
        native_even_polyval_exterior_output,
    )

    raw = np.asarray(bits, dtype=np.uint16)
    composite = NativeEvenPolyvalExteriorComposite(tuple(float(item) for item in overlay["coefficients"]))
    selected = native_even_polyval_exterior_finite_mask(
        raw,
        composite,
        open_interval_bound=float(overlay["open_interval_bound"]),
    )
    output_bits = native_even_polyval_exterior_output(raw, composite)
    return selected, _bf16_policy_classes(output_bits)


def _class_confusion(expected, observed, mask):
    """Compact expected->observed class counts for one proof population."""
    result = {}
    for expected_name, observed_name in zip(expected[mask], observed[mask]):
        key = f"{expected_name}->{observed_name}"
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


def _condensed_ieee_classes(classes):
    return {
        "zeros": classes["positive_zero"] + classes["negative_zero"],
        "subnormals": (classes["positive_subnormal"] + classes["negative_subnormal"]),
        "normals": classes["positive_normal"] + classes["negative_normal"],
        "infinities": (classes["positive_infinity"] + classes["negative_infinity"]),
        "nans": sum(classes[name] for name in _IEEE_CLASS_NAMES if "nan" in name),
    }


def _write_metrics_record(metrics):
    """Write the stable 11-column accuracy record consumed by run_csv.sh."""
    print(
        f"{metrics['mae']},{metrics['rmse']},{metrics['max_error']},"
        f"{metrics['mean_relative_error']},"
        f"{metrics['max_ulp_error']},{metrics['mean_ulp_error']},"
        f"{metrics['median_ulp_error']},{metrics['p99_ulp_error']},"
        f"{metrics.get('max_ulp_pure', float('nan'))},"
        f"{metrics.get('mean_ulp_pure', float('nan'))},"
        f"{metrics.get('ml_pass_rate', float('nan'))}"
    )


def _atomic_write_coverage(path, coverage):
    """Publish a complete proof without exposing a partial JSON document."""
    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(coverage, sort_keys=True, indent=2) + "\n"
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=summary_path.parent,
            prefix=f".{summary_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, summary_path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _ml_tolerance_counts(y_true, y_pred, precision):
    """Return the exact numerator/denominator used by ``ml_pass_rate``."""
    y_true = np.atleast_1d(np.asarray(y_true, dtype=np.float64))
    y_pred = np.atleast_1d(np.asarray(y_pred, dtype=np.float64))
    comp = _units._downcast_ftz(y_pred, precision, flush_to_zero=True).astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        absolute_error = np.abs(y_true - comp)
        tolerance = 1e-3 + 1e-3 * np.abs(y_true)
    defined = np.isfinite(y_true) & np.isfinite(tolerance)
    passed = defined & np.isfinite(absolute_error) & (absolute_error <= tolerance)
    return int(np.count_nonzero(defined)), int(np.count_nonzero(passed))


def _run_bf16_exhaustive_raw(
    activation,
    data_file,
    coverage_summary_path=None,
    class_mismatch_details_path=None,
    skip_subnormal_inputs=False,
    coefficient_csv=None,
    semantic_sidecar=None,
    class_reference_bf16=None,
    retained_raw_path=None,
    attest_fused_gradient_one=False,
    target_architecture="bh",
    intermediate_precision="fp32",
    input_daz=True,
    output_ftz=True,
    rounding_order="post_round",
    package_rows_identity=None,
    package_rows_output=None,
    *,
    compiler=None,
    reference_provider=reference_value,
    domain_mask_provider=real_domain_mask,
):
    """Reduce raw little-endian BF16 outputs for the FULL 2^16 input space.

    Deliberately not a port of the fp32 streaming reducer: 65,536 points is 128
    KiB, so the whole space is classified in one shot. Numeric metrics use the
    canonical high-precision reference only on finite, real-domain inputs; all
    remaining encodings stay visible in separate conformance populations.
    """
    expected_bytes = _BF16_SPACE_SIZE * 2
    close_stream = str(data_file) != "-"
    stream = open(data_file, "rb") if close_stream else sys.stdin.buffer
    try:
        raw = stream.read(expected_bytes)
        if len(raw) != expected_bytes:
            raise RuntimeError(f"raw BF16 output truncated: {len(raw)}/{expected_bytes} bytes")
        if stream.read(1):
            raise ValueError(f"raw BF16 output contains trailing data: expected exactly " f"{expected_bytes} bytes")
    finally:
        if close_stream:
            stream.close()

    def _widen(b16):
        # bf16 is a truncated binary32: shifting the pattern left 16 bits gives
        # the EXACT value with no rounding, and preserves NaN payloads.
        return (b16.astype(np.uint32) << np.uint32(16)).view(np.float32)

    bits = np.arange(_BF16_SPACE_SIZE, dtype=np.uint16)
    out_bits = np.frombuffer(raw, dtype="<u2")
    class_reference_bits = None
    if class_reference_bf16 is not None:
        reference_raw = Path(class_reference_bf16).read_bytes()
        if len(reference_raw) != expected_bytes:
            raise ValueError("BF16 class reference must contain exactly 65536 raw words")
        class_reference_bits = np.frombuffer(reference_raw, dtype="<u2")
    input_classes = _classify_bf16_bits(bits)
    output_classes = _classify_bf16_bits(out_bits)
    if input_classes != _FULL_BF16_CLASS_COUNTS:
        raise RuntimeError("full-space bfloat16 input class cardinalities mismatch")
    input_sum = int(np.sum(bits.astype(np.uint64), dtype=np.uint64)) & ((1 << 64) - 1)
    input_xor = int(np.bitwise_xor.reduce(bits))
    if input_sum != _FULL_BF16_INPUT_SUM64 or input_xor != _FULL_BF16_INPUT_XOR32:
        raise RuntimeError("full-space bfloat16 input checksum constants mismatch")
    inputs = _widen(bits)
    outputs = _widen(out_bits)
    if coefficient_csv is not None and compiler is None:
        raise ValueError("CSV scoring requires an explicit compiler adapter; use semantic_sidecar for runtime scoring")
    compiled = (
        compiler(
            coefficient_csv,
            activation=activation,
            precision="bf16",
            architecture=(
                {
                    "bh": "bh",
                    "blackhole": "bh",
                    "wh": "wh",
                    "wormhole": "wh",
                    "wormhole_b0": "wh",
                }.get(target_architecture, target_architecture)
                if package_rows_output is not None
                else target_architecture
            ),
            intermediate_precision=intermediate_precision,
            input_daz=input_daz,
            output_ftz=output_ftz,
            rounding_order=rounding_order,
            compliance="ttnn" if package_rows_output is not None else None,
            bind_ttmetal_llk_schedule=package_rows_output is not None,
        )
        if coefficient_csv is not None
        else None
    )
    if semantic_sidecar is not None:
        if compiled is not None:
            raise ValueError("use either semantic sidecar or coefficient CSV, not both")
        compiled = _load_compiled_semantics(semantic_sidecar, activation=activation, precision="bf16")

    auxiliary_inputs = None
    if attest_fused_gradient_one:
        from ttpoly.spec.activation_config import operation_io_contract_from_config

        io_contract = getattr(compiled, "semantic_io_contract", None)
        if io_contract is None:
            activation_spec = getattr(compiled, "activation_spec", None)
            io_contract = (
                operation_io_contract_from_config(activation_spec) if isinstance(activation_spec, dict) else None
            )
        if io_contract is None or not io_contract.fuse_grad:
            raise ValueError("--attest-fused-gradient-one requires a bound fused-gradient " "activation I/O contract")
        auxiliary_inputs = [
            {
                "tensor_index": io_contract.gradient_input_index,
                "role": "incoming_gradient",
                "encoding": "constant_bfloat16",
                "raw_bits": "0x3f80",
                "value": 1.0,
                "count": _BF16_SPACE_SIZE,
            }
        ]

    # All 65,536 encodings receive one conformance disposition. Numeric metrics
    # are deliberately narrower: only finite inputs in the real mathematical
    # domain whose canonical golden is also finite. The campaign interval is an
    # independently reported subset; it never turns raw encoding coverage into a
    # misleading all-encoding numeric claim.
    target_policy = (
        _compiled_target(compiled)
        if compiled is not None
        else TargetPolicy(
            "bf16",
            target_architecture,
            intermediate_precision,
            input_daz,
            output_ftz,
            rounding_order,
        )
    )
    target_class_semantics = _target_class_contract(compiled)
    if class_reference_bits is not None and (compiled is None or getattr(compiled, "semantic_profile", None) != "ttnn"):
        raise ValueError("--bf16-class-reference is valid only for compiler-bound ttnn semantics")
    if (
        retained_raw_path is not None
        and str(data_file) != "-"
        and (Path(retained_raw_path).resolve() != Path(data_file).resolve())
    ):
        raise ValueError("retained BF16 raw path does not match scored artifact")
    raw_zero_boundary_override = None
    # The target contract owns the value presented to the mathematical
    # function.  On an input-DAZ target, headline accuracy therefore compares
    # the device output with f(DAZ(raw)); scoring f(raw) instead asks the SFPU to
    # recover information which its declared ingress has already discarded.
    # Keep the exact-raw view below as a diagnostic strengthened-contract axis.
    ingress_inputs = _bf16_effective_program_inputs(
        inputs,
        target_policy,
        target_class_semantics if target_policy.architecture == "wh" else None,
    )
    headline_inputs = ingress_inputs if target_policy.input_daz else inputs
    finite_inputs = np.isfinite(headline_inputs)
    real_valid = finite_inputs & (
        domain_mask_provider(activation, headline_inputs)
        if compiled is None
        else _compiled_real_domain_mask(compiled, activation, headline_inputs)
    )
    reference = np.full(inputs.shape, np.nan, dtype=np.float64)
    if np.any(real_valid):
        with np.errstate(all="ignore"):
            reference[real_valid] = reference_provider(activation, headline_inputs[real_valid], use_cache=False)
    if class_reference_bits is not None:
        # TTNN profiles may declaratively distinguish physical subnormal
        # encodings even though the mathematical target has input-DAZ.  The
        # exhaustive API oracle is value-authoritative only where the compiler
        # sidecar carries a matching finite-constant raw-class action.  Without
        # that declaration it remains class-only evidence; importing its numeric
        # value would create target semantics outside the compiler contract.
        # Ordinary normal/zero arithmetic remains independently compared with
        # the mathematical reference above.
        raw_class_actions = getattr(compiled, "semantic_raw_class_actions", {})
        value_authoritative_subnormal_classes = {
            raw_class
            for raw_class in ("pos_subnormal", "neg_subnormal")
            if raw_class_actions.get(raw_class, {}).get("kind") == "constant"
        }
        raw_subnormal_mask = ((bits & np.uint16(0x7F80)) == 0) & ((bits & np.uint16(0x007F)) != 0) & real_valid
        sign = (bits & np.uint16(0x8000)) != 0
        declared_value_mask = np.zeros(bits.shape, dtype=bool)
        if "pos_subnormal" in value_authoritative_subnormal_classes:
            declared_value_mask |= raw_subnormal_mask & ~sign
        if "neg_subnormal" in value_authoritative_subnormal_classes:
            declared_value_mask |= raw_subnormal_mask & sign
        api_values = _widen(class_reference_bits).astype(np.float64)
        api_finite = np.isfinite(api_values)
        reference[declared_value_mask & api_finite] = api_values[declared_value_mask & api_finite]
    _materialize_declared_real_singularities(activation, headline_inputs, reference)
    (
        raw_action_expected,
        raw_action_claimed,
        _raw_action_values,
        _raw_action_value_claimed,
    ) = _declared_raw_class_actions(
        bits,
        compiled,
        precision="bf16",
        output_ftz=target_policy.output_ftz,
    )
    finite_action_expected_bits = np.zeros(bits.shape, dtype=np.uint16)
    finite_action_claimed = np.zeros(bits.shape, dtype=bool)
    if compiled is not None:
        finite_action_expected_bits, finite_action_claimed = _declared_bf16_finite_action_values(
            inputs, compiled, target_class_semantics
        )
        # A finite constant DomainAction is part of the compiled target
        # function, not merely a class epilogue.  Score exactly that declared
        # BF16 value on owned finite lanes.
        reference[finite_action_claimed] = _widen(finite_action_expected_bits[finite_action_claimed]).astype(np.float64)
    rounded_reference = _units.round_to_target(
        reference,
        target_policy.precision,
        flush_to_zero=target_policy.output_ftz,
    )
    finite_target_reference = np.isfinite(rounded_reference)
    finite_reference = real_valid & finite_target_reference
    valid_expected_nonfinite = real_valid & ~finite_target_reference
    if class_reference_bits is not None:
        # A compliance profile may declaratively replace an otherwise numeric
        # Torch lane with an API terminal (for example a signed-zero action).
        # The exhaustive TTNN class oracle is authoritative for that ownership:
        # do not leave such a lane in the numeric population and manufacture an
        # infinite ULP error against the pre-profile mathematical value.
        api_reference_class = _bf16_policy_classes(class_reference_bits)
        api_numeric = np.isin(api_reference_class, ("finite_other", "pos_zero", "neg_zero"))
        finite_reference &= api_numeric
        valid_expected_nonfinite |= real_valid & ~api_numeric
    invalid_domain = finite_inputs & ~real_valid
    special_inputs = ~finite_inputs

    normal_or_zero = ((bits & np.uint16(0x7F80)) != 0) | ((bits & np.uint16(0x7FFF)) == 0)
    raw_finite = np.isfinite(inputs)
    raw_real_valid = raw_finite & _compiled_real_domain_mask(compiled, activation, inputs)
    raw_subnormal = raw_finite & ~normal_or_zero
    diagnostic_subnormal_numeric = raw_real_valid & raw_subnormal
    diagnostic_subnormal_reference = np.full(inputs.shape, np.nan, dtype=np.float64)
    if np.any(diagnostic_subnormal_numeric):
        with np.errstate(all="ignore"):
            diagnostic_subnormal_reference[diagnostic_subnormal_numeric] = reference_provider(
                activation,
                inputs[diagnostic_subnormal_numeric],
                use_cache=False,
            )
    diagnostic_subnormal_numeric &= np.isfinite(
        _units.round_to_target(
            diagnostic_subnormal_reference,
            target_policy.precision,
            flush_to_zero=target_policy.output_ftz,
        )
    )
    input_policy_excluded = np.zeros(bits.shape, dtype=bool)
    contract = finite_reference
    contract_points = int(np.count_nonzero(contract))
    if contract_points == 0:
        raise RuntimeError("BF16 finite-valid numeric population is empty")
    contract_input_classes = _classify_bf16_bits(bits[contract])
    contract_output_classes = _classify_bf16_bits(out_bits[contract])
    outside_contract_output_classes = _classify_bf16_bits(out_bits[~contract])
    with np.errstate(all="ignore"):
        contract_outputs = outputs[contract]
        y_true = reference[contract]
        metrics = _compute_single_precision_metrics(
            y_true,
            contract_outputs,
            include_percentiles=True,
            precision="bf16",
            inputs=headline_inputs[contract],
            flush_order=target_policy.rounding_order,
        )

    # Exact integers are the source of truth for the CSV rate; no rounded
    # percentage is reverse-engineered into a numerator.
    ml_valid, ml_pass = _ml_tolerance_counts(y_true, contract_outputs, "bf16")
    metrics["ml_pass_rate"] = 100.0 * ml_pass / ml_valid if ml_valid else float("nan")
    _write_metrics_record(metrics)

    finite_reference_metric = np.isfinite(y_true)
    finite_output = np.isfinite(contract_outputs)
    with np.errstate(all="ignore"):
        finite_error = np.isfinite(y_true - contract_outputs.astype(np.float64))
        pure = _units.ulp_error_pure(
            y_true,
            contract_outputs,
            precision="bf16",
            inputs=headline_inputs[contract],
            flush_to_zero=True,
            flush_order=target_policy.rounding_order,
        )
    pure_count = int(np.count_nonzero(np.isfinite(pure)))
    # Keep exact gate numerators alongside the floating summaries.  Non-finite
    # pure-ULP values are failures for a finite target reference, just as they
    # are for the ML gate; excluding them would reward a broken finite path.
    numeric_population_count = ml_valid
    numeric_failure_count = numeric_population_count - ml_pass
    pure_ulp_ge_one_count = int(np.count_nonzero(~np.isfinite(pure) | (pure >= 1.0)))
    condensed_contract_outputs = _condensed_ieee_classes(contract_output_classes)
    condensed_outside_outputs = _condensed_ieee_classes(outside_contract_output_classes)
    population_masks = {
        "finite_valid_numeric": contract,
        "invalid_domain": invalid_domain,
        "valid_expected_nonfinite": valid_expected_nonfinite,
        "all_encoding_special": special_inputs,
        "input_policy_excluded": input_policy_excluded,
    }
    if sum(int(np.count_nonzero(mask)) for mask in population_masks.values()) != _BF16_SPACE_SIZE:
        raise RuntimeError("BF16 conformance populations do not partition the input space")
    population_counts = {name: int(np.count_nonzero(mask)) for name, mask in population_masks.items()}
    population_input_classes = {name: _classify_bf16_bits(bits[mask]) for name, mask in population_masks.items()}
    population_output_classes = {name: _classify_bf16_bits(out_bits[mask]) for name, mask in population_masks.items()}

    subnormal_outputs = outputs[diagnostic_subnormal_numeric]
    subnormal_reference = diagnostic_subnormal_reference[diagnostic_subnormal_numeric]
    subnormal_ml_valid, subnormal_ml_pass = _ml_tolerance_counts(subnormal_reference, subnormal_outputs, "bf16")
    with np.errstate(all="ignore"):
        subnormal_pure = _units.ulp_error_pure(
            subnormal_reference,
            subnormal_outputs,
            precision="bf16",
            inputs=inputs[diagnostic_subnormal_numeric],
            flush_to_zero=True,
            flush_order=target_policy.rounding_order,
        )
    finite_subnormal_pure = subnormal_pure[np.isfinite(subnormal_pure)]
    mathematical_subnormal_quality = {
        "authority": "diagnostic_non_headline",
        "input_reference_policy": "exact_raw_bf16_before_declared_input_daz",
        "count": int(np.count_nonzero(diagnostic_subnormal_numeric)),
        "input_classes": _classify_bf16_bits(bits[diagnostic_subnormal_numeric]),
        "output_classes": _classify_bf16_bits(out_bits[diagnostic_subnormal_numeric]),
        "pure_ulp_count": int(finite_subnormal_pure.size),
        "max_pure_ulp": (float(np.max(finite_subnormal_pure)) if finite_subnormal_pure.size else None),
        "mean_pure_ulp": (float(np.mean(finite_subnormal_pure)) if finite_subnormal_pure.size else None),
        "ml_valid": subnormal_ml_valid,
        "ml_pass": subnormal_ml_pass,
        "ml_pass_rate": (100.0 * subnormal_ml_pass / subnormal_ml_valid if subnormal_ml_valid else None),
    }
    certified_pre_daz_raw_boundary_quality = None

    observed_policy = _bf16_policy_classes(out_bits)
    policy_expected = np.full(bits.shape, "undeclared", dtype="<U12")
    expected_nonfinite_policy = _reference_policy_classes(
        reference,
        precision="bf16",
        flush_to_zero=target_policy.output_ftz,
    )
    expected_nonfinite_policy = _target_egress_policy_classes(expected_nonfinite_policy, target_class_semantics)

    declared_action_mismatch_count = None
    declared_finite_action_claim_count = None
    declared_finite_action_mismatch_count = None
    invalid_domain_mismatch_count = None
    expected_nonfinite_mismatch_count = None
    signed_zero_mismatch_count = None
    total_class_policy_mismatch_count = None
    class_wrong = np.zeros(bits.shape, dtype=bool)
    if compiled is not None:
        # Resolve raw encodings through the target ingress transport before
        # Program predicates, while retaining the original bits separately for
        # encoding-population evidence.
        action_expected, action_claimed = _declared_bf16_class_policy(inputs, compiled, target_class_semantics)
        # Raw-class actions are a late compiler phase and supersede the base
        # transport and earlier Program-owned classes on their raw partitions.
        # This is the scorer-side composition performed by S55/S60.
        action_expected[raw_action_claimed] = raw_action_expected[raw_action_claimed]
        action_claimed |= raw_action_claimed
        finite_action_wrong = finite_action_claimed & (out_bits != finite_action_expected_bits)
        finite_action_expected_policy = _bf16_policy_classes(finite_action_expected_bits)
        action_expected[finite_action_claimed] = finite_action_expected_policy[finite_action_claimed]
        action_claimed |= finite_action_claimed
        if raw_zero_boundary_override is not None:
            raw_owned = raw_zero_boundary_override["selected"]
            action_expected[raw_owned] = raw_zero_boundary_override["expected"][raw_owned]
            action_claimed[raw_owned] = True
        composite_owned, composite_expected = _bf16_target_composite_class_override(bits, compiled)
        if composite_expected is not None:
            # The target graph supersedes the selected evaluator's terminal
            # action on this typed partition; S60 makes the same ownership
            # transfer before emission.
            action_claimed[composite_owned] = True
            action_expected[composite_owned] = composite_expected[composite_owned]
            finite_action_wrong[composite_owned] = False
            expected_nonfinite_policy[composite_owned] = composite_expected[composite_owned]
        policy_expected[action_claimed] = action_expected[action_claimed]
        if composite_expected is not None:
            policy_expected[composite_owned] = composite_expected[composite_owned]
        # Invalid-domain values need an explicit typed action: there is no
        # mathematical result for the core to own.  In contrast, a valid input
        # whose correctly rounded target result is non-finite (for example a
        # legitimate BF16 overflow) is an ordinary result of the evaluator.
        # It must match the target IEEE class, but must not be forced through a
        # terminal action merely to make that natural overflow admissible.
        invalid_expected = expected_nonfinite_policy
        invalid_wrong = invalid_domain & (
            ~action_claimed | (action_expected != invalid_expected) | (observed_policy != invalid_expected)
        )
        expected_nonfinite_wrong = _valid_expected_nonfinite_mismatch(
            valid_expected_nonfinite, observed_policy, expected_nonfinite_policy
        )
        declared_wrong = (action_claimed & (observed_policy != action_expected)) | finite_action_wrong
        class_wrong = invalid_wrong | expected_nonfinite_wrong | declared_wrong
        declared_action_mismatch_count = int(np.count_nonzero(declared_wrong))
        declared_finite_action_claim_count = int(np.count_nonzero(finite_action_claimed))
        declared_finite_action_mismatch_count = int(np.count_nonzero(finite_action_wrong))
        invalid_domain_mismatch_count = int(np.count_nonzero(invalid_wrong))
        expected_nonfinite_mismatch_count = int(np.count_nonzero(expected_nonfinite_wrong))
        signed_zero_claims = action_claimed & np.isin(action_expected, ("pos_zero", "neg_zero"))
        signed_zero_mismatch_count = int(np.count_nonzero(signed_zero_claims & (observed_policy != action_expected)))
        total_class_policy_mismatch_count = int(np.count_nonzero(class_wrong))
    else:
        action_claimed = np.zeros(bits.shape, dtype=bool)
        action_expected = np.full(bits.shape, "undeclared", dtype="<U12")
        finite_action_claimed = np.zeros(bits.shape, dtype=bool)

    zero_inputs = (bits & np.uint16(0x7FFF)) == 0
    target_class_expected = _bf16_target_expected_classes(bits, compiled, target_class_semantics)
    if target_class_expected is not None and raw_zero_boundary_override is not None:
        raw_owned = raw_zero_boundary_override["selected"]
        # This is a post-action finite-class claim. It changes only ownership
        # evidence; DAZ class conformance remains separate from the exact-input
        # mathematical numeric population.
        target_class_expected[raw_owned] = raw_zero_boundary_override["expected"][raw_owned]
    zero_expected = (
        target_class_expected[zero_inputs]
        if target_class_expected is not None
        else np.full(np.count_nonzero(zero_inputs), "undeclared", dtype="<U12")
    )
    zero_wrong = zero_inputs.copy()
    zero_wrong[zero_inputs] = observed_policy[zero_inputs] != zero_expected
    signed_zero_mismatch_count = int(np.count_nonzero(zero_wrong))
    signed_zero_policy_status = "declared_target_transport_class" if target_class_expected is not None else "undeclared"
    policy_expected[zero_inputs] = zero_expected

    subnormal_inputs = ~normal_or_zero & finite_inputs
    subnormal_expected = (
        target_class_expected[subnormal_inputs]
        if target_class_expected is not None
        else np.full(np.count_nonzero(subnormal_inputs), "undeclared", dtype="<U12")
    )
    subnormal_wrong = subnormal_inputs.copy()
    subnormal_wrong[subnormal_inputs] = observed_policy[subnormal_inputs] != subnormal_expected
    subnormal_policy_mismatch_count = int(np.count_nonzero(subnormal_wrong))
    subnormal_policy_status = (
        (
            "declared_target_transport_with_certified_pre_DAZ_ownership"
            if raw_zero_boundary_override is not None
            else "declared_target_transport_class"
        )
        if target_class_expected is not None
        else "undeclared"
    )
    policy_expected[subnormal_inputs] = subnormal_expected

    # Special policies are supplied by the precision-instantiated target
    # manifest.  Domain actions can claim infinities, but NaNs never satisfy an
    # ordered comparison, so actions alone cannot constitute a complete policy.
    special_policy_mismatch_count = None
    special_policy_status = "undeclared"
    if target_class_expected is not None:
        special_expected = target_class_expected
        special_policy_mismatch_count = int(np.count_nonzero(special_inputs & (observed_policy != special_expected)))
        special_policy_status = "declared_target_transport"
        policy_expected[special_inputs] = special_expected[special_inputs]

    special_wrong = (
        special_inputs & (observed_policy != policy_expected)
        if target_class_expected is not None
        else np.zeros(bits.shape, dtype=bool)
    )
    all_class_wrong = class_wrong | zero_wrong | subnormal_wrong | special_wrong
    if class_reference_bits is not None:
        # TTNN compliance is differential by definition.  Bind its class gate
        # to the exhaustive output of the exact TTNN baseline instead of a
        # guessed generic-SFPU quotient. Numeric accuracy remains independently
        # scored against the mathematical reference above.
        reference_policy = _bf16_policy_classes(class_reference_bits)
        class_overlay = getattr(compiled, "semantic_target_composite_class_overlay", None)
        finite_reference_precedence = (
            isinstance(class_overlay, dict)
            and class_overlay.get("population_scope") == "declared_invalid_nonfinite_or_raw_special_only"
        )
        # TTNN is the differential authority for transported input classes,
        # not for the numeric value of compiler-declared finite terminal
        # actions.  Those actions have already been checked, bit for bit,
        # against their typed constant above (``finite_action_wrong``), and
        # their mathematical quality is independently scored against the
        # golden.  Letting TTNN replace that expectation turns a more accurate
        # finite terminal into a class failure whenever TTNN itself flushes or
        # overflows it to a different class (softsign's signed-unit tails and
        # softsign_bw's two min-normal endpoints are concrete examples).
        class_action_owned = action_claimed & ~finite_action_claimed
        if finite_reference_precedence:
            class_action_owned &= ~finite_reference
        reference_owned = (
            class_action_owned
            | invalid_domain
            | valid_expected_nonfinite
            | special_inputs
            | zero_inputs
            | subnormal_inputs
        )
        policy_expected[reference_owned] = reference_policy[reference_owned]
        reference_wrong = reference_owned & (observed_policy != reference_policy)
        all_class_wrong = finite_action_wrong | reference_wrong
        declared_action_mismatch_count = int(
            np.count_nonzero(finite_action_wrong | (class_action_owned & (observed_policy != reference_policy)))
        )
        invalid_domain_mismatch_count = int(np.count_nonzero(invalid_domain & (observed_policy != reference_policy)))
        expected_nonfinite_mismatch_count = int(
            np.count_nonzero(valid_expected_nonfinite & (observed_policy != reference_policy))
        )
        signed_zero_mismatch_count = int(np.count_nonzero(zero_inputs & (observed_policy != reference_policy)))
        subnormal_policy_mismatch_count = int(
            np.count_nonzero(subnormal_inputs & (observed_policy != reference_policy))
        )
        special_policy_mismatch_count = int(np.count_nonzero(special_inputs & (observed_policy != reference_policy)))
        signed_zero_policy_status = "declared_ttnn_exhaustive_reference"
        subnormal_policy_status = "declared_ttnn_exhaustive_reference"
        special_policy_status = "declared_ttnn_exhaustive_reference"
    total_class_policy_mismatch_count = int(np.count_nonzero(all_class_wrong))
    policies_complete = (
        special_policy_status.startswith("declared")
        and signed_zero_policy_status.startswith("declared")
        and subnormal_policy_status.startswith("declared")
    )
    if compiled is None:
        device_conformance_status = "unbound_traversal_only"
    elif total_class_policy_mismatch_count:
        device_conformance_status = "declared_class_fail"
    elif policies_complete:
        device_conformance_status = "declared_class_pass"
    else:
        device_conformance_status = "declared_class_incomplete"

    expected_class_counts = {
        name: int(np.count_nonzero(policy_expected == name))
        for name in (
            "nan",
            "pos_inf",
            "neg_inf",
            "pos_zero",
            "neg_zero",
            "finite_other",
            "undeclared",
        )
    }
    class_confusion_matrices = {
        name: _class_confusion(policy_expected, observed_policy, mask) for name, mask in population_masks.items()
    }

    # Field NAMES deliberately match PRODUCER_PROOF_FIELDS in
    # tools/ttnn_ref_csv.py so verify-coverage can cross-check bf16 the same way
    # it cross-checks fp32, instead of the gate being bypassed for bf16.
    coverage = {
        "schema": BF16_EXHAUSTIVE_EVIDENCE_SCHEMA,
        "complete": True,
        "precision": "bf16",
        "input_encoding": "sequential_ieee754_bfloat16_bit_patterns",
        "input_transport_policy": _bf16_ingress_policy(target_policy, target_class_semantics),
        "output_encoding": "raw_little_endian_bfloat16",
        "semantic_profile": getattr(compiled, "semantic_profile", "unbound_reference"),
        "class_reference_kind": (
            "ttnn_bf16_exhaustive" if class_reference_bits is not None else "compiled_target_semantics"
        ),
        "start_bit": 0,
        "count": _BF16_SPACE_SIZE,
        "end_bit_exclusive": _BF16_SPACE_SIZE,
        "chunks": 1,
        "expected_bytes": expected_bytes,
        "observed_bytes": len(raw),
        "last_bit": _BF16_SPACE_SIZE - 1,
        "finite_valid_numeric_points": contract_points,
        "contract_points": contract_points,
        "outside_contract_points": _BF16_SPACE_SIZE - contract_points,
        "metric_scope": "finite_valid_numeric_normals_and_zeros",
        "mathematical_quality_scope": (
            "finite_valid_bf16_after_declared_input_daz" if target_policy.input_daz else "all_finite_valid_bf16_inputs"
        ),
        "finite_valid_subnormals_in_headline_metrics": not target_policy.input_daz,
        "same_ingress_daz_conformance_reported_separately": False,
        "same_ingress_daz_is_headline_reference": target_policy.input_daz,
        "finite_input_daz_transformed_encoding_count": int(
            np.count_nonzero(
                finite_inputs & ~normal_or_zero & (inputs.view(np.uint32) != ingress_inputs.view(np.uint32))
            )
        ),
        "mathematical_subnormal_quality": mathematical_subnormal_quality,
        "coverage": "bf16_all_encoding_traversal_with_declared_class_checks",
        "coverage_kind": "all_encoding_traversal_with_declared_class_checks",
        "device_conformance_status": device_conformance_status,
        "raw_zero_boundary_ownership": (
            {key: value for key, value in raw_zero_boundary_override.items() if key not in {"selected", "expected"}}
            if raw_zero_boundary_override is not None
            else None
        ),
        "expected_class_counts": expected_class_counts,
        "class_confusion_matrices": class_confusion_matrices,
        "declared_action_claim_count": int(np.count_nonzero(action_claimed)),
        "declared_action_mismatch_count": declared_action_mismatch_count,
        "declared_finite_action_claim_count": declared_finite_action_claim_count,
        "declared_finite_action_mismatch_count": (declared_finite_action_mismatch_count),
        "invalid_domain_mismatch_count": invalid_domain_mismatch_count,
        "expected_nonfinite_mismatch_count": expected_nonfinite_mismatch_count,
        "special_policy_status": special_policy_status,
        "special_policy_mismatch_count": special_policy_mismatch_count,
        "signed_zero_policy_status": signed_zero_policy_status,
        "signed_zero_mismatch_count": signed_zero_mismatch_count,
        "subnormal_policy_status": subnormal_policy_status,
        "subnormal_policy_mismatch_count": subnormal_policy_mismatch_count,
        "total_class_policy_mismatch_count": total_class_policy_mismatch_count,
        "output_checksum": hashlib.sha256(raw).hexdigest(),
        "retained_raw_artifact": (
            {
                "path": str(Path(retained_raw_path).resolve()),
                "output_checksum": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
            if retained_raw_path is not None
            else None
        ),
        "input_classes": input_classes,
        "output_classes": output_classes,
        "contract_input_classes": contract_input_classes,
        "contract_output_classes": contract_output_classes,
        "outside_contract_output_classes": outside_contract_output_classes,
        "population_counts": population_counts,
        "population_input_classes": population_input_classes,
        "population_output_classes": population_output_classes,
        "contract_nonfinite_outputs": (condensed_contract_outputs["infinities"] + condensed_contract_outputs["nans"]),
        "outside_contract_nonfinite_outputs": (
            condensed_outside_outputs["infinities"] + condensed_outside_outputs["nans"]
        ),
        "outside_contract_references_evaluated": False,
        "contract_input_reference_policy": (
            "declared_input_daz_before_mathematical_golden"
            if target_policy.input_daz
            else "exact_bf16_value_before_mathematical_golden"
        ),
        "contract_input_ftz_threshold": _units.MIN_NORMAL,
        "metric_counts": {
            "finite_reference": int(np.count_nonzero(finite_reference_metric)),
            "nonfinite_reference": int(np.count_nonzero(~finite_reference_metric)),
            "finite_output": int(np.count_nonzero(finite_output)),
            "nonfinite_output": int(np.count_nonzero(~finite_output)),
            "finite_error": int(np.count_nonzero(finite_error)),
            "pure_ulp": pure_count,
            "ml_valid": ml_valid,
            "ml_pass": ml_pass,
            "invalid_output_for_finite_reference": int(np.count_nonzero(finite_reference_metric & ~finite_output)),
            "numeric_failure_count": numeric_failure_count,
            "pure_ulp_ge_one_count": pure_ulp_ge_one_count,
        },
        "numeric_population_count": numeric_population_count,
        "numeric_failure_count": numeric_failure_count,
        "pure_ulp_ge_one_count": pure_ulp_ge_one_count,
    }
    if certified_pre_daz_raw_boundary_quality is not None:
        coverage["certified_pre_daz_raw_boundary_quality"] = certified_pre_daz_raw_boundary_quality
    if auxiliary_inputs is not None:
        coverage["auxiliary_inputs"] = auxiliary_inputs
    if package_rows_output is not None:
        if class_reference_bits is None:
            raise ValueError("BF16 package rows require an exhaustive TTNN raw reference")
        if compiled is None or getattr(compiled, "semantic_profile", None) != "ttnn":
            raise ValueError("BF16 package rows require compiler-bound ttnn semantics")
        if not isinstance(package_rows_identity, dict):
            raise ValueError("BF16 package rows require an identity JSON object")
        identity_fields = (
            "operation",
            "family",
            "architecture",
            "source_commit",
            "tt_metal_commit",
            "job",
            "chip",
        )
        missing_identity = [field for field in identity_fields if not package_rows_identity.get(field)]
        if missing_identity:
            raise ValueError(f"BF16 package identity missing fields: {missing_identity}")
        if str(package_rows_identity["operation"]) != activation:
            raise ValueError("BF16 package identity operation does not match activation")
        architecture_names = {
            "bh": "blackhole",
            "blackhole": "blackhole",
            "wh": "wormhole_b0",
            "wormhole": "wormhole_b0",
            "wormhole_b0": "wormhole_b0",
        }
        target_architecture_name = architecture_names.get(str(target_architecture))
        if target_architecture_name is None:
            raise ValueError(f"unsupported BF16 package architecture {target_architecture!r}")
        if package_rows_identity["architecture"] != target_architecture_name:
            raise ValueError("BF16 package identity architecture does not match compiler target")
        if architecture_names.get(str(compiled.target.architecture)) != target_architecture_name:
            raise ValueError("BF16 package architecture does not match bound compiler semantics")

        # Reuse the exact reference, target rounding, domain partition, class
        # composition, and PURE-ULP implementation above.  This is a raw-row
        # projection of the authoritative exhaustive scorer, not a second
        # scorer with subtly different policy.
        package_expected = policy_expected.copy()
        numeric_class_owned = finite_reference & (package_expected == "undeclared")
        package_expected[numeric_class_owned] = expected_nonfinite_policy[numeric_class_owned]
        undeclared = np.flatnonzero(package_expected == "undeclared")
        if undeclared.size:
            raise ValueError(
                "compiler semantics left package result classes undeclared " f"for {undeclared.size} BF16 encodings"
            )
        stock_outputs = _widen(class_reference_bits)
        with np.errstate(all="ignore"):
            stock_pure = _units.ulp_error_pure(
                y_true,
                stock_outputs[contract],
                precision="bf16",
                inputs=headline_inputs[contract],
                flush_to_zero=True,
                flush_order=target_policy.rounding_order,
            )
        candidate_pure_ulp = [None] * _BF16_SPACE_SIZE
        ttnn_pure_ulp = [None] * _BF16_SPACE_SIZE
        for raw_index, candidate_value, stock_value in zip(np.flatnonzero(contract), pure, stock_pure):
            index = int(raw_index)
            candidate_pure_ulp[index] = float(candidate_value) if np.isfinite(candidate_value) else None
            ttnn_pure_ulp[index] = float(stock_value) if np.isfinite(stock_value) else None
        package_expected_counts = {
            name: int(np.count_nonzero(package_expected == name))
            for name in (
                "nan",
                "pos_inf",
                "neg_inf",
                "pos_zero",
                "neg_zero",
                "finite_other",
            )
        }
        package_rows = {
            "schema": "ttmetal_llk_exhaustive_rows_v1",
            **{field: package_rows_identity[field] for field in identity_fields},
            "semantic_profile": "ttnn",
            "coverage_kind": "all_encoding_traversal_with_declared_class_checks",
            "input_words": [int(value) for value in bits],
            "reference_is_finite": [bool(value) for value in contract],
            "candidate_is_finite": [bool(value) for value in np.isfinite(outputs)],
            "expected_classes": [str(value) for value in package_expected],
            "candidate_classes": [str(value) for value in observed_policy],
            "candidate_pure_ulp": candidate_pure_ulp,
            "ttnn_pure_ulp": ttnn_pure_ulp,
            "expected_class_counts": package_expected_counts,
        }
        _atomic_write_coverage(package_rows_output, package_rows)
    if coverage_summary_path:
        _atomic_write_coverage(coverage_summary_path, coverage)
    if class_mismatch_details_path:
        raw_input_class = np.empty(bits.shape, dtype="<U16")
        for name, mask in _bf16_raw_class_masks(bits).items():
            raw_input_class[mask] = name

        def mismatch_records(mask):
            records = []
            for raw_input in np.flatnonzero(mask):
                raw_input = int(raw_input)
                records.append(
                    {
                        "expected_output_class": str(policy_expected[raw_input]),
                        "input_raw_class": str(raw_input_class[raw_input]),
                        "observed_output_class": str(observed_policy[raw_input]),
                        "output_raw": f"0x{int(out_bits[raw_input]):04x}",
                        "raw": f"0x{raw_input:04x}",
                    }
                )
            return records

        finite_mismatches = mismatch_records(all_class_wrong & finite_inputs)
        nonfinite_mismatches = mismatch_records(all_class_wrong & special_inputs)
        mismatch_sets = {
            "finite": finite_mismatches,
            "nonfinite": nonfinite_mismatches,
        }
        details = {
            "schema": "bf16_class_mismatch_details_v1",
            "input_count": _BF16_SPACE_SIZE,
            "finite_input_count": int(np.count_nonzero(finite_inputs)),
            "nonfinite_input_count": int(np.count_nonzero(special_inputs)),
            "partition_complete": bool(
                np.all(finite_inputs ^ special_inputs)
                and not np.any(finite_inputs & special_inputs)
                and len(finite_mismatches) + len(nonfinite_mismatches) == total_class_policy_mismatch_count
            ),
            "coverage_total_class_policy_mismatch_count": (total_class_policy_mismatch_count),
            "finite_class_mismatches": finite_mismatches,
            "nonfinite_class_mismatches": nonfinite_mismatches,
            "output_checksum": hashlib.sha256(raw).hexdigest(),
        }
        _atomic_write_coverage(class_mismatch_details_path, details)
    print(
        "BF16_EXHAUSTIVE_COVERAGE,"
        f"count={_BF16_SPACE_SIZE},contract={contract_points},"
        f"scope={coverage['metric_scope']}",
        file=sys.stderr,
        flush=True,
    )
    return coverage
