# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Validation helpers for activation-level search and asymptote declarations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from ttpoly.spec.program import ProgramIOContract
from ttpoly.spec.program import TargetPolicy


SPECIAL_INPUT_CLASSES = ("nan", "pos_inf", "neg_inf", "pos_zero", "neg_zero")
SPECIAL_OUTPUT_CLASSES = frozenset({"nan", "pos_inf", "neg_inf", "pos_zero", "neg_zero", "finite_other"})
SPECIAL_VALUE_VALIDATION_FIELDS = frozenset({"first_pass", "transport", "mathematical"})
SPECIAL_VALUE_VALIDATION_VALUES = {
    "first_pass": "first_pass_ttnn_special_parity",
    "transport": "raw_ieee_transport_pending",
    "mathematical": "second_pass_mathematical_conformance",
}

TARGET_ACCURACY_FIRST_PASS = "first_pass_ttnn_finite_parity"
TARGET_ACCURACY_MATHEMATICAL = "second_pass_mathematical_conformance"
TARGET_ACCURACY_EXCEPTION_KINDS = frozenset({"origin_half_slope_ttnn_ftz"})

TARGET_EXECUTION_KINDS = frozenset(
    {
        "sfpu_ingress_quotient",
        "sfpu_raw_nan",
        "tti_signed_nonfinite_quotient",
        "tti_signed_negative_nan",
        "tti_signed_raw_nan_finalizer",
        "tti_signed_positive_nan",
    }
)


@dataclass(frozen=True)
class RationalSearch:
    enabled: bool
    segment_counts: tuple[int, ...]
    max_denominator_excess: int
    computed_segment_counts: tuple[int, ...] = ()


@dataclass(frozen=True)
class PiecewiseBoundaryPolicy:
    """Typed ownership for mathematical boundaries between adjacent pieces.

    The historical segment cascade is right-owned (``x >= threshold``).  An
    omitted owner vector therefore retains that behavior byte-for-byte.  New
    declarations keep the mathematical breakpoint exact and state only which
    adjacent piece owns equality; target-specific physical thresholds are a
    separate lowering concern.
    """

    breakpoints: tuple[float, ...]
    boundary_owners: tuple[str, ...]
    explicit: bool


def piecewise_boundary_policy_from_config(
    config: Mapping[str, Any],
) -> PiecewiseBoundaryPolicy | None:
    """Validate the closed piecewise-boundary schema without parsing prose.

    ``domain_condition`` remains documentation and is intentionally not an
    executable grammar.  Boundary ownership is compiler data.  Missing
    ``boundary_owners`` means the legacy all-right convention so existing
    configs and coefficient regeneration remain stable until explicitly
    migrated.
    """
    if not isinstance(config, Mapping):
        raise ValueError("activation config must be an object")
    piecewise = config.get("piecewise")
    if piecewise is None:
        return None
    if not isinstance(piecewise, Mapping):
        raise ValueError("piecewise must be an object")

    raw_breakpoints = piecewise.get("breakpoints")
    raw_pieces = piecewise.get("pieces")
    if not isinstance(raw_breakpoints, list):
        raise ValueError("piecewise.breakpoints must be an array")
    if not isinstance(raw_pieces, list) or not raw_pieces:
        raise ValueError("piecewise.pieces must be a nonempty array")
    if len(raw_pieces) != len(raw_breakpoints) + 1:
        raise ValueError("piecewise.pieces must contain exactly one more entry than breakpoints")

    breakpoints: list[float] = []
    for index, value in enumerate(raw_breakpoints):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"piecewise.breakpoints[{index}] must be a finite number")
        point = float(value)
        if not math.isfinite(point):
            raise ValueError(f"piecewise.breakpoints[{index}] must be a finite number")
        breakpoints.append(point)
    if any(left >= right for left, right in zip(breakpoints, breakpoints[1:])):
        raise ValueError("piecewise.breakpoints must be strictly increasing with no duplicates")

    for index, piece in enumerate(raw_pieces):
        if not isinstance(piece, Mapping) or piece.get("index") != index:
            raise ValueError("piecewise.pieces indices must be contiguous and ordered from zero")

    raw_owners = piecewise.get("boundary_owners")
    explicit = raw_owners is not None
    if raw_owners is None:
        owners = ("right",) * len(breakpoints)
    else:
        if not isinstance(raw_owners, list) or len(raw_owners) != len(breakpoints):
            raise ValueError("piecewise.boundary_owners must contain one entry per breakpoint")
        if any(owner not in {"left", "right"} for owner in raw_owners):
            raise ValueError("piecewise.boundary_owners entries must be 'left' or 'right'")
        owners = tuple(raw_owners)

    return PiecewiseBoundaryPolicy(tuple(breakpoints), owners, explicit)


def bh_daz_fp32_piecewise_breakpoints(config: Mapping[str, Any]) -> tuple[float, ...]:
    """Materialize a piecewise policy on Blackhole's decoded FP32 fit lane.

    The mathematical policy remains authoritative. This helper only derives
    the historical fit coordinate: right-owned boundaries use the exact FP32
    value, while strict left ownership uses the FP32 successor (MIN_NORMAL at
    decoded zero after DAZ). No activation name or coefficient artifact is
    consulted.
    """
    policy = piecewise_boundary_policy_from_config(config)
    if policy is None or not policy.explicit:
        raise ValueError("bh_daz_fp32 piecewise boundaries require explicit " "piecewise.boundary_owners")
    thresholds: list[float] = []
    for breakpoint, owner in zip(policy.breakpoints, policy.boundary_owners):
        point = np.float32(breakpoint)
        if float(point) != float(breakpoint):
            raise ValueError("bh_daz_fp32 piecewise breakpoint is not exactly FP32")
        if owner == "right":
            threshold = point
        elif point == np.float32(0.0):
            threshold = np.float32(np.finfo(np.float32).tiny)
        else:
            threshold = np.nextafter(point, np.float32(np.inf), dtype=np.float32)
        thresholds.append(float(threshold))
    if thresholds != sorted(set(thresholds)):
        raise ValueError("bh_daz_fp32 piecewise thresholds are not strictly ordered")
    return tuple(thresholds)


def piecewise_special_factors_from_config(
    config: Mapping[str, Any],
) -> dict[str, float]:
    """Materialize nonfinite factors from the piecewise mathematical contract.

    A class-only ``finite_other`` policy cannot tell the golden which finite
    number to return.  Explicit piecewise specs therefore carry one scalar or
    IEEE class token for each nonfinite input class.  The resolver applies the
    I/O contract's optional gradient scale and verifies each resulting class
    against ``special_value_policy``.  It never derives a value from an
    activation name or depends on an executable lowering graph.
    """
    policy = piecewise_boundary_policy_from_config(config)
    if policy is None or not policy.explicit:
        raise ValueError("piecewise special factors require explicit boundary_owners")
    io_contract = operation_io_contract_from_config(config)
    piecewise = config["piecewise"]
    factors = piecewise.get("special_factors")
    input_classes = frozenset({"nan", "pos_inf", "neg_inf"})
    if not isinstance(factors, Mapping) or set(factors) != input_classes:
        raise ValueError("piecewise.special_factors must contain exactly nan, pos_inf, " "and neg_inf")
    class_values = {
        "nan": math.nan,
        "pos_inf": math.inf,
        "neg_inf": -math.inf,
        "pos_zero": 0.0,
        "neg_zero": -0.0,
    }
    materialized: dict[str, float] = {}
    for input_class in sorted(input_classes):
        value = factors[input_class]
        if isinstance(value, bool):
            raise ValueError(f"piecewise.special_factors.{input_class} must be numeric or " "an IEEE class token")
        if isinstance(value, (int, float)):
            factor = float(value)
        elif value in class_values:
            factor = class_values[value]
        else:
            raise ValueError(f"piecewise.special_factors.{input_class} must be numeric or " "an IEEE class token")
        if io_contract.gradient_scale is not None:
            factor *= float(io_contract.gradient_scale)
        materialized[input_class] = factor

    declared_classes = special_value_policy_from_config(config)

    def output_class(value: float) -> str:
        if math.isnan(value):
            return "nan"
        if value == math.inf:
            return "pos_inf"
        if value == -math.inf:
            return "neg_inf"
        if value == 0.0:
            return "neg_zero" if math.copysign(1.0, value) < 0.0 else "pos_zero"
        return "finite_other"

    for input_class, factor in materialized.items():
        observed = output_class(factor)
        expected = declared_classes[input_class]
        if observed != expected:
            raise ValueError(
                f"materialized {input_class} piecewise factor class {observed!r} "
                f"disagrees with special_value_policy {expected!r}"
            )
    return materialized


def piecewise_boundary_lowering_certificates(config, target):
    """Bind every explicit mathematical owner to one target-lattice threshold."""
    policy = piecewise_boundary_policy_from_config(config)
    if policy is None or not policy.explicit:
        return ()
    from ttpoly.spec.target_lattice import certify_piecewise_boundary_threshold

    return tuple(
        certify_piecewise_boundary_threshold(target, breakpoint=breakpoint, owner=owner)
        for breakpoint, owner in zip(policy.breakpoints, policy.boundary_owners)
    )


def operation_io_contract_from_config(
    config: Mapping[str, Any],
) -> ProgramIOContract:
    """Validate schema-v3 tensor arity and optional fused-gradient composition.

    Unary forward specs predate ``tensor_inputs`` and retain the closed default
    role ``activation_input``.  Fused-gradient operations have no such default:
    they must explicitly declare both tensor roles in ABI order so a missing or
    reversed gradient layout cannot silently compile.
    """
    if not isinstance(config, Mapping):
        raise ValueError("activation config must be an object")
    version = config.get("activation_schema_version")
    if type(version) is not int or version < 3:
        raise ValueError("operation I/O contract requires activation_schema_version >= 3")
    contract = config.get("contract")
    if contract is None:
        return ProgramIOContract()
    if not isinstance(contract, Mapping):
        raise ValueError("activation config contract must be an object")
    arity = contract.get("arity")
    if type(arity) is not int or arity < 1:
        raise ValueError("contract.arity must be a positive integer")
    fuse_grad = contract.get("fuse_grad", False)
    if type(fuse_grad) is not bool:
        raise ValueError("contract.fuse_grad must be boolean")
    gradient_composition = contract.get("gradient_composition")
    if fuse_grad and gradient_composition not in {
        "multiply",
        "zero_mask_select",
        "zone_select_multiply",
    }:
        raise ValueError("contract.fuse_grad=true requires explicit gradient_composition")
    if not fuse_grad and gradient_composition is not None:
        raise ValueError("contract.gradient_composition requires contract.fuse_grad=true")
    gradient_scale = contract.get("gradient_scale")
    if "gradient_scale" in contract:
        if (
            isinstance(gradient_scale, bool)
            or not isinstance(gradient_scale, (int, float))
            or not math.isfinite(float(gradient_scale))
        ):
            raise ValueError("contract.gradient_scale must be a finite number")
        if gradient_composition == "zone_select_multiply":
            raise ValueError("contract.zone_select_multiply does not accept gradient_scale")
        if gradient_composition not in {"multiply", "zero_mask_select"}:
            raise ValueError("contract.gradient_scale requires a fused gradient composition")

    raw_inputs = contract.get("tensor_inputs")
    if raw_inputs is None:
        if fuse_grad:
            raise ValueError("contract.fuse_grad=true requires explicit contract.tensor_inputs")
        if arity != 1:
            raise ValueError("multi-input contract requires explicit contract.tensor_inputs")
        return ProgramIOContract(arity, False, ("activation_input",))
    if not isinstance(raw_inputs, list) or len(raw_inputs) != arity:
        raise ValueError("contract.tensor_inputs must contain exactly contract.arity entries")

    names: set[str] = set()
    roles = []
    allowed_roles = {"activation_input", "incoming_gradient"}
    for index, raw in enumerate(raw_inputs):
        path = f"contract.tensor_inputs[{index}]"
        if not isinstance(raw, Mapping) or set(raw) != {"name", "role"}:
            raise ValueError(f"{path} must contain exactly name and role")
        name, role = raw["name"], raw["role"]
        if not isinstance(name, str) or not name:
            raise ValueError(f"{path}.name must be a nonempty string")
        if name in names:
            raise ValueError(f"contract.tensor_inputs repeats name {name!r}")
        if role not in allowed_roles:
            raise ValueError(f"{path}.role has unknown tensor role {role!r}")
        names.add(name)
        roles.append(role)

    role_tuple = tuple(roles)
    if fuse_grad:
        if arity != 2:
            raise ValueError("contract.fuse_grad=true requires contract.arity=2")
        if role_tuple != ("incoming_gradient", "activation_input"):
            raise ValueError("fused-gradient tensor layout must be incoming_gradient then activation_input")
    elif "incoming_gradient" in role_tuple:
        raise ValueError("incoming_gradient tensor role requires contract.fuse_grad=true")
    if role_tuple.count("activation_input") != 1:
        raise ValueError("contract.tensor_inputs must contain one activation_input role")
    return ProgramIOContract(
        arity,
        fuse_grad,
        role_tuple,
        gradient_composition,
        gradient_scale,
    )


def campaign_domain_from_config(config: Mapping[str, Any]) -> tuple[float, float]:
    """Return the finite fit/search interval, never the runtime valid domain.

    ``campaign_domain`` is the canonical name.  During the compatibility
    window an activation may also carry the legacy ``domain`` spelling, but if
    both are present they must describe the same interval.  This keeps old CSV
    regeneration byte-stable while preventing new code from silently treating
    a fitting interval as the mathematical contract.
    """
    if not isinstance(config, Mapping):
        raise ValueError("activation config must be an object")
    canonical = config.get("campaign_domain")
    legacy = config.get("domain")
    if canonical is None and legacy is None:
        raise ValueError("activation config must declare campaign_domain")

    def normalize(value: Any, path: str) -> tuple[float, float]:
        if not isinstance(value, Mapping):
            raise ValueError(f"{path} must be an object")
        required = {"min", "max"}
        missing = required - set(value)
        unknown = set(value) - required
        if missing:
            raise ValueError(f"{path} missing fields: {sorted(missing)}")
        if unknown:
            raise ValueError(f"{path} has unknown fields: {sorted(unknown)}")
        if any(isinstance(value[key], bool) for key in required):
            raise ValueError(f"{path} bounds must be finite numbers")
        try:
            lo, hi = float(value["min"]), float(value["max"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path} bounds must be finite numbers") from exc
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError(f"{path} must be a finite nonempty interval")
        return lo, hi

    selected = normalize(
        canonical if canonical is not None else legacy,
        "campaign_domain" if canonical is not None else "domain",
    )
    if canonical is not None and legacy is not None:
        legacy_bounds = normalize(legacy, "domain")
        if selected != legacy_bounds:
            raise ValueError("campaign_domain and deprecated domain disagree")
    return selected


def fixed_parameter_bindings_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return closed compile-time parameter bindings declared by a schema-v3 spec.

    Parameterized activation specs duplicate their selected constants in the
    mathematical ``contract`` and the legacy top-level ``parameters`` object.
    Treating either copy as a default would silently compile a different
    operation when they drift.  This validator therefore requires exact names,
    JSON-scalar types, and values before a Program template can be bound.
    """
    if not isinstance(config, Mapping):
        raise ValueError("activation config must be an object")
    contract = config.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("activation config must declare a contract object")
    declared = contract.get("parameters")
    if not isinstance(declared, list):
        raise ValueError("contract.parameters must be an array")
    legacy = config.get("parameters")
    if not declared:
        if legacy not in (None, {}):
            raise ValueError("top-level parameters must be empty when contract.parameters is empty")
        return {}
    if not isinstance(legacy, Mapping):
        raise ValueError("parameterized activation must declare top-level parameters")

    bindings: dict[str, Any] = {}
    for index, raw in enumerate(declared):
        path = f"contract.parameters[{index}]"
        if not isinstance(raw, Mapping) or set(raw) != {"name", "value"}:
            raise ValueError(f"{path} must contain exactly name and value")
        name, value = raw["name"], raw["value"]
        if not isinstance(name, str) or not name:
            raise ValueError(f"{path}.name must be a nonempty string")
        if name in bindings:
            raise ValueError(f"contract.parameters repeats {name!r}")
        if value is None or not isinstance(value, (bool, int, float, str)):
            raise ValueError(f"{path}.value must be a JSON scalar")
        bindings[name] = value

    if set(legacy) != set(bindings):
        raise ValueError("top-level parameters disagree with contract.parameters names")
    for name, value in bindings.items():
        actual = legacy[name]
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f"top-level parameters.{name} disagrees with contract binding")
    return bindings


def public_parameter_bindings_from_config(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return an explicitly declared public subset, without removing math constants.

    Absence retains the legacy interpretation: every fixed binding is public.
    An explicit empty list declares an API without selectable scalar arguments;
    mathematical constants remain in fixed_parameter_bindings_from_config.
    """
    if not isinstance(config, Mapping):
        raise ValueError("activation config must be an object")
    contract = config.get("contract")
    if not isinstance(contract, Mapping) or "public_parameters" not in contract:
        return None
    names = contract["public_parameters"]
    if not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("contract.public_parameters must be an array of parameter names")
    if len(names) != len(set(names)):
        raise ValueError("contract.public_parameters contains duplicate names")
    fixed = fixed_parameter_bindings_from_config(config)
    if not set(names) <= set(fixed):
        raise ValueError("contract.public_parameters must be a subset of fixed parameter names")
    return {name: fixed[name] for name in names}


def _counts(value: Any, path: str, *, allow_empty: bool = False) -> tuple[int, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{path} must be a {'possibly empty ' if allow_empty else 'nonempty '}array")
    if any(type(item) is not int or item < 1 for item in value):
        raise ValueError(f"{path} entries must be positive integers")
    result = tuple(value)
    if tuple(sorted(set(result))) != result:
        raise ValueError(f"{path} must be sorted with no duplicates")
    return result


def rational_search_from_config(config: Mapping[str, Any]) -> RationalSearch:
    """Return the closed rational-grid policy declared by one activation spec."""
    value = config.get("rational_search")
    if not isinstance(value, Mapping):
        raise ValueError("activation config must declare a rational_search object")
    required = {"enabled", "segment_counts", "max_denominator_excess"}
    allowed = required | {"computed_segment_counts"}
    missing = required - set(value)
    unknown = set(value) - allowed
    if missing:
        raise ValueError(f"rational_search missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"rational_search has unknown fields: {sorted(unknown)}")
    if type(value["enabled"]) is not bool:
        raise ValueError("rational_search.enabled must be boolean")
    excess = value["max_denominator_excess"]
    if type(excess) is not int or excess < 0:
        raise ValueError("rational_search.max_denominator_excess must be a nonnegative integer")
    segments = _counts(value["segment_counts"], "rational_search.segment_counts")
    computed = _counts(
        value.get("computed_segment_counts", []),
        "rational_search.computed_segment_counts",
        allow_empty=True,
    )
    if not value["enabled"] and (excess != 0 or computed):
        raise ValueError("disabled rational_search cannot declare expanded search axes")
    return RationalSearch(value["enabled"], segments, excess, computed)


def asymptotes_from_config(config: Mapping[str, Any]) -> dict[str, float]:
    """Validate optional finite mathematical limits, independently of fit tails."""
    value = config.get("asymptotes")
    if value is None:
        return {}
    if not isinstance(value, Mapping) or not value:
        raise ValueError("asymptotes must be a nonempty object")
    unknown = set(value) - {"left", "right"}
    if unknown:
        raise ValueError(f"asymptotes has unknown fields: {sorted(unknown)}")
    result: dict[str, float] = {}
    for direction, constant in value.items():
        if isinstance(constant, bool) or not isinstance(constant, (int, float)):
            raise ValueError(f"asymptotes.{direction} must be a finite number")
        number = float(constant)
        if not math.isfinite(number):
            raise ValueError(f"asymptotes.{direction} must be finite")
        result[direction] = number
    return result


def special_value_policy_from_config(config: Mapping[str, Any]) -> dict[str, str]:
    """Validate the complete input-class to expected output-class contract."""
    value = config.get("special_value_policy")
    if not isinstance(value, Mapping):
        raise ValueError("activation config must declare special_value_policy")
    required = set(SPECIAL_INPUT_CLASSES)
    missing = required - set(value)
    unknown = set(value) - required
    if missing:
        raise ValueError(f"special_value_policy missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"special_value_policy has unknown fields: {sorted(unknown)}")
    for input_class, output_class in value.items():
        if output_class not in SPECIAL_OUTPUT_CLASSES:
            raise ValueError(f"special_value_policy.{input_class} has unknown class {output_class!r}")
    return {key: value[key] for key in SPECIAL_INPUT_CLASSES}


def special_value_validation_from_config(
    config: Mapping[str, Any],
) -> dict[str, str] | None:
    """Validate the optional, explicitly separated special-value authority axes.

    First-pass TTNN parity is defined on identical physical ingress.  It must
    not be confused with either raw IEEE transport preservation or a scalar
    mathematical expression evaluated at a non-finite input; those are
    intentionally retained as independent second-pass obligations.
    """
    value = config.get("special_value_validation")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("special_value_validation must be an object")
    missing = SPECIAL_VALUE_VALIDATION_FIELDS - set(value)
    unknown = set(value) - SPECIAL_VALUE_VALIDATION_FIELDS
    if missing:
        raise ValueError(f"special_value_validation missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"special_value_validation has unknown fields: {sorted(unknown)}")
    for axis, expected in SPECIAL_VALUE_VALIDATION_VALUES.items():
        if value[axis] != expected:
            raise ValueError(f"special_value_validation.{axis} must be {expected!r}")
    return {key: value[key] for key in sorted(SPECIAL_VALUE_VALIDATION_FIELDS)}


def target_accuracy_exceptions_from_config(
    config: Mapping[str, Any],
    target: TargetPolicy | Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Resolve explicitly separated finite first-pass parity exceptions.

    These declarations never alter the mathematical reference or hide its
    error.  They only authorize a target-bound first-pass error comparison
    against TTNN on the same physical ingress: candidate pure ULP must be no
    worse than TTNN, but may be mathematically better without matching TTNN's
    output bits. Second-pass mathematical conformance remains named and
    independently reportable.
    """
    records = config.get("target_accuracy_validation")
    if records is None:
        return ()
    if not isinstance(records, list) or not records:
        raise ValueError("target_accuracy_validation must be a nonempty array")
    requested = TargetPolicy.from_value(target)
    matches: list[tuple[dict[str, Any], ...]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or set(record) != {"target", "first_pass", "mathematical", "exceptions"}:
            raise ValueError(f"target_accuracy_validation[{index}] has invalid fields")
        declared = TargetPolicy.from_value(record["target"])
        if record["first_pass"] != TARGET_ACCURACY_FIRST_PASS:
            raise ValueError(
                f"target_accuracy_validation[{index}].first_pass must be " f"{TARGET_ACCURACY_FIRST_PASS!r}"
            )
        if record["mathematical"] != TARGET_ACCURACY_MATHEMATICAL:
            raise ValueError(
                f"target_accuracy_validation[{index}].mathematical must be " f"{TARGET_ACCURACY_MATHEMATICAL!r}"
            )
        raw_exceptions = record["exceptions"]
        if not isinstance(raw_exceptions, list) or not raw_exceptions:
            raise ValueError(f"target_accuracy_validation[{index}].exceptions must be nonempty")
        parsed: list[dict[str, Any]] = []
        for exception_index, exception in enumerate(raw_exceptions):
            if not isinstance(exception, Mapping) or set(exception) != {"kind", "slope"}:
                raise ValueError(
                    f"target_accuracy_validation[{index}].exceptions" f"[{exception_index}] has invalid fields"
                )
            kind = exception["kind"]
            slope = exception["slope"]
            if kind not in TARGET_ACCURACY_EXCEPTION_KINDS:
                raise ValueError(f"unsupported target accuracy exception {kind!r}")
            if isinstance(slope, bool) or not isinstance(slope, (int, float)):
                raise ValueError("target accuracy exception slope must be numeric")
            parsed.append({"kind": kind, "slope": float(slope)})
        if declared == requested:
            matches.append(tuple(parsed))
    if len(matches) > 1:
        raise ValueError("target_accuracy_validation repeats the requested target")
    return matches[0] if matches else ()


def target_execution_kind_from_config(
    config: Mapping[str, Any],
    target: TargetPolicy | Mapping[str, Any],
    *,
    default: str,
) -> str:
    """Resolve an optional exact-target execution transport declaration.

    Mathematical special policy remains independent.  This declaration binds
    the physical producer quotient used for first-pass target parity; it never
    selects an activation, evaluator, coefficient set, or runtime fallback.
    """
    records = config.get("target_execution")
    if records is None:
        return default
    if not isinstance(records, list) or not records:
        raise ValueError("target_execution must be a nonempty array")
    typed_target = TargetPolicy.from_value(target)
    matches: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or set(record) != {"target", "kind"}:
            raise ValueError(f"target_execution[{index}] must contain exactly target and kind")
        declared_target = TargetPolicy.from_value(record["target"])
        kind = record["kind"]
        if kind not in TARGET_EXECUTION_KINDS:
            raise ValueError(f"target_execution[{index}].kind is unsupported: {kind!r}")
        if declared_target == typed_target:
            matches.append(kind)
    if len(matches) > 1:
        raise ValueError("target_execution repeats the requested exact target")
    return matches[0] if matches else default


__all__ = [
    "RationalSearch",
    "SPECIAL_INPUT_CLASSES",
    "SPECIAL_OUTPUT_CLASSES",
    "asymptotes_from_config",
    "campaign_domain_from_config",
    "fixed_parameter_bindings_from_config",
    "public_parameter_bindings_from_config",
    "rational_search_from_config",
    "operation_io_contract_from_config",
    "special_value_policy_from_config",
    "special_value_validation_from_config",
    "target_accuracy_exceptions_from_config",
    "target_execution_kind_from_config",
]
