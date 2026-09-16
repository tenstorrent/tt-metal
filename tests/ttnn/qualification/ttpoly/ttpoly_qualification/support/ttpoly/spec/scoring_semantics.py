# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Compact semantic handoff from one CSV compilation to accuracy scoring.

The payload deliberately excludes paths, digests, certificates, and receipts.
It contains only facts that change class/domain-action scoring.
"""

from __future__ import annotations

from typing import Any, Mapping

from .domain_action import parse_domain_actions
from .program import ProgramIOContract, TargetPolicy
from .target_class import (
    compile_raw_class_terminal_plan,
    compile_target_class_semantics,
)


BF16_EXHAUSTIVE_EVIDENCE_SCHEMA = "bf16_exhaustive_accuracy_v3"


_FIELDS = {
    "schema",
    "activation",
    "target",
    "domain_actions",
    "special_value_policy",
    "execution_kind",
    "closed_form_terminals",
    "io_contract",
    "semantic_profile",
    "real_domain",
    "raw_class_actions",
    "target_composite_class_overlay",
}


def _target_composite_class_overlay(compiled: Any) -> dict[str, Any] | None:
    """Carry a typed target-owned class override into the offline scorer."""
    from .closed_form import closed_forms_from_config, selected_artifact_closed_form_block
    from .finite_sum_class_abstract import abstract_finite_sum_class_repair
    from .target_composite import (
        finite_sum_power_tail_class_repair_from_compiled,
        native_even_polyval_exterior_from_compiled,
        native_piecewise_rational_asymptotic_from_compiled,
    )
    from .sfpu_nan_abstract import abstract_exponent_bucket_class_repair

    exponent_bucket = native_piecewise_rational_asymptotic_from_compiled(compiled)
    if (
        exponent_bucket is not None
        and exponent_bucket.class_merge.different_class_result == "target_class_representative"
    ):
        repair = abstract_exponent_bucket_class_repair(exponent_bucket)
        return {
            "kind": "exponent_bucket_target_class_repair",
            "positive_nan_guard": repair.positive_nan.guard,
            "positive_nan_output_class": "finite_other",
            "negative_nan_result": repair.negative_nan_result,
            "zero_subnormal_output_class": "finite_other",
            "negative_integer_transitions": [
                {"first_raw": raw, "output_class": kind} for raw, kind, _ in repair.negative_integer_transitions
            ],
        }

    composite = native_even_polyval_exterior_from_compiled(compiled)
    if composite is not None:
        block_id = selected_artifact_closed_form_block(compiled)
        forms = [
            form for form in closed_forms_from_config(compiled.activation_spec) if form.correction_block == block_id
        ]
        if len(forms) != 1 or not hasattr(forms[0], "finite_terminal_bound"):
            raise ValueError("native-even target composite lacks one selected bound")
        return {
            "kind": composite.kind,
            "coefficients": [float(value) for value in composite.coefficients],
            "open_interval_bound": float(forms[0].finite_terminal_bound),
            "different_class_result": (
                composite.class_merge.different_class_result if composite.class_merge is not None else None
            ),
        }
    finite_sum = finite_sum_power_tail_class_repair_from_compiled(compiled)
    if finite_sum is None:
        # A closed target graph may supply raw-special representatives without
        # owning ordinary finite Program lanes. Carry that population boundary
        # explicitly so a fresh TTNN class oracle cannot replace a finite
        # mathematical reference or a finite DomainAction.
        from .native_clamped_rational import (
            native_clamped_rational_representative,
        )

        declaration = compiled.activation_spec.get("target_composite")
        if (
            native_clamped_rational_representative(declaration) is not None
            and declaration["scope"] == "declared_invalid_nonfinite_or_raw_special_only"
        ):
            return {
                "kind": "raw_special_class_parity_scope",
                "population_scope": declaration["scope"],
            }
    if finite_sum is None or finite_sum.class_merge.different_class_result != "target_class_representative":
        return None
    repair = abstract_finite_sum_class_repair(finite_sum)
    return {
        "kind": "finite_sum_power_tail_class_repair",
        "population_scope": finite_sum.population_scope,
        "negative_integer_transitions": [
            {"first_raw": raw, "output_class": kind, "representative": output}
            for raw, kind, output in repair.negative_integer_transitions
        ],
        "positive_zero_first_raw": repair.positive_zero_first_raw,
        "positive_zero_representative": repair.zero_representative,
    }


def real_domain_declaration_from_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize both modern and legacy domain declarations for scoring."""
    valid_domain = spec.get("valid_domain")
    contract = spec.get("contract") or {}
    finite_domain = contract.get("finite_domain", "all_real")
    # Older contracts put interval bounds in ``finite_domain``.  The activation
    # level ``valid_domain`` is authoritative when present; otherwise translate
    # the legacy inclusive spelling into the scorer's interval vocabulary.
    if isinstance(finite_domain, Mapping):
        if valid_domain is None:
            valid_domain = {}
            for edge in ("min", "max"):
                if edge in finite_domain:
                    valid_domain[edge] = finite_domain[edge]
                    inclusive = bool(finite_domain.get(f"{edge}_inclusive", False))
                    valid_domain[f"{edge}_exclusive"] = not inclusive
        finite_domain = "all_real"
    return {
        "valid_domain": valid_domain,
        "finite_domain": finite_domain,
    }


def _closed_form_terminals(compiled: Any) -> list[dict[str, str]]:
    from .closed_form import closed_forms_from_config, selected_artifact_closed_form_block

    block_id = selected_artifact_closed_form_block(compiled)
    if not block_id:
        return []
    forms = [form for form in closed_forms_from_config(compiled.activation_spec) if form.correction_block == block_id]
    if len(forms) > 1:
        raise ValueError(f"multiple closed structural forms consume fit block {block_id!r}")
    if not forms:
        return []
    form = forms[0]
    declared_kind = compiled.metadata.get("closed_form_kind")
    if declared_kind and declared_kind != form.kind:
        raise ValueError("closed structural form kind metadata drift")
    return [
        {
            "coordinate": terminal.coordinate,
            "predicate": terminal.predicate,
            "result_class": terminal.result_class,
        }
        for terminal in getattr(form, "terminals", ())
    ]


def scoring_semantics_from_compiled(compiled: Any) -> dict[str, Any]:
    """Return the JSON-compatible, receipt-free scorer input for ``compiled``."""
    from .activation_config import operation_io_contract_from_config

    class_semantics = compiled.target_class_semantics
    policy = dict(class_semantics["mathematical_special_policy"])
    execution_kind = str(class_semantics["execution"]["kind"])
    # Recompose once here so malformed compiler state cannot be serialized.
    compile_target_class_semantics(compiled.target, policy, execution_kind=execution_kind)
    payload = {
        "schema": "ttpoly_scoring_semantics_v5",
        "activation": compiled.approx.activation,
        "semantic_profile": compiled.semantic_profile,
        "target": compiled.target.to_dict(),
        "domain_actions": [action.to_dict() for action in compiled.approx.domain_actions],
        "raw_class_actions": dict(compiled.activation_spec.get("raw_class_actions") or {}),
        "target_composite_class_overlay": _target_composite_class_overlay(compiled),
        "special_value_policy": policy,
        "execution_kind": execution_kind,
        "closed_form_terminals": _closed_form_terminals(compiled),
        "real_domain": real_domain_declaration_from_spec(compiled.activation_spec),
        "io_contract": operation_io_contract_from_config(compiled.activation_spec).to_dict(),
    }
    return validate_scoring_semantics(payload)


def validate_scoring_semantics(value: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly validate and canonicalize a scoring-semantics payload."""
    if not isinstance(value, Mapping) or set(value) != _FIELDS:
        raise ValueError("scoring semantics has invalid fields")
    if value.get("schema") != "ttpoly_scoring_semantics_v5":
        raise ValueError("scoring semantics has unsupported schema")
    activation = value.get("activation")
    if not isinstance(activation, str) or not activation:
        raise ValueError("scoring semantics activation must be nonempty")
    from .semantic_profile import semantic_profile

    profile = semantic_profile(value.get("semantic_profile"))
    target = TargetPolicy.from_value(value["target"])
    actions = parse_domain_actions(value["domain_actions"])
    policy = value["special_value_policy"]
    execution_kind = value["execution_kind"]
    class_semantics = compile_target_class_semantics(target, policy, execution_kind=execution_kind)
    raw_class_actions = value["raw_class_actions"]
    compile_raw_class_terminal_plan(class_semantics.raw_to_output, raw_class_actions)
    io_contract = ProgramIOContract.from_value(value["io_contract"])
    overlay = value["target_composite_class_overlay"]
    if overlay is not None:
        if not isinstance(overlay, Mapping) or "kind" not in overlay:
            raise ValueError("target composite class overlay has invalid fields")
        if overlay["kind"] == "native_even_polyval_exterior":
            if set(overlay) != {
                "kind",
                "coefficients",
                "open_interval_bound",
                "different_class_result",
            }:
                raise ValueError("native-even target composite overlay has invalid fields")
            coefficients = overlay["coefficients"]
            bound = overlay["open_interval_bound"]
            if (
                not isinstance(coefficients, list)
                or len(coefficients) != 11
                or not all(isinstance(item, (int, float)) for item in coefficients)
                or not isinstance(bound, (int, float))
                or float(bound) <= 0.0
                or overlay["different_class_result"] not in {"target_value", "target_class_representative"}
            ):
                raise ValueError("target composite class overlay payload is invalid")
        elif overlay["kind"] == "finite_sum_power_tail_class_repair":
            if set(overlay) != {
                "kind",
                "population_scope",
                "negative_integer_transitions",
                "positive_zero_first_raw",
                "positive_zero_representative",
            }:
                raise ValueError("finite-sum target composite overlay has invalid fields")
            transitions = overlay["negative_integer_transitions"]
            if (
                overlay["population_scope"] != "declared_invalid_nonfinite_or_raw_special_only"
                or not isinstance(transitions, list)
                or any(
                    not isinstance(item, Mapping)
                    or set(item) != {"first_raw", "output_class", "representative"}
                    or type(item["first_raw"]) is not int
                    or type(item["representative"]) is not int
                    or not 0 <= item["first_raw"] <= 0xFFFF
                    or not 0 <= item["representative"] <= 0xFFFF
                    for item in transitions
                )
                or [item["output_class"] for item in transitions] != ["pos_inf", "finite", "pos_zero"]
                or type(overlay["positive_zero_first_raw"]) is not int
                or not 0 < overlay["positive_zero_first_raw"] < 0x7F80
                or type(overlay["positive_zero_representative"]) is not int
                or not 0 <= overlay["positive_zero_representative"] <= 0xFFFF
            ):
                raise ValueError("finite-sum target composite overlay payload is invalid")
        elif overlay["kind"] == "exponent_bucket_target_class_repair":
            if set(overlay) != {
                "kind",
                "positive_nan_guard",
                "positive_nan_output_class",
                "negative_nan_result",
                "zero_subnormal_output_class",
                "negative_integer_transitions",
            }:
                raise ValueError("exponent-bucket class overlay has invalid fields")
            transitions = overlay["negative_integer_transitions"]
            if (
                overlay["positive_nan_guard"] != "sign_zero_and_biased_exponent_255_and_mantissa_nonzero"
                or overlay["positive_nan_output_class"] != "finite_other"
                or overlay["negative_nan_result"] != "preserve_selected"
                or overlay["zero_subnormal_output_class"] != "finite_other"
                or not isinstance(transitions, list)
                or [item.get("output_class") for item in transitions] != ["finite", "neg_inf", "pos_inf"]
                or any(
                    not isinstance(item, Mapping)
                    or set(item) != {"first_raw", "output_class"}
                    or type(item["first_raw"]) is not int
                    or not 0 <= item["first_raw"] <= 0xFFFF
                    for item in transitions
                )
            ):
                raise ValueError("exponent-bucket class overlay payload is invalid")
        elif overlay["kind"] == "raw_special_class_parity_scope":
            if set(overlay) != {"kind", "population_scope"} or (
                overlay["population_scope"] != "declared_invalid_nonfinite_or_raw_special_only"
            ):
                raise ValueError("raw-special class-parity scope is invalid")
        else:
            raise ValueError("target composite class overlay kind is unsupported")
    terminals = value["closed_form_terminals"]
    if not isinstance(terminals, list):
        raise ValueError("closed_form_terminals must be an array")
    canonical_terminals = []
    for index, terminal in enumerate(terminals):
        expected = ("coordinate", "predicate", "result_class")
        if not isinstance(terminal, Mapping) or set(terminal) != set(expected):
            raise ValueError(f"closed_form_terminals[{index}] has invalid fields")
        canonical_terminals.append({name: str(terminal[name]) for name in expected})
    real_domain = value["real_domain"]
    if not isinstance(real_domain, Mapping) or set(real_domain) != {"valid_domain", "finite_domain"}:
        raise ValueError("real_domain must declare valid_domain and finite_domain")
    if real_domain["valid_domain"] is not None and not isinstance(real_domain["valid_domain"], Mapping):
        raise ValueError("real_domain.valid_domain must be an object or null")
    if real_domain["finite_domain"] not in {"all_real", "all_real_except_nonpositive_integer_poles", "nonnegative"}:
        raise ValueError("real_domain.finite_domain is unsupported")
    if overlay is None:
        canonical_overlay = None
    elif overlay["kind"] == "native_even_polyval_exterior":
        canonical_overlay = {
            "kind": str(overlay["kind"]),
            "coefficients": [float(item) for item in overlay["coefficients"]],
            "open_interval_bound": float(overlay["open_interval_bound"]),
            "different_class_result": str(overlay["different_class_result"]),
        }
    elif overlay["kind"] == "exponent_bucket_target_class_repair":
        canonical_overlay = {
            "kind": str(overlay["kind"]),
            "positive_nan_guard": str(overlay["positive_nan_guard"]),
            "positive_nan_output_class": str(overlay["positive_nan_output_class"]),
            "negative_nan_result": str(overlay["negative_nan_result"]),
            "zero_subnormal_output_class": str(overlay["zero_subnormal_output_class"]),
            "negative_integer_transitions": [dict(item) for item in overlay["negative_integer_transitions"]],
        }
    elif overlay["kind"] == "raw_special_class_parity_scope":
        canonical_overlay = {
            "kind": str(overlay["kind"]),
            "population_scope": str(overlay["population_scope"]),
        }
    else:
        canonical_overlay = {
            "kind": str(overlay["kind"]),
            "population_scope": str(overlay["population_scope"]),
            "negative_integer_transitions": [dict(item) for item in overlay["negative_integer_transitions"]],
            "positive_zero_first_raw": int(overlay["positive_zero_first_raw"]),
            "positive_zero_representative": int(overlay["positive_zero_representative"]),
        }
    return {
        "schema": "ttpoly_scoring_semantics_v5",
        "activation": activation,
        "semantic_profile": profile,
        "target": target.to_dict(),
        "domain_actions": [action.to_dict() for action in actions],
        "raw_class_actions": {str(raw_class): dict(action) for raw_class, action in raw_class_actions.items()},
        "target_composite_class_overlay": canonical_overlay,
        "special_value_policy": dict(policy),
        "execution_kind": str(execution_kind),
        "closed_form_terminals": canonical_terminals,
        "real_domain": {
            "valid_domain": (None if real_domain["valid_domain"] is None else dict(real_domain["valid_domain"])),
            "finite_domain": str(real_domain["finite_domain"]),
        },
        "io_contract": io_contract.to_dict(),
    }


__all__ = [
    "real_domain_declaration_from_spec",
    "scoring_semantics_from_compiled",
    "validate_scoring_semantics",
]
