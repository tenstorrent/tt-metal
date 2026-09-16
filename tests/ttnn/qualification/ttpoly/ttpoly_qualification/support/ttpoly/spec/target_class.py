# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Target-bound composition of raw IEEE classes and activation policy.

The activation JSON describes mathematical behaviour at the input seen by the
activation.  It does not describe how a tensor encoding reaches that input.
This module keeps those two authorities separate and composes them once at the
artifact compiler boundary:

    raw storage class -> target ingress class -> mathematical result class

The resulting record is consumed by both evidence reduction and kernel
codegen.  It is deliberately architecture/precision based; activation names
and evaluator forms never participate.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from .activation_config import SPECIAL_INPUT_CLASSES, SPECIAL_OUTPUT_CLASSES
from .domain_action import parse_domain_actions
from .program import TargetPolicy


RAW_INPUT_CLASSES = (
    "pos_nan",
    "neg_nan",
    "pos_inf",
    "neg_inf",
    "pos_zero",
    "neg_zero",
    "pos_subnormal",
    "neg_subnormal",
    "finite_other",
)

_INGRESS_CLASSES = frozenset((*SPECIAL_INPUT_CLASSES, "finite_other"))


def bh_bf16_domain_actions_need_coordinate(actions: Any) -> bool:
    """Prove whether DAZ can change an ordered raw-domain action outcome.

    All 256 BF16 exponent-zero encodings are enumerated at compile time. If
    they select one constant/class owner (or no owner), replacing their action
    coordinate with +0 cannot affect the terminal action. Actions that consume
    the raw value/sign remain conservatively sensitive even with one owner.
    """
    typed = parse_domain_actions(actions)
    if any(action.coordinate != "raw" for action in typed):
        return True

    def fp32(value: float) -> float:
        try:
            return struct.unpack("<f", struct.pack("<f", value))[0]
        except OverflowError:
            return math.copysign(math.inf, value)

    def owner(value: float) -> int | None:
        for index, action in enumerate(typed):
            bound = fp32(action.bound)
            if action.direction == "below":
                matches = value <= bound if action.inclusive else value < bound
            else:
                matches = value >= bound if action.inclusive else value > bound
            if matches:
                return index
        return None

    min_subnormal = math.ldexp(1.0, -133)
    exponent_zero_values = tuple(sign * fraction * min_subnormal for sign in (1.0, -1.0) for fraction in range(128))
    owners = {owner(value) for value in exponent_zero_values}
    if len(owners) != 1:
        return True
    selected = next(iter(owners))
    if selected is None:
        return False
    return typed[selected].action.kind not in {"constant", "return_class"}


@dataclass(frozen=True)
class TargetClassSemantics:
    """Closed class-level semantics for one precision/architecture target."""

    target: TargetPolicy
    execution_kind: str
    ingress_kind: str
    raw_to_ingress: Mapping[str, str]
    mathematical_policy: Mapping[str, str]
    result_to_egress: Mapping[str, str]
    raw_to_output: Mapping[str, str]
    post_action_raw_override: Mapping[str, str]

    def __post_init__(self) -> None:
        target = TargetPolicy.from_value(self.target)
        raw_to_ingress = dict(self.raw_to_ingress)
        mathematical_policy = dict(self.mathematical_policy)
        result_to_egress = dict(self.result_to_egress)
        raw_to_output = dict(self.raw_to_output)
        post_action_raw_override = dict(self.post_action_raw_override)
        if self.execution_kind not in {
            "sfpu",
            "sfpu_ingress_quotient",
            "tti_signed_nonfinite_quotient",
            "llk_decoded_signed_nonfinite",
            "sfpu_raw_nan",
            "sfpu_post_round_raw_class_repair",
            "tti_signed_negative_nan",
            "tti_signed_positive_nan",
            "tti_signed_raw_nan_finalizer",
            "tile_copy",
            "pack_relu",
        }:
            raise ValueError("target class execution kind is unsupported")
        if set(raw_to_ingress) != set(RAW_INPUT_CLASSES):
            raise ValueError("raw_to_ingress must cover the closed raw class set")
        if not set(raw_to_ingress.values()) <= _INGRESS_CLASSES:
            raise ValueError("raw_to_ingress names an unknown ingress class")
        if set(mathematical_policy) != set(SPECIAL_INPUT_CLASSES):
            raise ValueError("mathematical_policy must cover every special class")
        if not set(mathematical_policy.values()) <= SPECIAL_OUTPUT_CLASSES:
            raise ValueError("mathematical_policy names an unknown result class")
        if set(result_to_egress) != set(SPECIAL_OUTPUT_CLASSES):
            raise ValueError("result_to_egress must cover every result class")
        if not set(result_to_egress.values()) <= SPECIAL_OUTPUT_CLASSES:
            raise ValueError("result_to_egress names an unknown output class")
        if set(raw_to_output) != set(RAW_INPUT_CLASSES):
            raise ValueError("raw_to_output must cover the closed raw class set")
        if not set(raw_to_output.values()) <= SPECIAL_OUTPUT_CLASSES:
            raise ValueError("raw_to_output names an unknown result class")
        if not set(post_action_raw_override) <= set(RAW_INPUT_CLASSES):
            raise ValueError("post-action override names an unknown raw class")
        if not set(post_action_raw_override.values()) <= set(SPECIAL_INPUT_CLASSES):
            raise ValueError("post-action override names an unknown policy class")
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "raw_to_ingress", MappingProxyType(raw_to_ingress))
        object.__setattr__(self, "mathematical_policy", MappingProxyType(mathematical_policy))
        object.__setattr__(self, "result_to_egress", MappingProxyType(result_to_egress))
        object.__setattr__(self, "raw_to_output", MappingProxyType(raw_to_output))
        object.__setattr__(
            self,
            "post_action_raw_override",
            MappingProxyType(post_action_raw_override),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema": ("target_class_semantics_v3" if self.post_action_raw_override else "target_class_semantics_v2"),
            "target_sha256": self.target.sha256,
            "execution": {"kind": self.execution_kind},
            "ingress": {
                "kind": self.ingress_kind,
                "raw_to_effective_class": {name: self.raw_to_ingress[name] for name in RAW_INPUT_CLASSES},
            },
            "mathematical_special_policy": {name: self.mathematical_policy[name] for name in SPECIAL_INPUT_CLASSES},
            "raw_to_output_class": {name: self.raw_to_output[name] for name in RAW_INPUT_CLASSES},
            "egress": {
                "kind": "target_round_then_ftz",
                "precision": self.target.precision,
                "output_ftz": self.target.output_ftz,
                "rounding_order": self.target.rounding_order,
                "result_to_output_class": {
                    name: self.result_to_egress[name] for name in sorted(SPECIAL_OUTPUT_CLASSES)
                },
            },
        }
        if self.post_action_raw_override:
            result["post_action_raw_class_override"] = {
                name: self.post_action_raw_override[name]
                for name in RAW_INPUT_CLASSES
                if name in self.post_action_raw_override
            }
        return result


@dataclass(frozen=True)
class RawClassTerminalPlan:
    """Late semantic-profile actions keyed by raw storage class."""

    output_classes: Mapping[str, str]
    constants: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_classes", MappingProxyType(dict(self.output_classes)))
        object.__setattr__(self, "constants", MappingProxyType(dict(self.constants)))


def compile_target_class_semantics(
    target: TargetPolicy | Mapping[str, Any],
    mathematical_policy: Mapping[str, str],
    *,
    execution_kind: str = "sfpu",
) -> TargetClassSemantics:
    """Compose target transport with a validated mathematical class policy."""
    typed_target = TargetPolicy.from_value(target)
    policy = dict(mathematical_policy)
    if set(policy) != set(SPECIAL_INPUT_CLASSES):
        raise ValueError("mathematical special policy is incomplete")
    if not set(policy.values()) <= SPECIAL_OUTPUT_CLASSES:
        raise ValueError("mathematical special policy names an unknown class")

    if execution_kind not in {
        "sfpu",
        "sfpu_ingress_quotient",
        "tti_signed_nonfinite_quotient",
        "llk_decoded_signed_nonfinite",
        "sfpu_raw_nan",
        "sfpu_post_round_raw_class_repair",
        "tti_signed_negative_nan",
        "tti_signed_positive_nan",
        "tti_signed_raw_nan_finalizer",
        "tile_copy",
        "pack_relu",
    }:
        raise ValueError("target class execution kind is unsupported")
    if execution_kind in {"sfpu_raw_nan", "sfpu_post_round_raw_class_repair"} and not (
        typed_target.precision == "bf16" and typed_target.architecture == "bh"
    ):
        raise ValueError("raw-class SFPU repair requires a Blackhole BF16 target")
    if execution_kind in {
        "sfpu_ingress_quotient",
        "tti_signed_nonfinite_quotient",
        "tti_signed_negative_nan",
        "tti_signed_positive_nan",
        "tti_signed_raw_nan_finalizer",
    } and not (typed_target.precision == "bf16" and typed_target.architecture == "bh"):
        raise ValueError("SFPU ingress quotient requires a Blackhole BF16 target")
    if execution_kind == "llk_decoded_signed_nonfinite" and not (
        typed_target.precision == "bf16" and typed_target.architecture == "wh"
    ):
        raise ValueError("decoded LLK signed-nonfinite transport requires Wormhole BF16")
    if execution_kind == "pack_relu" and not (typed_target.precision == "bf16" and typed_target.architecture == "bh"):
        raise ValueError("packer ReLU class transport requires a Blackhole BF16 target")

    if execution_kind == "llk_decoded_signed_nonfinite":
        ingress_kind = "wh_float16_b_llk_decoded_signed_dst"
        ingress = {
            "pos_nan": "pos_inf",
            "neg_nan": "neg_inf",
            "pos_inf": "pos_inf",
            "neg_inf": "neg_inf",
            "pos_zero": "pos_zero",
            "neg_zero": "pos_zero",
            "pos_subnormal": "pos_zero",
            "neg_subnormal": "pos_zero",
            "finite_other": "finite_other",
        }
    elif typed_target.precision == "bf16" and typed_target.architecture == "bh":
        # Blackhole tile-copy and SFPU evaluation have different class
        # quotients. Bind that structural execution choice explicitly; neither
        # activation identity nor coefficient values participate.
        if execution_kind in {
            "tile_copy",
            "tti_signed_nonfinite_quotient",
            "tti_signed_negative_nan",
            "tti_signed_positive_nan",
            "tti_signed_raw_nan_finalizer",
            "pack_relu",
        }:
            ingress_kind = (
                "bh_float16_b_tile_copy"
                if execution_kind == "tile_copy"
                else ("bh_float16_b_pack_relu" if execution_kind == "pack_relu" else "bh_float16_b_tti_signed_dst")
            )
            ingress = {
                "pos_nan": "pos_inf",
                "neg_nan": "neg_inf",
                "pos_inf": "pos_inf",
                "neg_inf": "neg_inf",
                "pos_zero": "pos_zero",
                "neg_zero": "pos_zero",
                "pos_subnormal": "pos_zero",
                "neg_subnormal": "pos_zero",
                "finite_other": "finite_other",
            }
        else:
            # The common SFPU dst-load path canonicalizes either NaN sign to
            # +Inf. Exponent-zero coordinates are normalized by the common
            # target-action hook before actions/finalization.
            ingress_kind = "bh_float16_b_sfpu_dst"
            ingress = {
                "pos_nan": "pos_inf",
                "neg_nan": "pos_inf",
                "pos_inf": "pos_inf",
                "neg_inf": "neg_inf",
                "pos_zero": "pos_zero",
                "neg_zero": "pos_zero",
                "pos_subnormal": "pos_zero",
                "neg_subnormal": "pos_zero",
                "finite_other": "finite_other",
            }
    else:
        ingress_kind = "target_numeric_policy"
        ingress = {
            "pos_nan": "nan",
            "neg_nan": "nan",
            "pos_inf": "pos_inf",
            "neg_inf": "neg_inf",
            "pos_zero": "pos_zero",
            "neg_zero": "neg_zero",
            "pos_subnormal": ("pos_zero" if typed_target.input_daz else "finite_other"),
            "neg_subnormal": ("neg_zero" if typed_target.input_daz else "finite_other"),
            "finite_other": "finite_other",
        }

    result_to_egress = {name: name for name in SPECIAL_OUTPUT_CLASSES}
    if execution_kind == "pack_relu":
        # STACC_RELU applies after the early format conversion. Its sign test
        # clears negative zero while its signed non-finite input quotient lets
        # the activation policy distinguish the two raw NaN signs. This is a
        # structural packer fact, not an activation-name exception.
        result_to_egress["neg_zero"] = "pos_zero"
    if (
        typed_target.precision == "bf16"
        and typed_target.architecture == "bh"
        and execution_kind
        in {
            "sfpu",
            "sfpu_ingress_quotient",
            "tti_signed_nonfinite_quotient",
            "sfpu_raw_nan",
            "tti_signed_negative_nan",
            "tti_signed_positive_nan",
            "tti_signed_raw_nan_finalizer",
        }
    ):
        # FP32 qNaN and explicit/computed negative zero returned by a common
        # SFPU action/finalizer narrow through the existing Float16_b
        # conversion to canonical +Inf and +0 respectively.  These are egress
        # facts, distinct from the activation's mathematical class policy.
        result_to_egress["nan"] = "pos_inf"
        result_to_egress["neg_zero"] = "pos_zero"
    if execution_kind == "llk_decoded_signed_nonfinite":
        result_to_egress["nan"] = "pos_inf"
        result_to_egress["neg_zero"] = "pos_zero"

    if execution_kind == "sfpu_raw_nan":
        post_action_raw_override = {"pos_nan": "nan", "neg_nan": "nan"}
    elif execution_kind == "tti_signed_positive_nan":
        post_action_raw_override = {"pos_nan": "nan"}
    elif execution_kind == "tti_signed_negative_nan":
        post_action_raw_override = {"neg_nan": "nan"}
    elif execution_kind == "sfpu_post_round_raw_class_repair":
        post_action_raw_override = {
            "pos_nan": "nan",
            "neg_nan": "nan",
            "pos_zero": "pos_zero",
            "neg_zero": "neg_zero",
            "pos_subnormal": "pos_zero",
            "neg_subnormal": "neg_zero",
        }
    else:
        post_action_raw_override = {}
    raw_to_output = {
        raw_class: (
            result_to_egress[policy[post_action_raw_override[raw_class]]]
            if raw_class in post_action_raw_override
            else ("finite_other" if ingress_class == "finite_other" else result_to_egress[policy[ingress_class]])
        )
        for raw_class, ingress_class in ingress.items()
    }
    return TargetClassSemantics(
        target=typed_target,
        execution_kind=execution_kind,
        ingress_kind=ingress_kind,
        raw_to_ingress=ingress,
        mathematical_policy=policy,
        result_to_egress=result_to_egress,
        raw_to_output=raw_to_output,
        post_action_raw_override=post_action_raw_override,
    )


def compile_raw_class_terminal_plan(raw_to_output: Mapping[str, str], actions: Any) -> RawClassTerminalPlan:
    """Compile return-class and finite-constant raw actions without routing."""
    result = dict(raw_to_output)
    if set(result) != set(RAW_INPUT_CLASSES):
        raise ValueError("raw class output map must cover the closed raw class set")
    constants: dict[str, float] = {}
    if actions is None:
        return RawClassTerminalPlan(result, constants)
    if not isinstance(actions, Mapping):
        raise ValueError("raw_class_actions must be an object")
    for raw_class, action in actions.items():
        if raw_class not in RAW_INPUT_CLASSES:
            raise ValueError(f"raw_class_actions names unknown class {raw_class!r}")
        if not isinstance(action, Mapping):
            raise ValueError("raw_class_actions values must be action objects")
        if set(action) == {"kind"} and action["kind"] == "evaluate":
            result[str(raw_class)] = "finite_other"
        elif set(action) == {"kind", "class"} and action["kind"] == "return_class":
            if action["class"] not in SPECIAL_OUTPUT_CLASSES:
                raise ValueError("raw_class_actions names an unsupported output class")
            result[str(raw_class)] = str(action["class"])
        elif set(action) == {"kind", "value"} and action["kind"] == "constant":
            value = action["value"]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("raw_class_actions constant value must be numeric")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("raw_class_actions constant value must be finite")
            result[str(raw_class)] = "constant"
            constants[str(raw_class)] = value
        else:
            raise ValueError("raw_class_actions values must be return_class, constant, or evaluate actions")
    return RawClassTerminalPlan(result, constants)


def target_class_semantics_from_manifest(
    value: Mapping[str, Any],
    *,
    expected_target: TargetPolicy | Mapping[str, Any],
) -> TargetClassSemantics:
    """Parse a compiled record by reproducing and byte-comparing composition."""
    if not isinstance(value, Mapping):
        raise ValueError("target class semantics must be an object")
    schema = value.get("schema")
    required = {
        "schema",
        "target_sha256",
        "execution",
        "ingress",
        "mathematical_special_policy",
        "raw_to_output_class",
        "egress",
    }
    if schema == "target_class_semantics_v3":
        required.add("post_action_raw_class_override")
    elif schema != "target_class_semantics_v2":
        raise ValueError("target class semantics has invalid fields or schema")
    if set(value) != required:
        raise ValueError("target class semantics has invalid fields or schema")
    target = TargetPolicy.from_value(expected_target)
    execution = value.get("execution")
    if not isinstance(execution, Mapping) or set(execution) != {"kind"}:
        raise ValueError("target class semantics has invalid execution binding")
    compiled = compile_target_class_semantics(
        target,
        value["mathematical_special_policy"],
        execution_kind=str(execution["kind"]),
    )
    if dict(value) != compiled.to_dict():
        raise ValueError("target class semantics disagrees with target composition")
    return compiled


__all__ = [
    "RAW_INPUT_CLASSES",
    "RawClassTerminalPlan",
    "TargetClassSemantics",
    "bh_bf16_domain_actions_need_coordinate",
    "compile_raw_class_terminal_plan",
    "compile_target_class_semantics",
    "target_class_semantics_from_manifest",
]
