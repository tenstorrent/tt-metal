# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Typed, target-instantiated region Program IR.

This module owns structure and validation only.  It deliberately contains no
numeric evaluator: ``ttpoly.precision.eval`` remains the sole arithmetic owner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import struct
from types import MappingProxyType
from typing import Any, Mapping


class ProgramOp(str, Enum):
    CONSTANT = "constant"
    COPY = "copy"
    NEGATE = "negate"
    ABS = "abs"
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    FMA = "fma"
    RECIPROCAL = "reciprocal"
    MIN = "min"
    MAX = "max"
    COPYSIGN = "copysign"
    SELECT = "select"
    ROUND_NEAREST_INTEGER = "round_nearest_integer"
    PERIODIC_REDUCE_Q192 = "periodic_reduce_q192"
    PERIODIC_REDUCE_WINDOW = "periodic_reduce_window"
    NORMALIZE_EXPONENT = "normalize_exponent"
    SCALE_POW2 = "scale_pow2"
    RETURN_CLASS = "return_class"
    EVAL_BLOCK = "eval_block"

    @classmethod
    def coerce(cls, value: "ProgramOp | str") -> "ProgramOp":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError as exc:
            raise ValueError(f"unknown ProgramOp: {value!r}") from exc


class TargetExecutionKind(str, Enum):
    """Closed physical execution quotient carried into target lowering."""

    SFPU = "sfpu"
    SFPU_INGRESS_QUOTIENT = "sfpu_ingress_quotient"
    SFPU_RAW_NAN = "sfpu_raw_nan"
    TTI_SIGNED_NONFINITE_QUOTIENT = "tti_signed_nonfinite_quotient"
    TTI_SIGNED_NEGATIVE_NAN = "tti_signed_negative_nan"
    TTI_SIGNED_RAW_NAN_FINALIZER = "tti_signed_raw_nan_finalizer"
    TTI_SIGNED_POSITIVE_NAN = "tti_signed_positive_nan"

    @classmethod
    def from_value(cls, value: "TargetExecutionKind | str") -> "TargetExecutionKind":
        try:
            return cls(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unsupported target_execution_kind: {value!r}") from exc


class SpecialPredicate(str, Enum):
    IS_NAN = "is_nan"
    IS_POS_INF = "is_pos_inf"
    IS_NEG_INF = "is_neg_inf"
    IS_POS_ZERO = "is_pos_zero"
    IS_NEG_ZERO = "is_neg_zero"
    INVALID_DOMAIN = "invalid_domain"


class SpecialResult(str, Enum):
    NAN = "nan"
    POS_INF = "pos_inf"
    NEG_INF = "neg_inf"
    POS_ZERO = "pos_zero"
    NEG_ZERO = "neg_zero"
    CONSTANT = "constant"


_PRECISIONS = frozenset({"bf16", "fp16", "fp32"})
_ARCHITECTURES = frozenset({"bh", "wh", "ieee", "rvv"})
_ROUNDING_ORDERS = frozenset({"pre_round", "post_round", "none"})


@dataclass(frozen=True)
class TargetPolicy:
    """Complete numerical target identity for one instantiated Program."""

    precision: str
    architecture: str
    intermediate_precision: str
    input_daz: bool
    output_ftz: bool
    rounding_order: str

    def __post_init__(self) -> None:
        precision = str(self.precision).lower()
        architecture = str(self.architecture).lower()
        intermediate = str(self.intermediate_precision).lower()
        rounding_order = str(self.rounding_order).lower()
        if precision not in _PRECISIONS:
            raise ValueError(f"unsupported target precision: {self.precision!r}")
        if architecture not in _ARCHITECTURES:
            raise ValueError(f"unsupported target architecture: {self.architecture!r}")
        if intermediate not in _PRECISIONS:
            raise ValueError(f"unsupported intermediate precision: {self.intermediate_precision!r}")
        if type(self.input_daz) is not bool or type(self.output_ftz) is not bool:
            raise ValueError("target input_daz and output_ftz must be booleans")
        if rounding_order not in _ROUNDING_ORDERS:
            raise ValueError(f"unsupported target rounding_order: {self.rounding_order!r}")
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "architecture", architecture)
        object.__setattr__(self, "intermediate_precision", intermediate)
        object.__setattr__(self, "rounding_order", rounding_order)

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision": self.precision,
            "architecture": self.architecture,
            "intermediate_precision": self.intermediate_precision,
            "input_daz": self.input_daz,
            "output_ftz": self.output_ftz,
            "rounding_order": self.rounding_order,
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_value(cls, value: "TargetPolicy | Mapping[str, Any]") -> "TargetPolicy":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("target must be a TargetPolicy or object")
        expected = {
            "precision",
            "architecture",
            "intermediate_precision",
            "input_daz",
            "output_ftz",
            "rounding_order",
        }
        missing = expected - set(value)
        unknown = set(value) - expected
        if missing:
            raise ValueError(f"target missing fields: {sorted(missing)}")
        if unknown:
            raise ValueError(f"target has unknown fields: {sorted(unknown)}")
        return cls(**{key: value[key] for key in expected})


DEFAULT_BF16_TARGET = TargetPolicy("bf16", "bh", "fp32", True, True, "post_round")
DEFAULT_FP32_TARGET = TargetPolicy("fp32", "bh", "fp32", True, True, "post_round")


@dataclass(frozen=True)
class ProgramIOContract:
    """Artifact-bound tensor ABI and optional fused-gradient composition."""

    arity: int = 1
    fuse_grad: bool = False
    tensor_input_roles: tuple[str, ...] = ("activation_input",)
    gradient_composition: str | None = None
    gradient_scale: float | None = None

    def __post_init__(self) -> None:
        if type(self.arity) is not int or self.arity < 1:
            raise ValueError("Program I/O arity must be a positive integer")
        if type(self.fuse_grad) is not bool:
            raise ValueError("Program I/O fuse_grad must be boolean")
        composition = self.gradient_composition
        if composition is None:
            composition = "multiply" if self.fuse_grad else "none"
        if composition not in {
            "none",
            "multiply",
            "zero_mask_select",
            "zone_select_multiply",
        }:
            raise ValueError("Program I/O gradient_composition is unknown")
        if (composition != "none") != self.fuse_grad:
            raise ValueError("Program I/O fuse_grad must agree with gradient_composition")
        scale = self.gradient_scale
        if scale is not None:
            if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(float(scale)):
                raise ValueError("Program I/O gradient_scale must be a finite number")
            if composition == "zone_select_multiply":
                raise ValueError("Program I/O zone_select_multiply does not accept gradient_scale")
            if composition not in {"multiply", "zero_mask_select"}:
                raise ValueError("Program I/O gradient_scale requires a fused gradient composition")
            # The fused arithmetic is performed in fp32 on the target.  Store
            # that exact scalar in the typed artifact so decimal spelling and
            # host float width cannot perturb Program identity or replay.
            try:
                scale = struct.unpack("<f", struct.pack("<f", float(scale)))[0]
            except OverflowError as exc:
                raise ValueError("Program I/O gradient_scale must be representable as finite fp32") from exc
            if not math.isfinite(scale):
                raise ValueError("Program I/O gradient_scale must be representable as finite fp32")
        roles = tuple(self.tensor_input_roles)
        if len(roles) != self.arity or any(role not in {"activation_input", "incoming_gradient"} for role in roles):
            raise ValueError("Program I/O tensor roles must match arity and be known")
        if roles.count("activation_input") != 1:
            raise ValueError("Program I/O requires exactly one activation_input")
        if self.fuse_grad:
            if self.arity != 2 or roles != (
                "incoming_gradient",
                "activation_input",
            ):
                raise ValueError("fused Program I/O requires incoming_gradient then activation_input")
        elif "incoming_gradient" in roles:
            raise ValueError("incoming_gradient requires fused Program I/O")
        object.__setattr__(self, "tensor_input_roles", roles)
        object.__setattr__(self, "gradient_composition", composition)
        object.__setattr__(self, "gradient_scale", scale)

    @property
    def gradient_input_index(self) -> int | None:
        return 0 if self.fuse_grad else None

    @property
    def activation_input_index(self) -> int:
        return self.tensor_input_roles.index("activation_input")

    @property
    def activation_dst_tile_index(self) -> int:
        """Physical Program-row placement after transport reorders the ABI."""
        return 0

    @property
    def gradient_dst_tile_index(self) -> int | None:
        """Physical fused-grad placement; external input order is independent."""
        return 1 if self.fuse_grad else None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "arity": self.arity,
            "fuse_grad": self.fuse_grad,
            "tensor_input_roles": list(self.tensor_input_roles),
            "gradient_composition": self.gradient_composition,
        }
        # Preserve every existing unary/unscaled Program identity byte-for-byte.
        if self.gradient_scale is not None:
            payload["gradient_scale"] = self.gradient_scale
        return payload

    @classmethod
    def from_value(cls, value: "ProgramIOContract | Mapping[str, Any]") -> "ProgramIOContract":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("Program I/O contract must be an object")
        required_fields = {
            "arity",
            "fuse_grad",
            "tensor_input_roles",
            "gradient_composition",
        }
        unknown_fields = set(value) - required_fields
        if not required_fields <= set(value) or unknown_fields - {"gradient_scale"}:
            raise ValueError("Program I/O contract has invalid fields")
        roles = value["tensor_input_roles"]
        if not isinstance(roles, (list, tuple)):
            raise ValueError("Program I/O tensor_input_roles must be an array")
        return cls(
            value["arity"],
            value["fuse_grad"],
            tuple(roles),
            value["gradient_composition"],
            value.get("gradient_scale"),
        )


def _bound(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if math.isnan(result):
        raise ValueError(f"{name} cannot be NaN")
    return result


@dataclass(frozen=True)
class ZoneBounds:
    lower: float
    upper: float
    lower_inclusive: bool
    upper_inclusive: bool

    def __post_init__(self) -> None:
        lower = _bound(self.lower, "zone lower bound")
        upper = _bound(self.upper, "zone upper bound")
        if type(self.lower_inclusive) is not bool or type(self.upper_inclusive) is not bool:
            raise ValueError("zone bound inclusivity must be boolean")
        if lower > upper or (lower == upper and not (self.lower_inclusive and self.upper_inclusive)):
            raise ValueError("zone bounds describe an empty interval")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


_OP_ARITY = {
    ProgramOp.CONSTANT: 0,
    ProgramOp.COPY: 1,
    ProgramOp.NEGATE: 1,
    ProgramOp.ABS: 1,
    ProgramOp.ADD: 2,
    ProgramOp.SUBTRACT: 2,
    ProgramOp.MULTIPLY: 2,
    ProgramOp.FMA: 3,
    ProgramOp.RECIPROCAL: 1,
    ProgramOp.MIN: 2,
    ProgramOp.MAX: 2,
    ProgramOp.COPYSIGN: 2,
    ProgramOp.SELECT: 3,
    ProgramOp.ROUND_NEAREST_INTEGER: 1,
    ProgramOp.PERIODIC_REDUCE_Q192: 1,
    ProgramOp.PERIODIC_REDUCE_WINDOW: 1,
    ProgramOp.NORMALIZE_EXPONENT: 1,
    ProgramOp.SCALE_POW2: 2,
    ProgramOp.RETURN_CLASS: 0,
    ProgramOp.EVAL_BLOCK: 1,
}


def _freeze_params(value: Mapping[str, Any], *, allow_strings: bool = False) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("ProgramStep params must be an object")
    frozen: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("ProgramStep parameter names must be nonempty strings")
        if allow_strings and isinstance(item, str):
            frozen[key] = item
        elif isinstance(item, bool) or (isinstance(item, (int, float)) and not isinstance(item, bool)):
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("ProgramStep numeric parameters must be finite")
            frozen[key] = item
        else:
            raise ValueError("ProgramStep params may contain only numeric or boolean typed values")
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class ProgramStep:
    id: str
    op: ProgramOp | str
    inputs: tuple[str, ...] = field(default_factory=tuple)
    params: Mapping[str, Any] = field(default_factory=dict)
    block: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id or self.id == "input":
            raise ValueError("ProgramStep id must be a nonempty non-reserved string")
        op = ProgramOp.coerce(self.op)
        inputs = tuple(self.inputs)
        if any(not isinstance(item, str) or not item for item in inputs):
            raise ValueError("ProgramStep inputs must be nonempty value ids")
        if len(inputs) != _OP_ARITY[op]:
            raise ValueError(f"ProgramOp {op.value!r} requires {_OP_ARITY[op]} inputs, got {len(inputs)}")
        params = _freeze_params(
            self.params,
            allow_strings=op
            in {
                ProgramOp.NORMALIZE_EXPONENT,
                ProgramOp.PERIODIC_REDUCE_WINDOW,
                ProgramOp.RETURN_CLASS,
            },
        )
        if op is ProgramOp.CONSTANT:
            if (
                set(params) != {"value"}
                or isinstance(params["value"], bool)
                or not isinstance(params["value"], (int, float))
            ):
                raise ValueError("constant requires one numeric 'value' parameter")
        elif op is ProgramOp.NORMALIZE_EXPONENT:
            lower = params.get("lower")
            if (
                set(params) != {"lower"}
                or isinstance(lower, bool)
                or not isinstance(lower, (int, float))
                or not 0.5 <= float(lower) < 1.0
            ):
                raise ValueError("normalize_exponent requires numeric lower in [0.5, 1.0)")
        elif op is ProgramOp.PERIODIC_REDUCE_Q192:
            phase = params.get("phase_quadrants")
            if set(params) != {"phase_quadrants"} or type(phase) is not int or phase not in {0, 1, 2, 3}:
                raise ValueError("periodic_reduce_q192 requires integer phase_quadrants in {0,1,2,3}")
        elif op is ProgramOp.PERIODIC_REDUCE_WINDOW:
            window_bits = params.get("window_bits")
            period = params.get("period")
            phase = params.get("phase_quadrants")
            if (
                set(params) != {"window_bits", "period", "phase_quadrants"}
                or type(window_bits) is not int
                or not 16 <= window_bits <= 63
                or period not in {"pi", "pi_over_2"}
                or type(phase) is not int
                or phase not in {0, 1, 2, 3}
            ):
                raise ValueError(
                    "periodic_reduce_window requires integer window_bits in "
                    "[16, 63], period pi|pi_over_2, and integer "
                    "phase_quadrants in {0,1,2,3}"
                )
        elif op is ProgramOp.RECIPROCAL:
            iterations = params.get("iterations", 2)
            if set(params) not in (set(), {"iterations"}) or type(iterations) is not int or iterations not in {1, 2}:
                raise ValueError("reciprocal accepts optional integer iterations in {1,2}")
        elif op is ProgramOp.SCALE_POW2:
            intermediate_ftz = params.get("intermediate_ftz")
            if set(params) not in (set(), {"intermediate_ftz"}) or (
                intermediate_ftz is not None and intermediate_ftz is not True
            ):
                raise ValueError("scale_pow2 accepts optional intermediate_ftz=true")
        elif op is ProgramOp.RETURN_CLASS:
            allowed = {
                SpecialResult.NAN.value,
                SpecialResult.POS_INF.value,
                SpecialResult.NEG_INF.value,
                SpecialResult.POS_ZERO.value,
                SpecialResult.NEG_ZERO.value,
            }
            if set(params) != {"class"} or params["class"] not in allowed:
                raise ValueError("return_class requires class nan|pos_inf|neg_inf|pos_zero|neg_zero")
        elif params:
            raise ValueError(f"ProgramOp {op.value!r} does not accept params")
        if op is ProgramOp.EVAL_BLOCK:
            if not isinstance(self.block, str) or not self.block:
                raise ValueError("eval_block requires a nonempty block reference")
        elif self.block is not None:
            raise ValueError(f"ProgramOp {op.value!r} cannot carry a block reference")
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "params", params)


@dataclass(frozen=True)
class Zone:
    id: str
    bounds: ZoneBounds
    steps: tuple[ProgramStep, ...]
    result: str

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("Zone id must be a nonempty string")
        if not isinstance(self.bounds, ZoneBounds):
            raise ValueError("Zone bounds must be typed ZoneBounds")
        steps = tuple(self.steps)
        seen = {"input"}
        for step in steps:
            if not isinstance(step, ProgramStep):
                raise ValueError("Zone steps must be typed ProgramStep records")
            if step.id in seen:
                raise ValueError(f"duplicate Program value id: {step.id!r}")
            missing = [item for item in step.inputs if item not in seen]
            if missing:
                raise ValueError(f"ProgramStep {step.id!r} has forward/unknown inputs: {missing}")
            seen.add(step.id)
        if self.result not in seen:
            raise ValueError(f"Zone result references unknown value id: {self.result!r}")
        object.__setattr__(self, "steps", steps)


@dataclass(frozen=True)
class SpecialAction:
    predicate: SpecialPredicate | str
    result: SpecialResult | str
    value: float | None = None

    def __post_init__(self) -> None:
        try:
            predicate = (
                self.predicate
                if isinstance(self.predicate, SpecialPredicate)
                else SpecialPredicate(str(self.predicate))
            )
            result = self.result if isinstance(self.result, SpecialResult) else SpecialResult(str(self.result))
        except ValueError as exc:
            raise ValueError("unknown special predicate or result class") from exc
        if result is SpecialResult.CONSTANT:
            if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
                raise ValueError("constant special result requires a finite numeric value")
            value = float(self.value)
            if not math.isfinite(value):
                raise ValueError("constant special result must be finite")
            object.__setattr__(self, "value", value)
        elif self.value is not None:
            raise ValueError("only a constant special result may carry a value")
        object.__setattr__(self, "predicate", predicate)
        object.__setattr__(self, "result", result)


def validate_zone_partition(zones: tuple[Zone, ...], domain: ZoneBounds) -> None:
    if not zones:
        raise ValueError("Program requires at least one finite Zone")
    if zones[0].bounds.lower != domain.lower or (domain.lower_inclusive and not zones[0].bounds.lower_inclusive):
        raise ValueError("Program zones leave a gap at the valid-domain lower bound")
    if zones[-1].bounds.upper != domain.upper or (domain.upper_inclusive and not zones[-1].bounds.upper_inclusive):
        raise ValueError("Program zones leave a gap at the valid-domain upper bound")
    for left, right in zip(zones, zones[1:]):
        if right.bounds.lower < left.bounds.lower:
            raise ValueError("Program zones must be ordered by lower bound")
        if left.bounds.upper < right.bounds.lower:
            raise ValueError(f"Program zones {left.id!r}/{right.id!r} have a gap")
        if left.bounds.upper > right.bounds.lower:
            raise ValueError(f"Program zones {left.id!r}/{right.id!r} overlap")
        owners = int(left.bounds.upper_inclusive) + int(right.bounds.lower_inclusive)
        if owners == 0:
            raise ValueError(f"Program zones {left.id!r}/{right.id!r} leave an endpoint gap")
        if owners == 2:
            raise ValueError(f"Program zones {left.id!r}/{right.id!r} overlap at endpoint")


@dataclass(frozen=True)
class ProgramLoweringContract:
    """Typed, closed compiler intent consumed by target lowering."""

    target_execution_kind: TargetExecutionKind = TargetExecutionKind.SFPU
    piecewise_boundary_ownership: tuple[tuple[float, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_execution_kind",
            TargetExecutionKind.from_value(self.target_execution_kind),
        )
        ownership = tuple(self.piecewise_boundary_ownership)
        for item in ownership:
            if not (
                isinstance(item, tuple)
                and len(item) == 2
                and isinstance(item[0], (int, float))
                and not isinstance(item[0], bool)
                and math.isfinite(float(item[0]))
                and item[1] in {"left", "right"}
            ):
                raise ValueError(
                    "piecewise_boundary_ownership entries must be finite " "(breakpoint, 'left'|'right') pairs"
                )
        if any(float(left[0]) >= float(right[0]) for left, right in zip(ownership, ownership[1:])):
            raise ValueError("piecewise_boundary_ownership breakpoints must be strictly increasing")
        object.__setattr__(
            self,
            "piecewise_boundary_ownership",
            tuple((float(point), owner) for point, owner in ownership),
        )


@dataclass(frozen=True)
class Program:
    activation: str
    activation_schema_version: int
    target: TargetPolicy
    valid_domain: ZoneBounds
    zones: tuple[Zone, ...]
    blocks: Mapping[str, Any]
    io_contract: ProgramIOContract = field(default_factory=ProgramIOContract)
    special_actions: tuple[SpecialAction, ...] = field(default_factory=tuple)
    required_primitives: tuple[str, ...] = field(default_factory=tuple)
    lowering_contract: ProgramLoweringContract = field(default_factory=ProgramLoweringContract)

    def __post_init__(self) -> None:
        if not isinstance(self.activation, str) or not self.activation:
            raise ValueError("Program activation must be a nonempty string")
        if type(self.activation_schema_version) is not int or self.activation_schema_version < 1:
            raise ValueError("activation_schema_version must be a positive integer")
        target = TargetPolicy.from_value(self.target)
        io_contract = ProgramIOContract.from_value(self.io_contract)
        if not isinstance(self.valid_domain, ZoneBounds):
            raise ValueError("Program valid_domain must be typed ZoneBounds")
        zones = tuple(self.zones)
        if len({zone.id for zone in zones}) != len(zones):
            raise ValueError("Program Zone ids must be unique")
        validate_zone_partition(zones, self.valid_domain)
        blocks = dict(self.blocks)
        if any(not isinstance(key, str) or not key for key in blocks):
            raise ValueError("Program block ids must be nonempty strings")
        for zone in zones:
            for step in zone.steps:
                if step.op is ProgramOp.EVAL_BLOCK and step.block not in blocks:
                    raise ValueError(f"unknown Program block reference: {step.block!r}")
        for name, block in blocks.items():
            validate = getattr(block, "validate", None)
            if not callable(validate):
                raise ValueError(f"Program block {name!r} is not a typed approximation")
            if getattr(block, "domain_actions", ()):
                raise ValueError(
                    f"Program block {name!r} cannot carry domain actions; "
                    "the enclosing Program owns finite-zone routing"
                )
            block_target = getattr(block, "target", None)
            if block_target is not None and TargetPolicy.from_value(block_target) != target:
                raise ValueError(f"Program block {name!r} target does not match Program target")
            validate()
        specials = tuple(self.special_actions)
        if any(not isinstance(item, SpecialAction) for item in specials):
            raise ValueError("special_actions must contain typed SpecialAction records")
        predicates = [item.predicate for item in specials]
        if len(set(predicates)) != len(predicates):
            raise ValueError("special_actions predicates must be unique")
        primitives = tuple(self.required_primitives)
        if any(not isinstance(item, str) or not item for item in primitives):
            raise ValueError("required_primitives must be nonempty strings")
        if len(set(primitives)) != len(primitives):
            raise ValueError("required_primitives must be unique")
        if not isinstance(self.lowering_contract, ProgramLoweringContract):
            raise ValueError("lowering_contract must be a typed ProgramLoweringContract")
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "io_contract", io_contract)
        object.__setattr__(self, "zones", zones)
        object.__setattr__(self, "blocks", MappingProxyType(blocks))
        object.__setattr__(self, "special_actions", specials)
        object.__setattr__(self, "required_primitives", primitives)


__all__ = [
    "DEFAULT_BF16_TARGET",
    "DEFAULT_FP32_TARGET",
    "Program",
    "ProgramIOContract",
    "ProgramLoweringContract",
    "ProgramOp",
    "ProgramStep",
    "SpecialAction",
    "SpecialPredicate",
    "SpecialResult",
    "TargetExecutionKind",
    "TargetPolicy",
    "Zone",
    "ZoneBounds",
    "validate_zone_partition",
]
