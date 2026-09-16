# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Typed region actions shared by activation specs and approximation artifacts.

The vocabulary in this module is intentionally small and closed.  Activation
JSON may bind numbers to these records, but it may not supply predicates or
executable expressions.  Lowering therefore depends only on record structure,
never on an activation name (CHARTER C-PIPE-1 / C-LOW-1).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence

from ttpoly.precision import bf16

from .program import TargetPolicy


DOMAIN_ACTIONS_FIELD = "domain_actions"
DOMAIN_ACTIONS_SHA256_FIELD = "domain_actions_sha256"
DOMAIN_ACTION_PROGRAMS_FIELD = "domain_action_programs"
DOMAIN_ACTION_MANIFEST_SCHEMA = "domain_action_manifest_v1"
TARGETED_DOMAIN_ACTION_MANIFEST_SCHEMA = "domain_action_manifest_v2"
DOMAIN_ACTION_MANIFEST_SHA256_FIELD = "manifest_sha256"

COORDINATES = frozenset({"raw", "transformed"})
DIRECTIONS = frozenset({"below", "above"})
PHASES = ("pre_evaluation", "post_reconstruction")
ACTION_KINDS = frozenset({"constant", "identity", "affine", "return_class", "signed_inf"})
RETURN_CLASSES = frozenset({"nan", "pos_inf", "neg_inf", "pos_zero", "neg_zero"})

_RECORD_FIELDS = frozenset({"coordinate", "direction", "bound", "inclusive", "phase", "action"})
_ACTION_FIELDS = {
    "constant": frozenset({"kind", "value"}),
    "identity": frozenset({"kind", "source"}),
    "affine": frozenset({"kind", "source", "scale", "bias"}),
    "return_class": frozenset({"kind", "class"}),
    "signed_inf": frozenset({"kind", "source"}),
}


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite; use return_class for nonfinite output")
    return result


def _closed_fields(value: Mapping[str, Any], allowed: frozenset[str], path: str) -> None:
    missing = allowed - set(value)
    unknown = set(value) - allowed
    if missing:
        raise ValueError(f"{path} missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{path} has unknown fields: {sorted(unknown)}")


@dataclass(frozen=True)
class ResultAction:
    """Closed, immutable action body; unused parameters are structurally absent."""

    kind: str
    value: float | None = None
    source: str | None = None
    scale: float | None = None
    bias: float | None = None
    return_class: str | None = None

    def __post_init__(self):
        if self.kind not in ACTION_KINDS:
            raise ValueError(f"unknown result action kind: {self.kind!r}")
        if self.kind == "constant":
            object.__setattr__(self, "value", _finite_number(self.value, "action.value"))
            populated = {"value"}
        elif self.kind in {"identity", "signed_inf"}:
            if self.source not in COORDINATES:
                raise ValueError(f"action.source must be one of {sorted(COORDINATES)}")
            populated = {"source"}
        elif self.kind == "affine":
            if self.source not in COORDINATES:
                raise ValueError(f"action.source must be one of {sorted(COORDINATES)}")
            object.__setattr__(self, "scale", _finite_number(self.scale, "action.scale"))
            object.__setattr__(self, "bias", _finite_number(self.bias, "action.bias"))
            populated = {"source", "scale", "bias"}
        else:
            if self.return_class not in RETURN_CLASSES:
                raise ValueError(f"action.return_class must be one of {sorted(RETURN_CLASSES)}")
            populated = {"return_class"}
        values = {
            "value": self.value,
            "source": self.source,
            "scale": self.scale,
            "bias": self.bias,
            "return_class": self.return_class,
        }
        extras = sorted(key for key, value in values.items() if value is not None and key not in populated)
        if extras:
            raise ValueError(f"action {self.kind!r} has inapplicable fields: {extras}")

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "constant":
            return {"kind": self.kind, "value": self.value}
        if self.kind in {"identity", "signed_inf"}:
            return {"kind": self.kind, "source": self.source}
        if self.kind == "affine":
            return {
                "kind": self.kind,
                "source": self.source,
                "scale": self.scale,
                "bias": self.bias,
            }
        return {"kind": self.kind, "class": self.return_class}


@dataclass(frozen=True)
class DomainAction:
    """One one-sided, explicitly-owned region action.

    Records are evaluated in list order.  The first matching record claims the
    lane; pre-evaluation claims are terminal and cannot be overwritten by a
    post-reconstruction action.
    """

    coordinate: str
    direction: str
    bound: float
    inclusive: bool
    phase: str
    action: ResultAction

    def __post_init__(self):
        if self.coordinate not in COORDINATES:
            raise ValueError(f"coordinate must be one of {sorted(COORDINATES)}")
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {sorted(DIRECTIONS)}")
        object.__setattr__(self, "bound", _finite_number(self.bound, "bound"))
        if type(self.inclusive) is not bool:
            raise ValueError("inclusive must be a boolean")
        if self.phase not in PHASES:
            raise ValueError(f"phase must be one of {list(PHASES)}")
        if not isinstance(self.action, ResultAction):
            raise ValueError("action must be a typed ResultAction")

    def to_dict(self) -> dict[str, Any]:
        return {
            "coordinate": self.coordinate,
            "direction": self.direction,
            "bound": self.bound,
            "inclusive": self.inclusive,
            "phase": self.phase,
            "action": self.action.to_dict(),
        }


@dataclass(frozen=True)
class SymmetricConstantRawTails:
    """Target-float lowering certificate for two identical constant tails."""

    bound: float
    inclusive: bool
    value: float
    phase: str


@dataclass(frozen=True)
class SymmetricSignedConstantRawTails:
    """Target-float certificate for equal-magnitude, sign-restored tails."""

    bound: float
    inclusive: bool
    magnitude: float
    phase: str


@dataclass(frozen=True)
class RawTerminalActionEnvelope:
    """Closed raw-coordinate tails surrounding one selected-CSV core.

    The selected evaluator may be called only after its input is clamped to
    ``[lower_bound, upper_bound]``.  The original raw coordinate remains live
    for the two terminal result actions.  This is a shape certificate: it
    contains no activation identity and no executable expression.
    """

    lower_bound: float
    upper_bound: float
    lower_inclusive: bool
    upper_inclusive: bool


def raw_terminal_action_envelope_bounds(
    actions: Sequence[DomainAction] | Sequence[Mapping[str, Any]],
) -> RawTerminalActionEnvelope | None:
    """Recognize a closed lower/upper typed-terminal partition.

    One action per side owns a whole tail.  Two actions per side may split an
    exact boundary value from its strict tail, but must share one bound and
    complementary inclusivity.  Unsupported shapes fail closed.
    """

    typed = parse_domain_actions(actions)
    below = [record for record in typed if record.direction == "below"]
    above = [record for record in typed if record.direction == "above"]
    if (
        len(typed) not in {2, 4}
        or len(below) not in {1, 2}
        or len(above) not in {1, 2}
        or len(below) + len(above) != len(typed)
    ):
        return None
    if len(below) != len(above):
        return None

    for side in (below, above):
        if len(side) == 2 and not (
            side[0].bound == side[1].bound and {side[0].inclusive, side[1].inclusive} == {False, True}
        ):
            return None
    if any(record.coordinate != "raw" or record.phase != "post_reconstruction" for record in typed):
        return None
    for record in typed:
        body = record.action
        if body.kind in {"identity", "affine", "signed_inf"} and body.source != "raw":
            return None

    lower = _target_float32(below[0].bound)
    upper = _target_float32(above[0].bound)
    if lower is None or upper is None or not lower < upper:
        return None
    return RawTerminalActionEnvelope(
        lower_bound=lower,
        upper_bound=upper,
        lower_inclusive=any(record.inclusive for record in below),
        upper_inclusive=any(record.inclusive for record in above),
    )


def detect_raw_terminal_action_envelope(
    actions: Sequence[DomainAction] | Sequence[Mapping[str, Any]],
    *,
    callable_lower: float,
    callable_upper: float,
) -> RawTerminalActionEnvelope | None:
    """Recognize typed tails whose untouched core is callable by the CSV."""

    envelope = raw_terminal_action_envelope_bounds(actions)
    if envelope is None:
        return None
    domain_lower = _target_float32(callable_lower)
    domain_upper = _target_float32(callable_upper)
    if domain_lower is None or domain_upper is None:
        return None
    if not (domain_lower <= envelope.lower_bound < envelope.upper_bound <= domain_upper):
        return None
    return envelope


def _target_float32(value: float) -> float | None:
    try:
        return struct.unpack(">f", struct.pack(">f", float(value)))[0]
    except OverflowError:
        return None


def _target_float32_bits(value: float) -> bytes | None:
    rounded = _target_float32(value)
    return None if rounded is None else struct.pack(">f", rounded)


def detect_symmetric_constant_raw_tails(
    actions: Sequence[DomainAction] | Sequence[Mapping[str, Any]],
) -> SymmetricConstantRawTails | None:
    """Recognize an exactly equivalent ``abs(raw)`` constant-tail lowering.

    The recognition is intentionally closed: the complete action program must
    be exactly one below/above pair, both actions must be post-reconstruction
    raw-coordinate constants, and inclusivity, phase, target-float bounds, and
    target-float result bits must agree. Signed-zero results do not coalesce.
    Matching target-float bits mirrors the literals consumed by the device and
    avoids making an optimization decision from host double values.
    """

    typed = parse_domain_actions(actions)
    if len(typed) != 2:
        return None
    by_direction = {record.direction: record for record in typed}
    if set(by_direction) != {"below", "above"}:
        return None
    below = by_direction["below"]
    above = by_direction["above"]
    if not (
        below.coordinate == above.coordinate == "raw"
        and below.phase == above.phase == "post_reconstruction"
        and below.inclusive == above.inclusive
        and below.action.kind == above.action.kind == "constant"
    ):
        return None

    low_bound = _target_float32(below.bound)
    high_bound = _target_float32(above.bound)
    low_bits = _target_float32_bits(below.bound)
    neg_high_bits = None if high_bound is None else _target_float32_bits(-high_bound)
    low_value_bits = _target_float32_bits(below.action.value)
    high_value_bits = _target_float32_bits(above.action.value)
    if (
        low_bound is None
        or high_bound is None
        or high_bound < 0.0
        or low_bits != neg_high_bits
        or low_value_bits is None
        or low_value_bits != high_value_bits
    ):
        return None
    value = _target_float32(above.action.value)
    assert value is not None
    return SymmetricConstantRawTails(
        bound=high_bound,
        inclusive=above.inclusive,
        value=value,
        phase=above.phase,
    )


def detect_symmetric_signed_constant_raw_tails(
    actions: Sequence[DomainAction] | Sequence[Mapping[str, Any]],
) -> SymmetricSignedConstantRawTails | None:
    """Recognize ``copysign(c, raw)`` over symmetric constant tails.

    This is deliberately disjoint from identical-constant compaction.  The
    complete program must be one inclusive-compatible below/above pair with
    exact opposite FP32 constant bits and a strictly positive magnitude.
    """

    typed = parse_domain_actions(actions)
    if len(typed) != 2:
        return None
    by_direction = {record.direction: record for record in typed}
    if set(by_direction) != {"below", "above"}:
        return None
    below = by_direction["below"]
    above = by_direction["above"]
    if not (
        below.coordinate == above.coordinate == "raw"
        and below.phase == above.phase == "post_reconstruction"
        and below.inclusive == above.inclusive
        and below.action.kind == above.action.kind == "constant"
    ):
        return None
    low_bound = _target_float32(below.bound)
    high_bound = _target_float32(above.bound)
    low_value = _target_float32(below.action.value)
    high_value = _target_float32(above.action.value)
    if (
        low_bound is None
        or high_bound is None
        or low_value is None
        or high_value is None
        or high_bound <= 0.0
        or high_value <= 0.0
        or _target_float32_bits(low_bound) != _target_float32_bits(-high_bound)
        or _target_float32_bits(low_value) != _target_float32_bits(-high_value)
    ):
        return None
    return SymmetricSignedConstantRawTails(
        bound=high_bound,
        inclusive=above.inclusive,
        magnitude=high_value,
        phase=above.phase,
    )


def parse_domain_action(value: Any, *, path: str = "domain_actions[]") -> DomainAction:
    """Validate and normalize one JSON-compatible action record."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    _closed_fields(value, _RECORD_FIELDS, path)

    coordinate = value["coordinate"]
    if coordinate not in COORDINATES:
        raise ValueError(f"{path}.coordinate must be one of {sorted(COORDINATES)}")
    direction = value["direction"]
    if direction not in DIRECTIONS:
        raise ValueError(f"{path}.direction must be one of {sorted(DIRECTIONS)}")
    if type(value["inclusive"]) is not bool:
        raise ValueError(f"{path}.inclusive must be a boolean")
    phase = value["phase"]
    if phase not in PHASES:
        raise ValueError(f"{path}.phase must be one of {list(PHASES)}")

    action = value["action"]
    if not isinstance(action, Mapping):
        raise ValueError(f"{path}.action must be an object")
    kind = action.get("kind")
    if kind not in ACTION_KINDS:
        raise ValueError(f"{path}.action.kind must be one of {sorted(ACTION_KINDS)}")
    _closed_fields(action, _ACTION_FIELDS[kind], f"{path}.action")

    normalized_action: ResultAction
    if kind == "constant":
        normalized_action = ResultAction(
            kind=kind,
            value=_finite_number(action["value"], f"{path}.action.value"),
        )
    elif kind in {"identity", "affine", "signed_inf"}:
        if action["source"] not in COORDINATES:
            raise ValueError(f"{path}.action.source must be one of {sorted(COORDINATES)}")
        if kind in {"identity", "signed_inf"}:
            normalized_action = ResultAction(kind=kind, source=action["source"])
        else:
            normalized_action = ResultAction(
                kind=kind,
                source=action["source"],
                scale=_finite_number(action["scale"], f"{path}.action.scale"),
                bias=_finite_number(action["bias"], f"{path}.action.bias"),
            )
    else:
        if action["class"] not in RETURN_CLASSES:
            raise ValueError(f"{path}.action.class must be one of {sorted(RETURN_CLASSES)}")
        normalized_action = ResultAction(kind=kind, return_class=action["class"])

    return DomainAction(
        coordinate=coordinate,
        direction=direction,
        bound=_finite_number(value["bound"], f"{path}.bound"),
        inclusive=value["inclusive"],
        phase=phase,
        action=normalized_action,
    )


def parse_domain_actions(value: Any) -> tuple[DomainAction, ...]:
    """Validate an ordered action sequence and return immutable typed records."""
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("domain_actions must be an array")
    actions = tuple(
        item if isinstance(item, DomainAction) else parse_domain_action(item, path=f"domain_actions[{i}]")
        for i, item in enumerate(value)
    )
    phase_indices = [PHASES.index(item.phase) for item in actions]
    if phase_indices != sorted(phase_indices):
        raise ValueError("domain_actions must be phase-ordered: pre_evaluation before " "post_reconstruction")
    return actions


def domain_actions_from_spec(spec: Mapping[str, Any]) -> tuple[DomainAction, ...]:
    """Read only the typed field; dormant legacy boundary strings stay ignored."""
    if not isinstance(spec, Mapping):
        raise ValueError("activation spec must be an object")
    return parse_domain_actions(spec.get(DOMAIN_ACTIONS_FIELD))


def _proof_from_program(
    program: Mapping[str, Any], target: TargetPolicy, actions: tuple[DomainAction, ...]
) -> dict[str, str]:
    proof = program.get("proof")
    if not isinstance(proof, Mapping):
        raise ValueError("target domain-action program proof must be an object")
    required = {"kind", "precision", "population"}
    optional = {
        "derivation_kind",
        "derivation_sha256",
    }
    allowed = required | optional
    missing = required - set(proof)
    unknown = set(proof) - allowed
    if missing:
        raise ValueError("target domain-action program proof missing fields: " f"{sorted(missing)}")
    if unknown:
        raise ValueError("target domain-action program proof has unknown fields: " f"{sorted(unknown)}")
    kind = proof["kind"]
    if kind not in {"exhaustive", "mathematical_contract"}:
        raise ValueError("domain-action proof kind must be exhaustive or mathematical_contract")
    if proof["precision"] != target.precision:
        raise ValueError("domain-action proof precision does not match target precision")
    if not isinstance(proof["population"], str) or not proof["population"].strip():
        raise ValueError("domain-action proof population must be a nonempty string")
    derivation_fields = {"derivation_kind", "derivation_sha256"} & set(proof)
    if derivation_fields and derivation_fields != {
        "derivation_kind",
        "derivation_sha256",
    }:
        raise ValueError("domain-action derivation kind and SHA-256 must be declared together")
    if derivation_fields:
        if proof["derivation_kind"] not in {
            "target_independent_return_class_closure_v1",
            "target_rounded_finite_endpoint_closure_v1",
        }:
            raise ValueError("unknown domain-action derivation kind")
        derivation_sha256 = _sha256_hex(
            proof["derivation_sha256"],
            "target domain-action derivation SHA-256",
        )
        if derivation_sha256 != domain_actions_sha256(actions):
            raise ValueError("target domain-action derivation SHA-256 does not match actions")
        if proof["derivation_kind"] == (
            "target_rounded_finite_endpoint_closure_v1"
        ) and not _is_target_exact_finite_endpoint_program(actions, target):
            raise ValueError(
                "target-rounded finite endpoint derivation requires exact " "single-encoding endpoint actions"
            )
    has_finite_result = any(action.action.kind not in {"return_class"} for action in actions)
    finite_endpoint_derivation = proof.get("derivation_kind") == ("target_rounded_finite_endpoint_closure_v1")
    if has_finite_result and kind != "exhaustive" and not finite_endpoint_derivation:
        raise ValueError("finite domain actions require a target-specific exhaustive proof")
    return {key: str(proof[key]) for key in sorted(allowed) if key in proof}


def _close_target_independent_return_classes(
    spec: Mapping[str, Any],
    selected: tuple[DomainAction, ...],
    proof: Mapping[str, str],
) -> tuple[tuple[DomainAction, ...], dict[str, str]]:
    """Add omitted target-independent class actions without inventing behavior.

    ``domain_actions`` is the canonical ordered mathematical contract.  A
    target program may historically contain only a subsequence of that list.
    When the complete canonical contract consists only of ``return_class``
    records, it has no destination-precision coefficient or rounding
    dependence.  When (and only when) the explicit target list is an exact
    subsequence, close it to that canonical first-match program.

    Any contract containing finite constants, identity/affine tails, or signed
    results is excluded as a whole: it requires separate target-lattice proof.
    A target-specific action that is not present byte-for-byte in the canonical
    list also fails closed and is left untouched rather than being reordered
    around inferred records.
    """
    canonical = domain_actions_from_spec(spec)
    if not canonical:
        return selected, dict(proof)
    if any(action.action.kind != "return_class" for action in canonical):
        # Mixed finite/class contracts need a target-lattice proof for the
        # complete first-match program.  Lifting only their class subset can
        # change which lanes reach a finite tail, so refuse the partial closure.
        return selected, dict(proof)

    paired_overflow = (
        len(canonical) == 2
        and [action.direction for action in canonical] == ["below", "above"]
        and all(action.coordinate == "raw" for action in canonical)
        and all(action.phase == "pre_evaluation" for action in canonical)
        and all(action.inclusive for action in canonical)
        and canonical[0].bound == -canonical[1].bound
    )
    paired_invalid_and_poles = (
        len(canonical) == 4
        and [action.direction for action in canonical] == ["below", "above", "below", "above"]
        and all(action.coordinate == "raw" for action in canonical)
        and all(action.phase == "pre_evaluation" for action in canonical)
        and [action.inclusive for action in canonical] == [False, False, True, True]
        and canonical[0].bound == canonical[2].bound
        and canonical[1].bound == canonical[3].bound
        and canonical[0].action.return_class == "nan"
        and canonical[1].action.return_class == "nan"
    )
    one_sided_invalid_and_pole = (
        len(canonical) == 2
        and canonical[0].coordinate == canonical[1].coordinate == "raw"
        and canonical[0].phase == canonical[1].phase == "pre_evaluation"
        and canonical[0].direction == canonical[1].direction
        and [action.inclusive for action in canonical] == [False, True]
        and canonical[0].bound == canonical[1].bound
        and canonical[0].action.return_class == "nan"
        and canonical[1].action.return_class in {"neg_inf", "pos_inf"}
    )
    if not (paired_overflow or paired_invalid_and_poles or one_sided_invalid_and_pole):
        return selected, dict(proof)

    cursor = 0
    for action in selected:
        try:
            index = canonical.index(action, cursor)
        except ValueError:
            return selected, dict(proof)
        cursor = index + 1
    closed = canonical
    if closed == selected:
        return selected, dict(proof)

    closed_proof = dict(proof)
    closed_proof.update(
        derivation_kind="target_independent_return_class_closure_v1",
        derivation_sha256=domain_actions_sha256(canonical),
    )
    return closed, closed_proof


def _target_rounded_finite(value: float, target: TargetPolicy) -> float | None:
    """Return the target's deterministic RNE/FTZ encoding of a finite scalar."""
    if target.rounding_order != "post_round":
        return None
    rounded = bf16.quantize(
        [value],
        target.precision,
        flush_to_zero=target.output_ftz,
        rounding_mode="rne",
    )[0]
    result = float(rounded)
    return result if math.isfinite(result) else None


def _is_paired_finite_endpoint_shape(
    actions: tuple[DomainAction, ...],
) -> bool:
    return (
        len(actions) == 4
        and [action.direction for action in actions] == ["below", "above", "below", "above"]
        and all(action.coordinate == "raw" for action in actions)
        and [action.phase for action in actions]
        == ["pre_evaluation", "pre_evaluation", "post_reconstruction", "post_reconstruction"]
        and [action.inclusive for action in actions] == [False, False, True, True]
        and actions[0].bound == actions[2].bound
        and actions[1].bound == actions[3].bound
        and actions[0].bound < actions[1].bound
        and actions[0].action.return_class == "nan"
        and actions[1].action.return_class == "nan"
        and actions[2].action.kind == "constant"
        and actions[3].action.kind == "constant"
    )


def _is_target_exact_finite_endpoint_program(actions: tuple[DomainAction, ...], target: TargetPolicy) -> bool:
    if not _is_paired_finite_endpoint_shape(actions):
        return False
    # DAZ would alias zero/subnormal bounds with multiple raw encodings, so the
    # single-encoding theorem is restricted to normal endpoint coordinates.
    if target.input_daz and any(abs(action.bound) < bf16.MIN_NORMAL for action in actions[:2]):
        return False
    target_values = tuple(
        _target_rounded_finite(value, target)
        for value in (
            actions[0].bound,
            actions[1].bound,
            actions[2].action.value,
            actions[3].action.value,
        )
    )
    return target_values == (
        actions[0].bound,
        actions[1].bound,
        actions[2].action.value,
        actions[3].action.value,
    )


def _close_target_rounded_finite_endpoints(
    spec: Mapping[str, Any],
    target: TargetPolicy,
    selected: tuple[DomainAction, ...],
    proof: Mapping[str, str],
) -> tuple[tuple[DomainAction, ...], dict[str, str]]:
    """Materialize target-rounded constants that own exact endpoint encodings.

    The only admitted shape is a paired finite interval whose exclusive invalid
    exteriors precede paired inclusive constant actions at the identical raw
    bounds.  First-match ownership therefore reduces each finite action to one
    exactly representable input encoding; it cannot silently introduce a tail
    or interval plateau.  The canonical constants are mathematical values and
    are rounded once to the declared output target.
    """
    canonical = domain_actions_from_spec(spec)
    if not _is_paired_finite_endpoint_shape(canonical):
        return selected, dict(proof)
    if target.input_daz and any(abs(action.bound) < bf16.MIN_NORMAL for action in canonical[:2]):
        return selected, dict(proof)

    rounded_bounds = tuple(_target_rounded_finite(action.bound, target) for action in canonical[:2])
    if rounded_bounds != tuple(action.bound for action in canonical[:2]):
        return selected, dict(proof)
    rounded_values = tuple(_target_rounded_finite(action.action.value, target) for action in canonical[2:])
    if any(value is None for value in rounded_values):
        return selected, dict(proof)

    closed = canonical[:2] + tuple(
        replace(action, action=replace(action.action, value=value))
        for action, value in zip(canonical[2:], rounded_values)
    )
    cursor = 0
    for action in selected:
        try:
            index = closed.index(action, cursor)
        except ValueError:
            return selected, dict(proof)
        cursor = index + 1
    if closed == selected:
        return selected, dict(proof)

    closed_proof = dict(proof)
    closed_proof.update(
        derivation_kind="target_rounded_finite_endpoint_closure_v1",
        derivation_sha256=domain_actions_sha256(closed),
    )
    return closed, closed_proof


def _target_program_from_spec(
    spec: Mapping[str, Any], target: TargetPolicy
) -> tuple[tuple[DomainAction, ...], dict[str, str]]:
    programs = spec.get(DOMAIN_ACTION_PROGRAMS_FIELD)
    if programs is None:
        legacy = domain_actions_from_spec(spec)
        if legacy:
            raise ValueError("activation domain actions are not target-scoped; add domain_action_programs")
        return (), {
            "kind": "mathematical_contract",
            "precision": target.precision,
            "population": "empty_program",
        }
    if isinstance(programs, (str, bytes)) or not isinstance(programs, Sequence):
        raise ValueError("domain_action_programs must be an array")
    matches: list[tuple[tuple[DomainAction, ...], dict[str, str]]] = []
    for index, program in enumerate(programs):
        path = f"domain_action_programs[{index}]"
        if not isinstance(program, Mapping):
            raise ValueError(f"{path} must be an object")
        _closed_fields(
            program,
            frozenset({"target", DOMAIN_ACTIONS_FIELD, "proof"}),
            path,
        )
        program_target = TargetPolicy.from_value(program["target"])
        actions = parse_domain_actions(program[DOMAIN_ACTIONS_FIELD])
        proof = _proof_from_program(program, program_target, actions)
        if program_target == target:
            matches.append((actions, proof))
    if not matches:
        raise ValueError(
            "activation has no domain-action program for target "
            f"{target.precision}/{target.architecture}/{target.sha256[:12]}"
        )
    if len(matches) != 1:
        raise ValueError("activation has duplicate domain-action programs for target")
    actions, proof = matches[0]
    actions, proof = _close_target_independent_return_classes(spec, actions, proof)
    return _close_target_rounded_finite_endpoints(spec, target, actions, proof)


def target_domain_actions_from_spec(
    spec: Mapping[str, Any], target: TargetPolicy | Mapping[str, Any]
) -> tuple[DomainAction, ...]:
    """Select exactly one independently proven action program for ``target``."""
    if not isinstance(spec, Mapping):
        raise ValueError("activation spec must be an object")
    typed_target = TargetPolicy.from_value(target)
    return _target_program_from_spec(spec, typed_target)[0]


def invalid_domain_actions(valid_domain: Mapping[str, Any] | None) -> tuple[DomainAction, ...]:
    """Translate a generic interval contract into terminal invalid-region actions.

    This helper is validation/migration machinery, not name-based activation
    behavior.  Specs still carry their explicit compiled actions so the action
    manifest—not a later inference—drives host and kernel evaluation.
    """
    if valid_domain is None:
        return ()
    if not isinstance(valid_domain, Mapping):
        raise ValueError("valid_domain must be an object")
    allowed = {
        "min",
        "max",
        "min_exclusive",
        "max_exclusive",
        "min_result",
        "max_result",
    }
    unknown = set(valid_domain) - allowed
    if unknown:
        raise ValueError(f"valid_domain has unknown fields: {sorted(unknown)}")
    records = []
    for edge, direction, exclusive in (
        ("min", "below", "min_exclusive"),
        ("max", "above", "max_exclusive"),
    ):
        if edge not in valid_domain:
            continue
        records.append(
            {
                "coordinate": "raw",
                "direction": direction,
                "bound": valid_domain[edge],
                # If the valid interval excludes its endpoint, the invalid
                # exterior owns that endpoint.
                "inclusive": bool(valid_domain.get(exclusive, False)),
                "phase": "pre_evaluation",
                "action": {"kind": "return_class", "class": "nan"},
            }
        )
    return parse_domain_actions(records)


def serialize_domain_actions(actions: Any) -> str:
    """Canonical artifact/CSV serialization used for provenance hashes."""
    records = [action.to_dict() for action in parse_domain_actions(actions)]
    return json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False)


def domain_actions_sha256(actions: Any) -> str:
    """Stable content identity, including the explicit empty-program identity."""
    encoded = serialize_domain_actions(actions).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_hex(value: Any, path: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{path} must be a 64-character SHA-256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{path} must be hexadecimal") from exc
    return value.lower()


def _manifest_payload(
    actions: Any,
    binding: Mapping[str, Any],
    *,
    target: TargetPolicy | None = None,
    proof: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    typed = parse_domain_actions(actions)
    payload = {
        "schema": (DOMAIN_ACTION_MANIFEST_SCHEMA if target is None else TARGETED_DOMAIN_ACTION_MANIFEST_SCHEMA),
        "binding": dict(binding),
        DOMAIN_ACTIONS_FIELD: [action.to_dict() for action in typed],
        DOMAIN_ACTIONS_SHA256_FIELD: domain_actions_sha256(typed),
    }
    if target is not None:
        payload.update(
            {
                "target": target.to_dict(),
                "target_sha256": target.sha256,
                "proof": dict(proof or {}),
            }
        )
    return payload


def _manifest_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def synthetic_domain_action_manifest(
    actions: Any,
    *,
    target: TargetPolicy | Mapping[str, Any] | None = None,
    proof: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build an explicitly anonymous manifest for synthetic conformance only."""
    typed_target = None if target is None else TargetPolicy.from_value(target)
    binding = {"kind": "anonymous_synthetic"}
    if typed_target is not None:
        binding["target_sha256"] = typed_target.sha256
        if proof is None:
            typed_actions = parse_domain_actions(actions)
            proof = {
                "kind": (
                    "exhaustive"
                    if any(item.action.kind != "return_class" for item in typed_actions)
                    else "mathematical_contract"
                ),
                "precision": typed_target.precision,
                "population": "synthetic_fixture",
            }
    payload = _manifest_payload(actions, binding, target=typed_target, proof=proof)
    return {**payload, DOMAIN_ACTION_MANIFEST_SHA256_FIELD: _manifest_sha256(payload)}


def bound_domain_action_manifest(
    spec: Mapping[str, Any],
    spec_path: str | Path,
    *,
    target: TargetPolicy | Mapping[str, Any] | None = None,
    source_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind a production action program to an activation and exact spec bytes."""
    if not isinstance(spec, Mapping) or not isinstance(spec.get("name"), str):
        raise ValueError("bound activation spec must declare a string name")
    path = Path(spec_path)
    try:
        raw = path.read_bytes()
        disk_spec = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read bound activation spec {path}") from exc
    expected_disk_spec = dict(spec if source_spec is None else source_spec)
    if disk_spec != expected_disk_spec:
        raise ValueError("bound activation spec mapping does not match spec_path bytes")
    binding = {
        "kind": "activation_spec",
        "activation": spec["name"],
        "activation_spec_sha256": hashlib.sha256(raw).hexdigest(),
    }
    typed_target = None if target is None else TargetPolicy.from_value(target)
    if typed_target is None:
        actions = domain_actions_from_spec(spec)
        if actions:
            raise ValueError("action-bearing production manifests require an explicit target")
        proof = None
    else:
        actions, proof = _target_program_from_spec(spec, typed_target)
        binding["target_sha256"] = typed_target.sha256
    payload = _manifest_payload(actions, binding, target=typed_target, proof=proof)
    return {**payload, DOMAIN_ACTION_MANIFEST_SHA256_FIELD: _manifest_sha256(payload)}


def domain_actions_from_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_activation: str | None = None,
    expected_spec_sha256: str | None = None,
    allow_synthetic: bool = False,
    expected_target: TargetPolicy | Mapping[str, Any] | None = None,
) -> tuple[DomainAction, ...]:
    """Verify a bound manifest and optionally its expected production identity."""
    if not isinstance(manifest, Mapping):
        raise ValueError("domain action manifest must be an object")
    base_fields = {
        "schema",
        "binding",
        DOMAIN_ACTIONS_FIELD,
        DOMAIN_ACTIONS_SHA256_FIELD,
        DOMAIN_ACTION_MANIFEST_SHA256_FIELD,
    }
    schema = manifest.get("schema")
    targeted = schema == TARGETED_DOMAIN_ACTION_MANIFEST_SCHEMA
    expected_fields = base_fields | ({"target", "target_sha256", "proof"} if targeted else set())
    _closed_fields(manifest, frozenset(expected_fields), "domain action manifest")
    if schema not in {DOMAIN_ACTION_MANIFEST_SCHEMA, TARGETED_DOMAIN_ACTION_MANIFEST_SCHEMA}:
        raise ValueError(f"unknown domain action manifest schema: {manifest['schema']!r}")
    typed_target = None
    if targeted:
        typed_target = TargetPolicy.from_value(manifest["target"])
        if manifest["target_sha256"] != typed_target.sha256:
            raise ValueError("manifest target_sha256 does not match target")
        if expected_target is not None and typed_target != TargetPolicy.from_value(expected_target):
            raise ValueError("manifest target does not match expected target")
    elif expected_target is not None:
        raise ValueError("legacy manifest cannot satisfy a target binding")
    binding = manifest["binding"]
    if not isinstance(binding, Mapping):
        raise ValueError("domain action manifest binding must be an object")
    kind = binding.get("kind")
    if kind == "anonymous_synthetic":
        binding_fields = {"kind"} | ({"target_sha256"} if targeted else set())
        _closed_fields(binding, frozenset(binding_fields), "domain action manifest binding")
        if not allow_synthetic:
            raise ValueError("anonymous synthetic domain action manifest is not production evidence")
        if expected_activation is not None or expected_spec_sha256 is not None:
            raise ValueError("anonymous synthetic manifest cannot satisfy activation binding")
    elif kind == "activation_spec":
        binding_fields = {"kind", "activation", "activation_spec_sha256"}
        if targeted:
            binding_fields.add("target_sha256")
        _closed_fields(
            binding,
            frozenset(binding_fields),
            "domain action manifest binding",
        )
        activation = binding["activation"]
        if not isinstance(activation, str) or not activation:
            raise ValueError("manifest binding activation must be a nonempty string")
        spec_sha = _sha256_hex(
            binding["activation_spec_sha256"],
            "manifest binding activation_spec_sha256",
        )
        if expected_activation is not None and activation != expected_activation:
            raise ValueError(f"manifest activation {activation!r} != expected {expected_activation!r}")
        if expected_spec_sha256 is not None:
            expected_sha = _sha256_hex(expected_spec_sha256, "expected_spec_sha256")
            if spec_sha != expected_sha:
                raise ValueError("manifest activation spec SHA does not match expected spec")
    else:
        raise ValueError(f"unknown domain action manifest binding kind: {kind!r}")

    if targeted and binding["target_sha256"] != typed_target.sha256:
        raise ValueError("manifest binding target hash does not match target")

    actions = parse_domain_actions(manifest[DOMAIN_ACTIONS_FIELD])
    if targeted:
        _proof_from_program({"proof": manifest["proof"]}, typed_target, actions)
    recorded_hash = manifest.get(DOMAIN_ACTIONS_SHA256_FIELD)
    if recorded_hash != domain_actions_sha256(actions):
        raise ValueError(f"{DOMAIN_ACTIONS_SHA256_FIELD} does not match {DOMAIN_ACTIONS_FIELD}")
    payload = {key: manifest[key] for key in expected_fields if key != DOMAIN_ACTION_MANIFEST_SHA256_FIELD}
    if manifest[DOMAIN_ACTION_MANIFEST_SHA256_FIELD] != _manifest_sha256(payload):
        raise ValueError("manifest_sha256 does not match bound manifest content")
    return actions


__all__ = [
    "ACTION_KINDS",
    "COORDINATES",
    "DIRECTIONS",
    "DOMAIN_ACTIONS_FIELD",
    "DOMAIN_ACTION_PROGRAMS_FIELD",
    "DOMAIN_ACTIONS_SHA256_FIELD",
    "DOMAIN_ACTION_MANIFEST_SCHEMA",
    "TARGETED_DOMAIN_ACTION_MANIFEST_SCHEMA",
    "DOMAIN_ACTION_MANIFEST_SHA256_FIELD",
    "DomainAction",
    "PHASES",
    "RETURN_CLASSES",
    "ResultAction",
    "RawTerminalActionEnvelope",
    "SymmetricConstantRawTails",
    "SymmetricSignedConstantRawTails",
    "bound_domain_action_manifest",
    "domain_actions_from_manifest",
    "domain_actions_sha256",
    "invalid_domain_actions",
    "domain_actions_from_spec",
    "detect_symmetric_constant_raw_tails",
    "detect_symmetric_signed_constant_raw_tails",
    "detect_raw_terminal_action_envelope",
    "raw_terminal_action_envelope_bounds",
    "parse_domain_action",
    "parse_domain_actions",
    "serialize_domain_actions",
    "synthetic_domain_action_manifest",
    "target_domain_actions_from_spec",
]
