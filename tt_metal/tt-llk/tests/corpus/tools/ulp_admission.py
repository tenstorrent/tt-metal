#!/usr/bin/env python3
"""Lightweight same-oracle, per-class ULP non-regression admission."""

from __future__ import annotations

import math
import re


_CLASS_RE = re.compile(r"[a-z][a-z0-9_]*")
IEEE_BF16_CLASSES = frozenset(
    {"nan", "pos_inf", "neg_inf", "pos_zero", "neg_zero", "pos_subnormal", "neg_subnormal"}
)
UNARY_DOMAIN_PARTITION_OPS = frozenset({"erfinv-fresh"})


def unary_class_names(domain_partition: bool) -> frozenset[str]:
    names = {f"{name}_input" for name in IEEE_BF16_CLASSES}
    names.add("in_domain_finite_normal")
    if domain_partition:
        names.update(
            {
                "domain_lower_boundary",
                "domain_upper_boundary",
                "out_of_domain_finite_normal",
            }
        )
    return frozenset(names)


BINARY_CLASS_NAMES = frozenset(
    {f"base_{name}" for name in IEEE_BF16_CLASSES}
    | {f"normal_base_exp_{name}" for name in IEEE_BF16_CLASSES}
    | {"pos_normal_base_normal_exp", "neg_normal_base_normal_exp"}
)


def parse_class_ulp(encoded: str) -> dict[str, tuple[int, float]]:
    """Parse ``class:count:max_ulp|...`` and reject partial/ambiguous data."""
    if not encoded:
        raise ValueError("missing class_ulp")
    result: dict[str, tuple[int, float]] = {}
    for item in encoded.split("|"):
        fields = item.split(":")
        if len(fields) != 3 or _CLASS_RE.fullmatch(fields[0]) is None:
            raise ValueError(f"malformed class_ulp item: {item!r}")
        name = fields[0]
        if name in result:
            raise ValueError(f"duplicate class_ulp class: {name}")
        try:
            count = int(fields[1])
            maximum = float(fields[2])
        except ValueError as error:
            raise ValueError(f"malformed class_ulp numbers: {item!r}") from error
        if (
            count <= 0
            or not math.isfinite(maximum)
            or maximum < 0
            or not maximum.is_integer()
        ):
            raise ValueError(f"invalid class_ulp values: {item!r}")
        result[name] = (count, maximum)
    return result


def fold_class_ulp(
    aggregate: dict[str, tuple[int, float]], encoded: str
) -> None:
    for name, (count, maximum) in parse_class_ulp(encoded).items():
        old_count, old_maximum = aggregate.get(name, (0, 0.0))
        aggregate[name] = (old_count + count, max(old_maximum, maximum))


def candidate_not_worse(
    candidate: dict[str, tuple[int, float]],
    hand: dict[str, tuple[int, float]],
) -> tuple[bool, str]:
    """Compare per-class maxima, not pointwise errors, on identical populations."""
    if not candidate or not hand:
        return False, "missing-class-ulp"
    if set(candidate) != set(hand):
        return False, "class-set-mismatch"
    for name in sorted(candidate):
        candidate_count, candidate_ulp = candidate[name]
        hand_count, hand_ulp = hand[name]
        try:
            candidate_max = float(candidate_ulp)
            hand_max = float(hand_ulp)
        except (TypeError, ValueError):
            return False, f"invalid-class-metric-{name}"
        if (
            type(candidate_count) is not int
            or type(hand_count) is not int
            or candidate_count <= 0
            or hand_count <= 0
            or not math.isfinite(candidate_max)
            or not math.isfinite(hand_max)
            or candidate_max < 0
            or hand_max < 0
            or not candidate_max.is_integer()
            or not hand_max.is_integer()
        ):
            return False, f"invalid-class-metric-{name}"
        if candidate_count != hand_count:
            return False, f"class-count-mismatch-{name}"
        if candidate_max > hand_max:
            return False, f"candidate-ulp-regression-{name}"
    return True, "candidate-max-ulp-le-hand-every-class"


def format_class_ulp(values: dict[str, tuple[int, float]]) -> str:
    return "|".join(
        f"{name}:{count}:{maximum:.17g}"
        for name, (count, maximum) in sorted(values.items())
        if count > 0
    )
