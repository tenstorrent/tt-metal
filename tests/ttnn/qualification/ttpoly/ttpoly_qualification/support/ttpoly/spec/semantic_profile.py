# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Typed selection of API-level activation semantics.

Compliance profiles are activation-spec data. They may refine semantic
fields (terminal actions, special factors, and other contract declarations),
but they never select coefficient artifacts, evaluator families, or runtime
implementations.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


SEMANTIC_PROFILES = frozenset({"torch", "ttnn"})
DEFAULT_SEMANTIC_PROFILE = "torch"


def semantic_profile(value: str | None) -> str:
    """Validate and canonicalize a public compliance-profile name."""
    result = DEFAULT_SEMANTIC_PROFILE if value is None else str(value).strip().lower()
    if result not in SEMANTIC_PROFILES:
        raise ValueError(f"unsupported compliance profile {value!r}; expected torch or ttnn")
    return result


def _merge_contract(base: dict[str, Any], patch: Mapping[str, Any]) -> None:
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            child = dict(base[key])
            _merge_contract(child, value)
            base[key] = child
        else:
            base[key] = deepcopy(value)


def _replace_special_actions(value: Any, replacements: Mapping[str, Any]) -> None:
    """Apply predicate-keyed terminal replacements throughout typed Programs."""
    if isinstance(value, dict):
        actions = value.get("special_actions")
        if isinstance(actions, list):
            for index, action in enumerate(actions):
                if not isinstance(action, Mapping):
                    continue
                predicate = action.get("predicate")
                if predicate in replacements:
                    replacement = replacements[predicate]
                    if not isinstance(replacement, Mapping):
                        raise ValueError("special_action_overrides values must be action objects")
                    actions[index] = {"predicate": predicate, **deepcopy(replacement)}
        for child in value.values():
            _replace_special_actions(child, replacements)
    elif isinstance(value, list):
        for child in value:
            _replace_special_actions(child, replacements)


def _replace_domain_actions(value: Any, replacements: list[Any]) -> None:
    """Apply exact structural domain-action replacements recursively."""
    if isinstance(value, dict):
        actions = value.get("domain_actions")
        if isinstance(actions, list):
            for action in actions:
                if not isinstance(action, dict):
                    continue
                for replacement in replacements:
                    if not isinstance(replacement, Mapping) or set(replacement) not in (
                        {"match", "action"},
                        {"match", "replace"},
                    ):
                        raise ValueError(
                            "domain_action_overrides entries require match and either " "action or replace"
                        )
                    match = replacement["match"]
                    if not isinstance(match, Mapping):
                        raise ValueError("domain_action_overrides match must be an object")
                    if all(action.get(key) == expected for key, expected in match.items()):
                        if "action" in replacement:
                            new_action = replacement["action"]
                            if not isinstance(new_action, Mapping):
                                raise ValueError("domain_action_overrides action must be an object")
                            action["action"] = deepcopy(dict(new_action))
                        else:
                            new_record = replacement["replace"]
                            if not isinstance(new_record, Mapping):
                                raise ValueError("domain_action_overrides replace must be an object")
                            action.clear()
                            action.update(deepcopy(dict(new_record)))
        for child in value.values():
            _replace_domain_actions(child, replacements)
    elif isinstance(value, list):
        for child in value:
            _replace_domain_actions(child, replacements)


def _replace_program_steps(value: Any, replacements: list[Any]) -> None:
    """Apply zone/step keyed replacements without evaluator-name dispatch."""
    if isinstance(value, dict):
        zones = value.get("zones")
        if isinstance(zones, list):
            for zone in zones:
                if not isinstance(zone, dict):
                    continue
                steps = zone.get("steps")
                if not isinstance(steps, list):
                    continue
                for step in steps:
                    if not isinstance(step, dict):
                        continue
                    for replacement in replacements:
                        if not isinstance(replacement, Mapping) or set(replacement) != {"match", "step"}:
                            raise ValueError("program_step_overrides entries require match and step")
                        match = replacement["match"]
                        new_step = replacement["step"]
                        if not isinstance(match, Mapping) or not isinstance(new_step, Mapping):
                            raise ValueError("program_step_overrides match/step must be objects")
                        zone_id = match.get("zone_id")
                        step_match = {key: expected for key, expected in match.items() if key != "zone_id"}
                        if zone.get("id") == zone_id and all(
                            step.get(key) == expected for key, expected in step_match.items()
                        ):
                            step.clear()
                            step.update(deepcopy(dict(new_step)))
        for child in value.values():
            _replace_program_steps(child, replacements)
    elif isinstance(value, list):
        for child in value:
            _replace_program_steps(child, replacements)


def resolve_semantic_profile(config: Mapping[str, Any], profile: str | None = None) -> tuple[dict[str, Any], str]:
    """Return one coherent activation contract for the requested profile."""
    if not isinstance(config, Mapping):
        raise ValueError("activation config must be an object")
    selected = semantic_profile(profile)
    result = deepcopy(dict(config))
    profiles = result.pop("semantic_profiles", None)
    if profiles is None:
        return result, selected
    if not isinstance(profiles, Mapping) or not profiles:
        raise ValueError("semantic_profiles must be a nonempty object")
    unknown = set(profiles) - SEMANTIC_PROFILES
    if unknown:
        raise ValueError(f"semantic_profiles has unknown profiles: {sorted(unknown)}")
    overlay = profiles.get(selected)
    if overlay is None:
        return result, selected
    if not isinstance(overlay, Mapping):
        raise ValueError(f"semantic_profiles.{selected} must be an object")
    overlay = deepcopy(dict(overlay))
    action_overrides = overlay.pop("special_action_overrides", None)
    domain_action_overrides = overlay.pop("domain_action_overrides", None)
    program_step_overrides = overlay.pop("program_step_overrides", None)
    if action_overrides is not None and not isinstance(action_overrides, Mapping):
        raise ValueError(f"semantic_profiles.{selected}.special_action_overrides must be an object")
    if domain_action_overrides is not None and not isinstance(domain_action_overrides, list):
        raise ValueError(f"semantic_profiles.{selected}.domain_action_overrides must be an array")
    if program_step_overrides is not None and not isinstance(program_step_overrides, list):
        raise ValueError(f"semantic_profiles.{selected}.program_step_overrides must be an array")
    _merge_contract(result, overlay)
    if action_overrides:
        symbolic = {
            key: dict(value) if isinstance(value, Mapping) else value for key, value in action_overrides.items()
        }
        needs_graph_representative = [
            key
            for key, value in symbolic.items()
            if isinstance(value, Mapping) and value.get("result") == "target_composite_class_representative"
        ]
        if needs_graph_representative:
            from ttpoly.spec.native_clamped_rational import (
                native_clamped_rational_representative,
            )

            representative = native_clamped_rational_representative(result.get("target_composite"))
            if representative is None:
                raise ValueError("symbolic special action has no supported typed target composite")
            for key in needs_graph_representative:
                symbolic[key] = {"result": "constant", "value": representative}
        action_overrides = symbolic
        _replace_special_actions(result, action_overrides)
    raw_class_actions = result.get("raw_class_actions")
    if isinstance(raw_class_actions, Mapping):
        symbolic_raw = {
            key: dict(value) if isinstance(value, Mapping) else value for key, value in raw_class_actions.items()
        }
        symbolic_classes = [
            key
            for key, value in symbolic_raw.items()
            if isinstance(value, Mapping)
            and value.get("kind") == "constant"
            and value.get("value") == "target_composite_class_representative"
        ]
        if symbolic_classes:
            from ttpoly.spec.native_clamped_rational import (
                native_clamped_rational_representative,
            )

            representative = native_clamped_rational_representative(result.get("target_composite"))
            if representative is None:
                raise ValueError("symbolic raw-class action has no supported typed target composite")
            for key in symbolic_classes:
                symbolic_raw[key] = {"kind": "constant", "value": representative}
            result["raw_class_actions"] = symbolic_raw
    if domain_action_overrides:
        _replace_domain_actions(result, domain_action_overrides)
    if program_step_overrides:
        _replace_program_steps(result, program_step_overrides)
    return result, selected


__all__ = [
    "DEFAULT_SEMANTIC_PROFILE",
    "SEMANTIC_PROFILES",
    "resolve_semantic_profile",
    "semantic_profile",
]
