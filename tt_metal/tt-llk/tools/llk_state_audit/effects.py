# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Normalized, source-traceable state-effect extraction for LLK functions.

Every extracted record maps one *parameter-to-state* relationship (or a fixed
effect with ``parameter.name == "-"``) discovered inside an ``_llk_*`` function
body, using the reviewed effect model for the state vocabulary. Parameter flow
is deliberately conservative: an effect only claims a parameter when that
parameter's identifier (or a local it provably flows through) appears in the
recovered value expression, or when it is forwarded positionally into a direct
``_llk_*`` call whose callee effect is already established.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .candidates import classify_candidates, load_candidate_model
from .inventory import (
    AuditModelError,
    _compact,
    _matching_delimiter,
    _skip_space,
    _split_top_level,
    scan_functions,
    scan_helpers,
)

_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
_ASSIGNMENT = re.compile(r"\b([A-Za-z_]\w*)\s*=(?!=)")
_INDEX_ASSIGNMENT_OPERATOR = re.compile(r"(?:<<|>>|[+\-*/%&|^])?=(?!=)")
_IF = re.compile(r"\bif\b")
_SKIP = object()


def build_effects(
    root: Path | str,
    effect_model: Path | str | None = None,
    *,
    _records: list[dict[str, Any]] | None = None,
    _model: dict[str, Any] | None = None,
    _helpers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return the JSON-ready normalized state-effect dataset for *root*."""
    if _records is None or _model is None:
        records, model = scan_functions(root, effect_model)
    else:
        records, model = _records, _model
    helpers = (
        _helpers
        if _helpers is not None
        else scan_helpers(root, effect_model, _model=model)
    )
    effects_model = model["effects"]
    restore_contracts = model.get("restore_contracts", [])
    candidate_model = load_candidate_model()

    all_records = records + helpers
    direct_effects = {
        id(record): (
            _direct_effects(record, effects_model)
            + _transient_datapath_summaries(record, candidate_model)
        )
        for record in all_records
    }
    propagated_effects = {
        id(record): list(direct_effects[id(record)]) for record in all_records
    }
    candidate_names = {record["name"] for record in all_records}

    for _ in range(len(all_records) + 1):
        overloads_by_key = _overload_map(all_records, propagated_effects)
        changed = False
        next_effects: dict[int, list[dict[str, Any]]] = {}
        for record in all_records:
            combined = direct_effects[id(record)] + _transitive_effects(
                record,
                overloads_by_key,
                candidate_names,
            )
            summarized = _dedupe(combined)
            summarized.sort(key=_sort_key)
            next_effects[id(record)] = summarized
            if _effect_identity_set(summarized) != _effect_identity_set(
                propagated_effects[id(record)]
            ):
                changed = True
        propagated_effects = next_effects
        if not changed:
            break
    else:
        raise AuditModelError("LLK/helper effect propagation did not converge")

    overloads_by_key = _overload_map(all_records, propagated_effects)
    for record in all_records:
        _transitive_effects(
            record,
            overloads_by_key,
            candidate_names,
            fail_on_ambiguous=True,
        )
    all_effects: list[dict[str, Any]] = []
    for record in records:
        all_effects.extend(propagated_effects[id(record)])

    _assign_restore_contracts(all_effects, restore_contracts)
    deduped = _dedupe(all_effects)
    deduped.sort(key=_sort_key)
    return {"schema_version": 2, "effects": deduped}


# --------------------------------------------------------------------------- #
# Direct effects
# --------------------------------------------------------------------------- #
def _transient_datapath_summaries(
    record: dict[str, Any],
    candidate_model: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates = [
        candidate
        for candidate in classify_candidates(
            [record],
            [],
            candidate_model,
        )
        if candidate["category"] == "transient_datapath"
    ]
    by_family: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        family = _transient_family(candidate["base"] or "")
        if family is not None:
            by_family.setdefault(family, []).append(candidate)

    summaries: list[dict[str, Any]] = []
    alias_of = record["canonical_target"] if record["lifecycle"] == "wrapper" else None
    for family, members in sorted(by_family.items()):
        members.sort(key=lambda item: (item["line"], item["token"]))
        first = members[0]
        deferred = sum(member["token"].startswith("TT_OP_") for member in members)
        immediate = len(members) - deferred
        activation = "deferred" if deferred and not immediate else "immediate"
        activation_note = (
            f"; immediate={immediate}, deferred={deferred}" if deferred else ""
        )
        sink = {
            "token": first["token"],
            "line": first["line"],
            "source_path": record["source"]["path"],
            "body_fingerprint": record["body_fingerprint"],
        }
        summaries.append(
            {
                "architecture": record["architecture"],
                "thread": record["thread"],
                "stage": record["stage"],
                "stability": record["stability_tier"],
                "function": record["name"],
                "alias_of": alias_of,
                "lifecycle": record["lifecycle"],
                "parameter": {
                    "name": "-",
                    "type": "-",
                    "kind": "fixed",
                },
                "condition": None,
                "condition_kind": None,
                "domain": "transient_datapath",
                "resource": family,
                "operation": "summarize",
                "value_expr": (
                    f"{len(members)} candidates; first={first['token']}"
                    f"{activation_note}"
                ),
                "persistence": "transient",
                "retention_contract": (
                    "Transient datapath contents remain only until consumed "
                    "or overwritten and do not create a restore contract."
                ),
                "activation": activation,
                "restore": None,
                "confidence": "high",
                "evidence": {
                    "kind": "direct",
                    "token": first["token"],
                    "line": first["line"],
                    "source_path": record["source"]["path"],
                    "body_fingerprint": record["body_fingerprint"],
                    "via": None,
                    "via_chain": [],
                    "sink": sink,
                },
            }
        )
    return summaries


def _transient_family(base: str) -> str | None:
    if base.startswith("MOV") or base in {"ZEROACC", "ZEROSRC"}:
        return "MOV"
    if base.startswith("ELW"):
        return "ELW"
    if base.startswith("MVMUL"):
        return "MATMUL"
    if base in {"GAPOOL", "GMPOOL"}:
        return "POOL"
    if base.startswith("UNPACR"):
        return "UNPACR_DATA_PLANE"
    if base.startswith("PACR"):
        return "PACR_DATA_PLANE"
    if base.startswith("SFP"):
        return "SFPU_DATA_PLANE"
    return None


def _direct_effects(
    record: dict[str, Any], effects_model: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    body = record["_body"]
    masked = record["_masked_body"]
    body_start_line = record["_body_start_line"]
    parameters = {parameter["name"]: parameter for parameter in record["_parameters"]}
    local_flow = _local_flow(masked, body, parameters)
    control = _control_blocks(masked, body)
    alias_of = record["canonical_target"] if record["lifecycle"] == "wrapper" else None

    results: list[dict[str, Any]] = []
    for spec in effects_model:
        if record["name"] in spec.get("skip_functions", []):
            continue
        if (
            spec.get("architectures") is not None
            and record["architecture"] not in spec["architectures"]
        ):
            continue
        extractor = spec.get("extractor") or {"kind": "token"}
        for match in re.finditer(spec["pattern"], masked):
            extracted = _extract(masked, body, match, extractor, spec)
            if extracted is _SKIP:
                continue
            resource, value_expr = extracted
            if value_expr in spec.get("skip_values", []):
                continue
            value_pattern = spec.get("value_pattern")
            if value_pattern is not None and (
                value_expr is None or re.search(value_pattern, value_expr) is None
            ):
                continue
            if resource in spec.get("skip_resources", []):
                continue
            resource_template = spec.get("resource_template")
            if resource_template is not None:
                resource = resource_template.format(resource=resource or "-")
            line = body_start_line + body.count("\n", 0, match.start())
            token = body[match.start() : match.end()]
            condition, condition_kind = _condition_at(control, match.start())
            flows = _flow_parameters(value_expr, parameters, local_flow)
            evidence = {
                "kind": "direct",
                "token": token,
                "line": line,
                "source_path": record["source"]["path"],
                "body_fingerprint": record["body_fingerprint"],
                "via": None,
                "via_chain": [],
                "sink": {
                    "token": token,
                    "line": line,
                    "source_path": record["source"]["path"],
                    "body_fingerprint": record["body_fingerprint"],
                },
            }
            common = {
                "architecture": record["architecture"],
                "thread": record["thread"],
                "stage": record["stage"],
                "stability": record["stability_tier"],
                "function": record["name"],
                "alias_of": alias_of,
                "lifecycle": record["lifecycle"],
                "condition": condition,
                "condition_kind": condition_kind,
                "domain": spec["domain"],
                "resource": resource or "-",
                "operation": spec["operation"],
                "value_expr": value_expr,
                "persistence": spec.get("persistence", "persistent"),
                "retention_contract": spec["retention"],
                "activation": _activation(token, spec),
                "restore": None,
            }
            if flows:
                for parameter, confidence in flows:
                    results.append(
                        {
                            **common,
                            "parameter": {
                                "name": parameter["name"],
                                "type": parameter["type"],
                                "kind": parameter["kind"],
                            },
                            "confidence": confidence,
                            "evidence": dict(evidence),
                        }
                    )
            else:
                results.append(
                    {
                        **common,
                        "parameter": {"name": "-", "type": "-", "kind": "fixed"},
                        "confidence": "high",
                        "evidence": dict(evidence),
                    }
                )
    return results


# --------------------------------------------------------------------------- #
# Transitive effects (one hop through direct _llk_* calls)
# --------------------------------------------------------------------------- #
def _transitive_effects(
    record: dict[str, Any],
    overloads_by_key: dict[
        tuple[str, str], list[tuple[dict[str, Any], list[dict[str, Any]]]]
    ],
    candidate_names: set[str],
    *,
    fail_on_ambiguous: bool = False,
) -> list[dict[str, Any]]:
    body = record["_body"]
    masked = record["_masked_body"]
    body_start_line = record["_body_start_line"]
    parameters = {parameter["name"]: parameter for parameter in record["_parameters"]}
    local_flow = _local_flow(masked, body, parameters)
    control = _control_blocks(masked, body)
    architecture = record["architecture"]
    alias_of = record["canonical_target"] if record["lifecycle"] == "wrapper" else None

    results: list[dict[str, Any]] = []
    for site in _call_sites(
        masked,
        body,
        record["name"],
        candidate_names,
    ):
        key = (architecture, site["qualified_name"])
        overloads = overloads_by_key.get(key, [])
        if not site["is_qualified"] and record["_namespace"]:
            local_key = (
                architecture,
                f'{record["_namespace"]}::{site["name"]}',
            )
            overloads = overloads_by_key.get(
                local_key,
                overloads_by_key.get((architecture, site["name"]), []),
            )
        compatible = [
            overload
            for overload in overloads
            if overload[0] is not record
            and _compatible_arity(
                overload[0]["_parameters"],
                site["template_args"],
                site["call_args"],
            )
        ]
        if len(compatible) > 1 and not site["is_qualified"]:
            same_source = [
                overload
                for overload in compatible
                if overload[0]["source"]["path"] == record["source"]["path"]
            ]
            if same_source:
                compatible = same_source
            elif record["stage"] in {"unpack", "math", "pack"}:
                stage_tokens = {
                    "unpack": ("unpack",),
                    "math": ("math", "sfpu"),
                    "pack": ("pack",),
                }[record["stage"]]
                same_stage = [
                    overload
                    for overload in compatible
                    if any(
                        token in overload[0]["source"]["path"].lower()
                        for token in stage_tokens
                    )
                ]
                if same_stage:
                    compatible = same_stage
            owner_type = record.get("_owner_type")
            if owner_type and len(compatible) > 1:
                same_owner = [
                    overload
                    for overload in compatible
                    if overload[0].get("_owner_type") == owner_type
                ]
                if same_owner:
                    compatible = same_owner
        if len(compatible) > 1:
            compatible = _best_typed_overloads(
                compatible,
                site["call_args"],
                record["_parameters"],
            )
        if not compatible and site["is_qualified"]:
            compatible_by_final_name = [
                overload
                for overload in overloads_by_key.get(
                    (architecture, site["name"]),
                    [],
                )
                if overload[0] is not record
                and _compatible_arity(
                    overload[0]["_parameters"],
                    site["template_args"],
                    site["call_args"],
                )
            ]
            if len(compatible_by_final_name) == 1:
                compatible = compatible_by_final_name
            elif fail_on_ambiguous and any(
                effects for _, effects in compatible_by_final_name
            ):
                _raise_call_resolution_error(record, site, "unresolved")
        if len(compatible) != 1:
            if (
                fail_on_ambiguous
                and len(compatible) > 1
                and any(effects for _, effects in compatible)
            ):
                _raise_call_resolution_error(record, site, "ambiguous")
            continue
        callee_record, callee_effects = compatible[0]
        if not callee_effects:
            continue
        argmap = _positional_argmap(
            callee_record["_parameters"], site["template_args"], site["call_args"]
        )
        line = body_start_line + body.count("\n", 0, site["pos"])
        caller_condition, caller_condition_kind = _condition_at(
            control,
            site["pos"],
        )
        for effect in callee_effects:
            callee_condition = _substitute_expression(
                effect["condition"],
                argmap,
            )
            truth = _condition_truth(callee_condition)
            if truth is False:
                continue
            if truth is True:
                callee_condition = None
            condition = _combine_conditions(
                caller_condition,
                callee_condition,
            )
            condition_kind = (
                caller_condition_kind
                if callee_condition is None
                else effect["condition_kind"] or caller_condition_kind
            )
            parameter_name = effect["parameter"]["name"]
            evidence = {
                "kind": "transitive",
                "token": site["qualified_name"],
                "line": line,
                "source_path": record["source"]["path"],
                "body_fingerprint": record["body_fingerprint"],
                "via": site["name"],
                "via_chain": [
                    {
                        "function": site["name"],
                        "line": line,
                        "source_path": record["source"]["path"],
                        "body_fingerprint": record["body_fingerprint"],
                    },
                    *effect["evidence"].get("via_chain", []),
                ],
                "sink": effect["evidence"].get(
                    "sink",
                    {
                        "token": effect["evidence"]["token"],
                        "line": effect["evidence"]["line"],
                        "source_path": effect["evidence"]["source_path"],
                        "body_fingerprint": effect["evidence"]["body_fingerprint"],
                    },
                ),
            }
            common = {
                "architecture": architecture,
                "thread": record["thread"],
                "stage": record["stage"],
                "stability": record["stability_tier"],
                "function": record["name"],
                "alias_of": alias_of,
                "lifecycle": record["lifecycle"],
                "condition": condition,
                "condition_kind": condition_kind,
                "domain": effect["domain"],
                "resource": _substitute_expression(
                    effect["resource"],
                    argmap,
                ),
                "operation": effect["operation"],
                "persistence": effect["persistence"],
                "retention_contract": effect["retention_contract"],
                "activation": effect["activation"],
                "restore": None,
            }
            if parameter_name == "-":
                results.append(
                    {
                        **common,
                        "parameter": {"name": "-", "type": "-", "kind": "fixed"},
                        "value_expr": _substitute_expression(
                            effect["value_expr"],
                            argmap,
                        ),
                        "confidence": "medium",
                        "evidence": dict(evidence),
                    }
                )
                continue
            caller_expr = argmap.get(parameter_name)
            if caller_expr is None:
                continue
            flows = _flow_parameters(caller_expr, parameters, local_flow)
            if not flows:
                results.append(
                    {
                        **common,
                        "parameter": {"name": "-", "type": "-", "kind": "fixed"},
                        "value_expr": caller_expr,
                        "confidence": "medium",
                        "evidence": dict(evidence),
                    }
                )
                continue
            for parameter, confidence in flows:
                results.append(
                    {
                        **common,
                        "parameter": {
                            "name": parameter["name"],
                            "type": parameter["type"],
                            "kind": parameter["kind"],
                        },
                        "value_expr": caller_expr,
                        "confidence": "low" if confidence == "medium" else "medium",
                        "evidence": dict(evidence),
                    }
                )
    return results


# --------------------------------------------------------------------------- #
# Activation classification
# --------------------------------------------------------------------------- #
def _activation(token: str, spec: dict[str, Any]) -> str:
    """Derive when a normalized effect activates.

    ``immediate`` covers ``TTI_*`` instructions and direct MMIO/software writes;
    ``deferred`` covers ``TT_OP_*`` opcode construction whose effect activates
    when the enclosing MOP/replay program executes; ``programming`` covers
    MOP/replay program configuration itself. A reviewed spec may pin activation
    explicitly, but the token prefix is authoritative so a broad regex that
    matches both ``TTI_`` and ``TT_OP_`` is never conflated into one activation.
    """
    if token.startswith("TT_OP_"):
        return "deferred"
    explicit = spec.get("activation")
    if explicit is not None:
        return explicit
    if spec["domain"] == "mop_replay":
        return "programming"
    return "immediate"


# --------------------------------------------------------------------------- #
# Extraction helpers
# --------------------------------------------------------------------------- #
def _extract(
    masked: str,
    body: str,
    match: re.Match[str],
    extractor: dict[str, Any],
    spec: dict[str, Any],
):
    kind = extractor.get("kind", "token")
    if kind == "token":
        resource = _resource_literal(extractor.get("resource"))
        return (
            (
                resource
                if resource is not None
                else _compact(body[match.start() : match.end()])
            ),
            None,
        )
    if kind == "regex_group":
        resource_group = extractor.get("resource_group", "resource")
        value_group = extractor.get("value_group")
        try:
            resource = match.group(resource_group)
            value = match.group(value_group) if value_group else None
        except (IndexError, KeyError):
            return _SKIP
        return (_compact(resource), _compact(value) if value else None)
    if kind == "assign_index":
        return _extract_assign_index(masked, body, match, extractor)
    if kind == "adc_call":
        extracted = _extract_call(
            masked,
            body,
            match,
            extractor,
            template=False,
        )
        if extracted is _SKIP:
            return _SKIP
        resource, value = extracted
        return (_normalize_adc_resource(resource), value)
    if kind == "semaphore_call":
        extracted = _extract_call(
            masked,
            body,
            match,
            extractor,
            template=True,
        )
        if extracted is _SKIP:
            return _SKIP
        resource, value = extracted
        return (_normalize_semaphore_resource(resource), value)
    if kind == "sfpu_config_call":
        extracted = _extract_call(
            masked,
            body,
            match,
            extractor,
            template=False,
        )
        if extracted is _SKIP:
            return _SKIP
        resource, value = extracted
        return (
            f"SFPU_CONFIG[{resource}]" if resource else "SFPU_CONFIG",
            value,
        )
    return _extract_call(
        masked, body, match, extractor, template=(kind == "template_call")
    )


def _extract_call(
    masked: str,
    body: str,
    match: re.Match[str],
    extractor: dict[str, Any],
    *,
    template: bool,
):
    position = _skip_space(masked, match.end())
    template_args: list[str] = []
    if template and position < len(masked) and masked[position] == "<":
        end = _matching_delimiter(masked, position, "<", ">")
        if end is None:
            return _SKIP
        template_args = [
            _compact(item) for item in _split_top_level(body[position + 1 : end])
        ]
        position = _skip_space(masked, end + 1)
    if position >= len(masked) or masked[position] != "(":
        return _SKIP
    end = _matching_delimiter(masked, position, "(", ")")
    if end is None:
        return _SKIP
    call_args = (
        [_compact(item) for item in _split_top_level(body[position + 1 : end])]
        if end > position + 1
        else []
    )
    resource = _resolve(
        extractor.get("resource"), template_args, call_args, index=None, rhs=None
    )
    value = _resolve(
        extractor.get("value"), template_args, call_args, index=None, rhs=None
    )
    return (resource, value)


def _extract_assign_index(
    masked: str, body: str, match: re.Match[str], extractor: dict[str, Any]
):
    bracket = match.end() - 1
    if bracket < 0 or masked[bracket] != "[":
        return _SKIP
    close = _matching_delimiter(masked, bracket, "[", "]")
    if close is None:
        return _SKIP
    index_expr = _compact(body[bracket + 1 : close])
    position = _skip_space(masked, close + 1)
    operator_match = _INDEX_ASSIGNMENT_OPERATOR.match(masked, position)
    if operator_match is None:
        return _SKIP  # a read or comparison, not a persistent write
    operator = body[position : operator_match.end()]
    semicolon = _statement_end(masked, operator_match.end())
    rhs_value = _compact(body[operator_match.end() : semicolon])
    rhs = rhs_value if operator == "=" else f"{operator} {rhs_value}"
    resource = _resolve(extractor.get("resource"), [], [], index=index_expr, rhs=rhs)
    value = _resolve(extractor.get("value"), [], [], index=index_expr, rhs=rhs)
    return (resource, value)


def _resolve(
    spec: str | None,
    template_args: list[str],
    call_args: list[str],
    *,
    index: str | None,
    rhs: str | None,
):
    if spec is None:
        return None
    if spec == "index":
        return index
    if spec == "rhs":
        return rhs
    if spec.startswith("literal:"):
        return spec[len("literal:") :]
    kind, _, raw = spec.partition(":")
    if kind == "args":
        try:
            positions = [int(value) for value in raw.split(",")]
        except ValueError:
            return None
        values = [
            call_args[position]
            for position in positions
            if 0 <= position < len(call_args)
        ]
        return ", ".join(values) if values else None
    try:
        position = int(raw)
    except ValueError:
        return None
    source = template_args if kind == "template" else call_args if kind == "arg" else []
    return source[position] if 0 <= position < len(source) else None


def _resource_literal(spec: str | None) -> str | None:
    if spec is None:
        return None
    if spec.startswith("literal:"):
        return spec[len("literal:") :]
    return spec


def _normalize_adc_resource(resource: str | None) -> str | None:
    return {
        "0b001": "p_setadc::UNP_A",
        "0b010": "p_setadc::UNP_B",
        "0b011": "p_setadc::UNP_AB",
        "0b11": "p_setadc::UNP_AB",
        "0b100": "p_setadc::PAC",
        "1": "p_setadc::UNP_A",
        "2": "p_setadc::UNP_B",
        "3": "p_setadc::UNP_AB",
        "4": "p_setadc::PAC",
    }.get(resource or "", resource)


def _normalize_semaphore_resource(
    resource: str | None,
) -> str | None:
    if resource is None:
        return None
    numeric = resource.removeprefix("0b")
    return f"semaphore[{resource}]" if numeric.isdigit() else resource


# --------------------------------------------------------------------------- #
# Parameter-flow helpers
# --------------------------------------------------------------------------- #
def _flow_parameters(
    value_expr: str | None,
    parameters: dict[str, dict[str, Any]],
    local_flow: dict[str, set[str]],
):
    if not value_expr:
        return []
    identifiers = set(_IDENTIFIER.findall(value_expr))
    best: dict[str, str] = {}
    for identifier in identifiers:
        if identifier in parameters:
            best[identifier] = "high"
        elif identifier in local_flow:
            for source in local_flow[identifier]:
                if best.get(source) != "high":
                    best[source] = "medium"
    ordered = sorted(
        best.items(),
        key=lambda item: (
            parameters[item[0]].get("position", 0) if item[0] in parameters else 0
        ),
    )
    return [
        (parameters[name], confidence)
        for name, confidence in ordered
        if name in parameters
    ]


def _substitute_expression(
    expression: str | None,
    arguments: dict[str, str],
) -> str | None:
    if expression is None:
        return None
    result = expression
    for name, value in sorted(
        arguments.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        result = re.sub(rf"\b{re.escape(name)}\b", value, result)
    return _compact(result)


def _condition_truth(condition: str | None) -> bool | None:
    if condition is None:
        return True
    expression = condition.strip()
    if expression.startswith("!(") and expression.endswith(")"):
        inner = _condition_truth(expression[2:-1])
        return None if inner is None else not inner
    parts = [part.strip() for part in expression.split("&&")]
    if len(parts) > 1:
        values = [_condition_truth(part) for part in parts]
        if any(value is False for value in values):
            return False
        return True if all(value is True for value in values) else None
    match = re.fullmatch(r"(.+?)\s*(==|!=)\s*(.+)", expression)
    if match is None:
        return None
    left, operator, right = (
        _compact(match.group(1)),
        match.group(2),
        _compact(match.group(3)),
    )
    if not (left in {"true", "false"} or "::" in left or left.isdigit()) or not (
        right in {"true", "false"} or "::" in right or right.isdigit()
    ):
        return None
    equal = left == right
    return equal if operator == "==" else not equal


def _combine_conditions(
    caller: str | None,
    callee: str | None,
) -> str | None:
    if caller and callee:
        return f"{caller} && {callee}"
    return caller or callee


def _local_flow(
    masked: str, body: str, parameters: dict[str, dict[str, Any]]
) -> dict[str, set[str]]:
    flow: dict[str, set[str]] = {}
    for match in _ASSIGNMENT.finditer(masked):
        name = match.group(1)
        if name in parameters:
            continue
        end = _statement_end(masked, match.end())
        rhs = body[match.end() : end]
        sources: set[str] = set()
        for identifier in set(_IDENTIFIER.findall(rhs)):
            if identifier in parameters:
                sources.add(identifier)
            elif identifier in flow:
                sources |= flow[identifier]
        if sources:
            flow.setdefault(name, set()).update(sources)
    return flow


# --------------------------------------------------------------------------- #
# Condition + call-site helpers
# --------------------------------------------------------------------------- #
def _control_blocks(masked: str, body: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for match in _IF.finditer(masked):
        position = _skip_space(masked, match.end())
        constexpr = False
        if masked[position : position + 9] == "constexpr":
            constexpr = True
            position = _skip_space(masked, position + 9)
        if position >= len(masked) or masked[position] != "(":
            continue
        close = _matching_delimiter(masked, position, "(", ")")
        if close is None:
            continue
        condition = _compact(body[position + 1 : close])
        after = _skip_space(masked, close + 1)
        if after < len(masked) and masked[after] == "{":
            block_close = _matching_delimiter(masked, after, "{", "}")
            if block_close is None:
                continue
            start, end = after + 1, block_close
        else:
            start, end = close + 1, _statement_end(masked, close + 1)
        blocks.append(
            {"start": start, "end": end, "condition": condition, "constexpr": constexpr}
        )
        else_position = _skip_space(masked, end + 1)
        if masked[else_position : else_position + 4] != "else":
            continue
        else_body = _skip_space(masked, else_position + 4)
        if masked[else_body : else_body + 2] == "if":
            else_start, else_end = else_body, _if_chain_end(masked, else_body)
        elif else_body < len(masked) and masked[else_body] == "{":
            else_close = _matching_delimiter(masked, else_body, "{", "}")
            if else_close is None:
                continue
            else_start, else_end = else_body + 1, else_close
        else:
            else_start, else_end = else_body, _statement_end(masked, else_body)
        blocks.append(
            {
                "start": else_start,
                "end": else_end,
                "condition": f"!({condition})",
                "constexpr": constexpr,
            }
        )
    return blocks


def _condition_at(blocks: list[dict[str, Any]], position: int):
    enclosing = sorted(
        (block for block in blocks if block["start"] <= position < block["end"]),
        key=lambda block: block["start"],
    )
    if not enclosing:
        return None, None
    condition = " && ".join(block["condition"] for block in enclosing)
    return condition, (
        "compile_time" if all(block["constexpr"] for block in enclosing) else "runtime"
    )


def _if_chain_end(masked: str, if_position: int) -> int:
    condition_start = _skip_space(masked, if_position + 2)
    condition_end = _matching_delimiter(masked, condition_start, "(", ")")
    if condition_end is None:
        return if_position
    body_start = _skip_space(masked, condition_end + 1)
    if body_start < len(masked) and masked[body_start] == "{":
        body_end = _matching_delimiter(masked, body_start, "{", "}")
        if body_end is None:
            return len(masked)
    else:
        body_end = _statement_end(masked, body_start)
    else_position = _skip_space(masked, body_end + 1)
    if masked[else_position : else_position + 4] != "else":
        return body_end + 1
    else_body = _skip_space(masked, else_position + 4)
    if masked[else_body : else_body + 2] == "if":
        return _if_chain_end(masked, else_body)
    if else_body < len(masked) and masked[else_body] == "{":
        close = _matching_delimiter(masked, else_body, "{", "}")
        return len(masked) if close is None else close + 1
    return _statement_end(masked, else_body) + 1


def _call_sites(
    masked: str,
    body: str,
    current_name: str,
    candidate_names: set[str],
) -> list[dict[str, Any]]:
    sites: list[dict[str, Any]] = []
    for match in re.finditer(
        r"(?<![A-Za-z0-9_])(?:[A-Za-z_]\w*::)*[A-Za-z_]\w*",
        masked,
    ):
        if match.start() > 0 and masked[match.start() - 1] in {".", ">"}:
            continue
        qualified_name = match.group(0)
        name = qualified_name.rsplit("::", 1)[-1]
        if name not in candidate_names:
            continue
        position = _skip_space(masked, match.end())
        template_args: list[str] = []
        if position < len(masked) and masked[position] == "<":
            end = _matching_delimiter(masked, position, "<", ">")
            if end is None:
                continue
            template_args = [
                _compact(item) for item in _split_top_level(body[position + 1 : end])
            ]
            position = _skip_space(masked, end + 1)
        if position >= len(masked) or masked[position] != "(":
            continue
        end = _matching_delimiter(masked, position, "(", ")")
        if end is None:
            continue
        call_args = (
            [_compact(item) for item in _split_top_level(body[position + 1 : end])]
            if end > position + 1
            else []
        )
        sites.append(
            {
                "name": name,
                "qualified_name": qualified_name,
                "is_qualified": "::" in qualified_name,
                "template_args": template_args,
                "call_args": call_args,
                "pos": match.start(),
            }
        )
    return sites


def _positional_argmap(
    parameters: list[dict[str, Any]], template_args: list[str], call_args: list[str]
) -> dict[str, str]:
    argmap: dict[str, str] = {}
    template_params = [
        parameter for parameter in parameters if parameter["kind"] == "template"
    ]
    runtime_params = [
        parameter for parameter in parameters if parameter["kind"] == "runtime"
    ]
    for parameter, argument in zip(template_params, template_args):
        argmap[parameter["name"]] = argument
    for parameter, argument in zip(runtime_params, call_args):
        argmap[parameter["name"]] = argument
    for parameter in template_params[len(template_args) :]:
        if parameter.get("default") is not None:
            argmap[parameter["name"]] = parameter["default"]
    for parameter in runtime_params[len(call_args) :]:
        if parameter.get("default") is not None:
            argmap[parameter["name"]] = parameter["default"]
    return argmap


def _compatible_arity(
    parameters: list[dict[str, Any]], template_args: list[str], call_args: list[str]
) -> bool:
    template_params = [
        parameter for parameter in parameters if parameter["kind"] == "template"
    ]
    runtime_params = [
        parameter for parameter in parameters if parameter["kind"] == "runtime"
    ]
    return _arity_matches(template_params, template_args) and _arity_matches(
        runtime_params, call_args
    )


def _arity_matches(parameters: list[dict[str, Any]], arguments: list[str]) -> bool:
    required = sum(parameter.get("default") is None for parameter in parameters)
    return required <= len(arguments) <= len(parameters)


def _best_typed_overloads(
    overloads: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    call_args: list[str],
    caller_parameters: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Prefer an overload whose parameter type matches a scoped enum argument."""
    caller_by_name = {
        parameter["name"]: parameter
        for parameter in caller_parameters
        if parameter["kind"] == "runtime"
    }
    scored: list[tuple[int, tuple[dict[str, Any], list[dict[str, Any]]]]] = []
    for overload in overloads:
        parameters = overload[0]["_parameters"]
        runtime_params = [
            parameter for parameter in parameters if parameter["kind"] == "runtime"
        ]
        score = 0
        for parameter, argument in zip(runtime_params, call_args):
            caller_parameter = caller_by_name.get(argument)
            if caller_parameter is not None:
                caller_type_tokens = set(
                    _IDENTIFIER.findall(caller_parameter["type"])
                ) - {caller_parameter["name"], "const", "volatile"}
                candidate_type_tokens = set(_IDENTIFIER.findall(parameter["type"])) - {
                    parameter["name"],
                    "const",
                    "volatile",
                }
                if caller_type_tokens & candidate_type_tokens:
                    score += 2
            scoped_values = re.findall(
                r"\b((?:[A-Za-z_]\w*::)+[A-Za-z_]\w*)::[A-Za-z_]\w*\b",
                argument,
            )
            if any(value in parameter["type"] for value in scoped_values):
                score += 1
        scored.append((score, overload))
    best = max(score for score, _ in scored)
    return [overload for score, overload in scored if score == best]


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
def _statement_end(text: str, position: int) -> int:
    depth = 0
    openers = {"(": ")", "[": "]", "{": "}"}
    closers = {")", "]", "}"}
    while position < len(text):
        token = text[position]
        if token in openers:
            depth += 1
        elif token in closers:
            depth = max(0, depth - 1)
        elif token == ";" and depth == 0:
            return position
        position += 1
    return len(text)


def _overload_map(
    records: list[dict[str, Any]],
    effects: dict[int, list[dict[str, Any]]],
) -> dict[
    tuple[str, str],
    list[tuple[dict[str, Any], list[dict[str, Any]]]],
]:
    overloads: dict[
        tuple[str, str],
        list[tuple[dict[str, Any], list[dict[str, Any]]]],
    ] = {}
    for record in records:
        item = (record, effects[id(record)])
        overloads.setdefault(
            (record["architecture"], record["name"]),
            [],
        ).append(item)
        if record["_qualified_name"] != record["name"]:
            overloads.setdefault(
                (record["architecture"], record["_qualified_name"]),
                [],
            ).append(item)
    return overloads


def _raise_call_resolution_error(
    record: dict[str, Any],
    site: dict[str, Any],
    reason: str,
) -> None:
    line = record["_body_start_line"] + record["_body"].count(
        "\n",
        0,
        site["pos"],
    )
    raise AuditModelError(
        f'{record["source"]["path"]}:{line}: {reason} state-carrying call '
        f'{site["qualified_name"]}'
    )


def _effect_identity_set(
    effects: list[dict[str, Any]],
) -> set[tuple[Any, ...]]:
    return {
        (
            effect["domain"],
            effect["resource"],
            effect["operation"],
            effect["parameter"]["name"],
            effect["condition"],
            effect["value_expr"],
        )
        for effect in effects
    }


def _assign_restore_contracts(
    effects: list[dict[str, Any]], contracts: list[dict[str, Any]]
) -> None:
    """Attach only explicitly reviewed checked-in restore contracts."""
    by_function = {
        (
            contract["architecture"],
            contract["function"],
            contract["source"]["body_fingerprint"],
        ): contract
        for contract in contracts
    }
    for effect in effects:
        if effect["persistence"] == "transient":
            effect["restore"] = None
            continue
        contract = by_function.get(
            (
                effect["architecture"],
                effect["function"],
                effect["evidence"]["body_fingerprint"],
            )
        )
        selector = contract.get("effect_selector") if contract is not None else None
        matches_selector = contract is not None and (
            selector is None
            or all(
                (
                    effect.get(key) in value
                    if isinstance(value, list)
                    else effect.get(key) == value
                )
                for key, value in selector.items()
            )
        )
        effect["restore"] = dict(contract) if matches_selector else None


def _dedupe(effects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple] = set()
    unique: list[dict[str, Any]] = []
    for effect in effects:
        signature = (
            effect["architecture"],
            effect["function"],
            effect["domain"],
            effect["resource"],
            effect["operation"],
            effect["parameter"]["name"],
            effect["condition"],
            effect["evidence"]["kind"],
            effect["evidence"]["line"],
            effect["evidence"]["via"],
            effect["value_expr"],
        )
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(effect)
    return unique


def _sort_key(effect: dict[str, Any]) -> tuple:
    return (
        effect["architecture"],
        effect["evidence"]["source_path"],
        effect["evidence"]["line"],
        effect["function"],
        effect["domain"],
        effect["resource"],
        effect["operation"],
        effect["parameter"]["name"],
        effect["evidence"]["kind"],
        effect["evidence"]["via"] or "",
    )
