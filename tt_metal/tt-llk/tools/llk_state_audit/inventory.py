"""Source scanner used to inventory LLK functions without compiling them."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

_ARCHITECTURES = ("wormhole_b0", "blackhole", "quasar")
_DOMAINS = {
    "cfg_register",
    "counter",
    "addr_mod",
    "mop_replay",
    "dvalid_semaphore",
    "destination",
    "gpr",
    "debug_register",
    "software_shadow",
    "config_context",
    "sfpu_config",
    "sfpu_lreg",
    "dest_data",
    "src_data",
    "mutex",
    "mailbox",
    "transient_datapath",
}
_ACTIVATIONS = {"immediate", "deferred", "programming"}
_PERSISTENCE = {"persistent", "transient"}
_REQUIRED_EFFECT_KEYS = {"pattern", "domain", "operation", "retention"}
_OPTIONAL_EFFECT_KEYS = {
    "extractor",
    "persistence",
    "note",
    "resource_default",
    "skip_values",
    "skip_resources",
    "architectures",
    "activation",
    "skip_functions",
    "value_pattern",
    "resource_template",
}
_EXTRACTOR_KINDS = {
    "template_call",
    "call",
    "assign_index",
    "token",
    "regex_group",
    "adc_call",
    "semaphore_call",
    "sfpu_config_call",
}
_RESTORE_CONTRACT_KINDS = {
    "snapshot_restore",
    "canonical_reset",
    "no_op_transient",
    "retained_until_reconfigure",
}
_RESTORE_CONTRACT_KEYS = {
    "architecture",
    "function",
    "kind",
    "owner",
    "pair",
    "owner_source",
    "pair_source",
    "rationale",
    "source",
}
_OPTIONAL_RESTORE_CONTRACT_KEYS = {"effect_selector"}
_RESTORE_SELECTOR_KEYS = {"domain", "resource", "operation"}
_RESTORE_SOURCE_KEYS = {"path", "line", "token", "body_fingerprint"}
_FUNCTION_NAME = re.compile(r"\b_llk_[A-Za-z0-9_]+\b")
_ANY_FUNCTION_NAME = re.compile(r"\b[A-Za-z_]\w*\b")
_NON_FUNCTION_NAMES = {
    "if",
    "for",
    "while",
    "switch",
    "return",
    "sizeof",
    "static_assert",
    "constexpr",
    "inline",
    "static",
    "template",
    "else",
    "do",
    "alignof",
    "decltype",
    "requires",
    "noexcept",
    "__attribute__",
    "static_cast",
    "reinterpret_cast",
    "const_cast",
    "dynamic_cast",
    "void",
    "bool",
    "char",
    "int",
    "float",
    "double",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "int8_t",
    "int16_t",
    "int32_t",
}


class AuditModelError(ValueError):
    """Raised when a checked-in effect model is malformed."""


def load_effect_model(
    path: Path | str | None = None,
    *,
    root: Path | str | None = None,
    _validate_source_contracts: bool = True,
) -> dict[str, Any]:
    """Load and validate an explicit effect model; never infer unknown semantics."""
    default_model = path is None
    if default_model:
        path = Path(__file__).with_name("state_effects.json")
    try:
        model = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditModelError(f"cannot load state-effect model: {error}") from error
    if not isinstance(model, dict) or model.get("schema_version") != 1:
        raise AuditModelError("state-effect model requires schema_version 1")
    effects = model.get("effects")
    if not isinstance(effects, list) or not effects:
        raise AuditModelError("state-effect model requires a non-empty effects list")
    for effect in effects:
        if not isinstance(effect, dict):
            raise AuditModelError("each effect must be an object")
        keys = set(effect)
        if not _REQUIRED_EFFECT_KEYS.issubset(keys) or not (
            keys - _REQUIRED_EFFECT_KEYS
        ).issubset(_OPTIONAL_EFFECT_KEYS):
            raise AuditModelError(
                "each effect requires pattern, domain, operation, and retention (plus only known optional keys)"
            )
        if not isinstance(effect["pattern"], str) or not effect["pattern"]:
            raise AuditModelError("effect pattern must be a non-empty string")
        if (
            effect["domain"] not in _DOMAINS
            or not isinstance(effect["operation"], str)
            or not effect["operation"]
        ):
            raise AuditModelError("effect has an unknown domain or empty operation")
        if not isinstance(effect["retention"], str) or not effect["retention"].strip():
            raise AuditModelError(
                "effect retention must be a non-empty reviewed string"
            )
        if effect.get("persistence", "persistent") not in _PERSISTENCE:
            raise AuditModelError("effect persistence must be persistent or transient")
        try:
            re.compile(effect["pattern"])
        except re.error as error:
            raise AuditModelError(
                f"invalid effect pattern {effect['pattern']!r}: {error}"
            ) from error
        extractor = effect.get("extractor")
        if extractor is not None:
            if (
                not isinstance(extractor, dict)
                or extractor.get("kind") not in _EXTRACTOR_KINDS
            ):
                raise AuditModelError("effect extractor must specify a known kind")
        skip_values = effect.get("skip_values")
        if skip_values is not None and (
            not isinstance(skip_values, list)
            or not all(isinstance(value, str) for value in skip_values)
        ):
            raise AuditModelError("effect skip_values must be a list of strings")
        skip_functions = effect.get("skip_functions")
        if skip_functions is not None and (
            not isinstance(skip_functions, list)
            or not all(isinstance(value, str) for value in skip_functions)
        ):
            raise AuditModelError("effect skip_functions must be a list of strings")
        value_pattern = effect.get("value_pattern")
        if value_pattern is not None:
            if not isinstance(value_pattern, str) or not value_pattern:
                raise AuditModelError("effect value_pattern must be a non-empty string")
            try:
                re.compile(value_pattern)
            except re.error as error:
                raise AuditModelError(
                    f"invalid effect value_pattern {value_pattern!r}: {error}"
                ) from error
        resource_template = effect.get("resource_template")
        if resource_template is not None and (
            not isinstance(resource_template, str)
            or "{resource}" not in resource_template
        ):
            raise AuditModelError("effect resource_template must contain {resource}")
        skip_resources = effect.get("skip_resources")
        if skip_resources is not None and (
            not isinstance(skip_resources, list)
            or not all(isinstance(value, str) for value in skip_resources)
        ):
            raise AuditModelError("effect skip_resources must be a list of strings")
        architectures = effect.get("architectures")
        if architectures is not None and (
            not isinstance(architectures, list)
            or not architectures
            or not set(architectures).issubset(_ARCHITECTURES)
        ):
            raise AuditModelError(
                "effect architectures must name supported architectures"
            )
        activation = effect.get("activation")
        if activation is not None and activation not in _ACTIVATIONS:
            raise AuditModelError(
                "effect activation must be immediate, deferred, or programming"
            )
    contracts = model.get("restore_contracts", [])
    if not isinstance(contracts, list):
        raise AuditModelError("restore_contracts must be a list")
    identities: set[tuple[str, str]] = set()
    for contract in contracts:
        if (
            not isinstance(contract, dict)
            or not _RESTORE_CONTRACT_KEYS.issubset(contract)
            or not (set(contract) - _RESTORE_CONTRACT_KEYS).issubset(
                _OPTIONAL_RESTORE_CONTRACT_KEYS
            )
        ):
            raise AuditModelError(
                "each restore contract requires architecture, function, kind, owner, pair, rationale, and source"
            )
        if contract["architecture"] not in _ARCHITECTURES:
            raise AuditModelError("restore contract has an unknown architecture")
        if not isinstance(contract["function"], str) or not contract[
            "function"
        ].startswith("_llk_"):
            raise AuditModelError("restore contract function must name an _llk_ family")
        if contract["kind"] not in _RESTORE_CONTRACT_KINDS:
            raise AuditModelError("restore contract has an unknown kind")
        for key in ("owner", "pair"):
            if contract[key] is not None and (
                not isinstance(contract[key], str)
                or not contract[key].startswith("_llk_")
            ):
                raise AuditModelError(
                    f"restore contract {key} must be null or an _llk_ function"
                )
        if (
            not isinstance(contract["rationale"], str)
            or not contract["rationale"].strip()
        ):
            raise AuditModelError("restore contract requires a source rationale")
        selector = contract.get("effect_selector")
        if selector is not None and (
            not isinstance(selector, dict)
            or not selector
            or not set(selector).issubset(_RESTORE_SELECTOR_KEYS)
            or not all(
                (isinstance(value, str) and value)
                or (
                    isinstance(value, list)
                    and value
                    and all(isinstance(item, str) and item for item in value)
                )
                for value in selector.values()
            )
        ):
            raise AuditModelError(
                "restore contract effect_selector must contain reviewed string fields"
            )
        if not _valid_restore_source(contract["source"]):
            raise AuditModelError(
                "restore contract source requires path, positive line, token, and body fingerprint"
            )
        for reference in ("owner", "pair"):
            evidence = contract[f"{reference}_source"]
            if contract[reference] is None:
                if evidence is not None:
                    raise AuditModelError(
                        f"restore contract null {reference} requires null {reference}_source"
                    )
            elif not _valid_restore_source(evidence):
                raise AuditModelError(
                    f"restore contract {reference} requires immutable {reference}_source evidence"
                )
        identity = (contract["architecture"], contract["function"])
        if identity in identities:
            raise AuditModelError("restore contract identities must be unique")
        identities.add(identity)
    if contracts and _validate_source_contracts:
        source_root = (
            Path(root).resolve()
            if root is not None
            else (
                Path(__file__).resolve().parents[2]
                if default_model
                else Path(path).resolve().parent
            )
        )
        _validate_restore_contract_sources(source_root, contracts)
    return model


def _valid_restore_source(source: Any) -> bool:
    return (
        isinstance(source, dict)
        and set(source) == _RESTORE_SOURCE_KEYS
        and isinstance(source["path"], str)
        and bool(source["path"])
        and isinstance(source["line"], int)
        and source["line"] >= 1
        and isinstance(source["token"], str)
        and bool(source["token"])
        and isinstance(source["body_fingerprint"], str)
        and re.fullmatch(r"[0-9a-f]{64}", source["body_fingerprint"]) is not None
    )


def scan_functions(
    root: Path | str, effect_model: Path | str | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return richly annotated in-memory function records plus the loaded model.

    Records include private ``_body`` / ``_masked_body`` / ``_body_start_line`` /
    ``_parameters`` fields that downstream analyzers (effect extraction and
    classification) need but which never appear in the JSON-ready inventory.
    """
    root_path = Path(root).resolve()
    complete_source_tree = all(
        (root_path / f"tt_llk_{architecture}").is_dir()
        for architecture in _ARCHITECTURES
    )
    model = load_effect_model(
        effect_model,
        root=root_path,
        _validate_source_contracts=effect_model is not None or complete_source_tree,
    )
    records: list[dict[str, Any]] = []
    for architecture in _ARCHITECTURES:
        architecture_root = root_path / f"tt_llk_{architecture}"
        if not architecture_root.is_dir():
            continue
        for source in sorted(architecture_root.rglob("*.h")):
            records.extend(
                _scan_file(root_path, source, architecture, model["effects"])
            )
    records.sort(
        key=lambda item: (
            item["architecture"],
            item["source"]["path"],
            item["source"]["start_line"],
            item["name"],
        )
    )
    return records, model


def scan_helpers(
    root: Path | str,
    effect_model: Path | str | None = None,
    *,
    _model: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Discover private non-LLK helper definitions for effect propagation."""
    root_path = Path(root).resolve()
    model = _model
    if model is None:
        complete_source_tree = all(
            (root_path / f"tt_llk_{architecture}").is_dir()
            for architecture in _ARCHITECTURES
        )
        model = load_effect_model(
            effect_model,
            root=root_path,
            _validate_source_contracts=(
                effect_model is not None or complete_source_tree
            ),
        )
    records: list[dict[str, Any]] = []
    for architecture in _ARCHITECTURES:
        architecture_root = root_path / f"tt_llk_{architecture}"
        if not architecture_root.is_dir():
            continue
        for source in sorted(architecture_root.rglob("*.h")):
            records.extend(
                record
                for record in _scan_file(
                    root_path,
                    source,
                    architecture,
                    model["effects"],
                    definition_pattern=_ANY_FUNCTION_NAME,
                )
                if not record["name"].startswith("_llk_")
            )
    records.sort(
        key=lambda item: (
            item["architecture"],
            item["source"]["path"],
            item["source"]["start_line"],
            item["name"],
        )
    )
    return records


def inventory(
    root: Path | str, effect_model: Path | str | None = None
) -> dict[str, Any]:
    """Return a sorted JSON-ready inventory for headers below *root*."""
    records, _ = scan_functions(root, effect_model)
    public = [
        {key: value for key, value in record.items() if not key.startswith("_")}
        for record in records
    ]
    return {"schema_version": 1, "functions": public}


def _validate_restore_contract_sources(
    root: Path, contracts: list[dict[str, Any]]
) -> None:
    architectures = {contract["architecture"] for contract in contracts}
    definitions: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for architecture in architectures:
        architecture_root = root / f"tt_llk_{architecture}"
        if not architecture_root.is_dir():
            raise AuditModelError(
                f"restore contract architecture source is missing: {architecture_root}"
            )
        for source in sorted(architecture_root.rglob("*.h")):
            text = source.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            for definition in _discover_definitions(text):
                close = _matching_brace(text, definition["open_brace"])
                if close is None:
                    continue
                name = definition["name"]
                body = text[definition["open_brace"] + 1 : close]
                definitions.setdefault((architecture, name), []).append(
                    {
                        "path": source.relative_to(root).as_posix(),
                        "start_line": text.count("\n", 0, definition["start"]) + 1,
                        "end_line": text.count("\n", 0, close) + 1,
                        "body_fingerprint": _fingerprint(body),
                        "lines": lines,
                    }
                )

    for contract in contracts:
        architecture = contract["architecture"]
        _validate_restore_reference(
            architecture,
            "function",
            contract["function"],
            contract["source"],
            definitions,
        )
        for reference in ("owner", "pair"):
            target = contract[reference]
            if target is None:
                continue
            _validate_restore_reference(
                architecture,
                reference,
                target,
                contract[f"{reference}_source"],
                definitions,
            )


def _validate_restore_reference(
    architecture: str,
    label: str,
    function: str,
    source: dict[str, Any],
    definitions: dict[tuple[str, str], list[dict[str, Any]]],
) -> None:
    candidates = definitions.get((architecture, function), [])
    if not candidates:
        raise AuditModelError(
            f"restore contract {label} does not exist in {architecture}: {function}"
        )
    fingerprint_matches = [
        candidate
        for candidate in candidates
        if candidate["body_fingerprint"] == source["body_fingerprint"]
    ]
    if not fingerprint_matches:
        raise AuditModelError(
            f"restore contract {label} body fingerprint is stale: {architecture}/{function}"
        )
    anchor_matches = [
        candidate
        for candidate in fingerprint_matches
        if candidate["path"] == source["path"]
        and candidate["start_line"] <= source["line"] <= candidate["end_line"]
        and source["line"] <= len(candidate["lines"])
        and source["token"] in candidate["lines"][source["line"] - 1]
    ]
    if len(anchor_matches) != 1:
        raise AuditModelError(
            f"restore contract {label} source token/line/path is stale or ambiguous: {architecture}/{function}"
        )


def _scan_file(
    root: Path,
    source: Path,
    architecture: str,
    effects: list[dict[str, str]],
    *,
    definition_pattern: re.Pattern[str] = _FUNCTION_NAME,
) -> list[dict[str, Any]]:
    text = source.read_text(encoding="utf-8", errors="replace")
    analysis_text = _mask_inactive_preprocessor(text)
    records: list[dict[str, Any]] = []
    covered_until = -1
    for definition in _discover_definitions(analysis_text, definition_pattern):
        if definition["start"] <= covered_until:
            continue
        open_brace = definition["open_brace"]
        close_brace = _matching_brace(analysis_text, open_brace)
        if close_brace is None:
            continue
        covered_until = close_brace
        name = definition["name"]
        body = text[open_brace + 1 : close_brace]
        masked_body = _mask_comments_and_strings(
            analysis_text[open_brace + 1 : close_brace]
        )
        namespace = _namespace_at(analysis_text, definition["start"])
        start_line = text.count("\n", 0, definition["start"]) + 1
        body_start_line = text.count("\n", 0, open_brace + 1) + 1
        end_line = text.count("\n", 0, close_brace) + 1
        calls = _discover_calls(masked_body, name)
        sinks = _discover_sinks(
            body,
            masked_body,
            body_start_line,
            architecture,
            effects,
        )
        lifecycle = _lifecycle(name, body, calls, sinks)
        template_parameters = _parameters(definition["template"])
        runtime_parameters = _parameters(definition["params"])
        signature = _compact(text[definition["start"] : open_brace])
        owner_match = re.search(
            rf"\b((?:[A-Za-z_]\w*::)*[A-Za-z_]\w*)::{re.escape(name)}\s*\(",
            signature,
        )
        records.append(
            {
                "name": name,
                "signature": signature,
                "template_parameters": template_parameters,
                "runtime_parameters": runtime_parameters,
                "source": {
                    "path": source.relative_to(root).as_posix(),
                    "start_line": start_line,
                    "end_line": end_line,
                },
                "architecture": architecture,
                "thread": _thread(name, source),
                "stage": _stage(name, source),
                "stability_tier": _stability(source),
                "lifecycle": lifecycle,
                "body_fingerprint": _fingerprint(body),
                "direct_calls": calls,
                "canonical_target": (
                    calls[0] if lifecycle == "wrapper" and len(calls) == 1 else None
                ),
                "state_sinks": sinks,
                "_body": body,
                "_masked_body": masked_body,
                "_body_start_line": body_start_line,
                "_parameters": _typed_parameters(
                    template_parameters, runtime_parameters
                ),
                "_namespace": namespace,
                "_owner_type": owner_match.group(1) if owner_match else None,
                "_qualified_name": (f"{namespace}::{name}" if namespace else name),
            }
        )
    return records


def _stability(source: Path) -> str:
    parts = set(source.parts)
    if "debug" in parts:
        return "debug"
    if "experimental" in parts:
        return "experimental"
    if "test" in parts or "tests" in parts:
        return "test_only"
    return "stable"


def _typed_parameters(
    template_parameters: list[str], runtime_parameters: list[str]
) -> list[dict[str, str]]:
    """Return ordered parameter descriptors carrying name, type and kind."""
    typed: list[dict[str, str]] = []
    for index, declaration in enumerate(template_parameters):
        name = _parameter_name(declaration)
        typed.append(
            {
                "name": name or f"__template_{index}",
                "type": declaration,
                "kind": "template",
                "position": index,
                "default": _default_expression(declaration),
            }
        )
    for index, declaration in enumerate(runtime_parameters):
        name = _parameter_name(declaration)
        typed.append(
            {
                "name": name or f"__runtime_{index}",
                "type": declaration,
                "kind": "runtime",
                "position": index,
                "default": _default_expression(declaration),
            }
        )
    return typed


def _default_expression(declaration: str) -> str | None:
    parts = _split_top_level_assignment(declaration)
    return _compact(parts[1]) if parts is not None else None


def _split_top_level_assignment(value: str) -> tuple[str, str] | None:
    depths = {"<": 0, "(": 0, "[": 0, "{": 0}
    pairs = {">": "<", ")": "(", "]": "[", "}": "{"}
    for position, token in enumerate(value):
        if token in depths:
            depths[token] += 1
        elif token in pairs and depths[pairs[token]] > 0:
            depths[pairs[token]] -= 1
        elif (
            token == "="
            and not any(depths.values())
            and value[position : position + 2] not in {"==", "=>"}
            and value[position - 1 : position + 1] not in {"!=", "<=", ">="}
        ):
            return value[:position], value[position + 1 :]
    return None


def _parameter_name(declaration: str) -> str | None:
    """Best-effort extraction of the declared identifier from a parameter."""
    assignment = _split_top_level_assignment(declaration)
    cleaned = (assignment[0] if assignment is not None else declaration).strip()
    cleaned = re.sub(r"\b(class|typename)\b", "", cleaned).strip()
    # A trailing identifier that is not immediately part of a template/type token.
    match = None
    for match in re.finditer(r"[A-Za-z_]\w*", cleaned):
        pass
    if match is None:
        return None
    # Guard against variadic/anonymous forms where the last token is a type keyword.
    identifier = match.group(0)
    if identifier in {
        "void",
        "int",
        "bool",
        "float",
        "double",
        "char",
        "auto",
        "std",
        "uint32_t",
        "uint8_t",
    }:
        # These appear as the last token only for unnamed parameters; treat as unnamed.
        if cleaned.endswith(identifier) and not re.search(
            r"[\*&>\s]" + re.escape(identifier) + r"$", cleaned
        ):
            return None
    return identifier


def _discover_definitions(
    text: str,
    function_pattern: re.Pattern[str] = _FUNCTION_NAME,
) -> list[dict[str, Any]]:
    masked = _mask_comments_and_strings(text)
    definitions: list[dict[str, Any]] = []
    for name_match in function_pattern.finditer(masked):
        if name_match.group(0) in _NON_FUNCTION_NAMES:
            continue
        position = _skip_space(masked, name_match.end())
        if position < len(masked) and masked[position] == "<":
            template_call_end = _matching_delimiter(masked, position, "<", ">")
            if template_call_end is None:
                continue
            position = _skip_space(masked, template_call_end + 1)
        if position >= len(masked) or masked[position] != "(":
            continue
        params_end = _matching_delimiter(masked, position, "(", ")")
        if params_end is None:
            continue
        open_brace = _definition_brace(masked, params_end + 1)
        if open_brace is None:
            continue

        boundary = max(
            masked.rfind(delimiter, 0, name_match.start()) for delimiter in ";{}"
        )
        header_start = boundary + 1
        directive = masked.rfind("\n#", header_start, name_match.start())
        if directive != -1:
            directive_end = masked.find("\n", directive + 1)
            header_start = directive_end + 1 if directive_end != -1 else header_start
        header_start = _skip_space(masked, header_start)

        template_value = None
        declaration_start = header_start
        template_match = re.search(
            r"\btemplate\s*<", masked[header_start : name_match.start()]
        )
        if template_match is not None:
            template_start = header_start + template_match.start()
            angle_start = masked.find("<", template_start, name_match.start())
            angle_end = _matching_delimiter(masked, angle_start, "<", ">")
            if angle_end is None or angle_end >= name_match.start():
                continue
            declaration_start = template_start
            template_value = text[angle_start + 1 : angle_end]
            return_prefix = _compact(masked[angle_end + 1 : name_match.start()])
        else:
            return_prefix = _compact(masked[header_start : name_match.start()])
        if not return_prefix or not re.search(r"[A-Za-z_]", return_prefix):
            continue
        if not _looks_like_definition_prefix(return_prefix):
            continue
        definitions.append(
            {
                "name": name_match.group(0),
                "start": declaration_start,
                "open_brace": open_brace,
                "template": template_value,
                "params": text[position + 1 : params_end],
            }
        )
    return definitions


def _looks_like_definition_prefix(prefix: str) -> bool:
    compact = _compact(prefix)
    if "=" in compact or compact.startswith((".", "->")):
        return False
    if re.search(
        r"\b(return|if|else|for|while|switch|case|throw)\b",
        compact,
    ):
        return False
    return True


def _discover_calls(masked_body: str, current_name: str) -> list[str]:
    calls: set[str] = set()
    for name_match in _FUNCTION_NAME.finditer(masked_body):
        name = name_match.group(0)
        position = _skip_space(masked_body, name_match.end())
        if position < len(masked_body) and masked_body[position] == "<":
            template_end = _matching_delimiter(masked_body, position, "<", ">")
            if template_end is None:
                continue
            position = _skip_space(masked_body, template_end + 1)
        if (
            position < len(masked_body)
            and masked_body[position] == "("
            and name != current_name
        ):
            calls.add(name)
    return sorted(calls)


def _mask_comments_and_strings(text: str) -> str:
    return _mask_lexical(text, mask_literals=True)


def _mask_inactive_preprocessor(text: str) -> str:
    """Blank literal ``#if 0`` branches while retaining byte/line offsets."""
    masked = list(text)
    stack: list[dict[str, bool]] = []
    active = True
    offset = 0
    for line in text.splitlines(keepends=True):
        directive = re.match(
            r"\s*#\s*(ifdef|ifndef|if|elif|else|endif)\b(.*)",
            line,
        )
        line_active = active
        if directive:
            kind, tail = directive.group(1), directive.group(2).strip()
            if kind in {"if", "ifdef", "ifndef"}:
                truth = _static_preprocessor_truth(tail) if kind == "if" else None
                stack.append(
                    {
                        "parent_active": active,
                        "known": truth is not None,
                        "branch_taken": truth is True,
                    }
                )
                active = active and (truth if truth is not None else True)
            elif kind == "elif" and stack:
                frame = stack[-1]
                if not frame["known"]:
                    active = frame["parent_active"]
                elif frame["branch_taken"]:
                    active = False
                else:
                    truth = _static_preprocessor_truth(tail)
                    if truth is None:
                        frame["known"] = False
                        active = frame["parent_active"]
                    else:
                        frame["branch_taken"] = truth
                        active = frame["parent_active"] and truth
            elif kind == "else" and stack:
                frame = stack[-1]
                active = frame["parent_active"] and (
                    not frame["branch_taken"] if frame["known"] else True
                )
                frame["branch_taken"] = True
            elif kind == "endif" and stack:
                active = stack.pop()["parent_active"]
            line_active = False
        if not line_active:
            for index in range(offset, offset + len(line)):
                if masked[index] != "\n":
                    masked[index] = " "
        offset += len(line)
    return "".join(masked)


def _static_preprocessor_truth(expression: str) -> bool | None:
    match = re.fullmatch(
        r"\(?\s*([01])(?:[uUlL]*)\s*\)?(?:\s*(?://.*|/\*.*\*/))?",
        expression,
    )
    return None if match is None else match.group(1) == "1"


def _namespace_at(text: str, position: int) -> str:
    namespaces: list[tuple[int, str]] = []
    masked = _mask_comments_and_strings(text)
    for match in re.finditer(r"\bnamespace\s+([A-Za-z_]\w*)\s*\{", masked[:position]):
        opening = match.end() - 1
        closing = _matching_delimiter(masked, opening, "{", "}")
        if closing is not None and position < closing:
            namespaces.append((opening, match.group(1)))
    return "::".join(name for _, name in sorted(namespaces))


def _mask_lexical(text: str, *, mask_literals: bool) -> str:
    masked = list(text)
    position = 0
    while position < len(text):
        following = text[position + 1] if position + 1 < len(text) else ""
        if text[position] == "/" and following in {"/", "*"}:
            if following == "/":
                ending = text.find("\n", position + 2)
                ending = len(text) if ending == -1 else ending
            else:
                closing = text.find("*/", position + 2)
                ending = len(text) if closing == -1 else closing + 2
            for index in range(position, ending):
                if masked[index] != "\n":
                    masked[index] = " "
            position = ending
            continue
        if _is_quote_start(text, position):
            ending = _skip_quoted(text, position, text[position])
            if mask_literals:
                for index in range(position, ending):
                    if masked[index] != "\n":
                        masked[index] = " "
            position = ending
            continue
        position += 1
    return "".join(masked)


def _skip_space(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position


def _is_quote_start(text: str, position: int) -> bool:
    if text[position] == '"':
        return True
    if text[position] != "'":
        return False
    preceding = text[position - 1] if position > 0 else ""
    following = text[position + 1] if position + 1 < len(text) else ""
    return not (preceding.isalnum() and following.isalnum())


def _matching_delimiter(text: str, opening: int, left: str, right: str) -> int | None:
    depth = 0
    for position in range(opening, len(text)):
        if text[position] == left:
            depth += 1
        elif text[position] == right:
            depth -= 1
            if depth == 0:
                return position
    return None


def _definition_brace(text: str, position: int) -> int | None:
    while position < len(text):
        if text[position] == "{":
            return position
        if text[position] in ";}":
            return None
        position += 1
    return None


def _matching_brace(text: str, opening: int) -> int | None:
    depth = 0
    position = opening
    while position < len(text):
        token = text[position]
        following = text[position + 1] if position + 1 < len(text) else ""
        if token == "/" and following == "/":
            newline = text.find("\n", position + 2)
            position = len(text) if newline == -1 else newline
            continue
        if token == "/" and following == "*":
            ending = text.find("*/", position + 2)
            position = len(text) if ending == -1 else ending + 2
            continue
        if _is_quote_start(text, position):
            position = _skip_quoted(text, position, token)
            continue
        if token == "{":
            depth += 1
        elif token == "}":
            depth -= 1
            if depth == 0:
                return position
        position += 1
    return None


def _skip_quoted(text: str, position: int, quote: str) -> int:
    position += 1
    while position < len(text):
        if text[position] == "\\":
            position += 2
            continue
        if text[position] == quote:
            return position + 1
        position += 1
    return len(text)


def _parameters(value: str | None) -> list[str]:
    if not value or not value.strip() or value.strip() == "void":
        return []
    return [_compact(item) for item in _split_top_level(value)]


def _split_top_level(value: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depths = {"<": 0, "(": 0, "[": 0, "{": 0}
    pairs = {">": "<", ")": "(", "]": "[", "}": "{"}
    for position, token in enumerate(value):
        if token in depths:
            if token == "<" and (
                (position > 0 and value[position - 1] == "<")
                or (position + 1 < len(value) and value[position + 1] == "<")
            ):
                continue
            depths[token] += 1
        elif token in pairs and depths[pairs[token]] > 0:
            depths[pairs[token]] -= 1
        elif token == "," and not any(depths.values()):
            parts.append(value[start:position])
            start = position + 1
    parts.append(value[start:])
    return parts


def _compact(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _fingerprint(body: str) -> str:
    comment_masked_body = _mask_lexical(body, mask_literals=False)
    return hashlib.sha256(_compact(comment_masked_body).encode("utf-8")).hexdigest()


def _stage(name: str, source: Path | None = None) -> str:
    if name.startswith("_llk_sync_") or name == "_llk_stall_cfg_on_":
        return "shared"
    for stage in ("unpack", "math", "pack"):
        if name.startswith(f"_llk_{stage}_"):
            return stage
    if source is not None:
        source_name = source.as_posix().lower()
        if "sfpu" in source_name:
            return "math"
        for stage in ("unpack", "math", "pack"):
            if stage in source_name:
                return stage
    return "unknown"


def _thread(name: str, source: Path | None = None) -> str:
    return {"unpack": "T0", "math": "T1", "pack": "T2", "shared": "shared"}.get(
        _stage(name, source), "unknown"
    )


def _lifecycle(
    name: str, body: str, calls: list[str], sinks: list[dict[str, Any]]
) -> str:
    if "wrapper" in name or (
        len(calls) == 1
        and not sinks
        and re.fullmatch(r"\s*_llk_[\w<> ,:]*\([^;]*\);\s*", body)
    ):
        return "wrapper"
    for lifecycle in ("reinit", "uninit", "clear", "reset", "done", "restore"):
        if f"_{lifecycle}_" in name:
            return lifecycle
    if "_init_" in name:
        return "init"
    if any(token in name for token in ("_config_", "_configure_", "_reconfigure_")):
        return "configure"
    return "execute"


def _discover_sinks(
    body: str,
    masked_body: str,
    body_start_line: int,
    architecture: str,
    effects: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sinks: list[dict[str, Any]] = []
    for effect in effects:
        if (
            effect.get("architectures") is not None
            and architecture not in effect["architectures"]
        ):
            continue
        for match in re.finditer(effect["pattern"], masked_body):
            if _sink_match_is_skipped(body, masked_body, match, effect):
                continue
            sinks.append(
                {
                    "domain": effect["domain"],
                    "operation": effect["operation"],
                    "evidence": {
                        "token": body[match.start() : match.end()],
                        "line": body_start_line + body.count("\n", 0, match.start()),
                    },
                }
            )
    return sorted(
        sinks,
        key=lambda item: (item["evidence"]["line"], item["domain"], item["operation"]),
    )


def _sink_match_is_skipped(
    body: str,
    masked_body: str,
    match: re.Match[str],
    effect: dict[str, Any],
) -> bool:
    extractor = effect.get("extractor") or {}
    if extractor.get("kind") not in {"call", "template_call"}:
        return False
    position = _skip_space(masked_body, match.end())
    if (
        extractor["kind"] == "template_call"
        and position < len(masked_body)
        and masked_body[position] == "<"
    ):
        end = _matching_delimiter(masked_body, position, "<", ">")
        if end is None:
            return False
        position = _skip_space(masked_body, end + 1)
    if position >= len(masked_body) or masked_body[position] != "(":
        return False
    end = _matching_delimiter(masked_body, position, "(", ")")
    if end is None:
        return False
    arguments = [
        _compact(value) for value in _split_top_level(body[position + 1 : end])
    ]
    resource = _model_argument(extractor.get("resource"), arguments)
    value = _model_argument(extractor.get("value"), arguments)
    value_pattern = effect.get("value_pattern")
    return (
        resource in effect.get("skip_resources", [])
        or value in effect.get("skip_values", [])
        or (
            value_pattern is not None
            and (value is None or re.search(value_pattern, value) is None)
        )
    )


def _model_argument(spec: str | None, arguments: list[str]) -> str | None:
    if spec is None:
        return None
    if spec.startswith("literal:"):
        return spec[len("literal:") :]
    kind, _, raw = spec.partition(":")
    if kind != "arg":
        return None
    try:
        position = int(raw)
    except ValueError:
        return None
    return arguments[position] if 0 <= position < len(arguments) else None
