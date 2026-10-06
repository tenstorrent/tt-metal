# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""AST-derived finite parameter sweeps with conservative assertion filtering."""

import itertools
import json
import math
import re
from collections import defaultdict

from .ast_scan import llk_ast

MAX_COMBINATIONS = 100_000


def other_arguments(arguments: list[str], parameters: list[dict]) -> tuple:
    values = []
    for parameter in parameters:
        if parameter["values"]:
            continue
        value = arguments[parameter["index"]]
        cast = re.fullmatch(r"\([^()]+\)\s*(.+)", value)
        numeric = cast[1] if cast and parameter["kind"] == "integer" else value
        integer = llk_ast.integer_value(numeric)
        values.append(
            (parameter["name"], str(integer) if integer is not None else value)
        )
    return tuple(values)


def label(parameter: dict, value: int) -> str:
    return next(
        (
            " / ".join(item["names"])
            for item in parameter["values"]
            if item["value"] == value
        ),
        str(value),
    )


def audit_group(definition, parameters, observations, group):
    function = llk_ast.function_from_json(json.dumps(definition))
    declared = {parameter.index: parameter for parameter in function.parameters}
    finite = [parameter for parameter in parameters if parameter["values"]]
    domains = [
        set(item["value"] for item in parameter["values"]) for parameter in finite
    ]
    executed, observed, unknown = set(), set(), set()
    for observation in observations:
        if not finite:
            continue
        values = tuple(
            llk_ast.argument_value(
                declared[p["index"]], observation["arguments"][p["index"]]
            )
            for p in finite
        )
        if any(value not in domain for value, domain in zip(values, domains)):
            unknown.add(observation["signature"])
        else:
            observed.add(values)
            if observation["execution_count"] > 0:
                executed.add(values)
    total = math.prod(map(len, domains)) if finite else 0
    fixed = {
        ("template", p["index"]): value
        for p in parameters
        if p["kind"] == "integer" and not p["values"]
        if (value := llk_ast.integer_value(dict(group[2]).get(p["name"], "")))
        is not None
    }
    excluded, unresolved_constraints, conflicts = {}, 0, []
    assertions = definition.get("assertions")
    checked = bool(finite) and assertions is not None and total <= MAX_COMBINATIONS
    candidate_domains = [set() for _ in finite] if checked else domains
    if checked:
        for values in itertools.product(*(sorted(domain) for domain in domains)):
            environment = {
                **fixed,
                **{("template", p["index"]): value for p, value in zip(finite, values)},
            }
            results = [
                llk_ast.check(assertion, environment)
                for assertion in function.assertions
            ]
            rejected = [i for i, result in enumerate(results) if result is False]
            if rejected:
                excluded[values] = rejected
                continue
            for domain, value in zip(candidate_domains, values):
                domain.add(value)
            unresolved_constraints += any(result is None for result in results)
    else:
        unresolved_constraints = total
    contradictions = (executed & excluded.keys()) | {
        values
        for values in observed & excluded.keys()
        if any(assertions[i]["kind"] == "static_assert" for i in excluded[values])
    }
    for values in sorted(contradictions):
        conflicts.append(
            {
                "values": {
                    p["name"]: label(p, value) for p, value in zip(finite, values)
                },
                "assertions": excluded[values],
            }
        )
        del excluded[values]
        for domain, value in zip(candidate_domains, values):
            domain.add(value)
    remaining = total - len(excluded)
    axes = []
    for index, parameter in enumerate(finite):
        seen = {values[index] for values in executed}
        axes.append(
            {
                **parameter,
                "seen": [label(parameter, value) for value in sorted(seen)],
                "excluded": [
                    label(parameter, value)
                    for value in sorted(domains[index] - candidate_domains[index])
                ],
                "missing": [
                    label(parameter, value)
                    for value in sorted(candidate_domains[index] - seen)
                ],
            }
        )
    untracked = [p for p in parameters if not p["values"]]
    unresolved = (
        unknown
        or conflicts
        or not finite
        or remaining == 0
        or any(p["kind"] == "unresolved" for p in untracked)
    )
    state = "sweep_gap" if len(executed) < remaining else "sweep_complete"
    if unresolved:
        state = "unresolved"
    if not observations:
        state = "absent"
    missing = itertools.islice(
        (
            values
            for values in itertools.product(*(sorted(domain) for domain in domains))
            if values not in executed and values not in excluded
        ),
        32,
    )
    return {
        "arch": definition["arch"],
        "source": definition["header"],
        "function": definition["function"],
        "trisc": group[0],
        "build_axes": json.loads(group[1]),
        "other_arguments": dict(group[2]),
        "status": state,
        "observed": bool(observations),
        "parameters": axes,
        "untracked_parameters": untracked,
        "unmapped_instantiations": sorted(unknown),
        "declared_combinations": total,
        "candidate_combinations": remaining,
        "assertion_excluded_combinations": len(excluded),
        "constraint_unknown_combinations": unresolved_constraints,
        "constraints_checked": checked,
        "constraint_conflicts": conflicts,
        "assertions": assertions or [],
        "excluded_preview": [
            {
                "values": {
                    p["name"]: label(p, value) for p, value in zip(finite, values)
                },
                "assertions": reasons,
            }
            for values, reasons in itertools.islice(excluded.items(), 32)
        ],
        "executed_combinations": len(executed),
        "missing_combinations": remaining - len(executed),
        "missing_preview": (
            [
                {p["name"]: label(p, value) for p, value in zip(finite, values)}
                for values in missing
            ]
            if finite
            else []
        ),
    }


def finite_sweeps(baseline: dict, records: list[dict]) -> dict:
    indexed = defaultdict(list)
    for record in records:
        indexed[record.get("definition")].append(record)
    rows, unresolved = [], []
    for definition in baseline["definitions"]:
        declarations = definition["template_parameters"]
        if not declarations:
            continue
        if (
            any("..." in declaration for declaration in declarations)
            or definition["status"] == "stale"
        ):
            unresolved.append(
                {
                    "function": definition["function"],
                    "reason": "Parameter pack or changed source snapshot",
                }
            )
            continue
        parameters = definition["parameters"]
        groups = defaultdict(list)
        for record in indexed[definition["key"]]:
            if len(record["arguments"]) != len(parameters):
                unresolved.append(
                    {
                        "function": definition["function"],
                        "reason": "Mismatched AST template arguments",
                    }
                )
                continue
            group = (
                record["trisc"],
                json.dumps(record.get("build_axes", {}), sort_keys=True),
                other_arguments(record["arguments"], parameters),
            )
            groups[group].append(record)
        rows.extend(
            audit_group(definition, parameters, observations, group)
            for group, observations in (
                groups
                or {
                    (
                        definition["trisc"],
                        json.dumps(definition["build_axes"], sort_keys=True),
                        (),
                    ): []
                }
            ).items()
        )
    return {
        "rows": rows,
        "unresolved_definitions": unresolved,
        "counts": {
            state: sum(row["status"] == state for row in rows if row["observed"])
            for state in ("sweep_complete", "sweep_gap", "unresolved")
        },
    }
