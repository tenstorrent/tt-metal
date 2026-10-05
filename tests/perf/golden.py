# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Golden files: accepted values per environment, plus the measurement identity they were taken with."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from tests.perf.contract import MetricSpec

SCHEMA_VERSION = 1
SIGNIFICANT_DIGITS = 4


class GoldenError(ValueError):
    pass


@dataclass
class Environment:
    repetitions: int | None = None
    provenance: dict[str, str] = field(default_factory=dict)
    context: dict[str, str] = field(default_factory=dict)
    cases: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class Golden:
    suite: str
    metrics: dict[str, MetricSpec] = field(default_factory=dict)
    environments: dict[str, Environment] = field(default_factory=dict)


def load(path: Path, suite: str) -> Golden:
    """Loads a golden. A missing file is an empty golden so a new suite can be recorded with --force."""
    if not path.exists():
        return Golden(suite=suite)
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise GoldenError(f"cannot read golden {path}: {error}") from error
    if document.get("schema_version") != SCHEMA_VERSION:
        raise GoldenError(f"golden {path} has schema_version {document.get('schema_version')!r}")
    if document.get("suite") != suite:
        raise GoldenError(f"golden {path} is for suite {document.get('suite')!r}, not {suite!r}")
    golden = Golden(suite=suite, metrics={k: MetricSpec.from_dict(v) for k, v in document["metrics"].items()})
    for name, env in document.get("environments", {}).items():
        cases = env.get("cases", {})
        for case, values in cases.items():
            for metric, value in values.items():
                if metric not in golden.metrics:
                    raise GoldenError(f"golden {path} case {case!r} uses undeclared metric {metric!r}")
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise GoldenError(f"golden {path} case {case!r} has invalid {metric}: {value!r}")
        golden.environments[name] = Environment(
            repetitions=env.get("repetitions"),
            provenance=dict(env.get("provenance", {})),
            context=dict(env.get("context", {})),
            cases={case: {m: float(v) for m, v in values.items()} for case, values in cases.items()},
        )
    return golden


def round_value(value: float) -> float:
    return float(f"{value:.{SIGNIFICANT_DIGITS}g}")


def dumps(golden: Golden) -> str:
    """Serializes deterministically with one case per line so concurrent updates rarely conflict."""
    lines = [
        "{",
        f'  "schema_version": {SCHEMA_VERSION},',
        f'  "suite": {json.dumps(golden.suite)},',
        f'  "metrics": {json.dumps({k: v.as_dict() for k, v in sorted(golden.metrics.items())}, sort_keys=True)},',
        '  "environments": {',
    ]
    env_blocks = []
    for name, env in sorted(golden.environments.items()):
        block = [
            f"    {json.dumps(name)}: {{",
            f'      "repetitions": {json.dumps(env.repetitions)},',
            f'      "provenance": {json.dumps(env.provenance, sort_keys=True)},',
            f'      "context": {json.dumps(env.context, sort_keys=True)},',
            '      "cases": {',
        ]
        case_lines = [
            f"        {json.dumps(case)}: {json.dumps({m: round_value(v) for m, v in sorted(values.items())})}"
            for case, values in sorted(env.cases.items())
        ]
        block.append(",\n".join(case_lines))
        block += ["      }", "    }"]
        env_blocks.append("\n".join(line for line in block if line))
    lines.append(",\n".join(env_blocks))
    lines += ["  }", "}"]
    return "\n".join(line for line in lines if line) + "\n"


def save(golden: Golden, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(golden))
