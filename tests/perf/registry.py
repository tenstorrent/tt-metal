# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Loads tests/perf/suites.yaml. Entries are data only: what to run and the policy to judge it by."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from tests.perf.compare import Policy

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = Path(__file__).with_name("suites.yaml")
KINDS = ("google_benchmark", "contract")


class RegistryError(ValueError):
    pass


@dataclass(frozen=True)
class Suite:
    name: str
    binary: Path
    golden: Path
    policy: Policy
    kind: str = "google_benchmark"
    args: tuple = ()
    repetitions: int | dict[str, int] | None = None
    process_repetitions: int = 1
    env: dict = field(default_factory=dict)
    env_unset: tuple = ()

    def repetitions_for(self, environment: str) -> int | None:
        if isinstance(self.repetitions, dict):
            if environment not in self.repetitions:
                raise RegistryError(f"suite {self.name} has no repetition count for environment {environment!r}")
            return int(self.repetitions[environment])
        return None if self.repetitions is None else int(self.repetitions)


def _policy(raw: dict, suite: str) -> Policy:
    try:
        return Policy(
            regression_pct=float(raw["regression_pct"]),
            improvement_pct=float(raw["improvement_pct"]),
            enforce=bool(raw.get("enforce", True)),
            overrides=tuple((pattern, dict(values)) for pattern, values in (raw.get("overrides") or {}).items()),
        )
    except KeyError as missing:
        raise RegistryError(f"suite {suite} policy is missing {missing}") from None


def load(path: Path = REGISTRY) -> dict[str, Suite]:
    raw = yaml.safe_load(path.read_text()) or {}
    suites = {}
    for name, entry in raw.items():
        kind = entry.get("kind", "google_benchmark")
        if kind not in KINDS:
            raise RegistryError(f"suite {name} has kind {kind!r}; expected one of {KINDS}")
        suites[name] = Suite(
            name=name,
            binary=REPO_ROOT / entry["binary"],
            golden=REPO_ROOT / entry["golden"],
            policy=_policy(entry.get("policy") or {}, name),
            kind=kind,
            args=tuple(entry.get("args") or ()),
            repetitions=entry.get("repetitions"),
            process_repetitions=int(entry.get("process_repetitions", 1)),
            env={k: str(v) for k, v in (entry.get("env") or {}).items()},
            env_unset=tuple(entry.get("env_unset") or ()),
        )
    return suites
