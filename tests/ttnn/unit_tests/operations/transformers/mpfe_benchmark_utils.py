# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared configuration and result helpers for Tensor Prefetcher MPFE benchmarks."""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_PREFIX = "TT_METAL_BENCHMARK_TENSOR_PREFETCHER_"


@dataclass(frozen=True)
class MpfeBenchmarkPolicy:
    name: str
    idle_weights: tuple[int, int, int]
    active_weights: tuple[int, int, int]
    request_sync: bool


def _weight(name: str, default: int) -> int:
    value = int(os.environ.get(f"{_PREFIX}{name}", str(default)))
    assert 0 <= value <= 7, f"{name} must be in [0, 7], got {value}"
    return value


def _tuple(name: str) -> tuple[int, int, int] | None:
    value = os.environ.get(f"{_PREFIX}{name}")
    if value is None:
        return None
    parts = value.split(",")
    assert len(parts) == 3, f"{name} must contain free,noc1,ordinary weights"
    weights = tuple(int(part) for part in parts)
    assert all(0 <= weight <= 7 for weight in weights), f"{name} weights must be in [0, 7]"
    return weights


def resolve_mpfe_benchmark_policy() -> MpfeBenchmarkPolicy:
    idle_override = _tuple("IDLE_WEIGHTS")
    active_override = _tuple("ACTIVE_WEIGHTS")
    assert (idle_override is None) == (active_override is None), "IDLE_WEIGHTS and ACTIVE_WEIGHTS must be set together"

    force_sync = os.environ.get(f"{_PREFIX}FORCE_REQUEST_SYNC", "0")
    assert force_sync in ("0", "1"), f"FORCE_REQUEST_SYNC must be 0 or 1, got {force_sync!r}"

    if idle_override is not None:
        conflicting = [
            name
            for name in ("PRIORITY_POLICY", "HIGH_WEIGHT", "MEDIUM_WEIGHT", "ACTIVE_WEIGHT")
            if os.environ.get(f"{_PREFIX}{name}") is not None
        ]
        assert not conflicting, f"tuple overrides cannot be combined with {', '.join(conflicting)}"
        idle = idle_override
        active = active_override
        name = "custom"
    else:
        name = os.environ.get(f"{_PREFIX}PRIORITY_POLICY", "dynamic-007")
        high = _weight("HIGH_WEIGHT", 7)
        medium = _weight("MEDIUM_WEIGHT", 3) if name == "static-037" else 3
        active_weight = _weight("ACTIVE_WEIGHT", 0) if name == "dynamic-007" else 0
        if name != "dynamic-007":
            assert os.environ.get(f"{_PREFIX}ACTIVE_WEIGHT") is None, (
                "ACTIVE_WEIGHT can only be combined with dynamic-007"
            )
        policies = {
            "dynamic-007": ((high, high, high), (active_weight, active_weight, high)),
            "dynamic-000": ((0, 0, 0), (0, 0, high)),
            "static-000": ((0, 0, 0), (0, 0, 0)),
            "static-777": ((high, high, high), (high, high, high)),
            "static-007": ((0, 0, high), (0, 0, high)),
            "static-037": ((0, medium, high), (0, medium, high)),
            "static-770": ((high, high, 0), (high, high, 0)),
        }
        assert name in policies, f"unknown MPFE policy {name!r}"
        if name == "static-037":
            assert medium <= high, "MEDIUM_WEIGHT must not exceed HIGH_WEIGHT"
        idle, active = policies[name]

    return MpfeBenchmarkPolicy(
        name=name,
        idle_weights=idle,
        active_weights=active,
        request_sync=idle[2] != active[2] or force_sync == "1",
    )


def append_benchmark_jsonl(result: dict[str, Any]) -> None:
    output_path = os.environ.get("TT_METAL_BENCHMARK_RESULT_JSONL")
    if output_path is None:
        return
    run_label = os.environ.get("TT_METAL_BENCHMARK_RUN_LABEL")
    suite_iteration = os.environ.get("TT_METAL_BENCHMARK_SUITE_ITERATION")
    if run_label is not None:
        result["run_label"] = run_label
    if suite_iteration is not None:
        result["suite_iteration"] = int(suite_iteration)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        json.dump(result, output, sort_keys=True)
        output.write("\n")


def policy_result_fields(policy: MpfeBenchmarkPolicy) -> dict[str, Any]:
    return asdict(policy)
