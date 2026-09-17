# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared configuration and result helpers for Tensor Prefetcher MPFE benchmarks."""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


_PREFIX = "TT_METAL_BENCHMARK_TENSOR_PREFETCHER_"


@dataclass(frozen=True)
class MpfeBenchmarkWeights:
    name: str
    weights: tuple[Optional[int], Optional[int], Optional[int]]
    idle_weights: tuple[Optional[int], Optional[int], Optional[int]]

    def start_kwargs(self) -> dict[str, int]:
        free_sender, noc1_sender, ordinary = self.weights
        idle_free_sender, idle_noc1_sender, idle_ordinary = self.idle_weights
        return {
            name: weight
            for name, weight in (
                ("free_sender_mpfe_weight", free_sender),
                ("noc1_sender_mpfe_weight", noc1_sender),
                ("ordinary_mpfe_weight", ordinary),
                ("idle_free_sender_mpfe_weight", idle_free_sender),
                ("idle_noc1_sender_mpfe_weight", idle_noc1_sender),
                ("idle_ordinary_mpfe_weight", idle_ordinary),
            )
            if weight is not None
        }


def _weight(name: str) -> Optional[int]:
    value = os.environ.get(f"{_PREFIX}{name}")
    if value is None:
        return None
    assert len(value) == 1 and "0" <= value <= "7", f"{name} must be one digit in [0, 7], got {value!r}"
    return int(value)


def resolve_mpfe_benchmark_weights() -> MpfeBenchmarkWeights:
    weights = (
        _weight("FREE_SENDER_WEIGHT"),
        _weight("NOC1_SENDER_WEIGHT"),
        _weight("ORDINARY_WEIGHT"),
    )
    idle_weights = (
        _weight("IDLE_FREE_SENDER_WEIGHT"),
        _weight("IDLE_NOC1_SENDER_WEIGHT"),
        _weight("IDLE_ORDINARY_WEIGHT"),
    )
    if all(weight is None for weight in weights) and all(weight is None for weight in idle_weights):
        name = "default"
    elif all(weight is None for weight in idle_weights) and all(weight is not None for weight in weights):
        name = f"static-{''.join(str(weight) for weight in weights)}"
    else:
        active_label = "".join("-" if weight is None else str(weight) for weight in weights)
        idle_label = "".join("-" if weight is None else str(weight) for weight in idle_weights)
        name = f"dynamic-{idle_label}-to-{active_label}"
    return MpfeBenchmarkWeights(name=name, weights=weights, idle_weights=idle_weights)


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


def weight_result_fields(weights: MpfeBenchmarkWeights) -> dict[str, Any]:
    return asdict(weights)
