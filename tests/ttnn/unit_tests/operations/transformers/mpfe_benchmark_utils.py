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
class MpfeBenchmarkWeights:
    name: str
    weights: tuple[int, int, int]


def _weight(name: str, default: int) -> int:
    value = os.environ.get(f"{_PREFIX}{name}", str(default))
    assert len(value) == 1 and "0" <= value <= "7", f"{name} must be one digit in [0, 7], got {value!r}"
    return int(value)


def resolve_mpfe_benchmark_weights() -> MpfeBenchmarkWeights:
    weights = (
        _weight("FREE_SENDER_WEIGHT", 0),
        _weight("NOC1_SENDER_WEIGHT", 1),
        _weight("ORDINARY_WEIGHT", 5),
    )
    return MpfeBenchmarkWeights(name=f"static-{''.join(str(weight) for weight in weights)}", weights=weights)


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
