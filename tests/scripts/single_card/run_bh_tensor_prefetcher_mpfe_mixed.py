#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Compare static and dynamic MPFE policies on mixed Llama-8B traffic."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TT_METAL_HOME = Path(os.environ.get("TT_METAL_HOME", SCRIPT_DIR.parents[2])).resolve()
MIXED_TEST = (
    "tests/ttnn/unit_tests/operations/transformers/"
    "test_prefetcher_BH_mpfe_mixed_bench.py::test_mpfe_mixed_llama8b_ff1_sdpa"
)
MPFE_ENV_PREFIX = "TT_METAL_BENCHMARK_TENSOR_PREFETCHER_"
MPFE_ENV_NAMES = (
    "FREE_SENDER_WEIGHT",
    "NOC1_SENDER_WEIGHT",
    "ORDINARY_WEIGHT",
    "IDLE_FREE_SENDER_WEIGHT",
    "IDLE_NOC1_SENDER_WEIGHT",
    "IDLE_ORDINARY_WEIGHT",
)


@dataclass(frozen=True)
class Policy:
    label: str
    active: tuple[int, int, int]
    idle: tuple[int, int, int] | None = None


POLICIES = (
    Policy("static-000", (0, 0, 0)),
    Policy("static-015", (0, 1, 5)),
    Policy("dynamic-000-to-015", (0, 1, 5), (0, 0, 0)),
)


def env_int(name: str, default: int, minimum: int = 1) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def parse_contexts() -> tuple[int, ...]:
    contexts = tuple(
        int(value.strip())
        for value in os.environ.get("MPFE_MIXED_CONTEXTS", "512,1024,2048,4096").split(",")
    )
    if not contexts or any(context < 128 or context % 128 != 0 for context in contexts):
        raise ValueError("MPFE_MIXED_CONTEXTS must contain comma-separated multiples of 128")
    return contexts


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def percent_change(value: float, baseline: float) -> float:
    return 100.0 * (value / baseline - 1.0)


class MixedRunner:
    def __init__(self, pytest_args: list[str]) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.output_dir = Path(
            os.environ.get("OUTPUT_DIR", TT_METAL_HOME / "generated" / f"mpfe-mixed-{timestamp}")
        ).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.output_dir / "results.jsonl"
        self.temp_result_path = self.output_dir / ".current-result.jsonl"
        self.pytest_log_path = self.output_dir / "pytest.log"
        self.manifest_path = self.output_dir / "manifest.json"
        self.python = shlex.split(os.environ.get("PYTHON", sys.executable))
        self.pytest_args = pytest_args
        self.contexts = parse_contexts()
        self.iterations = env_int("MPFE_MIXED_ITERATIONS", 5)
        self.trace_repeats = env_int("BENCH_TRACE_REPEATS", 20)
        self.seed = env_int("MPFE_RANDOM_SEED", 0x4D495845, minimum=0)
        self.manifest = {
            "schema_version": 1,
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=TT_METAL_HOME, text=True
            ).strip(),
            "python": self.python,
            "pytest_args": self.pytest_args,
            "contexts": list(self.contexts),
            "iterations": self.iterations,
            "trace_repeats": self.trace_repeats,
            "random_seed": self.seed,
            "policies": [
                {
                    "label": policy.label,
                    "active": list(policy.active),
                    "idle": list(policy.idle) if policy.idle is not None else None,
                }
                for policy in POLICIES
            ],
        }
        self._validate_manifest()
        self.existing: dict[tuple[int, int, str], dict] = {}
        self.hardware_signature: tuple[int, int] | None = None
        if self.results_path.exists():
            with self.results_path.open(encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    self.existing[self.key(record)] = record
                    signature = (record["num_dram_banks"], record["ring_size"])
                    if self.hardware_signature not in (None, signature):
                        raise RuntimeError(f"{self.results_path} mixes device topologies")
                    self.hardware_signature = signature

    @staticmethod
    def key(record: dict) -> tuple[int, int, str]:
        return (record["sdpa_context"], record["suite_iteration"], record["run_label"])

    def _validate_manifest(self) -> None:
        if self.manifest_path.exists():
            with self.manifest_path.open(encoding="utf-8") as source:
                saved = json.load(source)
            if saved != self.manifest:
                raise RuntimeError(
                    f"{self.output_dir} belongs to a different run; choose a new OUTPUT_DIR or restore its settings"
                )
            return
        with self.manifest_path.open("w", encoding="utf-8") as output:
            json.dump(self.manifest, output, indent=2, sort_keys=True)
            output.write("\n")

    def run_policy(self, context: int, iteration: int, sequence: int, policy: Policy) -> dict:
        record_key = (context, iteration, policy.label)
        if record_key in self.existing:
            return self.existing[record_key]

        environment = os.environ.copy()
        for name in MPFE_ENV_NAMES:
            environment.pop(f"{MPFE_ENV_PREFIX}{name}", None)
        environment.pop("TT_METAL_SLOW_DISPATCH_MODE", None)
        environment.update(
            {
                "ARCH_NAME": "blackhole",
                "PYTHONPATH": str(TT_METAL_HOME)
                + (f":{environment['PYTHONPATH']}" if environment.get("PYTHONPATH") else ""),
                "BENCH_SDPA_CONTEXT": str(context),
                "BENCH_TRACE_REPEATS": str(self.trace_repeats),
                "TT_METAL_BENCHMARK_RESULT_JSONL": str(self.temp_result_path),
                "TT_METAL_BENCHMARK_RUN_LABEL": policy.label,
                "TT_METAL_BENCHMARK_SUITE_ITERATION": str(iteration),
                f"{MPFE_ENV_PREFIX}FREE_SENDER_WEIGHT": str(policy.active[0]),
                f"{MPFE_ENV_PREFIX}NOC1_SENDER_WEIGHT": str(policy.active[1]),
                f"{MPFE_ENV_PREFIX}ORDINARY_WEIGHT": str(policy.active[2]),
            }
        )
        if policy.idle is not None:
            environment.update(
                {
                    f"{MPFE_ENV_PREFIX}IDLE_FREE_SENDER_WEIGHT": str(policy.idle[0]),
                    f"{MPFE_ENV_PREFIX}IDLE_NOC1_SENDER_WEIGHT": str(policy.idle[1]),
                    f"{MPFE_ENV_PREFIX}IDLE_ORDINARY_WEIGHT": str(policy.idle[2]),
                }
            )

        self.temp_result_path.unlink(missing_ok=True)
        command = self.python + ["-m", "pytest", "-q", "--tb=short", *self.pytest_args, MIXED_TEST]
        with self.pytest_log_path.open("a", encoding="utf-8") as output:
            output.write(
                f"\n== context={context} iteration={iteration} sequence={sequence} policy={policy.label} ==\n"
                f"$ {shlex.join(command)}\n"
            )
            output.flush()
            completed = subprocess.run(
                command,
                cwd=TT_METAL_HOME,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"context={context}/iteration={iteration}/{policy.label} failed; see {self.pytest_log_path}"
            )
        if not self.temp_result_path.exists():
            raise RuntimeError(f"{policy.label} produced no benchmark record; see {self.pytest_log_path}")
        with self.temp_result_path.open(encoding="utf-8") as source:
            records = [json.loads(line) for line in source if line.strip()]
        self.temp_result_path.unlink(missing_ok=True)
        if len(records) != 1:
            raise RuntimeError(f"{policy.label} produced {len(records)} records instead of one")

        record = records[0]
        record.update(
            {
                "suite_iteration": iteration,
                "run_label": policy.label,
                "run_sequence": sequence,
                "active_weights": list(policy.active),
                "idle_weights": list(policy.idle if policy.idle is not None else policy.active),
            }
        )
        signature = (record["num_dram_banks"], record["ring_size"])
        if self.hardware_signature not in (None, signature):
            raise RuntimeError(
                f"device topology changed from {self.hardware_signature} to {signature}; use a new OUTPUT_DIR"
            )
        self.hardware_signature = signature
        with self.results_path.open("a", encoding="utf-8") as output:
            json.dump(record, output, sort_keys=True)
            output.write("\n")
        self.existing[record_key] = record
        print(
            f"PASS context={context} iteration={iteration} {policy.label}: "
            f"{record['per_step_us']:.3f} us",
            flush=True,
        )
        return record

    def run(self) -> list[dict]:
        for context_index, context in enumerate(self.contexts):
            for iteration in range(self.iterations):
                policies = list(POLICIES)
                random.Random(self.seed + context_index * 1009 + iteration).shuffle(policies)
                for sequence, policy in enumerate(policies):
                    self.run_policy(context, iteration, sequence, policy)
        return list(self.existing.values())


def write_reports(output_dir: Path, records: list[dict], contexts: tuple[int, ...]) -> None:
    summary_path = output_dir / "summary.csv"
    comparison_path = output_dir / "paired-comparisons.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "context",
                "policy",
                "mean_step_us",
                "stdev_step_us",
                "mean_ff1_tflops",
                "prefetch_to_ordinary_ratio",
                "n",
            ]
        )
        for context in contexts:
            for policy in POLICIES:
                group = [
                    record
                    for record in records
                    if record["sdpa_context"] == context and record["run_label"] == policy.label
                ]
                step_values = [record["per_step_us"] for record in group]
                writer.writerow(
                    [
                        context,
                        policy.label,
                        f"{mean(step_values):.6f}",
                        f"{statistics.stdev(step_values):.6f}" if len(step_values) > 1 else "0.000000",
                        f"{mean([record['ff1_tflops'] for record in group]):.6f}",
                        f"{mean([record['prefetch_to_ordinary_ratio'] for record in group]):.6f}",
                        len(group),
                    ]
                )

    comparisons = (
        ("active-priority", "static-015", "static-000"),
        ("dynamic-restoration", "dynamic-000-to-015", "static-015"),
    )
    with comparison_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "context",
                "comparison",
                "candidate",
                "baseline",
                "mean_delta_pct",
                "ci95_low",
                "ci95_high",
                "n",
            ]
        )
        for context in contexts:
            context_records = [record for record in records if record["sdpa_context"] == context]
            by_key = {
                (record["suite_iteration"], record["run_label"]): record for record in context_records
            }
            for comparison, candidate, baseline in comparisons:
                deltas = [
                    percent_change(
                        by_key[(iteration, baseline)]["per_step_us"],
                        by_key[(iteration, candidate)]["per_step_us"],
                    )
                    for iteration in range(max(record["suite_iteration"] for record in context_records) + 1)
                    if (iteration, candidate) in by_key and (iteration, baseline) in by_key
                ]
                delta_mean = mean(deltas)
                margin = (
                    1.96 * statistics.stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
                )
                writer.writerow(
                    [
                        context,
                        comparison,
                        candidate,
                        baseline,
                        f"{delta_mean:.6f}",
                        f"{delta_mean - margin:.6f}",
                        f"{delta_mean + margin:.6f}",
                        len(deltas),
                    ]
                )

    print(f"Results: {output_dir}")
    print(f"Summary: {summary_path}")
    print(f"Paired comparisons: {comparison_path}")


def main() -> None:
    runner = MixedRunner(sys.argv[1:])
    records = runner.run()
    write_reports(runner.output_dir, records, runner.contexts)


if __name__ == "__main__":
    main()
