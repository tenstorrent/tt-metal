#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Find and validate MPFE weights for the Device-side Tensor Prefetcher.

Stage 1 exhaustively sweeps static 0/M/H tuples, Stage 2 confirms the six
leaders, and Stage 3 compares static, dynamic, and forced-sync forms of the
two finalists on both FF1 matmul throughput and repeated-request contention.
"""

from __future__ import annotations

import csv
import json
import os
import random
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
TT_METAL_HOME = Path(os.environ.get("TT_METAL_HOME", SCRIPT_DIR.parents[2])).resolve()
MATMUL_TEST = (
    "tests/ttnn/unit_tests/operations/transformers/"
    "test_prefetcher_BH_bench.py::test_bench_dram_core_repeats_recv_contig"
)
CONTENTION_TEST = (
    "tests/ttnn/unit_tests/operations/transformers/"
    "test_prefetcher_BH_bw_bench.py::test_mpfe_priority_contention"
)
MPFE_ENV_PREFIX = "TT_METAL_BENCHMARK_TENSOR_PREFETCHER_"
MPFE_ENV_NAMES = (
    "PRIORITY_POLICY",
    "HIGH_WEIGHT",
    "MEDIUM_WEIGHT",
    "ACTIVE_WEIGHT",
    "IDLE_WEIGHTS",
    "ACTIVE_WEIGHTS",
    "FORCE_REQUEST_SYNC",
)
BENCHMARK_ENV_NAMES = (
    "BENCH_K",
    "BENCH_N",
    "BENCH_DTYPE",
    "BENCH_RECV_PER_BANK",
    "BENCH_ORDINARY_READ_BYTES",
    "BENCH_GCB_WINDOW_BLOCKS",
    "BENCH_DUAL_SENDERS",
    "BENCH_TRACE_REPEATS",
)


@dataclass(frozen=True)
class Case:
    label: str
    idle: tuple[int, int, int] | None = None
    active: tuple[int, int, int] | None = None
    force_sync: bool = False
    named_policy: str | None = None
    weight_tuple: tuple[int, int, int] | None = None
    mode: str = "static"


def env_int(name: str, default: int, minimum: int = 1) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def tuple_string(weights: tuple[int, int, int]) -> str:
    return ",".join(str(weight) for weight in weights)


def tuple_label(weights: tuple[int, int, int]) -> str:
    return "".join(str(weight) for weight in weights)


class OptimizationRunner:
    def __init__(self, pytest_args: list[str]) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.output_dir = Path(
            os.environ.get("OUTPUT_DIR", TT_METAL_HOME / "generated" / f"mpfe-optimize-{timestamp}")
        ).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.output_dir / "results.jsonl"
        self.pytest_log_path = self.output_dir / "pytest.log"
        self.summary_log_path = self.output_dir / "summary.log"
        self.temp_result_path = self.output_dir / ".current-result.jsonl"
        self.manifest_path = self.output_dir / "manifest.json"
        self.python = shlex.split(os.environ.get("PYTHON", sys.executable))
        self.pytest_args = pytest_args
        self.matmul_shape = os.environ.get("MPFE_MATMUL_SHAPE", "3B_FF1")
        self.matmul_repeats = env_int("BENCH_TRACE_REPEATS", 100)
        self.contention_repeats = env_int("CONTENTION_TRACE_REPEATS", 50)
        self.seed = env_int("MPFE_RANDOM_SEED", 0x4D504645, minimum=0)
        self.tuning_iterations = env_int("MPFE_TUNING_ITERATIONS", 5)
        self.confirm_iterations = env_int("MPFE_CONFIRM_ITERATIONS", 15)
        self.lifecycle_iterations = env_int("MPFE_LIFECYCLE_ITERATIONS", 20)
        self.manifest = {
            "schema_version": 1,
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=TT_METAL_HOME,
                text=True,
            ).strip(),
            "python": self.python,
            "pytest_args": self.pytest_args,
            "matmul_shape": self.matmul_shape,
            "matmul_trace_repeats": self.matmul_repeats,
            "contention_trace_repeats": self.contention_repeats,
            "random_seed": self.seed,
            "tuning_iterations": self.tuning_iterations,
            "confirmation_iterations": self.confirm_iterations,
            "lifecycle_iterations": self.lifecycle_iterations,
        }
        if self.manifest_path.exists():
            with self.manifest_path.open(encoding="utf-8") as source:
                saved_manifest = json.load(source)
            if saved_manifest != self.manifest:
                raise RuntimeError(
                    f"{self.output_dir} belongs to a different benchmark configuration; "
                    "choose a new OUTPUT_DIR or restore the settings in manifest.json"
                )
        else:
            with self.manifest_path.open("w", encoding="utf-8") as output:
                json.dump(self.manifest, output, indent=2, sort_keys=True)
                output.write("\n")
        self.existing: dict[tuple[str, str, int, str], dict] = {}
        self.hardware_signature: tuple[int, int] | None = None
        if self.results_path.exists():
            with self.results_path.open(encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        record = json.loads(line)
                        self.existing[self.key(record)] = record
                        signature = (record["num_dram_banks"], record["ring_size"])
                        if self.hardware_signature not in (None, signature):
                            raise RuntimeError(f"{self.results_path} mixes multiple device topologies")
                        self.hardware_signature = signature

    @staticmethod
    def key(record: dict) -> tuple[str, str, int, str]:
        return (
            record["optimization_stage"],
            record["optimization_benchmark"],
            record["suite_iteration"],
            record["run_label"],
        )

    def log(self, message: str) -> None:
        print(message, flush=True)
        with self.summary_log_path.open("a", encoding="utf-8") as output:
            output.write(message + "\n")

    def test_args(self, benchmark: str) -> tuple[list[str], int]:
        if benchmark == "matmul":
            return [MATMUL_TEST, "-k", f"{self.matmul_shape} and shard_contiguous"], self.matmul_repeats
        if benchmark == "contention":
            return [CONTENTION_TEST], self.contention_repeats
        raise ValueError(f"unknown benchmark {benchmark}")

    def run_case(
        self,
        stage: str,
        benchmark: str,
        iteration: int,
        case: Case,
        sequence: int,
        sequence_count: int,
    ) -> dict:
        record_key = (stage, benchmark, iteration, case.label)
        if record_key in self.existing:
            return self.existing[record_key]

        test_args, trace_repeats = self.test_args(benchmark)
        environment = os.environ.copy()
        for name in MPFE_ENV_NAMES:
            environment.pop(f"{MPFE_ENV_PREFIX}{name}", None)
        for name in BENCHMARK_ENV_NAMES:
            environment.pop(name, None)
        environment.pop("TT_METAL_SLOW_DISPATCH_MODE", None)
        environment.update(
            {
                "ARCH_NAME": "blackhole",
                "BENCH_DUAL_SENDERS": "1",
                "BENCH_TRACE_REPEATS": str(trace_repeats),
                "PYTHONPATH": str(TT_METAL_HOME)
                + (f":{environment['PYTHONPATH']}" if environment.get("PYTHONPATH") else ""),
                "TT_METAL_BENCHMARK_RESULT_JSONL": str(self.temp_result_path),
                "TT_METAL_BENCHMARK_RUN_LABEL": case.label,
                "TT_METAL_BENCHMARK_SUITE_ITERATION": str(iteration),
            }
        )
        if case.named_policy is not None:
            environment[f"{MPFE_ENV_PREFIX}PRIORITY_POLICY"] = case.named_policy
        else:
            if case.idle is None or case.active is None:
                raise ValueError(f"{case.label} has no MPFE weights")
            environment[f"{MPFE_ENV_PREFIX}IDLE_WEIGHTS"] = tuple_string(case.idle)
            environment[f"{MPFE_ENV_PREFIX}ACTIVE_WEIGHTS"] = tuple_string(case.active)
        if case.force_sync:
            environment[f"{MPFE_ENV_PREFIX}FORCE_REQUEST_SYNC"] = "1"

        self.temp_result_path.unlink(missing_ok=True)
        command = self.python + ["-m", "pytest", "-q", "--tb=short", *self.pytest_args, *test_args]
        with self.pytest_log_path.open("a", encoding="utf-8") as output:
            output.write(
                f"\n== {stage} {benchmark} iteration={iteration} case={case.label} ==\n"
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
                f"{stage}/{benchmark}/{iteration}/{case.label} failed; see {self.pytest_log_path}"
            )
        if not self.temp_result_path.exists():
            raise RuntimeError(
                f"{stage}/{benchmark}/{iteration}/{case.label} produced no result (possibly skipped); "
                f"see {self.pytest_log_path}"
            )
        with self.temp_result_path.open(encoding="utf-8") as source:
            records = [json.loads(line) for line in source if line.strip()]
        self.temp_result_path.unlink(missing_ok=True)
        if len(records) != 1:
            raise RuntimeError(f"{case.label} produced {len(records)} records instead of one")

        record = records[0]
        record.update(
            {
                "optimization_stage": stage,
                "optimization_benchmark": benchmark,
                "optimization_mode": case.mode,
                "optimization_sequence": sequence,
                "optimization_sequence_count": sequence_count,
                "weight_tuple": list(case.weight_tuple) if case.weight_tuple is not None else None,
            }
        )
        signature = (record["num_dram_banks"], record["ring_size"])
        if self.hardware_signature not in (None, signature):
            raise RuntimeError(
                f"device topology changed from {self.hardware_signature} to {signature}; "
                "resume with the original device or choose a new OUTPUT_DIR"
            )
        self.hardware_signature = signature
        with self.results_path.open("a", encoding="utf-8") as output:
            json.dump(record, output, sort_keys=True)
            output.write("\n")
        self.existing[record_key] = record
        metric_name = "tflops" if benchmark == "matmul" else "combined_gbps"
        self.log(
            f"PASS {stage} {benchmark} iteration={iteration} {case.label}: "
            f"{record[metric_name]:.6f} {metric_name}"
        )
        return record

    def randomized(self, stage: str, iteration: int, cases: Iterable[Case]) -> list[Case]:
        shuffled = list(cases)
        stage_seed = sum((index + 1) * ord(character) for index, character in enumerate(stage))
        random.Random(self.seed + stage_seed + iteration).shuffle(shuffled)
        return shuffled


def static_case(weights: tuple[int, int, int], label_prefix: str = "static") -> Case:
    return Case(
        label=f"{label_prefix}-{tuple_label(weights)}",
        idle=weights,
        active=weights,
        weight_tuple=weights,
        mode="static",
    )


def baseline_case(position: str) -> Case:
    weights = (0, 0, 0)
    return Case(
        label=f"static-000-{position}",
        idle=weights,
        active=weights,
        weight_tuple=weights,
        mode="baseline",
    )


def mean(values: Iterable[float]) -> float:
    return statistics.fmean(values)


def write_ranking(
    path: Path,
    records: list[dict],
    candidate_labels: set[str],
) -> list[tuple[int, int, int]]:
    baselines: dict[int, tuple[dict, dict]] = {}
    iterations = sorted({record["suite_iteration"] for record in records})
    for iteration in iterations:
        sentinels = [
            record
            for record in records
            if record["suite_iteration"] == iteration
            and record["run_label"] in {"static-000-start", "static-000-end"}
        ]
        if len(sentinels) != 2:
            raise RuntimeError(f"iteration {iteration} has {len(sentinels)} baseline sentinels")
        start = next(record for record in sentinels if record["run_label"] == "static-000-start")
        end = next(record for record in sentinels if record["run_label"] == "static-000-end")
        baselines[iteration] = (start, end)

    def interpolated_baseline(record: dict) -> float:
        start, end = baselines[record["suite_iteration"]]
        span = end["optimization_sequence"] - start["optimization_sequence"]
        fraction = (record["optimization_sequence"] - start["optimization_sequence"]) / span
        return start["tflops"] + fraction * (end["tflops"] - start["tflops"])

    rows = []
    for label in sorted(candidate_labels):
        samples = [record for record in records if record["run_label"] == label]
        if label == "static-000" and not samples:
            samples = [
                record
                for record in records
                if record["run_label"] in {"static-000-start", "static-000-end"}
            ]
            deltas = []
            for record in samples:
                start, end = baselines[record["suite_iteration"]]
                iteration_baseline = mean((start["tflops"], end["tflops"]))
                deltas.append((record["tflops"] / iteration_baseline - 1.0) * 100.0)
        else:
            deltas = [
                (record["tflops"] / interpolated_baseline(record) - 1.0) * 100.0
                for record in samples
            ]
        weights_value = samples[0]["weight_tuple"]
        weights = tuple(weights_value) if weights_value is not None else None
        rows.append(
            {
                "tuple": tuple_string(weights) if weights is not None else label,
                "mean_tflops": mean(record["tflops"] for record in samples),
                "mean_vs_static_000_percent": mean(deltas),
                "stdev_vs_static_000_percent": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                "samples": len(samples),
                "_weights": weights,
            }
        )
    rows.sort(key=lambda row: row["mean_vs_static_000_percent"], reverse=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        fieldnames = [
            "rank",
            "tuple",
            "mean_tflops",
            "mean_vs_static_000_percent",
            "stdev_vs_static_000_percent",
            "samples",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "tuple": row["tuple"],
                    "mean_tflops": f"{row['mean_tflops']:.6f}",
                    "mean_vs_static_000_percent": f"{row['mean_vs_static_000_percent']:.3f}",
                    "stdev_vs_static_000_percent": f"{row['stdev_vs_static_000_percent']:.3f}",
                    "samples": row["samples"],
                }
            )
    return [row["_weights"] for row in rows if row["_weights"] is not None]


def write_lifecycle_summary(path: Path, records: list[dict]) -> None:
    grouped: dict[tuple[str, tuple[int, int, int], str], list[dict]] = {}
    for record in records:
        weights_value = record.get("weight_tuple")
        if weights_value is None:
            continue
        key = (
            record["optimization_benchmark"],
            tuple(weights_value),
            record["optimization_mode"],
        )
        grouped.setdefault(key, []).append(record)

    rows = []
    for (benchmark, weights, mode), samples in grouped.items():
        metric_name = "tflops" if benchmark == "matmul" else "combined_gbps"
        static_by_iteration = {
            record["suite_iteration"]: record[metric_name]
            for record in grouped.get((benchmark, weights, "static"), [])
        }
        baseline_by_iteration = {
            record["suite_iteration"]: record[metric_name]
            for record in grouped.get((benchmark, (0, 0, 0), "baseline"), [])
        }
        vs_static = [
            (record[metric_name] / static_by_iteration[record["suite_iteration"]] - 1.0) * 100.0
            for record in samples
            if record["suite_iteration"] in static_by_iteration
        ]
        vs_baseline = [
            (record[metric_name] / baseline_by_iteration[record["suite_iteration"]] - 1.0) * 100.0
            for record in samples
            if record["suite_iteration"] in baseline_by_iteration
        ]
        rows.append(
            {
                "benchmark": benchmark,
                "tuple": tuple_string(weights),
                "mode": mode,
                "metric": metric_name,
                "mean": mean(record[metric_name] for record in samples),
                "vs_same_tuple_static_percent": mean(vs_static) if vs_static else "",
                "stdev_vs_same_tuple_static_percent": (
                    statistics.stdev(vs_static) if len(vs_static) > 1 else ""
                ),
                "vs_static_000_percent": mean(vs_baseline) if vs_baseline else "",
                "stdev_vs_static_000_percent": (
                    statistics.stdev(vs_baseline) if len(vs_baseline) > 1 else ""
                ),
                "samples": len(samples),
            }
        )
    rows.sort(key=lambda row: (row["benchmark"], row["tuple"], row["mode"]))
    with path.open("w", encoding="utf-8", newline="") as output:
        fieldnames = [
            "benchmark",
            "tuple",
            "mode",
            "metric",
            "mean",
            "vs_same_tuple_static_percent",
            "stdev_vs_same_tuple_static_percent",
            "vs_static_000_percent",
            "stdev_vs_static_000_percent",
            "samples",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row["mean"] = f"{row['mean']:.6f}"
            for field in (
                "vs_same_tuple_static_percent",
                "stdev_vs_same_tuple_static_percent",
                "vs_static_000_percent",
                "stdev_vs_static_000_percent",
            ):
                if row[field] != "":
                    row[field] = f"{row[field]:.3f}"
            writer.writerow(row)


def records_for(runner: OptimizationRunner, stage: str, benchmark: str) -> list[dict]:
    return [
        record
        for record in runner.existing.values()
        if record["optimization_stage"] == stage and record["optimization_benchmark"] == benchmark
    ]


def main() -> int:
    runner = OptimizationRunner(sys.argv[1:])
    runner.log(f"MPFE optimization output: {runner.output_dir}")
    runner.log(
        f"shape={runner.matmul_shape} tuning={runner.tuning_iterations} "
        f"confirmation={runner.confirm_iterations} lifecycle={runner.lifecycle_iterations}"
    )

    all_tuples = [(0, medium, high) for medium in range(8) for high in range(medium, 8)]
    tuning_candidates = [static_case(weights) for weights in all_tuples if weights != (0, 0, 0)]
    runner.log(f"Stage 1: exhaustive static sweep ({len(all_tuples)} tuples)")
    for iteration in range(1, runner.tuning_iterations + 1):
        ordered_cases = runner.randomized("tuning", iteration, tuning_candidates)
        sequence_count = len(ordered_cases) + 2
        runner.run_case("tuning", "matmul", iteration, baseline_case("start"), 0, sequence_count)
        for sequence, case in enumerate(ordered_cases, start=1):
            runner.run_case("tuning", "matmul", iteration, case, sequence, sequence_count)
        runner.run_case(
            "tuning",
            "matmul",
            iteration,
            baseline_case("end"),
            sequence_count - 1,
            sequence_count,
        )
    tuning_records = records_for(runner, "tuning", "matmul")
    tuning_ranking_path = runner.output_dir / "stage1-tuning-ranking.csv"
    tuning_ranking = write_ranking(
        tuning_ranking_path,
        tuning_records,
        {case.label for case in tuning_candidates} | {"static-000"},
    )
    finalists = tuning_ranking[:6]
    runner.log(f"Stage 1 leaders: {', '.join(tuple_string(weights) for weights in finalists)}")

    confirmation_candidates = [static_case(weights) for weights in finalists]
    production = Case(
        label="production-default",
        named_policy="dynamic-007",
        mode="production",
    )
    runner.log("Stage 2: randomized finalist confirmation")
    for iteration in range(1, runner.confirm_iterations + 1):
        cases = [*confirmation_candidates, production]
        ordered_cases = runner.randomized("confirmation", iteration, cases)
        sequence_count = len(ordered_cases) + 2
        runner.run_case("confirmation", "matmul", iteration, baseline_case("start"), 0, sequence_count)
        for sequence, case in enumerate(ordered_cases, start=1):
            runner.run_case("confirmation", "matmul", iteration, case, sequence, sequence_count)
        runner.run_case(
            "confirmation",
            "matmul",
            iteration,
            baseline_case("end"),
            sequence_count - 1,
            sequence_count,
        )
    confirmation_records = records_for(runner, "confirmation", "matmul")
    confirmation_ranking_path = runner.output_dir / "stage2-confirmation-ranking.csv"
    confirmation_ranking = write_ranking(
        confirmation_ranking_path,
        confirmation_records,
        {case.label for case in confirmation_candidates} | {production.label},
    )
    # A 000 winner is already represented by the lifecycle baseline and has no
    # meaningful idle-to-active transition. Compare the best two nonzero tuples.
    winners = [weights for weights in confirmation_ranking if weights != (0, 0, 0)][:2]
    runner.log(f"Stage 2 winners: {', '.join(tuple_string(weights) for weights in winners)}")

    lifecycle_cases = []
    for weights in winners:
        label = tuple_label(weights)
        lifecycle_cases.extend(
            [
                static_case(weights),
                Case(
                    label=f"dynamic-000-to-{label}",
                    idle=(0, 0, 0),
                    active=weights,
                    weight_tuple=weights,
                    mode="dynamic",
                ),
                Case(
                    label=f"static-{label}-forced-sync",
                    idle=weights,
                    active=weights,
                    force_sync=True,
                    weight_tuple=weights,
                    mode="static-forced-sync",
                ),
            ]
        )
    lifecycle_baseline = Case(
        label="static-000",
        idle=(0, 0, 0),
        active=(0, 0, 0),
        weight_tuple=(0, 0, 0),
        mode="baseline",
    )
    runner.log("Stage 3: static versus dynamic lifecycle comparison")
    for iteration in range(1, runner.lifecycle_iterations + 1):
        for benchmark in ("matmul", "contention"):
            cases = [lifecycle_baseline, *lifecycle_cases]
            ordered_cases = runner.randomized(f"lifecycle-{benchmark}", iteration, cases)
            for sequence, case in enumerate(ordered_cases):
                runner.run_case(
                    "lifecycle",
                    benchmark,
                    iteration,
                    case,
                    sequence,
                    len(ordered_cases),
                )
    lifecycle_records = records_for(runner, "lifecycle", "matmul") + records_for(
        runner, "lifecycle", "contention"
    )
    lifecycle_summary_path = runner.output_dir / "stage3-static-vs-dynamic.csv"
    write_lifecycle_summary(lifecycle_summary_path, lifecycle_records)

    runner.log(f"Stage 1 ranking: {tuning_ranking_path}")
    runner.log(f"Stage 2 ranking: {confirmation_ranking_path}")
    runner.log(f"Static/dynamic summary: {lifecycle_summary_path}")
    runner.log(f"Raw results: {runner.results_path}")
    runner.log(f"Pytest log: {runner.pytest_log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
