#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tune static MPFE weights for the Device-side Tensor Prefetcher.

Stage 1 exhaustively sweeps static 0/M/H tuples. Stage 2 confirms the six
leaders with more randomized samples. Static 0/0/0 runs before and after each
pass so the rankings can compensate for linear performance drift.
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
MPFE_ENV_PREFIX = "TT_METAL_BENCHMARK_TENSOR_PREFETCHER_"
MPFE_ENV_NAMES = ("FREE_SENDER_WEIGHT", "NOC1_SENDER_WEIGHT", "ORDINARY_WEIGHT")
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
    weights: tuple[int, int, int]


def env_int(name: str, default: int, minimum: int = 1) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def tuple_string(weights: tuple[int, int, int]) -> str:
    return ",".join(str(weight) for weight in weights)


def tuple_label(weights: tuple[int, int, int]) -> str:
    return "".join(str(weight) for weight in weights)


def static_case(weights: tuple[int, int, int]) -> Case:
    return Case(label=f"static-{tuple_label(weights)}", weights=weights)


def baseline_case(position: str) -> Case:
    return Case(label=f"static-000-{position}", weights=(0, 0, 0))


def mean(values: Iterable[float]) -> float:
    return statistics.fmean(values)


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
        self.trace_repeats = env_int("BENCH_TRACE_REPEATS", 100)
        self.seed = env_int("MPFE_RANDOM_SEED", 0x4D504645, minimum=0)
        self.tuning_iterations = env_int("MPFE_TUNING_ITERATIONS", 5)
        self.confirm_iterations = env_int("MPFE_CONFIRM_ITERATIONS", 15)
        self.manifest = {
            "schema_version": 2,
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=TT_METAL_HOME,
                text=True,
            ).strip(),
            "python": self.python,
            "pytest_args": self.pytest_args,
            "matmul_shape": self.matmul_shape,
            "trace_repeats": self.trace_repeats,
            "random_seed": self.seed,
            "tuning_iterations": self.tuning_iterations,
            "confirmation_iterations": self.confirm_iterations,
        }
        self._validate_manifest()
        self.existing: dict[tuple[str, int, str], dict] = {}
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

    def _validate_manifest(self) -> None:
        if self.manifest_path.exists():
            with self.manifest_path.open(encoding="utf-8") as source:
                saved_manifest = json.load(source)
            if saved_manifest != self.manifest:
                raise RuntimeError(
                    f"{self.output_dir} belongs to a different benchmark configuration; "
                    "choose a new OUTPUT_DIR or restore the settings in manifest.json"
                )
            return
        with self.manifest_path.open("w", encoding="utf-8") as output:
            json.dump(self.manifest, output, indent=2, sort_keys=True)
            output.write("\n")

    @staticmethod
    def key(record: dict) -> tuple[str, int, str]:
        return (record["optimization_stage"], record["suite_iteration"], record["run_label"])

    def log(self, message: str) -> None:
        print(message, flush=True)
        with self.summary_log_path.open("a", encoding="utf-8") as output:
            output.write(message + "\n")

    def randomized(self, stage: str, iteration: int, cases: Iterable[Case]) -> list[Case]:
        shuffled = list(cases)
        stage_seed = sum((index + 1) * ord(character) for index, character in enumerate(stage))
        random.Random(self.seed + stage_seed + iteration).shuffle(shuffled)
        return shuffled

    def run_case(
        self,
        stage: str,
        iteration: int,
        case: Case,
        sequence: int,
        sequence_count: int,
    ) -> dict:
        record_key = (stage, iteration, case.label)
        if record_key in self.existing:
            return self.existing[record_key]

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
                "BENCH_TRACE_REPEATS": str(self.trace_repeats),
                "PYTHONPATH": str(TT_METAL_HOME)
                + (f":{environment['PYTHONPATH']}" if environment.get("PYTHONPATH") else ""),
                "TT_METAL_BENCHMARK_RESULT_JSONL": str(self.temp_result_path),
                "TT_METAL_BENCHMARK_RUN_LABEL": case.label,
                "TT_METAL_BENCHMARK_SUITE_ITERATION": str(iteration),
                f"{MPFE_ENV_PREFIX}FREE_SENDER_WEIGHT": str(case.weights[0]),
                f"{MPFE_ENV_PREFIX}NOC1_SENDER_WEIGHT": str(case.weights[1]),
                f"{MPFE_ENV_PREFIX}ORDINARY_WEIGHT": str(case.weights[2]),
            }
        )

        self.temp_result_path.unlink(missing_ok=True)
        command = self.python + [
            "-m",
            "pytest",
            "-q",
            "--tb=short",
            *self.pytest_args,
            MATMUL_TEST,
            "-k",
            f"{self.matmul_shape} and shard_contiguous",
        ]
        with self.pytest_log_path.open("a", encoding="utf-8") as output:
            output.write(
                f"\n== {stage} iteration={iteration} case={case.label} ==\n"
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
            raise RuntimeError(f"{stage}/{iteration}/{case.label} failed; see {self.pytest_log_path}")
        if not self.temp_result_path.exists():
            raise RuntimeError(
                f"{stage}/{iteration}/{case.label} produced no result (possibly skipped); "
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
                "optimization_sequence": sequence,
                "optimization_sequence_count": sequence_count,
                "weight_tuple": list(case.weights),
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
        self.log(f"PASS {stage} iteration={iteration} {case.label}: {record['tflops']:.6f} tflops")
        return record


def write_ranking(path: Path, records: list[dict], candidate_labels: set[str]) -> list[tuple[int, int, int]]:
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
        weights = tuple(samples[0]["weight_tuple"])
        rows.append(
            {
                "tuple": tuple_string(weights),
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
    return [row["_weights"] for row in rows]


def records_for(runner: OptimizationRunner, stage: str) -> list[dict]:
    return [record for record in runner.existing.values() if record["optimization_stage"] == stage]


def run_bracketed_stage(
    runner: OptimizationRunner,
    stage: str,
    iterations: int,
    candidates: list[Case],
) -> None:
    for iteration in range(1, iterations + 1):
        ordered_cases = runner.randomized(stage, iteration, candidates)
        sequence_count = len(ordered_cases) + 2
        runner.run_case(stage, iteration, baseline_case("start"), 0, sequence_count)
        for sequence, case in enumerate(ordered_cases, start=1):
            runner.run_case(stage, iteration, case, sequence, sequence_count)
        runner.run_case(
            stage,
            iteration,
            baseline_case("end"),
            sequence_count - 1,
            sequence_count,
        )


def main() -> int:
    runner = OptimizationRunner(sys.argv[1:])
    runner.log(f"MPFE optimization output: {runner.output_dir}")
    runner.log(
        f"shape={runner.matmul_shape} tuning={runner.tuning_iterations} "
        f"confirmation={runner.confirm_iterations}"
    )

    all_tuples = [(0, medium, high) for medium in range(8) for high in range(medium, 8)]
    tuning_candidates = [static_case(weights) for weights in all_tuples if weights != (0, 0, 0)]
    runner.log(f"Stage 1: exhaustive static sweep ({len(all_tuples)} tuples)")
    run_bracketed_stage(runner, "tuning", runner.tuning_iterations, tuning_candidates)
    tuning_ranking_path = runner.output_dir / "stage1-tuning-ranking.csv"
    tuning_ranking = write_ranking(
        tuning_ranking_path,
        records_for(runner, "tuning"),
        {case.label for case in tuning_candidates} | {"static-000"},
    )
    finalists = tuning_ranking[:6]
    runner.log(f"Stage 1 leaders: {', '.join(tuple_string(weights) for weights in finalists)}")

    confirmation_candidates = [static_case(weights) for weights in finalists]
    runner.log("Stage 2: randomized finalist confirmation")
    run_bracketed_stage(runner, "confirmation", runner.confirm_iterations, confirmation_candidates)
    confirmation_ranking_path = runner.output_dir / "stage2-confirmation-ranking.csv"
    confirmation_ranking = write_ranking(
        confirmation_ranking_path,
        records_for(runner, "confirmation"),
        {case.label for case in confirmation_candidates},
    )
    runner.log(f"Stage 2 winner: {tuple_string(confirmation_ranking[0])}")
    runner.log(f"Stage 1 ranking: {tuning_ranking_path}")
    runner.log(f"Stage 2 ranking: {confirmation_ranking_path}")
    runner.log(f"Raw results: {runner.results_path}")
    runner.log(f"Pytest log: {runner.pytest_log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
