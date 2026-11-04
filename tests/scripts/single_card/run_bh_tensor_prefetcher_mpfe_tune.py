#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tune Tensor Prefetcher MPFE weights around an unchanged model command."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


POLICY_MARKER = "TENSOR_PREFETCHER_MPFE_POLICY "
TIMER_MARKER = "TENSOR_PREFETCHER_MPFE_ACTIVE_LIFETIME -- elapsed: "
ENV_PREFIX = "TT_METAL_BENCHMARK_TENSOR_PREFETCHER_"
ENV_NAMES = (
    f"{ENV_PREFIX}ENABLE",
    f"{ENV_PREFIX}FREE_SENDER_WEIGHT",
    f"{ENV_PREFIX}NOC1_SENDER_WEIGHT",
    f"{ENV_PREFIX}ORDINARY_WEIGHT",
    f"{ENV_PREFIX}DYNAMIC_MPFE_WEIGHTING",
)
SCHEMA_VERSION = 1


@dataclass(frozen=True, order=True)
class Candidate:
    dynamic: bool
    low: int
    medium: int
    high: int

    @property
    def label(self) -> str:
        mode = "dynamic" if self.dynamic else "static"
        return f"{mode}-{self.low}{self.medium}{self.high}"

    @property
    def idle(self) -> tuple[int, int, int]:
        return (self.high, self.high, self.high) if self.dynamic else self.active

    @property
    def active(self) -> tuple[int, int, int]:
        return self.low, self.medium, self.high


def parse_candidate_values(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(sorted({int(value.strip()) for value in raw.split(",") if value.strip()}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("candidate values must be comma-separated integers") from error
    if not values or values[0] < 0 or values[-1] > 7:
        raise argparse.ArgumentTypeError("candidate values must contain at least one value in [0, 7]")
    return values


def generate_candidates(values: Sequence[int]) -> list[Candidate]:
    weights = [(low, medium, high) for low in values for medium in values for high in values if low <= medium <= high]
    return [Candidate(dynamic, *weight) for dynamic in (False, True) for weight in weights]


def candidate_environment(candidate: Candidate, base: dict[str, str] | None = None) -> dict[str, str]:
    environment = dict(os.environ if base is None else base)
    for name in ENV_NAMES:
        environment.pop(name, None)
    environment.update(
        {
            f"{ENV_PREFIX}ENABLE": "1",
            f"{ENV_PREFIX}FREE_SENDER_WEIGHT": str(candidate.low),
            f"{ENV_PREFIX}NOC1_SENDER_WEIGHT": str(candidate.medium),
            f"{ENV_PREFIX}ORDINARY_WEIGHT": str(candidate.high),
            f"{ENV_PREFIX}DYNAMIC_MPFE_WEIGHTING": "1" if candidate.dynamic else "0",
        }
    )
    return environment


def _parse_weight_tuple(value: str) -> tuple[int, int, int]:
    fields = value.split("/")
    if len(fields) != 3:
        raise ValueError(f"invalid weight tuple: {value!r}")
    weights = tuple(int(field) for field in fields)
    if any(weight < 0 or weight > 7 for weight in weights):
        raise ValueError(f"weight outside [0, 7]: {value!r}")
    return weights


def parse_policy_markers(output: str) -> list[dict[str, object]]:
    policies: list[dict[str, object]] = []
    for line in output.splitlines():
        marker_index = line.find(POLICY_MARKER)
        if marker_index < 0:
            continue
        fields = {}
        for field in line[marker_index + len(POLICY_MARKER) :].split():
            if "=" in field:
                key, value = field.split("=", 1)
                fields[key] = value
        if not {"active", "dynamic"} <= fields.keys():
            raise ValueError(f"incomplete policy marker: {line}")
        if fields["dynamic"] not in {"0", "1"}:
            raise ValueError(f"invalid dynamic policy value: {line}")
        policies.append(
            {
                "active": _parse_weight_tuple(fields["active"]),
                "dynamic": fields["dynamic"] == "1",
            }
        )
    return policies


def parse_timer_total_us(output: str) -> tuple[float, int]:
    scale_to_us = {"ns": 0.001, "us": 1.0, "µs": 1.0, "ms": 1_000.0, "s": 1_000_000.0}
    durations = []
    for line in output.splitlines():
        marker_index = line.find(TIMER_MARKER)
        if marker_index < 0:
            continue
        value_with_unit = line[marker_index + len(TIMER_MARKER) :].strip()
        match = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*(ns|us|µs|ms|s)\b", value_with_unit)
        if match is None:
            raise ValueError(f"unknown timer unit: {line}")
        number, unit = match.groups()
        durations.append(float(number) * scale_to_us[unit])
    if not durations:
        raise ValueError(f"model output did not contain {TIMER_MARKER!r}")
    return sum(durations), len(durations)


def verify_policy(candidate: Candidate, policies: Sequence[dict[str, object]]) -> None:
    if not policies:
        raise ValueError(f"model output did not contain {POLICY_MARKER!r}")
    expected = {"active": candidate.active, "dynamic": candidate.dynamic}
    for policy in policies:
        if policy != expected:
            raise ValueError(f"expected policy {expected}, observed {policy}")


def rank_candidates(records: Iterable[dict[str, object]], phase: str) -> list[dict[str, object]]:
    samples: dict[str, list[float]] = {}
    candidates: dict[str, dict[str, object]] = {}
    for record in records:
        if record.get("phase") != phase or record.get("status") != "passed":
            continue
        label = str(record["candidate"]["label"])
        samples.setdefault(label, []).append(float(record["elapsed_us"]))
        candidates[label] = record["candidate"]
    ranking = []
    for label, elapsed in samples.items():
        ranking.append(
            {
                "rank": 0,
                "label": label,
                "median_us": statistics.median(elapsed),
                "min_us": min(elapsed),
                "max_us": max(elapsed),
                "runs": len(elapsed),
                "candidate": candidates[label],
            }
        )
    ranking.sort(key=lambda row: (row["median_us"], row["label"]))
    for index, row in enumerate(ranking, start=1):
        row["rank"] = index
    return ranking


def _git_revision() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _hardware_metadata() -> dict[str, object]:
    metadata: dict[str, object] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "arch_name": os.environ.get("ARCH_NAME"),
        "tt_metal_home": os.environ.get("TT_METAL_HOME"),
    }
    tt_smi = shutil.which("tt-smi")
    if tt_smi:
        try:
            result = subprocess.run([tt_smi, "-s"], text=True, capture_output=True, timeout=5, check=False)
            if result.returncode == 0:
                metadata["tt_smi"] = result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    return metadata


def _manifest(
    command: Sequence[str],
    values: Sequence[int],
    search_runs: int,
    rank_runs: int,
    top_k: int,
    seed: int,
    timeout_seconds: float,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "command": list(command),
        "candidate_values": list(values),
        "search_runs": search_runs,
        "rank_runs": rank_runs,
        "top_k": top_k,
        "seed": seed,
        "timeout_seconds": timeout_seconds,
        "git_revision": _git_revision(),
        "hardware": _hardware_metadata(),
    }


def validate_or_write_manifest(path: Path, manifest: dict[str, object], resume: bool) -> None:
    if path.exists():
        existing = json.loads(path.read_text())
        if not resume:
            raise RuntimeError(f"{path} already exists; pass --resume to continue it")
        existing_comparable = dict(existing)
        manifest_comparable = dict(manifest)
        for comparable in (existing_comparable, manifest_comparable):
            hardware = dict(comparable.get("hardware", {}))
            hardware.pop("tt_smi", None)
            comparable["hardware"] = hardware
        if existing_comparable != manifest_comparable:
            raise RuntimeError(
                "resume manifest does not match the current command, configuration, revision, or hardware"
            )
        return
    if resume:
        raise RuntimeError(f"cannot resume: {path} does not exist")
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _load_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _candidate_dict(candidate: Candidate) -> dict[str, object]:
    data = asdict(candidate)
    data.update({"label": candidate.label, "idle": list(candidate.idle), "active": list(candidate.active)})
    return data


def run_once(
    candidate: Candidate,
    phase: str,
    run_index: int,
    command: Sequence[str],
    log_dir: Path,
    timeout_seconds: float,
) -> dict[str, object]:
    log_path = log_dir / f"{phase}-{candidate.label}-r{run_index}.log"
    try:
        result = subprocess.run(
            command,
            env=candidate_environment(candidate),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
        output = result.stdout
        returncode = result.returncode
        timeout_error = None
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        returncode = None
        timeout_error = f"model command timed out after {timeout_seconds:g} seconds"
    log_path.write_text(output)
    record: dict[str, object] = {
        "phase": phase,
        "run_index": run_index,
        "candidate": _candidate_dict(candidate),
        "returncode": returncode,
        "log": str(log_path),
        "status": "failed",
    }
    if timeout_error is not None:
        record["error"] = timeout_error
        return record
    if returncode != 0:
        record["error"] = f"model command exited with status {returncode}"
        return record
    try:
        policies = parse_policy_markers(output)
        verify_policy(candidate, policies)
        elapsed_us, lifetime_count = parse_timer_total_us(output)
        if len(policies) != lifetime_count:
            raise ValueError(
                f"policy/timer marker count mismatch: observed {len(policies)} policies and "
                f"{lifetime_count} lifetimes"
            )
    except ValueError as error:
        record["error"] = str(error)
        return record
    record.update(
        {
            "status": "passed",
            "elapsed_us": elapsed_us,
            "lifetime_count": lifetime_count,
            "policy_count": len(policies),
        }
    )
    return record


def _append_record(path: Path, record: dict[str, object]) -> None:
    with path.open("a") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")


def _completed_keys(records: Iterable[dict[str, object]]) -> set[tuple[str, str, int]]:
    return {
        (str(record["phase"]), str(record["candidate"]["label"]), int(record["run_index"]))
        for record in records
        if record.get("status") == "passed"
    }


def _run_jobs(
    jobs: list[tuple[Candidate, str, int]],
    command: Sequence[str],
    records: list[dict[str, object]],
    results_path: Path,
    log_dir: Path,
    rng: random.Random,
    timeout_seconds: float,
) -> None:
    rng.shuffle(jobs)
    completed = _completed_keys(records)
    for job_number, (candidate, phase, run_index) in enumerate(jobs, start=1):
        key = phase, candidate.label, run_index
        if key in completed:
            continue
        print(f"[{job_number}/{len(jobs)}] {phase} {candidate.label} run {run_index}", flush=True)
        record = run_once(candidate, phase, run_index, command, log_dir, timeout_seconds)
        _append_record(results_path, record)
        records.append(record)
        if record["status"] != "passed":
            raise RuntimeError(f"{candidate.label} failed; see {record['log']}: {record.get('error')}")


def write_ranking(path: Path, ranking: Sequence[dict[str, object]]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["rank", "candidate", "mode", "active", "idle", "runs", "median_us", "min_us", "max_us"])
        for row in ranking:
            candidate = row["candidate"]
            writer.writerow(
                [
                    row["rank"],
                    row["label"],
                    "dynamic" if candidate["dynamic"] else "static",
                    "/".join(str(value) for value in candidate["active"]),
                    "/".join(str(value) for value in candidate["idle"]),
                    row["runs"],
                    row["median_us"],
                    row["min_us"],
                    row["max_us"],
                ]
            )


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("generated/bh-tensor-prefetcher-mpfe-tune"))
    parser.add_argument("--search-runs", type=int, default=3)
    parser.add_argument("--rank-runs", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-values", type=parse_candidate_values, default=parse_candidate_values("0,1,3,5,7"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="model command, preceded by --")
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("provide a model command after --")
    if args.search_runs < 1 or args.rank_runs < 1 or args.top_k < 1 or args.timeout_seconds <= 0:
        parser.error("--search-runs, --rank-runs, --top-k, and --timeout-seconds must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    results_path = args.output_dir / "results.jsonl"

    manifest = _manifest(
        args.command,
        args.candidate_values,
        args.search_runs,
        args.rank_runs,
        args.top_k,
        args.seed,
        args.timeout_seconds,
    )
    if not args.resume and results_path.exists():
        raise RuntimeError(f"{results_path} already exists; pass --resume to continue it")
    validate_or_write_manifest(manifest_path, manifest, args.resume)
    records = _load_records(results_path)
    candidates = generate_candidates(args.candidate_values)
    rng = random.Random(args.seed)

    search_jobs = [
        (candidate, "search", run_index) for candidate in candidates for run_index in range(args.search_runs)
    ]
    _run_jobs(search_jobs, args.command, records, results_path, log_dir, rng, args.timeout_seconds)
    search_ranking = rank_candidates(records, "search")
    leaders = search_ranking[: min(args.top_k, len(search_ranking))]
    if not leaders:
        raise RuntimeError("search produced no successful candidates")

    candidate_by_label = {candidate.label: candidate for candidate in candidates}
    rank_jobs = [
        (candidate_by_label[row["label"]], "rank", run_index) for row in leaders for run_index in range(args.rank_runs)
    ]
    _run_jobs(rank_jobs, args.command, records, results_path, log_dir, rng, args.timeout_seconds)
    final_ranking = rank_candidates(records, "rank")
    if not final_ranking:
        raise RuntimeError("ranking produced no successful candidates")
    write_ranking(args.output_dir / "ranking.csv", final_ranking)
    print(f"Best policy: {final_ranking[0]['label']} ({final_ranking[0]['median_us']:.1f} us median)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
