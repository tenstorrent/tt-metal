#!/usr/bin/env python3
"""Run MPFE policies through the real Llama-3.1-8B Tensor Prefetcher demo."""

from __future__ import annotations

import csv
import json
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEMO = ROOT / "models/tt_transformers/demo/simple_text_demo.py"
ENV_PREFIX = "TT_METAL_BENCHMARK_TENSOR_PREFETCHER_"
ENV_SUFFIXES = (
    "FREE_SENDER_WEIGHT",
    "NOC1_SENDER_WEIGHT",
    "ORDINARY_WEIGHT",
    "IDLE_FREE_SENDER_WEIGHT",
    "IDLE_NOC1_SENDER_WEIGHT",
    "IDLE_ORDINARY_WEIGHT",
)
METRIC_RE = re.compile(
    r"Average speed: ([0-9.]+)ms @ ([0-9.]+) tok/s/user \(([0-9.]+) tok/s throughput\)"
)
POLICY_RE = re.compile(
    r"\[mpfe_model_benchmark\] controller=independent idle=(\d)/(\d)/(\d) active=(\d)/(\d)/(\d)"
)

DEFAULT_HIGHS = (3, 5, 7)
MODES = ("static", "dynamic")


@dataclass(frozen=True)
class Case:
    workload: str
    high: int
    mode: str
    iteration: int

    @property
    def idle_weights(self) -> tuple[int, int, int]:
        return (self.high, self.high, self.high)

    @property
    def active_weights(self) -> tuple[int, int, int]:
        return self.idle_weights if self.mode == "static" else (0, 0, self.high)

    @property
    def label(self) -> str:
        idle = "".join(str(value) for value in self.idle_weights)
        active = "".join(str(value) for value in self.active_weights)
        policy = idle if self.mode == "static" else f"{idle}-to-{active}"
        return f"{self.workload}-{self.mode}-{policy}-i{self.iteration}"


def parse_csv(name: str, default: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in os.environ.get(name, default).split(",") if value.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{name} must contain unique comma-separated values")
    return values


def parse_highs() -> tuple[int, ...]:
    raw = os.environ.get("MPFE_MODEL_HIGHS")
    if raw is None:
        return DEFAULT_HIGHS
    highs = []
    for value in raw.split(","):
        value = value.strip()
        if len(value) != 1 or value < "0" or value > "7":
            raise ValueError("MPFE_MODEL_HIGHS must contain single digits in [0, 7], such as 3,5,7")
        highs.append(int(value))
    if not highs or len(highs) != len(set(highs)):
        raise ValueError("MPFE_MODEL_HIGHS must contain unique values")
    return tuple(highs)


def case_environment(case: Case) -> dict[str, str]:
    environment = os.environ.copy()
    for suffix in ENV_SUFFIXES:
        environment.pop(f"{ENV_PREFIX}{suffix}", None)

    active_free, active_noc1, active_ordinary = case.active_weights
    idle_free, idle_noc1, idle_ordinary = case.idle_weights
    environment.update(
        {
            "ARCH_NAME": "blackhole",
            "PYTHONPATH": str(ROOT)
            + (f":{environment['PYTHONPATH']}" if environment.get("PYTHONPATH") else ""),
            f"{ENV_PREFIX}FREE_SENDER_WEIGHT": str(active_free),
            f"{ENV_PREFIX}NOC1_SENDER_WEIGHT": str(active_noc1),
            f"{ENV_PREFIX}ORDINARY_WEIGHT": str(active_ordinary),
            f"{ENV_PREFIX}IDLE_FREE_SENDER_WEIGHT": str(idle_free),
            f"{ENV_PREFIX}IDLE_NOC1_SENDER_WEIGHT": str(idle_noc1),
            f"{ENV_PREFIX}IDLE_ORDINARY_WEIGHT": str(idle_ordinary),
        }
    )
    return environment


def test_selector(workload: str) -> str:
    if workload not in ("batch-1", "batch-32"):
        raise ValueError("MPFE_MODEL_WORKLOADS supports batch-1 and batch-32")
    exclusion = " and not log-probs" if workload == "batch-32" else ""
    return f"performance and {workload}{exclusion}"


def load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    records = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            records[record["label"]] = record
    return records


def run_case(case: Case, output_dir: Path) -> dict:
    log_path = output_dir / "logs" / f"{case.label}.log"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--tb=short",
        str(DEMO),
        "-k",
        test_selector(case.workload),
        "--use_prefetcher",
        "True",
        "--skip_perf_report",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=case_environment(case),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(
        json.dumps(
            {
                "label": case.label,
                "command": command,
                "idle_weights": case.idle_weights,
                "active_weights": case.active_weights,
                "mode": case.mode,
            },
            sort_keys=True,
        )
        + "\n"
        + completed.stdout,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{case.label} failed; see {log_path}")

    policy_markers = POLICY_RE.findall(completed.stdout)
    if len(policy_markers) != 1:
        raise RuntimeError(
            f"{case.label} emitted {len(policy_markers)} MPFE policy markers; rebuild the host library and see {log_path}"
        )
    marker = policy_markers[0]
    observed_idle = tuple(int(value) for value in marker[:3])
    observed_active = tuple(int(value) for value in marker[3:6])
    if (observed_idle, observed_active) != (case.idle_weights, case.active_weights):
        raise RuntimeError(
            f"{case.label} applied idle={observed_idle} active={observed_active}; see {log_path}"
        )

    metrics = METRIC_RE.findall(completed.stdout)
    if len(metrics) != 1:
        raise RuntimeError(f"{case.label} emitted {len(metrics)} final metrics; see {log_path}")
    latency_ms, tok_s_user, tok_s = (float(value) for value in metrics[0])
    return {
        "label": case.label,
        "workload": case.workload,
        "mode": case.mode,
        "high": case.high,
        "active_weights": list(case.active_weights),
        "idle_weights": list(case.idle_weights),
        "iteration": case.iteration,
        "decode_latency_ms": latency_ms,
        "decode_tok_s_user": tok_s_user,
        "decode_tok_s": tok_s,
        "log": str(log_path),
    }


def write_summary(records: list[dict], path: Path) -> None:
    groups: dict[tuple, list[dict]] = {}
    for record in records:
        key = (
            record["workload"],
            record["mode"],
            record["high"],
            tuple(record["idle_weights"]),
            tuple(record["active_weights"]),
        )
        groups.setdefault(key, []).append(record)

    rows = []
    for (workload, mode, high, idle_weights, active_weights), values in groups.items():
        rows.append(
            {
                "workload": workload,
                "mode": mode,
                "high": high,
                "idle_weights": "".join(str(value) for value in idle_weights),
                "active_weights": "".join(str(value) for value in active_weights),
                "samples": len(values),
                "mean_decode_latency_ms": sum(value["decode_latency_ms"] for value in values) / len(values),
                "mean_decode_tok_s_user": sum(value["decode_tok_s_user"] for value in values) / len(values),
                "mean_decode_tok_s": sum(value["decode_tok_s"] for value in values) / len(values),
            }
        )
    rows.sort(key=lambda row: (row["workload"], -row["mean_decode_tok_s"], row["mode"], row["high"]))
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if "Llama-3.1-8B" not in os.environ.get("HF_MODEL", ""):
        raise RuntimeError("Set HF_MODEL to a Llama-3.1-8B checkpoint")

    workloads = parse_csv("MPFE_MODEL_WORKLOADS", "batch-1,batch-32")
    modes = parse_csv("MPFE_MODEL_MODES", ",".join(MODES))
    if any(mode not in MODES for mode in modes):
        raise ValueError(f"MPFE_MODEL_MODES values must come from {MODES}")
    highs = parse_highs()
    iterations = int(os.environ.get("MPFE_MODEL_ITERATIONS", "1"))
    if iterations < 1:
        raise ValueError("MPFE_MODEL_ITERATIONS must be positive")

    output_dir = Path(os.environ.get("MPFE_MODEL_OUTPUT_DIR", ROOT / "generated/mpfe-llama-model-independent-drisc"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    results_path = output_dir / "results.jsonl"
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "schema_version": 3,
        "benchmark": "llama-3.1-8b-simple-text-demo-tensor-prefetcher-mpfe",
        "hf_model": os.environ["HF_MODEL"],
        "mesh_device": os.environ.get("MESH_DEVICE"),
        "workloads": list(workloads),
        "modes": list(modes),
        "highs": list(highs),
        "iterations": iterations,
        "seed": int(os.environ.get("MPFE_MODEL_SEED", "49778")),
    }
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError(f"{output_dir} belongs to a different benchmark configuration")
    elif results_path.exists():
        raise RuntimeError(f"{results_path} exists without a manifest; choose a new MPFE_MODEL_OUTPUT_DIR")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    existing = load_existing(results_path)

    cases = [
        Case(workload, high, mode, iteration)
        for iteration in range(iterations)
        for workload in workloads
        for high in highs
        for mode in modes
    ]
    random.Random(manifest["seed"]).shuffle(cases)

    for sequence, case in enumerate(cases, start=1):
        if case.label in existing:
            print(f"[{sequence}/{len(cases)}] resume {case.label}", flush=True)
            continue
        print(f"[{sequence}/{len(cases)}] run {case.label}", flush=True)
        record = run_case(case, output_dir)
        with results_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")
        existing[case.label] = record

    write_summary(list(existing.values()), output_dir / "summary.csv")
    print(f"Results: {results_path}")
    print(f"Summary: {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
