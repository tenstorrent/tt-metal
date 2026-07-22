#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Multi-process driver for the remote JIT compile-server stress test.

Spawns N copies of the compile-stress gtest (mock-device-backed), aligns
their timed sections via a wall-clock T_ZERO rendezvous, and aggregates
results from each client's JSON output.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from contextlib import ExitStack
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


# Env vars that would route MetalContext off the mock path; we strip them from
# every child env so the pre-flight check can catch misconfiguration before
# spawning the full cohort.
SILICON_FORCING_ENV_VARS = (
    "TT_METAL_SIMULATOR",
    "TT_METAL_EMULE_MODE",
    "TT_METAL_MOCK_CLUSTER_DESC_PATH",
)


@dataclass
class ClientSpec:
    client_id: int
    endpoints: list[str]
    private_seed: int
    cache_dir: Path
    output_file: Path
    stdout_log: Path
    stderr_log: Path


@dataclass
class SpawnedClient:
    spec: ClientSpec
    proc: subprocess.Popen
    _log_files: ExitStack

    def wait(self) -> int:
        try:
            return self.proc.wait()
        finally:
            self._log_files.close()


@dataclass
class ClientResult:
    client_id: int
    private_seed: int
    shared_fraction: float
    shared_seed: int
    num_shared: int
    arch: str
    num_chips: int
    num_kernels: int
    num_programs: int
    target_device_type: str
    start_unix_ns: int
    end_unix_ns: int
    total_elapsed_ms: float

    @classmethod
    def from_json_file(cls, path: Path) -> "ClientResult":
        with open(path) as f:
            data = json.load(f)
        return cls(
            client_id=int(data["client_id"]),
            private_seed=int(data["private_seed"]),
            shared_fraction=float(data["shared_fraction"]),
            shared_seed=int(data["shared_seed"]),
            num_shared=int(data["num_shared"]),
            arch=str(data["arch"]),
            num_chips=int(data["num_chips"]),
            num_kernels=int(data["num_kernels"]),
            num_programs=int(data["num_programs"]),
            target_device_type=str(data["target_device_type"]),
            start_unix_ns=int(data["start_unix_ns"]),
            end_unix_ns=int(data["end_unix_ns"]),
            total_elapsed_ms=float(data["total_elapsed_ms"]),
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--test-binary",
        required=True,
        type=Path,
        help="Path to unit_tests_jit_build (the compiled gtest binary).",
    )
    p.add_argument("--num-clients", type=int, default=1, help="Number of stress-client processes to spawn.")
    p.add_argument(
        "--kernels-per-client",
        type=int,
        default=1000,
        help="TT_METAL_COMPILE_STRESS_NUM_KERNELS passed to each client.",
    )
    p.add_argument(
        "--endpoints",
        type=str,
        required=True,
        help="Comma-separated remote JIT server endpoints (host:port,host:port,...).",
    )
    p.add_argument(
        "--endpoint-mode",
        choices=("shared", "split"),
        default="shared",
        help=(
            "shared: every client gets the full endpoint list (default). "
            "split: endpoints partitioned round-robin across clients (1-app-per-server isolation)."
        ),
    )
    p.add_argument(
        "--shared-fraction",
        type=float,
        default=0.0,
        help="Fraction in [0,1] of kernels in the shared pool compiled identically by every client.",
    )
    p.add_argument(
        "--shared-seed",
        type=int,
        default=0,
        help="Seed used for the shared pool; identical across clients.",
    )
    p.add_argument("--arch", default="wormhole_b0", help="Mock architecture.")
    p.add_argument("--num-chips", type=int, default=1, help="Mock num_chips.")
    p.add_argument(
        "--startup-slack-sec",
        type=float,
        default=5.0,
        help="Seconds of head-start given to every client before T_ZERO. Increase for heavy fixtures / large N.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/compile_stress_harness"),
        help="Directory for per-client logs, JSON results, isolated caches, and the aggregate summary.",
    )
    p.add_argument(
        "--keep-output",
        action="store_true",
        help="Do NOT wipe --output-dir at start. Default is to wipe for a clean run.",
    )
    p.add_argument(
        "--preflight-kernels",
        type=int,
        default=8,
        help="Kernel count for the pre-flight mock-mode check run.",
    )
    p.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the pre-flight dry-run. Only use if you've already validated the env.",
    )
    p.add_argument(
        "--master-seed",
        type=int,
        default=None,
        help="Seed the harness RNG that picks per-client private seeds (for reproducibility).",
    )
    return p.parse_args(argv)


def partition_endpoints(endpoints: list[str], num_clients: int, mode: str) -> list[list[str]]:
    if mode == "shared":
        return [list(endpoints) for _ in range(num_clients)]
    buckets: list[list[str]] = [[] for _ in range(num_clients)]
    for i, ep in enumerate(endpoints):
        buckets[i % num_clients].append(ep)
    empty = [i for i, b in enumerate(buckets) if not b]
    if empty:
        raise SystemExit(
            f"endpoint-mode=split with {num_clients} clients and {len(endpoints)} endpoints leaves "
            f"clients {empty} with no endpoint. Use --endpoint-mode shared or add more endpoints."
        )
    return buckets


def build_client_env(
    spec: ClientSpec,
    *,
    num_kernels: int,
    arch: str,
    num_chips: int,
    shared_fraction: float,
    shared_seed: int,
    t_zero_ns: int = 0,
) -> dict[str, str]:
    env = dict(os.environ)
    for var in SILICON_FORCING_ENV_VARS:
        env.pop(var, None)
    env.update(
        {
            "TT_METAL_JIT_SERVER_ENABLE": "1",
            "TT_METAL_JIT_SERVER_ENDPOINTS": ",".join(spec.endpoints),
            "TT_METAL_CACHE": str(spec.cache_dir),
            "TT_METAL_COMPILE_STRESS_NUM_KERNELS": str(num_kernels),
            "TT_METAL_COMPILE_STRESS_ARCH": arch,
            "TT_METAL_COMPILE_STRESS_NUM_CHIPS": str(num_chips),
            "TT_METAL_COMPILE_STRESS_CLIENT_ID": str(spec.client_id),
            "TT_METAL_COMPILE_STRESS_SEED": str(spec.private_seed),
            "TT_METAL_COMPILE_STRESS_SHARED_FRACTION": f"{shared_fraction:.6f}",
            "TT_METAL_COMPILE_STRESS_SHARED_SEED": str(shared_seed),
            "TT_METAL_COMPILE_STRESS_OUTPUT": str(spec.output_file),
        }
    )
    if t_zero_ns != 0:
        env["TT_METAL_COMPILE_STRESS_T_ZERO_NS"] = str(t_zero_ns)
    return env


def make_client_specs(args: argparse.Namespace, endpoints: list[str]) -> list[ClientSpec]:
    rng = random.Random(args.master_seed) if args.master_seed is not None else random.Random()
    buckets = partition_endpoints(endpoints, args.num_clients, args.endpoint_mode)
    specs: list[ClientSpec] = []
    for i in range(args.num_clients):
        client_dir = args.output_dir / f"client_{i:03d}"
        cache_dir = client_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        specs.append(
            ClientSpec(
                client_id=i,
                endpoints=buckets[i],
                private_seed=rng.randint(1, 2**31 - 1),
                cache_dir=cache_dir,
                output_file=client_dir / "result.json",
                stdout_log=client_dir / "stdout.log",
                stderr_log=client_dir / "stderr.log",
            )
        )
    return specs


def spawn_client(test_binary: Path, spec: ClientSpec, env: dict[str, str]) -> SpawnedClient:
    cmd = [
        str(test_binary),
        "--gtest_also_run_disabled_tests",
        "--gtest_filter=*TensixCompileStress*",
        "--gtest_color=no",
    ]
    with ExitStack() as log_files:
        stdout_fh = log_files.enter_context(open(spec.stdout_log, "w"))
        stderr_fh = log_files.enter_context(open(spec.stderr_log, "w"))
        proc = subprocess.Popen(cmd, env=env, stdout=stdout_fh, stderr=stderr_fh)
        return SpawnedClient(spec=spec, proc=proc, _log_files=log_files.pop_all())


def wait_all(clients: list[SpawnedClient]) -> list[int]:
    exit_codes: list[int] = []
    for client in clients:
        rc = client.wait()
        exit_codes.append(rc)
        if rc != 0:
            print(
                f"[harness] client {client.spec.client_id} exited with code {rc}. "
                f"See {client.spec.stdout_log} and {client.spec.stderr_log}.",
                file=sys.stderr,
            )
    return exit_codes


def run_preflight(args: argparse.Namespace, endpoints: list[str]) -> None:
    print(f"[harness] Running pre-flight mock-mode check with {args.preflight_kernels} kernels ...")
    preflight_dir = args.output_dir / "preflight"
    cache_dir = preflight_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    spec = ClientSpec(
        client_id=0,
        endpoints=endpoints,
        private_seed=42,
        cache_dir=cache_dir,
        output_file=preflight_dir / "result.json",
        stdout_log=preflight_dir / "stdout.log",
        stderr_log=preflight_dir / "stderr.log",
    )
    env = build_client_env(
        spec,
        num_kernels=args.preflight_kernels,
        arch=args.arch,
        num_chips=args.num_chips,
        shared_fraction=0.0,
        shared_seed=0,
    )
    client = spawn_client(args.test_binary, spec, env)
    rc = client.wait()
    if rc != 0:
        raise SystemExit(
            f"[harness] pre-flight client failed with exit code {rc}. " f"See {spec.stdout_log} and {spec.stderr_log}."
        )
    if not spec.output_file.exists():
        raise SystemExit(f"[harness] pre-flight client produced no JSON at {spec.output_file}.")
    result = ClientResult.from_json_file(spec.output_file)
    if result.target_device_type != "Mock":
        raise SystemExit(
            f"[harness] pre-flight target_device_type={result.target_device_type!r} (expected 'Mock'). "
            "Aborting: running multiple silicon clients on one host would fight over UMD driver locks."
        )
    print(f"[harness] pre-flight ok (target_device_type=Mock, elapsed={result.total_elapsed_ms:.1f}ms)")


def aggregate(results: list[ClientResult]) -> dict:
    assert results, "aggregate called with no results"
    total_kernels = sum(r.num_kernels for r in results)
    start_min = min(r.start_unix_ns for r in results)
    start_max = max(r.start_unix_ns for r in results)
    end_max = max(r.end_unix_ns for r in results)
    makespan_ns = end_max - start_min
    makespan_s = makespan_ns / 1e9 if makespan_ns > 0 else 0.0
    aggregate_throughput = total_kernels / makespan_s if makespan_s > 0 else 0.0

    per_client_throughput = [
        (r.num_kernels * 1000.0) / r.total_elapsed_ms if r.total_elapsed_ms > 0 else 0.0 for r in results
    ]

    sanity_warnings: list[str] = []
    for field_name in ("shared_fraction", "shared_seed", "num_shared", "num_kernels"):
        values = {getattr(r, field_name) for r in results}
        if len(values) > 1:
            sanity_warnings.append(f"{field_name} mismatch across clients: {values}")

    return {
        "num_clients": len(results),
        "total_kernels": total_kernels,
        "makespan_ms": makespan_ns / 1e6,
        "start_skew_ms": (start_max - start_min) / 1e6,
        "aggregate_throughput_kernels_per_sec": aggregate_throughput,
        "per_client_throughput_kernels_per_sec": {
            "min": min(per_client_throughput) if per_client_throughput else 0.0,
            "median": statistics.median(per_client_throughput) if per_client_throughput else 0.0,
            "max": max(per_client_throughput) if per_client_throughput else 0.0,
            "stdev": statistics.pstdev(per_client_throughput) if len(per_client_throughput) > 1 else 0.0,
            "values": per_client_throughput,
        },
        "sanity_warnings": sanity_warnings,
        "clients": [
            {
                "client_id": r.client_id,
                "num_kernels": r.num_kernels,
                "num_shared": r.num_shared,
                "num_programs": r.num_programs,
                "total_elapsed_ms": r.total_elapsed_ms,
                "throughput_kernels_per_sec": (r.num_kernels * 1000.0) / r.total_elapsed_ms
                if r.total_elapsed_ms > 0
                else 0.0,
                "start_unix_ns": r.start_unix_ns,
                "end_unix_ns": r.end_unix_ns,
            }
            for r in results
        ],
    }


def print_report(summary: dict, endpoints: list[str], args: argparse.Namespace) -> None:
    print()
    print("=" * 72)
    print("Compile-stress harness summary")
    print("=" * 72)
    print(f"clients:               {summary['num_clients']}")
    print(f"kernels per client:    {args.kernels_per_client}")
    print(f"total kernels:         {summary['total_kernels']}")
    print(f"endpoints:             {','.join(endpoints)}  ({args.endpoint_mode})")
    print(f"shared_fraction:       {args.shared_fraction:.3f}")
    print()
    print(f"makespan:              {summary['makespan_ms']:.1f} ms")
    print(f"start_skew:            {summary['start_skew_ms']:.1f} ms   (want << makespan)")
    print(f"aggregate throughput:  {summary['aggregate_throughput_kernels_per_sec']:.1f} kernels/sec")
    pct = summary["per_client_throughput_kernels_per_sec"]
    print(
        f"per-client throughput: min={pct['min']:.1f}  median={pct['median']:.1f}  max={pct['max']:.1f}  "
        f"stdev={pct['stdev']:.1f}  kernels/sec"
    )
    if summary["sanity_warnings"]:
        print()
        print("Sanity warnings:")
        for w in summary["sanity_warnings"]:
            print(f"  - {w}")
    print()


def collect_results(specs: list[ClientSpec]) -> tuple[list[ClientResult], list[int]]:
    results: list[ClientResult] = []
    missing: list[int] = []
    for spec in specs:
        if not spec.output_file.exists():
            missing.append(spec.client_id)
            continue
        try:
            results.append(ClientResult.from_json_file(spec.output_file))
        except Exception as e:
            print(f"[harness] failed to parse {spec.output_file}: {e}", file=sys.stderr)
            missing.append(spec.client_id)
    return results, missing


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.test_binary.exists():
        raise SystemExit(f"--test-binary not found: {args.test_binary}")

    endpoints = [e.strip() for e in args.endpoints.split(",") if e.strip()]
    if not endpoints:
        raise SystemExit("--endpoints must contain at least one host:port entry")
    if args.num_clients < 1:
        raise SystemExit("--num-clients must be >= 1")
    if not (0.0 <= args.shared_fraction <= 1.0):
        raise SystemExit("--shared-fraction must be in [0, 1]")

    if args.output_dir.exists() and not args.keep_output:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_preflight:
        run_preflight(args, endpoints)

    specs = make_client_specs(args, endpoints)

    t_zero_ns = time.time_ns() + int(args.startup_slack_sec * 1e9)
    print(
        f"[harness] Spawning {args.num_clients} clients, "
        f"T_ZERO in {args.startup_slack_sec:.1f}s "
        f"({time.strftime('%H:%M:%S', time.localtime(t_zero_ns / 1e9))})..."
    )

    clients: list[SpawnedClient] = []
    for spec in specs:
        env = build_client_env(
            spec,
            num_kernels=args.kernels_per_client,
            arch=args.arch,
            num_chips=args.num_chips,
            shared_fraction=args.shared_fraction,
            shared_seed=args.shared_seed,
            t_zero_ns=t_zero_ns,
        )
        clients.append(spawn_client(args.test_binary, spec, env))

    exit_codes = wait_all(clients)
    results, missing = collect_results(specs)

    if missing:
        print(f"[harness] {len(missing)} clients produced no parseable JSON: {missing}", file=sys.stderr)

    if not results:
        print("[harness] no client results parsed; aborting aggregation.", file=sys.stderr)
        return 1

    for r in results:
        if r.target_device_type != "Mock":
            raise SystemExit(
                f"[harness] client {r.client_id} reported target_device_type={r.target_device_type!r}; expected 'Mock'."
            )

    summary = aggregate(results)
    summary["exit_codes"] = exit_codes
    summary["missing_client_ids"] = missing
    summary["config"] = {
        "num_clients": args.num_clients,
        "kernels_per_client": args.kernels_per_client,
        "endpoints": endpoints,
        "endpoint_mode": args.endpoint_mode,
        "shared_fraction": args.shared_fraction,
        "shared_seed": args.shared_seed,
        "arch": args.arch,
        "num_chips": args.num_chips,
        "startup_slack_sec": args.startup_slack_sec,
        "t_zero_unix_ns": t_zero_ns,
    }

    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print_report(summary, endpoints, args)
    print(f"Wrote full summary to {summary_path}")

    if any(rc != 0 for rc in exit_codes) or missing:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
