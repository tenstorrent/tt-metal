#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Run the reduce-helper migration inventory's unit tests through run_safe_pytest."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / (
    "ttnn/cpp/ttnn/kernel_lib/reduce_migration_inventory_2026-09-08_f808380a87b/unit_test_suite.json"
)


def parse_args(argv=None, default_manifest=DEFAULT_MANIFEST):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="List selected tests without importing pytest or TTNN")
    mode.add_argument("--dry-run", action="store_true", help="Print wrapper commands without running them")
    mode.add_argument(
        "--collect-only", action="store_true", help="Collect parametrized cases without running test bodies"
    )
    parser.add_argument(
        "--factory", action="append", default=[], help="Select groups associated with a report factory ID"
    )
    parser.add_argument("--group", action="append", default=[], help="Select group IDs or substrings of source paths")
    parser.add_argument("--lane", action="append", default=[], help="Select sanity hardware lanes; repeat to combine")
    parser.add_argument("--kernel", action="append", default=[], help="Select sanity cases by inventory kernel ID")
    parser.add_argument("--output-dir", type=Path, help="New directory for logs, counts and JUnit XML (must not exist)")
    parser.add_argument("--include-disabled-gtests", action="store_true", help="Opt in to upstream DISABLED_ C++ tests")
    parser.add_argument(
        "--precompile", action="store_true", help="Warm Python kernels before execution; never during collection"
    )
    parser.add_argument(
        "pytest_args", nargs=argparse.REMAINDER, help="Additional pytest arguments after --, e.g. -- -k cache"
    )
    args = parser.parse_args(argv)
    if args.pytest_args[:1] == ["--"]:
        args.pytest_args = args.pytest_args[1:]
    # Profile mode masks pytest failures in run_safe_pytest; xdist workers do not share this plugin's counts.
    for arg in args.pytest_args:
        if arg.startswith(("-n", "--numprocesses", "--dist", "--sim-workers", "--jit-server")) or arg in {
            "--profile",
            "--precompile",
            "--collect-only",
            "--co",
        }:
            parser.error(
                "use the runner's collection/precompile options; profiling and worker overrides are unsupported"
            )
    return parser, args


def main(argv=None, default_manifest=DEFAULT_MANIFEST):
    parser, args = parse_args(argv, default_manifest)
    manifest = json.loads(args.manifest.read_text())
    groups = manifest["groups"]
    if args.factory:
        wanted = {factory.upper() for factory in args.factory}
        groups = [group for group in groups if wanted.intersection(group["factories"])]
    if args.group:
        groups = [
            group for group in groups if any(query == group["id"] or query in group["source"] for query in args.group)
        ]
    if args.lane:
        groups = [group for group in groups if group.get("lane") in args.lane]
    if args.kernel:
        wanted = {kernel.upper() for kernel in args.kernel}
        groups = [group for group in groups if wanted.intersection(group.get("kernel_ids", []))]
    if not groups:
        parser.error("no test groups matched; see the manifest's gaps for factories without tests")

    python_count = sum(len(group["tests"]) for group in groups if group["kind"] == "pytest")
    cpp_count = sum(len(group["tests"]) for group in groups if group["kind"] == "gtest")
    disabled_count = sum("DISABLED_" in test for group in groups if group["kind"] == "gtest" for test in group["tests"])
    python_unit = "cases" if manifest.get("exact_cases") else "test functions"
    print(
        f"Selected {python_count} Python {python_unit} and {cpp_count} C++ cases in {len(groups)} groups.", flush=True
    )
    print(f"C++ cases include {disabled_count} disabled upstream (skipped unless explicitly enabled).", flush=True)
    if args.list:
        for group in groups:
            print(f"\n{group['id']} | {group['family']} | {', '.join(group['factories'])}")
            if "kernel_ids" in group:
                print(f"Lane: {group['lane']} | Kernels: {', '.join(group['kernel_ids'])} | {group['hardware']}")
            for test in group["tests"]:
                print(f"{group['source']}::{test}")
        return 0

    if args.dry_run:
        output = args.output_dir or Path("<new-results-directory>")
    elif args.output_dir:
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=False)
    else:
        parent = ROOT / "generated/test_reports"
        parent.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="reduce-migration-", dir=parent))
    print(f"Results: {output}", flush=True)
    results = []
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if not args.dry_run:
        # A running suite must keep its selection if the migration inventory is edited concurrently.
        snapshot = output / "manifest.json"
        snapshot.write_text(json.dumps(manifest, indent=2) + "\n")
        env["TT_REDUCE_SUITE_MANIFEST"] = str(snapshot)
    else:
        env["TT_REDUCE_SUITE_MANIFEST"] = str(args.manifest.resolve())
    env["TT_REDUCE_SUITE_DISABLED_GTESTS"] = "1" if args.include_disabled_gtests else "0"

    for group in groups:
        prefix = output / group["id"]
        env["TT_REDUCE_SUITE_GROUP"] = group["id"]
        env["TT_REDUCE_SUITE_RESULT"] = str(prefix.with_suffix(".json"))
        group_env = env.copy()
        for name, value in group.get("environment_defaults", {}).items():
            group_env.setdefault(name, value)
        # Sanity selections pin supported test-loop controls so ambient settings cannot change coverage.
        group_env.update(group.get("environment", {}))
        command = ["bash", str(ROOT / "scripts/run_safe_pytest.sh"), "--run-all"]
        if env.get("TT_METAL_SIMULATOR"):
            command.extend(["--sim-workers", "1"])
        warm = args.precompile and not args.collect_only and group["kind"] == "pytest"
        command.append("--precompile" if warm else "--no-precompile")
        command.extend(["-p", "scripts.reduce_migration_pytest_plugin", "-o", "addopts=", "--import-mode=importlib"])
        command.append(f"--junitxml={prefix.with_suffix('.xml')}")
        if args.collect_only:
            command.extend(["--collect-only", "-q"])
        else:
            command.extend(["-v", "-rA", "--durations=25"])
        if group["kind"] == "pytest":
            # The option is defined in tests/ttnn/conftest.py, not in model-test conftests.
            if group["source"].startswith("tests/ttnn/"):
                command.append("--runslow")
            for test in group["tests"]:
                # Pytest resolves its architecture fixture before the plugin chooses the exact case.
                if group.get("nodeid_arch_template"):
                    test = test.split("[", 1)[0]
                command.append(f"{group['source']}::{test}")
        else:
            command.append("scripts/reduce_migration_gtest_adapter.py::test_cpp_factory")
        command.extend(args.pytest_args)
        if args.dry_run:
            print(f"\n# {group['id']} {group['family']}; TT_REDUCE_SUITE_GROUP={group['id']}")
            displayed_env = {name: value for name, value in group_env.items() if name.startswith("TT_REDUCE_SUITE_")}
            displayed_env.update({name: group_env[name] for name in group.get("environment_defaults", {})})
            displayed_env.update(group.get("environment", {}))
            print(shlex.join(["env", *(f"{name}={value}" for name, value in displayed_env.items()), *command]))
            continue
        print(f"[{group['id']}] {group['source']} ... log: {prefix.with_suffix('.log')}", flush=True)
        with prefix.with_suffix(".log").open("w") as log:
            completed = subprocess.run(
                command, cwd=ROOT, env=group_env, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        counts_path = prefix.with_suffix(".json")
        counts = json.loads(counts_path.read_text()) if counts_path.exists() else {}
        returncode = completed.returncode
        if returncode == 0 and (not counts or counts["collected"] != counts["unique_nodeids"]):
            print(f"[{group['id']}] Missing collection counts or duplicate node IDs; treating as failure.", flush=True)
            returncode = 1
        if returncode == 0 and "expected_cases" in group and counts["collected"] != group["expected_cases"]:
            print(
                f"[{group['id']}] Expected {group['expected_cases']} cases; treating selection drift as failure.",
                flush=True,
            )
            returncode = 1
        if returncode == 0 and group.get("require_pass") and not args.collect_only:
            outcomes = counts.get("outcomes", {})
            if outcomes.get("passed", 0) != group["expected_cases"] or any(
                count for outcome, count in outcomes.items() if outcome != "passed"
            ):
                print(f"[{group['id']}] Sanity requires every selected case to pass; outcomes={outcomes}.", flush=True)
                returncode = 1
        result = {"group": group["id"], "source": group["source"], "returncode": returncode, **counts}
        results.append(result)
        summary = {
            "mode": "collection" if args.collect_only else "execution",
            "manifest": str(args.manifest.resolve()),
            "python_cases" if manifest.get("exact_cases") else "python_functions": python_count,
            "cpp_cases": cpp_count,
            "disabled_cpp_cases": disabled_count,
            "groups_requested": len(groups),
            "groups_completed": len(results),
            "collected_cases": sum(row.get("collected", 0) for row in results),
            "failed_groups": [row["group"] for row in results if row["returncode"] != 0],
            "results": results,
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[{group['id']}] exit={returncode}, collected={counts.get('collected', 'unknown')}", flush=True)
    if args.dry_run:
        return 0
    print(f"Collected cases: {summary['collected_cases']:,}. Failed groups: {len(summary['failed_groups'])}.")
    print(f"Summary: {output / 'summary.json'}")
    return 1 if summary["failed_groups"] else 0


if __name__ == "__main__":
    sys.exit(main())
