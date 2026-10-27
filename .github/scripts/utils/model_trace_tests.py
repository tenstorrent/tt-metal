#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Run LLM e2e commands with the trace allocation tracker, or build the monthly audit matrix."""

import argparse
import os
import shlex
from pathlib import Path

import yaml

from prepare_test_matrix import (
    build_test_matrix,
    collect_skus_from_tests,
    load_sku_config,
    load_tests,
    parse_enabled_skus,
    write_matrix_output,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
E2E_TESTS = REPO_ROOT / "tests/pipeline_reorg/models_e2e_tests.yaml"
SKU_CONFIG = REPO_ROOT / ".github/sku_config.yaml"
TRACE_CONFIG = REPO_ROOT / "tests/pipeline_reorg/models_trace_config.yaml"
SWEEP_TESTS = REPO_ROOT / "tests/pipeline_reorg/models_sweep_tests.yaml"
GENERATED_START = "# BEGIN GENERATED TRACE ALLOCATION SWEEPS"
GENERATED_END = "# END GENERATED TRACE ALLOCATION SWEEPS"
# Reserve extra job time for allocation diagnostics; this estimate needs per-SKU measurements.
DIAGNOSTIC_TIMEOUT_MULTIPLIER = 2
TRACKER_ENV = {
    "TT_METAL_TRACE_ALLOC_TRACKING": "1",
    "TT_METAL_TRACE_ALLOC_TRACEBACKS": "1",
    "TT_METAL_TRACE_ALLOC_REFERRER_DEPTH": "10",
    # Include implicit program-cache allocations even if the parent shell excludes them.
    "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE": "0",
}


def tracked_command(command):
    """Set flags in the test shell, before any Python/TTNN or distributed launcher starts."""
    exports = "\n".join(f"export {key}={value}" for key, value in TRACKER_ENV.items())
    return f"{exports}\n{command}"


def load_trace_config():
    with TRACE_CONFIG.open() as file:
        return yaml.safe_load(file)


def select_test(tests, name, sku, config=None):
    config = load_trace_config() if config is None else config
    matches = [test for test in tests if test["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one e2e test named {name!r}; found {len(matches)}")
    test = matches[0]
    if not is_llm(test, config):
        raise ValueError(f"{name!r} is not an LLM inference test")
    if sku not in test["skus"]:
        raise ValueError(f"{name!r} does not support SKU {sku!r}")
    return test


def is_llm(test, config):
    return test.get("model_family") in config["families"] and test.get("model") not in config["exclude_models"]


def audit_matrix(tests, sku_config, skus="all", model="all", config=None):
    config = load_trace_config() if config is None else config
    tests = [test for test in tests if is_llm(test, config)]
    enabled_skus = collect_skus_from_tests(tests) if skus == "all" else parse_enabled_skus(skus)
    matrix = build_test_matrix(tests, enabled_skus, sku_config)
    if model != "all":
        models = {value.strip().lower() for value in model.split(",") if value.strip()}
        matrix = [entry for entry in matrix if entry.get("model", "").lower() in models]
    if not matrix:
        raise ValueError(f"No e2e tests match model={model!r}, skus={skus!r}")
    for entry in matrix:
        entry["name"] = f"Trace allocation - {entry['name']}"
        entry["cmd"] = tracked_command(entry["cmd"])
        # Diagnostics add host work. Preserve individual test timeouts and assertions;
        # only reserve more job time for the complete e2e entry.
        entry["timeout"] *= DIAGNOSTIC_TIMEOUT_MULTIPLIER
    return matrix


def focused_tests(tests, config):
    """Resolve model/SKU selections against e2e; never maintain a second tier/owner list."""
    result = []
    for model, skus in config["sweeps"].items():
        if not isinstance(skus, list) or not skus or len(set(skus)) != len(skus):
            raise ValueError(f"Sweep {model!r} needs a non-empty list of unique SKUs")
        matched = set()
        for test in tests:
            if test.get("model") != model:
                continue
            if not is_llm(test, config):
                raise ValueError(f"Sweep {model!r} is excluded from the LLM audit")
            selected = [sku for sku in skus if sku in test["skus"]]
            if not selected:
                continue
            matched.update(selected)
            result.append(
                {
                    "name": f"Trace allocation - {test['name']}",
                    "cmd": "python .github/scripts/utils/model_trace_tests.py run "
                    f"--test-name {shlex.quote(test['name'])} --sku {{sku}}",
                    "model": model,
                    "model_family": test["model_family"],
                    "skus": {
                        sku: {
                            "timeout": test["skus"][sku]["timeout"] * DIAGNOSTIC_TIMEOUT_MULTIPLIER,
                            "tier": test["skus"][sku]["tier"],
                        }
                        for sku in selected
                    },
                    "owner_id": test["owner_id"],
                    "team": test["team"],
                    "budget_type": "sweep",
                }
            )
        if missing := set(skus) - matched:
            raise ValueError(f"Sweep {model!r} has no e2e entry for SKUs: {sorted(missing)}")
    return result


def sync_sweeps(tests, config, check=False):
    """Keep a standard CI registry so the existing budget and changed-test gates can read it."""
    text = SWEEP_TESTS.read_text()
    if text.count(GENERATED_START) != 1 or text.count(GENERATED_END) != 1:
        raise ValueError(f"{SWEEP_TESTS} must contain one generated trace allocation section")
    before, rest = text.split(GENERATED_START)
    _, after = rest.split(GENERATED_END)
    entries = focused_tests(tests, config)
    generated = "\n# Generated by model_trace_tests.py sync-sweeps; edit models_trace_config.yaml instead.\n\n"
    generated += yaml.safe_dump(entries, sort_keys=False, width=110)
    updated = before + GENERATED_START + generated + GENERATED_END + after
    if check and updated != text:
        raise ValueError("Trace sweeps are stale. Run: python .github/scripts/utils/model_trace_tests.py sync-sweeps")
    if not check:
        SWEEP_TESTS.write_text(updated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run one canonical e2e entry for a focused sweep")
    run.add_argument("--test-name", required=True)
    run.add_argument("--sku", required=True)
    run.add_argument("--dry-run", action="store_true", help="Print the command without running it")
    matrix = commands.add_parser("matrix", help="Generate the all-tier monthly audit matrix")
    matrix.add_argument("--skus", default="all", help="all, or comma-separated canonical SKU names")
    matrix.add_argument("--model", default="all", help="all, or comma-separated exact model identifiers")
    sync = commands.add_parser(
        "sync-sweeps", help="Generate focused CI entries from the trace configuration and e2e registry"
    )
    sync.add_argument("--check", action="store_true", help="Fail if regeneration would change the sweep registry")
    args = parser.parse_args()
    tests = load_tests(str(E2E_TESTS))
    sku_config = load_sku_config(str(SKU_CONFIG))
    try:
        if args.command == "sync-sweeps":
            sync_sweeps(tests, load_trace_config(), check=args.check)
            return
        if args.command == "matrix":
            write_matrix_output(audit_matrix(tests, sku_config, args.skus, args.model))
            return
        test = select_test(tests, args.test_name, args.sku)
        entry = build_test_matrix([test], [args.sku], sku_config)[0]
        command = tracked_command(entry["cmd"])
    except ValueError as error:
        parser.error(str(error))
    print(f"Trace allocation check: {entry['name']}\n{command}", flush=True)
    if not args.dry_run:
        os.chdir(REPO_ROOT)
        # Keep shell pipelines and multi-command entries intact. A failed command must
        # fail the sweep, even when another pytest invocation follows it.
        os.execvp("bash", ["bash", "-e", "-o", "pipefail", "-c", command])


if __name__ == "__main__":
    main()
