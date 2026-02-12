#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Parse sweep artifacts as fallback when database is unavailable."""

import json
import os
from collections import defaultdict
from pathlib import Path

# Output file for results
RESULTS_FILE = os.environ.get("RESULTS_FILE", "sweep_results.json")
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "./sweep-results")


def parse_artifacts(artifacts_dir: str) -> dict:
    """Parse oprun_*.json files from artifacts directory."""
    artifacts_path = Path(artifacts_dir)

    if not artifacts_path.exists():
        print(f"WARNING: Artifacts directory not found: {artifacts_dir}")
        return create_empty_results()

    # Collect all test results from JSON files
    tests = []
    for json_file in artifacts_path.glob("oprun_*.json"):
        print(f"Parsing {json_file}")
        try:
            with open(json_file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    tests.extend(data)
                elif isinstance(data, dict):
                    if "tests" in data:
                        tests.extend(data["tests"])
                    else:
                        # Single test result
                        tests.append(data)
        except json.JSONDecodeError as e:
            print(f"WARNING: Failed to parse {json_file}: {e}")
            continue

    if not tests:
        print("WARNING: No test results found in artifacts")
        return create_empty_results()

    print(f"Parsed {len(tests)} test results from artifacts")

    # Aggregate results
    pass_count = sum(1 for t in tests if t.get("status") == "pass")
    fail_count = sum(1 for t in tests if str(t.get("status", "")).startswith("fail"))
    test_count = len(tests)

    # Calculate pass rate
    pass_pct = round(pass_count * 100.0 / test_count, 2) if test_count > 0 else 0

    # Extract unique models
    models_tested = sorted(set(t.get("model_name") for t in tests if t.get("model_name")))

    # Get failures by model (for display purposes)
    failures_by_model = defaultdict(int)
    for t in tests:
        if str(t.get("status", "")).startswith("fail") and t.get("model_name"):
            failures_by_model[t["model_name"]] += 1

    # Build models_affected from failures (no comparison available)
    models_affected = [
        {"model_name": model, "new_failures": count}
        for model, count in sorted(failures_by_model.items(), key=lambda x: -x[1])
    ]

    # Get card_type and git info from environment or first test
    card_type = os.environ.get("ARCH_NAME", "unknown")
    git_sha = os.environ.get("GITHUB_SHA", "")[:8] if os.environ.get("GITHUB_SHA") else ""
    git_branch = os.environ.get("GITHUB_REF_NAME", "")

    results = {
        "run_id": None,  # No run_id when parsing artifacts
        "run_summary": {
            "test_count": test_count,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "pass_pct": pass_pct,
            "prev_pass_pct": None,  # No comparison available
            "card_type": card_type,
            "git_sha": git_sha,
            "git_branch": git_branch,
        },
        "pass_rate_regressions": [],  # Cannot detect without comparison
        "perf_regressions_by_op": [],  # Cannot detect without comparison
        "perf_regressions_by_test": [],  # Cannot detect without comparison
        "models_affected": models_affected,
        "models_tested": models_tested,
        "comparison_available": False,
    }

    return results


def create_empty_results() -> dict:
    """Create empty results structure when no data available."""
    return {
        "run_id": None,
        "run_summary": {
            "test_count": 0,
            "pass_count": 0,
            "fail_count": 0,
            "pass_pct": 0,
            "prev_pass_pct": None,
            "card_type": os.environ.get("ARCH_NAME", "unknown"),
            "git_sha": os.environ.get("GITHUB_SHA", "")[:8] if os.environ.get("GITHUB_SHA") else "",
            "git_branch": os.environ.get("GITHUB_REF_NAME", ""),
        },
        "pass_rate_regressions": [],
        "perf_regressions_by_op": [],
        "perf_regressions_by_test": [],
        "models_affected": [],
        "models_tested": [],
        "comparison_available": False,
    }


def main():
    print(f"Parsing artifacts from: {ARTIFACTS_DIR}")

    results = parse_artifacts(ARTIFACTS_DIR)

    # Write results to file
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results written to {RESULTS_FILE}")
    print(f"Test count: {results['run_summary']['test_count']}")
    print(f"Pass rate: {results['run_summary']['pass_pct']}%")
    print(f"Models tested: {len(results['models_tested'])}")


if __name__ == "__main__":
    main()
