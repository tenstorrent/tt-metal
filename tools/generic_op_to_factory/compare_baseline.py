# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Compare a new JUnit baseline with one explicit phase of a frozen DB export.

Outcome parity is not operation correctness or proof of historical replay.
Metrics/performance and undocumented environment differences are not compared.
"""

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from tools.generic_op_to_factory.classify_failures import parse_junit_xml
from tools.generic_op_to_factory.export_run import ExportError, _hash_file, verify_export


def compare_outcomes(recorded, observed):
    def index(rows):
        indexed = {}
        for row in rows:
            key = (row.get("test_file"), row.get("test_name"))
            if not all(isinstance(value, str) and value for value in key):
                raise ExportError("A result has no unambiguous test file/name identity")
            if key in indexed:
                raise ExportError(f"Duplicate case identity in selected results: {key}")
            if row.get("status") not in (
                "passed",
                "failed",
                "error",
                "skipped",
                "xfail",
                "xpass",
            ):
                raise ExportError(f"Unrecognized outcome: {row.get('status')!r}")
            indexed[key] = row["status"]
        if not indexed:
            raise ExportError("Selected results are empty")
        return indexed

    before, after = index(recorded), index(observed)

    def identity(key):
        return {"test_file": key[0], "test_name": key[1]}

    missing = [identity(key) for key in sorted(before.keys() - after.keys())]
    added = [identity(key) for key in sorted(after.keys() - before.keys())]
    changed = [
        {**identity(key), "recorded": before[key], "observed": after[key]}
        for key in sorted(before.keys() & after.keys())
        if before[key] != after[key]
    ]
    return {
        "recorded_counts": dict(sorted(Counter(before.values()).items())),
        "observed_counts": dict(sorted(Counter(after.values()).items())),
        "same_case_set": not missing and not added,
        "outcomes_match": not missing and not added and not changed,
        "recorded_failures": sum(status in ("failed", "error", "xpass") for status in before.values()),
        "observed_failures": sum(status in ("failed", "error", "xpass") for status in after.values()),
        "missing": missing,
        "added": added,
        "changed": changed,
        "scope": "case identities and outcome classes only; excludes metrics, failure-message equivalence and runtime provenance",
        "migration_ready": False,
    }


def compare(export, junit, phase):
    manifest = verify_export(export)
    recorded = []
    with (Path(export) / "records/test_results.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("phase") == phase:
                recorded.append(row)
    result = compare_outcomes(recorded, parse_junit_xml(Path(junit)))
    result.update(input_snapshot_sha256=manifest["snapshot_sha256"], selected_phase=phase)
    result["junit_sha256"] = _hash_file(Path(junit))[0]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--phase")
    phase.add_argument("--unphased", action="store_true")
    args = parser.parse_args()
    try:
        result = compare(args.export, args.junit, args.phase)
    except (ExportError, OSError, ValueError, ET.ParseError) as error:
        parser.exit(2, f"Baseline comparison failed: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["outcomes_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
