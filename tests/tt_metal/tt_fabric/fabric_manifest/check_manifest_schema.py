#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Checks fabric manifests against tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_schema.json.

Usage: check_manifest_schema.py PATH [PATH ...]

Checks the schema against the JSON Schema metaschema, then every fabric_manifest_rank_*.json under each directory
PATH, or each file PATH, several at once. Fails when a manifest doesn't match the schema, or when it finds none.
"""

import concurrent.futures
import json
import os
import pathlib
import sys

import jsonschema

SCHEMA_PATH = (
    pathlib.Path(__file__).resolve().parents[4]
    / "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_schema.json"
)
MAX_ERRORS_SHOWN = 20
# A galaxy's manifest is tens of MB of JSON, and each check holds one loaded, so only a few run at once.
MAX_PARALLEL_CHECKS = 4


def manifest_paths(args):
    paths = []
    for arg in args:
        path = pathlib.Path(arg)
        paths.extend(sorted(path.rglob("fabric_manifest_rank_*.json")) if path.is_dir() else [path])
    return paths


def describe(error):
    # A oneOf or anyOf error says only that no branch matched. The branch that got deepest into the value says why;
    # a branch that fails only on the value's type or on a constant, such as a field's null or its kind, does not.
    while error.context:
        error = max(error.context, key=lambda e: (len(e.absolute_path), e.validator not in ("type", "const")))
    location = "/".join(str(part) for part in error.absolute_path)
    return f"/{location}: {error.message}"


def load_validator():
    return jsonschema.Draft202012Validator(json.loads(SCHEMA_PATH.read_text()))


def find_problems(path):
    """What in the manifest at path doesn't match the schema, one line each. Runs in a worker process."""
    manifest = json.loads(path.read_text())
    errors = sorted(load_validator().iter_errors(manifest), key=lambda e: list(map(str, e.path)))
    return [describe(error) for error in errors]


def main(args):
    jsonschema.Draft202012Validator.check_schema(json.loads(SCHEMA_PATH.read_text()))

    missing = [arg for arg in args if not pathlib.Path(arg).exists()]
    if missing:
        print(f"Fabric manifest: {' '.join(missing)} does not exist")
        return 1
    paths = manifest_paths(args)
    if not paths:
        print(f"Fabric manifest: no manifests found under {' '.join(args)}")
        return 1
    failed = 0
    # Each result is printed as it arrives, so a run cut short shows how far it got.
    num_workers = min(MAX_PARALLEL_CHECKS, len(paths), os.cpu_count() or 1)
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(find_problems, path): path for path in paths}
        for future in concurrent.futures.as_completed(futures):
            path = futures[future]
            problems = future.result()
            if not problems:
                print(f"{path}: matches the schema", flush=True)
                continue
            failed += 1
            print(f"Fabric manifest: {path} does not match {SCHEMA_PATH.name} ({len(problems)} problems)")
            for problem in problems[:MAX_ERRORS_SHOWN]:
                print(f"  {problem}")
            sys.stdout.flush()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
