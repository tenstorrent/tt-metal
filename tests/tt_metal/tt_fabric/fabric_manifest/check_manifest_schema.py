#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Checks fabric manifests against tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_schema.json.

Usage: check_manifest_schema.py PATH [PATH ...]

Checks the schema against the JSON Schema metaschema, then every fabric_manifest_rank_*.json under each directory
PATH, or each file PATH. Fails when a manifest does not match, or when it finds none.
"""

import json
import pathlib
import sys

import jsonschema

SCHEMA_PATH = (
    pathlib.Path(__file__).resolve().parents[4]
    / "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_schema.json"
)
MAX_ERRORS_SHOWN = 20


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


def main(args):
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)

    missing = [arg for arg in args if not pathlib.Path(arg).exists()]
    if missing:
        print(f"Fabric manifest: {' '.join(missing)} does not exist")
        return 1
    paths = manifest_paths(args)
    if not paths:
        print(f"Fabric manifest: no manifests found under {' '.join(args)}")
        return 1
    failed = 0
    for path in paths:
        errors = sorted(validator.iter_errors(json.loads(path.read_text())), key=lambda e: list(map(str, e.path)))
        if not errors:
            print(f"{path}: matches the schema")
            continue
        failed += 1
        print(f"Fabric manifest: {path} does not match {SCHEMA_PATH.name} ({len(errors)} errors)")
        for error in errors[:MAX_ERRORS_SHOWN]:
            print(f"  {describe(error)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
