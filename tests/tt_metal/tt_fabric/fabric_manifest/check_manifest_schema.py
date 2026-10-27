#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Checks fabric manifests against tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_schema.json.

Usage: check_manifest_schema.py PATH [PATH ...]

Checks the schema against the JSON Schema metaschema, then every fabric_manifest_rank_*.json under each directory
PATH, or each file PATH. Each manifest must match the schema, and its types must hold exactly what its schemas reach:
every struct:, enum: or union: schema names an entry in types, and every entry in types is reached from a schema
outside types, directly or through other entries. Fails when a manifest doesn't, or when it finds none.
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
# A schema's prefix, and the types block it names an entry of.
TYPE_KINDS = {"struct:": "structs", "enum:": "enums", "union:": "tagged_unions"}


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


def find_schemas(node):
    """Every schema string in node, at any depth."""
    if isinstance(node, list):
        for element in node:
            yield from find_schemas(element)
    elif isinstance(node, dict):
        for key, value in node.items():
            if key == "schema" and isinstance(value, str):
                yield value
            else:
                yield from find_schemas(value)


def find_type_problems(manifest):
    """Schemas naming no entry in types, then entries in types no schema reaches."""
    types = manifest.get("types", {})
    described = {kind: types.get(kind, {}) for kind in TYPE_KINDS.values()}
    reached = {kind: set() for kind in TYPE_KINDS.values()}
    problems = []
    pending = list(find_schemas({key: value for key, value in manifest.items() if key != "types"}))
    while pending:
        schema = pending.pop()
        for prefix, kind in TYPE_KINDS.items():
            if not schema.startswith(prefix):
                continue
            name = schema[len(prefix) :]
            if name not in described[kind]:
                problems.append(f"the schema {schema} names no entry in types.{kind}")
            elif name not in reached[kind]:
                reached[kind].add(name)
                pending.extend(find_schemas(described[kind][name]))
    for kind, entries in described.items():
        unreached = sorted(set(entries) - reached[kind])
        problems.extend(f"types.{kind}.{name} isn't reached from any schema" for name in unreached)
    return problems


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
        manifest = json.loads(path.read_text())
        errors = sorted(validator.iter_errors(manifest), key=lambda e: list(map(str, e.path)))
        problems = [describe(error) for error in errors] + find_type_problems(manifest)
        if not problems:
            print(f"{path}: matches the schema")
            continue
        failed += 1
        print(f"Fabric manifest: {path} does not match {SCHEMA_PATH.name} ({len(problems)} problems)")
        for problem in problems[:MAX_ERRORS_SHOWN]:
            print(f"  {problem}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
