# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Checks that the route decoder, routes.py, can read fabric manifests.

Usage, from the repository root:
    python3 -m tt_metal.fabric.debug.visualizer.decode.tests.check_manifest_routes PATH [PATH ...]

Runs get_encodings_from_manifest, then get_mesh_view on each mesh, on every fabric_manifest_rank_*.json under each
directory PATH, or each file PATH. Fails when either raises ManifestError, or when it finds no manifests. A manifest
can match its schema and still fail here: a fabric change can rename or remove a key or name routes.py looks up, or
add a value it doesn't know.
"""

import json
import pathlib
import sys

from tt_metal.fabric.debug.visualizer.decode.routes import ManifestError, get_encodings_from_manifest, get_mesh_view

ROUTES_DOC = "tt_metal/fabric/debug/visualizer/decode/ROUTES.md"


def find_manifests(args):
    paths = []
    for arg in args:
        path = pathlib.Path(arg)
        paths.extend(sorted(path.rglob("fabric_manifest_rank_*.json")) if path.is_dir() else [path])
    return paths


def run_route_loaders(manifest):
    get_encodings_from_manifest(manifest)
    if not manifest.get("meshes"):
        raise ManifestError("manifest has no meshes")
    for mesh_key in manifest["meshes"]:
        get_mesh_view(manifest, mesh_key)
    return list(manifest["meshes"])


def main(args):
    missing = [arg for arg in args if not pathlib.Path(arg).exists()]
    if missing:
        print(f"Fabric manifest: {' '.join(missing)} does not exist")
        return 1
    paths = find_manifests(args)
    if not paths:
        print(f"Fabric manifest: no manifests found under {' '.join(args)}")
        return 1
    failed = 0
    for path in paths:
        try:
            mesh_keys = run_route_loaders(json.loads(path.read_text()))
        except ManifestError as error:
            failed += 1
            print(f"Fabric manifest: the route decoder can't read {path}: {error}")
            continue
        print(f"{path}: the route decoder reads its encodings and meshes {', '.join(mesh_keys)}")
    if failed:
        print(f'Fabric manifest: "When the fabric changes" in {ROUTES_DOC} says what to update')
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
