# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Checks that the route decoder, routes.py, can read fabric manifests.

Usage, from the repository root:
    python3 -m tt_metal.fabric.debug.visualizer.decode.tests.check_manifest_routes [--expect-z-peers] PATH [PATH ...]

Runs get_encodings_from_manifest, then get_mesh_view on each mesh, on every fabric_manifest_rank_*.json under each
directory PATH, or each file PATH. Fails when either raises ManifestError, or when it finds no manifests. A manifest
can match its schema and still fail here: a fabric change can rename or remove a key or name routes.py looks up,
change a value's type, or add a value it doesn't know.

With --expect-z-peers, it also fails a manifest in which get_mesh_view finds no Z peer. Pass it the manifests of a
mesh with express chords: a change to a Z router's key or to its peer's format raises nothing, but leaves every Z peer
unknown.
"""

import argparse
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
    """The view of each of the manifest's meshes, by its key."""
    get_encodings_from_manifest(manifest)
    if not manifest.get("meshes"):
        raise ManifestError("manifest has no meshes")
    return {mesh_key: get_mesh_view(manifest, mesh_key) for mesh_key in manifest["meshes"]}


def main(argv):
    parser = argparse.ArgumentParser(description="Checks that the route decoder can read fabric manifests.")
    parser.add_argument("paths", nargs="+", metavar="PATH", help="a manifest, or a directory to search for them")
    parser.add_argument("--expect-z-peers", action="store_true", help="fail a manifest with no known Z peer")
    args = parser.parse_args(argv)

    missing = [arg for arg in args.paths if not pathlib.Path(arg).exists()]
    if missing:
        print(f"Fabric manifest: {' '.join(missing)} does not exist")
        return 1
    paths = find_manifests(args.paths)
    if not paths:
        print(f"Fabric manifest: no manifests found under {' '.join(args.paths)}")
        return 1
    failed = 0
    for path in paths:
        try:
            views = run_route_loaders(json.loads(path.read_text()))
        except ManifestError as error:
            failed += 1
            print(f"Fabric manifest: the route decoder can't read {path}: {error}")
            continue
        z_peers = sum(len(view.z_peers) for view in views.values())
        print(f"{path}: the route decoder reads its encodings and meshes {', '.join(views)}; Z peers found: {z_peers}")
        if args.expect_z_peers and not z_peers:
            failed += 1
            print(f"Fabric manifest: the route decoder finds no Z peer in {path}")
    if failed:
        print(f'Fabric manifest: "When the fabric changes" in {ROUTES_DOC} says what to update')
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
