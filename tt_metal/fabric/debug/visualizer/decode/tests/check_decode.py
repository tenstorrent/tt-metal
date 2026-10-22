# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Checks that decode can decode every manifest's types and snapshots, and that a decoded file holds what the manifest
names.

Usage, from the repository root:
    python3 -m tt_metal.fabric.debug.visualizer.decode.tests.check_decode PATH [PATH ...]

For every fabric_manifest_rank_*.json under each directory PATH, or each file PATH:
- every struct in the manifest's types decodes from zeros and from random bytes, and every tagged union decodes under
  each value of its tag, and under one that names no member, picking the member its union says;
- a snapshot taken through capture with a device that returns random bytes decodes. The decoded file has one entry, with
  a value, for each area, region, stream reference and element reference of every router. It has health for every
  router, a sample per liveness round and the main pass, and matches fabric_decoded_schema.json.
Each snapshot and decoded file is deleted once checked: express's snapshot is over 100 MB a rank. Fails when a check
fails, or when it finds no manifests.
"""

import argparse
import datetime
import json
import pathlib
import random
import sys
import tempfile
import time

import jsonschema

from tt_metal.fabric.debug.visualizer.capture.snapshot import OpenedRouter, take_snapshot
from tt_metal.fabric.debug.visualizer.common.manifest_lookup import ManifestError, lookup
from tt_metal.fabric.debug.visualizer.common.router_paths import LeafKind, get_router, walk_router
from tt_metal.fabric.debug.visualizer.decode.decoded import decode_snapshot, write_decoded
from tt_metal.fabric.debug.visualizer.decode.inputs import InputError, load_input
from tt_metal.fabric.debug.visualizer.decode.routing_tables import ROUTING_TABLE
from tt_metal.fabric.debug.visualizer.decode.type_parser import decode_struct, get_types_from_manifest

SCHEMA_PATH = pathlib.Path(__file__).resolve().parents[1] / "fabric_decoded_schema.json"
MAX_PROBLEMS_SHOWN = 20
LIVENESS_ROUNDS = 2
SEED = 10


class RandomDevice:
    """A device whose bytes are random, the same for every run."""

    version = "check_decode"

    def __init__(self):
        self.opened = 0
        self.random = random.Random(SEED)

    def open_router(self, router, arch):
        self.opened += 1
        return OpenedRouter(handle=self.opened - 1, ttexalens_id=self.opened - 1)

    def read(self, opened, address, size):
        return self.random.randbytes(size)


def find_manifests(args):
    paths = []
    for arg in args:
        path = pathlib.Path(arg)
        paths.extend(sorted(path.rglob("fabric_manifest_rank_*.json")) if path.is_dir() else [path])
    return paths


def check_types(manifest: dict, problems: list) -> str:
    """Decodes every struct, and every tagged union under each value of its tag. Describes what it decoded."""
    types = get_types_from_manifest(manifest)
    rng = random.Random(SEED)
    unions = 0
    for name, struct in types.structs.items():
        decode_struct(name, bytes(struct.size), 0, types)
        decode_struct(name, rng.randbytes(struct.size), 0, types)
        for member in struct.members:
            if member.shape.tag is None:
                continue
            union = types.unions[member.shape.schema[len("union:") :]]
            tag = next(sibling for sibling in struct.members if sibling.name == member.shape.tag)
            names = types.enums[tag.shape.schema[len("enum:") :]].names
            # Every value the tag's enum names, and one it doesn't.
            for value in [*names, max(names, default=-1) + 1]:
                data = bytearray(struct.size)
                data[tag.offset : tag.offset + tag.shape.size] = value.to_bytes(tag.shape.size, "little")
                picked = decode_struct(name, bytes(data), 0, types)[member.name]["member"]
                expected = union.members.get(names.get(value), union.default_member)
                if picked != (None if expected is None else expected.name):
                    problems.append(f"{name}.{member.name}: tag value {value} picked {picked!r}")
                unions += 1
    return f"{len(types.structs)} structs, {unions} tagged union decodes"


def check_router(manifest: dict, path: str, router: dict, problems: list):
    """
    The decoded router has one entry, with a value, for each item the manifest locates, and health. Its routing table
    is stored in the file's routing_tables instead of its areas.
    """
    node, where = get_router(manifest, path)
    leaves = {kind: [leaf.path for leaf in walk_router(node, where) if leaf.kind is kind] for kind in LeafKind}
    areas = lookup(manifest, f"archs.{manifest['run']['arch']}.areas", "object")
    expected = {
        "areas": [name for name, area in areas.items() if area is not None and name != ROUTING_TABLE],
        "regions": leaves[LeafKind.REGION],
        "streams": leaves[LeafKind.STREAM],
        "elements": leaves[LeafKind.ELEMENT],
    }
    for key, paths in expected.items():
        if list(router[key]) != paths:
            missing = [item for item in paths if item not in router[key]]
            extra = [item for item in router[key] if item not in paths]
            problems.append(f"{path}.{key}: missing {missing[:3]}, not in the manifest {extra[:3]}, or out of order")
    unread = [
        key
        for key, stream in router["streams"].items()
        if None in (stream["before"], stream["after"], stream["before_word"], stream["after_word"])
    ]
    unread += [key for key, element in router["elements"].items() if element["value"] is None]
    if unread:
        problems.append(f"{path}: no value for {unread[:3]}")
    health = router["health"]
    if len(health["t_ns"]) != LIVENESS_ROUNDS + 1:
        problems.append(f"{path}.health has {len(health['t_ns'])} samples, not {LIVENESS_ROUNDS + 1}")
    if list(health["eriscs"]) != list(node["eriscs"]):
        problems.append(f"{path}.health.eriscs is {list(health['eriscs'])}, not the router's {list(node['eriscs'])}")


def check_manifest(manifest_path: pathlib.Path, validator) -> tuple:
    """The problems with decoding the manifest at `manifest_path`, and a line describing what was decoded."""
    manifest = json.loads(manifest_path.read_bytes())
    problems = []
    types = check_types(manifest, problems)
    with tempfile.TemporaryDirectory() as out_dir:
        taken = take_snapshot(
            manifest_path,
            pathlib.Path(out_dir),
            RandomDevice(),
            host="check_decode",
            argv=[str(manifest_path)],
            liveness_rounds=LIVENESS_ROUNDS,
            liveness_interval_s=0,
            sleep=lambda seconds: None,
        )
        started = time.monotonic()
        snapshot = load_input(taken.json_path, None)
        decoded_path = pathlib.Path(out_dir) / "decoded.json"
        write_decoded(decode_snapshot(snapshot, None, datetime.datetime.now(datetime.timezone.utc)), decoded_path)
        seconds = time.monotonic() - started
        decoded = json.loads(decoded_path.read_text())
        size = decoded_path.stat().st_size

    problems += [
        f"/{'/'.join(map(str, error.absolute_path))}: {error.message[:200]}" for error in validator.iter_errors(decoded)
    ]
    if list(decoded["routers"]) != list(snapshot.json["routers"]):
        problems.append("the decoded routers aren't the snapshot's")
    for path, router in decoded["routers"].items():
        check_router(manifest, path, router, problems)
    routers = decoded["routers"].values()
    counts = ", ".join(
        f"{sum(len(router[key]) for router in routers)} {key}" for key in ("regions", "streams", "elements")
    )
    description = f"{types}; {len(routers)} routers, {counts}; decoded in {seconds:.1f} s, {size:,} bytes"
    return problems, description


def main(argv):
    parser = argparse.ArgumentParser(description="Checks that decode can decode fabric manifests' types and snapshots.")
    parser.add_argument("paths", nargs="+", metavar="PATH", help="a manifest, or a directory to search for them")
    args = parser.parse_args(argv)

    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)

    missing = [arg for arg in args.paths if not pathlib.Path(arg).exists()]
    if missing:
        print(f"Fabric decode: {' '.join(missing)} does not exist")
        return 1
    paths = find_manifests(args.paths)
    if not paths:
        print(f"Fabric decode: no manifests found under {' '.join(args.paths)}")
        return 1
    failed = 0
    for path in paths:
        try:
            problems, description = check_manifest(path, validator)
        except (ManifestError, InputError, ValueError) as error:
            failed += 1
            print(f"Fabric decode: can't decode {path}: {error}")
            continue
        if not problems:
            print(f"{path}: {description}")
            continue
        failed += 1
        print(f"Fabric decode: decoding {path} is wrong ({len(problems)} problems)")
        for problem in problems[:MAX_PROBLEMS_SHOWN]:
            print(f"  {problem}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
