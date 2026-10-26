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
- a snapshot taken through capture with a device that returns random bytes decodes. Each ring buffer gets random bytes
  in its first slot only, and zeros in the rest. The decoded file has one entry, with a value, for each area but the
  routing table, region, stream reference and element reference of every router. It has health for every router, a
  sample per liveness round and the main pass. Every active channel has its counts and a slot per ring slot, the first
  used and the rest empty, and no other channel is listed. Every router names a routing table copy of its chip, every
  copy holds one router, since random tables never match, and a 2D copy has a route toward each chip of its mesh. The
  file matches fabric_decoded_schema.json.
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
    """
    A device whose bytes are random, the same for every run, except in a ring buffer: its first slot is random, and
    the others are zero, as slots the router hasn't used since it started. Every channel's slot rules still meet random
    bytes, but the decoded file stays small: with every slot random, an express rank's is over 400 MB.
    """

    version = "check_decode"

    def __init__(self, manifest: dict):
        self.manifest = manifest
        self.opened = 0
        self.random = random.Random(SEED)
        self.rings = {}  # each opened router's ring buffers' slot sizes, by address

    def open_router(self, router, arch):
        node, where = get_router(self.manifest, router.path)
        nodes = {leaf.path: leaf.node for leaf in walk_router(node, where)}
        self.rings[self.opened] = {
            read.address: nodes[path]["size_per_element"]
            for path, read in router.regions.items()
            if path.endswith("/ring_buffer")
        }
        self.opened += 1
        return OpenedRouter(handle=self.opened - 1, ttexalens_id=self.opened - 1)

    def read(self, opened, address, size):
        data = self.random.randbytes(size)
        slot_size = self.rings[opened.handle].get(address)
        return data if slot_size is None else data[:slot_size] + bytes(size - slot_size)


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
    check_channels(node, path, router, problems)


def check_channels(node: dict, path: str, router: dict, problems: list):
    """
    Every active channel has its counts and one slot per ring slot: the first used, since RandomDevice fills it, and the
    rest empty. No other channel is listed.
    """
    expected = {}
    for kind in ("senders", "receivers"):
        for vc, channels in node["channels"][kind].items():
            for key, channel in channels.items():
                if channel["status"] == "active":
                    expected[f"channels/{kind}/{vc}/{key}"] = (
                        kind[: -len("s")],
                        channel["ring_buffer"]["num_elements"],
                    )
    if list(router["channels"]) != list(expected):
        problems.append(
            f"{path}.channels are {list(router['channels'])[:3]}..., not the active {list(expected)[:3]}..."
        )
        return
    for channel_path, (kind, depth) in expected.items():
        channel = router["channels"][channel_path]
        where = f"{path}.{channel_path}"
        if (channel["kind"], channel["depth"]) != (kind, depth):
            problems.append(f"{where} is a {channel['kind']} of depth {channel['depth']}, not a {kind} of {depth}")
        slots = channel["slots"]
        if slots is None or len(slots) != depth:
            problems.append(f"{where} has {None if slots is None else len(slots)} slots, not {depth}")
            continue
        if [slot["empty"] for slot in slots] != [False] + [True] * (depth - 1):
            problems.append(f"{where}'s slots aren't used in slot 0 and empty after it, as RandomDevice wrote them")
        if any((slot["header"] is None) != slot["empty"] for slot in slots):
            problems.append(f"{where} has a used slot with no header, or an empty one with one")


def check_routing_tables(manifest: dict, decoded: dict, problems: list):
    """
    Every router names a copy holding it, on its chip. RandomDevice gives every router different table bytes, so every
    copy has one router. A 2D copy has a route toward every chip of its mesh.
    """
    is_2d = manifest["fabric_context"]["is_2d_routing"]
    tables = decoded["routing_tables"]
    for path, router in decoded["routers"].items():
        key = None if router["routing"] is None else router["routing"]["table_key"]
        table = tables.get(key)
        if table is None or path not in table["routers"] or table["chip"] != path.rsplit("/", 1)[0]:
            problems.append(f"{path}.routing names {key!r}, which isn't a copy of its chip holding it")
    for key, table in tables.items():
        if table["routers"] != [key]:
            problems.append(f"routing_tables.{key} holds {table['routers']}; random tables never match, so just {key}")
        mesh_key = key.split("/")[0]
        chips = [f"{mesh_key}/{chip}" for chip in manifest["meshes"][mesh_key]["chips"]]
        if is_2d and (table["route_table_2d"] is None or list(table["route_table_2d"]) != chips):
            problems.append(
                f"routing_tables.{key}.route_table_2d doesn't have a route toward each of {mesh_key}'s chips"
            )
        if not is_2d and table["path_table_1d"] is None:
            problems.append(f"routing_tables.{key} has no path_table_1d")


def check_manifest(manifest_path: pathlib.Path, validator) -> tuple:
    """The problems with decoding the manifest at `manifest_path`, and a line describing what was decoded."""
    manifest = json.loads(manifest_path.read_bytes())
    problems = []
    types = check_types(manifest, problems)
    with tempfile.TemporaryDirectory() as out_dir:
        taken = take_snapshot(
            manifest_path,
            pathlib.Path(out_dir),
            RandomDevice(manifest),
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
    check_routing_tables(manifest, decoded, problems)
    routers = decoded["routers"].values()
    counts = ", ".join(
        f"{sum(len(router[key]) for router in routers)} {key}" for key in ("regions", "streams", "elements", "channels")
    )
    description = (
        f"{types}; {len(routers)} routers, {counts}, {len(decoded['routing_tables'])} routing tables; "
        f"decoded in {seconds:.1f} s, {size:,} bytes"
    )
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
