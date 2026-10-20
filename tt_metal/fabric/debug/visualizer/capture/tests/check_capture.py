# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Checks that capture can take a snapshot of fabric manifests, and that the snapshot holds what the manifest names.

Usage, from the repository root:
    python3 -m tt_metal.fabric.debug.visualizer.capture.tests.check_capture PATH [PATH ...]

Takes a snapshot of every fabric_manifest_rank_*.json under each directory PATH, or each file PATH, with a fake device
whose bytes say what was read: a read's first word is its address and its second is the router's index. Then checks:
- the snapshot has every router, area, region, core register and stream register get_read_plan lists, and nothing else;
- each area's name and each region's path find, in the manifest, the address and size its entry copies;
- each area's and region's bytes are at its offset in the .bin, and the entries fill the .bin end to end;
- each register's value is the address the manifest gives it, base + id * stride + offset for a stream register;
- each router has two liveness samples, taken before its main pass, whose heartbeat and core register values are the
  addresses the manifest gives them;
- the snapshot matches fabric_snapshot_schema.json, manifest.sha256 is the manifest file's and raw is the .bin's.
Each snapshot is deleted once checked: express's is over 100 MB a rank. Fails when a check fails, or when it finds no
manifests.
"""

import argparse
import hashlib
import json
import pathlib
import struct
import sys
import tempfile
import time

import jsonschema

from tt_metal.fabric.debug.visualizer.capture.reads import ManifestError, get_read_plan
from tt_metal.fabric.debug.visualizer.capture.snapshot import OpenedRouter, take_snapshot

SCHEMA_PATH = pathlib.Path(__file__).resolve().parents[1] / "fabric_snapshot_schema.json"
MAX_PROBLEMS_SHOWN = 20
LIVENESS_ROUNDS = 2


class RecordingDevice:
    """A device whose bytes say what was read: the address, then the router's index in the plan, then zeros."""

    version = "check_capture"

    def __init__(self):
        self.opened = 0

    def open_router(self, router, arch):
        self.opened += 1
        return OpenedRouter(handle=self.opened - 1, ttexalens_id=self.opened - 1)

    def read(self, opened, address, size):
        return (struct.pack("<II", address, opened.handle) + bytes(size))[:size]


def find_manifests(args):
    paths = []
    for arg in args:
        path = pathlib.Path(arg)
        paths.extend(sorted(path.rglob("fabric_manifest_rank_*.json")) if path.is_dir() else [path])
    return paths


def resolve(router: dict, path: str):
    """
    The manifest node a region's path names inside its router, object keys and array indices joined by "/", or None.
    """
    node = router
    try:
        for key in path.split("/"):
            node = node[int(key)] if isinstance(node, list) else node[key]
    except (KeyError, IndexError, ValueError, TypeError):
        return None
    return node


def sha256_of(path: pathlib.Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            sha256.update(chunk)
    return sha256.hexdigest()


def check_keys(problems: list, where: str, found, expected):
    if list(found) != list(expected):
        missing = [key for key in expected if key not in found]
        extra = [key for key in found if key not in expected]
        problems.append(f"{where}: missing {missing[:5]}, not in the plan {extra[:5]}, or out of plan order")


def check_routers(manifest: dict, snapshot: dict, raw_path: pathlib.Path) -> list:
    problems = []
    plan = get_read_plan(manifest)
    arch = manifest["archs"][plan.arch]
    core_registers = {register["name"]: register["address"] for register in arch["core_registers"]}
    areas = arch["areas"]
    streams = arch["stream_registers"]
    sample_words = {"heartbeat": None if areas.get("heartbeat") is None else areas["heartbeat"]["address"]}
    sample_words.update(core_registers)

    paths = [router.path for router in plan.routers]
    if list(snapshot["routers"]) != paths:
        check_keys(problems, "routers", snapshot["routers"], paths)
        return problems

    l1 = []  # (where, entry, router index) for every area and region
    for index, router in enumerate(plan.routers):
        entry = snapshot["routers"][router.path]
        if entry["error"] is not None:
            problems.append(f"{router.path}: error {entry['error']}")
            continue
        if entry["device"]["ttexalens_id"] != index:
            problems.append(f"{router.path}: device.ttexalens_id is {entry['device']['ttexalens_id']}, not {index}")
        check_keys(problems, f"{router.path}.core_registers", entry["core_registers"], plan.core_registers)
        check_keys(problems, f"{router.path}.areas", entry["areas"], plan.areas)
        check_keys(problems, f"{router.path}.regions", entry["regions"], router.regions)
        stream_keys = [(int(stream_id), name) for stream_id, names in entry["streams"].items() for name in names]
        check_keys(problems, f"{router.path}.streams", stream_keys, router.streams)

        for name, value in entry["core_registers"].items():
            expected = core_registers.get(name)
            if expected is not None and value != expected:
                problems.append(f"{router.path}.core_registers.{name} is {value:#x}, not {expected:#x}")
        for stream_id, names in entry["streams"].items():
            for name, values in names.items():
                address = streams["base"] + int(stream_id) * streams["stride"] + streams["register_offsets"][name]
                if values != {"before": address, "after": address}:
                    problems.append(f"{router.path}.streams.{stream_id}.{name} is {values}, not {address:#x}")

        if len(entry["liveness"]) != LIVENESS_ROUNDS:
            problems.append(f"{router.path}.liveness has {len(entry['liveness'])} samples, not {LIVENESS_ROUNDS}")
        for round_index, sample in enumerate(entry["liveness"]):
            words = {"heartbeat": sample["heartbeat"], **sample["core_registers"]}
            if list(words.items()) != list(sample_words.items()):
                problems.append(f"{router.path}.liveness[{round_index}] is {words}, not {sample_words}")
        times = [sample["t_ns"] for sample in entry["liveness"]] + [entry["started_ns"]]
        if times != sorted(times):
            problems.append(f"{router.path}: the liveness samples' and main pass's times {times} are out of order")

        mesh_key, chip_key, router_key = router.path.split("/")
        router_node = manifest["meshes"][mesh_key]["chips"][chip_key]["routers"][router_key]
        for kind in ("areas", "regions"):
            for key, item in entry[kind].items():
                where = f"{router.path}.{kind}.{key}"
                node = areas.get(key) if kind == "areas" else resolve(router_node, key)
                copied = (item["address"], item["size"])
                if not isinstance(node, dict) or copied != (node.get("address"), node.get("size")):
                    problems.append(f"{where}: {item} doesn't match the manifest's {str(node)[:200]}")
                l1.append((where, item, index))

    raw_size = raw_path.stat().st_size
    with raw_path.open("rb") as raw:
        for where, item, index in l1:
            raw.seek(item["offset"])
            data = raw.read(min(item["size"], 8))
            if data != struct.pack("<II", item["address"], index)[: len(data)]:
                problems.append(f"{where}: the bytes at offset {item['offset']} are {data.hex()}")
    end = 0
    for where, item, _ in sorted(l1, key=lambda l1_item: l1_item[1]["offset"]):
        if item["offset"] != end:
            problems.append(f"{where}: offset {item['offset']}, where the entry before it ends at {end}")
            break
        end += item["size"]
    if end != raw_size:
        problems.append(f"the entries end at {end}; the .bin is {raw_size} bytes")
    return problems


def check_manifest(manifest_path: pathlib.Path, validator) -> tuple:
    """The problems with a snapshot of the manifest at `manifest_path`, and a line describing the snapshot."""
    with tempfile.TemporaryDirectory() as out_dir:
        started = time.monotonic()
        result = take_snapshot(
            manifest_path,
            pathlib.Path(out_dir),
            RecordingDevice(),
            host="check_capture",
            argv=[str(manifest_path)],
            liveness_rounds=LIVENESS_ROUNDS,
            liveness_interval_s=0,
        )
        seconds = time.monotonic() - started
        snapshot = json.loads(result.json_path.read_text())
        problems = [
            f"/{'/'.join(map(str, error.absolute_path))}: {error.message}" for error in validator.iter_errors(snapshot)
        ]
        if snapshot["manifest"]["sha256"] != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
            problems.append("manifest.sha256 isn't the manifest file's")
        if (snapshot["raw"]["size"], snapshot["raw"]["sha256"]) != (
            result.bin_path.stat().st_size,
            sha256_of(result.bin_path),
        ):
            problems.append("raw.size or raw.sha256 isn't the .bin's")
        if not problems:
            problems = check_routers(json.loads(manifest_path.read_bytes()), snapshot, result.bin_path)
        routers = snapshot["routers"].values()
        regions = sum(len(router["regions"]) for router in routers)
        streams = sum(len(names) for router in routers for names in router["streams"].values())
        description = (
            f"{len(routers)} routers, {regions} regions, {streams} stream registers; "
            f".bin {result.bin_path.stat().st_size:,} bytes, .json {result.json_path.stat().st_size:,} bytes; "
            f"taken in {seconds:.1f} s"
        )
    return problems, description


def main(argv):
    parser = argparse.ArgumentParser(description="Checks that capture can take a snapshot of fabric manifests.")
    parser.add_argument("paths", nargs="+", metavar="PATH", help="a manifest, or a directory to search for them")
    args = parser.parse_args(argv)

    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)

    missing = [arg for arg in args.paths if not pathlib.Path(arg).exists()]
    if missing:
        print(f"Fabric snapshot: {' '.join(missing)} does not exist")
        return 1
    paths = find_manifests(args.paths)
    if not paths:
        print(f"Fabric snapshot: no manifests found under {' '.join(args.paths)}")
        return 1
    failed = 0
    for path in paths:
        try:
            problems, description = check_manifest(path, validator)
        except ManifestError as error:
            failed += 1
            print(f"Fabric snapshot: capture can't read {path}: {error}")
            continue
        if not problems:
            print(f"{path}: {description}")
            continue
        failed += 1
        print(f"Fabric snapshot: the snapshot of {path} is wrong ({len(problems)} problems)")
        for problem in problems[:MAX_PROBLEMS_SHOWN]:
            print(f"  {problem}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
