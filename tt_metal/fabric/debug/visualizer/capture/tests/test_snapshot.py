# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Cases for taking a snapshot, with a fake device and clock.

Run from the repository root: python3 -m unittest tt_metal.fabric.debug.visualizer.capture.tests.test_snapshot
"""

import collections
import copy
import datetime
import hashlib
import itertools
import json
import pathlib
import tempfile
import unittest

from tt_metal.fabric.debug.visualizer.capture.reads import REGISTER_BYTES, ManifestError, get_read_plan
from tt_metal.fabric.debug.visualizer.capture.snapshot import CaptureError, OpenedRouter, take_snapshot
from tt_metal.fabric.debug.visualizer.capture.tests.test_reads import MANIFEST

STARTED = datetime.datetime(2026, 10, 20, 12, 0, 0, tzinfo=datetime.timezone.utc)
FINISHED = datetime.datetime(2026, 10, 20, 12, 0, 5, tzinfo=datetime.timezone.utc)
PLAN = get_read_plan(MANIFEST)
E1, S1 = PLAN.routers
REMOTE_SRC_31 = E1.streams[(31, "remote_src")].address
NAME = "fabric_snapshot_rank_1_of_1_20261020T120000Z"


def device_bytes(path: str, address: int, size: int, nth: int) -> bytes:
    """What the fake device returns for the `nth` read of `size` bytes at `address` on router `path`."""
    seed = hashlib.sha256(f"{path} {address:#x} {nth}".encode()).digest()
    return (seed * (size // len(seed) + 1))[:size]


def device_word(path: str, address: int, nth: int) -> int:
    return int.from_bytes(device_bytes(path, address, REGISTER_BYTES, nth), "little")


class FakeDevice:
    version = "fake 1.0"

    def __init__(self, unopenable=(), failing=None, short=None):
        self.unopenable = set(unopenable)
        self.failing = failing  # (router path, address) whose read raises
        self.short = short  # (router path, address) whose read returns 2 bytes
        self.opened = 0
        self.archs = set()  # the arch each open_router was given
        self.reads = []  # (router path, address, size) in the order read
        self.counts = collections.Counter()

    def open_router(self, router, arch):
        self.archs.add(arch)
        if router.path in self.unopenable:
            raise CaptureError("chip not found")
        self.opened += 1
        return OpenedRouter(handle=router.path, ttexalens_id=self.opened)

    def read(self, opened, address, size):
        path = opened.handle
        self.reads.append((path, address, size))
        if (path, address) == self.failing:
            raise CaptureError("timed out")
        data = device_bytes(path, address, size, self.counts[(path, address)])
        self.counts[(path, address)] += 1
        return data[:2] if (path, address) == self.short else data


def main_pass_reads(router) -> list:
    """The reads of the main pass on `router`, in order."""
    words = [(r.address, REGISTER_BYTES) for r in PLAN.core_registers.values()]
    streams = [(r.address, REGISTER_BYTES) for r in router.streams.values()]
    l1 = [(r.address, r.size) for r in [*PLAN.areas.values(), *router.regions.values()]]
    return [(router.path, *read) for read in words + streams + l1 + streams]


class SnapshotCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.dir = pathlib.Path(directory.name)
        self.manifest_path = self.dir / "fabric_manifest_rank_1_of_1.json"
        self.out_dir = self.dir / "out"

    def take(self, device=None, manifest=MANIFEST):
        self.manifest_path.write_text(json.dumps(manifest))
        self.device = device or FakeDevice()
        self.result = take_snapshot(
            self.manifest_path,
            self.out_dir,
            self.device,
            host="test-host",
            argv=["capture", str(self.manifest_path)],
            monotonic_ns=itertools.count(1000, 1000).__next__,
            utc_now=iter([STARTED, FINISHED]).__next__,
        )
        self.snapshot = json.loads(self.result.json_path.read_text())
        self.raw = self.result.bin_path.read_bytes()

    def assert_l1(self, router, kind: str, plan_reads: dict):
        entries = self.snapshot["routers"][router.path][kind]
        for key, read in plan_reads.items():
            entry = entries[key]
            self.assertEqual((entry["address"], entry["size"]), (read.address, read.size), key)
            data = self.raw[entry["offset"] : entry["offset"] + entry["size"]]
            self.assertEqual(data, device_bytes(router.path, read.address, read.size, 0), key)


class Snapshots(SnapshotCase):
    def test_files(self):
        self.take()
        self.assertEqual(self.result.json_path, self.out_dir / f"{NAME}.json")
        self.assertEqual(self.result.bin_path, self.out_dir / f"{NAME}.bin")
        self.assertEqual(sorted(p.name for p in self.out_dir.iterdir()), [f"{NAME}.bin", f"{NAME}.json"])
        self.assertEqual(self.result.routers, 2)
        self.assertEqual(self.result.errors, {})

    def test_header(self):
        self.take()
        self.assertEqual(self.snapshot["kind"], "fabric_snapshot")
        self.assertEqual(self.snapshot["snapshot_version"], 1)
        self.assertEqual(
            self.snapshot["manifest"],
            {
                "file": str(self.manifest_path.resolve()),
                "sha256": hashlib.sha256(self.manifest_path.read_bytes()).hexdigest(),
                "run": MANIFEST["run"],
            },
        )
        self.assertEqual(
            self.snapshot["capture"],
            {
                "started_at": "2026-10-20T12:00:00Z",
                "finished_at": "2026-10-20T12:00:05Z",
                "host": "test-host",
                "argv": ["capture", str(self.manifest_path)],
                "ttexalens": "fake 1.0",
            },
        )
        self.assertEqual(
            self.snapshot["raw"],
            {"file": f"{NAME}.bin", "size": len(self.raw), "sha256": hashlib.sha256(self.raw).hexdigest()},
        )

    def test_routers(self):
        self.take()
        self.assertEqual(list(self.snapshot["routers"]), ["M0/C0/E1", "M1/C0/S1"])
        self.assertEqual(self.device.archs, {MANIFEST["run"]["arch"]})
        entry = self.snapshot["routers"]["M0/C0/E1"]
        self.assertEqual(entry["device"], {"unique_id": "0x000000036190302b", "ttexalens_id": 1})
        self.assertEqual(entry["core"], {"translated": [25, 16], "logical": [0, 0]})
        self.assertLess(entry["started_ns"], entry["finished_ns"])
        self.assertIsNone(entry["error"])

    def test_l1_entries_are_at_their_offsets_in_order(self):
        self.take()
        self.assert_l1(E1, "areas", PLAN.areas)
        self.assert_l1(E1, "regions", E1.regions)
        self.assert_l1(S1, "areas", PLAN.areas)
        self.assert_l1(S1, "regions", S1.regions)
        offsets = [
            entry["offset"]
            for router in self.snapshot["routers"].values()
            for entry in [*router["areas"].values(), *router["regions"].values()]
        ]
        sizes = [
            entry["size"]
            for router in self.snapshot["routers"].values()
            for entry in [*router["areas"].values(), *router["regions"].values()]
        ]
        self.assertEqual(offsets, list(itertools.accumulate([0, *sizes[:-1]])))
        self.assertEqual(len(self.raw), sum(sizes))

    def test_words(self):
        self.take()
        entry = self.snapshot["routers"]["M0/C0/E1"]
        self.assertEqual(
            entry["core_registers"],
            {name: device_word(E1.path, r.address, 0) for name, r in PLAN.core_registers.items()},
        )
        self.assertEqual(
            entry["streams"]["31"]["remote_src"],
            {"before": device_word(E1.path, REMOTE_SRC_31, 0), "after": device_word(E1.path, REMOTE_SRC_31, 1)},
        )
        self.assertEqual(
            sorted((int(i), name) for i, registers in entry["streams"].items() for name in registers),
            sorted(E1.streams),
        )
        self.assertEqual(self.snapshot["routers"]["M1/C0/S1"]["streams"], {})

    def test_main_pass_order(self):
        self.take()
        self.assertEqual(self.device.reads, main_pass_reads(E1) + main_pass_reads(S1))


class Errors(SnapshotCase):
    def test_a_router_that_does_not_open(self):
        self.take(FakeDevice(unopenable={"M0/C0/E1"}))
        entry = self.snapshot["routers"]["M0/C0/E1"]
        error = {"stage": "open", "path": None, "message": "chip not found"}
        self.assertEqual(entry["error"], error)
        self.assertEqual(self.result.errors, {"M0/C0/E1": error})
        self.assertIsNone(entry["device"])
        self.assertIsNone(entry["started_ns"])
        self.assertEqual((entry["core_registers"], entry["areas"], entry["regions"], entry["streams"]), ({},) * 4)
        self.assertEqual(self.device.reads, main_pass_reads(S1))
        self.assert_l1(S1, "regions", S1.regions)

    def test_a_failed_read_stops_its_router_and_keeps_what_was_read(self):
        failing = E1.regions["channels/receivers/vc0/ch0/ring_buffer"]
        self.take(FakeDevice(failing=("M0/C0/E1", failing.address)))
        entry = self.snapshot["routers"]["M0/C0/E1"]
        self.assertEqual(
            entry["error"],
            {"stage": "regions", "path": "channels/receivers/vc0/ch0/ring_buffer", "message": "timed out"},
        )
        regions_read = list(E1.regions)[: list(E1.regions).index("channels/receivers/vc0/ch0/ring_buffer")]
        self.assertEqual(list(entry["regions"]), regions_read)
        self.assert_l1(E1, "regions", {key: E1.regions[key] for key in regions_read})
        self.assert_l1(E1, "areas", PLAN.areas)
        self.assertTrue(all("after" not in r for registers in entry["streams"].values() for r in registers.values()))
        self.assertIsNotNone(entry["finished_ns"])
        reads_until_failure = len(PLAN.core_registers) + len(E1.streams) + len(PLAN.areas) + len(regions_read) + 1
        self.assertEqual(self.device.reads, main_pass_reads(E1)[:reads_until_failure] + main_pass_reads(S1))
        self.assertIsNone(self.snapshot["routers"]["M1/C0/S1"]["error"])

    def test_a_short_read_is_an_error(self):
        address = PLAN.core_registers["wall_clock_high"].address
        self.take(FakeDevice(short=("M0/C0/E1", address)))
        self.assertEqual(
            self.snapshot["routers"]["M0/C0/E1"]["error"],
            {"stage": "core_registers", "path": "wall_clock_high", "message": f"read 2 bytes at {address:#x}, not 4"},
        )

    def test_a_failed_stream_read_names_the_stream_and_register(self):
        self.take(FakeDevice(failing=("M0/C0/E1", REMOTE_SRC_31)))
        self.assertEqual(
            self.snapshot["routers"]["M0/C0/E1"]["error"],
            {"stage": "streams_before", "path": "31/remote_src", "message": "timed out"},
        )

    def test_a_bad_manifest_writes_nothing(self):
        manifest = copy.deepcopy(MANIFEST)
        manifest["kind"] = "something_else"
        with self.assertRaises(ManifestError):
            self.take(manifest=manifest)
        self.assertFalse(self.out_dir.exists())

    def test_an_existing_snapshot_is_not_overwritten(self):
        self.take()
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            self.take()


if __name__ == "__main__":
    unittest.main()
