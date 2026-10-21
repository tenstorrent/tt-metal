# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Cases for loading a snapshot with its .bin and its manifest, from files written to a temporary directory.

Run from the repository root: python3 -m unittest tt_metal.fabric.debug.visualizer.decode.tests.test_inputs
"""

import hashlib
import json
import pathlib
import shutil
import subprocess
import tempfile
import unittest

from tt_metal.fabric.debug.visualizer.common.manifest_lookup import ManifestError
from tt_metal.fabric.debug.visualizer.decode.inputs import (
    Checkout,
    InputError,
    get_build_warnings,
    get_checkout,
    load_input,
)

SHA = "a" * 40
OTHER_SHA = "b" * 40
RAW = bytes(range(16))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest_bytes(**run) -> bytes:
    return json.dumps({"kind": "fabric_manifest", "run": {"git_sha": SHA, "git_dirty": False, **run}}).encode()


MANIFEST = manifest_bytes(arch="wormhole_b0")


def router_entry():
    """A router's entry with every key decode reads, as capture writes it for a router it read without an error."""
    return {
        "liveness": [],
        "started_ns": 0,
        "core_registers": {},
        "areas": {},
        "regions": {},
        "streams": {},
        "error": None,
    }


class LoadInputTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = pathlib.Path(temporary.name)

    def write(self, change=None):
        """A manifest, then a snapshot and its .bin in another directory, as capture writes them with -o."""
        self.manifest_path = self.directory / "generated" / "fabric_manifest_rank_1_of_1.json"
        self.snapshot_path = self.directory / "snapshots" / "fabric_snapshot_rank_1_of_1_20261021T120000Z.json"
        self.bin_path = self.snapshot_path.with_suffix(".bin")
        for path in (self.manifest_path, self.snapshot_path):
            path.parent.mkdir(exist_ok=True)
        self.manifest_path.write_bytes(MANIFEST)
        self.bin_path.write_bytes(RAW)
        snapshot = {
            "kind": "fabric_snapshot",
            "snapshot_version": 1,
            "manifest": {"file": str(self.manifest_path), "sha256": sha256(MANIFEST), "run": {}},
            "raw": {"file": self.bin_path.name, "size": len(RAW), "sha256": sha256(RAW)},
            "routers": {"M0/C0/E0": router_entry()},
        }
        if change is not None:
            change(snapshot)
        self.snapshot_path.write_text(json.dumps(snapshot))
        return self.snapshot_path

    def test_a_snapshot_loads_with_its_bin_and_manifest(self):
        loaded = load_input(self.write(), Checkout(SHA, False))
        self.assertEqual(loaded.raw, RAW)
        self.assertEqual(loaded.manifest_path, self.manifest_path)
        self.assertEqual(loaded.manifest["run"]["arch"], "wormhole_b0")
        self.assertEqual(loaded.snapshot_sha256, sha256(self.snapshot_path.read_bytes()))
        self.assertEqual(loaded.warnings, ())

    def test_a_moved_snapshot_finds_its_manifest_beside_it(self):
        snapshot_path = self.write(lambda snapshot: snapshot["manifest"].update(file="/elsewhere/m.json"))
        beside = snapshot_path.parent / "m.json"
        shutil.copy(self.manifest_path, beside)
        self.assertEqual(load_input(snapshot_path, None).manifest_path, beside)

    def test_manifests_given_are_chosen_by_sha256(self):
        snapshot_path = self.write()
        other = self.directory / "other.json"
        other.write_bytes(manifest_bytes(arch="blackhole"))
        self.assertEqual(load_input(snapshot_path, None, [other, self.manifest_path]).manifest_path, self.manifest_path)

    def test_another_runs_manifest_is_refused(self):
        snapshot_path = self.write()
        self.manifest_path.write_bytes(manifest_bytes(arch="blackhole"))
        with self.assertRaisesRegex(InputError, r"has sha256 \w+, but snapshot .* was taken with"):
            load_input(snapshot_path, None)

    def test_manifests_given_that_dont_match_are_refused(self):
        snapshot_path = self.write()
        other = self.directory / "other.json"
        other.write_bytes(manifest_bytes(arch="blackhole"))
        with self.assertRaisesRegex(InputError, "none of the 1 manifests given"):
            load_input(snapshot_path, None, [other])

    def test_a_missing_manifest_is_refused(self):
        snapshot_path = self.write(lambda snapshot: snapshot["manifest"].update(file="/elsewhere/m.json"))
        with self.assertRaisesRegex(InputError, "pass it with --manifest"):
            load_input(snapshot_path, None)

    def test_a_bin_that_isnt_the_snapshots_is_refused(self):
        for name, change, named in (
            ("missing", lambda: self.bin_path.unlink(), "can't read the snapshot's .bin"),
            ("short", lambda: self.bin_path.write_bytes(RAW[:-1]), "is 15 bytes; snapshot .* says 16"),
            ("other bytes", lambda: self.bin_path.write_bytes(bytes(16)), "sha256 isn't the one"),
        ):
            with self.subTest(name):
                snapshot_path = self.write()
                change()
                with self.assertRaisesRegex(InputError, named):
                    load_input(snapshot_path, None)

    def test_a_file_that_isnt_a_snapshot_is_refused(self):
        def replace(key, value):
            return lambda snapshot: snapshot.__setitem__(key, value)

        for name, change, named in (
            ("another kind", replace("kind", "fabric_manifest"), "isn't a fabric snapshot"),
            ("another version", replace("snapshot_version", 2), "is version 2; decode reads version 1"),
            ("no routers", lambda snapshot: snapshot.pop("routers"), "has no routers"),
            ("a size that isn't a number", lambda snapshot: snapshot["raw"].update(size="16"), r"raw\.size is '16'"),
            ("no manifest hash", lambda snapshot: snapshot["manifest"].pop("sha256"), r"has no manifest\.sha256"),
            (
                "a router with no liveness samples",
                lambda snapshot: snapshot["routers"]["M0/C0/E0"].pop("liveness"),
                r"has no routers\.M0/C0/E0\.liveness",
            ),
            (
                "a router whose start time isn't a number",
                lambda snapshot: snapshot["routers"]["M0/C0/E0"].update(started_ns="0"),
                r"routers\.M0/C0/E0\.started_ns is '0', not an int",
            ),
            (
                "a router with no error key",
                lambda snapshot: snapshot["routers"]["M0/C0/E0"].pop("error"),
                r"has no routers\.M0/C0/E0\.error",
            ),
        ):
            with self.subTest(name), self.assertRaisesRegex(InputError, named):
                load_input(self.write(change), None)

    def test_a_router_capture_couldnt_open_loads(self):
        def unopened(snapshot):
            snapshot["routers"]["M0/C0/E0"].update(started_ns=None, error={"stage": "open"})

        self.assertIsNone(load_input(self.write(unopened), None).snapshot["routers"]["M0/C0/E0"]["started_ns"])

    def test_a_file_that_isnt_json_is_refused(self):
        snapshot_path = self.write()
        snapshot_path.write_text("{")
        with self.assertRaisesRegex(InputError, "isn't JSON"):
            load_input(snapshot_path, None)

    def test_build_warnings_come_with_the_input(self):
        warnings = load_input(self.write(), Checkout(OTHER_SHA, False)).warnings
        self.assertEqual(warnings, ("the run was built from aaaaaaaaaa; decode is at bbbbbbbbbb",))


class GetBuildWarningsTest(unittest.TestCase):
    def warnings(self, checkout, **run):
        return get_build_warnings({"run": run}, checkout)

    def test_the_same_clean_commit_has_no_warning(self):
        self.assertEqual(self.warnings(Checkout(SHA, False), git_sha=SHA, git_dirty=False), [])

    def test_each_case_decode_cant_vouch_for(self):
        for name, checkout, run, named in (
            ("no build identity", Checkout(SHA, False), {}, r"no run\.git_sha"),
            ("built outside git", Checkout(SHA, False), {"git_sha": "", "git_dirty": False}, "built outside a git"),
            ("decode outside git", None, {"git_sha": SHA, "git_dirty": False}, "decode isn't in a git checkout"),
            ("another commit", Checkout(OTHER_SHA, False), {"git_sha": SHA, "git_dirty": False}, "decode is at bbbb"),
            (
                "the run had local changes",
                Checkout(SHA, False),
                {"git_sha": SHA, "git_dirty": True},
                "built from aaaaaaaaaa with local changes; decode is at aaaaaaaaaa$",
            ),
            (
                "decode has local changes",
                Checkout(SHA, True),
                {"git_sha": SHA, "git_dirty": False},
                "built from aaaaaaaaaa; decode is at aaaaaaaaaa with local changes",
            ),
        ):
            with self.subTest(name):
                (warning,) = self.warnings(checkout, **run)
                self.assertRegex(warning, named)

    def test_a_dirty_flag_that_isnt_a_boolean_is_refused(self):
        with self.assertRaisesRegex(ManifestError, r"run\.git_dirty"):
            self.warnings(None, git_sha=SHA, git_dirty="no")


@unittest.skipUnless(shutil.which("git"), "needs git")
class GetCheckoutTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = pathlib.Path(temporary.name)

    def git(self, *args):
        identity = ["-c", "user.name=test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false"]
        return subprocess.run(
            ["git", "-C", str(self.directory), *identity, *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    def test_a_checkout_is_its_commit_and_whether_tracked_files_changed(self):
        self.git("init", "-q")
        (self.directory / "tracked.txt").write_text("one")
        self.git("add", "tracked.txt")
        self.git("commit", "-q", "-m", "one")
        head = self.git("rev-parse", "HEAD")
        self.assertEqual(get_checkout(self.directory), Checkout(head, False))
        (self.directory / "untracked.txt").write_text("new")
        self.assertEqual(get_checkout(self.directory), Checkout(head, False))
        (self.directory / "tracked.txt").write_text("two")
        self.assertEqual(get_checkout(self.directory), Checkout(head, True))

    def test_a_directory_outside_git_has_no_checkout(self):
        self.assertIsNone(get_checkout(self.directory))


if __name__ == "__main__":
    unittest.main()
