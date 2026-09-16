# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import unittest
from pathlib import Path

import jsonschema

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema" / "fabric_debug_decoded_schema.json"
INDEX_NAMES = {
    "line_1d.json",
    "mesh_2d.json",
    "torus_xy.json",
    "two_mesh.json",
    "stalled_link.json",
    "coverage_holes.json",
}


def load(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


class FixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(cls.schema)
        cls.validator = jsonschema.Draft202012Validator(
            cls.schema,
            format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER,
        )

    def test_index_names_existing_files(self):
        index = load("index.json")
        files = {entry["file"] for entry in index}
        self.assertEqual(files, INDEX_NAMES)
        for name in files:
            self.assertTrue((FIXTURE_DIR / name).is_file(), name)

    def test_every_fixture_validates(self):
        for path in sorted(FIXTURE_DIR.glob("*.json")):
            if path.name == "index.json":
                continue
            with self.subTest(path.name):
                self.validator.validate(json.loads(path.read_text(encoding="utf-8")))

    def test_stalled_link_is_the_hot_edge(self):
        decoded = load("stalled_link.json")
        scores = [link["stall_score"] for link in decoded["topology"]["links"]]
        self.assertIn(1.0, scores)
        self.assertEqual(max(score for score in scores if score is not None), 1.0)
        hot = next(link for link in decoded["topology"]["links"] if link["stall_score"] == 1.0)
        src = next(router for router in decoded["routers"] if router["id"] == hot["src"])
        self.assertEqual(src["stall_score"], 1.0)
        self.assertTrue(any(edge["free_slots"] == 0 for edge in src["channels"]["downstream"]))
        self.assertTrue(
            any(
                sender["depth"] and sender["occupied"] == sender["depth"]
                for sender in src["channels"]["senders"]
            )
        )

    def test_torus_has_wrap_arcs(self):
        decoded = load("torus_xy.json")
        self.assertTrue(any(link.get("wrap") for link in decoded["topology"]["links"]))
        self.assertFalse(any(link.get("wrap") for link in load("mesh_2d.json")["topology"]["links"]))

    def test_coverage_holes_statuses(self):
        statuses = {router["capture"]["status"] for router in load("coverage_holes.json")["routers"]}
        self.assertTrue({"not_captured", "reset", "torn"} <= statuses)
        self.assertEqual(load("coverage_holes.json")["coverage"]["identity_mismatch"], 1)

    def test_two_mesh_has_intermesh_gap(self):
        decoded = load("two_mesh.json")
        self.assertEqual(len(decoded["topology"]["meshes"]), 2)
        self.assertTrue(any(link.get("link_class") == "intermesh" for link in decoded["topology"]["links"]))

    def test_line_coords_are_a_row(self):
        chips = load("line_1d.json")["topology"]["meshes"][0]["chips"]
        self.assertEqual([chip["mesh_coord"] for chip in chips], [[0, 0], [0, 1], [0, 2], [0, 3]])

    def test_mesh_contains_clickable_header_candidates(self):
        decoded = load("mesh_2d.json")
        slots = [
            slot
            for router in decoded["routers"]
            for ring in router["rings"]
            for slot in ring.get("slots", [])
        ]
        self.assertTrue(slots)
        self.assertTrue(any(slot["header"] and slot["header"]["plausible"] for slot in slots))
        self.assertTrue(any(slot["header"] and not slot["header"]["plausible"] for slot in slots))
        plausible = next(slot["header"] for slot in slots if slot["header"] and slot["header"]["plausible"])
        self.assertIn("payload_size_bytes", plausible)
        self.assertIn("noc_send_type", plausible)
        self.assertIn("routing", plausible)


if __name__ == "__main__":
    unittest.main()
