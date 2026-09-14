# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path

from tt_metal.fabric.debug.visualizer.capture.manifest import ManifestError, load_manifest


def router(eth_chan: int, direction: str = "E") -> dict:
    return {
        "eth_chan": eth_chan,
        "direction": direction,
        "routing_plane": 0,
        "link_class": "intramesh",
        "logical_core": [0, eth_chan],
        "virtual_core": [18 + eth_chan, 16],
    }


def chip(
    chip_id: int,
    *,
    is_local: bool,
    physical_chip_id: int | None,
    routers: list[dict],
) -> dict:
    return {
        "fabric_chip_id": chip_id,
        "mesh_coord": [0, chip_id],
        "physical_chip_id": physical_chip_id,
        "asic_id": None,
        "is_local": is_local,
        "routers": routers,
    }


def manifest(chips: list[dict]) -> dict:
    return {
        "manifest_version": 1,
        "kind": "fabric_debug_manifest",
        "run": {
            "arch": "WORMHOLE_B0",
            "fabric_config": "FABRIC_2D",
            "fabric_type": "MESH",
            "host_rank": 0,
            "mpi_rank": 0,
            "world_size": 2,
        },
        "meshes": [
            {
                "mesh_id": 0,
                "shape": [1, len(chips)],
                "chips": chips,
            }
        ],
        "links": [],
    }


class ManifestTest(unittest.TestCase):
    def load(self, data: dict):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            return load_manifest(path)

    def test_enumerates_only_this_hosts_routers(self):
        loaded = self.load(
            manifest(
                [
                    chip(0, is_local=True, physical_chip_id=4, routers=[router(1)]),
                    chip(1, is_local=False, physical_chip_id=None, routers=[]),
                    chip(2, is_local=True, physical_chip_id=7, routers=[router(8, "W")]),
                ]
            )
        )

        # physical chip 7 models a UMD-remote chip on the same T3K host. It is
        # still local to this process and therefore remains a legal peek target.
        self.assertEqual(
            [(target.chip_id, target.physical_chip_id, target.eth_chan) for target in loaded.router_targets],
            [(0, 4, 1), (2, 7, 8)],
        )
        self.assertEqual(
            loaded.router_targets[1].endpoint(),
            {"mesh_id": 0, "chip_id": 2, "eth_chan": 8},
        )

    def test_targets_are_sorted_by_global_endpoint(self):
        loaded = self.load(
            manifest(
                [
                    chip(
                        1,
                        is_local=True,
                        physical_chip_id=0,
                        routers=[router(9, "W"), router(0)],
                    ),
                    chip(0, is_local=True, physical_chip_id=4, routers=[router(6)]),
                ]
            )
        )

        self.assertEqual(
            [(target.chip_id, target.eth_chan) for target in loaded.router_targets],
            [(0, 6), (1, 0), (1, 9)],
        )

    def test_rejects_duplicate_router_endpoint(self):
        data = manifest(
            [chip(0, is_local=True, physical_chip_id=4, routers=[router(1), router(1)])]
        )

        with self.assertRaisesRegex(ManifestError, "duplicate local router endpoint"):
            self.load(data)

    def test_rejects_local_chip_without_physical_id(self):
        data = manifest([chip(0, is_local=True, physical_chip_id=None, routers=[router(1)])])

        with self.assertRaisesRegex(ManifestError, "physical_chip_id must be an integer"):
            self.load(data)

    def test_rejects_unsupported_manifest_version(self):
        data = manifest([])
        data["manifest_version"] = 123456

        with self.assertRaisesRegex(ManifestError, "unsupported manifest_version"):
            self.load(data)


if __name__ == "__main__":
    unittest.main()
