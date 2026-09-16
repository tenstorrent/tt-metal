# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generate committed viewer fixtures from decode's synthetic artifacts."""

from __future__ import annotations

import hashlib
import json
import tempfile
from copy import deepcopy
from pathlib import Path

from tt_metal.fabric.debug.visualizer.capture.tests.test_manifest import chip, manifest, router
from tt_metal.fabric.debug.visualizer.decode.inputs import discover_inputs
from tt_metal.fabric.debug.visualizer.decode.output import build_decoded
from tt_metal.fabric.debug.visualizer.decode.tests.fixtures import router_sample, write_input

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"

INDEX = [
    {"id": "line_1d", "title": "1D line", "file": "line_1d.json"},
    {"id": "mesh_2d", "title": "2×2 mesh", "file": "mesh_2d.json"},
    {"id": "torus_xy", "title": "2×2 torus wrap", "file": "torus_xy.json"},
    {"id": "two_mesh", "title": "Two meshes", "file": "two_mesh.json"},
    {"id": "stalled_link", "title": "Stalled link", "file": "stalled_link.json"},
    {"id": "coverage_holes", "title": "Coverage holes", "file": "coverage_holes.json"},
]


def _ep(mesh_id: int, chip_id: int, eth_chan: int) -> dict[str, int]:
    return {"mesh_id": mesh_id, "chip_id": chip_id, "eth_chan": eth_chan}


def _link(src, dst, direction: str, *, wrap: bool = False, link_class: str = "intramesh") -> dict:
    return {
        "src": src,
        "dst": dst,
        "direction": direction,
        "routing_plane": 0,
        "link_class": link_class,
        "wrap": wrap,
        "cross_host": False,
    }


def _spec_shape(rows: list[dict]) -> list[int]:
    coords = [row["mesh_coord"] for row in rows if row.get("mesh_coord")]
    if not coords:
        return [1, len(rows)]
    return [max(coord[0] for coord in coords) + 1, max(coord[1] for coord in coords) + 1]


def _sanitize(decoded: dict) -> dict:
    decoded["generated_at"] = "2026-09-15T00:00:00Z"
    for item in decoded["inputs"]:
        item["manifest"]["path"] = "fixtures/source/manifest.json"
        if item.get("snapshot"):
            item["snapshot"]["path"] = "fixtures/source/snapshot.json"
            item["snapshot"]["provenance"]["hostname"] = "fixture-host"
        if item.get("raw"):
            item["raw"]["file"] = "snapshot.bin"
    return decoded


def _force_identity(decoded: dict) -> dict:
    for router_row in decoded["routers"]:
        if router_row["capture"]["status"] == "not_captured":
            continue
        router_row["identity"] = {
            "my_mesh_id": router_row["id"]["mesh_id"],
            "my_device_id": router_row["id"]["chip_id"],
            "matches": True,
        }
    decoded["coverage"]["identity_mismatch"] = 0
    return decoded


def _decode_case(chip_specs: list[dict], links: list[dict], *, topology: str = "Mesh") -> dict:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        template_manifest, template_snapshot, raw_path = write_input(root / "template")
        template = json.loads(template_manifest.read_text(encoding="utf-8"))
        snapshot_template = json.loads(template_snapshot.read_text(encoding="utf-8"))
        blobs = snapshot_template["samples"][0]["routers"][0]["blobs"]
        instance = deepcopy(template["meshes"][0]["chips"][0]["routers"][0]["instance"])

        chips_by_mesh: dict[int, list[dict]] = {}
        samples = []
        for spec in chip_specs:
            mesh_id = spec.get("mesh_id", 0)
            captured = spec.get("captured", True)
            routers = []
            for entry in spec["routers"]:
                row = router(entry["eth_chan"], entry["direction"])
                row["instance"] = deepcopy(instance)
                if spec.get("link_class"):
                    row["link_class"] = spec["link_class"]
                    if spec["link_class"] == "intermesh" and "is_inter_mesh" in row["instance"]:
                        row["instance"]["is_inter_mesh"] = True
                routers.append(row)
            chip_row = chip(
                spec["chip_id"],
                is_local=captured,
                physical_chip_id=spec["chip_id"] if captured else None,
                asic_id=f"0x{spec['chip_id'] + 1:016x}" if captured else None,
                routers=routers,
            )
            chip_row["mesh_coord"] = spec["coord"]
            chips_by_mesh.setdefault(mesh_id, []).append(chip_row)
            if not captured:
                continue
            for entry in spec["routers"]:
                sample = router_sample(spec["chip_id"], entry["eth_chan"], status=spec.get("status", "ok"))
                sample["id"]["mesh_id"] = mesh_id
                sample["blobs"] = deepcopy(blobs)
                samples.append(sample)

        manifest_data = manifest(chips_by_mesh.get(0, []))
        manifest_data["meshes"] = [
            {"mesh_id": mesh_id, "shape": _spec_shape(rows), "chips": rows}
            for mesh_id, rows in sorted(chips_by_mesh.items())
        ]
        manifest_data["links"] = links
        manifest_data["run"]["world_size"] = 1
        manifest_data["fabric_context"] = deepcopy(template["fabric_context"])
        manifest_data["fabric_context"]["topology"] = topology
        manifest_data["hal"] = deepcopy(template["hal"])
        manifest_data["layouts"] = deepcopy(template["layouts"])
        case = root / "case"
        case.mkdir()
        manifest_path = case / "manifest_0.json"
        manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
        sidecar = case / raw_path.name
        sidecar.write_bytes(raw_path.read_bytes())
        snapshot = snapshot_template
        snapshot["samples"][0]["routers"] = samples
        snapshot["raw"]["file"] = sidecar.name
        snapshot["manifest"]["run"]["world_size"] = 1
        snapshot["manifest"]["sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        (case / "snapshot_0.json").write_text(json.dumps(snapshot), encoding="utf-8")
        decoded = build_decoded(
            discover_inputs([case]),
            generated_at="2026-09-15T00:00:00Z",
            slots="none",
        )
        return _sanitize(_force_identity(decoded))


def line_1d() -> dict:
    chips = [
        {"chip_id": index, "coord": [0, index], "routers": [{"eth_chan": 0, "direction": "E"}]}
        for index in range(4)
    ]
    links = [_link(_ep(0, index, 0), _ep(0, index + 1, 0), "E") for index in range(3)]
    return _decode_case(chips, links, topology="Linear")


def mesh_2d(*, wrap: bool = False) -> dict:
    ports = {"E": 0, "W": 1, "N": 2, "S": 3}
    coords = {0: [0, 0], 1: [0, 1], 2: [1, 0], 3: [1, 1]}
    chips = [
        {
            "chip_id": chip_id,
            "coord": coord,
            "routers": [{"eth_chan": eth, "direction": direction} for direction, eth in ports.items()],
        }
        for chip_id, coord in coords.items()
    ]
    neighbors = {
        (0, "E"): (1, "W"),
        (0, "S"): (2, "N"),
        (1, "W"): (0, "E"),
        (1, "S"): (3, "N"),
        (2, "N"): (0, "S"),
        (2, "E"): (3, "W"),
        (3, "N"): (1, "S"),
        (3, "W"): (2, "E"),
    }
    wrap_neighbors = {
        (0, "W"): (1, "E"),
        (0, "N"): (2, "S"),
        (1, "E"): (0, "W"),
        (1, "N"): (3, "S"),
        (2, "S"): (0, "N"),
        (2, "W"): (3, "E"),
        (3, "S"): (1, "N"),
        (3, "E"): (2, "W"),
    }
    links = []
    for (src_chip, direction), (dst_chip, dst_dir) in neighbors.items():
        links.append(_link(_ep(0, src_chip, ports[direction]), _ep(0, dst_chip, ports[dst_dir]), direction))
    if wrap:
        for (src_chip, direction), (dst_chip, dst_dir) in wrap_neighbors.items():
            links.append(
                _link(_ep(0, src_chip, ports[direction]), _ep(0, dst_chip, ports[dst_dir]), direction, wrap=True)
            )
    return _decode_case(chips, links, topology="Torus" if wrap else "Mesh")


def two_mesh() -> dict:
    chips = [
        {"mesh_id": 0, "chip_id": 0, "coord": [0, 0], "routers": [{"eth_chan": 0, "direction": "E"}]},
        {
            "mesh_id": 1,
            "chip_id": 0,
            "coord": [0, 0],
            "routers": [{"eth_chan": 0, "direction": "Z"}],
            "link_class": "intermesh",
        },
    ]
    links = [
        _link(_ep(0, 0, 0), _ep(1, 0, 0), "Z", link_class="intermesh"),
        _link(_ep(1, 0, 0), _ep(0, 0, 0), "Z", link_class="intermesh"),
    ]
    return _decode_case(chips, links, topology="Mesh")


def stalled_link() -> dict:
    decoded = mesh_2d()
    target = next(
        router_row
        for router_row in decoded["routers"]
        if router_row["id"] == {"mesh_id": 0, "chip_id": 0, "eth_chan": 0}
    )
    target["stall_score"] = 1.0
    for sender in target["channels"]["senders"]:
        if sender["depth"]:
            sender["occupied"] = sender["depth"]
            sender["free_slots"] = 0
    for edge in target["channels"]["downstream"]:
        edge["free_slots"] = 0
    for ring in target["rings"]:
        if ring["id"].startswith("sender.") and ring["depth"]:
            ring["occupied_count"] = ring["depth"]
            ring["occupancy_source"] = "stream"
            ring["occupancy_status"] = "ok"
    for link in decoded["topology"]["links"]:
        if link["src"] == target["id"]:
            link["stall_score"] = 1.0
            link["status"] = "ok"
            break
    return decoded


def coverage_holes() -> dict:
    chips = [
        {"chip_id": 0, "coord": [0, 0], "routers": [{"eth_chan": 0, "direction": "E"}], "status": "ok"},
        {"chip_id": 1, "coord": [0, 1], "routers": [{"eth_chan": 0, "direction": "W"}], "captured": False},
        {"chip_id": 2, "coord": [1, 0], "routers": [{"eth_chan": 0, "direction": "N"}], "status": "reset"},
        {"chip_id": 3, "coord": [1, 1], "routers": [{"eth_chan": 0, "direction": "S"}], "status": "torn"},
    ]
    links = [
        _link(_ep(0, 0, 0), _ep(0, 1, 0), "E"),
        _link(_ep(0, 2, 0), _ep(0, 0, 0), "N"),
        _link(_ep(0, 3, 0), _ep(0, 1, 0), "S"),
    ]
    decoded = _decode_case(chips, links)
    for router_row in decoded["routers"]:
        if router_row["id"]["chip_id"] == 0:
            router_row["identity"]["matches"] = False
            router_row["identity"]["my_device_id"] = 99
    decoded["coverage"]["identity_mismatch"] = 1
    return decoded


CASES = {
    "line_1d.json": line_1d,
    "mesh_2d.json": mesh_2d,
    "torus_xy.json": lambda: mesh_2d(wrap=True),
    "two_mesh.json": two_mesh,
    "stalled_link.json": stalled_link,
    "coverage_holes.json": coverage_holes,
}


def write_fixtures(directory: Path | None = None) -> Path:
    output = directory or FIXTURE_DIR
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.json").write_text(json.dumps(INDEX, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, factory in CASES.items():
        (output / name).write_text(json.dumps(factory(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def main() -> int:
    path = write_fixtures()
    print(f"Wrote fixtures to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
