# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``derive_fabric_line_route``: a ring under the 1x4 LINE descriptor opens in the order the fabric embedded the line.

The fixtures are cluster descriptors and fabric mappings read on real hosts: a QuietBox (4x p150, where the fabric's
order equals the ring walk from chip 0) and a QuietBox 2 (2x p300c, where it does not). The route is read before the
fabric is enabled, so the opened mesh is checked to hold fabric chip k at coordinate k (``live_mapping``).
"""

from __future__ import annotations

import re
from dataclasses import replace
from types import SimpleNamespace

import pytest
import yaml

import ttnn
from models.demos.blackhole.qwen38_flash_next.tools import hardware_profiles
from models.demos.blackhole.qwen38_flash_next.tools.hardware_profiles import HARDWARE_PROFILES, HardwareProfileError
from models.demos.blackhole.qwen38_flash_next.tools.physical_route import (
    PhysicalRouteError,
    derive_fabric_line_route,
    derive_ring_walk_route,
    parse_chips_with_mmio,
    route_device_nodes,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_session import resolve_route


def _links(pairs):
    return [[{"chip": a, "chan": ca}, {"chip": b, "chan": cb}] for (a, ca), (b, cb) in pairs]


# QuietBox (4x p150): chips 0-2, 0-3, 1-3, 1-2 (four channels each), chip -> node {0:1, 1:2, 2:3, 3:0};
# fabric (M0, D0..D3) -> physical 0, 2, 1, 3.
QUIETBOX_DESCRIPTOR = {
    "chips_with_mmio": [{0: 1}, {1: 2}, {2: 3}, {3: 0}],
    "ethernet_connections": _links(
        [((0, c), (2, c)) for c in (4, 5, 6, 7)]
        + [((0, c), (3, c)) for c in (8, 9, 10, 11)]
        + [((1, c), (3, c)) for c in (4, 5, 6, 7)]
        + [((1, c), (2, c)) for c in (8, 9, 10, 11)]
    ),
}
QUIETBOX_FABRIC = {0: 0, 1: 2, 2: 1, 3: 3}

# QuietBox 2 (2x p300c): ring 0-1 (on-card), 1-2 (cable), 2-3 (on-card), 3-0 (cable), two channels each,
# chip -> node identity; fabric (M0, D0..D3) -> physical 1, 0, 3, 2.
QUIETBOX_2_DESCRIPTOR = {
    "chips_with_mmio": [{0: 0}, {1: 1}, {2: 2}, {3: 3}],
    "ethernet_connections": _links(
        [
            ((0, 2), (3, 5)),
            ((0, 4), (3, 4)),
            ((0, 8), (1, 3)),
            ((0, 9), (1, 2)),
            ((1, 4), (2, 4)),
            ((1, 5), (2, 2)),
            ((2, 8), (3, 3)),
            ((2, 9), (3, 2)),
        ]
    ),
}
QUIETBOX_2_FABRIC = {0: 1, 1: 0, 2: 3, 3: 2}


# (profile, descriptor, fabric order, route, route nodes, ring walk): the QuietBox's fabric order equals the ring walk
# from chip 0, the QuietBox 2's does not.
RINGS = [
    pytest.param(
        "tt-quietbox", QUIETBOX_DESCRIPTOR, QUIETBOX_FABRIC, (0, 2, 1, 3), (1, 3, 2, 0), (0, 2, 1, 3), id="tt-quietbox"
    ),
    pytest.param(
        "tt-quietbox-2",
        QUIETBOX_2_DESCRIPTOR,
        QUIETBOX_2_FABRIC,
        (1, 0, 3, 2),
        (1, 0, 3, 2),
        (0, 1, 2, 3),
        id="tt-quietbox-2",
    ),
]


@pytest.mark.parametrize("name, descriptor, fabric, route, nodes, walk", RINGS)
def test_the_fabric_order_is_derived_deterministically_and_compared_with_the_ring_walk(
    name, descriptor, fabric, route, nodes, walk
):
    chips = parse_chips_with_mmio(descriptor)
    calls = []

    def lookup(chip_id):
        calls.append(chip_id)
        return fabric[chip_id]

    first = derive_fabric_line_route(descriptor, chips, lookup)
    second = derive_fabric_line_route(descriptor, chips, lookup)
    assert first == second == route, (first, second)
    assert calls == [0, 1, 2, 3] * 2, calls  # ascending fabric chip id, nothing else consulted
    assert derive_ring_walk_route(descriptor, chips) == walk
    assert route_device_nodes(route, chips) == nodes, route_device_nodes(route, chips)
    # derived on each host and recorded at start, never pinned to these IDs
    profile = HARDWARE_PROFILES[name]
    assert profile.route is None and profile.route_nodes is None, (profile.route, profile.route_nodes)


def test_fabric_order_refuses_unknown_ids_duplicates_and_non_paths(expect_error):
    chips = parse_chips_with_mmio(QUIETBOX_2_DESCRIPTOR)
    with expect_error(PhysicalRouteError, match="not one of the four local chips"):
        derive_fabric_line_route(QUIETBOX_2_DESCRIPTOR, chips, {**QUIETBOX_2_FABRIC, 2: 7}.__getitem__)
    for wrong_type in (str, float, bool):
        wrong = {fabric_chip: wrong_type(chip) for fabric_chip, chip in QUIETBOX_2_FABRIC.items()}
        with expect_error(PhysicalRouteError, match="not one of the four local chips"):
            derive_fabric_line_route(QUIETBOX_2_DESCRIPTOR, chips, wrong.__getitem__)
    with expect_error(PhysicalRouteError, match="four distinct local chips"):
        derive_fabric_line_route(
            QUIETBOX_2_DESCRIPTOR, chips, {**QUIETBOX_2_FABRIC, 3: QUIETBOX_2_FABRIC[0]}.__getitem__
        )
    # D0..D3 = chips 0, 2, 1, 3 is a bijection but not a path of the QuietBox 2 ring (0 and 2 are not linked).
    with expect_error(PhysicalRouteError, match="not an ethernet path"):
        derive_fabric_line_route(QUIETBOX_2_DESCRIPTOR, chips, {0: 0, 1: 2, 2: 1, 3: 3}.__getitem__)
    # A ring profile on a graph that is not a four-chip ring (the 3-0 cable missing) is refused before the fabric is
    # asked anything: the ring walk's check, the same message.
    broken_ring = {**QUIETBOX_2_DESCRIPTOR, "ethernet_connections": QUIETBOX_2_DESCRIPTOR["ethernet_connections"][2:]}
    asked = []
    with expect_error(PhysicalRouteError, match="not one four-chip ring"):
        derive_fabric_line_route(broken_ring, chips, asked.append)
    assert asked == [], asked


def _system_mesh(monkeypatch, tmp_path, descriptor, fabric, shape=(1, 4)):
    """A stand-in system mesh of ``shape`` whose coordinate (0, k) holds ``fabric[k]``, and ``descriptor`` as the
    cluster descriptor.  Returns the coordinates asked and the number of system meshes built."""

    asked, built = [], []

    class SystemMesh:
        def __init__(self):
            built.append(shape)

        def shape(self):
            return shape

        def get_device_id(self, coordinate):
            asked.append(tuple(int(value) for value in coordinate))
            return fabric[asked[-1][1]]

    for name in ("shape", "get_device_id"):  # the built bindings the lookup calls
        assert callable(getattr(ttnn._ttnn.multi_device.SystemMeshDescriptor, name, None)), name
    path = tmp_path / "cluster_descriptor.yaml"
    path.write_text(yaml.safe_dump(descriptor))
    monkeypatch.setattr(ttnn.cluster, "serialize_cluster_descriptor", lambda: str(path))
    monkeypatch.setattr(ttnn._ttnn.multi_device, "SystemMeshDescriptor", SystemMesh)
    monkeypatch.delattr(ttnn.cluster, "get_chip_unique_id_from_fabric_node_id", raising=False)
    return asked, built


@pytest.mark.parametrize("name, descriptor, fabric, route, nodes, walk", RINGS)
def test_resolve_route_reads_the_ring_order_from_the_system_mesh_before_any_mesh_opens(
    monkeypatch, tmp_path, name, descriptor, fabric, route, nodes, walk
):
    # No call needs an open mesh: _system_mesh removes ttnn.cluster's mesh-scoped FabricNodeId -> ASIC lookup.
    asked, _ = _system_mesh(monkeypatch, tmp_path, descriptor, fabric)
    profile, derivation = resolve_route(HARDWARE_PROFILES[name])
    assert (profile.route, profile.route_nodes) == (route, nodes), (profile.route, profile.route_nodes)
    assert asked == [(0, 0), (0, 1), (0, 2), (0, 3)], asked
    assert derivation["route_derivation"] == "derive_fabric_line_route", derivation
    assert (derivation["ring_walk_route"], derivation["ring_walk_agrees"]) == (list(walk), walk == route), derivation


def test_resolve_route_refuses_a_broken_ring_and_a_system_mesh_that_is_not_the_1x4_line(
    monkeypatch, tmp_path, expect_error
):
    # The 3-0 cable out: refused from the cluster descriptor before the system mesh (the topology mapper) is built.
    broken_ring = {**QUIETBOX_2_DESCRIPTOR, "ethernet_connections": QUIETBOX_2_DESCRIPTOR["ethernet_connections"][2:]}
    asked, built = _system_mesh(monkeypatch, tmp_path, broken_ring, QUIETBOX_2_FABRIC)
    with expect_error(PhysicalRouteError, match="not one four-chip ring"):
        resolve_route(HARDWARE_PROFILES["tt-quietbox-2"])
    assert (asked, built) == ([], []), (asked, built)
    # No mesh graph descriptor (a server started without the launcher): ttnn auto-discovers the ring as 2x2.
    monkeypatch.delenv("TT_MESH_GRAPH_DESC_PATH", raising=False)
    asked, built = _system_mesh(monkeypatch, tmp_path, QUIETBOX_DESCRIPTOR, QUIETBOX_FABRIC, shape=(2, 2))
    with expect_error(HardwareProfileError, match=r"system mesh is \(2, 2\), not the 1x4 line .*DESC_PATH=None"):
        resolve_route(HARDWARE_PROFILES["tt-quietbox"])
    assert (asked, built) == ([], [(2, 2)]), (asked, built)


class _OpenMesh:
    """An open 1x4 mesh: coordinate (0, k) holds local chip ``route[k]`` and fabric node ``fabric[k]`` (mesh, chip)."""

    def __init__(self, route, fabric):
        self.route, self.fabric = route, fabric

    @staticmethod
    def _column(coordinate):
        row, column = (int(value) for value in coordinate)
        assert row == 0, row
        return column

    def get_device_id(self, coordinate):
        return self.route[self._column(coordinate)]

    def get_fabric_node_id(self, coordinate):
        mesh_id, chip_id = self.fabric[self._column(coordinate)]
        return SimpleNamespace(mesh_id=mesh_id, chip_id=chip_id)


def _sysfs(monkeypatch, tmp_path, nodes):
    for node in nodes:
        device = tmp_path / "pci" / f"0000:0{node}:00.0"
        device.mkdir(parents=True)
        (device / "numa_node").write_text("0\n")
        link = tmp_path / "sysfs" / f"tenstorrent!{node}" / "device"
        link.parent.mkdir(parents=True)
        link.symlink_to(device)
    monkeypatch.setattr(hardware_profiles, "SYSFS_ROOT", tmp_path / "sysfs")


IN_ORDER = [(0, 0), (0, 1), (0, 2), (0, 3)]


@pytest.mark.parametrize("name, descriptor, fabric, route, nodes, walk", RINGS)
def test_live_mapping_refuses_a_ring_mesh_whose_coordinate_k_does_not_hold_fabric_chip_k(
    monkeypatch, tmp_path, expect_error, name, descriptor, fabric, route, nodes, walk
):
    # The route is read before set_fabric_config(FABRIC_1D) rebuilds the control plane, which does not compare the two
    # mappings: a rebuild that moved a chip fails at the open, not at the first all_gather.
    chips = parse_chips_with_mmio(descriptor)
    profile = replace(HARDWARE_PROFILES[name], route=route, route_nodes=nodes)
    _sysfs(monkeypatch, tmp_path, chips.values())
    mapping = hardware_profiles.live_mapping(_OpenMesh(route, IN_ORDER), chips, profile)
    fabric = [(item["fabric_mesh_id"], item["fabric_chip_id"]) for item in mapping]
    assert fabric == IN_ORDER, fabric
    for moved in ([(0, 1), (0, 0), (0, 2), (0, 3)], [(1, 0), (1, 1), (1, 2), (1, 3)]):
        refusal = "not in the fabric's line order: .*" + re.escape(f"actual={moved}")
        with expect_error(HardwareProfileError, match=refusal):
            hardware_profiles.live_mapping(_OpenMesh(route, moved), chips, profile)


def test_live_mapping_leaves_a_line_profile_its_fabric_orientation(monkeypatch, tmp_path):
    # A line's fabric orientation follows the topology mapper's physical node order, not the chip IDs the route is
    # oriented by, and both orientations are a line: a mesh whose fabric runs the other way opens.
    chips = {0: 0, 1: 1, 2: 2, 3: 3}
    profile = replace(HARDWARE_PROFILES["p150-line"], route=(0, 1, 2, 3), route_nodes=(0, 1, 2, 3))
    _sysfs(monkeypatch, tmp_path, chips.values())
    mapping = hardware_profiles.live_mapping(_OpenMesh((0, 1, 2, 3), IN_ORDER[::-1]), chips, profile)
    fabric = [(item["fabric_mesh_id"], item["fabric_chip_id"]) for item in mapping]
    assert fabric == IN_ORDER[::-1], fabric


def test_the_ring_profiles_mesh_graph_descriptors_are_one_1x4_mesh_with_the_checked_mesh_id():
    rings = {name: profile for name, profile in HARDWARE_PROFILES.items() if profile.ethernet_graph == "ring"}
    assert sorted(rings) == ["tt-quietbox", "tt-quietbox-2", "tt-quietbox-2-instance-1"], sorted(rings)
    for name, profile in rings.items():
        text = hardware_profiles.mesh_graph_descriptor_path(profile).read_text()
        shape = re.findall(r"device_topology \{ dims: \[ (\d+), (\d+) \]", text)
        mesh_ids = [int(mesh_id) for mesh_id in re.findall(r"mesh_id: (\d+)", text)]
        assert (shape, mesh_ids) == ([("1", "4")], [hardware_profiles.RING_FABRIC_MESH_ID]), (name, shape, mesh_ids)
