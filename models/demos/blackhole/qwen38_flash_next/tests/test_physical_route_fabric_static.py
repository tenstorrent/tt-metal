# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``derive_fabric_line_route``: a ring under the 1x4 LINE descriptor opens in the order the fabric embedded the line.

The fixtures are two real boxes' cluster descriptors and fabric mappings (``tools/qb_mesh_smoke.py`` results of
2026-09-18): the QuietBox (4x p150, where the fabric's order equals the ring walk from chip 0) and a QuietBox 2
(2x p300c, where it does not: the ring walk (0, 1, 2, 3) put the mesh pair 1-2 on the ring link the fabric line
does not carry and the first all_gather failed with "Could not find any forwarding direction").
"""

from __future__ import annotations

from models.demos.blackhole.qwen38_flash_next.tools.hardware_profiles import HARDWARE_PROFILES
from models.demos.blackhole.qwen38_flash_next.tools.physical_route import (
    PhysicalRouteError,
    derive_fabric_line_route,
    derive_ring_walk_route,
    parse_chip_unique_ids,
    parse_chips_with_mmio,
    route_device_nodes,
)


def _links(pairs):
    return [[{"chip": a, "chan": ca}, {"chip": b, "chan": cb}] for (a, ca), (b, cb) in pairs]


# QuietBox (tt-quietbox), 2026-09-18 20:31Z: chips 0-2, 0-3, 1-3, 1-2 (four channels each), chip -> node {0:1, 1:2, 2:3, 3:0};
# fabric (M0, D0..D3) -> physical 0, 2, 1, 3.
QUIETBOX_DESCRIPTOR = {
    "chips_with_mmio": [{0: 1}, {1: 2}, {2: 3}, {3: 0}],
    "chip_unique_ids": {1: 143238002116000, 3: 143238002115840, 2: 143238002115552, 0: 143238002114880},
    "ethernet_connections": _links(
        [((0, c), (2, c)) for c in (4, 5, 6, 7)]
        + [((0, c), (3, c)) for c in (8, 9, 10, 11)]
        + [((1, c), (3, c)) for c in (4, 5, 6, 7)]
        + [((1, c), (2, c)) for c in (8, 9, 10, 11)]
    ),
}
QUIETBOX_FABRIC = {0: 143238002114880, 1: 143238002115552, 2: 143238002116000, 3: 143238002115840}

# QuietBox 2 (2x p300c), 2026-09-18 20:44Z: ring 0-1 (on-card), 1-2 (cable), 2-3 (on-card), 3-0 (cable), two channels each,
# chip -> node identity; fabric (M0, D0..D3) -> physical 1, 0, 3, 2.
QUIETBOX_2_DESCRIPTOR = {
    "chips_with_mmio": [{0: 0}, {1: 1}, {2: 2}, {3: 3}],
    "chip_unique_ids": {2: 154095682330465, 3: 154095682330464, 0: 154095682323969, 1: 154095682323968},
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
QUIETBOX_2_FABRIC = {0: 154095682323968, 1: 154095682323969, 2: 154095682330464, 3: 154095682330465}


def _lookup(table):
    def fabric_node_unique_id(mesh_id, chip_id):
        assert mesh_id == 0
        return table[chip_id]

    return fabric_node_unique_id


def test_quietbox_fabric_order_is_derived_and_equals_this_fixture_ring_walk():
    chips = parse_chips_with_mmio(QUIETBOX_DESCRIPTOR)
    route = derive_fabric_line_route(QUIETBOX_DESCRIPTOR, chips, _lookup(QUIETBOX_FABRIC))
    assert route == (0, 2, 1, 3) == derive_ring_walk_route(QUIETBOX_DESCRIPTOR, chips)
    profile = HARDWARE_PROFILES["tt-quietbox"]
    assert (route, route_device_nodes(route, chips)) == ((0, 2, 1, 3), (1, 3, 2, 0))
    assert profile.route is None and profile.route_nodes is None  # derived on each host, never pinned to these IDs


def test_quietbox_2_fabric_order_differs_from_the_ring_walk():
    chips = parse_chips_with_mmio(QUIETBOX_2_DESCRIPTOR)
    route = derive_fabric_line_route(QUIETBOX_2_DESCRIPTOR, chips, _lookup(QUIETBOX_2_FABRIC))
    assert route == (1, 0, 3, 2)
    assert route_device_nodes(route, chips) == (1, 0, 3, 2)
    assert derive_ring_walk_route(QUIETBOX_2_DESCRIPTOR, chips) == (0, 1, 2, 3) != route
    assert HARDWARE_PROFILES["tt-quietbox-2"].route is None  # derived and recorded at start, not pinned


def test_parse_chip_unique_ids_is_exact(expect_error):
    chips = parse_chips_with_mmio(QUIETBOX_2_DESCRIPTOR)
    assert parse_chip_unique_ids(QUIETBOX_2_DESCRIPTOR, chips) == QUIETBOX_2_DESCRIPTOR["chip_unique_ids"]
    for broken in (
        {**QUIETBOX_2_DESCRIPTOR, "chip_unique_ids": None},
        {**QUIETBOX_2_DESCRIPTOR, "chip_unique_ids": {0: 1, 1: 2, 2: 3}},
        {**QUIETBOX_2_DESCRIPTOR, "chip_unique_ids": {0: 1, 1: 1, 2: 3, 3: 4}},
        {**QUIETBOX_2_DESCRIPTOR, "chip_unique_ids": {0: "1", 1: 2, 2: 3, 3: 4}},
        {**QUIETBOX_2_DESCRIPTOR, "chip_unique_ids": {0: 1.0, 1: 2, 2: 3, 3: 4}},
    ):
        with expect_error(PhysicalRouteError):
            parse_chip_unique_ids(broken, chips)


def test_fabric_order_refuses_unknown_ids_duplicates_and_non_paths(expect_error):
    chips = parse_chips_with_mmio(QUIETBOX_2_DESCRIPTOR)
    with expect_error(PhysicalRouteError, match="not one of the four local chips"):
        derive_fabric_line_route(QUIETBOX_2_DESCRIPTOR, chips, _lookup({**QUIETBOX_2_FABRIC, 2: 7}))
    with expect_error(PhysicalRouteError, match="not one of the four local chips"):
        derive_fabric_line_route(
            QUIETBOX_2_DESCRIPTOR, chips, _lookup({k: str(v) for k, v in QUIETBOX_2_FABRIC.items()})
        )
    with expect_error(PhysicalRouteError, match="four distinct local chips"):
        derive_fabric_line_route(QUIETBOX_2_DESCRIPTOR, chips, _lookup({**QUIETBOX_2_FABRIC, 3: QUIETBOX_2_FABRIC[0]}))
    # D0..D3 = chips 0, 2, 1, 3 is a bijection but not a path of the QuietBox 2 ring (0 and 2 are not linked).
    diagonal = {0: 154095682323969, 1: 154095682330465, 2: 154095682323968, 3: 154095682330464}
    with expect_error(PhysicalRouteError, match="not an ethernet path"):
        derive_fabric_line_route(QUIETBOX_2_DESCRIPTOR, chips, _lookup(diagonal))
    with expect_error(PhysicalRouteError, match="mesh_id"):
        derive_fabric_line_route(QUIETBOX_2_DESCRIPTOR, chips, _lookup(QUIETBOX_2_FABRIC), mesh_id="0")
    # A ring profile on a graph that is not a four-chip ring (the 3-0 cable missing) is refused before the fabric is
    # asked anything: the ring walk's check, the same message.
    broken_ring = {**QUIETBOX_2_DESCRIPTOR, "ethernet_connections": QUIETBOX_2_DESCRIPTOR["ethernet_connections"][2:]}
    asked = []
    with expect_error(PhysicalRouteError, match="not one four-chip ring"):
        derive_fabric_line_route(broken_ring, chips, lambda mesh_id, chip_id: asked.append((mesh_id, chip_id)))
    assert asked == []


def test_fabric_order_is_independent_of_the_open_order_and_deterministic():
    chips = parse_chips_with_mmio(QUIETBOX_2_DESCRIPTOR)
    calls = []

    def lookup(mesh_id, chip_id):
        calls.append((mesh_id, chip_id))
        return QUIETBOX_2_FABRIC[chip_id]

    first = derive_fabric_line_route(QUIETBOX_2_DESCRIPTOR, chips, lookup)
    second = derive_fabric_line_route(QUIETBOX_2_DESCRIPTOR, chips, lookup)
    assert first == second == (1, 0, 3, 2)
    assert calls == [(0, 0), (0, 1), (0, 2), (0, 3)] * 2  # ascending fabric chip id, nothing else consulted
