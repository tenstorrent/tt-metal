# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Pure validation for an exact four-device UMD Ethernet line route.

``chips_with_mmio`` keys are process-local UMD/TTNN chip IDs and values are
KMD PCIe device-node numbers.  Neither is a fabric ASIC unique ID or a stable
KMD board ID; callers must keep all four identity domains separate.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any


class PhysicalRouteError(ValueError):
    """The descriptor cannot prove one exact four-device physical line."""


def _require_exact_int(value: Any, *, label: str) -> int:
    if type(value) is not int:
        raise PhysicalRouteError(f"{label} must be an exact integer, got {value!r}")
    return value


def parse_chips_with_mmio(
    descriptor: Mapping[str, Any],
    *,
    expected_device_nodes: tuple[int, int, int, int] | set[int] | None = None,
) -> dict[int, int]:
    """Return process-local logical-chip ID -> KMD PCIe device-node ID."""

    if type(descriptor) is not dict:
        raise PhysicalRouteError("cluster descriptor must be an exact object")
    raw = descriptor.get("chips_with_mmio")
    if type(raw) is not list:
        raise PhysicalRouteError("cluster descriptor chips_with_mmio must be a list")
    chips_with_mmio: dict[int, int] = {}
    for item in raw:
        if type(item) is not dict or len(item) != 1:
            raise PhysicalRouteError(f"malformed chips_with_mmio entry: {item!r}")
        logical_chip_id, device_node = next(iter(item.items()))
        logical_chip_id = _require_exact_int(logical_chip_id, label="chips_with_mmio logical chip ID")
        device_node = _require_exact_int(device_node, label="chips_with_mmio device node")
        if logical_chip_id in chips_with_mmio:
            raise PhysicalRouteError(f"duplicate logical chip ID {logical_chip_id}")
        chips_with_mmio[logical_chip_id] = device_node

    if len(chips_with_mmio) != 4 or len(set(chips_with_mmio.values())) != 4:
        raise PhysicalRouteError(f"expected four distinct local chips and device nodes: {chips_with_mmio}")
    if expected_device_nodes is not None:
        if (
            type(expected_device_nodes) not in {tuple, set}
            or len(expected_device_nodes) != 4
            or any(type(value) is not int for value in expected_device_nodes)
        ):
            raise PhysicalRouteError("expected leased device nodes must be four exact integers")
        if set(chips_with_mmio.values()) != set(expected_device_nodes):
            raise PhysicalRouteError(
                "cluster descriptor PCIe device nodes differ from the leased partition: "
                f"{chips_with_mmio} != {set(expected_device_nodes)}"
            )
    return chips_with_mmio


def _ethernet_adjacency(descriptor: Mapping[str, Any], chips_with_mmio: Mapping[int, int]) -> dict[int, set[int]]:
    """Validated local-chip adjacency of the descriptor's ethernet_connections."""

    if type(descriptor) is not dict or type(chips_with_mmio) is not dict:
        raise PhysicalRouteError("route inputs must be exact objects")
    if any(type(chip) is not int or type(node) is not int for chip, node in chips_with_mmio.items()):
        raise PhysicalRouteError("route chip and device-node identities must be exact integers")
    chip_ids = set(chips_with_mmio)
    if len(chip_ids) != 4:
        raise PhysicalRouteError(f"route requires exactly four logical chip IDs: {sorted(chip_ids)}")
    adjacency = {chip_id: set() for chip_id in chip_ids}
    raw_links = descriptor.get("ethernet_connections")
    if type(raw_links) is not list:
        raise PhysicalRouteError("cluster descriptor ethernet_connections must be a list")
    for link in raw_links:
        if type(link) is not list or len(link) != 2:
            raise PhysicalRouteError(f"malformed ethernet connection: {link!r}")
        endpoints = []
        for endpoint in link:
            if type(endpoint) is not dict or "chip" not in endpoint:
                raise PhysicalRouteError(f"malformed ethernet endpoint: {endpoint!r}")
            endpoints.append(_require_exact_int(endpoint["chip"], label="ethernet endpoint chip ID"))
        left, right = endpoints
        if left not in chip_ids or right not in chip_ids or left == right:
            raise PhysicalRouteError(f"ethernet connection escapes the four local chips: {link!r}")
        adjacency[left].add(right)
        adjacency[right].add(left)
    return adjacency


def derive_canonical_line_route(
    descriptor: Mapping[str, Any],
    chips_with_mmio: Mapping[int, int],
) -> tuple[int, int, int, int]:
    """Return the unique four-chip line, oriented from the lowest endpoint ID."""

    adjacency = _ethernet_adjacency(descriptor, chips_with_mmio)
    chip_ids = set(chips_with_mmio)
    endpoints = sorted(chip_id for chip_id, neighbors in adjacency.items() if len(neighbors) == 1)
    degrees = sorted(len(neighbors) for neighbors in adjacency.values())
    if len(endpoints) != 2 or degrees != [1, 1, 2, 2]:
        raise PhysicalRouteError(f"local ethernet graph is not one four-chip line: {adjacency}")

    route: list[int] = []
    previous: int | None = None
    current = endpoints[0]
    while True:
        route.append(current)
        next_hops = adjacency[current] - ({previous} if previous is not None else set())
        if not next_hops:
            break
        if len(next_hops) != 1:
            raise PhysicalRouteError(f"ambiguous line route at chip {current}: {adjacency}")
        previous, current = current, next(iter(next_hops))
    if len(route) != 4 or set(route) != chip_ids:
        raise PhysicalRouteError(f"line route does not cover all local chips: {route}, graph={adjacency}")
    return tuple(route)


def derive_ring_walk_route(
    descriptor: Mapping[str, Any],
    chips_with_mmio: Mapping[int, int],
) -> tuple[int, int, int, int]:
    """Return the four-chip ring walked from the lowest chip ID, taking the lowest unvisited neighbour.

    A 1x4 LINE mesh graph descriptor over a ring uses three of its four links;
    this walk is the order that descriptor opened on the QuietBox (0, 2, 1, 3).
    """

    adjacency = _ethernet_adjacency(descriptor, chips_with_mmio)
    if sorted(len(neighbors) for neighbors in adjacency.values()) != [2, 2, 2, 2]:
        raise PhysicalRouteError(f"local ethernet graph is not one four-chip ring: {adjacency}")
    route = [min(adjacency)]
    while len(route) < 4:
        route.append(min(adjacency[route[-1]] - set(route)))
    return tuple(route)


def parse_chip_unique_ids(descriptor: Mapping[str, Any], chips_with_mmio: Mapping[int, int]) -> dict[int, int]:
    """Return process-local logical-chip ID -> the chip's 64-bit ASIC unique ID (``chip_unique_ids``), exactly the
    four local chips, distinct."""

    if type(descriptor) is not dict or type(chips_with_mmio) is not dict:
        raise PhysicalRouteError("route inputs must be exact objects")
    raw = descriptor.get("chip_unique_ids")
    if type(raw) is not dict:
        raise PhysicalRouteError("cluster descriptor chip_unique_ids must be a mapping")
    unique_ids: dict[int, int] = {}
    for chip_id, unique_id in raw.items():
        chip_id = _require_exact_int(chip_id, label="chip_unique_ids chip ID")
        unique_ids[chip_id] = _require_exact_int(unique_id, label="chip_unique_ids ASIC unique ID")
    if set(unique_ids) != set(chips_with_mmio) or len(set(unique_ids.values())) != len(unique_ids):
        raise PhysicalRouteError(
            f"chip_unique_ids must name the four local chips once each: {unique_ids}, chips={sorted(chips_with_mmio)}"
        )
    return unique_ids


def derive_fabric_line_route(
    descriptor: Mapping[str, Any],
    chips_with_mmio: Mapping[int, int],
    fabric_node_unique_id: Callable[[int, int], int],
    *,
    mesh_id: int = 0,
) -> tuple[int, int, int, int]:
    """Return the four chips in the order the fabric embedded the 1x4 LINE mesh graph descriptor: the physical chip
    of fabric node (mesh_id, 0), then (mesh_id, 1), (mesh_id, 2), (mesh_id, 3).

    A 1x4 LINE descriptor over a four-chip ring uses three of the four links, and the fabric's topology solver picks
    which three: the QuietBox (4x p150) came out as chips (0, 2, 1, 3), a QuietBox 2 (2x p300c) as (1, 0, 3, 2)
    (2026-09-18).  The collectives are Linear along the mesh row, so the mesh must be opened in exactly this order:
    any other ethernet path through the ring puts a mesh neighbour pair on a link the fabric line does not carry
    and the first all_gather fails ("Could not find any forwarding direction").  ``fabric_node_unique_id(mesh_id,
    chip_id)`` is ``ttnn.cluster.get_chip_unique_id_from_fabric_node_id`` (no mesh needs to be open; the control plane
    initializes from the descriptor ``TT_MESH_GRAPH_DESC_PATH`` names); the ASIC unique IDs are mapped back to the
    descriptor's local chip IDs through ``chip_unique_ids``.  The result is checked to be a bijection over the four
    local chips and an ethernet path (consecutive chips linked).  Deterministic: the fabric mapping is a function of
    the cluster descriptor and the mesh graph descriptor, not of any open order.
    """

    adjacency = _ethernet_adjacency(descriptor, chips_with_mmio)
    if sorted(len(neighbors) for neighbors in adjacency.values()) != [2, 2, 2, 2]:
        # the ring profile's promise; refused before the fabric is asked anything (the same check as the ring walk)
        raise PhysicalRouteError(f"local ethernet graph is not one four-chip ring: {adjacency}")
    unique_ids = parse_chip_unique_ids(descriptor, chips_with_mmio)
    chip_of_unique = {unique_id: chip_id for chip_id, unique_id in unique_ids.items()}
    if type(mesh_id) is not int:
        raise PhysicalRouteError("mesh_id must be an exact integer")
    route: list[int] = []
    for fabric_chip_id in range(4):
        unique_id = fabric_node_unique_id(mesh_id, fabric_chip_id)
        if type(unique_id) is not int or unique_id not in chip_of_unique:
            raise PhysicalRouteError(
                f"fabric node (mesh {mesh_id}, chip {fabric_chip_id}) resolves to ASIC unique ID {unique_id!r}, "
                f"not one of the four local chips {unique_ids}"
            )
        route.append(chip_of_unique[unique_id])
    if len(set(route)) != 4:
        raise PhysicalRouteError(f"fabric nodes do not map to four distinct local chips: {route}")
    for left, right in zip(route, route[1:]):
        if right not in adjacency[left]:
            raise PhysicalRouteError(
                f"fabric line order {tuple(route)} is not an ethernet path of the local chips: {adjacency}"
            )
    return tuple(route)


def route_device_nodes(
    route: Sequence[int],
    chips_with_mmio: Mapping[int, int],
) -> tuple[int, int, int, int]:
    """Translate a validated logical-chip route into KMD device-node order."""

    if type(route) not in {tuple, list} or any(type(value) is not int for value in route):
        raise PhysicalRouteError("route must be an exact integer sequence")
    if type(chips_with_mmio) is not dict or any(
        type(chip) is not int or type(node) is not int for chip, node in chips_with_mmio.items()
    ):
        raise PhysicalRouteError("route mapping identities must be exact integers")
    if len(route) != 4 or len(set(route)) != 4 or set(route) != set(chips_with_mmio):
        raise PhysicalRouteError(f"route and chips_with_mmio name different chips: {route}, {chips_with_mmio}")
    return tuple(chips_with_mmio[chip_id] for chip_id in route)
