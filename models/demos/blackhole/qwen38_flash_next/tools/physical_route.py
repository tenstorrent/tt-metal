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
    """Return the four-chip ring walked from the lowest chip ID, taking the lowest unvisited neighbour."""

    adjacency = _ethernet_adjacency(descriptor, chips_with_mmio)
    if sorted(len(neighbors) for neighbors in adjacency.values()) != [2, 2, 2, 2]:
        raise PhysicalRouteError(f"local ethernet graph is not one four-chip ring: {adjacency}")
    route = [min(adjacency)]
    while len(route) < 4:
        route.append(min(adjacency[route[-1]] - set(route)))
    return tuple(route)


def derive_fabric_line_route(
    descriptor: Mapping[str, Any],
    chips_with_mmio: Mapping[int, int],
    fabric_node_chip_id: Callable[[int], int],
) -> tuple[int, int, int, int]:
    """Return the four chips in the order the fabric embedded the 1x4 LINE mesh graph descriptor: the local chip of
    fabric chip 0 of the host's mesh, then fabric chips 1, 2, 3.

    The fabric picks which three of the ring's four links the line uses, and the Linear collectives need the mesh
    opened in exactly this order: any other path puts a mesh neighbour pair on a link the line does not carry and the
    first all_gather fails.  The result is checked to be a bijection over the four local chips and an ethernet path.
    """

    adjacency = _ethernet_adjacency(descriptor, chips_with_mmio)
    if sorted(len(neighbors) for neighbors in adjacency.values()) != [2, 2, 2, 2]:
        # the ring profile's promise; refused before the fabric is asked anything (the same check as the ring walk)
        raise PhysicalRouteError(f"local ethernet graph is not one four-chip ring: {adjacency}")
    route: list[int] = []
    for fabric_chip_id in range(4):
        chip_id = fabric_node_chip_id(fabric_chip_id)
        if type(chip_id) is not int or chip_id not in chips_with_mmio:
            raise PhysicalRouteError(
                f"fabric chip {fabric_chip_id} resolves to chip {chip_id!r}, "
                f"not one of the four local chips {sorted(chips_with_mmio)}"
            )
        route.append(chip_id)
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
