# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
What the route rules need of a manifest, read once per snapshot, and the keys decode names chips by. The routing tables
and the ring buffer slots both walk routes with it.
"""

from __future__ import annotations

from dataclasses import dataclass

from tt_metal.fabric.debug.visualizer.common.manifest_lookup import ManifestError, lookup
from tt_metal.fabric.debug.visualizer.decode.routes import (
    Coord,
    Direction,
    Encodings,
    LineView1D,
    MeshView,
    OddKind,
    Walk2D,
    get_encodings_from_manifest,
    get_line_view_1d,
    get_mesh_view,
)


@dataclass(frozen=True)
class RouteContext:
    """A manifest's routing values, meshes and wiring, and its chips' keys, M<mesh>/C<chip>, and coordinates."""

    enc: Encodings
    is_2d: bool
    extension_words_1d: int | None  # a 1D run's route words past the first; None in a 2D run
    meshes: dict[str, MeshView]  # by mesh key, such as M0
    chip_keys: dict[str, dict[Coord, str]]  # each mesh's chip keys, by coordinate
    coords: dict[str, Coord]  # every chip's coordinate, by its key
    line: LineView1D | None  # a 1D run's wiring; None in a 2D run


def get_route_context(manifest: dict) -> RouteContext:
    """Reads what the route rules need. Raises ManifestError, as the rules' loaders do."""
    enc = get_encodings_from_manifest(manifest)
    is_2d = lookup(manifest, "fabric_context.is_2d_routing", "boolean")
    meshes, chip_keys, coords = {}, {}, {}
    for mesh_key in lookup(manifest, "meshes", "object"):
        meshes[mesh_key] = get_mesh_view(manifest, mesh_key)
        chip_keys[mesh_key] = {}
        chips_where = f"meshes.{mesh_key}.chips"
        for chip_key in lookup(manifest, chips_where, "object"):
            coord = tuple(lookup(manifest, f"{chips_where}.{chip_key}.mesh_coord", "array", of="integer"))
            key = f"{mesh_key}/{chip_key}"
            chip_keys[mesh_key][coord] = key
            coords[key] = coord
    return RouteContext(
        enc=enc,
        is_2d=is_2d,
        extension_words_1d=None if is_2d else lookup(manifest, "fabric_context.routing_1d_extension_words", "integer"),
        meshes=meshes,
        chip_keys=chip_keys,
        coords=coords,
        line=None if is_2d else get_line_view_1d(manifest),
    )


@dataclass(frozen=True)
class ChipStep2D:
    """One link a 2D route takes, between chips named by their keys."""

    src: str
    direction: Direction
    dst: str


@dataclass(frozen=True)
class ChipOdd2D:
    """An Odd2D at a chip named by its key."""

    chip: str
    kind: OddKind
    direction: Direction | None


@dataclass(frozen=True)
class ChipWalk2D:
    """A Walk2D with its chips named by their keys, M<mesh>/C<chip>."""

    delivers: list[str]
    steps: list[ChipStep2D]
    exits: list[str]
    odd: list[ChipOdd2D]


def name_walk_2d(walk: Walk2D, mesh_key: str, ctx: RouteContext) -> ChipWalk2D:
    """`walk`, a walk through mesh `mesh_key`, with its coordinates replaced by chip keys."""
    # One dict lookup per chip, since a walk of random bytes can take dozens of steps.
    keys = ctx.chip_keys[mesh_key]
    try:
        return ChipWalk2D(
            delivers=[keys[chip] for chip in walk.delivers],
            steps=[ChipStep2D(keys[step.src], step.direction, keys[step.dst]) for step in walk.steps],
            exits=[keys[chip] for chip in walk.exits],
            odd=[ChipOdd2D(keys[odd.chip], odd.kind, odd.direction) for odd in walk.odd],
        )
    except KeyError as error:
        raise ManifestError(f"meshes.{mesh_key} has no chip at {error.args[0]}") from None
