# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Decodes the routing tables a snapshot holds. Each router holds its chip's routing table in L1, in its routing_table
area; this takes those bytes and gives the decoded table: its fields, and in a 2D run, the route it gives toward every
chip of its mesh and its multicast trees, or in a 1D run, its path table. The host writes one table to every core of a
chip, so a chip's routers hold the same bytes, apart from state_manager, which is each router's own. So each unique
table is decoded and stored once. Checking the tables against each other, or against what the host meant to write, is
left to whoever reads them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from tt_metal.fabric.debug.visualizer.common.manifest_lookup import ManifestError, lookup
from tt_metal.fabric.debug.visualizer.decode.inputs import Input
from tt_metal.fabric.debug.visualizer.decode.route_context import ChipWalk2D, RouteContext, name_walk_2d
from tt_metal.fabric.debug.visualizer.decode.routers import DecodedRouter
from tt_metal.fabric.debug.visualizer.decode.routes import (
    WORD_BYTES,
    McastTrees2D,
    get_mcast_tree_edges_2d,
    get_path_table_entry_1d,
    get_route_map_from_table_2d,
    walk_1d,
    walk_2d,
)
from tt_metal.fabric.debug.visualizer.decode.type_parser import Member, Types, get_shape

# The manifest names the routing tables read.
ROUTING_TABLE = "routing_table"
STATE_MANAGER = "state_manager"
ROUTE_TABLE_2D = "route_table_2d"
ACTION_VECTORS = "action_vectors"
MCAST_TREES = "mcast_trees"
PATH_TABLE_1D = "routing_path_table_1d"
PATHS = "paths"


@dataclass(frozen=True)
class PathEntry1D:
    """A nonzero entry of a 1D path table: the route words a worker gives a packet for the chip `hops_away` along."""

    hops_away: int
    hops: list[str] | None  # each hop's name, hop 0 first; None when the walk refuses
    problem: str | None  # why the walk refuses: the router would assert on the route


@dataclass(frozen=True)
class PathTable1D:
    entries: list[PathEntry1D]  # the nonzero entries, nearest first
    zero_entries: int  # the all-zero entries, which no route uses


@dataclass(frozen=True)
class RoutingTable:
    """One unique routing table, decoded once: the bytes some of a chip's routers hold, apart from state_manager."""

    chip: str  # M<mesh>/C<chip>
    routers: list[str]  # every router holding it, in snapshot order; the first one's path is its key
    raw: dict  # the first router's area in the .bin, as a bytes value: {"bytes": {"offset", "size"}}
    fields: dict  # the area as the type parser decodes it, without state_manager
    route_table_2d: dict[str, ChipWalk2D] | None  # in a 2D run, its route toward each chip of its mesh, by chip key
    mcast_trees: McastTrees2D | None  # in a 2D run
    path_table_1d: PathTable1D | None  # in a 1D run


@dataclass(frozen=True)
class RouterTableRef:
    """A router's reference to the RoutingTable it holds, and what's its own in its area: its state and its bytes."""

    table_key: str  # the key of its RoutingTable in the store's copies
    state_manager: dict  # its own, as the type parser decodes it
    raw: dict  # its own area in the .bin, as a bytes value


@dataclass(frozen=True)
class RoutingTableStore:
    copies: dict[str, RoutingTable]  # by key
    routers: dict[str, RouterTableRef]  # by router path; a router whose area capture didn't read has none


def _get_state_manager(manifest: dict, types: Types) -> Member:
    """The area's state_manager member, whose bytes are left out when copies are compared."""
    where = f"archs.{lookup(manifest, 'run.arch', 'string')}.areas.{ROUTING_TABLE}"
    shape = get_shape(lookup(manifest, where, "object"), types, where)
    struct = types.structs.get(shape.schema[len("struct:") :]) if shape.schema.startswith("struct:") else None
    members = {} if struct is None else {member.name: member for member in struct.members}
    if STATE_MANAGER not in members:
        raise ManifestError(f"{where}.schema is {shape.schema!r}, which has no {STATE_MANAGER} member")
    return members[STATE_MANAGER]


def _get_member(struct: dict, name: str, where: str):
    """A member of a struct as the type parser decodes it."""
    if name not in struct:
        raise ManifestError(f"{where} has no {name} member; its members are {sorted(struct)}")
    return struct[name]


def _get_bytes(snapshot: Input, value: dict) -> bytes:
    """The bytes a decoded bytes value points at in the .bin."""
    at = value["bytes"]
    return snapshot.bin[at["offset"] : at["offset"] + at["size"]]


def _get_routes_2d(
    fields: dict, chip: str, snapshot: Input, ctx: RouteContext
) -> tuple[dict[str, ChipWalk2D], McastTrees2D]:
    """Each route the table gives from its chip, toward every chip of the mesh, and its multicast trees."""
    where = f"{ROUTING_TABLE}.{ROUTE_TABLE_2D}"
    table = _get_member(fields, ROUTE_TABLE_2D, ROUTING_TABLE)
    action_vectors = _get_bytes(snapshot, _get_member(table, ACTION_VECTORS, where))
    mcast_trees = _get_bytes(snapshot, _get_member(table, MCAST_TREES, where))
    mesh_key = chip.split("/")[0]
    mesh = ctx.meshes[mesh_key]
    y_size, x_size = mesh.shape
    try:
        routes = {}
        for dst, dst_key in ctx.chip_keys[mesh_key].items():
            route_map = get_route_map_from_table_2d(action_vectors, y_size, x_size, dst, ctx.enc.two_d)
            walk = walk_2d(route_map, ctx.coords[chip], mesh, ctx.enc.two_d, mesh.mesh_id)
            routes[dst_key] = name_walk_2d(walk, mesh_key, ctx)
        trees = get_mcast_tree_edges_2d(mcast_trees, y_size, x_size, ctx.enc.two_d)
    except ValueError as error:
        raise ManifestError(f"{where} doesn't hold {mesh_key}'s {y_size}x{x_size} mesh: {error}") from None
    return routes, trees


def _get_path_table_1d(fields: dict, snapshot: Input, ctx: RouteContext) -> PathTable1D:
    """Every entry of the table, walked. An entry the walk refuses gets the reason, rather than raising."""
    table = _get_member(fields, PATH_TABLE_1D, ROUTING_TABLE)
    paths = _get_bytes(snapshot, _get_member(table, PATHS, f"{ROUTING_TABLE}.{PATH_TABLE_1D}"))
    entry_bytes = (1 + ctx.extension_words_1d) * WORD_BYTES
    entries, zero_entries = [], 0
    for hops_away in range(len(paths) // entry_bytes):
        words = get_path_table_entry_1d(paths, hops_away, ctx.extension_words_1d)
        if not any(words):
            zero_entries += 1
            continue
        try:
            hops, problem = [hop.name for hop in walk_1d(words, ctx.enc.one_d)], None
        except ValueError as error:
            hops, problem = None, str(error)
        entries.append(PathEntry1D(hops_away, hops, problem))
    return PathTable1D(entries, zero_entries)


def get_routing_table_store(
    snapshot: Input, types: Types, decoded_routers: Mapping[str, DecodedRouter], ctx: RouteContext
) -> RoutingTableStore:
    """
    The routing tables `snapshot` holds, decoded from each router's routing_table area. `decoded_routers` holds every
    router of the snapshot, in its order. Routers of one chip whose areas match byte for byte, apart from
    state_manager, share a copy; routers of different chips never do. Raises ManifestError when the area lacks a member
    the tables read, or doesn't fit the mesh.
    """
    # Find the routers whose areas hold routing tables
    routers_captured = [router for router in decoded_routers.values() if ROUTING_TABLE in router.areas]
    if not routers_captured:
        return RoutingTableStore({}, {})
    state_manager = _get_state_manager(snapshot.manifest, types)
    skip_start, skip_end = state_manager.offset, state_manager.offset + state_manager.shape.size

    # Collect every router's routing info
    copy_keys, holders, routers = {}, {}, {}
    for router in routers_captured:
        area = snapshot.json["routers"][router.path]["areas"][ROUTING_TABLE]
        raw = {"bytes": {"offset": area["offset"], "size": area["size"]}}
        data = _get_bytes(snapshot, raw)
        chip = router.path.rsplit("/", 1)[0]
        # Routers share a copy when they're on one chip and their bytes match, apart from state_manager
        compared = (chip, data[:skip_start] + data[skip_end:])
        key = copy_keys.setdefault(compared, router.path)
        holders.setdefault(key, []).append(router.path)
        state = _get_member(router.areas[ROUTING_TABLE], STATE_MANAGER, ROUTING_TABLE)
        routers[router.path] = RouterTableRef(key, state, raw)

    # Reduce the dict of every router's routing info to only storing unique copies
    copies = {}
    for key, paths in holders.items():
        chip = key.rsplit("/", 1)[0]
        area_fields = decoded_routers[key].areas[ROUTING_TABLE]
        fields = {name: value for name, value in area_fields.items() if name != STATE_MANAGER}
        routes = trees = path_table = None
        if ctx.is_2d:
            routes, trees = _get_routes_2d(fields, chip, snapshot, ctx)
        else:
            path_table = _get_path_table_1d(fields, snapshot, ctx)
        copies[key] = RoutingTable(chip, paths, routers[key].raw, fields, routes, trees, path_table)
    return RoutingTableStore(copies, routers)
