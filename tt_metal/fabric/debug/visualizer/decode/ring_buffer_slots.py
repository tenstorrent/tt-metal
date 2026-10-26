# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Decodes each ring-buffer slot's packet header, read from the slot's L1 bytes, and where that packet goes.
The router zeroes every slot's header when it starts and never clears one after, so a slot whose header is
all zero bytes is empty, and any other slot holds the last packet written there. Whether that packet is still
waiting, already sent or done isn't in the snapshot: the router keeps its slot positions in its own variables.
"""

from __future__ import annotations

from dataclasses import dataclass

from tt_metal.fabric.debug.visualizer.common.manifest_lookup import ManifestError, lookup
from tt_metal.fabric.debug.visualizer.common.router_paths import get_router
from tt_metal.fabric.debug.visualizer.decode.inputs import Input
from tt_metal.fabric.debug.visualizer.decode.route_context import ChipWalk2D, RouteContext, name_walk_2d
from tt_metal.fabric.debug.visualizer.decode.routers import DecodedRouter
from tt_metal.fabric.debug.visualizer.decode.routes import (
    OPPOSITE_2D,
    Direction,
    ScatterChunk,
    get_chips_1d,
    get_landing_route_map_2d,
    get_scatter_chunks,
    walk_1d,
    walk_2d,
)
from tt_metal.fabric.debug.visualizer.decode.routing_tables import (
    ACTION_VECTORS,
    ROUTE_TABLE_2D,
    RoutingTable,
)
from tt_metal.fabric.debug.visualizer.decode.type_parser import Member, Types, decode_struct

# The manifest names the slots read: the packet header's members, and the router's and its channels' fields.
COMMAND_FIELDS = "command_fields"
NOC_SEND_TYPE = "noc_send_type"
PAYLOAD_SIZE_BYTES = "payload_size_bytes"
ROUTING_FIELDS = "routing_fields"
VALUE = "value"
ROUTE_BUFFER = "route_buffer"
DST_START_MESH_ID = "dst_start_mesh_id"
DST_START_CHIP_ID = "dst_start_chip_id"
MCAST_PARAMS = "mcast_params"
SCATTER_WRITE = "unicast_scatter_write"
CHUNK_COUNT = "chunk_count"
CHUNK_SIZE = "chunk_size"
CHUNK_ENCODING = "chunk_encoding"
EXIT_NODE_TABLE = "exit_node_table"
MY_DIRECTION = "my_direction"
INTERMESH_INGRESS = "intermesh_ingress"
RING_BUFFER = "ring_buffer"
ACTIVE = "active"
CHANNEL_KINDS = ("senders", "receivers")


@dataclass(frozen=True)
class SlotContext:
    """What decoding slots needs of a manifest, read and checked once per snapshot."""

    route_context: RouteContext
    header: str  # the packet header's struct name
    header_size: int
    directions: dict[int, Direction]  # in a 2D run, the eth_chan_directions values the rules know, by value


@dataclass(frozen=True)
class Hop1D:
    hop: str  # the hop field's name
    chip: str | None  # the chip that reads it; None once the line can't be followed


@dataclass(frozen=True)
class SlotRoute1D:
    hops: list[Hop1D]  # hop 0 first


@dataclass(frozen=True)
class SlotRoute2D:
    start: str  # the chip whose router reads the map first
    entered_from: Direction | None  # the side the packet enters `start` from; None at a landing, read as a route's start
    walk: ChipWalk2D | None  # None when `start` is in another mesh, which rebuilds the map when the packet lands
    landing: bool  # whether the walk follows the map `start`'s router rebuilds at landing, rather than the slot's
    rebuilt: bool | None  # at a landing, whether the slot's map is already the rebuilt one; None elsewhere


@dataclass(frozen=True)
class RingBufferSlot:
    """One ring-buffer slot. A used slot's packet may be waiting, sent or done: the snapshot doesn't say which."""

    empty: bool  # the header is all zero bytes: the slot hasn't held a packet since the router started
    header: dict | None  # as the type parser decodes it; None when empty
    route: SlotRoute1D | SlotRoute2D | None  # None when empty, or when a rule refuses (see problems)
    scatter: list[ScatterChunk] | None  # a scatter write's chunks, as the router carries them out
    problems: list[str]  # each rule that refused the slot's bytes, and why


EMPTY_RING_BUFFER_SLOT = RingBufferSlot(True, None, None, None, [])


def _get_struct_members(types: Types, schema: str, where: str) -> dict[str, Member]:
    struct = types.structs.get(schema[len("struct:") :]) if schema.startswith("struct:") else None
    if struct is None:
        raise ManifestError(f"{where} is {schema!r}; the slot rules read a struct there")
    return {member.name: member for member in struct.members}


def _check_members(members: dict[str, Member], names: list[str], where: str):
    missing = [name for name in names if name not in members]
    if missing:
        raise ManifestError(f"{where} has no {', '.join(missing)}; its members are {sorted(members)}")


def _check_header(types: Types, header: str, route_context: RouteContext):
    """Checks that the header has every member the slot rules read, which a fabric change could rename."""
    where = f"{types.where}.{header}"
    members = _get_struct_members(types, f"struct:{header}", where)
    _check_members(members, [COMMAND_FIELDS, NOC_SEND_TYPE, PAYLOAD_SIZE_BYTES], where)
    if route_context.is_2d:
        _check_members(members, [ROUTE_BUFFER, DST_START_MESH_ID, DST_START_CHIP_ID, MCAST_PARAMS], where)
        map_bytes = members[ROUTE_BUFFER].shape.count
        for mesh_key, mesh in route_context.meshes.items():
            if sum(mesh.shape) > map_bytes:
                raise ManifestError(
                    f"{where}.{ROUTE_BUFFER} holds {map_bytes} bytes; {mesh_key}'s {mesh.shape[0]}x{mesh.shape[1]} "
                    f"map needs {sum(mesh.shape)}"
                )
    else:
        _check_members(members, [ROUTING_FIELDS], where)
        routing_where = f"{where}.{ROUTING_FIELDS}"
        routing_members = _get_struct_members(types, members[ROUTING_FIELDS].shape.schema, routing_where)
        _check_members(
            routing_members, [VALUE] + ([ROUTE_BUFFER] if route_context.extension_words_1d else []), routing_where
        )

    command_schema = members[COMMAND_FIELDS].shape.schema
    union = types.unions.get(command_schema[len("union:") :]) if command_schema.startswith("union:") else None
    if union is None:
        raise ManifestError(f"{where}.{COMMAND_FIELDS} is {command_schema!r}; the slot rules read a union there")
    scatter = [member for member in union.members.values() if member.name == SCATTER_WRITE]
    if not scatter:
        raise ManifestError(f"{where}.{COMMAND_FIELDS} has no {SCATTER_WRITE} member")
    scatter_where = f"{where}.{COMMAND_FIELDS}.{SCATTER_WRITE}"
    scatter_members = _get_struct_members(types, scatter[0].shape.schema, scatter_where)
    _check_members(scatter_members, [CHUNK_COUNT, CHUNK_SIZE, CHUNK_ENCODING], scatter_where)


def get_slot_context(manifest: dict, types: Types, route_context: RouteContext) -> SlotContext:
    """
    Reads the packet header's struct and checks that it has every member the slot rules read. Raises ManifestError
    naming the first one missing, so a renamed member fails even when every slot is empty.
    """
    schema = lookup(manifest, "fabric_context.packet_header.schema", "string")
    size = lookup(manifest, "fabric_context.packet_header.size", "integer")
    header = schema[len("struct:") :] if schema.startswith("struct:") else None
    if header is None or header not in types.structs:
        raise ManifestError(f"fabric_context.packet_header.schema is {schema!r}, which names no struct of the types")
    if types.structs[header].size != size:
        raise ManifestError(
            f"fabric_context.packet_header.size is {size}; {header} is {types.structs[header].size} bytes"
        )
    _check_header(types, header, route_context)

    directions = {}
    if route_context.is_2d:
        values = lookup(manifest, "enums.eth_chan_directions.values", "object", of="integer")
        known = {direction.value for direction in Direction}
        if not known <= set(values):
            raise ManifestError(f"enums.eth_chan_directions names {sorted(values)}; the 2D rules need {sorted(known)}")
        directions = {value: Direction(name) for name, value in values.items() if name in known}
    return SlotContext(route_context, header, size, directions)


@dataclass(frozen=True)
class _Start:
    """Where the routes of one channel's packets start."""

    receiver_1d: str | None = None  # 1D: the router whose receiver reads hop 0
    chip: str | None = None  # 2D: the chip whose router reads the map first
    entered_from: Direction | None = None  # 2D: the side the packet enters `chip` from
    walks: bool = True  # 2D: False when `chip` is in another mesh, which rebuilds the map at landing
    landing: bool = False  # 2D: whether `chip`'s router rebuilds the map from its own table first
    problem: str | None = None  # why no slot of the channel gets a route


@dataclass(frozen=True)
class _LandingTable:
    """What a landing rebuilds a map from: this chip's route table."""

    action_vectors: bytes
    exit_node_table: list[int]


def _get_direction(router: dict, where: str, ctx: SlotContext) -> Direction:
    value = lookup(router, f"fields.{MY_DIRECTION}.value", "integer", where=where)
    direction = ctx.directions.get(value)
    if direction is None:
        raise ManifestError(f"{where}.fields.{MY_DIRECTION} is {value}, a direction the 2D rules don't know")
    return direction


def _get_start(
    kind: str, channel: dict, channel_where: str, path: str, router: dict, router_where: str, ctx: SlotContext
) -> _Start:
    """The start rule for a channel of router `path`, whose manifest node is `router` at `router_where`."""
    route_context = ctx.route_context
    chip = path.rsplit("/", 1)[0]
    if not route_context.is_2d:
        # A packet in a sender slot crosses the link before any router reads hop 0; in a receiver slot, it's read here.
        return _Start(receiver_1d=route_context.line.peers.get(path) if kind == "senders" else path)

    direction = _get_direction(router, router_where, ctx)
    if kind == "receivers":
        if lookup(channel, f"fields.{INTERMESH_INGRESS}.value", "boolean", where=channel_where):
            return _Start(chip=chip, landing=True)
        return _Start(chip=chip, entered_from=direction)

    peer = lookup(router, "link.peer", ("string", "null"), where=router_where)
    if peer is None:
        return _Start(problem=f"{path}'s link has no peer, so its packets go nowhere")
    peer_chip = "/".join(peer.split("/")[:2])
    if peer_chip not in route_context.coords:
        return _Start(problem=f"{path}'s link peer {peer} is on a chip the manifest doesn't have")
    leaves_mesh = peer_chip.split("/")[0] != chip.split("/")[0]
    return _Start(chip=peer_chip, entered_from=OPPOSITE_2D[direction], walks=not leaves_mesh)


def _get_landing_table(table: RoutingTable | None, snapshot: Input) -> _LandingTable | None:
    if table is None:
        return None
    route_table = table.fields.get(ROUTE_TABLE_2D)
    if route_table is None or ACTION_VECTORS not in route_table or EXIT_NODE_TABLE not in table.fields:
        raise ManifestError(
            f"the routing table has no {ROUTE_TABLE_2D}.{ACTION_VECTORS} or {EXIT_NODE_TABLE}, which a landing reads"
        )
    at = route_table[ACTION_VECTORS]["bytes"]
    return _LandingTable(snapshot.bin[at["offset"] : at["offset"] + at["size"]], table.fields[EXIT_NODE_TABLE])


def _get_route_1d(header: dict, start: _Start, ctx: SlotContext) -> SlotRoute1D:
    """Raises ValueError when the router would assert on the route."""
    route_context = ctx.route_context
    routing = header[ROUTING_FIELDS]
    words = [routing[VALUE]] + (list(routing[ROUTE_BUFFER]) if route_context.extension_words_1d else [])
    hops = walk_1d(words, route_context.enc.one_d)
    chips = get_chips_1d(start.receiver_1d, len(hops), route_context.line)
    return SlotRoute1D([Hop1D(hop.name, chip) for hop, chip in zip(hops, chips)])


def _get_route_2d(header: dict, start: _Start, landing_table: _LandingTable | None, ctx: SlotContext) -> SlotRoute2D:
    """Raises ValueError, with the reason, when a landing can't rebuild the map."""
    route_context = ctx.route_context
    if not start.walks:
        return SlotRoute2D(start.chip, start.entered_from, None, False, None)
    mesh_key = start.chip.split("/")[0]
    mesh = route_context.meshes[mesh_key]
    route_map = bytes(header[ROUTE_BUFFER])
    rebuilt = None
    if start.landing:
        if landing_table is None:
            raise ValueError(f"{start.chip}'s routing table wasn't captured, so its landing can't be rebuilt")
        landing_map = get_landing_route_map_2d(
            header[DST_START_MESH_ID],
            header[DST_START_CHIP_ID],
            header[MCAST_PARAMS],
            mesh,
            landing_table.action_vectors,
            landing_table.exit_node_table,
            route_context.enc.two_d,
        )
        if landing_map is None:
            raise ValueError(
                "a multicast for this mesh landed here; intermesh multicast isn't supported, and decode doesn't "
                "rebuild its map"
            )
        rebuilt = route_map[: len(landing_map)] == landing_map
        route_map = landing_map
    walk = walk_2d(
        route_map,
        route_context.coords[start.chip],
        mesh,
        route_context.enc.two_d,
        header[DST_START_MESH_ID],
        start.entered_from,
    )
    return SlotRoute2D(
        start.chip, start.entered_from, name_walk_2d(walk, mesh_key, route_context), start.landing, rebuilt
    )


def _get_scatter(header: dict, ctx: SlotContext) -> list[ScatterChunk]:
    """Raises ValueError for a scatter the router can't carry out."""
    command = header[COMMAND_FIELDS]["value"]
    return get_scatter_chunks(
        command[CHUNK_COUNT],
        command[CHUNK_SIZE],
        [entry["name"] for entry in command[CHUNK_ENCODING]],
        header[PAYLOAD_SIZE_BYTES],
        ctx.route_context.enc.scatter,
    )


def _decode_slot(
    offset: int, start: _Start, landing_table: _LandingTable | None, snapshot: Input, types: Types, ctx: SlotContext
) -> RingBufferSlot:
    if not any(snapshot.bin[offset : offset + ctx.header_size]):
        return EMPTY_RING_BUFFER_SLOT
    header = decode_struct(ctx.header, snapshot.bin, offset, types)
    problems = []
    send_type = header[NOC_SEND_TYPE]
    if send_type["name"] is None:
        problems.append(f"{NOC_SEND_TYPE} is {send_type['raw']}, which names no send type")

    scatter = None
    if header[COMMAND_FIELDS].get("member") == SCATTER_WRITE:
        try:
            scatter = _get_scatter(header, ctx)
        except ValueError as error:
            problems.append(f"the router can't carry out this scatter write: {error}")

    route = None
    if start.problem is not None:
        problems.append(start.problem)
    else:
        try:
            if ctx.route_context.is_2d:
                route = _get_route_2d(header, start, landing_table, ctx)
            else:
                route = _get_route_1d(header, start, ctx)
        except ValueError as error:
            problems.append(f"no route: {error}")
    return RingBufferSlot(False, header, route, scatter, problems)


def get_ring_buffer_slots(
    snapshot: Input, types: Types, ctx: SlotContext, router: DecodedRouter, table: RoutingTable | None
) -> dict[str, list[RingBufferSlot]]:
    """
    The slots of each of `router`'s active channels, by the channel's path in the router, such as
    channels/senders/vc0/ch0. `table` is the router's routing table, which a receiver at an intermesh ingress rebuilds
    a landing packet's map from; None when capture didn't read it. A channel no ERISC runs has no slots: its ring holds
    whatever L1 held. Neither has one whose ring capture didn't read. A rule that refuses a slot's bytes goes in its
    problems and is never raised. Raises ManifestError for a manifest value the rules can't use.
    """
    node, where = get_router(snapshot.manifest, router.path)
    landing_table = None  # read only for a router with an intermesh ingress, the one channel that lands packets
    slots = {}
    for kind in CHANNEL_KINDS:
        for vc, channels in lookup(node, f"channels.{kind}", "object", of="object", where=where).items():
            for key, channel in channels.items():
                channel_path = f"channels/{kind}/{vc}/{key}"
                channel_where = f"{where}.channels.{kind}.{vc}.{key}"
                ring = router.regions.get(f"{channel_path}/{RING_BUFFER}")
                if lookup(channel, "status", "string", where=channel_where) != ACTIVE or ring is None:
                    continue
                slot_size = lookup(channel, f"{RING_BUFFER}.size_per_element", "integer", where=channel_where)
                if slot_size < ctx.header_size:
                    raise ManifestError(
                        f"{channel_where}.{RING_BUFFER}'s slots are {slot_size} bytes, smaller than the "
                        f"{ctx.header_size}-byte packet header"
                    )
                start = _get_start(kind, channel, channel_where, router.path, node, where, ctx)
                if start.landing and landing_table is None:
                    landing_table = _get_landing_table(table, snapshot)
                slots[channel_path] = [
                    _decode_slot(entry["bytes"]["offset"], start, landing_table, snapshot, types, ctx) for entry in ring
                ]
    return slots
