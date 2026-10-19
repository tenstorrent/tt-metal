# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Reads fabric routes from raw bytes, using only the values the fabric manifest carries.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

# Route words are uint32, little-endian as on the device. The loader checks that the manifest's 1D hops fill one.
WORD_BYTES = 4

# The 1D hop names the rules know, each mapped to (writes, forwards).
HOP_ACTIONS = {
    "noop": (False, False),
    "write_only": (True, False),
    "forward_only": (False, True),
    "write_and_forward": (True, True),
}


class Direction(str, Enum):
    """A 2D direction the rules know, by its name in routing_encodings.2d.action_bits, in action-bit order."""

    EAST = "east"
    WEST = "west"
    NORTH = "north"
    SOUTH = "south"
    Z = "z"


# North is decreasing y and west decreasing x, so a step is a (dy, dx); a Z step instead follows the chip's Z link.
STEPS_2D = {Direction.EAST: (0, 1), Direction.WEST: (0, -1), Direction.NORTH: (-1, 0), Direction.SOUTH: (1, 0)}
OPPOSITE_2D = {
    Direction.EAST: Direction.WEST,
    Direction.WEST: Direction.EAST,
    Direction.NORTH: Direction.SOUTH,
    Direction.SOUTH: Direction.NORTH,
    Direction.Z: Direction.Z,
}
X_DIRECTIONS = (Direction.EAST, Direction.WEST)
Y_DIRECTIONS = (Direction.NORTH, Direction.SOUTH)
LOCAL_DELIVER = "local_deliver"

# The Z port roles the 2D rules know, by their names in the manifest's vocabulary.z_port_roles. Only an express chord
# is a Z step within the mesh.
Z_PORT_ROLES = {"none", "intermesh_boundary", "express_chord"}


class ChunkEncoding(str, Enum):
    """A scatter write chunk type the rules know, by its name in NocScatterWriteChunkEncoding."""

    NOP = "chunk_encoding_nop"
    UNICAST_WRITE = "chunk_encoding_unicast_write"
    SEMINC_NO_FLUSH = "chunk_encoding_seminc_no_flush"
    SEMINC_FLUSH = "chunk_encoding_seminc_flush"


# A chip's place in its mesh, (y, x), as its mesh_coord.
Coord = tuple[int, int]


class ManifestError(ValueError):
    """The manifest lacks a value the route rules need, or holds one they can't use."""


@dataclass(frozen=True)
class Hop:
    """What a 1D router does at one hop: the hop field's name in routing_encodings.1d.hop, and its actions."""

    name: str
    writes: bool
    forwards: bool


@dataclass(frozen=True)
class Encodings1D:
    bits_per_hop: int
    hops_per_word: int
    hops: tuple[Hop, ...]  # indexed by a hop field's value


@dataclass(frozen=True)
class BitField:
    """
    A field of `bits` bits, `shift` bits up from the bottom of a packed value. For example, a tree edge's parent row is
    BitField(shift=6, bits=6), so BitField(6, 6).read(0x2083) is 2: 0x2083 is 0b00_10_000010_000011, and bits 6 to 11
    are 0b000010.
    """

    shift: int
    bits: int

    def read(self, value: int) -> int:
        return (value >> self.shift) & ((1 << self.bits) - 1)


@dataclass(frozen=True)
class Encodings2D:
    direction_bits: tuple[tuple[Direction, int], ...]  # (direction, its action bit), in Direction order
    local_deliver_bit: int
    table_entry_bits: int
    table_entries_per_byte: int
    y_code_action: tuple[int, ...]  # the action a Y table code widens to, indexed by code
    x_code_action: tuple[int, ...]  # the action an X table code widens to, indexed by code
    tree_edge_bytes: int
    tree_edge_child: BitField
    tree_edge_parent: BitField
    tree_edge_output: BitField  # a table code


@dataclass(frozen=True)
class EncodingsScatter:
    chunk_type_bits: int
    max_chunks: int
    chunk_encodings: tuple[ChunkEncoding, ...]  # indexed by a chunk type's value


@dataclass(frozen=True)
class Encodings:
    """The manifest's routing_encodings, checked."""

    one_d: Encodings1D
    two_d: Encodings2D
    scatter: EncodingsScatter


# JSON's name for each type json.load returns. bool comes first: a JSON true or false is also a Python int.
_JSON_TYPES = (
    (bool, "boolean"),
    (int, "integer"),
    (float, "number"),
    (str, "string"),
    (type(None), "null"),
    (list, "array"),
    (dict, "object"),
)


def _json_type(value) -> str:
    return next((name for cls, name in _JSON_TYPES if isinstance(value, cls)), type(value).__name__)


def _a(json_type: str) -> str:
    """A JSON type's name with its article, such as "an integer"."""
    if json_type == "null":
        return json_type
    return f"{'an' if json_type[0] in 'aeiou' else 'a'} {json_type}"


def _lookup(manifest: dict, path: str, kind: str | tuple[str, ...], of: str | None = None):
    """
    The manifest's value at a dotted path, such as "routing_encodings.1d.hop". `kind` is the JSON type the value must
    have, or a tuple of them, and `of` the type of each item of an array or each value of an object.
    """
    value = manifest
    keys = path.split(".")
    for depth, key in enumerate(keys):
        if not isinstance(value, dict):
            parent = ".".join(keys[:depth]) or "the manifest"
            raise ManifestError(f"{parent} is {_a(_json_type(value))}; the route rules need an object")
        if key not in value:
            raise ManifestError(f"manifest has no {path}")
        value = value[key]

    kinds = (kind,) if isinstance(kind, str) else kind
    need = " or ".join(_a(name) for name in kinds) + (f" of {of}s" if of else "")
    if _json_type(value) not in kinds:
        raise ManifestError(f"{path} is {_a(_json_type(value))}; the route rules need {need}")
    if of is not None:
        items = value.items() if isinstance(value, dict) else enumerate(value)
        for item_key, item in items:
            if _json_type(item) != of:
                where = f"{path}.{item_key}" if isinstance(value, dict) else f"{path}[{item_key}]"
                raise ManifestError(f"{where} is {_a(_json_type(item))}; the route rules need {path} to be {need}")
    return value


def _encodings_1d(manifest: dict) -> Encodings1D:
    bits_per_hop = _lookup(manifest, "routing_encodings.1d.bits_per_hop", "integer")
    hops_per_word = _lookup(manifest, "routing_encodings.1d.hops_per_word", "integer")
    hop = _lookup(manifest, "routing_encodings.1d.hop", "object", of="integer")
    if bits_per_hop * hops_per_word != 8 * WORD_BYTES:
        raise ManifestError(
            f"routing_encodings.1d: {hops_per_word} hops of {bits_per_hop} bits don't fill a {WORD_BYTES}-byte word"
        )
    if set(hop) != set(HOP_ACTIONS):
        raise ManifestError(f"routing_encodings.1d.hop names {sorted(hop)}; the 1D rules know {sorted(HOP_ACTIONS)}")
    if sorted(hop.values()) != list(range(1 << bits_per_hop)):
        raise ManifestError(
            f"routing_encodings.1d.hop values {sorted(hop.values())} aren't each {bits_per_hop}-bit value once"
        )
    by_value = sorted(hop.items(), key=lambda item: item[1])
    return Encodings1D(bits_per_hop, hops_per_word, tuple(Hop(name, *HOP_ACTIONS[name]) for name, _ in by_value))


def _encodings_2d(manifest: dict) -> Encodings2D:
    action_bits = _lookup(manifest, "routing_encodings.2d.action_bits", "object", of="integer")
    table_entry_bits = _lookup(manifest, "routing_encodings.2d.table_entry_bits", "integer")
    table_entries_per_byte = _lookup(manifest, "routing_encodings.2d.table_entries_per_byte", "integer")
    y_code_action = _lookup(manifest, "routing_encodings.2d.y_code_action", "array", of="integer")
    x_code_action = _lookup(manifest, "routing_encodings.2d.x_code_action", "array", of="integer")
    edge = "routing_encodings.2d.mcast_tree_edge"
    tree_edge_bytes = _lookup(manifest, f"{edge}.bytes", "integer")
    tree_edge_fields = {
        name: BitField(
            _lookup(manifest, f"{edge}.{name}.shift", "integer"), _lookup(manifest, f"{edge}.{name}.bits", "integer")
        )
        for name in ("child", "parent", "output")
    }
    known = {*(direction.value for direction in Direction), LOCAL_DELIVER}

    # Catch any inconsistencies in different action bit values between the manifest and the decoder
    if set(action_bits) != known:
        raise ManifestError(
            f"routing_encodings.2d.action_bits names {sorted(action_bits)}; the 2D rules know {sorted(known)}"
        )

    # Ensure that action bit values are distinct from one-another and that they all fit within a byte
    bits = list(action_bits.values())
    if len(set(bits)) != len(bits) or not set(bits) <= {1 << shift for shift in range(8)}:
        raise ManifestError(f"routing_encodings.2d.action_bits values {bits} aren't distinct bits of a byte")

    # Ensure that we are filling up the byte correctly
    if table_entry_bits * table_entries_per_byte != 8:
        raise ManifestError(
            f"routing_encodings.2d: {table_entries_per_byte} table entries of {table_entry_bits} bits don't fill a byte"
        )

    # Ensure that we can actually represent the number of actions with the bits available
    for name, actions in (("y_code_action", y_code_action), ("x_code_action", x_code_action)):
        if len(actions) != 1 << table_entry_bits:
            raise ManifestError(
                f"routing_encodings.2d.{name} has {len(actions)} actions; {table_entry_bits}-bit codes need "
                f"{1 << table_entry_bits}"
            )

    # Ensure each tree edge field lies within an edge
    for name, field in tree_edge_fields.items():
        if field.shift + field.bits > 8 * tree_edge_bytes:
            raise ManifestError(
                f"{edge}.{name} ends at bit {field.shift + field.bits}, past a {tree_edge_bytes}-byte edge"
            )

    # Ensure an edge's output is a table code, so it widens through the code-to-action lists
    if tree_edge_fields["output"].bits != table_entry_bits:
        raise ManifestError(
            f"{edge}.output is {tree_edge_fields['output'].bits} bits; table codes are {table_entry_bits}"
        )

    return Encodings2D(
        direction_bits=tuple((direction, action_bits[direction.value]) for direction in Direction),
        local_deliver_bit=action_bits[LOCAL_DELIVER],
        table_entry_bits=table_entry_bits,
        table_entries_per_byte=table_entries_per_byte,
        y_code_action=tuple(y_code_action),
        x_code_action=tuple(x_code_action),
        tree_edge_bytes=tree_edge_bytes,
        tree_edge_child=tree_edge_fields["child"],
        tree_edge_parent=tree_edge_fields["parent"],
        tree_edge_output=tree_edge_fields["output"],
    )


def _encodings_scatter(manifest: dict) -> EncodingsScatter:
    chunk_type_bits = _lookup(manifest, "routing_encodings.scatter.chunk_encoding_bits", "integer")
    max_chunks = _lookup(manifest, "routing_encodings.scatter.max_chunks", "integer")
    types = _lookup(manifest, "enums.NocScatterWriteChunkEncoding.values", "object", of="integer")
    known = {encoding.value for encoding in ChunkEncoding}
    if set(types) != known:
        raise ManifestError(
            f"enums.NocScatterWriteChunkEncoding names {sorted(types)}; the scatter rule knows {sorted(known)}"
        )
    if sorted(types.values()) != list(range(1 << chunk_type_bits)):
        raise ManifestError(
            f"enums.NocScatterWriteChunkEncoding values {sorted(types.values())} aren't each {chunk_type_bits}-bit "
            "value once"
        )
    by_value = sorted(types.items(), key=lambda item: item[1])
    return EncodingsScatter(chunk_type_bits, max_chunks, tuple(ChunkEncoding(name) for name, _ in by_value))


def get_encodings_from_manifest(manifest: dict) -> Encodings:
    """
    Reads the manifest's routing_encodings, and the scatter chunk types from its enums. Raises ManifestError naming
    the first value it can't use.
    """
    return Encodings(one_d=_encodings_1d(manifest), two_d=_encodings_2d(manifest), scatter=_encodings_scatter(manifest))


def read_packed(data: bytes, bits_per_entry: int, num_entries: int) -> list[int]:
    """`num_entries` entries of `bits_per_entry` bits, read lowest bits first from one run of bits across the bytes."""
    if bits_per_entry * num_entries > 8 * len(data):
        raise ValueError(
            f"{num_entries} entries of {bits_per_entry} bits need {bits_per_entry * num_entries} bits; "
            f"got {8 * len(data)}"
        )
    run = int.from_bytes(data, "little")
    mask = (1 << bits_per_entry) - 1
    return [(run >> (bits_per_entry * index)) & mask for index in range(num_entries)]


def walk_1d(words: Sequence[int], enc: Encodings1D) -> list[Hop]:
    """
    Walks the 1D route.
    `words` are the route words, word 0 first: a header's routing_fields.value, then its route_buffer. Hop 0 is what
    the next router to read the route does (this router's hop has already been consumed), hop 1 the router one chip
    further on, and so on. The last hop is the first one that doesn't forward.
    Raises ValueError at a hop that neither writes nor forwards (noop), which the router asserts on. A route that
    forwards past its last word reaches one, as does an all-zero route.
    """
    live = list(words)
    mask = (1 << enc.bits_per_hop) - 1
    hops = []
    # Each forward consumes a field, so once every field is used the next read is of an all-zero word.
    for _ in range(enc.hops_per_word * len(live) + 1):
        # Add the next hop
        hop = enc.hops[live[0] & mask]
        hops.append(hop)

        # The router only acts on hops that write or forward
        if not (hop.writes or hop.forwards):
            raise ValueError(f"hop {len(hops) - 1} is {hop.name}, which the router asserts on")

        # Terminal hop, so we can early return
        if not hop.forwards:
            return hops

        # Go to the next hop, including refilling the first word if needed with an extension word
        live[0] >>= enc.bits_per_hop
        if live[0] == 0:
            live = live[1:] + [0]
    raise ValueError(f"1D route forwards past its {len(words)} route words / has no terminal hop")


def get_path_table_entry_1d(paths: bytes, hops: int, extension_words: int) -> list[int]:
    """The route words the 1D path table holds for a destination `hops` hops away, word 0 first."""
    entry_bytes = (1 + extension_words) * WORD_BYTES
    start = hops * entry_bytes
    if start + entry_bytes > len(paths):
        raise ValueError(f"entry {hops} ends at byte {start + entry_bytes}, past the {len(paths)}-byte table")
    offsets = range(start, start + entry_bytes, WORD_BYTES)
    return [int.from_bytes(paths[at : at + WORD_BYTES], "little") for at in offsets]


@dataclass(frozen=True)
class MeshView:
    """What the 2D rules need of one mesh in the manifest."""

    mesh_id: int
    shape: Coord  # (y_size, x_size)
    torus: tuple[bool, bool]  # whether y and x wrap respectively
    z_peers: dict[Coord, Coord]  # the chip a chip's Z express chord reaches (derived from the manifest)


def get_mesh_view(manifest: dict, mesh_key: str) -> MeshView:
    """
    Reads mesh `mesh_key`, such as "M0", from the manifest.
    A chip's Z peer is known when its z_port_role is "express_chord" and all its Z routers' links reach the same chip
    of this mesh. Only local chips have a z_port_role and routers, so a chip on another host has no known Z peer.
    Raises ManifestError for a value it reads that's missing or of the wrong type, when the manifest's Z port roles
    aren't the ones the rules know, or for a local chip whose z_port_role names a role the rules don't know.
    """
    if not (mesh_key.startswith("M") and mesh_key[1:].isdigit()):
        raise ManifestError(f"mesh key {mesh_key!r} isn't M<mesh id>")
    path = f"meshes.{mesh_key}"
    shape = _lookup(manifest, f"{path}.shape", "array", of="integer")
    if len(shape) != 2:
        raise ManifestError(f"{path}.shape is {shape}; the 2D rules need [y, x]")
    torus = (_lookup(manifest, f"{path}.torus.y", "boolean"), _lookup(manifest, f"{path}.torus.x", "boolean"))
    chips = _lookup(manifest, f"{path}.chips", "object")
    coord_of = {key: tuple(_lookup(manifest, f"{path}.chips.{key}.mesh_coord", "array", of="integer")) for key in chips}

    roles = _lookup(manifest, "vocabulary.z_port_roles", "array", of="string")
    if set(roles) != Z_PORT_ROLES:
        raise ManifestError(f"vocabulary.z_port_roles names {sorted(roles)}; the 2D rules know {sorted(Z_PORT_ROLES)}")
    z_peers = {}
    for key in chips:
        chip_path = f"{path}.chips.{key}"
        if not _lookup(manifest, f"{chip_path}.is_local", "boolean"):
            continue
        role = _lookup(manifest, f"{chip_path}.z_port_role", "string")
        if role not in Z_PORT_ROLES:
            raise ManifestError(f"{chip_path}.z_port_role is {role!r}; the 2D rules know {sorted(Z_PORT_ROLES)}")
        if role != "express_chord":
            continue
        # A peer is a router path, such as "M0/C5/Z0"; an unconnected link has none.
        links = [
            _lookup(manifest, f"{chip_path}.routers.{router_key}.link.peer", ("string", "null"))
            for router_key in _lookup(manifest, f"{chip_path}.routers", "object")
            if router_key.startswith("Z")
        ]
        peers = {None if link is None else tuple(link.split("/")[:2]) for link in links}
        if len(peers) == 1:
            (peer,) = peers
            if peer is not None and peer[0] == mesh_key and peer[1] in coord_of:
                z_peers[coord_of[key]] = coord_of[peer[1]]
    return MeshView(int(mesh_key[1:]), (shape[0], shape[1]), torus, z_peers)


def read_action_2d(
    route_map: bytes, router_y_coord: int, router_x_coord: int, y_size: int, entered_from: Direction | None
) -> int:
    """
    The action byte the router at (y, x) reads from a 2D route map: `y_size` row bytes, then a byte per column.
    `entered_from` is the direction of the router that received the packet. If the packet entered from the east or west,
    we read its column's byte. Any other, we read its row's byte, or its column's if the row's is 0.
    """
    column_action = route_map[y_size + router_x_coord]
    if entered_from is not None and Direction(entered_from) in X_DIRECTIONS:
        return column_action
    return route_map[router_y_coord] or column_action


@dataclass(frozen=True)
class Step2D:
    """One link a 2D route takes: from chip `src`, in `direction`, to chip `dst`."""

    src: Coord
    direction: Direction
    dst: Coord


@dataclass(frozen=True)
class Odd2D:
    """
    A chip where the route map asks for something the router there doesn't do, or that the fabric doesn't build.
    `kind` is one of:
    - "off_edge": a direction bit that steps off an edge of the mesh that doesn't wrap;
    - "no_z_peer": a Z bit at a chip whose Z peer isn't known;
    - "ignored_bit": a direction bit the router ignores: the way the packet came in, or north or south at a router
      the packet entered from the east or west;
    - "unknown_bits": bits that are neither a direction nor local delivery;
    - "empty": an action byte of 0;
    - "revisit": a chip the route reaches again from the same side, which `direction` names.
    """

    chip: Coord
    kind: str
    direction: Direction | None = None


@dataclass(frozen=True)
class Walk2D:
    delivers: list[Coord]  # the chips that deliver, in the order the walk reaches them
    steps: list[Step2D]
    exits: list[Coord]  # the chips where the packet leaves for its destination mesh
    odd: list[Odd2D]


def _step_2d(chip: Coord, direction: Direction, mesh: MeshView) -> Coord | None:
    """The chip one step from `chip`, or None if the step leaves an edge that doesn't wrap or the Z peer is unknown."""
    # Z peer
    if direction == Direction.Z:
        return mesh.z_peers.get(chip)

    # Step
    (dy, dx), (y_size, x_size), (y_wraps, x_wraps) = STEPS_2D[direction], mesh.shape, mesh.torus
    y, x = chip[0] + dy, chip[1] + dx

    # Torus
    if y_wraps:
        y %= y_size
    if x_wraps:
        x %= x_size

    # A step off an edge that doesn't wrap has no chip
    return (y, x) if 0 <= y < y_size and 0 <= x < x_size else None


def walk_2d(
    route_map: bytes,
    start: Coord,
    mesh: MeshView,
    enc: Encodings2D,
    dst_mesh: int,
    entered_from: Direction | None = None,
) -> Walk2D:
    """
    Walks a 2D route map through `mesh` from the chip at `start`, following every direction bit each router acts on.
    `route_map` is a header's route_buffer, or a map get_route_map_from_table_2d builds. `entered_from` is as for
    read_action_2d, for the start chip. `dst_mesh` is the header's dst_start_mesh_id: where a chip's byte says only
    "deliver here" and `dst_mesh` is another mesh, the packet leaves the mesh there instead.
    """
    y_size, x_size = mesh.shape
    # Even if the route doesn't span across all chips, the route map is expected
    # to contain the maximum number of bytes for the configured mesh
    if len(route_map) < y_size + x_size:
        raise ValueError(f"a {y_size}x{x_size} mesh's route map is {y_size + x_size} bytes; got {len(route_map)}")

    # Create a mask of all the "known" bits (bits that have a value)
    known_bits = enc.local_deliver_bit
    for _, bit in enc.direction_bits:
        known_bits |= bit

    delivers, steps, exits, odd = [], [], [], []
    seen = set()

    # Initialize the todo list with the starting chip and direction
    todo = deque([(start, None if entered_from is None else Direction(entered_from))])
    while todo:
        # Pop the next chip and direction from the todo list
        chip, side = todo.popleft()
        # We should never revisit the same chip and direction in a single walk
        if (chip, side) in seen:
            odd.append(Odd2D(chip, "revisit", side))
            continue
        seen.add((chip, side))

        # Read the action byte
        action = read_action_2d(route_map, chip[0], chip[1], y_size, side)

        # If the action is "deliver here" and the destination mesh is different from the current mesh,
        # this packet is expected to leave the mesh here (intermesh hop)
        if action == enc.local_deliver_bit and dst_mesh != mesh.mesh_id:
            exits.append(chip)
            continue

        # If the action is "deliver here" and the destination mesh is the same as the current mesh,
        # this packet is expected to be delivered here
        if action & enc.local_deliver_bit:
            delivers.append(chip)

        # Unknown bits that we don't know how to process
        if action & ~known_bits:
            odd.append(Odd2D(chip, "unknown_bits"))
        elif action == 0:
            odd.append(Odd2D(chip, "empty"))

        for direction, bit in enc.direction_bits:
            # Action != bit
            if not action & bit:
                continue

            # A router doesn't send back the way the packet came, and packets entered through x don't turn into y
            if direction == side or (side in X_DIRECTIONS and direction in Y_DIRECTIONS):
                odd.append(Odd2D(chip, "ignored_bit", direction))
                continue

            # Step to the next chip
            next_chip = _step_2d(chip, direction, mesh)
            if next_chip is None:
                odd.append(Odd2D(chip, "no_z_peer" if direction == Direction.Z else "off_edge", direction))
                continue
            steps.append(Step2D(chip, direction, next_chip))
            todo.append((next_chip, OPPOSITE_2D[direction]))
    return Walk2D(delivers, steps, exits, odd)


def get_route_map_from_table_2d(action_vectors: bytes, y_size: int, x_size: int, dst: Coord, enc: Encodings2D) -> bytes:
    """
    The route map a worker or an intermesh landing builds from its chip's 2D route table, toward the chip at `dst`.
    `action_vectors` holds, for each destination row, a table code for every row; then, for each destination column,
    one for every column. Each destination's codes start on a new byte. The map is dst's row codes widened through
    y_code_action, then its column codes widened through x_code_action, with "deliver here" added at dst's column.
    """
    dst_y, dst_x = dst
    # Validate the destination coordinates
    if not (0 <= dst_y < y_size and 0 <= dst_x < x_size):
        raise ValueError(f"{dst} isn't in a {y_size}x{x_size} mesh")

    # Calculate and validate the starting indices for the y and x codes
    per_byte = enc.table_entries_per_byte
    y_row_bytes = (y_size + per_byte - 1) // per_byte
    x_row_bytes = (x_size + per_byte - 1) // per_byte
    y_start = dst_y * y_row_bytes
    x_start = y_size * y_row_bytes + dst_x * x_row_bytes
    if x_start + x_row_bytes > len(action_vectors):
        raise ValueError(f"{dst}'s column codes end at byte {x_start + x_row_bytes}, past {len(action_vectors)} bytes")

    # Read the y and x codes
    y_codes = read_packed(action_vectors[y_start : y_start + y_row_bytes], enc.table_entry_bits, y_size)
    x_codes = read_packed(action_vectors[x_start : x_start + x_row_bytes], enc.table_entry_bits, x_size)

    # Create the route map
    route_map = [enc.y_code_action[code] for code in y_codes] + [enc.x_code_action[code] for code in x_codes]

    # Add the "deliver here" bit to the destination column
    route_map[y_size + dst_x] |= enc.local_deliver_bit
    return bytes(route_map)


@dataclass(frozen=True)
class TreeEdge2D:
    """One edge of a multicast tree: routers in row (or column) `parent` reach `child` by sending in `direction`."""

    child: int
    parent: int
    direction: Direction | None  # None when the edge's code names no direction


@dataclass(frozen=True)
class McastTrees2D:
    y: list[TreeEdge2D]  # over rows, rooted at the chip's row
    x: list[TreeEdge2D]  # over columns, rooted at the chip's column


def _read_tree_2d(
    region: bytes, start: int, count: int, code_action: tuple[int, ...], enc: Encodings2D
) -> list[TreeEdge2D]:
    direction_of = {bit: direction for direction, bit in enc.direction_bits}
    edges = []
    for at in range(start, start + count * enc.tree_edge_bytes, enc.tree_edge_bytes):
        packed = int.from_bytes(region[at : at + enc.tree_edge_bytes], "little")
        action = code_action[enc.tree_edge_output.read(packed)]
        edges.append(
            TreeEdge2D(enc.tree_edge_child.read(packed), enc.tree_edge_parent.read(packed), direction_of.get(action))
        )
    return edges


def get_mcast_tree_edges_2d(region: bytes, y_size: int, x_size: int, enc: Encodings2D) -> McastTrees2D:
    """
    A chip's two multicast trees, from its 2D route table's mcast_trees region: the Y tree, then the X tree. An axis of
    n chips has n - 1 edges, each tree_edge_bytes bytes, lowest byte first. Edges keep their stored order, which puts
    every edge below a row before the edge into it. An edge's output is a table code, widened through y_code_action or
    x_code_action to its direction.
    """
    y_edges, x_edges = y_size - 1, x_size - 1

    # Our byte region does not cover all the edges
    if (y_edges + x_edges) * enc.tree_edge_bytes > len(region):
        raise ValueError(
            f"a {y_size}x{x_size} mesh's trees are {(y_edges + x_edges) * enc.tree_edge_bytes} bytes; got {len(region)}"
        )

    # Read the y and x tree edges
    return McastTrees2D(
        y=_read_tree_2d(region, 0, y_edges, enc.y_code_action, enc),
        x=_read_tree_2d(region, y_edges * enc.tree_edge_bytes, x_edges, enc.x_code_action, enc),
    )


@dataclass(frozen=True)
class ScatterChunk:
    """
    What one chunk of a scatter write does at its noc_address, by the type the router treats it as: a unicast write
    writes `amount` bytes of the payload, and a semaphore increment adds `amount` to the semaphore there, flushing the
    packet's writes first for SEMINC_FLUSH.
    """

    encoding: ChunkEncoding
    amount: int


def get_scatter_chunks(
    chunk_count: int, chunk_size: Sequence[int], chunk_encoding: int, payload_size_bytes: int, enc: EncodingsScatter
) -> list[ScatterChunk]:
    """
    What each chunk of a scatter write does, chunk 0 first, as the router carries it out. Every chunk but the last
    writes its chunk_size, whatever its type. The last chunk's type (chunk i's is chunk_type_bits bits at bit
    i * chunk_type_bits of `chunk_encoding`) decides what it does: a write gets the rest of the payload, and an
    increment adds its chunk_size slot. Raises ValueError for a scatter the router can't carry out.
    """
    # The router writes chunk 0 before it reads the last chunk's type, so chunk 0 can't be the last.
    if not 2 <= chunk_count <= enc.max_chunks:
        raise ValueError(f"chunk_count is {chunk_count}; a scatter has 2 to {enc.max_chunks} chunks")

    # We write every chunk but the last
    last = chunk_count - 1
    written = sum(chunk_size[:last])
    chunks = [ScatterChunk(ChunkEncoding.UNICAST_WRITE, size) for size in chunk_size[:last]]

    # Decode the last chunk's type
    encoding = enc.chunk_encodings[BitField(last * enc.chunk_type_bits, enc.chunk_type_bits).read(chunk_encoding)]
    if encoding == ChunkEncoding.NOP:
        raise ValueError(f"the last chunk, chunk {last}, is a nop")
    if encoding == ChunkEncoding.UNICAST_WRITE:
        if written > payload_size_bytes:
            raise ValueError(f"chunks 0 to {last - 1} write {written} bytes, past a {payload_size_bytes}-byte payload")
        return chunks + [ScatterChunk(encoding, payload_size_bytes - written)]
    if last >= len(chunk_size):
        raise ValueError(f"chunk {last} increments, but chunk_size has no slot {last}")
    return chunks + [ScatterChunk(encoding, chunk_size[last])]
