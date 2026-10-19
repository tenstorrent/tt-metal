# Route rules

[`routes.py`](routes.py) reads fabric routes from raw bytes: the route fields of a packet header, the routing tables in a router's L1, and the chunks of a scatter write. The manifest gives it every value routes are encoded with (`routing_encodings`, and the `compressed_routing_values` and `NocScatterWriteChunkEncoding` enums). How to read a route with those values isn't data the manifest can carry, so `routes.py` holds it as rules, each written to match the fabric's C++.

This page lists the rules, for fabric owners to check against the fabric, and names the code each one mirrors. A fabric
change to that code should update the rule, its function and its tests in the same change: nothing else catches it
(see [When the fabric changes](#when-the-fabric-changes)).

## What it takes and returns

The caller passes the manifest and bytes it has already cut out through the manifest's layouts, such as a header's
`route_buffer` or a routing table's `action_vectors`. `routes.py` returns plain data: ints, lists and small frozen
dataclasses. It doesn't find the bytes, decide whether a captured packet is live, or know which way a channel faces.
The caller does, and passes what a rule needs, such as `entered_from`, the direction of the router that received the
packet.

Every rule takes the encodings for its routing mode, read once per manifest:

```python
enc = get_encodings_from_manifest(manifest)  # enc.one_d, enc.two_d, enc.scatter
mesh = get_mesh_view(manifest, "M0")
walk = walk_2d(route_buffer, start=(0, 0), mesh=mesh, enc=enc.two_d, dst_mesh=dst_start_mesh_id)
walk.delivers, walk.exits, walk.odd
```

## Design decisions

- **Values come from the manifest, by name.** `get_encodings_from_manifest` looks up each value by its key, checks its JSON type and its value (for example, that the action bits are distinct bits of a byte, and that the 1D hop fields fill a route word), and raises `ManifestError` naming the first key it can't use. `get_mesh_view` does the same for the mesh values it reads.
- **Meaning is tied to names, and only the known names are accepted.** The rules hardcode what some names mean: which 1D hop names write and which forward (`HOP_ACTIONS`), the direction names (`Direction`), `local_deliver`, the Z port roles (`Z_PORT_ROLES`), and the scatter chunk type names (`ChunkEncoding`). The loaders check those names against the manifest's full list of each (`routing_encodings.1d.hop`, `routing_encodings.2d.action_bits`, `vocabulary.z_port_roles` and `enums.NocScatterWriteChunkEncoding`), not only the values a run uses, so a renamed, added or removed value fails at load, naming what the manifest holds and what the rules know, instead of decoding wrong.
- **Names are enum values.** `Direction` and `ChunkEncoding` are `str` enums whose values are the manifest's names. A member equals its name, looks up the manifest's keys, and writes to JSON as the name.
- **The rules mirror the router, not the API.** Where the code that builds a route and the router that reads it
  disagree, the rules follow the router. For example, a plain scatter write marks all four chunk types as writes,
  whatever its `chunk_count`. The router goes by `chunk_count`, and so does the rule.
- **Odd routes are reported; routes the router asserts on raise.** `walk_2d` returns what it can't follow as `Odd2D` entries next to the route, since a hung fabric is where odd routes show up. A function raises `ValueError` for input it can't read, such as a buffer too short for the mesh, and for a route the router asserts on: a 1D route that reaches a `noop` hop, or a scatter the router can't carry out.
- **One function per rule.** Names start with a verb. A function for one routing mode ends in `_1d` or `_2d`: the 1D routing path table and the 2D route table share one slot of the routing table, so a build has only one of them.

## The rules

Each rule says what the fabric does, the function that does the same, and the fabric code it mirrors. The files are listed with their paths under [Files cited](#files-cited). Diagrams use the values today's manifest gives; the code reads them from the manifest.

### Packed values

Packed entries are read lowest bits first, from one continuous run of bits across the bytes, so an entry can straddle two bytes.

- **Function:** `read_packed`. `get_route_map_from_table_2d` uses it for the 2D route table's codes.
- **Mirrors:** `direction_table_t::get_direction` (`fabric_direction_table_interface.h`), which reads the first-hop direction tables, 3-bit `compressed_routing_values` entries.
- A value no enum name covers, such as 7 in a direction table, comes back as is, for the caller to flag.

### 1D: walking a route

A 1D route is a list of route words: the header's `routing_fields.value`, then its `routing_fields.route_buffer`
extension words. Each word holds `hops_per_word` hop fields of `bits_per_hop` bits, hop 0 lowest:

```
routing_fields.value          route_buffer[0]         route_buffer[1] ...
[ h15 | ... | h1 | h0 ]       [ h31 | ... | h16 ]
bit 31            bit 0
```

The router reads the lowest field of the first word. Its name in `routing_encodings.1d.hop` says what to do:
`write_only` and `write_and_forward` deliver at this chip, and `forward_only` and `write_and_forward` send the packet on. Before sending it on, the router shifts the first word down by one field; when the first word is then 0, the next word moves up into its place. The route ends at the first hop that doesn't forward.

The router has no case for `noop`, and asserts on it. With the watcher off, the assert compiles away (`assert.h`) and the packet is neither written nor forwarded. In today's manifest an all-zero field is `noop`, so a route that forwards past its last word reaches one, as does an all-zero route.

- **Function:** `walk_1d`. Hop 0 is what the next router to read the header does: a router shifts its own hop out
  before it sends the packet on. It raises `ValueError` at a `noop` hop.
- **Mirrors:** `receiver_forward_packet` (`fabric_erisc_router.cpp`) reads the field, with `ASSERT(false)` as its
  `default` case, and `update_packet_header_for_next_hop` (`fabric_edm_packet_transmission.hpp`) shifts and refills.

### 1D: the routing path table

`routing_path_table_1d.paths` holds one entry per distance: entry `i` is the route words to the chip `i` hops away, `1 + routing_1d_extension_words` words each, packed back to back.

- **Function:** `get_path_table_entry_1d`.
- **Mirrors:** `intra_mesh_routing_path_t<1, false>::calculate_chip_to_all_routing_fields`
  (`compressed_routing_path.cpp`), which fills the table by hop count, and `intra_mesh_routing_path_t<1, false>::decode_route_to_buffer` (`fabric_routing_path_interface.h`), which reads it. Its parameter is named `dst_chip_id`, but its caller, `fabric_set_unicast_route` (`tt_fabric_api.h`), passes a hop count.

### 2D: the route map

A 2D header's `route_buffer` is a route map: one action byte per row of the mesh, then one per column. An action byte has a bit for each direction (east, west, north, south, Z) and one for `local_deliver`, as `routing_encodings.2d.action_bits` gives them. North is decreasing y and west is decreasing x.

```
route_buffer for a 3 x 4 mesh (y_size 3, x_size 4):

  byte:  0    1    2    3    4    5    6
       [ Y0 | Y1 | Y2 | X0 | X1 | X2 | X3 ]
         one per row    one per column

the router at (y 1, x 2) reads Y1, or X2
```

### 2D: which byte a router reads

A router the packet entered from the east or west reads only its column's byte. Any other router reads its row's byte, or its column's byte if the row's is 0. At a route's start (a worker sending the packet, or an intermesh landing that has just rebuilt the map), the row's byte is read first.

- **Function:** `read_action_2d`. `entered_from` is the direction of the router that received the packet: a packet
  sent east arrives at the next chip's west router, so it entered from the west. `None` is a route's start.
- **Mirrors:** `Routing2DCodec::decode_action` and `Routing2DCodec::decode_action_y_first` (`fabric_common.h`).

### 2D: following the route

At each chip, the router delivers locally if `local_deliver` is set, and sends the packet one chip on in every
direction whose bit is set, wrapping around on a torus axis. Z goes to the chip's Z peer. A router ignores two kinds of bit: the direction the packet came in from, and north or south at a router entered from the east or west, since a packet turns from Y to X but not back.

- **Function:** `walk_2d`, with `get_mesh_view` for the mesh's shape, torus axes and Z peers.
- **Mirrors:** `admit_2d_dispatch` and `live_eth_mask_2d` (`fabric_erisc_router.cpp`).
- A chip's Z peer is known when its `z_port_role` is `express_chord` and all its Z routers' `link.peer`s name the same chip of this mesh. A chip has one Z port, used either as an express chord along Y or as an intermesh boundary
(`ZPortRole`, `fabric_edge_capability.hpp`).
- What the walk can't follow, it reports as an `Odd2D` and doesn't follow:
  - `off_edge`: a step off an edge that doesn't wrap;
  - `no_z_peer`: a Z bit at a chip whose Z peer isn't known;
  - `ignored_bit`: a bit the router ignores, as above;
  - `unknown_bits`: bits that are neither a direction nor `local_deliver`;
  - `empty`: an action byte of 0;
  - `revisit`: a chip reached again from the same side, which also ends a loop.

### 2D: leaving the mesh

A chip whose action byte is exactly `local_deliver`, in a packet whose `dst_start_mesh_id` is another mesh, is where the packet leaves the mesh. That chip neither delivers nor forwards: it sends the packet over its intermesh link.

- **Function:** `walk_2d`, which lists these chips in `Walk2D.exits`.
- **Mirrors:** `Routing2DCodec::action_is_intermesh_exit` (`fabric_common.h`), which the router checks on its edge
  routers (`fabric_erisc_router.cpp`).

### 2D: a route map from the route table

`route_table_2d.action_vectors` is destination-major. For each destination row, it holds a code for every row; then, for each destination column, a code for every column. Codes are `table_entry_bits` bits, `table_entries_per_byte` to a
byte, and each destination's codes start on a new byte. The route map toward the chip at (row r, column c) is row r's codes widened through `y_code_action`, then column c's codes widened through `x_code_action`, with `local_deliver` added at column c.

```
action_vectors for a 3 x 4 mesh (2-bit codes, 4 to a byte):

  [ dst row 0 | dst row 1 | dst row 2 | dst col 0 | dst col 1 | dst col 2 | dst col 3 ]
    1 byte each: codes for rows 0-2       1 byte each: codes for columns 0-3

route map toward (2, 1) = widen_y(dst row 2) + widen_x(dst col 1), with local_deliver at column 1
```

- **Function:** `get_route_map_from_table_2d`.
- **Mirrors:** `widen_2d_route_to_chip` (`fabric_2d_route_interface.h`), with `Routing2DCodec::y_row`, `x_row`,
  `widen_y` and `widen_x` (`fabric_common.h`). Workers call it to send a unicast and to route toward their mesh's
  exit, and an intermesh landing calls it to rebuild a unicast's map in the new mesh.

### 2D: multicast trees

`route_table_2d.mcast_trees` holds this chip's two multicast trees: a Y tree over the rows, rooted at the chip's row, then an X tree over the columns, rooted at its column. An axis of n chips has n - 1 edges, and none for one chip, so the X tree starts at byte `tree_edge_bytes * (y_size - 1)`. Each edge is `tree_edge_bytes` bytes, lowest byte first, with the `child`, `parent` and `output` fields `routing_encodings.2d.mcast_tree_edge` gives. `output` is a table code of the edge's axis: widened through `y_code_action` or `x_code_action`, it's the direction routers in the parent row or column send in to reach the child. Edges are stored with every edge below a row before the edge into it.

```
mcast_trees for a 4 x 3 mesh:

  bytes 0-5: Y tree, 3 edges       bytes 6-9: X tree, 2 edges
  [ edge | edge | edge ]           [ edge | edge ]

one edge (2 bytes):

  bits  15-14  | 13-12  | 11-6   | 5-0
        unused | output | parent | child

  0x2083 = 0b00_10_000010_000011: child row 3, parent row 2, output 2 (south on Y)
  -> routers in row 2 reach row 3 by sending south
```

- **Function:** `get_mcast_tree_edges_2d`. An edge whose code names no direction has `direction` `None`; the host never writes one.
- **Mirrors:** `Routing2DCodec::mcast_tree_edge_count`, `mcast_tree_x_offset`, `mcast_edge_child`, `mcast_edge_parent` and `mcast_edge_output` (`fabric_common.h`). The host writes the trees with `pack_mcast_reverse_tree` and `embed_mcast_reverse_trees` (`mcast_reverse_tree.cpp`).
- The trees are for showing a chip's routing table. Decoding a header doesn't need them: a worker builds a multicast's route map from its trees when it sends (`encode_2d_mcast_maps`, `fabric_common.h`), and the header carries the finished map, which `walk_2d` reads.

### Scatter writes

A scatter write has `chunk_count` chunks, 2 to `max_chunks`. Every chunk but the last writes `chunk_size[i]` bytes of
the payload to its `noc_address[i]`, whatever its type. The last chunk's type, `chunk_encoding_bits` bits at bit
`i * chunk_encoding_bits` of `chunk_encoding`, decides what it does:

- `chunk_encoding_unicast_write`: writes the rest of the payload;
- `chunk_encoding_seminc_no_flush`: adds `chunk_size[last]` to the semaphore at its address;
- `chunk_encoding_seminc_flush`: the same, after flushing the packet's earlier writes;
- `chunk_encoding_nop`: the router asserts.

```
chunk_count 3, chunk_size [64, 192, 1], chunk_encoding 0b11_01_01, a 256-byte payload:

  chunk 0: writes 64 bytes        (its type isn't read)
  chunk 1: writes 192 bytes       (its type isn't read)
  chunk 2: type 0b11, seminc_flush: flushes chunks 0 and 1, then adds 1
```

- **Function:** `get_scatter_chunks`. It raises for a scatter the router can't carry out: fewer than 2 or more than
  `max_chunks` chunks, a nop last, earlier writes longer than the payload, or an increment as the fourth chunk.
  `chunk_size` has `max_chunks - 1` slots, so the router would read past it.
- **Mirrors:** the `NOC_UNICAST_SCATTER_WRITE` case of `execute_chip_unicast_to_local_chip_impl`
  (`fabric_edm_packet_transmission.hpp`), and `NocUnicastScatterCommandHeader` (`fabric_edm_packet_header.hpp`).

## Where the walk differs from the router, on purpose

- **It checks for an exit at every chip.** The router checks only on routers built as edge routers
  (`is_intramesh_router_on_edge`), and the manifest doesn't say which those are. A route map from the route table
  holds exactly `local_deliver`, with another mesh as the destination, only at the exit chip.
- **It doesn't know which outputs a router was built with** (`live_eth_mask_2d`). It follows a bit toward a neighbour by coordinate, whether or not the chip has a link there.
- **At a route's start, it ignores no bits.** A walk from an intermesh landing starts with `entered_from=None`, so it doesn't drop a bit pointing back out of the landing router, as the router would. A map rebuilt toward a chip in this mesh has no reason to point back.
- **It follows Z only to a known peer.** Only chips on this host have routers in the manifest, so a Z step from a chip on another host is reported as `no_z_peer`.

## What the rules don't cover

- **A multicast into another mesh.** The fabric has code for it (the carrier leg in `fabric_set_2d_mcast_route` and the multicast branch of `fabric_set_2d_intermesh_landing_route`, both in `tt_fabric_api.h`), but intermesh multicast isn't supported, and the fabric's test infrastructure refuses to generate it (`tt_fabric_test_config.cpp`). The rules don't mirror the landing's rebuild, so `walk_2d` follows such a packet only as far as its exit.
- **Finding the bytes.** Reading a header's members, deciding which captured slots hold a live packet, and knowing
  which way a channel faces belong to the caller.

## When the fabric changes

The rows follow the three cases in [README.md](README.md#what-decode-reads-and-what-it-knows): a value whose meaning
the manifest carries, a value whose meaning is in `routes.py`, and a change to the logic alone.

| Change | What fails | What to edit |
| --- | --- | --- |
| An encoding's value or width changes, such as a bit position, the table code width or a tree edge field | Nothing, while the loader's checks still hold: the rules read these from the manifest | Nothing |
| A key the rules read is renamed or removed | The route decoder check in `run_fabric_manifest_tests.sh`: `get_encodings_from_manifest` or `get_mesh_view` raises `ManifestError`, "manifest has no ...", naming the manifest | The key in `routes.py` |
| A value the rules read changes type, such as an object where an array was | The same check: `ManifestError`, naming the key, the type it has and the type the rules need | The read in `routes.py`, and its `_lookup` type |
| A 1D hop, 2D action bit, Z port role or scatter chunk type is renamed, added or removed | The same check: `ManifestError`, naming what the manifest holds and what the rules know | `HOP_ACTIONS`, `Direction` and `LOCAL_DELIVER`, `Z_PORT_ROLES`, or `ChunkEncoding`, and the rule on this page |
| The code a rule mirrors changes what it does | Nothing | The rule on this page, its function in `routes.py`, and its tests |

`tests/check_manifest_routes.py` runs the loaders on every manifest `run_fabric_manifest_tests.sh` writes, which is
what makes the second to fourth rows fail in CI. What it can't reach is the path to a Z router's peer (`routers.<Z key>.link.peer`): decode reads it only at an express chord chip, and no mock-cluster chip is one, so a key renamed or a type changed on that path fails only when decode reads a manifest that has one.

`tests/test_routes.py` holds hand-written cases, one or two for each branch of each rule. They catch a change to
`routes.py` that breaks a rule, not a fabric change the rules missed.

## Files cited

| File | Path |
| --- | --- |
| `fabric_common.h` | `tt_metal/hostdevcommon/api/hostdevcommon/` |
| `fabric_direction_table_interface.h`, `fabric_routing_path_interface.h`, `fabric_edm_packet_header.hpp` | `tt_metal/hw/inc/hostdev/` |
| `fabric_erisc_router.cpp` | `tt_metal/fabric/impl/kernels/edm_fabric/` |
| `assert.h` | `tt_metal/hw/inc/api/debug/` |
| `fabric_edm_packet_transmission.hpp` | `tt_metal/fabric/hw/inc/edm_fabric/` |
| `fabric_2d_route_interface.h`, `tt_fabric_api.h` | `tt_metal/fabric/hw/inc/` |
| `compressed_routing_path.cpp`, `mcast_reverse_tree.cpp` | `tt_metal/fabric/` |
| `fabric_edge_capability.hpp` | `tt_metal/fabric/builder/` |
| `tt_fabric_test_config.cpp` | `tests/tt_metal/tt_fabric/test_infra/` |
