# Routing tables

[`routing_tables.py`](../routing_tables.py) decodes the routing tables a snapshot holds. Each router holds its chip's
routing table in L1, in its `routing_table` area. It takes those bytes and gives the decoded table:
- its fields, as the type parser decodes them;
- in a 2D run, the route it gives toward every chip of its mesh, and its multicast trees;
- in a 1D run, its path table.

A chip's routers hold the same table, so each unique table is decoded and stored once (see "Storing each table once"
below).

It reports what the tables say. Whether they agree, or match what the host meant to write, is left to the reader.

## What it takes and returns

`get_routing_table_store(snapshot, types, decoded_routers, ctx)` takes:
- the snapshot, as `inputs.py` loads it, and its types;
- every router of the snapshot decoded by [`routers.py`](../routers.py), by path, in snapshot order;
- a `RouteContext` from [`route_context.py`](../route_context.py).

It gives a `RoutingTableStore`, with two maps:

| Field | What it holds |
| --- | --- |
| `copies` | Each unique table, a `RoutingTable`, by its key |
| `routers` | For each router whose area capture read, its `RouterTableRef`, by router path |

A `RoutingTable`:

| Field | What it holds |
| --- | --- |
| `chip` | The chip it belongs to, `M<mesh>/C<chip>` |
| `routers` | Every router holding it, in snapshot order. The first one's path is the copy's key |
| `raw` | The first router's area in the `.bin`, as a bytes value, `{"bytes": {"offset", "size"}}` |
| `fields` | The area's fields, as the type parser decodes them, without `state_manager` |
| `route_table_2d` | In a 2D run, the route toward each chip of the mesh, by chip key. `None` in a 1D run |
| `mcast_trees` | In a 2D run, the multicast trees. `None` in a 1D run |
| `path_table_1d` | In a 1D run, the path table. `None` in a 2D run |

A `RouterTableRef` is a router's reference to the `RoutingTable` it holds, plus what's its own in its area:

| Field | What it holds |
| --- | --- |
| `table_key` | The key of its `RoutingTable` in `copies` |
| `state_manager` | Its own `state_manager`, decoded |
| `raw` | Its own area in the `.bin`, so its own bytes can be shown too |

A router whose area capture didn't read has no `RouterTableRef`, and isn't in any copy.

From the T3K snapshot:

```python
copy = store.copies["M0/C0/E1"]
copy.routers                          # ['M0/C0/E1', 'M0/C0/E0', 'M0/C0/S1', 'M0/C0/S0']
copy.raw                              # {'bytes': {'offset': 164, 'size': 2704}}
store.routers["M0/C0/E0"].table_key   # 'M0/C0/E1'
store.routers["M0/C0/E0"].state_manager
# {'state': {'raw': 1, 'name': 'running'}, 'command': {'raw': 0, 'name': 'run'}}
```

**The route context.** `get_route_context(manifest)` reads, once per snapshot, what the route rules need of the
manifest: the encodings, the routing mode, each mesh's view, each chip's key and coordinate, and in a 1D run the
extension words and the line view. The ring buffer slots use it too, so the tables and the slots name chips the same way.
`name_walk_2d` turns a `Walk2D`'s coordinates into chip keys, as a `ChipWalk2D`.

## Storing each table once

**One table per chip.** The host builds one `routing_l1_info_t` per chip and writes it to every core of the chip:
Tensix, idle Ethernet and active Ethernet (`write_routing_info_to_devices`). So a chip's routers hold the same bytes,
apart from `state_manager`.

**`state_manager` is the router's own.** The host writes it with the rest of the table, as `INITIALIZING` and `RUN`.
After that, each router writes its own state there, and the host writes commands to each router's copy, such as a
pause. So it differs per router, and stays with the router.

**What's compared.** The area's bytes, pad bytes included, with `state_manager`'s bytes left out. Routers of one chip
whose bytes match share a copy. A router with any other byte different gets its own copy, and the bytes that differ are
in the two copies' `raw`.

**Chips never share.** Routers of different chips get different copies, even when the bytes match:
- a table's routes start at its chip;
- most tables hold their chip's ids (`my_device_id`, `my_mesh_coord_*`).

**Keys.** A copy is keyed by the first router holding it, in snapshot order. So every key is a real router path, and
keys stay unique when the ranks of a fabric are put together.

In each of the three real snapshots, every chip has one copy: 8 copies for 40 routers.

## 2D: routes and trees

**Routes.** For every chip of the mesh, `get_route_map_from_table_2d` builds the map toward it, as a worker on the
copy's chip does for a unicast packet. `walk_2d` walks it from the copy's chip, as a route's start (`entered_from` is
`None`), with this mesh as the destination mesh. Each route is a `ChipWalk2D`:

| Field | What it holds |
| --- | --- |
| `delivers` | The chips that deliver, in the order the walk reaches them |
| `steps` | Each link it takes, as `src`, `direction` and `dst` |
| `exits` | Always empty: the destination is in this mesh |
| `odd` | Each chip where the map asks for something the router there doesn't do, as `walk_2d` reports it |

The route toward the copy's own chip delivers where it starts, with no steps. From the T3K snapshot, C0 is at (0, 0),
and C6 at (1, 2):

```python
copy.route_table_2d["M0/C0"]   # ChipWalk2D(delivers=['M0/C0'], steps=[], exits=[], odd=[])
copy.route_table_2d["M0/C6"]
# ChipWalk2D(delivers=['M0/C6'],
#            steps=[ChipStep2D('M0/C0', 'south', 'M0/C4'), ChipStep2D('M0/C4', 'east', 'M0/C5'),
#                   ChipStep2D('M0/C5', 'east', 'M0/C6')],
#            exits=[], odd=[])
```

In the T3K and Blackhole 2D snapshots, every route delivers to its destination and nothing else, with no odd chips.
Each takes as many steps as the chips are apart, the rows first.

**Trees.** `mcast_trees` is `get_mcast_tree_edges_2d`'s `McastTrees2D`: the Y tree's edges, then the X tree's. Each
edge is `child`, `parent` and `direction`.

**Routes toward other meshes** aren't built here: they go to `exit_node_table[dst_mesh]`, which is in `fields`. The slots
build them where a packet lands, with `get_landing_route_map_2d`.

## 1D: the path table

`path_table_1d` is a `PathTable1D`:

| Field | What it holds |
| --- | --- |
| `entries` | Each nonzero entry, nearest first, as a `PathEntry1D` |
| `zero_entries` | How many entries are all zero |

A `PathEntry1D`:

| Field | What it holds |
| --- | --- |
| `hops_away` | The entry's index: a worker reads entry `i` for the chip `i` hops along |
| `hops` | Each hop's name, hop 0 first, from `walk_1d`. `None` when the walk refuses |
| `problem` | Why the walk refuses: a hop the router would assert on. `None` otherwise |

- **Entries are `1 + routing_1d_extension_words` words each.**
- **Which entries the host fills.** Entries 1 to n - 1, where n is the mesh's chips on this host, up to
  `MAX_CHIPS_LOWLAT_1D` (`compute_and_embed_1d_routing_path_table`). Entry `i` is `i - 1` forward hops, then a write
  (`encode_1d_unicast`). Entry 0, the chip itself, and the entries past n stay zero. A nonzero entry 0 is listed like
  any other.
- **No chips.** An entry is a hop count. Which way it goes depends on the router the worker sends on, so the entry alone
  doesn't place its hops on chips.
- **The same on every chip.** The host builds the path table the same way for every chip, but the rest of the area
  holds the chip's own ids, so each chip still has its own copy.

In the Blackhole 1D snapshot, each of the 8 copies has 7 entries, as expected, and 249 zero ones.

## The names it reads

The meaning of these names is in decode's code (case 2 of "What decode reads, and what it knows" in the
[README](../README.md)). A renamed one fails with `ManifestError`, naming the member and the members there are:

- the `routing_table` area, and its struct's `state_manager` member;
- in a 2D run, `route_table_2d`, with its `action_vectors` and `mcast_trees`;
- in a 1D run, `routing_path_table_1d`, with its `paths`, and `fabric_context.routing_1d_extension_words`.

A table that can't hold its mesh fails with `ManifestError` too: action vectors or trees too short for the mesh's shape.

## What it doesn't cover

- **Judging the tables.** Whether a chip's copies agree is whether it has more than one. Whether a table matches what
  the host meant to write is left to the reader.
- **The direction tables and `exit_node_table`.** They're in `fields`, decoded, but not interpreted.
- **The table's own coordinates and mesh size** (`my_mesh_coord_*`, `mesh_y_size`, `mesh_x_size`). Chips' coordinates
  come from the manifest. The host sets the mesh size only in a 2D run.

## Files cited

| File | Symbols |
| --- | --- |
| `tt_metal/fabric/control_plane.cpp` | `ControlPlane::write_routing_info_to_devices`, `compute_and_embed_1d_routing_path_table` |
| `tt_metal/fabric/compressed_routing_path.cpp` | `intra_mesh_routing_path_t<1, false>::calculate_chip_to_all_routing_fields` |
| `tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h` | `RouterStateManager`, `routing_encoding::encode_1d_unicast`, `MAX_CHIPS_LOWLAT_1D` |
| `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp` | the router's writes to `state_manager->state` |
| `tests/tt_metal/tt_fabric/common/fabric_command_interface.cpp` | `FabricCommandInterface::issue_command_to_routers` |
| `tt_metal/fabric/hw/inc/fabric_2d_route_interface.h` | `widen_2d_route_to_chip` |
