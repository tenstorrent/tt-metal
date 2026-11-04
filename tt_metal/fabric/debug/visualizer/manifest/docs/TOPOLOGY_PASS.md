# Topology pass

The topology pass (`fabric_manifest_topology_pass.cpp`) builds the manifest's `meshes`: every mesh, every chip in it,
and each local chip's routers. Its entry point is `manifest::describe_meshes`.

It runs at write time. By then the router builders are gone, but what they published is kept, and ControlPlane, the
mesh graph and the cluster can say how everything connects. The topology pass joins the two.

## What it builds on

The router pass describes one router at a time, while its builder exists ([ROUTER_PASS.md](ROUTER_PASS.md)). It knows
each router by its Ethernet channel, and knows nothing about the rest of the chip or the fabric. Each chip's
`FabricBuilder` publishes those routers, with a few chip-wide facts, to `PublishedChips`
([BUILDER_HOOKS.md](BUILDER_HOOKS.md)).

The topology pass finishes what the router pass couldn't:

| The router pass leaves | The topology pass adds |
| --- | --- |
| Each router's Ethernet channel | Its key (direction and routing plane, such as `E0`) and its cores |
| Each link's edge capability, and whether it's a dispatch link | The `peer` router on the other end, `cross_host`, and `wrap` |
| Each downstream edge's direction and landing channel | The sibling router it lands on (`downstream_channel`), and whether it ends at a tensix mux |
| Producers for worker and tensix mux channels | Producers for channels a sibling feeds |
| The local sync master's Ethernet channel | The master router's path |
| The chip's Z port role, local sync, routers, and the router pairs its builder connected | The chip's place in its mesh, its ids, and whether it's local |

It also lays out what the router pass never sees: every mesh, and every chip, including chips on other hosts.

## Where each fact comes from

The topology pass reads each fact from the object that already knows it, through that object's own accessor. It
doesn't work out again something fabric has already decided, so it can't drift from what fabric did.

| Fact | Read from |
| --- | --- |
| The meshes, in mesh id order | `mesh_graph.get_all_mesh_ids()` |
| A mesh's `shape` | `mesh_graph.get_mesh_shape()` |
| A mesh's `torus` (2D only) | `has_genuine_torus_axis()`, fabric's own rule, with the fabric type from `get_fabric_type()`: the same rule and fabric type the mesh graph connected the chips with |
| A mesh's `express_routing` | `control_plane.express_routing_enabled()` |
| A mesh's `credit_transport` | The builder context's credit transport plan, for the meshes on this host (`get_local_mesh_id_bindings()`) ([DESCRIBE.md](DESCRIBE.md)) |
| A mesh's chips, in order | `mesh_graph.get_chip_ids()` |
| A chip's `mesh_coord` | `mesh_graph.chip_to_coordinate()` |
| Whether a chip is local, and its `physical_chip_id` | `control_plane.try_get_physical_chip_id_from_fabric_node_id()`: a chip this process opened has one |
| A chip's `asic_id` | `control_plane.get_asic_id_from_fabric_node_id()` |
| A chip's routers, and everything the router pass collected about them | What its `FabricBuilder` published (`builder_context.get_manifest_chips()`) |
| A router's direction and routing plane | `control_plane.get_active_fabric_eth_channels()` and `control_plane.get_routing_plane_id()` |
| A router's cores | The cluster's SoC descriptor (`get_eth_core_for_channel`) and `cluster.get_virtual_coordinate_from_logical_coordinates()` |
| A link's peer | `control_plane.try_get_connected_mesh_chip_chan_ids()`, then the active fabric router on that channel |
| A link's `cross_host` | `control_plane.is_cross_host_eth_link()` |
| An edge's sibling | The router pairs the builder connected (`FabricBuilder::get_router_connection_pairs()`, published with the chip) |
| An edge's `through_tensix_mux` | `builder_type_for_vc()`, the builder's rule for which builder owns a VC's senders, on the sibling's `has_tensix_extension` |

Two facts are the topology pass's own, because no object holds them directly. Each is built from the lookups above:

- **`wrap`**: a link closes a torus axis when the mesh is 2D, the peer is in the same mesh, the link's direction lies
  on an axis that wraps (`has_genuine_torus_axis()`), and the two chips are at opposite ends of that axis.
- **Sibling producers**: a sender channel's producer is the router whose edge lands on it. This reads the builder's
  edges backwards, so it adds no rule of its own.

## Chips

Every chip in the mesh graph is described, local or not, so a reader can draw the whole fabric and see where each
host's part ends (`describe_chip`).

- **A local chip** is one this process opened. It starts from what its `FabricBuilder` published, adds its
  `mesh_coord`, `physical_chip_id` and `asic_id`, and then has its routers resolved (below).
- **A remote chip** belongs to another host rank. This rank knows only its place in the mesh, so it has its
  `mesh_coord`, null ids, and `is_local` false. Its own rank's manifest describes it in full.

A local chip whose builder published nothing has no routers. That's only valid when ControlPlane lists no active
fabric channels for it; otherwise resolving its routers fails.

## Resolving a chip's routers

`resolve_chip_routers` turns every Ethernet channel the builders recorded into the router it names, in four steps.

**1. Identities** (`resolve_router_identities`). Each router gets its key and cores, and routers are listed in the
order ControlPlane lists the chip's active channels.

**2. Local sync master** (`get_master_router`). The master's channel, from the builder's `KernelCreationContext`,
becomes the master router.

**3. Link peers** (`resolve_link_peer`). Each router's link gets its `peer`, `cross_host` and `wrap`. The peer is null
when the link connects to nothing, or to a channel trimmed from the peer's routing planes. A peer can be on another
chip, another mesh, or another host.

**4. Edges and sibling producers** (`resolve_edges_through_tensix_mux`, then `resolve_sibling_producers`). The router
pass knows only the direction each downstream edge points in. The sibling is the router the builder paired with this
one that faces that direction. Reading the builder's pairs, rather than looking for any router facing that direction,
gives the routers the builder actually connected. That matters where it pairs routers that don't face opposite ways,
such as the corner fold in 1D. Each step writes only to the router it resolves, and reads its siblings:

- A router's edges get `through_tensix_mux` from `builder_type_for_vc()`: the tensix mux owns VC0 on a sibling with a
  tensix extension. An edge whose direction has no paired router isn't through the mux.
- A router's sender channel gets a sibling `producer` when a sibling's edge lands on it and doesn't end at this
  router's tensix mux. An edge landing past the router's sender channels on its VC feeds none.

## Checks

The topology pass checks only what it puts together itself, where the builders and ControlPlane could disagree:

- **Published routers match ControlPlane's active channels.** Every active fabric channel has a published router,
  every published router is on an active channel, and no two share a channel or a key.
- **The local sync master has a router.**

It doesn't check the builder's wiring against itself: that an edge's direction has a paired router, that it lands on a
sender channel the sibling has, that each sender channel has at most one edge, that a `static_connection` channel has
one, or that the worker channel has none. The edges come from the builder, so that's fabric's to test.

## Router paths

Wherever the manifest refers to another router, it writes the router's path, `M<mesh>/C<chip>/<key>`, such as
`M0/C7/E0`:

| Reference | Path |
| --- | --- |
| A link's `peer` | The peer router's path, on whatever chip, mesh or host it's on |
| `local_sync.master` | The master router's path |
| A sender channel's sibling `producer` | The sibling router's path |
| A downstream edge's `downstream_channel` | The sibling's path, then `channels/senders/vc<v>/ch<c>` for the channel it lands on |

A peer on another host is named by its path even though this rank's manifest has no routers for it. That host's
manifest does.

## Adding a fact that needs the topology

1. Add the fact to the model (`fabric_manifest_model.hpp`) where it sits.
2. Find the object that already knows it (ControlPlane, the mesh graph, the cluster, or what the builder published),
   and read it through that object's accessor. If the builder knows it but doesn't publish it, publish it from the
   builder ([BUILDER_HOOKS.md](BUILDER_HOOKS.md)) rather than working it out again here.
3. Read it in the right place: a router fact as a step in `resolve_chip_routers`, a mesh or chip fact in
   `describe_mesh` or `describe_chip`.
4. Write it in the writer (`fabric_manifest.cpp`), and add it to `fabric_manifest_schema.json` and
   [FORMAT.md](FORMAT.md).
5. Describe it in this doc, with any check it brings.
