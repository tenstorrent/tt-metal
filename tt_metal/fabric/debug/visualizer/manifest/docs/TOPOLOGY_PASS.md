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
| Each router's Ethernet channel and the direction its builder built it facing | Its routing plane, which makes its key (such as `E0`), and its cores |
| Each link's edge capability, and whether it's a dispatch link | The `peer` router on the other end, `cross_host`, and `wrap` |
| Producers for worker and tensix mux channels | Producers for channels a sibling feeds |
| The local sync master's Ethernet channel | The master router's path |
| The chip's Z port role, local sync and routers | The chip's place in its mesh, its ids, and whether it's local |

It also lays out what the router pass never sees: every mesh, and every chip, including chips on other hosts.

## Where each fact comes from

The topology pass reads each fact from the object that already knows it, through that object's own accessor. It
doesn't work out again something fabric has already decided, so it can't drift from what fabric did.

| Fact | Read from |
| --- | --- |
| The meshes, in mesh id order | `mesh_graph.get_all_mesh_ids()` |
| A mesh's `shape` | `mesh_graph.get_mesh_shape()` |
| A mesh's `torus` (2D only) | `mesh_graph.get_intra_mesh_connectivity()`: the mesh graph connects each mesh from its own fabric type, and gives the corner chip (0, 0) an N edge only when the mesh wraps rows and a W edge only when it wraps columns |
| A mesh's `express_routing` | `control_plane.express_routing_enabled()` |
| A mesh's `credit_transport` | The builder context's credit transport plan, for the meshes on this host (`get_local_mesh_id_bindings()`) ([DESCRIBE.md](DESCRIBE.md)) |
| A mesh's chips, in order | `mesh_graph.get_chip_ids()` |
| A chip's `mesh_coord` | `mesh_graph.chip_to_coordinate()` |
| A chip's `host_rank` | `mesh_graph.get_host_rank_for_chip()`: the host rank of its mesh that owns it, for local and remote chips alike |
| Whether a chip is local, and its `physical_chip_id` | `control_plane.try_get_physical_chip_id_from_fabric_node_id()`: a chip this process opened has one |
| A chip's `asic_id` | `control_plane.get_asic_id_from_fabric_node_id()` |
| A chip's routers, and everything the router pass collected about them | What its `FabricBuilder` published (`builder_context.get_manifest_chips()`) |
| A router's routing plane | `control_plane.get_routing_plane_id()` |
| A router's cores | The cluster's SoC descriptor (`get_eth_core_for_channel`) and `cluster.get_virtual_coordinate_from_logical_coordinates()` |
| A link's peer | `control_plane.try_get_connected_mesh_chip_chan_ids()`, then the router the peer's builder built on that channel: one of its routing planes in the direction (`get_active_fabric_eth_routing_planes_in_direction()`), which `FabricBuilder` builds routers on |
| A link's `cross_host` | `control_plane.is_cross_host_eth_link()` |

Two facts are the topology pass's own, because no object holds them directly. Each is built from the lookups above:

- **`wrap`**: a link closes a torus axis when the peer is in the same mesh, the link's direction lies on an axis the
  mesh's `torus` wraps, and the two chips are at opposite ends of that axis.
- **Sibling producers**: a sender channel's producer is the router whose edge lands on it. This reads the builder's
  edges backwards, so it adds no rule of its own.

## Chips

Every chip in the mesh graph is described, local or not, so a reader can draw the whole fabric and see where each
host's part ends (`describe_chip`).

- **A local chip** is one this process opened. It starts from what its `FabricBuilder` published, adds its
  `mesh_coord`, `host_rank`, `physical_chip_id` and `asic_id`, and then has its routers resolved (below).
- **A remote chip** belongs to another process. This rank knows only its place in the mesh and the host rank that owns
  it, so it has its `mesh_coord`, `host_rank`, null ids, and `is_local` false. Its own rank's manifest describes it in
  full. In a mesh split across hosts, the remote chips are the other host ranks' share.

A local chip whose builder published nothing has no routers.

## Resolving a chip's routers

`resolve_chip_routers` turns every Ethernet channel the builders recorded into the router it names, in four steps.

**1. Identities** (`resolve_router_identities`). Each router gets its routing plane, which with its direction makes its
key, and its cores. Routers stay in the order they were published, by Ethernet channel.

**2. Local sync master** (`get_master_router`). The master's channel, from the builder's `KernelCreationContext`,
becomes the master router.

**3. Link peers** (`resolve_link_peer`). Each router's link gets its `peer`, `cross_host` and `wrap`. The peer is null
when the link connects to nothing, or to a channel trimmed from the peer's routing planes. A peer can be on another
chip, another mesh, or another host.

**4. Sibling producers** (`resolve_sibling_producers`). Each downstream edge names the sibling it lands on, by its
Ethernet channel, from the connection the builder recorded. A router's sender channel gets a sibling `producer` when a
sibling's edge lands on it and doesn't end at this router's tensix mux. An edge landing past the router's sender
channels on its VC feeds none. Each router writes only its own channels, and reads its siblings' edges.

## Checks

The topology pass checks only what it puts together itself:

- **No two routers on a chip share a key.**
- **The local sync master has a router.**

It doesn't check the builder's wiring against itself: that an edge lands on a router that exists, that it lands on a
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
