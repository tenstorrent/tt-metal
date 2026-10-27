# Topology pass

The topology pass (`fabric_manifest_topology_pass.cpp`) builds the manifest's `meshes`: every mesh, every chip in it,
and each local chip's routers. It runs at write time, when the router builders are gone but ControlPlane, the mesh
graph and the cluster can say how everything connects. Its one entry point is `manifest::describe_meshes`, which
returns a `std::vector<Mesh>` for `ManifestFacts`.

Its inputs are ControlPlane (the mesh graph, which channels are active fabric routers and their routing planes, which
channels connect, which links cross hosts), the cluster (Ethernet cores), and the routers each chip published during
the build ([BUILDER_HOOKS.md](BUILDER_HOOKS.md), [ROUTER_PASS.md](ROUTER_PASS.md)).

## Meshes

Meshes are listed by mesh id. For each (`describe_mesh`):

| Key | From |
| --- | --- |
| `shape` | The mesh graph's shape for the mesh |
| `torus` | 2D meshes only: whether each axis (`y` for north-south, `x` for east-west) wraps. It uses fabric's own rule, `has_genuine_torus_axis`, with the fabric type from the fabric config, the same one the mesh graph used to connect the chips. An axis too short to form a distinct wrap edge doesn't wrap |
| `express_routing` | ControlPlane's `express_routing_enabled` for the mesh |
| `credit_transport` | Meshes on this host only: each VC's credit backing, from the builder context ([DESCRIBE.md](DESCRIBE.md)) |
| `chips` | Every chip in the mesh graph, in its order |

## Chips

Every chip in the mesh graph is described, local or not, so a reader can draw the whole fabric and see where each
host's part ends (`describe_chip`).

**Local and remote chips.** A chip is local when this process opened it, which is when ControlPlane maps it to a
physical chip id. A remote chip belongs to another host rank's process: this rank knows its place in the mesh, but
can't read its hardware. So a remote chip has only its `mesh_coord`, with `physical_chip_id` and `asic_id` null and
`is_local` false. The manifest its own rank writes describes it in full.

A local chip starts from what its `FabricBuilder` published: its routers, its Z port role and its local sync. It adds
its `mesh_coord`, its physical chip id, and its ASIC id from ControlPlane, then resolves its routers (below).

A local chip whose builder published nothing has no routers. That's only valid when ControlPlane lists no active fabric
channels for it. Otherwise resolving its routers fails, since ControlPlane expects a router on each active channel.

## Resolving a chip's routers

The builders know their routers by Ethernet channel. A reader wants routers by what they are: the router facing east on
routing plane 0, or the router on the other end of this link. Resolving (`resolve_chip_routers`) turns every channel
the builders recorded into the router it names, in four steps.

### 1. Identities

`resolve_router_identities` gives each router its key and cores:

- **Key**: its direction and routing plane, from ControlPlane's active fabric channels, written as one letter and the
  plane, such as `E0`. A router's key names it within its chip.
- **Cores**: its logical and virtual Ethernet core, from the cluster's SoC descriptor.

Routers are listed in the order ControlPlane lists the chip's active channels. The checks:

- every active channel has a published router, and every published router is on an active channel;
- no two published routers share a channel, and no two routers share a key.

### 2. Local sync master

The chip's `local_sync` records the master router's channel, from the builder's `KernelCreationContext`. It's resolved
to that router, which must exist.

### 3. Link peers

`resolve_link_peer` adds what only ControlPlane knows about each router's link:

- **`peer`**: the router on the other end. ControlPlane gives the chip and channel the link connects to. The peer is the
  active fabric router there, or null when the link connects to nothing, or to a channel that was trimmed from the
  peer's routing planes. A peer can be on another chip, another mesh, or another host.
- **`cross_host`**: whether the link crosses to another host, from ControlPlane.
- **`wrap`**: whether the link closes a torus axis. It wraps when the mesh is 2D, the peer is in the same mesh, the
  link's direction lies on an axis that wraps (as `torus` above decides), and the two chips are at opposite ends of
  that axis.

### 4. Edge siblings

Each downstream edge the router pass collected names its sibling by direction only. `resolve_edge_siblings` finds the
sibling router: the router facing that direction on the same chip and routing plane. Then:

- The edge's landing channel must be one of the sibling's sender channels on the edge's VC.
- **`through_tensix_mux`** is set when the core the edge writes to isn't the sibling's ERISC, which means it writes to
  the sibling's tensix mux.
- Only one edge may land on each sibling sender channel.

Each edge that lands on a sender channel, other than through a tensix mux, makes that sibling the channel's
**producer**. Two checks tie edges and producers together:

- A sender channel fed by the local worker or the tensix mux can't also have a sibling's edge land on it.
- A sender channel whose `static_connection` field is set waits at startup for a sibling to connect. Some sibling's
  edge must land on it.

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
2. Compute it here, from ControlPlane, the mesh graph or the cluster. If it's about a router, add it as a resolve step
   in `resolve_chip_routers`. If it's about a mesh or chip, add it in `describe_mesh` or `describe_chip`.
3. Write it in the writer (`fabric_manifest.cpp`), and add it to `fabric_manifest_schema.json` and
   [FORMAT.md](FORMAT.md).
4. Describe it in this doc, with any check it brings.
