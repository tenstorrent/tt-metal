# TT-Fabric: An Architecture Orientation

**Audience.** This report assumes you are comfortable with the host-local TT-Metalium
runtime: you know what a `Program` is, how kernels are placed on Tensix worker cores,
how the NOC moves data between cores on one chip, what L1 and semaphores are, and how a
command queue dispatches work to a single device. It assumes you know *nothing* about
multi-chip or multi-host systems.

**Relationship to existing docs.** The formal specification is
[`TT-Fabric-Architecture.md`](TT-Fabric-Architecture.md) in this same directory. That
document is the authority on protocol details, and this report does not attempt to
replace it. What it does not do is bridge from the single-device mental model, and it
predates a fair amount of the code that now exists. This report is the bridge: it
introduces the concepts in dependency order, defines every term before using it, and
points at the specific files where each concept is implemented so you can go read the
real thing.

---

## Table of contents

1. [The problem fabric solves](#1-the-problem-fabric-solves)
2. [Glossary](#2-glossary)
3. [The physical substrate: Ethernet cores and links](#3-the-physical-substrate-ethernet-cores-and-links)
4. [Naming the machine: meshes, nodes, and the control plane](#4-naming-the-machine-meshes-nodes-and-the-control-plane)
5. [The router: Erisc Data Mover](#5-the-router-erisc-data-mover)
6. [The packet](#6-the-packet)
7. [Routing](#7-routing)
8. [The worker's view: how a kernel sends to another chip](#8-the-workers-view-how-a-kernel-sends-to-another-chip)
9. [Bootstrap sequence](#9-bootstrap-sequence)
10. [Fabric and `distributed/`](#10-fabric-and-distributed)
11. [File map](#11-file-map)
12. [Suggested reading order](#12-suggested-reading-order)

---

## 1. The problem fabric solves

On a single chip, the NOC gives every core a flat, uniform way to reach every other
core. A dataflow kernel writes to a NOC address built from a destination core's
coordinates plus an offset, and the hardware delivers it. There is no routing decision
to make, no buffering to manage, and no possibility of the transfer failing partway.
`noc_async_write` is, from the kernel's point of view, a fire-and-forget store to a
slightly unusual address.

That model stops at the edge of the chip. A Tensix core on chip A has no NOC path to a
Tensix core on chip B. The only physical connection between chips is Ethernet, and
Ethernet is a point-to-point link between two specific *Ethernet cores*, one on each
chip. It connects neighbors, not arbitrary pairs. If chip A and chip C are not directly
cabled together, nothing in hardware will carry a write from A to C.

TT-Fabric is the software and firmware layer that closes this gap. Its goal, stated as
directly as possible:

> Make a write to a core on a remote chip look as much as possible like a write to a
> core on the local chip, for any pair of chips in the system, including chips that are
> not directly connected and chips managed by a different host process.

It achieves this by running a **router** kernel on every Ethernet core. Routers
cooperate to form a network. A worker kernel hands a **packet** to a local router; the
packet carries both a route (which chips to traverse) and a NOC command (what to do
once it arrives). Routers forward the packet hop by hop until it reaches the
destination chip, where the final router executes the embedded NOC command locally —
typically a `noc_async_write` into the destination core's L1.

Two consequences follow, and they explain most of the design:

- **Fabric is packetized, not connection-oriented.** There is no stream or session at
  the base layer. Each packet is independent and self-describing, and carries the full
  destination address. Flow control exists between adjacent routers, not end to end.
- **Fabric is an extension of the NOC, not a replacement.** A remote write is a NOC
  write that happens to have been carried across some Ethernet links first. The
  *intra*-chip half of every transfer is still ordinary NOC traffic.

From the spec's overview of layers 1–5, the whole thing is deliberately structured like
a network stack: Ethernet hardware at layers 1–2, TT-routing at layer 3, TT-transport
(virtual channels and flow control) at layer 4, and TT-session (the APIs kernels call)
at layer 5.

---

## 2. Glossary

Read this once now; the rest of the document uses these terms freely.

**Device / chip.** One AI accelerator ASIC — a Wormhole or a Blackhole. Everything you
know about the single-device runtime applies to one of these.

**Ethernet core (eth core).** A core on the chip that owns one Ethernet link. It is not
a Tensix core: it has no compute engines, a much smaller L1, and a different RISC
complement. It sits on the chip's NOC like any other core, so Tensix cores can read and
write its L1 over the NOC.

**ERISC.** The RISC-V processor inside an Ethernet core. Wormhole has one usable ERISC
per Ethernet core; Blackhole has two, which is why some Blackhole builder code splits
router duties across processors (see `erisc_datamover_builder.cpp`, which refers to
"ERISC0: Handle sender channels only").

**EDM — Erisc Data Mover.** The router kernel that runs on an ERISC. "EDM" and "fabric
router" are used interchangeably throughout the codebase. Source:
`tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp`.

**Link.** One physical Ethernet connection between an Ethernet core on chip A and an
Ethernet core on chip B. Chips typically have several links per direction.

**Hop.** One traversal of a link, i.e. one chip-to-chip step. A packet from chip 0 to
chip 3 along a line takes three hops.

**Mesh.** A group of chips that are fully and uniformly connected in a grid — every
chip has the same number of Ethernet links to each of its neighbors. Meshes are the
unit of *scale-up*. A T3000 is a 2×4 mesh; a Galaxy is 8×4.

**Torus / ring.** A mesh whose edges wrap around: the last chip in each row or column
is also cabled to the first. A 1D wrapped line is a **ring**; a 2D wrapped mesh is a
**torus**. Wrapping creates cycles, which creates deadlock risk (§7.4).

**Fabric node.** One chip's identity within the fabric network, expressed as a
`FabricNodeId` — a `{MeshId, chip_id}` pair, where `chip_id` is the row-major index of
the chip within its mesh. Defined in
`tt_metal/api/tt-metalium/experimental/fabric/fabric_types.hpp`. This is a *logical*
identity, distinct from the physical chip id the driver uses.

**Routing plane.** Chips have multiple links per direction (four on Wormhole, two on
Blackhole). Rather than load-balancing across them arbitrarily, fabric partitions them
into independent parallel networks: link 0 on every edge routes only to link 0 on other
edges, link 1 only to link 1, and so on. Each such independent network is a routing
plane. Traffic never crosses between planes, which keeps traffic patterns deterministic
and makes reasoning about congestion tractable. Clients can pick a plane per
transaction.

**Channel.** A ring buffer of packet-sized slots in a router's L1, used to stage packets
in or out. **Sender channels** hold packets on their way out over the Ethernet link;
**receiver channels** hold packets arriving from the link. Note that "sender" and
"receiver" are relative to the *link*, not the chip.

**Virtual channel (VC).** A logically independent flow through the routers, made of its
own set of channel buffers. Separating traffic into VCs prevents one class of traffic
from blocking another — fabric reserves a dedicated control VC so control messages never
queue behind bulk data.

**1D vs 2D fabric.** A *routing mode*, not a cabling description. In 1D, routers only
forward along one axis; a packet cannot turn from a row into a column. The system
becomes a set of independent lines (or rings). In 2D, packets can turn, and one fabric
network spans the whole mesh. You can run 1D fabric on physically 2D-cabled hardware.
The modes are enumerated in `FabricConfig` (`fabric_types.hpp`).

**Unicast / multicast.** *Chip*-unicast delivers to one destination chip; *chip*-multicast
delivers to a contiguous range of chips along a route, with routers both consuming and
forwarding as the packet passes. Do not confuse this with *NOC* unicast/multicast, which
concerns cores within the destination chip. A packet independently specifies both.

**Fabric mux.** A kernel running on a *Tensix* core that aggregates many worker
connections into a single router connection. Discussed in §8.3.

**Control plane.** The host-side software (class `ControlPlane`) that discovers the
physical system, assigns logical identities, computes routing tables, and writes them
into Ethernet cores over PCIe. Also the general term for the out-of-band path used to
configure and monitor fabric, as opposed to the **data plane** that carries workload
traffic.

---

## 3. The physical substrate: Ethernet cores and links

A useful way to picture a chip once fabric is running: the familiar Tensix grid is
unchanged, and around its edge sit Ethernet cores, each owning one cable to a neighbor.
Each of those cores is running a router kernel, occupying its ERISC and most of its L1.

Three properties of Ethernet cores drive the design:

**They are NOC-reachable.** A Tensix core can write into an Ethernet core's L1 with an
ordinary `noc_async_write`. This is how a worker injects a packet into fabric: it writes
the packet into a slot in the router's sender channel buffer. No special hardware path
is involved — the worker→router leg is plain NOC traffic.

**They are small and slow relative to Tensix.** The ERISC is a single scalar processor
and L1 on an Ethernet core is a fraction of a Tensix core's. Router buffer depth is
directly limited by this, which is why channel slot counts are modest (the spec's
examples use 8 sender slots and 16 receiver slots) and why the router's main loop is
performance-critical. The repo's own review rules call this out explicitly: in
`.github/instructions/fabric-distributed.instructions.md`, "heartbeat/telemetry
increments in the router main loop add per-iteration cost. Amortize by posting every N
iterations."

**The link itself is reliable.** Tenstorrent Ethernet controllers run in "TT-link mode,"
in which hardware adds a 16-byte header implementing a Go-Back-N ARQ protocol. Frames
corrupted in transit are retransmitted by hardware. The router therefore never has to
implement retransmission: per the spec, "any TT-routing payload that is accepted by
TT-link layer is guaranteed to be delivered to TT-routing layer on the receiver side."
This is a significant simplification versus a conventional network stack — fabric's
software layers handle routing and flow control, but not reliability.

---

## 4. Naming the machine: meshes, nodes, and the control plane

Before any packet can move, the system needs a consistent logical map of itself. Two
pieces of machinery produce it.

### 4.1 The Mesh Graph Descriptor

A **Mesh Graph Descriptor (MGD)** is a textproto file declaring the *intended* logical
topology of a system: how many meshes, what shape each is, what architecture, and how
many links to expect between neighbors. They live in
`tt_metal/fabric/mesh_graph_descriptors/`. The T3000 one is short enough to quote in
full:

```
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 2, 4 ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2  policy: RELAXED }
}
top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
```

That declares one mesh, id 0, of eight Wormhole chips in a 2×4 grid, managed by a single
host, with two links expected per neighbor pair. Larger systems get correspondingly
larger descriptors — the directory contains everything from a single `n150` up to
`9_quad_bh_galaxy_9x32x4_torus_graph_descriptor.textproto`.

The MGD is the declared intent. Separately, `physical_system_discovery.cpp` and
`physical_system_descriptor.cpp` walk the actual hardware to find out which chips and
links really exist and are healthy, and `topology_mapper.cpp` reconciles the two —
deciding which physical chip plays the role of each logical mesh coordinate.

What happens when reality does not match the descriptor is governed by
`FabricReliabilityMode`: `STRICT_SYSTEM_HEALTH_SETUP_MODE` treats any downed link as an
error, while `RELAXED_SYSTEM_HEALTH_SETUP_MODE` allows fabric to come up with fewer
routing planes than declared.

### 4.2 The control plane

`ControlPlane` (declared in
`tt_metal/api/tt-metalium/experimental/fabric/control_plane.hpp`, implemented in
`tt_metal/fabric/control_plane.cpp`) is the host-side owner of all of this. It is
reachable as a singleton through `MetalContext::instance().get_control_plane()`. Its
responsibilities, visible in its API:

- **Identity translation.** `get_fabric_node_id_from_physical_chip_id()` maps the
  driver's chip id to the logical `FabricNodeId`, and `get_physical_mesh_shape()`
  reports mesh dimensions.
- **Routing table computation and installation.**
  `configure_routing_tables_for_fabric_ethernet_channels()` computes routes;
  `write_routing_tables_to_all_chips()` pushes them into every Ethernet core's L1.
- **Link and plane bookkeeping.** `get_active_fabric_eth_channels_in_direction()`,
  `get_num_usable_routing_planes()`, `reserve_routing_planes()`. Note the last one:
  planes can be reserved (for instance, for dispatch traffic) and are then excluded from
  the pool available to workloads.
- **Route queries used by workload builders.** `get_forwarding_direction()`,
  `get_fabric_route()`, `get_forwarding_eth_chans_to_chip()`.

Critically, the control plane reaches devices over **PCIe**, not over fabric. The spec is
emphatic about why: "The control plane is persistent, always functional and does not rely
on a properly functioning Data Plane." You cannot bootstrap the network over the network.
This separation also means fabric can be monitored and repaired when the data plane is
wedged.

---

## 5. The router: Erisc Data Mover

The EDM is the heart of the system. Its design is documented unusually well in a long
comment block at the top of `fabric_erisc_router.cpp`; that comment is worth reading in
full, and this section summarizes it.

### 5.1 Channel structure

A 1D router has **two sender channels and one receiver channel**, and the split reflects
the two possible sources of outbound traffic:

- **Sender channel 0** — packets injected by workers on this chip.
- **Sender channel 1** — passthrough packets that arrived from an upstream router and
  are continuing onward.
- **Receiver channel** — packets arriving off the Ethernet link.

The router round-robins between the two sender channels when transmitting. When a packet
lands in the receiver channel, the router inspects its header and does one or both of:

- write it into local chip memory, if this chip is a destination (the NOC command in the
  header says exactly what to do); and/or
- forward it to the next router, if the packet is a unicast that has not arrived yet, or
  a multicast whose range extends beyond this chip.

A 2D router needs **four sender channels**, because outbound traffic on any one edge can
come from the local worker *or* from any of the three other edges (a packet turning from
north into east, for example).

Each channel is a ring of fixed-size buffers, one packet per buffer. Packets always
advance to the next logical buffer even if they do not fill the current one, wrapping at
the end — so short packets leave gaps, which the router does not transmit.

### 5.2 Flow control

Flow control is strictly **hop by hop**, between adjacent routers, using credits: a
sender only writes into a downstream buffer slot it knows to be free, and the downstream
router returns credits as it drains slots. There is no end-to-end acknowledgment. A
packet accepted by the first router will be delivered, but the sender learns nothing
about when.

On rings and toruses this is augmented with **bubble flow control**. A channel is an
*injection channel* if it accepts locally generated traffic, or traffic turning into
itself from another direction. Injection channels are required to leave a free slot: a
packet may only enter if two or more slots are free downstream. That reserved "bubble"
guarantees the cycle can always drain, which breaks the cyclic-dependency deadlock that
wrapped topologies would otherwise permit. §7.4 explains the underlying hazard.

### 5.3 Intra-chip hops

One subtlety that surprises people: not every hop is an Ethernet hop. A packet arriving
on the north edge of a chip and leaving on the east edge lands in the *north* Ethernet
core's receiver channel and must reach the *east* Ethernet core's sender channel. Those
are two different cores on the same chip, so the transfer between them is an ordinary
NOC write. The spec diagrams this case explicitly under "Dataflow between two fabric
routers over NOC." Every 2D turn costs a NOC traversal in addition to the two Ethernet
traversals.

---

## 6. The packet

A fabric packet is a **header** followed by a **payload**. Everything about the packet's
journey and its final effect is encoded in the header, which is defined in
`tt_metal/fabric/fabric_edm_packet_header.hpp`.

### 6.1 Two orthogonal questions

Every header answers two independent questions, and keeping them separate is the key to
reading this code:

**Where does it go?** — the *chip*-level routing. Encoded by `ChipSendType`
(`CHIP_UNICAST` or `CHIP_MULTICAST`) plus routing fields whose layout depends on
topology (§6.2).

**What happens when it arrives?** — the *NOC*-level command, encoded by `NocSendType`
and an accompanying command-field union. The full set:

| `NocSendType` | Effect at the destination chip |
| --- | --- |
| `NOC_UNICAST_WRITE` | Write the payload to one NOC address |
| `NOC_UNICAST_INLINE_WRITE` | Write a small immediate value carried in the header itself |
| `NOC_UNICAST_ATOMIC_INC` | Atomically increment a location (this is how remote semaphores work) |
| `NOC_FUSED_UNICAST_ATOMIC_INC` | Write, then atomically increment — one packet instead of two |
| `NOC_UNICAST_SCATTER_WRITE` | Write to up to four addresses on the destination chip |
| `NOC_UNICAST_READ` | Read from the destination |
| `NOC_MULTICAST_WRITE` / `NOC_MULTICAST_ATOMIC_INC` | NOC-multicast to a core rectangle on the destination chip |
| `NOC_SPARSE_MCAST_WRITE` | Chip-multicast to non-contiguous chips, each with its own destination address (1D only) |

The fused write-plus-increment is the workhorse for collectives: a producer writes data
to a remote consumer and bumps the consumer's semaphore in a single packet, which is
exactly the remote analogue of the local write-then-signal idiom.

### 6.2 Header variants

There is no single header type. The concrete type is selected at kernel compile time by
the `ROUTING_MODE` define, resolving to the macro `PACKET_HEADER_TYPE` at the bottom of
`fabric_edm_packet_header.hpp`. The variants:

- **`LowLatencyPacketHeaderT<ExtensionWords>`** — 1D. The routing field is a
  pre-computed bit-encoded sequence of per-hop actions, so the router does not decode a
  hop count or consult a table; it reads the next action directly. `ExtensionWords = 0`
  yields a 48-byte header supporting 16 hops; 1 through 3 yield 64-byte headers
  supporting 32, 48, and 64 hops. A static assert blocks going further: "1D routing with
  >64 hops (ExtensionWords > 3) requires memory map updates."
- **`HybridMeshPacketHeaderT<RouteBufferSize>`** — 2D. Carries `dst_start_chip_id` and
  `dst_start_mesh_id` (the `FabricNodeId` of the destination), a four-entry
  `mcast_params` array giving multicast hop counts per direction, and a route buffer of
  action-map bytes. Sizes range from 80 to 128 bytes depending on route buffer capacity.
- **`PacketHeader`** — the older 1D format, with a one-byte `RoutingFields` packing a
  4-bit start distance and 4-bit hop range. Still present; superseded by the low-latency
  form.
- **`UDMHybridMeshPacketHeader`** — the 2D header plus UDM control fields (§8.4).

The practical implication: header size is a compile-time constant that varies by mode,
and it is subtracted from every payload. `get_tt_fabric_max_payload_size_bytes()` exists
precisely so host code does not hardcode it.

---

## 7. Routing

### 7.1 Source routing

Within a mesh, fabric uses **source routing**: the sending worker computes the entire
route and writes it into the packet header. Routers do not make independent routing
decisions for intra-mesh traffic; they read the next action out of the header and
execute it. The spec's justification is that "a worker is always able to fully determine
the route to any other device in the mesh," so there is no reason to pay for a table
lookup at every hop.

Routers *do* hold fully instantiated routing tables, and those tables are what the source
consults (and what mesh entry points use for inter-mesh traffic, §7.3). Two levels:

- **L0 / intra-mesh** — indexed by destination chip id within the mesh, yielding a source
  route. This is scale-up.
- **L1 / inter-mesh** — indexed by destination mesh id, yielding an *exit node*: a chip in
  the current mesh that has a link toward that mesh. This is scale-out.

### 7.2 1D routing

In 1D there is nothing to decide beyond distance. The API takes `num_hops` directly:

```cpp
packet_header->to_chip_unicast(num_hops);
```

Each router decrements toward zero (in the low-latency encoding, consumes one action
entry) and delivers when it reaches the destination. Multicast is a start distance plus a
range: begin delivering at hop *N*, keep delivering and forwarding for *R* more hops.

`NOC_SPARSE_MCAST_WRITE` generalizes this with a bitmask, documented inline:

```
// Each bit represents a single hop in the target direction
// The router will WRITE AND FORWARD at hops set to 1 and FORWARD ONLY at unset hops.
// ...
// 0 --> 1 --> 2 --> 3 --> 4 --- 5
//      [X]               [X]
// We would set a hop mask of 0b01001
```

### 7.3 2D routing

In 2D the destination is a `FabricNodeId`, and routes follow **dimension-ordered
routing**: traverse the X dimension to completion, then Y. So from chip 0 to chip 8 in a
3×3 mesh the route is `EESS` — east, east, south, south. The spec's L0 table for a 3×3
mesh shows this pattern for every source-destination pair.

Dimension ordering is not an arbitrary convention; it is the deadlock-avoidance mechanism
(§7.4).

Inter-mesh routing composes the two tables. The sending worker sets a route to its local
mesh's exit node. When the packet crosses a mesh boundary, the entry-node router in the
next mesh *rewrites* the source route in the header — either a path to the final
destination if this is the destination mesh, or a path to another exit node if not. This
is the one place where a router recomputes routing rather than following the header, and
you can see it guarded in the router kernel:

```cpp
#ifdef FABRIC_2D
// The router re-encodes an intermesh packet's route when it lands, via
// fabric_set_2d_intermesh_landing_route().
#endif
```

Scale limits: up to 1024 meshes of 256 devices each, roughly 250,000 devices.

### 7.4 Deadlock

This is the one genuinely unfamiliar hazard if your background is single-device, and it
is worth understanding because it explains several otherwise arbitrary-looking design
choices.

Routers have finite buffers. A router cannot forward a packet until the next router has a
free slot. If router A is waiting on B, B on C, C on D, and D on A, nothing can ever
progress — every router is full and every router is waiting on a full neighbor. No packet
is lost and no error is raised; the system simply stops. This is **cyclic dependency
deadlock**.

Fabric prevents it in layers:

- **Dimension-ordered routing** eliminates cycles in a plain mesh. Because every route
  goes X-then-Y and never Y-then-X, the dependency graph among buffers is acyclic by
  construction. This is why the routing order is fixed rather than shortest-path.
- **Bubble flow control** handles rings and toruses, where wraparound reintroduces cycles
  that dimension ordering cannot remove. Reserving a free slot on injection guarantees
  drain capacity.
- **A separate control virtual channel** keeps control messages from queueing behind
  blocked data.
- **Timeouts** are the backstop on the roadmap: if a packet head cannot make progress
  within an interval, the router reports to the control plane and optionally drops and
  drains.

The repo's review rules reflect how easy this is to get subtly wrong: "ring topology does
not imply torus. 1D ring is a distinct routing constraint from 2D mesh."

---

## 8. The worker's view: how a kernel sends to another chip

This section is the one that most directly maps onto what you already know.

### 8.1 The connection

A router sender channel can serve **exactly one worker at a time**. Before sending, a
worker must *connect*, and the protocol is spelled out in the router kernel comment:

> Worker:
> - Read from EDM sender channel buffer_index address
>   - Required so that the worker knows where to write its first packet (since the
>     channel may already contain packets from a previous connection)
> - Write worker core X/Y (NOC 0 based)
> - Write worker flow control semaphore L1 address

The router then observes its connection semaphore go valid and assumes the other fields
are populated. Teardown is symmetric and mandatory: the worker writes `0` to the
connection semaphore, after which another kernel may connect. The comment is blunt about
the failure modes — if two workers connect to the same channel simultaneously, or a
worker pushes to a channel it has not connected to, "behavior is undefined. Undefined ==
likely hang."

You rarely implement this by hand. `WorkerToFabricEdmSender` in
`tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp` wraps the protocol
behind `open()` / `close()` and exposes the send primitives.

### 8.2 The send path

The device-side APIs live under `tt_metal/fabric/hw/inc/<topology>/api.h` —
`linear/api.h` for 1D, `mesh/api.h` for 2D. Here is the entire 1D unicast write:

```cpp
template <typename FabricSenderType>
FORCE_INLINE void fabric_unicast_noc_unicast_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header,
    uint8_t num_hops) {
    packet_header->to_chip_unicast(num_hops);
    packet_header->to_noc_unicast_write(noc_unicast_command_header, size);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(src_addr, size);
    client_interface->send_payload_flush_non_blocking_from_address(
        (uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}
```

Five lines, and each one maps to a concept from earlier sections. Set the chip-level
route. Set the NOC-level command. Wait for a free slot in the router's channel (this is
the credit-based flow control of §5.2, and it is the only place this call can block).
Push the payload. Push the header — last, because the header's arrival is what makes the
slot valid to the router.

Note what the "send" calls actually are: NOC writes into the Ethernet core's L1. There is
no new mechanism here, which is the payoff of the design.

Payload and header being separate also explains the `_set_state` / `_with_state` API
variants that dominate both headers by line count. A kernel streaming many packets to the
same destination sets the invariant header fields once, then per packet updates only the
fields selected by a compile-time `UpdateMask`. Same idea as the stateful NOC APIs you
already use, applied to packet headers.

The 2D path differs only in route construction: instead of `num_hops` it takes a
destination `FabricNodeId` and calls `fabric_set_unicast_route(packet_header, dst_dev_id,
dst_mesh_id)`, or `fabric_set_mcast_route(...)` with per-direction hop counts for
multicast.

**Portability note, from the repo's review rules.** Prefer the topology APIs in
`linear/api.h` and `mesh/api.h` over the legacy private APIs in `tt_fabric_api.h`; the
latter "are not portable to 2D fabric topologies and are slated for removal." A useful
property that follows: collectives written against the 1D linear API build and run
unchanged on a 2D mesh fabric.

### 8.3 Host-side wiring, and the mux

A worker kernel needs to know where its router is, what addresses to handshake on, and
which semaphores to use. Host code supplies all of this as runtime args via
`append_fabric_connection_rt_args()` (declared in
`tt_metal/api/tt-metalium/experimental/fabric/fabric.hpp`), which takes source and
destination `FabricNodeId`s and a link index and appends everything the device-side
adapter needs to reconstruct itself. `get_forwarding_link_indices()` tells you which link
indices are valid for a given chip pair. For multi-route workers there is
`append_routing_plane_connection_manager_rt_args()`, which wires up several routes at
once. These are the functions every TTNN collective calls — see
`ttnn/cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.cpp` and the various
`all_gather` / `all_to_all` program factories.

The one-worker-per-sender-channel rule is a real constraint: a chip has far more Tensix
cores that might want to send than it has Ethernet cores to send through. The **fabric
mux** resolves this. It is a kernel on a Tensix core that presents many logical channels
to many workers, multiplexes them onto a single router connection, and is configured by
`FabricMuxConfig` (V1) or `FabricMuxV2Config` (V2, which splits a forwarder onto RISCV_0
and a manager onto RISCV_1 and auto-terminates when clients disconnect). Enabled through
`FabricTensixConfig::MUX`. Mux channels come in two flavors — full-size channels carrying
header plus payload, and header-only channels for returning credits, which are cheaper
because they need no payload buffer.

### 8.4 UDM

`FabricUDMMode` (Unified DataMovement) is a newer 2D-only mode that builds both mux and
relay kernels as a Tensix extension and adds `UDMControlFields` to the packet header,
including the source's first-hop direction. It is worth knowing the name exists; the
constraints are enforced by static asserts in the packet header file, which reject 1D and
reject express routing with an explanation of why ("the mux fabric is cardinal-only").

---

## 9. Bootstrap sequence

Fabric bring-up is two layers: **host objects** that name the machine and compute routes,
then **device firmware** that actually runs the routers. Both the old `MetalContext`
singleton path and the new `MetalEnv` path run the same host objects; they differ in who
owns them and when they are created.

### 9.1 Who owns what

`MetalEnv` is the RAII owner of cluster, HAL, runtime options, fabric config,
`ControlPlane`, and `SystemMesh`. `MetalContext` is a session around one env: it owns
`DeviceManager` and dispatch managers, and today still forwards fabric calls into the env
(`MetalContext::get_control_plane()` is a one-line delegate). `SetFabricConfig()` still
goes through `MetalContext::instance()`, which lazily constructs a **default env +
context** if you never created a `MetalEnv` yourself.

So:

- **Legacy.** `SetFabricConfig(...)` then `MeshDevice::create(...)`. First
  `MetalContext::instance()` allocates a default `MetalEnv` and owns it
  (`env_owned_ = true`). Fabric config is mutable on that env until devices open.
- **MetalEnv-first.** Construct `MetalEnv(MetalEnvDescriptor{..., FabricConfigDescriptor{FABRIC_2D}})`
  then `env.create_mesh_device(...)`. Config is part of env construction. Creating a
  mesh device registers a `MetalContext` bound to that env
  (`ensure_context_registered`). Only one silicon context is allowed at a time.

`ControlPlane` is still lazy: it is not built in the `MetalEnv` constructor. It is built
on first `get_control_plane()` / `get_system_mesh()`, or when `set_fabric_config` finds
an existing plane and has to rebuild it.

### 9.2 Host object order (and why)

Each step depends on the previous:

1. **`RunTimeOptions` → `Cluster` → `Hal`.** `MetalEnvImpl::initialize_base_objects()`.
   Cluster talks to UMD/PCIe. HAL needs architecture and whether base-routing eth firmware
   is present.
2. **`DistributedContext` (MPI world).** Copied in at env construction so later topology
   mapping can run a multi-host collective.
3. **Fabric config state.** From the env descriptor, or from `SetFabricConfig`. This is
   just host-side knobs until step 5. Changing a non-DISABLED config while devices are
   open is rejected: it would rebuild the control plane under live routers.
4. **`ControlPlane`.** Loads the MGD (or auto-discovers on a single host), runs physical
   system discovery, maps logical `FabricNodeId`s onto physical chips (`TopologyMapper`),
   splits per-mesh MPI sub-contexts, **all-gathers fabric config across ranks**, then
   builds `RoutingTableGenerator` and inter-mesh exit nodes. Constructor also creates
   `FabricContext` (packet/header sizes, builder context). Depends on Cluster, HAL,
   DistributedContext, and the chosen `FabricConfig`.
5. **`initialize_fabric_config()`.** Reserves Ethernet cores for routers
   (`Cluster::configure_ethernet_cores_for_fabric_routers`) and fills in routing tables
   in the control plane. Must happen **before** devices are constructed so default
   sub-device state sees a stable eth-core reservation. A no-op if fabric is DISABLED.
6. **`SystemMesh`.** Virtual coordinates over the mapped chips. Depends on ControlPlane.
7. **Open devices.** `DeviceManager::open_devices`. If fast dispatch needs remote chips
   and the user never enabled fabric, this **implicitly** sets `FABRIC_1D` with one
   routing plane — dispatch-only fabric. Then tensix datamover config is initialized
   (mux/relay L1 maps), after devices exist so SOC descriptors are available.
8. **Dispatch kernel *topology* only.** `populate_fd_kernels_only` claims dispatch cores
   before fabric firmware launches, so mux placement and dispatch placement do not
   collide. Firmware for both fabric and dispatch is **not** launched here in the
   MeshDevice path.

Routing tables exist in host memory after step 5. They are not on device until firmware
init writes them over PCIe.

### 9.3 Firmware launch

`MeshDevice` opens devices with `initialize_fabric_and_dispatch_fw = false`, barriers
across host ranks, then calls `DeviceManager::initialize_fabric_and_dispatch_fw()`. The
barrier is required so both ends of a cross-host Ethernet link exist before handshake.

`FabricFirmwareInitializer::init` then, if `FabricManagerMode` includes `INIT_FABRIC`:

1. `ControlPlane::write_routing_tables_to_all_chips()` — PCIe write into every eth core.
2. Per device: `compile_fabric()` → `FabricBuilder`
   (`discover_channels` → `create_routers` → `connect_routers` → ancillary mux/relay
   kernels → `create_kernels`). This is an ordinary persistent `Program` on Ethernet
   cores.
3. `configure_fabric()` — zero router scratch, write runtime args, configure the program.

`configure()` waits until every master router reports `READY_FOR_TRAFFIC`
(`wait_for_fabric_router_sync`). Router postcodes are `EDMStatus` in
`fabric_edm_packet_header.hpp`; `fabric_firmware_initializer.cpp` maps a stuck postcode
to a diagnostic (never started vs eth handshake vs intra-chip handshake).

Dispatch firmware is initialized **after** fabric compile/configure, then fabric
`configure()` (the sync wait) runs last. Teardown is the reverse: dispatch first, then
fabric (host writes a termination signal into the master router).

### 9.4 Runtime dependencies (short)

```
Cluster/HAL ─► ControlPlane ─► routing tables (host)
                    │
                    ├─► SystemMesh / MeshDeviceView (FabricNodeId per coord)
                    │
                    └─► FabricBuilder / mux maps ─► ERISC router Program
                              │
Ethernet cores reserved ──────┤
                              ▼
              write tables + launch routers ─► eth handshake ─► READY_FOR_TRAFFIC
                              │
Worker kernels ── append_fabric_connection_rt_args ──► WorkerToFabricEdmSender
```

Workers cannot connect until routers are `READY_FOR_TRAFFIC`. Collectives compiled
earlier only stash connection runtime args; they do not talk to fabric until enqueue.

---

## 10. Fabric and `distributed/`

This is the relationship the rest of the report has been building toward. The short
version:

> **`tt_metal/fabric/` is the network. `tt_metal/distributed/` is the programming model
> that assumes the network exists.** Fabric moves bytes between chips. `distributed/`
> lets you treat many chips as one device. They meet at exactly two points: `distributed/`
> asks the control plane to bring fabric up, and the kernels dispatched through
> `distributed/` use fabric APIs to talk to each other.

### 10.1 What `distributed/` provides

`tt_metal/distributed/` is the multi-device runtime. Its core abstractions mirror the
single-device ones you know:

| Single device | Mesh |
| --- | --- |
| `IDevice` | `MeshDevice` |
| `Program` | `MeshWorkload` |
| `CommandQueue` | `MeshCommandQueue` |
| `Buffer` | `MeshBuffer` |
| `Event` | `MeshEvent` |
| trace | `MeshTrace` |

A **`MeshDevice`** is a virtual device over a rectangular set of chips. As the
TT-Distributed architecture report defines it, a *Virtual Mesh* is "a user-defined fully
connected mesh of TT-Devices, of arbitrary shape, with one or more host machines," carved
out of the *Physical Mesh* of everything that is actually cabled together. The design goal
is explicit: "The API contract for the MeshDevice class must be identical to the Device
class." Allocation happens in lockstep across all chips in the mesh, so a `MeshBuffer` has
the same address everywhere.

### 10.2 The two contact points

**Contact point one: bring-up.** `distributed/` does not program Ethernet cores itself.
It calls into the control plane for identity (`system_mesh.cpp` builds its coordinate
translation map from `ControlPlane`, and `MeshDeviceView` stores a `FabricNodeId` per
coordinate) and it triggers fabric initialization at mesh creation via
`initialize_fabric_and_dispatch_fw()`. Which fabric mode comes up was decided earlier by
the application's `SetFabricConfig()` call. Grep the directory and you will find fabric
references are almost entirely `FabricNodeId` and `ControlPlane` — identity and
initialization, not data movement.

**Contact point two: kernels.** When a `MeshWorkload` contains kernels that communicate
across chips — a TTNN all-gather, say — those kernels call the fabric device APIs of §8.2,
and the *host* side of the op called `append_fabric_connection_rt_args()` to wire them up.
`distributed/` dispatched the program; fabric carries its traffic. Neither layer knows much
about the other.

### 10.3 Three planes, not two

Multi-host is where the separation becomes sharpest, and where the naming can mislead. In
a multi-host system there are three distinct communication paths:

**Host-to-device: PCIe.** How a host talks to its local chips. Command queues, buffer
reads and writes, and control-plane configuration all ride this. Unchanged from
single-device.

**Host-to-host: MPI.** Multi-host TT-Metalium runs SPMD — one process per host, executing
the same program. `DistributedContext`
(`tt_metal/api/tt-metalium/distributed_context.hpp`) is the abstraction, and it is a
conventional MPI-shaped interface: `barrier`, `send`, `recv`, `broadcast`, `all_reduce`,
`Rank`. The MultiHostMeshRuntime report is emphatic that this is *not* a data path:

> It is critical to understand that the coordination layer is **not** intended for bulk
> data movement between host processes or between hosts and devices. All substantial data
> movement, including device-to-device transfers and collective communications across the
> logical mesh, is assumed to occur over the **underlying TT-fabric**.

MPI keeps the host processes in lockstep so they agree on what the global mesh looks like
and what work is being defined. Then each host dispatches its own slice.

**Device-to-device: fabric.** Every byte of tensor data that moves between chips — whether
those chips share a host or not. Fabric does not care about host boundaries; a cross-host
Ethernet link is just a link (`ControlPlane::is_cross_host_eth_link()` exists mainly so
bring-up can sequence the handshake correctly).

The confusing part is that `DistributedContext::all_reduce` and TTNN's `all_reduce` share
a name and share nothing else. The first reduces small host-side values across MPI ranks;
the second reduces tensors across chips over fabric. Keeping the three planes distinct
resolves most confusion about which layer is responsible for what.

### 10.4 Where they blend: `MeshSocket`

`tt_metal/distributed/mesh_socket.cpp` is the clearest place the layers meet. A socket is
a flow-controlled point-to-point connection between mesh coordinates, and it is exactly
the "sockets over TT-fabric" concept from the spec: send/receive built on fabric's
asynchronous write APIs plus explicit flow-control messages, giving you synchronous
semantics that the base packet layer deliberately does not provide. The socket is set up
through `distributed/` (mesh coordinates, host-side descriptor exchange, serialization for
the peer) and carries data over fabric (it resolves both endpoints to `FabricNodeId`s via
`generate_fabric_node_id_map`). Host-to-device and device-to-host variants
(`h2d_socket.cpp`, `d2h_socket.cpp`) round out the set.

### 10.5 TTNN on top

One layer up, TTNN's collectives (`ttnn/cpp/ttnn/operations/ccl/` — `all_gather`,
`reduce_scatter`, `all_broadcast`, `all_to_all_dispatch`, `broadcast`) are the main
consumers. Each is a normal TTNN op whose program factory places worker kernels and calls
`append_fabric_connection_rt_args()` or
`append_routing_plane_connection_manager_rt_args()` to attach them to fabric. From a model
author's perspective this is invisible: you call `ttnn.all_gather` on a tensor sharded
across a `MeshDevice`, and fabric is what makes it work.

---

## 11. File map

### Host-side, `tt_metal/fabric/`

| File | Role |
| --- | --- |
| `../impl/context/metal_env.cpp` | Owns cluster/HAL, fabric config, ControlPlane, SystemMesh |
| `../impl/context/metal_context.cpp` | Session over an env; DeviceManager; still the `SetFabricConfig` entry |
| `../impl/device/firmware/fabric_firmware_initializer.cpp` | Write tables, compile/launch routers, handshake wait, teardown |
| `control_plane.cpp` | System-wide owner: identity, routing tables, link state |
| `mesh_graph.cpp`, `mesh_graph_descriptor.cpp` | Parse and represent the declared logical topology |
| `physical_system_discovery.cpp`, `physical_system_descriptor.cpp` | Discover actual hardware |
| `topology_mapper.cpp`, `topology_solver*.cpp` | Reconcile declared topology against physical reality |
| `routing_table_generator.cpp`, `routing_2d_table_builder.hpp` | Compute L0/L1 routing tables |
| `fabric_init.cpp` | `create_and_compile_fabric_program()`, `configure_fabric_cores()` |
| `fabric_builder.cpp` | Phase-ordered construction of the per-device fabric program |
| `erisc_datamover_builder.cpp` | Per-ERISC router configuration |
| `fabric_tensix_builder*.cpp` | Mux / relay kernels on Tensix cores |
| `fabric_mux_config.cpp`, `fabric_mux_v2_config.cpp` | Mux memory maps and kernel args |
| `fabric_context.cpp` | Runtime fabric parameters (header size, payload size, buffer size) |
| `fabric_edm_packet_header.hpp` | Packet header definitions — shared host/device |
| `mesh_graph_descriptors/*.textproto` | Topology declarations per system |

### Device-side, `tt_metal/fabric/hw/inc/` and `impl/kernels/`

| File | Role |
| --- | --- |
| `impl/kernels/edm_fabric/fabric_erisc_router.cpp` | The router kernel; read its header comment first |
| `impl/kernels/tt_fabric_mux*.cpp` | Mux kernels |
| `hw/inc/linear/api.h` | 1D worker-facing send APIs |
| `hw/inc/mesh/api.h` | 2D worker-facing send APIs |
| `hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp` | `WorkerToFabricEdmSender` — connection protocol |
| `hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp` | Channel buffer management |
| `hw/inc/edm_fabric/fabric_router_flow_control.hpp` | Credit-based flow control |
| `hw/inc/tt_fabric_api.h` | Legacy API — prefer the topology headers |
| `hw/inc/udm/` | Unified DataMovement |

### Public API, `tt_metal/api/tt-metalium/experimental/fabric/`

`fabric.hpp` (`SetFabricConfig`, `append_fabric_connection_rt_args`,
`get_forwarding_link_indices`, mux configs), `fabric_types.hpp` (`FabricConfig`,
`FabricNodeId`, `MeshId`, reliability and topology enums), `control_plane.hpp`,
`fabric_telemetry.hpp`.

---

## 12. Suggested reading order

If you want to go deeper, this order builds each concept on the previous one:

1. **`fabric_erisc_router.cpp`, lines 1–200.** The design comment. Channel structure, the
   connection protocol, and the buffer model, with diagrams. The single highest-value
   thing to read.
2. **`fabric_edm_packet_header.hpp`.** Start at the `NocSendType` enum, then
   `PacketHeaderBase`, then the `LowLatencyPacketHeaderT` and `HybridMeshPacketHeaderT`
   variants, then the `PACKET_HEADER_TYPE` macro at the bottom.
3. **`hw/inc/linear/api.h`, first 130 lines.** The 1D unicast write end to end. Short, and
   it closes the loop between header construction and the NOC.
4. **`fabric_init.cpp` and §9.** Builder phases plus env/context bootstrap order.
5. **`ttnn/cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.cpp`.** A real consumer —
   how an actual collective attaches its workers to fabric.
6. **[`TT-Fabric-Architecture.md`](TT-Fabric-Architecture.md).** The formal spec, now that
   you have the vocabulary. Especially §2.2 (routing tables), §3 (router and topology
   diagrams), and §6 (deadlock).
7. **[`../TT-Distributed/MultiHostMeshRuntime.md`](../TT-Distributed/MultiHostMeshRuntime.md).**
   The host-coordination side of §10.
8. **`.github/instructions/fabric-distributed.instructions.md`.** The PR review rules. A
   compact list of the mistakes people actually make in this subsystem, which is a good
   proxy for where the sharp edges are.
