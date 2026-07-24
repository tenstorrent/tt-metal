# FabricBuilder ↔ Routing-Configuration Contract

Companion contract for what FabricBuilder (and closely related build-time owners) need from the
routing configuration / ControlPlane under the **cardinal N/S + Z express-link** design.

Authority split:

- `GALAXY_CARDINAL_NS_Z_SKIP_ROUTING_ASSESSMENT.md` owns topology/system assumptions, command
  semantics, the canonical route oracle and routing-generation policy, the CDG/SCC and VC0 BFC
  proof, system invariants, and Appendix A validation oracles.
- `GALAXY_CONTROL_PLANE_ROUTING_GENERATION_CONTRACT.md` owns detailed MGD/MeshGraph express
  materialization, ring synthesis, canonical-route generation, and the ControlPlane-owned facts
  consumed through this document's §4 surface.
- **This document** owns the CP ↔ builder query surface, builder-local effect derivation, router
  wiring and allocation, and BFC compile-time arguments.
- The codec and kernel contracts own device-visible bytes and ERISC realization respectively; this
  document supplies their routing-relevant builder bindings rather than restating either contract.

Related:

- `GALAXY_WORKING_MODEL.md` — physical/logical context and current software stack baseline
- `GALAXY_CARDINAL_NS_Z_SKIP_ROUTING_ASSESSMENT.md` — central routing oracle/proof/generation design,
  “Routing configuration and consumer boundary,” and Appendix A validation
- `GALAXY_CONTROL_PLANE_ROUTING_GENERATION_CONTRACT.md` — detailed pre-builder graph, synthesis,
  route-generation, and ControlPlane contract
- `GALAXY_DEVICE_ROUTE_CODEC_CONTRACT.md` — L1/device artifacts and packet/header ABI
- `GALAXY_DEVICE_ROUTER_KERNEL_CONTRACT.md` — ERISC decode/admit/forward realization

---

## 0. Scope and terminology

**§4 is normative.** §2 records the current baseline, §3 states builder realization deltas, §5 is
the remaining-requirements/checklist section, and §6 contains non-normative `[32,4]` examples.
Exact C++ signatures, packet bytes, kernel execution, and the arbitrary-pattern VC1 safety proof are
outside this builder contract; the open portion is broad multi-mesh support, not the retained current
two-link, predominantly linear envelope.

### 0.1 Terms

| Prefer | Avoid (in builder/API text) | Meaning |
|---|---|---|
| **express** | skip | Intramesh non-nearest-neighbor chord; command **Z** |
| **cardinal** | base *(for edge capability / commands)* | Ordinary geometric N/S/E/W edge or command |
| **intermesh** | “Z means boundary” | Cross-mesh edge capability (any command letter) |
| **source multi-inject** | root RX fanout | Worker sends one canonical-map copy per connected root output; required for express multicast, which is out of V1 scope |

Builder edge capabilities:

```text
INTRAMESH_CARDINAL   // N/S/E/W ordinary intramesh edge
INTRAMESH_EXPRESS    // Z express chord
INTERMESH            // mesh-boundary edge (direction independent)
```

Cardinal capability does not imply protected-domain membership. Qualify cardinal edges by axis when
needed; proof-only terms such as Y-NN ring remain owned by the assessment.

---

## 1. Ownership and runtime domains

```text
RT gen     → canonical next-hop relation + internal ring/domain derivation;
             MGD express → Z neighbors
ControlPlane → existing connectivity + §4 node/direction predicate surface
FabricBuilder → total effects on wired producers; capability + static DOR unwiring;
                trust routes for LAND_ONLY/FORBIDDEN
Kernel     → output direction + static injection flags only
```

Hard rule: routing facts the builder needs must be on the §4 CP surface (or locally derived from
it). Builder-private tables (sender indices, injection bools, ENTER/REMAIN rows) stay in the
builder.

**Runtime-domain rule:** RT-gen derives physical domains, oriented edges, dimensions, membership,
and transition policy from the current topology scan, or derives equivalent predicate results by
another conforming process. These details remain internal. FabricBuilder consumes the §4
node/direction predicates and must not depend on ring IDs, fixed domain names, counts, memberships,
or topology-node lists. Concrete domain labels and Y nodes appear only in the explicitly labeled
`[32,4]` examples in §6 and must never become implementation identifiers.

**Normative detail: §4.** Realization deltas vs today’s code: §3. Worked injection tables: §6.

---
## 2. Current builder baseline (code as of this writing)

This section records today’s behavior so the required changes are concrete. Primary paths:

| Phase | File(s) | Today’s behavior |
|---|---|---|
| Discover | `fabric_builder.cpp` `discover_channels()` | Iterate `RoutingDirection`; one neighbor mesh; one neighbor chip per direction; cache `chip_neighbors_[dir]` and `channels_by_direction_[dir]`; reserve dispatch planes **during** discovery |
| Create routers | `fabric_builder.cpp` `create_routers()` | One router per eth chan; `RouterLocation.direction` is the discovery key |
| Variant select | `compute_mesh_router_builder.cpp` | `direction == Z → RouterVariant::Z_ROUTER`, else `MESH` |
| Has Z? | `fabric_context.cpp` `has_z_router_on_device()` | True if any active eth chan has `eth_chan_directions::Z` |
| Connection map | `router_connection_mapping.cpp` | Mesh: N/E/S/W VC0 (+ optional VC1); optional `MESH_TO_Z`. Z router: **VC1-only** `Z_TO_MESH` to N/E/S/W |
| Injection flags | `compute_sender_channel_injection_flags_for_vc()` | VC1 never injection; Linear/Mesh → all false; Torus → worker + **axis-turn** channels |
| BFC enable | `fabric_context.cpp` | `bubble_flow_control_enabled_ = is_ring_or_torus(topology)`; `need_deadlock_avoidance_support(dir)` gated by torus dimension |
| CT emission | `erisc_datamover_builder.cpp` | `ENABLE_DEADLOCK_AVOIDANCE`, `ENABLE_FIRST_LEVEL_ACK_VC0/VC1` (VC1 hardwired off today), flat `SENDER_CH_i_IS_INJECTION`, and `UPDATE_PKT_HDR_ON_RX_CH` selecting legacy RX- vs sender-side hop mutation |
| Connect | `fabric_builder.cpp` `get_router_connection_pairs()` | Orthogonal direction pairs, including Z↔{N,S,E,W} |

### 2.1 Critical current invariants that break under express routing

1. **One neighbor chip per `RoutingDirection`.**
   Express + Y-cardinal neighbors are different chips that would both be “Y-ish” if encoded only as
   N/S. The design keeps them distinct by making cardinal Y = N/S and express = Z, so *command*
   collision is avoided in the initial cut — but builder discovery is still keyed only by
   direction. That is acceptable **only while** each direction still has one logical neighbor.

2. **`Z_ROUTER` means intermesh today.**
   `RouterConnectionMapping::for_z_router()` wires VC1 `Z_TO_MESH` only. There is no VC0
   intramesh N/S↔Z continuation map for express, and no same-VC VC1 N/S↔Z realization for a landed
   carrier. The target requires both: VC0 for local/same-mesh traffic and VC1 for traffic that has
   crossed an intermesh boundary.

3. **Injection = torus axis turn.**
   Leaf→anchor→express acquisition occurs on the first cyclic express sender, which today’s turn
   heuristic can mark as transit. Injection must instead follow §4.4 effects.

4. **BFC = topology enum.**
   Express rings can exist without `Topology::Torus` on that dimension (carve-out / LINE on one axis
   with express chords still forming protected rings). Conversely, a leaf-only output may not need
   a protected-ring bubble.

5. **Intermesh identity inferred from remote mesh_id and/or Z direction.**
   Express Z can be same-mesh; intermesh may sit on a cardinal edge. Capability must be queried, not
   inferred from `direction == Z` or solely from `local.mesh_id != remote.mesh_id` without edge
   metadata (mesh-id inequality remains a useful check, but is not the capability source of truth).

---

## 3. Builder change map (condensed)

Normative CP ↔ builder contract is **§4**. This section only records *what breaks in today’s
code* and the realization deltas. Do not duplicate query APIs here.

### 3.1 Capability vs `direction == Z`

| Today | Target |
|---|---|
| `direction == Z` → `Z_ROUTER` + VC1 intermesh map | Z is a normal mesh-like forwarding direction; **express** is a capability on that edge. Materialize `INTRAMESH_EXPRESS` on every carrier VC that uses the shared canonical maps: VC0 for local/same-mesh traffic and VC1 after intermesh crossover. **Intermesh** is a capability (any direction), not “Z means boundary.” |
| `has_z_router_on_device` ↔ any Z chan | Same-mesh Z + `express_routing_enabled` → express wiring; different `mesh_id` / `INTERMESH` → intermesh template |

Express Z and intermesh on another edge of the same chip are **different router instances** /
capabilities, not a special Z-only variant family.

Packet route maps are VC-agnostic. The receiver channel / concrete downstream sender preserves the
carrier VC; `mode = INTRAMESH` after landing does not imply VC0. Once traffic crosses to VC1 it stays
on VC1 through intermediate/destination meshes, and a decoded Z action therefore requires a VC1
express-Z realization. A cardinal-only VC1 route family is not part of v1.

### 3.2 Discovery (v1)

Keep **direction-keyed** neighbors/chans (§4.3). Express assigned as Z; one logical express
neighbor per node. Query or cache the §4.2 protected-edge/turn predicate results needed by §4.4.

`LogicalEdgeId` deferred. Reject multi-express-per-node at generation.

### 3.3 Injection / BFC guards

Today: torus axis-turn heuristic before connections.
Target: §4.4 on **wired** producers only → `SENDER_CH_i_IS_INJECTION`.

| Effect | Flag |
|---|---|
| `REMAIN D` | 0 (`free ≥ 1`) |
| `ENTER D` | 1 (`free ≥ 2`) |
| `NON_RING` | non-injection |

For a protected Y egress, an off-domain Y ingress is `ENTER`, an ingress in the same oriented domain
is `REMAIN`, and an intramesh X E/W ingress is statically DOR-unwired before classification. An
E/W-facing `INTERMESH` landing is exempt and may wire to Y. Use ingress/egress capabilities and
ports, not only the previous command. Reject an ENTER+REMAIN alias on one sender. Representative
`[32,4]` results are in §6.

### 3.4 BFC enable / CT args

Keep advertising **Torus** (FabricConfig/topology). Express presence comes from **MGD express
links** + `express_routing_enabled`, not a new topology token.

`IS_INJECTION`: exact per-concrete-VC sender from day one (§4.4).

`ENABLE_DEADLOCK_AVOIDANCE`: **per outbound router (axis)**, not per-chip; enable when any serviced
carrier VC on that router requires protected flow control. `ENABLE_FIRST_LEVEL_ACK_VC0` and
`ENABLE_FIRST_LEVEL_ACK_VC1` are selected independently for the concrete receiver queues.
The current restricted VC1 envelope does not by itself require enabling a VC1 bubble. If the
arbitrary-pattern VC1 decision enables BFC, its receiver slots, ACK policy, and sender roles are
materialized independently from VC0.
Optional **leaf elision**: if `has_protected_ring(local, dim) == false` (§4.2.1), skip BFC enable on
routers whose outbound axis is that dim only after the relevant VC safety proof permits it. Leaf
chips still enable BFC on axes that
have a domain (e.g. X on a Y-leaf).

| Symbol | Express-aware source |
|---|---|
| `ENABLE_DEADLOCK_AVOIDANCE` | per-axis / per-router; set if any serviced VC requires it |
| `ENABLE_FIRST_LEVEL_ACK_VC0/VC1` | independently per concrete VC receiver; leaf-elide only when valid for that VC |
| flat `SENDER_CH_i_IS_INJECTION` | §4.4 exact for the VC owning flat sender `i` |
| Speedy / trim | **disabled** until proven |
| VC1 on express Z instance | required when intermesh is enabled and shared maps can select Z after crossover |

Landing/header rebuild itself does not acquire a protected domain. The first VC1 egress selected
after landing is classified normally and may be `ENTER`.

### 3.5 Connection maps / wiring trust (v1)

| Today | Target |
|---|---|
| Mesh: opposite + cross; Z: VC1-only `Z_TO_MESH` | If express is enabled, realize the same legal cardinal/Z transitions on VC0 and VC1: N/S↔Z within Y, and N/S/Z→E/W at the Y→X phase change. Preserve carrier VC across local forwarding. Intermesh boundary edges keep their capability-specific template. No new `ConnectionType` is required for v1; the target VC remains part of each connection record. |

**Decided (v1):** generic orthogonal/cross pairing on each required VC followed by:

1. capability unwiring; and
2. a static DOR precheck that unwires an `INTRAMESH_CARDINAL` X ingress (E/W) from every
   intramesh Y egress (`INTRAMESH_CARDINAL` N/S or `INTRAMESH_EXPRESS` Z).

An E/W-facing `INTERMESH` ingress is a landing/root and is exempt from the X→Y check.
LAND_ONLY / FORBIDDEN turn legality owned by **RT-gen + encoders**. Injection flags do not encode
LAND_ONLY. Selective filtering of those policy-forbidden turns remains optional later hardening.
Details: §4.4.

VC-preserving rule:

```text
VC0 receiver / worker producer → VC0 cardinal or express sender
VC1 receiver                   → VC1 cardinal or express sender
VC0 → VC1                      → only the initial intermesh crossover
VC1 → VC1 across a boundary    → intermediate-mesh pass-through
```

There is no VC1→VC0 landing crossover. A boundary landing on VC1 may feed N/S/Z/E/W according to the
installed shared maps; the selected VC1 sender gets its own §4.4 guard class.

Current-code cutover:

- `for_mesh_router(...)`: VC1 mesh→Z pass-through is no longer an experimental optional path when Z
  is `INTRAMESH_EXPRESS` and VC1 is active. Wire N/S→Z on VC1; DOR-unwire an E/W producer from
  intramesh N/S/Z only when its ingress edge is `INTRAMESH_CARDINAL` X. An E/W-facing `INTERMESH`
  landing must remain eligible to enter N/S/Z.
- Express-router mapping: its VC0 and VC1 receivers must each connect to same-VC N/S/E/W targets
  selected by the shared maps (subject to the DOR/capability policy above). Current
  `for_z_router()` provides only VC1 `Z_TO_MESH`.
- `target_vc == source_vc` for all of these intramesh connections.
- The capability-specific intermesh template owns the initial VC0→VC1 crossover. Subsequent
  boundary exits consume and produce VC1; none of the landing connections target VC0.
- The initial crossover is an intermesh link-endpoint rule: the source side can exit through its VC0
  boundary sender and the remote side lands in a VC1 receiver. It does not add a VC1 worker or a
  local VC0-receiver→VC1-intramesh-sender shortcut.
- No producer is wired from a receiver back to a sender on the same physical link. A canonical route
  never reverses on the link it arrived on, and an infrastructure producer that did so would add an
  `e → reverse(e)` arc to the dependency graph that assessment §5.7.3 assumes absent.

### 3.6 Allocators, intermesh context, lifecycle

All sender/receiver counts in this section are logical firmware channel slots, not physical Ethernet
links, TX queues, or hardware sender engines.

- VC0: ≤5 senders (one worker + up to four non-self producer slots), 1 receiver.
- VC1: ≤4 senders (no worker; up to four non-self producer slots), 1 receiver when intermesh is
  enabled.
- VC2: ≤1 sender; ≤1 receiver on configurations that service a VC2 receiver.
- Aggregate logical sender-channel count: `5 + 4 + 1 = 10`; configured ceiling `10`; margin `0`.
- Aggregate receiver ceiling: `1 + 1 + 1 = 3`; configured ceiling `3`; margin `0`.
- Per receiving router, cardinal/express fanout can require four downstream EDMs on VC0 and four on
  VC1: `4 + 4 = 8`; configured aggregate downstream ceiling `8`; margin `0`.
- Enforce ≥2 slots on every BFC-protected VC0 receiver and, if arbitrary-pattern VC1 BFC is enabled,
  on every corresponding VC1 receiver.
- Intermesh/VC1 shape derives from multi-mesh connectivity (`FabricBuilderContext`) plus the shared
  map requirement that cardinal and express-Z outputs remain realizable after crossover.
- Production guards: derive one BFC class per concrete sender/VC so VC1 can be enabled independently;
  no undeclared waits; landing rebuild is not acquisition, but its first egress may be.

The global `num_max_sender_channels == 10`, `num_max_receiver_channels == 3`, and
`max_downstream_edms == 8` ceilings are numerically sufficient for the express profile. Express
routing does not raise the global logical sender-channel array size; margins on all three are `0`,
so any later feature that adds a producer, a receiver VC, or a downstream identity must re-derive
these numbers rather than assume headroom.

What express does change is the per-router shape inside those ceilings. An express-aware cardinal
router needs `5 VC0 + 4 VC1 = 9` logical sender channels before the optional VC2, rather than the
current `4 + 4 = 8`, and VC0 downstream capacity rises from `3` to `4`. In current code that means
`num_sender_channels_2d` moves from `8` to `9` and `num_downstream_edms_2d_vc0` from `3` to `4`,
with the VC1 flat base placed after the five-entry VC0 range. This is firmware resource-mapping
work, not only a host constant edit: the hard-coded per-sender firmware CT tables, kernel
initialization ranges, flat VC bases, free-slot stream/counter assignment, and L1 sizing all move
together. The current 22–29 sender free-slot stream block and the dual uses of stream IDs 30–31
require an explicit mapping plan.

### 3.7 Express multicast source injection (out of V1 scope)

Assessment §9.5 and codec “Express multicast and worker connections” own this boundary. The summary
for builder purposes:

A source worker injects directly into one sender channel and bypasses source RX, so a canonical
multicast root with more than one output cannot be launched from a single connection. The sound
realization is `RoutingPlaneConnectionManager` source multi-inject: one direction-tagged slot per
canonical root output, one byte-identical copy through each. N/S/Z source fanout fits the current
four-slot manager; a broader persistent N/S/E/W/Z policy would require at least five slots and
remains an open manager policy question.

Multi-inject needs **no builder support beyond what already exists** — the worker channel on each
router already carries the correct `ENTER` flag, and the copies are independent source injections
requiring no atomicity between them. Builder must continue to retain the selected connection
direction (`ConnectionSlot::tag` or raw-sender `edm_direction`) and assert that the two
representations agree when both exist.

Express multicast is out of V1 scope because the existing single-connection worker adapters cannot
hold a multi-output root, not because of a missing builder mechanism. Builder must not attempt to
close that gap with a same-link return producer: routing a copy back over the link it arrived on
adds an `e → reverse(e)` dependency arc that the VC0 proof assumes absent, and no BFC flag or
producer-boundary ordering repairs it.

### 3.8 Universal indexed-2D cutover cleanup

At the universal indexed-2D cutover, builder retires the `UPDATE_PKT_HDR_ON_RX_CH` policy, CT
argument, and plumbing. Do not retire it while any 2D producer still emits the legacy hop program;
the cutover boundary is codec §4.5.1 and kernel §3.9.

`SENDER_CH_i_IS_INJECTION` remains required. Replace its current turn heuristic with §4.4 effects,
but retain per-concrete-sender flags, connection maps, direction mappings, VC wiring, and capability
templates. This requirement does not change 1D behavior.

---
## 4. CP ↔ builder predicate surface and local derivation

This section is the **normative** ControlPlane ↔ FabricBuilder contract. Other sections
point here. Names remain illustrative; the information split is what matters.

**One-line picture:**

```text
CP:  connectivity + express_routing_enabled (from MGD→Z neighbors)
     + node/direction predicates backed by internal ring/domain state
FB:  total NON_RING|REMAIN|ENTER|NON_CANONICAL → IS_INJECTION on wired producers;
     capability + static X→Y DOR unwiring; trust routes for LAND_ONLY/FORBIDDEN
```

The uppercase effect and policy names in this document are semantic labels, not required C++ enums,
packet fields, or persistent ControlPlane values.

Keep next-hop **direction** routing tables / `get_forwarding_direction`-style APIs for parity with
non-express systems. Indexed packet vectors are an encoder/ABI concern, not a builder CP dependency.

### 4.0 Query index

- §4.1: validated `express_routing_enabled(mesh)` and existing connectivity.
- §4.2: node/direction predicate sketch; ring IDs, ordered cycles, membership maps, and transition
  policy remain internal to RT-gen/ControlPlane.
- §4.3: exhaustive edge-capability derivation.
- §4.4: builder-owned wiring filters and total per-concrete-VC effect derivation.
- §4.5–§4.6: storage/ownership split and prohibited production dependencies.

Existing node/chip mapping, neighbor/channel/plane queries, direction conversion, route helpers, and
connection `edm_direction` remain available. Intermesh VC shape remains builder-derived from
multi-mesh connectivity rather than “has Z.”

### 4.1 Mesh flag / express enable

```text
express_routing_enabled(mesh) → bool
```

**Derived from validated MGD express connectivity**, not an independent knob that can disagree with
the neighbor graph. Owner chain:

```text
MGD express_connections
  → RT-gen / MeshGraph materializes them as Z (or equivalent) neighbors + validates
  → ControlPlane exposes neighbors + express_routing_enabled == true when that succeeded
FabricBuilder reads neighbors + flag (does not parse MGD C itself)
```

If MGD lists express links but they are not materialized as neighbors, setup is incomplete
(today’s gap: parse as C without Z neighbor materialization).

Express routing *is* the protected-ring regime for the initial cut (with Torus topology retained).

### 4.2 Protected-ring predicate surface

```text
is_protected_ring_edge(local, egress_direction) → bool

are_same_directed_ring_edges(
    local,
    ingress_direction,
    egress_direction) → bool

continuation_allowed(
    local,
    ingress_direction,
    egress_direction) → bool

has_protected_ring(local, dimension) → bool
```

Names and exact signatures are illustrative. For one wired local router pair, ControlPlane resolves:

```text
U = neighbor(local, ingress_direction)
V = local
W = neighbor(local, egress_direction)
```

The predicates describe the logical `U→V→W` turn. `continuation_allowed` is needed only after the
egress is known to be protected and the turn is not same-ring transit. It returns the Builder-visible
distinction between an allowed protected-ring acquisition and a route-illegal turn; RT-gen may keep
terminal-only versus fully forbidden policy separate internally.

Worker source injection has no ingress direction and is handled directly as first acquisition when
its egress is protected; it does not call these turn predicates.

RT-gen/ControlPlane may internally use opaque ring IDs, orientations, ordered cycles, successor maps,
`domains_of`, and a richer transition policy. Those are not external Builder API values.

**Required internal invariants behind the predicates:**

- Chip: ≤1 **physical** domain per `domain_dim` → ≤2 physical ids (FWD/REV are views).
- Edge: each directed edge ∈ at most one `(physical_id, orientation)` per VC/plane.
- **Planes:** the supported deployment supplies the same ordered active plane set on every
  participating cardinal, express, and X edge. One physical id applies across those planes (not
  per-plane ids), and Builder preserves plane index on every connection. Builder does not perform a
  physical-link-class homogeneity qualification.
- **Columns / rows:** Y physical ids column-local; X row-local.
- Express-Z / Y N/S protected resources ↔ `domain_dim == Y`; E/W X-ring ↔ `domain_dim == X`.

**Supported homogeneity input:** patterns match along a dimension; columns/rows may still use
distinct internal physical ids.

#### 4.2.1 Required predicate behavior

A leaf on one dimension may still belong to a protected ring on the other:

```text
has_protected_ring(y_leaf, Y) = false
has_protected_ring(y_leaf, X) = true
has_protected_ring(anchor, Y) = true
```

Do not use one mesh-wide leaf bit. Optional BFC elision is per dimension; Builder still runs the
per-producer §4.4 checks for exact `IS_INJECTION`.

Ingress and egress axes are derived from directions:

```text
N/S/Z → Y
E/W   → X
```

The external predicates must reproduce these results:

- source worker → protected egress: first acquisition;
- `INTERMESH` ingress → protected egress: first acquisition on the landed VC;
- Y ingress → protected X egress: first X acquisition;
- intramesh X ingress → Y egress: statically unwired before predicate use;
- same directed protected ring on ingress and egress: transit, not acquisition;
- same-dimension transition between different rings or from a non-ring attachment:
  `continuation_allowed(local, ingress, egress)` decides whether acquisition is permitted.

This avoids exposing the internal predecessor-domain recovery that implements those answers.

Representative `[32,4]` CONTINUE, LAND_ONLY, leaf, and Y→X results are in §6. They are examples of
this generic recovery algorithm, not additional CP facts.

### 4.3 Connectivity and edge capability (exhaustive)

“Local edges” = ordinary discovery — **not** a new edge-role product:

```text
get_chip_neighbors(node, direction)
get_active_fabric_eth_routing_planes_in_direction(...)
get_connected_mesh_chip_chan_ids(...)
routing_direction_to_eth_direction(...)
```

Under the initial cut (one logical express neighbor as Z; cardinal N/S/E/W), direction-keyed
discovery remains viable **after** RT-gen materializes MGD express links as Z neighbors (§4.1).

**Capability truth table** (derive in builder / FabricContext):

```text
remote.mesh_id != local.mesh_id              →  INTERMESH
  (any local direction)

remote.mesh_id == local.mesh_id
  && direction ∈ {N,E,S,W}                   →  INTRAMESH_CARDINAL

remote.mesh_id == local.mesh_id
  && direction == Z
  && express_routing_enabled                 →  INTRAMESH_EXPRESS

remote.mesh_id == local.mesh_id
  && direction == Z
  && !express_routing_enabled                →  INVALID (generation / config failure)
```

Stop using “has Z” alone as intermesh. Express-Z BFC participates in protected **Y-axis** rings.

**Consequence for the edge-facing CT flags.** `IS_INTERMESH_ROUTER_ON_EDGE` and
`IS_INTRAMESH_ROUTER_ON_EDGE` are per-Ethernet-channel and are computed from the ControlPlane channel
split: a channel is intramesh-facing when it appears in the node's direction map and is *not*
registered as an exit-node direction. A same-mesh express-Z channel must therefore land on the
intramesh-facing side. Two things depend on it:

- kernel §4.2's exit check is gated by `is_intramesh_router_on_edge`, so a Z router on an exit chip
  only inherits the gate when express-Z is intramesh-facing. Misclassifying it would make a packet
  that reaches the exit chip over a Z chord deliver locally instead of crossing the boundary;
- the VC0/VC1 wiring templates in §3.5 are selected from the same capability, so a misclassified
  express-Z edge would also receive the intermesh boundary template rather than the shared cardinal
  and express maps.

No new flag or builder mechanism is required for either; the requirement is that the classification
is capability-based rather than `direction == Z`.

### 4.4 Builder derivation (effects → `IS_INJECTION`)

ControlPlane does **not** expose builder-shaped effect rows. Builder derives a **total** effect for
every **wired** producer. The builder applies the fixed Y-before-X local DOR precheck before this
derivation and rejects any intramesh X→Y producer that survives connection-map filtering. Route
legality for LAND_ONLY/FORBIDDEN remains RT-gen/encoder-owned; those wired non-canonical slots still
need a defined compile-time result.

Derivation is per concrete carrier VC. Physical-domain membership and transition policy are shared,
but `(egress edge, VC0)` and `(egress edge, VC1)` are distinct queues, protected receivers, sender
flags, and—when BFC is enabled on that VC—bubbles. A VC0 result must not be reused as the VC1 CT
value without separately materializing and validating the VC1 producer.

This reuse of physical transition facts covers VC1 traversal of the shared intramesh rings. It does
not prove that those are the only VC1 dependency cycles under arbitrary multi-mesh traffic. The
current two-link, predominantly linear operating envelope remains supported as existing behavior.
Before broadening that envelope, the multi-mesh VC1 CDG analysis may require additional protected
domains / guard facts; the builder must not infer those missing classes.

**Internal route policy ≠ Builder effect:**

| Internal transition result | Builder effect if that turn is wired |
|---|---|
| `CONTINUE_ALLOWED` + cyclic egress | `ENTER` (if off-ring in) or handled via REMAIN path |
| (n/a) both edges in same `(P,orient)` | `REMAIN` |
| no cyclic egress | `NON_RING` |
| `LAND_ONLY` or `FORBIDDEN` | `NON_CANONICAL` |

| Effect | `IS_INJECTION` | Alias reject |
|---|---|---|
| `REMAIN` | 0 | participates with ENTER |
| `ENTER` | 1 | conflicts with REMAIN |
| `NON_RING` | 0 | no |
| `NON_CANONICAL` | 0 | **excluded** from ACQ/TRANSIT alias reject |

**Wiring policy (v1):**

| Turn class | Connect? |
|---|---|
| Capability-illegal | **Unwire** |
| Static DOR-illegal: `INTRAMESH_CARDINAL` X E/W ingress → intramesh Y N/S/Z egress | **Unwire**; fail configuration if still present after mapping |
| `LAND_ONLY` / `FORBIDDEN` but still in generic map | **May stay wired** → classify `NON_CANONICAL`; trust routes not to use |

The DOR precheck is capability-aware:

```text
is_intramesh_x_ingress =
    ingress_capability == INTRAMESH_CARDINAL && D_in ∈ {E,W}

is_intramesh_y_egress =
    (egress_capability == INTRAMESH_CARDINAL && D_out ∈ {N,S}) ||
    (egress_capability == INTRAMESH_EXPRESS && D_out == Z)

if is_intramesh_x_ingress && is_intramesh_y_egress:
    unwire
```

An `INTERMESH` ingress is a landing/root, not an intramesh X ingress, even if its local physical port
is E or W. This check is static and local; it does not attempt to distinguish route-state-dependent
misuse of an otherwise legal Y→X producer. Those cases remain RT-gen/encoder validation obligations.

**No same-link producer class exists.** A `U→V→U` turn is outside the canonical route set and is
not wired, and there is no infrastructure producer that routes a copy back over the link it arrived
on. Such a producer would add an `e → reverse(e)` dependency arc that assessment §5.7.3 assumes
absent; see §3.7.

#### Explicit builder algorithm (per chip)

The builder does not start from a multi-hop path. It discovers all local neighbors, creates routers,
wires producer slots, then classifies each wired `(ingress, egress)`.

```text
1. discover_channels (existing)
     for each dir in {N,E,S,W,Z} (as present):
       neighbors_[dir]  = remote chip
       chans_[dir]      = eth / planes

2. create_routers
     one router instance per active eth chan / direction

3. connect / connection map (v1)
     generic orthogonal/cross pairing per required VC
     + capability unwiring (intramesh X E/W ↛ express Z)
     + static DOR unwiring (intramesh X E/W ↛ intramesh Y N/S/Z)
     + landing exception (INTERMESH ingress may feed intramesh N/S/Z)
     each outbound router gets producer slots, e.g. Z-router:
       VC0: worker + wired non-self VC0 producers
       VC1: wired non-self VC1 producers; no worker
       …

4. for each outbound router, carrier VC, and egress direction D_out:
     V = local_chip
     for each wired producer slot i:
       if VC0 worker:
         effect = is_protected_ring_edge(V, D_out) ? ENTER : NON_RING
       else:
         assert !is_static_dor_forbidden(D_in, ingress_capability, D_out, egress_capability)
         effect = classify(V, D_in, D_out)  // node/direction predicates below
       flat_i = flat_sender_index(VC, i)
       SENDER_CH_<flat_i>_IS_INJECTION = (effect == ENTER)
     reject only if ENTER and REMAIN would alias onto the same sender
```

For a VC1 landing, `U` is the boundary-facing/intermesh ingress. If the first local egress enters a
protected Y or X ring, the corresponding VC1 sender is `ENTER`; the landing map rebuild does not
itself assign the guard.

#### Domain effect rules (one wired direction pair at local node `V`)

```text
if is_static_dor_forbidden(D_in, ingress_capability, D_out, egress_capability):
    FAIL CONFIGURATION                  // connection-map bug; this producer must be unwired

!is_protected_ring_edge(V, D_out)
    → NON_RING

ingress_capability == INTERMESH
    → ENTER                                  // landed carrier acquires protected egress

are_same_directed_ring_edges(V, D_in, D_out)
    → REMAIN                                 // IS_INJECTION=0

ingress_dim(D_in) != egress_dim(D_out)
    → ENTER                                  // legal remaining case is Y→X

continuation_allowed(V, D_in, D_out)
    → ENTER                                  // leaf/non-ring entry or allowed Y-ring transition

otherwise
    → NON_CANONICAL                          // route generation must not use this turn
```

Classify is called with local node `V`, `D_in`, `D_out`, and both edge capabilities known from the
producer slot and egress router. ControlPlane resolves the conceptual ingress/egress neighbors
internally. Only `INTRAMESH_CARDINAL` E/W is an X ingress; an `INTERMESH` E/W port takes the landing
branch above.

Representative `[32,4]` applications of the algorithm are in §6.

#### Why ENTER/REMAIN still matter in the builder

Only to set BFC gates: ENTER → `free ≥ 2`, REMAIN → `free ≥ 1`. Same Z output has both (e.g. from
chip 1 vs from leaf 3). Kernel never sees these semantic effects.

### 4.5 Storage / ownership split

```text
RT gen
  owns the canonical logical next-hop relation and reconstructible routes
  may build physical domains, oriented edge sets, membership maps, and transition policy internally
  materializes MGD express as Z neighbors; sets express_routing_enabled
  enforces LAND_ONLY / FORBIDDEN on paths; CDG private

ControlPlane
  exposes: connectivity, express_routing_enabled, and §4.2 node/direction predicates
  exposes logical next-hop/route queries to codec/host setup, not to Builder
  may store internal domain records and route-generation working state
  does not store: domain_kind, IS_INJECTION, builder effect tables, SCC-count API

FabricBuilder
  reads §§4.1–4.4; derives total effects on wired producers;
  generic maps + capability/static-DOR unwiring
  retains connection directions
```

| Builder need | CP provides | Builder derives |
|---|---|---|
| Neighbors / chans | existing APIs (express as Z neighbors) | — |
| Express on? | `express_routing_enabled` (from MGD validate) | capability + CT |
| Protected ring on local dimension? | `has_protected_ring(local, dim)` | optional BFC elision |
| Protected egress? | `is_protected_ring_edge(local, D_out)` | NON_RING vs protected |
| Same directed ring transit? | `are_same_directed_ring_edges(local, D_in, D_out)` | REMAIN |
| Non-transit acquisition legal? | `continuation_allowed(local, D_in, D_out)` | ENTER vs NON_CANONICAL |
| Injection | — | total effect → flag |
| Static X→Y DOR legality | directions + edge capabilities | unwire; assert absent before classification |
| Terminal-only / forbidden turn | `continuation_allowed == false` | NON_CANONICAL if wired |
| Capability | mesh ids + dirs + flag (§4.3) | EXPRESS / CARDINAL / INTERMESH |
| Source inject direction | existing connection-table `edm_direction`; manager tags | retain/expose on sender; tag-direction consistency assert |

### 4.6 Explicitly not on the builder production path

- Full all-pairs route set / CDG dumps
- Canonical logical next-hop or complete-route queries (codec/host consumers)
- Precomputed domain-effect transition tables
- Internal ring IDs, ordered cycles, membership maps, or transition-policy enums
- Destination-indexed vectors / multicast maps
- Edge-role *product* DB (roles beyond cyclic membership) and SCC census APIs

---

## 5. Remaining requirements and builder implementation checklist

Open requirements:

- Decide and complete the system-level VC1 CDG/BFC treatment before supporting arbitrary cross-mesh
  traffic beyond the current two-link, predominantly linear envelope. It must either confirm that §4
  supplies all required builder inputs or define additional VC1 domain/guard facts.
- Implement the per-router shape change inside the existing ceilings: `num_sender_channels_2d`
  `8 → 9`, `num_downstream_edms_2d_vc0` `3 → 4`, VC1 flat base after the five-entry VC0 range, and
  the matching CT tables, stream/counter assignment, initialization ranges, and L1 sizing.
- Decide the broader manager policy before express multicast is enabled: persistent N/S/E/W/Z
  provisioning needs at least five slots; the current four are sufficient for the N/S/Z source
  multi-inject requirement in §3.7.

Builder implementation checklist:

1. Consume the runtime-derived §4 facts and reject domain/cardinality/capability violations; preserve
   the ordered plane identity supplied by the supported deployment.
2. Realize express as same-VC cardinal/Z wiring on VC0 and VC1, preserve VC1 after crossover, and
   apply capability plus static intramesh X→Y DOR unwiring.
3. Classify every wired concrete producer with §4.4; emit exact per-VC `IS_INJECTION`, per-router
   deadlock-avoidance enable, and independent first-level ACK flags.
4. Apply the §3.6 sender/receiver/downstream ceilings and protected-receiver slot requirements.
5. Wire no producer from a receiver back to a sender on the same physical link (§3.7).
6. Retire `UPDATE_PKT_HDR_ON_RX_CH` only at the codec §4.5.1 / kernel §3.9 indexed-2D cutover.
7. Regress builder derivation and projection against assessment Appendix A.6; require zero effect
   mismatches, mixed-role aliases, unmapped/multiply mapped transitions, and connected
   DOR-forbidden arcs.

Builder verification:

- Check every router/VC/flat slot against its ingress/egress capabilities, `U→V→W`, total effect,
  emitted `IS_INJECTION`, and concrete receiver.
- Check that no capability-illegal or static intramesh-X→Y connection survives mapping.
- Check VC0 and VC1 CT flags and first-level ACK settings independently.
- Check flat VC bases, streams/counters, L1 queue bytes, receiver slots, and downstream counts with
  and without VC2.
- Check that no producer connects a receiver to a sender on the same physical link.
- At indexed cutover, check that no remaining 2D producer depends on `UPDATE_PKT_HDR_ON_RX_CH`.

---
## 6. Worked `[32,4]` examples: nodes Y=2 and Y=3

This section is a **non-normative `[32,4]` example**. Its ex4 names, concrete Y nodes, and sender
labels are explanatory only: they are neither CP storage nor required implementation identifiers.
Other topology shapes use domains and policy derived from their own runtime scan. Assessment
Appendix A.6 owns the independent route-occurrence regression against §4 builder projection.

### 6.0 Example assumptions

- One fixed-X column plus E/W neighbors at the same Y; no intermesh edge on Y=2 or Y=3.
- `e(u→v)` labels Y edges; `e_E@Y` / `e_W@Y` label X-ring edges.
- `(D_ex4,FWD|REV)` is one example Y physical domain with two orientations;
  `(D_x@Y,FWD|REV)` is the example X domain at row Y.
- The same physical id applies across homogeneous planes.

```text
Y phase (N|S|Z)* completes before X phase (E|W)*
intramesh X E/W → intramesh Y N/S/Z is statically unwired
Y → E/W enters X when the E/W hop is the first X resource
```

An E/W-facing `INTERMESH` landing is exempt and may begin Y. Leaf attachments in this example are
cardinal but not members of `D_ex4`.

```text
VC0 output: worker + each wired VC0 ingress
VC1 output: each wired VC1 ingress; no worker; carrier remains VC1
```

Tables below show VC0 for compactness. A realized VC1 turn receives the same physical effect but an
independent sender index, flag, queue, and protected receiver.

---

### 6.1 Node Y=2 (ex4 express node)

#### 6.1.1 Local edges

| Edge id | Command | Capability | Remote | Ring role |
|---|---|---|---|---|
| `e(2→1)` | N | INTRAMESH_CARDINAL | Y=1 | ex4-rev cyclic (with reverse-Z) |
| `e(2→3)` | S | INTRAMESH_CARDINAL | Y=3 leaf | leaf attachment only (not ex4 cyclic) |
| `e(2→5)` | Z | INTRAMESH_EXPRESS | Y=5 | ex4-fwd cyclic |
| `e_E@2` | E | INTRAMESH_CARDINAL | (x+1, Y=2) | X-fwd cyclic at Y=2 |
| `e_W@2` | W | INTRAMESH_CARDINAL | (x−1, Y=2) | X-rev cyclic at Y=2 |

Ingress names below are local ports: N-face=`e(1→2)`, S-face=`e(3→2)`, and
Z-face=`e(5→2)`. Builder keys off the ingress edge/port, not only the wire command.

#### 6.1.2 Local transitions (routing-config facts → builder guards)

**Onto express `e(2→5)` (Z):**

| Ingress producer | Effect | `IS_INJECTION` | Gate / treatment |
|---|---|---|---|
| source worker | ENTER `(D_ex4,FWD)` | 1 | `free(Q(2→5)) ≥ 2` |
| N-face `e(1→2)` | REMAIN `(D_ex4,FWD)` | 0 | `free(Q(2→5)) ≥ 1` |
| S-face `e(3→2)` from leaf | ENTER `(D_ex4,FWD)` | 1 | `free(Q(2→5)) ≥ 2` |
| Z-face | — | — | same-link reversal is not wired (§3.7) |
| intramesh E/W-face | — | — | static X→Y unwiring |

The reverse-domain cardinal output `e(2→1)` has the symmetric distinction: source or leaf
attachment enters `(D_ex4,REV)`, while Z-face transit remains in it.

**Onto `e_E@2` (E):**

| Ingress producer | Effect | `IS_INJECTION` |
|---|---|---|
| source or N/S/Z-face after Y completes | ENTER `(D_x@Y2,FWD)` | 1 |
| W-face X transit | REMAIN `(D_x@Y2,FWD)` | 0 |
| E-face | same-link reversal is not wired (§3.7) | — |

W egress is symmetric with `REV`. A route-state-dependent Y→X misuse may be
`NON_CANONICAL`; static intramesh X→Y is unwired.

**Builder coverage takeaway for Y=2:** an EXPRESS Z router and the E/W MESH routers both need
**mixed** acquisition and transit producers. Enabling BFC only from `Topology::Torus` or only on
“axis turns” is insufficient: Y→E/W is an X-domain ENTER even when the producer is Z or N/S.

---

### 6.2 Node Y=3 (leaf)

Y=3 has cardinal attachments `e(3→2)` and `e(3→4)`, no express neighbor, and X-ring E/W
edges. Its representative effects are:

| Ingress | Egress | Domain effect | `IS_INJECTION` | Notes |
|---|---|---|---|---|
| source worker | `e(3→2)` or `e(3→4)` | NON_RING | 0 | Y attachment does not acquire ex4 |
| source or N/S-face | `e_E@3` | ENTER `(D_x@Y3,FWD)` | 1 | first X resource |
| W-face | `e_E@3` | REMAIN `(D_x@Y3,FWD)` | 0 | X transit |

W egress is symmetric with X-rev.

**Builder coverage takeaway for Y=3:**

- No EXPRESS router.
- Cardinal N/S to anchor / paired leaf are **not** ex4 acquisition.
- E/W still need X-ring BFC, including Y→X after landing on the leaf row.
- Therefore BFC cannot be enabled or disabled per chip merely from Y-leaf status.

---

### 6.3 Same-dimension continuation outcomes

The `[32,4]` internal policy produces two useful external predicate results:

```text
CONTINUE example:
  0 (ex8) → 1 (land) → 2 (first ex4-fwd cyclic edge)
  continuation_allowed(
      local=1,
      ingress=the local direction receiving 0→1,
      egress=the local direction sending 1→2) = true
  builder effect: ENTER (D_ex4,FWD), IS_INJECTION=1

LAND_ONLY example:
  6 (ex4) → 7 (land) → 8 (first ex8-fwd cyclic edge)
  continuation_allowed(
      local=7,
      ingress=the local direction receiving 6→7,
      egress=the local direction sending 7→8) = false
  builder effect if wired: NON_CANONICAL, IS_INJECTION=0
```

The first case demonstrates why same-dimension continuation cannot rely only on whether the ingress
edge itself is cyclic. The second remains physically wireable under v1 but must not appear in a
generated route. Neither example adds named domains or transition enums to the production CP
surface.

---

### 6.4 VC1 landing / pass-through connectivity

This sequence fixes the builder-visible VC transitions; it is not the arbitrary-pattern VC1 CDG/BFC
closure.

```text
source mesh:
  worker / local route on VC0
  initial intermesh exit: capability template crosses VC0 → VC1

landing mesh:
  INTERMESH-capability VC1 receiver intercepts the boundary landing before ordinary map decode
  maps are re-rooted; carrier remains VC1
  first action Z (when the boundary and express-Z edges are distinct):
    boundary VC1 receiver → EXPRESS-Z VC1 sender
    ingress_capability=INTERMESH → A=NONE → ENTER local Y ring (free ≥ 2)

after express hop:
  Z-facing VC1 receiver → N/S VC1 sender: REMAIN or ENTER from §4.4
  Z-facing VC1 receiver → E/W VC1 sender: ENTER X when Y is complete

intermediate-mesh exit:
  VC1 receiver → boundary VC1 sender → next mesh still on VC1

destination mesh:
  same VC1 cardinal/express wiring until LOCAL_DELIVER
  no VC1 → VC0 landing connection
```

Required realization checks:

- every action selected by a landed map has a same-VC1 concrete sender;
- the VC1 sender owns an independent flat channel index and `IS_INJECTION` value;
- every protected VC1 egress targets a protected VC1 receiver with the required slot count;
- an E/W-facing `INTERMESH` landing is allowed to start Y, while intramesh-X E/W→Y remains
  statically DOR-unwired.

---

### 6.5 Worked multi-output source root: canonical `S|Z` at Y=2

Suppose source Y=2 has canonical root action `S|Z`. With S and Z manager slots open, the worker
injects one byte-identical canonical-map copy on each edge. The builder-visible result is two
ordinary worker-channel producers, classified independently:

| Concrete producer | Builder effect | `IS_INJECTION` |
|---|---|---|
| Y=2 worker → S output `e(2→3)` | NON_RING (leaf attachment) | 0 |
| Y=2 worker → Z output `e(2→5)` | ENTER `(D_ex4,FWD)` | 1; require `free(Q(2→5)) ≥ 2` |

Each copy is an independent source injection, so no atomicity is required between them and neither
producer holds a fabric receiver while the other waits.

If the worker owns only one of the two connections, the operation is not realizable: the unconnected
root subtree is unreachable. Builder does not close that gap — routing a copy back over the injected
link would add the dependency arc described in §3.7. Express multicast is therefore out of V1 scope
until every participating client holds one connection per canonical root output; see assessment
§9.5.

### 6.6 Tie-back to §4

§6 tables are **expected derived results for the `[32,4]` example** — not CP storage or required
domain names. Effects are total (`NON_RING` / `REMAIN` / `ENTER` / `NON_CANONICAL`). Labels `ex4-fwd` mean
`(D_ex4, FWD)`. Express-Z BFC belongs to the protected Y axis. “Unwired” covers both
capability-unwiring and the static intramesh-X E/W → intramesh-Y N/S/Z DOR unwiring required by
§4.4. `LAND_ONLY` / `FORBIDDEN` slots are different: they may stay wired and are classified
`NON_CANONICAL`.

`TERMINAL` is a route-oracle semantic, not a builder effect. A wired terminal/noncyclic egress maps
to builder `NON_RING`; a wired route-illegal `LAND_ONLY` / `FORBIDDEN` slot maps to
`NON_CANONICAL`. Neither term changes the static DOR unwiring rule above.

Assessment Appendix A.6 is the regression authority for these VC0 examples. Its independent
route-occurrence projection must match §4 effects, map each connected transition to exactly one
concrete sender, find no ENTER/REMAIN alias, and find no connected static DOR-forbidden arc. Those
test-only rows are not a production ControlPlane API.

---
