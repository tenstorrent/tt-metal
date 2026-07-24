# Cardinal N/S with Z Skip-Link Routing: Design and Assessment

## 0. Executive summary, scope, and status

This document is the central high-level system design and assessment for routing nested Galaxy
express links while preserving the fabric client's cardinal coordinate contract:

- **N/S retain global cardinal meaning** and traverse one ordinary Y-base edge.
- **Z means one intramesh express hop** on an ex4 or ex8 chord.
- **E/W retain global cardinal meaning** on the orthogonal four-chip X ring.

The constrained baseline route function is defined in §3; no sibling document is needed
to determine a route. The design first selects a safe physical route, then encodes each selected edge
as N/S/E/W/Z. It does not discover routes by interpreting command letters.

The same command contract and route rules apply to all four supported fixtures. The `[32,4]`
four-Galaxy route/CDG and destination-vector results are validated topology-specific results. A standalone
`[8,4]` torus plus two- and three-Galaxy `[16,4]` and `[24,4]` carve-outs from the quad setup are
mandatory scale fixtures for ring synthesis and table generation; they do not inherit the `[32,4]`
SCC result automatically.

This is an architecture and system-design assessment. It defines target behavior and implementation
checks; it does not claim that the described ControlPlane / RT-gen route changes, builder, packet ABI,
or kernel work is implemented. The current physical and software baseline is recorded in
`GALAXY_WORKING_MODEL.md`.

### 0.1 Initial-cut boundaries

The route rules and required setup checks are shared across the supported fixtures, while the initial
implementation and proof target remain deliberately narrow:

- **Y-only express links:** express edges exist only in the N/S (Y) dimension. E/W remains the
  ordinary four-chip X ring; E/W express links are out of scope.
- **Physical command contract:** S follows increasing logical Y/downward in team diagrams, N follows
  decreasing logical Y/upward, Z selects one Y express edge, and E/W traverse ordinary X edges.
- **Deterministic DOR:** complete `(N/S/Z)*` Y routing before `(E/W)*` X routing. X→Y return,
  adaptive orientation changes, opportunistic shortcuts, and runtime degraded detours are forbidden.
- **Primary and scale fixtures:** `[32,4]` is the primary deployment fixture. `[8,4]`, `[16,4]`, and
  `[24,4]` are mandatory synthesis and representation fixtures, but each requires its own route/CDG
  check. V1 does not claim admission of arbitrary future mesh or pod shapes.
- **Constrained `[32,4]` domains:** leaves are endpoints, base-32 is not an active transit domain,
  same-ring routes remain on ex4 or ex8, each route uses at most one ex4/ex8 crossover, ex8→ex4 may
  CONTINUE, and ex4→ex8 is TERMINAL in Y.
- **One logical intramesh express neighbor:** each node has at most one logical intramesh express
  adjacency, so bare Z identifies one neighbor directly. Parallel physical lanes that realize that
  same logical edge are allowed; routing generation rejects multiple logical express edges or
  neighbors. Future support requires a richer output identity and the extension process in §10.
- **VC scope:** same-mesh cardinal and express traffic uses VC0; no new VC is introduced for express
  routing or multicast. A packet that has crossed an intermesh boundary remains on VC1 while
  traversing cardinal or express links in later meshes. Optional existing VC2 behavior remains
  separate. A pair of current four-Galaxy meshes has two intermesh links and predominantly linear
  intermesh traffic; that restricted operating envelope is retained. Arbitrary cross-mesh traffic
  requires the separate VC1 BFC decision and safety closure in §5.7.5.
- **Homogeneous deployment input:** the supported cluster provides the same Y/ring structure in every
  X column, the same four-chip X-ring structure in every Y row, and one uniform ordered routing-plane
  set. This is an input guarantee for the declared fixtures, not a topology-generality goal or a new
  physical-link qualification performed by this routing design.
- **Plane and buffer realization:** software preserves the provided plane identity on every turn.
  Every BFC-protected receiver has at least two packet slots.
- **Operational model:** topology, link, and routing-plane changes are fail-stop. Recovery routing is
  a separate extension.
- **Deployment ABI:** 2D software cuts over atomically to one indexed packet/table ABI after unicast,
  intermesh, multicast, and E/W feature parity; legacy and indexed modes are never deployed
  together.
- **Extension non-goals:** opportunistic Z, enabling the forbidden Y nearest-neighbor
  **base-32** cycle, multiple logical express neighbors, adaptive routing, and degraded routing are
  outside this proof. They require a new canonical route set, CDG/SCC analysis, representation
  validation, and component conformance under §10.

### 0.2 Design finding

Cardinal N/S with explicit Z express hops is the selected command and representation direction. It is
viable under the initial-cut boundaries above and the fixed route contract in §3.

The complete `[32,4]` baseline Y-route set for one fixed X contains 992 directed routes. Its
edge-level Channel Dependency Graph (CDG) has exactly four nontrivial Y SCCs per fixed X: ex4
forward/reverse and ex8 forward/reverse. §5 gives the complete paper reduction from those SCCs to
per-ring BFC for VC0 unicast and multicast. The remaining VC0 work is implementation and machine
conformance, not an unresolved deadlock theory. Current two-link, predominantly linear VC1 traffic is
not declared broken by this design, but arbitrary multi-mesh traffic remains a separate BFC/proof
obligation.

The important implementation difference is route encoding and concrete output wiring: an express
chord is emitted through Z rather than a logical ring-local N/S output. The protected ex4/ex8 domains
and their acquisition order are properties of the canonical all-pairs route set rather than command
letters. RT-gen owns the canonical next-hop relation, from which complete routes are reconstructible.
ControlPlane exposes logical route queries to codec/host setup and node/direction protected-ring
predicates to FabricBuilder; internal ring IDs, membership maps, and transition policy remain private.
FabricBuilder derives every wired producer's total local effect and concrete per-sender flag from that
surface; it does not query precomputed ENTER/REMAIN rows or `list_local_transitions`.

### 0.3 Status taxonomy

The document uses the following status terms deliberately:

- **Required contract:** behavior this design requires.
- **Validated model result:** a finite route, codec, execution, or capacity space has been
  exhaustively checked against the stated assessment oracle.
- **Derived capacity result:** complete arithmetic over the declared router variants and current
  static limits; concrete allocation and queue-graph conformance remain separate.
- **Paper proof:** a complete checkable argument under explicit architectural assumptions, but not a
  machine-checked implementation proof.
- **Implementation gap:** target behavior is defined but current software does not yet realize or
  validate it.
- **Open obligation:** the contract or proof is not complete. The principal open safety obligation is
  VC1 under arbitrary multi-mesh traffic, not the existence of the current restricted VC1 path.
- **Finalized ControlPlane route state:** generic route and topology facts available after the
  applicable §6.6 setup checks pass. Consumers do not repeat the route search during normal setup;
  FabricBuilder only checks that its bounded local realization is possible. Neither result is a
  runtime proof or a new persisted artifact.
- **Component contract:** the authoritative implementation-level contract for one owned surface. If
  a central summary conflicts with a component contract in that component's area, the component
  contract governs and this assessment must be corrected.

The current assessment status is:

```text
command and route contract                 required
[32,4] canonical Y-route set and edge SCCs validated model result
cardinal destination-vector suffix sweep  validated model result
[32,4] multicast action-map sweep          validated model result
[supported fixtures] row/column route uniformity  guaranteed topology input; mandatory setup check
[32,4] logical software-channel budget     derived capacity result; array counts only
[32,4] VC0 domain-effect→sender sweep      required validation; not yet run
VC0 unicast + multicast reduction          complete paper proof
ControlPlane / RT-gen route setup, builder, packet, kernel    specified, not implemented
VC1 current two-link, predominantly linear envelope  retained operational scope; no all-pattern claim
VC1 arbitrary multi-mesh traffic                    BFC decision and safety proof open
```

The validated-model rows are assessment-time exhaustive enumerations with their complete oracles and
expected counters recorded in Appendix A. Repository-owned regression coverage remains to be
implemented; these rows do not claim that one concrete build conforms to the paper-proof premises.

This subsection is the authoritative status ledger. The scoped conclusions in §5.7.6 and §12 must
be updated with it whenever the finding changes.

### 0.4 Authority and document precedence

Ownership is normative:

- **This assessment** owns physical/logical system assumptions, command semantics, the canonical
  route oracle, routing-generation policy, CDG/SCC analysis, the VC0 BFC proof, system-level
  invariants, the central cross-layer dependency ledger, and validation oracles/evidence.
- **`GALAXY_CONTROL_PLANE_ROUTING_GENERATION_CONTRACT.md`** owns the detailed pre-builder
  MGD/MeshGraph updates, runtime-derived ring synthesis, canonical-route generation procedure,
  ControlPlane state, setup checks, and handoffs to downstream consumers. It implements this
  assessment's route policy rather than defining another oracle.
- **`GALAXY_BUILDER_ROUTING_CONFIG_CONTRACT.md`** owns the ControlPlane↔FabricBuilder surface,
  local-effect derivation, wiring, channel/queue allocation, BFC compile-time flags, and U-turn
  sender realization.
- **`GALAXY_DEVICE_ROUTE_CODEC_CONTRACT.md`** owns L1/device route artifacts, packet/header ABI,
  encode/load/decode semantics, multicast encoding, and source fanout/reroot overlay.
- **`GALAXY_DEVICE_ROUTER_KERNEL_CONTRACT.md`** owns ERISC decode/admit/forward dispatch,
  intermesh-transition execution, header-mutation retirement, sender-step BFC consumption, and
  controlled same-link return.
- **`GALAXY_WORKING_MODEL.md`** owns the physical and current-code baseline context.

If a central summary conflicts with a component contract in that component's area, the component
contract governs and this assessment must be corrected. Appendix B is the normative ownership index.

### 0.5 Reading guide

- **Decision and current status:** §§0 and 12.
- **Physical route and safety contract:** §§1–5.
- **Routing generation and consumer boundary:** §§6–7.
- **Representation requirements and multicast safety:** §§8–9.
- **Alternative/extension record and implementation checklist:** §§10–11.
- **Finite validation oracles and evidence:** Appendix A.
- **Normative document ownership/index:** Appendix B.

---

## 1. Terminology and governing system model

### 1.1 Terminology

- **MGD (Mesh Graph Descriptor):** logical mesh and intermesh topology intent.
- **PGD (Physical Grouping Descriptor):** allowed physical carve-outs and groupings.
- **PSD (Physical System Descriptor):** discovered physical ASIC and link graph.
- **DOR (dimension-ordered routing):** all Y work completes before X work; no E/W dependency may
  return to N/S/Z.
- **CDG (Channel Dependency Graph):** one node per modeled directed resource and one edge for each
  possible hold-and-wait dependency between consecutive resources.
- **SCC (strongly connected component):** a maximal mutually reachable CDG region. A nontrivial SCC
  contains a potential dependency cycle.
- **BFC (bubble flow control):** admission discipline that preserves at least one free receiver slot
  in each protected directed ring. Acquisition requires `free ≥ 2`; same-ring transit requires
  `free ≥ 1`.
- **Routing plane:** an independent physical/software resource lane. Packets do not change planes in
  the baseline; every protected directed ring has a separate bubble on each usable plane.
- **VC0 / VC1:** software virtual channels. VC0 carries the intramesh route family analyzed here.
  VC1 carries intermesh traffic; the current two-link, predominantly linear envelope is retained,
  while arbitrary multi-mesh traffic is a separate BFC and proof domain.
- **Routing path:** one ordered directed-edge sequence for one source/destination pair.
- **Canonical route:** the one selected routing path for a legal source/destination pair, annotated
  with commands, route family, and protected-domain effects.
- **Canonical all-pairs route set:** the collection of canonical routes for every legal ordered
  source/destination pair in the topology. For the deterministic policy in this document it is the
  global routing answer, not one packet's path. The complete `[32,4]` 2D topology has
  `128 × 127 = 16,256` such routes; its Y-only oracle for one fixed X has `32 × 31 = 992`.
- **Ring domain:** a synthesized directed cycle of concrete edge resources protected by one bubble.
  A ring domain is not a command direction and is not identified by the letter Z.
- **ex4 / ex8:** the active 16-node and 8-node long-axis ring families in the validated `[32,4]`
  fixture, formed by alternating selected base edges with 4×4 or 4×8 wrap chords.
- **Leaf:** a Y endpoint without an express neighbor; leaves may source or receive traffic but are not
  Y-transit nodes in the baseline.
- **Anchor:** the ex4 node used to enter or leave the constrained route set for a leaf.
- **CONTINUE / TERMINAL:** a cross-domain route may acquire and transit the destination domain, or
  land at its final Y destination without consuming a destination-domain ring-transit resource.

### 1.2 Physical context

The validated quad topology contains:

```text
4 Galaxies × 32 chips per Galaxy
= 128 chips

logical shape:
4 chips on the short X axis × 32 chips on the long Y axis
```

Each Galaxy contributes one physical 4×8 segment. Staggered 4×4 subtori may lie within a Galaxy or
span a Galaxy boundary; their 4×4 wrap links remain physical links in the complete 4×32 system.
Galaxy-local 4×8 wraps and the complete 4×32 wrap likewise remain physical edges. For routing, the
relevant edge classes are:

- ordinary X direct/wrap edges forming one four-chip X ring at each Y;
- ordinary Y direct edges, including Galaxy-boundary links;
- the complete 4×32 Y wrap;
- retained 4×4 wraps, encoded as ex4 Z chords;
- retained 4×8 wraps, encoded as ex8 Z chords.

The logical labels in the following fixture are Y coordinates within one fixed-X column. The same
32-node Y graph is replicated independently at each of the four X coordinates. Hardware-local chip
labels and logical coordinates are distinct; §8.2 defines the pinning requirement that makes the
logical layout deterministic.

Cardinal end-wrap availability is a physical property of the selected carve-out, not a ring-domain
decision:

```text
standalone [8,4]:                 7↔0 exists and is cardinal
two-Galaxy quad carve-out [16,4]: 15↔0 does not exist
three-Galaxy carve-out [24,4]:    23↔0 does not exist
full four-Galaxy [32,4]:          31↔0 exists and is cardinal
```

The two- and three-Galaxy fixtures are guaranteed by deployment/physical grouping to contain
consecutive Galaxies. This routing stack consumes the resulting logical carve-out; it does not
select the participating hosts or define how the deployment layer forms that grouping.

§6.3 derives the protected ring families from these physical graphs. A missing end wrap does not by
itself prove that any replacement cycle is reachable or safe.

### 1.3 Validated four-Galaxy fixture: one 32-node Y graph per fixed X

For each fixed X position:

```text
base ring:
0↔1↔2↔…↔30↔31↔0

ex4 chords:
2↔5, 6↔9, 10↔13, 14↔17,
18↔21, 22↔25, 26↔29, 30↔1

ex8 chords:
0↔7, 8↔15, 16↔23, 24↔31
```

The routing roles are:

```text
ex4:
{1,2,5,6,9,10,13,14,17,18,21,22,25,26,29,30}

ex8:
{0,7,8,15,16,23,24,31}

leaves:
{3,4,11,12,19,20,27,28}
```

The membership count is:

```text
16 ex4 nodes + 8 ex8 nodes + 8 leaves
= 24 + 8
= 32 nodes
```

The physical Y-edge count per fixed X is:

```text
32 base edges + 8 ex4 chords + 4 ex8 chords
= 40 + 4
= 44 undirected edges

44 × 2 directions
= 88 directed channels
```

### 1.4 Validated four-Galaxy logical ring orders

Applying the canonical-orientation rule in §6.3 to the validated fixture produces:

```text
ex4:
[1,2,5,6,9,10,13,14,17,18,21,22,25,26,29,30]

ex8:
[0,7,8,15,16,23,24,31]

X at every fixed Y:
[0,1,2,3] in forward E order
```

Each Y ring alternates between one selected base edge and one express chord. The X ring uses the
ordinary `3→0` wrap. Reversing a listed order defines the opposite directed ring. The fixed
exact-half-ring tie rule is:

```text
ex4 8-vs-8 tie: choose the listed forward order
ex8 4-vs-4 tie: choose the listed forward order
X    2-vs-2 tie: choose the fixed forward E orientation
```

These lists are reference outputs of topology-driven synthesis, not production inputs. The canonical
orientation and tie rule are fixed policy; a topology, coordinate mapping, or pinning change that
derives another order requires regeneration and validation.

---

## 2. Direction and edge-role contract

### 2.1 Required command meanings

| Command | Required meaning | Edge class |
|---|---|---|
| N | One ordinary base-edge step toward decreasing logical Y / upward in team diagrams | Y-direct/base |
| S | One ordinary base-edge step toward increasing logical Y / downward in team diagrams | Y-direct/base |
| Z | One intramesh ex4 or ex8 express hop | Y express/chord |
| E | One X-ring step in one global X orientation | X-direct/wrap |
| W | One X-ring step in the opposite global X orientation | X-direct/wrap |
| drain | Deliver locally | Local |

Logical rows `0,1,...,31` run from top to bottom in the usual team topology diagrams. S increases
logical Y, including the `[32,4]` wrap `31→0`; N decreases logical Y, including `0→31`. E increases
logical X around the four-chip ring and W decreases it. If an existing host enum uses another name
convention, the topology/ABI boundary must translate it consistently; the canonical route set and
packet constants may not disagree.

### 2.2 Z is not a domain transition

In this design, Z describes a physical edge class. It does not mean:

- enter a new protected ring;
- leave a protected ring;
- cross a logical mesh boundary;
- change VC;
- recompute a route;
- apply the two-slot injection gate.

Those are independent route and transport actions.

A Z hop can be:

- same-ring transit;
- the first protected ring resource after a leaf entry;
- source injection from a worker;
- final movement toward a local destination;
- an intramesh output while an unrelated cardinal output is intermesh-facing.

### 2.3 The initial contract satisfies the one-express-neighbor constraint

The ex4 and ex8 memberships are disjoint:

- every ex4 node terminates exactly one ex4 chord;
- every ex8 node terminates exactly one ex8 chord;
- every leaf terminates no express chord.

Therefore each chip has at most one logical intramesh express neighbor in the exact `[32,4]` fixture.
The induced `[8,4]`, `[16,4]`, and `[24,4]` fixtures preserve the same property. A local bare Z
command therefore identifies one logical edge in every baseline fixture.

Several physical Ethernet lanes may realize that edge on one or more routing planes; they do not
create additional logical next-hop choices. A topology with multiple logical express edges or
neighbors at one node is rejected by the initial contract.

### 2.4 Intermesh remains an edge capability

Current software often gives Z special intermesh treatment. That cannot remain an implicit rule.

The route/builder contract must keep these properties separate:

```text
output identity:
N / S / E / W / Z

intra-mesh role:
ordinary transit / ring acquisition / ring ejection

mesh-boundary action:
remain intramesh / cross intermesh

transport action:
VC selection / BFC role / drain / route recomputation
```

For the initial Galaxy contract, Z may be intramesh express while a cardinal edge is intermesh-facing.
Existing non-express deployments may continue using a Z-facing edge for intermesh if its topology
capability says so.

---

## 3. Required baseline route function

This section is the complete reference oracle for the validated `[32,4]` fixture. It defines the
expected result used by assessment enumeration and fixture regression; it is not a table or node array
loaded by ControlPlane / RT-gen setup. Setup derives the ring mapping and routes from the
topology during fabric setup as described in §6. Every validation result, table entry, packet vector,
multicast tree, and BFC role in this document must match this reference route set.

The reference is complete rather than a collection of examples because the route, SCC, suffix,
multicast, and BFC results depend on the exact canonical all-pairs route set. Implementations may
encode the same policy algorithmically and are not required to use the presentation below as their
internal representation.

During setup, ControlPlane / RT-gen derives ex4/ex8 crossovers as ordinary base edges whose endpoints belong
to different synthesized ring families. It orders them by the ex8 endpoint's position in the
canonical forward ex8 order, with the directed endpoint tuple as the final tie-breaker. For the `[32,4]`
reference fixture this produces:

```text
0↔1, 7↔6, 8↔9, 15↔14,
16↔17, 23↔22, 24↔25, 31↔30
```

In each displayed pair, the ex8 endpoint is listed first and its ex4 partner second.

Route setup likewise identifies adjacent endpoint-only leaf pairs and each leaf's unique adjacent
transit node as its anchor. For the reference fixture this produces:

```text
3↔4,   anchors 3→2 and 4→5
11↔12, anchors 11→10 and 12→13
19↔20, anchors 19→18 and 20→21
27↔28, anchors 27→26 and 28→29
```

For a ring order, `distance(a,b)` means the shorter forward/reverse walk; an exact tie uses the
canonical forward order derived by §6.3. Section 1.4 records that order for the `[32,4]` reference
fixture. The Y oracle is:

1. **Same ex4 or same ex8:** take the shortest walk within that ring.
2. **ex4→ex8:** use the crossover paired with the ex8 destination. Walk ex4 to the paired ex4 node
   and cross one ordinary base edge. The landing is terminal in Y.
3. **ex8→ex4:** evaluate every ordered crossover `(a,b)` from ex8 node `a` to ex4 node `b` and choose
   the lexicographically minimum tuple:

   ```text
   (
       remaining ex4 distance from b to destination,
       ex8 distance from source to a + 1 crossover hop + remaining ex4 distance,
       crossover-list index
   )
   ```

   Walk ex8 to `a`, cross to `b`, acquire ex4, and continue only within ex4. This is the required
   late-exit rule.

   The second key is source-referential and is not claimed to be suffix-preserving for arbitrary
   synthesized multi-family topologies. In V1 this rule is used only by the `[32,4]` two-family
   fixture, whose complete reachable-intermediate-state suffix sweep passes `992/992` (Appendix A.4).
   The `[8,4]`, `[16,4]`, and `[24,4]` fixtures use one spanning family and do not invoke this rule.
   V1 has no alternate route representation if this sweep fails; supporting such a topology requires
   a route-policy or ABI redesign under §10.

4. **Source leaf:** use its anchor before applying the corresponding non-leaf route class, except
   when the destination is its paired leaf.
5. **Destination leaf:** route to its anchor, then take the ordinary base edge to the leaf.
6. **Paired leaves:** use their direct ordinary base edge. Other leaf pairs use source anchor,
   constrained non-leaf routing, and destination anchor.
7. **Cross-X:** complete the Y route at the source X coordinate, including terminal leaf
   movement. Then take the shortest direction around the four-chip X ring at the destination Y,
   using forward E for an exact 2-vs-2 tie. Never return to Y.

The oracle is deterministic because crossover order, ring orientation ties, X ties, leaf anchors,
and terminal behavior are all fixed. These are routing-policy choices; the exact crossover and anchor
identities are derived from the topology and compared with this reference for the `[32,4]` fixture.

### 3.1 Physical-edge-to-command mapping

For every selected directed edge `(u,v)`:

```text
if {u,v} is an ex4 or ex8 chord:
    emit Z
else if {u,v} is a Y base edge:
    emit N or S from the global cardinal orientation
else if {u,v} is an X edge:
    emit E or W
else:
    reject the route
```

This maps the selected physical hop to a command after the route is selected. It does not use command
letters to discover the route.

### 3.2 Global phase machine

```text
source/leaf entry
    ↓
Y phase: (N | S | Z)*
    - same-ring walk
    - at most one ex4/ex8 crossover
    - no leaf transit
    ↓
X-ring acquisition
    ↓
X phase: (E | W)*
    ↓
drain
```

Forbidden:

```text
E/W → N/S
E/W → Z
ex8 → ex4 → ex8
ex4 → ex8 → ex4
leaf → leaf → ring transit
adaptive ring-orientation reversal
runtime degraded reroute
```

### 3.3 Representative route traces

#### Same ex4

```text
physical: 1 → 2 → 5 → 6 → 9 → 10
command:      S   Z   S   Z   S
```

The command sequence exposes which selected physical edges are ordinary base edges and which are
express chords.

#### Same ex8

```text
physical: 0 → 7 → 8 → 15
command:      Z   S    Z
```

#### ex8→ex4

```text
physical: 31 → 0 → 1 → 2
command:       S   S   S
role:        ex8  cross ex4
```

The three commands are all cardinal S, but they have different domain roles:

- `31→0` is same-ex8 transit after initial injection;
- `0→1` is a crossover/ejection edge;
- `1→2` is the first continuing ex4 cyclic resource.

This demonstrates why injection cannot be inferred from a direction change.

#### Leaf→ex4

```text
physical: 3 → 2 → 5 → 6 → 9 → 10
command:      N   Z   S   Z   S
role:       entry ex4 transit...
```

The first protected ex4 resource is `2→5`, even though the packet entered the anchor over `3→2`.

#### ex4→ex8, terminal in Y

```text
physical: ... → 6 → 7
command:           S
role:         final Y crossover
```

The route may drain at Y=7 or enter the X ring. It may not continue around ex8.

---

## 4. Edge-level CDG and SCC result

### 4.1 Command encoding preserves physical resources

Let:

- `R_physical(s,d)` be the physical path selected by the §3 oracle;
- `encode_cardinal_z(edge)` map each physical edge to N/S/Z/E/W under §3.1;
- `R_encoded(s,d)` be the resulting command-labelled path.

By construction:

```text
physical_edges(R_encoded(s,d)) = R_physical(s,d)
```

for every source/destination pair in the baseline fixture.

A CDG node represents a directed resource, not a command letter. Encoding a selected physical chord
as Z does not add or remove a dependency.

Therefore:

```text
CDG_edges(R_encoded) = CDG_edges(R_physical)
```

at the directed-chip-edge abstraction.

### 4.2 Validated Y-only result

Exhaustive enumeration of the §3 oracle gives:

```text
route count:              32 × 31 = 992
used directed Y channels: 88
nontrivial SCC sizes:     [16,16,8,8]
cyclic directed channels: 48
directed diameter:        10
total route hops:         4,576
mean hops:                4,576 / 992
                         = 143 / 31
                         = 4.6129032258…
```

This result belongs only to the exact §3 canonical route set. A new greedy Z policy does not acquire it
merely by using the same command vocabulary.

### 4.3 Validated Y-then-X composition

For the assessed 4×32 composition:

```text
Y SCCs:
4 SCCs per fixed X × 4 X positions
= 16 SCCs

X SCCs:
1 selected directed four-chip cycle per Y position × 32 Y positions
= 32 SCCs

total:
16 + 32
= 48 nontrivial SCCs
```

No route may introduce a dependency from an X resource back into a Y N/S or Z resource.

### 4.4 Abstraction and implementation boundary

The edge-level identity does not prove:

- concrete software sender/receiver queue expansion;
- VC and routing-plane expansion;
- local router-to-router queues;
- multicast branch-and-wait dependencies;
- generic VC1 all-to-all;
- optimized paths that bypass normal admission;
- runtime degraded rerouting.

§5 explains why VC0 is deadlock-free when the required queue structure and BFC rules are followed.
Before claiming implementation support, the builder-derived queue graph and FabricBuilder
implementation must satisfy the §11 checklist. Arbitrary-pattern VC1 support requires the separate
§5.7.5 safety closure.

---

## 5. Bubble flow control and VC0 safety argument

Sections 5.1–5.6 define the required BFC and injection contract. Section 5.7 gives the complete
conditional VC0 paper proof for the constrained route set and states the remaining
implementation and VC1 obligations.

### 5.1 The protected ring is not a command direction

An ex4 forward cycle is:

```text
1→2 [S]
2→5 [Z]
5→6 [S]
6→9 [Z]
...
29→30 [S]
30→1 [Z]
```

The reverse ex4 cycle alternates N and reverse-Z resources. ex8 follows the same pattern with eight
resources per directed ring.

The invariant is one bubble in each complete directed ex4/ex8 cycle per VC and routing plane. It is
not:

- one independent bubble for all N outputs;
- one independent bubble for all S outputs;
- one independent bubble for all Z outputs.

### 5.2 Occupancy effects

The existing conceptual BFC rules remain valid:

```text
ring acquisition:
downstream free slots before admission ≥ 2
net protected-ring population change = +1

same-ring transit:
downstream free slots before movement ≥ 1
one old protected receiver is released
net protected-ring population change = 0

destination/ejection drain:
net protected-ring population change = -1
```

The command letter does not determine which row applies. The assessment route oracle can classify
each route occurrence as entering a protected ring, remaining in it, leaving it, or taking a terminal
landing. In production, FabricBuilder derives the corresponding total local effect from connectivity
and ControlPlane's node/direction protected-ring predicates; it then maps that effect onto the
concrete software sender and emits the static flag.

### 5.3 Required injection binding

Injection binds to the producer→sender path whose immediate downstream receiver is the first cyclic
resource of the acquired ring.

Example:

```text
31→0 [ex8]
0→1  [cardinal crossover]
1→2  [first ex4 cyclic resource]
```

The protected admission check is:

```text
free(Q(1→2)) ≥ 2
```

It is not sufficient to guard only `Q(0→1)`.

Same-ex4 transit from the opposite producer remains:

```text
30→1 [ex4 chord]
1→2  [ex4 base]

free(Q(1→2)) ≥ 1
```

The two producer paths may share a physical TXQ. FabricBuilder is responsible for mapping them onto
correctly classified software sender channels and supplying the per-sender compile-time flags. This
is the same responsibility in both command schemes.

### 5.4 Z sender can be injection or transit

At node 2:

```text
1→2 [ex4 base] → 2→5 [ex4 chord]
classification: same-ex4 transit
gate: free(Q(2→5)) ≥ 1
```

but:

```text
3→2 [leaf entry] → 2→5 [first ex4 chord]
classification: ex4 acquisition
gate: free(Q(2→5)) ≥ 2
```

No direction is globally injection. In particular, Z is not default-injection. FabricBuilder derives
the protected-domain occupancy effect locally from the frozen ControlPlane surface and realizes the
corresponding sender role:

- the leaf-produced `3→2 → 2→5` path is injection;
- the ex4-produced `1→2 → 2→5` path is transit.

The kernel does not need to infer either role from Z.

Here, **producer** means the immediate local producer feeding the outgoing sender, not necessarily
the packet's original source. If the packet originates at node 1, the complete assignment is:

```text
node 1 local worker → 1→2 [first protected ex4 resource]
classification: ex4 acquisition
gate: free(Q(1→2)) ≥ 2

1→2 [occupied ex4 resource] → 2→5 [next ex4 resource]
classification at node 2: same-ex4 transit
gate: free(Q(2→5)) ≥ 1
```

The route therefore acquires ex4 exactly once, at `1→2`. At node 2, the immediate producer of the Z
sender is the local ingress carrying `1→2`; the original worker on node 1 does not feed node 2's
worker sender channel.

### 5.5 Repeated Z is not repeated domain acquisition

A same-ex4 route can legitimately contain several Z commands:

```text
1→2 [S] → 5 [Z] → 6 [S] → 9 [Z] → 10 [S]
```

Both Z hops are same-domain transit. This does not create the repeated-domain sequence:

```text
ex8 → ex4 → ex8
```

Route validation and BFC classification must track protected domain state separately from command
letters.

### 5.6 ControlPlane and builder responsibility

The intended responsibility boundary is:

1. Routing generation identifies protected physical rings, both oriented edge views, their
   dimensions, membership, and permitted cross-domain transition policy.
2. ControlPlane exposes only builder-facing connectivity, express-enable state, and the
   node/direction protected-ring predicates defined by the builder contract. Ring IDs, ordered
   cycles, membership maps, and transition-policy storage remain internal.
3. FabricBuilder enumerates its own wired producer/ingress→egress paths and derives a total
   `NON_RING`, `REMAIN`, `ENTER`, or `NON_CANONICAL` effect for each one. It does not query
   precomputed ENTER/REMAIN rows, domain-effect records, or `list_local_transitions`.
4. FabricBuilder verifies that every concrete sender has one BFC guard class, then emits
   `sender_channel_is_traffic_injection_channel`.
5. The kernel reads the builder-provided flag and chooses `free ≥ 2` for injection or `free ≥ 1` for
   transit.

The kernel therefore remains unaware of ring identity and does not classify N/S/Z commands.

The builder-facing routing configuration does not contain a builder-shaped forwarding-arc record,
software-channel layout, canonical local-transition list, or route-occurrence effect table. The
builder contract owns the exact local derivation, wiring, allocation, and U-turn realization.

`ComputeMeshRouterBuilder::compute_sender_channel_injection_flags_for_vc()` currently derives VC0
turn injection mainly from cardinal axis changes. The target design replaces that heuristic with the
builder-contract derivation from the frozen ControlPlane surface. For the cardinal encoding:

- S→S can be ex8→ex4 acquisition;
- S→Z can be same-ring transit or leaf/domain acquisition;
- Z→S can be same-ring transit;
- Z→E/W is X-ring acquisition after Y completion.

Representative required assignments are:

```text
physical producer→sender path       cardinal+Z commands        flag
0→1 → 1→2                           S→S                        injection
30→1 → 1→2                          Z→S                        transit
3→2 → 2→5                           N→Z                        injection
1→2 → 2→5                           S→Z                        transit
```

The command pair names packet hops. At the receiving node, the local ingress-port direction is the
opposite of the producing hop: for example, the N hop `3→2` feeds node 2's S-facing ingress slot
before the Z hop `2→5`.

Injection flags therefore need locally derived domain effects rather than the current axis-turn test.

This is an implementation and finite-validation gap, not an edge-level counterexample. Appendix A.6
defines an independent route-occurrence oracle and compares it with builder-local derivation without
making that oracle part of the production ControlPlane surface.

### 5.7 VC0 safety proof

The following subsections give the complete conditional paper argument for VC0 intramesh routing
under the required §3 canonical route set, for unicast and multicast. They close the edge-level
abstraction boundary in §4.4. This is not a machine-checked proof of one concrete build:
implementation of the required sender roles, five-output fanout, allocator checks, and builder
conformance remains mandatory.

#### 5.7.1 Scope, assumptions, and topology portability

The concrete SCC reduction is the frozen `[32,4]` result. ControlPlane / RT-gen route setup does not
transfer that SCC set to `[8,4]`, the two-Galaxy `[16,4]` carve-out, the
three-Galaxy `[24,4]` carve-out, or another synthesized topology. A generated topology inherits
only the proof method, not the `[32,4]` result. Its edge-level CDG must be checked independently to
re-establish the contraction, BFC, routing-plane, acyclic stratum-order, and multicast-fork premises
in §§5.7.2–5.7.4.

This proof:

- does not re-prove BFC on an isolated directed ring; it assumes acquisition `free ≥ 2`, transit
  `free ≥ 1`, at least two slots in every protected receiver, accurate completion credits, a valid
  initial bubble, fair service, and eventual endpoint drain;
- does not replace the implementation checklist for sender marking, receiver identity, capacity, or
  fast-path exclusion;
- does prove that the canonical route set's only cyclic dependency strata are independently protected
  directed ex4, ex8, and X rings, with no cyclic dependency between strata.

#### 5.7.2 Why the edge-level CDG is faithful

The assessment CDG uses one node per directed chip edge. Two architectural invariants preserve cycle
structure when concrete resources are contracted to that edge:

1. **Linear intra-edge pipeline.**

   ```text
   software sender → hardware TX queue → hardware RX queue → software receiver
   ```

   This chain has no internal back-edge. A deadlock cycle cannot hide inside one contracted edge; it
   must span at least two edge resources and is visible in the edge-level CDG.
2. **Independent routing planes.** A packet is born on one plane and remains there. Planes do not
   share packet buffers or credits. The proof therefore applies to one plane and replicates to every
   closed plane.

The supported deployment supplies one uniform, ordered routing-plane set across every participating
cardinal, express, and X edge. The routing design does not qualify TRACE, linking-board, QSFP, or
cluster-wrap lane counts independently. Its realization requirement is only to preserve the supplied
plane identity: plane `p` enters and leaves as plane `p`, and planes do not alias queues or credits.

#### 5.7.3 Unicast: stratified BFC

The acyclic condensation of the unicast CDG is:

```text
ex8 rings ⇒ ex4 rings ⇒ X rings ⇒ local drain

⇒ means "may wait on"; there are no back-edges
```

- ex8 may wait on ex4 through a CONTINUE crossover;
- ex4→ex8 is TERMINAL, so ex4 does not wait on ex8 transit;
- Y may wait on X after Y completion;
- X never waits on N/S/Z because DOR forbids X→Y;
- ex4 forward/reverse and ex8 forward/reverse are independent directed-ring SCCs, each with its own
  bubble per plane.

Termination follows bottom-up. Local drain terminates by assumption. X then advances using drain and
its own bubble; ex4 advances using only lower X/drain strata and its own bubble; ex8 advances using
only lower ex4/X/drain strata and its own bubble. No copy can wait forever because every cross-stratum
dependency descends and every cyclic stratum preserves its bubble.

#### 5.7.4 Multicast: atomic fanout and structural fork lemma

Multicast uses two distinct admission stages:

1. Receiver-side fanout is atomic with respect to local child queues.
   `can_forward_packet_completely()` checks capacity in every selected downstream software sender.
   The router clones only when all local child queues can accept a copy; otherwise it retains the
   parent receiver.
2. Each concrete sender independently applies its remote-receiver BFC gate in
   `run_sender_channel_step_impl()`: ACQUISITION requires `free ≥ 2`; TRANSIT requires `free ≥ 1`.
   Local child-queue capacity and first-level ACK do not replace this remote protected-receiver count.

The generalized N/S/E/W/Z fanout must preserve this division. If acquisition and transit uses of one
physical output would alias one static-role software sender, the fixed sender mapping is invalid and
FabricBuilder rejects the realization.

The multicast tree is safe under the canonical route set because:

- each node has at most one Z chord child;
- a node's chord and same-ring base edge reach opposite directed-ring neighbors, hence different
  SCCs and bubbles;
- therefore a multicast action has at most one child in any directed-ring SCC;
- every other child descends to a lower stratum: leaf drain, ex8→ex4 CONTINUE, ex4→ex8 TERMINAL, or
  an E/W tooth.

Within one directed-ring SCC, the multicast tree is a simple monotonic path rather than a fork. The
unicast ring-bubble argument therefore remains valid, and cross-stratum children are handled by the
§5.7.3 induction.

A crossover such as `Y=0` may have three Y children—`0→7` on ex8 forward, `0→31` on ex8 reverse, and
`0→1` toward ex4—but those children occupy three different SCCs. Likewise, the two sides of a
two-sided source fork are separate ring acquisitions and must use independently classified
`free ≥ 2` senders.

#### 5.7.5 Remaining implementation and VC1 obligations

VC0 implementation and conformance must still establish:

- route-set conformance to §3, including no leaf transit, ex4→ex8 terminal behavior, and at most one
  crossover;
- role-separated sender-side `free ≥ 2` / `free ≥ 1` admission for every unicast and multicast arc;
- five-output atomic local fanout and cloning;
- at least two packet slots in every protected receiver;
- no speedy, trimming, channel-remapping, or degraded path that changes the proved queue graph;
- use of the immediate protected ring receiver's completion count rather than an early-ACK credit;
- the standard BFC progress assumptions listed in §5.7.1.

VC1 is not treated as unusable while this obligation remains open. Between a pair of current
four-Galaxy meshes, two intermesh links connect the meshes and current workloads are predominantly
linear. The existing VC1 path has no bubble; that restricted operating envelope is retained without
claiming arbitrary all-to-all traffic. It is not, however, a proof for every possible traffic
pattern.

The remaining decision is whether to enable BFC on VC1 to support arbitrary cross-mesh traffic.
Before claiming that broader traffic envelope, the selected cabling's entry-restricted VC1 CDG must
either be shown acyclic for the allowed patterns or each cyclic VC1 domain must receive the
corresponding BFC treatment. Some bindings preserve a clean cut while others produce cyclic SCCs;
multi-mesh pass-through introduces additional dependencies. No VC0 result transfers automatically to
those arbitrary VC1 dependencies.

#### 5.7.6 VC0 conclusion

VC0 unicast and multicast deadlock freedom has a complete checkable paper argument:

```text
per-ring BFC
+ validated edge-level SCC reduction
+ linear-pipeline contraction
+ routing-plane independence
+ stratified termination
+ multicast structural fork lemma
```

The remaining VC0 work is implementation and conformance to those premises, not additional deadlock
theory. Existing restricted VC1 operation remains in scope; arbitrary cross-mesh traffic requires the
§5.7.5 BFC decision and safety closure.

---

## 6. Routing generation and canonical route planning

This section defines the route-setup changes for the initial-cut contract in §0.1. ControlPlane /
RT-gen derives the exact ring arrangement, attachments, crossovers, and routes from the logical and
physical topology; it does not select a hard-coded node array based on fixture dimensions. The `[8,4]`,
`[16,4]`, `[24,4]`, and `[32,4]` descriptions below are complete expected results for regression and
assessment. Normal setup applies the same synthesis rules and directed checks without consulting
fixture-oracle tables. Section 8 defines representation requirements; exact L1 and packet artifacts
are owned by `GALAXY_DEVICE_ROUTE_CODEC_CONTRACT.md`.

`GALAXY_CONTROL_PLANE_ROUTING_GENERATION_CONTRACT.md` owns the detailed production algorithm,
current-path integration, retained ControlPlane facts, and consumer handoffs for this section. The
assessment retains the route policy, required outcomes, proof premises, and regression oracles.

Sections 6 and 7 anchor system responsibilities to the current setup path without prescribing new
host-side types or internal storage. Component implementation details are owned by the contracts
indexed in Appendix B.

### 6.1 Existing setup path and required extensions

Sections 6 and 7 describe responsibilities and required results within the existing fabric setup.
They do not prescribe a new top-level generator, a separate finalization phase, or a reordering of
ControlPlane and FabricBuilder initialization.

The current pre-builder path has two ControlPlane initialization variants. The `ControlPlane`
constructor invokes the selected path and initializes `FabricContext` after that path returns:

- with an explicit MGD, `ControlPlane::init_control_plane()` constructs `MeshGraph`, runs physical
  system discovery, runs `TopologyMapper`, constructs `RoutingTableGenerator`, and refreshes
  intermesh connectivity;
- with auto-discovery, physical discovery precedes construction of the generated `MeshGraph`, after
  which topology mapping and routing-table generation follow;
- after either path returns, the constructor calls `initialize_fabric_context()`.

Later, `MetalEnvImpl::initialize_fabric_config()` calls
`ControlPlane::configure_routing_tables_for_fabric_ethernet_channels()`. That existing pass maps
logical directions to physical Ethernet channels, establishes the usable routing-plane view, and
converts `RoutingTableGenerator` output into per-channel routing tables. During later fabric firmware
initialization, `ControlPlane::write_routing_tables_to_all_chips()` writes device routing information
before `create_and_compile_tt_fabric_program()` constructs `FabricBuilder`.

The target design extends those existing points:

1. MGD/MeshGraph handling materializes validated intramesh express adjacencies as Z neighbors.
2. ControlPlane / RT-gen route setup derives the selected ring arrangement, canonical route relation,
   physical-domain facts, and the applicable §6.6 checks.
3. Existing ControlPlane physical-channel setup verifies the required edge/channel/plane realization
   when those bindings are available and exposes the §7 builder-facing facts.
4. Existing host/L1 setup derives the codec-owned indexed artifacts from the same route relation;
   FabricBuilder and the kernel consume their component-owned views.

These are implementation updates at existing ownership boundaries, not a requirement to create a new
orchestration layer or report object.

The logical results required by the design are:

1. §6.2 defines the input facts and existing logical connectivity used by route setup.
2. §6.3 defines the selected ring arrangement plus transition and edge-use semantics.
3. §6.4 defines physical realization checks performed when edge/channel/plane bindings are available.
4. §6.5 defines the complete annotated all-pairs route relation.
5. §6.6 defines the setup checks that reject an unsupported or inconsistent configuration.

After the target §6 updates, ControlPlane keeps routing identities consistent and makes connectivity,
express enable, and node/direction protected-ring predicates available to FabricBuilder. Each
consumer owns its derived tables, forwarding view, queue model, and other internal structures.
FabricBuilder owns concrete ERISC/Tensix allocation, connections, service
assignment, compile-time arguments, and bounded realization checks. Worker/edge setup and router
kernels consume codec-installed tables/packet state and builder-emitted values without reconstructing
routing decisions.

Checks run where their required logical or physical state is available. A reachability,
suffix-consistency, CDG, topology, physical-link, pinning, or internal route-setup failure rejects
the configuration. Row/column route uniformity comes from the supported homogeneous-topology input
but C15 still verifies the generated relation before L1 writes; a mismatch indicates a descriptor,
configuration, or implementation error and rejects setup. FabricBuilder likewise rejects a concrete
setup that lacks a required edge, channel, capacity, feature, or unambiguous sender guard. Appendix A
defines reference validation cases; it is not an additional runtime phase.

No consumer performs a second route search. Route artifacts are derived from the canonical route
relation; FabricBuilder separately derives wired local effects from the ControlPlane facts in §7.
ControlPlane does not prescribe consumer-internal layouts, and FabricBuilder does not infer domain
acquisition from topology enums or N/S/Z commands.

### 6.2 Inputs and logical topology generation

The existing ControlPlane setup depends on four groups of facts. They become available through the
current initialization and channel-configuration paths described in §6.1; this list is not an
execution sequence.

1. **MGD / MeshGraph topology intent**
   - mesh dimensions and LINE/RING boundary semantics;
   - target architecture and host-layout dimensions;
   - channel count and policy;
   - explicit express-connection endpoint pairs;
   - graph/intermesh connectivity;
   - optional logical-to-physical pinnings.
2. **Fixed initial routing contract**
   - ring-synthesis constraints and the selection rule for the discovered topology;
   - required transit-node inclusion;
   - DOR order;
   - legal domain transitions and terminal transitions;
   - late-exit and tie-breaking rules;
   - fail-stop behavior;
   - command/action vocabulary.
3. **Discovered physical system**
   - physical ASIC links;
   - Ethernet channels;
   - host ownership.
4. **Later physical-channel and routing-plane setup**
   - direction-to-Ethernet-channel assignments;
   - the deployment-provided ordered routing-plane set reflected by ControlPlane channel
     configuration;
   - any additional reservation state applied by existing FabricBuilder discovery.

The current MGD does not supply leaf/transit labels or ring-domain hints. Route setup derives the
required ring arrangement and endpoint behavior from existing intra-/intermesh connectivity,
logical coordinates, N/E/S/W/Z directions, express enable, and the routing contract. Exact Ethernet
channels and routing-plane instances come from physical discovery and later ControlPlane channel
configuration, not from the MGD.

Logical route selection does not require existing channel reservation work to move earlier. Physical
realization checks use the channel/plane state available when the current setup establishes each
binding; an inconsistent required binding rejects setup.

For ring synthesis, route setup must be able to distinguish every directed logical hop and recover:

- source and destination logical nodes;
- logical axis and N/S/E/W/Z command;
- whether the hop belongs to the selected ring arrangement or is legal only for crossover,
  attachment, or terminal use.

The initial cut does not require persistent per-edge capability or transit annotations. Existing
intra-/intermesh connectivity plus direction and express enable derive ordinary, express, and
intermesh distinctions. Sorted `(mesh, source, destination, direction)` tuples are sufficient when
temporary deterministic keys are useful. MeshGraph, ControlPlane, and RT-gen may instead use another
representation that produces the same route/domain outputs and validation results.

Direction is relative to the selected logical carve-out, not just physical endpoints. For example:

```text
standalone [8,4] torus:
    7↔0 is the ordinary cardinal Y wrap

two-Galaxy [16,4] carve-out from the quad setup:
    there is no 15↔0 link
    7↔8 is an ordinary cardinal inter-Galaxy edge
    0↔7 and 8↔15 are retained 4×8 wrap links and are Z express links

three-Galaxy [24,4] carve-out from the quad setup:
    there is no 23↔0 link
    7↔8 and 15↔16 are ordinary cardinal inter-Galaxy edges
    0↔7, 8↔15, and 16↔23 are retained 4×8 wrap links and are Z express links
```

The current `IntraMeshConnectivity` representation is indexed by destination chip and stores one
`RouterEdge`. The initial contract deliberately adopts the corresponding boundary: one logical
intramesh express adjacency per node, with any parallel lanes treated as physical realizations of that
same edge. Several independently routable logical edges between the same endpoint pair or several
express neighbors are rejected rather than compressed into bare Z.

### 6.3 Ring-domain synthesis

Ring domains are synthesized from the selected carve-out. They are not hard-coded ex4/ex8 node arrays,
and not every closed loop in the physical graph is a valid protected ring.

Route setup derives the required **ring arrangement**. An arrangement selects one or more
physical ring families and must satisfy:

1. within each selected physical ring family, every member node has exactly one selected predecessor
   and one selected successor;
2. following a family's selected edges returns to the starting node after visiting every member once;
3. no transit node belongs to more than one selected physical ring family;
4. the selected families together include every required transit node;
5. leaf/end-point-only nodes are excluded from transit;
6. every selected edge exists in the logical multigraph;
7. no selected family contains an undeclared branch, dead end, or premature repeated edge.

The initial selection rule is topology-driven:

1. Endpoint-only leaves are the interior base-path nodes bypassed by an ex4 chord that are not
   endpoints of another retained express edge. All other required Y nodes remain transit candidates.
2. When the selected Y graph has its cardinal end wrap, route setup forms one ring family per
   detected express-link class, using that class's express edges and the base edges required to close
   the cycle. A topology with one express class therefore produces one family; the `[32,4]` topology
   produces ex4 and ex8 families.
3. When the cardinal end wrap is absent, route setup requires one system-spanning family over the
   union of the express-link classes, using the retained express edges and selected base edges.
4. For a bidirectional physical ring, rotate both orientations to begin at the smallest logical node
   and call the lexicographically smaller node sequence the canonical forward orientation. This
   derives the displayed forward/reverse orders and deterministic half-ring tie behavior.
5. If these rules leave no valid arrangement or more than one indistinguishable arrangement,
   generation fails rather than loading a fixture-specific answer.

Passing these initial checks is not sufficient. The arrangement must satisfy the fixed synthesis rule
for the discovered topology, the §6.4 transition/physical preconditions, and §6.6 all-pairs
validation. For the supported fixtures, the expected derived results are:

1. `[8,4]`, `[16,4]`, and `[24,4]` produce the single system-spanning physical ring family shown
   below, with leaves excluded from transit.
2. `[32,4]` produces the ex4 and ex8 physical ring families in §1.4.
3. No valid arrangement, or more than one arrangement remaining after the fixed selection rule,
   fails generation.

The initial cut has no soft ring objective or fallback arrangement. Supporting another arrangement
or selection policy belongs in the §10 extension process.
Implementations may reject an invalid possibility as soon as a required check fails; how they perform
the search and represent candidates is owned by the ControlPlane / RT-gen route-setup path. The reference ring
sequences below are test expectations, not production inputs.

At the ring-synthesis → route-planning and physical-mapping boundary, downstream components must be
able to recover:

- a stable identity for each protected directed ring;
- its axis, orientation, and ordered node cycle, directed endpoint pairs, or equivalent successor
  relation;
- the corresponding reverse directed ring, whether stored directly or derived from the ordered edges;
- an optional node order for diagnostics;
- canonical route behavior for ring transit, continuing ring transition, terminal ring transition
  with its landing node, terminal attachment, and forbidden transit;
- the allowed transition relationships between rings.

Parallel physical channels remain later bindings to one logical hop. The semantic role names above
describe required route behavior, not required per-edge enum names. ControlPlane / RT-gen may use
ordered cycles, temporary sets, graph annotations, indexed tables, or another representation as long
as every downstream dependency can recover the same domain and route results.

Explicitly excluding an edge from transit is stronger than simply omitting it from a ring. It records
that the ring arrangement and safety argument depend on canonical routes never using that directed
edge for transit. For the two-Galaxy spanning arrangement, both directions of `6↔7` and `8↔9` are
excluded from transit; allowing them as off-ring transit can restore the `{6,7,8,9}` cross-ring cycle.

#### Partial-Galaxy carve-out policy

One physical ring family produces two directed protected-ring descriptions: the listed forward
orientation and its reverse. Each directed ring has its own BFC state on each usable routing plane.
Therefore “one spanning ring” below means one physical ring family, not one shared bidirectional bubble.

For the canonical contiguous two- and three-Galaxy carve-outs from the quad setup, the selected
logical Y range has no cardinal end wrap. The fixed initial contract requires one valid physical ring
to include every transit-eligible row:

1. synthesize a system-spanning cycle from the available cardinal and retained-wrap edges;
2. exclude leaves and every non-selected transit edge from ring transit;
3. require the complete canonical all-pairs route set and §6.6 checks to pass;
4. emit the forward and reverse directed domains;
5. reject the configuration if no such cycle exists.

This is not a blind `ring_count = 1` setting and is not degraded-routing fallback. The fixed contract
requires one system-spanning ring family; that is the derived valid result for the closed
`[16,4]` and `[24,4]` fixtures below. A missing end wrap alone cannot bypass edge existence,
reachability, CDG, suffix, or builder-conformance checks.

#### One-Galaxy `[8,4]` fixture reference

For a standalone torused Galaxy, `7→0` is cardinal. Replacing the leaf-only cardinal arc
`2→3→4→5` with the `2→5` express chord produces:

```text
0 → 1 → 2 → 5 → 6 → 7 → 0
S   S   Z   S   S   S
```

Rows 3 and 4 remain destinations, but are not ring-transit nodes. The complementary mathematical
cycle `2→3→4→5→2` is not a routing/BFC domain.

#### Two-Galaxy `[16,4]` fixture reference from the quad setup

The induced 16-row topology has no `15↔0` loopback. Its relevant edges include:

```text
cardinal:
    0↔1↔...↔15

ex4 wraps:
    2↔5
    6↔9
    10↔13

retained 4×8 wraps:
    0↔7
    8↔15
```

With rows `{3,4,11,12}` excluded from transit, the selected spanning ring is:

```text
0 → 1 → 2 → 5 → 6 → 9 → 10 → 13 → 14 → 15 → 8 → 7 → 0
S   S   Z   S   Z   S    Z    S    S    Z    N   Z
```

The reverse domain is:

```text
0 → 7 → 8 → 15 → 14 → 13 → 10 → 9 → 6 → 5 → 2 → 1 → 0
Z   S   Z    N    N    Z    N   Z   N   Z   N   N
```

This fixture establishes two requirements shared by the supported fixtures:

- ring orientation is not equivalent to one cardinal orientation; one ring can contain N, S, and Z;
- ex4/ex8 identify edge/link classes, not protected-domain identity. One protected ring can contain
  several express-link classes.

#### Three-Galaxy `[24,4]` fixture reference from the quad setup

The induced 24-row topology has no `23↔0` loopback. Its relevant edges include:

```text
cardinal:
    0↔1↔...↔23

ex4 wraps:
    2↔5
    6↔9
    10↔13
    14↔17
    18↔21

retained 4×8 wraps:
    0↔7
    8↔15
    16↔23
```

With rows `{3,4,11,12,19,20}` excluded from transit, the selected spanning ring is:

```text
0 → 1 → 2 → 5 → 6 → 9 → 10 → 13 → 14 → 17 → 18 → 21 → 22 → 23 → 16 → 15 → 8 → 7 → 0
S   S   Z   S   Z   S    Z    S    Z    S    Z    S    S    Z    N    Z    N   Z
```

The transit-node count is:

```text
24 total rows - 6 leaves
= 18 ring-transit rows
```

The reverse domain is:

```text
0 → 7 → 8 → 15 → 16 → 23 → 22 → 21 → 18 → 17 → 14 → 13 → 10 → 9 → 6 → 5 → 2 → 1 → 0
Z   S   Z    S    Z     N    N    Z     N    Z     N    Z     N    Z   N   Z   N   N
```

Both directions of `6↔7`, `8↔9`, `14↔15`, and `16↔17` are excluded from transit. The retained
`0↔7`, `8↔15`, and `16↔23` edges are Z express links in this carve-out; none is a cardinal end wrap.

#### Four-Galaxy `[32,4]` reference relationship

The full four-Galaxy fixture has the cardinal `31↔0` end wrap and uses the frozen ex4 and ex8 ring
families in §1.4 rather than the partial-carve-out spanning-ring policy. Its 24 transit rows are
partitioned into the 16-node ex4 family and 8-node ex8 family, with
`{3,4,11,12,19,20,27,28}` remaining leaves. The two physical ring families produce four protected
directed domains: ex4 forward/reverse and ex8 forward/reverse.

#### Complete reference route oracles for the scale fixtures

The displayed `[8,4]`, `[16,4]`, and `[24,4]` rings define complete validation oracles, not
production configuration tables or just example traces:

1. The listed forward ring order and its reverse are the only Y-transit ring family.
2. A route between two transit nodes uses the shorter orientation; an exact half-ring tie uses the
   listed forward order.
3. `[8,4]` leaves are `{3,4}`, with anchors `3→2` and `4→5`; `[16,4]` leaves are
   `{3,4,11,12}`, with anchors `3→2`, `4→5`, `11→10`, and `12→13`; `[24,4]` additionally has leaves
   `{19,20}`, with anchors `19→18` and `20→21`.
4. Paired leaves `3↔4`, `11↔12` where present, and `19↔20` where present use their direct base edge.
   Every other leaf route enters or leaves through its anchor, and no leaf is transit.
5. Complete Y before X. At the destination Y, use the `[0,1,2,3]` X order from §1.4, with E selected
   for an exact 2-vs-2 tie.
6. No omitted edge may carry transit. In particular, `[8,4]` excludes the complementary
   `2→3→4→5→2` cycle; `[16,4]` excludes both directions of `6↔7` and `8↔9`; and `[24,4]`
   additionally excludes both directions of `14↔15` and `16↔17`.

These rules determine every ordered source/destination route for the three fixtures. Their exhaustive
route, SCC, suffix, and table-encoding tests remain to be run; the oracle itself is closed.

Local 4×8 cycles remain visible in the physical graph, but they are not selected as the
system-spanning domain under the fixed contract.

### 6.4 Transition and physical preconditions

The ring, edge-use, and transition semantics from §6.3 must agree with the discovered physical
system. This requirement does not choose another ring arrangement, define new routing rules, or
require physical channel assignment to move ahead of the existing ControlPlane pass. The `[32,4]`
CONTINUE/TERMINAL policy is defined in §3; the single-domain scale-fixture expectations are defined
in §6.3. Their safety significance is explained once in §5.

Current setup establishes the needed physical view in more than one place: `TopologyMapper` resolves
logical nodes against the discovered ASIC topology, while
`ControlPlane::configure_routing_tables_for_fabric_ethernet_channels()` later assigns physical
Ethernet channels and the usable routing-plane view. The target checks are added where the
corresponding bindings already become available; no new handoff object is required.

The combined setup requires:

- every selected ring, attachment, crossover, and X edge exists physically;
- software preserves the deployment-provided plane index around each ring and across turns;
- every terminal landing maps to its declared logical destination;
- no nonexistent regular-grid edge, such as `15↔0` in the two-Galaxy carve-out, was synthesized;
- ordinary and express edges sharing endpoints remain distinguishable.

Uniform plane availability is a supported-deployment input, not another check in this list. The
remaining fail-fast checks do not replace the all-pairs reachability, terminal-use, and CDG checks in
§6.6.

### 6.5 Deterministic canonical route construction

Route setup consumes the synthesized ring arrangement and applies the fixed route policy to its
derived ring orders, attachments, and crossovers. The policy operations are the ones defined in §3:
same-ring shortest orientation with deterministic ties, constrained cross-domain routing with
late-exit and terminal behavior, leaf attachment handling, and Y-before-X DOR. It does not
load a fixture route table and does not perform another ring or policy search.

For every source/destination pair, route setup emits one deterministic annotated logical-hop path.
Fixture regression separately compares the generated route set with the §3 or §6.3 reference oracle.
At the route-planning → consumer boundary, downstream
components must be able to recover:

- the ordered source/destination hops and their N/S/E/W/Z commands;
- protected-ring membership for each cyclic edge and the permitted physical-domain transition policy;
- terminal-in-axis behavior.

As defined in §1.1, one such annotated path is a **canonical route** and their complete collection is
the **canonical all-pairs route set**. These facts are sufficient for the independent Appendix A.6
route-occurrence oracle; they are not a builder-facing table of precomputed local effects. Their
internal record types, storage, and APIs remain an implementation choice.

### 6.6 Route and representation validation

Before the corresponding device tables are written, route setup requires:

C1. every route hop references an existing logical and mapped physical edge;
C2. every route is deterministic and cycle-free;
C3. selected ring edges carry their synthesized domain identity;
C4. same-domain N/S/Z transitions remain transit;
C5. the CONTINUE acquisition graph is acyclic as required by §5.7.3;
C6. leaves are never transit;
C7. Y completes before X;
C8. terminal transitions do not continue in that axis;
C9. every required source/destination pair has one deterministic route over the derived topology;
C10. no route uses an edge that the ring arrangement excludes from transit;
C11. every TERMINAL crossing lands at its recorded destination and is never an intermediate hop;
C12. the generated CDG contains no cyclic edge outside the selected protected rings;
C13. every node has at most one logical intramesh express adjacency, and the selected action encoding
     identifies every local edge unambiguously;
C14. at every reachable intermediate state, the remaining route equals
     `R(current,destination)` as required by the suffix-consistency predicate in §8.2.
C15. the supported homogeneous-topology input yields row/column route uniformity: the generated Y
     relation is identical across X columns, the generated X relation is identical across Y rows,
     and all supplied routing planes use that same logical relation.

Checks C5, C9, C10, C11, and C12 validate the required ring arrangement. Any failure rejects the
configuration with the missing required node or route, forbidden edge, unexpected SCC, or unresolved
ambiguity.

Checks C13–C14 are initial-contract topology/representation gates. Multiple physical lanes for one
logical express edge are legal, but multiple logical express adjacencies at one node reject the
configuration rather than selecting another runtime encoding. C15 records the homogeneous-cluster
input and verifies the route-uniformity result expected from applying one deterministic policy to its
repeated ring structure. It is a mandatory logical setup check, not a new physical-link
qualification.

Check C12 is one required validity check, not the entire route-setup validation. The complete
edge-level CDG for the exact ring arrangement and canonical route set must also be checked. Consumers
do not rerun these checks. FabricBuilder applies only its bounded local realization guards; Appendix A
defines regression coverage for equivalence with the selected implementation.

Multi-domain ring arrangements are not intrinsically suffix-inconsistent, but each crossover increases the
chance that two packets at the same `(current node, destination)` require different next hops because
they arrived through different domains. Destination-only indexing is structurally simplest for a
single spanning domain; the validated `[32,4]` ex4/ex8 destination-vector result demonstrates that a
multi-domain arrangement can still satisfy it. Every arrangement must pass C14's
reachable-intermediate-state suffix check using the equality defined in §8.2. A failing arrangement
may still route safely, but it is outside the initial contract and cannot use the V1 compact-vector
ABI. V1 has no alternate representation; a different route policy or ABI is an extension under §10.

**Row/column route uniformity** means:

```text
project_Y(R((x, ys), (x, yd))) is identical for every valid x
project_X(R((xs, y), (xd, y))) is identical for every valid y
```

The supported cluster and ring layout guarantees these equalities: every X column repeats the same Y
structure, every Y row repeats the same four-chip X ring, and every supplied plane realizes the same
logical relation. C15 nevertheless compares the generated route projections before L1 writes to
catch asymmetric MGD input, configuration mistakes, or implementation drift. A mismatch rejects
setup. This does not attempt to admit nonhomogeneous topologies, and V1 has no coordinate- or
context-indexed fallback.

---

## 7. Routing configuration and consumer boundary

### 7.1 High-level routing configuration and central dependency ledger

RT-gen owns one canonical next-hop relation, and ControlPlane retains or can reconstruct the
logical routes and internal ring/domain facts needed to realize it. This central assessment owns
that relation, its ring/transition policy, the checks in §6, and the
following cross-layer dependency ledger. The ledger lists existing setup owners plus the required
updates at their current boundaries; arrows between rows are not a proposed execution order. It does
not introduce another API, storage layout, report, or consumer-private artifact. The detailed owner
named in the last column remains authoritative for its component surface.

| Existing owner / setup area | Existing input or required update | Consumer and required use | Check or failure behavior | Detailed authority |
|---|---|---|---|---|
| MeshGraph and topology mapping inside current ControlPlane initialization | Existing MGD/auto-discovery inputs plus the required materialization of validated intramesh express endpoint pairs as same-mesh Z neighbors; existing intra-/intermesh connectivity, logical coordinates, directions, express enable, and optional pinning | Current ControlPlane / RT-gen setup consumes the resulting logical graph and logical↔physical node mapping; no persistent per-edge capability product is required | Multiple logical express neighbors or an unrealizable pinned graph reject the configuration; explicit-MGD and auto-discovery initialization retain their existing order | `GALAXY_CONTROL_PLANE_ROUTING_GENERATION_CONTRACT.md`; this assessment, §§6.1–6.2 |
| ControlPlane / RoutingTableGenerator route setup | Fixed ring-selection rules, required transit nodes, Y-before-X DOR, legal/terminal transitions, late-exit/tie rules, fail-stop behavior, N/S/E/W/Z vocabulary, and the supported homogeneous-cluster input | Extend the existing route setup to derive the ring arrangement, physical-domain facts, and one deterministic canonical route relation; C15 verifies row/column route uniformity before L1 writes | Logical route, CDG/SCC, suffix, route-uniformity, or representation failures reject the configuration at the applicable §6.6 check; V1 has no alternate representation | `GALAXY_CONTROL_PLANE_ROUTING_GENERATION_CONTRACT.md`; this assessment, §§3 and 6.2–6.6 |
| Current ControlPlane physical-channel setup | Discovered ASIC links, Ethernet channels, host ownership, logical direction/edge requirements from route setup, and the uniform ordered plane set guaranteed by the supported deployment | `configure_routing_tables_for_fabric_ethernet_channels()` or its existing ownership area realizes direction→channel mappings and preserves supplied plane identity without changing route policy | Missing required links or inconsistent bindings reject setup; this routing design does not introduce a TRACE/linking-board/QSFP/cluster-wrap plane-homogeneity qualification | `GALAXY_CONTROL_PLANE_ROUTING_GENERATION_CONTRACT.md`; this assessment, §§6.1 and 6.4 |
| ControlPlane state consumed by FabricBuilder | Existing node↔physical mapping; neighbors, Ethernet channels, routing planes, direction↔peer/channel connectivity; add `express_routing_enabled` and the builder contract's local-node/ingress-direction/egress-direction protected-ring predicates | Builder derives edge capability, DOR-safe connection maps, a total local effect for every wired producer, concrete allocation, and BFC compile-time flags; internal ring IDs and transition-policy enums are not exposed | `express_routing_enabled` agrees with materialized MGD express neighbors; missing or inconsistent predicate results reject builder setup | `GALAXY_BUILDER_ROUTING_CONFIG_CONTRACT.md`, “Protected-ring predicate surface” |
| Canonical route relation used by codec/host setup | RT-gen-owned canonical logical next-hop relation with Z-visible logical hops, pinned logical coordinates/shape, suffix consistency, and row/column route uniformity from the supported homogeneous topology; ControlPlane exposes a logical next-hop or complete-route query, and may reconstruct complete `R(source,destination)` rather than storing each path | Codec setup derives destination-major vectors and performs its multicast arborescence/legality gates while deriving local-root reverse trees from the same relation | A channel-conditioned current `get_fabric_route` result is not a substitute for the canonical logical relation; a failed suffix or codec check rejects setup, with no V1 representation fallback | This assessment, §§6.5–6.6 and 8–9; `GALAXY_CONTROL_PLANE_ROUTING_GENERATION_CONTRACT.md`, §8; `GALAXY_DEVICE_ROUTE_CODEC_CONTRACT.md`, “Host generator / L1 embed / device loader (target contract)” |
| Existing host generation / ControlPlane L1-write path | Destination-major Y/X two-bit tables; deployment-wide indexed-2D ABI selection; express feature state; pinned shape and local `(my_x,my_y)`; retained `exit_node_table` and `inter_mesh_direction_table`; per-chip reverse tree after every root passes the mesh-wide arborescence gate | Workers and landing encoders widen routes and construct packet action maps; intermesh setup retains exit/boundary behavior | Any failed root rejects reverse-tree multicast for that mesh/configuration; V1 has no alternate encoder; `intra_mesh_direction_table` becomes removable only after indexed cutover | `GALAXY_DEVICE_ROUTE_CODEC_CONTRACT.md`, “What host must provide (device-facing)” and “Loader responsibilities” |
| Deployment/FabricContext policy → all 2D producers and consumers | One indexed-2D ABI selection plus route-buffer/header sizing for the supported dimensional bound | Worker, edge, builder, header typedefs, and kernel select one compatible ABI | Cutover is atomic across the 2D deployment; `express_routing_enabled` does not select the packet ABI | `GALAXY_DEVICE_ROUTE_CODEC_CONTRACT.md`, “Host generator / L1 embed / device loader (target contract)” and “Codec implementation checklist” |
| Fabric connection setup → source encoder/worker | Connection direction identity: manager slot direction tag or retained raw-sender `edm_direction` | Source fanout selects the actual connected root output; reroot fallback selects its inject edge | Direction is connection metadata, not a packet field, canonical-route field, or value inferred from multicast extents | `GALAXY_DEVICE_ROUTE_CODEC_CONTRACT.md`, “Same-mesh 2D express source multi-output injection”; `GALAXY_BUILDER_ROUTING_CONFIG_CONTRACT.md`, “Same-mesh express multicast source fanout” |
| FabricBuilder → ERISC/kernel | Concrete normal and same-link-return wiring; queue/channel allocation; `SENDER_CH_i_IS_INJECTION`; deadlock-avoidance and header/buffer compile-time arguments | Kernel performs local RX admission and applies the builder-selected BFC guard in sender-step | Every concrete sender has one guard class; capacity or ambiguous-role failure rejects builder setup | `GALAXY_BUILDER_ROUTING_CONFIG_CONTRACT.md`; `GALAXY_DEVICE_ROUTER_KERNEL_CONTRACT.md`, “BFC consumption” |
| Codec/loader → ERISC/kernel | Packet `route_buffer_y` / `route_buffer_x`, retained final destination/anchor and multicast extents, local coordinate indexing, multicast action maps, and packet/header ABI | Kernel decodes, atomically admits, fans out, forwards, and performs capability-aware intermesh transitions without reconstructing routes | Transit maps are immutable; codec owns byte meanings and kernel owns their execution | `GALAXY_DEVICE_ROUTE_CODEC_CONTRACT.md`; `GALAXY_DEVICE_ROUTER_KERNEL_CONTRACT.md` |
| Intermesh transition → continued intramesh routing on VC1 | Retained final destination/anchor and multicast extents; each router's current mesh identity; retained exit/intermesh-direction state; builder-wired `INTERMESH` egress and ingress capability plus cardinal and express-Z VC1 senders; landing-installed maps | At an exit, current/final mesh inequality plus the selected intermesh egress identifies the boundary hop. After landing, current/final mesh equality distinguishes destination from intermediate behavior: an intermediate installs a unicast-style segment to its next exit, while only the destination installs final unicast or multicast maps; both continue on VC1 | No new packet intermesh mode or boundary-direction field is required. Intermediate meshes must not begin final multicast fanout; there is no VC1→VC0 crossover. The current two-link, predominantly linear operating envelope is retained; arbitrary-pattern VC1 support requires a separate VC1 CDG proof or BFC treatment, including multi-mesh pass-through | This assessment, §5.7.5 and §11; codec and kernel intermesh-transition contracts; builder VC1 contract |
| Builder/kernel queue realization → VC0 proof premises | Role-separated sender guards; five-output atomic local fanout; at least two packet slots in each protected receiver; immediate receiver completion count | Concrete execution preserves the queue graph used by the §5.7 proof | Speedy, trimming, channel remapping, degraded routing, early-ACK credit, or another bypass may not alter the proved wait graph without a new proof | This assessment, §5.7.5; builder and kernel BFC/conformance contracts |
| Codec + builder + kernel → same-mesh reroot fallback | Codec pre-inject overlay; builder-owned dedicated VC0 same-link sender with worker-mirror guard; kernel-controlled full-packet same-link return | Applies only to same-mesh, worker-originated indexed 2D express multicast with an N/S/Z multi-output root when not every root output has a connection and one canonical root output is connected | It excludes 1D/sparse, intermesh source/landing, and X-only traffic; canonical U-turns remain forbidden; claiming fallback support requires the reverse-tree all-roots gate, full-packet producer boundary, differential execution, queue ordering, capacity, and BFC-role conformance | This assessment, §9.5; all three component contracts in their owned areas |
| Universal indexed-2D cutover → legacy-path retirement | All 2D producers/consumers stop writing or consuming `hop_index` / `branch_*`; both header-update paths and `UPDATE_PKT_HDR_ON_RX_CH` retire; profiler/debug decoding and host/device header layout move with the ABI | Worker, edge, builder, headers, profiler/debug tooling, and ERISC use only the indexed interpretation | No legacy/indexed pairing is valid; the deployment-wide cutover is one gate and includes non-express 2D configurations | Codec and kernel cutover contracts |

For one configured fabric instance, once the applicable setup checks pass, consumers must not change
the selected route relation until teardown or explicit reconfiguration. This does not require a new
immutable configuration object. Consumer-private tables, queues, channels, sender indices, kernel
arguments, and packet layouts are not added to the generic ControlPlane route state.

### 7.2 Ownership boundaries and prohibited dependencies

The ledger is completed by the existing negative requirements below. These prevent a consumer
implementation detail from becoming a second routing contract:

| Boundary | Required exclusion or invariant | Existing authority |
|---|---|---|
| ControlPlane → FabricBuilder | Do not expose internal ring IDs, ordered cycles, membership maps, transition-policy enums, precomputed ENTER/REMAIN rows or `list_local_transitions`, a heavy per-edge `edge_role` product, `domain_kind`, a mesh-wide SCC census, `route_family_to_vc`, sender indices/`IS_INJECTION`, or builder-shaped forwarding arcs; expose the builder contract's node/direction predicates, and let FabricBuilder derive effects only for producers its connection maps actually wire | Builder contract, “CP ↔ builder predicate surface and local derivation” |
| ControlPlane → codec | Codec does not consume protected-domain ENTER/REMAIN or builder sender roles; it consumes the canonical route relation and device-facing route artifacts | Codec contract, “ControlPlane and host handoff” |
| ControlPlane → kernel | No runtime ControlPlane query; kernel receives routing behavior only through codec-installed L1/packet bytes and builder compile-time arguments/wiring | Kernel contract, “Cross-document dependencies” |
| ControlPlane route state → all consumers | No consumer performs a second route search or changes the ring arrangement; a route, topology, CDG, representation, or physical-mapping failure rejects setup at the owning check | This assessment, §§6.1 and 7.3 |
| Direction vs capability/effect | N/S/E/W/Z selects an output; cardinal/express/intermesh capability selects transport behavior; protected-domain effect selects BFC; “has Z” alone does not imply intermesh or injection | This assessment, §§2 and 5; builder and kernel contracts |
| Express enable vs packet ABI | `express_routing_enabled` controls materialized express topology and associated Z/BFC/reroot artifacts; it never selects legacy versus indexed packet interpretation | Codec and builder contracts |
| Component-private realization | Builder forwarding maps/queues/channels, codec layouts/bytes, and kernel dispatch mechanics remain owned by their component contracts and are not generic ControlPlane route facts | Authority split in §0.4 and Appendix B |

Appendix A.6 may independently compute route-occurrence effects for regression testing, but that
test-only oracle is not a production ControlPlane query.

Component-private derivation and realization remain with the owners in §0.4; the ledger does not
transfer that authority.

The system-level capacity headline for the primary fixture counts logical firmware sender-channel
slots serviced by one router. It does not count Ethernet links, physical TX queues, or hardware
sender engines:

```text
logical sender maxima by VC       = 5 / 4 / 1
normal logical aggregate          = 5 + 4 + 1 = 10
required logical-channel ceiling  = 11
normal margin                     = 11 - 10 = 1

reroot logical maxima by VC       = 6 / 4 / 1
reroot logical aggregate          = 6 + 4 + 1 = 11
reroot margin                     = 11 - 11 = 0

receiver maxima by VC        = 1 / 1 / 1
receiver aggregate           = 1 + 1 + 1 = 3
receiver ceiling             = 3
receiver margin              = 3 - 3 = 0
```

These are logical array-count results, not concrete allocation proof. Detailed arithmetic, wiring,
`UTURN_REROOT`, sender/receiver layouts, and queue conformance are owned by the builder contract.

### 7.3 Setup checks and reference validation

ControlPlane / RT-gen setup checks topology, canonical routes, CDG/SCCs, representation, and physical
bindings at the existing points where the required state is available. This includes C15's generated
route-projection comparison before L1 writes. FabricBuilder then performs only bounded local
realization guards: required connectivity/capability exists, every wired producer has a total effect,
one concrete sender has one BFC class, capacities fit, and no implementation feature aliases or
removes a required path. A failure rejects builder setup; the builder does not choose another route or
ring arrangement.

This contract does not require plane/channel reservation or binding work to move to a new phase.
Existing ControlPlane channel configuration and FabricBuilder discovery may establish or reserve
their state at different points; the state consumed by a concrete router build must be consistent.
The existing generic Wormhole dispatch-plane reservation remains a separate builder concern and is
relevant to a BH express configuration only if that configuration consumes the affected resource.

Appendix A.6 defines an independent route-occurrence oracle that can compare outputs from the same
production route-generation and Builder logic. It does not prescribe a separate validation-only
implementation path. The oracle compares route-occurrence effects with builder-local derivation and
projection, requires zero mixed-guard aliases and zero DOR-forbidden connected arcs, and checks the
normal VC0 maximum of five. The temporary reroot profile raises VC0 capacity to six and has its own
producer-boundary, worker-mirror-guard, differential-execution, and queue-conformance gate.

The exact builder checks, allocation rules, BFC defines, `UTURN_REROOT` wiring, and local algorithms
are specified only in `GALAXY_BUILDER_ROUTING_CONFIG_CONTRACT.md`.

---

## 8. Route representation requirements

### 8.1 Current format is insufficient

The current 2D packet command mask has N/E/S/W bits and no inline Z bit. Current compressed routes
store one N/S segment and one E/W segment.

This design needs sequences such as:

```text
S, Z, S, Z, S, E, E
```

They cannot be represented losslessly as:

```text
N/S hop count + E/W hop count
```

The existing per-device direction table is also insufficient by itself. It stores only the first
direction from the local source to each destination. Chaining next-hop lookups would require a
persistent L1 table access at every hop or a much larger all-source table.

### 8.2 Destination-indexed routing vectors

The L1 relation is destination-major:

```text
y_vectors[destination_y][current_y]
x_vectors[destination_x][current_x]
```

For one fixed destination, each two-bit entry records the next canonical action from a possible
current coordinate:

```text
Y relation:
00 = Y phase complete
01 = N
10 = S
11 = Z

X relation:
00 = X phase complete / local destination
01 = E
10 = W
11 = INVALID
```

This relation is a semantic requirement, not the packet layout. Exact packing, placement, sizes,
alignment, loading, and compatibility bounds are codec-owned.

The compact relation has two mandatory logical setup predicates. Homogeneous topology is the input
reason C15 is expected to pass, not a reason to omit the check.

**Suffix consistency:**

```text
suffix of R(source, destination) beginning at current
= R(current, destination)
```

The healthy cardinal sweep passed all `1,044,640` route cases and `8,843,872` intermediate checks;
the frozen `[32,4]` express-aware Y sweep passed `992/992`. Appendix A.4 owns the complete oracle and
counters. Every declared fixture must pass its own sweep. V1 does not admit an arbitrary new shape
and has no alternate packet representation if suffix consistency fails.

**Row/column route uniformity:**

```text
project_Y(R((x, ys), (x, yd))) is identical for every valid x
project_X(R((xs, y), (xd, y))) is identical for every valid y
```

The supported deployment guarantees these equalities through homogeneous clusters, repeated ring
structure, and one logical relation shared by its supplied planes. C15 compares the generated
projections before L1 writes and rejects a mismatch. V1 does not perform generic topology admission
and has no coordinate/context-indexed fallback.

Indices are canonical logical mesh coordinates, never discovered physical chip IDs. The MGD must pin
enough nodes to eliminate routing-relevant rotations/reflections and fix X/Y orientation, host and
Galaxy boundaries, wrap placement, and express endpoints. Complete per-node pinning is unnecessary
when the remaining embedding is unique. Mapping fails if the pinned logical graph cannot be realized
on the required physical links.

### 8.3 Device ABI boundary

The working ABI widens selected L1 two-bit vectors during setup into immutable Y and X action-byte
maps carried by the packet. Intramesh routers index those maps by pinned local coordinates.
Intermesh transport compares the retained final mesh with the router's current mesh and uses
builder-provided `INTERMESH` edge capability to identify boundary egress and landing. It adds no
packet routing-mode or boundary-direction field, and direction letters do not imply a mesh-boundary
transition.

The following assessment-era sketches are rejected and are not alternate implementation paths:

- a packed two-bit packet overlay or transit `extract_2bit`;
- `execution_state`, `E_LINE`, `W_LINE`, and `remaining_hops`;
- route-buffer sizing from `max(packed-vector bytes, Y-only multicast bytes)`;
- treating old packet/header capacity numbers as current authority; codec §2.12 closes the
  vector-only L1 fit, codec §6.2 requires explicit `[64,4]` hybrid placement or aligned struct
  growth, and codec §4.7 owns packet-capacity and current axis-limit work.

`GALAXY_DEVICE_ROUTE_CODEC_CONTRACT.md` is authoritative for action-byte layout, dual maps, exact
fields, sizes, loading, encode/decode semantics, multicast artifacts, and source fanout/reroot
overlay. `GALAXY_DEVICE_ROUTER_KERNEL_CONTRACT.md` is authoritative for ERISC execution and
header-mutation retirement.

---

## 9. Multicast semantics and safety

### 9.1 Client target semantics are preserved

The client meaning of an N/S multicast range remains geometric. For example:

```text
source 1, client S=4
target Y geometry: {2,3,4,5}
```

Target-row construction follows the configured Y boundary semantics. On a Y torus, N/S offsets wrap
modulo Y. On a non-torus Y dimension, an extent that crosses an end is invalid and does not wrap.
Simultaneous N/S ranges form the set union of their intermediate geometric rows. The initial X
dimension is the ordinary four-chip ring; codec §5.8 owns its modulo-X target-column construction.

With `E=3, W=3`, each selected Y row also requests the existing E/W line branches. There are no
E/W express links in the assessed physical topology, so the E/W branch suffix remains cardinal.

The design must not reinterpret `S=4` as four logical ex4 ring steps:

```text
1→2→5→6→9
```

This avoids the client/fabric N/S semantic break identified for ring-domain N/S.

### 9.2 Why direct cardinal execution is not safe

Preserving the target contract does not prove that the current compact multicast execution is safe.
A direct cardinal chain:

```text
1→2→3→4→5
```

uses leaves 3 and 4 as transit and introduces a Y dependency path outside the constrained unicast
route set. In combination with the existing ex4 forward ring, it can add the cycle:

```text
1→2 → 2→3 → 3→4 → 4→5 → 5→6
 ↑                                  ↓
 └────── ex4 path 6→…→30→1 ────────┘
```

Thus “do not use express links for multicast” is not enough. It can introduce cardinal/leaf
dependencies that merge into the protected unicast SCCs.

### 9.3 Selected canonical-route union

The selected behavior is one Z-aware, source-reachable union/tree of canonical routes from the encode
root to every cardinally named target. It preserves the client target set while ensuring every
root-to-target path is a §3 route. For `source=1, S=4`, the target rows are `{2,3,4,5}` and the Y
union is:

```text
1→2
   ├─S→3
   └─Z→5→N→4
```

The selected route union, not a direct cardinal chain and not an OR of complete unreachable vector
states, defines multicast routing semantics. Exact action-byte layout, dual Y/X maps, reverse-tree
primary encoding and its mesh-wide arborescence gate, E/W X-map construction, and source injection
mechanics are owned by `GALAXY_DEVICE_ROUTE_CODEC_CONTRACT.md`.

### 9.4 Legality and validation

Every encoded multicast artifact must satisfy:

1. every root-to-target path is the corresponding canonical unicast route;
2. no branch reverses a protected-ring orientation or violates ex8→ex4 CONTINUE /
   ex4→ex8 TERMINAL ordering;
3. leaves remain non-transit and E/W children never return to N/S/Z;
4. source dispatch reaches every selected root child exactly once;
5. execution has no reconvergence that duplicates delivery or adds an unmodeled dependency;
6. atomic local fanout and each concrete sender's independently derived BFC role preserve the §5
   structural-fork proof.

Appendix A.2 retains the `992/1,984` restricted-codec counterexample result. Appendix A.3 retains the
`16,864/16,864` indexed-action validation result, including zero duplicate-arrival, cycle,
multi-context-row, and terminal-violation cases. These are assessment model results, not device
implementation evidence.

### 9.5 Source fanout and reroot boundary

Connection-manager source multi-inject is preferred when all canonical root outputs are available.
The temporary pre-inject reroot overlay is transport-only: it does not alter the canonical route
oracle, and canonical immediate U-turns remain forbidden.

For the fallback, the controlled same-link return must cross a full-packet producer boundary: child
RX commits an immutable copy and releases the incoming receiver before the dedicated sender waits on
remote credit. That sender mirrors the corresponding worker-output BFC guard. Subject to
differential execution and concrete queue/producer-boundary conformance, these invariants preserve the
current VC0 proof rather than creating a new route family. Overlay construction and source fanout are
codec-owned; sender allocation is builder-owned; same-link execution is kernel-owned.

---

## 10. Alternative and extension decision record

### 10.1 Command decision provenance

The principal alternative encoded protected-ring successor/predecessor movement as ring-local N/S
and used Z for transitions. When it selects the same physical routes, it has the same edge-level CDG,
BFC domains, and producer roles. It was not selected because N/S would cease to mean global cardinal
geometry, cardinal multicast targets would require an extra translation contract, paired-leaf
movement would need another rule, and diagnostics would need ring context to interpret directions.

Cardinal N/S/E/W plus express Z preserves client geometry and names physical outputs literally.
Command vocabulary alone proves neither deadlock freedom nor implementation conformance; those come
from §§3–6 and the component requirements.

### 10.2 Extension boundary

Opportunistic Z, enabling the Y nearest-neighbor base-32 cycle, multiple logical express
neighbors, adaptive route choice, and degraded routing do not inherit this proof. They can add leaf
transit, repeated or reversed domain acquisition, overlapping cyclic resources, or
context-dependent next hops.

Every such extension must define and validate a new canonical route relation, directed-resource
CDG/SCC inventory, queue/VC/plane contraction or expanded proof, BFC classification, full Y→X
composition, suffix consistency, row/column route behavior or a replacement representation,
multicast legality, and VC1 scope. Reusing N/S/E/W/Z names is not evidence that the baseline result
transfers.

---

## 11. Implementation checklist

The target design is not implemented. `GALAXY_WORKING_MODEL.md` remains authoritative for the current
physical and software baseline, §0.4 and Appendix B define ownership, and the component contracts
define detailed implementation requirements.

Before claiming support for this target, the implementation must satisfy all of the following:

1. every `[8,4]`, `[16,4]`, `[24,4]`, and `[32,4]` fixture passes its route, topology, CDG/SCC,
   suffix-consistency, and mandatory row/column route-uniformity setup checks;
2. multicast source-reachable route-union legality, every-root reverse-tree validation, and the
   Appendix A.2/A.3 regressions pass; any failed reverse-tree root rejects the mesh/configuration;
3. the Appendix A.6 builder projection has zero effect mismatches, zero sender-role aliases, zero
   unmapped/multiply mapped transitions, and zero DOR-forbidden connected arcs;
4. each component contract is satisfied, including at least two slots in every BFC-protected
   receiver and no unproved trim/speedy/bypass behavior;
5. before expanding VC1 beyond the current two-link, predominantly linear operating envelope, a
   separate VC1 dependency proof or BFC treatment exists for the broader traffic patterns;
6. when reroot fallback is enabled, differential execution, full-packet producer-boundary,
   worker-mirror sender-guard, and queue-ordering checks pass;
7. one atomic indexed-ABI cutover covers producers, L1 artifacts, edge handling, and kernels—no mixed
   legacy/indexed deployment.

---

## 12. Conclusion

Cardinal N/S/E/W with express Z is the selected high-level Galaxy routing design. It preserves client
geometry, derives one deterministic canonical route relation from the physical carve-out, and has a
complete conditional VC0 paper proof for the frozen `[32,4]` route family. The standalone and
partial-Galaxy fixtures remain mandatory independent synthesis and validation targets.

The target is not implemented. Claiming support requires the §11 checks, including suffix
consistency, preservation of the supported row/column-uniform route relation, multicast route-union
legality, A.6 builder projection, and component-contract conformance. Expanding VC1 beyond the current
restricted envelope additionally requires the §5.7.5 proof or BFC treatment. Component contracts own
all detailed builder, codec, and kernel realization; the assessment remains authoritative for the
route oracle, safety argument, system invariants, and validation cases.

Opportunistic Z, base-32 transit, multiple express neighbors, adaptive routing, and degraded routing
remain separate extensions and do not inherit this result.

---

## Appendix A. Validation oracles and regression cases

This appendix defines finite input spaces, route/codec oracles, assertions, and expected counters for
repository regression tests. The assessment-time enumerations summarized here are validated model
results except for the explicitly labeled derived channel-capacity result and the not-yet-run
domain-effect→sender sweep in A.6. Missing regression coverage is an implementation gap, but it does
not add another production artifact or runtime phase.

### A.1 `[32,4]` route and edge-CDG enumeration

Input:

```text
sources      = 0..31
destinations = 0..31 excluding source
cases        = 32 × 31 = 992
route oracle = §3
physical graph = §1.3
```

For every ordered pair:

1. generate the complete Y edge sequence from §3;
2. require every edge to exist in §1.3;
3. require no repeated route state and no leaf transit;
4. require at most one ex4/ex8 crossover;
5. require ex4→ex8 to be terminal and ex8→ex4 to obey late exit;
6. append a CDG dependency from every directed edge to its successor;
7. run SCC analysis over all used directed edges.

Expected result:

```text
routes                    = 992
used directed Y channels  = 88
total route hops          = 4,576
mean route hops           = 4,576 / 992
                          = 143 / 31
                          = 4.6129032258...
directed Y diameter       = 10
nontrivial SCC sizes      = [16, 16, 8, 8]
cyclic directed channels  = 48
```

The complete 4×32 composition independently replicates the Y graph at four X coordinates and appends
the four-chip X route after Y completion. Its expected nontrivial SCC inventory is 16 Y SCCs plus
32 X SCCs, with no X→Y dependency.

### A.2 Restricted multicast-codec counterexample sweep

The superseded global-template codec is tested over:

```text
32 sources × 2 one-sided directions × 31 non-empty extents
= 1,984 cases
```

For each cardinal target range, construct the union of §3 unicast paths and require the restricted
codec to use one inline fall-through child, Z-inline preference, and at most one reusable out-of-line
N shape and one reusable out-of-line S shape. Decode the candidate and compare exact target and
transit actions.

Expected result:

```text
passes   = 992 / 1,984
failures = 992 / 1,984
physical-chip reconvergence cases = 0
```

`source=0, S=20` and `source=0, N=13` are constructive counterexamples. This sweep disproves only the
restricted global-template representation; it does not disprove coordinate-indexed action maps.

### A.3 Indexed multicast action-map sweep

One-sided input count:

```text
32 × 2 × 31 = 1,984
```

For simultaneous non-overlapping N/S extents:

```text
N ≥ 1
S ≥ 1
N + S ≤ 31

pairs per source = 30 + 29 + ... + 1
                 = 30 × 31 / 2
                 = 465

32 × 465 = 14,880
```

Combined:

```text
1,984 + 14,880 = 16,864 cases
```

For each case:

1. derive exact target rows arithmetically from the source and extents;
2. trace the source-reachable suffix of each destination vector;
3. union one-hot N/S/Z actions and `LOCAL_DELIVER` into one byte per Y row;
4. simulate source dispatch and every resulting child until delivery or termination;
5. require exact targets, no duplicate delivery, no execution cycle, no multi-context row, and no
   ex4→ex8 terminal violation.

Expected result:

```text
representation passes       = 16,864 / 16,864
exact target sets           = 16,864 / 16,864
duplicate-arrival cases     = 0
cycle cases                 = 0
multi-context row cases     = 0
terminal violations         = 0
maximum active Y entries    = 32
maximum Y outputs at a row  = 3
```

### A.4 Destination-vector suffix sweep and route-uniformity assertion

The healthy regular-grid cardinal sweep is a logical arithmetic superset, not a parse of every MGD
or physical channel binding. It evaluates 22 valid descriptor shapes and 65,290 ordered
source/destination pairs under:

```text
shapes [Y,X]:
[1,1], [1,2], [1,8], [1,16], [1,32],
[2,2], [2,4], [2,8],
[4,1], [4,2], [4,4], [4,8], [4,16],
[8,1], [8,2], [8,4], [8,8], [8,16],
[16,1], [16,4], [16,8],
[32,4]
```

Each shape is exercised under mesh, torus-Y, torus-X, and torus-XY boundary semantics, including
synthetic combinations broader than one concrete MGD, and under Y ties N/S crossed with X ties E/W:

```text
4 boundary modes × 4 exact-half tie policies = 16 policy combinations

65,290 × 16
= 1,044,640 source-expanded route cases
```

At every intermediate state on every route, regenerate the route from that state to the same
destination and require exact suffix equality. Expected result:

```text
route cases         = 1,044,640 / 1,044,640
intermediate checks = 8,843,872 / 8,843,872
failures            = 0
```

This proves destination-indexed representability for the enumerated healthy cardinal DOR family. It
does not prove physical edge identity, express routing, irregular multigraph embeddings, or
builder realization.

The separate frozen `[32,4]` express-aware sweep applies the same state-based suffix assertion to all
`32 × 31 = 992` Y routes generated by §3. Expected result is `992/992`. Every newly synthesized
express topology, including `[8,4]`, `[16,4]`, and `[24,4]`, requires its own complete suffix sweep.

Row/column route uniformity follows from the supported homogeneous cluster and repeated ring input.
Setup must nevertheless run the following comparison for every declared fixture before writing L1
route artifacts:

```text
for each ys, yd:
    require project_Y(R((x, ys), (x, yd))) equal for all valid x

for each xs, xd:
    require project_X(R((xs, y), (xd, y))) equal for all valid y
```

Required result: zero projection mismatches. A mismatch rejects setup. This check detects asymmetric
MGD input, configuration errors, or implementation drift; it is not a topology-generalization
mechanism, and no separate assessment counter is required. All supplied routing planes use the same
logical relation by deployment guarantee.

### A.5 Primary `[32,4]` logical software-channel budget

Enumerate the ordinary N/S, ordinary E/W, and intramesh-Z output-router variants required by the
`[32,4]` express+XY fixture, with optional existing VC2 enabled and with/without the temporary
source-reroot sender. These are logical firmware sender-channel slots, not physical Ethernet channels
or TX queues. Compare the maximum count in each VC with the required target array ceilings; current
code's software-channel ceiling 10 is an implementation gap.

Expected result:

```text
normal logical maxima by VC    = 5 / 4 / 1
normal logical-channel count   = 5 + 4 + 1
                               = 10

fallback logical maxima by VC    = 6 / 4 / 1
fallback logical-channel count   = 6 + 4 + 1
                                 = 11

required logical-channel ceiling = 11
normal sender margin              = 11 - 10 = 1
fallback sender margin            = 11 - 11 = 0

receiver maxima by VC    = 1 / 1 / 1
aggregate receiver count = 1 + 1 + 1
                         = 3
receiver ceiling         = 3
receiver margin          = 3 - 3
                         = 0
```

This result proves logical array-count feasibility only. It does not prove that an independent
route-occurrence oracle agrees with builder-local effect derivation or that the builder maps
conflicting effects to distinct channels. It also does not validate L1 packet-buffer fit, RISC
servicing, stream assignment, reserved-channel teardown, or the concrete queue/wait graph.

### A.6 Primary `[32,4]` VC0 domain-effect and builder-projection sweep

**Status:** required validation; not yet run. The zero-conflict values below are required pass conditions,
not validated-model claims.

Input:

```text
logical nodes = 32 Y positions × 4 X positions
              = 128

ordered source/destination pairs = 128 × 127
                                 = 16,256

route oracle            = complete Y-before-X composition from §3
occurrence-effect oracle = §§5.2–5.6, computed independently from each annotated route
builder CP input         = connectivity + express enable
                           + local-node/ingress/egress protected-ring predicates
builder derivation       = GALAXY_BUILDER_ROUTING_CONFIG_CONTRACT.md,
                           "Builder derivation (effects → IS_INJECTION)"
VC scope                 = VC0
```

The full 2D canonical all-pairs route set is required rather than only the 992 Y routes from A.1. The
larger space includes local-worker→Y, Y→Y, local-worker→X, Y→X, and X→X forwarding while checking
that no X→Y arc is realized.

The sweep constructs two independent views. The first is an assessment oracle derived from route
history; the second is FabricBuilder's local derivation from only the frozen CP surface and its own
wired producer map. ControlPlane does not expose the oracle effects or a local-transition list.

For every route occurrence:

1. generate the complete annotated edge sequence;
2. identify each immediate producer as `LOCAL` or the exact ingress edge;
3. independently classify whether that occurrence enters, remains in, leaves, or terminates at a
   protected ring using the canonical route's domain history;
4. group route occurrences by:

   ```text
   (node, ingress edge or LOCAL, egress edge, VC, routing plane)
   ```

5. require every occurrence to have one oracle effect and all occurrences of one key to agree;
6. independently have FabricBuilder enumerate each producer actually wired by its connection maps
   and derive a total `NON_RING`, `REMAIN`, `ENTER`, or `NON_CANONICAL` effect from connectivity and
   the local-node/ingress-direction/egress-direction protected-ring predicates;
7. match each canonical route occurrence to exactly one wired producer path and require the builder
   effect/BFC class to agree with the independent occurrence oracle;
8. project each wired transition through physical port bindings onto:

   ```text
   (output-router instance, VC, routing plane, logical sender index)
   ```

9. require every derived transition to map exactly once, every sender key to have at most one BFC guard
   class, and every DOR-forbidden slot to remain unwired;
10. compare the normal projected per-VC sender/receiver maxima with A.5;
11. project every forwarding branch emitted by the A.3 multicast action-map sweep and require it to
    match an existing canonical local transition and the same builder-derived role;
12. gate the reroot fallback separately: add the dedicated U-turn producer, require the full-packet
    producer boundary and worker-mirror guard, then verify the fallback VC0 maximum of six and its
    differential/queue conformance.

Required result:

```text
complete 2D routes                         = 16,256
unclassified forwarding occurrences       = 0
route-occurrence effect conflicts           = 0
builder-vs-oracle effect mismatches         = 0
unmapped or multiply mapped transitions    = 0
sender guard-class aliases                 = 0
DOR-forbidden connected arcs               = 0
multicast arcs outside unicast arc set     = 0
multicast role mismatches                  = 0
normal projected VC0 sender maximum         = 5
normal projected VC0 receiver maximum       = 1
A.5 normal configured maxima by VC          = 5 / 4 / 1
fallback configured maxima by VC            = 6 / 4 / 1  # separately gated
A.5 configured receiver maxima by VC        = 1 / 1 / 1
```

The normal route projection has a VC0 maximum of five. Six is the separately enabled reroot profile,
not a route-oracle projection result. The regression should also count total forwarding occurrences,
unique occurrence keys, unique wired producer keys, unique logical sender keys, per-role counts, and
per-router-variant maxima. Those observed counters become regression expectations once the first
conforming enumeration is reviewed.

Passing this sweep proves that the assessment's `[32,4]` route-occurrence effects agree with the
selected builder's independent local derivation and that those roles project without a mixed-guard
alias. It does not prove that the current FabricBuilder implements the target contract, nor does it
validate ERISC/Tensix internal allocation, L1 capacity, servicing, teardown, the concrete queue/wait
graph, or VC1.

### A.7 Regression coverage

Validation coverage for each supported topology must cover:

- the exact logical and physical topology used by the test;
- fixed route-contract inputs, tie rules, and selected ring arrangement;
- route, edge-use, CDG/SCC, suffix, row/column route-uniformity, and multicast assertions;
- logical channel counts, domain-effect→sender projection, configured ceilings, and concrete
  buffer-capacity checks;
- builder-derived queue-graph and VC0-premise checks, with VC1 scope handled explicitly.

Normal FabricBuilder setup performs only the bounded local guards in §7.3 and does not recompute these
exhaustive regression sweeps.

---

## Appendix B. Normative document ownership and index

This assessment is the central system design, not a duplicate implementation contract. Ownership is:

- **System route and proof — this assessment:** physical/logical assumptions, command semantics,
  canonical route oracle, generation policy, CDG/SCC and VC0 proof, system invariants, the §7
  cross-layer dependency ledger, implementation checklist, and Appendix A validation cases.
- **ControlPlane / route generation — `GALAXY_CONTROL_PLANE_ROUTING_GENERATION_CONTRACT.md`:**
  MGD/MeshGraph express materialization, runtime ring synthesis, canonical-route generation,
  ControlPlane-owned route/domain facts, physical-plane checks, and pre-builder consumer handoffs.
- **Builder — `GALAXY_BUILDER_ROUTING_CONFIG_CONTRACT.md`:** CP↔builder surface, local-effect
  derivation, wiring, allocation, BFC flags, and U-turn sender realization.
- **Codec — `GALAXY_DEVICE_ROUTE_CODEC_CONTRACT.md`:** L1/device artifacts, packet/header ABI,
  encode/load/decode, multicast encoder, and source fanout/reroot overlay.
- **Kernel — `GALAXY_DEVICE_ROUTER_KERNEL_CONTRACT.md`:** ERISC decode/admit/forward dispatch,
  intermesh execution, header mutation retirement, sender-step BFC, and controlled same-link return.
- **Baseline — `GALAXY_WORKING_MODEL.md`:** physical/current-code context against which the target is
  assessed.
- **Decision provenance — `GALAXY_RING_DOMAIN_Z_ROUTING_ASSESSMENT.md`:** historical alternative in
  which N/S are ring-local and Z represents transitions; it is not the selected route oracle.

When a component detail in this assessment conflicts with its component contract, the component
contract governs and this assessment must be corrected. No sibling contract is optional for
implementing the owned component.
