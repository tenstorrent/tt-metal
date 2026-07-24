# Galaxy ControlPlane Routing-Generation Contract

Detailed pre-builder contract for materializing Galaxy express connectivity, synthesizing protected
ring domains, generating canonical routes, validating them, and handing the resulting generic facts
to existing consumers.

Status: **target contract plus verified current-code baseline**. The target behavior in this document
is not implemented unless a section explicitly says that it is current behavior.

---

## 0. Authority, scope, and precedence

### 0.1 What this document owns

This document owns the component-focused contract from topology input through validated ControlPlane
route state:

1. consume MGD/MeshGraph topology intent, the selected logical carve-out, optional pinnings, and the
   logical-to-physical mapping;
2. materialize each validated intramesh express connection as a Z neighbor;
3. derive leaves, anchors, express-link classes, the selected ring arrangement, directed domains,
   required edge-use behavior, and transition policy;
4. generate one deterministic annotated canonical route
   `R(source, destination)` for every required ordered pair;
5. run the logical, route, CDG/SCC, representation, and mapping checks owned by ControlPlane /
   routing generation, while preserving the deployment-provided plane identity;
6. retain generic route/domain facts and provide each consumer only the facts it owns.

This is a pre-builder contract. It does not define FabricBuilder queues, sender indices, concrete
sender effects, packet bytes, L1 layouts, or ERISC execution.

Sections 3–4 describe one implementation-compatible way to derive the required results using the
current connectivity model. They do not require new persistent per-edge annotations, a particular
graph representation, or a particular search algorithm. Another implementation is conforming when
it produces the same route/domain outputs and passes the same checks.

### 0.2 Relationship to the five existing documents

Authority remains split as follows:

- `GALAXY_WORKING_MODEL.md` owns the physical context and current-code baseline. This document gives
  the more detailed current ControlPlane order and target delta.
- `GALAXY_CARDINAL_NS_Z_SKIP_ROUTING_ASSESSMENT.md` remains the central high-level design and proof
  document. It owns command meanings, the canonical route policy, ring-selection constraints,
  CDG/SCC and VC0 safety reasoning, cross-layer invariants, and regression oracles. This document
  turns its route-generation requirements into a pre-builder component contract; it does not create
  another topology policy.
- `GALAXY_BUILDER_ROUTING_CONFIG_CONTRACT.md` owns the exact ControlPlane-to-FabricBuilder query
  surface, builder-local effect derivation, wiring, allocation, and BFC compile-time values. This
  document owns production and validation of the generic facts behind that surface.
- `GALAXY_DEVICE_ROUTE_CODEC_CONTRACT.md` owns device-facing route artifacts, host/L1 placement,
  packet/header ABI, encode/load/decode behavior, and multicast construction. This document supplies
  the canonical route relation and coordinate facts from which codec-owned artifacts are derived.
- `GALAXY_DEVICE_ROUTER_KERNEL_CONTRACT.md` owns ERISC execution. The kernel has no direct
  ControlPlane dependency.

If this document conflicts with the central assessment on route policy or proof, the assessment
governs. If it conflicts with a component contract in that component's realization area, that
component contract governs. The documents must then be reconciled.

### 0.3 Initial-cut boundary

This contract preserves the assessment's initial cut:

- express links exist only on logical Y;
- N/S are ordinary one-edge Y moves, Z is one intramesh express move, and E/W are ordinary X moves;
- Y completes before X;
- one node has at most one logical intramesh express neighbor;
- physical lanes may be parallel realizations of one logical edge, not additional logical choices;
- supported clusters and rings are homogeneous: Y/ring structure repeats across X columns and the
  four-chip X ring repeats across Y rows;
- the supported deployment supplies one ordered routing-plane set across participating 2D edges;
  software preserves its plane index;
- no opportunistic shortcut, adaptive orientation change, degraded detour, or recovery route is
  generated;
- topology, mapping, link, or route ambiguity fails closed.

Names used by fixture descriptions are never production class identifiers. Production decisions use
logical coordinates, topology intent, derived class properties, and deterministic endpoint ordering.

---

## 1. Current baseline and exact initialization order

The sequence below is verified against current code in `tt_metal/fabric` and the existing callers.
The target extends these points; it does not add a new top-level coordinator or reorder
ControlPlane and FabricBuilder.

### 1.1 Explicit-MGD constructor path

A `ControlPlane` constructor with an MGD path calls `init_control_plane(...)`. That function currently
runs in this order:

1. Construct `MeshGraph` from the explicit textproto and current `FabricConfig`.
2. Run physical system discovery and construct the PSD.
3. Initialize the local mesh binding.
4. Prepare topology-mapping constraints:
   - use a supplied logical-to-physical chip map when that constructor overload is selected; or
   - collect applicable Galaxy corner pinnings and MGD pinnings.
5. Construct `TopologyMapper` against the logical `MeshGraph` and discovered PSD.
6. Load the resulting local logical-to-physical chip mapping into `ControlPlane`.
7. Attempt the current mesh-coordinate and ASIC-to-fabric-node mapping serializations. Serialization
   failure is logged and does not change the remaining order.
8. Construct `RoutingTableGenerator`; its constructor immediately generates the current intramesh
   and initial intermesh direction tables.
9. Initialize distributed contexts.
10. Call `generate_intermesh_connectivity()`, which binds requested intermesh connectivity, passes the
   resulting connections to `RoutingTableGenerator::load_intermesh_connections()`, and regenerates
   the intermesh table.
11. Return from `init_control_plane(...)`.
12. The constructor then calls `initialize_fabric_context()`. When fabric is enabled, this constructs
    `FabricContext`, which derives topology behavior, routing mode, and packet-resource sizing from
    the now-initialized ControlPlane.

The overload with a caller-supplied logical-to-physical map still performs physical discovery. It
uses the supplied map in the mapping constructor instead of solving for a new placement.

### 1.2 Auto-discovery constructor path

The constructor without an explicit MGD calls `init_control_plane_auto_discovery()`. Current order is:

1. Require the currently supported single-host auto-discovery case.
2. Run physical system discovery and construct the PSD.
3. Generate a regular `MeshGraph` from the PSD, `FabricConfig`, and reliability mode.
4. Initialize the local mesh binding.
5. Prepare applicable Galaxy fixed pinnings.
6. Construct `TopologyMapper`.
7. Load the resulting local logical-to-physical chip mapping.
8. Attempt the current mapping serializations.
9. Construct `RoutingTableGenerator`.
10. Initialize distributed contexts.
11. Generate and load intermesh connectivity.
12. Return from the auto-discovery function.
13. The constructor then calls `initialize_fabric_context()`.

Auto-discovery currently generates regular LINE/RING geometry. Express routing must not be inferred
from arbitrary PSD links without equivalent logical topology intent.

### 1.3 Later channel setup, L1 write, and builder construction

The later order is separate from both constructor paths:

1. `MetalEnvImpl::initialize_fabric_config()` configures Ethernet cores for fabric routers and then
   calls `ControlPlane::configure_routing_tables_for_fabric_ethernet_channels()`.
2. That ControlPlane pass:
   - maps logical directions to physical Ethernet channels;
   - checks required link counts under the selected reliability policy;
   - establishes and orders routing planes;
   - trims channels not mapped to live planes;
   - converts logical direction tables to per-source-channel routing tables.
3. This pass builds host-side state only. It does not write routing data to device L1.
4. During later fabric firmware initialization,
   `ControlPlane::write_routing_tables_to_all_chips()` writes routing information to Tensix and
   Ethernet-core L1.
5. Firmware setup then reaches `create_and_compile_tt_fabric_program()`.
6. That function constructs `FabricBuilder`, then calls its discovery, router creation, connection,
   ancillary-kernel, and kernel-creation phases before compiling the program.

Target route/domain derivation may run inside the existing ControlPlane / routing-generation
ownership area. Checks that require physical channels run when the existing channel pass makes those
bindings available. All checks required before device consumption must pass before the corresponding
L1 write. `FabricBuilderContext` remains lazy under `FabricContext`; it is not constructed as part of
the ControlPlane constructor sequence above.

### 1.4 Current routing limitations at these points

Current code has the following relevant behavior:

- The MGD schema and descriptor parser contain `express_connections`.
- `MeshGraph` constructs `MeshGraphDescriptor` with backward-compatible validation enabled, which
  rejects nonempty express connections.
- The descriptor population code can form bidirectional connection records for express endpoint
  pairs, currently tagged with compatibility direction C.
- `MeshGraph::initialize_from_mgd()` independently builds intramesh connectivity from regular
  two-dimensional LINE/RING geometry. It does not merge descriptor express records into runtime
  intramesh connectivity.
- Current in-tree mesh graph descriptors under `tt_metal/fabric/mesh_graph_descriptors/` contain no
  `express_connections`.
- `RouterEdge` is stored by destination chip and carries one port direction plus parallel connected
  chip entries. This is sufficient for the initial logical adjacency representation; ring/domain
  results are target additions beside it rather than required `RouterEdge` fields.
- `RoutingTableGenerator` emits one `RoutingDirection` per
  `[mesh][source chip][destination chip]`. It routes coordinate 0 through N/S before coordinate 1
  through E/W and uses the first direction passed to its helper for exact ties.
- `ControlPlane::configure_routing_tables_for_fabric_ethernet_channels()` remains direction-centric:
  channels in one direction are treated as parallel planes to the same logical neighbor.
- No current ControlPlane or routing-generator state contains the canonical annotated all-pairs
  relation, physical ring domains, `domain_dim`, `domains_of`, or `transition_policy`.

These are current gaps, not reasons to change the initialization order.

---

## 2. Topology intent, carve-out, mapping, and physical truth

### 2.1 Input authority

The inputs have distinct jobs:

- **MGD / MeshGraph** supplies logical dimensions, LINE/RING boundary intent, logical coordinates,
  channel policy, express endpoint pairs, intermesh intent, host layout, and optional pinnings.
- **PGD** constrains allowed physical groupings and carve-outs.
- **PSD** is the discovered physical ASIC/link graph and is the physical-conformance boundary.
- **TopologyMapper** binds logical nodes to PSD ASICs while satisfying the selected constraints and
  pinnings.
- **ControlPlane channel setup** later binds logical edges/directions to physical Ethernet channels
  and usable routing planes.

Logical topology intent chooses what the graph means. PSD validation determines whether the selected
logical graph is physically realizable. PSD must not silently add a logical edge that MGD/topology
intent did not select, and MGD must not make a nonexistent physical edge usable.

### 2.2 Consecutive-Galaxy carve-out assumption

The `[16,4]` and `[24,4]` cases in this contract are guaranteed consecutive-Galaxy carve-outs of the
quad setup. This layer consumes that selected logical carve-out and its mapping. It does not define
how deployment chooses those Galaxies.

Consequences visible to routing generation are:

- `[16,4]` has ordinary Y edges through the selected consecutive range and no cardinal `15↔0` end
  wrap.
- `[24,4]` has ordinary Y edges through the selected consecutive range and no cardinal `23↔0` end
  wrap.
- retained local wraps inside the selected range are express edges, not synthetic cardinal end
  wraps.

The route layer must reject any mapped graph that does not realize these facts.

### 2.3 Pinning and logical coordinates

Routing indices are logical coordinates, not physical chip IDs. Pinnings must eliminate every
routing-relevant rotation or reflection and fix:

- which logical dimension is Y and which is X;
- increasing and decreasing coordinate orientation;
- host and Galaxy boundaries;
- the cardinal end-wrap location when present;
- express endpoints;
- any placement distinction needed to keep logical route and physical channel bindings unambiguous.

Complete per-node pinning is not required if the remaining embedding is unique. Mapping fails when
the pinned logical graph cannot be realized in the PSD.

Pinnings constrain placement; they do not define ring membership or route policy.

---

## 3. Extending the existing logical connectivity

### 3.1 Existing representation is sufficient

The initial cut does not require adding capability, class, transit-eligibility, or stable-ID fields to
`RouterEdge`. Existing topology state already separates:

- same-mesh adjacencies in `IntraMeshConnectivity`;
- cross-mesh adjacencies in `InterMeshConnectivity`;
- the N/E/S/W/Z command in `RouterEdge::port_direction`; and
- parallel physical realizations in `RouterEdge::connected_chip_ids`.

At logical route-generation time, an implementation can therefore derive the needed distinction:

```text
entry from InterMeshConnectivity                         → intermesh
entry from IntraMeshConnectivity with direction N/E/S/W → ordinary intramesh
entry from IntraMeshConnectivity with direction Z
    and express_routing_enabled                          → intramesh express
```

A same-mesh Z entry while express routing is disabled is invalid for this contract. At the later
physical-channel stage, existing ControlPlane intermesh/intramesh-facing channel queries provide the
corresponding channel-level distinction to FabricBuilder.

If route setup needs deterministic temporary keys, `(mesh, source, destination, direction)` sorted in
a total coordinate order is sufficient for the initial graph. Such keys are working data, not a
required persistent edge product or ControlPlane API.

### 3.2 Materializing MGD express connections

For each validated intramesh express endpoint pair:

1. Resolve both endpoints in one logical mesh.
2. Require both endpoints to differ only on Y in the initial cut.
3. Create both directed logical adjacencies.
4. Assign command Z to both directions.
5. Insert the adjacency into runtime `IntraMeshConnectivity` so existing direction-keyed neighbor
   queries can recover it.
6. Require each endpoint to have at most one logical intramesh express neighbor.
7. Treat multiple physical lanes to that same neighbor as parallel realizations of the one logical
   edge.
8. Reject multiple independently routable logical express edges between the same endpoints or
   multiple logical express neighbors at one node.

`express_routing_enabled` becomes true only when nonempty express intent has been materialized and all
logical checks pass. A flag that disagrees with the runtime neighbor graph is invalid.

### 3.3 Determinism without persistent edge annotations

Required results must not depend on unordered-container iteration, physical channel numbering, host
discovery order, or fixture names. One straightforward implementation sorts directed endpoint tuples
before ring derivation, crossover ordering, route construction, CDG construction, and regression
comparison.

No opaque logical edge ID must be retained when source, destination, and command identify the logical
hop unambiguously. The initial contract already rejects independently routable duplicate logical
edges that the existing destination-keyed connectivity or action encoding cannot distinguish.
Physical channel and routing-plane identity remain later bindings and are not part of the logical
route key.

---

## 4. Required Y-ring results and a possible derivation

This section illustrates a topology-derived process that produces the required initial-cut ring
results. It is not a mandated internal algorithm. An implementation may extend the current
`RoutingTableGenerator` directly, build a temporary graph, use indexed tables, or use another method,
provided it does not consume fixture-specific node arrays and produces the same deterministic
domains, routes, exclusions, and validation results.

### 4.1 Possible Y working view

A straightforward implementation can form the following temporary view for each fixed X coordinate:

1. Project the logical graph onto Y.
2. Treat N/S entries in `IntraMeshConnectivity` as ordinary Y edges and same-mesh Z entries as
   express Y edges.
3. Record whether topology intent includes the cardinal Y end wrap.
4. Verify that ordinary edges agree with LINE/RING intent and logical coordinate orientation.
5. Collapse parallel physical lanes only at the logical level; retain their later physical bindings.
6. Use the supported deployment guarantee that every X column repeats the same Y/ring structure and
   every Y row repeats the same X-ring structure. One logical route relation is shared across those
   homogeneous copies.

Equivalent implementations need not construct this view explicitly. The required result is that
fixed-X projections have identical logical topology after replacing column-local node and endpoint
tuples with their coordinate-relative forms. Otherwise C15 in §7 fails.

### 4.2 Possible express-link class derivation

For the supported initial fixtures, one compatible way to derive a class is from the ordinary-Y arc
that a Z edge replaces:

1. Temporarily remove express edges.
2. For each undirected express endpoint pair, find the ordinary-Y base walk between its endpoints.
3. In a LINE dimension there is at most one such walk.
4. In a RING dimension choose the unique shorter ordinary-Y walk.
5. If there is no ordinary walk, or the two ring walks tie, class detection is ambiguous and fails.
6. Define the express span as:

   ```text
   express span = ordinary base hops on the bypassed walk + 1
   ```

7. Group express edges by `(axis, express span)`.
8. Sort classes by axis and increasing span; use sorted directed endpoint tuples only when a
   deterministic order within a class is needed.

For the supported fixtures this recovers the retained four-row and eight-row wrap classes without
using fixture names. Class membership comes from logical geometry; PSD later verifies that each edge
is physically present.

This detector intentionally has a fail-closed limit: the current MGD has endpoint pairs but no
explicit class identity. If two physically distinct retained-wrap classes have the same axis and
span but must remain distinct, the available topology intent cannot distinguish them. That topology
is unsupported until intent carries enough class information; setup must not split it by descriptor
order or a fixture-specific node list. Another class-derivation method is valid when it makes the same
distinctions for the supported fixtures and fails on unresolved ambiguity.

### 4.3 Required leaf, pair, and anchor result

One direct derivation of the assessment's endpoint-only rule is:

1. Select the minimum-span express class as the leaf-forming class.
2. For every edge in that class, inspect the interior nodes of its unique bypassed ordinary-Y walk.
3. A node is a leaf candidate when it is interior to such a walk and is not an endpoint of any
   retained express edge.
4. Every other required Y node is a transit candidate.
5. In the ordinary-Y graph, leaf candidates must form adjacent two-node pairs.
6. Each leaf must have exactly:
   - one ordinary edge to its paired leaf; and
   - one ordinary edge to a transit candidate.
7. The unique adjacent transit candidate is that leaf's anchor.
8. Any extra leaf adjacency, missing pair, shared ambiguity, or non-unique anchor fails synthesis.

A leaf remains a legal source or destination. It is never a Y-transit node. The direct edge between a
paired leaf pair is legal only for that pair's source/destination route. Anchor edges are terminal
attachments, not protected-ring edges.

The minimum-span rule is closed for the four supported fixtures. A future topology in which the
leaf-forming class is not the minimum-span class is outside the initial policy and must not be
guessed. Implementations need not retain leaf or anchor annotations on every edge; route generation
only needs to reproduce the same endpoint behavior and validation results.

### 4.4 Required ring-arrangement result

Regardless of construction method, every selected physical ring family must satisfy:

1. every member has selected degree two;
2. following selected edges visits every member exactly once and returns to the start;
3. every selected edge exists in the logical connectivity;
4. no leaf is a member;
5. no branch, dead end, or premature repeated edge exists;
6. every express edge required for that family is selected;
7. only ordinary Y edges may be added to close it.

The required arrangement depends on cardinal end-wrap intent:

- **Cardinal end wrap present:** produce one ring family per detected express
  class. A family contains all express edges in its class and no express edge from another class.
  Families may select the ordinary base edges needed to close their cycles.
- **Cardinal end wrap absent:** produce exactly one system-spanning family over the union of all
  express classes. It contains every retained express edge and may select ordinary base edges needed
  to close the cycle.

At the arrangement level:

1. no transit candidate belongs to more than one family;
2. the union of family members is exactly the transit-candidate set;
3. all leaves are excluded;
4. every selected family is a simple cycle;
5. every required express edge appears exactly once in the physical family set.

An implementation may construct these cycles directly from the existing connectivity, enumerate
possibilities, or use another equivalent approach. Backtracking or constraint solving is not a
contract requirement. Fixture-specific node arrays are not production inputs.

### 4.5 Canonical orientation and ambiguity handling

Each selected cycle needs a deterministic orientation. One compatible canonicalization is:

1. For one bidirectional cycle, rotate both node orientations to begin at the smallest logical node.
2. Compare the two rotated node sequences lexicographically.
3. If node sequences are equal, compare their ordered directed endpoint tuples.
4. The smaller `(node sequence, edge sequence)` is the canonical forward orientation.

Another orientation method is valid if it produces the same fixture result and deterministic tie
behavior. Deterministic traversal order is not permission to select the first otherwise valid
topology. Apply all constraints in §4.4, physical preconditions in §6, and route checks in §7:

- zero valid arrangements: fail;
- exactly one valid arrangement: select it;
- more than one distinct valid arrangement: fail with unresolved ambiguity.

The assessment provides no soft objective or preference among multiple surviving arrangements.
Choosing by search order, node labels beyond canonicalization, fixture size, or class name would
invent a topology policy.

### 4.6 Required selected and non-selected edge behavior

Route generation must enforce these semantic cases:

- **ring transit:** the edge belongs to one oriented view of a selected physical ring;
- **continuing crossover:** an ordinary non-ring edge enters another physical ring and canonical
  routing may continue around that destination ring;
- **terminal crossover:** an ordinary non-ring edge lands at its final Y destination in another
  ring and no destination-ring transit follows;
- **terminal attachment:** a leaf-anchor edge or paired-leaf edge used only for endpoint movement;
- **forbidden transit:** every other non-selected edge.

For a multi-family arrangement, a crossover candidate is an ordinary base edge whose endpoints
belong to different physical ring families. For a one-family arrangement, an unselected ordinary edge
between two transit members is not a crossover; it is forbidden transit.

These names do not require stored per-edge enums or annotations. An implementation may enforce the
same behavior through its route-construction logic, temporary sets, ordered cycles, or another
representation. The required output is that no canonical route uses a forbidden edge for transit and
that continuing, terminal, and attachment behavior matches the route policy. Merely omitting an edge
from a ring is insufficient if route construction can still use it and recreate an unprotected cycle.

### 4.7 Orientations and domain identities

One selected physical ring family produces:

- one opaque physical domain identity;
- one canonical forward directed view;
- one reverse directed view over the reversed logical hops.

Each directed view has its own BFC state on each usable routing plane. Forward and reverse are views
of one physical identity, not separate physical IDs.

For each physical domain, ControlPlane must retain or make recoverable:

- `domain_dim` = X or Y;
- canonical ordered node cycles, directed endpoint pairs, or an equivalent successor relation for
  both orientations;
- successor lookup or equivalent edge-membership lookup;
- `domains_of(node)`;
- optional member order for diagnostics.

Domain invariants are:

- a node belongs to at most one physical domain per dimension;
- a directed edge belongs to at most one oriented view per VC/plane;
- Y physical IDs are column-local and X physical IDs are row-local;
- one physical ID is shared across homogeneous planes; IDs are not multiplied by plane;
- class identity is not domain identity;
- command Z is not domain identity.

The ordinary four-chip X ring is handled as the orthogonal protected family at every Y row. Pinned
logical increasing X is the canonical E orientation; reverse is W. Exact half-ring X ties choose E.

### 4.8 Transition policy

Policy is attached to physical domain identities, `SOURCE`, and `NONE`; it is not inferred from
N/S/Z commands.

Always define:

```text
transition_policy(SOURCE, P) = CONTINUE_ALLOWED
transition_policy(NONE,   P) = CONTINUE_ALLOWED
```

Same-oriented-domain movement is REMAIN by edge membership and does not require a cross-domain policy
lookup. Unknown real physical-domain pairs default to `FORBIDDEN`, never to `CONTINUE_ALLOWED`.

For a one-family Y arrangement there is no real Y-domain-to-Y-domain transition. Leaf and source
entry can acquire the one family through `SOURCE` or `NONE`.

For the supported two-family nested arrangement:

1. order classes by derived express span;
2. transition from the larger-span family to the smaller-span family is `CONTINUE_ALLOWED`;
3. transition from the smaller-span family to the larger-span family is `LAND_ONLY`;
4. the `LAND_ONLY` crossing is terminal in Y;
5. require the resulting CONTINUE acquisition graph to be acyclic.

This is the generic geometric spelling of the assessment's supported two-family policy. It is not a
rule for arbitrary nesting. Three or more physical Y families, equal-span families needing separate
identity, or another desired acquisition order have no policy in the source documents and must fail
as an unresolved extension rather than infer an order from names or IDs.

### 4.9 Crossover order

For the supported two-family arrangement, first orient every crossover from the
`CONTINUE_ALLOWED` source family A toward destination family B:

1. orient every crossover as `(a, b)`, with `a` in A and `b` in B;
2. order crossovers by `a`'s position in A's canonical forward order;
3. use the directed endpoint tuple as the final ordering key.

The reverse `LAND_ONLY` view uses the same ordered physical crossover list, with each edge reversed.
Each possible destination member in A must have the unique paired crossover required by the
canonical policy. Missing or multiple paired landings fail route generation.

For A→B `CONTINUE_ALLOWED`, route generation evaluates every ordered crossover using the late-exit rule
in §5.2. The ordering index is the final deterministic tie-break.

---

## 5. Canonical route generation

### 5.1 Route relation and annotations

For every legal ordered pair, generate exactly one:

```text
R(source, destination)
```

`R` is an ordered sequence of logical hops. Each occurrence must make recoverable:

- source and destination node;
- command N, S, E, W, or Z;
- selected physical-domain membership and orientation when the edge is cyclic;
- crossover or attachment semantics;
- whether a crossing is terminal in the current axis.

For this initial cut, ordinary versus express is derivable from same-mesh direction and validated
express enable; intermesh routes are already separate from intramesh route generation. An
implementation may retain richer temporary records, but persistent per-edge capability annotations
are not required. The recoverable route facts support validation and consumer derivation; they are
not precomputed FabricBuilder `ENTER`/`REMAIN` rows, sender indices, or forwarding maps.

### 5.2 Non-leaf Y routes

Apply these rules after ring synthesis:

1. **Same physical family:** choose the shorter forward or reverse ring walk. An exact half-ring tie
   uses the canonical forward orientation from §4.5.
2. **`LAND_ONLY` source family to destination family:**
   - choose the crossover paired with the destination;
   - walk the source family to its source-side crossover endpoint using the same shortest/tie rule;
   - cross the ordinary base edge;
   - stop Y immediately at the destination landing.
3. **`CONTINUE_ALLOWED` source family to destination family:** evaluate every ordered crossover
   `(a,b)` and choose the lexicographically minimum tuple:

   ```text
   (
       destination-family distance from b to destination,
       source-family distance from source to a
           + 1 crossover hop
           + destination-family distance from b to destination,
       crossover order index
   )
   ```

   Walk the source family to `a`, cross to `b`, acquire the destination family, and continue only
   within that family. This is the late-exit rule.

   The source-distance key is not a generic suffix-consistency theorem. In V1 this two-family rule is
   used only for `[32,4]`, whose complete reachable-intermediate-state sweep passes `992/992`. The
   `[8,4]`, `[16,4]`, and `[24,4]` fixtures use one spanning family and do not invoke it. A failed
   suffix sweep makes the topology unsupported; V1 has no alternate route representation.

4. A route may not leave a family and later reacquire it.
5. A route may not reverse orientation adaptively after its shortest/tie decision.

For a one-family arrangement, only rule 1 is needed between transit nodes.

### 5.3 Leaves and anchors

Apply leaf handling around the non-leaf route:

1. Paired leaves use their direct ordinary edge.
2. A source leaf first takes its unique edge to its anchor, unless the destination is its paired leaf.
3. A destination leaf is reached from its anchor by the final ordinary edge.
4. Two non-paired leaves use source leaf to source anchor, the canonical non-leaf route, then
   destination anchor to destination leaf.
5. No route may use a leaf as an intermediate node.

Leaf-anchor and paired-leaf edges remain non-ring attachments even though they are Y edges; they do
not inherit Y-ring domain membership or protected-ring BFC treatment.

### 5.4 Complete Y before X

For a two-dimensional source and destination:

```text
source      = (xs, ys)
destination = (xd, yd)
```

1. Generate the complete Y route from `ys` to `yd` at fixed `xs`.
2. Include terminal crossover or destination-leaf movement before leaving Y.
3. At `(xs, yd)`, generate the shortest X-ring route to `(xd, yd)`.
4. For an exact X half-ring tie, choose forward E.
5. Never return from E/W to N/S/Z.

The command is assigned from the selected edge after path selection:

- express Y edge → Z;
- ordinary Y edge toward decreasing coordinate → N;
- ordinary Y edge toward increasing coordinate → S;
- ordinary X edge in forward orientation → E;
- ordinary X edge in reverse orientation → W.

Wrap commands retain their global cardinal orientation.

### 5.5 Stability and immutability during one configured instance

The same logical topology, pinning, physical realization, and policy must generate byte-for-byte
equivalent ordered logical edge relations and annotations. Once the applicable checks pass,
consumers must not alter `R`, the ring arrangement, or transition policy until teardown or explicit
reconfiguration.

No consumer performs another route search.

---

## 6. Physical mapping and routing-plane preconditions

Logical synthesis does not replace physical checks. The following checks run at the existing points
where their inputs become available.

### 6.1 TopologyMapper preconditions

After logical-to-physical mapping:

- every selected ring edge, crossover, attachment, and X edge maps to a PSD edge;
- every express edge maps to the intended retained physical wrap;
- no nonexistent regular-grid edge was synthesized;
- cardinal and express edges remain distinguishable even if endpoints or physical resources would
  otherwise be confused;
- each terminal landing maps to its declared logical node;
- pinnings preserve the coordinate and boundary interpretation used during synthesis.

An unrealizable pinning or logical edge rejects setup.

### 6.2 Channel realization under the deployment plane guarantee

When `configure_routing_tables_for_fabric_ethernet_channels()` establishes channels and planes:

- every required logical edge has a physical Ethernet realization;
- the supported deployment has already supplied the same ordered active plane set on every
  participating cardinal, express, and X edge;
- plane identity is preserved around the complete directed ring;
- a packet never changes plane on the canonical route;
- parallel lanes realize the same logical neighbor and remain consistently ordered at both ends;
- direction/channel bindings preserve ordinary-versus-express identity.

This design does not add a TRACE/linking-board/QSFP/cluster-wrap lane-homogeneity survey. It consumes
the deployment guarantee and preserves the established plane mapping.

### 6.3 Write gate

No device-facing route artifact derived from this target relation may be written until all logical
checks and all then-available physical checks pass. A later channel/plane failure prevents the L1
write and therefore prevents FabricBuilder from consuming an inconsistent configuration.

---

## 7. Required setup validation

### 7.1 C1-C15 setup checks

The ControlPlane / routing-generation path enforces all C1-C15 checks before device-facing route
artifacts are written. C15 verifies the generated relation expected from the homogeneous-topology
input:

- **C1:** every route hop references an existing logical and mapped physical edge.
- **C2:** every route is deterministic and cycle-free.
- **C3:** selected ring edges carry their synthesized domain identity.
- **C4:** same-domain N/S/Z transitions remain transit.
- **C5:** the CONTINUE acquisition graph is acyclic.
- **C6:** leaves are never transit.
- **C7:** Y completes before X.
- **C8:** terminal transitions do not continue in that axis.
- **C9:** every required source/destination pair has exactly one deterministic route.
- **C10:** no route uses an edge excluded from transit by the selected arrangement.
- **C11:** every terminal crossing lands at its recorded destination and is never intermediate.
- **C12:** the generated CDG contains no cyclic edge outside the selected protected rings.
- **C13:** every node has at most one logical intramesh express adjacency, and the selected action
  encoding identifies every local edge unambiguously.
- **C14:** at every reachable intermediate state, the remaining route equals
  `R(current, destination)`.
- **C15:** the supported homogeneous-topology input yields row/column route uniformity: Y routes are
  identical across X columns, X routes are identical across Y rows, and supplied planes share the
  same logical relation.

Any failure rejects the configuration with enough mesh/node/endpoint, route, domain, or ambiguity
context to identify the failing input.

### 7.2 CDG and SCC check

Build the complete edge-level CDG from the selected canonical all-pairs route set:

1. create one CDG node per used directed logical hop, keyed by its endpoint tuple or an equivalent
   temporary representation;
2. for each route, add a dependency from every edge occurrence to its immediate successor;
3. run SCC analysis;
4. require every nontrivial SCC to equal one selected directed protected ring;
5. require no excluded edge, leaf attachment, terminal crossover, or inter-domain edge to appear in
   an unexpected cyclic SCC;
6. require the condensation graph to respect the acyclic CONTINUE order and Y-before-X order.

The SCC inventory is generated per topology. Results from one fixture do not transfer to another.

### 7.3 Suffix consistency

For every ordered pair and every reachable intermediate node on its route:

```text
suffix of R(source, destination) beginning at current
= R(current, destination)
```

Compare directed logical-hop sequences and terminal semantics, not only command letters. A failure
rejects destination-only route representation for that arrangement.

### 7.4 Row/column route uniformity

For every valid coordinate:

```text
project_Y(R((x, ys), (x, yd))) is identical for every valid x
project_X(R((xs, y), (xd, y))) is identical for every valid y
```

The supported cluster and ring layout guarantees these equalities, and all supplied routing planes
share the same logical relation. Setup still compares the generated projections before L1 writes to
catch asymmetric MGD input, configuration errors, or route-generation drift. A mismatch rejects
setup; this is a logical consistency check, not a physical topology-qualification pass.

### 7.5 Other required invariants

Validation also requires:

- complete reachability for every required ordered pair;
- no repeated route state;
- no immediate canonical U-turn;
- no forbidden edge use;
- no route with more than the allowed cross-domain transition count;
- terminal landing correctness;
- stable deterministic tie behavior;
- ordinary and express output identity preserved through physical mapping;
- at least two packet slots in every protected receiver as a downstream realization precondition;
- a separate VC1 dependency proof or BFC treatment whenever intermesh VC1 is claimed.

ControlPlane owns the route and topology checks. FabricBuilder later owns only its bounded concrete
realization checks.

---

## 8. ControlPlane-owned facts and consumer handoffs

### 8.1 Facts retained by ControlPlane / routing generation

RT-gen owns the canonical logical next-hop relation. It may retain the internal ring/domain
representation used to derive and validate that relation, but it need not store every complete
`R(source,destination)` path separately. Because C14 requires suffix consistency, a complete route
can be reconstructed by repeatedly applying the canonical next-hop relation.

ControlPlane is the facade for consumers. A compatible logical-route query sketch is:

```cpp
std::optional<RoutingDirection> get_forwarding_direction(
    FabricNodeId source,
    FabricNodeId destination) const;

std::vector<FabricNodeId> get_canonical_intramesh_route(
    FabricNodeId source,
    FabricNodeId destination) const;
```

The existing channel-conditioned `get_fabric_route(source, destination, source_channel)` remains the
physical channel route query and is not a substitute for this canonical logical route. Exact names,
return containers, and whether the complete route is reconstructed or cached remain implementation
choices.

The validated state must make these generic facts recoverable:

- logical-to-physical node mapping and pinned logical coordinates;
- existing logical connectivity and deterministic route/domain views;
- validated `express_routing_enabled`;
- selected physical ring identities;
- each ring's X/Y dimension;
- canonical forward and reverse ordered edge views;
- `domains_of(node)`;
- `transition_policy`;
- the canonical next-hop relation and complete logical routes reconstructed from it;
- existing next-hop direction information where retained for compatibility.

Internal storage and exact C++ names remain implementation choices; the builder contract defines the
required predicate results. Temporary class, leaf, candidate, or forbidden-edge working sets do not
become persistent ControlPlane APIs merely because one synthesis implementation uses them.

### 8.2 FabricBuilder handoff

Internal ring IDs, `domains_of`, ordered cycles, and transition-policy storage remain owned by
RT-gen/ControlPlane. FabricBuilder does not need those identities as external API values. A
compatible node/direction predicate sketch is:

```cpp
bool express_routing_enabled(MeshId mesh) const;

bool is_protected_ring_edge(
    FabricNodeId local,
    RoutingDirection egress) const;

bool are_same_directed_ring_edges(
    FabricNodeId local,
    RoutingDirection ingress,
    RoutingDirection egress) const;

bool continuation_allowed(
    FabricNodeId local,
    RoutingDirection ingress,
    RoutingDirection egress) const;

bool has_protected_ring(
    FabricNodeId node,
    RoutingDimension dimension) const;
```

For a wired router pair, ControlPlane resolves the ingress neighbor `U` and egress neighbor `W` from
the two directions around local node `V`; the semantic turn is `U→V→W`. `continuation_allowed`
returns only the distinction Builder needs: whether a non-transit protected-ring acquisition is
allowed. RT-gen may distinguish terminal-only and forbidden transitions internally. Worker source
injection and the dedicated reroot sender are handled separately and do not use an artificial ingress
direction.

Names and exact signatures remain illustrative. The information split and predicate results are the
contract. FabricBuilder also continues consuming existing:

- existing neighbors, physical channels, routing planes, and direction/peer connectivity;
- `express_routing_enabled`;
- intermesh/intramesh channel identity.

ControlPlane must not expose builder-shaped effects or layouts. In particular, it does not provide:

- precomputed `ENTER`/`REMAIN` rows;
- `list_local_transitions`;
- sender indices or `IS_INJECTION`;
- builder forwarding arcs;
- queue/channel allocation;
- a domain-kind enum tied to fixture names;
- internal ring/domain IDs or transition-policy enums;
- a production SCC census API;
- destination vectors or multicast maps as builder dependencies.

FabricBuilder derives total local effects only for producers it actually wires.

### 8.3 Codec/host handoff

Codec/host setup consumes:

- the canonical logical next-hop/route queries above, or route-derived generic vectors produced
  beside routing generation;
- pinned mesh shape and logical coordinates;
- validated express state;
- suffix consistency and the supported row/column-uniform route relation.

The codec does not consume internal ring IDs, Builder `ENTER`/`REMAIN` effects, or transition-policy
storage.

Exact vector layout, multicast artifacts, L1 placement, packet state, and ABI selection remain codec
owned.

### 8.4 Kernel boundary

There is no ControlPlane-to-kernel query. Kernel routing inputs come only from:

- codec-installed L1/packet state; and
- builder-emitted wiring and compile-time values.

The kernel does not recover ring identity, rerun route generation, or infer protected-domain effects
from N/S/Z.

---

## 9. Fixture regression examples

All examples in this section are required regression outputs, independent of the implementation used
to derive them. They are not production node arrays, class names, or dimension-dispatch tables.

### 9.1 `[8,4]` regression example

For one fixed X:

```text
ordinary Y ring:
0↔1↔2↔3↔4↔5↔6↔7↔0

express:
2↔5
```

The express span is:

```text
base hops from 2 to 5 = 3
express span          = 3 + 1 = 4
```

Derived output:

```text
forward:
0 → 1 → 2 → 5 → 6 → 7 → 0
S   S   Z   S   S   S

reverse:
0 → 7 → 6 → 5 → 2 → 1 → 0
N   N   N   Z   N   N
```

Leaves and anchors:

```text
leaves  = {3,4}
pair    = 3↔4
anchors = 3→2, 4→5

8 total Y rows - 2 leaves = 6 transit rows
```

The complementary cycle through `2→3→4→5→2` is not a protected routing domain.

### 9.2 `[16,4]` regression example

The consecutive two-Galaxy carve-out has no `15↔0` cardinal edge.

```text
ordinary:
0↔1↔...↔15

span-4 express:
2↔5, 6↔9, 10↔13

span-8 express:
0↔7, 8↔15
```

Leaves and anchors:

```text
leaves  = {3,4,11,12}
pairs   = 3↔4, 11↔12
anchors = 3→2, 4→5, 11→10, 12→13

16 total Y rows - 4 leaves = 12 transit rows
```

The no-end-wrap rule selects one spanning physical family:

```text
forward:
0 → 1 → 2 → 5 → 6 → 9 → 10 → 13 → 14 → 15 → 8 → 7 → 0
S   S   Z   S   Z   S    Z    S    S    Z    N   Z

reverse:
0 → 7 → 8 → 15 → 14 → 13 → 10 → 9 → 6 → 5 → 2 → 1 → 0
Z   S   Z    N    N    Z     N   Z   N   Z   N   N
```

Both directions of `6↔7` and `8↔9` are forbidden transit edges.

### 9.3 `[24,4]` regression example

The consecutive three-Galaxy carve-out has no `23↔0` cardinal edge.

```text
ordinary:
0↔1↔...↔23

span-4 express:
2↔5, 6↔9, 10↔13, 14↔17, 18↔21

span-8 express:
0↔7, 8↔15, 16↔23
```

Leaves and anchors:

```text
leaves  = {3,4,11,12,19,20}
pairs   = 3↔4, 11↔12, 19↔20
anchors = 3→2, 4→5, 11→10, 12→13, 19→18, 20→21

24 total Y rows - 6 leaves = 18 transit rows
```

The no-end-wrap rule selects one spanning physical family:

```text
forward:
0 → 1 → 2 → 5 → 6 → 9 → 10 → 13 → 14 → 17 → 18 → 21 → 22 → 23 → 16 → 15 → 8 → 7 → 0
S   S   Z   S   Z   S    Z    S    Z    S    Z    S    S    Z     N    Z    N   Z

reverse:
0 → 7 → 8 → 15 → 16 → 23 → 22 → 21 → 18 → 17 → 14 → 13 → 10 → 9 → 6 → 5 → 2 → 1 → 0
Z   S   Z    S    Z     N    N    Z     N    Z     N    Z     N    Z   N   Z   N   N
```

Both directions of `6↔7`, `8↔9`, `14↔15`, and `16↔17` are forbidden transit edges.

### 9.4 Explicitly labeled `[32,4]` ex4/ex8 regression example

The full four-Galaxy fixture has cardinal `31↔0`.

```text
ex4:
[1,2,5,6,9,10,13,14,17,18,21,22,25,26,29,30]

ex8:
[0,7,8,15,16,23,24,31]

leaves:
{3,4,11,12,19,20,27,28}
```

Arithmetic:

```text
16 ex4 nodes + 8 ex8 nodes + 8 leaves
= 24 transit nodes + 8 leaves
= 32 Y rows

32 ordinary Y edges + 8 ex4 edges + 4 ex8 edges
= 44 undirected Y edges

44 × 2 directions
= 88 directed Y edges
```

Leaves, pairs, and anchors:

```text
3↔4,   anchors 3→2 and 4→5
11↔12, anchors 11→10 and 12→13
19↔20, anchors 19→18 and 20→21
27↔28, anchors 27→26 and 28→29
```

Crossovers, ordered by the ex8 endpoint's canonical forward position:

```text
0↔1, 7↔6, 8↔9, 15↔14,
16↔17, 23↔22, 24↔25, 31↔30
```

The selected physical families produce four directed Y domains per fixed X:

```text
ex4 forward and reverse
ex8 forward and reverse
```

The fixed policy is:

```text
ex8 → ex4 = CONTINUE_ALLOWED with late exit
ex4 → ex8 = LAND_ONLY / terminal in Y
```

Exact half-ring ties choose the displayed forward order:

```text
ex4: 8 versus 8
ex8: 4 versus 4
X:   2 versus 2 chooses E
```

Reference route arithmetic for one fixed X:

```text
ordered Y routes = 32 × 31 = 992
total route hops = 4,576
mean route hops  = 4,576 / 992
                 = 143 / 31
                 = 4.6129032258...
```

Reference CDG result:

```text
nontrivial Y SCC sizes = [16,16,8,8]
cyclic directed Y edges = 48
directed Y diameter = 10
```

For the complete two-dimensional mesh:

```text
chips = 32 × 4 = 128
ordered non-self routes = 128 × 127 = 16,256
```

These names and arrays are regression expectations only.

---

## 10. Current-to-target gaps

### 10.1 MGD and MeshGraph

- Stop rejecting validated initial-cut express endpoint pairs solely because backward-compatible MGD
  validation is enabled.
- Convert descriptor express endpoints into runtime bidirectional Z adjacencies.
- Keep same-mesh express Z adjacency separate from the existing intermesh connectivity.
- Reject unsupported axes, multiple logical express neighbors, and ambiguous duplicate edges.
- Use deterministic endpoint ordering wherever unordered connectivity is traversed.

### 10.2 Routing generation

- Replace the regular-grid-only intramesh direction result for express-enabled meshes with the
  synthesis and canonical-route pipeline in §§4-5.
- Derive classes, leaves, anchors, ring families, orientations, required edge-use behavior, domains,
  and transitions from runtime topology; these may remain route-generation working data where no
  consumer query requires them.
- Generate a canonical suffix-consistent next-hop relation from which every complete
  `R(source,destination)` can be reconstructed.
- Retain ordinary direction tables where needed for compatibility, but derive them from the same
  canonical relation.

### 10.3 ControlPlane validation and state

- Add C1-C15 at the existing ownership points.
- Validate CDG/SCCs from directed logical-hop endpoint tuples or an equivalent temporary key.
- Validate suffix consistency and row/column route uniformity for the supported homogeneous fixtures.
- Bind logical edges to the TopologyMapper result and later channel/plane state.
- Retain the generic facts in §8.1 without adding builder-shaped effects.

### 10.4 Existing channel and write path

- Make channel mapping edge-aware enough to distinguish ordinary and express edges.
- Verify complete, plane-preserving physical realization before L1 consumption.
- Derive codec-owned artifacts from the same validated relation.
- Keep the existing order: constructor route setup, FabricContext, later channel setup, later L1
  write, then FabricBuilder.

---

## 11. Focused implementation checklist

- [ ] Preserve both current constructor orders and the later channel/write/builder order in §1.
- [ ] Materialize validated MGD express endpoint pairs as bidirectional intramesh Z neighbors.
- [ ] Make `express_routing_enabled` agree exactly with materialized express connectivity.
- [ ] Produce deterministic results independent of unordered iteration and physical channel
      numbering; persistent logical edge IDs are not required.
- [ ] Derive the supported-fixture express classes and reject ambiguous equal-span provenance; §4.2
      is one compatible process, not a required implementation.
- [ ] Produce the required leaves, pairs, and unique anchors; §4.3 is one compatible derivation.
- [ ] Produce and canonicalize an arrangement satisfying §§4.4-4.5; direct construction, temporary
      graph search, or another equivalent method is allowed, but unresolved multiple arrangements
      are not selected by traversal order.
- [ ] Internally materialize enough ring/domain state to answer the §8.2 node/direction predicates;
      do not expose internal IDs or transition-policy enums.
- [ ] Generate the canonical next-hop relation by §5 so every reconstructed
      `R(source,destination)` has the required same-ring ties, late exit, terminal behavior, leaves,
      and Y-before-X.
- [ ] Run C1-C15, complete route/CDG/SCC checks, suffix checks, and the generated row/column
      route-projection comparison before writing L1 artifacts.
- [ ] Validate logical-to-physical edges, pinnings, terminal landings, and channels; preserve the
      deployment-provided plane identity at the existing setup points.
- [ ] Prevent the corresponding L1 write on any unresolved logical or physical failure.
- [ ] Expose only the node/direction predicates and existing connectivity facts in §8.2; do not expose
      internal ring IDs, local effect tables, sender roles, or consumer layouts.
- [ ] Provide the codec with the canonical logical route query or equivalent route-derived generic
      inputs; do not substitute channel-conditioned `get_fabric_route()`.
- [ ] Add independent regression coverage for `[8,4]`, `[16,4]`, `[24,4]`, and `[32,4]` matching
      §9 and the assessment's complete oracles.
- [ ] Retain the current two-link, predominantly linear VC1 envelope; gate arbitrary-pattern
      multi-mesh expansion on its separate dependency proof or required BFC extension.

---

## 12. Explicit unresolved contract items

The supported fixture outputs are closed. The following generalizations are not closed by the source
documents and must fail rather than guess:

1. **Same-span class provenance.** Current MGD express intent gives endpoint pairs but no explicit
   class identity. The `(axis, span)` detector is sufficient for the supported fixtures, but cannot
   separate two physical classes with the same span when topology policy needs them distinct.
2. **More than two nested Y families.** The assessment defines one-family behavior and the supported
   two-family CONTINUE/LAND_ONLY order. It does not define a transition order for three or more
   families or another nesting relation.
3. **Multiple valid ring arrangements.** The assessment supplies hard constraints, canonical
   orientation, and fail-closed ambiguity behavior, but no soft ranking among multiple distinct
   survivors. Deterministic traversal or working-set ordering is for reproducibility, not selection.
4. **Non-minimum leaf-forming class.** The supported fixtures derive leaves from the minimum-span
   class. A topology requiring another leaf-forming class needs explicit topology policy.

Resolving any item requires updating the central assessment's topology policy and validation oracles
before changing production synthesis.
