# Per-connection STRICT / RELAXED for inter-mesh connections

Status: design for review. Nothing here is implemented. Related:
[tt-metal#49960](https://github.com/tenstorrent/tt-metal/issues/49960) (mixed inter-mesh
connections in the control plane), #56762 (RELAXED zero-link tolerance, landed), and the
pipeline design in `mgd-pipeline-graph-design.md` (its MGD merge of several parts needs this).

Canvas: `mgd-intermesh-policy-plan.canvas.tsx` (same content, with diagrams).

## 0. Summary

The MGD proto already stores a policy on **every** connection (`Channels { count policy }`),
but the code collapses it into one boolean for the whole MGD: the policy of the first FABRIC
connection. That boolean then becomes two **global solver arguments**:
`ConnectionValidationMode` (are channel counts hard?) and
`MappingConstraints::allow_unmatched_target_edges` (may an edge have zero links?). Parsing
rejects an MGD that mixes policies, merging rejects parts that disagree, and the control plane
rejects an MGD with both device-level and mesh-level connections.

The design carries the policy per connection, end to end:

1. **MGD.** Each mesh pair gets a channel demand `{strict, relaxed}`, summed over the
   connections on that pair. Mixing is allowed.
2. **Solver.** The policy becomes a **constraint on a target edge**, like the existing
   node constraints, and every edge resolves to one `EdgeDemand { required, preferred }`.
   A caller sets it per edge (`add_required_edge_constraint`,
   `add_preferred_edge_constraint`, `set_edge_policy`) or for the **whole solve**
   (`set_default_edge_policy`). The default lives in `MappingConstraints`, so
   `solve_topology_mapping` no longer takes a `ConnectionValidationMode`, and
   `allow_unmatched_target_edges` becomes one value of the default. The SAT encoding already
   has a per-edge "realized" literal; today it is created for every edge or none, and it is
   created only for edges with `required == 0`.
3. **Both solves use it.** The intra-mesh solve sets its mesh's policy as the default edge
   policy; the inter-mesh seat solve sets per-pair demands. Neither passes a mode to the
   solver (section 3.4).
4. **Mapper and control plane.** Exit-node constraints, intermesh binding and validation look
   up each mesh pair's demand instead of the global mode.
5. **Cable type.** The same per-edge structure carries the cable `PortType`: each global edge
   gets a channel count per type; each target edge, or the whole solve, says which types may
   count (hard) and which it prefers (soft, tiers best first). The MGD says it with
   `allowed_port_types` and `preferred_port_types` in `Channels` (section 7). Without them,
   every edge still gets an **implicit tie-break preference** from a built-in cable order,
   added by the mapper like its other implicit preferred constraints (section 7.6).
6. **The mapper hands its allowed inter-mesh channels to the control plane.** After the
   solve, `TopologyMapper::get_inter_mesh_candidates()` lists every channel between the meshes
   it mapped that is up and on an allowed cable type, more than requested when the cabling has
   more. The control plane keeps channel selection and direction assignment as they are and
   drops its gather of port descriptors (section 10). Moving channel selection and directions
   into the mapper too is a stretch goal, not planned (section 13).

So yes, STRICT / RELAXED can be a required constraint instead of a solver argument, for the
whole solve or per edge. The MGD side and the solver side land together, in one PR (section 9):
the per-pair demand is useless until the solver can encode it, and the solver change has no
caller until the MGD supplies it.

## 1. How the policy flows today

| Step | Where | What it does with the policy |
|---|---|---|
| Proto | `mesh_graph_descriptor.proto`, `Channels` | `count` and an optional `policy` (STRICT if omitted) on every connection, and on `graph_topology` for the edges it generates |
| Parse | `MeshGraphDescriptor::populate_inter_mesh_policy` (`mesh_graph_descriptor.cpp:789`) | one bool: the first FABRIC connection's policy, else the top-level `graph_topology.channels` policy |
| Validate | MGD 1.0 compatibility check (`mesh_graph_descriptor.cpp:1198`) | rejects "Cannot mix STRICT and RELAXED policies in the same graph" |
| Merge | `MeshGraphDescriptor::merge` (`:384`), `validate_shared_inter_mesh_policy` (`topology_mapper_utils.cpp:378`), `generate_rank_bindings.cpp:177` | three separate fatals when parts disagree |
| MeshGraph | `mesh_graph.cpp:345-390` | device-level connections go to `requested_intermesh_ports_` (src chip, dst chip, count); mesh-level ones to `requested_intermesh_connections_` (count, assigned with `=`, so a second connection on the same pair overwrites the first). Neither map keeps the policy |
| Mapper config | `topology_mapper.cpp:563` (TODO pointing at #49960), `topology_mapper_utils.cpp:1624` | `TopologyMappingConfig::inter_mesh_validation_mode` = the bool |
| Seat solve | `SatPlacementEnumerationSession` (`physical_grouping_descriptor_matching.cpp:3131, 3170, 3276`) | `relaxed_inter_mesh_policy_` picks RELAXED channel validation and enables the soft-seam rungs of `kSolveLadder` (`:2953`), which set `allow_unmatched_target_edges` for **every** seam |
| Solver | `topology_solver_sat.cpp` | soft edges switch off the degree filter (`:856`) and AC-3 (`:887`) for the whole problem, and give every edge a realized literal (`:1121`); RELAXED adds channel-threshold literals for every edge (`:1564`) |
| Exit nodes | `add_exit_node_constraints` (`topology_mapper_utils.cpp:955`) | per destination mesh, but the STRICT / RELAXED math reads the one global mode |
| Control plane | `control_plane.cpp:3595, 3747, 4152, 4606` | "must specify either RelaxedGraph or Graph connections, not both": device-level and mesh-level cannot coexist |
| Control plane | `:3613`, `validate_requested_intermesh_connections` (`:3818`) | shortfall and per-pair checks read the global bool; the device-level map is labelled "strict" and the mesh-level map "relaxed" |
| Control plane | `reject_used_downed_links_under_strict_policy` (`:1213`), `validate_system_health_against_mesh_graph_policy` (`:1015`) | "Intermesh is one policy for the whole graph" |

### Two separate things are conflated

- **Granularity:** a device-level connection pins exit chips; a mesh-level connection asks for
  a channel total between two meshes.
- **Policy:** STRICT requires the channels; RELAXED prefers them and tolerates fewer,
  including zero (#56762).

The code and #49960 treat device-level as strict and mesh-level as relaxed. They are
independent: a mesh-level connection may be STRICT, and a device-level one RELAXED. This
design makes policy per connection, and lets the two granularities coexist, per mesh pair.

## 2. Semantics per connection

| Connection | STRICT | RELAXED |
|---|---|---|
| mesh-level, `count: c` | the two meshes must be joined by at least `c` channels | up to `c` channels, maximized; zero is allowed and warns |
| device-level, `count: c` | the two named chips must be joined by `c` channels | the chips stay pinned as exit nodes; up to `c` channels between them, zero allowed |
| `graph_topology` with `channels` | each generated edge is STRICT with that count | each generated edge is RELAXED with that count |
| policy omitted | STRICT (unchanged default) | - |

**Several connections on one mesh pair** add up into a demand:

```
demand(A, B) = { strict:  sum of counts of STRICT connections between A and B,
                 relaxed: sum of counts of RELAXED connections between A and B }
```

The edge A–B must exist if `strict > 0`; its hard channel minimum is `strict`; the solver then
prefers up to `strict + relaxed` channels. For example, a device-level STRICT pin of 1 channel
plus a mesh-level RELAXED count of 2 on the same pair means: one pinned channel is required,
two more are wanted, and the pair is never zero-link. Device-level connections are demands on
`(src chip, dst chip)` in the same way.

## 3. Solver: STRICT / RELAXED as a per-edge channel requirement, not a solver-wide mode

### 3.1 One structure per edge

Every target edge with `c` channels in the target graph resolves to one demand:

```cpp
struct EdgeDemand {
    std::size_t required = 0;   // hard minimum; > 0 means the edge must exist
    std::size_t preferred = 0;  // soft extra on top of `required`; required + preferred == c
};
```

Today's two solver arguments are three points on that scale, so one policy enum names all
three:

| `EdgePolicy` | `required` | `preferred` | Today |
|---|---|---|---|
| `STRICT` | `c` | 0 | `ConnectionValidationMode::STRICT` |
| `RELAXED` | 1 | `c - 1` | `ConnectionValidationMode::RELAXED`: the edge must exist, its channel count is best effort |
| `BEST_EFFORT` | 0 | `c` | `RELAXED` plus `allow_unmatched_target_edges` (#56762): the edge may have zero links |

The MGD word RELAXED already means two different things: an intra-mesh RELAXED mesh still
needs every link (solver `RELAXED`), and an inter-mesh RELAXED connection tolerates zero
links (solver `BEST_EFFORT`). Naming the three by effect keeps the solver unambiguous.

### 3.2 API: the whole-solve default is a constraint too

`MappingConstraints<TargetNode, GlobalNode>` already has node constraints (required,
preferred, forbidden, cardinality). It gains edge constraints at three scopes:

```cpp
enum class EdgePolicy { STRICT, RELAXED, BEST_EFFORT };

// Whole solve: every target edge with no per-edge constraint. Unset means RELAXED, today's
// default argument, so a caller that sets nothing is unchanged.
void set_default_edge_policy(EdgePolicy policy);
std::optional<EdgePolicy> default_edge_policy() const;

// One edge, sized from that edge's channel count in the target graph.
bool set_edge_policy(TargetNode a, TargetNode b, EdgePolicy policy);

// One edge, explicit numbers (an inter-mesh pair's {strict, relaxed} demand).
// Calls on the same edge add up. Return false if a or b is not a target node.
bool add_required_edge_constraint(TargetNode a, TargetNode b, std::size_t min_channels = 1);
bool add_preferred_edge_constraint(TargetNode a, TargetNode b, std::size_t channels);

// For the backends and the validator: one demand per target edge.
std::map<std::pair<TargetNode, TargetNode>, EdgeDemand> resolve_edge_demands(
    const AdjacencyGraph<TargetNode>& target_graph) const;
```

Rules:

- **Precedence:** explicit numbers, then `set_edge_policy`, then the default. Using both
  explicit numbers and `set_edge_policy` on one edge is rejected (the call returns false).
- **Counts:** for an edge with explicit numbers, `required + preferred` must equal the target
  edge's channel count, so the target graph stays the one source of counts. A mismatch fails
  the solve with a message naming the edge.
- **`merge`:** explicit numbers add up; two different defaults or two different
  `set_edge_policy` values on one edge are rejected, like conflicting same-rank partitions.
- **`allow_unmatched_target_edges`:** `set_allow_unmatched_target_edges(true)` becomes a
  deprecated alias for `set_default_edge_policy(BEST_EFFORT)`. Every caller that sets it today
  also solves in RELAXED mode (`physical_grouping_descriptor_matching.cpp:3254, 3276`), so the
  alias loses nothing.

**The solver call no longer takes a mode.** `solve_topology_mapping`,
`solve_topology_mapping_n`, `solve_topology_mapping_all` and the
`TopologyMappingEnumerationSession` constructor get overloads without
`ConnectionValidationMode`:

```cpp
template <typename TargetNode, typename GlobalNode>
MappingResult<TargetNode, GlobalNode> solve_topology_mapping(
    const AdjacencyGraph<TargetNode>& target_graph,
    const AdjacencyGraph<GlobalNode>& global_graph,
    const MappingConstraints<TargetNode, GlobalNode>& constraints,
    bool quiet_mode = false,
    TopologyMappingSolverEngine solver_engine = TopologyMappingSolverEngine::Auto);
```

The old signatures stay, deprecated, with the mode made a required argument. They copy the
constraints and set the mode as the default edge policy when none is set; if the constraints
already set a different default, the solve fails. A three-argument call resolves to the new
overload, and since an `enum class` and `bool` don't convert into each other, no call is
ambiguous. About 190 uses of `ConnectionValidationMode::` in 13 files (127 in
`test_topology_solver.cpp`) can move over later; nothing forces them.

Inside the backends, `ConnectionValidationMode` disappears from the SAT encoder, the DFS
search and `MappingValidator`; they read the resolved `EdgeDemand` per edge.

### 3.3 SAT encoding (`topology_solver_sat.cpp`)

"Required edge" below means `required > 0`; "preferred-only edge" means `required == 0`.

| Step | Today | Per edge |
|---|---|---|
| 1. Domain degree filter (`:856`) | skipped for every target when soft | a global must have at least as many neighbours as the target has **required** edges |
| 2. AC-3 (`:887`) | skipped for the whole problem when soft; STRICT also checks channel counts (`:902, 918`) | runs over required arcs; an arc checks channels against its `required` minimum |
| 3b. `prune_active_lit` (`:1364`) | guards the hard domains of a soft encoding | the same, built from the required-only domains |
| 6. Adjacency support (`:1121`) | every edge hard, or every edge guarded by a realized literal; STRICT supports need the full count (`:1155`) | required edge: unguarded support clauses on global edges with at least `required` channels. Preferred-only edge: guarded by its realized literal |
| Channel thresholds (`:1564`) | RELAXED: levels 1..count on every edge, built but discarded (`:1861`), so never optimized | levels `required+1 .. required+preferred`, only on edges with `preferred > 0`, maximized as their own stage (11.5) |
| Realized-edge objective (`:1867`, binary search at `:2033`) | over every edge when soft | over preferred-only edges |
| Seam toggle (`set_allow_unmatched_target_edges`, `:1926`) | assume every realized literal | `set_best_effort_edges_required(bool)`: assume the preferred-only edges' realized literals, so `BEST_EFFORT` edges act as `RELAXED`; required edges are hard in both modes |

A uniform `STRICT` or `RELAXED` default produces exactly today's encoding for that mode, and a
uniform `BEST_EFFORT` default produces today's soft encoding. That is the property test in
section 9.

The order of today's objectives is unchanged: host cap and host packing, then preferred node
hits, then the number of realized preferred-only edges. Preferred channels follow as a new
stage; today the channel-threshold literals are built but never optimized. Section 11.5 has
the full schedule.

**Validation** (`MappingValidator`, `topology_solver.tpp:3800`) checks each edge against its
demand: a missing required edge or a required-channel shortfall is an error; a missing
preferred edge or a preferred-channel shortfall is a warning naming the edge.

**DFS backend.** Required edges are what DFS does today in STRICT mode. A preferred edge that
may end with zero links needs the SAT engine, as `allow_unmatched_target_edges` does today
(`topology_solver.tpp:1354`). The DFS backend rejects such constraints with the same message.

**Why this is also faster for mixed MGDs.** Today one RELAXED seam makes every seam soft and
switches off the degree filter and AC-3 for the whole master solve. Per edge, those prunings
stay on for STRICT edges.

### 3.4 Both solves use the same structure

| Solve | Where | Today | Per edge |
|---|---|---|---|
| Intra-mesh (fabric node to ASIC) | `complete_intra_mesh_for_placement` (`topology_mapper_utils.cpp:1459-1468`) | `determine_intra_mesh_validation_mode` picks STRICT / RELAXED per mesh and passes it to `solve_topology_mapping`; a `TT_FATAL` forbids `allow_unmatched_target_edges` | `intra_mesh_constraints.set_default_edge_policy(...)` from the mesh's policy (`STRICT` or `RELAXED`), then the new overload. The fatal becomes: no intra-mesh edge may resolve to `required == 0` |
| Intra-mesh candidate pools in the seat solve | `CandidatePool`, `create_sat_placement_pools` (`physical_grouping_descriptor_matching.cpp:2211, 2558`) | `sat_intra_mesh_mode_by_mesh_` (`:3126, 3157`) holds a mode per mesh; it is part of the pool hash | the map holds an `EdgePolicy`; each pool sets it as the default on its constraints; the hash uses the policy |
| PGD grouping matches | `physical_grouping_descriptor_matching.cpp:586, 1361, 1698, 1779` | mode argument (STRICT, or the instance's intra policy at `:1779`) | default edge policy on the constraints they already build |
| Inter-mesh seat solve | `SatPlacementEnumerationSession` | RELAXED or STRICT argument plus the soft-seam bool | explicit numbers per mesh pair from the demand table (section 5); default `STRICT` |
| Single-mesh fallback | `topology_mapper.cpp:1746` | RELAXED argument | `set_default_edge_policy(RELAXED)` |

`TopologyMappingConfig::mesh_validation_modes` keeps its type for now; the mapper translates
each entry to an `EdgePolicy` (`STRICT` to `STRICT`, `RELAXED` to `RELAXED`). Intra-mesh
`BEST_EFFORT` stays forbidden: a missing link inside a mesh breaks routing, and nothing in the
MGD can ask for it.

**Per-connection intra-mesh policy** is not expressible in the MGD today: a mesh has one
`channels { count policy }`, and express links inherit it (`mesh_graph_descriptor.proto:246`,
"there is no per-pattern override"). The solver side would support it with `set_edge_policy`
per edge; giving `ExpressConnection` / `ExpressLink` their own `channels` is a separate proto
change (open question 5).

### 3.5 The seat solve's ladder

`kSolveLadder` (`physical_grouping_descriptor_matching.cpp:2953`) tries hard seams first, then
soft seams, each with and without the host cap. Under the per-edge design:

- The soft-seam rungs run only if some mesh pair has `strict == 0` and `relaxed > 0`.
  This replaces `relaxed_inter_mesh_policy_`.
- "Hard seams" means `set_best_effort_edges_required(true)`: every preferred-only edge must
  be realized. STRICT edges are hard in every rung.
- The rung order and the zero-link warning stay as they are.

## 4. MGD and MeshGraph

1. **Parse.** Replace `populate_inter_mesh_policy` with a per-pair demand table, built from
   every FABRIC connection and from `graph_topology`. Drop the "Cannot mix STRICT and RELAXED
   policies in the same graph" check.
2. **API** on `MeshGraphDescriptor` and `MeshGraph`:

   ```cpp
   struct InterMeshChannelDemand {
       std::uint32_t strict_channels = 0;
       std::uint32_t relaxed_channels = 0;
   };
   // Summed over every connection between the two meshes, both granularities.
   InterMeshChannelDemand get_inter_mesh_channel_demand(MeshId a, MeshId b) const;
   bool has_relaxed_inter_mesh_connection() const;
   ```

   `is_inter_mesh_policy_relaxed()` and `is_inter_mesh_policy_specified()` are deprecated,
   and their callers move to the new calls. Until then they answer "any RELAXED connection"
   and "any connection with a policy", not "the first connection".
3. **Requested maps.** `RequestedIntermeshPorts` entries get a policy, and
   `RequestedIntermeshConnections` becomes a demand map. A second mesh-level connection on a
   pair adds to the first instead of overwriting it.
4. **Merge.** Drop the policy-agreement fatal in `merge`, `validate_shared_inter_mesh_policy`,
   and the conflict throw in `generate_rank_bindings`. Seams are a disjoint union of the
   parts, so each keeps its own policy.
5. **`LogicalMultiMeshGraph`** carries the demand per mesh-level edge and per exit-node edge.
   `build_logical_multi_mesh_adjacency_graph_impl` fills both granularities in one pass (the
   TODO at `topology_mapper_utils.cpp:198`).
6. **Proto.** Comments only: the policy is per connection, and connections with different
   policies may be mixed. No schema change.

## 5. Mapper

1. **Seat solve.** `SatPlacementEnumerationSession` turns each mesh pair's demand into edge
   constraints on its `MappingConstraints<MeshId, ...>`: `add_required_edge_constraint(a, b,
   strict)` when `strict > 0`, and `add_preferred_edge_constraint(a, b, relaxed)` when
   `relaxed > 0`. Every mesh pair has a demand, so the default edge policy is never read; it
   is set to `STRICT` so that a missing demand fails loudly instead of going soft.
2. **Exit nodes.** `add_exit_node_constraints` reads each pair's demand. It applies the STRICT
   pair math to `strict` channels and the capped RELAXED math to `relaxed`. A pair may skip its
   constraint with a zero-link warning only when `strict == 0`.
3. **`complete_intra_mesh_for_placement`** takes the demand table instead of one inter-mesh
   mode (for the exit nodes), and sets the mesh's intra-mesh policy as the default edge
   policy instead of passing it to the solver (section 3.4).
4. **Config.** `TopologyMappingConfig::inter_mesh_validation_mode` becomes an override for
   tools and tests ("treat every inter-mesh connection as X"). It is unset by default, and the
   MGD's demands are used.
5. **`unique_shapes`.** This dedup collapses permutations of same-shape meshes. With mixed
   policies, swapping two meshes can move a STRICT seam onto a pair with fewer links, so the
   permutations are no longer equivalent. Turn `unique_shapes` off when an MGD mixes policies,
   the same gate the pipeline design uses.

## 6. Control plane (#49960)

1. Remove the four "either RelaxedGraph or Graph connections, not both" fatals.
2. **Binding** (`pair_logical_intermesh_ports`, single-host path, port descriptors): pick the
   quota per boundary. A boundary with device-level connections uses per-exit-node quotas; one
   with only mesh-level connections uses the channel total.
3. **Shortfall and validation** (`:3613`, `validate_requested_intermesh_connections`). Check
   each boundary against its demand:
   - fewer than `strict` channels bound: throw, except under a factory descriptor, as today;
   - a shortfall in the `relaxed` part: warn;
   - zero links: allowed only when `strict == 0`.

   The labels "strict (device-level)" and "relaxed (graph-level)" become device-level and
   mesh-level.
4. **Downed links.** `reject_used_downed_links_under_strict_policy` counts an inter-mesh
   downed link only if its boundary has `strict > 0`.
5. **System health.** `validate_system_health_against_mesh_graph_policy` treats the MGD as
   RELAXED if any connection is RELAXED.

## 7. Cable type: hard and preferred constraints by port type

Every Ethernet channel in the physical system descriptor has a `PortType`
(`EthConnection::port_type`, `physical_system_descriptor.hpp:74`): `TRACE`, `QSFP_DD`,
`WARP100`, `WARP400`, `LINKING_BOARD_1..3`, or `UNKNOWN`. The solver never sees it today:
`build_adjacency_graph_physical` (`topology_solver.cpp:164`) and the seat graph
(`build_sat_placement_seat_graph`, `physical_grouping_descriptor_matching.cpp:2667`) turn each
channel into one parallel edge, so a global edge is just a channel count. The per-edge
structure from section 3 is the natural place to add it: each global edge gets a count per
cable type, and each target edge says which types may count (hard) and which it prefers
(soft).

### 7.1 Solver API: channel classes

The solver stays generic and knows nothing about `PortType`. A **channel class** is an
integer the caller assigns; the mapper uses `static_cast<ChannelClass>(port_type)`.

```cpp
using ChannelClass = std::uint32_t;

// Global side: how many channels of each class a global edge has. The counts must add up to the
// edge's multiplicity in the global graph. Edges left unannotated have all channels in class 0.
bool set_global_edge_channel_classes(
    GlobalNode a, GlobalNode b, const std::map<ChannelClass, std::size_t>& channels_per_class);

// Hard: only channels of these classes count toward an edge's demand. Empty set = all classes.
void set_default_allowed_channel_classes(std::set<ChannelClass> classes);
bool set_edge_allowed_channel_classes(TargetNode a, TargetNode b, std::set<ChannelClass> classes);

// Soft: preference tiers, best first. A class that is not listed is in the last tier.
// PRIMARY is optimized before TIE_BREAK; both come after capacity and node hits (section 11.5).
enum class PreferenceStrength { PRIMARY, TIE_BREAK };
void set_default_channel_class_preference(
    std::vector<ChannelClass> best_first, PreferenceStrength strength = PreferenceStrength::PRIMARY);
bool set_edge_channel_class_preference(
    TargetNode a, TargetNode b, std::vector<ChannelClass> best_first,
    PreferenceStrength strength = PreferenceStrength::PRIMARY);
```

An edge has at most one preference at each strength. A PRIMARY and a TIE_BREAK preference
on the same edge are two separate objectives: the solver first minimizes the PRIMARY tiers,
then, among those solutions, the TIE_BREAK tiers.

These scope like the edge policy: per edge beats the whole-solve default. The resolved edge
from section 3 grows two fields:

```cpp
struct EdgeDemand {
    std::size_t required = 0;
    std::size_t preferred = 0;
    std::set<ChannelClass> allowed_classes;          // empty = all
    std::vector<ChannelClass> primary_preference;    // best first; empty = none
    std::vector<ChannelClass> tie_break_preference;  // best first; empty = none
};
```

### 7.2 Semantics

- **Hard (allowed classes).** For target edge `e` on global edge `g`, the usable channel count
  is the sum of `g`'s channels in `e`'s allowed classes. Every channel rule in section 3 uses
  that count instead of the multiplicity: a STRICT edge needs `required` allowed channels, a
  RELAXED edge needs at least one, and a BEST_EFFORT edge is realized only on allowed
  channels.
- **Soft (preference).** The tier of `e` on `g` is the worst tier among the best `c_e` allowed
  channels of `g` (`c_e` is `e`'s channel count): the edge is charged for the cables it would
  actually use. The objective minimizes the number of target edges in the worst tier, then
  the next worst, and so on (lexicographic). Tiers read directly as "prefer A over B" and need
  no weight tuning; integer costs per class could be added later if tiers aren't enough.
- **Objective order.** host cap and host packing, then preferred node hits, then realized
  preferred-only edges, then preferred channels, then **primary cable tiers** (what an MGD
  connection or mesh asks for), then **tie-break cable tiers** (the implicit default, section
  7.6), then tie-break node preferences. The first three are today's order. Capacity comes
  before cable type (open question 7). Section 11.5 has the encoding.
- **`UNKNOWN`.** Descriptors without cabling data report `UNKNOWN`. An `UNKNOWN` channel is
  always allowed, so an allowed list can't silently drop every link on such a system, and it
  sits in the last tier, with a single warning per solve. When every channel is `UNKNOWN`,
  preferences have no effect and the mapping is the same as today.

### 7.3 SAT encoding

| Piece | Encoding |
|---|---|
| Allowed classes | Per (target edge, global pair) usable count, computed before encoding. Support clauses and AC-3 arcs use pairs with enough usable channels; that is the same clause shape as today, with the neighbour list filtered per target edge |
| Channel thresholds | Count usable channels, not the multiplicity |
| Preference | For each target edge `e` and tier `T` worse than the best: a literal `z[e,T]`, with clauses `x[t1,g1] ∧ x[t2,g2] → z[e,T]` for each global pair whose tier for `e` is `T`. Pairs in the best tier add no clause. Minimize `Σ z[e,worst]`, then the next tier, with the binary search the realized-edge objective already uses (`:2033`) |
| Cost | `O(|E_target| × |E_global|)` ternary clauses for pairs outside the best tier, the same order as the support clauses. Nothing is added when no preference is set |

DFS: the allowed-class filter is a per-edge channel check, which DFS can do. Preferences are an
objective, so they need SAT; under DFS they only order candidates, as RELAXED channel
preference does today.

### 7.4 Where the classes come from

| Solve | Global edge | Where the counts per class are built |
|---|---|---|
| Intra-mesh | ASIC pair | `build_adjacency_graph_physical` already loops `get_eth_connections` (`topology_solver.cpp:164`); it also emits counts per `port_type` |
| Seat solve | footprint pair (`const Candidate*`) | `AdjacencyMatrixCache` / `saturated_link_count` (`physical_grouping_descriptor_matching.cpp:2395`) counts links between two footprints; it counts them per port type too, and `build_sat_placement_seat_graph` annotates each seat pair |
| Exit nodes | chip | `add_exit_node_constraints` can prefer exit chips whose inter-mesh links are in a better tier, as preferred node constraints, so the intra-mesh solve and the seat solve agree |
| Control plane binding | channel | the solver picks chip pairs, not channels. When a boundary has more channels than its quota, `pair_logical_intermesh_ports` binds the allowed channels in tier order. Shortfall checks count allowed channels only |

Intra-mesh fabric uses every channel between two adjacent chips, so an intra-mesh preference
changes which chip pairs a smaller mesh uses, not which channels.

### 7.5 How you say it

**In the MGD.** `Channels` gets two optional fields, so the same words work per inter-mesh
connection and per mesh (intra-mesh):

```proto
message Channels {
    int32 count = 1;
    optional Policy policy = 2;
    // Hard: only these cable types count toward `count`. Empty = any. UNKNOWN always counts.
    repeated PortType allowed_port_types = 3;
    // Soft: preferred cable types, best first. Unlisted types rank last.
    repeated PortType preferred_port_types = 4;
}
```

`PortType` is already defined in `physical_system_descriptor.proto`. The MGD proto imports it
rather than adding a fourth copy (open question 8).

For the whole MGD, a new top-level field replaces or switches off the built-in default
(section 7.6):

```proto
message PortTypeDefaults {
    // Replaces the built-in tie-break order for this MGD. Best first.
    repeated PortType preferred_port_types = 1;
    // No implicit preference at all; only explicit Channels fields apply.
    bool disable = 2;
}

message MeshGraphDescriptor {
    // ... fields 1-5 unchanged ...
    optional PortTypeDefaults port_type_defaults = 6;
}
```

Example: inter-mesh, prefer WARP400 over QSFP-DD, and only accept those two.

```
connections {
  nodes { mesh { mesh_descriptor: "M" mesh_id: 0 } }
  nodes { mesh { mesh_descriptor: "M" mesh_id: 1 } }
  channels {
    count: 2
    policy: STRICT
    allowed_port_types: [ WARP400, QSFP_DD ]
    preferred_port_types: [ WARP400 ]
  }
}
```

Two seatings that both give two links between M0 and M1, one over WARP400 and one over
QSFP-DD, are both valid; the solver picks the WARP400 one. A seating whose only links are
WARP100 is rejected, because WARP100 isn't allowed and the connection is STRICT. Under
RELAXED it would be accepted as zero-link with a warning.

Example: intra-mesh, prefer on-board traces and linking boards over cables when a smaller
mesh fits a larger system in several ways.

```
mesh_descriptors {
  name: "M"
  arch: BLACKHOLE
  device_topology { dims: [ 4, 4 ] dim_types: [ RING, RING ] }
  host_topology   { dims: [ 1, 1 ] }
  channels {
    count: 2
    policy: STRICT
    preferred_port_types: [ TRACE, LINKING_BOARD_1, LINKING_BOARD_2, LINKING_BOARD_3 ]
  }
}
```

**In C++, directly against the solver:**

```cpp
MappingConstraints<FabricNodeId, PhysicalNodeId> c;
c.set_default_edge_policy(EdgePolicy::STRICT);
for (const auto& [pair, per_type] : channels_per_port_type(psd, physical_graph)) {
    c.set_global_edge_channel_classes(pair.first, pair.second, per_type);
}
c.set_default_channel_class_preference({cls(PortType::TRACE), cls(PortType::LINKING_BOARD_1)});
// One edge must not use QSFP-DD at all:
c.set_edge_allowed_channel_classes(n0, n1, {cls(PortType::TRACE), cls(PortType::WARP400)});
auto result = solve_topology_mapping(logical_graph, physical_graph, c);
```

The orderings in these examples are illustrations, not recommendations: which cable to
prefer is the MGD author's call.

**Config override.** `TopologyMappingConfig` gets optional `port_type_preference`,
`allowed_port_types` and `disable_implicit_port_type_preference` for tools and tests. They
sit between the MGD-wide defaults and the built-in table (section 7.6).

### 7.6 Implicit preference: on by default

Most MGDs will never mention cables. The mapper should still prefer better cables when
everything else is equal. So every edge gets a **tie-break preference by default**, without
the MGD saying anything. It is an implicit preferred constraint, added the same way the mapper
already adds PGD pinnings as preferred node constraints
(`add_pgd_pinning_preferred_constraints`, `topology_mapper_utils.cpp:889`, called at `:1450`).

**Precedence, highest first:**

| Source | Scope | Strength |
|---|---|---|
| `Channels.preferred_port_types` on a connection | that inter-mesh connection | PRIMARY |
| `Channels.preferred_port_types` on a mesh descriptor | every intra-mesh edge of that mesh | PRIMARY |
| `MeshGraphDescriptor.port_type_defaults` | every edge in the MGD | TIE_BREAK |
| `TopologyMappingConfig::port_type_preference` | every edge in the solve | TIE_BREAK |
| built-in table, `default_port_type_preference()` | every edge in the solve | TIE_BREAK |

`port_type_defaults { disable: true }` or `disable_implicit_port_type_preference` removes the
three tie-break rows. Explicit preferences still apply.

**Why tie-break, below preferred node hits.** An implicit preference must never override
anything the user or the system asked for. Below the PGD pinning preferences, it only chooses
among mappings that are equal on everything else: same host use, same realized edges, same
channel counts, same pinning hits. That keeps the PGD-chosen layout stable and limits golden
changes to cases that were ties.

**The built-in table.** `default_port_type_preference()` lives in a new
`tt_metal/fabric/port_type_preference.hpp`, the one place that orders cable types. Nothing in
the codebase ranks port types today (`link_health` records `medium` but doesn't order it), so
the order needs a sign-off from the hardware and scaleout owners. The starting proposal follows
the order in [#36425](https://github.com/tenstorrent/tt-metal/issues/36425): TRACE first (same
host), then WARP400 (400G), then QSFP-DD (400G, lower priority than WARP400), then WARP100
(100G). The issue leaves the linking boards "based on their context"; they are placed after
TRACE here, to be confirmed:

```cpp
inline std::vector<PortType> default_port_type_preference() {
    return {PortType::TRACE,
            PortType::LINKING_BOARD_1, PortType::LINKING_BOARD_2, PortType::LINKING_BOARD_3,
            PortType::WARP400, PortType::QSFP_DD,
            PortType::WARP100};
    // UNKNOWN is never listed: it always ranks last.
}
```

The table may later depend on the arch or the cluster type; the function takes no arguments
until a second order is needed.

### 7.7 Implementation

The port-type rules are one more family of mapper-built constraints, next to rank bindings,
exit nodes, MGD pinnings and PGD preferred pinnings. One function builds them for both solves:

```cpp
// topology_mapper_utils.cpp
// Annotates every global edge with its channels per PortType and adds the allowed and
// preferred port-type constraints for this solve, in the precedence of section 7.6.
template <typename TargetNode, typename GlobalNode>
void add_port_type_constraints(
    MappingConstraints<TargetNode, GlobalNode>& constraints,
    const PortTypeRules& rules,                         // resolved from MGD + config, below
    const std::map<std::pair<GlobalNode, GlobalNode>,
                   std::map<PortType, std::size_t>>& channels_per_port_type);
```

```cpp
// What one solve needs, resolved once from the MGD and the config.
struct PortTypeRules {
    std::set<PortType> default_allowed;                                // mesh-level allowed, or empty
    std::vector<PortType> default_primary;                             // mesh-level preferred, or empty
    std::vector<PortType> tie_break;                                   // after precedence; empty if disabled
    std::map<std::pair<MeshId, MeshId>, std::set<PortType>> pair_allowed;     // inter-mesh only
    std::map<std::pair<MeshId, MeshId>, std::vector<PortType>> pair_primary;  // inter-mesh only
};
```

Steps:

1. **MGD.** `MeshGraphDescriptor` parses `allowed_port_types` / `preferred_port_types` per
   mesh and per connection, and `port_type_defaults`, into lookups next to the demand table
   (section 4): `get_mesh_port_type_rules(MeshId)`, `get_inter_mesh_port_type_rules(MeshId,
   MeshId)`, `get_port_type_defaults()`. Several connections on one pair with different
   explicit lists are rejected at parse ("conflicting preferred_port_types on M0-M1").
2. **Counts per type.** `build_adjacency_graph_physical` (`topology_solver.cpp:164`) returns
   the channels per `PortType` per ASIC pair alongside the graph. The seat solve's
   `AdjacencyMatrixCache` keeps a count per type next to `saturated_link_count`, and
   `build_sat_placement_seat_graph` returns a map per seat pair.
3. **Intra-mesh.** `complete_intra_mesh_for_placement` calls `add_port_type_constraints` after
   `add_pgd_pinning_preferred_constraints` (`topology_mapper_utils.cpp:1450`), with the mesh's
   rules. The intra-mesh candidate pools and the PGD grouping matches do the same with the
   tie-break preference only, so the pool's choice and the final mapping agree.
4. **Seat solve.** `SatPlacementEnumerationSession` calls it on its
   `MappingConstraints<GlobalMeshId, const Candidate*>`, with `pair_allowed` and
   `pair_primary` as per-edge constraints and the tie-break as the default.
5. **Exit nodes.** `add_exit_node_constraints` adds a preferred node constraint for exit chips
   whose inter-mesh channels are in the best available tier, at tie-break strength, so the
   intra-mesh solve picks the exit chips the seat solve assumed.
6. **Solver.** The tie-break objective runs after the preferred node hits. When no edge has a
   tie-break preference, or every global edge is in one tier, it adds nothing.
7. **Results.** `MappingResult` reports edges per tier for both strengths, and the mapper logs
   one line per solve ("cable tiers: 12 TRACE, 4 QSFP_DD"), so a golden diff says why it
   changed.
8. **Control plane.** Binding takes allowed channels in the same order: explicit first, then
   the tie-break table.
9. **`unique_shapes`.** It lists every distinct footprint rather than the best one, so the
   cable objectives are skipped there, as node hits are today. The cable choice is made by the
   seat solve and the final intra-mesh solve, which don't use `unique_shapes` (section 11.5).

**Goldens.** The implicit default changes which mapping wins a tie, so it can move goldens on
systems whose descriptors carry cabling data. A report test lists every golden that changes,
and the same PR updates them (section 9).

## 8. Implications

| Area | Effect |
|---|---|
| Existing MGDs | Unchanged. Every connection in a graph shares one policy today, so the demands produce the same constraints; goldens pin it |
| Newly accepted | MGDs that mix policies (rejected at parse today), merged parts that disagree (rejected at merge today), and device-level plus mesh-level connections in one MGD (rejected by the control plane today) |
| Hierarchical MGDs | The compatibility check is per graph, so two graphs may already differ, and the first FABRIC connection silently wins. After this change each is honoured. That is a behaviour change: list the MGDs it affects before landing |
| Solver API | The mode argument and `allow_unmatched_target_edges` become one constraint, `set_default_edge_policy`. Old signatures keep working, deprecated; about 190 uses in 13 files can move later |
| Intra-mesh | Same behaviour, now expressed as the default edge policy; ready for per-edge intra-mesh policy if the MGD ever gets it |
| Cable type, explicit | Opt-in. An allowed list can turn a STRICT connection UNSAT on hardware that only has other cables; the error names the connection and the types found |
| Cable type, implicit | On by default. It only breaks ties left after every other objective, so the constraints met never change, but which tied mapping wins can, on systems whose descriptors carry cabling data. Systems reporting only `UNKNOWN` are unchanged. `port_type_defaults { disable: true }` restores today's choice |
| Solve time | Mixed MGDs solve faster than all-RELAXED ones, because STRICT edges keep their pruning. Uniform MGDs are unchanged |
| Symmetry | `unique_shapes` must be off when policies mix (section 5) |
| Warnings | Zero-link and shortfall warnings name the connection and its policy |
| Pipeline design | A STRICT pipeline edge crossing a RELAXED connection may end with zero links. Tier B keeps the declared channels as an upper bound; Tier C decides on the bound links. Merging parts with different policies, which the pipeline design needs, stops failing |

## 9. Implementation plan

One PR, with the MGD side, the solver side, the mapper, the control plane and cable types
together. Every piece needs another to be useful (the demand table needs the solver encoding,
the solver change needs a caller, cable preferences need the per-edge structure), so splitting
it would only land code with no caller.

| Area | What the PR changes |
|---|---|
| Solver | `EdgePolicy`, `EdgeDemand`, `set_default_edge_policy`, `set_edge_policy`, `add_required_edge_constraint`, `add_preferred_edge_constraint`, `resolve_edge_demands`, `merge` (section 3.2); SAT steps from 3.3; `IndexedEdgeDemand` and resolution in `ConstraintIndexData` (11.1); the `SatObjective` schedule and the preferred-channel stage (11.5); validator and DFS per edge; overloads without `ConnectionValidationMode`, old ones deprecated; `set_allow_unmatched_target_edges` becomes an alias; session `set_best_effort_edges_required` |
| Solver, cable types | `ChannelClass`, `set_global_edge_channel_classes`, allowed classes, preferences at both scopes and both strengths (7.1); usable-count and tier tables, tier literals, TIE_BREAK node preferences (11.2, 11.4); tiers in `MappingResult` |
| MGD and MeshGraph | demand table and `get_inter_mesh_channel_demand` (section 4); requested maps carry policy and sum; `allowed_port_types` / `preferred_port_types` in `Channels` and `port_type_defaults` (7.5); drop the "Cannot mix" check and the merge-agreement fatals; proto and `MGD_README.md` comments; deprecate `is_inter_mesh_policy_relaxed` |
| Mapper | intra-mesh callers set the default edge policy (3.4); seat solve and exit nodes read demands (3.5, 5); `PortTypeRules`, `add_port_type_constraints`, counts per `PortType`, `default_port_type_preference()` (7.6, 7.7); config overrides; `unique_shapes` gate |
| Control plane | sections 6.1-6.5 (#49960); binding in cable-tier order; shortfall counts allowed channels |

The ~190 existing uses of `ConnectionValidationMode::` keep compiling through the deprecated
overloads, so the PR doesn't have to touch them.

**Behaviour changes in the PR, and how they're reviewed:**

- MGDs may mix policies, merged parts may disagree, and device-level and mesh-level
  connections may coexist. Nothing accepted today is rejected.
- Hierarchical MGDs whose graphs had different policies now honour each one (section 8). The
  PR lists the affected textprotos in its description.
- The implicit cable preference can change which tied mapping wins on systems whose
  descriptors carry cabling data. A report test lists every golden that moves, and the PR
  updates them, so the diff shows each change next to the cable tiers that caused it. The
  built-in order (open question 10) is settled in review before merge.
- The preferred-channel stage (section 11.5) makes the SAT engine maximize channels, which
  it never did. It can change which tied mapping wins, so the same report test covers it.

### Tests

- **Solver.** A required edge with no global edge is UNSAT. A preferred edge with no
  global edge solves with a warning, and is realized when a placement allows it. A
  required-plus-preferred edge meets its minimum and prefers the total. Degree and AC-3
  pruning still apply to required edges when other edges are preferred. Property test over
  the existing solver suites: a uniform default of `STRICT`, `RELAXED` or `BEST_EFFORT` gives
  the same mappings and warnings as the matching mode argument (and `allow_unmatched` for
  `BEST_EFFORT`). Precedence, the count check, conflicting defaults in `merge`, and a mode
  argument that disagrees with the default all fail as section 3.2 says.
- **Intra-mesh.** Every intra-mesh golden in `TopologyMapperUtilsTest` and the PGD suites
  is unchanged; a mesh whose default would resolve an edge to `required == 0` fails.
- **MGD.** A pair with a device-level STRICT 1 and a mesh-level RELAXED 2 gives demand
  `{1, 2}`. Two mesh-level connections on one pair sum. Each existing textproto gives the same
  demands as its global policy.
- **Mapper.** Three meshes: M0–M1 STRICT and M1–M2 RELAXED. A seating with no M1–M2 link
  is accepted with a warning; one with no M0–M1 link is rejected. Merging a STRICT part with a
  RELAXED part succeeds. `unique_shapes` is off for mixed MGDs.
- **Control plane.** The #49960 acceptance criteria. A STRICT boundary shortfall throws;
  a RELAXED one warns; a downed link on a RELAXED boundary is not rejected. Single-host and
  multi-host golden paths.
- **Cable type.** Two seatings with equal link counts, one over the preferred type: the
  preferred one wins. A STRICT edge whose only links are a disallowed type is UNSAT; the same
  edge under BEST_EFFORT is zero-link with a warning. `UNKNOWN` channels always count. A
  system with no cabling data maps exactly as today. Capacity beats cable tier (a RELAXED edge
  takes 4 QSFP-DD links over 2 WARP400 links), as section 7.2 orders them. The control plane
  binds preferred channels first.
- **Implicit default.** With no MGD fields, a tie between a TRACE edge and a QSFP-DD
  edge goes to TRACE; a PGD preferred pinning still beats the implicit order; an explicit
  `preferred_port_types` beats `port_type_defaults`, which beats the config, which beats the
  built-in table; `disable: true` gives today's mapping. A report test lists every golden the
  default changes.
- **Encoding.** The brute-force cross-check, CNF parity, per-piece, objective-order and
  limit tests in section 11.10.
- **Regression.** `TopologyMapperUtilsTest`, `MeshGraphDescriptorTests`, control-plane
  suites and the CPU-only fabric suites, unchanged for uniform MGDs.

## 10. Inter-mesh candidate channels from the topology mapper

With per-pair demands and cable rules in the mapper, the mapper is the one place that knows
which inter-mesh channels are **allowed**: the ones between the meshes it mapped, on cable
types the rules allow, on links that are up. This section adds an API that hands that set to
the control plane after the solve. The control plane keeps everything else it does today:
**channel selection** (budgets, trimming, round-robin) and **direction assignment** (Z claim,
N/E/S/W, overflow onto Z). What goes away is the gather: ranks no longer read their own cables
from the PSD and send them to rank 0, because the mapper already has the full PSD on rank 0.

Moving channel selection and direction assignment into the mapper as well is a stretch goal,
in section 13. It is not planned for implementation.

### 10.1 What the control plane does today

`generate_intermesh_connectivity` (`control_plane.cpp:3580`) runs after the mapper and does
five things:

| Step | Where | What |
|---|---|---|
| Gather | `generate_port_descriptor_table` (`:3737`), `gather_intermesh_cables_for_exit_nodes` (`:3649`) | each rank lists its exit-node cables toward each neighbor rank from the PSD; filters by MPI ownership, by requested exit nodes (device-level), and by `is_ethernet_link_up` (`:3725`); fatals if one src chip cables to more than one dst chip on a neighbor mesh (`:3696`) |
| Send to rank 0 | `forward_descriptors_to_controller` | MPI gather of every rank's `PortDescriptorTable` |
| Pair | `pair_logical_intermesh_ports` (`:4144`), rank 0 only | joins both sides of each cable by `connection_hash`; budgets per boundary (mesh-level) or per exit chip (device-level); trims a link to the remaining budget (`has_budget_after_trim`, `:4379`); work list boundary-major (device-level) or round-robin (mesh-level); places channels in three phases: `assign_z` boundaries claim Z, others take N/E/S/W, then overflow onto leftover Z; one neighbor mesh per (chip, direction) |
| Broadcast and apply | `forward_intermesh_connections_from_controller`, `convert_port_descriptors_to_intermesh_connections` (`:4476`) | every rank maps each `(mesh, port_id, hash)` back to its physical channel |
| Check and load | `:3613` shortfall, `validate_requested_intermesh_connections` (`:3818`), `rebuild_intermesh_exit_maps_from_connections` (`:3549`), `load_intermesh_connections` | STRICT shortfall throws, RELAXED warns; exit maps; routing tables |

The single-host path, `generate_intermesh_connections_on_local_host` (`:4588`), is a different
greedy algorithm for the same job.

The gather re-derives from scratch what the mapper already knew. The mapper solves on rank 0
with the full PSD (`TopologyMapper::build_mapping`, `topology_mapper.cpp:491`) and, with
section 7, decides which cable types may count. The gather knows nothing about cable types, so
the pairing can bind a channel on a cable type the mapper was told not to use, and the
mapper's choice and the control plane's choice can disagree.

### 10.2 What changes and what stays

| Concern | Today | After |
|---|---|---|
| Which channels exist between two mapped meshes | each rank reads its own cables from the PSD; MPI gather to rank 0 | the mapper on rank 0, from the full PSD, after the solve (`TopologyMapper::get_inter_mesh_candidates()`) |
| Cable type | ignored | channels on a disallowed `PortType` are left out of the table; each channel carries its type and tier |
| Liveness (`is_ethernet_link_up`) | checked per rank during the gather | each rank reports its down channels in the chip info it **already sends** to rank 0 before the solve (`topology_mapper.cpp:515`); the mapper leaves them out |
| Requested exit nodes (device-level) | filtered per rank during the gather | filtered by the mapper, same rule |
| How many channels | every live cable on the requested boundaries | the same: **every** allowed channel, not capped to the demand. A RELAXED boundary with more cabling than requested gets all of it as candidates |
| One dst chip per (src chip, neighbor mesh) | fatal per rank at gather | same check and same message, run once on rank 0 over the table |
| Channel selection: budgets, trim, round-robin / boundary-major | `pair_logical_intermesh_ports`, rank 0 | **unchanged**, with one change: within a link, channels are taken best cable tier first, so trimming keeps the preferred cables |
| Direction assignment | `pair_logical_intermesh_ports`, rank 0 | **unchanged** |
| Broadcast of the result, apply to ETH channels, routing tables, shortfall, validation | control plane | **unchanged** |

### 10.3 API

```cpp
// tt-metalium/experimental/fabric/inter_mesh_candidates.hpp

// One physical inter-mesh channel the mapper allows. Both ends are mapped fabric nodes on
// different meshes.
struct InterMeshCandidateChannel {
    FabricNodeId src;                 // exit node on the src mesh
    FabricNodeId dst;                 // peer exit node
    tt::tt_metal::AsicID src_asic;
    tt::tt_metal::AsicID dst_asic;
    std::uint8_t src_chan = 0;        // physical ETH channels
    std::uint8_t dst_chan = 0;
    tt::tt_metal::PortType port_type = tt::tt_metal::PortType::UNKNOWN;
    std::uint8_t tier = 0;            // 0 = most preferred, from PortTypeRules (section 7.7)
    std::size_t connection_hash = 0;  // std::hash<ExitNodeConnection>, as today
};

enum class InterMeshCandidateExclusion : std::uint8_t {
    LINK_DOWN,             // reported down by the owning rank
    PORT_TYPE_DISALLOWED,  // PortType not in the boundary's allowed list
    NOT_REQUESTED_EXIT,    // device-level boundary, src chip is not a requested exit node
};

struct InterMeshCandidates {
    // Keyed (src mesh, dst mesh), both directions, like PortDescriptorTable. Within a key,
    // sorted by (src chip, tier, src chan, dst chan).
    std::map<std::pair<MeshId, MeshId>, std::vector<InterMeshCandidateChannel>> channels;

    // What was left out, and why. For logs, the yaml dump, and shortfall messages.
    std::vector<std::pair<InterMeshCandidateChannel, InterMeshCandidateExclusion>> excluded;

    // The table pair_logical_intermesh_ports reads today, so the pairing code stays as it is.
    PortDescriptorTable to_port_descriptor_table() const;
};

struct InterMeshCandidateOptions {
    std::set<std::size_t> down_connection_hashes;  // from the ranks' chip info
    PortTypeRules port_type_rules;                 // section 7.7
};

// Pure: no cluster, no MPI. Same inputs, same table, in a CPU-only test.
InterMeshCandidates collect_inter_mesh_candidates(
    const MeshGraph& mesh_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& psd,
    const std::map<FabricNodeId, tt::tt_metal::AsicID>& fabric_node_to_asic,
    const InterMeshCandidateOptions& options);
```

`TopologyMapper` calls it on rank 0 after the mapping is final and keeps the result:

```cpp
// Valid on the rank that solved (rank 0, or every rank when generate_mapping_locally_).
// Other ranks get the paired result through the existing broadcast.
const InterMeshCandidates& TopologyMapper::get_inter_mesh_candidates() const;
```

Callers that build a control plane without the mapper (custom fabric topology) call
`collect_inter_mesh_candidates` themselves with their mapping; it is the same function.

### 10.4 How the table is built

For every requested boundary (mesh-level or device-level, using the demand table from
section 2), walk the PSD cables between the mapped ASICs of the two meshes and keep a channel
when all of these hold:

1. **Both ends are mapped**, to fabric nodes on the two meshes of the boundary. Channels into
   unmapped ASICs or onto a third mesh are not candidates.
2. **The link is up**: its `connection_hash` is not in `down_connection_hashes`.
3. **The cable type is allowed** for the boundary by `PortTypeRules`. `UNKNOWN` always counts,
   as in section 7.2. The channel's `tier` comes from the boundary's preferred list, then
   `port_type_defaults`, then the built-in order (section 7.6).
4. **Device-level boundaries only:** the src chip is a requested exit node, the same rule as
   `get_requested_exit_nodes` today.

Everything that passes goes in, **including channels beyond the demand**. Choosing among them
is the pairing's job.

**Order.** Within a key the channels are sorted by (src chip id, tier, src chan, dst chan).
That is today's gather order (`control_plane.cpp:3674`) with `tier` added, so with no cable
data every tier is 0 and the order is unchanged.

**Down links.** Before the solve, every rank already sends its chip info to rank 0
(`broadcast_chip_info_to_hosts`, `topology_mapper.cpp:515`). It adds one field: the
`connection_hash` of every PSD cable whose src ASIC this rank owns, whose dst ASIC another
rank owns, and whose src ETH link is down (`cluster.is_ethernet_link_up`, the check the gather
does today at `:3725`). `build_mapping` already has the `Cluster`. That set is a superset of
any inter-mesh channel the multi-host path can use, and it is computed before the mapping
exists, so the rank doesn't need to know which cables will be inter-mesh.

**Agreement with the solve.** Section 7 makes the seat solve and the exit-node constraints
count only allowed types, and the down links are excluded before the solve as well. So the
table contains only channels of the kind the mapper counted when it accepted the placement.

### 10.5 What the control plane changes

The multi-host path in `generate_intermesh_connectivity` becomes:

```cpp
if (!generate_mapping_locally_ && *(this->distributed_context_.get().size()) > 1) {
    AnnotatedIntermeshConnections paired;
    if (my_rank == 0) {
        const auto& candidates = topology_mapper_->get_inter_mesh_candidates();
        check_one_dst_chip_per_src_chip(candidates);   // today's :3696 fatal, same message
        auto port_descriptors = candidates.to_port_descriptor_table();
        paired = pair_logical_intermesh_ports(port_descriptors);  // unchanged except tier order
    }
    forward_intermesh_connections_from_controller(paired);        // unchanged broadcast
    intermesh_connections = apply_intermesh_connections_to_local_chips(paired);  // the apply half of :4476
} else {
    intermesh_connections = generate_intermesh_connections_on_local_host();      // filtered, see below
}
// shortfall, validation, exit maps, routing tables: unchanged
```

| Function | Change |
|---|---|
| `generate_port_descriptor_table`, `gather_intermesh_cables_for_exit_nodes`, `get_requested_exit_nodes`, `forward_descriptors_to_controller` | removed |
| `convert_port_descriptors_to_intermesh_connections` | split: the gather and pairing half goes; the apply half (build `cable_lookup` from the local PSD, then set `exit_node_directions_` and `intermesh_chan_to_peer_` per broadcast entry, `:4489-4585`) stays as `apply_intermesh_connections_to_local_chips` |
| `pair_logical_intermesh_ports` | one change: `PortDescriptor` gains `std::uint8_t tier`, and the per-link sort at `:4305` sorts by (tier, hash) instead of hash. `has_budget_after_trim` keeps the first channels of a link, so a trimmed link keeps its preferred cables. With every tier 0 the order is the same as today |
| `generate_intermesh_connections_on_local_host` | keeps its algorithm; it skips any cable whose `connection_hash` is not in the table, so single-host also respects cable types and down links |
| shortfall (`:3613`), `validate_requested_intermesh_connections`, `rebuild_intermesh_exit_maps_from_connections`, `load_intermesh_connections` | unchanged. A STRICT shortfall caused by disallowed types or down links names them, from `excluded` |

`PortDescriptor` / `PortDescriptorTable` stay, as the pairing's input type.

### 10.6 Out of scope: a link that goes down after the solve

Today's gather checks `is_ethernet_link_up` once, and nothing handles a link that goes down
after that. The down-link report in the chip info replaces that check at about the same point
in initialization, so behaviour is unchanged. Handling later link-downs is left for a
follow-up.

### 10.7 Implications

| Area | Effect |
|---|---|
| Consistency | The control plane can only bind channels the mapper allows: no channel on a disallowed cable type, none on a down link, none between meshes the mapper didn't map together |
| MPI | The gather of port descriptors goes away; the down-link report rides the existing chip-info message. The broadcast of the paired result stays |
| Control plane code | Pairing and direction assignment are unchanged apart from the tier sort; the gather code is removed |
| RELAXED boundaries | Get every allowed channel as a candidate, more than requested when the cabling has more, exactly as the gather gave today |
| Cable preference | Trimmed links keep their preferred cables. Which links the round-robin picks is unchanged; that is a stretch-goal concern (section 13) |
| Single-host | Same algorithm as today, restricted to the table |
| Testing | `collect_inter_mesh_candidates` is pure: CPU-only tests on PSD fixtures |
| Debugging | The table can be dumped next to `asic_to_fabric_node_mapping_rank_*.yaml` as `inter_mesh_candidates.yaml`: every boundary, its channels with type and tier, and every excluded channel with its reason |
| Parity | With no cable data and no down links, `to_port_descriptor_table()` equals the gathered table, so every multi-host pairing is identical to today's |

### 10.8 Implementation

One PR, after the policy PR (it needs the demand table and `PortTypeRules`):

| Area | Change |
|---|---|
| New | `inter_mesh_candidates.hpp/.cpp`: the types, `collect_inter_mesh_candidates`, `to_port_descriptor_table` |
| Mapper | down-link hashes in the chip info sent to rank 0; `collect_inter_mesh_candidates` after the final mapping; `get_inter_mesh_candidates()`; `inter_mesh_candidates.yaml` dump |
| Control plane | the new multi-host branch in 10.5; `apply_intermesh_connections_to_local_chips`; `PortDescriptor::tier` and the tier sort; the single-host filter; the removals in 10.5 |

Tests:
- **Parity.** For every multi-host golden in the control-plane and CPU-only fabric suites,
  `to_port_descriptor_table()` equals today's gathered table, and the pairing gives the same
  `AnnotatedIntermeshConnections`.
- **Filters.** A down link, a disallowed cable type, and a non-requested exit chip on a
  device-level boundary are each left out and listed in `excluded` with the right reason;
  `UNKNOWN` always passes.
- **More than requested.** A RELAXED boundary with 4 live channels and a demand of 2 gets 4
  candidates; the pairing binds 2.
- **Tier order.** A link with 2 QSFP-DD and 2 TRACE channels trimmed to 2 keeps the TRACE
  channels; with no cable data the bound channels are the same as today.
- **Single-host.** A cable missing from the table is never bound.
- **Multi-host.** Every rank gets the same paired result.

## 11. SAT encoding and constraint API in detail

Sections 3 and 7 say what changes. This section is the implementation reference: how each
constraint is set, how it reaches the SAT encoder, and the exact variables, clauses and
objective stages. File references are to `topology_solver.hpp` (API and index types) and
`topology_solver_sat.cpp` (encoder) unless noted.

### 11.1 From API call to encoder

A constraint goes through the same three layers as today's node constraints:
`MappingConstraints` stores what the caller said, `ConstraintIndexData` (`:879`) resolves it
to indices once per solve, and `TopologySatConstraintView` (`:1040`) hands it to the encoder.

| API call | Stored in `MappingConstraints` | Resolved in `ConstraintIndexData` | Read by |
|---|---|---|---|
| `set_default_edge_policy(p)` | `std::optional<EdgePolicy> default_edge_policy_` | `edge_demands` (below) | support clauses, realized literals, channel indicators |
| `set_edge_policy(a, b, p)` | `std::map<std::pair<T, T>, EdgePolicy> edge_policies_`, key `(min, max)` | `edge_demands` | same |
| `add_required_edge_constraint(a, b, n)`, `add_preferred_edge_constraint(a, b, n)` | `std::map<std::pair<T, T>, EdgeDemand> explicit_edge_demands_`; calls add up | `edge_demands` | same |
| `set_global_edge_channel_classes(a, b, counts)` | `std::map<std::pair<G, G>, std::map<ChannelClass, std::size_t>> global_edge_classes_` | `global_class_count` | usable-count table (11.2) |
| `set_default_allowed_channel_classes`, `set_edge_allowed_channel_classes` | `default_allowed_classes_`, `edge_allowed_classes_` | `IndexedEdgeDemand::allowed` | usable-count table |
| `set_default_channel_class_preference`, `set_edge_channel_class_preference` | one default and one per-edge map per `PreferenceStrength` | `IndexedEdgeDemand::primary`, `::tie_break` | tier table, tier literals |
| `add_preferred_constraint(t, globals, strength)` (new optional `strength`, default `PRIMARY`) | `preferred_mappings_` (PRIMARY, today's) and `tie_break_preferred_mappings_` | `preferred_global_indices`, `tie_break_preferred_global_indices` | node-hit indicators |
| `set_allow_unmatched_target_edges(true)` | deprecated alias: sets `default_edge_policy_ = BEST_EFFORT` | | |
| session: `set_best_effort_edges_required(bool)` | live session flag, replaces `set_allow_unmatched_target_edges` on the session (`:1671`) | | assumptions in `next()` |

New index types in `ConstraintIndexData`:

```cpp
// One per undirected target edge (t1 < t2) of the target graph.
struct IndexedEdgeDemand {
    std::size_t t1 = 0;
    std::size_t t2 = 0;
    std::size_t required = 0;
    std::size_t preferred = 0;
    std::vector<ChannelClass> allowed;    // sorted; empty = every class
    std::vector<ChannelClass> primary;    // best first; empty = none
    std::vector<ChannelClass> tie_break;  // best first; empty = none
};
std::vector<IndexedEdgeDemand> edge_demands;
std::vector<std::map<std::size_t, std::size_t>> edge_demand_of;  // t1 -> t2 -> index into edge_demands

// Global side: channels per class for each adjacent global pair; absent = all in class 0.
std::vector<std::map<std::size_t, std::vector<std::pair<ChannelClass, std::size_t>>>> global_class_count;
std::vector<std::vector<std::size_t>> tie_break_preferred_global_indices;
```

**Resolution** happens once, in the `ConstraintIndexData` constructor, for every target edge
in `GraphIndexData::target_adj_idx`:

1. Explicit numbers if any, else `set_edge_policy`, else the default, else `RELAXED`.
2. A policy becomes numbers from the edge's channel count `c = target_conn_count[t1][t2]`:
   STRICT `{c, 0}`, RELAXED `{1, c - 1}`, BEST_EFFORT `{0, c}`.
3. Explicit numbers must satisfy `required + preferred == c`; class counts on a global pair must
   add up to `global_conn_count`. A violation fails the solve with a message naming the edge,
   through the existing `trivial_reason` path, before anything is encoded.
4. Allowed classes and preferences: per edge if set, else the default.

`merge` combines two `MappingConstraints` by the rules in section 3.2: explicit numbers add,
conflicting policies or preferences on one edge are rejected.

### 11.2 Precomputed tables

Before encoding, for each target edge `e = (t1, t2)` and each candidate pair `(ga, gb)` with
`ga` in the domain of `t1`, `gb` in the domain of `t2`, and `ga`, `gb` adjacent:

- **usable count** `u_e(ga, gb)`: the channels of `(ga, gb)` in `e`'s allowed classes, plus
  every `UNKNOWN` channel. With no class annotation it is `global_conn_count[ga][gb]`, today's
  number.
- **tier** `τ_e(ga, gb)`, once per strength that `e` has a preference at: sort the usable
  channels by the preference (unlisted classes last), take the best `c_e`, and use the worst
  tier among them. An edge pays for the cables it would actually use.

The tables are sparse vectors per edge of `(i1, i2, u, τ_primary, τ_tie_break)`, built in the
same loop that `topology_sat_encode_adjacency_support` (`:1116`) already runs over domain
pairs, so they cost no more than the support clauses do.

### 11.3 Hard constraints

Unchanged: one assignment literal `x[t, g]` per target and domain global
(`topology_sat_create_assignment_variables`, `:979`), exactly one per target (`:1012`),
injectivity (`:1027`), resources, same-rank groups, node cardinality.

| Piece | Today | Per edge |
|---|---|---|
| Domain degree filter (`topology_sat_build_initial_domains`, `:842`) | skipped when soft | `g` stays in `dom(t)` only if it has at least `deg_req(t)` neighbours, where `deg_req(t)` counts `t`'s edges with `required > 0` |
| AC-3 (`topology_sat_apply_arc_consistency`, `:880`) | skipped when soft; STRICT compares multiplicity with `c` | arcs only for required edges; `ga` is supported if some `gb` in `dom(t2)` has `u_e(ga, gb) >= required_e` |
| Required edge, `required_e >= 1` | STRICT: `multiplicity >= c`; RELAXED: adjacent | for each `ga` in `dom(t1)`: `¬x[t1,ga] ∨ ⋁ { x[t2,gb] : u_e(ga,gb) >= required_e }`, and the same from `t2`. Today's clause shape with `required_e` and `u` |
| Preferred-only edge, `required_e == 0` | every edge guarded when soft | one realized literal `r_e`: `¬r_e ∨ ¬x[t1,ga] ∨ ⋁ { x[t2,gb] : u_e(ga,gb) >= 1 }`, both directions. Only these edges get `r_e` |
| `prune_active_lit` (`:1364`) | guards the hard domains of a soft encoding | built from the domains that required edges pruned |

A uniform STRICT default gives every edge `required = c`, which is today's STRICT clause
exactly; uniform RELAXED gives `required = 1`, today's RELAXED clause; uniform BEST_EFFORT gives
today's soft encoding.

### 11.4 Indicators for the objectives

**Realized edges.** The `r_e` literals above, with today's full-width counter
(`topology_sat_build_at_least_threshold_lits`, `:512`): threshold literal `R[k]` true means at
least `k` of the `r_e` are true.

**Preferred channels.** For each edge with `preferred_e > 0` and each level
`k = required_e + 1 .. required_e + preferred_e` (capped at 24 levels per edge, as today):

```
I[e,k]  ⇔  ⋁ { x[t1,ga] ∧ x[t2,gb] : u_e(ga,gb) >= k }      (topology_sat_define_indicator_as_or_of_pairwise_and, :1491)
I[e,k+1] → I[e,k]                                              (chain, new; helps the counter)
```

and one counter over all `I` literals, giving thresholds `C[k]`. Maximizing `Σ I` is
maximizing `Σ_e min(c_e, u_e)`, the same score DFS uses to order candidates.

Today these indicators are built under RELAXED
(`topology_sat_append_relaxed_channel_threshold_literals`, `:1564`), but the result `ch_lits`
is discarded (`:1858-1862`), so the SAT engine never maximizes channels today. The per-edge
design makes it a real objective stage (11.5). The existing limits stay: at most 512 pairs per
indicator and 2048 indicators per solve; past them the stage is skipped with one debug line
naming the edge, and the mapping is still valid.

**Cable tiers.** For each edge with a preference at strength `S` whose candidate pairs span
more than one tier, with tiers `0 .. L-1` (0 best), one literal per tier above the best:

```
z_S[e,j]  means "e uses a pair of tier >= j",  j = 1 .. L-1
for each candidate pair with τ = τ_e(ga,gb) >= 1:   ¬x[t1,ga] ∨ ¬x[t2,gb] ∨ z_S[e,τ]
z_S[e,j] → z_S[e,j-1]                               j = 2 .. L-1
```

Pairs in tier 0 add nothing, and an edge whose pairs all share one tier gets no literals. A
BEST_EFFORT edge left with zero links selects no adjacent pair, so it isn't charged. One
counter per `(S, j)` over the `z_S[·,j]` literals.

**Node hits.** Today's PRIMARY indicators `p_t ⇔ ⋁ x[t,g]` over preferred globals
(`topology_sat_append_preferred_hit_indicators`, `:1439`) stay as they are. TIE_BREAK node
preferences get their own indicators `q_t`, built the same way, with a full-width counter.

### 11.5 Objective schedule

Today `Impl` holds a list of assumption stages (`s.stages`, `:1874-1883`): it tries host
packing and preferred node hits together, then each alone, then neither, and keeps the first
that solves. After that it maximizes realized edges by binary search on `R[k]` under the kept
stage (`:2005-2084`). The design generalizes the second half into an ordered list:

```cpp
struct SatObjective {
    enum class Kind : std::uint8_t { MAXIMIZE, MINIMIZE };
    Kind kind;
    std::vector<int> thresholds;  // thresholds[k-1] true => at least k counted literals true
    std::string name;             // for PlacementSolveStats / MappingResult and logs
};
std::vector<SatObjective> objectives;  // in Impl, after `stages`
```

`next()` first runs today's stage fallback unchanged, then each objective in order. A
MAXIMIZE objective binary-searches the largest `k` with `thresholds[k-1]` satisfiable; a
MINIMIZE objective the smallest `k` with `¬thresholds[k]` satisfiable. The optimum is then
**kept as an assumption** for every later objective, so the order is strictly lexicographic.
Each probe tries the symmetry hint first, as today; an undecided probe counts as UNSAT, and two
undecided probes end that objective with the best model found
(`kZeroLinkMaximizeMaxUndecidedSolves`, today's budget).

| Order | Objective | Kind | Status |
|---|---|---|---|
| 1 | host packing (`minimize_lit`) | assumption stage | today |
| 2 | preferred node hits, at least the computed lower bound (`preferred_lit`) | assumption stage | today |
| 3 | realized preferred-only edges, `R[k]` | MAXIMIZE | today |
| 4 | preferred channels, `C[k]` | MAXIMIZE | new (indicators exist today, unused) |
| 5 | PRIMARY cable tiers, worst tier first: one MINIMIZE per `j = L-1 .. 1` | MINIMIZE | new |
| 6 | TIE_BREAK cable tiers, same way | MINIMIZE | new |
| 7 | TIE_BREAK node hits, `q_t` | MAXIMIZE | new |

**Why node hits stay above realized edges.** In today's code, node hits come first: the
node-hit stage stays assumed while realized edges are maximized. Keeping today's results for
today's inputs fixes rows 1 to 3. So PRIMARY cable tiers rank below node hits as well as below capacity, and
TIE_BREAK below PRIMARY. Ranking PRIMARY tiers above node hits would mean moving node hits
below realized edges, which changes the result of every soft solve that has pinnings (open
question 11).

**`unique_shapes`.** Today node hits are skipped under `unique_shapes` (`:1813`), because that
mode lists every distinct footprint rather than the best one. Objectives 4 to 7 are skipped the
same way. The candidate pools are listed with `unique_shapes` on, and the cable choice is made
later, by the seat solve and by the final intra-mesh solve on the chosen footprint, neither of
which uses `unique_shapes`. This replaces step 9 of section 7.7.

**Cost when unused.** An objective with no literals is not added. With no classes, no
preferences, and no tie-break node preferences, only objective 4 is new, and only for edges
with `preferred > 0`. Objective 4 can change which tied mapping wins, so it is covered by the
same golden report test as the implicit cable default (section 9).

### 11.6 How the mapper sets the constraints

**Intra-mesh solve** (`complete_intra_mesh_for_placement`, `topology_mapper_utils.cpp:1459`):

```cpp
MappingConstraints<FabricNodeId, PhysicalNodeId> intra = ...;  // pinnings, ranks, exit nodes as today
intra.set_default_edge_policy(mesh_is_relaxed ? EdgePolicy::RELAXED : EdgePolicy::STRICT);
add_port_type_constraints(intra, rules.for_mesh(mesh_id), channels_per_port_type);  // section 7.7
auto result = solve_topology_mapping(logical_graph, physical_graph, intra);       // no mode argument
```

**Seat solve** (`build_sat_placement_constraints`,
`physical_grouping_descriptor_matching.cpp:2732`; target `GlobalMeshId`, global
`const Candidate*`):

```cpp
constraints.set_default_edge_policy(EdgePolicy::STRICT);
for (const auto& [pair, demand] : inter_mesh_channel_demands) {    // section 4
    if (demand.strict > 0) {
        constraints.add_required_edge_constraint(pair.first, pair.second, demand.strict);
    }
    if (demand.relaxed > 0) {
        constraints.add_preferred_edge_constraint(pair.first, pair.second, demand.relaxed);
    }
}
for (const auto& [seats, links_per_type] : seat_pair_links_per_port_type) {  // AdjacencyMatrixCache, per type
    constraints.set_global_edge_channel_classes(seats.first, seats.second, to_classes(links_per_type));
}
add_port_type_constraints(constraints, rules.for_seat_solve(), seat_pair_links_per_port_type);
```

The mesh-level target graph carries `strict + relaxed` parallel edges per pair, so the count
check in 11.1 holds. The seat graph already carries one parallel edge per link
(`build_sat_placement_seat_graph`, `:2723`), and the per-type counts add up to it. A pair with
`strict == 0` is preferred-only and gets an `r_e`. The ladder's hard-seam rungs call
`set_best_effort_edges_required(true)`, which assumes those `r_e`; STRICT pairs are hard in
every rung.

**Exit nodes** (`add_exit_node_constraints`, `topology_mapper_utils.cpp:955`) keep their
required and cardinality node constraints. The minimum exit-pair count comes from the pair's
`strict` part only, so a RELAXED part never forces an exit chip. The cable preference is a
TIE_BREAK node preference:

```cpp
intra.add_preferred_constraint(fabric_node, exit_asics_in_best_tier, PreferenceStrength::TIE_BREAK);
```

**Stretch goal (section 13).** Its seat-solve conditions need no new encoding: placeable
counts change the global edge multiplicities and class counts, and the per-seat filter is
`add_forbidden_constraint`, which removes the seat from the domain (a unit clause through
`refresh_constraints`, `:1900`). Only the pair-level block is a new clause, described there.

### 11.7 DFS backend

| Constraint | DFS |
|---|---|
| Required edge, allowed classes | the edge check compares `u_e` with `required_e`, where it compares the multiplicity with `c` under STRICT today |
| Preferred-only edge | rejected with today's message: zero-link edges need SAT (`topology_solver.tpp:1354`) |
| Preferred channels | candidate order by `Σ min(c_e, u_e)`, today's channel-match score using `u` |
| Cable tiers, TIE_BREAK node hits | candidate order only, best tier first |

### 11.8 Incremental sessions

| Change | Without re-encoding? |
|---|---|
| node forbidden / required constraints | yes: unit clauses through `refresh_constraints` (`:1900`), as today |
| `set_best_effort_edges_required` | yes: assumptions on `r_e`, replacing today's seam toggle (`:1924-1934`) |
| blocking clauses (`exclude_mapping`) | yes, as today |
| edge demands, allowed classes, preferences | no: they shape the clauses, so the session is rebuilt. The seat solve already rebuilds when its pools grow |

### 11.9 Size

| Piece | Variables | Clauses |
|---|---|---|
| Support clauses | none | unchanged: one per (edge, endpoint, domain global), each up to the size of the other domain |
| Realized literals | one per preferred-only edge | unchanged shape, guarded |
| Preferred-channel indicators | at most 24 per edge, 2048 per solve, plus one per qualifying pair | 3 per pair plus 1 + pairs per indicator, plus one chain clause per level |
| Cable tiers | at most `L - 1` per edge with a preference | one ternary clause per candidate pair above tier 0, plus the chain |
| Counters | `m × k` per counter | about `4 m k` per counter (`topology_sat_build_counter_matrix`, `:434`) |

### 11.10 Tests for the encoding

- **Brute-force cross-check.** Random small instances (up to 5 targets, 7 globals) with random
  demands, classes, allowed sets and preferences at both strengths. Enumerate every injective
  mapping, compute feasibility and the objective vector of 11.5, and check that SAT returns a
  feasible mapping with the optimal vector, and UNSAT exactly when nothing is feasible.
- **Encoding parity.** A uniform STRICT, RELAXED or BEST_EFFORT default produces the same
  variables and clauses as the matching mode argument, apart from objective 4. Checked through
  a CNF dump hook, a debug-only `TopologySatSolver` sink.
- **Pieces.** A STRICT edge whose only pairs use disallowed classes is UNSAT; `UNKNOWN`
  channels always count; a `z` literal fires only for its tier and the chain holds; an edge
  whose pairs share one tier adds no literals.
- **Order.** Hand-built cases where two objectives conflict, one per adjacent pair of rows in
  11.5, check that the higher row wins.
- **Limits.** Past the indicator caps the stage is skipped, the mapping is valid, and one debug
  line names the edge.

## 12. Open questions

1. **Device-level RELAXED.** Should the named chips stay pinned as exit nodes (proposed), or
   should RELAXED also let the link move to other chips?
2. **`graph_topology` plus an explicit connection on the same pair:** add up (proposed), or
   the explicit connection overrides?
3. **Ladder or objective.** The realized-edge objective already prefers hard seams. Keep the
   hard-seam rungs for their interplay with the host cap, or drop them?
4. **Override lifetime.** How long should `TopologyMappingConfig::inter_mesh_validation_mode`
   stay as a forcing override?
5. **Per-connection intra-mesh policy.** Give express links (or device-topology dimensions)
   their own `channels { count policy }` so the intra-mesh solve gets per-edge policies, or
   keep one policy per mesh?
6. **Unset default.** An unset default edge policy means `RELAXED`, today's default argument.
   Flip it to `STRICT`, matching the MGD default, once every caller sets it explicitly?
7. **Capacity versus cable type.** Section 7.2 ranks channel count above cable tier. Should a
   preferred cable type win over more channels instead, or be configurable?
8. **`PortType` in the MGD proto.** Import it from `physical_system_descriptor.proto`
   (proposed), or define an MGD-owned enum so the MGD schema doesn't depend on the PSD's?
9. **Tiers or weights.** Lexicographic tiers (proposed), or integer costs per type for cases
   such as "two QSFP-DD links are as good as one WARP400"?
10. **Built-in cable order.** Who signs off on `default_port_type_preference()`? Is the
    #36425 order right, and where do the linking boards go? Does it need to depend on arch or
    cluster type?
11. **Cable tiers versus node hits.** Today preferred node hits outrank realized edges
    (section 11.5), so keeping today's results puts both cable strengths below node hits.
    Keep that (proposed), or move node hits below realized edges so an explicit MGD cable
    preference can outrank a PGD-preferred layout, at the cost of changing soft solves that
    have pinnings?
12. **Candidates per rank.** The table lives on rank 0, where the pairing runs. Should it also
    ride the chip-info broadcast, so every rank can log or dump it, or stay rank-0 only
    (proposed)?
13. **Stretch goal (section 13).** If it is picked up: is the per-seat filter enough to keep
    binding rejections rare on SC16 / SC20 and the dual- and quad-galaxy systems, or does it need
    capacity per exit chip from the start?

## 13. Stretch goal (experimental): channel selection and direction assignment in the mapper

**Status: optional, not planned for implementation.** Section 10 is the plan. This section
records the design for doing more in the mapper, so it can be picked up later.

### 13.1 Why both move together

Section 10 gives the mapper's allowed channels to the control plane and leaves the choice to
the control plane's pairing. A natural next step is to let the mapper also **choose** the
channels, since it has the global view: every boundary, every cable, every cable type.

Choosing channels without choosing directions is the wrong split, though. A chip's channels
toward one neighbor mesh must all sit in one logical direction, and each direction serves one
neighbor (the `find_free_dir` / `dir_owner` rule, `control_plane.cpp:4183`). A selection that
ignores directions can be legal in count and still bad in direction:

- **Overflow.** It picks 4 channels from chip 3 toward M1 when chip 3's only free direction
  has 2 ports. The control plane then drops 2 channels, or fails a STRICT boundary that a
  different selection would have met.
- **Congestion.** It piles a boundary's channels onto one chip and one direction, while other
  exit chips of the same boundary have idle ports. The count is met but one router direction
  carries the whole boundary.
- **Contention.** Two boundaries share an exit chip. The selection gives the first one the
  chip's best direction and leaves the second one only Z, or nothing.

All three are decided by which channels get which ports. So if the mapper chooses channels, it
should assign directions in the same step, and the control plane only applies the result.

### 13.2 What moves

| Concern | Section 10 | Stretch |
|---|---|---|
| Allowed channels | mapper | mapper |
| Channel selection (budgets, one dst chip per src chip, round-robin, cable tier) | control plane | mapper |
| Direction assignment (Z claim, N/E/S/W, overflow onto Z) | control plane | mapper, in the same pass, with a congestion objective |
| Unbindable placement | control plane throws | mapper rejects the placement and tries the next one |
| Rank-0 pairing and its broadcast | control plane | gone: the plan rides the mapper's existing chip-info broadcast |
| Apply to ETH channels, routing tables, shortfall policy, validation | control plane | control plane |

### 13.3 API

```cpp
// tt-metalium/experimental/fabric/inter_mesh_binding.hpp

struct InterMeshChannelBinding {
    InterMeshCandidateChannel channel;  // section 10.3
    port_id_t src_port;                 // (direction, index) on src
    port_id_t dst_port;                 // (direction, index) on dst
    bool strict = true;                 // fills the STRICT part of the demand, else the RELAXED part
};

struct InterMeshBoundaryBinding {
    MeshId a;                           // a < b
    MeshId b;
    InterMeshChannelDemand demand;      // section 4
    std::size_t allowed_channels = 0;   // candidates for this boundary
    std::vector<InterMeshChannelBinding> channels;
    std::vector<std::string> warnings;  // RELAXED shortfall, zero-link, tier fallback
};

struct InterMeshBindingPlan {
    std::vector<InterMeshBoundaryBinding> boundaries;  // sorted by (a, b)
    std::size_t strict_shortfall() const;
    AnnotatedIntermeshConnections to_annotated_connections() const;  // what the control plane loads
};

// What a failed binding reports, so the mapper knows what to forbid (13.5).
struct InterMeshBindingFailure {
    MeshId a;
    MeshId b;
    enum class Kind : std::uint8_t {
        FOOTPRINT,  // not enough allowed channels between the two footprints, whatever the chip mapping
        PLACEMENT,  // depends on which fabric node sits on which chip: device-level pins, ports per direction
    } kind;
    std::string message;
};

// Port layout per fabric node: directions with edge ports, ports per direction, assign_z
// boundaries. Built once from MeshGraph::get_mesh_edge_ports_to_chip_id (mesh_graph.cpp:560-615).
class PortCapacityModel { /* ... */ };

// Exactly one of plan / failure is set.
struct InterMeshBindingResult {
    std::optional<InterMeshBindingPlan> plan;
    std::optional<InterMeshBindingFailure> failure;
};

// Pure: no cluster, no MPI.
InterMeshBindingResult resolve_inter_mesh_binding(
    const MeshGraph& mesh_graph,
    const InterMeshCandidates& candidates,
    const PortCapacityModel& port_capacity);

const InterMeshBindingPlan& TopologyMapper::get_inter_mesh_binding() const;
```

### 13.4 How channels and directions are chosen

`resolve_inter_mesh_binding` reuses the body of `pair_logical_intermesh_ports`, moved to the
mapper, with the candidate table as input and three changes.

1. **Links.** Group each boundary's candidates by (src chip, dst chip). Per src chip and
   neighbor mesh, keep one link: most channels, then best tier, then lowest dst chip id. This
   replaces the one-dst-chip fatal with a rule.
2. **Demand.** Device-level pins first, boundary-major, as today. Then mesh-level `strict`,
   then up to `relaxed` more, round-robin across boundaries and links as today.
3. **Directions, with a congestion objective.** Today's phases stay (Z claim for `assign_z`,
   then N/E/S/W, then overflow onto Z). Within a phase, when a channel can go to more than one
   link, take the link whose src and dst directions have the **fewest channels already
   assigned**, then the best tier, then the lowest chip id. The result spreads a boundary over
   its exit chips and directions instead of filling the first one. With one candidate link per
   choice, which is the common case, it picks the same channels as today.
4. **Result.** If any STRICT channel can't be placed, return a failure with the boundary and
   its kind: `FOOTPRINT` when the boundary has fewer allowed channels than its STRICT demand
   (that depends only on which chips the two footprints cover), otherwise `PLACEMENT`.

### 13.5 Keeping the seat solve and the binding in step

Once the binding can fail inside the mapper, the mapper should rarely propose a placement that
fails it, and should recover when it does.

**Decision: no channel selection inside the seat solve.** The seat solve works on footprints
(`const Candidate*`), not chips. Which channels can be bound depends on the exit chips, which
the intra-mesh solve picks later. Encoding channel selection means solving seats, intra-mesh
mappings and channels jointly, with variables per channel for every candidate seat pair. That
is the largest problem in the stack, paid on every solve. Instead, three cheap necessary
conditions, and the exact binding after each placement.

**Condition 1: placeable link count per seat pair.** Today `AdjacencyMatrix::build`
(`physical_grouping_descriptor_matching.cpp:2434`) counts, for each (from seat, to seat), every
boundary link of the from seat that lands in the to seat (`boundary_fabric_links()`, one entry
per parallel link). `build_sat_placement_seat_graph` then adds one seat-graph edge per link
(`:2723`). The change: count **placeable** channels instead.

- `Candidate` keeps the src chip of each boundary link next to its dst
  (`boundary_link_src_dense_`, parallel to `boundary_link_dense_`), and only counts links whose
  cable type is allowed for the boundary (section 7).
- For each (from seat, to seat), group the links by (src chip, dst chip). Per src chip keep the
  largest group (the one-dst-chip rule) and cap it at `max_ports_per_direction(mesh)`, the
  largest port count of any direction of any fabric node of the mesh, from the
  `PortCapacityModel`. The cap has to be orientation-free, because the seat solve doesn't know
  which fabric node lands on which chip; the max is an upper bound, so the condition stays
  necessary, never too strict.
- The matrix entry is the sum. `uint8_t` saturation at 255 stays.

The SAT clauses don't change; they get smaller numbers.

**Condition 2: aggregate per seat.** For each candidate seat `s` of mesh `m`, the sum over all
neighbors of its placeable channels (condition 1, against the best seat of each neighbor) must
cover the sum of the STRICT demands of `m`'s boundaries. This is a unary test on the
candidate, so it isn't a SAT constraint at all: `build_sat_placement_constraints` (`:2732`)
adds the failing seats as forbidden seats for `m`. It catches an exit region too small for all
of a mesh's boundaries together, which condition 1 misses because it looks at one pair at a
time.

**Condition 3: exit-node capacity.** `add_exit_node_constraints`
(`topology_mapper_utils.cpp:955`) derives the minimum number of exit chips from
`max_physical_exit_edges_per_asic_toward_mesh` (`:1110`), raw link counts. It counts with the
same rule as condition 1 instead: per exit ASIC, `min(links to its one dst chip, ports per
direction)`. This runs inside the intra-mesh solve, where the fabric node is known, so it can
use the exact port count of the node rather than the mesh-wide max.

**The backstop.** In `MultiMeshSolutionEnumerator::next` (`topology_mapper_utils.cpp:1770`),
after `complete_intra_mesh_for_placement` succeeds, call `resolve_inter_mesh_binding`. On
failure:

| Failure kind | Action | Why |
|---|---|---|
| `FOOTPRINT` | `add_forbidden_seat_pair(a, footprint_a, b, footprint_b)` | the failure depends only on the two footprints, so no seating with both can bind; each footprint stays usable next to other neighbors |
| `PLACEMENT` | `exclude_mapping(seating)`, the existing whole-seating exclusion (`physical_grouping_descriptor_matching.cpp:3452`) | the same footprints with a different chip mapping might bind, so only this exact combination is ruled out |

Today a failed intra-mesh solve calls `add_forbidden_constraint(*failed_placed)` (`:1830`),
which forbids that mesh's footprint everywhere. That is too strong for either kind of binding
failure.

**New primitive: a partial-mapping blocking clause.** `add_forbidden_seat_pair` needs to
forbid two variables together, `¬x(a, s_a) ∨ ¬x(b, s_b)`, without saying anything about the
other meshes. Nothing does that today:

- `topology_sat_add_blocking_clause_for_mapping_impl` (`topology_solver_sat.cpp:329`) builds a
  clause over every target and returns false on an unmapped (`-1`) entry (`:361`).
- The DFS engine's `mapping_is_excluded` (`topology_solver.tpp:3416`) compares whole mappings
  with `==`.

The change:

1. `topology_sat_add_partial_blocking_clause(solver, enc, partial)`: one negative literal per
   mapped entry, `-1` entries skipped, at least one entry required. It ignores `unique_shapes`,
   because a pair block is about two specific seats, not a shape class.
2. DFS: `blocked_partial_mappings_`, checked as a subset match (every mapped entry of the block
   equals the mapping's entry) during the search, so a blocked pair is pruned as soon as both
   meshes are assigned.
3. `TopologyMappingEnumerationSession::exclude_partial_mapping(const std::map<TargetNode,
   GlobalNode>&)`, routed to both.
4. `SatPlacementEnumerationSession` stores `forbidden_seat_pairs_` as footprints, like
   `extra_forbidden_` (`physical_grouping_descriptor.hpp:501`), because pools grow and
   candidates are rebuilt. `MasterSolve::restart` (`physical_grouping_descriptor_matching.cpp:3258`)
   replays them next to `excluded_seat_maps()`. Footprint dedup can map one footprint to several
   `Candidate*` (`seats_matching`), so a pair expands to one clause per (seat_a, seat_b)
   combination; in practice each side has one or two.

**Stats.** `PlacementSolveStats` (`physical_grouping_descriptor.hpp:158`) gains
`binding_rejections_footprint`, `binding_rejections_placement`, and `binding_elapsed`. If real
systems show more than a handful of rejections per solve, the next step is to encode capacity
per exit chip in the seat solve. Full channel selection still isn't needed for that.

### 13.6 What the control plane keeps

```cpp
void ControlPlane::generate_intermesh_connectivity() {
    const auto& plan = topology_mapper_->get_inter_mesh_binding();
    enforce_inter_mesh_shortfall_policy(plan);  // STRICT throws (FSD: records), RELAXED warns
    auto intermesh_connections = plan.to_annotated_connections();
    apply_intermesh_connections_to_local_chips(intermesh_connections);
    validate_requested_intermesh_connections(intermesh_connections);
    rebuild_intermesh_exit_maps_from_connections(intermesh_connections);
    routing_table_generator_->load_intermesh_connections(intermesh_connections);
}
```

Single-host and multi-host take the same path. `pair_logical_intermesh_ports` (moved to the
mapper), `forward_intermesh_connections_from_controller`, and
`generate_intermesh_connections_on_local_host` are removed. `validate_requested_intermesh_connections`
stays as an independent check of the mapper's output.

### 13.7 When to pick it up

Any one of these is a reason:

- section 10 is in and bindings show the problems in 13.1: STRICT shortfalls that a
  different selection would meet, or one direction carrying a whole boundary;
- a solve objective needs directions, for example keeping a ring's inter-mesh hops on
  consistent directions;
- a consumer needs channels and directions at mapping time, for example a pipeline builder
  planning routes before the control plane exists.

The cost to weigh: the mapper's output, its broadcast payload and its goldens then carry
logical ports, so a change to the port layout (torus reservations, Z ports, fabric config)
shows up as a mapping change.

### 13.8 Implementation sketch (not scheduled)

| Area | Change |
|---|---|
| New | `inter_mesh_binding.hpp/.cpp`: the types, `PortCapacityModel`, `resolve_inter_mesh_binding` (body moved from `pair_logical_intermesh_ports`) |
| Solver | partial blocking clause in SAT; subset-match blocks in DFS; `exclude_partial_mapping` on the session |
| Seat solve | placeable counts in `AdjacencyMatrix::build`; per-seat filter in `build_sat_placement_constraints`; `forbidden_seat_pairs_` and replay in `MasterSolve::restart`; `add_forbidden_seat_pair` |
| Mapper | capacity-aware `add_exit_node_constraints`; binding after each intra-mesh success in `MultiMeshSolutionEnumerator::next`; failure kinds to pair block or seating exclusion; stats; plan in the chip-info broadcast; `get_inter_mesh_binding()`; `inter_mesh_binding.yaml` dump |
| Control plane | the `generate_intermesh_connectivity` in 13.6; the removals |

Tests it would need:
- **Parity.** With one candidate link per choice, the bindings equal section 10's pairing on
  every multi-host golden.
- **Congestion.** A boundary with two exit chips of 2 channels each and a demand of 2 binds one
  channel on each chip, not both on the first.
- **Contention.** Two boundaries sharing an exit chip both get a direction when one exists.
- **Seat-solve conditions.** A seat pair with enough cables but too few ports in one direction
  isn't proposed; a seat too small for all its boundaries is filtered out.
- **Backstop.** A `FOOTPRINT` failure blocks only that pair: each seat is still used next to a
  different neighbor. A `PLACEMENT` failure excludes only that seating.
- **Partial blocking clause.** SAT and DFS agree on a pair block; the block survives a pool
  growth and session rebuild.
