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
6. **Inter-mesh binding moves to the mapper.** With demands and cable rules in the mapper,
   choosing which physical channels realize each inter-mesh connection moves there too, behind
   `resolve_inter_mesh_binding` and `TopologyMapper::get_inter_mesh_binding()`. The control
   plane keeps direction assignment for now, as a pure function every rank runs on the same
   plan, so the rank-0 gather and broadcast go away (section 10).

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

## 3. The solver: policy as a constraint, not an argument

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
| Channel thresholds (`:1564`) | RELAXED: levels 1..count on every edge | levels `required+1 .. required+preferred`, only on edges with `preferred > 0` |
| Realized-edge objective (`:1867`, binary search at `:2033`) | over every edge when soft | over preferred-only edges |
| Seam toggle (`set_allow_unmatched_target_edges`, `:1926`) | assume every realized literal | `set_best_effort_edges_required(bool)`: assume the preferred-only edges' realized literals, so `BEST_EFFORT` edges act as `RELAXED`; required edges are hard in both modes |

A uniform `STRICT` or `RELAXED` default produces exactly today's encoding for that mode, and a
uniform `BEST_EFFORT` default produces today's soft encoding. That is the property test in
section 9.

The objective order is unchanged: host cap and host objective, then the number of realized
preferred edges, then channel thresholds, then preferred node hits.

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
// PRIMARY ranks above preferred node hits; TIE_BREAK ranks below them (section 7.6).
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
- **Objective order.** host cap and host objective, then realized preferred edges, then
  channel thresholds, then **primary cable tiers** (what an MGD connection or mesh asks for),
  then preferred node hits, then **tie-break cable tiers** (the implicit default, section
  7.6). Capacity comes before cable type (open question 7).
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
the order needs a sign-off from the hardware and scaleout owners. A starting proposal, by
fewest connectors and shortest reach, to be confirmed:

```cpp
inline std::vector<PortType> default_port_type_preference() {
    return {PortType::TRACE,
            PortType::LINKING_BOARD_1, PortType::LINKING_BOARD_2, PortType::LINKING_BOARD_3,
            PortType::WARP400, PortType::WARP100,
            PortType::QSFP_DD};
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
9. **`unique_shapes`.** It collapses mappings on the same set of global nodes, but two such
   mappings can use different edges and so different cables. With a preference active, the
   enumerator keeps the best-tier representative: SAT finds it first under the objective, so
   it is enough to record the first.

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
| Solver | `EdgePolicy`, `EdgeDemand`, `set_default_edge_policy`, `set_edge_policy`, `add_required_edge_constraint`, `add_preferred_edge_constraint`, `resolve_edge_demands`, `merge` (section 3.2); SAT steps from 3.3; validator and DFS per edge; overloads without `ConnectionValidationMode`, old ones deprecated; `set_allow_unmatched_target_edges` becomes an alias; session `set_best_effort_edges_required` |
| Solver, cable types | `ChannelClass`, `set_global_edge_channel_classes`, allowed classes, preferences at both scopes and both strengths (7.1); SAT pieces from 7.3; tiers in `MappingResult` |
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
- **Regression.** `TopologyMapperUtilsTest`, `MeshGraphDescriptorTests`, control-plane
  suites and the CPU-only fabric suites, unchanged for uniform MGDs.

## 10. Inter-mesh binding in the topology mapper

With per-pair demands and cable rules in the mapper, the control plane's inter-mesh
resolution is mostly a second, separate solve of the same problem. This section moves the
**binding** (which physical channels realize each inter-mesh connection) into the topology
mapper behind an API, and leaves **direction assignment** (which logical port, N/E/S/W/Z, each
channel gets) in the control plane for now, as a pure function that can move later.

### 10.1 What the control plane does today

`generate_intermesh_connectivity` (`control_plane.cpp:3580`) runs after the mapper and does
five things:

| Step | Where | What |
|---|---|---|
| Gather | `generate_port_descriptor_table` (`:3737`), `gather_intermesh_cables_for_exit_nodes` (`:3649`) | each rank lists its exit-node cables toward each neighbor rank from the PSD; filters by MPI ownership, by requested exit nodes (device-level), and by `is_ethernet_link_up` (`:3725`); fatals if one src chip cables to more than one dst chip on a neighbor mesh (`:3696`) |
| Send to rank 0 | `forward_descriptors_to_controller` | MPI gather of every rank's `PortDescriptorTable` |
| Pair | `pair_logical_intermesh_ports` (`:4144`), rank 0 only | joins both sides of each cable by `connection_hash`; budgets per boundary (mesh-level) or per exit chip (device-level); work list boundary-major (device-level) or round-robin (mesh-level); places channels in three phases: `assign_z` boundaries claim Z, others take N/E/S/W, then overflow onto leftover Z; one neighbor mesh per (chip, direction) |
| Broadcast and apply | `forward_intermesh_connections_from_controller`, `convert_port_descriptors_to_intermesh_connections` (`:4476`) | every rank maps each `(mesh, port_id, hash)` back to its physical channel |
| Check and load | `:3613` shortfall, `validate_requested_intermesh_connections` (`:3818`), `rebuild_intermesh_exit_maps_from_connections` (`:3549`), `load_intermesh_connections` | STRICT shortfall throws, RELAXED warns; exit maps; routing tables |

The single-host path, `generate_intermesh_connections_on_local_host` (`:4588`), is a different
greedy algorithm for the same job.

Meanwhile the mapper has already done most of the thinking. It solves on rank 0 with the full
PSD (`TopologyMapper::build_mapping`, `topology_mapper.cpp:491`), chooses which chips are exit
nodes (`add_exit_node_constraints`), and broadcasts the result to every rank. The control plane
then re-derives the cables from scratch, with its own MPI round trip and its own rules. The
two can disagree: the mapper can accept a placement that the control plane's pairing can't
bind (for example the one-dst-chip-per-src fatal, or a direction with too few ports), and the
failure only shows up after the mapping is fixed.

### 10.2 What moves

| Concern | Today | After |
|---|---|---|
| Which cables exist between two mapped meshes | control plane, per rank, then gathered | mapper, from the full PSD on rank 0 |
| Liveness (`is_ethernet_link_up`) | checked per rank during gather | each rank reports its down inter-mesh channels in the chip info it **already sends** to rank 0 before the solve (`topology_mapper.cpp:515`); the mapper excludes them |
| Budgets per boundary / per exit chip, STRICT vs RELAXED | `pair_logical_intermesh_ports` | mapper, from the demand table (section 2) |
| Cable type preference | none | mapper (section 7) |
| One dst chip per (src chip, neighbor mesh) | fatal at gather | a binding rule: the mapper picks one dst chip per src chip (most allowed channels, then best tier), so cabling that has more than one no longer fails |
| Port capacity per chip | implicit in `find_free_dir` | an explicit capacity model, shared with the seat solve (10.4) |
| Placement that can't be bound | throws in the control plane | the mapper rejects the placement and tries the next one, as it already does for intra-mesh failures (`failing_pair_out`) |
| Direction / port assignment (Z claim, N/E/S/W, overflow) | `pair_logical_intermesh_ports`, rank 0 | **stays in the control plane**, extracted as a pure function run on every rank (10.5) |
| Applying to ETH channels, routing tables, link health, shortfall policy, validation | control plane | control plane, reading the mapper's binding |

### 10.3 API

```cpp
// tt-metalium/experimental/fabric/inter_mesh_binding.hpp

// One bound physical channel of an inter-mesh connection.
struct InterMeshChannelBinding {
    FabricNodeId src;                 // exit node on the lower mesh id of the boundary
    FabricNodeId dst;                 // peer exit node
    tt::tt_metal::AsicID src_asic;
    tt::tt_metal::AsicID dst_asic;
    std::uint8_t src_chan = 0;        // physical ETH channels
    std::uint8_t dst_chan = 0;
    tt::tt_metal::PortType port_type = tt::tt_metal::PortType::UNKNOWN;
    std::size_t connection_hash = 0;  // std::hash<ExitNodeConnection>, as today
    bool strict = true;               // fills the STRICT part of the demand, else the RELAXED part
};

// Everything decided for one mesh pair.
struct InterMeshBoundaryBinding {
    MeshId a;                          // a < b
    MeshId b;
    InterMeshChannelDemand demand;     // section 4
    std::size_t live_channels = 0;     // physical channels available after liveness
    std::vector<InterMeshChannelBinding> channels;  // chosen, in a deterministic order
    bool assign_z = false;             // MGD assign_z_direction, for direction assignment
    std::vector<std::string> warnings; // RELAXED shortfall, zero-link, tier fallbacks
};

struct InterMeshBindingPlan {
    std::vector<InterMeshBoundaryBinding> boundaries;  // sorted by (a, b)
    const InterMeshBoundaryBinding* find(MeshId a, MeshId b) const;
    std::vector<FabricNodeId> exit_nodes(MeshId from, MeshId to) const;
    std::size_t strict_shortfall() const;              // > 0 means the plan is invalid under STRICT
};

// What the binding may use.
struct InterMeshBindingOptions {
    std::set<std::size_t> excluded_connection_hashes;  // down links reported by the ranks
    PortTypeRules port_type_rules;                     // section 7.7
    PortCapacityModel port_capacity;                   // 10.4
};

// Pure: no cluster, no MPI. Same inputs, same plan, on any rank or in a CPU-only test.
InterMeshBindingPlan resolve_inter_mesh_binding(
    const MeshGraph& mesh_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& psd,
    const std::map<FabricNodeId, tt::tt_metal::AsicID>& fabric_node_to_asic,
    const InterMeshBindingOptions& options);
```

`TopologyMapper` calls it after the mapping is final, keeps the plan, broadcasts it with the
chip info it already broadcasts, and exposes it:

```cpp
const InterMeshBindingPlan& TopologyMapper::get_inter_mesh_binding() const;
```

Callers that build a control plane without the mapper (custom fabric topology) call
`resolve_inter_mesh_binding` themselves with their mapping; it is the same function.

### 10.4 How the binding is chosen

Per boundary, the candidates are the PSD cables between the mapped ASICs of the two meshes,
minus excluded hashes and disallowed port types, grouped into **links** (one link = one
(src chip, dst chip) pair and its channels). Then, deterministically (sorted by chip id, then
channel):

1. **Device-level pins first.** Each device-level connection takes its `count` channels from its
   pinned link, best cable tier first. A STRICT pin that can't be filled makes the plan invalid;
   a RELAXED one warns.
2. **One dst chip per src chip and neighbor mesh.** Among a src chip's links into one neighbor
   mesh, keep one: most allowed channels, then best tier, then lowest dst chip id.
3. **Mesh-level demand.** Fill `strict` channels, then up to `relaxed` more, from the remaining
   links. Order: today's round-robin across boundaries and links (so a shared exit chip is not
   drained by whichever boundary comes first), with cable tier as the tie-break within a round.
4. **Capacity.** A chip's channels toward one neighbor mesh must fit in one logical direction,
   and the neighbors on one chip must fit in its directions, one neighbor per direction (the
   rule `find_free_dir` enforces today). The `PortCapacityModel` holds, per chip, the
   directions it has edge ports in and their port counts, built from
   `MeshGraph::get_mesh_edge_ports_to_chip_id` (`mesh_graph.cpp:560-615`), plus which
   boundaries are `assign_z`. Before accepting a channel the binding checks that a direction
   assignment still exists, by calling the direction-assignment function (10.5) on the partial
   plan. That function is cheap (a few dozen channels per chip at most), so the check costs
   little.
5. **Result.** `strict_shortfall()` is the STRICT channels the plan couldn't place. If it is
   non-zero, the mapper treats the placement as failed and enumerates the next one, the same
   way `complete_intra_mesh_for_placement` reports a failing mesh pair. Only when no placement
   binds does the mapper fail, with a message naming the boundary, the demand, and the live
   channels found.

The seat solve uses the same `PortCapacityModel` when it counts links between two footprints
(`AdjacencyMatrixCache` / `saturated_link_count`), so a seating it accepts has enough
placeable channels, not just enough cables.

### 10.5 Direction assignment: extracted now, moved later

The three placement phases of `pair_logical_intermesh_ports` (Z claim for `assign_z`,
N/E/S/W, overflow onto leftover Z) become a pure function:

```cpp
// tt_metal/fabric/intermesh_port_assignment.hpp
// Assigns a logical port (direction, index) on both sides of every bound channel.
// Pure and deterministic; returns std::nullopt if no assignment exists.
std::optional<AnnotatedIntermeshConnections> assign_intermesh_ports(
    const InterMeshBindingPlan& plan, const PortCapacityModel& port_capacity);
```

In this step the **control plane** calls it, on every rank, on the broadcast-identical plan. No
rank-0 pairing and no second broadcast are needed, because every rank computes the same
answer from the same input. The mapper calls the same function only as the feasibility check
in 10.4 and discards the ports.

Moving direction assignment into the mapper later is then a small change: the mapper keeps the
ports it already computed, `InterMeshChannelBinding` gains `src_port` / `dst_port`, and the
control plane stops calling the function. Doing it later is useful for two reasons:
- the seat solve could then reject placements that have enough channels but no valid
  direction assignment, instead of only checking capacity per chip;
- direction choices could join the solve's objective, for example to keep a ring's inter-mesh
  hops on consistent directions.

### 10.6 What the control plane keeps

`generate_intermesh_connectivity` becomes:

```cpp
void ControlPlane::generate_intermesh_connectivity() {
    const auto& plan = topology_mapper_->get_inter_mesh_binding();
    enforce_inter_mesh_shortfall_policy(plan);       // STRICT throws (FSD: records), RELAXED warns
    auto intermesh_connections = assign_intermesh_ports(plan, port_capacity_model());
    TT_FATAL(intermesh_connections.has_value(), "...");  // the mapper already checked it
    apply_intermesh_connections_to_local_chips(*intermesh_connections, plan);  // ETH chan directions, chan_to_peer
    validate_requested_intermesh_connections(*intermesh_connections);
    rebuild_intermesh_exit_maps_from_connections(*intermesh_connections);
    routing_table_generator_->load_intermesh_connections(*intermesh_connections);
}
```

Single-host and multi-host take the same path. Removed:
`generate_port_descriptor_table`, `gather_intermesh_cables_for_exit_nodes`,
`get_requested_exit_nodes`, `forward_descriptors_to_controller`,
`forward_intermesh_connections_from_controller`, `pair_logical_intermesh_ports` (its body moves
to `assign_intermesh_ports`), `generate_intermesh_connections_on_local_host`, and
`PortDescriptor` / `PortDescriptorTable`. `convert_port_descriptors_to_intermesh_connections`
shrinks to `apply_intermesh_connections_to_local_chips`, which looks up channels in the plan
instead of re-reading the PSD.

**A link that goes down between the solve and fabric init.** Each rank re-checks
`is_ethernet_link_up` on its own bound channels. A bound channel that is now down is reported
through `LinkHealth` like any downed link. All ranks agree on the verdict through one
all-reduce of the down hashes, the only MPI step left in inter-mesh setup, and the STRICT /
RELAXED / FSD policy then applies on every rank.

### 10.7 Implications

| Area | Effect |
|---|---|
| Consistency | The mapper and the control plane can no longer disagree: a placement the mapper accepts is bindable by construction |
| MPI | One gather and one broadcast of port descriptors go away; the binding rides on the existing chip-info broadcast; one small all-reduce for late link-downs |
| Single-host | Uses the same algorithm as multi-host. Bindings that the old single-host greedy chose differently can change; the golden report lists them |
| Cabling with several dst chips per src chip | Works instead of fataling |
| Testing | `resolve_inter_mesh_binding` and `assign_intermesh_ports` are pure, so CPU-only tests can run them on PSD fixtures without a cluster or MPI |
| Debugging | The plan can be dumped next to the existing `asic_to_fabric_node_mapping_rank_*.yaml` as `inter_mesh_binding.yaml`: every boundary, its demand, and each bound channel with its cable type |
| Parity | With uniform policies, no cable data, and cabling that the old path accepted, multi-host bindings are identical to today's: same candidates, same round-robin or boundary-major order, same phases |

### 10.8 Implementation

One PR, after the policy PR (it needs the demand table and the port-type rules):

| Area | Change |
|---|---|
| New | `inter_mesh_binding.hpp/.cpp` (types, `resolve_inter_mesh_binding`, `PortCapacityModel`); `intermesh_port_assignment.hpp/.cpp` (`assign_intermesh_ports`, moved from `pair_logical_intermesh_ports`) |
| Mapper | down-link hashes in the chip info sent to rank 0; binding after the final mapping; rejected placements feed the enumerator; plan in the chip-info broadcast; `get_inter_mesh_binding()`; seat solve uses `PortCapacityModel`; `inter_mesh_binding.yaml` dump |
| Control plane | the new `generate_intermesh_connectivity`; the removals in 10.6; late link-down all-reduce |

Tests:
- **Parity.** For every multi-host golden in the control-plane and CPU-only fabric suites, the
  new path gives the same `AnnotatedIntermeshConnections` as today.
- **Binding.** Device-level pins filled first; the one-dst-chip rule picks the larger link
  instead of fataling; round-robin keeps a shared exit chip from being drained; cable tiers
  break ties; excluded hashes are never bound.
- **Capacity.** A placement whose channels fit per chip but have no direction assignment is
  rejected by the mapper, and the next placement is tried.
- **Multi-host.** Every rank gets the same plan; a link that goes down after the solve gives the
  same verdict on every rank.
- **Single-host.** Golden report of bindings that differ from the old single-host greedy.

## 11. Open questions

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
10. **Built-in cable order.** Who signs off on `default_port_type_preference()`, and is
    "fewest connectors, shortest reach" the right rule? Does it need to depend on arch or
    cluster type?
11. **Strength of the implicit default.** Tie-break below preferred node hits (proposed), or
    above them, so cables can outrank a PGD-preferred layout?
12. **Binding in the solve or after it.** Section 10 binds after each candidate placement and
    rejects failures. Should channel selection instead be encoded in the seat solve, so the
    solver never proposes an unbindable placement?
13. **When to move direction assignment.** Right after section 10 lands, or only once a solve
    objective needs it (ring direction consistency)?
14. **Late link-downs.** One all-reduce and the existing policy (proposed), or re-run the binding
    on the surviving links, which needs a second broadcast?
