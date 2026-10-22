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
   node constraints: `add_required_edge_constraint` (STRICT) and
   `add_preferred_edge_constraint` (RELAXED). The SAT encoding already has a per-edge
   "realized" literal; today it is created for every edge or none, and it is created only for
   RELAXED edges. `ConnectionValidationMode` and `allow_unmatched_target_edges` stay only as
   defaults for edges with no edge constraint.
3. **Mapper and control plane.** Exit-node constraints, intermesh binding and validation look
   up each mesh pair's demand instead of the global mode.

So yes, STRICT / RELAXED can be a required constraint instead of a solver argument. The MGD
side and the solver side should land together (section 8): the per-pair demand is useless
until the solver can encode it, and the solver change has no caller until the MGD supplies it.

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

### 3.1 API

`MappingConstraints<TargetNode, GlobalNode>` already has node constraints (required,
preferred, forbidden, cardinality). It gains edge constraints:

```cpp
// The target edge (a, b) must map onto a global edge with at least min_channels channels.
// Calls on the same edge add up. Returns false if a or b is not a target node.
bool add_required_edge_constraint(TargetNode a, TargetNode b, std::size_t min_channels = 1);

// The target edge (a, b) is best effort: realized when the placement allows, preferring up to
// `channels` channels on top of any required minimum. Zero is allowed. Calls add up.
void add_preferred_edge_constraint(TargetNode a, TargetNode b, std::size_t channels);

// Per-edge demand, for the backends and for validation.
struct EdgeDemand {
    std::size_t required = 0;   // hard minimum; > 0 means the edge must exist
    std::size_t preferred = 0;  // soft extra on top of `required`
};
const std::map<std::pair<TargetNode, TargetNode>, EdgeDemand>& edge_demands() const;
```

A target edge with no edge constraint keeps today's behaviour, chosen by the solve's
`ConnectionValidationMode` and `allow_unmatched_target_edges`. Both stay, as the default for
unconstrained edges, so every existing caller is unchanged. A caller that gives every edge a
constraint never reads them.

### 3.2 SAT encoding (`topology_solver_sat.cpp`)

| Step | Today | Per edge |
|---|---|---|
| 1. Domain degree filter (`:856`) | skipped for every target when soft | a global must have at least as many neighbours as the target has **required** edges |
| 2. AC-3 (`:887`) | skipped for the whole problem when soft | runs over required arcs only |
| 3b. `prune_active_lit` (`:1364`) | guards the hard domains of a soft encoding | the same, built from the required-only domains |
| 6. Adjacency support (`:1121`) | every edge hard, or every edge guarded by a realized literal | required edge: unguarded support clauses, channel-compatible with its `required` minimum. Preferred-only edge: guarded by its realized literal |
| Channel thresholds (`:1564`) | RELAXED: levels 1..count on every edge | levels `required+1 .. required+preferred`, only on edges with `preferred > 0` |
| Realized-edge objective (`:1867`, binary search at `:2033`) | over every edge when soft | over preferred-only edges |
| Seam toggle (`set_allow_unmatched_target_edges`, `:1926`) | assume every realized literal | assume the preferred-only edges' realized literals; required edges are hard in both modes |

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

### 3.3 The seat solve's ladder

`kSolveLadder` (`physical_grouping_descriptor_matching.cpp:2953`) tries hard seams first, then
soft seams, each with and without the host cap. Under the per-edge design:

- The soft-seam rungs run only if some mesh pair has `strict == 0` and `relaxed > 0`.
  This replaces `relaxed_inter_mesh_policy_`.
- "Hard seams" means every preferred-only edge is realized; STRICT edges are hard in every
  rung.
- The rung order and the zero-link warning stay as they are.

## 4. MGD and MeshGraph

1. **Parse.** Replace `populate_inter_mesh_policy` with a per-pair demand table, built from
   every FABRIC connection and from `graph_topology`. Drop the "Cannot mix STRICT and RELAXED
   policies in the same graph" check (section 8 says when).
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
   `relaxed > 0`.
2. **Exit nodes.** `add_exit_node_constraints` reads each pair's demand. It applies the STRICT
   pair math to `strict` channels and the capped RELAXED math to `relaxed`. A pair may skip its
   constraint with a zero-link warning only when `strict == 0`.
3. **`complete_intra_mesh_for_placement`** takes the demand table instead of one mode.
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

## 7. Implications

| Area | Effect |
|---|---|
| Existing MGDs | Unchanged. Every connection in a graph shares one policy today, so the demands produce the same constraints; goldens pin it |
| Newly accepted | MGDs that mix policies (rejected at parse today), merged parts that disagree (rejected at merge today), and device-level plus mesh-level connections in one MGD (rejected by the control plane today) |
| Hierarchical MGDs | The compatibility check is per graph, so two graphs may already differ, and the first FABRIC connection silently wins. After this change each is honoured. That is a behaviour change: list the MGDs it affects before landing |
| Solve time | Mixed MGDs solve faster than all-RELAXED ones, because STRICT edges keep their pruning. Uniform MGDs are unchanged |
| Symmetry | `unique_shapes` must be off when policies mix (section 5) |
| Warnings | Zero-link and shortfall warnings name the connection and its policy |
| Pipeline design | A STRICT pipeline edge crossing a RELAXED connection may end with zero links. Tier B keeps the declared channels as an upper bound; Tier C decides on the bound links. Merging parts with different policies, which the pipeline design needs, stops failing |

## 8. Implementation plan

Land the MGD side and the solver side as one PR stack. The parse-time "Cannot mix" check stays
until the last PR, so no MGD is accepted before every consumer can handle it.

| PR | Contents | Behaviour change |
|---|---|---|
| A. Solver edge constraints | `add_required_edge_constraint`, `add_preferred_edge_constraint`, `EdgeDemand`; SAT steps from 3.2; validator per edge; DFS rejects zero-link preferred edges | none: no caller sets edge constraints yet |
| B. MGD demand tables | `InterMeshChannelDemand`, `get_inter_mesh_channel_demand`, `has_relaxed_inter_mesh_connection`; requested maps carry policy; sum instead of overwrite | none for valid MGDs, except the overwrite fix |
| C. Mapper | seat solve and exit nodes read demands (sections 3.3, 5); config override; `unique_shapes` gate; drop the merge-agreement fatals | merged parts with different policies are accepted |
| D. Control plane | sections 6.1-6.5 (#49960) | device-level and mesh-level connections may coexist |
| E. Allow mixing | drop the MGD 1.0 "Cannot mix" check; proto and `MGD_README.md` comments; deprecate `is_inter_mesh_policy_relaxed` | MGDs may mix policies |

A and B are independent and can be reviewed in parallel. C needs both; D and E need C.

### Tests

- **Solver (A).** A required edge with no global edge is UNSAT. A preferred edge with no
  global edge solves with a warning, and is realized when a placement allows it. A
  required-plus-preferred edge meets its minimum and prefers the total. Degree and AC-3
  pruning still apply to required edges when other edges are preferred. Uniform demands give
  the same mappings as the global modes (property test over the existing solver suites).
- **MGD (B).** A pair with a device-level STRICT 1 and a mesh-level RELAXED 2 gives demand
  `{1, 2}`. Two mesh-level connections on one pair sum. Each existing textproto gives the same
  demands as its global policy.
- **Mapper (C).** Three meshes: M0–M1 STRICT and M1–M2 RELAXED. A seating with no M1–M2 link
  is accepted with a warning; one with no M0–M1 link is rejected. Merging a STRICT part with a
  RELAXED part succeeds. `unique_shapes` is off for mixed MGDs.
- **Control plane (D).** The #49960 acceptance criteria. A STRICT boundary shortfall throws;
  a RELAXED one warns; a downed link on a RELAXED boundary is not rejected. Single-host and
  multi-host golden paths.
- **Regression.** `TopologyMapperUtilsTest`, `MeshGraphDescriptorTests`, control-plane
  suites and the CPU-only fabric suites, unchanged for uniform MGDs.

## 9. Open questions

1. **Device-level RELAXED.** Should the named chips stay pinned as exit nodes (proposed), or
   should RELAXED also let the link move to other chips?
2. **`graph_topology` plus an explicit connection on the same pair:** add up (proposed), or
   the explicit connection overrides?
3. **Ladder or objective.** The realized-edge objective already prefers hard seams. Keep the
   hard-seam rungs for their interplay with the host cap, or drop them?
4. **Override lifetime.** How long should `TopologyMappingConfig::inter_mesh_validation_mode`
   stay as a forcing override?
