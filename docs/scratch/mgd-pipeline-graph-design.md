<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

SPDX-License-Identifier: Apache-2.0
-->

# Encoding pipeline-stage adjacency into MGD

Brainstorm / design notes for [tt-metal#52016](https://github.com/tenstorrent/tt-metal/issues/52016).

**Status:** design. Nothing below is implemented yet except Tier D (2.4), which already
runs in the pipeline builder and is not changed. Rewritten after the SAT joint-placement
rework of the topology mapper landed; the Tier C plan (2.3, 4, 5) follows that code.

---

## 0. Problem recap

The pipeline the mapper has to place is a graph of stages. GPT-OSS single galaxy
(4 processes, double torus) is a ring of embedding, windowed decoder, global decoder and
LM head, each on a 4x2, closing from the last stage back to the first:

```python
nodes = {
    "s0": Node(shape=SUBMESH_SHAPE_4X2, name="Embedding", factory=s_embed),
    "s1": Node(shape=SUBMESH_SHAPE_4X2, name="Windowed",  factory=...),
    "s2": Node(shape=SUBMESH_SHAPE_4X2, name="Global",    factory=...),
    "s3": Node(shape=SUBMESH_SHAPE_4X2, name="LMHead",    factory=s_lm_head),
}
edges = [
    Edge("s0", "s1"),
    Edge("s1", "s2"),
    Edge("s2", "s3"),
    Edge("s3", "s0", is_loopback=True),
]
```

That dict is the shape of the graph, not how blaze authors it. Most models never write
it. `build_pipeline` in `blaze/stages/stage.py` (tt-blaze) builds the same ring by
iterating stage harnesses, which already carry `mesh_shape()` and the other stage
info. A few older models still write the graph in `entry.py`; those should move onto
the harness loop. See 1.2 and 2.4.

Two things make the real graphs harder than this ring:

- **Joint stages.** A stage is a rectangle of whole host slices, so it can be wider than one
  rank's slice and be co-owned by several ranks (see 6).
- **Fork graphs.** The DFlash / DSpark router graphs fan out from one router stage into
  several branches, each returning to the router over fabric or over host MPI.

![Pipeline graph](https://raw.githubusercontent.com/tenstorrent/tt-metal/ridvan/mgd-ticket-52016-diagrams/docs/scratch/ticket-52016-diagrams/1-pipeline-graph.png)

### The Problem

The MGD only partly describes this. In the multi-mesh pipeline MGDs (the blaze
`tools/pipeline_sweep` generator, `subtorus_4x4_pipeline_8stage`), each mesh hosts a stage
group and the ring is written as `connections` between meshes, which the SAT seating
already enforces. What the MGD never says is which stages exist, how big each one is, where
a stage boundary falls inside a mesh, or what a fork looks like. A single-mesh MGD says
nothing at all - four anonymous 4x2 slices:

```textproto
mesh_descriptors {
  name: "M0"
  arch: BLACKHOLE
  device_topology { dims: [ 8, 4 ] }
  host_topology   { dims: [ 2, 2 ] }
  channels { count: 2 policy: RELAXED }
}

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
```

![MGD submesh view](https://raw.githubusercontent.com/tenstorrent/tt-metal/ridvan/mgd-ticket-52016-diagrams/docs/scratch/ticket-52016-diagrams/2-mgd-submesh-view.png)

The mapper commits without knowing the pipeline. The builder only carves and resolves the
graph afterwards and fails with `"no valid submesh assignment found"` or a `CarvePlanError`.
Today the recourse is to enumerate every mapping offline (`generate_rank_bindings
--all-solutions`) and launch each one on hardware (`tools/scaleout/sweep_rank_binding_solutions.py`)
until one works.

![Current flow](https://raw.githubusercontent.com/tenstorrent/tt-metal/ridvan/mgd-ticket-52016-diagrams/docs/scratch/ticket-52016-diagrams/3-current-flow.png)

---

Intended end state: the mapper checks pipeline feasibility while it can still steer, using
the pipeline builder's own placement code.

![Proposed flow](https://raw.githubusercontent.com/tenstorrent/tt-metal/ridvan/mgd-ticket-52016-diagrams/docs/scratch/ticket-52016-diagrams/4-proposed-flow.png)

### Big-picture steps

1. **MGD pipeline descriptor format** - section 1.
2. **Verification tiers** - section 2. Tiers A and B run on the MGD alone; Tier C runs inside
   the topology mapper; Tier D is the existing pipeline builder and is not changed.
3. **Shared code** - section 3. Reuse the C++ graph-layout resolver, port the carve planner to
   C++, and share the inter-mesh binding rule with the control plane.
4. **Where it runs in the topology mapper** - section 4, including the rank-binding output.
5. **SAT encoding** - section 5. Plan 1 (constraints) is primary; Plan 2 (check and learn) is
   the backstop.
6. **Stage-to-rank mapping** - section 6.

---

## 1. MGD pipeline descriptor format

### 1.1 Scope

This adds **one new top-level section** to MGD, `pipeline_descriptors`, and changes
nothing else.

- `MeshDescriptor`, `GraphDescriptor`, `NodeRef` and `top_level_instance` are **not**
  modified. In particular `graph_topology: RING` and `connections` keep their meaning.
- The only schema edit is a new field on the top-level `MeshGraphDescriptor` container,
  alongside `mesh_descriptors`, `graph_descriptors`, `switch_descriptors`,
  `top_level_instance` and `pinnings`.

`connections` are not overloaded, but the pipeline must agree with them: **every pipeline
edge that crosses between two meshes must ride a declared connection between those meshes,
and one that the control plane actually binds to physical links** (2.2, 2.3). Connections
the pipeline does not use are allowed.

**Purpose:** let the topology mapper answer *"can this pipeline be built on the mapping I am
about to commit?"* during its solve. The plan is for MGD to become the source of truth for
that topology. The builder cannot load a graph from MGD yet, and most models do not keep a
hand-written copy either: `build_pipeline` derives the graph from stage harnesses (1.2,
2.4). Host-side fields the mapper never uses stay out of the descriptor.

> **Note:** the format does **not** need to mirror the argument list of `resolve_graph_layout`.
> It only has to carry what the feasibility check needs, and may be a superset of what the
> builder checks.

### 1.2 The pipeline descriptor

The descriptor carries how many stages there are, how big each one is, and how they
connect.

```textproto
pipeline_descriptors {
  name: "gpt_oss_120b_single_galaxy"
  stages { id: "s0" label: "Embedding" shape { dims: [ 4, 2 ] } }
  stages { id: "s1" label: "Windowed"  shape { dims: [ 4, 2 ] } }
  stages { id: "s2" label: "Global"    shape { dims: [ 4, 2 ] } }
  stages { id: "s3" label: "LMHead"    shape { dims: [ 4, 2 ] } }
  edges { src: "s0" dst: "s1" }
  edges { src: "s1" dst: "s2" }
  edges { src: "s2" dst: "s3" }
  edges { src: "s3" dst: "s0" }
}
```

**No stage names a mesh.** A stage may bind to any mesh it is *able* to bind to: one whose
host slice divides the stage shape into a whole-slice rectangle that fits the mesh. The
mapper chooses (2.2).

#### How blaze builds the same graph

The textproto above is what the mapper should see. The builder does not read it, and for
most models nobody writes the nodes and edges by hand.

`get_pipeline_harnesses` (per model; GPT-OSS is
`blaze/models/gpt_oss_120b/gpt_oss_120b_blaze_config.py`) returns one harness per stage.
Each harness implements `mesh_shape()` and `num_ranks()`. `build_pipeline` in
`blaze/stages/stage.py` walks that list into a ring:

```python
nodes = {f"s{i}": _make_node(i, harness) for i, harness in enumerate(harnesses)}

num_stages = len(harnesses)
edges = [Edge(f"s{i}", f"s{i + 1}") for i in range(num_stages - 1)]
edges.append(Edge(f"s{num_stages - 1}", "s0", is_loopback=True))

graph = PipelineGraph(nodes=nodes, edges=edges)
```

`_make_node` sets `shape=harness.mesh_shape()`. The ring is sized in ranks, not
harnesses: a joint stage counts `num_ranks()` ranks, and `build_pipeline` pads with
forward-passthrough stages until every rank carries one. The rest of this function,
including `build_topology`, is Tier D and does not change (2.4).

Older models still declare the graph in `entry.py`: dspark and dflash
(`ring_graph_from_stages` in `blaze/models/pipeline_graph_helpers.py`) and the router forks
(`build_gpt_oss_dflash_router_pipeline_graph`, `build_kimi_dflash_router_pipeline_graph`,
`build_kimi_k3_dspark_router_pipeline_graph`). They should move onto the harness loop; that
is a blaze cleanup, not an MGD change.

**TODO:** the pipeline builder has no path that loads stages and edges from MGD. Once it
does, read `pipeline_descriptors` as the source of truth for the topology the mapper and the
builder share. Until then the harness loop and the leftover `entry.py` graphs stay.

Two fields the builder sets stay out of MGD, since neither changes the shape being mapped:

- **`factory`** - host-side construction (`GenericStage` over the harness list).
- **`is_loopback`** - a builder conversion mark for which edge closes the ring. The ring is
  already the `s3 -> s0` edge (2.1).

#### A heterogeneous example

The llama_8b_pod graph (`_build_heterogeneous_two_mesh_graph` in
`tests/pipeline_builder/test_pipeline_builder_infra.py`) is 40 stages: 8 of `4x2` on M0 plus
32 of `1x2` on M1, chained with the ring closing back to `s0`. Abbreviated:

```textproto
pipeline_descriptors {
  name: "llama_8b_pod"
  stages { id: "s0"  label: "Embedding" shape { dims: [ 4, 2 ] } }
  stages { id: "s1"  label: "LMHead"    shape { dims: [ 4, 2 ] } }
  stages { id: "s2"  label: "Token4x2"  shape { dims: [ 4, 2 ] } }
  # ... s3 .. s7 likewise 4x2 ...
  stages { id: "s8"  label: "Token1x2"  shape { dims: [ 1, 2 ] } }
  # ... s9 .. s39 likewise 1x2 ...
  edges { src: "s0" dst: "s1" }
  # ... linear chain ...
  edges { src: "s39" dst: "s0" }
}
```

Shapes vary per stage, so `shape` lives on the stage. Nothing says which mesh the `4x2`
stages land on; the binding set (2.2) decides.

#### A joint-stage example

A `4x4` LM head over two `4x2` host slices is one stage co-owned by two ranks:

```textproto
stages { id: "s7" label: "LMHead" shape { dims: [ 4, 4 ] } }   # 2 slices of 4x2, 2 ranks
```

#### A fork example

A router with a fabric-return target branch and a host-MPI-return draft branch, mirroring
`build_gpt_oss_dflash_router_pipeline_graph`:

```textproto
pipeline_descriptors {
  name: "gpt_oss_dflash_router"
  stages { id: "router"  shape { dims: [ 4, 2 ] } }
  stages { id: "target0" shape { dims: [ 4, 2 ] } }
  # ... target1 .. target37 ...
  stages { id: "draft0"  shape { dims: [ 4, 2 ] } }
  # ... draft1 .. draft10 ...
  edges { src: "router"   dst: "target0" }
  # ... target chain ...
  edges { src: "target37" dst: "router" }
  edges { src: "router"   dst: "draft0" }
  # ... draft chain ...
  edges { src: "draft10"  dst: "router" host_return: true }   # returns over host MPI
}
```

### 1.3 Schema

The proto shape only. **All validation lives in section 2.**

```proto
message PipelineDescriptor {
  string name = 1;                    // non-empty, unique across descriptors
  repeated PipelineStage stages = 2;  // >= 1
  repeated PipelineEdge edges = 3;
}

message PipelineStage {
  string id = 1;             // non-empty, unique, referenced by edges
  string label = 2;          // human-facing, unconstrained, may repeat
  TorusTopology shape = 3;   // required, rank 2, all dims > 0
}

message PipelineEdge {
  string src = 1;            // resolves to a stage id
  string dst = 2;            // resolves to a stage id
  bool host_return = 3;      // returns over host MPI; needs no fabric link
}
```

`shape` is rank 2 because the resolver models a chip as `(row, col)`. `host_return` mirrors
the builder's `Edge.host_mpi_return`: the resolver drops those edges, and the carve planner
places that branch as an open path. Edges carry no loopback marking and no channel count
(2.1).

None of the comments above are enforced by protobuf. They are obligations on
`validate_pipeline_descriptors`.

---

## 2. Verification tiers

> **Note - builder rules come from builder code, not from restated rules.**
>
> Several checks below are pipeline builder rules: stage 0 must be one host slice, a stage is
> a rectangle of whole slices on one mesh, a fork router is one slice with enough fabric
> neighbours. Earlier drafts kept these out of MGD validation so they would not go stale.
> This design instead calls the builder's own code for them - the C++ resolver as-is and a
> C++ port of the carve planner (section 3) - so the rule lives in one place per language,
> and parity tests keep the two copies aligned.

### 2.1 Tier A - descriptor only

**When it runs:** MGD parse, before anything else.

- **Control-plane stage:** `MeshGraph` construction in `ControlPlane::init_control_plane`.
- **Available:** the proto only.
- **Site:** a new `validate_pipeline_descriptors`, next to the existing validators.
- **On failure:** load error.

```cpp
// mesh_graph_descriptor.cpp, MeshGraphDescriptor validation
validate_mesh_topology(proto, all_errors);
validate_express_connections(proto, all_errors);
validate_switch_descriptors(proto, all_errors);
validate_graph_descriptors(proto, all_errors);
validate_graph_topology_and_connections(proto, all_errors);
validate_pinnings(proto, all_errors);
validate_pipeline_descriptors(proto, all_errors);   // <-- Tier A
```

**Field-level obligations** are the schema comments in 1.3. Exactly one descriptor is
active, so capacity (2.2) is evaluated for it alone. Every stage is passed to the resolver as
an explicit node, so a stage no edge mentions is kept, not silently dropped.

**Graph structure:**

- **No self-edges.** `edges { src: "s0" dst: "s0" }` is rejected. No submesh is adjacent to
  itself, and the resolver would otherwise report a misleading cycle error.
- **Repeating an edge declares channels.** Edges are directional; `s0 -> s1` twice is two
  channels on that hop, and `s0 -> s1` plus `s1 -> s0` is a two-stage ring. Multiplicity is
  repetition, so edges need no `channels` field. The builder reads one link per edge today;
  that is an implementation limit, not a schema one.
- **Cycles are allowed.** Conversion marks one edge per cycle as the loopback.
- **Linear or fork.** Either every stage has out-degree <= 1 (chain or ring), or the graph is
  a fork: exactly one in-degree-0 router, each branch a chain, each branch ending in an edge
  back to the router (fabric or `host_return`). This is the C++ port of
  `fork_structure` (3.2).
- **`host_return` only on a branch's return edge** to the router.

### 2.2 Tier B - descriptor plus the declared fabric

**When it runs:** after the mesh graph is built, before any hardware is looked at.

- **Control-plane stage:** `MeshGraph` construction, after `initialize_from_mgd` resolved
  `top_level_instance` and slice shapes.
- **Available:** mesh instances, slice shapes, intra-mesh topology (`device_topology` dims and
  dim types), declared `connections` with channel counts. **Not** the PSD.
- **On failure:** reject before the mapper is constructed.

These reject pipelines that are impossible on *any* mapping of this MGD.

#### Which meshes a stage can bind to

`host_topology` partitions `device_topology`; the slice is the elementwise quotient. A stage
can bind to a mesh when its shape is a whole-slice rectangle of that mesh's slice and fits
the mesh:

```
M0 device 8x4, host_topology 2x2  ->  slice 4x2, 4 slices

  stage 4x2   = 1 slice   (one rank)          can bind to M0
  stage 4x4   = 2 slices  (joint, two ranks)  can bind to M0
  stage 2x2   = 1/2 slice                     cannot - a rank's slice cannot split
```

- The binding set of every stage must be non-empty.
- **Stage 0 binds to a single slice.** It owns host-to-device, the loopback entry and
  device-to-host, so one rank must drive it.
- **A fork router binds to a single slice.**
- A stage never spans two meshes.

#### Capacity

Counted in host slices, not stages:

- **Total:** the slices the pipeline needs (each stage's slice count, summed) must not exceed
  the slices the MGD declares. Unused slices become idle filler slots. Blaze then pads the
  ring with passthrough stages until every rank carries one, so in practice the two are equal.
- **Per mesh class:** stages that can only bind to one mesh class need enough slices of that
  class, counted per mesh instance (each contributes `prod(host_topology.dims)` slices).

```
llama_8b_pod
  instance       slices available   slices needed
  M0 mesh_id 0    8 x [ 4, 2 ]       8 x [ 4, 2 ]      OK
  M1 mesh_id 1   32 x [ 1, 2 ]      32 x [ 1, 2 ]      OK

5 stages of [ 4, 2 ] against M0's 4 slices   ->   REJECT
```

#### Projection onto meshes - pipeline edges must ride declared connections

Search for an assignment of stages to meshes (each from its binding set, respecting
capacity) such that:

- **Every crossing rides a declared connection.** For each non-`host_return` pipeline edge
  whose stages land on different meshes A and B, the MGD declares a connection between A and
  B.
- **Crossings fit the channels.** The number of pipeline edges crossing A–B is at most that
  connection's channel count.

Extra connections the pipeline does not use are fine. Tier C tightens "declared" to
"actually bound by the control plane" (2.3).

Rotations and reflections of a uniform ring give equivalent projections and collapse to one.
The surviving projections, usually one, are handed to Tier C. Zero projections is a
reject.

```
subtorus_4x4_pipeline_8stage: 8 meshes, connections 0-1-2-...-7-0, 8 stages of 4x4

  s_i on mesh i, crossings s_i -> s_{i+1} ride connection (i, i+1)   OK

same MGD, connections 0-1-2-...-7 (no 7-0)                           REJECT
  the loopback crossing has no declared connection
```

#### Each mesh's segment can be carved

For each mesh in the projection, run the C++ carve planner (3.2) on that mesh's logical
slices. Slice adjacency comes from `device_topology` dims and RING wraps. Crossing endpoints
are left open, because which slice a crossing touches is not known until seating. This checks
that the stages assigned to the mesh can be cut into whole-slice rectangles along a
path or cycle, including joint stages and, for a fork, the router's degree:
2 neighbours per fabric-return branch plus 1 per host-return branch.

### 2.3 Tier C - inside the topology mapper

The only tier that sees physical adjacency.

**When it runs:** during the SAT joint-placement solve in `MultiMeshSolutionEnumerator`,
before any mapping is committed.

- **Control-plane stage:** `TopologyMapper` construction, after the PSD exists, before the
  mapping is loaded.
- **Available:** the PSD, the candidate seats (each fixes a mesh's footprint and its chip
  layout), and, after intra-mesh completion, the full logical-to-physical mapping.
- **No control-plane dependency.** The resolver already takes injected links (3.1), and the
  carve planner port and the inter-mesh binding helper are pure (3.2, 3.3).

It works in two parts, detailed in sections 4 and 5:

- **Plan 1 - constraints in the SAT solve (primary).** Before the master solve, prune seats
  and seat pairs that cannot carry the pipeline (5.1).
- **Exact check per candidate (backstop).** After intra-mesh completion, run the carve planner
  and the resolver on the candidate. On failure, feed the result back to SAT (5.2).

#### Crossings must ride connections the control plane actually binds

After mapping, `ControlPlane::generate_intermesh_connectivity()` binds each requested MGD
connection to physical cables (`AnnotatedIntermeshConnections`), validates them
(`validate_requested_intermesh_connections`), builds the exit maps and loads them into the
routing tables. Only those links carry fabric traffic. The builder's runtime check matches:
`discover_slice_adjacency` and the resolver only count links confirmed by
`get_forwarding_direction`.

So for Tier C a crossing between A and B may only use links that the control plane *will*
bind for the A–B connection on this mapping, not any PSD cable between the two footprints.
That binding runs after mapping, so Tier C must compute the same set with the same rule
(3.3). Intra-mesh links follow the same idea: only links the intra-mesh topology routes
over count.

#### Why the location of a connection matters, not just the count

Take a 4-stage ring split two stages per mesh, which puts **two edges across the cut**:

```
pipeline   s0 -> s1 -> s2 -> s3 -> s0        placement   s0 s1 on M0, s2 s3 on M1
cut edges  s1 -> s2   and   s3 -> s0


CASE A   2 links, 2 distinct slice pairs                               MAP OK

    M0                          M1
  +------+------+  A0<->B0    +------+------+
  |  s0  |  s1  |=============|  s2  |  s3  |
  +------+------+             +------+------+
      ^                                   |
      +===================================+
                   A1<->B1


CASE B   2 links, both on the SAME slice pair                          REJECT

    M0                          M1
  +------+------+  A0<->B0    +------+------+
  |  s0  |  s1  |=============|  s2  |  s3  |
  +------+------+  A0<->B0    +------+------+

  s3 -> s0 needs a link touching s3's slice and s0's slice, and there is none.


CASE C   1 link                                            REJECT as ring
                                                           OK as an open chain
    M0                          M1
  +------+------+  A0<->B0    +------+------+
  |  s0  |  s1  |-------------|  s2  |  s3  |
  +------+------+             +------+------+
```

Tier B can see the count. Only Tier C can see where the bound links land, because that is
decided by the seats the mapper chooses. `mesh4x4_stage4x2` in the blaze sweep (4 stages,
2 meshes) is exactly this shape.

### 2.4 Tier D - inside the pipeline builder

**Already done. Do not change this.**

This is the pipeline builder's own carve and resolution, on the committed mapping. It is the
only tier that exists today, and nothing in this design modifies it. Tiers A–C reject bad
mappings earlier; without them Tier D still fails hard, as it does now.

**When it runs:** during pipeline build, after the mapping has committed.

- `PipelineGraph.__init__` calls `_validate`. `build_topology` carves submeshes (uniform
  tiling, or `carve_planner.plan_carve` when no `submesh_shape` is given), then calls
  `resolve_graph_layout`.
- **Site:** `build_pipeline` in `blaze/stages/stage.py` (tt-blaze), through
  `PipelineGraph.build_topology` in `pipeline_builder/graph.py`.
- **Cannot steer anything.** The mapping is fixed, so a failure is a hard error. Fixing that
  is Tiers A–C.

```python
# blaze/stages/stage.py — build_pipeline
def _make_node(i, harness):
    name = f"s{i}"
    return Node(
        shape=harness.mesh_shape(),
        name=name,
        pipeline_core_coords=harness.pipeline_core_coords(),
        factory=lambda d: GenericStage(
            harnesses=harnesses,
            run_persistently=persistent_mode,
            io_socket_descriptor_prefix=io_socket_descriptor_prefix if i == 0 else None,
        ),
    )

nodes = {f"s{i}": _make_node(i, harness) for i, harness in enumerate(harnesses)}

num_stages = len(harnesses)
edges = [Edge(f"s{i}", f"s{i + 1}") for i in range(num_stages - 1)]
edges.append(Edge(f"s{num_stages - 1}", "s0", is_loopback=True))

graph = PipelineGraph(nodes=nodes, edges=edges)
```

```python
# pipeline_builder/graph.py — build_topology, after carving:
result = ttnn._ttnn.multi_device.experimental.resolve_graph_layout(
    nodes=node_names,
    edges=edge_tuples,
    submesh_chips=global_submesh_chips,   # carved by _collect_submeshes
    node_chip_counts=node_chip_counts,
    node_pipeline_core_counts=node_pipeline_core_counts,
)
```

---

## 3. Shared code

Tier C runs the pipeline builder's own logic wherever it can, so mapper and runtime agree by
construction rather than by matching rules.

### 3.1 Reuse the C++ graph-layout resolver as-is

`tt_metal/fabric/pipeline_builder.cpp` already exposes the resolver without the control
plane (since [#55313](https://github.com/tenstorrent/tt-metal/pull/55313)):

- `detail::resolve_graph_layout_with_connections(nodes, edges, submesh_chips,
  node_chip_counts, node_pipeline_core_counts, pipeline_core_count, direct_links)` - takes
  injected `DirectLinks`; `nullptr` falls back to the control plane.
- `detail::resolve_pipeline_placement(...)` - the exact stage/link/core-slot search.

Tier C calls `resolve_graph_layout_with_connections` with `DirectLinks` built from the
candidate mapping and the bound links (3.3). It is the same function Tier D calls through
`resolve_graph_layout`, so the two cannot drift. It handles fork graphs today: edges form a
DAG after dropping loopbacks, host-return edges are left out, and per-node core capacity
covers the router's endpoint budget.

The one change: the resolver throws with a message string on failure. Tier C needs to know
*which* stage or edge failed to attribute the failure (5.2). Add a structured failure
(failing stage, failing edge, or capacity conflict) alongside the existing message.

### 3.2 Port the carve planner to C++ (a second copy)

The carve planner (`pipeline_builder/carve_planner.py`, tt-blaze) decides which host slices
make up each stage. Its planning core is pure: no ttnn, no collectives; slices and slice
adjacency come in as integers. Port that core to C++ in `tt_metal/fabric/pipeline_builder.cpp`,
next to the resolver:

| Python (tt-blaze, unchanged) | C++ port (tt-metal) |
|---|---|
| `HostSlice`, `CarveSlot` | `detail::HostSlice`, `detail::CarveSlot` |
| `plan_ring_carve` | `detail::plan_ring_carve` |
| `plan_fork_carve` | `detail::plan_fork_carve` |
| `fork_structure` | `detail::fork_structure` |
| `validate_carve_plan` | `detail::validate_carve_plan` |
| `CarvePlanError` (phase + reason) | structured result with the same phase names |

- **Two copies.** The Python planner stays in tt-blaze and keeps running at runtime. Nothing
  in blaze moves or changes.
- **Same answers.** The plan must be a pure, deterministic function of slices and adjacency in
  both languages: geometry-keyed slices, sorted iteration, the same canonical walk, the same
  search budgets. Then the mapper's plan and the runtime plan match for the same inputs.
- **Parity tests.** A shared set of golden cases (ring, heterogeneous, joint stage, fork with
  fabric and host returns, the known failure phases) fed to both copies, comparing the emitted
  plan and its crc32 digest (the same digest `plan_carve` exchanges across ranks).
- **Discovery stays out.** `discover_host_slices` and `discover_slice_adjacency` are not
  ported. Tier C derives slices from the MGD plus the candidate mapping, and adjacency from
  the bound links (3.3).

### 3.3 Share the inter-mesh binding rule with the control plane

`ControlPlane::generate_intermesh_connectivity()` decides which physical cables carry each
MGD connection, after mapping. Tier C needs that same set during mapping. Factor the pairing
rule out of the control plane into a pure helper:

```cpp
// inputs: PSD, the candidate logical-to-physical mapping, the MGD's requested connections
// output: the AnnotatedIntermeshConnections the control plane would bind
AnnotatedIntermeshConnections bind_intermesh_connections(...);
```

The control plane then calls the helper, so there is one rule. Its invariants carry over,
for example that a source chip cables to at most one destination chip on a given
neighbouring mesh (from `gather_intermesh_cables_for_exit_nodes`).

## 4. Where it runs in the topology mapper

The topology mapper now places meshes with SAT joint seating. A *seat* is a physical
footprint for one mesh that also fixes its chip-by-chip layout. The loop in
`MultiMeshSolutionEnumerator::next()` (`topology_mapper_utils.cpp`) is:

```
seating = placement_session_->next()                 SAT joint seating
physical = build_hierarchical_from_flat_graph(...)
full = complete_intra_mesh_for_placement(...)        identity intra-mesh
if fail: add_forbidden_constraint(failed seat); loop
emit
```

With a pipeline descriptor present, it becomes:

```
[before the first solve]
  Tier B projections from MeshGraph
  Plan 1 pruning of seats and seat pairs (5.1)

loop:
  seating = placement_session_->next()
  full = complete_intra_mesh_for_placement(...)
  if fail: add_forbidden_constraint(failed seat); loop          (unchanged)

  links  = bind_intermesh_connections(psd, full, mgd)  (3.3) + routed intra-mesh links
  slices = host slices of every mesh, through full.fabric_node_to_asic
  plan   = detail::plan_carve(slices, slice adjacency, pipeline)   (3.2)
  layout = detail::resolve_graph_layout_with_connections(slots, links)  (3.1)
  if fail: feedback to SAT (5.2); loop

  emit full + plan + layout
```

#### Step by step

Worked example: `mesh4x4_stage4x2` from the blaze sweep. Two 4x4 meshes M0 and M1, each
`host_topology [1, 2]` (two 4x2 slices, one rank each), a 4-stage ring of 4x2 stages, and
two declared M0–M1 connections.

**Before the first solve.**

- *Tier B projection* (2.2): `s0, s1 → M0` and `s2, s3 → M1`; the mirror image collapses.
  Crossings `s1 → s2` and `s3 → s0` give `k(M0, M1) = 2`. Every slice may hold a boundary
  stage.
- *Plan 1 pruning* (5.1): a seat fixes each chip's ASIC, so splitting by `host_topology`
  (`MeshGraph::get_coord_range(mesh, host_rank)`) gives each slice's ASICs per seat. Drop
  M0–M1 seat pairs whose usable links (distinct slices on both sides) are fewer than 2, and
  seats with no outgoing link on an allowed boundary slice.

**Per seating.**

1. **Seat.** SAT returns one seat per mesh, using only seat pairs that survived pruning.
2. **Intra-mesh.** Fills `full.fabric_node_to_asic`. Unchanged; a failure forbids the seat.
3. **Links.** Only links the control plane will route over count:
   - *between meshes:* `bind_intermesh_connections` (3.3) returns the
     `AnnotatedIntermeshConnections` the control plane would build, honouring channel counts
     and the one-destination-chip-per-source-chip rule. The PSD holds every host's cables, so
     this needs no MPI exchange;
   - *inside a mesh:* the mapped pairs of logical neighbours in `device_topology`, RING wraps
     included. Extra cables outside the topology are ignored.
4. **Slices.** For each mesh and host rank, `get_coord_range` gives a
   `HostSlice(mesh_id, row0, col0, nrows, ncols)`. Two slices are adjacent when a step-3 link
   joins them, matching blaze's `discover_slice_adjacency`.
5. **Carve.** `plan_ring_carve` (or `plan_fork_carve`) finds a cycle of slices and cuts it
   into stage rectangles, joint stages included. Here: `s0 → M0@(0,0)`, `s1 → M0@(0,2)`,
   `s2 → M1@(0,2)`, `s3 → M1@(0,0)`. Case B (both crossings on one slice pair) fails here
   with `no-cycle-of-length-4`.
6. **Layout.** Feed the resolver what blaze feeds it:
   - stages as nodes; edges with one loopback per cycle; `host_return` edges dropped;
   - each slot's chips as `submesh_chips`, with `node_chip_counts = rows * cols`;
   - blaze's default core counts (router `2 + 2 * branches`, host-return tail 2);
   - the step-3 links between slots as `direct_links`.

   It catches per-chip core capacity, shared entry and exit chips, and the router's endpoint
   budget.
7. **Feedback.** On failure, tell SAT (5.2): forbid one seat, block one seat pair, or block
   the whole seating.
8. **Emit.** The mapping, carve plan and layout. Each rank's mesh and host rank give its
   slice, then its slot, then its `TT_PIPELINE_STAGE` (4.1).

**Gap.** Step 6 uses blaze's default core counts. Harness `pipeline_core_coords()` overrides
are not in the MGD, so Tier C could accept a layout runtime rejects (open question 6).

- **No descriptor, no change.** Without `pipeline_descriptors`, the loop is exactly today's.
- **Failure reporting.** `failure_stage = "pipeline_carve"` or `"pipeline_layout"`, carrying the
  carve phase or resolver reason. When the session is exhausted it surfaces through the
  existing *"Graph specified in MGD could not fit in the discovered physical topology"*.
- **Bonus.** `generate_rank_bindings --all-solutions` only emits mappings the pipeline can be
  built on, so `sweep_rank_binding_solutions.py` stops launching dead ones.

### 4.1 Writing the stage plan into rank bindings (written, not used)

When a mapping passes Tier C, `generate_rank_bindings` writes the stage each rank drives as
an `env_overrides` entry in `rank_bindings.yaml` (`RankBindingConfig::env_overrides`).
Co-owners of a joint stage all get the same stage id:

```yaml
    env_overrides:
      TT_PIPELINE_STAGE: "s7"                 # stage this rank drives, or "idle"
```

**Not consumed yet.** Nothing in tt-metal or blaze reads it. Blaze keeps planning with its
own Python carve planner. It records which stage the mapper validated for each rank, and
whether anything ever uses it is a separate decision.

---

## 5. SAT encoding

The SAT targets are logical meshes; the globals are seats. Each mesh-to-mesh connection must
land on a seat pair with links between them. Under STRICT the channel count is hard; under
RELAXED only adjacency is hard and the count is a preference. Because a seat fixes the chip
layout, the ASICs under every host slice are known per seat before solving.

### 5.1 Plan 1 - pipeline constraints in the solve (primary)

Plan 1 encodes necessary conditions, computed from the Tier B projection, before the master
solve. If Tier B returned several non-equivalent projections, run the solve once per
projection.

From the projection, for each mesh: its *segment* (stages, intra-mesh edges, which stages
have a crossing toward which neighbour mesh, and which slices may hold those boundary
stages per the logical carve in 2.2). For each mesh pair A–B: the crossing count `k_AB`.

**Seat-pair constraint - usable links, not raw links.** In `build_sat_placement_seat_graph`,
for a mesh pair the pipeline crosses, replace the raw link count between two seats with the
*usable* count: the largest set of bound links (3.3) whose endpoints sit on distinct
allowed boundary slices on both sides (a bipartite matching). Drop the seat-graph edge when
that count is below `k_AB`.

- This rejects Case B.
- Dropping the edge makes it hard under RELAXED too, not just STRICT.
- Mesh pairs the pipeline does not cross keep today's link count.

**Single-seat constraint.** In `build_sat_placement_constraints`, remove a seat from a mesh's
allowed set when:

- no allowed boundary slice in it has an outgoing bound link toward a required neighbour
  mesh; or
- for the router's mesh in a fork: no single slice reaches the router's required degree
  (2 per fabric-return branch plus 1 per host-return branch).

**Caveats.**

- **Shared seats.** The seat graph merges identical seats across mesh pairs. If two mesh pairs
  that share a seat pair need different `k`, edge-dropping is unsound for the looser pair.
  Then fall back to pairwise blocking (5.2) for those pairs.
- **What Plan 1 cannot encode.** Joint-stage alignment along the whole ring, per-chip
  pipeline-core capacity, a stage whose entry and exit land on the same chip, stage-0
  host-to-device and device-to-host slots, and the router's endpoint budget. These depend on
  the whole pipeline at once. The exact check (5.2) catches them.

### 5.2 Plan 2 - check each candidate and teach the solver (backstop)

After intra-mesh completion, run the carve planner and the resolver on the candidate
(section 4). On failure, feed back the strongest constraint the failure supports:

| Failure attributed to | Feedback |
|---|---|
| one mesh's segment cannot be carved in its seat | `add_forbidden_constraint(placed)` - forbid that seat for that mesh |
| one mesh pair's crossings | partial `exclude_mapping({A: seat_a, B: seat_b})` - block that pair |
| not attributable | `exclude_mapping(seating)` - block this combination |

- `SatPlacementEnumerationSession::exclude_mapping` already blocks a combination without
  forbidding any single footprint. The underlying `TopologyMappingEnumerationSession` takes a
  partial map. **To confirm:** that blocking a partial map adds a clause over only the given
  meshes.
- Attribution needs the structured failures from 3.1 and 3.2.
- Cap the number of pipeline rejections per run and report the last carve or resolver reason.

---

## 6. Stage-to-rank mapping

### 6.1 How it is set today

1. **Rank to host slice - the topology mapper.** `host_topology` splits each mesh into equal
   host slices, and the rank bindings say which rank drives which slice. A host slice is the
   smallest unit a stage can be built from.
2. **Stage to slices - the carve planner, at runtime.** `plan_carve` in
   `pipeline_builder/carve_planner.py` runs collectively on every rank. It gathers all host
   slices, finds which slice pairs the fabric links (routing-confirmed), searches for a cycle
   of the right length (or one cycle per fabric-return branch and one open path per
   host-return branch for a fork), and cuts it into stage rectangles. Ranks exchange a crc32
   digest of the plan and fail if they disagree.
3. **Co-ownership.** A stage covering several slices is co-owned by the ranks driving them.
   `_collect_submeshes` merges them into one node, and `allgather_end_owners` picks which
   co-owner drives the entry chip and which drives the exit chip. Slices the pipeline does
   not reach become filler slots and their ranks idle.
4. **Harness sizing.** `num_ranks()` sizes the ring in ranks; blaze pads with passthrough
   stages.

### 6.2 In this design

- The mapper only accepts a mapping once the C++ carve planner has produced a stage-to-slot
  plan for it and the resolver has laid it out (section 4). That plan is the stage-to-rank
  mapping: each stage, its slices, hence its ranks, and its entry and exit chips.
- Each rank's stage is recorded in `rank_bindings.yaml` as `TT_PIPELINE_STAGE` (4.1) but not
  consumed.
- At runtime blaze replans with its Python planner from the same slices and adjacency. The
  parity tests (3.2) are what make that plan the one the mapper checked.

---

## 7. Open questions

1. **Adjacency parity.** Tier C builds slice adjacency from bound inter-mesh links and routed
   intra-mesh links; runtime uses `get_forwarding_direction`-confirmed neighbours. Confirm the
   two agree, including links with no active routing planes.
2. **Projection count.** For heterogeneous or fork pipelines, how many non-equivalent
   projections survive Tier B in practice, and is a per-projection outer loop affordable?
3. **Shared seats with different `k`** (5.1): handle generally, or reject until a real
   descriptor needs it.
4. **Partial nogoods** (5.2): confirm `block` on a partial map, or add a dedicated API.
5. **Planned key name** for the rank-binding `env_overrides` entry (4.1).
6. **Per-stage core counts.** Harnesses can override pipeline-core pools, which the MGD does
   not carry. Add an optional core count to `PipelineStage`, or accept the gap (4, step 6).
