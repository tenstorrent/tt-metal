<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

SPDX-License-Identifier: Apache-2.0
-->

# Encoding pipeline-stage adjacency into MGD

Brainstorm / design notes for [tt-metal#52016](https://github.com/tenstorrent/tt-metal/issues/52016).

Worked examples that follow two pipelines through every step, from MGD parse through SAT
placement, the carve planner and the resolver to blaze's runtime build, are in section 8.

A visual version with diagrams, the same worked examples and the full per-PR implementation
plan is in [mgd-pipeline-graph-design.html](mgd-pipeline-graph-design.html). Download it and
open it in a browser; GitHub shows HTML files as source. Its source is the Cursor canvas
[mgd-pipeline-graph-design.canvas.tsx](mgd-pipeline-graph-design.canvas.tsx), which opens
as a live canvas in Cursor.

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
  edges { src: "draft10"  dst: "router" transport: HOST }   # returns over host MPI
}
```

### 1.3 Schema

The proto shape only. **All validation lives in section 2.**

```proto
message PipelineDescriptor {
  string name = 1;                    // non-empty, unique across descriptors
  repeated PipelineStage stages = 2;  // >= 1
  repeated PipelineEdge edges = 3;
  PipelinePlacement placement = 4;    // default STRICT
}

enum PipelinePlacement {
  STRICT = 0;     // every non-HOST edge must ride fabric
  FLEXIBLE = 1;   // the builder may move edges onto host MPI where the fabric ends
}

enum EdgeTransport {
  FABRIC = 0;
  HOST = 1;       // D2H -> MPI -> H2D; needs no fabric link
}

message PipelineStage {
  string id = 1;             // non-empty, unique, referenced by edges
  string label = 2;          // human-facing, unconstrained, may repeat
  TorusTopology shape = 3;   // required, rank 2, all dims > 0
}

message PipelineEdge {
  string src = 1;            // resolves to a stage id
  string dst = 2;            // resolves to a stage id
  EdgeTransport transport = 3;  // default FABRIC
}
```

`shape` is rank 2 because the resolver models a chip as `(row, col)`. `transport` mirrors
the builder's `Edge.transport` (`"fabric"` or `"host"`), which replaced `host_mpi_return` in
tt-blaze #4917: the resolver drops HOST edges, the graph splits into fabric chains that are
resolved one at a time, and a fork branch with a HOST return is placed as an open path.
`placement` mirrors `build_topology(placement="strict" | "flexible")` (tt-blaze #4513, CLI
`--pipeline-builder`), which decides how much the mapper may assume the builder can route
over host (2.3, 3.4). Edges carry no loopback marking and no channel count (2.1).

None of the comments above are enforced by protobuf. They are obligations on
`validate_pipeline_descriptors`.

### 1.4 Merging MGDs

The topology mapper never reads a part MGD directly. `MultiMeshSolutionEnumerator::init_from_parts`
(`topology_mapper_utils.cpp`) always calls `MeshGraphDescriptor::merge` on its parts and builds
the SAT problem from the result, so Tier B, Tier C and the `unique_shapes` gate (5.2) all see
the merged MGD.

**How `merge` works today** (`mesh_graph_descriptor.cpp`):

- **One part:** the whole proto is copied, so `pipeline_descriptors` survives.
- **Several parts:** a fresh proto is built. Each part is prefixed (`mgd<i>_`) and its mesh
  ids are remapped to global ids (`prefix_and_remap_mgd_proto`). Then only these are copied
  in: `mesh_descriptors`, `switch_descriptors`, `pinnings`, and the instances and
  connections of each part's top-level FABRIC graph, folded into one graph `G0`.
  **Anything else is dropped without a warning**, so `pipeline_descriptors` would vanish
  and Tier C would silently not run.

**Change.** In the per-part loop, copy `part.pipeline_descriptors()` into the merged proto:

```cpp
for (const auto& pipeline : part.pipeline_descriptors()) {
    *merged->add_pipeline_descriptors() = pipeline;
}
```

- **No remap needed.** Stages bind to meshes by shape (2.2), not by mesh id or descriptor
  name, and stage ids are scoped to their own descriptor. `prefix_and_remap_mgd_proto` does
  not touch the section, and the pipeline name stays as written so logs match the source
  file.
- **The binding scope is the merged fabric.** A stage may bind to any mesh of the merged
  MGD whose slice it fits, not just meshes from the part that declared it. A pipeline that
  spans the parts is the reason to merge (for example, per-galaxy MGDs merged into one
  supercluster).

**Rule for more than one part declaring a pipeline: reject.** `merge` fails if two or more
parts carry `pipeline_descriptors`, and names both parts:

```
MeshGraphDescriptor::merge: pipeline_descriptors declared in descriptor 0 ('p_ring') and
descriptor 2 ('p_ring'); declare the pipeline in exactly one part.
```

- One pipeline is active per run (2.1). Since every pipeline binds across the whole merged
  fabric, two parts' pipelines would compete for the same slices, and nothing says which
  one runtime builds.
- Byte-identical copies (the same file in two parts) are rejected as well. Being exact keeps
  the rule simple; relax it only if a real setup needs it.
- Uniting several parts' pipelines into one graph needs edges between parts, and no part
  can name another part's stages. That is out of scope.

**Validation order.** Tier A runs as each part is parsed, so a bad pipeline is reported
against its own file. The merged MGD re-runs the validators when constructed, which also
covers a pipeline that is valid only against the merged fabric. Tier B and later run on the
merged MGD only.

**Tests.**
- Two parts, a pipeline in one: present after merge, unchanged, and Tier C runs.
- Pipelines in two parts: merge fails with the message above.
- One part: unchanged, as today.
- No pipeline anywhere: the merged proto is byte-identical to today's.

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

**Stage shapes are in logical mesh orientation.** A stage's `shape` is `(rows, cols)` in the
mesh's logical coordinates. Neither blaze's carve planner nor Tier C tries the transpose, so
a 4x2 stage never lands on a 2x4 rectangle (3.2).

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
  back to the router (FABRIC or HOST). This is the C++ port of `fork_structure` (3.2).
- **HOST transport on any edge.** Blaze supports host hops anywhere in a ring or branch
  (host input, host output and host hops on both sides of a stage), not just on a branch's
  return. Under FLEXIBLE placement no edge may be declared HOST up front, because blaze only
  plans host cuts when the graph declares none (`_resolve_topology`).

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

> **Note - no early elimination.** Tiers A and B, and Plan 1 in Tier C, may only reject what
> is impossible under *every* stage-to-mesh split and every chip layout the mapper could
> still choose. None of them picks a split. Which stage lands on which mesh and slice is
> decided once per seating by the exact check (2.3, section 4), the same way the runtime
> carve planner decides it from the links that are actually there.

#### A split onto meshes exists - pipeline edges must ride declared connections

Check that at least one assignment of stages to meshes (a *projection*: each stage from its
binding set, respecting capacity) satisfies:

- **Every crossing rides a declared connection.** For each FABRIC pipeline edge whose stages
  land on different meshes A and B, the MGD declares a connection between A and B.
- **Crossings fit the channels.** The number of FABRIC pipeline edges crossing A–B is at most
  that connection's channel count.

Extra connections the pipeline does not use are fine. Tier C tightens "declared" to
"actually bound by the control plane" (2.3).

**Under FLEXIBLE placement these two rules do not apply.** The builder may turn any crossing
except stage 0's outgoing edge into a host hop, so a missing or too-small connection is not
a reason to reject. Tier B then checks only binding sets and capacity.

Zero projections is a reject. Otherwise Tier B only reports that the pipeline is possible;
it does not hand a chosen projection to Tier C. Rotations and reflections of a uniform ring
may collapse to one for this existence check, but nothing downstream is restricted to the
survivor. Plan 1 (5.1) reads only facts that hold for every surviving projection.

```
subtorus_4x4_pipeline_8stage: 8 meshes, connections 0-1-2-...-7-0, 8 stages of 4x4

  s_i on mesh i, crossings s_i -> s_{i+1} ride connection (i, i+1)   OK

same MGD, connections 0-1-2-...-7 (no 7-0)                           REJECT
  the loopback crossing has no declared connection
```

#### Each mesh's segment can be carved

A projection survives only if, for each mesh in it, the stages it assigns to that mesh can be
carved on the mesh's logical slices. Reject only when no projection survives. This is a
per-segment search with open crossing endpoints, which the runtime carve planner does not
have (it searches whole cycles), so it is new code built from the planner's pieces rather
than a straight port.

Slice adjacency comes from `device_topology` dims and RING wraps. Crossing endpoints are left
open, because which slice a crossing touches is not known until seating. This checks that
the stages assigned to the mesh can be cut into whole-slice rectangles along a path or cycle,
including joint stages and, for a fork, the router's degree: 2 neighbours per fabric-return
branch plus 1 per host-return branch.

### 2.3 Tier C - inside the topology mapper

The only tier that sees physical adjacency.

**Goal: prove a placement exists.** Tier C accepts a mapping when it finds *some* carve and
layout of the pipeline on it. That placement is a witness, not the one blaze will build:
blaze carves again at runtime, may take a different path (planned carve or uniform tiling,
see Tier D), and may land on a different placement. That is fine while nothing consumes the
mapper's placement (4.1). The one rule is the no-early-elimination rule from 2.2: Tier C
rejects a mapping only when **every** carve path it has fails, never because one path does
not apply.

**When it runs:** during the SAT joint-placement solve in `MultiMeshSolutionEnumerator`,
before any mapping is committed.

- **Control-plane stage:** `TopologyMapper` construction, after the PSD exists, before the
  mapping is loaded.
- **Available:** the PSD, the candidate seats, and, after intra-mesh completion, the full
  logical-to-physical mapping. A seat fixes a mesh's footprint. Its chip layout
  (`mesh_pgd_pinnings_`) only enters the intra-mesh solve as *preferred* constraints, so
  which ASICs sit under each host slice is known for sure only after intra-mesh completion,
  unless hard MGD pinnings fix it.
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
over count. Today's binding ignores the pipeline and can bind the wrong cables even when
good ones exist; 5.3 makes it bind the links Tier C's layout uses first.

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

Tier C runs the pipeline builder's own logic wherever it can, so the mapper applies the same
rules as runtime (core capacity, whole-slice rectangles, stage 0 on one slice, router
degree) rather than a restated copy of them. It does not need to produce the same placement
as runtime (2.3).

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
DAG after dropping loopbacks, HOST edges are left out, and per-node core capacity
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
| `plan_ring_placement`, `plan_fork_placement` (`host_edge_planner.py`, FLEXIBLE only) | `detail::plan_ring_placement`, `detail::plan_fork_placement` |
| `HostEdgePlanError`, `ForkShapeUnsupported` | structured result: reject vs fall back to STRICT |

- **Two copies.** The Python planner stays in tt-blaze and keeps running at runtime. Nothing
  in blaze moves or changes.
- **Same verdicts, not necessarily the same plans.** Because Tier C only proves existence
  (2.3), the two copies must agree on *whether* a carve exists for the same slices and
  adjacency, including the failure phase. Matching plans is not required. Keep the C++ copy
  deterministic anyway (geometry-keyed slices, sorted iteration, fixed search budgets) so the
  mapper's output is stable run to run.
- **Parity tests.** A shared set of golden cases (ring, heterogeneous, joint stage, fork with
  fabric and host returns, the known failure phases) fed to both copies, comparing accept or
  reject and the failure phase. Comparing the emitted plan and its crc32 digest is optional
  and only matters if the plan is ever consumed (4.1).
- **Discovery stays out.** `discover_host_slices` and `discover_slice_adjacency` are not
  ported. Tier C derives slices from the MGD plus the candidate mapping, and adjacency from
  the bound links (3.3, 5.3).

Planner behaviour the port must keep, or deliberately change:

- **Blaze commits to its first plan.** `plan_ring_carve` sorts cycles by fewest mesh
  crossings, then by slice keys, and returns the first one the stage shapes align onto.
  `plan_fork_carve` is first-found too. If the resolver then fails on that plan, blaze
  raises; it never tries the next cycle. Tier C instead **tries further cycles until one
  resolves**, because it only proves that a placement exists (2.3). The gap is that Tier C
  may accept a seating where blaze's first cycle fails in the resolver and a later one would
  have worked. Closing it later means either porting blaze's exact order and checking only
  the first cycle, or teaching blaze to retry. The final rerun over the bound links (5.3)
  narrows the gap because it uses the same links runtime sees.
- **Running out of budget is "unknown", not "reject".** The cycle search stops at
  `_MAX_SEARCH_EXPANSIONS = 200_000` with `budget-exhausted`. For Tier C that means
  "accept, and leave the final call to blaze". Rejecting would eliminate placements early on
  large fabrics. Log it so the case is visible.
- **Adjacency is coarse.** Two slices are adjacent if one confirmed link joins them. The
  planner does not count links or channels and does not know which chips they land on, so a
  planner pass is a necessary condition only; the resolver gives the real answer.
- **No rotation.** `_as_rectangle` requires each stage's rectangle to be exactly the declared
  `(rows, cols)`, so a 4x2 stage does not fit a 2x4 run of slices. Tier C does not try
  transposes either (2.1).
- **Filler slots and coverage.** Every slice the ring does not use becomes its own idle
  slot, and `_check_covers` requires the plan to tile every chip of every mesh, including
  meshes with no stage.
- **Leaf pruning.** Slices with fewer than two fabric neighbours are dropped before the cycle
  search. This is sound for rings and is why sparse fabrics report
  `no-cycle-of-length-N`.
- **No digest exchange.** `plan_carve`'s crc32 allgather exists because carving is
  collective per mesh. Tier C plans centrally and does not need it.

### 3.3 Share the inter-mesh binding rule with the control plane

`ControlPlane::generate_intermesh_connectivity()` decides which physical cables carry each
MGD connection, after mapping. Tier C needs that same set during mapping. Factor the gather
and pairing rules (`gather_intermesh_cables_for_exit_nodes`, `pair_logical_intermesh_ports`)
out of the control plane into pure helpers:

```cpp
// inputs: PSD, the candidate logical-to-physical mapping
// output: every cable the pairing could bind (link up, gathered from both sides, one dst chip per src chip)
IntermeshCandidates gather_intermesh_candidates(...);

// inputs: the candidates, the MGD's requested connections, links that must be bound first
// output: the AnnotatedIntermeshConnections the control plane binds
AnnotatedIntermeshConnections bind_intermesh_connections(
    const IntermeshCandidates&, const MeshGraph&, const std::vector<IntermeshLink>& required_links);
```

The control plane then calls the helpers, so there is one rule. Its invariants carry over,
for example that a source chip cables to at most one destination chip on a given
neighbouring mesh. `required_links` is empty without a pipeline descriptor, which keeps
today's pairing exactly; with one, Tier C picks the links, they are bound first, and the
rest of the budget is filled to spread across host slices (5.3). The single-host path
(`generate_intermesh_connections_on_local_host`) moves onto the same two helpers.

### 3.4 Carve paths: planned carve, per-rank tiling and flexible placement, as the pipeline builder does them

Blaze has two ways to turn a mesh into submeshes, and Tier C needs both (2.3). Both end in
the same code, so Tier C shares one path from "slots" to the resolver and the two carve
paths differ only in how they produce slots. On top of them, FLEXIBLE placement (tt-blaze
#4513, #4693, #4917) first plans which edges go over host, then reuses the same two carve
paths, then resolves each fabric chain on its own (see "Flexible placement" below).

This section reads tt-blaze `origin/main` (34e783a631). An older checkout (45d3be26c4, before
#4513) has neither `Edge.transport` nor `placement="flexible"`.

#### What blaze does

`PipelineGraph._collect_submeshes` (`pipeline_builder/graph.py`, tt-blaze):

| Step | Multi-mesh, planned carve | Multi-mesh, per-rank tiling | Single mesh |
|---|---|---|---|
| when | no `submesh_shape` | `submesh_shape` given | one mesh in the MGD |
| 1. slots | `plan_carve` → one `create_submesh(shape, offset)` per slot on this rank's mesh | `create_submeshes(submesh_shape)` on this rank's mesh | `create_submeshes(shape)`; every stage must have the same shape |
| 2. this rank's slot | the one slot holding its chips (exactly one) | first tile whose `(0,0)` chip is local, else first tile with any local chip | n/a (all tiles are local) |
| 3. global list | allgather each rank's slot chips | same | every tile, no allgather |
| 4. joint stages | ranks with identical chip sets collapse into one submesh, ordered by first rank | same | n/a |
| 5. layout | `resolve_graph_layout` (links from the control plane) | same | same |

`create_submeshes` (`MeshDeviceImpl::create_submeshes`, `tt_metal/distributed/mesh_device.cpp`)
stamps the shape in row-major order and fails unless it divides the mesh in both dims.
Chips are `(mesh_id, chip_id, row, col)` with `row, col` local to the submesh and `chip_id`
the mesh's row-major id (`MeshGraph::coordinate_to_chip`). The submesh shape for per-rank
tiling comes from the caller, in practice the harness's `mesh_shape()`, and every rank of a
mesh carves the same shape.

#### What Tier C does

Per seating, after intra-mesh completion (section 4):

1. **Slots.** One list of `(mesh_id, shape, offset)` per carve path:
   - *planned carve:* the C++ `plan_ring_carve` / `plan_fork_carve` (3.2), filler slots
     included;
   - *per-rank tiling:* for each mesh, a tile shape (below), stamped row-major over the mesh
     exactly like `create_submeshes`;
   - *single mesh:* the one stage shape stamped over the mesh.
2. **Each rank's slot.** For each mesh and host rank, `MeshGraph::get_coord_range(mesh,
   host_rank)` gives its chips. Apply blaze's step-2 rule to pick its slot.
3. **Submesh list.** Build each picked slot's chip tuples and collapse identical chip sets
   in first-rank order. On a single mesh, take every tile.
4. **Links.** Build `DirectLinks` between the submeshes from the candidate links (3.3) and
   routed intra-mesh links.
5. **Layout.** Call `resolve_graph_layout_with_connections` with those submeshes and links.

Which paths Tier C may use is what blaze has for that MGD, so an accepted mapping is one
blaze could build some way:

| MGD | STRICT | FLEXIBLE |
|---|---|---|
| one mesh | single-mesh tiling | single-mesh tiling |
| several meshes | planned carve or per-rank tiling | flexible placement; if a fork raises `ForkShapeUnsupported`, the STRICT paths |

A mapping is accepted if any allowed path succeeds. A FLEXIBLE ring whose planner fails
(for example non-uniform host slices) is rejected, because blaze raises there rather than
falling back.

**Tile shape per mesh (per-rank tiling).** Blaze gets it from the caller; Tier C enumerates
it. For each mesh, the candidates are the distinct stage shapes that divide the mesh, plus
the host-slice shape (for a mesh that carries no stage, whose ranks still carve something).
Try combinations across meshes; there is usually one candidate per mesh. One shape per mesh,
since every rank of a mesh carves the same shape.

#### Flexible placement

**What blaze does.** `PipelineGraph._resolve_topology` (`graph.py`) with
`placement="flexible"`, when the graph declares no HOST edge and has a loopback:

1. **Plan the host cuts** (`pipeline_builder/host_edge_planner.py`):
   - *ring:* `plan_host_edges` → `plan_ring_placement(slices, adjacency, stage_shapes,
     page_sizes)`. It first looks for an all-fabric cycle that the stage shapes align onto,
     the same search as `plan_ring_carve`. If none exists, it covers the ring's slices with
     `k` vertex-disjoint fabric chains, trying the fewest chains first up to one chain per
     stage. Stages are grouped along each chain, and consecutive chains are joined over host.
     A cut falls only between stages (never inside a wide stage) and never on stage 0's
     outgoing edge. Ties go to the smallest host-hop page sizes, then to cutting the loopback.
     Inputs follow the carve planner's rules: uniform slices, whole-slice stage shapes,
     stage 0 on one slice. Breaking them raises `HostEdgePlanError`, which is not caught.
   - *fork:* `plan_fork_host_edges` → `plan_fork_placement`. It tries `plan_fork_carve`
     first (all fabric). Failing that, it walks each branch from a distinct router neighbour
     and cuts to host where a branch runs out of fabric. Non-uniform slices or a router
     wider than one slice raise `ForkShapeUnsupported`; the caller then resolves the fork as
     declared, which is STRICT behaviour.
2. **Apply the plan.** The chosen edges are marked `transport="host"`.
3. **Carve.** If the plan has a carve (some stage spans several slices), its slots are used,
   as in the planned carve. Otherwise, if every stage has the same shape, blaze sets
   `submesh_shape` to that shape and uses **per-rank tiling**. This is the planner and
   per-rank tiling working together: the planner picks the cuts, and tiling makes the
   submeshes. A ring that is already all fabric on single-slice stages produces no plan,
   and placement goes to the resolver as in STRICT.
4. **Resolve each chain.** HOST edges are dropped from the resolver's edges, and the fabric
   graph splits into chains. `_resolve_chains_separately` calls the resolver once per chain.
   A chain's pool is the planner's chip sets for its stages when the plan fixed them,
   otherwise the submeshes not yet claimed by earlier chains. That second pool is greedy: an
   earlier chain may take submeshes a later one needed.

`host_hop_page_size_bytes`, required on every node under FLEXIBLE, only breaks ties, so it
does not affect whether a placement exists. It stays out of MGD.

**What Tier C does** for a FLEXIBLE descriptor, per seating:

1. Run the C++ `plan_ring_placement` / `plan_fork_placement` (3.2) on the slices and slice
   adjacency (section 4, steps 3–4).
2. Mark the planned edges HOST.
3. Produce slots as blaze does in step 3: the plan's carve, or per-rank tiling by the
   uniform stage shape (steps 1–3 of "What Tier C does" above). With a fixed tile shape
   there is no tile-shape enumeration.
4. Build `DirectLinks` once, then call `resolve_graph_layout_with_connections` per fabric
   chain with blaze's pool rule, including the greedy order. Matching that order keeps the
   verdict the same as blaze's.
5. Accept if the plan and every chain resolve. On `ForkShapeUnsupported`, fall back to the
   STRICT paths.

**What FLEXIBLE changes upstream of Tier C.** Under FLEXIBLE, most mappings that STRICT
rejects for missing fabric become buildable. Tier B drops the crossing rules (2.2), and
Plan 1 prunes nothing (5.1). Rejections that remain come from shapes, capacity, stage 0's
outgoing edge, and what the planner cannot cover.

**Staging.** FLEXIBLE support can land after STRICT. Until it does, Tier C must accept
FLEXIBLE descriptors without checking them. A STRICT check on a FLEXIBLE descriptor would
reject placements blaze can build.

#### Reuse or rebuild

| Piece | Today | Tier C |
|---|---|---|
| resolver | `resolve_graph_layout_with_connections` (C++) | **use directly** (3.1) |
| submesh links | `discover_submesh_links` (C++, anonymous namespace in `pipeline_builder.cpp`, asks the control plane per chip pair) | **refactor and share:** split out a pure core that takes "are these two chips linked, in which direction" as a callback. Tier D passes the control-plane lookup it uses today; Tier C passes the candidate links. One loop, two link sources |
| host-rank chip ranges, chip ids | `MeshGraph::get_coord_range`, `coordinate_to_chip` (C++) | **use directly** |
| row-major tiling | `MeshDeviceImpl::create_submeshes` (C++, needs a live `MeshDevice`) | **rebuild**, a few lines of offset arithmetic with the same divisibility rule. Factor it into a shared helper both call so the order cannot drift |
| planned slots | `plan_ring_carve`, `plan_fork_carve` (Python) | **port** (3.2) |
| host-cut planning | `plan_ring_placement`, `plan_fork_placement` (Python, `host_edge_planner.py`) | **port** (3.2), FLEXIBLE only; built on the carve planner's internals, so it mostly extends that port |
| per-chain resolution, chain split | `_resolve_chains_separately`, fabric component split (Python) | **rebuild**: a loop over the existing resolver, with the same pool rule and order |
| pick a rank's slot, collapse joint stages | `_collect_submeshes` (Python) | **rebuild** in C++, small and pure; add to the parity tests (3.2) |
| default core counts | `_pipeline_core_defaults` (Python) | **rebuild** (open question 7) |

#### Examples

- **GPT-OSS single galaxy:** one 8x4 mesh, four 4x2 stages. Single-mesh tiling gives four
  4x2 tiles; the resolver places the ring on them. The planner is never used, as at runtime.
- **`llama_8b_pod`:** M0 gets 4x2 tiles (its slice), M1 gets 1x2 tiles (its slice): 8 + 32
  submeshes, one rank each, and the resolver places all 40 stages, matching stage chip counts
  to submeshes. The planned carve rejects this MGD up front because the host slices are not
  all one shape, so per-rank tiling is the path that accepts it.
- **Joint stage:** a 4x4 stage on a 4x4 mesh with `host_topology [1, 2]`. The single 4x4 tile
  is picked by both ranks, and the collapse in step 3 merges it into one co-owned submesh.

#### Edge case: a tile smaller than a host slice

With per-rank tiling a rank can own several tiles (a 2x2 shape on a 4x2 slice). Blaze keeps
only the first, by the step-2 rule, and the others drop out of the submesh list. Tier C must
apply the same rule. The binding rule in 2.2 says such a stage cannot bind at all, which is
stricter than per-rank tiling; if runtime supports that layout, the binding rule should
allow it for the tiling path, or it eliminates placements early.

## 4. Where it runs in the topology mapper

The topology mapper now places meshes with SAT joint seating. A *seat* is a physical
footprint for one mesh; its chip-by-chip layout is a preference to the intra-mesh solve
(2.3). The loop in
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
  Tier B: at least one projection exists (no projection is chosen)
  Plan 1 pruning of seats and seat pairs, sound for every projection (5.1)

loop:
  seating = placement_session_->next()
  full = complete_intra_mesh_for_placement(...)
  if fail: add_forbidden_constraint(failed seat); loop          (unchanged)

  cands  = gather_intermesh_candidates(psd, full)  (3.3) + routed intra-mesh links
  slices = host slices of every mesh, through full.fabric_node_to_asic
  for each carve path allowed for this MGD and placement mode                 (3.4)
      STRICT:   planned carve, per-rank tiling, single-mesh tiling
      FLEXIBLE: host-cut plan, then its carve or uniform-shape tiling
    slots  = that path's slots over slices and cands                          (3.2, 3.4)
    subs   = pick each rank's slot, collapse joint stages                     (3.4)
    layout = detail::resolve_graph_layout_with_connections(subs, cands)       (3.1)
             (once per fabric chain when HOST edges split the graph)
    stop at the first path that succeeds
  R      = crossing links the layout uses
  bound  = bind_intermesh_connections(cands, mgd, required_links = R)  (3.3; 5.3, steps 2-4)
  rerun plan + layout over bound; it must succeed                     (5.3, step 5)
  if fail: feedback to SAT (5.2); loop                    ("pipeline_binding")

  emit full + plan + layout
```

#### Step by step

Worked example: `mesh4x4_stage4x2` from the blaze sweep. Two 4x4 meshes M0 and M1, each
`host_topology [1, 2]` (two 4x2 slices, one rank each), a 4-stage ring of 4x2 stages, and
two declared M0–M1 connections.

**Before the first solve.**

- *Tier B* (2.2): a projection exists, for example `s0, s1 → M0` and `s2, s3 → M1`. Its
  mirror `s0, s1 → M1` is equally valid and is **not** ruled out; neither is chosen. The ring
  needs 4 slices and each mesh has 2, so every projection puts two stages on each mesh and
  crosses M0–M1 at least twice (an interleaved split such as `s0, s2 → M0` crosses four
  times). Plan 1 uses the minimum over projections, `k(M0, M1) = 2`.
- *Plan 1 pruning* (5.1): drop M0–M1 seat pairs with fewer than 2 links in total. The
  slice-level form (2 links on distinct slices on both sides) needs each slice's ASICs, so it
  applies only where the layout is hard-pinned; otherwise it waits for the exact check.

**Per seating.**

1. **Seat.** SAT returns one seat per mesh, using only seat pairs that survived pruning.
2. **Intra-mesh.** Fills `full.fabric_node_to_asic`. Unchanged; a failure forbids the seat.
3. **Links.** Only links the control plane can route over count:
   - *between meshes:* `gather_intermesh_candidates` (3.3) returns every cable the pairing
     could bind, honouring link state and the one-destination-chip-per-source-chip rule. The
     PSD holds every host's cables, so this needs no MPI exchange. Which of them get bound is
     decided after step 6: the layout's crossing links are bound first, the rest of the
     budget is filled, and the check reruns over the bound set (5.3, steps 2–5);
   - *inside a mesh:* the mapped pairs of logical neighbours in `device_topology`, RING wraps
     included. Extra cables outside the topology are ignored.
4. **Slices.** For each mesh and host rank, `get_coord_range` gives a
   `HostSlice(mesh_id, row0, col0, nrows, ncols)`. Two slices are adjacent when a step-3 link
   joins them, matching blaze's `discover_slice_adjacency`.
5. **Carve.** Produce slots for each carve path blaze has for this MGD (3.4):
   - *Planned carve* (several meshes): `plan_ring_carve` (or `plan_fork_carve`) finds a
     cycle of slices and cuts it into stage rectangles, joint stages included. Here:
     `s0 → M0@(0,0)`, `s1 → M0@(0,2)`, `s2 → M1@(0,2)`, `s3 → M1@(0,0)`.
   - *Per-rank tiling* (several meshes): each mesh tiled by one shape, each rank keeps its
     tile, identical tiles collapse into joint stages, and the resolver in step 6 assigns
     stages to tiles. Here: 4x2 tiles, two per mesh, one per rank. This covers what the
     planner rejects up front: host slices of different shapes across meshes
     (`llama_8b_pod`), and graphs with no loopback (open chains).
   - *Single-mesh tiling* (one mesh): every tile of the one stage shape.
   - *Flexible placement* (FLEXIBLE descriptors): `plan_ring_placement` (or
     `plan_fork_placement`) picks host cuts, then the plan's carve or tiling by the uniform
     stage shape, then one resolve per fabric chain (3.4).

   Steps 6 and 7 run per path; a mapping is rejected only when every path fails. In Case B
   (both crossings on one slice pair) under STRICT, the planned carve fails with
   `no-cycle-of-length-4` and the per-rank tiling layout fails in the resolver, so the
   mapping is rejected. Under FLEXIBLE, the same Case B is accepted: with both cables
   between slices `M0a` and `M1a`, the slices still form one fabric path
   `M0b → M0a → M1a → M1b`. The planner lays `s0..s3` along it and cuts only the loopback
   `s3 → s0` to host, which is allowed because it is not stage 0's outgoing edge.
6. **Layout.** Feed the resolver what blaze feeds it:
   - stages as nodes; edges with one loopback per cycle; HOST edges dropped, and the
     remaining fabric chains resolved one at a time when HOST edges split the graph
     (`_resolve_chains_separately`);
   - each slot's chips as `submesh_chips`, with `node_chip_counts = rows * cols`;
   - blaze's default core counts (`_pipeline_core_defaults`: router `2 + 2 * branches`; a
     HOST-return tail 2 on a fused router's small submesh; ring head 4 with
     `co_locate_ring_host_io`; chain head and tail 2);
   - the step-3 links between slots as `direct_links`.

   It catches per-chip core capacity, shared entry and exit chips, and the router's endpoint
   budget.
7. **Feedback.** On failure, tell SAT (5.2) in terms of seats: block this set of seats by
   default, or a proven seat pair. Never forbid one mesh on one seat for a pipeline failure.
8. **Emit.** The mapping, carve plan and layout. Each rank's mesh and host rank give its
   slice, then its slot, then its `TT_PIPELINE_STAGE` (4.1).

**Gap.** Step 6 uses blaze's default core counts. Harness `pipeline_core_coords()` overrides
are not in the MGD, so Tier C could accept a layout runtime rejects (open question 7).

- **No descriptor, no change.** Without `pipeline_descriptors`, the loop is exactly today's.
- **Failure reporting.** `failure_stage = "pipeline_carve"`, `"pipeline_layout"` or
  `"pipeline_binding"` (5.3), carrying the carve phase, resolver reason, or the link that
  could not be bound. When the session is exhausted it surfaces through the
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
own Python carve planner. It records the mapper's witness placement (2.3), which may differ
from what blaze builds, and whether anything ever uses it is a separate decision.

---

## 5. SAT encoding

The SAT targets are logical meshes; the globals are seats. Each mesh-to-mesh connection must
land on a seat pair with links between them. Under STRICT the channel count is hard; under
RELAXED only adjacency is hard and the count is a preference. A seat fixes a mesh's
footprint, but its chip layout is only a preference to the intra-mesh solve (2.3), so the
ASICs under each host slice are not known before solving unless hard pinnings fix them.

### 5.1 Plan 1 - pipeline constraints in the solve (primary)

Plan 1 encodes necessary conditions before the master solve, in one solve. It does not
commit to a projection and does not run once per projection. **Every condition must hold for
all surviving projections and for every chip layout the intra-mesh solve may still pick.** A
condition that holds for only some of them would drop seatings the runtime could build on,
so it is left to the exact check (5.2).

From the surviving projections, take only what they all agree on:

- for each mesh pair A–B, `k_AB` = the **minimum** number of crossings over all projections
  (0 if any projection does not cross A–B);
- for each mesh, the boundary slices are the **union** over projections of the slices that
  may hold a boundary stage.

When exactly one projection survives (for example, every stage's binding set is a single
mesh), these are just that projection's values.

**Under FLEXIBLE placement**, any crossing may become a host hop, so the minimum `k_AB` is 0
for every mesh pair and the seat-pair and single-seat constraints below do not apply. Plan 1
then prunes nothing, and all pipeline checking falls to the exact check (5.2).

**Seat-pair constraint - usable links, not raw links.** In `build_sat_placement_seat_graph`,
for a mesh pair with `k_AB > 0`, drop the seat-graph edge between two seats when they cannot
carry `k_AB` crossings:

- **Footprint-level (always sound):** fewer than `k_AB` bound links (3.3) between the two
  footprints.
- **Slice-level (only with hard-pinned layouts):** the largest set of bound links whose
  endpoints sit on distinct allowed boundary slices on both sides (a bipartite matching) is
  below `k_AB`. This is what rejects Case B before the solve. Without hard pins the slice
  under each ASIC is not known yet, and Case B is rejected by the exact check instead.

- Dropping the edge makes it hard under RELAXED too, not just STRICT.
- Mesh pairs with `k_AB = 0` keep today's link count.

**Single-seat constraint.** In `build_sat_placement_constraints`, remove a seat from a mesh's
allowed set only when it fails under every projection and layout:

- the footprint has no bound link toward a neighbour mesh that every projection crosses
  to; or
- for a router whose binding set is this mesh alone, and only with a hard-pinned layout: no
  slice reaches the router's required degree (2 per fabric-return branch plus 1 per
  host-return branch). Otherwise the exact check catches it.

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

The exact check carves freely over the seating's slices, so a failure says "the pipeline
cannot be built on this set of seats with these layouts". It does not say which logical mesh
was at fault. Feedback must therefore be about seats, not about a mesh on a seat:

| Failure attributed to | Feedback |
|---|---|
| default: the seating as a whole | `exclude_mapping(seating)` - block this set of seats |
| one pair of seats, proven for every carve over them | block "seat_a and seat_b both used" (new API, below) |
| a seat that no carve could use under any seating | forbid that seat for every mesh |

- **Do not** use `add_forbidden_constraint(placed)` (one mesh on one seat) for a pipeline
  failure. Another mesh on the same seat, or the same mesh with the stages split differently,
  may succeed, so this would eliminate placements early. It stays for intra-mesh failures,
  as today.
- **Partial nogoods need a new API.** `exclude_mapping` with a partial map does not work
  today. Without `unique_shapes` (the setting this design uses, below) a partial map adds no
  clause at all. Under `unique_shapes` the clause becomes "some mesh uses a seat outside
  `{seat_a, seat_b}`", which with more than two meshes is always true and blocks nothing. The new clause should range
  over seats: "not (any mesh on seat_a and any mesh on seat_b)", keyed on seat identity
  including orientation, not on the ASIC set `block_assigned` matches today.
- Attribution needs the structured failures from 3.1 and 3.2.
- Cap the number of pipeline rejections per run and report the last carve or resolver reason.

#### Turn `unique_shapes` off for SAT placement

**Action:** when `pipeline_descriptors` is present, build the SAT placement session (the
master seating solve in `MultiMeshSolutionEnumerator`) with `unique_shapes = false`. Without
a pipeline descriptor, keep today's setting.

Where it is set today:

| Call site | `unique_shapes` | Level |
|---|---|---|
| `map_multi_mesh_to_physical` (both overloads, `topology_mapper_utils.cpp`), used by `TopologyMapper` at runtime | `true` | master seating |
| `generate_rank_bindings --all-solutions` | `true` unless `--allow-shape-permutations` | master seating |
| `run_topology_mapping` (`generate_rank_bindings` single-solution path) | `false` | master seating |
| seat-pool enumeration (`physical_grouping_descriptor_matching.cpp`) | `false`, must stay so to keep every orientation | per-mesh candidates |

So the runtime path, which is where Tier C runs, has it **on** today and needs the change.

**Code change.** Inside `SatPlacementEnumerationSession` the flag is read in one place, where
`MasterSolve::restart` builds the master `TopologyMappingEnumerationSession`
(`physical_grouping_descriptor_matching.cpp`). It arrives through the
`MultiMeshSolutionEnumerator` constructor, so the change is at the callers that pass `true`:

1. `map_multi_mesh_to_physical`, both overloads (`topology_mapper_utils.cpp`): pass
   `unique_shapes = !mgd_has_pipeline_descriptors`, read from the merged MGD (1.4). These
   two are the runtime path.
2. `generate_rank_bindings --all-solutions` (`tools/scaleout/src/generate_rank_bindings.cpp`):
   the same gate, so the offline sweep enumerates the seatings runtime would consider.
   `--allow-shape-permutations` keeps forcing it off.
3. `run_topology_mapping` (same file): already `false`; no change.

Leave these alone:

- seat-pool enumeration (`/*unique_shapes=*/false`), which must stay off to keep every
  orientation;
- the three `enumerate_flat_grouping_embeddings` calls that use the default `true` (MGD
  fallback check, grouping-match gate, `enumerate_distinct_placements_for_grouping`). The
  first two only ask whether one placement exists (`max_solutions = 1`), so the dedup cannot
  change their answer, and the third is meant to return distinct placements.

The gate depends on `pipeline_descriptors` existing in the proto (1.3), so it lands with or
after the schema change. Without a pipeline descriptor nothing changes, so today's mappings
and goldens stay as they are.

Why off: `unique_shapes` keys a seating by its sorted set of seats, so `{M0→a, M1→b}` and
`{M0→b, M1→a}` count as one, and only the first one tried is ever checked. That is only
safe if swapping meshes between the same seats cannot change the pipeline answer. It can:

- **The intra-mesh layout is per mesh.** Rank-binding, pinning and exit-node constraints
  differ by logical mesh, so `M0` on seat `a` and `M1` on seat `a` can complete to different
  chip layouts, hence different host slices and slice adjacency for the carve.
- **Per-mesh feedback strands the swap.** After a pipeline failure on `{M0→a, M1→b}`, the
  swapped seating has the same key and is never offered, even when it would build.

The rules elsewhere in this design (no split fixed early, 2.2 and 5.1; feedback about seats,
5.2) still apply; with `unique_shapes` off they no longer have to carry this alone.

**Cost.** Off, the master solve can yield every mesh permutation of a seat set, which is what
the dedup was added to avoid on large symmetric machines. Cap pipeline rejections per run
(5.2) and measure on the ring-stress and superpod cases before turning it off by default.

### 5.3 Pipeline-aware exit-node binding

The checks above decide whether a seating *can* carry the pipeline. The control plane then
decides which cables actually carry each MGD connection, and today it does that without
looking at the pipeline. This section makes that choice follow what the pipeline builder
needs, so the links Tier C relied on are the links that get bound.

#### How exit nodes are chosen today

`ControlPlane::generate_intermesh_connectivity` (`control_plane.cpp`):

1. **Gather** (`gather_intermesh_cables_for_exit_nodes`, per host and neighbour rank): every
   PSD cable toward a requested neighbour mesh, sorted by `(src chip, src chan, dst chan)`.
   Cables whose ethernet link is down are dropped. Under STRICT `Graph` connections, only
   pinned exit nodes are kept. A source chip may cable to only one destination chip per
   neighbour mesh.
2. **Pair** (`pair_logical_intermesh_ports`, rank 0, after an allgather): cables seen from
   both sides become links, ordered by `(src node, dst node)`. The work-list is
   mesh-pair-major under STRICT and round-robin across mesh pairs under RELAXED. The phases
   are: assign_z mesh pairs on Z, then NESW, then NESW overflow on leftover Z. A link is
   placed only if both ends find free ports in one direction that face that neighbour, and
   each mesh pair stops at its requested count.
3. **Single host** (`generate_intermesh_connections_on_local_host`): a separate greedy pass
   over each local mesh's exit nodes with the same one-destination-chip rule. It does not
   use the pairing allocator.

Under RELAXED, the exit nodes are therefore just **the lowest-numbered chips that fit the
budget**. With `count: 2` between M0 and M1 and four candidate cables, the two links with
the smallest `(src, dst)` chip ids are bound, even if both sit on one slice pair (Case B)
and cables on another slice pair would have closed the ring. The PSD could carry the
pipeline, but the binding throws that away. Under STRICT `Graph` connections the user pins
exit chips, so there is no freedom to use; the pipeline must fit the pins, and Tier C
checks against exactly those.

#### Plan

1. **Split gather from bind (3.3).** `gather_intermesh_candidates(psd, mapping)` returns
   every link the pairing could bind for a mapping: both-sides join, link up, the
   one-destination-chip rule, STRICT pins applied. It is pure over the PSD, so Tier C and the
   control plane call the same function. `bind_intermesh_connections(candidates, mgd,
   required_links)` runs today's phases.
2. **Tier C picks the required links.** Tier C runs the planner and resolver over the
   *candidates* (section 4, steps 3–6). `R` is the cable behind each FABRIC crossing edge in
   the resolver's `resolved_edges` (its exit and entry chips identify the link). `R` holds at
   most one link per crossing edge, so it never exceeds the requested count when Tier B's
   channel rule holds.
3. **Bind `R` first, then fill.** In `bind_intermesh_connections`, add a phase 0 ahead of
   phase 1a. It places each link in `R` with its mesh pair's Z rule (assign_z → Z, otherwise
   NESW then Z) and counts it against the budget. The remaining budget is then filled by
   today's work-list, skipping placed links. With `R` empty the result is exactly today's.
4. **Fill toward spread, not chip id (pipeline descriptors only).** For the fill, order
   each mesh pair's remaining links so that consecutive picks land on different host-slice
   pairs (round-robin over slice pairs, chip id within one). Spare bound links then add slice
   adjacency rather than doubling an existing pair. That gives blaze's runtime carve planner,
   which reruns on the bound links, more cycles to choose from (step 5). Without a pipeline
   descriptor, keep today's order so existing bindings and goldens do not move.
5. **Rerun over the bound links.** Run the planner and resolver again over the bound set,
   with the same routing-plane rule blaze applies: a link counts only if
   `get_forwarding_direction` would confirm it, and inter-mesh that means exactly the bound
   links (`discover_slice_adjacency`). Accept only if this rerun succeeds. That makes the
   accepted seating hold for the links runtime will actually see, not just for the PSD.
6. **Failures feed back.** If phase 0 cannot place a link in `R` (a direction already owned
   toward another mesh, Z-ness, the one-peer-per-direction rule, or no ports left), or the
   rerun fails, try the next layout from the resolver if one is cheap to get. Otherwise
   report `failure_stage = "pipeline_binding"` and block the seating (5.2). Blaze's own
   runtime planner might pick a different cycle than Tier C's witness. Accepting in step 5
   still means *some* carve works on the bound links, which is the existence guarantee
   (2.3).
7. **Same answer on every rank.** The mapping is computed identically everywhere, so `R`
   is too; the control plane gets it from the `TopologyMapper` (for example
   `pipeline_required_intermesh_links()`). The pairing still runs on rank 0, so only rank 0
   consumes it. `validate_requested_intermesh_connections` runs unchanged on every rank.
8. **Single host joins the same rule.** Route `generate_intermesh_connections_on_local_host`
   through `gather_intermesh_candidates` + `bind_intermesh_connections` (with every cable
   local), so single-galaxy multi-mesh MGDs get phase 0 and the spread fill too. Until then,
   Tier C on a single host must model that path's greedy order, or skip the rerun there and
   note the gap.

#### Gaps

- **Live link state.** The control plane drops cables whose link is down at init. Offline
  (`generate_rank_bindings` on a factory PSD), every cable looks up. A required link that is
  down at runtime fails phase 0, and the runtime mapper (which runs Tier C itself) then
  picks another seating. The offline answer can be optimistic.
- **Intra-mesh links.** Inside a mesh, Tier C counts the mapped logical neighbours in
  `device_topology`. Links the topology does not route over (extra cables, inactive routing
  planes) are excluded, matching what blaze's adjacency and the resolver confirm.

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
- At runtime blaze carves again, by planned carve or uniform tiling depending on how it is
  called, and may pick a different placement than the mapper's witness. That is accepted
  (2.3): the mapper only guarantees that a placement exists on the mapping. If
  `TT_PIPELINE_STAGE` is ever consumed, the two would have to be made to agree first.

---

## 7. Open questions

1. **Adjacency parity.** Tier C builds slice adjacency from bound inter-mesh links and routed
   intra-mesh links; runtime uses `get_forwarding_direction`-confirmed neighbours. Confirm the
   two agree, including links with no active routing planes.
2. **Projection count.** For heterogeneous or fork pipelines, how many projections survive
   Tier B in practice? With many survivors the minimum `k_AB` and union of boundary slices
   (5.1) get weak, and more of the work falls to the exact check. Is that affordable?
3. **Shared seats with different `k`** (5.1): handle generally, or reject until a real
   descriptor needs it.
4. **Seat-pair nogoods** (5.2): partial `block` does not work today, in either
   `unique_shapes` mode. Design the seat-level API, and decide when a failure is provably
   about one pair of seats rather than the whole seating.
5. **Cost of `unique_shapes` off** (5.2): how many extra seatings the master solve yields on
   the large symmetric cases once mesh permutations are no longer collapsed.
6. **Planned key name** for the rank-binding `env_overrides` entry (4.1).
7. **Per-stage core counts.** Harnesses can override pipeline-core pools, which the MGD does
   not carry. Add an optional core count to `PipelineStage`, or accept the gap (4, step 6).
8. **STRICT or FLEXIBLE per descriptor** (1.3, 3.4). Blaze picks the placement mode at
   runtime (`--pipeline-builder`), not in the graph. If one MGD is run both ways, the
   descriptor must say FLEXIBLE, or Tier C will reject placements the flexible run could
   build. Decide whether the MGD carries the mode, or Tier C always checks the most
   permissive mode the deployment uses.
9. **Blaze's first-plan commitment** (3.2). Tier C accepts if any cycle resolves, but blaze
   uses only its first. Either teach blaze's planner to try the next cycle when the resolver
   fails, or have Tier C check only blaze's first cycle once the mapper's output is consumed
   (4.1).
10. **Spread fill for the binding** (5.3, step 4). Turn it on only with a pipeline descriptor,
    or everywhere once the goldens are regenerated.

---

## 8. Worked examples, end to end

Everything in this section is hypothetical and traced by hand through the code as it stands:
tt-blaze `origin/main` (34e783a631) for the carve planner, `_collect_submeshes` and the host
edge planner, and tt-metal on this branch for SAT placement, the intra-mesh solve and the
graph-layout resolver. Steps this design adds are marked *new*. Nothing here was run.

8.1 explains the carve planner on its own, with small examples. 8.2 and 8.3 then follow two
pipelines through every stage: parse (Tier A), the declared fabric (Tier B), SAT placement,
the intra-mesh solve, the per-seating check (Tier C), cable binding, and blaze's build at
runtime (Tier D). 8.4 lists the special cases the examples hit.

All examples use the same slice names. A mesh driven by four ranks with `host_topology
[1, 4]` has four host slices; on mesh M0 they are A0..A3 (ranks 0–3), on M1 they are B0..B3
(ranks 4–7). Inside a mesh the slices form a line (A0–A1–A2–A3). Cables between meshes are
drawn as vertical bars.

### 8.1 How the carve planner works

The carve planner (`pipeline_builder/carve_planner.py`) answers one question: given every
rank's host slice and which slice pairs the fabric links, which slices does each stage get?

It works on host slices, not chips. `_collect_submeshes` requires each rank's chips to sit
inside exactly one slot, so a slot is always a union of whole slices. That keeps the problem
small: a 512-chip system with 4x2 slices is a graph of 128 vertices.

Every rank runs the planner on its own and must reach the same plan, because carving is
collective. So the plan is a pure function of allgathered integers. Slices are ordered by
`(mesh_id, row0, col0)`, never by rank, every loop runs over a sorted sequence, and
`plan_carve` finishes by exchanging a crc32 of the plan and failing if any rank disagrees.

#### The phases of `plan_ring_carve`

| Phase | Code | What it does |
|---|---|---|
| 1. discover | `discover_host_slices`, `discover_slice_adjacency` | Allgather each rank's local chips; each rank's bounding box becomes a `HostSlice(mesh_id, row0, col0, nrows, ncols)`. For each chip and direction (N, E, S, W, Z), a neighbour on another slice counts only if `get_forwarding_direction` confirms it. The result is a set of slice pairs; one confirmed link is enough, counts are not kept. |
| 0. input checks | top of `plan_ring_carve` | All slices one shape. Each stage shape a whole number of slices (`r % nrows == 0`, `c % ncols == 0`), giving `stage_counts`. `need = sum(stage_counts)` no more than the number of slices. Stage 0 is exactly one slice. |
| 2. reduce | `_prune_leaves`, `_contract` | Repeatedly drop slices with fewer than two live neighbours (they cannot lie on a cycle). Slices with more than two neighbours are *branch slices*; each run of two-neighbour slices between branch slices becomes one weighted chain. Several chains may join the same two branch slices. |
| 3. cycle search | `_cycles_of_length`, `_cycles_no_branch` | Find simple cycles of exactly `need` slices. With no branch slices the graph is disjoint loops, and each loop is walked once. Otherwise a DFS over branch slices picks one unused chain per step, adding its weight plus one, and prunes any partial cycle above `need`. Stops at 200,000 expansions (`budget-exhausted`) or 64 cycles. |
| 4. align | `_group_cycle`, `_canonical_walk`, `_as_rectangle` | Cycles are sorted by mesh crossings, fewest first. For each cycle, start from the canonical walk (anchored at the lowest slice, stepping toward its lower neighbour), then try every rotation and the reverse direction. Cut consecutive groups of `stage_counts[i]` slices; each group must be one gap-free rectangle on one mesh with exactly the declared shape (no transpose). The first alignment that works wins. |
| 5. emit | end of `plan_ring_carve`, `validate_carve_plan` | One `CarveSlot(mesh_id, shape, offset)` per stage in stage order, then one filler slot per unused slice (its rank idles). Every chip of every mesh must be covered exactly once. |

The failures name their phase: `host slices are not uniform`, `stage i declares shape ...
not a whole number of host slices`, `input-infeasible`, `no-cycle-of-length-N` (with the
cycle lengths that are achievable), `budget-exhausted`, and `no-alignment` (with the
cycle's same-mesh run lengths).

After the plan, `_collect_submeshes` turns slots into submeshes. Each rank calls
`create_submesh(shape, offset)` for every slot on its own mesh, in plan order (a collective
over that mesh's ranks), and keeps the one slot that holds its chips. Chip lists are
allgathered and identical chip sets collapse into one submesh, which is how a stage
spanning two ranks becomes one co-owned node. The resolver (`resolve_graph_layout`) then
decides which stage goes on which submesh; it does not take the planner's stage order as
given, it re-derives the assignment from chip counts and links.

In Tier C, phase 1 comes from the mapper instead of ttnn: slices from
`MeshGraph::get_coord_range` and adjacency from the candidate inter-mesh links plus routed
intra-mesh links (section 4, steps 3–4). Phases 0 and 2–5 are the C++ port (3.2).

#### Carver example A: one loop, no branch slices

Two meshes, cables A0–B0 and A2–B2. Ring: s0 4x2, s1 4x4, s2 4x4, s3 4x2.

```
   A0 ── A1 ── A2 ── A3
   │            │
   B0 ── B1 ── B2 ── B3

phase 0   slices all 4x2; stage_counts = [1, 2, 2, 1]; need = 6 <= 8; stage 0 is 1 slice
phase 2   A3 has one live neighbour (A2), B3 has one (B2): drop both
          alive = {A0 A1 A2 B0 B1 B2}; every alive slice has 2 neighbours -> no branch slices
phase 3   _cycles_no_branch walks from the lowest slice A0:
          A0 A1 A2 B2 B1 B0              length 6 == need
phase 4   canonical walk: anchor A0; A1 (0,0,2) < B0 (1,0,0), so step to A1
          forward, offset 0:
            s0 <- [A0]      M0 rows 0-3 cols 0-1   4x2   ok
            s1 <- [A1 A2]   M0 rows 0-3 cols 2-5   4x4   ok
            s2 <- [B2 B1]   M1 rows 0-3 cols 2-5   4x4   ok
            s3 <- [B0]      M1 rows 0-3 cols 0-1   4x2   ok
phase 5   slots: M0 4x2@(0,0), M0 4x4@(0,2), M1 4x4@(0,2), M1 4x2@(0,0),
          fillers M0 4x2@(0,6), M1 4x2@(0,6)        (ranks 3 and 7 idle)
```

Twelve alignments exist (six rotations, two directions); four of them work. The other three
are forward offset 3 (s0 B2, s1 B1+B0, s2 A0+A1, s3 A2), reverse offset 1 (s0 B0, s1 B1+B2,
s2 A2+A1, s3 A0), and reverse offset 4 (s0 A2, s1 A1+A0, s2 B0+B1, s3 B2). The rest fail in
`_as_rectangle`, because a 4x4 group would take one slice from each mesh. The planner returns
the first; Tier C only needs one to exist.

#### Carver example B: branch slices and contraction

Add a third cable, A3–B3:

```
   A0 ── A1 ── A2 ── A3
   │            │     │
   B0 ── B1 ── B2 ── B3

phase 2   nothing pruned. Branch slices: A2 (A1, A3, B2) and B2 (B1, B3, A2).
          _contract walks out of A2 in neighbour order and finds three chains A2 -> B2:
            c0  interior A1 A0 B0 B1   weight 4
            c1  interior A3 B3         weight 2
            c2  direct                 weight 0
phase 3   a cycle is two different chains between A2 and B2: 2 + weight + weight slices
            c0 + c2 = 6     c1 + c2 = 4     c0 + c1 = 8
          need 6: the DFS finds c0 + c2, i.e. A2 A1 A0 B0 B1 B2 (four times: two start
          slices, two directions; it is one ring). c0 + c1 exceeds 6 and is pruned.
phase 4   same canonical walk and plan as example A.
```

The extra cable changes nothing for this ring. A ring of four 4x2 stages (need 4) would take
`c1 + c2`, the cycle A2 A3 B3 B2.

One quirk shows up here. With branch slices, the search records only lengths up to `need`,
because longer partial cycles are pruned before they close. A ring needing 5 slices on this
fabric fails with `no-cycle-of-length-5` and `achievable cycle lengths: [4]`, although an
8-cycle also exists. With no branch slices (example A) every loop's length is reported.

#### Carver example C: right length, no alignment

Same fabric as example A, ring s0 4x2, s1 4x2, s2 4x8 (`stage_counts = [1, 1, 4]`, need 6).

```
phase 3   cycle A0 A1 A2 B2 B1 B0, length 6
phase 4   s2 needs 4 slices in one row of one mesh. The cycle's same-mesh runs are
          [3, 3] (A0 A1 A2, then B2 B1 B0), so every rotation puts s2 across the crossing.
error     no-alignment: found 1 cycle(s) of the right length (6 slices) but no rotation
          places every stage as its declared rectangle ...
            cycle's same-mesh run lengths: [3, 3]
            stage shapes: ['4x2', '4x2', '4x8']
```

#### Carver example D: a fork (`plan_fork_carve`)

A fork is a router stage plus branches. A branch that returns over fabric needs a cycle
through the router (router, head, ..., tail, router). A branch that returns over host needs
only an open path starting next to the router. The branches share nothing but the router.

| Step | What it does |
|---|---|
| input checks | Router is exactly one slice; branch stages are whole slices. `need_edges = 2 x fabric-return branches + 1 x host-return branches`; `need_slices = 1 + all branch slices`. |
| router candidates | Every slice with at least `need_edges` neighbours, in slice order. None gives `no slice has the N fabric neighbours this fork needs`. |
| branch order | Fabric-return branches first, shortest first (a short cycle must hug the router), then host-return branches. |
| walks per branch | Fabric return: `_cycles_through` reuses prune, contract and cycle search on the free slices and keeps cycles through the router, offered in both directions. Host return: `_path_of_length` from each free router neighbour, depth first, lowest index first. Each walk is cut into the branch's rectangles. |
| search | Backtracking over each branch's walks, capped at 2,000 attempts, because one branch's first walk often eats slices a later branch needs. |
| emit | Router slot, branch slots, fillers. |

On example B's fabric, take a 4x2 router R, branch X of three 4x2 stages returning over
fabric, and branch Y of one 4x2 stage returning over host:

```
need_edges = 2 + 1 = 3; need_slices = 1 + 3 + 1 = 5
router candidates: A2 and B2 (three neighbours each); try A2
branch X (fabric, 3 slices) needs a 4-cycle through A2: A2 A3 B3 B2
  walks [A3 B3 B2] and [B2 B3 A3]; take [A3 B3 B2]: X0 A3, X1 B3, X2 B2, return B2 -> A2
branch Y (host, 1 slice): A2's free neighbours are {A1}; path [A1]: Y0 A1, return over host
plan: R A2 | X A3, B3, B2 | Y A1 | fillers A0, B0, B1
```

If Y also returned over fabric, the router would need four neighbours and no slice has more
than three, so the planner fails at the candidate step.

#### Flexible placement (`plan_ring_placement`)

Under FLEXIBLE, blaze may turn ring edges into host hops. The host edge planner runs the
same input checks, then:

1. **Phase 1, a fabric cycle.** The same prune, contract and cycle search as the carve
   planner, then an alignment with no cuts. If one exists there are no host edges.
2. **Phase 3, cover with chains.** Count the fewest fabric components that hold `need`
   slices and start `k` there. `_cover_with_chains(k)` finds covers of `need` slices by `k`
   disjoint simple paths, depth first: start a chain at the smallest free slice, extend to
   the smallest free neighbour.
3. **Phase 4, align.** `_group_walk` lays the stages along the covers' concatenated walk.
   The joints between chains must fall between stages, never inside a wide stage, and never
   on stage 0's outgoing edge (cut 0). Among valid alignments the key is (total page size
   of cut edges, 0 if the loopback is cut else 1, cut indices); the smallest wins. Each cut
   costs `HOST_HOP_COST = 10` in the plan.
4. The plan's carve (if a stage is wide) or per-rank tiling by the uniform stage shape makes
   the submeshes, the cut edges become HOST, and `_resolve_chains_separately` resolves each
   fabric chain on its own.

On the fabric where cables are A0–B0 and A3–B3 (one loop of 8), with example A's ring:

```
   A0 ── A1 ── A2 ── A3
   │                  │
   B0 ── B1 ── B2 ── B3

phase 1   no 6-cycle (only the 8-cycle). Census note: 2 passthrough stages would make an
          8-stage ring that closes over fabric (passthrough_ring = 2).
phase 3   one component of 8 >= 6, so k starts at 1. Covers in search order:
            1. A0 A1 A2 A3 B3 B2    2. A0 B0 B1 B2 B3 A3    3. A1 A0 B0 B1 B2 B3    ...
phase 4   cover 1, best: s0 A3, s1 B3+B2, s2 A0+A1, s3 A2; cuts s1 -> s2     key (1, 1, (1,))
          cover 2, best: s0 A3, s1 B3+B2, s2 B1+B0, s3 A0; cuts s3 -> s0     key (1, 0, (3,))
          cover 2 wins: one cut, and it is the loopback.
result    host edge s3 -> s0; carve M0 4x2@(0,6), M1 4x4@(0,4), M1 4x4@(0,0), M0 4x2@(0,0);
          A1 and A2 idle. One fabric chain s0 -> s1 -> s2 -> s3 is resolved.
```

### 8.2 Example 1: stages that span two ranks (planned carve)

**Setup.** Two 4x8 meshes, each driven by four ranks (`host_topology [1, 4]`, so 4x2
slices). The ring has two 4x4 stages, which cannot fit in one rank.

```protobuf
mesh_descriptors {
  name: "M"  arch: BLACKHOLE
  device_topology { dims: [ 4, 8 ] }
  host_topology   { dims: [ 1, 4 ] }
  channels { count: 2 policy: RELAXED }
}
# graph: M0 <-> M1, channels { count: 2 policy: RELAXED }

pipeline_descriptors {
  name: "mixed4"  placement: STRICT
  stages { id: "s0" shape { dims: [ 4, 2 ] } }   # 1 slice
  stages { id: "s1" shape { dims: [ 4, 4 ] } }   # 2 slices, two ranks
  stages { id: "s2" shape { dims: [ 4, 4 ] } }   # 2 slices, two ranks
  stages { id: "s3" shape { dims: [ 4, 2 ] } }   # 1 slice
  edges { src: "s0" dst: "s1" }  edges { src: "s1" dst: "s2" }
  edges { src: "s2" dst: "s3" }  edges { src: "s3" dst: "s0" }
}
```

The hypothetical machine is three 32-chip pods, each a 4x8 grid driven by four hosts
(h0..h3, one 4x2 block each), with single cables between some hosts:

```
P.h0 ── Q.h0   P.h3 ── Q.h3      P–Q: 2 links
P.h0 ── R.h0   P.h2 ── R.h2      P–R: 2 links
Q.h1 ── R.h1                     Q–R: 1 link
```

Assume the intra-mesh solve puts mesh host rank i on host hi of its seat (the PGD's
preferred layout), so with M0 on P, slice A2 is P.h2.

#### Step 1: Tier A (MGD parse)

`validate_pipeline_descriptors` checks the descriptor alone: stage ids are unique, every
edge names declared stages, there is no self-edge, every stage has out-degree 1 (a ring),
one cycle exists and `s3 -> s0` becomes its loopback, and every shape is positive. Pass.

#### Step 2: Tier B (declared fabric, no physical system)

- **Slices.** 4x8 divided by `[1, 4]` gives 4x2 slices, four per mesh.
- **Binding sets.** 4x2 is one slice and 4x4 is two, both on either mesh. Stage 0 is one
  slice.
- **Capacity.** The ring needs 6 slices; the MGD has 8.
- **Projections.** Each mesh holds at most 4 slices, so neither mesh can take the whole ring
  (6). For example, `s0, s1 -> M0` and `s2, s3 -> M1` (3 slices each) crosses twice, within
  the 2 channels. `s0, s1, s3 -> M0` and `s2 -> M1` (4 and 2) also crosses twice. An
  interleaved split such as `s0, s2 -> M0` crosses four times and is dropped.
- **Segment carve.** In the first projection, M0's segment (1 + 2 slices) is a path of three
  slices in a line of four, which fits.
- **Result.** Pass. Every projection crosses M0–M1, and a ring crosses a cut an even number
  of times, so Plan 1 gets `k(M0, M1) = 2`.

#### Step 3: SAT placement setup

- **Seat pools.** `create_sat_placement_pools` embeds each mesh's PGD grouping (a 4x8 over
  four hosts) into the physical graph. Both meshes get the pool {P, Q, R}.
  `sat_footprint_canonical` merges orientations of the same footprint (the key is grouping
  name, type and sorted ASICs), so each pod is one seat.
- **Seat graph.** `build_sat_placement_seat_graph` counts links between every seat pair for
  each MGD mesh pair (`saturated_link_count`) and adds that many parallel edges: P–Q x2,
  P–R x2, Q–R x1.
- **Constraints.** `build_sat_placement_constraints` requires each mesh on a seat from its
  pool. M0 and M1 share a pool but are adjacent, so they are not twins and get no symmetry
  window. The footprints are disjoint, so injectivity alone keeps chips unshared.
- ***New:* Plan 1 (5.1).** With `k = 2`, drop seat-graph edges between seats with fewer than
  2 links: Q–R goes. This is the footprint-level rule; the layout is not hard-pinned, so the
  slice-level rule cannot run. It would not help here anyway: P–Q's two links are on
  distinct slice pairs, and P–Q fails for a reason only the exact check sees (step 4).
- ***New:* `unique_shapes = false`** because a pipeline descriptor is present (5.2).
- **Master solve.** `MasterSolve::restart` builds the master session; it maps the mesh graph
  (M0–M1, count 2, RELAXED) onto the seat graph. The candidate seatings are (M0, M1) = (P,
  Q), (Q, P), (P, R), (R, P). Without Plan 1, (Q, R) and (R, Q) would also be offered,
  since RELAXED only needs adjacency. With `unique_shapes` on, (P, Q) and (Q, P) would count
  as one.

Suppose the solver offers them in the order (P, Q), (Q, P), (P, R).

#### Step 4: seating (P, Q)

**Intra-mesh** (`complete_intra_mesh_for_placement`, per mesh):

| Constraint | Function | Here |
|---|---|---|
| rank binding (hard) | `add_rank_binding_constraints` | M0's host-rank-i fabric nodes map to P.hi's ASICs; M1's to Q.hi's |
| exit nodes (hard) | `add_exit_node_constraints` | the connection is mesh-level, so one cardinality constraint: at least 2 of M0's nodes on ASICs cabled toward Q (capped by the physical link total under RELAXED). P has exactly two such ASICs, on h0 and h3 |
| MGD pins (hard) | `add_pinning_constraints` | none declared |
| PGD layout (soft) | `add_pgd_pinning_preferred_constraints` | biases toward the PGD layout |

Both meshes solve. `full.fabric_node_to_asic` now fixes which slice each cable touches:
A0–B0 and A3–B3.

***New:* Tier C.**

1. **Links.** `gather_intermesh_candidates` returns the two P–Q cables. The routed intra-mesh
   links give the two lines.
2. **Slices and adjacency.** A0..A3, B0..B3; lines plus A0–B0 and A3–B3, one loop of 8.
3. **Planned carve.** No slice is pruned and none is a branch slice, so
   `_cycles_no_branch` finds the single 8-cycle: `no-cycle-of-length-6`, achievable [8].
4. **Per-rank tiling.** The tile shape per mesh is one of {4x2, 4x4}. The resolver's chip-count
   filter then decides:
   - 4x2 on both meshes gives eight 8-chip submeshes and nothing with 16 chips for s1.
   - 4x4 on both gives four 16-chip submeshes (two ranks each) and nothing with 8 chips for
     s0.
   - 4x2 on M0 and 4x4 on M1 puts s0 and s3 on A slices and s1, s2 on M1's two 4x4 tiles.
     s3 -> s0 then needs two neighbouring A slices that both have cables to M1, but the
     cables are on A0 and A3, which are not neighbours. The mirror (4x4 on M0) fails the same
     way.
5. **Result.** Every path fails, so the seating is rejected with `failure_stage =
   "pipeline_carve"` and the last reason attached.
6. **Feedback (5.2).** `exclude_mapping({M0 -> P, M1 -> Q})`. Not
   `add_forbidden_constraint(M0 on P)`: M0 on P is exactly what succeeds in step 6.

#### Step 5: seating (Q, P)

Same two cables with the meshes swapped. Under the assumed layout the slice geometry is the
same, so it fails and is excluded the same way. With `unique_shapes` on, this seating would
never be offered. That is harmless here, but if the two meshes' layouts differed (different
pins or rank bindings), the swap could be the one that works.

#### Step 6: seating (P, R)

Intra-mesh succeeds as in step 4, with cables on A0–B0 and A2–B2.

***New:* Tier C.**

1. **Planned carve.** Carver example A: plan `M0 4x2@(0,0)`, `M0 4x4@(0,2)`, `M1 4x4@(0,2)`,
   `M1 4x2@(0,0)`, fillers `M0 4x2@(0,6)` and `M1 4x2@(0,6)`.
2. **Slots to submeshes** (the `_collect_submeshes` model):

   ```
   M0 slots in plan order: 4x2@(0,0), 4x4@(0,2), 4x2@(0,6)
     rank 0 -> slot 4x2@(0,0)    rank 1 -> 4x4@(0,2)    rank 2 -> 4x4@(0,2)    rank 3 -> filler
     (a rank whose chips fall in zero or two slots is a planner bug and raises)
   collapse identical chip sets:
     submesh 0 = A0 (8)   1 = A1+A2 (16, ranks 1 and 2)   2 = A3 (8)
     submesh 3 = B0 (8)   4 = B1+B2 (16, ranks 5 and 6)   5 = B3 (8)
   6 distinct submeshes >= 4 stages
   ```

3. **Resolver** (`resolve_graph_layout` -> `resolve_pipeline_placement`):

   ```
   topological order, loopback excluded: s0 s1 s2 s3
   chip counts: s0 8, s1 16, s2 16, s3 8
   submesh links: 0-1, 1-2, 3-4, 4-5 (inside meshes), 0-3 and 1-4 (the cables)

   s0: needs 2 links + 2 host endpoints = 4 <= capacity 1 x 8 chips; needs 2 neighbours
       submesh 0 (neighbours 1, 3)  candidate      submesh 2 (neighbour 1 only)  skip
       submesh 3 (neighbours 0, 4)  candidate      submesh 5 (neighbour 4 only)  skip
       try submesh 0
   s1: 16 chips and linked to submesh 0: only submesh 1
   s2: 16 chips and linked to submesh 1: submesh 4, over the A2-B2 cable
   s3: 8 chips, linked from 4 and to 0: submesh 3, closing over the A0-B0 cable
   LinkSearch: one chip pair per edge, exclusive chips first; host endpoints on s0's
               unused chips
   result: s0 -> 0, s1 -> 1, s2 -> 4, s3 -> 3
   ```

   Had submesh 0 failed, the resolver would try submesh 3 next and find the mirror (s0 B0,
   s1 B1+B2, s2 A2+A1, s3 A0).

4. ***New:* binding (5.3).** The crossing edges are `s1 -> s2` (A2–B2) and `s3 -> s0`
   (A0–B0), so `R` holds those two cables. Phase 0 binds them first and fills the count of
   2. The rerun over the bound links gives the same plan and layout. Accept.

   Suppose the machine had a second A0–B0 cable on lower chip ids than A2–B2. Today's
   binding would take the two lowest ids, both A0–B0, and the ring could not close. Phase 0
   binds `R` first, so the A2–B2 cable is kept.

5. **Emit.** The mapping, plan and layout. `rank_bindings.yaml` records `TT_PIPELINE_STAGE`
   (4.1): rank 0 `s0`, ranks 1 and 2 `s1`, ranks 5 and 6 `s2`, rank 4 `s3`, ranks 3 and 7
   `idle`.

#### Step 7: runtime (Tier D, unchanged)

The control plane binds the two cables in `R`. Blaze's `build_topology` is called without
`submesh_shape`, so `_collect_submeshes` calls `plan_carve`. It allgathers the slices, builds
adjacency from links that `get_forwarding_direction` confirms (exactly the bound cables plus
the intra-mesh lines), and reaches the same plan as Tier C. The ranks exchange the crc32,
carve with `create_submesh`, collapse ranks 1+2 and 5+6, and resolve.

Without 5.3, on the machine with the extra A0–B0 cable, blaze would see only A0–B0. Leaf
pruning would then remove every slice (A3 and B3 first, then A2 and B2, and so on), and
blaze would fail with `no-cycle-of-length-6: only 0 slices can lie on any cycle`. That is
the hard runtime failure this design moves into the mapper.

### 8.3 Example 2: different slice shapes (per-rank tiling)

**Setup.** M0 is 4x8 with `host_topology [1, 4]` (4x2 slices, ranks 0–3). M1 is 2x4 with
`host_topology [2, 2]` (1x2 slices, ranks 4–7). This is the `llama_8b_pod` pattern at small
scale. The ring has eight stages: s0..s3 are 4x2, s4..s7 are 1x2.

```
M1 (2x4), one 1x2 slice per rank      slice adjacency inside M1: a 4-cycle
        cols 0-1   cols 2-3
row 0   T0 (r4)    T1 (r5)             T0 ── T1
row 1   T2 (r6)    T3 (r7)             │      │
                                       T2 ── T3
```

The machine has one pod P (as in 8.2) and two 2x4 boards U and V, each driven by four hosts
of two chips. Each board has two cables to P, from hosts P.h0 and P.h3.

#### Tier A and Tier B

- **Tier A.** Ring, eight stages, loopback `s7 -> s0`. Pass.
- **Binding sets.** 4x2 binds only to M0 (M1 has two rows). 1x2 binds only to M1; on M0 it
  is not a whole number of 4x2 slices.
- **Projections.** There is exactly one: s0..s3 on M0 and s4..s7 on M1. Capacity is 4 of 4
  slices on each mesh. It crosses twice (`s3 -> s4`, `s7 -> s0`), within the 2 channels.
- **Segment carve.** M0's segment must visit all four slices of a line, so it enters and
  leaves at the end slices A0 and A3. M1's segment is a path through all four slices of a
  4-cycle.
- **Result.** Pass, with `k(M0, M1) = 2`. The boundary slices are {A0, A3} on M0 and any
  slice on M1.

#### SAT placement

M0's pool is {P}; M1's pool is {U, V}. The seat graph has P–U x2 and P–V x2, and the
footprint-level Plan 1 keeps both. The slice-level rule would use the boundary slices, but
needs a hard-pinned layout. The master solve offers (P, U), then (P, V).

#### Seating (P, U)

After intra-mesh, the cables land on M0 A3 to M1 (0,1), which is in T0, and M0 A0 to M1
(1,3), which is in T3.

***New:* Tier C.**

1. **Planned carve.** It is rejected up front because the host slices are not uniform (4x2
   on M0, 1x2 on M1). This path does not apply.
2. **Per-rank tiling.** The only tile shapes that divide each mesh are 4x2 for M0 and 1x2
   for M1. Each tile is exactly one rank's slice, so each rank keeps its own and nothing
   collapses. That gives 8 submeshes: A0..A3 with 8 chips each, T0..T3 with 2 chips each.
3. **Resolver.**
   - **Chip counts.** These confine s0..s3 to the A submeshes and s4..s7 to the T
     submeshes.
   - **Capacity.** Submeshes of at least 8 chips get 1 slot per chip, smaller ones 2. So s0
     needs 4 of A's 8 slots, and each T submesh has 4 slots.
   - **M0.** s0..s3 must run along the line, A0 to A3 or A3 to A0. So `s3 -> s4` leaves
     from A3 (to T0) or from A0 (to T3).
   - **M1.** s4..s7 must visit all of T0..T3. In a 4-cycle, a path through every vertex
     ends next to where it started. Starting at T0, it ends at T1 or T2, but `s7 -> s0`
     needs the cable on T3. Starting at T3, it ends at T1 or T2, but the cable back is on
     T0.
   - The resolver reports `no valid submesh assignment found`.
4. **Result.** Reject, and `exclude_mapping({M0 -> P, M1 -> U})`.

The count of cables is right and both land on boundary slices on M0. What fails is that
on M1 they land on opposite corners. Only the layout shows that, which is why Plan 1 cannot
prune it without hard pins.

#### Seating (P, V)

The cables land on A3 to M1 (0,3), which is in T1, and A0 to M1 (0,0), which is in T0.

| Mapping | M0 | M1 | Who finds it |
|---|---|---|---|
| 1 | s0 A0, s1 A1, s2 A2, s3 A3 | s4 T1, s5 T3, s6 T2, s7 T0; loopback T0 to A0 | the resolver's first try (s0 on submesh 0) |
| 2 | s0 A3, s1 A2, s2 A1, s3 A0 | s4 T0, s5 T2, s6 T3, s7 T1; loopback T1 to A3 | valid; found if mapping 1 failed |
| impossible | a 4x2 stage on a T submesh, or a 1x2 stage on an A submesh | | chip-count filter (8 against 2) |

Binding puts both cables in `R`. Emit `TT_PIPELINE_STAGE` s0..s3 for ranks 0–3 and the M1
stages for the ranks that own T0..T3.

At runtime, blaze is called with `submesh_shape` from the harness (4x2 on M0's ranks, 1x2 on
M1's). It tiles, and each rank keeps the tile whose `(0,0)` chip is local. It allgathers
and resolves, reaching the same answer.

**A variation that Tier B stops.** If M1 were 4x8 with `host_topology [1, 4]` (4x2 slices)
and still carried 1x2 stages, Tier B as written rejects at load, because 1x2 is not a whole
slice. Blaze's per-rank tiling would build it. Each rank owns four 1x2 tiles and keeps the
first with a local `(0,0)` chip (row 0), and the other three drop out. Then only cables
landing in row 0 count. This is the edge case in 3.4: if runtime relies on it, the Tier B
binding rule must allow it for the tiling path.

### 8.4 Special cases these examples hit

| Case | Where in the code | What happens |
|---|---|---|
| Stage 0 wider than one slice | `plan_ring_carve`, `stage_counts[0] != 1` | `input-infeasible`: stage 0 owns host-to-device, the loopback entry and device-to-host, so one rank must drive it |
| Stage not a whole number of slices | `plan_ring_carve`, `r % nrows` or `c % ncols` | the planned carve rejects; only per-rank tiling can build it, and Tier B rejects it as written (8.3 variation) |
| Host slices of different shapes | `plan_ring_carve` uniform check | the planned carve rejects; per-rank tiling is the path (8.3) |
| Mixed stage shapes on per-rank tiling | one tile shape per mesh; resolver chip-count filter | no submesh of the needed size (8.2, step 4) |
| Wide stage across a mesh crossing | `_as_rectangle`, more than one mesh | that rotation is skipped; another may work (carver example A) |
| Loop of the wrong length | `_cycles_no_branch`, `_cycles_of_length` | `no-cycle-of-length-N` with achievable lengths, which list only lengths up to N when there are branch slices (carver example B) |
| Right length, no alignment | `_group_cycle` returns nothing | `no-alignment` with same-mesh run lengths (carver example C) |
| Unused slices | filler slots | their ranks idle; the plan still covers every chip |
| Two ranks carve the same chips | `_collect_submeshes` collapse | one co-owned submesh, one node (8.2, step 6) |
| Cable on a tile nobody keeps | `discover_submesh_links` sees only submesh chips | the cable does not count; Tier C applies the same rule (8.3 variation) |
| Hamiltonian path endpoints | resolver over per-rank tiles | the right cable count on the wrong slices fails (8.3, seating (P, U)) |
| Fork router degree | `plan_fork_carve` candidates | each fabric-return branch takes two router neighbours, each host-return branch one (carver example D) |
| Search budget hit | `_MAX_SEARCH_EXPANSIONS = 200,000` | `budget-exhausted`; Tier C treats it as unknown and accepts (3.2) |
| Pipeline failure feedback | `exclude_mapping`, never `add_forbidden_constraint` | the seat set is blocked, not the mesh on its seat; M0 on P still succeeds in 8.2, step 6 |
