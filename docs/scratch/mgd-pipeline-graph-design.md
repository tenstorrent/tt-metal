<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

SPDX-License-Identifier: Apache-2.0
-->

# Encoding pipeline-stage adjacency into MGD

Brainstorm / design notes for [tt-metal#52016](https://github.com/tenstorrent/tt-metal/issues/52016).

Each section ends with examples of what that step checks, including its edge cases. They
share the running examples defined at the start of section 2.

A visual version with diagrams, the same examples and the full per-PR implementation
plan is in [mgd-pipeline-graph-design.html](mgd-pipeline-graph-design.html). Download it and
open it in a browser; GitHub shows HTML files as source. Its source is the Cursor canvas
[mgd-pipeline-graph-design.canvas.tsx](mgd-pipeline-graph-design.canvas.tsx), which opens
as a live canvas in Cursor.

**Status:** design. Nothing below is implemented yet except Tier D (2.4), which already
runs in the pipeline builder and is not changed. Rewritten after the SAT joint-placement
rework of the topology mapper landed; the Tier C plan (2.3, 4, 5) follows that code.

### Scope: STRICT first, RELAXED later

> **Decision (fixed for the first version):** every pipeline edge is **STRICT**. The first
> version has no edge `policy` field, so a pipeline edge always rides a fabric link. RELAXED
> edges (host MPI) are a separate, later piece of work. Keep this split until it is decided
> otherwise.

The RELAXED design stays in this doc so that it can be built later. Every part of it is
tagged **RELAXED (later)** where it appears. Everything not tagged is needed for STRICT.

Two things are easy to mix up:

- **Edge policy** (`policy` on a *pipeline edge*, 1.4) is what is deferred.
- **Channel policy** (`channels { count: 2 policy: RELAXED }` on a *mesh or connection*) is
  an existing MGD feature. It stays in the first version unchanged. It says how many cables
  the control plane binds, not whether a pipeline edge may go over host.

| Piece | STRICT (first version) | RELAXED (later) |
|---|---|---|
| Schema (1.3) | `PipelineEdge` has `src` and `dst`; field 3 is reserved | add `EdgePolicy` and `policy = 3`, STRICT = 0, so first-version files keep their meaning |
| Tier A (2.1) | ids, edges, shapes, names | nothing new: RELAXED is allowed on any edge |
| Tier B (2.2) | every edge counts against `cap(A, B)`; every edge inside a mesh needs touching rectangles | RELAXED crossings and RELAXED edges inside a mesh are skipped |
| Fork router degree (2.2) | 2 per return, 1 per dead-end branch | 1 per RELAXED return |
| Tier C (2.3, 3.4) | planned carve, per-rank tiling, single-mesh tiling, resolver | host-cut path: port `plan_ring_placement` and `plan_fork_placement` |
| Plan 1 `k_AB` (4.4) | every crossing counts | RELAXED crossings do not count |
| blaze | nothing new | `Edge` gets a policy; `--pipeline-builder flexible` marks every edge RELAXED |
| Models | rings, `llama_8b_pod`, joint stages, forks whose returns are all on fabric | forks with a host return (dflash draft), rings with a host loopback |

**PR plan** (in the canvas). Milestones M1 to M4 (P1 to P10, P12, P13, P15) are STRICT only and
need nothing from RELAXED. M5, RELAXED (later), holds two PRs:

1. **P14, schema + Tier B + Plan 1.** Add `EdgePolicy` and un-reserve field 3. Tier B and
   `k_AB` skip RELAXED edges. Tier C rejects a pipeline with a RELAXED edge
   (`pipeline_relaxed_unsupported`) instead of accepting it unchecked.
2. **P11, host cuts in Tier C.** Port the host-cut planner, resolve each fabric chain on its
   own, and remove P14's reject guard.

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
   C++, and take the inter-mesh candidate table from the inter-mesh policy plan.
4. **Pipeline mapping in the topology mapper (Tier C)** - section 4, in four steps after
   intra-mesh completion: build the physical pipeline graph from host slices and carve paths
   (C1, 4.1), solve the pipeline onto it (C2, 4.2), on success emit and publish the cables to
   bind (C3, 4.3), on failure retry exit nodes and then exclude the seating (C4, 4.4). The
   per-seating check with feedback lands first; pruning inside the solves is an optional
   speed-up, added later only if measurements call for it.
5. **Stage-to-rank mapping** - section 5.

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
connect. It sits in the same file as the mesh it runs on. Here is the whole MGD for GPT-OSS
on one galaxy: the mesh part is blaze's existing
`single_bh_galaxy_4x2_mesh_graph_descriptor.textproto`, unchanged, and the pipeline part is
new.

```textproto
# --- Existing: the mesh -----------------------------------------------------
mesh_descriptors {
  name: "M0"
  arch: BLACKHOLE
  device_topology { dims: [ 8, 4 ] }   # chip grid: 8 rows x 4 columns = 32 chips
  host_topology   { dims: [ 2, 2 ] }   # hosts in a 2x2 grid = 4 ranks; each owns
                                       #   (8 / 2) x (4 / 2) = a 4x2 slice of chips
  channels { count: 2 policy: RELAXED }
}
top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }   # one mesh, mesh_id 0

# --- New: the pipeline ------------------------------------------------------
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

The slice shape is never written down. It follows from the two `dims` lines: the host grid
divides the chip grid, rows by rows and columns by columns:

```
              cols 0-1    cols 2-3
  rows 0-3    rank 0      rank 1        device_topology [8, 4] / host_topology [2, 2]
  rows 4-7    rank 2      rank 3        = four 4x2 slices, one per rank
```

Each stage here is `[4, 2]`, exactly one slice, so each runs on one rank and the four
stages use all four ranks.

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

A router with a target branch and a draft branch, both returning to the router, mirroring
`build_gpt_oss_dflash_router_pipeline_graph`. In the first version both returns are STRICT.
**RELAXED (later):** the draft return may be marked `policy: RELAXED` to go over host MPI, as
in the comment on its last edge.

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
  edges { src: "draft10"  dst: "router" }   # RELAXED (later): policy: RELAXED to return over host
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
  reserved 3;                // edge policy, RELAXED (later); every edge is STRICT for now
}
```

`shape` is rank 2 because the resolver models a chip as `(row, col)`. Every edge must ride
the fabric. Edges carry no loopback marking and no channel count (2.1). A file that writes
`policy` on an edge fails to parse, because the field does not exist yet.

**RELAXED (later)** adds this, as field 3 with STRICT as value 0, so an edge without
`policy` keeps meaning STRICT and first-version files are unchanged:

```proto
enum EdgePolicy {
  STRICT = 0;     // must ride a fabric link
  RELAXED = 1;    // fabric if the placement allows; otherwise host MPI (D2H -> MPI -> H2D)
}

message PipelineEdge {
  string src = 1;
  string dst = 2;
  EdgePolicy policy = 3;     // default STRICT (1.4)
}
```

None of the comments above are enforced by protobuf. They are obligations on
`validate_pipeline_descriptors`.

### 1.4 Edge policy: STRICT or RELAXED, per edge - RELAXED (later)

> **RELAXED (later).** Not in the first version: every edge is STRICT and there is no
> `policy` field (see "Scope" at the top). This section is the design for when it is added.

Each edge says whether it must ride the fabric. The words match `channels { policy }`:
STRICT is a requirement, RELAXED a preference. **STRICT is the default:** it is enum value
0, so an edge with no `policy` is STRICT, and a pipeline goes over host only where an edge
opts in with `policy: RELAXED`.

| Policy | Meaning | What the mapper requires |
|---|---|---|
| `STRICT` (default) | carried over a fabric link | a fabric link between the two stages' submeshes. If they are on different meshes, the edge rides a declared connection and counts against its channels |
| `RELAXED` | carried over fabric where the placement allows, otherwise over host MPI | nothing: the edge never needs a link |

```textproto
edges { src: "s0" dst: "s1" }                    # STRICT
edges { src: "s1" dst: "s2" }                    # STRICT
edges { src: "s2" dst: "s3" }                    # STRICT
edges { src: "s3" dst: "s0" policy: RELAXED }    # the loopback may go over host
```

- **There is no HOST value.** An edge the builder always sends over host needs no link, so
  to the mapper it is RELAXED. Whether a RELAXED edge ends up on fabric or host is decided
  when the placement is built, not in the descriptor.
- **The two whole-pipeline modes are special cases.** Blaze picks one mode per run today:
  `build_pipeline(placement="strict" | "flexible")`, set by `--pipeline-builder`. Every edge
  STRICT is `"strict"`. Every edge RELAXED is `"flexible"`, which may cut any edge except
  the two below. A mix is what neither mode can say: for example a STRICT ring whose only
  RELAXED edge is the loopback.
- **The builder reads the same policy.** Each blaze `Edge` carries its policy, and the
  host-cut planner may cut only RELAXED edges. `--pipeline-builder flexible` stays as a
  shortcut that marks every edge RELAXED. The MGD and the runtime graph then agree on which
  edges may go over host, so Tier C checks what the builder will build.
- **RELAXED is allowed on every edge.** An MGD stage does not have to map onto a blaze stage
  yet, so the MGD does not restrict where RELAXED goes. Tier B and Plan 1 leave every RELAXED
  crossing out. Tier C mirrors blaze's host-cut planner, which never cuts these edges, so a
  RELAXED one still needs a fabric link there:
  - *stage 0's outgoing edge* in a ring. `_group_walk` (`host_edge_planner.py`) skips any
    alignment with `0 in cuts`: stage 0's rank drives inference and cannot run a host-hop pump
    (`test_pipeline_builder_flexible_mode.py`);
  - *the router's edge into each branch* of a fork. The router sends to a branch through its
    `d2d_exchange`, and a host hop out of the router has no wiring: `_find_branches`
    (`pipeline_builder/fork.py`) asserts "a branch head must be a fabric neighbour of the
    router". A branch's *return* to the router may go over host.

  > **Note: rules that might be good to add.** Once MGD stages map onto blaze stages, Tier A
  > could reject RELAXED where blaze can never carry the edge over host: stage 0's outgoing
  > edge, a router's edge into a branch, and any edge of an open chain (blaze raises
  > `NotImplementedError("host edges are only supported on a looped-back ring")`,
  > `graph.py`). Tier B and Plan 1 would then count those edges as STRICT, which prunes more.

What each step does with the policy:

| Step | STRICT edge | RELAXED edge |
|---|---|---|
| Tier A (2.1) | - | - |
| Tier B crossing rules (2.2) | a crossing must ride a declared connection and fit its channels | left out |
| Plan 1 `k_AB` (4.4) | counted | not counted |
| Tier C (2.3, 3.4) | must be a fabric link in the layout | the host-cut planner may cut it; each fabric chain is then resolved on its own |
| Fork router degree (2.2, 3.2) | a STRICT return needs a cycle through the router: 2 router neighbours | a RELAXED return or a dead end needs an open path: 1 router neighbour |

A pipeline whose edges are all STRICT takes the carve paths of 3.4 unchanged; that is every
pipeline in the first version. One with a RELAXED edge also gets the host-cut path (3.4,
"Host cuts on RELAXED edges").

### 1.5 Merging MGDs

The topology mapper never reads a part MGD directly. `MultiMeshSolutionEnumerator::init_from_parts`
(`topology_mapper_utils.cpp`) always calls `MeshGraphDescriptor::merge` on its parts and builds
the SAT problem from the result. Tier C and the `unique_shapes` gate (4.4) see the merged MGD.
Tier B has already run on each part, because constructing a part is constructing an MGD, and it
runs again because `merge` constructs a new one.

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

**Several pipelines: checked together, as one graph.** An MGD may declare more than one
pipeline, and merging parts that each declare one gives several. The parts' pipelines are
disconnected from each other, because no part can name another part's stages. Rather than
check them one at a time, the mapper combines them into **one pipeline graph whose
components are the pipelines**, and sets every constraint on that graph in one go:

- **Names.** Pipeline names must be unique across the merged MGD; a clash is a merge error
  that names both parts. In the combined graph a stage is `<pipeline>/<stage id>`, so stage
  ids only need to be unique within their pipeline.
- **Tier A** checks each pipeline on its own: each is linear or one fork, with its own
  stage 0 or router (2.1).
- **Tier B** counts capacity over all pipelines together, and a projection places every
  stage of every pipeline at once. The pipelines compete for the same slices, so two that
  each fit alone can fail together (2.2).
- **Plan 1** takes `k_AB` over the combined projections, so it sums the STRICT crossings of
  every pipeline (4.4).
- **Tier C** places all components on disjoint slices of one seating, and resolves each
  component as its own graph (3.4, "Several pipelines").

A part's pipeline still binds across the whole merged fabric, not just to that part's
meshes. Byte-identical copies of one pipeline (the same file in two parts) clash on the name
and are rejected; relax that only if a real setup needs it.

**Validation order.** Every `MeshGraphDescriptor` constructor runs Tier A (in `static_validate`)
and Tier B (after `populate`). Loading a part therefore checks that part against its own meshes,
before merge. `merge` returns a new `MeshGraphDescriptor`, so both tiers run again on the merged
meshes and every pipeline together. A pipeline that fits its own part can still fail after merge,
when the pipelines share the combined slices. A pipeline that does not fit the meshes in its own
file fails when that file is loaded. `init_from_parts` reads the facts stored on the merged
descriptor; it does not run Tier B itself, and neither does `MeshGraph`.

**Tests.**
- Two parts, a pipeline in one: present after merge, unchanged, and Tier C runs.
- Pipelines with different names in two parts: both present after merge, and Tier C checks
  them together.
- Pipelines with the same name in two parts: merge fails, naming both parts.
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

#### Running examples

Each section below ends with examples of what that step checks, including its edge cases.
They are hypothetical, traced by hand through the code as it stands (tt-blaze `origin/main`
34e783a631, tt-metal on this branch); nothing was run. Most of them use the setup below.

**Slices.** A 4x8 mesh with `host_topology [1, 4]` has four 4x2 host slices, one per rank.
On M0 they are A0..A3 (ranks 0–3, columns 0–1, 2–3, 4–5, 6–7); on M1 they are B0..B3
(ranks 4–7). Inside a mesh the slices form a line, A0–A1–A2–A3. Cables between meshes are
drawn as vertical bars:

```
   A0 ── A1 ── A2 ── A3        M0
   │            │
   B0 ── B1 ── B2 ── B3        M1      (here: cables A0-B0 and A2-B2)
```

**`mixed4`: stages that span two ranks.** Two meshes of type `M`, M0 and M1, joined with 2
channels: 8 slices in all. The ring has two 4x4 stages, which need two neighbouring slices
each, so it needs 6 of the 8 and leaves two ranks idle:

```protobuf
mesh_descriptors {
  name: "M"  arch: BLACKHOLE
  device_topology { dims: [ 4, 8 ] }
  host_topology   { dims: [ 1, 4 ] }   # 4 ranks per mesh, each a 4x2 slice
  channels { count: 2 policy: RELAXED }
}

graph_descriptors {
  name: "G"  type: "FABRIC"
  instances { mesh { mesh_descriptor: "M" mesh_id: 0 } }   # M0: slices A0..A3, ranks 0-3
  instances { mesh { mesh_descriptor: "M" mesh_id: 1 } }   # M1: slices B0..B3, ranks 4-7
  graph_topology { layout_type: ALL_TO_ALL channels { count: 2 policy: RELAXED } }
}
top_level_instance { graph { graph_descriptor: "G" graph_id: 0 } }

# 2 meshes x 4 slices = 8 slices; the ring below needs 1 + 2 + 2 + 1 = 6
pipeline_descriptors {
  name: "mixed4"
  stages { id: "s0" shape { dims: [ 4, 2 ] } }   # 1 slice
  stages { id: "s1" shape { dims: [ 4, 4 ] } }   # 2 slices, two ranks
  stages { id: "s2" shape { dims: [ 4, 4 ] } }   # 2 slices, two ranks
  stages { id: "s3" shape { dims: [ 4, 2 ] } }   # 1 slice
  edges { src: "s0" dst: "s1" }  edges { src: "s1" dst: "s2" }
  edges { src: "s2" dst: "s3" }  edges { src: "s3" dst: "s0" }
}
```

**`pod8`: different slice shapes.** The `llama_8b_pod` pattern at small scale. Two meshes
of different shape and rank grid, joined with 2 channels, and one ring of eight stages.

```protobuf
mesh_descriptors {
  name: "Big"  arch: BLACKHOLE
  device_topology { dims: [ 4, 8 ] }
  host_topology   { dims: [ 1, 4 ] }   # 4 ranks, each a 4x2 slice
  channels { count: 2 policy: RELAXED }
}

mesh_descriptors {
  name: "Small"  arch: BLACKHOLE
  device_topology { dims: [ 2, 4 ] }
  host_topology   { dims: [ 2, 2 ] }   # 4 ranks, each a 1x2 slice
  channels { count: 2 policy: RELAXED }
}

graph_descriptors {
  name: "G"  type: "FABRIC"
  instances { mesh { mesh_descriptor: "Big"   mesh_id: 0 } }   # M0: slices A0..A3, ranks 0-3
  instances { mesh { mesh_descriptor: "Small" mesh_id: 1 } }   # M1: slices T0..T3, ranks 4-7
  graph_topology { layout_type: ALL_TO_ALL channels { count: 2 policy: RELAXED } }
}
top_level_instance { graph { graph_descriptor: "G" graph_id: 0 } }

# 2 meshes x 4 slices = 8 slices; the ring below needs 8 x 1 = 8
pipeline_descriptors {
  name: "pod8"
  stages { id: "s0" shape { dims: [ 4, 2 ] } }   # 1 slice of Big
  stages { id: "s1" shape { dims: [ 4, 2 ] } }
  stages { id: "s2" shape { dims: [ 4, 2 ] } }
  stages { id: "s3" shape { dims: [ 4, 2 ] } }
  stages { id: "s4" shape { dims: [ 1, 2 ] } }   # 1 slice of Small
  stages { id: "s5" shape { dims: [ 1, 2 ] } }
  stages { id: "s6" shape { dims: [ 1, 2 ] } }
  stages { id: "s7" shape { dims: [ 1, 2 ] } }
  edges { src: "s0" dst: "s1" }  edges { src: "s1" dst: "s2" }
  edges { src: "s2" dst: "s3" }  edges { src: "s3" dst: "s4" }
  edges { src: "s4" dst: "s5" }  edges { src: "s5" dst: "s6" }
  edges { src: "s6" dst: "s7" }  edges { src: "s7" dst: "s0" }
}
```

A slice is `device_topology` divided by `host_topology`. M0 is 4×8 / 1×4 = 4×2, and M1 is
2×4 / 2×2 = 1×2. Every chip is a dot below; each box is one rank's slice:

```
M0 (4x8), host_topology [1, 4]           M1 (2x4), host_topology [2, 2]
four 4x2 slices, one row of ranks        four 1x2 slices, a 2x2 grid of ranks

  cols 0-1  2-3   4-5   6-7                cols 0-1  2-3
 ┌─────┬─────┬─────┬─────┐               ┌─────┬─────┐
 │ . . │ . . │ . . │ . . │  row 0        │ . . │ . . │  row 0   T0 (r4)  T1 (r5)
 │ . . │ . . │ . . │ . . │  row 1        ├─────┼─────┤
 │ . . │ . . │ . . │ . . │  row 2        │ . . │ . . │  row 1   T2 (r6)  T3 (r7)
 │ . . │ . . │ . . │ . . │  row 3        └─────┴─────┘
 └─────┴─────┴─────┴─────┘
   A0    A1    A2    A3                   T0 ── T1
  (r0)  (r1)  (r2)  (r3)                  │      │
                                          T2 ── T3
 A0 ── A1 ── A2 ── A3   (a line)          (a 4-cycle)
```

A `[4, 2]` stage is exactly one A slice, and a `[1, 2]` stage is exactly one T slice. M1 has
only 2 rows, so `[4, 2]` cannot bind there. On M0, `[1, 2]` is a quarter of a 4×2 slice, so it
is not a whole number of slices and cannot bind there either. That leaves one projection:
s0..s3 on A0..A3, s4..s7 on T0..T3. It uses all 8 slices and crosses twice, `s3 -> s4` and
`s7 -> s0`, within the 2 channels.

**The machine.** Three 32-chip pods P, Q and R, each a 4x8 grid driven by four hosts h0..h3
(one 4x2 block each), plus two 2x4 boards U and V of four 2-chip hosts. Single cables:

```
P.h0 ── Q.h0   P.h3 ── Q.h3      P–Q: 2 links
P.h0 ── R.h0   P.h2 ── R.h2      P–R: 2 links
Q.h1 ── R.h1                     Q–R: 1 link
P.h0, P.h3 ── two chips of U     P–U: 2 links
P.h0, P.h3 ── two chips of V     P–V: 2 links
```

Assume the intra-mesh solve puts mesh host rank i on host hi of its seat (the PGD's
preferred layout). Then with M0 on P and M1 on Q, the cables touch A0–B0 and A3–B3; with M1
on R they touch A0–B0 and A2–B2.

### How Tier A and Tier B differ

Both run inside every `MeshGraphDescriptor` constructor. Tier A asks whether the pipeline text is a valid descriptor. Tier B asks whether those stages can fit on the meshes that same descriptor declares. Tier A never looks at a mesh. Tier B never looks at the physical machine, and it does not choose which mesh a stage lands on.

**Tier A** runs in `static_validate`, on the proto alone.

- A stage id is unique and non-empty, and an edge names stages that exist.
- An edge does not point at its own stage.
- A shape is two positive numbers. Pipeline names are unique.
- Writing the same edge twice means two channels on that hop. Every edge is STRICT (RELAXED is later).
- How the stages connect is not limited. Two stages with in-degree 0, a stage with three outgoing edges, and a split inside a branch all pass. The descriptor does not yet say which stage is the router, so those limits are notes for when it does.

**Tier B** runs after `populate`, once the meshes, their `host_topology` rank grids, and the declared connections exist. `host_topology.dims` is the rank grid, and the slice is `device_topology` divided by that grid. A 4×8 mesh with `host_topology [1, 4]` is four 4×2 slices, one rank each. It keeps every assignment that passes.

- A stage is a rectangle of those slices, in that orientation. On 4×2 slices, `[4, 2]` is one slice and `[4, 4]` is two. `[2, 4]` is not a whole number of 4×2 slices, and turning it sideways is not allowed, so its binding set is empty.
- All pipelines in this descriptor fit in the slices it declares. Two copies of `mixed4` need 12 slices where 8 are declared.
- Every hop that crosses meshes rides a declared connection, within that connection's channel count. RELAXED (later): a RELAXED crossing is left out of this count.

`mixed4` shows the split. Its stages and edges are well-formed, so Tier A passes. Tier B then tries assignments onto the two meshes. All four stages on M0 need 6 slices and M0 has 4, so that assignment is dropped. `s0, s1` on M0 and `s2, s3` on M1 uses 3 and 3 and crosses twice, which matches the 2 declared channels, so it survives. `s0, s2` on one mesh and `s1, s3` on the other crosses four times, so it is dropped. Several assignments survive. Tier B stops there.

Rows marked "pass here" are the same split. Tier A accepts a `[2, 4]` stage because the shape is two positive numbers. Tier B rejects it because no mesh's slice tiles that shape. Two pipelines with different names each pass Tier A. Tier B adds their slice counts on the descriptor that contains both.

Loading a part runs both tiers on that part. `merge` builds a new descriptor, so both run again. The second Tier B is the one that can reject two pipelines which each fit their own file but do not fit together.

### 2.1 Tier A - descriptor only

**When it runs:** inside every `MeshGraphDescriptor` constructor, in `static_validate`, before `populate` and before Tier B.

- **Available:** the proto only. No meshes, no `host_topology`, no connections.
- **Site:** a new `validate_pipeline_descriptors`, next to the existing validators.
- **On failure:** the constructor fails.

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

**Field-level obligations** are the schema comments in 1.3. Each pipeline is checked on its
own here; when an MGD declares several, the later tiers check them together as one graph
(1.5). Every stage is passed to the resolver as an explicit node, so a stage no edge
mentions is kept, not silently dropped.

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
- **Stage topology is not limited yet.** In-degree, out-degree, how many stages have
  in-degree 0, and whether a branch is a chain are all accepted. The descriptor does not
  identify which stage is the router. Once it does, the graph can be constrained to blaze's
  shape: one router, each branch a chain, and at most two branches (`_find_branches` builds
  exactly two; `fork_structure` accepts any number). Until then a graph blaze cannot build
  still passes Tier A and fails at build time. A branch may dead-end; blaze's
  `fork_structure` and `_find_branches` reject a branch with no return today.
- **Every edge is STRICT.** There is no `policy` field yet, so Tier A has nothing to check
  for it. RELAXED (later): any edge may be RELAXED, and Tier A will not restrict where; see
  the note in 1.4 for rules that might be good to add.

#### Examples

| Descriptor | Tier A result | Why |
|---|---|---|
| `mixed4` | pass | ids unique, edges name declared stages, every stage has out-degree 1, one cycle (`s3 -> s0` becomes its loopback), shapes positive |
| `pod8` | pass | a ring of 8; the loopback is `s7 -> s0` |
| `mixed4` plus `edges { src: "s1" dst: "s1" }` | reject | self-edge |
| `mixed4` with `s0 -> s1` repeated | pass | two channels on that hop; the builder reads one link per edge, which is a builder limit |
| dflash: `router -> target0 .. target37 -> router`, `router -> draft0 .. draft10 -> router` | pass | one router, two chain branches, both return to it on fabric |
| edges `s0 -> s1`, `s0 -> s2`, `s1 -> s0`; `s2` has no out-edge | pass | a fork whose branch `s2` dead-ends, which is allowed. Blaze rejects it today in `fork_structure` (TODO above) |
| a router with three branches | pass | Tier A does not limit stage topology. Blaze's fork runtime still builds exactly two |
| a branch stage with out-degree 2 (a fork inside a branch) | pass | Tier A does not limit stage topology. Once the MGD names the router, only that stage may split |
| two stages with in-degree 0 | pass | Tier A does not limit stage topology. Once the MGD names the router, a fork has exactly one |
| `mixed4` with `policy: RELAXED` on `s3 -> s0` | reject | no `policy` field in the first version; the textproto does not parse |
| `mixed4` plus a copy named `mixed4_b` | pass here | each is checked on its own; Tier B checks them together and rejects on capacity (1.5) |
| `mixed4` plus a copy also named `mixed4` | reject | pipeline names must be unique (1.5) |
| a stage declared `[2, 4]` where every mesh has 4x2 slices | pass here | shapes are logical and never transposed, so Tier B finds no mesh it binds to |

**RELAXED (later), Tier A examples.**

| Descriptor | Tier A result | Why |
|---|---|---|
| dflash with the draft return RELAXED | pass | any edge may be RELAXED |
| `mixed4` with `s3 -> s0` RELAXED | pass | any edge may be RELAXED |
| `mixed4` with `s0 -> s1` RELAXED | pass | allowed; Tier C's planner still never cuts stage 0's outgoing edge (1.4) |
| dflash with `router -> draft0` RELAXED | pass | allowed; Tier C still needs a fabric link from the router to a branch head (1.4) |

### 2.2 Tier B - descriptor plus the declared fabric

**When it runs:** at the end of every `MeshGraphDescriptor` constructor, after `populate`,
before any hardware is looked at and before `MeshGraph` exists.

`host_topology.dims` is already the rank grid. `validate_mesh_topology` requires each
`device_topology` dim to be divisible by the matching host dim, and `get_declared_topology`
exposes both as `DeclaredTopology::dims` and `host_dims`. The slice is the elementwise
quotient. Tier B does not need `MeshGraph` to know how many ranks a mesh has.

- **Each part, before merge:** the file and string constructors delegate to the proto
  constructor, so a part is checked against the meshes it declares.
- **After merge:** `MeshGraphDescriptor::merge` returns `MeshGraphDescriptor(merged_proto)`,
  including the one-part copy. That construction checks the merged meshes and every pipeline
  together.
- **Available:** mesh instances, `device_topology`, `host_topology`, declared `connections`
  with channel counts. **Not** the PSD.
- **On failure:** the constructor fails. `init_from_parts` only reads the facts the merged
  descriptor stored.

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
- **No stage is required to be one slice.** The descriptor does not say which stage owns
  host-to-device, so Tier B cannot know which stage, if any, a single rank must drive.
  Blaze's carve planner still rejects a multi-rank stage 0 at build time.
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

- **Every crossing rides a declared connection.** For each pipeline edge whose stages land
  on different meshes A and B, the MGD declares a connection between A and B.
- **Crossings fit the channels.** The number of pipeline edges crossing A–B is at most that
  connection's channel count.

Extra connections the pipeline does not use are fine. Tier C tightens "declared" to
"actually bound by the control plane" (2.3).

**RELAXED (later): RELAXED crossings are left out of both rules** (1.4). The builder may
carry them over host, so a missing or too-small connection is not a reason to reject them. A
pipeline whose edges are all RELAXED is checked only on binding sets and capacity.

Zero projections is a reject. Tier B only has to show that **one** projection survives, so
the search stops at the first. It does not store projections and does not choose one, and a
projection is not a slice placement. Tier C does not filter by projection either: any carve
Tier C finds on real slices already passes these rules (each stage sits on whole slices of a
mesh in its binding set, each crossing rides a bound link and bound links never exceed the
declared count, and each mesh's stages were carved), so a stored list could never reject
anything. A split that breaks the rules (for example `mixed4` with `s0, s2` on M0 and
`s1, s3` on M1) cannot be carved by Tier C anyway. The facts Plan 1 (4.4) reads, such as the
minimum crossing count, need every projection; they come from an optional second run of the
same search (below).

```
subtorus_4x4_pipeline_8stage: 8 meshes, connections 0-1-2-...-7-0, 8 stages of 4x4

  s_i on mesh i, crossings s_i -> s_{i+1} ride connection (i, i+1)   OK

same MGD, connections 0-1-2-...-7 (no 7-0)                           REJECT
  the loopback crossing has no declared connection
```

#### Each mesh's segment can be carved

A projection survives only if, for each mesh in it, the stages it assigns to that mesh can be
carved on the mesh's logical slices. Reject only when no projection survives. The check
depends only on the MGD, not on seats: if it fails, it fails on every seating, so it belongs
here and not in the SAT placement, where it could prune nothing and the mapper would try
every seating before failing with a generic error. This is a
per-segment search with open crossing endpoints, which the runtime carve planner does not
have (it searches whole cycles), so it is new code built from the planner's pieces rather
than a straight port.

Slice adjacency comes from `device_topology` dims and RING wraps. Crossing endpoints are left
open, because which slice a crossing touches is not known until seating. This checks that
the stages assigned to the mesh can be cut into whole-slice rectangles along a path or cycle,
including joint stages and, for a fork, the router's degree: 2 neighbours per return plus 1
per dead-end branch. RELAXED (later): a RELAXED return counts 1, like a dead end.

#### How the connectivity check runs (plan)

Tier B never looks at chips or cables. It works on two small graphs the MGD already holds:
the **mesh graph** (meshes, with a channel count per mesh pair) and, inside each mesh, the
**slice grid** (the `host_topology` grid, with RING wraps). One search covers every pipeline
in the descriptor at once, because they share slices and channels.

**Inputs, built once per constructor.**

| Input | Where it comes from | Notes |
|---|---|---|
| slices of mesh `m` | `get_declared_topology(m)`: `dims / host_dims`, count `prod(host_dims)` | grid of slices, in rank order |
| slice adjacency in `m` | neighbours in the `host_dims` grid; a wrap on axis `i` when `ring_dims[i]` and `host_dims[i] > 2` | with `host_dims[i] == 2` the wrap is the edge already there |
| `cap(A, B)` | sum of `count` over FABRIC `ConnectionData` with `nodes = [A, B]` | `populate` stores each connection in both directions, so read one direction only. ALL_TO_ALL, RING and manual `connections` all land here |
| stage `s` | id, shape, and for each mesh `m` the rectangle `shape / slice(m)` in slices, or none | the binding set is the meshes with a rectangle |
| edge `e` | `src`, `dst`, multiplicity | a repeated edge counts once per copy (Tier A). RELAXED (later): plus its policy |

The channel `policy` of a connection (an existing MGD feature, not the deferred edge policy)
does not change `cap`. Under STRICT the control plane
binds exactly `count`; under RELAXED it binds up to `count` (`budget_channels` in
`control_plane.cpp`). Either way `count` is the most Tier B may assume. Binding fewer under
RELAXED is a hardware question for Tier C.

**Step 1: binding sets.** For every stage and mesh, the shape must divide by the slice on each
axis (no transpose) and fit inside `dims`. An empty set rejects, naming the stage.

**Step 2: capacity bound.** Sum each stage's smallest slice count over its binding set. If the
total over every pipeline exceeds the declared slices, reject before searching. This catches
`mixed4` + `mixed4_b` (12 > 8) and the fifth `[4, 6]` stage (9 > 8) without a search.

**Step 3: projection search** (depth-first, all pipelines together, stop at the first
survivor). It needs only the capacities and the mesh-to-mesh channel counts.

*Inputs, built once:* `slices[m]`; `cap[m][n]` as a small matrix; `need[s][m]`, the slice count
of stage `s` on mesh `m`, or none when `m` is not in its binding set (step 1); and `nbrs[s]`, a
list of (neighbour stage, edge copies) across every pipeline.

*State:* `assign[s]`, `used[m]` and `crossings[m][n]`, changed in place and undone on
backtrack, so no state is copied.

*Stage order (variables).* Start from the most constrained stage: fewest meshes in its
binding set, then most slices. Then go breadth-first along pipeline edges, so every stage
after the first has an assigned neighbour and the crossing check fires at once. Pipelines
with no edge between them are visited one after another; they interact only through `used`.

*Mesh order (values).* Meshes that already hold a neighbour of `s` first, since they add no
crossing; then the mesh with the most free slices.

*Assigning stage `s` to mesh `m` is allowed when:*

1. **Slices:** `used[m] + need[s][m] <= slices[m]`.
2. **Crossings:** for every edge copy between `s` and an assigned stage `t` on mesh
   `n != m`, `crossings[m][n] + 1 <= cap[m][n]`. If `cap[m][n] == 0` there is no declared
   connection, and the assignment fails at once.
3. **Symmetry:** see below.

*Look-ahead after each assignment (cheap, sound):*

- every unassigned stage still has a mesh in its binding set with room for it and, if it has
  assigned neighbours, enough channels left toward them;
- the smallest needs of the unassigned stages fit in the free slices left.

*Interchangeable meshes.* Two meshes are interchangeable when they have the same declared
topology (`dims`, `host_dims`, ring axes) and the same `cap` toward every other mesh, so
swapping them maps every projection to another one. Within such a class, an **empty** mesh may
be used only when it is the lowest-numbered empty mesh of its class. This is sound, because a
projection that uses a higher empty mesh has a mirror that uses the lower one. On ALL_TO_ALL
with 8 identical meshes and 8 one-slice stages, it turns 8! = 40320 orderings into one. A ring
of meshes `0-1-...-7-0` has no interchangeable pair (only rotations), but the crossing rule
already prunes it hard.

*Leaf.* When every stage has a mesh, run step 4 on each mesh that holds stages (memoised).
If any mesh fails, backtrack; at the first full success, Tier B passes.

```text
search(depth):
  if depth == |S|: return all(carve_ok(m, stages_on[m]) for m) ? FOUND : FAIL
  s = order[depth]
  for m in value_order(s):
    if !fits(s, m) or !channels_ok(s, m) or !symmetry_ok(m): continue
    apply(s, m)
    if lookahead_ok():
      r = search(depth + 1)
      if r != FAIL: return r
    undo(s, m)
  record_failure(s)            # deepest failure, for the error message
  return FAIL
```

RELAXED (later) adds one line: RELAXED edges are skipped in rule 2 and in the look-ahead, so
they never touch `crossings` or `cap`.

**Step 4: segment carve per mesh** (depth-first, stop at the first carve). For each mesh,
take the stages assigned to it and place each as a rectangle of slices, with no two
overlapping, so that every **edge between two stages on that mesh** joins rectangles that
touch (share a slice edge, wraps included). RELAXED (later): a RELAXED edge inside a mesh
needs no adjacency. An edge that crosses to another mesh puts no condition on the carve,
because which slice the cable lands on is not known until seating. Rectangles do not wrap,
as in blaze's `_as_rectangle`.

- *Representation.* A mesh has at most 32 slices in the repo's MGDs, so a placement is a
  `uint64_t` mask over the `host_dims` grid. For each stage, precompute every top-left
  position whose rectangle fits without wrapping, its mask, and its mask grown by one grid
  step (RING wraps included). Two rectangles touch when `grown(a) & b != 0`.
- *Order.* Largest stage first, then breadth-first along the edges inside the mesh.
- *A position is allowed when* `mask & occupied == 0` and it touches every placed neighbour.
  For a fork, the router's rectangle must still be able to reach its degree (2 neighbours per
  return plus 1 per dead-end branch; RELAXED (later): a RELAXED return counts 1).
- *Pruning.* The free slices must hold the stages left. When the grid has a flip symmetry
  (a flip maps every carve to another carve), the first stage is limited to one half of the
  grid.
- *Memo.* The result depends only on the mesh's grid and the stages on it, so it is memoised
  on `(mesh descriptor, sorted stage ids)` and reused across projections. A failure is
  memoised too.

**Step 5: result.** Pass at the first projection whose carves all succeed. Zero survivors
rejects, naming the deepest failure the search recorded (for example "s2: M0 full; M1
needs a 3rd M0–M1 channel, 2 declared", or "every split crosses M0–M1 on 4 edges, 2
channels declared"). Tier B stores no projection and no carve.

**No budget; fail fast.** Both searches run to a definite answer: pass or proved
impossible. There is no node budget and no UNDECIDED result (the same rule as Tier C, 4.2).
The one slow case is a fully used grid of one-slice stages that must close a ring, a
Hamiltonian-cycle search that is slow when none exists. Step 4 settles it before searching:
an odd grid holds no ring, and a mesh entered and left through fixed slices needs a
Hamiltonian path between them, which on a full rectangular grid has an exact O(1) test
(the Itai–Papadimitriou–Szwarcfiter conditions, 4.2).

**Cost.** Of the `host_topology` values in the repo's MGDs, about 200 are one slice (nothing
to search), about 60 are 2 to 4 slices, about 15 are 8 to 16, and 8 are 32 (4x8, 8x4, 2x16).
A stage with one mesh in its binding set (as in `llama_8b_pod`) makes no branch. Both
searches run in microseconds to milliseconds on all of them, and step 4 runs once per
distinct stage set on a mesh, not once per projection.

**Why not SAT.** CaDiCaL is already in the build, but the encoding costs more than the
search at these sizes: a variable per (stage, mesh), pseudo-Boolean sums for slice capacity,
per-edge crossing variables with an at-most-`cap` count per mesh pair, and for step 4 a
variable per (stage, position) with no-overlap and touching clauses. A SAT answer also
gives no readable reason. On the hard failing carves a plain encoding was no faster (4.2);
what makes them fail fast is the exact segment test, which works the same in front of
either solver.

**Optional: facts for Plan 1.** Plan 1 (4.4) needs facts that hold for **every** projection
and carve: the minimum crossings `k_AB` and the union of boundary slices. These need the
full set, not the first survivor, so they come from a second run of the same two searches in
"find all" mode: step 3 continues after each survivor, and step 4 enumerates every carve of a
stage set (memoised, recording for each slice whether some carve gives it to a stage with a
crossing). Interchangeable-mesh symmetry still applies; each found projection stands for its
class, and its mirrors are added by relabelling. This run has its own, smaller budget
(proposed: 10^5 nodes). Over the budget, Tier B saves the always-true defaults, `k = 0` and
every host rank as a boundary slice, which prune nothing. The first version may skip this
run entirely and save the defaults; the exact check (4.4) then catches what Plan 1 would
have pruned, at the cost of more seatings tried.

#### Output: mapping constraints for Tier C

Tier B already finds facts the mapper can use. It saves them on the descriptor as one
object, `PipelineMappingConstraints`. Helpers in the topology mapper turn that object into the
`MappingConstraints` the mapper already solves with. Every fact holds for **every**
surviving projection and every carve of it, so a constraint built from it never removes a seating
blaze could build on (the "no early elimination" note above).

**What Tier B saves** (MGD types only; the MGD library does not include `topology_solver.hpp`):

```cpp
// pipeline_feasibility.hpp
struct PipelineStageKey {
    std::string pipeline;   // PipelineDescriptor.name
    std::string stage;      // PipelineStage.id
    auto operator<=>(const PipelineStageKey&) const = default;
};

struct PipelineMappingConstraints {
    // Stage -> mesh.
    std::map<PipelineStageKey, std::set<MeshId>> binding_sets;          // step 1

    // The rest comes from the optional "find all" run. Over its budget, or when the run is
    // skipped, it holds the always-true defaults: k = 0, every host rank a boundary slice.
    bool enumerated = false;                                            // true: real values below

    // Mesh pairs, minimum over projections. Keys are ordered pairs (A, B), A != B.
    std::map<std::pair<MeshId, MeshId>, std::uint32_t> min_crossings;   // k_AB
    std::map<MeshId, std::uint32_t> min_crossings_out;                  // per mesh, all neighbours together

    // Slices, union over projections and carves: host ranks of A whose slice may hold a
    // stage with a crossing toward B. A host rank of A not listed never needs a cable to B.
    std::map<std::pair<MeshId, MeshId>, std::set<MeshHostRankId>> boundary_slices;
};

// MeshGraphDescriptor
const PipelineMappingConstraints* pipeline_mapping_constraints() const;   // null without pipelines
```

`min_crossings_out(m)` is the minimum over projections of the total crossings at `m`; it is
not the sum of the `k_AB`, because different projections may give the minimum for different
pairs. When the "find all" run is skipped or over its budget, Tier B cannot take a minimum.
It then saves `k = 0` and every host rank as a boundary slice. Those facts are always true,
and they prune nothing.

**Helpers that turn it into `MappingConstraints`** (new
`tt_metal/fabric/pipeline_mapping_constraints.{hpp,cpp}`, beside the topology mapper):

| Helper | Fills | Constraint added | Why it is safe |
|---|---|---|---|
| `stage_to_mesh_constraints(pmc)` | `MappingConstraints<PipelineStageKey, MeshId>` | required: each stage to its binding set | every projection respects it |
| `apply_pipeline_seat_constraints(pmc, pools, psd, c)` | `MappingConstraints<MeshId, const Candidate*>`, the SAT seating | forbidden: seats of mesh `m` whose footprint has fewer than `min_crossings_out(m)` cables leaving it | every projection needs that many cables out of `m` |
| `apply_pipeline_exit_constraints(pmc, mesh_graph, m, n, ..., c)` | `MappingConstraints<FabricNodeId, PhysicalNodeId>`, the intra-mesh solve of `m` | cardinality: at least `ceil(k(m, n) / links per exit ASIC)` of the chips in `boundary_slices(m, n)` land on ASICs cabled toward `n`'s seat | every crossing to `n` leaves from those slices, in every projection and carve |

The stage-to-mesh object is a filter and a report. It is not a solve, because the solver
maps one-to-one and many stages share a mesh. Capacity and channel counts do not fit in
`MappingConstraints`, so the object is only the per-stage view of the binding sets
(`get_valid_mappings(stage)`). The exact rules stay in Tier B's search and Tier C's carve.

The **seat helper** runs in `build_sat_placement_constraints`, next to
`apply_extra_constraints`, so it is rebuilt with the session. The mesh-pair form,
"two seats with fewer than `k_AB` links between them have no seat-graph edge", is a pair
condition, so it stays in `build_sat_placement_seat_graph` (Plan 1, 4.4).

The **exit helper** runs in `add_exit_node_constraints`, after the existing mesh-level
cardinality constraint for each destination mesh. Today that constraint lets any chip of
the mesh be an exit node. The pipeline adds a second, narrower one on the boundary-slice
chips, using the same links-per-ASIC rounding. The chips of a host rank come from
`MeshGraph::get_coord_range(mesh, host_rank)`.

**On the running examples.**

| Example | Seat helper | Exit helper |
|---|---|---|
| `mixed4` | M0 and M1: at least 2 cables out of the footprint | boundary slices are every slice, so the new constraint matches the existing one |
| `pod8` | M0 and M1: at least 2 cables out | M0 toward M1: at least 2 cables from A0 ∪ A3 (8 + 8 chips, not all 32). Seat U still passes, because its cables sit on h0 and h3. It fails later on M1's corners, which is a path fact, not a boundary one |
| `subtorus_4x4_pipeline_8stage` | each mesh: at least 2 cables out | each mesh toward each ring neighbour: at least 1 cable |

**Tests.** For each sweep MGD, build the constraints and check that every seating P8 accepts
satisfies them (the property test in P10). `pod8` with M0's layout pinned so that its cables
land only on A1 and A2 is infeasible in M0's intra-mesh solve, before Tier C runs. A "find all" run over its
budget gives `k = 0` and no added constraint.

**What RELAXED (later) changes.** The first version runs every row in the STRICT column.

| | STRICT edge (first version) | RELAXED edge (later) |
|---|---|---|
| crosses meshes A and B | needs `cap(A, B) >= 1` and uses one channel | no condition |
| both ends on one mesh | the two rectangles must touch | no condition |
| `k_AB`, boundary slices | counted | not counted |
| all edges of a pipeline | full search | steps 1, 2 and the no-overlap part of step 4 only |

**The edge cases above, through this plan.**

| Change | Step that decides | Why |
|---|---|---|
| fifth stage `[4, 6]` | 2 | 1 + 2 + 2 + 1 + 3 = 9 slices, 8 declared |
| fifth stage `[4, 4]` between s3 and s0 | 3, 4 | the search stops at `s3, s4, s0 -> M0` (1 + 2 + 1 = 4) and `s1, s2 -> M1` (2 + 2): 2 crossings (`s0 -> s1`, `s2 -> s3`), 2 channels; carve A0 / A1–A2 / A3. Nothing is stored. The "find all" run also finds the mirror (M0 and M1 swapped) and both carves on M0 (s3 or s0 on A0), so the boundary slices are {A0, A3} |
| a stage `[4, 10]` | 1 | 10 columns > 8: no mesh |
| `pod8`, M1 as 4x8 | 1 | `[1, 2] / [4, 2]` is not whole: no mesh |
| no MGD connection | 3 | `cap(M0, M1) = 0`; no single mesh holds 6 slices, so some edge must cross |
| `mixed4` + `mixed4_b` | 2 | 6 + 6 = 12 slices, 8 declared |
| two copies of `ring4`, 2 channels | 3, 4 | a ring4 on one mesh needs a 4-cycle of slices, and A0..A3 is a line; so each ring splits and crosses at least twice: 4 crossings, 2 channels |
| the same, `channels { count: 4 }` | pass | each ring splits 2 + 2; 4 crossings fit |

**RELAXED (later), the same plan.** With no MGD connection between M0 and M1:

| Change | Step that decides | Why |
|---|---|---|
| `s1 -> s2`, `s3 -> s0` RELAXED | 3, 4 | `s0, s1 -> M0`, `s2, s3 -> M1` crosses only on RELAXED edges; carve: s0 next to s1, s2 next to s3 |
| only `s3 -> s0` RELAXED | 3 | a ring crosses a cut at least twice, and only one crossing can be the RELAXED edge |

**Tests** (`test_mesh_graph_descriptor.cpp`): one per row above, plus a nested-graph MGD,
a RING `host_topology` axis (a ring4 on one mesh passes once A3–A0 wrap), a repeated edge
counting twice, the fail-fast cases (an odd grid ring and same-colour cable slices reject
before the first search node; "find all" over its cap saves the defaults), interchangeable-mesh symmetry (8 identical meshes on ALL_TO_ALL visit one
ordering), and a merged descriptor in which each part passes alone
but the merge fails capacity.

**Open points.**

- A connection between two sub-graphs (not two meshes) does not name the meshes it joins. The
  first version counts only mesh-to-mesh connections and reports the rest as unsupported.
- If a manual connection and `graph_topology` both cover a pair, step 3 sums them. Check that
  the control plane requests the sum too.

#### Examples

**`mixed4`: several projections survive.**

- **Slices.** 4x8 divided by `[1, 4]` gives 4x2 slices, four per mesh.
- **Binding sets.** 4x2 is one slice and 4x4 is two, so every stage binds to M0 or M1.
  Stage 0 is one slice.
- **Capacity.** The ring needs 6 slices and the MGD declares 8.

The table lists projections as the optional "find all" run sees them (M0 and M1 are
interchangeable, so each row also stands for its mirror). The existence search in step 3
stops at the first row that survives.

| Projection | Slices on M0 / M1 | Crossings | Result |
|---|---|---|---|
| all four stages on M0 | 6 / 0 | 0 | dropped: M0 holds only 4 slices |
| `s0, s1 -> M0`; `s2, s3 -> M1` | 3 / 3 | 2 (`s1 -> s2`, `s3 -> s0`) | survives |
| `s0, s1, s3 -> M0`; `s2 -> M1` | 4 / 2 | 2 (`s1 -> s2`, `s2 -> s3`) | survives |
| `s0, s2 -> M0`; `s1, s3 -> M1` | 3 / 3 | 4 | dropped: only 2 channels |

The slice counts of that last row fit (3 and 3, and each mesh holds 4). The ring does not.
`graph_topology` declares 2 channels between M0 and M1, and this assignment puts the ring's
neighbours on opposite meshes, so every edge crosses:

```
survives: s0, s1 on M0; s2, s3 on M1     dropped: s0, s2 on M0; s1, s3 on M1

s0(M0) ──── s1(M0)                       s0(M0) - - - s1(M1)
  ^            |                           |  \       /  |
  |            |                           |   \     /   |
  |            |                           |    \   /    |
s3(M1) ──── s2(M1)                       s3(M1) - - - s2(M0)

same mesh: s0→s1, s2→s3                  every edge crosses:
cross:     s1→s2, s3→s0                  s0→s1, s1→s2, s2→s3, s3→s0
2 crossings, 2 channels allowed          4 crossings, 2 channels allowed
```

How the step 3 search gets there. Every stage binds to both meshes, so the order starts
with the most slices: s1 (2 slices), on M0, the lowest empty mesh (M1 would be its mirror,
so it is never tried). Breadth-first from s1 gives s0, s2, then s3. s0 prefers M0, which
holds s1 (used 3 of 4). s2 also prefers M0 but needs 2 slices with 1 free, so it goes to M1
and uses one of the two channels (`s1 -> s2`). s3 has neighbours on both meshes, so the tie
goes to the mesh with more free slices, M1; the loopback `s3 -> s0` uses the second channel.
The leaf runs the step 4 carve: M0's 1 + 2 slices fit a path of three slices in a line of
four, and M1's 2 + 1 likewise. Tier B passes
after a handful of nodes and chooses nothing. In the "find all" run, every survivor crosses
M0–M1, and a ring crosses a cut an even number of times, so Plan 1 (4.4) gets
`k(M0, M1) = 2`.

**`pod8`: exactly one projection.**

- 4x2 binds only to M0, because M1 has two rows. 1x2 binds only to M1, because on M0 it is
  not a whole number of 4x2 slices.
- So s0..s3 go on M0 (4 of 4 slices) and s4..s7 on M1 (4 of 4). The projection crosses twice
  (`s3 -> s4`, `s7 -> s0`), within the 2 channels.
- **Segment carve.** M0's four stages fill a line of four slices, so the segment can only
  enter and leave at the end slices A0 and A3. M1's four stages are a path through all four
  slices of a 4-cycle.
- Tier B passes, with `k = 2` and boundary slices {A0, A3} on M0 and any slice on M1 (used
  by Plan 1).

**Edge cases.**

| Change | Result | Rule |
|---|---|---|
| `mixed4` with a fifth stage `[4, 6]` (3 slices) | reject | the ring needs 9 slices and the MGD declares 8 |
| `mixed4` with a fifth stage `[4, 4]` | pass | 8 of 8 slices: s0, s3 and one 4x4 stage on one mesh, the other two 4x4 stages on the other; every slice is used |
| `mixed4` with one stage `[4, 10]` | reject | wider than either mesh: empty binding set |
| `pod8` with M1 as 4x8, `host_topology [1, 4]` (4x2 slices), still 1x2 stages | reject | 1x2 is not a whole slice. Blaze's per-rank tiling could build it (3.4, "a tile smaller than a host slice"), so this rule is stricter than runtime |
| `mixed4` with the MGD connection removed | reject | every projection crosses on some edge, and no connection is declared |
| RELAXED (later): the same, with `s1 -> s2` and `s3 -> s0` RELAXED | pass | the projection `s0, s1 -> M0`, `s2, s3 -> M1` crosses only on those two edges, and RELAXED crossings need no connection |
| RELAXED (later): the same, with only `s3 -> s0` RELAXED | reject | every projection still crosses on a STRICT edge (`s1 -> s2` or `s2 -> s3`) |
| `mixed4` plus a copy named `mixed4_b` | reject | together they need 6 + 6 = 12 slices and the MGD has 8, although each fits alone (1.5) |
| two copies of `ring4` (four 4x2 stages each) | reject | 4 + 4 = 8 slices fit, but A0..A3 is a line, so neither ring closes on one mesh. Each splits and crosses at least twice: 4 STRICT crossings against 2 channels |
| the same, with `channels { count: 4 }` between M0 and M1 | pass | each ring splits 2 + 2; a projection places all eight stages at once |

### 2.3 Tier C - inside the topology mapper

The only tier that sees physical adjacency.

**Goal: prove a placement exists.** Tier C accepts a mapping when it finds *some* carve and
layout of the pipeline on it. That placement is a witness, not the one blaze will build:
blaze carves again at runtime, may take a different path (planned carve or uniform tiling,
see Tier D), and may land on a different placement. That is fine while nothing consumes the
mapper's placement (4.6). The one rule is the no-early-elimination rule from 2.2: Tier C
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
  carve planner port and the policy plan's candidate table are pure (3.2, 3.3).

It works in four steps after intra-mesh completion, detailed in section 4:

- **C1 - build the physical pipeline graph (4.1).** Turn host slices, slice adjacency and
  carve paths into slots, over every cable that could be bound. One graph per carve path
  and cycle.
- **C2 - pipeline solve (4.2).** Solve the pipeline graph (logical) onto each C1 graph
  (physical) with the builder's resolver. Accept when one solves and its crossing links `R`
  fit each connection's demand.
- **C3 - on success (4.3).** Emit the mapping and plan, and publish `R` through the
  `TopologyMapper` API; the control plane binds `R` first (phase 0).
- **C4 - on failure (4.4).** Retry the same seating with exit-node constraints that move the
  cable ends onto other host slices; if that cannot help, exclude the seating in the live SAT
  session and draw the next one. Later, optionally, constraints before the first solve, only
  where every carve fails.

Nothing before C1 may commit to a carve, a stage split or a cable choice: a different
layout or binding can make a different carve work.

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
good ones exist; 4.3 makes it bind the links Tier C's layout uses first.

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

#### Examples

Only Tier C sees where cables land, so it is the first tier that separates these seatings.
Section 4 traces each one step by step.

| Pipeline | Seating (M0, M1) | Cables land on | Tier C | Why |
|---|---|---|---|---|
| `mixed4` | (P, Q) | A0–B0, A3–B3 | reject | the slices form one loop of 8 and the ring needs a loop of exactly 6; every tiling fails in the resolver |
| `mixed4` | (P, R) | A0–B0, A2–B2 | accept | a 6-slice loop exists and the stages align onto it |
| RELAXED (later): `mixed4` with `s3 -> s0` RELAXED | (P, Q) | A0–B0, A3–B3 | accept | one fabric path covers 6 slices and the RELAXED loopback goes over host (3.4) |
| `pod8` | (P, U) | A3–T0, A0–T3 | reject | right count, both on M0's end slices, but on opposite corners of M1, so no path through M1's four tiles joins them |
| `pod8` | (P, V) | A3–T1, A0–T0 | accept | the cables land on neighbouring tiles of M1 |

Tier B passed all of these. Plan 1 (4.4) could only have caught `pod8` on U, and only with
a hard-pinned layout.

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

#### Examples

**`mixed4` on (P, R) at runtime.** The control plane binds the two cables Tier C relied on
(4.3). `build_topology` gets no `submesh_shape`, so `_collect_submeshes` calls `plan_carve`:

```
discover_host_slices      allgather: A0..A3 on M0, B0..B3 on M1, 4x2 each
discover_slice_adjacency  links confirmed by get_forwarding_direction:
                          the two lines plus the bound cables A0-B0, A2-B2
plan_ring_carve           the same 6-cycle and plan as Tier C (3.2, example A)
crc32 exchange            every rank derived the same plan
create_submesh            one call per slot on each rank's own mesh, in plan order
collapse                  ranks 1+2 and 5+6 carve the same chips: two joint submeshes
resolve_graph_layout      s0 A0, s1 A1+A2, s2 B1+B2, s3 B0 (3.1)
```

**The same machine today.** Add a second A0–B0 cable on lower chip ids than A2–B2. Today's
binding takes the two lowest ids, both A0–B0 (4.3), so blaze sees only A0–B0:

```
_prune_leaves    A3, B3 have one neighbour: drop; then A2, B2; then A1, B1; then A0, B0
CarvePlanError: no-cycle-of-length-6: only 0 slices can lie on any cycle.
```

The mapping is already committed, so this is a hard failure. That is the failure Tiers A–C
and 4.3 move earlier.

**`pod8` at runtime.** The harness gives `submesh_shape` 4x2 on M0's ranks and 1x2 on M1's.
`plan_carve` is not called; each rank tiles its mesh with `create_submeshes` and keeps the
tile whose `(0,0)` chip is local, then the resolver places all eight stages.

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
DAG after dropping loopbacks, edges carried over host are left out, and per-node core capacity
covers the router's endpoint budget.

The one change: the resolver throws with a message string on failure. Tier C needs to know
*which* stage or edge failed to attribute the failure (4.4). Add a structured failure
(failing stage, failing edge, or capacity conflict) alongside the existing message.

#### What the resolver checks

`resolve_pipeline_placement` orders the stages topologically (loopbacks excluded), then
`place_stages` assigns stages to submeshes one at a time, backtracking. For each stage it
tries the unused submeshes in order and keeps one that passes every check:

| Check | Rule |
|---|---|
| chip count | the submesh has exactly `node_chip_counts[stage]` chips |
| core capacity | links the stage needs (one per edge, plus 2 for stage 0's host-to-device and device-to-host) fit `capacity x chips`, where capacity is 1 slot per chip on submeshes of 8 or more chips and 2 on smaller ones |
| neighbour count | the submesh has at least as many distinct neighbouring submeshes as the stage has distinct logical neighbours |
| connectivity | it links to the submesh of every already-placed neighbour |
| lookahead | `remaining_stages_can_connect`: the stages still to place can still reach what they need |

`LinkSearch::solve` then picks one chip pair per edge, exclusive chips first, then shared;
host endpoints prefer chips no edge uses. Failure is `no valid submesh assignment found`.

#### Examples

**`mixed4` on (P, R), after the planned carve** (3.2, example A). Six submeshes: 0 = A0
(8 chips), 1 = A1+A2 (16, ranks 1 and 2), 2 = A3, 3 = B0, 4 = B1+B2 (16, ranks 5 and 6),
5 = B3.

```
topological order, loopback excluded: s0 s1 s2 s3
chip counts: s0 8, s1 16, s2 16, s3 8
submesh links: 0-1 1-2 3-4 4-5 inside meshes; 0-3 and 1-4 are the cables

s0: 2 links + 2 host endpoints = 4 <= 1 x 8; two logical neighbours (s1, s3)
    submesh 0 (neighbours 1, 3)  candidate     submesh 2 (neighbour 1 only)  skip
    submesh 3 (neighbours 0, 4)  candidate     submesh 5 (neighbour 4 only)  skip
    try submesh 0
s1: 16 chips and linked to submesh 0: only submesh 1
s2: 16 chips and linked to submesh 1: submesh 4, over the A2-B2 cable
s3: 8 chips, linked from 4 and to 0: submesh 3, closing over the A0-B0 cable
result: s0 -> 0, s1 -> 1, s2 -> 4, s3 -> 3
```

Had submesh 0 failed, it would try submesh 3 next and find the mirror (s0 B0, s1 B1+B2,
s2 A1+A2, s3 A0). The resolver re-derives the assignment from chip counts and links; it
does not take the carve planner's stage order as given.

**`pod8`, per-rank tiling.** Eight submeshes: A0..A3 (8 chips each) and T0..T3 (2 chips
each).

- **Chip counts.** These confine s0..s3 to the A submeshes and s4..s7 to the T submeshes.
- **Capacity.** s0 needs 4 of its 8 slots. Each T submesh has 2 x 2 = 4 slots, and s4..s7
  need 2 each.
- **M0.** s0..s3 must run along the line, A0 to A3 or A3 to A0.
- **M1.** s4..s7 visit all four tiles. In a 4-cycle a path through every vertex ends next to
  where it started.

| Seat | Cables | Result |
|---|---|---|
| U | A3–T0, A0–T3 (opposite corners) | starting at T0 the path ends at T1 or T2, but the cable back is on T3; starting at T3 it ends at T1 or T2, but the cable back is on T0: `no valid submesh assignment found` |
| V | A3–T1, A0–T0 (neighbours) | s0 A0, s1 A1, s2 A2, s3 A3, s4 T1, s5 T3, s6 T2, s7 T0; the mirror s0 A3 ... s3 A0, s4 T0, s5 T2, s6 T3, s7 T1 also works |

**Edge cases.**

| Case | What the resolver does |
|---|---|
| `mixed4` tiled 4x2 on both meshes | eight 8-chip submeshes; s1 finds nothing with 16 chips |
| `mixed4` tiled 4x4 on both meshes | four 16-chip submeshes; s0 finds nothing with 8 chips |
| a 1x2 stage that is stage 0 with 3 edges | 3 + 2 = 5 links against 2 chips x 2 = 4 slots: capacity fails |
| a ring stage on a submesh with one neighbouring submesh (A3 or B3 in `mixed4` on (P, R)) | the neighbour-count check skips it before any link search |
| harness overrides of pipeline-core pools | not in the MGD; Tier C uses blaze's defaults and can accept a layout runtime rejects (open question 7) |

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
| RELAXED (later): `plan_ring_placement`, `plan_fork_placement` (`host_edge_planner.py`) | `detail::plan_ring_placement`, `detail::plan_fork_placement`, taking the set of edges they may cut: the RELAXED edges (1.4) |
| RELAXED (later): `HostEdgePlanError`, `ForkShapeUnsupported` | structured result: reject vs fall back to the all-fabric paths |

- **Two copies.** The Python planner stays in tt-blaze and keeps running at runtime. Nothing
  in blaze moves or changes.
- **Same verdicts, not necessarily the same plans.** Because Tier C only proves existence
  (2.3), the two copies must agree on *whether* a carve exists for the same slices and
  adjacency, including the failure phase. Matching plans is not required. Keep the C++ copy
  deterministic anyway (geometry-keyed slices, sorted iteration, fixed search budgets) so the
  mapper's output is stable run to run.
- **Parity tests.** A shared set of golden cases (ring, heterogeneous, joint stage, joint
  stages on one mesh of 4x1 rank columns as in example F, fork with fabric and host returns,
  the known failure phases) fed to both copies, comparing accept or
  reject and the failure phase. Comparing the emitted plan and its crc32 digest is optional
  and only matters if the plan is ever consumed (4.6).
- **Discovery stays out.** `discover_host_slices` and `discover_slice_adjacency` are not
  ported. Tier C derives slices from the MGD plus the candidate mapping, and adjacency from
  the bound links (3.3, 4.3).

Planner behaviour the port must keep, or deliberately change:

- **Blaze commits to its first plan.** `plan_ring_carve` sorts cycles by fewest mesh
  crossings, then by slice keys, and returns the first one the stage shapes align onto.
  `plan_fork_carve` is first-found too. If the resolver then fails on that plan, blaze
  raises; it never tries the next cycle. Tier C instead **tries further cycles until one
  resolves**, because it only proves that a placement exists (2.3). The gap is that Tier C
  may accept a seating where blaze's first cycle fails in the resolver and a later one would
  have worked. Closing it later means either porting blaze's exact order and checking only
  the first cycle, or teaching blaze to retry. Binding `R` first (4.3) does not close it:
  the fill can add links that change which cycle blaze tries first.
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

#### How the carve planner works

The planner answers one question: given every rank's host slice and which slice pairs the
fabric links, which slices does each stage get? It works on whole host slices, because
`_collect_submeshes` needs each rank's chips inside exactly one slot. That keeps the problem
small: a 512-chip system with 4x2 slices is a graph of 128 vertices. Every rank must reach
the same plan, so slices are ordered by `(mesh_id, row0, col0)`, never by rank, and every
loop runs over a sorted sequence.

| Phase | Code | What it does |
|---|---|---|
| 1. discover (not ported) | `discover_host_slices`, `discover_slice_adjacency` | Each rank's local chips become a `HostSlice(mesh_id, row0, col0, nrows, ncols)`. For each chip and direction (N, E, S, W, Z), a neighbour on another slice counts only if `get_forwarding_direction` confirms it. One confirmed link per slice pair is enough; counts are not kept. Tier C gets slices from `MeshGraph::get_coord_range` and adjacency from the candidate links (section 4, steps 3–4). |
| 0. input checks | top of `plan_ring_carve` | All slices one shape. Each stage a whole number of slices (`r % nrows == 0`, `c % ncols == 0`), giving `stage_counts`. `need = sum(stage_counts)` no more than the slices. Stage 0 exactly one slice. |
| 2. reduce | `_prune_leaves`, `_contract` | Repeatedly drop slices with fewer than two live neighbours. Slices with more than two are *branch slices*; each run of two-neighbour slices between them becomes one weighted chain. Several chains may join the same two branch slices. |
| 3. cycle search | `_cycles_no_branch`, `_cycles_of_length` | Simple cycles of exactly `need` slices. With no branch slices the graph is disjoint loops, each walked once. Otherwise a DFS over branch slices takes one unused chain per step, adding its weight plus one, and prunes above `need`. Stops at 200,000 expansions or 64 cycles. |
| 4. align | `_group_cycle`, `_canonical_walk`, `_as_rectangle` | Cycles sorted by mesh crossings, fewest first. For each, the canonical walk (anchored at the lowest slice, stepping toward its lower neighbour), then every rotation, then the reverse. Each stage's group of `stage_counts[i]` slices must be one gap-free rectangle on one mesh with exactly the declared shape. First fit wins. |
| 5. emit | end of `plan_ring_carve`, `validate_carve_plan` | One `CarveSlot(mesh_id, shape, offset)` per stage in stage order, then a filler slot per unused slice (its rank idles). Every chip covered exactly once. |

Failures name their phase: `host slices are not uniform`, `not a whole number of host
slices`, `input-infeasible`, `no-cycle-of-length-N` (with achievable lengths),
`budget-exhausted`, and `no-alignment` (with the cycle's same-mesh run lengths).

#### Examples

A to E are on two 4x2-slice meshes, as in the running examples. F is one galaxy whose
ranks own 4x1 columns, with stages wider than a rank.

**A. One loop, no branch slices** (`mixed4`, cables A0–B0 and A2–B2; seating (P, R)).

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

Twelve alignments exist (six rotations, two directions) and four work, each putting the 4x4
stages on a different pair of ranks:

| Alignment | s0 | s1 | s2 | s3 |
|---|---|---|---|---|
| forward, offset 0 (returned) | A0 | A1+A2 | B2+B1 | B0 |
| forward, offset 3 | B2 | B1+B0 | A0+A1 | A2 |
| reverse, offset 1 | B0 | B1+B2 | A2+A1 | A0 |
| reverse, offset 4 | A2 | A1+A0 | B0+B1 | B2 |

The other eight fail in `_as_rectangle`, because a 4x4 group would take one slice from each
mesh. Blaze returns the first; Tier C only needs one.

**B. Wrong loop length** (`mixed4`, cables A0–B0 and A3–B3; seating (P, Q)).

```
   A0 ── A1 ── A2 ── A3
   │                  │
   B0 ── B1 ── B2 ── B3

phase 2   every slice has 2 neighbours: nothing pruned, no branch slices
phase 3   one loop: A0 A1 A2 A3 B3 B2 B1 B0, length 8
error     no-cycle-of-length-6: this fabric admits no simple cycle of exactly 6 host
          slices, which is what a 4-stage loopback ring needs.
            achievable cycle lengths: [8]
            nearest: none below, 8 above
```

**C. Branch slices and contraction** (add a third cable, A3–B3, to example A).

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

The extra cable changes nothing for `mixed4`. A ring of four 4x2 stages (need 4) would take
`c1 + c2`, the cycle A2 A3 B3 B2.

There is a quirk here. With branch slices, the search records only lengths up to `need`,
because longer partial cycles are pruned before they close. A ring needing 5 slices on this
fabric fails with `no-cycle-of-length-5` and `achievable cycle lengths: [4]`, although an
8-cycle exists. Without branch slices (example B) every loop's length is reported. The C++
port should keep the same verdict; the message may say more.

**D. Right length, no alignment** (example A's fabric; ring s0 4x2, s1 4x2, s2 4x8).

```
phase 0   stage_counts = [1, 1, 4], need 6
phase 3   cycle A0 A1 A2 B2 B1 B0, length 6
phase 4   s2 needs 4 slices in one row of one mesh. The cycle's same-mesh runs are
          [3, 3] (A0 A1 A2, then B2 B1 B0), so every rotation puts s2 across the crossing.
error     no-alignment: found 1 cycle(s) of the right length (6 slices) but no rotation
          places every stage as its declared rectangle ...
            cycle's same-mesh run lengths: [3, 3]
            stage shapes: ['4x2', '4x2', '4x8']
```

**E. A fork** (`plan_fork_carve`, example C's fabric). A branch that returns over fabric
needs a cycle through the router; one that returns over host needs only an open path
starting next to the router. A dead-end branch (allowed by Tier A, 2.1) is planned the same
way as a host return: an open path, one router neighbour. Blaze's planner rejects dead ends
today, so the port adds this.

| Step | What it does |
|---|---|
| input checks | Router exactly one slice; branch stages whole slices. `need_edges = 2 x fabric-return branches + 1 x host-return branches` (dead ends count as host returns); `need_slices = 1 + all branch slices`. |
| router candidates | Every slice with at least `need_edges` neighbours, in slice order. |
| branch order | Fabric-return branches first, shortest first (a short cycle must hug the router), then host-return branches. |
| walks | Fabric return: `_cycles_through` (prune, contract and cycle search on the free slices, keeping cycles through the router), both directions. Host return: `_path_of_length` from each free router neighbour. Each walk is cut into the branch's rectangles. |
| search | Backtracking over each branch's walks, capped at 2,000 attempts. |

Take a 4x2 router R, branch X of three 4x2 stages returning over fabric, and branch Y of one
4x2 stage returning over host:

```
need_edges = 2 + 1 = 3; need_slices = 1 + 3 + 1 = 5
router candidates: A2 and B2 (three neighbours each); try A2
branch X needs a 4-cycle through A2: A2 A3 B3 B2
  walks [A3 B3 B2] and [B2 B3 A3]; take [A3 B3 B2]: X0 A3, X1 B3, X2 B2, return B2 -> A2
branch Y: A2's free neighbours are {A1}; path [A1]: Y0 A1, return over host
plan: R A2 | X A3, B3, B2 | Y A1 | fillers A0, B0, B1
```

If Y also returned over fabric, the router would need four neighbours. No slice has more
than three, so it fails with `no slice has the 4 fabric neighbours this fork needs ... the
best any slice has is 3`.

**F. Stages wider than a rank, on one mesh** (`galaxy4x1`, a separate setup). One Blackhole
galaxy: a 4x8 mesh with `host_topology [1, 8]` and both axes RING, as in blaze's
`single_bh_galaxy_4x1_slices` MGD. Each rank owns one full-height 4x1 column, C0..C7 for ranks
0–7. The row wrap stays inside each column, so the only links between slices are column
neighbours, with the column wrap closing C7–C0. The slice graph is a single 8-cycle. The ring
is `s0 Embed 4x1 → s1 4x2 → s2 4x2 → s3 4x2 → s4 Head 4x1 → s0`, so each 4x2 stage is two
ranks.

```
   C0 ── C1 ── C2 ── C3 ── C4 ── C5 ── C6 ── C7
   r0    r1    r2    r3    r4    r5    r6    r7
    └──────────────── column wrap ───────────┘

path      one mesh, but s1..s3 are joint stages: build_pipeline sets multi_mesh and passes
          no submesh_shape, so this is the planned carve, not single-mesh tiling (3.4)
phase 0   slices all 4x1; stage_counts = [1, 2, 2, 2, 1]; need = 8 <= 8; stage 0 is 1 slice
phase 2   every column has exactly 2 neighbours: nothing pruned, no branch slices
phase 3   _cycles_no_branch: one loop C0 C1 ... C7, length 8 == need
phase 4   canonical walk: anchor C0; C1 (0,0,1) < C7 (0,0,7), so step toward C1
          forward, offset 0:
            s0 <- [C0]      col 0      4x1   ok
            s1 <- [C1 C2]   cols 1-2   4x2   ok
            s2 <- [C3 C4]   cols 3-4   4x2   ok
            s3 <- [C5 C6]   cols 5-6   4x2   ok
            s4 <- [C7]      col 7      4x1   ok
phase 5   slots 4x1@(0,0), 4x2@(0,1), 4x2@(0,3), 4x2@(0,5), 4x1@(0,7); no fillers
ring      C0 -> C1+C2 -> C3+C4 -> C5+C6 -> C7 -> column wrap -> C0
```

The 4x2 stages start on odd columns (1–2, 3–4, 5–6), not on the even pairs that tiling the
mesh by 4x2 would give. s0 takes column 0 by itself, and the planner cuts the walk wherever
the stage counts fall.

Sixteen alignments exist (eight starts, two directions) and ten work. The six that fail all
put a 4x2 stage on {C7, C0}: the columns are fabric neighbours through the wrap, but columns
{0, 7} are not contiguous, so `_as_rectangle` rejects the group.

| Alignment | s0 | s1 | s2 | s3 | s4 | Result |
|---|---|---|---|---|---|---|
| forward, start C0 (returned) | C0 | C1+C2 | C3+C4 | C5+C6 | C7 | ok; the loopback rides the wrap |
| forward, start C1 | C1 | C2+C3 | C4+C5 | C6+C7 | C0 | ok; s3 → s4 rides the wrap, the loopback is C0 → C1 |
| forward, start C2 | C2 | C3+C4 | C5+C6 | C7+C0 | | fails: s3 is not a rectangle |
| reverse, start C0 | C0 | C7+C6 | C5+C4 | C3+C2 | C1 | ok; s0 → s1 rides the wrap |
| reverse, start C1 | C1 | C0+C7 | | | | fails: s1 is not a rectangle |

After the carve, `_collect_submeshes` finds ranks 1 and 2 both in slot 4x2@(0,1) and collapses
them into one 8-chip submesh co-owned by two ranks; likewise ranks 3+4 and 5+6. In each 4x2
stage one co-owner is the entry rank and the other the exit rank; for each hop, the co-owner
that is not its endpoint builds no socket (`is_non_endpoint_coowner`). No rank idles.

What Tier B and Tier C see here. With one mesh there are no crossings, so Tier B's per-mesh
carve check (2.2, "Each mesh's segment can be carved") already runs the whole ring over the declared adjacency, RING wraps included,
and gives the verdict. Tier C reruns the same search over the mapped intra-mesh links (section
4, step 3), which are the same pairs, so it agrees. Its value on this MGD is the resolver
step, not the carve.

| Case on this galaxy | Phase | Result |
|---|---|---|
| ring of 4x1, 4x2, 4x2, 4x1 (need 6) | 3 | `no-cycle-of-length-6`, achievable [8]: the 8-cycle is the only simple cycle. Blaze pads such rings with 4x1 passthrough stages up to 8 |
| s0 declared 4x2 | 0 | `input-infeasible`: one rank must drive stage 0 |
| a stage declared 2x2 | 0 | `not a whole number of 4x1 host slices` |
| a 4x2 stage on {C7, C0} | 4 | `_as_rectangle` rejects it; this is why 6 of the 16 alignments fail |
| the column axis is LINE, not RING | 2 | C0 and C7 have one neighbour each; leaf pruning removes every column (`only 0 slices can lie on any cycle`) |
| a ring needing 9 slices | 0 | `input-infeasible`: the run has only 8 |

Blaze runs close versions of this on hardware and in unit tests:
`test_gemma_mixed_carve_ring_single_galaxy` (4x1, 4x2, 4x1, 4x4) and
`test_carve_planner_places_the_ring_on_a_galaxy`.

**Other edge cases.**

| Case | Phase | Result |
|---|---|---|
| `pod8` (4x2 and 1x2 slices) | 0 | `host slices are not uniform`; per-rank tiling is the path (3.4) |
| a 1x2 stage on 4x2 slices | 0 | `not a whole number of 4x2 host slices` |
| `mixed4` with s0 4x4 | 0 | `input-infeasible`: stage 0 owns host-to-device, the loopback entry and device-to-host, so one rank must drive it |
| only one cable, A0–B0 | 2 | leaf pruning removes every slice: `only 0 slices can lie on any cycle` |
| an open chain (no loopback) | before phase 0 | `plan_carve` refuses: there is no cycle to place; give it a `submesh_shape` |
| a huge fabric with many branch slices | 3 | `budget-exhausted`; Tier C treats it as unknown and accepts |

### 3.3 Inter-mesh candidates and binding: taken from the inter-mesh policy plan

Tier C needs, during mapping, every cable the control plane could bind for each MGD
connection. The [per-connection inter-mesh STRICT/RELAXED plan](https://riddy21.github.io/tt-metal/mgd-intermesh-policy-plan.html)
(the *policy plan*) already moves that set into the mapper. This design uses its pieces
instead of factoring its own helpers out of the control plane:

| Piece | Policy plan part | Used here for |
|---|---|---|
| `get_inter_mesh_channel_demand(a, b)` → `{strict, relaxed}` per mesh pair | 3, 5 | the budget `R` must fit (4.3); Tier B's `cap(A, B)` |
| `collect_inter_mesh_candidates(mesh_graph, psd, fabric_node_to_asic, options)` | 9 | C1's candidate cables, once per seating (4.1) |
| cable-type filter and tier per channel (`PortTypeRules`) | 6 | disallowed cables never reach C1; better tiers are tried first |
| `PortDescriptor::tier` and the tier sort in `pair_logical_intermesh_ports` | 9 | the fill order after the required links (4.3) |
| `resolve_inter_mesh_binding` in the mapper (stretch) | 11 | later: check that the required links can be placed before the mapping is emitted (4.3) |

This design adds two things on top: the required links `R`, and a phase 0 in the pairing
that binds them first (both in 4.3). Nothing else in the policy plan changes, and the
one-destination-chip-per-source-chip rule, link liveness and device-level exit pins all come
from the candidate table as it is.

```cpp
// From the policy plan, part 9 (pure: no cluster, no MPI).
InterMeshCandidates collect_inter_mesh_candidates(
    const MeshGraph&, const PhysicalSystemDescriptor&,
    const std::map<FabricNodeId, AsicID>& fabric_node_to_asic,
    const InterMeshCandidateOptions&);

// Added here (4.3): mark the channels of the required links, so the pairing binds them first.
PortDescriptorTable to_port_descriptor_table(
    const InterMeshCandidates&, const PipelineRequiredLinks& required);
```

**Called per seating.** The policy plan calls `collect_inter_mesh_candidates` once, after the
final mapping. Tier C calls it for every seating that passes intra-mesh. It is pure, so the
cost is a PSD scan over the two seats' ASICs, small next to the intra-mesh solve.

**Order of work.** The policy PR and the candidates PR (part 9) land first; P9 here then adds
`R` and phase 0. If P9 must land before part 9, factor today's gather
(`gather_intermesh_cables_for_exit_nodes`) into a pure helper with the same signature as
`collect_inter_mesh_candidates`, so the later swap is mechanical. That interim refactor runs
for every MGD and must keep `AnnotatedIntermeshConnections` byte-identical (4.7).

#### Example

`mixed4` on (P, R), on a machine with an extra A0–B0 cable whose chip ids are lower than the
A2–B2 cable's:

| Call | Result |
|---|---|
| `collect_inter_mesh_candidates(...)` | three candidates: A0–B0 (first), A0–B0 (second), A2–B2. Each source chip cables to one destination chip on M1, so all three pass |
| pairing, no required links | today's rule: the two lowest ids, both A0–B0 |
| pairing, `R = {A0–B0, A2–B2}` from Tier C | phase 0 binds both cables in `R`; the count of 2 is full |

Tier C and the control plane read the same candidate table, so the cables Tier C checks are
the cables the pairing can bind.

### 3.4 Carve paths: planned carve, per-rank tiling and host cuts, as the pipeline builder does them

Blaze has two ways to turn a mesh into submeshes, and Tier C needs both (2.3). Both end in
the same code, so Tier C shares one path from "slots" to the resolver and the two carve
paths differ only in how they produce slots. The first version uses only these two paths
(and single-mesh tiling), because every edge is STRICT. RELAXED (later): for a pipeline with
RELAXED edges (1.4), a host-cut planner (tt-blaze #4513, #4693, #4917) first picks which of them go over host,
then reuses the same two carve paths, then resolves each fabric chain on its own (see
"Host cuts on RELAXED edges" below).

This section reads tt-blaze `origin/main` (34e783a631). An older checkout (45d3be26c4, before
#4513) has neither `Edge.transport` nor `placement="flexible"`.

#### What blaze does

`PipelineGraph._collect_submeshes` (`pipeline_builder/graph.py`, tt-blaze):

| Step | Multi-mesh, planned carve | Multi-mesh, per-rank tiling | Single mesh |
|---|---|---|---|
| when | `multi_mesh`, no `submesh_shape` | `multi_mesh`, `submesh_shape` given | not `multi_mesh` |
| 1. slots | `plan_carve` → one `create_submesh(shape, offset)` per slot on this rank's mesh | `create_submeshes(submesh_shape)` on this rank's mesh | `create_submeshes(shape)`; every stage must have the same shape |
| 2. this rank's slot | the one slot holding its chips (exactly one) | first tile whose `(0,0)` chip is local, else first tile with any local chip | n/a (all tiles are local) |
| 3. global list | allgather each rank's slot chips | same | every tile, no allgather |
| 4. joint stages | ranks with identical chip sets collapse into one submesh, ordered by first rank | same | n/a |
| 5. layout | `resolve_graph_layout` (links from the control plane) | same | same |

`multi_mesh` does not mean "more than one mesh". `build_pipeline` (`blaze/stages/stage.py`)
sets it when the MGD has more than one mesh **or** any stage spans more than one rank
(`has_joint_stage`: the stages' rank counts add up to more than the number of stages). For a
joint stage it passes no `submesh_shape`, so the carve is planned. A single mesh with a joint
stage therefore takes the planned carve: for example a galaxy whose ranks own 4x1 columns,
running 4x2 stages (3.2, example F). Single-mesh tiling is only for a single mesh where every
stage sits on one rank's slice or less.

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
   - *single mesh, no joint stage:* the one stage shape stamped over the mesh.
2. **Each rank's slot.** For each mesh and host rank, `MeshGraph::get_coord_range(mesh,
   host_rank)` gives its chips. Apply blaze's step-2 rule to pick its slot.
3. **Submesh list.** Build each picked slot's chip tuples and collapse identical chip sets
   in first-rank order. On a single mesh, take every tile.
4. **Links.** Build `DirectLinks` between the submeshes from the candidate links (3.3) and
   routed intra-mesh links.
5. **Layout.** Call `resolve_graph_layout_with_connections` with those submeshes and links.

Which paths Tier C may use is what blaze has for that MGD, so an accepted mapping is one
blaze could build some way:

| MGD | every edge STRICT (first version) | some edge RELAXED (later) |
|---|---|---|
| one mesh, every stage on one rank | single-mesh tiling | single-mesh tiling |
| one mesh with a joint stage, or several meshes | planned carve or per-rank tiling | host cuts on the RELAXED edges, then the plan's carve or tiling; if a fork raises `ForkShapeUnsupported`, the all-STRICT paths |

A mapping is accepted if any allowed path succeeds. RELAXED (later): a ring with RELAXED edges whose
host-cut planner fails on its inputs (for example non-uniform host slices) is rejected,
because the builder raises there rather than falling back.

**Tile shape per mesh (per-rank tiling).** Blaze gets it from the caller; Tier C enumerates
it. For each mesh, the candidates are the distinct stage shapes that divide the mesh, plus
the host-slice shape (for a mesh that carries no stage, whose ranks still carve something).
Try combinations across meshes; there is usually one candidate per mesh. One shape per mesh,
since every rank of a mesh carves the same shape.

**Example: slots to submeshes on a planned carve** (`mixed4` on (P, R), plan from 3.2,
example A):

```
M0 slots in plan order: 4x2@(0,0), 4x4@(0,2), 4x2@(0,6)
  rank 0 -> 4x2@(0,0)   rank 1 -> 4x4@(0,2)   rank 2 -> 4x4@(0,2)   rank 3 -> filler 4x2@(0,6)
  a rank whose chips fall in zero or two slots is a planner bug and raises
collapse identical chip sets, in first-rank order:
  submesh 0 = A0 (8 chips)   1 = A1+A2 (16, ranks 1 and 2)   2 = A3 (8)
  submesh 3 = B0 (8)         4 = B1+B2 (16, ranks 5 and 6)   5 = B3 (8)
6 distinct submeshes >= 4 stages
```

**Example: per-rank tiling combinations** (`mixed4` on (P, Q), after the planned carve
failed with `no-cycle-of-length-6`). The candidate tile shapes are {4x2, 4x4} on each mesh:

| M0 tiles | M1 tiles | Submeshes | Result |
|---|---|---|---|
| 4x2 | 4x2 | eight with 8 chips | fails: nothing with 16 chips for s1 |
| 4x4 | 4x4 | four with 16 chips, two ranks each | fails: nothing with 8 chips for s0 |
| 4x2 | 4x4 | four with 8 on M0, two with 16 on M1 | fails: s0 and s3 must be neighbouring A slices that both have a cable to M1, but the cables are on A0 and A3 |
| 4x4 | 4x2 | the mirror | fails the same way |

Every path fails, so the seating is rejected (section 4).

**Example: different slice shapes** (`pod8`). The planned carve rejects up front because the
slices are not uniform. The only tile shapes that divide each mesh are 4x2 on M0 and 1x2 on
M1. Each tile is exactly one rank's slice, so every rank keeps its own and nothing
collapses: A0..A3 with 8 chips each, T0..T3 with 2 chips each. The resolver then decides by
where the cables land (3.1).

#### Host cuts on RELAXED edges - RELAXED (later)

> **RELAXED (later).** Not in the first version. Everything down to "Several pipelines" is
> the host-cut design for when edge policy is added.

**What the builder does** for a pipeline with RELAXED edges (1.4), when it has a loopback.
These are the steps of blaze's flexible placement (`PipelineGraph._resolve_topology` in
`graph.py`), with one change: the planner may cut only the RELAXED edges, where blaze today
may cut every edge but stage 0's outgoing one.

1. **Plan the host cuts** (`pipeline_builder/host_edge_planner.py`):
   - *ring:* `plan_host_edges` → `plan_ring_placement(slices, adjacency, stage_shapes,
     page_sizes)`. It first looks for an all-fabric cycle that the stage shapes align onto,
     the same search as `plan_ring_carve`. If none exists, it covers the ring's slices with
     `k` vertex-disjoint fabric chains, trying the fewest chains first up to one chain per
     stage. Stages are grouped along each chain, and consecutive chains are joined over host.
     A cut falls only between stages (never inside a wide stage), only on a RELAXED edge, and
     never on stage 0's outgoing edge. Ties go to the smallest host-hop page sizes, then to
     cutting the loopback. Inputs follow the carve planner's rules: uniform slices,
     whole-slice stage shapes, stage 0 on one slice. Breaking them raises
     `HostEdgePlanError`, which is not caught.
   - *fork:* `plan_fork_host_edges` → `plan_fork_placement`. It tries `plan_fork_carve`
     first (all fabric). Failing that, it walks each branch from a distinct router neighbour
     and cuts to host where a branch runs out of fabric, only on a RELAXED edge and never on
     the router's edge into the branch. Non-uniform slices or a router wider than one slice
     raise `ForkShapeUnsupported`; the caller then resolves the fork with every edge on
     fabric.
2. **Apply the plan.** The cut edges are carried over host; every other edge stays on fabric.
3. **Carve.** If the plan has a carve (some stage spans several slices), its slots are used,
   as in the planned carve. Otherwise, if every stage has the same shape, the builder sets
   `submesh_shape` to that shape and uses **per-rank tiling**. This is the planner and
   per-rank tiling working together: the planner picks the cuts, and tiling makes the
   submeshes. A ring that is already all fabric on single-slice stages produces no plan,
   and placement goes to the resolver as for an all-STRICT pipeline.
4. **Resolve each chain.** Cut edges are dropped from the resolver's edges, and the fabric
   graph splits into chains. `_resolve_chains_separately` calls the resolver once per chain.
   A chain's pool is the planner's chip sets for its stages when the plan fixed them,
   otherwise the submeshes not yet claimed by earlier chains. That second pool is greedy: an
   earlier chain may take submeshes a later one needed.

`host_hop_page_size_bytes`, required on the source of every edge the planner may cut, only
breaks ties, so it does not affect whether a placement exists. It stays out of MGD.

**What Tier C does** for a pipeline with a RELAXED edge, per seating:

1. Run the C++ `plan_ring_placement` / `plan_fork_placement` (3.2) on the slices and slice
   adjacency (section 4, steps 3–4), with the RELAXED edges as the edges it may cut.
2. Drop the cut edges from the graph the resolver sees.
3. Produce slots as the builder does in step 3: the plan's carve, or per-rank tiling by the
   uniform stage shape (steps 1–3 of "What Tier C does" above). With a fixed tile shape
   there is no tile-shape enumeration.
4. Build `DirectLinks` once, then call `resolve_graph_layout_with_connections` per fabric
   chain with the builder's pool rule, including the greedy order. Matching that order keeps
   the verdict the same as the builder's.
5. Accept if the plan and every chain resolve. On `ForkShapeUnsupported`, fall back to the
   all-STRICT paths.

**What RELAXED changes upstream of Tier C.** A RELAXED crossing drops out of Tier B's
crossing rules (2.2) and of `k_AB` (4.4), so mappings that lack fabric for it are not
rejected early. Rejections that remain come from shapes, capacity, STRICT crossings, and
what the planner cannot cover.

**Example: `mixed4` with `s3 -> s0` RELAXED, on (P, Q).** With every edge STRICT this
seating is rejected (3.2, example B). `plan_ring_placement` runs:

```
   A0 ── A1 ── A2 ── A3
   │                  │
   B0 ── B1 ── B2 ── B3

phase 1   no 6-cycle (only the 8-cycle). Census note: spare = 2 and 6 + 2 is an
          achievable length, so passthrough_ring = 2 (two passthrough stages would
          close an 8-stage ring over fabric).
phase 3   one component of 8 >= 6, so k starts at 1. _cover_with_chains(1) starts at
          the smallest slice and extends to the smallest free neighbour:
            1. A0 A1 A2 A3 B3 B2    2. A0 B0 B1 B2 B3 A3    3. A1 A0 B0 B1 B2 B3    ...
phase 4   _group_walk per cover; cuts only between stages and only on RELAXED edges:
            cover 1 best: s0 A3, s1 B3+B2, s2 A0+A1, s3 A2   cut s1 -> s2   skipped: STRICT
            cover 2 best: s0 A3, s1 B3+B2, s2 B1+B0, s3 A0   cut s3 -> s0   key (1, 0, (3,))
          key = (cut page sizes, 0 if the loopback is cut else 1, cut indices)
result    cover 2: its one cut is the RELAXED loopback. s3 -> s0 goes over host.
          carve M0 4x2@(0,6), M1 4x4@(0,4), M1 4x4@(0,0), M0 4x2@(0,0); A1, A2 idle
          _resolve_chains_separately: one fabric chain s0 -> s1 -> s2 -> s3
```

With every edge RELAXED, cover 1's cut would be allowed too, and cover 2 would still win on
the key, because it cuts the loopback.

**Edge cases.**

| Case | Result |
|---|---|
| a cover whose only alignment cuts a STRICT edge | skipped |
| a cover whose only alignment cuts `s0 -> s1` | skipped, even when `s0 -> s1` is declared RELAXED: stage 0's outgoing edge stays on fabric |
| a joint between chains inside a 4x4 stage | skipped: a wide stage cannot be split by a host hop |
| `pod8` with a RELAXED edge | `HostEdgePlanError: host slices are not uniform`, not caught; Tier C rejects, because the builder raises rather than falling back |
| a fork with a two-slice router | `ForkShapeUnsupported`; the fork is resolved with every edge on fabric, so Tier C uses the all-STRICT paths |
| a ring with RELAXED edges that is already all fabric on single-slice stages | no plan; placement goes to the resolver as for an all-STRICT pipeline |

#### Several pipelines

When the MGD declares more than one pipeline (1.5), Tier C checks them together on each
seating:

1. **Place the components.** Plan each pipeline on the slices the earlier ones left free,
   in declaration order. Each pipeline has its own stage 0 or router, so each owns its own
   host input and output. If one does not fit, retry with the next order of the pipelines;
   there are few, so trying every order is cheap. Unused slices become filler slots, once,
   after the last pipeline.
2. **Resolve each component as its own graph.** Pipelines share no edge, so the resolver
   runs once per pipeline, over that pipeline's slots, as `_resolve_chains_separately` does
   for chains.
3. **Accept** only if every pipeline is placed and resolves on this seating. Feedback to SAT
   (4.4) is about the seating, as for one pipeline.

**Example: two copies of `ring4`** (a ring of four 4x2 stages, needing a 4-cycle of slices)
on the `mixed4` meshes, 8 slices in all. The MGD declares `channels { count: 4 }` between
M0 and M1; with `mixed4`'s 2 channels Tier B already rejects it (2.2):

- **Cables A0–B0, A2–B2, A3–B3** (carver example C's fabric). `ring4_a` takes the only
  4-cycle, A2 A3 B3 B2. The free slices A0, A1, B0, B1 form the path A1–A0–B0–B1, with no
  cycle, so `ring4_b` fails. The other order fails the same way. The seating is rejected,
  although each pipeline fits on it alone.
- **Add a cable A1–B1.** Now there are three 4-cycles. `ring4_a` takes the first,
  A0 A1 B1 B0, and `ring4_b` takes A2 A3 B3 B2 from what is left. The seating is accepted,
  with every slice used.

#### Reuse or rebuild

| Piece | Today | Tier C |
|---|---|---|
| resolver | `resolve_graph_layout_with_connections` (C++) | **use directly** (3.1) |
| submesh links | `discover_submesh_links` (C++, anonymous namespace in `pipeline_builder.cpp`, asks the control plane per chip pair) | **refactor and share:** split out a pure core that takes "are these two chips linked, in which direction" as a callback. Tier D passes the control-plane lookup it uses today; Tier C passes the candidate links. One loop, two link sources |
| host-rank chip ranges, chip ids | `MeshGraph::get_coord_range`, `coordinate_to_chip` (C++) | **use directly** |
| row-major tiling | `MeshDeviceImpl::create_submeshes` (C++, needs a live `MeshDevice`) | **rebuild**, a few lines of offset arithmetic with the same divisibility rule. Factor it into a shared helper both call so the order cannot drift |
| planned slots | `plan_ring_carve`, `plan_fork_carve` (Python) | **port** (3.2) |
| RELAXED (later): host-cut planning | `plan_ring_placement`, `plan_fork_placement` (Python, `host_edge_planner.py`) | **port** (3.2), used only for pipelines with a RELAXED edge, taking the RELAXED edges as the ones it may cut; built on the carve planner's internals, so it mostly extends that port |
| per-chain resolution, chain split | `_resolve_chains_separately`, fabric component split (Python) | **rebuild**: a loop over the existing resolver, with the same pool rule and order |
| pick a rank's slot, collapse joint stages | `_collect_submeshes` (Python) | **rebuild** in C++, small and pure; add to the parity tests (3.2) |
| default core counts | `_pipeline_core_defaults` (Python) | **rebuild** (open question 7) |

#### Examples

- **GPT-OSS single galaxy:** one 8x4 mesh, four 4x2 stages. Single-mesh tiling gives four
  4x2 tiles; the resolver places the ring on them. The planner is never used, as at runtime.
- **`galaxy4x1` (3.2, example F):** also one mesh, but its ranks own 4x1 columns and three
  stages are 4x2, so they are joint. That makes it a planned carve: five slots, three of
  them co-owned by two ranks. Single-mesh tiling by 4x2 would give four tiles on even column
  pairs, which cannot hold the 4x1 stages or the odd-column 4x2s.
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

For example, take `pod8` with M1 changed to 4x8 and `host_topology [1, 4]` (4x2 slices),
still with 1x2 stages s4..s7. `create_submeshes(1x2)` on M1 makes 16 tiles, row-major. Each
of M1's ranks owns four of them and keeps the one whose `(0,0)` chip is local: rank 4 keeps
row 0, columns 0–1; rank 5 keeps row 0, columns 2–3; and so on. Rows 1–3 of M1 carry no
stage. A cable that lands on M1 row 3 then does not count, because `discover_submesh_links`
only sees chips inside submeshes. The same physical seat passes or fails depending on
whether the intra-mesh layout puts the cabled chips in row 0. Tier B as written rejects
this MGD at load (2.2).

## 4. Pipeline mapping in the topology mapper (Tier C)

The topology mapper places meshes with SAT joint seating. A *seat* is a physical footprint
for one mesh; its chip-by-chip layout is a preference to the intra-mesh solve (2.3). The loop
in `MultiMeshSolutionEnumerator::next()` (`topology_mapper_utils.cpp`) is today:

```
seating = placement_session_->next()                 SAT joint seating
physical = build_hierarchical_from_flat_graph(...)
full = complete_intra_mesh_for_placement(...)        identity intra-mesh
if fail: add_forbidden_constraint(failed seat); loop
emit
```

With a pipeline descriptor, Tier C adds four steps after intra-mesh success. Each step is its
own section with its own inputs, outputs and PRs:

| Step | What it does | Inputs | Outputs | Section |
|---|---|---|---|---|
| C1. Build the physical pipeline graph | host slices, slice adjacency and carve paths, turned into slots: the physical graph the pipeline is solved onto | seating, intra-mesh layout, candidate cables | one physical graph per carve path (and per cycle the carve planner finds) | 4.1 |
| C2. Pipeline solve | solve the pipeline graph (logical) onto a C1 graph (physical) with the builder's resolver | the pipeline, a C1 graph | on success: the layout and the crossing links `R`, within each connection's demand; on failure: a structured reason | 4.2 |
| C3. On success: emit and publish the cables | emit the mapping and plan; publish `R` through the `TopologyMapper` API; the control plane binds `R` first | C2's layout and `R` | the mapping, the plan, `get_pipeline_required_links()` | 4.3 |
| C4. On failure: constraints for the next solve | retry the same seating with exit-node constraints; if that does not help, exclude the seating in the SAT placement, and draw the next one | C2's failure | an intra-mesh retry, then `exclude_mapping(seating)` | 4.4 |

```
[before the first solve]
  Tier B: at least one projection exists (no projection is chosen)
  C4, later (optional): exit-node and seat constraints every carve agrees on      (4.4)

loop:
  seating = placement_session_->next()                                  existing
  full = complete_intra_mesh_for_placement(...)                         existing
  if fail: add_forbidden_constraint(failed seat); loop                  existing
  -- new from here, only when the MGD has a pipeline --
  for each physical graph G from C1(seating, full):                     (4.1)
      result = C2(pipeline, G)                                          (4.2)
      if result.accepted: C3(result); emit                              (4.3)
  C4 (4.4):
      retry the intra-mesh solve with exit-node constraints from the failure,
        at most kMaxLayoutRetries times, back to C1 each time
      then exclude_mapping(seating); loop
      after kMaxPipelineRejections rejected seatings: give up with the last reason
```

**Rule: do not commit to a carve before it is needed.** Different layouts and different
bound cables can make a different carve work, so no step may fix a carve, a stage-to-mesh
split or a cable choice ahead of the step that needs it:

- C1 builds its graphs over every cable that *could* be bound for the MGD connections
  (`collect_inter_mesh_candidates`, 3.3), not over a binding chosen in advance. The binding
  follows the solve: the cables the accepted layout uses become `R`, and C3 has them bound
  first.
- C1 and C2 together are an existence search. They try every carve path (3.4) and further
  cycles until one solves, and reject only when all fail (the no-early-elimination rule, 2.2).
- A constraint C4 derives from a failure holds only for the seating that failed: it steers
  that seating's retry, never another seating's solve. Constraints that apply to every
  seating (added before the intra-mesh or master solve) may only state what every carve
  agrees on: the minimum number of crossings and the union of boundary slices. When those
  facts cannot be computed within budget, nothing is added.
- A rejection is fed back to the SAT placement as a constraint on the seat set, never on one
  mesh on one seat (4.4).

**First version and later.**

| Step | First version (M3) | Later |
|---|---|---|
| C1 | slices, adjacency, every STRICT carve path over the candidate cables (P5, P6, P7, P8) | host-cut path for RELAXED edges (P11) |
| C2 | the resolver, the demand check on `R`, the tight-mesh segment test and colour pruning (P4, P8, P9) | a per-pair link budget inside the resolver, if the demand check fires in practice |
| C3 | `R` through `get_pipeline_required_links()`; phase 0 and the fill in the control plane's pairing (P9) | phase 0 inside the mapper with the policy plan's part 11 |
| C4 | `exclude_mapping(seating)`, `unique_shapes` off, rejection cap (P8); exit-node retry on the same seating (P15) | seat-pair clauses (P13); constraints before the first solve, only if rejection counts call for them (P10) |

### 4.1 Step C1 - build the physical pipeline graph

C1 turns one seating and its intra-mesh layout into the physical graph the pipeline is solved
onto. It runs after intra-mesh success, because the layout decides which chips sit under each
host slice, and so where each cable lands. Host slices, slice adjacency and carve paths are
one step: each depends on the one before, and none is useful alone.

- **Inputs.** The seating, `full.fabric_node_to_asic`, the candidate cables for every MGD
  connection (`collect_inter_mesh_candidates`, 3.3), and the links the intra-mesh topology
  routes over.
- **Output.** A sequence of physical graphs, one per carve path, and for the planned carve one
  per cycle the planner finds. Each graph is a set of *slots* (submeshes: the chips of one
  stage position) with their chips, the chip-level links between slots, and each chip's
  pipeline-core capacity. C2 is run on each in turn until one solves.
- **What it does not decide.** Which stage goes on which slot (C2), which cables are bound
  (C3), or what SAT offers next (C4).

```
cands  = collect_inter_mesh_candidates(mg, psd, full)  (3.3) + routed intra-mesh links
slices = host slices of every mesh, through full.fabric_node_to_asic
adj    = slice pairs joined by a link in cands
for each carve path allowed for this MGD and its edge policies             (3.4)
    every edge STRICT: planned carve, per-rank tiling, single-mesh tiling
    RELAXED (later) - some edge RELAXED: host cuts on RELAXED edges, then its carve or tiling
    (several pipelines: each placed on the slices still free)
  slots = that path's slots over slices and adj                             (3.2, 3.4)
  subs  = pick each rank's slot, collapse joint stages                      (3.4)
  yield G = (subs, chip links between subs from cands, per-chip capacity)
```

**The physical graph is chip-level, not a host-rank graph.** A graph of host ranks and their
adjacency would miss what the builder checks: per-chip pipeline-core capacity, a stage whose
entry and exit land on the same chip, which chips stage 0's host I/O uses, and the router's
endpoint budget. So each slot keeps its chips, and each link between slots keeps its two
chips, as the resolver's `submesh_chips` and `direct_links` (3.1).

**C1 already uses the pipeline.** The carve paths are not pipeline-independent: per-rank
tiling tiles by the stage shape, and the planned carve searches for a cycle of slices with the
pipeline's length and cuts it by stage shapes (3.2). So C1 is itself a search with a budget
(`_MAX_SEARCH_EXPANSIONS`), and running out is "unknown", not "reject" (3.2).

#### Links

Only links the control plane can route over count:

- *between meshes:* `collect_inter_mesh_candidates` (3.3) returns every cable the pairing
  could bind, honouring link state, cable types and the one-destination-chip-per-source-chip
  rule. The PSD holds every host's cables, so this needs no MPI exchange. Which of them get
  bound is decided in C3.
- *inside a mesh:* the mapped pairs of logical neighbours in `device_topology`, RING wraps
  included. Extra cables outside the topology are ignored.

To do:

- In `MultiMeshSolutionEnumerator::next`, after intra-mesh success, call
  `collect_inter_mesh_candidates(mesh_graph, psd, full.fabric_node_to_asic, options)` with the
  same `InterMeshCandidateOptions` as the final call (down-link hashes from the chip info,
  `PortTypeRules`).
- Build the resolver's `DirectLinks` from it: one `LinkPair` per candidate link (source chip,
  destination chip) in slot-local coordinates. Order each pair's links by best tier, then chip
  id, so the resolver's first-found layout crosses on the best cables.
- Keep the table with the accepted seating; the control plane uses the same one (policy plan,
  part 9), so it is not computed twice.

#### Slices and adjacency

For each mesh and host rank, `get_coord_range` gives a
`HostSlice(mesh_id, row0, col0, nrows, ncols)`. Two slices are adjacent when a link above
joins them, matching blaze's `discover_slice_adjacency`.

#### Carve paths to slots

Produce slots for each carve path blaze has for this MGD (3.4):

- *Planned carve* (several meshes, or one mesh with a joint stage): `plan_ring_carve` (or
  `plan_fork_carve`) finds a cycle of slices and cuts it into stage rectangles, joint stages
  included. Tier C tries further cycles when C2 fails on one (3.2).
- *Per-rank tiling* (several meshes, or one mesh with a joint stage): each mesh tiled by one
  shape, each rank keeps its tile, identical tiles collapse into joint stages, and C2 assigns
  stages to tiles. This covers what the planner rejects up front: host slices of different
  shapes across meshes (`llama_8b_pod`), and graphs with no loopback (open chains).
- *Single-mesh tiling* (one mesh, no joint stage): every tile of the one stage shape.
- *Host cuts*, RELAXED (later) (a pipeline with a RELAXED edge): `plan_ring_placement` (or
  `plan_fork_placement`) picks which RELAXED edges go over host, then the plan's carve or
  tiling by the uniform stage shape, then one solve per fabric chain (3.4).

C2 runs per graph; a seating fails only when every graph fails. In Case B (both crossings on
one slice pair) with every edge STRICT, the planned carve yields no graph
(`no-cycle-of-length-4`) and the per-rank tiling graph fails in C2, so the seating fails.
RELAXED (later): with `s3 → s0` RELAXED, the same Case B is accepted: with both cables
between slices `M0a` and `M1a`, the slices still form one fabric path
`M0b → M0a → M1a → M1b`. The planner lays `s0..s3` along it and cuts only the loopback
`s3 → s0` to host, which is allowed because it is RELAXED and not stage 0's outgoing edge.

**Failure.** A path that yields no graph reports `failure_stage = "pipeline_carve"` with the
planner's phase (`no-cycle-of-length-N`, `no-alignment`, `budget-exhausted`).

### 4.2 Step C2 - pipeline solve

C2 solves the pipeline graph (the logical side: stages and edges) onto one C1 graph (the
physical side: slots, chips and links). The solver is the builder's own resolver,
`detail::resolve_graph_layout_with_connections` (3.1), so Tier C and the runtime decide the
same way. It is not the topology solver (`solve_topology_mapping`): that maps nodes and edges
of two adjacency graphs, and does not model per-chip capacity, entry and exit chips, or host
I/O.

- **Inputs.** The pipeline and one graph `G` from C1.
- **Output on success.** The layout (stage to slot, each edge's exit and entry chips, core
  slots, host I/O chips) and `R`, the inter-mesh link behind each crossing edge.
- **Output on failure.** A structured reason: `pipeline_layout` with the failing stage, edge or
  capacity conflict (3.1), or `pipeline_binding` when `R` exceeds a connection's demand.

```
layout = detail::resolve_graph_layout_with_connections(G.subs, G.links)    (3.1)
         (once per fabric chain when host cuts split the graph)
R      = for each crossing edge in layout.resolved_edges: its (exit chip, entry chip)
accept if every mesh pair has |R(pair)| <= strict + relaxed
```

#### Resolver inputs

Feed the resolver what blaze feeds it:

- stages as nodes; edges with one loopback per cycle; edges cut to host dropped, and the
  remaining fabric chains resolved one at a time when cuts split the graph
  (`_resolve_chains_separately`);
- each slot's chips as `submesh_chips`, with `node_chip_counts = rows * cols`;
- blaze's default core counts (`_pipeline_core_defaults`: router `2 + 2 * branches`; a
  host-return tail 2 on a fused router's small submesh; ring head 4 with
  `co_locate_ring_host_io`; chain head and tail 2);
- C1's links between slots as `direct_links`.

It catches per-chip core capacity, shared entry and exit chips, and the router's endpoint
budget.

**Gap.** Blaze's default core counts are used. Harness `pipeline_core_coords()` overrides
are not in the MGD, so Tier C could accept a layout runtime rejects (open question 7).

#### `R` must fit each connection's demand

Phase 0 takes one channel per link in `R`, counted against the mesh pair's demand from
`get_inter_mesh_channel_demand` (policy plan, part 3):

| Rule | Detail |
|---|---|
| budget | links in `R` on a pair ≤ `strict + relaxed`. Tier B's channel rule does not guarantee this: it checks that *some* projection fits, and C1 may carve one with more crossings |
| counting | `R` uses the STRICT part of the demand first, then the RELAXED part |
| RELAXED and BEST_EFFORT connections | a pair with no link in `R` keeps the policy plan's rules, zero-link included. A pair with links in `R` needs them: a STRICT pipeline edge needs fabric, whatever the connection's policy |
| cable types | disallowed types are never candidates, so never in `R` |

To do:

- Check the budget in C2 after the resolver returns. Over budget, that graph fails with
  `failure_stage = "pipeline_binding"` and C2 moves to the next C1 graph; if every graph
  fails, C4 runs.
- If this check fires in practice, pass an optional per-mesh-pair link budget into
  `LinkSearch`, so the resolver looks for a layout within budget instead of failing after.
  The resolver is shared with Tier D, so the parameter defaults to no budget.

#### Solve cost: fail fast, no node budget

The resolver is a depth-first search: stages onto slots with backtracking, then links and
core slots per placed prefix. Measured on rings of single-slot stages over grids of 4x2-chip
slots:

| Case | Layout exists? | Resolver |
|---|---|---|
| one 4x8 grid, 32 stages | yes | 0.7 ms |
| one 8x8 grid, 64 stages | yes | 4.4 ms |
| odd 5x5 grid, 25 stages | no | 2.5 s |
| two 4x4 meshes, 32 stages, cables on same-colour slices | no | 3.4 s |
| two 4x8 meshes, 64 stages, same | no | over 120 s |

When a layout exists it is fast. When none exists, the search is exponential, and that is the
common case in this loop: every rejected seating is a proof that no layout exists. A plain
CaDiCaL encoding of the same placement was no better on the hard cases (3.0 s and over 230 s),
so SAT is not the fix. The planned carve path is cheap in C2 because its slots already pin each
stage; the risk is per-rank tiling with many identical one-slice stages.

**Rule: fail fast, no node budget.** Every C2 run must end in a definite answer, layout or
proved impossible, and an impossible graph must be proved so quickly that C4 can afford one
per seating. A slower average solve is acceptable. So there is no node budget and no
"undecided" result: a budget that passes would commit layouts blaze cannot build, and a budget
that rejects would turn slow-but-valid graphs into false failures.

The slow failures are all one shape, and it has an exact cheap test:

- **Tight meshes split into fixed segments.** When a mesh's slots are all used (one-slice
  stages tiling every rank, the hard case), the pipeline's visit to that mesh is a path through
  every slot between the two cable chips where it enters and leaves. Which cables those are is
  a small choice: C1's graph carries only the candidate cables between each mesh pair, so there
  are a few (entry, exit) pairs per mesh.
- **Each segment is decided in O(1).** A Hamiltonian path between two given cells of a full
  rectangular grid exists exactly under the Itai–Papadimitriou–Szwarcfiter conditions (1982):
  the colour condition (opposite colours on an even grid, both on the majority colour on an
  odd one) plus a few forbidden cases on grids 1, 2 or 3 slots wide. An odd grid holds no
  closed ring. Every infeasible row in the table fails here, before the first search node.
- **Pre-check order in C2.** (1) Per tight mesh, keep only (entry, exit) cable pairs that pass
  the segment test; if no pair is left, the graph fails with `pipeline_layout` and the mesh,
  cells and colour in the reason. (2) Run the resolver with the surviving pairs as its
  allowed crossings, so a found layout is never thrown away for a segment that was already
  ruled out. (3) Meshes with free slots skip the test: slack is what makes the DFS fast there.
- **The resolver also prunes with it.** In `remaining_stages_can_connect`, when the unused
  slots of a mesh exactly match the stages still to place there, apply the same colour test to
  the open path ends. That catches tight regions inside a partly placed prefix, the case the
  pre-check cannot see.

**What this does not cover.** Hamiltonian path is NP-complete on grids with holes (same
paper), so a mesh with dead slots that is also tight can still be slow. No MGD in the repo
does that today. The fallback for it is a **hard fail**, not an undecided pass: if C2 is still
running after a wall-clock cap (proposed 10 s), C2 fails that graph with `pipeline_layout`
and "solve exceeded 10 s on mesh M, N tight slots with holes", and the log says which mesh.
That keeps the failure fast and explicit; open question 13 asks whether to keep even this cap.

The experiments behind these numbers are in
[`pipeline_placement_experiments/`](pipeline_placement_experiments/README.md).

#### Failure reporting

`failure_stage = "pipeline_carve"` (C1), `"pipeline_layout"` or `"pipeline_binding"` (C2),
carrying the carve phase, the resolver reason, or the mesh pair over its demand. When the
session is exhausted it surfaces through the existing *"Graph specified in MGD could not fit
in the discovered physical topology"*.

- **No descriptor, no change.** Without `pipeline_descriptors`, the loop is exactly today's.
- **Bonus.** `generate_rank_bindings --all-solutions` only emits mappings the pipeline can be
  built on, so `sweep_rank_binding_solutions.py` stops launching dead ones.

### 4.3 Step C3 - on success: emit the mapping and publish the cables

When C2 accepts a graph, C3 emits the mapping and makes sure the cables the layout crosses on
are the cables that get bound. Today the control plane picks cables without looking at the
pipeline, so it can bind cables on the wrong slices even when good ones exist (the `mixed4`
example below). The MGD does not change for this: it already says how many channels each
connection gets and, with the policy plan, under which policy and on which cable types. Which
physical cables the pipeline needs depends on the seating and the intra-mesh layout, so it is
a mapper output, not MGD content.

**Emit.** The mapping, the carve plan and the layout. Each rank's mesh and host rank give its
slice, then its slot, then its `TT_PIPELINE_STAGE` (4.6).

Each new piece of the binding has its own subsection below, with what to build:

| Piece | Where it runs | When | New or reused |
|---|---|---|---|
| [Required links `R`](#new-the-required-links-r-and-where-they-live) | mapper, from C2's layout | when C2 accepts | new |
| [Phase 0 in the pairing](#new-phase-0-in-the-pairing-bind-r-first) | control plane, `pair_logical_intermesh_ports` on rank 0 | control-plane init | new |
| [Fill order](#new-fill-order-after-the-required-links) | same | after phase 0 | reused tier order, plus a new spread order |
| [Single host](#new-single-host) | `generate_intermesh_connections_on_local_host` | control-plane init | new |
| [Phase 0 inside the mapper](#later-phase-0-inside-the-mapper-policy-plan-part-11) | mapper, `resolve_inter_mesh_binding` | before emit | later, with the policy plan's part 11 |

The candidate table C3 binds from is built in C1 (4.1), and the check that `R` fits each
connection's demand is part of C2's acceptance (4.2).

**Order matters.** Binding comes after C2, so the carve is chosen over every candidate cable
and the binding follows it. Binding first and carving over the bound set would throw away
carves that a different binding allows (the `mixed4` example below).

**No rerun over the bound cables.** An earlier version of this plan reran the solve over the
bound set. Once phase 0 binds all of `R`, that rerun cannot fail: the bound set contains
`R`, and the witness layout crosses only on `R`, so it is still a layout over the bound set.

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

#### New: the required links `R`, and where they live

`R` is, for each crossing edge in the layout's `resolved_edges`, the link it uses: the exit
chip on the source mesh and the entry chip on the destination mesh. Several crossing edges
may share one link.

```cpp
// tt-metalium/experimental/fabric/pipeline_required_links.hpp (new)
struct PipelineRequiredLink {
    FabricNodeId src;  // exit node on the source mesh
    FabricNodeId dst;  // entry node on the destination mesh
    std::string pipeline, src_stage, dst_stage;  // for logs and failure messages
};
// Keyed by (lower mesh id, higher mesh id); both directions of an edge fold into one pair.
using PipelineRequiredLinks = std::map<std::pair<MeshId, MeshId>, std::vector<PipelineRequiredLink>>;

const PipelineRequiredLinks& TopologyMapper::get_pipeline_required_links() const;
```

Where `R` lives, and why:

| Option | Verdict |
|---|---|
| in the MGD, as device-level connections | no: `R` depends on the seating and the intra-mesh layout. Pinning it in the MGD fixes a carve before the solve (the rule in section 4) and needs a different MGD per machine |
| in `rank_bindings.yaml` | not in the first version: the runtime mapper runs Tier C itself and gets the same `R`. It can be written for debugging, like the stage plan (4.6) |
| mapper output, next to the mapping and the candidate table | **yes** |
| the `PortDescriptorTable` handed to the pairing | **yes**, as a `required` flag on each channel of a link in `R` (`to_port_descriptor_table`, 3.3) |

To do:

- Fill `R` at the end of C2 from the accepted layout and store it on the `TopologyMapper`
  with the mapping. Empty without a pipeline descriptor.
- Add `required` to `PortDescriptor`, next to the policy plan's `tier`, and set it in
  `to_port_descriptor_table`.
- Write `required_by: <pipeline>/<src_stage> -> <dst_stage>` on those channels in the policy
  plan's `inter_mesh_candidates.yaml` dump.
- Multi-host: the pairing runs on rank 0, which is the rank that holds the candidate table
  (policy plan, open question 12). Every rank computes the same mapping, so `R` is the same
  everywhere; only rank 0 consumes it.

#### New: phase 0 in the pairing, bind `R` first

In `pair_logical_intermesh_ports`, add a phase 0 ahead of phase 1a:

1. Take every link with a `required` channel, mesh pair by mesh pair.
2. Place it with its mesh pair's Z rule (assign_z → Z, otherwise NESW then Z), taking the
   link's best-tier channel, and count it against the pair's budget.
3. Run today's work-list for the rest of the budget, skipping links phase 0 placed.

With no required links, the result is exactly today's.

**When phase 0 cannot place a link** (a direction already owned toward another mesh,
Z-ness, the one-peer-per-direction rule, or no ports left): in the first version the mapping
is already committed, so the control plane fails at init with the link and the pipeline edge
that needed it (`failure_stage = "pipeline_binding"`). `R` is at most the requested count, so
this is rare. The likely case is an exit chip cabled toward two neighbouring meshes. Moving
phase 0 into the mapper turns this into a rejected seating instead (the last two
subsections).

#### New: fill order after the required links

After phase 0, the remaining budget is filled in the policy plan's order: best cable tier
first (`PortDescriptor::tier`). With a pipeline descriptor, ties within a tier are broken
toward spread: consecutive picks land on different host-slice pairs (round-robin over slice
pairs, chip id within one). Spare bound links then add slice adjacency rather than doubling
an existing pair, which gives blaze's runtime carve planner more cycles to choose from.

| Pipeline descriptor | Fill order |
|---|---|
| none | the policy plan's `(tier, chip id)`; with no cable data, today's order, so goldens do not move |
| present | `(tier, slice-pair round-robin, chip id)` |

Open question 9 asks whether spread should ever outrank tier.

#### New: single host

`generate_intermesh_connections_on_local_host` is a separate greedy pass. The policy plan
already makes it skip cables outside the candidate table. It also needs phase 0: place the
links in `R` first, then run today's greedy. The alternative is to route single host through
the pairing (with every cable local), which gives one rule but moves single-host goldens.
Until one of the two lands, a single-host MGD with a pipeline descriptor gets a warning that
`R` is not guaranteed to be bound.

#### Later: phase 0 inside the mapper (policy plan, part 11)

The policy plan's stretch goal moves channel selection and direction assignment into the
mapper as `resolve_inter_mesh_binding(mesh_graph, candidates, port_capacity)`. When it lands:

- Pass `R` as required links, so phase 0 runs inside it.
- Call it in the loop after C2 accepts. A failure to place `R` is the policy plan's
  `PLACEMENT` kind, so C4 handles it like a C2 failure (4.4), and the init-time failure above goes
  away.
- The control plane applies the binding plan as given; `R` no longer has to reach it
  separately.

Until then, phase 0 has no mapper-side check.

---

#### Binding examples

**`mixed4` on (P, R) with an extra cable.** Suppose the machine has two A0–B0 cables, both
on lower chip ids than the A2–B2 cable, and the MGD asks for 2 channels.

| Step | Today | With 4.3 |
|---|---|---|
| candidates | three: A0–B0, A0–B0, A2–B2 | same (3.3) |
| required links | none | Tier C's layout crosses on `s1 -> s2` (A2–B2) and `s3 -> s0` (A0–B0), so `R` holds one cable of each |
| budget | | 2 links in `R`, demand 2: fits |
| bind | the two lowest ids: both A0–B0 | phase 0 binds `R`; the count of 2 is full |
| runtime | blaze sees only A0–B0; leaf pruning removes every slice; `no-cycle-of-length-6` (2.4) | blaze sees A0–B0 and A2–B2 and builds |

**Spread fill.** Take the count as 3 instead. `R` takes two, and one more is filled from
the spare A0–B0. The spread order makes no difference here, since every spare cable is on
A0–B0. With a spare on A3–B3 instead, the fill would take it first. That adds a new slice
pair, which gives blaze's runtime planner the branch-slice fabric of 3.2, example C.

**Over budget.** With `count: 1`, `R` needs 2 links on M0–M1 and the layout fails the budget.
No carve of a ring across two meshes crosses fewer than twice, so the seating is rejected;
Tier B would already have rejected this MGD (2.2).

**Phase 0 cannot place a cable.** Suppose the A2–B2 cable's chip already uses its only free
direction toward a third mesh. In the first version, the control plane fails at init naming
A2–B2 and `s1 -> s2`. With part 11 in, the mapper rejects (P, R) and tries the next seating.

**STRICT device-level connections.** The user pins the exit chips, so the candidates are
exactly the pinned ones. If the pins give A0–B0 and A3–B3, Tier C rejects `mixed4`
(3.2, example B), and C3 has nothing to choose between.

#### Binding gaps

- **Live link state.** At runtime the candidate table leaves out down links (policy plan,
  part 9), so `R` never holds one. Offline (`generate_rank_bindings` on a factory PSD), every
  cable looks up, so the offline answer can be optimistic; the runtime mapper, which runs
  Tier C itself, then picks another seating.
- **Intra-mesh links.** Inside a mesh, Tier C counts the mapped logical neighbours in
  `device_topology`. Links the topology does not route over (extra cables, inactive routing
  planes) are excluded, matching what blaze's adjacency and the resolver confirm.

### 4.4 Step C4 - on failure: constraints for the next solve

When every C1 graph fails in C2, C4 decides what to try next. It has two moves, in order:

| Move | Scope | What it adds | When |
|---|---|---|---|
| 1. Exit-node retry | the same seating | an intra-mesh constraint that moves the crossings' exit chips onto different host slices, then back to C1 | first version (P15), at most `kMaxLayoutRetries` times per seating |
| 2. Exclude the seating | the SAT placement | `exclude_mapping(seating)` in the live master session, then the next seating | first version (P8) |
| later: seat-pair clause | the SAT placement | "not seat_a and seat_b together", when a failure is proven for every carve over the pair | P13 |
| later: constraints before the first solve | every seating | exit-node and seat constraints from facts every carve agrees on | P10, only if rejection counts call for them |

The first two moves are the loop. The other two only make it faster, and they are sound only
if they never remove a seating some carve could use.

#### Exit-node retry on the same seating (first version, P15)

C2 sees the layout the intra-mesh solve returned. A different layout of the same seating
puts the cabled ASICs under different host slices, and so can carve where this one does not.
So a C2 failure is exact for the layout, not for the seating. Before excluding the seating,
C4 retries the intra-mesh solve with the failing arrangement blocked.

**What is blocked.** The failure depends on which host slice each cabled exit ASIC sits
under: its *slice signature*. For each mesh m on a failing boundary, record
`sig(m) = { exit ASIC a -> slice of the fabric node on a }` over the ASICs cabled toward the
neighbouring seat. The retry adds one clause to m's intra-mesh solve: not every ASIC in
`sig(m)` is under the same slice as last time. Layouts that only move chips within a slice
keep the signature, so they are not retried; they cannot change the slice graph.

**Steps.**

1. On a C2 failure, compute `sig(m)` for each mesh on a boundary the pipeline crosses.
2. Skip the retry when it cannot help: if every ASIC in `sig(m)` is fixed under its slice by
   hard constraints (rank bindings or MGD pins), the signature cannot change. This is the
   runtime case with every host rank bound. The retry pays off when host-to-rank is free:
   `generate_rank_bindings`, and hosts left UNSET.
3. Re-solve one mesh at a time, keeping the other meshes' layouts. The failure is about the
   meshes jointly, but each intra-mesh solve is per mesh. With the others fixed, blocking
   `sig(m)` removes exactly the failing combination. Blocks are per retry, not kept: once
   another mesh's layout changes, an old signature may work again.
4. Back to C1 with the new layout. Stop after `kMaxLayoutRetries` (proposed 4) per seating,
   or when the intra-mesh solve has no other signature, then move 2.

**New solver API.** A slice-signature block is not expressible with today's
`MappingConstraints` (required, preferred, forbidden, cardinality). Add
`add_forbidden_joint_placement(groups)`: "not every (fabric-node group -> ASIC) in the list
holds", encoded with one indicator per (ASIC, slice), `y <-> OR x[f, a]` over the slice's
fabric nodes f, and one clause `OR NOT y`. It is related to the policy plan's partial
blocking clause (part 11), which blocks exact entries rather than groups.

**Example.** `pod8` offline, on (P, U), with host-to-rank free: U's cables land on opposite
corners of M1, and C2 fails (3.1). The retry blocks M1's signature, and the intra-mesh solve
puts a different host's slice under the corner ASIC. If that brings the two cable ends next
to each other along M1's loop, C2 passes on the same seating. With every host rank bound, the
signature is fixed and C4 goes straight to move 2.

#### Exit-node constraints for every seating, from Tier B (later, optional)

The intra-mesh solve decides which chips sit under each slice, and so where the cables
land. Today its exit-node constraint (`add_exit_node_constraints`) allows any mesh node. A
pipeline can narrow that, but only to what every carve agrees on, because C1 and C2 have not run
yet and must stay free to pick any carve:

- For each mesh pair (m, n) with `k(m, n) > 0`: at least `ceil(k / links per exit ASIC)` of
  the fabric nodes in `boundary_slices(m, n)` map to ASICs cabled toward n's seat
  (`apply_pipeline_exit_constraints`, a cardinality constraint next to the existing one).
- `k` is the minimum number of crossings and `boundary_slices` the union of slices that may
  hold a boundary stage, over every projection (Tier B's `PipelineMappingConstraints`, 2.2).
- When Tier B's "find all" run is skipped or over budget, `k = 0` and nothing is added.

This steers the layout toward ones C1 can carve, which narrows the one-layout gap,
and it never rules out a layout some carve could use. Like pruning before the first solve, it changes
speed, not answers, and it is added only if the first version's rejection counts call for
it.

#### Excluding the seating in the SAT placement (move 2)


The SAT targets are logical meshes; the globals are seats. Each mesh-to-mesh connection must
land on a seat pair with links between them. Under STRICT the channel count is hard; under
RELAXED only adjacency is hard and the count is a preference. A seat fixes a mesh's
footprint, but its chip layout is only a preference to the intra-mesh solve (2.3), so the
ASICs under each host slice are not known before solving unless hard pinnings fix them.

> **Decision: check each seating first; constraints in the solve only as a later speed-up.**
>
> Two ways were considered:
>
> | | Option 1: encode the pipeline in the SAT seating solve | Option 2: check each seating, feed back, re-solve |
> |---|---|---|
> | Solves | one | one master session; each rejected seating is blocked in place and the next one is drawn |
> | Exact? | no: only necessary conditions can be encoded at seat level | yes: runs the builder's carve planner and resolver on the real layout |
> | New code | seat-graph edge drop, seat and exit constraints (P10) | the check and one `exclude_mapping` call in the existing retry loop (P8) |
> | Cost | a larger master encoding | one intra-mesh solve and one check per rejected seating |
>
> Option 1 cannot be the whole check. The master solve chooses seats (footprints), not chips,
> and pipeline feasibility depends on facts that exist only after the intra-mesh solve: which
> chips sit under each host slice (and so slice adjacency), a stage whose entry and exit land
> on the same chip, per-chip pipeline-core capacity, stage alignment along the whole ring,
> and stage 0's host I/O slots. Encoding those exactly would need chip-level and stage-to-slice
> variables for every mesh in the master solve, which is close to one flat solve of the whole
> system, the thing the two-level seating design avoids. Seat-level rules can only prune: on
> `mixed4` the seating (P, Q) passes every seat-level rule and fails on loop length, and on
> `pod8` seat U fails because its cables land on opposite corners of M1. Neither is visible
> before the intra-mesh solve. Option 1 is also not clearly faster: the master solve is cheap
> and incremental, and the expensive per-seating step is the intra-mesh solve.
>
> So:
>
> 1. **First version: check each seating (C1 and C2, 4.1 and 4.2) and exclude rejected seatings (Plan 2,
>    below), in P8.** After intra-mesh success in
>    `MultiMeshSolutionEnumerator::next()`, run the check. On reject, call
>    `exclude_mapping(seating)` and continue. `block_assigned` adds the clause to the live
>    master session, so the next seating needs no re-encode. The loop, the retry and the
>    in-session blocking already exist for intra-mesh failures.
> 2. **Later, only if measured: pruning before the first solve (Plan 1, below), in P10.** Record pipeline rejections per run
>    (seatings tried before the first accept) on `mesh4x4_stage4x2`, the ring-stress cases and
>    the superpod cases. Add pruning if those counts are too high, starting with the
>    footprint-level `k_AB` edge drop, which needs only Tier B's `PipelineMappingConstraints`.
>    Pruning must never remove a seating the check would accept, so it changes speed, not
>    answers.
>
> The check runs after intra-mesh completion, not before: before it only seat-level facts
> exist, which is pruning again. It uses the builder's planner and resolver (the C++ port), not
> a second SAT model of the builder, so the mapper and the runtime decide the same way.

#### Feedback after a rejected seating - Plan 2 (first version)

This is what the first version ships (P8). When C2 (4.2), including its demand check on `R`, still fails after the exit-node retry,
rejects a seating, feed back the strongest constraint the failure supports:

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
- **First version: whole-seating exclusion only.** `exclude_mapping(seating)` is the only
  feedback P8 uses. The seat-pair clause and the per-seat forbid wait for attributed failures
  (P13), because a wrong stronger cut would eliminate seatings that work.
- **Rejection cap.** Count pipeline rejections per run (`pipeline_rejections_`) and give up
  after `kMaxPipelineRejections` (proposed default 256, overridable for sweeps), reporting the
  last carve or resolver reason through the existing "could not fit" error.
- **Measure.** Record the rejection count in the placement stats, so the sweep and CI show
  how many seatings each MGD tries before the first accept. These numbers decide whether
  Plan 1 (4.4) is needed.

**Example.** `mixed4` fails on (P, Q) (section 4, iteration 1).

| Feedback | Effect | Verdict |
|---|---|---|
| `exclude_mapping({M0 -> P, M1 -> Q})` | blocks this seat set only | correct: (P, R) is still offered and succeeds |
| `add_forbidden_constraint(M0 on P)` | M0 may never use P | wrong: it removes (P, R), one of the two seatings that work |
| `add_forbidden_constraint(M1 on Q)` | M1 may never use Q | wrong: harmless here, but Q could carry M1 next to a different M0 seat |
| block "P and Q both used" (new API) | blocks (P, Q) and (Q, P) | only if proven for every carve over P and Q. Here it holds, because the slice graph is the same either way, but proving it needs the attribution from 3.1 and 3.2 |

On `pod8`, (P, U) fails and `exclude_mapping({M0 -> P, M1 -> U})` leaves (P, V), which is
the only seating that works.

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
   `unique_shapes = !mgd_has_pipeline_descriptors`, read from the merged MGD (1.5). These
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

The rules elsewhere in this design (no split fixed early, 2.2 and 4.4; feedback about seats,
4.4) still apply; with `unique_shapes` off they no longer have to carry this alone.

**Cost.** Off, the master solve can yield every mesh permutation of a seat set, which is what
the dedup was added to avoid on large symmetric machines. Cap pipeline rejections per run
(4.4) and measure on the ring-stress and superpod cases before turning it off by default.

**Example.** On `mixed4`, (P, Q) and (Q, P) are the same set of seats {P, Q}, so with
`unique_shapes` on only the first is offered. In this example both fail with the same slice
geometry, so the dedup is harmless. Now change it so that M1 has an MGD pin putting its
host rank 0 on host h1. Then (Q, P) lays out differently from (P, Q), and the two can give
different answers. After (P, Q) fails, `exclude_mapping` blocks the key {P, Q}, which also
covers (Q, P), so the swap is never checked. With `unique_shapes` off, (Q, P) is its own
seating and gets its own check.

#### Pruning before the first solve - Plan 1 (later, optional)

**Not in the first version** (see the decision above). Plan 1 encodes necessary conditions
before the master solve, in one solve. It does not
commit to a projection and does not run once per projection. **Every condition must hold for
all surviving projections and for every chip layout the intra-mesh solve may still pick.** A
condition that holds for only some of them would drop seatings the runtime could build on,
so it is left to the exact check (4.4).

From the surviving projections, take only what they all agree on. These come from Tier B's
optional "find all" run (2.2); when it is skipped or over its budget, the values are the
defaults (`k = 0`, every slice a boundary slice) and Plan 1 prunes nothing:

- for each mesh pair A–B, `k_AB` = the **minimum** number of crossings over all
  projections (0 if any projection has none on A–B). In the first version every edge is
  STRICT, so every crossing counts. RELAXED (later): RELAXED crossings may go over host, so
  they will not count (1.4). With several pipelines, a projection places all of them, so its
  count is the sum over pipelines (1.5);
- for each mesh, the boundary slices are the **union** over projections of the slices that
  may hold a boundary stage.

When exactly one projection survives (for example, every stage's binding set is a single
mesh), these are just that projection's values.

**RELAXED (later): when every edge is RELAXED**, `k_AB` is 0 for every mesh pair and the seat-pair and
single-seat constraints below do not apply. Plan 1 then prunes nothing, and all pipeline
checking falls to the exact check (4.4).

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
  slice reaches the router's required degree (2 per return plus 1 per dead-end branch;
  RELAXED (later): a RELAXED return counts 1). Otherwise the exact check catches it.

**Caveats.**

- **Shared seats.** The seat graph merges identical seats across mesh pairs. If two mesh pairs
  that share a seat pair need different `k`, edge-dropping is unsound for the looser pair.
  Then fall back to pairwise blocking (4.4) for those pairs.
- **What Plan 1 cannot encode.** Joint-stage alignment along the whole ring, per-chip
  pipeline-core capacity, a stage whose entry and exit land on the same chip, stage-0
  host-to-device and device-to-host slots, and the router's endpoint budget. These depend on
  the whole pipeline at once. The exact check (4.4) catches them.

##### Examples

**How the master solve is set up today, on `mixed4` and pods P, Q, R:**

| Step | Code | Result |
|---|---|---|
| seat pools | `create_sat_placement_pools` | both meshes embed their PGD grouping (a 4x8 over four hosts): pool {P, Q, R}. `sat_footprint_canonical` merges orientations of one footprint (key: grouping name, type, sorted ASICs), so each pod is one seat |
| seat graph | `build_sat_placement_seat_graph` | for the MGD pair M0–M1, `saturated_link_count` parallel edges per seat pair: P–Q x2, P–R x2, Q–R x1 |
| constraints | `build_sat_placement_constraints` | each mesh on a seat from its pool. M0 and M1 share a pool but are adjacent, so they are not twins and get no symmetry window. The footprints are disjoint, so injectivity keeps chips unshared |
| master session | `MasterSolve::restart` | maps M0–M1 (count 2, RELAXED) onto the seat graph. RELAXED only needs adjacency, so all six ordered pairs are possible, including (Q, R) on one link |

**Plan 1 on `mixed4`.** `k(M0, M1) = 2` (2.2). The footprint-level rule drops the Q–R edge,
which has 1 link, so (Q, R) and (R, Q) are never offered. P–Q stays (2 links), although
`mixed4` fails on it: the reason is the loop length (3.2, example B), which no seat-pair
link count can see. The slice-level rule would not catch it either, because P–Q's two
links are on distinct slice pairs.

**Plan 1 on `pod8`.** `k = 2`, and every projection agrees that the boundary slices on M0
are A0 and A3 (2.2). Both P–U and P–V have 2 links, so the footprint rule keeps both. With a
hard-pinned layout the slice-level rule could check that both links touch {A0, A3}, which
they do on both seats. It still cannot reject U: U fails because its cables land on
opposite corners of M1 (3.1). That depends on the path through M1, not on boundary slices.

**What Plan 1 must not do.** It must not use one projection's values. For `mixed4`, the
projection `s0, s1 -> M0` has boundary stages s0 and s1 on M0, while `s0, s1, s3 -> M0` has
s1 and s3. Pruning by either one's slices would drop seats the other could use. The minimum
`k` and the union of boundary slices hold for both.

**RELAXED (later): with `s3 -> s0` RELAXED**, `k(M0, M1) = 1`. The projection `s0, s1 -> M0`,
`s2, s3 -> M1` crosses on `s1 -> s2` (STRICT) and `s3 -> s0` (RELAXED), so it has one STRICT
crossing, and no projection has fewer. The Q–R edge, with its one link, now stays: `mixed4`
can use (Q, R) by sending the loopback over host. **With every edge RELAXED**, `k = 0` and
Plan 1 prunes nothing.

### 4.5 The whole loop on two examples

#### `mesh4x4_stage4x2`, step by step

Worked example: `mesh4x4_stage4x2` from the blaze sweep. Two 4x4 meshes M0 and M1, each
`host_topology [1, 2]` (two 4x2 slices, one rank each), a 4-stage ring of 4x2 stages, and
two declared M0–M1 connections.

**Before the first solve.**

- *Tier B* (2.2): a projection exists, for example `s0, s1 → M0` and `s2, s3 → M1`. Its
  mirror `s0, s1 → M1` is equally valid and is **not** ruled out; neither is chosen. The ring
  needs 4 slices and each mesh has 2, so every projection puts two stages on each mesh and
  crosses M0–M1 at least twice (an interleaved split such as `s0, s2 → M0` crosses four
  times). Plan 1 uses the minimum over projections, `k(M0, M1) = 2`.
- *Plan 1 pruning* (4.4, later, optional; not in the first version): drop M0–M1 seat pairs
  with fewer than 2 links in total. The slice-level form (2 links on distinct slices on both
  sides) needs each slice's ASICs, so it applies only where the layout is hard-pinned;
  otherwise it waits for the exact check. Without it, the exact check rejects the same seat
  pairs one seating at a time.

**Per seating.**

1. **Seat.** SAT returns one seat per mesh. Once Plan 1 lands, only seat pairs that survived
   pruning are offered.
2. **Intra-mesh.** Fills `full.fabric_node_to_asic`. Unchanged; a failure forbids the seat.
3. **C1.** Candidates: the two M0–M1 cables and the routed intra-mesh links. Slices: two 4x2
   slices per mesh, `M0a`, `M0b`, `M1a`, `M1b`. The planned carve finds the 4-cycle and gives
   `s0 → M0@(0,0)`, `s1 → M0@(0,2)`, `s2 → M1@(0,2)`, `s3 → M1@(0,0)`; per-rank tiling gives 4x2
   tiles, two per mesh.
4. **C2.** The resolver places the ring on the planned-carve graph. `R` holds the two cables,
   one per crossing edge: 2 links, demand 2, fits.
5. **C3.** Emit the mapping and plan; publish `R`; phase 0 binds it.
6. **C4**, had it failed. In Case B (both cables on one slice pair) the planned carve yields no
   graph and the tiling graph fails in C2. With host-to-rank free, the exit-node retry asks the
   intra-mesh solve to put one cable end under the other slice; if that is impossible or the
   retries run out, `exclude_mapping(seating)`.

#### `mixed4` through the loop on pods P, Q, R

Before the first solve, Tier B passed with `k(M0, M1) = 2` (2.2). With `unique_shapes` off,
the master solve can offer (M0, M1) = (P, Q), (Q, P), (P, R) and (R, P). Suppose it offers
them in that order. In the first version there is no Plan 1, so when the M0–M1 channel
policy is RELAXED the master can also offer (Q, R) and (R, Q), which have one link. The exact
check rejects each of them (no carve has two crossings) and excludes it, which costs one
iteration each. Plan 1 (4.4), once added, drops the Q–R seat edge so they are never
offered.

| Iteration | Seating | Intra-mesh | Tier C | Next |
|---|---|---|---|---|
| 1 | (P, Q) | success; cables on A0–B0, A3–B3 | planned carve `no-cycle-of-length-6` (3.2, example B); every tiling fails (3.4) | no exit-node retry can help (the hosts form a fixed line, so the cables stay at its ends); `exclude_mapping({M0 -> P, M1 -> Q})` |
| 2 | (Q, P) | success; the same cables with the meshes swapped | same geometry, same failures | `exclude_mapping({M0 -> Q, M1 -> P})` |
| 3 | (P, R) | success; cables on A0–B0, A2–B2 | planned carve finds the 6-cycle (3.2, example A); resolver places the ring (3.1); `R` fits the demand of 2 (4.3) | emit |

**What the intra-mesh solve constrains** (`complete_intra_mesh_for_placement`, per mesh, on
each seating):

| Constraint | Function | On (P, R) |
|---|---|---|
| rank binding (hard) | `add_rank_binding_constraints` | M0's host-rank-i fabric nodes map to P.hi's ASICs; M1's to R.hi's |
| exit nodes (hard) | `add_exit_node_constraints` | the MGD connection is mesh-level, so one cardinality constraint: at least 2 of M0's nodes sit on ASICs cabled toward R, capped by the physical link total under RELAXED. P has exactly two such ASICs, on h0 and h2 |
| MGD pins (hard) | `add_pinning_constraints` | none declared |
| PGD layout (soft) | `add_pgd_pinning_preferred_constraints` | biases toward the PGD layout; this is what puts host rank i on host hi |

A failure here is an intra-mesh failure and keeps today's `add_forbidden_constraint`. It
is the solved layout, not the seat, that tells Tier C which slice each cable touches.

**Per-step outputs on iteration 3:**

1. **Seat.** M0 on P, M1 on R.
2. **Intra-mesh.** `full.fabric_node_to_asic` for both meshes.
3. **C1.** `collect_inter_mesh_candidates` returns the two P–R cables (3.3); the routed
   intra-mesh links give the two lines. `get_coord_range(M0, host_rank)` gives A0..A3
   (columns 0–1, 2–3, 4–5, 6–7), likewise B0..B3; adjacency is the lines plus A0–B0 and
   A2–B2. The planned carve returns six slots (3.2, example A).
4. **C2.** The resolver gives s0 A0, s1 A1+A2, s2 B1+B2, s3 B0 (3.1). `R` = {A2–B2 for
   `s1 -> s2`, A0–B0 for `s3 -> s0`}: 2 links, demand 2, fits.
5. **C3.** The mapping, the plan and the layout; `TT_PIPELINE_STAGE` per rank (4.6); `R` is
   published and phase 0 binds it.

**Edge cases in this loop.**

- **Feedback is about seats.** After iteration 1, `add_forbidden_constraint(M0 on P)` would
  have blocked iteration 3, which succeeds with M0 on P. `exclude_mapping` blocks only the
  pair of seats (4.4).
- **The swap matters in general.** Iteration 2 fails here only because the layout is the
  same. With `unique_shapes` on, it would never be offered; that is harmless in this example,
  but wrong when the two meshes' pins or rank bindings differ (4.4).
- **RELAXED (later): a RELAXED loopback accepts iteration 1.** With `s3 -> s0` RELAXED, `mixed4` on (P, Q)
  passes with the loopback over host (3.4), so the loop would stop there.
- **`pod8`** takes two iterations: (P, U) fails in the resolver because the cables land on
  opposite corners of M1, and (P, V) passes (3.1).

### 4.6 Writing the stage plan into rank bindings (written, not used)

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

For `mixed4` on (P, R) the entries would be:

| Rank | Slice | `TT_PIPELINE_STAGE` |
|---|---|---|
| 0 | A0 | `s0` |
| 1, 2 | A1, A2 | `s1` (co-owned) |
| 3 | A3 | `idle` |
| 4 | B0 | `s3` |
| 5, 6 | B1, B2 | `s2` (co-owned) |
| 7 | B3 | `idle` |

If blaze's runtime carve picked a different alignment of the same cycle (3.2, example A),
for example s0 on B0, every entry would differ while the mapping stays the same. That is
why the entry is a witness, not an instruction.

---

### 4.7 Without a pipeline descriptor: every hook off

Rule: an MGD with no `pipeline_descriptors` (after merge) maps, binds and writes rank
bindings exactly as today, with no extra search, no extra SAT clauses and no changed
setting. Generated MGDs (`generate_mesh_graph_descriptor_of_shape`) have no pipelines, so
they are always off.

**One source of truth.** Hooks do not each re-check the proto. There is one gate:

```cpp
// MeshGraphDescriptor (merged)
bool has_pipeline_descriptors() const;                                 // proto_->pipeline_descriptors_size() > 0
const PipelineMappingConstraints* pipeline_mapping_constraints() const; // null when off

// TopologyMapper, set once in the constructor from mesh_graph_.get_mesh_graph_descriptor()
const PipelineFeasibility* pipeline_ = nullptr;                         // null when off
```

Every hook below takes that pointer (or `has_pipeline_descriptors()` where only a setting
changes) and returns early, or keeps today's value, when it is null. A null pointer means the
hook adds nothing; it never means "empty pipeline, run with defaults".

| Hook | Where | When off |
|---|---|---|
| schema `pipeline_descriptors` (1.3) | MGD proto | absent field; existing textprotos parse unchanged |
| merge (1.5) | `MeshGraphDescriptor::merge` | merged proto byte-identical to today's |
| Tier A (2.1) | MGD constructor | loops over zero pipelines |
| Tier B (2.2) | MGD constructor | `PipelineFeasibility` not built; `pipeline_mapping_constraints()` is null |
| `stage_to_mesh_constraints`, `apply_pipeline_seat_constraints`, `apply_pipeline_exit_constraints` (2.2, 4.4) | mapper helpers | return without adding a constraint |
| seat-graph edge drop for `k_AB` (4.4) | `build_sat_placement_seat_graph` | today's `saturated_link_count` edges |
| `unique_shapes` (4.4) | `map_multi_mesh_to_physical`, `generate_rank_bindings --all-solutions` | `true`, as today |
| C1 to C4 (section 4) | `MultiMeshSolutionEnumerator::next` | loop is today's; no `pipeline_*` failure stage can be reported |
| exit-node retry (4.4) | `complete_intra_mesh_for_placement` | never called a second time for one seating; no joint-placement block added |
| required links `R` and the budget check (4.3) | mapper | `R` empty; no budget check |
| phase 0 (4.3) | `pair_logical_intermesh_ports` | no `required` channel, so today's phases |
| spread fill (4.3) | fill order | the policy plan's `(tier, chip id)` order |
| single-host phase 0 (4.3) | `generate_intermesh_connections_on_local_host` | nothing required, so today's greedy (with the policy plan's table filter) |
| stage plan in rank bindings (4.6) | `generate_rank_bindings` | no stage fields written |

**Not gated, and must be a pure refactor.** The candidate table (policy plan, part 9) runs
for every MGD; that plan's parity tests show identical `AnnotatedIntermeshConnections`. If P9
needs the interim gather helper (3.3), it lands as its own change with the same golden
tests before any pipeline logic uses it.

**Kill switch (proposed).** A runtime option, for example
`TT_METAL_FABRIC_IGNORE_PIPELINE_DESCRIPTORS=1` in `rtoptions`, makes the mapper treat the
descriptor as having no pipelines: Tier B and every mapper hook are off, Tier A still parses
and validates the format. It is an escape hatch for a Tier B false reject or a mapper bug, and
it gives an A/B comparison on the same MGD. Blaze (Tier D) is unaffected either way; it carves
at runtime as today.

**Tests.**

- For every MGD in the repo (none has pipelines today): mapping, rank bindings and
  `AnnotatedIntermeshConnections` match main exactly, on the single-host and multi-host
  paths.
- A counter (or a test-only hook) confirms no pipeline code runs without a descriptor: no
  `PipelineFeasibility`, no extra SAT clauses, no exact-check calls.
- A pipeline MGD with the kill switch set gives the same mapping as the same MGD with its
  `pipeline_descriptors` removed.

## 5. Stage-to-rank mapping

### 5.1 How it is set today

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

### 5.2 In this design

- The mapper only accepts a mapping once the C++ carve planner has produced a stage-to-slot
  plan for it and the resolver has laid it out (section 4). That plan is the stage-to-rank
  mapping: each stage, its slices, hence its ranks, and its entry and exit chips.
- Each rank's stage is recorded in `rank_bindings.yaml` as `TT_PIPELINE_STAGE` (4.6) but not
  consumed.
- At runtime blaze carves again, by planned carve or uniform tiling depending on how it is
  called, and may pick a different placement than the mapper's witness. That is accepted
  (2.3): the mapper only guarantees that a placement exists on the mapping. If
  `TT_PIPELINE_STAGE` is ever consumed, the two would have to be made to agree first.

#### Example

`mixed4` on (P, R):

| Step | Who decides | Result |
|---|---|---|
| rank to slice | the topology mapper (rank bindings and intra-mesh solve) | rank i drives Ai on M0, rank 4 + i drives Bi on M1 |
| stage to slices | the carve planner | s0 A0, s1 A1+A2, s2 B2+B1, s3 B0; A3 and B3 are fillers |
| co-ownership | `_collect_submeshes`, then `allgather_end_owners` | ranks 1 and 2 share s1 and ranks 5 and 6 share s2; `allgather_end_owners` picks which co-owner drives the entry chip and which the exit chip |
| idle ranks | filler slots | ranks 3 and 7 carve their own slice and carry no stage |

---

## 6. Open questions

1. **Adjacency parity.** Tier C builds slice adjacency from bound inter-mesh links and routed
   intra-mesh links; runtime uses `get_forwarding_direction`-confirmed neighbours. Confirm the
   two agree, including links with no active routing planes.
2. **Projection count.** For heterogeneous or fork pipelines, how many projections survive
   Tier B in practice? With many survivors the minimum `k_AB` and union of boundary slices
   (4.4) get weak, and more of the work falls to the exact check. Is that affordable, and is
   the optional "find all" run (2.2) worth shipping in the first version, or should it save
   the defaults until a sweep shows Plan 1 would cut many seatings?
3. **Shared seats with different `k`** (4.4): handle generally, or reject until a real
   descriptor needs it.
4. **Seat-pair nogoods** (4.4): partial `block` does not work today, in either
   `unique_shapes` mode. Design the seat-level API, and decide when a failure is provably
   about one pair of seats rather than the whole seating.
5. **Cost of `unique_shapes` off** (4.4): how many extra seatings the master solve yields on
   the large symmetric cases once mesh permutations are no longer collapsed.
6. **Planned key name** for the rank-binding `env_overrides` entry (4.6).
7. **Per-stage core counts.** Harnesses can override pipeline-core pools, which the MGD does
   not carry. Add an optional core count to `PipelineStage`, or accept the gap (4, step 6).
8. **Blaze's first-plan commitment** (3.2). Tier C accepts if any cycle resolves, but blaze
   uses only its first. Either teach blaze's planner to try the next cycle when the resolver
   fails, or have Tier C check only blaze's first cycle once the mapper's output is consumed
   (4.6).
9. **Spread fill for the binding** (4.3). Turn it on only with a pipeline descriptor, or
   everywhere once the goldens are regenerated? And should spread ever outrank cable tier?
10. **Ordering against the inter-mesh policy plan** (3.3). Wait for its part 9 (candidate
    table) before P9, or land an interim pure gather helper with the same signature first?
11. **Phase 0 failure before part 11** (4.3). Fail at control-plane init (proposed), or add a
    cheap mapper-side check first (an exit chip in `R` toward two meshes needs two free
    directions)?
12. **Exit-node retry** (4.4). Is `kMaxLayoutRetries = 4` per seating right, and is the
    slice-signature block the right granularity, or should a retry also move chips within a
    slice when C2 failed on chip-level capacity?
13. **Wall-clock cap on C2** (4.2). With no node budget, the only open-ended case is a tight
    mesh with dead slots. Keep a hard-fail cap for it (proposed 10 s, fails the graph with the
    mesh named), or drop it and let such an MGD run until it finishes?
