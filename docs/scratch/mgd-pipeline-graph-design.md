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

### Scope: STRICT first; RELAXED and the policy plan's candidate table later

> **Decision (fixed for the first version):** C1 reads inter-mesh cables straight from the
> PSD (`inter_mesh_cables`, P5, 3.3). It does not wait for the inter-mesh policy plan's
> candidate table (part 9). Switching C1 to that table is a next step (P16, M5), together
> with RELAXED below.

> **Decision (fixed for the first version):** every pipeline edge is **STRICT**. The first
> version accepts an edge `policy` only when it is omitted or `STRICT`, so a pipeline edge always rides a fabric link. RELAXED
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
| Schema (1.3) | `PipelineEdge` has `src` and `dst`; field 3 is reserved | `optional Policy policy = 3`, the enum `channels` already uses; omitted is STRICT, so first-version files keep their meaning |
| Tier A (2.1) | ids, edges, shapes, names | reject an explicit `INVALID_POLICY`; RELAXED is allowed on any edge |
| Tier B (2.2) | every edge that crosses meshes counts against `cap(A, B)`; touching rectangles inside a mesh are a backup if the solve is slow | edges with `needs_fabric = false` (RELAXED, other than stage 0's outgoing edge) are skipped |
| Tier C (2.3, 4) | one footprint solve on the position graph, then the resolver | RELAXED edges become soft target edges in the same solve; one resolver call per fabric chain (3.4) |
| Plan 1 `k_AB` (4.4; later and optional, P10) | every crossing counts | only `needs_fabric` crossings count, so P10's hooks prune less |
| blaze | nothing new | `Edge` gets a policy; `--pipeline-builder flexible` marks every edge RELAXED |
| Models | rings, `llama_8b_pod`, joint stages, forks whose returns are all on fabric | forks with a host return (dflash draft), rings with a host loopback |

**PR plan** (4.0, and in the canvas). Milestones M1 to M4 (P1 to P3, P5, P6, P8 to P10, P12, P13, P15) are STRICT only and
need nothing from M5. M5, Later, holds three PRs:

1. **P14, schema + Tier B + Plan 1.** Un-reserve field 3 as `optional Policy policy`.
   `PipelineDescriptorData::Edge::needs_fabric` marks the edges that must stay on fabric; Tier B, and
   Plan 1's `k_AB` (so P10's constraints into the first solve), skip the rest. Tier C keeps
   checking every edge as fabric until P11: sound, but it misses seatings that need a host
   hop (1.4, "Implementation (P14)").
2. **P11, host cuts in Tier C.** Make edges with `needs_fabric = false` soft target edges in
   the footprint solve, resolve each fabric chain on its own (3.4), and drop P14's
   RELAXED-as-STRICT warning.
3. **P16, C1 on the policy plan's candidate table.** Once part 9 lands, replace
   `inter_mesh_cables` with `collect_inter_mesh_candidates` per seating, so C1 skips down links
   and disallowed cable types and picks `R` by tier (3.3). Closes open question 20.

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
  rank's slice and be co-owned by several ranks (see 5).
- **Fork graphs.** DFlash / DSpark fan out from one stage into several branches, and a
  branch may return over fabric or over host MPI.

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
the pipeline builder's own resolver.

![Proposed flow](https://raw.githubusercontent.com/tenstorrent/tt-metal/ridvan/mgd-ticket-52016-diagrams/docs/scratch/ticket-52016-diagrams/4-proposed-flow.png)

The Tier C loop in this flow (one position graph, one carve, resolver check, `R`) is drawn in the canvas's loop diagram and in section 4.

### Big-picture steps

1. **MGD pipeline descriptor format** - section 1.
2. **Verification tiers** - section 2. Tiers A and B run on the MGD alone; Tier C runs inside
   the topology mapper; Tier D is the existing pipeline builder and is not changed.
3. **Shared code** - section 3. Reuse the C++ graph-layout resolver as is, keep blaze's carve
   planner as the runtime reference only, and read inter-mesh cables from the PSD (the
   inter-mesh policy plan's candidate table later, P16).
4. **Pipeline mapping in the topology mapper (Tier C)** - section 4, in four steps after
   intra-mesh completion: build one position graph from host slices and candidate cables
   (C1, 4.1), solve the pipeline onto it and check the result (C2, 4.2), on success emit and publish the cables to
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
and Tier C makes sure the cable it uses is the one the control plane binds** (2.2, 2.3, 4.3). Connections
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
  stages { id: 0 label: "Embedding" shape { dims: [ 4, 2 ] } }
  stages { id: 1 label: "Windowed"  shape { dims: [ 4, 2 ] } }
  stages { id: 2 label: "Global"    shape { dims: [ 4, 2 ] } }
  stages { id: 3 label: "LMHead"    shape { dims: [ 4, 2 ] } }
  edges { src: 0 dst: 1 }
  edges { src: 1 dst: 2 }
  edges { src: 2 dst: 3 }
  edges { src: 3 dst: 0 loopback: true }   # closes the ring
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

One field the builder sets stays out of MGD, since it does not change the shape being mapped:

- **`factory`** - host-side construction (`GenericStage` over the harness list).

`is_loopback` is in MGD, as `loopback: true` on the edge that closes a cycle (`s3 -> s0`
above). The resolver orders stages along the other edges and places host I/O from the mark
(4.2), so the descriptor says which edge it is rather than leaving Tier C to guess. Tier A
checks the marks (2.1), and Tier C copies them into the resolver's edges.

#### A heterogeneous example

The llama_8b_pod graph (`_build_heterogeneous_two_mesh_graph` in
`tests/pipeline_builder/test_pipeline_builder_infra.py`) is 40 stages: 8 of `4x2` on M0 plus
32 of `1x2` on M1, chained with the ring closing back to `s0`. Abbreviated:

```textproto
pipeline_descriptors {
  name: "llama_8b_pod"
  stages { id: 0  label: "Embedding" shape { dims: [ 4, 2 ] } }
  stages { id: 1  label: "LMHead"    shape { dims: [ 4, 2 ] } }
  stages { id: 2  label: "Token4x2"  shape { dims: [ 4, 2 ] } }
  # ... s3 .. s7 likewise 4x2 ...
  stages { id: 8  label: "Token1x2"  shape { dims: [ 1, 2 ] } }
  # ... s9 .. s39 likewise 1x2 ...
  edges { src: 0 dst: 1 }
  # ... linear chain ...
  edges { src: 39 dst: 0 }
}
```

Shapes vary per stage, so `shape` lives on the stage. Nothing says which mesh the `4x2`
stages land on; the binding set (2.2) decides.

#### A joint-stage example

A `4x4` LM head over two `4x2` host slices is one stage co-owned by two ranks:

```textproto
stages { id: 7 label: "LMHead" shape { dims: [ 4, 4 ] } }   # 2 slices of 4x2, 2 ranks
```

#### A fork example

One stage with a target branch and a draft branch, both returning to that stage, mirroring
`build_gpt_oss_dflash_router_pipeline_graph`. In the first version both returns are STRICT.
**RELAXED (later):** the draft return may be marked `policy: RELAXED` to go over host MPI, as
in the comment on its last edge.

```textproto
pipeline_descriptors {
  name: "gpt_oss_dflash_router"
  stages { id: 0  label: "router"  shape { dims: [ 4, 2 ] } }
  stages { id: 1  label: "target0" shape { dims: [ 4, 2 ] } }
  # ... 2 .. 38: target1 .. target37 ...
  stages { id: 39 label: "draft0"  shape { dims: [ 4, 2 ] } }
  # ... 40 .. 49: draft1 .. draft10 ...
  edges { src: 0  dst: 1 }    # router -> target0
  # ... target chain ...
  edges { src: 38 dst: 0 }    # target37 -> router
  edges { src: 0  dst: 39 }   # router -> draft0
  # ... draft chain ...
  edges { src: 49 dst: 0 }    # draft10 -> router; RELAXED (later): policy: RELAXED to return over host
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
  uint32 id = 1;             // unique within the pipeline, referenced by edges
  string label = 2;          // human-facing, unconstrained, may repeat
  TorusTopology shape = 3;   // required, rank 2, all dims > 0
}

message PipelineEdge {
  uint32 src = 1;            // resolves to a stage id
  uint32 dst = 2;            // resolves to a stage id
  optional Policy policy = 3; // omitted or STRICT; RELAXED is rejected until P14
  bool loopback = 4;          // closes a cycle; the other edges must have none
}
```

Stage ids are integers, like `mesh_id` and `graph_id` elsewhere in the MGD. The stored record
uses `PipelineStageId` (`uint32_t`, in `fabric_types.hpp` beside `MeshId`). An omitted id is 0, so a second stage without one
repeats it and Tier A rejects the pipeline. The label carries the human name. In prose, `sN`
means the stage with id N (`s3 -> s0` is the edge `src: 3 dst: 0`).

`shape` is rank 2 because the resolver models a chip as `(row, col)`. Every edge must ride
the fabric. The edge that closes a cycle is marked `loopback: true`; edges carry no channel
count (2.1). An edge may write
`policy: STRICT`; `RELAXED` and `INVALID_POLICY` are rejected by `validate_pipeline_descriptors`.

**RELAXED (later, P14)** keeps this field and stops rejecting `RELAXED`. The field reuses the
`Policy` enum `channels` already has (`INVALID_POLICY`, `STRICT`, `RELAXED`), so there is no new
enum and the textproto spelling is the same. Omitted means STRICT, so first-version files are
unchanged.

None of the comments above are enforced by protobuf. They are obligations on
`validate_pipeline_descriptors`.

### 1.4 Edge policy: STRICT or RELAXED, per edge - RELAXED (later)

> **RELAXED (later).** Not in the first version: every edge is STRICT: the `policy`
> field exists, but Tier A rejects `RELAXED` (see "Scope" at the top). This section is the design for when it is added.

Each edge says whether it must ride the fabric. The words match `channels { policy }`:
STRICT is a requirement, RELAXED a preference. **STRICT is the default:** it is enum value
0, so an edge with no `policy` is STRICT, and a pipeline goes over host only where an edge
opts in with `policy: RELAXED`.

| Policy | Meaning | What the mapper requires |
|---|---|---|
| `STRICT` (default) | carried over a fabric link | a fabric link between the two stages' submeshes. If they are on different meshes, the edge rides a declared connection and counts against its channels |
| `RELAXED` | carried over fabric where the placement allows, otherwise over host MPI | nothing: the edge never needs a link |

```textproto
edges { src: 0 dst: 1 }                                    # STRICT
edges { src: 1 dst: 2 }                                    # STRICT
edges { src: 2 dst: 3 }                                    # STRICT
edges { src: 3 dst: 0 policy: RELAXED loopback: true }     # the loopback may go over host
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
  crossing out. Tier C keeps these edges hard even when RELAXED, because blaze's host-cut planner never
  cuts them:
  - *stage 0's outgoing edge* in a ring. `_group_walk` (`host_edge_planner.py`) skips any
    alignment with `0 in cuts`: stage 0's rank drives inference and cannot run a host-hop pump
    (`test_pipeline_builder_flexible_mode.py`).

  > **Note: rules that might be good to add.** Once MGD stages map onto blaze stages, Tier A
  > could reject RELAXED where blaze can never carry the edge over host: stage 0's outgoing
  > edge, and any edge of an open chain (blaze raises
  > `NotImplementedError("host edges are only supported on a looped-back ring")`,
  > `graph.py`). Tier B and Plan 1 would then count those edges as STRICT, which prunes more.

What each step does with the policy:

| Step | STRICT edge | RELAXED edge |
|---|---|---|
| Tier A (2.1) | - | - |
| Tier B crossing rules (2.2) | a crossing must ride a declared connection and fit its channels | left out |
| Plan 1 `k_AB` (4.4) | counted | not counted |
| Tier C (2.3, 3.4) | a hard target edge: must be a fabric link in the layout | a soft target edge: the solve may leave it to host; each fabric chain is then resolved on its own |

A pipeline whose edges are all STRICT has only hard target edges; that is every pipeline in
the first version. One with a RELAXED edge gets soft edges in the same solve (3.4, "Host cuts
on RELAXED edges").

#### Implementation (P14)

P14 lets the existing field say RELAXED and makes the descriptor-only tiers use it. Tier C does not change
until P11.

**One rule for "must stay on fabric".** `PipelineDescriptorData::Edge` (in `mesh_graph_descriptor.hpp`) gains
`relaxed` and `needs_fabric`, set once in `populate_pipeline_descriptors`: `needs_fabric` is true for a STRICT edge, and
for a RELAXED edge out of stage 0, which blaze never cuts (above). Tier B, Plan 1 and later
Tier C (P11) all read this one flag, so they agree on which edges may go over host. Counting
stage 0's outgoing edge as fabric is still sound, because P11 keeps it hard.

**Tier B needs one filter.** Every Tier B rule reads the same neighbour lists, so building
them from `needs_fabric` edges only covers the crossing rule, the channel count, the
look-ahead and the carve adjacency.

**The constraints passed into the first solve follow.** `min_crossings` (`k_AB`),
`min_crossings_out` and `boundary_slices` come from the same filtered search, so P10's
`ExtraMappingConstraints` hooks, which carry those MGD facts into the first SAT placement and
intra-mesh solves, need no change; they just prune less.

**Tier C until P11: RELAXED is checked as STRICT.** The check is unchanged: every edge is a
fabric edge. That is sound (an accepted seating carries every edge on fabric, so it is
buildable) but it misses seatings that need a host hop. A pipeline with RELAXED edges logs one
warning, and a failed run says so. The earlier plan rejected every such pipeline
(`pipeline_relaxed_unsupported`); checking it as STRICT accepts the ones that fit on fabric
anyway, and P11 only widens what is accepted.

```proto
 // Every edge rides a fabric link. policy may be omitted or STRICT. RELAXED is
-// rejected by MeshGraphDescriptor validation.
+// An edge may set policy: RELAXED to say it may go over host MPI when the
+// placement has no fabric link for it. Omitted means STRICT.
 ...
 message PipelineEdge {
     ...
     uint32 dst = 2;

-    // Omitted means STRICT. RELAXED is rejected during descriptor validation.
+    // Optional; omitted is STRICT (same enum and default as Channels.policy).
+    // STRICT: the edge rides a fabric link. RELAXED: fabric where the placement
+    // allows, otherwise host MPI. INVALID_POLICY is rejected.
     optional Policy policy = 3;

     bool loopback = 4;
 }
```

```cpp
 // validate_pipeline_descriptors (Tier A), in the edge loop: RELAXED is allowed now.
-            // An omitted policy means STRICT. An explicit policy other than STRICT is rejected.
-            if (edge.has_policy() && edge.policy() != proto::Policy::STRICT) {
+            if (edge.has_policy() && edge.policy() == proto::Policy::INVALID_POLICY) {
                 ...
             }

 // populate_pipeline_descriptors:
-            record.edges.push_back(
-                PipelineDescriptorData::Edge{.src = edge.src(), .dst = edge.dst(), .loopback = edge.loopback()});
+            const bool relaxed = edge.has_policy() && edge.policy() == proto::Policy::RELAXED;
+            record.edges.push_back(PipelineDescriptorData::Edge{
+                .src = edge.src(),
+                .dst = edge.dst(),
+                .loopback = edge.loopback(),
+                .relaxed = relaxed,
+                .needs_fabric = !relaxed || edge.src() == host_io_stage});   // the one root Tier A found

 // mesh_graph_descriptor.hpp (proto::Policy is only forward-declared there, so a bool):
-// Edges stay directional pairs of stage ids, each with its loopback mark. Omitted and explicit
-// STRICT both survive validation, so a stored edge has no policy of its own.
+// Edges stay directional pairs of stage ids, each with its loopback mark. relaxed is
+// policy: RELAXED; omitted is STRICT.
 struct PipelineDescriptorData {
     struct Edge {
         PipelineStageId src = 0;
         PipelineStageId dst = 0;
         bool loopback = false;  // closes a cycle; the other edges have none
+        bool relaxed = false;
+        // STRICT, or RELAXED out of stage 0: blaze's host-cut planner never cuts that
+        // (_group_walk skips cuts at stage 0). The one rule Tier B, Plan 1 and (P11) Tier C all read.
+        bool needs_fabric = true;
     };
```

```cpp
 // mesh_graph_descriptor.cpp, check_pipeline_feasibility (Tier B): neighbour copies built once.
 // Every rule reads them, so one filter here covers the crossing rule, the channel count and, in
 // the backup plan, the look-ahead, the carve adjacency and the Plan 1 facts.
 for (const PipelineDescriptorData& pipeline : pipeline_descriptors_) {
     for (const PipelineDescriptorData::Edge& edge : pipeline.edges) {
+        if (!edge.needs_fabric) {
+            continue;   // may go over host: no crossing, no channel, no adjacency, not in k_AB
+        }
         // one neighbour copy each way, as today
     }
 }

 // P10 (if landed) needs no change: pipeline_extra_constraints reads min_crossings, min_crossings_out and
 // boundary_slices, which are now computed over needs_fabric edges only, so the constraints passed from
 // the MGD into the first solve shrink with them. Every edge RELAXED (except stage 0's outgoing edge) gives k = 0.
```

```cpp
 // topology_mapper_utils.cpp, MultiMeshSolutionEnumerator constructor (P8), once per run.
 // Until P11, Tier C checks every edge as a fabric edge. Sound: an accepted seating carries
 // every edge on fabric, so it is buildable; it can miss seatings that need a host hop.
+    for (const PipelineDescriptorData& pipeline : mesh_graph_descriptor_->pipeline_descriptors()) {
+        const auto count = std::ranges::count_if(pipeline.edges, &PipelineDescriptorData::Edge::relaxed);
+        if (count > 0) {
+            log_warning(tt::LogFabric,
+                        "pipeline '{}': {} RELAXED edge(s) are checked as STRICT until host cuts are supported",
+                        pipeline.name, count);
+            relaxed_checked_as_strict_ = true;
+        }
+    }

 // MultiMeshSolutionEnumerator::next, give_up(): say why a RELAXED pipeline found nothing.
+    if (relaxed_checked_as_strict_) {
+        last_failed_->error_message += " (RELAXED pipeline edges are checked as STRICT until host cuts are supported)";
+    }
```

**Tests** (`test_mesh_graph_descriptor.cpp`, Tier B and Tier C suites):

- `EdgePolicyDoesNotParse` becomes `EdgePolicyParses`: RELAXED stores `relaxed = true`;
  STRICT and omitted store false. `policy: INVALID_POLICY` is rejected by Tier A.
- `UnknownFieldElsewhereStillParses` and `ShippedMeshGraphDescriptorsStillParse` unchanged;
  every earlier golden gives the same Tier B result.
- `needs_fabric`: a ring with `s3 -> s0` RELAXED gives false; `s0 -> s1` RELAXED gives true.
- Tier B: the RELAXED rows of the examples in 2.2; `mesh4x4_stage4x2` with `s3 -> s0` RELAXED
  gives `k(M0, M1) = 1`.
- Tier C: a RELAXED pipeline accepts exactly what its all-STRICT copy accepts, with one warning.
- Merge keeps the policy.

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

**Several pipelines: separate in the MGD, one graph in the mapper, one solve.** The
descriptor keeps them separate. `pipeline_descriptors` is a repeated field, one
`PipelineDescriptorData` per pipeline, and Tier A checks each record on its own. Stage ids
are unique only inside that record, so two pipelines may both have a stage 0. An MGD may
declare more than one, and merging parts that each declare one appends those records. The
parts' pipelines stay disconnected, because no part can name another part's stages. Merge
does not join them.

The topology mapper does not place that list one pipeline at a time.
`build_logical_multi_mesh_adjacency_graph` turns every record of the merged MGD into **one
target graph**, `LogicalMultiMeshGraph::pipeline_stage_graph_`, and that graph is the
representation C1 and C2 solve:

- A node is a `PipelineStageNode`, the pipeline's name and the stage's id, so two stage 0s are
  two nodes. There are no edges between pipelines, so each pipeline is one component of the
  same graph.
- One seating builds one position graph for that target. One footprint solve places every
  stage of every pipeline onto disjoint slices. Two pipelines that each fit alone can still
  fail together, because they compete for the same slices and the same cables.
- The mapping that comes back is split per pipeline
  (`PipelineMappingResult::pipelines`). The chip-level resolver runs once per component on
  the carve that single solve returned. Those calls check the carve; they are not a second
  placement (3.4, "Several pipelines").

On that one graph:

- **Names.** Pipeline names must be unique across the merged MGD; a clash is a merge error
  that names both parts. In the combined graph a stage is `<pipeline>/<stage id>`, so stage
  ids only need to be unique within their pipeline.
- **Tier A** checks each pipeline on its own (2.1).
- **Tier B** counts capacity over all pipelines together, and a projection places every
  stage of every pipeline at once. Pipelines in one descriptor compete for the same slices, so
  two that each fit alone can fail together (`mixed4` plus `mixed4_b`, 2.2). Parts are
  different: each part already fit its own meshes, and merge keeps those meshes and their
  connections, so the parts' assignments together still fit and the merged check cannot reject
  on counts. Tier C can still reject, because the parts share seats and cables.
- **Plan 1** takes `k_AB` over the combined projections, so it sums the STRICT crossings of
  every pipeline (4.4).

A part's pipeline still binds across the whole merged fabric, not just to that part's
meshes. Byte-identical copies of one pipeline (the same file in two parts) clash on the name
and are rejected; relax that only if a real setup needs it.

**Validation order.** Every `MeshGraphDescriptor` constructor runs Tier A (in `static_validate`)
and Tier B (after `populate`). Loading a part therefore checks that part against its own meshes,
before merge. `merge` returns a new `MeshGraphDescriptor`, so both tiers run again on the merged
meshes and every pipeline together. That second Tier B run cannot reject a merge of parts that
each passed (above), unless a part only passed because its search ran out of steps. A pipeline
that does not fit the meshes in its own file fails when that file is loaded. Tier B stores nothing for the mapper:
`init_from_parts` does not run Tier B itself, and neither does `MeshGraph`.

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
> Several checks below are pipeline builder rules: a stage is a rectangle of whole slices on
> one mesh, and every stage fits its core budget.
> Tiers A and B check only necessary conditions that follow from the shapes. Tier C calls the
> builder's own C++ resolver as is (3.1), so the layout rules live in one place. The carve is
> the footprint solve's (4.2); blaze's Python carve planner stays the runtime reference (3.2).

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
  stages { id: 0 shape { dims: [ 4, 2 ] } }   # 1 slice
  stages { id: 1 shape { dims: [ 4, 4 ] } }   # 2 slices, two ranks
  stages { id: 2 shape { dims: [ 4, 4 ] } }   # 2 slices, two ranks
  stages { id: 3 shape { dims: [ 4, 2 ] } }   # 1 slice
  edges { src: 0 dst: 1 }  edges { src: 1 dst: 2 }
  edges { src: 2 dst: 3 }  edges { src: 3 dst: 0 loopback: true }
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
  stages { id: 0 shape { dims: [ 4, 2 ] } }   # 1 slice of Big
  stages { id: 1 shape { dims: [ 4, 2 ] } }
  stages { id: 2 shape { dims: [ 4, 2 ] } }
  stages { id: 3 shape { dims: [ 4, 2 ] } }
  stages { id: 4 shape { dims: [ 1, 2 ] } }   # 1 slice of Small
  stages { id: 5 shape { dims: [ 1, 2 ] } }
  stages { id: 6 shape { dims: [ 1, 2 ] } }
  stages { id: 7 shape { dims: [ 1, 2 ] } }
  edges { src: 0 dst: 1 }  edges { src: 1 dst: 2 }
  edges { src: 2 dst: 3 }  edges { src: 3 dst: 4 }
  edges { src: 4 dst: 5 }  edges { src: 5 dst: 6 }
  edges { src: 6 dst: 7 }  edges { src: 7 dst: 0 loopback: true }
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

Both run inside every `MeshGraphDescriptor` constructor, and again on the merged descriptor
(1.5). Neither looks at the physical machine.

| | Tier A (2.1) | Tier B (2.2) |
|---|---|---|
| runs in | `static_validate`, on the proto alone | after `populate`, with the meshes, rank grids and connections |
| asks | is the pipeline text a valid descriptor? | can these stages fit on the meshes this descriptor declares? |
| checks | ids unique and resolved, no self-edges, loopback marks (the other edges acyclic, each loopback closes a cycle), one host I/O stage, positive shapes, unique pipeline names | binding sets, slice capacity, crossings ride declared connections within their channels (the per-mesh carve is a backup if the solve is slow) |
| `[2, 4]` stage on 4x2 slices | pass: two positive numbers | reject: no mesh's slice tiles it (no transpose) |
| two copies of `mixed4` | pass: each is well-formed | reject: 12 slices needed, 8 declared |

Tier B keeps going until one assignment of stages to meshes survives, and chooses none. On
`mixed4`, `s0, s1 -> M0` and `s2, s3 -> M1` survives (3 + 3 slices, 2 crossings on 2
channels); `s0, s2` against `s1, s3` crosses four times and is dropped.

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

- **No self-edges.** `edges { src: 0 dst: 0 }` is rejected. No submesh is adjacent to
  itself, and the resolver would otherwise report a misleading cycle error.
- **Repeating an edge declares channels.** Edges are directional; `s0 -> s1` twice is two
  channels on that hop, and `s0 -> s1` plus `s1 -> s0 loopback: true` is a two-stage ring. Multiplicity is
  repetition, so edges need no `channels` field. The builder reads one link per edge today;
  that is an implementation limit, not a schema one.
- **A cycle needs a loopback edge.** The edges not marked `loopback: true` must have no
  cycle, because the resolver orders stages along them and throws otherwise. The error names
  the cycle: `pipeline 'p': cycle s0 -> s1 -> s2 -> s0 has no loopback edge; mark one of its
  edges loopback: true`. Blaze's ring marks `s{n-1} -> s0`; a fork marks each return.
- **A loopback edge must close a cycle.** Its `dst` must reach its `src` over the edges not
  marked loopback; otherwise `pipeline 'p': loopback edge s0 -> s1 closes no cycle; s1 does not
  reach s0 over the other edges`. Which edge of a cycle is marked is not checked (the resolver
  takes any), and a repeated loopback edge is another channel. Tier B counts loopback edges as
  channels like any other edge.
- **One host I/O stage.** Exactly one stage has no incoming edge besides loopbacks. It owns
  host I/O, and the resolver starts its stage order there; with two such stages the resolver
  would pick whichever id sorts first. Two chains, or a stage with no edges beside a chain,
  are rejected: `pipeline 'p': stages 0, 2 have no incoming edge besides loopbacks;
  exactly one may, the stage that owns host I/O`. In Tier C it is the resolver's
  `layout.stage_order.front()` (4).
- **Otherwise stage topology is not limited.** In-degree, out-degree, and whether a branch
  is a chain are all accepted. A graph blaze cannot build still passes Tier A and fails at
  build time. A branch may dead-end.
- **Every edge is STRICT.** `policy` may be omitted or `STRICT`; Tier A rejects anything else. RELAXED (later): any edge may be RELAXED, and Tier A will not restrict where; see
  the note in 1.4 for rules that might be good to add.

#### Examples

| Descriptor | Tier A result | Why |
|---|---|---|
| `mixed4` | pass | ids unique, edges name declared stages, every stage has out-degree 1, one cycle (closed by the loopback `s3 -> s0`), shapes positive |
| `pod8` | pass | a ring of 8, closed by the loopback `s7 -> s0` |
| `mixed4` with `s3 -> s0` not marked loopback | reject | the cycle `s0 -> s1 -> s2 -> s3 -> s0` has no loopback edge |
| `mixed4` with `s1 -> s2` marked loopback instead of `s3 -> s0` | pass | any edge of the cycle may be the loopback |
| a chain `s0 -> s1` with that edge marked loopback | reject | the loopback closes no cycle |
| `mixed4` plus `edges { src: 1 dst: 1 }` | reject | self-edge |
| `mixed4` with `s0 -> s1` repeated | pass | two channels on that hop; the builder reads one link per edge, which is a builder limit |
| dflash: `router -> target0 .. target37 -> router`, `router -> draft0 .. draft10 -> router`, both returns marked loopback | pass | ids unique, edges name declared stages, shapes positive, each cycle has its loopback |
| the same dflash with the draft return not marked | reject | the cycle `router -> draft0 .. draft10 -> router` has no loopback edge |
| edges `s0 -> s1`, `s0 -> s2`, `s1 -> s0` (loopback); `s2` has no out-edge | pass | a branch may dead-end; Tier A does not limit stage topology |
| a stage with three outgoing branches | pass | Tier A does not limit stage topology |
| a branch stage with out-degree 2 (a fork inside a branch) | pass | Tier A does not limit stage topology |
| two stages with in-degree 0 (two chains, or a stage with no edges beside a chain) | reject | exactly one stage may have no incoming edge besides loopbacks; it owns host I/O |
| `mixed4` with `policy: RELAXED` on `s3 -> s0` | reject | `RELAXED` is rejected by Tier A in the first version |
| `mixed4` plus a copy named `mixed4_b` | pass here | each is checked on its own; Tier B checks them together and rejects on capacity (1.5) |
| `mixed4` plus a copy also named `mixed4` | reject | pipeline names must be unique (1.5) |
| a stage declared `[2, 4]` where every mesh has 4x2 slices | pass here | shapes are logical and never transposed, so Tier B finds no mesh it binds to |

**RELAXED (later), Tier A examples.**

| Descriptor | Tier A result | Why |
|---|---|---|
| dflash with the draft return RELAXED | pass | any edge may be RELAXED |
| `mixed4` with `s3 -> s0` RELAXED | pass | any edge may be RELAXED |
| `mixed4` with `s0 -> s1` RELAXED | pass | allowed; Tier C still keeps stage 0's outgoing edge a hard edge (1.4) |
| dflash with `router -> draft0` RELAXED | pass | any edge may be RELAXED |

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
- **On failure:** the constructor fails. On success nothing is stored.

These reject pipelines that are impossible on *any* mapping of this MGD.

> **What the first version runs.** Counts and channels only:
>
> - **Binding** (step 1): the meshes each stage fits.
> - **Capacity** (step 2): each stage's smallest slice count, summed, against the declared slices, and the
>   stages that fit only one mesh against that mesh.
> - **Projection search** (step 3): some choice of one mesh per stage fits every mesh's slices, and the
>   crossings fit the declared channels. It stops at a step limit and then accepts.
>
> Where a stage sits inside its mesh is left to the SAT solve in Tier C (4.2), which places stages quickly.
> The check is a private method, `MeshGraphDescriptor::check_pipeline_feasibility` in
> `mesh_graph_descriptor.cpp`, and it stores nothing.
>
> **Backup if the Tier C solve is slow (not implemented).** These stay written below so they can be added
> later: the per-mesh segment carve (step 4), interchangeable-mesh symmetry and the look-ahead in step 3,
> the "find all" run, and `PipelineMappingConstraints` with its P10 hooks. Leaving the carve out has one
> cost: a pipeline whose stages cannot be carved on their meshes passes Tier B and fails in Tier C, after
> the mapper has tried the seatings.

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
the search stops at the first, and it stores none: any carve Tier C finds on real slices
already satisfies these rules, so a stored list could never reject anything. The facts Plan 1
(4.4) reads, such as the minimum crossing count, need every projection; they come from an
optional second run of the same search (below).

```
subtorus_4x4_pipeline_8stage: 8 meshes, connections 0-1-2-...-7-0, 8 stages of 4x4

  s_i on mesh i, crossings s_i -> s_{i+1} ride connection (i, i+1)   OK

same MGD, connections 0-1-2-...-7 (no 7-0)                           REJECT
  the s7 -> s0 crossing has no declared connection
```

#### Each mesh's segment can be carved - backup if the solve is slow

**Not in the first version.** Tier C's solve places stages inside each mesh. Add this to Tier B only if
that solve is slow on real descriptors.

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
including joint stages.

#### How the connectivity check runs (plan)

Tier B never looks at chips or cables. It works on two small graphs the MGD already holds:
the **mesh graph** (meshes, with a channel count per mesh pair) and, inside each mesh, the
**slice grid** (the `host_topology` grid, with RING wraps). One search covers every pipeline
in the descriptor at once, because they share slices and channels.

**Inputs, built once per constructor.**

| Input | Where it comes from | Notes |
|---|---|---|
| slices of mesh `m` | `get_declared_topology(m)`: `dims / host_dims`, count `prod(host_dims)` | grid of slices, in rank order |
| slice adjacency in `m` (backup carve only) | neighbours in the `host_dims` grid; a wrap on axis `i` when `ring_dims[i]` and `host_dims[i] > 2` | with `host_dims[i] == 2` the wrap is the edge already there |
| `cap(A, B)` | sum of `count` over the `ConnectionData` of every graph, whatever its type (`FABRIC`, `POD`, `CLUSTER`) | `populate` stores each connection in both directions, so read one direction only. ALL_TO_ALL, RING and manual `connections` all land here. An end that is a graph stands for every mesh under it, so each pair gets the channels; meshes joined through one switch get the smaller of their two counts. Both are upper bounds, and Tier C checks the cables |
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
3. **Symmetry** (backup if slow): see below.

*Look-ahead after each assignment (backup if slow; cheap, sound):*

- every unassigned stage still has a mesh in its binding set with room for it and, if it has
  assigned neighbours, enough channels left toward them;
- the smallest needs of the unassigned stages fit in the free slices left.

*Interchangeable meshes (backup if slow).* Two meshes are interchangeable when they have the same declared
topology (`dims`, `host_dims`, ring axes) and the same `cap` toward every other mesh, so
swapping them maps every projection to another one. Within such a class, an **empty** mesh may
be used only when it is the lowest-numbered empty mesh of its class. This is sound, because a
projection that uses a higher empty mesh has a mirror that uses the lower one. On ALL_TO_ALL
with 8 identical meshes and 8 one-slice stages, it turns 8! = 40320 orderings into one. A ring
of meshes `0-1-...-7-0` has no interchangeable pair (only rotations), but the crossing rule
already prunes it hard.

*Leaf.* When every stage has a mesh, Tier B passes. (Backup if slow: run step 4 on each mesh
that holds stages, memoised, and backtrack if any mesh fails.)

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

**Step 4: segment carve per mesh - backup if the solve is slow** (depth-first, stop at the first
carve). Not in the first version. For each mesh,
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

**A step limit.** Step 3 counts search nodes and stops after 10^6. Running out is not a proof,
so Tier B accepts without a warning and leaves the answer to Tier C; the constructor always
returns. The slow case is a fully used grid of one-slice meshes that must close a ring, a
Hamiltonian-cycle search that is slow when none exists: a ring of 49 stages on a 7x7 grid of
one-slice meshes reaches the limit in about 250 ms. (Backup if slow: with step 4, an odd grid
of one-slice stages holds no ring and can be rejected before searching. Tier B leaves crossing
endpoints open, so the fixed-endpoint segment test, Itai–Papadimitriou–Szwarcfiter, does not
apply here; Tier C runs it once the cable slices are known, 4.2.)

**Cost.** Of the `host_topology` values in the repo's MGDs, about 200 are one slice (nothing
to search), about 60 are 2 to 4 slices, about 15 are 8 to 16, and 8 are 32 (4x8, 8x4, 2x16).
A stage with one mesh in its binding set (as in `llama_8b_pod`) makes no branch. Both
searches run in microseconds to milliseconds on all of them, and step 4 runs once per
distinct stage set on a mesh, not once per projection.

**Why not SAT.** CaDiCaL is already in the build, but the encoding costs more than the
search at these sizes: a variable per (stage, mesh), pseudo-Boolean sums for slice capacity,
per-edge crossing variables with an at-most-`cap` count per mesh pair, and for step 4 a
variable per (stage, position) with no-overlap and touching clauses. A SAT answer also
gives no readable reason. What makes the hard failing carves fail fast is the exact segment
test, which works the same in front of either solver. (Tier C does use SAT, 4.2: there the
positions are fixed by real cables and the solve must also pick the crossings.)

**Optional: facts for Plan 1 - backup if the solve is slow.** Not in the first version. Plan 1 (4.4) needs facts that hold for **every** projection
and carve: the minimum crossings `k_AB` and the union of boundary slices. These need the
full set, not the first survivor, so they come from a second run of the same two searches in
"find all" mode: step 3 continues after each survivor, and step 4 enumerates every carve of a
stage set (memoised, recording for each slice whether some carve gives it to a stage with a
crossing). Interchangeable-mesh symmetry still applies; each found projection stands for its
class, and its mirrors are added by relabelling. This run has its own, smaller budget
(proposed: 10^5 nodes). Over the budget, Tier B saves the always-true defaults, `k = 0` and
every host rank as a boundary slice, which prune nothing. The first version may skip this
run entirely and save the defaults; the exact check (4.2) then catches what Plan 1 would
have pruned, at the cost of more seatings tried.

#### Output: mapping constraints for Tier C - backup if the solve is slow

**Not in the first version.** Tier B saves nothing, and the descriptor has no
`pipeline_mapping_constraints()`. The Tier C solve is fast enough without them. The rest of
this section is the plan if it turns out to be slow on real descriptors.

Tier B already finds facts the mapper can use. It saves them on the descriptor as one
object, `PipelineMappingConstraints`. Helpers in the topology mapper turn that object into the
`MappingConstraints` the mapper already solves with. Every fact holds for **every**
surviving projection and every carve of it, so a constraint built from it never removes a seating
blaze could build on (the "no early elimination" note above).

**Where the code lives.** The first version is a private method,
`MeshGraphDescriptor::check_pipeline_feasibility` in `mesh_graph_descriptor.cpp`, because it
saves nothing and needs only the descriptor. If the backup is built, Tier B moves to one entry
point beside the resolver in `tt_metal/fabric/pipeline_builder.cpp`, with its searches as
helpers in that file's anonymous namespace:

```cpp
// pipeline_builder.hpp, namespace detail
// Throws, naming the pipeline and the rule, when no mapping of the descriptor can hold it.
PipelineMappingConstraints check_pipeline_feasibility(const MeshGraphDescriptor& descriptor);
```

`mesh_graph.hpp` includes `mesh_graph_descriptor.hpp`, so the descriptor header cannot include
`pipeline_builder.hpp`. It forward-declares `PipelineMappingConstraints` and keeps it in a
`std::unique_ptr` (its destructor is already out of line). `mesh_graph_descriptor.cpp`
includes `pipeline_builder.hpp`. Tier C's per-seating check goes in the same two files, beside Tier B (4.0).

**What Tier B saves** (MGD types only; `pipeline_builder.hpp` does not include `topology_solver.hpp`):

```cpp
// pipeline_builder.hpp, namespace tt::tt_fabric (the descriptor's accessor returns this)
struct PipelineMappingConstraints {
    // Stage -> mesh. Keyed by (pipeline, stage id); Tier C's PipelineStageNode is the same pair (4.0).
    std::map<std::pair<std::string, PipelineStageId>, std::set<MeshId>> binding_sets;   // step 1

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

**How it reaches the solves: a generic hook on the mapper (P10).** The mapper gets no
pipeline-specific constraint code. Instead `TopologyMappingConfig` gains a generic
`ExtraMappingConstraints`, in public types only, which any caller can fill; `TopologyMapper`
fills it from `PipelineMappingConstraints` before the first solve. Empty (the default) is
today's behaviour.

```cpp
// topology_mapper_utils.hpp: generic, no pipeline types. Callers (TopologyMapper) fill it; empty = today.
struct ExtraMappingConstraints {
    using Asics = std::unordered_set<tt::tt_metal::experimental::PhysicalNodeId>;
    // SAT placement, unary: false forbids mesh on a seat with these ASICs.
    std::function<bool(MeshId mesh, const Asics& seat)> allow_seat;
    // SAT placement, pair: false drops the seat-graph edge between two seats of connected meshes.
    std::function<bool(MeshId a, const Asics& seat_a, MeshId b, const Asics& seat_b)> allow_seat_pair;
    // Intra-mesh: add hard constraints to mesh's solve once the seating is fixed, after rank, exit-node
    // and MGD-pin constraints. A returned message fails that mesh (failure_stage "extra_constraints").
    std::function<std::optional<std::string>(
        MeshId mesh,
        const std::map<MeshId, std::set<tt::tt_metal::experimental::PhysicalNodeId>>& seated_asics,
        MappingConstraints<FabricNodeId, tt::tt_metal::experimental::PhysicalNodeId>& constraints)> intra_mesh;
};

struct TopologyMappingConfig {
    // ...existing fields...
    ExtraMappingConstraints extra_constraints;
};
```

```cpp
// physical_grouping_descriptor_matching.cpp: the session takes them from the enumerator's config.
//   allow_seat:      in apply_extra_constraints, add_forbidden_constraint(mesh, {seats where false}); rebuilt with the session.
//   allow_seat_pair: in build_sat_placement_seat_graph, skip a canonical seat pair only when it is false for
//                    every mesh pair that visits it (footprints are shared across meshes).
// topology_mapper_utils.cpp, complete_intra_mesh_for_placement, after add_exit_node_constraints:
if (config.extra_constraints.intra_mesh) {
    if (auto failure = config.extra_constraints.intra_mesh(logical_mesh_id, seated_asics, intra_mesh_constraints)) {
        record_failing_meshes(result, logical_mesh_id, physical_mesh_id);
        set_mapping_failure(result, "extra_constraints", *failure);
        return result;
    }
}
```

```cpp
// pipeline_builder.cpp: P10 fills the generic hooks from Tier B; the mapper never sees pipeline types.
ExtraMappingConstraints detail::pipeline_extra_constraints(const PipelineMappingConstraints& pmc,
                                                           const PhysicalSystemDescriptor& psd);
//   allow_seat:      cables_leaving(seat, psd) >= pmc.min_crossings_out(mesh)
//   allow_seat_pair: k = pmc.min_crossings(a, b); k == 0 || candidate_links_between(seat_a, seat_b, psd) >= k
//   intra_mesh:      per neighbour n with k(m, n) > 0, add_cardinality_constraint(boundary chips of
//                    boundary_slices(m, n), ASICs cabled toward seated_asics[n], ceil(k / links per exit ASIC))

// topology_mapper.cpp, build_mapping, where the config is filled:
+   if (const auto* pmc = mesh_graph_descriptor.pipeline_mapping_constraints()) {
+       config.extra_constraints = detail::pipeline_extra_constraints(*pmc, physical_system_descriptor_);
+   }
```

| Hook | Solve, and where it is applied | What the pipeline adds | Why it is safe |
|---|---|---|---|
| `allow_seat` | SAT seating, `apply_extra_constraints` (rebuilt with the session) | forbid seats of mesh `m` whose footprint has fewer than `min_crossings_out(m)` cables leaving it | every projection needs that many cables out of `m` |
| `allow_seat_pair` | SAT seating, `build_sat_placement_seat_graph` | no edge between seats with fewer than `k_AB` links (Plan 1, 4.4) | `k_AB` is the minimum over every projection |
| `intra_mesh` | intra-mesh solve of `m`, after `add_exit_node_constraints` | at least `ceil(k(m, n) / links per exit ASIC)` chips of `boundary_slices(m, n)` on ASICs cabled toward `n`'s seat | every crossing to `n` leaves from those slices, in every projection and carve |

Seat footprints are deduplicated across meshes in the seat graph, so `allow_seat_pair` drops
a canonical pair only when it is false for every mesh pair that visits it; otherwise the
exact check handles it. The `intra_mesh` hook's constraint sits after the existing mesh-level
exit cardinality constraint, which lets any chip of the mesh be an exit node; it narrows that
to the boundary-slice chips (`MeshGraph::get_coord_range(mesh, host_rank)`) with the same
links-per-ASIC rounding.

`stage_to_mesh_constraints(pmc)` (`MappingConstraints<std::size_t, MeshId>`, required:
each stage to its binding set) stays in `pipeline_builder.cpp` as a filter and a report. It is
not a solve, because the solver maps one-to-one and many stages share a mesh. Capacity and
channel counts do not fit in `MappingConstraints`; the exact rules stay in Tier B's search
and Tier C's carve.

**On the running examples.**

| Example | `allow_seat` | `intra_mesh` |
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

**Tests** (`test_mesh_graph_descriptor.cpp`, first version): a stage that fits no mesh, total
slice demand, stages that fit only one mesh, a crossing with no connection, too many crossings
for the channels, a repeated edge counting twice, POD and CLUSTER connections between two pods,
and the step limit (the 7x7 ring above accepts). Backup if slow: a RING `host_topology` axis (a
ring4 on one mesh passes once A3–A0 wrap), an odd grid ring rejected before the first search
node, "find all" over its cap saving the defaults, interchangeable-mesh symmetry (8 identical
meshes on ALL_TO_ALL visit one ordering). Also: `subtorus_4x4_pipeline_8stage` with and without its 7-0 connection, and
`llama_8b_pod` with nine 4x2 stages and thirty-one 1x2 stages (the total fits, M0 does not).

**Open points.**

- A connection between two sub-graphs (not two meshes) does not name the meshes it joins. The
  first version gives its channels to every pair of meshes under the two sub-graphs, an upper
  bound that Tier C checks against the cables.
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
with the most slices: s1 (2 slices), on M0, the first of two meshes with the same free slices
(with the backup symmetry rule, M1 would be its mirror and never tried). Breadth-first from s1
gives s0, s2, then s3. s0 prefers M0, which
holds s1 (used 3 of 4). s2 also prefers M0 but needs 2 slices with 1 free, so it goes to M1
and uses one of the two channels (`s1 -> s2`). s3 has neighbours on both meshes, so the tie
goes to the mesh with more free slices, M1; the loopback `s3 -> s0` uses the second channel.
Every stage has a mesh, so Tier B passes after a handful of nodes and chooses nothing. (The
backup step 4 carve would also pass: M0's 1 + 2 slices fit a path of three slices in a line of
four, and M1's 2 + 1 likewise.) In the "find all" run, every survivor crosses
M0–M1, and a ring crosses a cut an even number of times, so Plan 1 (4.4) gets
`k(M0, M1) = 2`.

**`pod8`: exactly one projection.**

- 4x2 binds only to M0, because M1 has two rows. 1x2 binds only to M1, because on M0 it is
  not a whole number of 4x2 slices.
- So s0..s3 go on M0 (4 of 4 slices) and s4..s7 on M1 (4 of 4). The projection crosses twice
  (`s3 -> s4`, `s7 -> s0`), within the 2 channels.
- **Segment carve (backup if slow).** M0's four stages fill a line of four slices, so the segment can only
  enter and leave at the end slices A0 and A3. M1's four stages are a path through all four
  slices of a 4-cycle.
- Tier B passes. (The backup "find all" run would give `k = 2` and boundary slices {A0, A3}
  on M0 and any slice on M1, for Plan 1.)

**Edge cases.**

| Change | Result | Rule |
|---|---|---|
| `mixed4` with a fifth stage `[4, 6]` (3 slices) | reject | the ring needs 9 slices and the MGD declares 8 |
| `mixed4` with a fifth stage `[4, 4]` | pass | 8 of 8 slices: s0, s3 and one 4x4 stage on one mesh, the other two 4x4 stages on the other; every slice is used |
| `mixed4` with one stage `[4, 10]` | reject | wider than either mesh: empty binding set |
| `pod8` with M1 as 4x8, `host_topology [1, 4]` (4x2 slices), still 1x2 stages | reject | 1x2 is not a whole slice. Blaze's per-rank tiling could build it (3.4), so this rule is stricter than runtime (open question 15) |
| `mixed4` with the MGD connection removed | reject | every projection crosses on some edge, and no connection is declared |
| RELAXED (later): the same, with `s1 -> s2` and `s3 -> s0` RELAXED | pass | the projection `s0, s1 -> M0`, `s2, s3 -> M1` crosses only on those two edges, and RELAXED crossings need no connection |
| RELAXED (later): the same, with only `s3 -> s0` RELAXED | reject | every projection still crosses on a STRICT edge (`s1 -> s2` or `s2 -> s3`) |
| `mixed4` plus a copy named `mixed4_b` | reject | together they need 6 + 6 = 12 slices and the MGD has 8, although each fits alone (1.5) |
| two copies of `ring4` (four 4x2 stages each) | pass here; Tier C rejects | 4 + 4 = 8 slices fit with each ring on its own mesh and no crossing. A0..A3 is a line, so neither ring closes on one mesh; only the backup step 4 carve would see that and force each ring to split, 4 STRICT crossings against 2 channels |
| the same, with `channels { count: 4 }` between M0 and M1 | pass | a projection places all eight stages at once; Tier C finds the split that crosses 2 + 2 |

### 2.3 Tier C - inside the topology mapper

The only tier that sees physical adjacency.

**Goal: prove a placement exists.** Tier C accepts a mapping when it finds *some* carve and
layout of the pipeline on it. That placement is a witness, not the one blaze will build:
blaze carves again at runtime, may take a different path (planned carve or uniform tiling,
see Tier D), and may land on a different placement. That is fine while nothing consumes the
mapper's placement (4.6). It is not enough on its own: blaze's planner rejects some carves the
footprint finds (3.2), so an accepted mapping can still fail at build time until the two
agree (open question 8). The one rule is the no-early-elimination rule from 2.2: Tier C
rejects a seating only when the footprint solve proves no carve on it passes the check (and
the exit-node retry cannot help), never because one carve fails.

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
  footprint graph builder and `inter_mesh_cables` are pure (4.1, 3.3).

It works in four steps after intra-mesh completion, detailed in section 4:

- **C1 - build the physical pipeline graph (4.1).** One position graph per seating: every
  rectangle each stage shape fills, and which rectangles touch, over every cable that could
  be bound.
- **C2 - pipeline solve (4.2).** One footprint solve gives one carve (the rectangles, stage
  order and crossings). The resolver checks that one carve at chip level, the exact channel
  of every crossing is recorded as `R`, and `R` is checked against the demand. Any failure
  fails the seating (C4); C2 never asks for a second carve.
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

So Tier C solves over the cables the control plane *could* bind for the A–B connection
(`inter_mesh_cables`, 3.3: mapped ends, requested exit nodes, one destination chip), not any PSD cable between the two footprints, and
then makes sure the ones it used are bound: it records the exact channel of every crossing in
`R`, and phase 0 of the pairing binds those channels first (4.2, 4.3). Intra-mesh links follow the same idea: only links the intra-mesh topology routes
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
| `mixed4` | (P, Q) | A0–B0, A3–B3 | reject | the slices form one loop of 8 and the ring needs a loop of exactly 6: the solve finds no carve |
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

`build_pipeline` builds the graph from the harnesses with the loop in 1.2, then:

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
plan_ring_carve           the 6-cycle of Tier C's carve, or one of its mirrors (3.2, example A)
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

Tier C runs the pipeline builder's own resolver, so the mapper applies the same layout rules
as runtime (chip counts, core capacity, links) rather than a restated copy of
them. The carve itself comes from the footprint solve (4.2), not from blaze's planner (3.2).
Tier C does not need to produce the same placement as runtime (2.3).

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
DAG after dropping loopbacks, and edges carried over host are left out.

The first version of Tier C calls it with no change: the rerun benchmark drove the unmodified
search with injected links on every case in 4.2, and the wrapper reaches the control plane
only when `direct_links` is null. The resolver throws with a message string on failure; C2
catches it, rejects that carve and keeps the message as the reason. No structured failure is
needed. The resolver tries every stage-to-submesh assignment and link before it throws, so its
failure is about the whole carve and cannot soundly name one stage. Failures that do depend on
one stage and one position are filtered in C1 instead (4.2). The exit-node retry (P15) reads
its signature from the intra-mesh layout and `inter_mesh_cables`, not from the resolver.

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

**`mixed4` on (P, R), on the carve of example A** (3.2). Six submeshes: 0 = A0
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
does not take the carve's stage order as given.

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
| harness overrides of pipeline-core pools | not in the MGD; Tier C computes blaze's defaults from the graph and can accept a layout runtime rejects (open question 7) |

### 3.2 Blaze's carve planner: the runtime reference, not ported

Tier C does not port the carve planner. The footprint solve (4.1, 4.2) decides the carve, the
stage order and the crossings in one call, and on the 132 carve-requiring benchmark cases the
planner plus resolver placed none while the footprint plus resolver decided 127 (4.2,
"Speed"). This section keeps what the planner does, because blaze still runs it at runtime
and Tier C should agree with it on *whether* a carve exists (open question 8).

The planner (`pipeline_builder/carve_planner.py`, tt-blaze) decides which host slices make up
each stage. It works on whole host slices, ordered by `(mesh_id, row0, col0)` so every rank
reaches the same plan:

| Phase | Code | What it does |
|---|---|---|
| 0. input checks | top of `plan_ring_carve` | all slices one shape; every stage a whole number of slices; `need = sum(stage_counts)` fits; stage 0 exactly one slice |
| 1. reduce | `_prune_leaves`, `_contract` | drop slices with fewer than two live neighbours; contract runs of two-neighbour slices between *branch slices* into weighted chains |
| 2. cycle search | `_cycles_no_branch`, `_cycles_of_length` | simple cycles of exactly `need` slices; stops at 200,000 expansions or 64 cycles |
| 3. align | `_group_cycle`, `_canonical_walk`, `_as_rectangle` | cycles by fewest mesh crossings; every rotation and both directions; each stage's group must be one rectangle of exactly the declared shape; first fit wins |
| 4. emit | `plan_ring_carve`, `validate_carve_plan` | one `CarveSlot(mesh_id, shape, offset)` per stage, a filler slot per unused slice |

`plan_fork_carve` searches a fork the same way, with backtracking capped at 2,000 attempts.

Where the planner and the footprint differ, and so where Tier C can accept what blaze does
not build:

| Planner behaviour | Footprint |
|---|---|
| commits to its first aligned cycle; if the resolver then fails, blaze raises | enumerates carves until one passes the check |
| stage 0 must be exactly one slice | stage 0 on any position with room for host I/O (open question 14) |
| first-fit grouping misses alignments (`no-alignment` on 16 feasible benchmark cases) | exact |
| raw Hamiltonian-cycle search on grid-shaped slice graphs (no degree-2 chains to contract), out of budget on every 8x8 or larger grid | SAT with resource completeness |
| uniform host slices only (`pod8`, `llama_8b_pod` go to per-rank tiling) | positions per mesh from each stage's chip shape |
| no rotation of stage shapes | no rotation either (2.1) |

The planner's examples, on the running examples' fabrics, with both verdicts (the diagrams
are in the canvas, "How the carve planner works"):

| Example | Fabric and pipeline | Planner | Footprint |
|---|---|---|---|
| A | `mixed4`, cables A0–B0, A2–B2; ring 4x2, 4x4, 4x4, 4x2 | carve: s0 A0, s1 A1+A2, s2 B2+B1, s3 B0 | same carve or one of its three mirrors |
| B | `mixed4`, cables A0–B0, A3–B3 | `no-cycle-of-length-6` (only an 8-cycle) | no mapping |
| C | A plus cable A3–B3 (branch slices A2, B2) | same carve as A | same as A |
| D | A's fabric; ring 4x2, 4x2, 4x8 | `no-alignment` (runs [3, 3]) | no mapping: the 4x8 fills a mesh, and the two cable slices left on the other mesh are not adjacent |
| F | `galaxy4x1`: one 4x8 mesh of 4x1 rank columns, RING; ring 4x1, 4x2, 4x2, 4x2, 4x1 | carve on columns C0, C1+C2, C3+C4, C5+C6, C7; ranks 1 and 2 (and 3+4, 5+6) co-own a stage | same carve or a rotation |

D and F are derived, not in the benchmark. F is also why a single mesh with a joint stage
takes blaze's planned carve, not single-mesh tiling (3.4).

### 3.3 Inter-mesh cables: read from the PSD (policy plan later)

Tier C needs, during mapping, the cables between every pair of meshes an MGD connection
joins. In the first version C1 reads them straight from the PSD with calls that exist today
(`get_asic_neighbors`, `get_eth_connections`), as the PSD's own `ExitNodeConnection`, so
`std::hash` of each one is the `connection_hash` today's gather and pairing use:

```cpp
// pipeline_builder.cpp, file-local. Pure: PSD + mapping, no cluster, no MPI.
std::vector<tt::tt_metal::ExitNodeConnection> inter_mesh_cables(
    const tt::tt_metal::PhysicalSystemDescriptor& psd,
    const std::map<FabricNodeId, tt::tt_metal::AsicID>& fabric_node_to_asic,
    MeshId mesh_a, MeshId mesh_b);   // one entry per ETH channel, sorted by (src chip, src chan)
```

It keeps the gather's rules that need no cluster: both ends mapped onto the two meshes;
requested exit nodes only, for a device-level connection; one destination chip per (source
chip, neighbour mesh). It does **not** filter by cable type (no `PortTypeRules` yet) or link
state (rank 0's PSD has none); a required channel on a down link fails the check after the
broadcast on every rank (4.3, open question 20). The control plane's gather is unchanged.

**Next step (P16, M5): the policy plan's candidate table.** The
[per-connection inter-mesh STRICT/RELAXED plan](https://riddy21.github.io/tt-metal/mgd-intermesh-policy-plan.html)
(the *policy plan*) moves the gather into the mapper. Once its part 9 lands, C1 switches to it:

| Piece | Policy plan part | Used here for (P16) |
|---|---|---|
| `collect_inter_mesh_candidates(mesh_graph, psd, fabric_node_to_asic, options)` | 9 | replaces `inter_mesh_cables`, once per seating, with the same options as the final call |
| down-link hashes in the chip info | 9 | down links never reach C1, closing question 20 |
| cable-type filter and tier per channel (`PortTypeRules`) | 6 | disallowed cables never reach C1; `R` and `direct_links` take better tiers first |
| `PortDescriptor::tier` and the tier sort in `pair_logical_intermesh_ports` | 9 | the fill order after the required links (4.3) |
| `resolve_inter_mesh_binding` in the mapper (stretch) | 11 | later: check that the required links can be placed before the mapping is emitted (4.3) |

The demand the budget check uses (4.2) is the MGD connection's channel count today; with the
policy plan it becomes `get_inter_mesh_channel_demand(a, b)` (parts 3 and 5).

#### Example

`mixed4` on (P, R), on a machine with an extra A0–B0 cable whose chip ids are lower than the
A2–B2 cable's:

| Call | Result |
|---|---|
| `inter_mesh_cables(...)` | three cables: A0–B0 (first), A0–B0 (second), A2–B2. Each source chip cables to one destination chip on M1, so all three pass |
| pairing, no required links | today's rule: the two lowest ids, both A0–B0 |
| pairing, `R = {A0–B0, A2–B2}` from Tier C | phase 0 binds both cables in `R`; the count of 2 is full |

Tier C reads the same PSD entries the gather turns into the pairing's table, so the cables
Tier C checks are the cables the pairing can bind (down links aside, until P16).

### 3.4 How blaze turns meshes into submeshes

Tier C does not reproduce blaze's carve paths (4.1 replaces them with one position graph),
but what blaze can build bounds what Tier C should accept. `PipelineGraph._collect_submeshes`
(`pipeline_builder/graph.py`, tt-blaze `origin/main` 34e783a631):

| Step | Multi-mesh, planned carve | Multi-mesh, per-rank tiling | Single mesh |
|---|---|---|---|
| when | `multi_mesh`, no `submesh_shape` | `multi_mesh`, `submesh_shape` given | not `multi_mesh` |
| slots | `plan_carve` (3.2) | `create_submeshes(submesh_shape)` on the rank's mesh | `create_submeshes(shape)`; every stage the same shape |
| the rank's slot | the one slot holding its chips | first tile whose `(0,0)` chip is local | all tiles |
| joint stages | ranks with identical chip sets collapse into one submesh | same | n/a |
| layout | `resolve_graph_layout` (links from the control plane) | same | same |

`multi_mesh` is set by `build_pipeline` when the MGD has more than one mesh **or** a stage spans
more than one rank, so a single mesh with a joint stage takes the planned carve (3.2,
example F). `create_submeshes` stamps the shape row-major and fails unless it divides the mesh.

How the position graph covers each blaze path:

| Blaze path | In the position graph |
|---|---|
| planned carve | positions of every stage shape on every mesh; the solve picks the cycle and the cut together |
| per-rank tiling | positions of the tile shape; with tile = host slice, every rank keeps its own slice (`llama_8b_pod`: 4x2 on M0, 1x2 on M1) |
| single-mesh tiling | one mesh, one shape |
| tile smaller than a host slice (a 2x2 stage on 4x2 slices) | **not covered**: positions are whole slices. Blaze keeps the rank's first tile and drops the rest, so a cable landing outside that tile does not count. Tier B rejects such MGDs today (2.2); open question 15 |

#### Host cuts on RELAXED edges - RELAXED (later)

Blaze's flexible placement (`_resolve_topology`, `host_edge_planner.py`) first looks for an
all-fabric carve, then covers the ring with the fewest vertex-disjoint fabric chains joined
over host. A cut falls only between stages, never inside a wide stage, and never on stage 0's
outgoing edge; ties go to the smallest host-hop page sizes, then to cutting the loopback. Each
fabric chain is then resolved on its own (`_resolve_chains_separately`). Forks use
`plan_fork_placement`, falling back to all-fabric on `ForkShapeUnsupported`.

In Tier C the cut does not need a port: a RELAXED pipeline edge becomes a **soft** target edge
in the footprint solve (`set_allow_unmatched_target_edges`, #56762), so the solve may leave
it unmatched (over host) while STRICT edges and stage 0's outgoing edge stay hard. The check
then resolves each fabric chain separately, as blaze does. Whether Tier C must also prefer
blaze's tie-breaks is open question 16.

Example: `mixed4` with `s3 → s0` RELAXED on (P, Q). With every edge STRICT this seating has no
carve (3.2, example B). With the loopback soft, the chain s0 → s1 → s2 → s3 maps onto
A3, B3+B2, B1+B0, A0, and `s3 → s0` goes over host, which is also blaze's answer.

#### Several pipelines

The descriptor stores one record per pipeline (1.5). The mapper's representation is the
merge of those records: one target graph, one component per pipeline, on one position
graph. One footprint solve places every stage of every pipeline; the resource constraint
keeps the components on disjoint slices. Each pipeline is then checked by its own resolver
call on that carve. Two copies of `ring4` (four 4x2 stages) on the `mixed4` meshes with
cables A0–B0, A2–B2, A3–B3: only one 4-cycle exists (A2 A3 B3 B2), so the seating has no
mapping. Adding cable A1–B1 gives a second 4-cycle (A0 A1 B1 B0) and the seating is accepted
with every slice used.

#### Reuse or rebuild

| Piece | Today | Tier C |
|---|---|---|
| resolver | `resolve_graph_layout_with_connections` (C++) | **use directly** (3.1) |
| host slices | the MGD's `get_host_ranks` (a grid of host ranks) and `get_coord_range`, the same slices Tier B divides out | **use directly**: each host's range, laid out as the host grid, in `LogicalMultiMeshGraph::mesh_host_grids_` (4.0, 4.1) |
| stages and edges | `PipelineDescriptorData` (MGD) | **use directly**, read from the merged MGD the enumerator keeps; the stage graph beside today's mesh graphs is built from them; no separate logical pipeline type (4.0) |
| carve | `plan_ring_carve`, `plan_fork_carve` (Python) | **replaced** by the footprint solve (4.1, 4.2) |
| RELAXED (later): host cuts | `plan_ring_placement`, `plan_fork_placement` (Python) | **replaced** by soft target edges in the same solve |
| per-chain resolution | `_resolve_chains_separately` (Python) | **rebuild**: a loop over the resolver, one call per fabric chain |
| default core counts | `_pipeline_core_defaults` (Python) | **rebuild** (open question 7) |
| submesh links | `discover_submesh_links` (C++, asks the control plane) | not needed: C2 supplies `direct_links` |
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

With a pipeline descriptor, Tier C adds four steps after intra-mesh success. 4.0 has the code
and the PRs; 4.1 to 4.4 explain each step's design.

| Step | What it does | First version | Later | Section |
|---|---|---|---|---|
| C1. Build the physical pipeline graph | host slices, slice adjacency and candidate cables, turned into one position graph: every rectangle a stage shape fills, and which touch | one position graph per seating over the PSD's cables (P5) | the policy plan's candidate table (P16) | 4.1 |
| C2. Pipeline solve | pre-checks, one footprint solve on the position graph, the resolver on that carve as a chip-collision check, `R` as exact channels, the demand check | all of it, on today's solver API; any failure fails the seating; per-stage misfits are filtered before the solve (P6) | RELAXED edges as soft target edges, one resolver call per fabric chain (P11); the demand as a crossing cap in the encoding if the check fires; a second carve only if a real case needs one (open question 24) | 4.2 |
| C3. On success: emit and publish the cables | emit the mapping; publish `R`; the control plane binds `R` first | the mapping in the result and on rank 0, `R` broadcast (P8); phase 0 in the pairing (P9); the stage plan in rank bindings (P12) | phase 0 inside the mapper with the policy plan's part 11 | 4.3 |
| C4. On failure: constraints for the next solve | retry the same seating with exit-node constraints, then exclude the seating and draw the next one | `exclude_mapping(seating)`, `unique_shapes` off, rejection cap (P8); exit-node retry (P15) | seat-pair clauses (P13); constraints before the first solve, only if rejection counts call for them (P10) | 4.4 |

```
[before the first solve]
  Tier B: at least one projection exists (no projection is chosen)
  C4, later (optional): exit-node and seat constraints every carve agrees on      (4.4)

loop:
  seating = placement_session_->next()                                  existing
  full = complete_intra_mesh_for_placement(...)                         existing
  if fail: add_forbidden_constraint(failed seat); loop                  existing
  -- new from here, only when the MGD has a pipeline --
  G = C1(seating, full)                                                 (4.1)
  result = C2(pipeline, G)            one carve, resolver, R, demand    (4.2)
  if result.accepted: C3(result); emit                                  (4.3)
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
  (`inter_mesh_cables`, 3.3), not over a binding chosen in advance. The binding
  follows the solve: the cables the accepted layout uses become `R`, and C3 has them bound
  first.
- C2 is the one exception, by choice. It solves for one carve and checks only that carve, so
  a seating whose first carve fails the resolver is rejected even if another carve would
  pass. No benchmark case did that (4.2, "Why one carve is enough"); if a real one does, C2
  asks the same session for a few more carves before failing (open question 24). A "no carve"
  from the solve is still a proof over every carve.
- A constraint C4 derives from a failure holds only for the seating that failed: it steers
  that seating's retry, never another seating's solve. Constraints that apply to every
  seating (added before the intra-mesh or master solve) may only state what every carve
  agrees on: the minimum number of crossings and the union of boundary slices. When those
  facts cannot be computed within budget, nothing is added.
- A rejection is fed back to the SAT placement as a constraint on the seat set, never on one
  mesh on one seat (4.4).

### 4.0 Implementation plan

Everything an implementer needs for Tier C is in this section: where the code lives, the
types, the one per-seating function, the mapper hook, and the PRs. Sections 4.1 to 4.4 explain
the design of each step; they do not add code locations.

#### Where the code lives

One rule: **`topology_mapper_utils` owns the types and what runs once per run; `pipeline_builder`
owns everything that runs per seating.** The mapper calls one function per seating and reads
back accept or reject. The per-seating work sits beside the resolver it ends with and beside
Tier B, which already lives in `pipeline_builder.cpp`.

| File | Holds | PR |
|---|---|---|
| `topology_mapper_utils.hpp` | `PipelineStageNode`; three new `LogicalMultiMeshGraph` fields (the merged stage graph, the stage shapes, the host grids), `MultiMeshSolutionEnumerator::mesh_graph_descriptor_` (the merged MGD, kept) and four new `PhysicalMultiMeshGraph` fields (the positions); the two new Tier C structs, the mapping result (`PipelinePlacement`, `PipelineMappingResult`); `TopologyMappingConfig::check_pipelines`; `TopologyMappingResult::pipeline` | P5, P8 |
| `topology_mapper_utils.cpp` | `build_logical_multi_mesh_adjacency_graph` fills the three new fields; the constructors keep the merged MGD; the `next()` hook and C4 | P5, P8 |
| `pipeline_builder.hpp` (`detail`) | forward declarations; `build_physical_pipeline_adjacency_graph`, `solve_pipeline_footprint`, `check_pipeline_seating` | P5, P6 |
| `pipeline_builder.cpp` | per seating: `inter_mesh_cables`, the physical graph, the pre-checks, the footprint solve, the resolver arguments and one call per pipeline, `R`, the demand check | P5, P6 |
| `topology_mapper.cpp` | the kill switch; the mapping saved on rank 0; `R` broadcast with the chip info | P8 |
| `control_plane.cpp` | phase 0 of the pairing binds `R`; every rank checks it was bound | P9 |
| `test_topology_mapper_utils.cpp` | the logical graph's pipeline fields (P5); the hook on PSD fixtures (P8) | P5, P8 |
| `test_pipeline_builder.cpp` | the physical graph, the solve and the check, on CPU PSD fixtures | P5, P6 |

Includes: `pipeline_builder.cpp` includes `topology_mapper_utils.hpp` and `topology_solver.hpp`;
`pipeline_builder.hpp` includes neither (it forward-declares), so the ttnn ops and the nanobind
module that include it are unchanged. `topology_mapper_utils.hpp` includes `pipeline_builder.hpp`
for `ChipTuple`, `GraphLayoutResult` and `DirectLinks`, which the new types store as is. `topology_mapper_utils.cpp` includes
`pipeline_builder.hpp`. Both `.cpp` files are in the same library (`tt_metal/fabric/sources.cmake`).

```
MGD (merged)
  │ build_logical_multi_mesh_adjacency_graph → logical_: stage graph,    topology_mapper_utils.cpp   once per run   P5
  │   stage shapes, host grids beside today's mesh graphs; the merged MGD is kept as mesh_graph_descriptor_
  ▼
MultiMeshSolutionEnumerator::next()                                            topology_mapper_utils.cpp
                           seating → intra-mesh (today)
                           detail::check_pipeline_seating                      pipeline_builder.cpp      P6
                             C1 build_physical_pipeline_adjacency_graph          positions overlap          P5
                             C2 pipeline_pre_checks → solve_pipeline_footprint   one carve (MappingResult)  P5
                                → resolver on the carve → R → demand check
                           reject: exclude_mapping(seating) (C4)               topology_mapper_utils.cpp P8
                           accept: TopologyMappingResult::pipeline
  ◄── TopologyMapper (rank 0): pipeline_mapping_; R broadcast                  topology_mapper.cpp       P8
        └─ ControlPlane: phase 0 binds R, every rank checks it                 control_plane.cpp         P9
        └─ generate_rank_bindings: the stage plan                                                         P12
```

#### The two graphs

They mirror the mesh graphs and are solved onto each other with the same `MappingConstraints`
and SAT session the seating uses:

| Mesh mapping (today) | Pipeline mapping (Tier C) | Nodes | Built |
|---|---|---|---|
| `LogicalMultiMeshGraph::mesh_level_graph_`, from `build_logical_multi_mesh_adjacency_graph(mgd)` | `LogicalMultiMeshGraph::pipeline_stage_graph_`, from the same call | `PipelineStageNode` (pipeline, stage id) | once per run, topology_mapper_utils |
| `PhysicalMultiMeshGraph::mesh_level_graph_`, from `build_hierarchical_from_flat_graph(flat, seating)` | `PhysicalMultiMeshGraph::pipeline_position_graph_`, filled into the same per-seating `physical` by `build_physical_pipeline_adjacency_graph(logical, full, psd, physical)` | position index | once per seating, pipeline_builder |
| SAT seating: seats overlap; `add_resource_constraint(seat → ASICs)` keeps the chosen ones disjoint | footprint solve: positions overlap; `add_resource_constraint(position → slices)` keeps the chosen ones disjoint | | once per seating, pipeline_builder |

- **Logical: part of `LogicalMultiMeshGraph`.** The stage graph is a third graph beside the
  mesh graph and the exit-node graphs, built by the same `build_logical_multi_mesh_adjacency_graph`
  from the same MGD, so the enumerator's `logical_` carries it with no new member. A node is a
  `PipelineStageNode`: the pipeline's name and the stage's `PipelineStageId`, both from the
  MGD (stage ids are unique only inside their pipeline, so the pair is the identity). It formats
  as `<pipeline>/<stage id>`, which is what logs and error messages print.
- **One stage graph for all pipelines.** The builder merges every pipeline of the merged MGD
  into this one graph. That includes every part's pipelines: `MeshGraphDescriptor::merge`
  carries them and rejects a pipeline name declared in two parts (1.5). Each pipeline is a
  connected component of it, with no edges between pipelines. The footprint solve maps the
  whole graph at once, so stages of different pipelines compete for the same slices.
- **Each stage's shape is beside the graph.** `pipeline_stage_shapes_` maps every
  `PipelineStageNode` to its `MeshShape`, from the stage's `dims`. The graph itself has no
  attributes, and C1 (positions per shape), the solve (stage to positions it fits) and the
  pre-checks (area) need the shape, so they read only the logical graph.
- **Host slices are a grid of chip ranges.** Per mesh, `mesh_host_grids_` is a
  `MeshContainer<MeshCoordinateRange>` shaped as the host grid (the MGD's `get_host_ranks`).
  The cell at a host-grid coordinate is that host's slice, `get_coord_range(mesh, rank)`: its
  start and its shape (`.shape()`). The MGD stores each rank's range explicitly and slices of
  one mesh need not match (pod8 mixes 4x2 and 1x2), so the cells keep the range rather than a
  shape alone. It is a grid, not an adjacency graph: a position is a rectangle of slices, which
  needs coordinates, and which slices touch is a physical fact C1 computes per seating. The
  rank is not stored: slices are numbered by (mesh, grid coordinate), and P12 reads the rank at
  a coordinate from the MGD's `get_host_ranks`, which has the same layout.
- **Edge direction and loopback marks come from the merged MGD.** The stage graph stores every
  edge both ways, and the resolver needs `src → dst` and the loopback marks. Today
  `init_from_parts` returns the merged MGD and the constructors drop it once the SAT session is
  built. With P5 the enumerator keeps it as `mesh_graph_descriptor_` and passes it to
  `check_pipeline_seating`, the only Tier C function that takes it, for the resolver's edges
  and node names.
- **Physical: part of `PhysicalMultiMeshGraph`.** The positions are four fields of the
  `physical` graph `next()` already builds for each seating, the twin of the stage graph in
  `logical_`. `build_hierarchical_from_flat_graph` leaves them empty: a position's chips depend
  on the intra-mesh layout, so `build_physical_pipeline_adjacency_graph` fills them after
  intra-mesh success, and only when the MGD has pipelines. A node is a *position*: a rectangle of host slices on one mesh that some stage
  shape fills exactly. Positions are generated for every stage shape, so they overlap: on a
  2x2 grid of slices, a 1x2-slice shape has four positions and each slice is in two of them.
  That is intended. The graph lists every place a stage could go; the resource constraint
  (each slice in at most one chosen position) keeps the chosen ones apart. An edge joins two
  positions only when they are disjoint and border each other, so overlapping positions are
  never adjacent. Each position keeps its chips as the resolver's `ChipTuple`s and each edge
  its links as the resolver's `DirectLinks`, so the carve's resolver arguments are lookups,
  not conversions. The positions depend only on the stage shapes and host slices; only the links
  between meshes change per seating, so the positions may be cached if the stats call for it.

#### Types (`topology_mapper_utils.hpp`)

Tier C adds one node type, `PipelineStageNode`, three fields to `LogicalMultiMeshGraph`, one
member to the enumerator (the merged MGD) and four fields to `PhysicalMultiMeshGraph` (P5). It adds two small structs, `PipelinePlacement` and
`PipelineMappingResult` (P6). Every field is an existing type. What it reuses
instead of defining:

| Need | Existing type |
|---|---|
| stages, labels, edges, loopback | `PipelineDescriptorData` (MGD, Tier A), from the merged MGD's `pipeline_descriptors()` |
| a stage's shape | `MeshShape`, from the stage's `dims` |
| a stage's id | `PipelineStageId` (`fabric_types.hpp`), an integer |
| host slices per mesh | `MeshContainer<MeshCoordinateRange>`, shaped as the host grid; each cell the MGD's `get_coord_range` for that host |
| position nodes | `std::size_t` indices |
| a position's chips | `std::vector<ChipTuple>` (resolver input) |
| links between positions | `detail::DirectLinks` (resolver input) |
| the carve | `MappingResult<PipelineStageNode, std::size_t>` (solver) |
| a pipeline's layout | `GraphLayoutResult` (resolver output) |
| a reject | `TopologyMappingResult`: `success`, `error_message`, `stats.failure_stage` |
| an exact channel in `R` | `tt::tt_metal::ExitNodeConnection` (PSD) |

```cpp
// --- the stage graph's node (P5), beside LogicalExitNode ---
// A stage: its pipeline's name and its id in that pipeline. Stage ids are unique only inside a pipeline.
struct PipelineStageNode {
    std::string pipeline;
    ::tt::tt_fabric::PipelineStageId stage_id = 0;

    auto operator<=>(const PipelineStageNode&) const = default;
};
// fmt::formatter<PipelineStageNode> prints "<pipeline>/<stage id>", as for LogicalExitNode.

// --- additions to LogicalMultiMeshGraph (P5) ---
// Empty by default (default-constructed). build_logical_multi_mesh_adjacency_graph fills them only when
// the MGD has pipeline_descriptors, so no-pipeline runs build exactly what they build today. Nothing
// reads them unless pipeline_stage_graph_ has nodes, which it does exactly when the MGD has pipelines
// (Tier A rejects a pipeline with no stages).
struct LogicalMultiMeshGraph {
    // ... mesh_adjacency_graphs_, mesh_level_graph_, mesh_exit_node_graphs_ as today ...

    // All pipelines merged into one graph: every stage of every pipeline, isolated ones included.
    // Edges both ways, one entry per declared edge (a repeated edge is another channel). Each pipeline
    // is one component; no edges between pipelines.
    AdjacencyGraph<PipelineStageNode> pipeline_stage_graph_;

    // Every stage's shape, from its dims.
    std::map<PipelineStageNode, tt::tt_metal::distributed::MeshShape> pipeline_stage_shapes_;

    // Per mesh: the host grid (the MGD's get_host_ranks layout); each cell is that host's slice of
    // device coordinates, get_coord_range(mesh, rank).
    std::map<LogicalMeshNode, tt::tt_metal::distributed::MeshContainer<MeshCoordinateRange>> mesh_host_grids_;
};

// --- addition to MultiMeshSolutionEnumerator (P5) ---
// The merged MGD, kept from init_from_parts (today a local of each constructor). check_pipeline_seating
// reads the resolver's edges from it: direction and loopback marks.
std::unique_ptr<const ::tt::tt_fabric::MeshGraphDescriptor> mesh_graph_descriptor_;
```

```cpp
// topology_mapper_utils.cpp (P5): build_logical_multi_mesh_adjacency_graph adds the three fields.
LogicalMultiMeshGraph build_logical_multi_mesh_adjacency_graph(const MeshGraphDescriptor& mgd) {
    auto mesh_adjacency_graphs = ::tt::tt_fabric::build_adjacency_graph_logical(mgd);
    auto [requested_intermesh_connections, requested_intermesh_ports] = get_requested_intermesh_from_mgd(mgd);
    LogicalMultiMeshGraph logical = build_logical_multi_mesh_adjacency_graph_impl(
        mesh_adjacency_graphs, requested_intermesh_connections, requested_intermesh_ports);
+   if (mgd.has_pipeline_descriptors()) {
+       AdjacencyGraph<PipelineStageNode>::AdjacencyMap stages;               // one map for every pipeline
+       for (const PipelineDescriptorData& pipeline : mgd.pipeline_descriptors()) {
+           for (const PipelineDescriptorData::Stage& stage : pipeline.stages) {
+               const PipelineStageNode node{pipeline.name, stage.id};
+               stages[node];                                                 // isolated stages stay listed
+               logical.pipeline_stage_shapes_.emplace(node, MeshShape(stage.dims[0], stage.dims[1]));   // rank 2 (Tier A)
+           }
+           for (const PipelineDescriptorData::Edge& edge : pipeline.edges) {   // P11 later skips needs_fabric == false
+               const PipelineStageNode src{pipeline.name, edge.src};
+               const PipelineStageNode dst{pipeline.name, edge.dst};
+               stages[src].push_back(dst);
+               stages[dst].push_back(src);
+           }
+       }
+       logical.pipeline_stage_graph_ = AdjacencyGraph<PipelineStageNode>(std::move(stages));
+       for (const auto& [mesh, adjacency] : logical.mesh_adjacency_graphs_) {
+           const MeshContainer<MeshHostRankId>& ranks = mgd.get_host_ranks(mesh);
+           std::vector<MeshCoordinateRange> slices;                          // row-major, as ranks.values()
+           for (const MeshHostRankId rank : ranks.values()) {
+               slices.push_back(mgd.get_coord_range(mesh, rank));
+           }
+           logical.mesh_host_grids_.emplace(mesh, MeshContainer<MeshCoordinateRange>(ranks.shape(), std::move(slices)));
+       }
+   }
    return logical;
}
```

```cpp
// --- additions to PhysicalMultiMeshGraph (P5), the twin of the LogicalMultiMeshGraph fields ---
// Empty by default; build_hierarchical_from_flat_graph leaves them so. Filled per seating by
// build_physical_pipeline_adjacency_graph after intra-mesh success, because a position's chips depend
// on the intra-mesh layout, and only when the logical stage graph has nodes. Position i is a rectangle of host
// slices on one mesh that some stage shape fills exactly.
struct PhysicalMultiMeshGraph {
    // ... mesh_adjacency_graphs_, mesh_level_graph_, mesh_exit_node_graphs_, mesh_pgd_pinnings_ as today ...

    AdjacencyGraph<std::size_t> pipeline_position_graph_;               // disjoint, bordering; one entry per link
    std::vector<std::vector<ChipTuple>> pipeline_position_chips_;       // per position: (mesh, chip, row, col), slot-local
    std::vector<std::vector<std::uint32_t>> pipeline_position_slices_;  // per position: slice ids, for add_resource_constraint
    detail::DirectLinks pipeline_position_links_;                       // keyed by (position, position), both ways
};

// --- the solve's answer (P5): no new type. solve_pipeline_footprint returns the solver's own
// MappingResult<PipelineStageNode, std::size_t>: success, error_message, stats (elapsed_time feeds
// pipeline_solve_seconds), and target_to_global, which is the carve (stage -> position index).

// --- the mapping result (P6 fills it, P8 returns it) ---
// Each pipeline keeps the resolver's own input and output, the same layout blaze gets at runtime.
struct PipelinePlacement {
    std::vector<std::vector<ChipTuple>> submesh_chips;   // resolver input: pipeline_position_chips_ of each stage's position
    GraphLayoutResult layout;                            // resolver output: stage order, node_to_submesh,
                                                         // resolved_edges (exit/entry chips), H2D and D2H chips
};

struct PipelineMappingResult {
    std::map<std::string, PipelinePlacement> pipelines;               // by pipeline name
    std::vector<tt::tt_metal::ExitNodeConnection> required_links;     // R: one PSD entry per crossing (4.3)
};

// --- additions to existing structs (P8) ---
struct TopologyMappingConfig  { /* ... */ bool check_pipelines = true; };   // false: no pipeline code runs
struct TopologyMappingResult  { /* ... */ std::optional<PipelineMappingResult> pipeline; };
struct TopologyMappingStats   { /* ... */ std::size_t pipeline_seatings_rejected = 0;
                                          double pipeline_solve_seconds = 0;   // from MappingResult::stats
                                          std::string pipeline_last_reason; };
```

Derived when needed, not stored:

- **Which `R` entry an edge uses.** `R` is appended in `layout.resolved_edges` order, one entry
  per edge whose exit and entry chips are on different meshes, so the edge for `R[k]` is the
  k-th crossing edge.
- **Host ranks of a stage, and idle host ranks.** A stage's host ranks are the ranks at the
  `mesh_host_grids_` coordinates its position covers, read from the MGD's `get_host_ranks`; the
  idle ranks are the rest. P12 computes them when it writes the stage plan.
- **The host I/O stage.** `layout.stage_order.front()`, the one root Tier A guarantees (2.1).

#### Entry points (`pipeline_builder.hpp`, `namespace detail`)

```cpp
// Forward declarations only, so the ttnn ops and nanobind module that include this header pull in
// neither topology_mapper_utils.hpp nor topology_solver.hpp. Everything per run comes in through the
// LogicalMultiMeshGraph (the stage graph, the stage shapes, the host grids and the intra-mesh links),
// plus the merged MGD for the resolver's edges (mesh_graph.hpp, already included, declares it).
// The positions live in the seating's PhysicalMultiMeshGraph.
namespace tt::tt_metal::experimental::tt_fabric {
struct LogicalMultiMeshGraph;
struct PhysicalMultiMeshGraph;
struct PipelineStageNode;
struct TopologyMappingResult;
}  // namespace tt::tt_metal::experimental::tt_fabric

namespace tt::tt_fabric {
template <typename TargetNode, typename GlobalNode>
struct MappingResult;           // topology_solver.hpp

namespace detail {
using ::tt::tt_metal::experimental::tt_fabric::LogicalMultiMeshGraph;
using ::tt::tt_metal::experimental::tt_fabric::PhysicalMultiMeshGraph;
using ::tt::tt_metal::experimental::tt_fabric::PipelineStageNode;
using ::tt::tt_metal::experimental::tt_fabric::TopologyMappingResult;

// C1 (P5). Clears and fills physical's pipeline_position_* fields. Public for the tests.
void build_physical_pipeline_adjacency_graph(
    const LogicalMultiMeshGraph& logical,               // stage shapes (pipeline_stage_shapes_), host grids
                                                        // (mesh_host_grids_), intra-mesh links
                                                        // (mesh_adjacency_graphs_)
    const TopologyMappingResult& full,                  // this seating's chip layout
    const tt::tt_metal::PhysicalSystemDescriptor& psd,  // inter-mesh cables
    PhysicalMultiMeshGraph& physical);                  // this seating's graph, from build_hierarchical_from_flat_graph

// C2's solve (P5), the solver's own result. The target is logical.pipeline_stage_graph_.
// success false: a stage with no position, no carve, or the conflict cap; target_to_global is the
// carve. Public for the tests.
MappingResult<PipelineStageNode, std::size_t> solve_pipeline_footprint(
    const LogicalMultiMeshGraph& logical, const PhysicalMultiMeshGraph& physical, std::size_t conflict_cap);

// The one call the mapper makes per seating (P6). Accept: sets full.pipeline, returns true.
// Reject: sets full.success = false, full.stats.failure_stage (pipeline_carve | pipeline_layout |
// pipeline_binding) and full.error_message, returns false. C4 stays with the caller.
bool check_pipeline_seating(
    const MeshGraphDescriptor& mgd,                     // the merged MGD: the resolver's edges and loopbacks
    const LogicalMultiMeshGraph& logical,               // also the declared channels per mesh pair
    PhysicalMultiMeshGraph& physical,                   // C1 fills its pipeline_position_* fields
    const tt::tt_metal::PhysicalSystemDescriptor& psd,
    TopologyMappingResult& full);
}  // namespace detail
}  // namespace tt::tt_fabric
```

#### One seating, end to end (`pipeline_builder.cpp`, P6)

```cpp
bool detail::check_pipeline_seating(const MeshGraphDescriptor& mgd,
                                    const LogicalMultiMeshGraph& logical,
                                    PhysicalMultiMeshGraph& physical,
                                    const tt::tt_metal::PhysicalSystemDescriptor& psd,
                                    TopologyMappingResult& full) {
    // C1 (4.1): every position of every stage shape; edges join disjoint, bordering positions,
    // through grid links or inter_mesh_cables (3.3).
    build_physical_pipeline_adjacency_graph(logical, full, psd, physical);

    // C2 (4.2): O(1) pre-checks first: area, lattice count, colour and segment on tight meshes.
    if (auto pre = pipeline_pre_checks(logical, physical)) {
        return reject(full, "pipeline_layout", pre->reason);
    }
    // One footprint solve: the stage graph, required (stage → positions it fits), resource
    // (position → slices), resource completeness when tight, conflict_cap. One carve, no loop.
    const auto footprint = solve_pipeline_footprint(logical, physical, kPipelineConflictCap);   // P5
    if (!footprint.success) {
        return reject(full, "pipeline_carve", footprint.error_message);
    }
    // The resolver on that carve, one call per pipeline: the runtime's own chip-level check (3.1).
    // Node names are the stage ids as text; every argument is read from the descriptor or looked up
    // in physical.
    PipelineMappingResult result;
    for (const PipelineDescriptorData& pipeline : mgd.pipeline_descriptors()) {
        // positions in stage order: footprint.target_to_global.at(PipelineStageNode{pipeline.name, stage.id})
        const std::vector<std::size_t> carve = carve_of(pipeline, footprint.target_to_global);
        PipelinePlacement placement;
        for (const std::size_t position : carve) {
            placement.submesh_chips.push_back(physical.pipeline_position_chips_[position]);   // already slot-local
        }
        const DirectLinks links = direct_links(carve, physical);   // pipeline_position_links_ re-keyed to submesh indices
        try {
            placement.layout = resolve_graph_layout_with_connections(
                stage_ids(pipeline), edge_tuples(pipeline), placement.submesh_chips,
                chip_counts(pipeline), core_counts(pipeline), std::nullopt, &links);
        } catch (const std::runtime_error& e) {
            return reject(full, "pipeline_layout", fmt::format("pipeline '{}': {}", pipeline.name, e.what()));
        }
        // IMPORTANT: R is the exact channel of every crossing, so phase 0 binds the cable the check used.
        // Chips come back through submesh_chips: layout.node_to_submesh + (row, col) -> (mesh, chip);
        // the channel is the PSD's next unused one between those two ASICs.
        record_required_links(placement, full, psd, result.required_links);
        result.pipelines.emplace(pipeline.name, std::move(placement));
    }
    if (auto over = first_over_demand(logical, result.required_links)) {   // channels per pair <= strict + relaxed
        return reject(full, "pipeline_binding", over->reason);
    }
    full.pipeline = std::move(result);
    return true;
}
```

File-local helpers beside it: `inter_mesh_cables` (3.3), `pipeline_pre_checks` (4.2, "Solve
cost"), `add_resource_completeness` (4.2, "No solver changes"), `carve_of` (a pipeline's
positions in stage order), the resolver arguments `direct_links`, `stage_ids`, `edge_tuples`,
`chip_counts` and `core_counts` (4.2, "From the chosen rectangles to the resolver call"),
`record_required_links` (4.2, "Which cable each crossing uses"), `first_over_demand` (4.2, "`R`
must fit each connection's demand"), and `reject`, which sets `full.success = false`,
`full.stats.failure_stage` and `full.error_message` and returns false. They return existing
types (`ChipTuple`, `EdgeInputTuple`, `DirectLinks`, `ExitNodeConnection`); there is no wrapper struct.

#### The mapper hook and where the mapping is saved (P8)

Lines without `+` are today's code.

```cpp
// topology_mapper_utils.cpp, init_from_parts: today's line, unchanged. With P5 it also fills
// logical_.pipeline_stage_graph_, pipeline_stage_shapes_ and mesh_host_grids_ from the merged MGD.
    logical_ = build_logical_multi_mesh_adjacency_graph(merged);

// topology_mapper_utils.cpp, each MultiMeshSolutionEnumerator constructor (P5): keep the merged MGD.
-   MeshGraphDescriptor merged = init_from_parts(physical_system_descriptor, parts, session_pinnings);
+   mesh_graph_descriptor_ = std::make_unique<const MeshGraphDescriptor>(
+       init_from_parts(physical_system_descriptor, parts, session_pinnings));
+   const MeshGraphDescriptor& merged = *mesh_graph_descriptor_;

// topology_mapper_utils.cpp, MultiMeshSolutionEnumerator::next
    while (true) {
        AssignedMeshes seating = placement_session_->next();
        if (seating.empty()) {
            return give_up();
        }
        TopologyMappingResult full = complete_intra_mesh_for_placement(identity, logical_, physical, ...);
        if (full.success) {
+           if (config_.check_pipelines && !logical_.pipeline_stage_graph_.get_nodes().empty() &&   // no stage graph: today's loop
+               !detail::check_pipeline_seating(
+                   *mesh_graph_descriptor_, logical_, physical, *physical_system_descriptor_, full)) {
+               last_failed_ = full;
+               if (++pipeline_rejections_ > kMaxPipelineRejections || !placement_session_->exclude_mapping(seating)) {
+                   return give_up();
+               }
+               continue;
+           }
            ++emitted_;
            return split_mapping_to_local_parts(full, per_part_local_to_global_mesh_ids_);   // remaps pipeline too
        }
        // intra-mesh failure: add_forbidden_constraint(failed seat), as today
    }

// topology_mapper.cpp, TopologyMapper::build_mapping, control host (rank 0)
+   config.check_pipelines = !rtoptions.get_ignore_pipeline_descriptors();    // kill switch (4.7)
    const auto mapping_result = map_multi_mesh_to_physical(physical_system_descriptor_, ..., config, ...);
    TT_FATAL(mapping_result.success, "Graph specified in MGD could not fit ... {}", mapping_result.error_message);
+   pipeline_mapping_ = mapping_result.pipeline;                            // rank 0 only
+   if (pipeline_mapping_) pipeline_required_links_ = pipeline_mapping_->required_links;   // broadcast with the chip info

// TopologyMapper
+   const std::optional<PipelineMappingResult>& get_pipeline_mapping() const;
+   const std::vector<tt::tt_metal::ExitNodeConnection>& get_pipeline_required_links() const;   // every rank
```

- **Saved twice, at two levels.** `TopologyMappingResult::pipeline` is what the utils return
  (per part, local mesh ids). `TopologyMapper::pipeline_mapping_` is rank 0's copy, for the
  control plane and `generate_rank_bindings`. Only `R` is broadcast, because every rank checks
  after the pairing that it was bound (4.3). A pipeline that spans parts appears in every part
  it touches (open question 23).
- **Every caller gets it the same way.** Nothing but the switch goes through the config, so
  every `map_multi_mesh_to_physical` overload and `generate_rank_bindings` (which already
  merges its parts) get the pipelines through the `logical_` they already build from the
  merged MGD.
- **`unique_shapes` off** whenever the MGD has pipelines: swapping meshes between the same seats
  can change the pipeline answer (4.4).

#### PRs

First version (STRICT), in order. Each PR lands with its tests; P5 and P6 change no behaviour.

| PR | Scope | Files | Done when |
|---|---|---|---|
| **P5** Pipeline graphs and the footprint solve (C1, C2's solve) | `PipelineStageNode` and its formatter; `LogicalMultiMeshGraph::pipeline_stage_graph_` (all pipelines merged), `pipeline_stage_shapes_` and `mesh_host_grids_`, filled by `build_logical_multi_mesh_adjacency_graph`; `MultiMeshSolutionEnumerator::mesh_graph_descriptor_`, the merged MGD kept; `PhysicalMultiMeshGraph::pipeline_position_graph_`, `pipeline_position_chips_`, `pipeline_position_slices_` and `pipeline_position_links_`; `inter_mesh_cables`; `build_physical_pipeline_adjacency_graph(logical, full, psd, physical)`; `solve_pipeline_footprint(logical, physical, cap)` on the stage graph, with resource completeness, returning the solver's `MappingResult` | utils `.hpp/.cpp`, pipeline_builder `.hpp/.cpp`, `test_topology_mapper_utils.cpp`, `test_pipeline_builder.cpp` | the stage graph of the GPT-OSS ring, the dflash fork, two pipelines that both have a stage 0, two merged MGD parts whose pipelines land in one stage graph, a repeated edge and an isolated stage; each `mesh_host_grids_` cell equals `get_coord_range` for the rank at that coordinate; stage shapes match `dims`; no pipeline: the three fields are empty; positions on mixed slice shapes; overlapping positions are never adjacent and never both chosen; brute-force agreement up to 8 stages; a stage that cannot fit is filtered before the solve; cable hashes equal today's gather on the SC16 and galaxy PSDs |
| **P6** The per-seating check (C2) | `PipelinePlacement`, `PipelineMappingResult`; pre-checks; the resolver arguments; one resolver call per pipeline on P5's carve; `R`; the demand check; `check_pipeline_seating(mgd, logical, physical, psd, full)` | utils `.hpp`, pipeline_builder `.hpp/.cpp`, `test_pipeline_builder.cpp` | blaze-rejected rings accepted; `mixed4` (P,R) accepts, (P,Q) and Case B reject; area and counting failures stop at the pre-check; a ring over a count-1 connection rejects with `pipeline_binding`; a reject sets `failure_stage` and `error_message` on the result |
| **P8** Tier C in the mapper (C4) | `check_pipelines`; the `next()` hook on `logical_` with `exclude_mapping` and the rejection cap; `unique_shapes` off; result, stats and part split; `TopologyMapper` storage and `R` broadcast | utils `.hpp/.cpp`, `topology_mapper.cpp`, `generate_rank_bindings.cpp` | `mesh4x4_stage4x2` Cases A, B, C: accept, reject, reject; GPT-OSS galaxy and llama_8b_pod accepted and built by blaze; the swapped-seat regression; no pipeline: every mapper golden unchanged |
| **P9** Bind `R` (C3) | phase 0 in `pair_logical_intermesh_ports`; single-host phase 0; every rank checks `R` was bound | `control_plane.cpp` | the required channel wins over a lower hash; an unplaceable required channel fails on every rank together, no hang; no pipeline: `AnnotatedIntermeshConnections` byte-identical |
| **P15** Exit-node retry (C4) | retry the intra-mesh solve with the failing slice signature blocked, up to `kMaxLayoutRetries` | `topology_mapper_utils.cpp` | a seating whose first layout fails and second passes is accepted after one retry |
| **P12** Witness plan | write the stage plan into rank bindings (4.6) | `generate_rank_bindings.cpp` | the plan round-trips; nothing reads it yet |

Later, each only if needed: P10 (constraints before the first solve, 4.4), P13 (seat-pair
nogoods, 4.4), P14 and P11 (RELAXED, 1.4 and 3.4), P16 (the policy plan's candidate table in
place of `inter_mesh_cables`). The canvas's *Implementation plan* tab has every PR's steps,
code and tests.

#### Example: `mixed4` on (P, R)

Input: one pipeline, stages s0 (4x2), s1 (4x4), s2 (4x4), s3 (4x2), edges s0→s1, s1→s2,
s2→s3, s3→s0 (loopback). Output:

| Field | Value |
|---|---|
| `submesh_chips` | four submeshes: s0 M0 cols 0–1, s1 M0 cols 2–5, s2 M1 cols 2–5, s3 M1 cols 0–1 |
| `layout` | `stage_order` s0, s1, s2, s3; `resolved_edges` s0→s1 and s2→s3 inside a mesh, s1→s2 exits on A2 and enters on B2, s3→s0 on B0 → A0; H2D and D2H on two chips of A0 |
| `required_links` (`R`) | the two exact channels, with `connection_hash`: `R[0]` is s1→s2 and `R[1]` is s3→s0, the two crossing edges in `resolved_edges` order |
| derived by P12 | host ranks s0 {0}, s1 {1, 2}, s2 {1, 2}, s3 {0}; idle M0 {3}, M1 {3} |

### 4.1 Step C1 - build the physical pipeline graph

C1 turns one seating and its intra-mesh layout into the one graph the pipeline is solved on:
the *position graph*. It runs after intra-mesh success, because the layout decides which chips
sit under each host slice, and so where each cable lands. C1 does not carve. The carve, the
stage order and the crossings are all decided by the solve in C2.

- **Inputs.** The seating, `full.fabric_node_to_asic`, the inter-mesh cables for every MGD
  connection (`inter_mesh_cables`, 3.3), and the links the intra-mesh topology
  routes over.
- **Output.** The seating's `PhysicalMultiMeshGraph` gains its `pipeline_position_*` fields, in the resolver's own types: the chips
  of each position (`ChipTuple`s) and, for each pair of touching positions, the candidate links
  between them (`DirectLinks`). Channels are not stored; `R` looks up the exact channel in the
  PSD once the resolver has picked a crossing's two chips (4.2).
- **What it does not decide.** The carve, stage order or crossings (C2), which channels are
  bound (C3), or what SAT offers next (C4).

```
G = build_physical_pipeline_adjacency_graph(logical, full, psd, physical)  pipeline_builder.cpp, fills physical.pipeline_position_*
      slices:    logical.mesh_host_grids_[m]: one coordinate range per host, chips through full.fabric_node_to_physical
      shapes:    logical.pipeline_stage_shapes_, the distinct stage shapes
      cands:     inter_mesh_cables(psd, full, a, b) per connected pair (3.3) + routed intra-mesh links
      positions: every rectangle of slices that a stage shape fills, per mesh (they overlap); index i
      edges:     two disjoint positions, a slice of one bordering a slice of the other
      links:     per edge, the candidate links behind it, as DirectLinks keyed (i, j)
C = built inside solve_pipeline_footprint (4.2), target logical.pipeline_stage_graph_
      required: stage -> positions of its shape; resource: position -> covered slices
```

**The graph is slice-level, and the chips stay attached.** The solve only needs positions and
which positions touch. The check after it does not: per-chip pipeline-core capacity, a stage
whose entry and exit land on the same chip, and the chips stage 0's host I/O uses, are
chip facts. So each position keeps its chips and each edge
keeps the links behind it, ready to become the resolver's `submesh_chips` and `direct_links`
(3.1) for whichever rectangles the solve picks.

#### Links

Only links the control plane can route over count:

- *between meshes:* `inter_mesh_cables` (3.3) returns every PSD cable between the two mapped
  meshes, with the one-destination-chip-per-source-chip rule. Link state and cable types come
  later, with the policy plan's table (P16). The PSD holds every host's cables, so this needs no MPI exchange. Which of them get
  bound is decided in C3, from the channels C2 records in `R`.
- *inside a mesh:* the mapped pairs of logical neighbours in `device_topology`, RING wraps
  included. Extra cables outside the topology are ignored.

To do:

- In `build_physical_pipeline_adjacency_graph`, call
  `inter_mesh_cables(psd, full.fabric_node_to_asic, a, b)` for each mesh pair the MGD connects.
- For each position-graph edge, keep the links behind it as `LinkPair`s in
  `PhysicalMultiMeshGraph::pipeline_position_links_` (`DirectLinks`, keyed by the two position indices). C2
  re-keys the chosen positions' entries to submesh indices for the resolver, and looks up the
  exact channel for `R` in the PSD from the two chips the resolver picked (4.2, "Which cable
  each crossing uses").
- Order each edge's links by chip id (by best tier first once P16 lands).

#### Slices and adjacency

For each mesh, `logical.mesh_host_grids_` is the host grid: the cell at a coordinate is that
host's slice, a rectangle of device coordinates (`get_coord_range`). It is the same slice Tier B
divides out of the declared topology. Slices are numbered by (mesh, grid coordinate); the ids
are global across meshes, so `pipeline_position_slices_` can feed
`add_resource_constraint` directly. Two slices are
adjacent when a link above joins them, matching blaze's `discover_slice_adjacency`.

#### The position graph

`build_physical_pipeline_adjacency_graph`: for each stage shape and each mesh, every rectangle
of host slices that the shape fills exactly is a *position*. Positions overlap: the graph lists
every place a stage could go, and the solve's resource constraint (a slice is used by at most
one chosen position) keeps the chosen ones apart, the same way SAT seating keeps overlapping
seat footprints apart. Two positions are adjacent when they do not overlap and some slice of
one borders some slice of the other, by the mesh's own adjacency or by a candidate cable. This one graph covers blaze's planned carve, per-rank tiling and
single-mesh tiling (3.4), except a stage smaller than a host slice (open question 15).
Positions are generated from each stage's chip shape per mesh, so host slices of different
shapes need no special case: an 8-chip stage is one 4x2 slice on one mesh and a 2x2 block of
2x2 slices on another. Several pipelines are already one target graph here (1.5): one
component per pipeline, no edges between them, and one solve of this position graph places
every component. The resource constraint keeps them on disjoint slices.

In Case B (both crossings on one slice pair, 2.3) with every edge STRICT, the solve finds no
mapping, and the seating goes to C4. RELAXED (later): with `s3 → s0` a soft edge (3.4), the
same Case B is accepted: `s3 → s0` is left unmatched (host), and the chain `s0..s3` maps onto
`M0b → M0a → M1a → M1b`, with both crossings between slices `M0a` and `M1a`.

**Failure.** A stage with no position on any mesh of its binding set (wider than the mesh, or a
shape no whole number of slices builds) reports `failure_stage = "pipeline_carve"` before any
solve. Everything else is decided in C2.

### 4.2 Step C2 - pipeline solve

C2 maps the pipeline onto C1's position graph with the topology solver, once, then checks
that one carve with the builder's own resolver. There is **one C1 graph and one carve per
seating**. If the carve fails the resolver or the demand check, the seating fails and C4 runs;
C2 does not ask for another carve ("When C2 fails", below).

```
G, C  = C1's position graph and constraints                               (4.1)
pre   = area, lattice count, colour and segment tests; drop ruled-out cables
carve = one solve of (pipeline, G, C) under conflict_cap                  (P5, existing API)
    none             -> C4 (exit-node retry, then exclude the seating)     (4.4)
layout = resolver on the carve's rectangles (the check)                   (3.1)
    resolver throws  -> C4 (the message is the reason)
R = for each crossing edge in layout.resolved_edges: its exact channel    (IMPORTANT)
    R over demand    -> C4 (the mesh pair is the reason)
accept: carve, layout, R                                                  (C3)
```

**The solve.** `solve_pipeline_footprint` in `pipeline_builder.cpp`: one SAT solve over
`<PipelineStageNode, std::size_t>`. The target is `logical.pipeline_stage_graph_` (one node per
stage, its STRICT fabric edges, both directions, a repeated edge as a duplicate entry); the global graph is C1's
`PhysicalMultiMeshGraph::pipeline_position_graph_`; the constraints it builds: each stage required onto
the positions it fits on its own (its shape; at least as many core slots on the position's
chips as the stage has endpoints; for stage 0, room for host I/O), and a resource
constraint from each position to the slices it covers. A mapping is a carve: one rectangle per
stage, disjoint, every pipeline edge between touching rectangles, the ring closed. It is also
the stage-to-slot assignment, and for every crossing edge it fixes the mesh pair and the two
rectangles the edge crosses between. `Auto` selects the SAT engine whenever a resource
constraint is present, which is always here. The call is a `TopologyMappingEnumerationSession`
built with `conflict_cap` and asked for one `next()`; `solve_topology_mapping` is the same solve,
but it fixes the cap at the SAT default (300,000 conflicts).

**No solver changes.** Everything C2 needs is in today's `MappingConstraints` and
solver API. The pipeline code builds the constraints; the
topology solver, and `topology_solver_sat.cpp` in particular, gets no pipeline-specific or
session-specific code.

| Need | Existing API | How |
|---|---|---|
| stage on positions of its shape | `add_required_constraint(stage, positions)` | per stage, in C1 |
| disjoint rectangles | `add_resource_constraint(position → slices)` | in C1 |
| resource completeness (tight carves) | `add_cardinality_constraint(pairs, 1)` | when the stage areas add up to exactly the slice count, one constraint per slice over every (stage, position) pair whose position covers it. With `min_count = 1` the encoder emits one plain clause (`topology_sat_add_at_least_k_literals`), the same clause as the experiment (report 5.4: 66 s to 0.59 s, 170 s to 0.32 s) |
| a stage that can never sit on a position | left out of `add_required_constraint` | in C1, before the session (below) |
| bound the solve | the session's `conflict_cap` (constructor) | deterministic, unlike a wall clock |

The experiment's in-solver completeness clause (`TT_TOPOLOGY_SAT_RESOURCE_COMPLETENESS` in
`topology_solver_sat.cpp`, uncommitted) is not part of the plan; the cardinality constraint
replaces it. One limit of the existing API: a solve that hits `conflict_cap` returns the same
"no mapping" as a proof (`solve_limited` status other than SAT is decoded as false), so C2
cannot tell the two apart (open question 13).

**Pre-checks, before the solve.** Area (stage areas must fit the slices of the binding set);
the lattice count for a shape (how many disjoint copies the grid holds); and, for every tight
mesh with one candidate cable per neighbour pair, the colour and segment test ("Solve cost"
below). These decide the cases a SAT solve is worst at, infeasibility, and each is O(1).

**The check: can the runtime build this carve?** The solve works on slices. The runtime, when
blaze builds the pipeline, decides the chips inside each slice: the exit and entry chip of every
edge, the core slot on each chip, and stage 0's host I/O chips. It does that by calling the
resolver (3.1) whatever the mapper did. The one way the runtime can fail on a carve the mapper
accepted is a chip collision: the endpoints a slice must host do not fit its chips (two cables
landing on the same chip, or more endpoints than the slice has core slots). The carve cannot
see that. So C2 runs the resolver itself on one submesh per chosen rectangle
(`resolve_graph_layout_with_connections`, the contract below). The chip counts force its
placement step, so its link step is the check. It runs once, on the one carve.

#### Which cable each crossing uses (IMPORTANT)

The mapping says two rectangles on different meshes touch through *some* candidate cable.
When several cables join them, the mapping does not pick one. The resolver's link step does:
`direct_links` carries every candidate link between the two slots, ordered by chip id,
and `resolved_edges` returns the exit and entry chip of the link it used. So `R` comes from the
resolver, not from the mapping:

1. For each resolved edge whose two slots are on different meshes, read its exit and entry
   chips back from `submesh_chips`: `layout.node_to_submesh` gives the submesh, the edge's
   `(row, col)` the `ChipTuple`, and its `(mesh_id, chip_id)` the `FabricNodeId`.
2. Look that chip pair up in `inter_mesh_cables` (3.3), which holds the PSD's
   `ExitNodeConnection` entries between the two ASICs, and take one channel: the lowest
   source ETH channel (best tier first once P16 lands). The pairing has no order of its own to
   match here (it trims by hash value); phase 0 binds the channel named.
3. Append that PSD `ExitNodeConnection` to `R` as is (4.3); its `std::hash` is the pairing's
   `connection_hash`.

> **IMPORTANT: record the exact channel the solve used, and bind that channel.** The carve is
> only valid on the cables the check placed it on. If the control plane's inter-mesh pairing
> (`pair_logical_intermesh_ports`) binds any other channel between the two meshes, even one
> between the same two slices, the cable can land on a different chip or slice and the
> runtime's resolver can fail on a layout the mapper accepted. So:
>
> - C2 stores the channel, not just the mesh pair or the chip pair, for every crossing edge.
> - C3 publishes it through `get_pipeline_required_links()`, and phase 0 of the pairing binds
>   exactly these channels, by `connection_hash` (the PSD entry's hash), before any other channel
>   (4.3).
> - A recorded channel that cannot be bound at control-plane init is a hard error naming the
>   channel and its two chips. It is never silently replaced by another cable.
> - The demand check below counts these recorded channels.

#### When C2 fails

| Failure | Found by | Result |
|---|---|---|
| stage `s` cannot fit on position `p` on its own (more endpoints than `p`'s core slots; stage 0's host I/O does not fit) | C1, before the solve | `p` left out of `s`'s required positions; never offered |
| a pre-check fails | before the solve | seating fails: C4, with the mesh and reason |
| no carve (or the conflict cap hit) | the solve | seating fails: C4 |
| the resolver throws on the carve | the check | seating fails: C4, the message is the reason |
| `R` over demand on mesh pair (A, B) | the demand check | seating fails: C4, `pipeline_binding` and the pair |

**Why one carve is enough.** On every benchmark case the resolver accepted the first carve the
footprint solve returned: 101 of 101 across the designed, repo, encoding, heterogeneous,
no-1x1, multi-mesh and sweep groups (`results/rerun_2026-10-27/footprint_then_resolver/`,
at most 4 ms each). The solve already decides what the resolver checks at slice level (shape,
disjointness, touching rectangles, a candidate cable behind every crossing), and C1 removes the
per-stage chip limits. What is left, two cables of one stage landing on the same chip, is rare.
The one case built to hit it (`chip_level/reject_demo_*`) has no passing carve at all, so a
second carve would not have saved it.

**What it costs.** A seating can fail on its first carve while another carve on the same
seating would pass. The mapper then tries the next seating, so a good mapping is still found if
another seating carries it, but the seating itself is lost. Open question 24 tracks it. If a
real MGD hits it, the fix needs no new API: ask the same session for another carve with
`exclude_mapping(carve)` and `next()`, a few times, before failing the seating.

#### Speed: blaze's carve planner + resolver against the footprint + resolver

The two ways of getting from a pipeline to a checked layout, on the same 132 carve-requiring
cases on one machine, 60 s cap (report section 5.3;
`results/rerun_2026-10-27/carve_comparison.md`). *Planner + resolver* is blaze's ring carve
planner (`plan_ring_carve`, ported to C++ in `topo/carve_planner_port.hpp` with blaze's 200k
expansion and 64-cycle budgets), then the resolver on its slots. *Footprint + resolver* is the
solve above with resource completeness, then the resolver on its rectangles.

| Group | Cases | Planner + resolver: placed | Footprint + resolver: decided | Resolver after the footprint, worst |
|---|---|---|---|---|
| original carve cases | 17 | 0 (17 budget-exhausted) | 16 | 0.029 s |
| heterogeneous sweep, 60 to 95% fill | 30 | 0 (30 budget-exhausted) | 30 | 0.001 s |
| single mesh, no 1x1 stages | 45 | 0 (stage-0 rule; with it off, 2) | 43 | 0.029 s |
| heterogeneous multi-mesh | 12 | 0 (stage-0 rule; with it off, 0) | 12 | 0.001 s |
| multi-mesh, no 1x1 stages | 28 | 0 (stage-0 rule; with it off, 0) | 26 | 0.036 s |
| **all** | **132** | **0** | **127** | **0.036 s** |

Per case, seconds as carve + resolver:

| Case | Planner + resolver | Footprint + resolver |
|---|---|---|
| 4x4: 1x1, 3x 2x2, 1x1 | 0.000 + 0.000 | 0.000 + 0.000 |
| 4x8: 1x1, 2x4, 2x4, 2x2, 1x1 | 0.009 + 0.000 | 0.001 + 0.000 |
| 4x8 ring of 32 x 1x1 | 0.038 + 0.001 | 0.004 + 0.001 |
| 4x8 ring of 16 x 1x2 (planner with its stage-0 rule off) | 0.036 + 0.000 | 0.002 + 0.000 |
| 4x4: 1x1, 3x 2x2 (feasible) | wrong reject: `no-cycle-of-length-13` | 0.000 + 0.000 |
| 4x6 and 4x8: 1x1, 2x2 ring, 1x1 (feasible) | wrong reject: `no-alignment` | 0.001 + 0.000 |
| 8x8 fill: 1x1, 15x 2x2, 3x 1x1 | budget exhausted | 0.026 + 0.000 |
| 16x16: 1x1, 62x 2x2, 1x1 | budget exhausted | 0.13 + 0.004 |
| random 8x8, 30 stages | budget exhausted | 0.60 + 0.001 |
| random 8x16, 37 stages | budget exhausted | 5.6 + 0.001 |
| 4 meshes of 4x8, 1 cable per pair, ring of 112 x 1x1 | budget exhausted | 9.9 + 0.020 |
| 7x7: 1x1, 10x 2x2 (infeasible) | budget exhausted, no verdict | no, 2.1 |
| 5x5 ring of 25 x 1x1 (odd, infeasible) | budget exhausted, no verdict | no, 0.003 |

**Use the footprint + resolver.** The planner only completes on small rings of columns, where
both take milliseconds. On grid-shaped meshes its slice graph has no degree-2 chains to
contract, so its cycle search is a raw Hamiltonian-cycle search and runs out of budget on every
8x8 or larger grid and every multi-mesh case. Its stage-0 rule rejects every case without a
1x1 stage, and its first-fit grouping wrongly rejects 16 cases the footprint carves in
milliseconds. In the footprint path the carve is the whole cost: the resolver checks every
carve in at most 36 ms.

#### From the chosen rectangles to the resolver call

C2 feeds the resolver what blaze feeds it, with the solve's rectangles as the slots, one per
stage. It catches per-chip core capacity, shared entry and exit chips, and stage 0's host I/O.
The six arguments:

| Argument | What C2 builds | Rule the resolver enforces |
|---|---|---|
| `nodes` | every stage name, in declaration order, isolated stages included | every edge endpoint must be listed; no duplicates |
| `edges` | `(src, dst, is_loopback)` per STRICT fabric edge; `is_loopback` is the MGD edge's `loopback`, which Tier A has checked (2.1). RELAXED (later): edges left to host are dropped, and each fabric chain is its own call (`_resolve_chains_separately`) | the non-loopback edges must form a DAG, or the sort throws `cycle detected` |
| `submesh_chips` | one entry per slot: `(mesh_id, chip_id, row, col)` for each chip, `chip_id` from `full.fabric_node_to_asic` | `(row, col)` are **slot-local**: relative to the slot's own origin, recomputed when a slot spans several host slices |
| `node_chip_counts` | `rows x cols` of each stage's `shape` (already in chips) | a stage only binds to a slot with exactly that many chips, so the placement step is forced to the solve's carve |
| `node_pipeline_core_counts` | blaze's `_pipeline_core_defaults` per node: ring head 4 with `co_locate_ring_host_io`; chain head and tail 2 | empty means 1 slot per chip on slots of 8 or more chips, 2 on smaller ones |
| `direct_links` | `DirectLinks`: per ordered slot pair, every `LinkPair{exit_row, exit_col, entry_row, entry_col}` | **directed per key**: a cable usable both ways appears under `{i, j}` and `{j, i}`, each with its own exit and entry chips |

Steps, in order:

1. **Links and slices.** C1 already stores both in the resolver's types (4.1): each position's
   chips as slot-local `ChipTuple`s in `pipeline_position_chips_`, and every candidate link between two
   touching positions (not only the ones that will be bound) as `LinkPair`s in `pipeline_position_links_`, under
   both `{i, j}` and `{j, i}`, ordered by chip id (best cable tier first once P16 lands).
2. **Slots.** `carve_of(pipeline, target_to_global)`: the positions of the pipeline's stages,
   in stage order, each looked up as `target_to_global.at(PipelineStageNode{pipeline.name,
   stage.id})`. A slot may span several host slices.
3. **`submesh_chips`.** `pipeline_position_chips_[p]` for each slot `p`, copied as is: the `(row, col)`
   were made slot-local when C1 built the position. No reverse map is kept: each `ChipTuple`
   already holds `(mesh_id, chip_id)`, so `R` reads a resolved chip back from `submesh_chips`.
4. **`direct_links`.** For each pair of slots `(a, b)` at submesh indices `(i, j)`, copy
   `pipeline_position_links_[{a, b}]` under `{i, j}`. Pairs with no entry do not touch. Nothing is recomputed.
5. **Stage inputs.** `nodes`, `edges` with the MGD's loopback marks, `node_chip_counts` from each
   stage's `dims`, and the core counts, all read from the pipeline's entry in
   the merged MGD's `pipeline_descriptors()`. The resolver names nodes with strings, so a node's name is its
   `PipelineStageId` as text (`std::to_string(stage.id)`), and `layout` reads back the same
   way. They cost a few string copies per pipeline per seating.
6. **Guard stage 0.** Stage 0 needs its edge links plus 2 slots for H2D and D2H. A one-chip slot
   has 2 slots and fails on capacity before any search, so C1 never offers stage 0 a one-chip
   position (with 4x2-chip slices stage 0 has 8 slots and needs 4).
7. **Call and read back.** `resolve_graph_layout_with_connections(nodes, edges, submesh_chips,
   node_chip_counts, node_pipeline_core_counts, std::nullopt, &direct_links)`.
   `resolved_edges` gives each edge's exit and entry chip in slot-local coordinates; H2D and
   D2H chips are separate fields. `R` is recorded from it as in "Which cable each crossing
   uses".

Everything in this list is new code in the resolver-argument helpers (P6), returning the
resolver's own types; none of it is a resolver change.

**Core counts.** C2 computes blaze's defaults from the graph shape when it builds the call:
the host I/O stage is the head, and it is a ring head when a loopback enters it. Nothing is stored in
the MGD. **Gap:** harness `pipeline_core_coords()` overrides are not in the MGD, so Tier C
could accept a layout runtime rejects (open question 7).

#### `R` must fit each connection's demand

Phase 0 takes one channel per link in `R`, counted against the mesh pair's demand from
`get_inter_mesh_channel_demand` (policy plan, part 3):

| Rule | Detail |
|---|---|
| budget | channels in `R` on a pair ≤ `strict + relaxed`. Tier B's channel rule does not guarantee this: it checks that *some* projection fits, and the solve may pick a carve with more crossings |
| counting | `R` uses the STRICT part of the demand first, then the RELAXED part |
| RELAXED and BEST_EFFORT connections | a pair with no link in `R` keeps the policy plan's rules, zero-link included. A pair with links in `R` needs them: a STRICT pipeline edge needs fabric, whatever the connection's policy |
| cable types | disallowed types are never candidates, so never in `R` |

To do:

- Check the budget in C2 as soon as `R` is recorded. Over budget, the seating fails with
  `failure_stage = "pipeline_binding"` and the mesh pair as the reason, and C4 runs.
- If this check fires in practice, make the demand part of the solve instead: a per-mesh-pair
  cap on crossing edges in the footprint encoding, so over-demand carves are never offered.

#### Solve cost: fail fast

A SAT solve is fast when a carve exists and slow when none does, and "none does" is the common
case in this loop: every rejected seating is a proof. So every C2 run ends in a definite
answer, carve or proved impossible, and an impossible graph must be proved quickly enough that
C4 can afford one per seating.

The slow failures are all one shape, and it has an exact cheap test:

- **Tight meshes split into fixed segments.** When a mesh's slices are all used (one-slice
  stages tiling every rank), the pipeline's visit to that mesh is a path through every slice
  between the two cable slices where it enters and leaves. With one candidate cable per
  neighbour pair the ring must use every cable once, so the two slices are fixed.
- **Each segment is decided in O(1).** A Hamiltonian path between two given cells of a full
  rectangular grid exists exactly under the Itai-Papadimitriou-Szwarcfiter conditions (1982):
  the colour condition (opposite colours on an even grid, both on the majority colour on an
  odd one) plus a few forbidden cases on grids 1, 2 or 3 slices wide. An odd grid holds no
  closed ring. This decides every tight 1-cable case before the first clause.
- **Pre-check order in C2.** (1) Per tight mesh, keep only (entry, exit) cable pairs that pass
  the segment test; if none is left, the seating fails with `pipeline_layout` and the mesh,
  cells and colour in the reason. (2) Build the position graph with only the surviving
  cables, so the solve never finds a carve on a segment already ruled out. (3) Meshes with
  spare slices skip the test; with several cables per pair the test only certifies, it does
  not refute, because a tight mesh may be visited in two segments.

**What this does not cover.** Hamiltonian path is NP-complete on grids with holes (same
paper), so a mesh with dead slices that is also tight can still be slow, and so can a tight
ring over several meshes with several cables per pair. The bound is the session's
`conflict_cap`; a solve that hits it never passes, but with today's API it reads as "no
mapping", so the seating goes to C4 like a proof would. Open question 13 asks whether that is
acceptable or the result should carry a stopped flag.

The experiments behind these numbers are in
[`pipeline_placement_experiments/`](pipeline_placement_experiments/README.md).

#### What changes in the resolver, and when

| Change | Needed by | Where | Size |
|---|---|---|---|
| none | P8 first version: accept or reject the one carve | | 0 lines |
| none later either | the C1 position filter replaces stage-level forbids; P15 reads its signature from the intra-mesh layout | | 0 lines |

A structured failure (`try_resolve_graph_layout_with_connections` returning a variant) was
planned and is dropped: nothing reads it soundly. Splitting `discover_submesh_links` behind a
link-query callback is not needed
by C2: C2 never discovers links, it supplies them. It is a Tier D refactor and is deferred.

#### Failure reporting

`failure_stage = "pipeline_carve"` (C1), `"pipeline_layout"` or `"pipeline_binding"` (C2),
carrying the pre-check that fired, "no carve" from the solve, the resolver's message, or the
mesh pair over its demand. When the master session is exhausted it surfaces
through the existing *"Graph specified in MGD could not fit in the discovered physical
topology"*.

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
| [Fill order](#new-fill-order-after-the-required-links) | same | after phase 0 | reused: today's order |
| [Single host](#new-single-host) | `generate_intermesh_connections_on_local_host` | control-plane init | new |
| [Phase 0 inside the mapper](#later-phase-0-inside-the-mapper-policy-plan-part-11) | mapper, `resolve_inter_mesh_binding` | before emit | later, with the policy plan's part 11 |

The cables C3 binds from are read in C1 (4.1), and the check that `R` fits each
connection's demand is part of C2's acceptance (4.2).

**Order matters.** Binding comes after C2, so the carve is chosen over every candidate cable
and the binding follows it. Binding first and carving over the bound set would throw away
carves that a different binding allows (the `mixed4` example below).

**No rerun over the bound cables.** Once phase 0 binds all of `R`, the witness layout, which
crosses only on `R`, is still a layout over the bound set, so the solve is not rerun.

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

`R` is, for each crossing edge in the layout's `resolved_edges`, the exact channel it uses:
the exit chip and ETH channel on the source mesh, the entry chip and ETH channel on the
destination mesh, and the cable's connection hash. C2 records it when it accepts the mapping
(4.2, "Which cable each crossing uses"). Several crossing edges may share one channel; `R`
holds it once, with every edge that needs it.

**What `R` is not.** It is not every bound link. It holds only the channels pipeline edges
cross between meshes: one per crossing edge (loopback edges included). Edges inside a mesh
are not in it; they use intra-mesh fabric, which the pairing does not touch. The MGD's
channel count between two meshes is usually larger than the pipeline's crossings; the pairing
fills the rest after phase 0, in the policy plan's order. So `R` is a subset of what the
control plane binds, and the pairing's output is still the full list.

**Taken from the PSD, in the pairing's own terms.** The pairing names a channel only by its
`connection_hash`, and every rank turns that hash back into its local ETH channel from its own
PSD copy (`convert_port_descriptors_to_intermesh_connections`, `cable_lookup`). So `R` must
carry the PSD's own record, not a rebuilt one:

- The hash is `std::hash<ExitNodeConnection>` of the PSD entry from
  `get_connecting_exit_nodes`. It covers the two ASIC ids, both channels, `is_local` and
  `port_type`, and is the same from either end. A hash rebuilt from `(asic, chan)` alone would
  not match, so `R` stores the `ExitNodeConnection` and takes the hash from it.
- One PSD `ExitNodeConnection` is one ETH channel. A cable with several channels is several
  entries with different hashes. `R` names exactly one of them per crossing.

> **IMPORTANT.** `R` is a list of channels, not of mesh pairs or chip pairs. The control
> plane's inter-mesh pairing must bind exactly these channels in phase 0, so the cables the
> runtime gets are the ones the mapper's check placed the carve on. Binding any other cable
> between the same two meshes, even between the same two slices, can move an endpoint onto a
> chip the runtime's resolver then rejects.

No new type. The PSD's `ExitNodeConnection` already holds everything the control plane
needs: both ASICs, both ETH channels, `is_local` and `port_type`, and `std::hash` of it is the
`connection_hash` the pairing and the broadcast use. The two `FabricNodeId`s come from the
mapping (`get_fabric_node_id_from_asic_id`), and the pipeline edge a channel carries follows
from the order: `R` is appended in `layout.resolved_edges` order, one entry per crossing edge,
so neither is stored again.

```cpp
// PipelineMappingResult (above)
std::vector<tt::tt_metal::ExitNodeConnection> required_links;   // R: one PSD entry per crossing

// TopologyMapper, every rank (broadcast with the chip info)
const std::vector<tt::tt_metal::ExitNodeConnection>& get_pipeline_required_links() const;

// ControlPlane, rank 0: phase 0 binds these hashes first
AnnotatedIntermeshConnections pair_logical_intermesh_ports(
    const PortDescriptorTable& port_descriptors, const std::unordered_set<std::size_t>& required_hashes);
```

Broadcast reuses `exit_node_connection_to_proto` / `proto_to_exit_node_connection` from the
PSD serialization (`physical_system_descriptor.proto` already has `ExitNodeConnection`).

Where `R` lives, and why:

| Option | Verdict |
|---|---|
| in the MGD, as device-level connections | no: `R` depends on the seating and the intra-mesh layout. Pinning it in the MGD fixes a carve before the solve (the rule in section 4) and needs a different MGD per machine |
| in `rank_bindings.yaml` | not in the first version: the runtime mapper runs Tier C itself and gets the same `R`. It can be written for debugging, like the stage plan (4.6) |
| mapper output, next to the mapping | **yes** |
| the `PortDescriptorTable` handed to the pairing | **no**: each rank builds its own table and sends it to rank 0 as proto with only `connection_hash`, `src_chip` and `dst_chip`; the other ranks have no `R`, and a `required` flag would not survive the serialization |
| an argument to `pair_logical_intermesh_ports` on rank 0 | **yes**: the set of `R`'s connection hashes, matched against the merged table there |

To do:

- Fill `R` at the end of C2 from the accepted layout, one recorded channel per crossing edge
  (IMPORTANT, 4.2), and store it on the `TopologyMapper` with the mapping. Empty without a
  pipeline descriptor.
- Pass `R` to `pair_logical_intermesh_ports(port_descriptors, required_hashes)` on rank 0.
  No change to `PortDescriptor` or its serialization.
- Broadcast `R` with the chip info, so every rank can check the pairing's result (below).

#### How `R` reaches the pairing, and how the result reaches every rank

Today's multi-host path (`convert_port_descriptors_to_intermesh_connections`):

1. **Every rank** gathers its own cables from the PSD (`gather_intermesh_cables_for_exit_nodes`):
   only chips it owns, only channels whose ETH link is up on its own cluster, one
   `PortDescriptor{connection_hash, src_chip, dst_node}` per channel.
2. **Every rank but 0** sends its table to rank 0 (`forward_descriptors_to_controller`,
   proto: hash, src chip, dst chip). Rank 0 merges them.
3. **Rank 0** pairs (`pair_logical_intermesh_ports`). It joins the two sides of each boundary by
   hash: a channel is eligible only if both endpoint ranks gathered it.
4. **Rank 0 broadcasts** `AnnotatedIntermeshConnections` (`forward_intermesh_connections_from_controller`):
   `((mesh, port_id), (mesh, port_id), connection_hash)` per channel, both directions.
5. **Every rank** looks each hash up in its own PSD copy to find its physical ETH channel and
   peer, and commits `exit_node_directions_` and `intermesh_chan_to_peer_`.
6. **Every rank** runs `validate_requested_intermesh_connections` on the same broadcast list,
   so a violation throws on every rank together.

Where `R` fits, and what this means for it:

| Step | With `R` | Correct? |
|---|---|---|
| 1-2 gather and forward | unchanged; `R` is not in the table | yes: rank 0 already sees every gathered channel by hash |
| 3 pair | phase 0 takes `R`'s hashes as an argument and places each link holding one first | yes, with the trim fix below |
| 4 broadcast | unchanged: a required channel goes out as one more tuple with its hash | yes: the hash is the PSD's, so step 5 binds exactly that channel on both ends |
| 5 apply | unchanged | yes |
| 6 validate | new rule: every hash in `R` appears in the broadcast list | needs `R` on every rank, hence the broadcast with the chip info |

Three things today's code would get wrong without care:

- **The trim drops channels by hash order.** `has_budget_after_trim` cuts a link's sorted hash
  list to the remaining budget, so it can cut the required channel and keep another one on the
  same two chips. Phase 0 must move the required hashes to the front before trimming.
- **A throw on rank 0 hangs the others.** Phase 0 runs on rank 0 before the broadcast; a
  `TT_FATAL` there leaves every other rank waiting in `recv`. Phase 0 must not throw. It places
  what it can, and the step-6 check fails on every rank together, naming the channel and its
  two chips (`pipeline_binding`).
- **A required channel may be down, or not gathered.** The mapper reads the PSD on rank 0,
  which has no link state; step 1 drops channels whose link is down on the owning rank. Such a
  channel never reaches the pairing, and step 6 fails as above. The policy plan's candidate
  table is meant to carry liveness (open question 20).

#### New: phase 0 in the pairing, bind `R` first

In `pair_logical_intermesh_ports(port_descriptors, required_hashes)`, add a phase 0 ahead of
phase 1a:

1. Take every link (chip pair) holding a hash in `required_hashes`, boundary by boundary.
2. Move its required hashes to the front of `connection_hashes`, then trim to the budget, so
   the trim never drops a required channel.
3. Place it with its boundary's Z rule (assign_z: Z; otherwise NESW, then Z) and count it
   against the budget. The broadcast tuples carry exactly the required hashes, so every rank
   binds exactly those channels.
4. Run today's work-list for the rest of the budget, skipping links phase 0 placed.

With no required links, the result is exactly today's.

**When phase 0 cannot place a link** (a direction already owned toward another mesh,
Z-ness, the one-peer-per-direction rule, no ports left, or the channel was not gathered):
phase 0 does not throw on rank 0. The broadcast goes out, and the new rule in
`validate_requested_intermesh_connections` fails on every rank together, naming the channel
and its two chips (`failure_stage = "pipeline_binding"`). `R` is at most the requested
count, so this is rare; the likely case is an exit chip cabled toward two neighbouring meshes.
Moving phase 0 into the mapper turns it into a rejected seating instead (the last two
subsections).

#### New: fill order after the required links

After phase 0, the remaining budget is filled in today's order (best cable tier first,
`PortDescriptor::tier`, once the policy plan lands), with or without a pipeline, so goldens do
not move. The pipeline only needs `R`, which phase 0 has already bound; spare links are not
reordered.

#### New: single host

`generate_intermesh_connections_on_local_host` is a separate greedy pass. It needs phase 0: place the
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

**Spare links.** Take the count as 3 instead. `R` takes two, and one more is filled from
the spare A0–B0 in today's order.

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

- **Live link state.** Until P16, C1 reads the PSD on rank 0, which has no link state, so `R`
  can name a down channel; the check after the broadcast then fails on every rank (question
  20). With the policy plan's candidate table (P16) down links are left out at runtime. Offline (`generate_rank_bindings` on a factory PSD), every
  cable looks up, so the offline answer can be optimistic; the runtime mapper, which runs
  Tier C itself, then picks another seating.
- **Intra-mesh links.** Inside a mesh, Tier C counts the mapped logical neighbours in
  `device_topology`. Links the topology does not route over (extra cables, inactive routing
  planes) are excluded, matching what blaze's adjacency and the resolver confirm.

### 4.4 Step C4 - on failure: constraints for the next solve

When C2 fails (no carve, the resolver rejects the carve, `R` over demand, or a pre-check), C4 decides what to try
next. (A solve that hit its conflict cap also lands here; open question 13.) It has two moves, in order:

| Move | Scope | What it adds | When |
|---|---|---|---|
| 1. Exit-node retry | the same seating | an intra-mesh constraint that moves the crossings' exit chips onto different host slices, then back to C1 | first version (P15), at most `kMaxLayoutRetries` times per seating |
| 2. Exclude the seating | the SAT placement | `exclude_mapping(seating)` in the live master session, then the next seating | first version (P8) |
| later: seat-pair nogood | the SAT placement | `add_seat_pair_nogood`: not mesh_a on seat_a with mesh_b on seat_b (and their twins), when a failure is proven for every carve over the pair | P13 |
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

**No new solver API.** The retry builds a fresh intra-mesh solve anyway, so the block is an
existing cardinality constraint on it: "at least one exit ASIC moves to another slice" is
`add_cardinality_constraint(pairs, 1)` over every (fabric node f, exit ASIC a) with f not in
the slice a sat under last time.

**Example.** `pod8` offline, on (P, U), with host-to-rank free: U's cables land on opposite
corners of M1, and C2 fails (3.1). The retry blocks M1's signature, and the intra-mesh solve
puts a different host's slice under the corner ASIC. If that brings the two cable ends next
to each other along M1's loop, C2 passes on the same seating. With every host rank bound, the
signature is fixed and C4 goes straight to move 2.

#### Exit-node constraints for every seating, from Tier B (backup if the solve is slow)

The intra-mesh solve decides which chips sit under each slice, and so where the cables
land. Today its exit-node constraint (`add_exit_node_constraints`) allows any mesh node. A
pipeline can narrow that, but only to what every carve agrees on, because C1 and C2 have not run
yet and must stay free to pick any carve:

- For each mesh pair (m, n) with `k(m, n) > 0`: at least `ceil(k / links per exit ASIC)` of
  the fabric nodes in `boundary_slices(m, n)` map to ASICs cabled toward n's seat
  (a cardinality constraint added through `ExtraMappingConstraints::intra_mesh`, after the
  existing one).
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
> | Exact? | no: only necessary conditions can be encoded at seat level | yes: runs the footprint solve and the builder's resolver on the real layout |
> | New code | seat-graph edge drop, seat and exit constraints (P10) | the check and one `exclude_mapping` call in the existing retry loop (P8) |
> | Cost | a larger master encoding | one intra-mesh solve and one check per rejected seating |
>
> Option 1 cannot be the whole check: the master solve chooses seats, not chips, and
> feasibility depends on facts that exist only after the intra-mesh solve (which chips sit
> under each slice, per-chip core capacity, stage alignment along the ring, stage 0's host
> I/O). Encoding those would be close to one flat solve of the whole system. On `mixed4`,
> (P, Q) passes every seat-level rule and fails on loop length; on `pod8`, seat U fails on
> M1's corners. Neither is visible before the intra-mesh solve.
>
> So the first version checks each seating (C1, C2) and excludes rejected ones (Plan 2, P8).
> Pruning before the first solve (Plan 1, P10) is added later only if the measured rejection
> counts on `mesh4x4_stage4x2`, the ring-stress and the superpod cases call for it, and it
> must never remove a seating the check would accept.

#### Feedback after a rejected seating - Plan 2 (first version)

This is what the first version ships (P8). When a seating still fails after the exit-node
retry (C2 failed again), feed back the strongest
constraint the failure supports:

The exact check carves freely over the seating's slices, so a failure says "the pipeline
cannot be built on this set of seats with these layouts". It does not say which logical mesh
was at fault. Feedback must therefore be about seats, not about a mesh on a seat:

| Failure attributed to | Feedback |
|---|---|
| default: the seating as a whole | `exclude_mapping(seating)` - block this set of seats |
| two meshes' seats, proven for every carve over them | `add_seat_pair_nogood` - block that mesh pair on those two seats (P13, below) |
| a seat that no carve could use under any seating | forbid that seat for every mesh |

- **Do not** use `add_forbidden_constraint(placed)` (one mesh on one seat) for a pipeline
  failure. Another mesh on the same seat, or the same mesh with the stages split differently,
  may succeed, so this would eliminate placements early. It stays for intra-mesh failures,
  as today.
- **Partial nogoods need a new API.** `exclude_mapping` with a partial map does not work
  today. Without `unique_shapes` (the setting this design uses, below) a partial map adds no
  clause at all. Under `unique_shapes` the clause becomes "some mesh uses a seat outside
  `{seat_a, seat_b}`", which with more than two meshes is always true and blocks nothing.
  P13 adds `add_seat_pair_nogood` on the session for this (below).
- Attribution needs a proof over every carve of the seats, such as the pre-check that fired (4.2). One resolver failure is about one carve, so it never attributes a seating failure.
- **First version: whole-seating exclusion only.** `exclude_mapping(seating)` is the only
  feedback P8 uses. The seat-pair clause and the per-seat forbid wait for attributed failures
  (P13), because a wrong stronger cut would eliminate seatings that work.
- **Rejection cap.** Count pipeline rejections per run (`pipeline_rejections_`) and give up
  after `kMaxPipelineRejections` (proposed default 256, overridable for sweeps), reporting the
  last carve or resolver reason through the existing "could not fit" error.
- **Measure.** Record the rejection count in the placement stats, so the sweep and CI show
  how many seatings each MGD tries before the first accept. These numbers decide whether
  Plan 1 (4.4) is needed.

**Example.** `mixed4` fails on (P, Q) (4.5, iteration 1).

| Feedback | Effect | Verdict |
|---|---|---|
| `exclude_mapping({M0 -> P, M1 -> Q})` | blocks this seat set only | correct: (P, R) is still offered and succeeds |
| `add_forbidden_constraint(M0 on P)` | M0 may never use P | wrong: it removes (P, R), one of the two seatings that work |
| `add_forbidden_constraint(M1 on Q)` | M1 may never use Q | wrong: harmless here, but Q could carry M1 next to a different M0 seat |
| `add_seat_pair_nogood(M0, P, M1, Q)` (P13) | blocks M0 on P with M1 on Q in every seating | only with a proof (P13's producers). Here (P, Q) fails on loop length, which is a resolver failure, so no proof: P8's exclusion only |

On `pod8`, (P, U) fails and `exclude_mapping({M0 -> P, M1 -> U})` leaves (P, V), which is
the only seating that works.

#### Seat-pair nogoods (later, P13)

P8 blocks one rejected seating. When the failure is provably about two meshes' seats, P13
blocks that pair of seats in every seating, so the master solve never offers it again. There is
no solver change: the clause is one existing cardinality constraint.

**API.** `SatPlacementEnumerationSession::add_seat_pair_nogood(mesh_a, seat_a, mesh_b, seat_b)`,
built like `add_forbidden_constraint`: it stores the nogood in `extra_nogoods_`, does a trial
rebuild (if no seating is left, it pops the nogood and returns false), then calls
`invalidate_pending_solve()`. Seatings already yielded stay blocked through
`excluded_seat_maps()`.

**Encoding.** `apply_extra_constraints` adds "mesh_a not on seat_a OR mesh_b not on seat_b".
The solver takes positive literals only, and each mesh takes exactly one seat, so this is
`add_cardinality_constraint(pairs, 1)` over mesh_a's other seats and mesh_b's other seats
(from `get_valid_mappings`). Seats compare by ASIC set, because the domains hold canonical,
footprint-shared candidates.

**Scope.**

- **Per mesh pair, with twins.** The nogood applies to every pair of meshes from the twin
  groups of mesh_a and mesh_b, because twin windowing keeps one representative per orbit, and
  blocking one twin alone could cut the only representative of a working orbit. It is not
  "any mesh on seat_a and any mesh on seat_b": two other meshes on those seats carry different
  stages and cables, so the proof does not cover them.
- **Every orientation.** Matching by ASIC set covers every orientation of a footprint. That is
  sound because each producer reads only orientation-free facts. A future producer that reads
  chip positions must key on the canonical candidate instead.
- **Hard on every rung.** The nogood is hard on every solve-ladder rung, including soft seams.

**Producers.** A producer adds a proof, a mesh pair, to an optional out-parameter P13 adds to
`check_pipeline_seating` (`std::vector<std::pair<MeshId, MeshId>>*`, default null), only when the
verdict reads nothing but the two seats' footprints, layouts fixed by hard constraints, and
facts true in every projection. The P8 loop turns each proof into `add_seat_pair_nogood`, with
the two meshes' seats from the seating, before counting the rejection.

| Producer | Fires when | Notes |
|---|---|---|
| cable count, before C1 | not added | needs `k_AB`, which Tier B does not save in the first version; with P10 in, `allow_seat_pair` already keeps those seat pairs out of the first solve |
| C2 segment pre-check on mesh m | the verdict holds for every projection, m's only MGD neighbour is n, and m's slice signature is fixed by hard constraints (P15's skip test) | all of m's cables land on n's seat, so no other seat of any other mesh changes the verdict |
| resolver failure | never | one resolver run is one carve |

On `pod8` with every host bound, (P, U) fails on M1's corners. M1's only neighbour is M0, so
P13 adds a nogood for M1 on U with M0 on P. With unbound hosts, M1's signature is not fixed, so
there is no proof and only P8's exclusion applies. The rejection cap and stats from P8 stay;
`pipeline_seat_pair_nogoods` is counted next to `pipeline_rejections`.

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

#### Pruning before the first solve - Plan 1 (backup if the solve is slow)

**Not in the first version** (see the decision above). Plan 1 encodes necessary conditions
before the master solve, in one solve. It does not
commit to a projection and does not run once per projection. **Every condition must hold for
all surviving projections and for every chip layout the intra-mesh solve may still pick.** A
condition that holds for only some of them would drop seatings the runtime could build on,
so it is left to the exact check (4.2).

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
checking falls to the exact check (4.2).

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
  to. Otherwise the exact check catches it.

**Caveats.**

- **Shared seats.** The seat graph merges identical seats across mesh pairs. If two mesh pairs
  that share a seat pair need different `k`, edge-dropping is unsound for the looser pair.
  Then fall back to pairwise blocking (4.4) for those pairs.
- **What Plan 1 cannot encode.** Joint-stage alignment along the whole ring, per-chip
  pipeline-core capacity, a stage whose entry and exit land on the same chip, and stage-0
  host-to-device and device-to-host slots. These depend on
  the whole pipeline at once. The exact check (4.2) catches them.

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
   slices per mesh, `M0a`, `M0b`, `M1a`, `M1b`. Every stage is 4x2, so the positions are the
   four slices, joined by the two in-mesh borders and the two cables.
4. **C2.** The pre-checks pass (4 slices, 4 stages). The one carve is the
   4-cycle, for example `s0 → M0a`, `s1 → M0b`, `s2 → M1b`, `s3 → M1a`. The resolver places
   the ring on those four slots, and `R` records the exact channel of each crossing:
   2 channels, demand 2, fits.
5. **C3.** Emit the mapping and plan; publish `R`; phase 0 binds exactly those channels.
6. **C4**, had it failed. In Case B (both cables on one slice pair) the position graph has no
   4-cycle, so the solve finds no carve. With host-to-rank free, the exit-node retry asks the
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
| 1 | (P, Q) | success; cables on A0–B0, A3–B3 | no carve: the only cycle has 8 slices (3.2, example B) | no exit-node retry can help (the hosts form a fixed line, so the cables stay at its ends); `exclude_mapping({M0 -> P, M1 -> Q})` |
| 2 | (Q, P) | success; the same cables with the meshes swapped | same geometry, no mapping | `exclude_mapping({M0 -> Q, M1 -> P})` |
| 3 | (P, R) | success; cables on A0–B0, A2–B2 | the carve is the 6-cycle (3.2, example A); the resolver places the ring (3.1); `R` fits the demand of 2 (4.2) | emit |

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
3. **C1.** `inter_mesh_cables` returns the two P–R cables (3.3); the routed
   intra-mesh links give the two lines. `get_coord_range(M0, host_rank)` gives A0..A3
   (columns 0–1, 2–3, 4–5, 6–7), likewise B0..B3; adjacency is the lines plus A0–B0 and
   A2–B2. Positions: the eight slices for the 4x2 stages, and the six adjacent slice pairs
   (A0+A1 ... B2+B3) for the 4x4 stages.
4. **C2.** The solve returns s0 A0, s1 A1+A2, s2 B2+B1, s3 B0 (or a mirror). The resolver
   places the ring on those slots (3.1). `R` = {the A2–B2 channel for `s1 -> s2`, the A0–B0
   channel for `s3 -> s0`}: 2 channels, demand 2, fits.
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
- **`pod8`** takes two iterations: on (P, U) the solve finds no carve, because the cables
  land on opposite corners of M1's tile loop, and (P, V) passes (3.1).

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

**One source of truth.** Hooks do not each re-check the proto. The pipeline fields of both graphs
are empty by default and are only read when `logical_.pipeline_stage_graph_` has nodes, which it
does exactly when the merged MGD has `pipeline_descriptors` (Tier A rejects a pipeline with no
stages). Hooks that run before the logical graph exists use `has_pipeline_descriptors()`:

```cpp
// MeshGraphDescriptor (merged)
bool has_pipeline_descriptors() const;                                 // proto_->pipeline_descriptors_size() > 0
// Backup if the solve is slow (not implemented):
// const PipelineMappingConstraints* pipeline_mapping_constraints() const; // null when off

// LogicalMultiMeshGraph, filled once per run by build_logical_multi_mesh_adjacency_graph (P5)
AdjacencyGraph<PipelineStageNode> pipeline_stage_graph_;               // empty by default; the gate: nodes iff pipelines
std::map<PipelineStageNode, tt::tt_metal::distributed::MeshShape> pipeline_stage_shapes_;   // empty likewise
std::map<LogicalMeshNode, tt::tt_metal::distributed::MeshContainer<MeshCoordinateRange>> mesh_host_grids_;   // empty likewise
```

Every hook below takes the merged MGD's `pipeline_descriptors()` (or `has_pipeline_descriptors()`
where only a setting changes) and returns early, or keeps today's value, when it is empty. An empty list means the
hook adds nothing; it never means "empty pipeline, run with defaults".

| Hook | Where | When off |
|---|---|---|
| schema `pipeline_descriptors` (1.3) | MGD proto | absent field; existing textprotos parse unchanged |
| merge (1.5) | `MeshGraphDescriptor::merge` | merged proto byte-identical to today's |
| Tier A (2.1) | MGD constructor | loops over zero pipelines |
| Tier B (2.2) | MGD constructor | `check_pipeline_feasibility` returns at once; nothing is stored |
| `ExtraMappingConstraints` (2.2, 4.4) | `TopologyMappingConfig` | empty hooks: no constraint added |
| seat-graph edge drop for `k_AB` (4.4) | `build_sat_placement_seat_graph` | today's `saturated_link_count` edges |
| `unique_shapes` (4.4) | `map_multi_mesh_to_physical`, `generate_rank_bindings --all-solutions` | `true`, as today |
| `pipeline_stage_graph_`, `pipeline_stage_shapes_`, `mesh_host_grids_` (4.0) | `build_logical_multi_mesh_adjacency_graph` | stay empty (the default); the hook in `next()` sees an empty stage graph and skips, and also skips when `check_pipelines` is off, and `TopologyMappingResult::pipeline` stays unset |
| C1 to C4 (section 4) | `MultiMeshSolutionEnumerator::next` | loop is today's; `PhysicalMultiMeshGraph::pipeline_position_*` stay empty; no `pipeline_*` failure stage can be reported |
| exit-node retry (4.4) | `complete_intra_mesh_for_placement` | never called a second time for one seating; no signature block added |
| required links `R` and the budget check (4.3) | mapper | `R` empty; no budget check |
| phase 0 (4.3) | `pair_logical_intermesh_ports` | empty `required_hashes`, so today's phases |
| single-host phase 0 (4.3) | `generate_intermesh_connections_on_local_host` | nothing required, so today's greedy |
| stage plan in rank bindings (4.6) | `generate_rank_bindings` | no stage fields written |

**No ungated change to the gather.** `inter_mesh_cables` only runs inside the mapper, for MGDs
with a pipeline; the control plane's gather is untouched. The move to the policy plan's
candidate table (P16) rides on that plan's own parity tests.

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
  `check_pipeline_feasibility`, `build_physical_pipeline_adjacency_graph` or `check_pipeline_seating`
  call, no extra SAT clauses.
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

- The mapper only accepts a mapping once the footprint solve has produced a carve for it and
  the resolver has laid it out (4.2). That carve is the stage-to-rank mapping: each stage, its
  slices, hence its ranks, and its entry and exit chips.
- Each rank's stage is recorded in `rank_bindings.yaml` as `TT_PIPELINE_STAGE` (4.6) but not
  consumed.
- At runtime blaze carves again, with its own carve planner or uniform tiling depending on
  how it is called (3.4), and may pick a different placement than the mapper's witness. That is accepted
  (2.3): the mapper only guarantees that a placement exists on the mapping. If
  `TT_PIPELINE_STAGE` is ever consumed, the two would have to be made to agree first.

#### Example

`mixed4` on (P, R):

| Step | Who decides | Result |
|---|---|---|
| rank to slice | the topology mapper (rank bindings and intra-mesh solve) | rank i drives Ai on M0, rank 4 + i drives Bi on M1 |
| stage to slices | the footprint solve in the mapper (a witness); blaze's carve planner at runtime | s0 A0, s1 A1+A2, s2 B2+B1, s3 B0; A3 and B3 are fillers |
| co-ownership | `_collect_submeshes`, then `allgather_end_owners` | ranks 1 and 2 share s1 and ranks 5 and 6 share s2; `allgather_end_owners` picks which co-owner drives the entry chip and which the exit chip |
| idle ranks | filler slots | ranks 3 and 7 carve their own slice and carry no stage |

---

## 6. Open questions

1. **Adjacency parity.** Tier C builds slice adjacency from candidate inter-mesh links (and
   then binds the ones in `R`) and routed intra-mesh links; runtime uses
   `get_forwarding_direction`-confirmed neighbours of the bound links. Confirm the two agree on
   the links in `R`, including links with no active routing planes.
2. **Projection count.** For heterogeneous or fork pipelines, how many projections survive
   Tier B in practice? With many survivors the minimum `k_AB` and union of boundary slices
   (4.4) get weak, and more of the work falls to the exact check. Is that affordable, and is
   the optional "find all" run (2.2) worth shipping in the first version, or should it save
   the defaults until a sweep shows Plan 1 would cut many seatings?
3. **Shared seats with different `k`** (4.4): handle generally, or reject until a real
   descriptor needs it.
4. **Seat-pair nogoods** (4.4): planned as P13 (`add_seat_pair_nogood`, a cardinality
   clause per twin-orbit mesh pair, two producers). Still open: whether a per-seat forbid
   ("no carve could use this seat under any seating") has a sound producer.
5. **Cost of `unique_shapes` off** (4.4): how many extra seatings the master solve yields on
   the large symmetric cases once mesh permutations are no longer collapsed.
6. **Planned key name** for the rank-binding `env_overrides` entry (4.6).
7. **Per-stage core counts.** Decided for now: no MGD field. C2 computes blaze's defaults
   from the graph shape (4.2). Harnesses can still override pipeline-core pools, which the MGD
   does not carry; add an optional core count to `PipelineStage` only if a model needs one.
8. **Agreement with blaze's planner** (3.2). Tier C accepts a seating if *any* carve passes
   the check; blaze commits to its first aligned cycle and also rejects carves the footprint
   finds (stage 0 on several slices, the 16 `no-alignment` cases, every budget-exhausted
   grid). So Tier C can accept a mapping blaze then fails to build. Options: have blaze call
   the same C++ footprint solve through its bindings; teach blaze's planner to try the next
   cycle; or, until then, restrict Tier C to what blaze builds (questions 14 and 15).
9. **Spread fill for the binding** (4.3). Dropped: spare links after phase 0 keep today's
   order, because the pipeline only needs `R`.
10. **Ordering against the inter-mesh policy plan** (3.3). Decided: do not wait for part 9;
    C1 reads the PSD (P5) and P16 switches to the candidate table later. Must P16 land before
    the first pipeline model ships?
11. **Phase 0 failure before part 11** (4.3). Fail at control-plane init on every rank, through the check after the broadcast (proposed), or add a
    cheap mapper-side check first (an exit chip in `R` toward two meshes needs two free
    directions)?
12. **Exit-node retry** (4.4). Is `kMaxLayoutRetries = 4` per seating right, and is the
    slice-signature block the right granularity, or should a retry also move chips within a
    slice when C2 failed on chip-level capacity?
13. **The C2 bound** (4.2). `conflict_cap` bounds the one solve and is deterministic.
    With today's API a capped solve reads as "no carve", so C4 excludes the seating as if it
    were proved infeasible, which can drop a seating that has a carve. Accept that (with the
    pre-checks deciding the known hard cases), or add a one-field "stopped" flag to
    `MappingResult`, the only solver change this design would then need?
14. **Stage 0 on one slice.** Blaze's planner requires stage 0 to be exactly one host slice;
    the footprint only requires room for host I/O (4.2). Restrict stage 0's positions in C1 to
    single slices for as long as blaze keeps the rule? Without it, Tier C accepts multi-slice
    stage-0 carves that blaze rejects at build time.
15. **A stage smaller than a host slice** (3.4). Positions are whole slices, so a 2x2 stage on
    4x2 slices is not modelled; blaze's per-rank tiling builds it by keeping the rank's first
    tile. Model sub-slice positions (the first tile only, to match blaze), or keep Tier B's
    reject (2.2)?
16. **RELAXED as soft edges** (3.4, later). Does the solver's soft-edge mode
    (`set_allow_unmatched_target_edges`) give blaze's preferences: fewest host cuts first,
    never stage 0's outgoing edge, then the smallest host-hop pages? Or are those extra
    constraints and an objective the solver does not have? One solve keeps every fabric chain
    on disjoint slices, which blaze's per-chain planning does not have to prove; confirm the
    per-chain resolver calls cannot then disagree on a shared slice's chips.
17. **The C1 position filter** (4.2). Shape, core slots against endpoint count, and stage 0's
    host I/O are what the resolver checks per stage today. If question 7 adds per-stage core
    counts, the filter
    must follow. A filter stricter than the resolver removes carves that would pass; a looser
    one only costs extra rejects.
18. **Demand inside the encoding** (4.2). The demand check rejects after the solve. If it
    fires often, add a per-mesh-pair cap on crossing edges to the footprint encoding? That
    needs a cardinality constraint over edge variables the solver does not expose today.
19. **Resource completeness off the tight case** (4.2). The cardinality constraints are added
    only when the stage areas fill the slices exactly. Carves with one or two spare slices
    (fillers) get no help; measure whether they are slow.
20. **A recorded channel that is down** (4.3). The mapper reads the PSD on rank 0, which has
    no link state, while each rank's gather drops channels whose link is down on its own
    cluster. Until the candidate table carries liveness, `R` can name a down channel, and the
    check after the pairing fails. Fail (proposed), or rerun Tier C without that channel?
21. **Broadcasting the pipeline mapping** (4, "The pipeline graph in, the pipeline mapping
    out"). `R` is broadcast with the chip info (every rank checks it after the pairing); the rest
    of the pipeline mapping stays on rank 0. Broadcast the rest once another rank consumes it?
    Also: with `generate_mapping_locally_` every rank runs the mapper itself on its own ASIC
    allowlist, so ranks can compute different `R`; take rank 0's, or disallow pipelines there?
22. **Total time per mapping** (4.2, 4.4). Each seating can take up to one C2 solve at the conflict cap, times
    `kMaxLayoutRetries` exit-node retries, and C4 may try up to `kMaxPipelineRejections`
    seatings: in the worst case hours. Add an overall cap on the pipeline loop (and report the
    seatings tried), or rely on the rejection cap alone?
23. **Pipelines across parts** (4, "The pipeline graph in, the pipeline mapping out").
    Results use each part's local mesh ids. For an edge whose two ends are in different parts,
    name the far end as (part index, local `FabricNodeId`), or return the pipeline mapping once
    for the whole run instead of per part?
24. **One carve per seating** (4.2, "When C2 fails"). C2 checks only the first carve, so a
    seating can be lost when another carve on it would pass the resolver or the demand check.
    No benchmark case did (101 of 101 first carves passed). Keep one carve, or allow a few more
    (`exclude_mapping(carve)`, `next()` on the same session, existing API) before failing the
    seating, and if so how many?
