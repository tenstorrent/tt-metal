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

**`mixed4`: stages that span two ranks.** Two such meshes, joined with 2 channels. The ring
has two 4x4 stages, which need two neighbouring slices each:

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

**`pod8`: different slice shapes.** The `llama_8b_pod` pattern at small scale. M0 is as
above. M1 is 2x4 with `host_topology [2, 2]`, so each of its four ranks drives a 1x2 slice,
T0..T3:

```
M1 (2x4)  cols 0-1   cols 2-3        slice adjacency inside M1
row 0     T0 (r4)    T1 (r5)          T0 ── T1
row 1     T2 (r6)    T3 (r7)          │      │
                                      T2 ── T3
ring of 8: s0..s3 shape [4, 2], s4..s7 shape [1, 2], loopback s7 -> s0, 2 channels
```

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

#### Examples

| Descriptor | Tier A result | Why |
|---|---|---|
| `mixed4` | pass | ids unique, edges name declared stages, every stage has out-degree 1, one cycle (`s3 -> s0` becomes its loopback), shapes positive |
| `pod8` | pass | a ring of 8; the loopback is `s7 -> s0` |
| `mixed4` plus `edges { src: "s1" dst: "s1" }` | reject | self-edge |
| `mixed4` with `s0 -> s1` repeated | pass | two channels on that hop; the builder reads one link per edge, which is a builder limit |
| edges `s0 -> s1`, `s0 -> s2`, `s1 -> s0`; `s2` has no out-edge | reject | out-degree 2 makes it a fork, and the branch ending at `s2` has no edge back to the router (`fork_structure`) |
| `mixed4` as FLEXIBLE with `s3 -> s0` declared `transport: HOST` | reject | FLEXIBLE plans its own host cuts and only does so when none are declared |
| `mixed4` as STRICT with `s3 -> s0` declared `transport: HOST` | pass | a host hop anywhere in a ring is allowed |
| a stage declared `[2, 4]` where every mesh has 4x2 slices | pass here | shapes are logical and never transposed, so Tier B finds no mesh it binds to |

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

#### Examples

**`mixed4`: several projections survive.**

- **Slices.** 4x8 divided by `[1, 4]` gives 4x2 slices, four per mesh.
- **Binding sets.** 4x2 is one slice and 4x4 is two, so every stage binds to M0 or M1.
  Stage 0 is one slice.
- **Capacity.** The ring needs 6 slices and the MGD declares 8.

| Projection | Slices on M0 / M1 | Crossings | Result |
|---|---|---|---|
| all four stages on M0 | 6 / 0 | 0 | dropped: M0 holds only 4 slices |
| `s0, s1 -> M0`; `s2, s3 -> M1` | 3 / 3 | 2 (`s1 -> s2`, `s3 -> s0`) | survives |
| `s0, s1, s3 -> M0`; `s2 -> M1` | 4 / 2 | 2 (`s1 -> s2`, `s2 -> s3`) | survives |
| `s0, s2 -> M0`; `s1, s3 -> M1` | 3 / 3 | 4 | dropped: only 2 channels |

Segment carve for the first survivor: M0's 1 + 2 slices fit a path of three slices in a
line of four. Tier B passes and chooses nothing. Every survivor crosses M0–M1, and a ring
crosses a cut an even number of times, so Plan 1 (5.1) gets `k(M0, M1) = 2`.

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
| `mixed4` with s0 declared `[4, 4]` | reject | stage 0 must bind to a single slice |
| `mixed4` with a fifth stage `[4, 6]` (3 slices) | reject | the ring needs 9 slices and the MGD declares 8 |
| `mixed4` with a fifth stage `[4, 4]` | pass | 8 of 8 slices: s0, s3 and one 4x4 stage on one mesh, the other two 4x4 stages on the other; every slice is used |
| `mixed4` with one stage `[4, 10]` | reject | wider than either mesh: empty binding set |
| `pod8` with M1 as 4x8, `host_topology [1, 4]` (4x2 slices), still 1x2 stages | reject | 1x2 is not a whole slice. Blaze's per-rank tiling could build it (3.4, "a tile smaller than a host slice"), so this rule is stricter than runtime |
| `mixed4` with the MGD connection removed | reject | every projection crosses, and no connection is declared |
| the same, but `placement: FLEXIBLE` | pass | crossings may become host hops, so only binding sets and capacity apply |

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

#### Examples

Only Tier C sees where cables land, so it is the first tier that separates these seatings.
Section 4 traces each one step by step.

| Pipeline | Seating (M0, M1) | Cables land on | Tier C | Why |
|---|---|---|---|---|
| `mixed4` | (P, Q) | A0–B0, A3–B3 | reject | the slices form one loop of 8 and the ring needs a loop of exactly 6; every tiling fails in the resolver |
| `mixed4` | (P, R) | A0–B0, A2–B2 | accept | a 6-slice loop exists and the stages align onto it |
| `mixed4`, FLEXIBLE | (P, Q) | A0–B0, A3–B3 | accept | one fabric path covers 6 slices and the loopback goes over host (3.4) |
| `pod8` | (P, U) | A3–T0, A0–T3 | reject | right count, both on M0's end slices, but on opposite corners of M1, so no path through M1's four tiles joins them |
| `pod8` | (P, V) | A3–T1, A0–T0 | accept | the cables land on neighbouring tiles of M1 |

Tier B passed all of these. Plan 1 (5.1) could only have caught `pod8` on U, and only with
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
(5.3). `build_topology` gets no `submesh_shape`, so `_collect_submeshes` calls `plan_carve`:

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
binding takes the two lowest ids, both A0–B0 (5.3), so blaze sees only A0–B0:

```
_prune_leaves    A3, B3 have one neighbour: drop; then A2, B2; then A1, B1; then A0, B0
CarvePlanError: no-cycle-of-length-6: only 0 slices can lie on any cycle.
```

The mapping is already committed, so this is a hard failure. That is the failure Tiers A–C
and 5.3 move earlier.

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
DAG after dropping loopbacks, HOST edges are left out, and per-node core capacity
covers the router's endpoint budget.

The one change: the resolver throws with a message string on failure. Tier C needs to know
*which* stage or edge failed to attribute the failure (5.2). Add a structured failure
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

All on two 4x2-slice meshes, as in the running examples.

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
starting next to the router.

| Step | What it does |
|---|---|
| input checks | Router exactly one slice; branch stages whole slices. `need_edges = 2 x fabric-return branches + 1 x host-return branches`; `need_slices = 1 + all branch slices`. |
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

**Other edge cases.**

| Case | Phase | Result |
|---|---|---|
| `pod8` (4x2 and 1x2 slices) | 0 | `host slices are not uniform`; per-rank tiling is the path (3.4) |
| a 1x2 stage on 4x2 slices | 0 | `not a whole number of 4x2 host slices` |
| `mixed4` with s0 4x4 | 0 | `input-infeasible`: stage 0 owns host-to-device, the loopback entry and device-to-host, so one rank must drive it |
| only one cable, A0–B0 | 2 | leaf pruning removes every slice: `only 0 slices can lie on any cycle` |
| an open chain (no loopback) | before phase 0 | `plan_carve` refuses: there is no cycle to place; give it a `submesh_shape` |
| a huge fabric with many branch slices | 3 | `budget-exhausted`; Tier C treats it as unknown and accepts |

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

#### Example

`mixed4` on (P, R), on a machine with an extra A0–B0 cable whose chip ids are lower than the
A2–B2 cable's:

| Call | Result |
|---|---|
| `gather_intermesh_candidates(psd, mapping)` | three candidates: A0–B0 (first), A0–B0 (second), A2–B2. Each source chip cables to one destination chip on M1, so all three pass |
| `bind_intermesh_connections(cands, mgd, {})` | today's rule: the two lowest ids, both A0–B0 |
| `bind_intermesh_connections(cands, mgd, R)` with `R = {A0–B0, A2–B2}` from Tier C | phase 0 binds both cables in `R`; the count of 2 is full |

Tier C and the control plane call the same two functions, so the cables Tier C checks are
the cables that get bound.

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

**Example: `mixed4` as FLEXIBLE on (P, Q).** STRICT rejects this seating (3.2, example B).
`plan_ring_placement` runs:

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
phase 4   _group_walk per cover; joints only between stages, never cut 0:
            cover 1 best: s0 A3, s1 B3+B2, s2 A0+A1, s3 A2   cut s1 -> s2   key (1, 1, (1,))
            cover 2 best: s0 A3, s1 B3+B2, s2 B1+B0, s3 A0   cut s3 -> s0   key (1, 0, (3,))
          key = (cut page sizes, 0 if the loopback is cut else 1, cut indices)
result    cover 2 wins: one cut, and it is the loopback. s3 -> s0 becomes HOST.
          carve M0 4x2@(0,6), M1 4x4@(0,4), M1 4x4@(0,0), M0 4x2@(0,0); A1, A2 idle
          _resolve_chains_separately: one fabric chain s0 -> s1 -> s2 -> s3
```

**FLEXIBLE edge cases.**

| Case | Result |
|---|---|
| a cover whose only alignment cuts `s0 -> s1` | skipped: stage 0's outgoing edge stays on fabric |
| a joint between chains inside a 4x4 stage | skipped: a wide stage cannot be split by a host hop |
| `pod8` as FLEXIBLE | `HostEdgePlanError: host slices are not uniform`, not caught; Tier C rejects, because blaze raises rather than falling back |
| a fork with a two-slice router | `ForkShapeUnsupported`; blaze resolves the fork as declared, so Tier C uses the STRICT paths |
| a FLEXIBLE ring that is already all fabric on single-slice stages | no plan; placement goes to the resolver as in STRICT |

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

For example, take `pod8` with M1 changed to 4x8 and `host_topology [1, 4]` (4x2 slices),
still with 1x2 stages s4..s7. `create_submeshes(1x2)` on M1 makes 16 tiles, row-major. Each
of M1's ranks owns four of them and keeps the one whose `(0,0)` chip is local: rank 4 keeps
row 0, columns 0–1; rank 5 keeps row 0, columns 2–3; and so on. Rows 1–3 of M1 carry no
stage. A cable that lands on M1 row 3 then does not count, because `discover_submesh_links`
only sees chips inside submeshes. The same physical seat passes or fails depending on
whether the intra-mesh layout puts the cabled chips in row 0. Tier B as written rejects
this MGD at load (2.2).

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

#### Example: `mixed4` through the loop on pods P, Q, R

Before the first solve, Tier B passed with `k(M0, M1) = 2` (2.2) and Plan 1 dropped the Q–R
seat edge (5.1). With `unique_shapes` off, the master solve can offer (M0, M1) = (P, Q),
(Q, P), (P, R) and (R, P). Suppose it offers them in that order.

| Iteration | Seating | Intra-mesh | Tier C | Next |
|---|---|---|---|---|
| 1 | (P, Q) | success; cables on A0–B0, A3–B3 | planned carve `no-cycle-of-length-6` (3.2, example B); every tiling fails (3.4) | `exclude_mapping({M0 -> P, M1 -> Q})` |
| 2 | (Q, P) | success; the same cables with the meshes swapped | same geometry, same failures | `exclude_mapping({M0 -> Q, M1 -> P})` |
| 3 | (P, R) | success; cables on A0–B0, A2–B2 | planned carve finds the 6-cycle (3.2, example A); resolver places the ring (3.1); binding rerun passes (5.3) | emit |

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
3. **Links.** `gather_intermesh_candidates` returns the two P–R cables (3.3); the routed
   intra-mesh links give the two lines.
4. **Slices.** `get_coord_range(M0, host_rank)` gives A0..A3 (columns 0–1, 2–3, 4–5, 6–7),
   likewise B0..B3. Adjacency is the lines plus A0–B0 and A2–B2.
5. **Carve.** The planned carve returns six slots (3.2, example A). Per-rank tiling is not
   needed, since the first path succeeded.
6. **Layout.** The resolver gives s0 A0, s1 A1+A2, s2 B1+B2, s3 B0 (3.1).
7. **Binding.** `R` = {A2–B2 for `s1 -> s2`, A0–B0 for `s3 -> s0`}; bound first; rerun
   passes (5.3).
8. **Emit.** The mapping, the plan and the layout; `TT_PIPELINE_STAGE` per rank (4.1).

**Edge cases in this loop.**

- **Feedback is about seats.** After iteration 1, `add_forbidden_constraint(M0 on P)` would
  have blocked iteration 3, which succeeds with M0 on P. `exclude_mapping` blocks only the
  pair of seats (5.2).
- **The swap matters in general.** Iteration 2 fails here only because the layout is the
  same. With `unique_shapes` on, it would never be offered; that is harmless in this example,
  but wrong when the two meshes' pins or rank bindings differ (5.2).
- **FLEXIBLE accepts iteration 1.** As FLEXIBLE, `mixed4` on (P, Q) passes with the loopback
  over host (3.4), so the loop would stop there.
- **`pod8`** takes two iterations: (P, U) fails in the resolver because the cables land on
  opposite corners of M1, and (P, V) passes (3.1).

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

#### Examples

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

**Under FLEXIBLE** `k = 0` for every pair, and Plan 1 prunes nothing: `mixed4` as FLEXIBLE
can use (Q, R) with its one link by sending one crossing over host.

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

**Example.** On `mixed4`, (P, Q) and (Q, P) are the same set of seats {P, Q}, so with
`unique_shapes` on only the first is offered. In this example both fail with the same slice
geometry, so the dedup is harmless. Now change it so that M1 has an MGD pin putting its
host rank 0 on host h1. Then (Q, P) lays out differently from (P, Q), and the two can give
different answers. After (P, Q) fails, `exclude_mapping` blocks the key {P, Q}, which also
covers (Q, P), so the swap is never checked. With `unique_shapes` off, (Q, P) is its own
seating and gets its own check.

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

#### Examples

**`mixed4` on (P, R) with an extra cable.** Suppose the machine has two A0–B0 cables, both
on lower chip ids than the A2–B2 cable, and the MGD asks for 2 channels.

| Step | Today | With 5.3 |
|---|---|---|
| gather | three candidates: A0–B0, A0–B0, A2–B2 | same (3.3) |
| required links | none | Tier C's layout crosses on `s1 -> s2` (A2–B2) and `s3 -> s0` (A0–B0), so `R` holds one cable of each |
| bind | the two lowest ids: both A0–B0 | phase 0 binds `R`; the count of 2 is full |
| rerun over bound links | not done | the same plan and layout: accept |
| runtime | blaze sees only A0–B0; leaf pruning removes every slice; `no-cycle-of-length-6` (2.4) | blaze sees A0–B0 and A2–B2 and builds |

**Spread fill.** Take the count as 3 instead. `R` takes two, and one more is filled from
the spare A0–B0. Step 4 orders the fill by slice pair, but every spare cable here is on
A0–B0, so it makes no difference. With a spare on A3–B3 instead, the fill would take it
first. That adds a new slice pair, which gives blaze's runtime planner the branch-slice
fabric of 3.2, example C.

**Phase 0 cannot place a cable.** Suppose the A2–B2 cable's chip has no free port left in a
direction facing M1. Phase 0 fails, Tier C finds no other layout cheaply, and the seating is
blocked with `failure_stage = "pipeline_binding"` (step 6).

**STRICT `Graph` connections.** The user pins the exit chips, so the cables Tier C may use
are exactly the pinned ones. If the pins give A0–B0 and A3–B3, Tier C rejects `mixed4`
(3.2, example B), and 5.3 has nothing to choose between.

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

#### Example

`mixed4` on (P, R):

| Step | Who decides | Result |
|---|---|---|
| rank to slice | the topology mapper (rank bindings and intra-mesh solve) | rank i drives Ai on M0, rank 4 + i drives Bi on M1 |
| stage to slices | the carve planner | s0 A0, s1 A1+A2, s2 B2+B1, s3 B0; A3 and B3 are fillers |
| co-ownership | `_collect_submeshes`, then `allgather_end_owners` | ranks 1 and 2 share s1 and ranks 5 and 6 share s2; `allgather_end_owners` picks which co-owner drives the entry chip and which the exit chip |
| idle ranks | filler slots | ranks 3 and 7 carve their own slice and carry no stage |

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
