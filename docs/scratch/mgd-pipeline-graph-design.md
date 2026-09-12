<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

SPDX-License-Identifier: Apache-2.0
-->

# Encoding pipeline-stage adjacency into MGD

Brainstorm / design notes for [tt-metal#52016](https://github.com/tenstorrent/tt-metal/issues/52016).

**Status:** sections 1 and 2 are filled in, with all validation in section 2. Sections 3, 4 and
5 are reserved placeholders, carrying only seed pointers and the two notes in 4.1 and 5.1, and
are expected to be filled in as follow-up passes.

---

## 0. Problem recap

The pipeline graph is already declared in blaze — `build_single_galaxy_pipeline_graph`
in `blaze/models/gpt_oss_120b/entry.py`:

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

![Pipeline graph](https://raw.githubusercontent.com/tenstorrent/tt-metal/ridvan/mgd-ticket-52016-diagrams/docs/scratch/ticket-52016-diagrams/1-pipeline-graph.png)

### The Problem
The MGD this runs on has none of that — four anonymous 4x2 slices, no stage names, no
adjacency, no ring:

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

The mapper commits blind. The builder sees `entry.py` only after, and fails with
`"no valid submesh assignment found"`. The only recourse is to re-run mapping blind.

![Current flow](https://raw.githubusercontent.com/tenstorrent/tt-metal/ridvan/mgd-ticket-52016-diagrams/docs/scratch/ticket-52016-diagrams/3-current-flow.png)

---

Intended end state: a three-phase solve. The new middle phase checks pipeline/ring
feasibility while mapping can still be steered.

![Proposed flow](https://raw.githubusercontent.com/tenstorrent/tt-metal/ridvan/mgd-ticket-52016-diagrams/docs/scratch/ticket-52016-diagrams/4-proposed-flow.png)

### Big-picture steps

1. **MGD graph format for graph layout** - the pipeline descriptor. *Filled in below.*
2. **MGD pipeline graph verification** - every check the declared graph must pass before it can be mapped. *Filled in below.*
3. **Encoding graph layout information in MGD** - the concrete schema and parser/validator work. *Reserved.*
4. **Mapping the graph as a stage in topology mapper util** - where the new phase runs. *Reserved.*
5. **How to solve this in the SAT solver** - clauses versus a separate feasibility check. *Reserved.*

---

## 1. MGD graph format for graph layout

### 1.1 Scope

This adds **one new top-level section** to MGD, `pipeline_descriptors`, and changes
nothing else.

Specifically:

- `MeshDescriptor` is **not** modified.
- `GraphDescriptor` is **not** modified. In particular this does not overload
  `graph_topology: RING` or `connections` to mean anything pipeline-related.
- `NodeRef` and `top_level_instance` are **not** modified. The descriptor reads
  `top_level_instance` to know what fabric it is being placed on, but reading is not
  modifying.
- The only edit to the existing schema is a new field on the top-level
  `MeshGraphDescriptor` *container* message, alongside `mesh_descriptors`,
  `graph_descriptors`, `switch_descriptors`, `top_level_instance` and `pinnings`. That
  container is not the same thing as `MeshDescriptor`; adding a member to it is what
  makes a new section exist at all.

**Purpose:** the descriptor exists so the topology mapper can answer *"is this pipeline
embeddable in the mapping I am about to commit?"* during its solve. It is deliberately
**not** a layout source of truth - `blaze`'s `PipelineGraph` remains that at runtime.
That scoping is what keeps the format as small as it is.

> **Note:** The format does **not** need to mirror the argument list of `resolve_graph_layout`. The MGD pipeline graph merely checks if this pipeline can be fullfilled during mapping, could be a superset of the one checked in PIpeline graph.

### 1.2 The pipeline descriptor

The descriptor carries three things: how many stages there are, how big each one is, and
how they connect.

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

#### Dropped from `entry.py`

Same graph, for comparison:

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

Two fields are omitted. Both matter to the pipeline builder; neither changes the shape
being mapped, since they don't affect the topology mapping or graph isomorphism check. They stay out of MGD. See 2.1 for more details.

- **`factory`** (`s_embed`, `s_lm_head`, ...) — host-side construction. The builder
  needs it; the mapper never calls it.
- **`is_loopback`** — a builder conversion mark. The ring is already the `s3 -> s0`
  edge. See 2.1.

#### A heterogeneous example

The gpt-oss ring is uniform, so it does not exercise per-stage shapes. The llama_8b_pod
graph does - `_build_heterogeneous_two_mesh_graph` in
`tests/pipeline_builder/test_pipeline_builder_infra.py` builds 40 stages as 8 of `4x2`
on M0 plus 32 of `1x2` on M1, chained linearly with the ring closing from the last stage
back to `s0`. Abbreviated:

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

Two things this establishes. Shapes vary per stage, so `shape` has to live on the stage. And
the shape mix is **uniform within each mesh instance** but differs between instances, which
is forced by `host_topology` being a uniform partition rather than being a choice - see 2.2.
Nothing in the descriptor says which mesh the `4x2` stages land on; shape fitting decides.

### 1.3 Schema

The proto shape only. **All validation lives in section 2**, organized by when each check can
run.

```proto
message PipelineDescriptor {
  string name = 1;                  // non-empty, unique across descriptors
  repeated PipelineStage stages = 2;  // >= 2
  repeated PipelineEdge edges = 3;    // >= 1
}

message PipelineStage {
  string id = 1;             // non-empty, unique, referenced by edges
  string label = 2;          // human-facing, unconstrained, may repeat
  TorusTopology shape = 3;   // required, rank 2, all dims > 0
}

message PipelineEdge {
  string src = 1;            // resolves to a stage id
  string dst = 2;            // resolves to a stage id
}
```

`shape` is rank 2 because the resolver models a chip as `(row, col)`; MGD itself allows any
rank. Edges carry no loopback marking and no channel count - see 2.1 for why.

None of the comments above are enforced by protobuf - proto3 has no `required`, and
uniqueness and cross-references are beyond what a schema can state. They are field-level
obligations on `validate_pipeline_descriptors`.

---

## 2. MGD pipeline graph verification

Every check the declared pipeline graph must pass before a mapping can be committed.


> **Note - the tiers check topology, not the pipeline builder.**
>
> Some checks that look like they belong above are left out on purpose. They constrain **how the
> pipeline builder happens to be implemented**, not whether the topology can host the pipeline,
> and **the builder already checks each one itself**. Examples:
>
> - **Chips per stage.** A `1x1` stage in a ring has one chip for both entry and exit. The
>   builder's deconfliction step fails with *"has only one chip at both the entry and exit
>   boundary"*.
> - **A link per repeated hop.** `s0 -> s1` declared twice needs two distinct ethernet links.
>   The builder resolves each edge through `discover_connections`.
> - **Ring closure.** A `1x4` row of blocks with no wrap admits a path but never a loop; an
>   odd-length ring on a bipartite grid is likewise impossible. The builder's backtracking
>   search simply finds no assignment.
> - **Inter-mesh cut.** Two stages on different meshes need a `connections` entry between that
>   mesh pair. The builder's per-edge link lookup throws when it is absent.
>
> Duplicating these would put **builder internals into MGD validation**, which goes stale as
> soon as the search or deconfliction changes.
>
> This is also why **repeated edges are allowed** (2.1) even though the builder ignores the
> extras today - it reads `links[0]` for every edge. That is an **implementation limit, not a
> topology limit**, so the declaration is permitted now and **real multi-channel support needs
> no schema change later**.

### 2.1 Tier A - descriptor only

**When it runs:** MGD parse, before anything else.

- **Control-plane stage:** first one - `MeshGraph` construction in `ControlPlane::init_control_plane`.
- **Available:** the proto only. No PSD, no mapper, no fabric.
- **Site:** a new `validate_pipeline_descriptors`, next to the existing validators.
- **On failure:** load error, not a map-time error.

```cpp
// mesh_graph_descriptor.cpp, MeshGraphDescriptor validation
validate_mesh_topology(proto, all_errors);
validate_express_connections(proto, all_errors);
validate_graph_topology_and_connections(proto, all_errors);
validate_pinnings(proto, all_errors);
validate_pipeline_descriptors(proto, all_errors);   // <-- Tier A
```

**Field-level obligations are the schema comments in 1.3** - non-empty and unique `name` and
`id`, `shape` present at rank 2 with positive dims, `src`/`dst` resolving to declared ids, and
the minimum counts. Two of them are worth a word each: exactly one
descriptor is active, so 2.2's capacity is evaluated for it alone under exclusive use of the
fabric; and the minimum-count rule exists because the resolver builds its node set by scanning
edges, so a stage no edge mentions is silently dropped rather than rejected.

What the schema cannot state at all is the graph structure. Two rules with the declaration that
violates each, then two things that look like rules and are not.

#### No self-edges

```textproto
edges { src: "s0" dst: "s0" }        REJECT
```

`discover_connections` skips `i == j`, so no block is ever adjacent to itself and no self link
can exist. Left unchecked, the resolver reports this as `"cycle detected in non-loopback
edges"`, naming neither the stage nor the real cause.

#### Repeating an edge declares channels

Edges are **directional**, so the pair is ordered. Both of these are legal:

```textproto
edges { src: "s0" dst: "s1" }
edges { src: "s0" dst: "s1" }        OK - two channels on the s0 -> s1 hop

edges { src: "s0" dst: "s1" }
edges { src: "s1" dst: "s0" }        OK - antiparallel, a 2-stage ring
```

Multiplicity is expressed by **repetition**, so `edges` needs no `channels` field. The
antiparallel form is the only way to declare a two-stage ring: conversion marks one direction
as the return path and the remainder is acyclic.

#### Cycles are allowed - conversion breaks them

```textproto
edges { src: "s0" dst: "s1" }
edges { src: "s1" dst: "s2" }
edges { src: "s2" dst: "s0" }        OK - a 3-stage ring
```

### 2.2 Tier B - descriptor plus the declared fabric

**When it runs:** after the mesh graph is built, before any hardware is looked at.

- **Control-plane stage:** still `MeshGraph` construction, but after `initialize_from_mgd` has
  resolved `top_level_instance` and computed slice shapes.
- **Available:** mesh instances, slice shapes, declared connections. **Not** the PSD - this runs
  before `run_physical_system_discovery`, so there is no physical anything.
- **On failure:** reject before the mapper is ever constructed.

```cpp
// control_plane.cpp, ControlPlane::init_control_plane
this->mesh_graph_ = std::make_unique<MeshGraph>(cluster.get_cluster_type(), mesh_graph_desc_file, fabric_config);
//  <-- Tier A during that parse, Tier B right here, on the built graph

auto psd = tt::tt_metal::run_physical_system_discovery(...);       // physical world appears
this->physical_system_descriptor_ = std::make_unique<PhysicalSystemDescriptor>(std::move(psd));
```

Purely declarative, and that is the point: these reject pipelines impossible on *any* mapping of
this MGD, which makes them the highest value checks in the whole list.

#### Stage shape must equal the mesh's implied slice shape

A `MeshDescriptor` never states a submesh shape; `host_topology` partitions `device_topology`
and the slice is the elementwise quotient, as `MeshGraph::initialize_from_mgd` computes it:

```
mesh_descriptors { M0 }                pipeline_descriptors
  device_topology  [ 8, 4 ]              s0 shape [ 4, 2 ]
  host_topology    [ 2, 2 ]              s1 shape [ 4, 2 ]
        |                                s2 shape [ 4, 2 ]
        | 8/2 , 4/2                      s3 shape [ 4, 2 ]
        v                                        |
  slice shape [ 4, 2 ]  === must equal ===========+
  slice count  4        === must cover === 4 stages
```

```
M0 device grid 8 x 4, host_topology 2 x 2

  +-------+-------+
  |  4x2  |  4x2  |     4 slices of 4x2
  +-------+-------+     4 stages of 4x2     OK
  |  4x2  |  4x2  |
  +-------+-------+

same mesh, host_topology 4 x 2  ->  slice 2x2  !=  stage 4x2   REJECT
```

- Device dims divisible by host dims. Already checked by `validate_mesh_topology`.
- Every distinct stage `shape` equals some mesh instance's slice shape, **exactly** - not
  merely divides it. The multi-mesh carve takes the *first* submesh whose origin is local
  (`graph.py` line 461) so a rank contributes exactly one submesh however many fit; a stage
  smaller than its rank's slice silently strands the remainder.
- `host_topology` is a **uniform** partition, so every slice of a mesh instance has the same
  shape. Heterogeneous pipelines are therefore uniform *per mesh instance* and vary *between*
  instances, which is why `shape` lives on the stage rather than on the descriptor.

#### Capacity

- **Total capacity:** stage count equals total slot count exactly. `_collect_submeshes`
  asserts `#ranks == #stages` and `#submeshes == #nodes`, and `model_pipeline.py` raises on
  `num_stages != num_procs`. The weaker "does not exceed" form only becomes correct if
  partial allocation lands.
- **Per-shape capacity:** for each distinct shape, the number of stages requesting it matches
  the slices the MGD declares at that shape, counted per mesh *instance*. Each instance
  contributes `prod(host_topology.dims)` slices, so the same descriptor instantiated twice
  contributes twice as many slots.

```
gpt_oss_120b_single_galaxy
  instance       slices available   stages needed
  M0 mesh_id 0    4 x [ 4, 2 ]       4 x [ 4, 2 ]      OK  exact
  ---------------------------------------------------------------
  total           4                  4

llama_8b_pod
  instance       slices available   stages needed
  M0 mesh_id 0    8 x [ 4, 2 ]       8 x [ 4, 2 ]      OK  exact
  M1 mesh_id 1   32 x [ 1, 2 ]      32 x [ 1, 2 ]      OK  exact
  ---------------------------------------------------------------
  total          40                 40

5 stages of [ 4, 2 ] against M0's 4 slices   ->   REJECT
41 stages total against 40 slots             ->   REJECT
```

### 2.3 Tier C - topology mapping constraints

This is the new phase the ticket asks for, and the only tier that needs physical adjacency.

**When it runs:** after the topology mapper starts, during its solving stage.

- **Control-plane stage:** `TopologyMapper` construction. After the PSD exists, **before
  `load_physical_chip_mapping` commits**.
- **Site:** inside `map_multi_mesh_to_physical`, once per *candidate* mapping.
- **Available:** the logical-to-physical binding, so real adjacency.
- **Blocked on:** control-plane decoupling (4.1) - the control plane is mid-construction here and
  cannot be queried.

**It constrains which mappings are valid.** Not a report on a finished mapping - **a candidate
that fails Tier C is rejected as a solution** and the mapper keeps searching. Same mechanism the
auto-mapper already uses for its other constraints:

```cpp
// topology_mapper_utils.cpp, map_multi_mesh_to_physical - the retry loop
auto reject_mesh_pair_mapping = [&](MeshId logical_mesh_id, MeshId physical_mesh_id,
                                   const std::string& failure_reason) -> bool {
    return handle_forbidden_constraint(...);   // add_forbidden_constraint, then re-solve
};

if (!exit_node_constraints_success) {                       // existing rejection
    if (!reject_mesh_pair_mapping(logical_mesh_id, physical_mesh_id,
            "exit node constraints cannot be satisfied ...")) { return result; }
    continue;
}
if (pinning_constraint_failure.has_value()) { ... }         // existing rejection

// Tier C would sit alongside these, rejecting on pipeline infeasibility
```

- **On failure:** forbid that logical/physical mesh pair, re-solve.
- **On repeated failure:** bails out like the rest - when the forbid over-constrains, or
  `max_retry_attempts` is exceeded, surfacing as the existing
  *"Graph specified in MGD could not fit in the discovered physical topology"*.

Tiers A and B read the MGD as a declaration. This tier reads it **as mapped**: once the mapper
has bound each logical `FabricNodeId` to a physical `AsicID` from the PSD, every logical chip has
a real ASIC behind it and every declared connection has a real cabled link. Only then can the
pipeline's edges be tested against links that actually exist. The question is not whether the
graph is well-formed, but whether the **mapped** graph has **enough connections in the right
places** to carry every edge.

#### Why the location of a connection matters, not just the count

MGD declares inter-mesh connectivity, but a declaration is satisfiable only if the links land
where the pipeline needs them. Take a 4-stage ring split two stages per mesh, which puts **two
edges across the cut**, one in each direction:

```
pipeline   s0 -> s1 -> s2 -> s3 -> s0        placement   s0 s1 on M0, s2 s3 on M1
cut edges  s1 -> s2   and   s3 -> s0


CASE A   2 connections, 2 distinct chip pairs                          MAP OK

    M0                          M1
  +------+------+  A0<->B0    +------+------+
  |  s0  |  s1  |=============|  s2  |  s3  |
  +------+------+             +------+------+
      ^                                   |
      +===================================+
                   A1<->B1

  s1 -> s2  rides A0<->B0        s3 -> s0  rides A1<->B1


CASE B   2 connections, both pinned to the SAME chip pair              REJECT

    M0                          M1
  +------+------+  A0<->B0    +------+------+
  |  s0  |  s1  |=============|  s2  |  s3  |
  +------+------+  A0<->B0    +------+------+

  count 2, locations 1.  s1 -> s2 is fine and has a spare lane it cannot use.
  s3 -> s0 needs a link touching s3's block and s0's block, and there is none.


CASE C   1 connection                                     REJECT as ring
                                                          OK as an open chain
    M0                          M1
  +------+------+  A0<->B0    +------+------+
  |  s0  |  s1  |-------------|  s2  |  s3  |
  +------+------+             +------+------+

  s1 -> s2  rides A0<->B0        s3 -> s0  has nothing to ride
```

**Reading the diagram.**

- **Case A** - 2 crossings, 2 different chip pairs. Both cut edges have a link at their own
  endpoints. Maps.
- **Case B** - 2 crossings, 2 channels declared, but only 1 chip pair. `s1 -> s2` is fine, the
  spare lane is useless to `s3 -> s0`, which has no link at *its* endpoints. Rejected.
- **Case C** - 1 chip pair. Enough for an open chain, never enough to close a ring.

The point of Case B: **counting is not enough, the links have to be in the right place.** Count
is declared in the MGD, so Tier B can see it. *Where* the links land is decided by the binding the
mapper chose, so only this tier can see it.

**The check itself.** For a candidate mapping, is there a way to put each stage on a block such
that every edge - loopback included - joins two blocks that a real link already connects? If not,
reject the candidate.

Two clarifications:

- **Block** means one submesh slice, not one chip. Adjacent blocks are blocks with at least one
  direct ethernet link between them.
- An edge repeated `k` times needs `k` *distinct* links between the same two blocks. The resolver
  reads `links[0]` for every edge today, so this is unimplemented rather than just unchecked (2.1).

Finding that placement is subgraph isomorphism, and a Hamiltonian cycle when the pipeline is a
ring spanning every block - both NP-complete, which is what section 5 has to encode.

### 2.4 Tier D - inside the pipeline builder

**When it runs:** during pipeline build itself, on the committed mapping. This is what happens
today, and it is the only tier that exists right now.

- **Not a separate pass.** Tier D *is* the pipeline builder's own resolution - `_validate` on the
  declared graph, then `resolve_graph_layout` on the carved submeshes.
- **Site:** `PipelineGraph.build_topology` in `pipeline_builder/graph.py`.
- **Available:** everything, since the mapping has committed and the control plane is live.
- **Cannot steer anything.** The mapping is fixed by the time it runs, so a failure here is a hard
  error rather than a rejected candidate. That is the whole problem the ticket is about.

```python
# pipeline_builder/graph.py - PipelineGraph.build_topology
self._validate()                     # every edge endpoint is a declared node

result = ttnn._ttnn.multi_device.experimental.resolve_graph_layout(
    nodes=list(self.nodes.keys()),
    edges=edge_tuples,
    submesh_chips=global_submesh_chips,   # carved by _collect_submeshes
    node_chip_counts=node_chip_counts,
)
```

Open question: does it survive as defense in depth once Tier C exists, or is it removed as a
duplicated constraint that can drift?

---

## 3. Encoding graph layout information in MGD

**Reserved.** *TBD.*

The question: given the descriptor from section 1, the concrete proto change and the full
set of code that has to learn about a new top-level MGD section.


## 4. Mapping the graph as a stage in topology mapper util

**Reserved.** *TBD.*

The question: where the new pipeline-feasibility phase runs, and how it reports failure
back into the mapping search.

Seed pointers:

- `map_multi_mesh_to_physical` in `tt_metal/fabric/topology_mapper_utils.cpp` today runs
  an inter-mesh solve, then a per-pair intra-mesh solve, with a retry loop that adds
  forbidden constraints and re-solves inter-mesh when an intra-mesh pair fails. The new
  pipeline phase sits between those two.
- Open: does the new phase reuse `resolve_graph_layout` once decoupled, or does it want a
  cheaper feasibility-only variant that answers yes/no without computing entry/exit
  coordinates and H2D/D2H placement? Given 1.1 scopes the descriptor to feasibility, the
  cheaper variant may be the better fit, and it would not need the loopback designation
  or host IO at all.
