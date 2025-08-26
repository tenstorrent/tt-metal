## Mesh Graph Descriptor (MGD) 2.0 — Quick Use Guide

This guide explains how to define and load a Mesh Graph Descriptor (MGD) 2.0 for TT‑Fabric using the up‑to‑date schema in `tt_metal/fabric/protobuf/mesh_graph_descriptor.proto`. It focuses on how to write a valid textproto, how node references work, and which fields are required.

### Where to look
- Schema: `tt_metal/fabric/protobuf/mesh_graph_descriptor.proto`
- Example textproto: `tests/tt_metal/tt_fabric/custom_mesh_descriptors/mgd2_syntax_check_mesh_graph_descriptor.textproto`
- C++ API: `tt_metal/api/tt-metalium/mesh_graph_descriptor.hpp`

---

## Minimal workflow
1) Write an MGD 2.0 textproto
2) Parse it with the C++ API
3) Provide the parsed descriptor to control‑plane logic (STM/FCP) for mapping/initialization

```cpp
#include <tt-metalium/mesh_graph_descriptor.hpp>

// From a string
const std::string mgd_text = R"proto(
  mesh_descriptors: {
    name: "M0"
    arch: WORMHOLE_B0
    device_topology: { dims: [ 2, 4 ] }
    host_topology:   { dims: [ 1, 1 ] }
    channels:        { count: 2 }
  }
  top_level_instance: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
)proto";
tt::tt_fabric::MeshGraphDescriptor mgd_from_string(mgd_text);

// From a file
tt::tt_fabric::MeshGraphDescriptor mgd_from_file(
  std::filesystem::path("tests/tt_metal/tt_fabric/custom_mesh_descriptors/mgd2_syntax_check_mesh_graph_descriptor.textproto"));
```

Notes:
- Unknown fields in the textproto are tolerated for forward/backward compatibility during parsing.

---

## Writing an MGD 2.0 textproto

An MGD has three parts:
- mesh_descriptors: definitions of reusable meshes
- graph_descriptors: logical groupings and connectivity across meshes/graphs
- top_level_instance: the single root instance to instantiate

### 1) Mesh descriptors
Represents a uniform, fully connected grid of devices with optional express links.

Required fields:
- `name` (string): logical identifier (e.g., "M0").
- `arch` (Architecture): `WORMHOLE_B0`, `BLACKHOLE`.
- `device_topology` (TorusTopology): `dims` list is required; `types` is optional per dimension.
- `channels` (Channels): `count` is required; `policy` is optional.
- `host_topology` (MeshTopology): `dims` list is required.

Optional fields:
- `express_connections` (list of `src`, `dst` device indices within this mesh).

Schema (from code):
```proto
message MeshDescriptor {
  string name = 1;
  Architecture arch = 2;
  TorusTopology device_topology = 3;  // dims required; types optional
  Channels channels = 4;               // count required; policy optional
  MeshTopology host_topology = 5;      // dims required
  message ExpressConnection { int32 src = 1; int32 dst = 2; }
  repeated ExpressConnection express_connections = 7;
}
```

Torus vs Mesh topology:
- `device_topology` uses `TorusTopology` with two parallel arrays: `dims` and `types` (LINE or RING) per dimension. If `types` is omitted or shorter than `dims`, treat unspecified dimensions as LINE.
- `host_topology` uses `MeshTopology` and only accepts `dims` (no per‑dimension type).

Example:
```proto
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 8, 4 ] types: [ LINE, RING ] }
  host_topology   { dims: [ 1, 2 ] }
  channels        { count: 4 policy: STRICT }
  express_connections { src: 0 dst: 4 }
}
```

### 2) Graph descriptors
Groups meshes or graphs of a single kind and defines their connectivity.

Required fields:
- `name` (string): logical identifier (e.g., "G0").
- `type` (string): freeform grouping label (e.g., "FABRIC", "POD", "CLUSTER").
- `instances` (list of `NodeRef`): all entries must be either meshes or graphs (do not mix types).

Connectivity (choose one style):
- Shorthand `graph_topology` (e.g., `ALL_TO_ALL`, `RING`), typically paired with `channels` to define edge multiplicity.
- Explicit `connections` list, each with two `nodes` and `channels`; `directional` defaults to false when omitted.

Schema (from code):
```proto
message GraphDescriptor {
  string name = 1;
  string type = 2;                    // grouping label
  repeated NodeRef instances = 3;     // must be uniform type: all mesh or all graph
  optional Channels channels = 4;     // commonly used with graph_topology
  optional GraphTopology graph_topology = 5; // ALL_TO_ALL, RING
  repeated Connection connections = 6;       // alternative to graph_topology
}
```

Example (mixed features):
```proto
graph_descriptors {
  name: "G0"
  type: "POD"
  instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
  instances { mesh { mesh_descriptor: "M1" mesh_id: 0 } }
  connections {
    nodes { mesh { mesh_descriptor: "M0" mesh_id: 0 device_id: 0 } }
    nodes { mesh { mesh_descriptor: "M1" mesh_id: 0 device_id: 0 } }
    channels { count: 2 policy: RELAXED }
    directional: false
  }
}

graph_descriptors {
  name: "G1"
  type: "CLUSTER"
  channels { count: 2 policy: RELAXED }
  instances { graph { graph_descriptor: "G0" graph_id: 0 } }
  instances { graph { graph_descriptor: "G0" graph_id: 1 } }
  graph_topology: ALL_TO_ALL
}
```

### 3) Top‑level instance
A single `NodeRef` that is the root of the allocation.

Schema (from code):
```proto
message MeshGraphDescriptor {
  repeated MeshDescriptor mesh_descriptors = 1;
  repeated GraphDescriptor graph_descriptors = 2; // optional
  NodeRef top_level_instance = 3;                 // required, single
}
```

Example:
```proto
top_level_instance { graph { graph_descriptor: "G1" graph_id: 0 } }
```

---

## Node references (NodeRef) explained
Node references identify a mesh, a graph, or drill down to a specific device inside a mesh. They can be nested when referencing down through graphs to meshes.

Types:
- Mesh reference:
  ```proto
  mesh { mesh_descriptor: "M0" mesh_id: 0 }
  ```
- Mesh device reference:
  ```proto
  mesh { mesh_descriptor: "M0" mesh_id: 0 device_id: 3 }
  ```
- Graph reference:
  ```proto
  graph { graph_descriptor: "G0" graph_id: 2 }
  ```
- Nested: graph → mesh → device (read as G0(0):M0(1):D3)
  ```proto
  graph {
    graph_descriptor: "G0" graph_id: 0
    mesh { mesh_descriptor: "M0" mesh_id: 1 device_id: 3 }
  }
  ```

From schema:
```proto
message NodeRef { oneof node_ref { MeshRef mesh = 1; GraphRef graph = 2; } }
message MeshRef { string mesh_descriptor = 1; int32 mesh_id = 2; optional int32 device_id = 3; }
message GraphRef {
  string graph_descriptor = 1; int32 graph_id = 2;
  oneof sub_ref { MeshRef mesh = 3; GraphRef graph = 4; }
}
```

Nuances:
- `mesh_id`/`graph_id` are per‑descriptor instance indices assigned by you in the textproto; use them consistently across references.
- `device_id` indexes into the linearized device list of the targeted mesh. Use the same indexing scheme used by your router/runtime.
- When specifying `GraphRef`, you may further narrow to a sub‑mesh (and optionally a device) using the nested `sub_ref` fields.

---

## Required fields per section (concise)
- MeshDescriptor: `name`, `arch`, `device_topology.dims`, `channels.count`, `host_topology.dims`.
- GraphDescriptor: `name`, `type`, `instances` (uniform node type). One of `graph_topology` (+ usually `channels`) or explicit `connections`.
- Top level: exactly one `top_level_instance` NodeRef.

Enums (use exact spellings):
- `Architecture`: `WORMHOLE_B0`, `BLACKHOLE`.
- `GraphTopology`: `ALL_TO_ALL`, `RING`.
- `Policy`: `STRICT`, `RELAXED`.

Channels:
```proto
message Channels { int32 count = 1; optional Policy policy = 2; }
```
If `policy` is omitted, treat as STRICT unless your control plane overrides this default.

Connection:
```proto
message Connection {
  repeated NodeRef nodes = 1;     // typically exactly two nodes
  Channels channels = 2;          // required
  optional bool directional = 3;  // default false
}
```

---

## Common pitfalls
- Use `name` (not `id`) for `MeshDescriptor`/`GraphDescriptor` identifiers; references use `mesh_descriptor`/`graph_descriptor` plus `mesh_id`/`graph_id`.
- `device_topology` is `TorusTopology` with optional per‑dimension `types`; `host_topology` is `MeshTopology` with only `dims`.
- Do not mix meshes and graphs in the same `instances` list of a `GraphDescriptor`.
- Ensure all `dims` and `channels.count` are positive; ensure `types.size()` matches `dims.size()` or omit missing `types` to default to LINE.
- When using `graph_topology`, also provide `channels` to determine edge multiplicity.

---

## Additional examples

Single mesh (no graphs):
```proto
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology: { dims: [ 2, 4 ] }
  host_topology:   { dims: [ 1, 1 ] }
  channels:        { count: 2 }
}
top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
```

All‑to‑all between two graphs:
```proto
graph_descriptors {
  name: "G0" type: "POD"
  instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
  instances { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
  graph_topology: ALL_TO_ALL
  channels: { count: 2 policy: STRICT }
}
graph_descriptors {
  name: "G1" type: "CLUSTER"
  instances { graph { graph_descriptor: "G0" graph_id: 0 } }
  instances { graph { graph_descriptor: "G0" graph_id: 1 } }
  graph_topology: ALL_TO_ALL
  channels: { count: 2 policy: RELAXED }
}
top_level_instance { graph { graph_descriptor: "G1" graph_id: 0 } }
```

---

## Terminology updates vs older drafts
- `id` → `name` on descriptors; references use `mesh_descriptor`/`graph_descriptor` plus `mesh_id`/`graph_id`.
- `topology` → `graph_topology` for graphs; value is `ALL_TO_ALL` (underscore), not `ALL-TO-ALL`.
- `device_ref`/nested `DeviceRef` → flattened `device_id` on `MeshRef`.
- `TorusTopology` now uses parallel `dims` and `types` arrays.

If in doubt, follow the `.proto` in code; it is the source of truth.
