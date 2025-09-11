# Mesh Graph Descriptor (MGD) 2.0 — Quick Use Guide

This guide explains how to define and load a Mesh Graph Descriptor (MGD) 2.0 for TT‑Fabric using the up‑to‑date schema in [`tt_metal/fabric/protobuf/mesh_graph_descriptor.proto`](protobuf/mesh_graph_descriptor.proto). It focuses on how to write a valid textproto, how node references work, and which fields are required.

A Mesh Graph Descriptor is the input to the Fabric Control Plane to specify a partition and topology of a device that a user would like to initialize fabric for.

---

### Where to look
- Schema: [`tt_metal/fabric/protobuf/mesh_graph_descriptor.proto`](protobuf/mesh_graph_descriptor.proto)
- Example textproto: [`tests/tt_metal/tt_fabric/custom_mesh_descriptors/mgd2_syntax_check_mesh_graph_descriptor.textproto`](../../tests/tt_metal/tt_fabric/custom_mesh_descriptors/mgd2_syntax_check_mesh_graph_descriptor.textproto)
- C++ API: [`tt_metal/api/tt-metalium/mesh_graph_descriptor.hpp`](../../tt_metal/api/tt-metalium/mesh_graph_descriptor.hpp)

## Background

A Mesh Graph Descriptor (MGD) specifies the logical topology that a user specifies for running their workload on an multi-host Exabox cluster.

This descriptor will capture information about how to compose a “big-mesh” (intra-mesh) across multiple host systems and how to connect meshes together (inter-mesh) in some user-topology. The MGD represents the minimum hardware allocation requirements for a workload to multi-host Exabox cluster.

Read more about Text proto at [Mesh Graph Descriptor 2.0](https://docs.google.com/document/d/1291H1Wl_pSkIGHP9B_L6oikaD3MflAGXg3Lox1O8S0c/edit?usp=sharing)


## Minimal workflow
> This is currently TBD
1) Write an MGD 2.0 textproto
2) Provide Control Plane with the Mesh graph descriptor via `TT_MESH_GRAPH_DESC_PATH`
3) Run metal workload using fabric

```bash
TT_MESH_GRAPH_DESC_PATH="/home/my_custom_desc.textproto" ./your_fabric_script
```

Notes:
- Unknown fields in the textproto are tolerated for forward/backward compatibility during parsing.

---

## Writing an MGD 2.0 textproto

An MGD has three parts:
- mesh_descriptors: definitions of reusable meshes
- graph_descriptors: logical groupings and connectivity across meshes/graphs
- top_level_instance: the single root instance to instantiate

Follow comments written in .proto file for detailed instructions for how to write an MGD file.
[`tt_metal/fabric/protobuf/mesh_graph_descriptor.proto`](protobuf/mesh_graph_descriptor.proto)

## Additional examples

### Single mesh (no graphs):
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

### All‑to‑all between two graphs:
```proto
graph_descriptors {
  name: "G0" type: "POD"
  instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
  instances { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
  graph_topology: { layout_type: ALL_TO_ALL }
  channels: { count: 2 policy: STRICT }
}
graph_descriptors {
  name: "G1" type: "CLUSTER"
  instances { graph { graph_descriptor: "G0" graph_id: 0 } }
  instances { graph { graph_descriptor: "G0" graph_id: 1 } }
  graph_topology: { layout_type: ALL_TO_ALL }
  channels: { count: 2 policy: RELAXED }
}
top_level_instance { graph { graph_descriptor: "G1" graph_id: 0 } }
```

### 16 LoudBox Cluster
![16 lb cluster](../../docs/source/common/images/16LB_Cluster.png)

```proto
# --- Mesh Descriptors ------------------------------------------------------

mesh_descriptors {
  id: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [2, 4] }
  channels { count: 2 policy: STRICT }
}

# --- Graph Descriptors -----------------------------------------------------

graph_descriptors {
  id: "G0"
  type: "SUPERPOD"
  instances { mesh { mesh_descriptor: "M0" id: 0 } }
  instances { mesh { mesh_descriptor: "M0" id: 1 } }
  instances { mesh { mesh_descriptor: "M0" id: 2 } }
  instances { mesh { mesh_descriptor: "M0" id: 3 } }
  topology: { layout_type: ALL-TO-ALL }
  channels { count: 2 policy: STRICT }
}

graph_descriptors {
  id: "G1"
  type: "CLUSTER"
  instances { graph { graph_descriptor: "G0" id: 0 } }
  instances { graph { graph_descriptor: "G0" id: 1 } }
  instances { graph { graph_descriptor: "G0" id: 2 } }
  instances { graph { graph_descriptor: "G0" id: 3 } }
  topology: { layout_type: ALL-TO-ALL }
  channels { count: 2 policy: STRICT }
}

# --- Instantiations --------------------------------------------------------

top_level_instance { graph { graph_descriptor: "G1" id: 0 } }
```

If in doubt, follow the `.proto` in code; it is the source of truth.
