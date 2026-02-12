# Physical Groupings File Specification

## Overview

In replacement of the rankfile and rank_bindings file as needed for running tt-run on multi-host systems, we will be introducing a cluster physical groupings file deployed with each machine cluster specifying the valid physical groupings for each cluster of machines. This file is provided by the cluster administrator and is used by FM to understand which subsets of ASICs can be used as candidate physical meshes for a given logical mesh in the MGD.

The Physical Groupings file defines the hierarchical structure of physical resources (meshes, pods, superpods, clusters) in the cluster. This file uses a **declarative approach** that defines groupings in terms of ASIC locations and other groupings without requiring explicit ASIC IDs. The actual ASIC IDs are derived at runtime from the Physical System Descriptor (PSD).

The groupings file is complementary to the Physical System Descriptor (PSD):
- **PSD**: Flat graph of all ASICs + links
- **Groupings**: Allowed carve-outs (meshes/pods/superpods/clusters) over that flat graph

Files use **protobuf text format** (`.textproto`) with schema validation. The schema enforces validation rules and ensures type safety.

## Quick Example

The groupings file uses an **adjacency graph format** with instances and connections. Each instance must have a unique ID, and connections reference these IDs:

```protobuf
# Using custom names
groupings {
  custom_name: "trays"
  instances: [
    { id: 0 asic_location: ASIC_LOCATION_1 },
    { id: 1 asic_location: ASIC_LOCATION_2 },
    # ... through ASIC_LOCATION_8
  ]
}

groupings {
  custom_name: "hosts"
  instances: [
    { id: 0 grouping_ref { custom_name: "trays" } }
  ]
}

groupings {
  custom_name: "meshes"
  instances: [
    { id: 0 grouping_ref { custom_name: "hosts" } },  # Each mesh = 1 host
    { id: 1 grouping_ref { custom_name: "hosts" } },
    { id: 2 grouping_ref { custom_name: "hosts" } },
    { id: 3 grouping_ref { custom_name: "hosts" } },
    { id: 4 grouping_ref { custom_name: "hosts" } },
    { id: 5 grouping_ref { custom_name: "hosts" } },
    { id: 6 grouping_ref { custom_name: "hosts" } },
    { id: 7 grouping_ref { custom_name: "hosts" } }
  ]
  connections {
    row_major_mesh {
      dims: [2, 4]
      dim_types: [LINE, RING]
      num_connections: 2
    }
  }
}

groupings {
  custom_name: "pods"
  instances: [
    { id: 0 grouping_ref { custom_name: "meshes" } },  # Each pod contains 2 meshes
    { id: 1 grouping_ref { custom_name: "meshes" } }
  ]
  connections {
    all_to_all { num_connections: 2 }
  }
}

groupings {
  custom_name: "superpods"
  instances: [
    { id: 0 grouping_ref { custom_name: "pods" } },  # Each superpod contains 3 pods
    { id: 1 grouping_ref { custom_name: "pods" } },
    { id: 2 grouping_ref { custom_name: "pods" } }
  ]
  connections {
    row_major_mesh {
      dims: [3, 1]
      dim_types: [LINE, LINE]
      num_connections: 2
    }
  }
}

groupings {
  custom_name: "clusters"
  instances: [
    { id: 0 grouping_ref { custom_name: "superpods" } },  # Each cluster contains 2 superpods
    { id: 1 grouping_ref { custom_name: "superpods" } }
  ]
  connections {
    custom { src_instance: 0 dst_instance: 1 num_connections: 2 }
  }
}

# Using predefined keywords
groupings {
  preset_name: TRAY_1
  instances: [
    { id: 0 asic_location: ASIC_LOCATION_1 },
    { id: 1 asic_location: ASIC_LOCATION_2 },
    # ... through ASIC_LOCATION_8
  ]
}

groupings {
  preset_name: MESH
  instances: [
    { id: 0 grouping_ref { preset_name: TRAY_1 } },
    { id: 1 grouping_ref { preset_name: TRAY_2 } }
  ]
  connections {
    row_major_mesh {
      dims: [2, 1]
      dim_types: [LINE, LINE]
      num_connections: 2
    }
  }
}
```

The actual ASIC IDs are derived at runtime from the PSD, making this file completely hardware-agnostic and reusable across different clusters.

## Schema and Validation

The physical groupings file uses protobuf text format with schema validation. Key features:

- **Adjacency Graph Format**: Groupings are defined as graphs with instances and connections
- **Instances**: Each grouping contains a list of instances, each with a unique ID
- **Instance IDs**: Each instance must have a unique `id` field within its grouping (used for connections)
- **Grouping Names**: Can use either predefined keywords (`preset_name`) or custom names (`custom_name`)
  - `preset_name`: Predefined keyword enum values (TRAY_1, TRAY_2, TRAY_3, TRAY_4, MESH)
  - `custom_name`: Custom string names (e.g., "pods", "meshes", "superpods", "clusters")
- **Connections**: Separate section that defines how instances connect and their topology using connection types (all-to-all, row-major-mesh, or custom)
- **Topology**: Topology (mesh dimensions, dimension types) is defined in the connections section, not on individual instances

### Predefined Keywords

The schema supports predefined keywords that represent preset groups:

- **TRAY_1, TRAY_2, TRAY_3, TRAY_4**: Predefined tray groupings
- **MESH**: Predefined mesh grouping

These keywords can be used with `preset_name` instead of defining custom names. For example:

```protobuf
groupings {
  preset_name: TRAY_1
  instances: [
    { id: 0 asic_location: ASIC_LOCATION_1 },
    { id: 1 asic_location: ASIC_LOCATION_2 },
    # ... through ASIC_LOCATION_8
  ]
}

groupings {
  preset_name: MESH
  instances: [
    { id: 0 grouping_ref { preset_name: TRAY_1 } },
    { id: 1 grouping_ref { preset_name: TRAY_2 } }
  ]
  connections {
    row_major_mesh {
      dims: [2, 1]
      dim_types: [LINE, LINE]
      num_connections: 2
    }
  }
}
```
- **ASIC Locations as Constants**: ASIC locations 1-8 are predefined as enum constants (`ASIC_LOCATION_1` through `ASIC_LOCATION_8`)
- **Type Safety**: Protobuf enforces that each instance is either an ASIC location or a grouping reference
- **Required Groupings**: The "meshes" grouping must be defined (enforced by validation)
- **Multiple Definitions**: The same grouping name can be defined multiple times (useful for custom groupings)

See `tt_metal/fabric/protobuf/physical_grouping_descriptor.proto` for the complete schema definition.

## Groupings Explained

### Physical Groupings

**ASIC Locations**: Predefined constants (`ASIC_LOCATION_1` through `ASIC_LOCATION_8`) representing individual ASIC positions within a tray. These are defined as enum values in the protobuf schema and are always available.

**Trays**: Contains all 8 ASIC locations. Defined using ASIC location enum values. Each instance must have a unique ID.

```protobuf
# Using custom name
groupings {
  custom_name: "trays"
  instances: [
    { id: 0 asic_location: ASIC_LOCATION_1 },
    { id: 1 asic_location: ASIC_LOCATION_2 },
    # ... through ASIC_LOCATION_8
  ]
}

# Or using predefined keyword
groupings {
  preset_name: TRAY_1
  instances: [
    { id: 0 asic_location: ASIC_LOCATION_1 },
    { id: 1 asic_location: ASIC_LOCATION_2 },
    # ... through ASIC_LOCATION_8
  ]
}
```

**Hosts**: Contains multiple trays. Defined using grouping references. Each instance must have a unique ID.

```protobuf
groupings {
  custom_name: "hosts"
  instances: [
    { id: 0 grouping_ref { custom_name: "trays" } }  # Each host contains 1 tray instance
    # Or: { id: 0 grouping_ref { preset_name: TRAY_1 } }
  ]
}
```

### Logical Groupings

**Meshes**: The required logical grouping. Can be defined using hosts, trays, or ASIC locations. **Note**: Meshes can have 1 instance, but all other groupings (pods, superpods, clusters) must have at least 2 instances. Each instance must have a unique ID. Topology is defined in the connections section, not on individual instances.

```protobuf
# Using custom name
groupings {
  custom_name: "meshes"
  instances: [
    { id: 0 grouping_ref { custom_name: "hosts" } }  # Each mesh contains 1 host (meshes can have 1 instance)
    # OR multiple instances:
    # { id: 0 grouping_ref { custom_name: "hosts" } },
    # { id: 1 grouping_ref { custom_name: "hosts" } },
    # { id: 2 grouping_ref { custom_name: "hosts" } },
    # { id: 3 grouping_ref { custom_name: "hosts" } }  # Each mesh contains 4 hosts
    # OR { id: 0 grouping_ref { custom_name: "trays" } }, ...  # Multiple trays per mesh
    # OR { id: 0 asic_location: ASIC_LOCATION_1 }, { id: 1 asic_location: ASIC_LOCATION_2 }, ...  # Direct ASIC locations
  ]
  # If multiple instances, define topology in connections:
  # connections {
  #   row_major_mesh {
  #     dims: [2, 4]
  #     dim_types: [LINE, RING]
  #     num_connections: 2
  #   }
  # }
}

# Or using predefined keyword
groupings {
  preset_name: MESH
  instances: [
    { id: 0 grouping_ref { preset_name: TRAY_1 } },
    { id: 1 grouping_ref { preset_name: TRAY_2 } }
  ]
  connections {
    row_major_mesh {
      dims: [2, 1]
      dim_types: [LINE, LINE]
      num_connections: 2
    }
  }
}
```

**Subdividing ASIC Locations**: You can use a subset of ASIC locations to create smaller meshes. For example, using only locations 1-4 instead of all 8 creates a mesh with half the ASICs per tray.

```protobuf
groupings {
  custom_name: "meshes"
  instances: [
    { id: 0 asic_location: ASIC_LOCATION_1 },
    { id: 1 asic_location: ASIC_LOCATION_2 },
    { id: 2 asic_location: ASIC_LOCATION_3 },
    { id: 3 asic_location: ASIC_LOCATION_4 }
  ]
  # Only using 4 ASIC locations from each tray instead of all 8
}
```

**Pods**: Contains meshes. Defined using grouping references. **Must have at least 2 instances.** Each instance must have a unique ID. Can optionally define connections between instances.

```protobuf
groupings {
  custom_name: "pods"
  instances: [
    { id: 0 grouping_ref { custom_name: "meshes" } },  # Each pod contains 2 meshes
    { id: 1 grouping_ref { custom_name: "meshes" } }
    # OR for 3 meshes:
    # { id: 0 grouping_ref { custom_name: "meshes" } },
    # { id: 1 grouping_ref { custom_name: "meshes" } },
    # { id: 2 grouping_ref { custom_name: "meshes" } }
    # OR mixing keywords and custom names:
    # { id: 0 grouping_ref { preset_name: MESH } },
    # { id: 1 grouping_ref { custom_name: "meshes" } }
  ]
  connections {
    all_to_all { num_connections: 2 }  # All instances connect to all others with 2 connections
  }
}
```

**Superpods**: Contains pods and/or meshes. Can mix different grouping types. **Must have at least 2 instances.** Each instance must have a unique ID. Can use row-major-mesh connections.

```protobuf
groupings {
  custom_name: "superpods"
  instances: [
    { id: 0 grouping_ref { custom_name: "pods" } },  # Each superpod contains 3 pods (must have at least 2)
    { id: 1 grouping_ref { custom_name: "pods" } },
    { id: 2 grouping_ref { custom_name: "pods" } }
    # OR { id: 0 grouping_ref { custom_name: "meshes" } }, ...  # Each superpod contains meshes directly
    # OR mix: { id: 0 grouping_ref { custom_name: "pods" } }, { id: 1 grouping_ref { preset_name: MESH } }, ...
  ]
  connections {
    row_major_mesh {
      dims: [3, 1]
      dim_types: [LINE, LINE]
      num_connections: 2
    }
  }
}
```

**Clusters**: Contains superpods, pods, and/or meshes. Can mix different grouping types. **Must have at least 2 instances.** Each instance must have a unique ID. Can use custom connections.

```protobuf
groupings {
  custom_name: "clusters"
  instances: [
    { id: 0 grouping_ref { custom_name: "superpods" } },  # Each cluster contains 2 superpods (must have at least 2)
    { id: 1 grouping_ref { custom_name: "superpods" } }
    # OR { id: 0 grouping_ref { custom_name: "pods" } }, ...  # Each cluster contains pods directly
    # OR { id: 0 grouping_ref { custom_name: "meshes" } }, ...  # Each cluster contains meshes directly
    # OR mix: { id: 0 grouping_ref { custom_name: "superpods" } }, { id: 1 grouping_ref { preset_name: MESH } }, { id: 2 grouping_ref { custom_name: "meshes" } }
  ]
  connections {
    custom { src_instance: 0 dst_instance: 1 num_connections: 2 }  # Explicit connection from instance 0 to instance 1
  }
}
```

### Connection Types

Groupings can define how instances connect to each other using three connection types:

**1. All-to-All Connections**: Every instance connects to every other instance.

```protobuf
groupings {
  custom_name: "pods"
  instances: [
    { id: 0 grouping_ref { custom_name: "meshes" } },
    { id: 1 grouping_ref { custom_name: "meshes" } }
  ]
  connections {
    all_to_all { num_connections: 2 }  # Each pair of instances has 2 connections
  }
}
```

**2. Row-Major Mesh Connections**: Instances are arranged in a grid with mesh connectivity.

```protobuf
groupings {
  custom_name: "superpods"
  instances: [
    { id: 0 grouping_ref { custom_name: "pods" } },
    { id: 1 grouping_ref { custom_name: "pods" } },
    { id: 2 grouping_ref { custom_name: "pods" } },
    { id: 3 grouping_ref { custom_name: "pods" } },
    { id: 4 grouping_ref { custom_name: "pods" } },
    { id: 5 grouping_ref { custom_name: "pods" } }
  ]
  connections {
    row_major_mesh {
      dims: [2, 3]              # 2x3 grid of instances
      dim_types: [LINE, RING]   # First dimension is LINE, second is RING
      num_connections: 2         # 2 connections per edge
    }
  }
}
```

- `dims`: Dimensions of the mesh (e.g., `[2, 3]` for a 2x3 grid)
- `dim_types`: Per-dimension connectivity type (`LINE` or `RING`) - these enum values are nested within `RowMajorMeshConnection` and can only be used in this context
  - `LINE`: No wrap-around (endpoints not connected)
  - `RING`: Wrap-around (endpoints connected, forming a ring)
- `num_connections`: Number of connections per edge in the mesh

**3. Custom Connections**: Explicit connections between specific instances.

```protobuf
groupings {
  custom_name: "clusters"
  instances: [
    { id: 0 grouping_ref { custom_name: "superpods" } },
    { id: 1 grouping_ref { custom_name: "superpods" } }
  ]
  connections {
    custom { src_instance: 0 dst_instance: 1 num_connections: 2 }  # Instance 0 -> Instance 1 with 2 connections
  }
  # Can have multiple custom connections
  connections {
    custom { src_instance: 1 dst_instance: 0 num_connections: 2 }  # Instance 1 -> Instance 0 with 2 connections
  }
}
```

- `src_instance`: Source instance index (0-based, refers to instances list)
- `dst_instance`: Destination instance index (0-based, refers to instances list)
- `num_connections`: Number of connections from source to destination

### Custom Groupings

You can define your own custom groupings using ASIC locations. This is useful for creating reusable sub-units like "half trays" or other logical divisions. You can define the same grouping name multiple times with different ASIC location sets.

```protobuf
groupings {
  # Define a custom grouping called "halftray" - first definition
  custom_name: "halftray"
  instances: [
    { asic_location: ASIC_LOCATION_1 },
    { asic_location: ASIC_LOCATION_2 },
    { asic_location: ASIC_LOCATION_3 },
    { asic_location: ASIC_LOCATION_4 }
  ]  # Lower half
}

groupings {
  # Same name, different ASIC locations - second definition
  custom_name: "halftray"
  instances: [
    { asic_location: ASIC_LOCATION_5 },
    { asic_location: ASIC_LOCATION_6 },
    { asic_location: ASIC_LOCATION_7 },
    { asic_location: ASIC_LOCATION_8 }
  ]  # Upper half
}

# Then use the custom grouping in meshes or other groupings
groupings {
  custom_name: "meshes"
  instances: [
    { id: 0 grouping_ref { custom_name: "halftray" } }  # Each mesh uses 1 half tray (can be either definition)
  ]
}
```

## Specification Format

**ASIC Locations**: Use enum constants (`ASIC_LOCATION_1` through `ASIC_LOCATION_8`).

**Grouping References**: Use `grouping_ref` with either `preset_name` (for keywords) or `custom_name` (for custom names):
- `{ id: 0 grouping_ref { custom_name: "hosts" } }` → Instance 0 references the "hosts" grouping
- `{ id: 0 grouping_ref { preset_name: TRAY_1 } }` → Instance 0 references the TRAY_1 keyword grouping
- Instances are explicitly instantiated - list each instance separately with its ID

**Grouping Names**: Each grouping must have either:
- `preset_name`: Predefined keyword enum value (TRAY_1, TRAY_2, TRAY_3, TRAY_4, MESH)
- `custom_name`: Custom string name (e.g., "pods", "meshes", "superpods")

**Instances List**: Each grouping contains an `instances` list where each instance must have:
- A unique `id` field within the grouping (used for connections)
- Either an ASIC location or a grouping reference:
  - An ASIC location: `{ id: 0 asic_location: ASIC_LOCATION_1 }`
  - A grouping reference: `{ id: 0 grouping_ref { custom_name: "hosts" } }` or `{ id: 0 grouping_ref { preset_name: TRAY_1 } }`

**Connections**: Separate `connections` section (distinct from `instances`) defines how instances connect and their topology:
- `all_to_all`: Every instance connects to every other instance
- `row_major_mesh`: Instances arranged in a grid with mesh connectivity (defines topology with dims and dim_types)
- `custom`: Explicit connections between specific instances (references instance IDs)

**Important**: Topology (mesh dimensions, dimension types) is defined in the `connections` section, not on individual instances.

**Important**:
- Each instance must have a unique `id` within its grouping
- Instances are explicitly instantiated - you must list each instance separately with its ID
- Connections reference instance IDs (e.g., `src_instance: 0` refers to the instance with `id: 0`)

## Validation Rules

1. **Required Groupings**: A grouping with `custom_name == "meshes"` MUST exist
2. **Multiple Definitions**: The same grouping name can appear multiple times (explicitly allowed)
3. **Grouping References**: All `grouping_name` values must reference an existing grouping
4. **Instance Count Validation**:
   - If `custom_name == "meshes"`: At least 1 instance required
   - Otherwise: At least 2 instances required
5. **Instance ID Validation**: Each instance must have a unique `id` within its grouping
6. **Grouping Structure**: Each instance in `instances` must be either `asic_location` or `grouping_ref` (enforced by oneof in schema)
7. **Connection Validation**:
   - `all_to_all`: Requires at least 2 instances
   - `row_major_mesh`: `dims` product must equal number of instances
   - `custom`: `src_instance` and `dst_instance` must reference valid instance IDs (must exist in the instances list)

## Complete Example

```protobuf
# Physical Groupings File for 3-Pod 16x8 Blackhole Galaxy Cluster

groupings {
  custom_name: "trays"
  instances: [
    { id: 0 asic_location: ASIC_LOCATION_1 },
    { id: 1 asic_location: ASIC_LOCATION_2 },
    { id: 2 asic_location: ASIC_LOCATION_3 },
    { id: 3 asic_location: ASIC_LOCATION_4 },
    { id: 4 asic_location: ASIC_LOCATION_5 },
    { id: 5 asic_location: ASIC_LOCATION_6 },
    { id: 6 asic_location: ASIC_LOCATION_7 },
    { id: 7 asic_location: ASIC_LOCATION_8 }
  ]
}

groupings {
  custom_name: "hosts"
  instances: [
    { id: 0 grouping_ref { custom_name: "trays" } }
  ]
}

groupings {
  custom_name: "meshes"
  instances: [
    { id: 0 grouping_ref { custom_name: "hosts" } },  # Each mesh = 1 host
    { id: 1 grouping_ref { custom_name: "hosts" } },
    { id: 2 grouping_ref { custom_name: "hosts" } },
    { id: 3 grouping_ref { custom_name: "hosts" } },
    { id: 4 grouping_ref { custom_name: "hosts" } },
    { id: 5 grouping_ref { custom_name: "hosts" } },
    { id: 6 grouping_ref { custom_name: "hosts" } },
    { id: 7 grouping_ref { custom_name: "hosts" } }
  ]
  connections {
    row_major_mesh {
      dims: [2, 4]
      dim_types: [LINE, RING]
      num_connections: 2
    }
  }
}

groupings {
  custom_name: "pods"
  instances: [
    { id: 0 grouping_ref { custom_name: "meshes" } },  # Each pod contains 2 meshes
    { id: 1 grouping_ref { custom_name: "meshes" } }
  ]
  connections {
    all_to_all { num_connections: 2 }
  }
}

groupings {
  custom_name: "superpods"
  instances: [
    { id: 0 grouping_ref { custom_name: "pods" } },  # Each superpod contains 3 pods
    { id: 1 grouping_ref { custom_name: "pods" } },
    { id: 2 grouping_ref { custom_name: "pods" } }
  ]
  connections {
    row_major_mesh {
      dims: [3, 1]
      dim_types: [LINE, LINE]
      num_connections: 2
    }
  }
}

groupings {
  custom_name: "clusters"
  instances: [
    { id: 0 grouping_ref { custom_name: "superpods" } },  # Each cluster contains 2 superpods
    { id: 1 grouping_ref { custom_name: "superpods" } }
  ]
  connections {
    custom { src_instance: 0 dst_instance: 1 num_connections: 2 }
  }
}
```

## Examples

### Example 1: Basic 3-Pod Configuration

**File**: `examples/example_basic_3_pod.textproto`

Standard configuration where each pod contains 2 meshes, each superpod contains 3 pods, and the cluster contains 2 superpods.

```protobuf
groupings {
  custom_name: "meshes"
  instances: [
    { id: 0 grouping_ref { custom_name: "hosts" } }  # Each mesh = 1 host (meshes can have 1 instance)
  ]
}

groupings {
  custom_name: "pods"
  instances: [
    { id: 0 grouping_ref { custom_name: "meshes" } },  # Pods must have at least 2 instances
    { id: 1 grouping_ref { custom_name: "meshes" } }
  ]
  connections {
    all_to_all { num_connections: 2 }
  }
}

groupings {
  custom_name: "superpods"
  instances: [
    { id: 0 grouping_ref { custom_name: "pods" } },  # Superpods must have at least 2 instances
    { id: 1 grouping_ref { custom_name: "pods" } },
    { id: 2 grouping_ref { custom_name: "pods" } }
  ]
  connections {
    row_major_mesh {
      dims: [3, 1]
      dim_types: [LINE, LINE]
      num_connections: 2
    }
  }
}

groupings {
  custom_name: "clusters"
  instances: [
    { id: 0 grouping_ref { custom_name: "superpods" } },  # Clusters must have at least 2 instances
    { id: 1 grouping_ref { custom_name: "superpods" } }
  ]
  connections {
    custom { src_instance: 0 dst_instance: 1 num_connections: 2 }
  }
}
```

### Example 2: Pod with Multiple Meshes

**File**: `examples/example_pod_multiple_meshes.textproto`

Shows pods containing 2 meshes each instead of 1, allowing for larger pod configurations.

```protobuf
groupings {
  custom_name: "meshes"
  instances: [
    { id: 0 grouping_ref { custom_name: "hosts" } }  # Each mesh = 1 host (meshes can have 1 instance)
  ]
}

groupings {
  custom_name: "pods"
  instances: [
    { id: 0 grouping_ref { custom_name: "meshes" } },  # Each pod contains 2 meshes (must have at least 2)
    { id: 1 grouping_ref { custom_name: "meshes" } }
  ]
  connections {
    all_to_all { num_connections: 2 }
  }
}
```

### Example 3: Superpod Containing Meshes Directly

**File**: `examples/example_superpod_direct_meshes.textproto`

Demonstrates superpods referencing meshes directly while pods are still defined separately.

```protobuf
groupings {
  custom_name: "meshes"
  instances: [
    { id: 0 grouping_ref { custom_name: "hosts" } }  # Each mesh = 1 host (meshes can have 1 instance)
  ]
}

groupings {
  custom_name: "pods"
  instances: [
    { id: 0 grouping_ref { custom_name: "meshes" } },  # Pods must have at least 2 instances
    { id: 1 grouping_ref { custom_name: "meshes" } }
  ]
  connections {
    all_to_all { num_connections: 2 }
  }
}

groupings {
  custom_name: "superpods"
  instances: [
    { id: 0 grouping_ref { custom_name: "meshes" } },  # Each superpod contains 3 meshes directly (must have at least 2)
    { id: 1 grouping_ref { custom_name: "meshes" } },
    { id: 2 grouping_ref { custom_name: "meshes" } }
  ]
  connections {
    row_major_mesh {
      dims: [3, 1]
      dim_types: [LINE, LINE]
      num_connections: 2
    }
  }
}
```

### Example 4: Cluster Containing Pods Directly

**File**: `examples/example_cluster_direct_pods.textproto`

Shows clusters referencing pods directly while superpods are still defined separately.

```protobuf
groupings {
  custom_name: "meshes"
  instances: [
    { id: 0 grouping_ref { custom_name: "hosts" } }  # Each mesh = 1 host
  ]
}

groupings {
  custom_name: "pods"
  instances: [
    { id: 0 grouping_ref { custom_name: "meshes" } },
    { id: 1 grouping_ref { custom_name: "meshes" } }
  ]
  connections {
    all_to_all { num_connections: 2 }
  }
}

groupings {
  custom_name: "clusters"
  instances: [
    { id: 0 grouping_ref { custom_name: "pods" } },  # Cluster contains 6 pods directly
    { id: 1 grouping_ref { custom_name: "pods" } },
    { id: 2 grouping_ref { custom_name: "pods" } },
    { id: 3 grouping_ref { custom_name: "pods" } },
    { id: 4 grouping_ref { custom_name: "pods" } },
    { id: 5 grouping_ref { custom_name: "pods" } }
  ]
  connections {
    row_major_mesh {
      dims: [2, 3]
      dim_types: [LINE, LINE]
      num_connections: 2
    }
  }
}
```

### Example 5: Mixed Counts

**File**: `examples/example_mixed_counts.textproto`

Demonstrates mixing different grouping types within a single grouping, such as a superpod containing both pods and meshes.

```protobuf
groupings {
  custom_name: "meshes"
  instances: [
    { id: 0 grouping_ref { custom_name: "hosts" } }  # Each mesh = 1 host (meshes can have 1 instance)
  ]
}

groupings {
  custom_name: "pods"
  instances: [
    { id: 0 grouping_ref { custom_name: "meshes" } },  # Pods must have at least 2 instances
    { id: 1 grouping_ref { custom_name: "meshes" } }
  ]
  connections {
    all_to_all { num_connections: 2 }
  }
}

groupings {
  custom_name: "superpods"
  instances: [
    { id: 0 grouping_ref { custom_name: "pods" } },  # Superpods must have at least 2 instances
    { id: 1 grouping_ref { custom_name: "pods" } },
    { id: 2 grouping_ref { custom_name: "meshes" } },  # Also contains 4 meshes directly
    { id: 3 grouping_ref { custom_name: "meshes" } },
    { id: 4 grouping_ref { custom_name: "meshes" } },
    { id: 5 grouping_ref { custom_name: "meshes" } }
  ]
  connections {
    all_to_all { num_connections: 2 }
  }
}

groupings {
  custom_name: "clusters"
  instances: [
    { id: 0 grouping_ref { custom_name: "superpods" } },  # Clusters must have at least 2 instances
    { id: 1 grouping_ref { custom_name: "superpods" } },
    { id: 2 grouping_ref { custom_name: "pods" } },  # Also contains 2 pods directly
    { id: 3 grouping_ref { custom_name: "pods" } },
    { id: 4 grouping_ref { custom_name: "meshes" } },  # Also contains 3 meshes directly
    { id: 5 grouping_ref { custom_name: "meshes" } },
    { id: 6 grouping_ref { custom_name: "meshes" } }
  ]
  connections {
    row_major_mesh {
      dims: [7, 1]
      dim_types: [LINE, LINE]
      num_connections: 2
    }
  }
}
```

### Example 6: Mesh Defined by Trays

**File**: `examples/example_mesh_by_trays.textproto`

Shows meshes defined using tray count instead of host count, providing more granular control over mesh size.

```protobuf
groupings {
  custom_name: "meshes"
  instances: [
    { id: 0 grouping_ref { custom_name: "trays" } },
    { id: 1 grouping_ref { custom_name: "trays" } },
    { id: 2 grouping_ref { custom_name: "trays" } },
    { id: 3 grouping_ref { custom_name: "trays" } },
    { id: 4 grouping_ref { custom_name: "trays" } },
    { id: 5 grouping_ref { custom_name: "trays" } },
    { id: 6 grouping_ref { custom_name: "trays" } },
    { id: 7 grouping_ref { custom_name: "trays" } },
    { id: 8 grouping_ref { custom_name: "trays" } },
    { id: 9 grouping_ref { custom_name: "trays" } },
    { id: 10 grouping_ref { custom_name: "trays" } },
    { id: 11 grouping_ref { custom_name: "trays" } },
    { id: 12 grouping_ref { custom_name: "trays" } },
    { id: 13 grouping_ref { custom_name: "trays" } },
    { id: 14 grouping_ref { custom_name: "trays" } },
    { id: 15 grouping_ref { custom_name: "trays" } }
  ]
  connections {
    row_major_mesh {
      dims: [4, 4]
      dim_types: [LINE, RING]
      num_connections: 2
    }
  }
}
```

### Example 7: Mesh Defined by ASIC Locations

**File**: `examples/example_mesh_by_asic_locations.textproto`

Shows meshes defined directly at the ASIC location level using enum constants.

```protobuf
groupings {
  custom_name: "meshes"
  instances: [
    { id: 0 asic_location: ASIC_LOCATION_1 }  # Each mesh uses 1 ASIC location (smaller mesh example)
  ]
}
```

### Example 8: Custom Groupings - Half Tray

**File**: `examples/example_mesh_subdivided_asic_locations.textproto`

Demonstrates defining custom groupings (halftray) to represent subsets of ASIC locations, then using them in meshes.

```protobuf
# Custom grouping: Half tray - defined twice with different ASIC locations
# Both represent a "half tray" concept
groupings {
  custom_name: "halftray"
  instances: [
    { asic_location: ASIC_LOCATION_1 },
    { asic_location: ASIC_LOCATION_2 },
    { asic_location: ASIC_LOCATION_3 },
    { asic_location: ASIC_LOCATION_4 }
  ]  # Lower half (locations 1-4)
}

groupings {
  custom_name: "halftray"
  instances: [
    { asic_location: ASIC_LOCATION_5 },
    { asic_location: ASIC_LOCATION_6 },
    { asic_location: ASIC_LOCATION_7 },
    { asic_location: ASIC_LOCATION_8 }
  ]  # Upper half (locations 5-8)
}

groupings {
  custom_name: "meshes"
  instances: [
    { id: 0 grouping_ref { custom_name: "halftray" } }  # Each mesh uses 1 half tray (can be either lower or upper)
  ]
}
```

### Example 9: Complex Multi-Level Mixing

**File**: `examples/example_complex_multi_level.textproto`

Shows complex configurations where groupings contain multiple types of lower-level groupings at different levels.

```protobuf
groupings {
  custom_name: "meshes"
  instances: [
    { id: 0 grouping_ref { custom_name: "hosts" } }  # Each mesh = 1 host (meshes can have 1 instance)
  ]
}

groupings {
  custom_name: "pods"
  instances: [
    { id: 0 grouping_ref { custom_name: "meshes" } },  # Must have at least 2 instances
    { id: 1 grouping_ref { custom_name: "meshes" } }
  ]
  connections {
    all_to_all { num_connections: 2 }
  }
}

groupings {
  custom_name: "superpods"
  instances: [
    { id: 0 grouping_ref { custom_name: "pods" } },  # Must have at least 2 instances
    { id: 1 grouping_ref { custom_name: "pods" } },
    { id: 2 grouping_ref { custom_name: "meshes" } },  # Also contains 2 meshes directly
    { id: 3 grouping_ref { custom_name: "meshes" } }
  ]
  connections {
    all_to_all { num_connections: 2 }
  }
}

groupings {
  custom_name: "clusters"
  instances: [
    { id: 0 grouping_ref { custom_name: "superpods" } },  # Must have at least 2 instances
    { id: 1 grouping_ref { custom_name: "superpods" } },
    { id: 2 grouping_ref { custom_name: "pods" } },  # Also contains 2 pods directly
    { id: 3 grouping_ref { custom_name: "pods" } },
    { id: 4 grouping_ref { custom_name: "meshes" } },  # Also contains 3 meshes directly
    { id: 5 grouping_ref { custom_name: "meshes" } },
    { id: 6 grouping_ref { custom_name: "meshes" } }
  ]
  connections {
    row_major_mesh {
      dims: [7, 1]
      dim_types: [LINE, LINE]
      num_connections: 2
    }
  }
}
```

## Key Benefits

- ✅ **Hardware-agnostic**: No ASIC IDs in the file - completely reusable across different clusters
- ✅ **Declarative**: Define structure using counts and ASIC locations, not explicit IDs
- ✅ **Flexible**: Can mix and match grouping types, reference multiple levels
- ✅ **Maintainable**: Physical structure (trays, hosts) is more stable than ASIC IDs
- ✅ **Schema-validated**: Protobuf schema enforces structure and type safety
- ✅ **Type-safe**: ASIC locations are enum constants, preventing typos

## File Location

Physical groupings files are located in: `tests/tt_metal/tt_fabric/physical_groupings/`

- **`3_pod_16x8_bh_galaxy_physical_groupings.textproto`**: Main configuration file for 3-pod 16x8 Blackhole Galaxy cluster
- **`triple_16x8_quad_bh_galaxy_physical_groupings.textproto`**: Blitz pipelined 48-stage configuration
- **`examples/`**: Directory containing example files demonstrating different grouping patterns

## Schema Definition

The protobuf schema is defined in: `tt_metal/fabric/protobuf/physical_groupings.proto`

This schema provides:
- Type definitions for all grouping structures
- Validation rules encoded in the schema structure
- Enum constants for ASIC locations (ASIC_LOCATION_1 through ASIC_LOCATION_8)
