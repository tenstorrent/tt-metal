# Physical Grouping Descriptor — Quick Use Guide

This guide explains how to define and load a Physical Grouping Descriptor for TT‑Fabric using the schema in [`tt_metal/fabric/protobuf/physical_grouping_descriptor.proto`](protobuf/physical_grouping_descriptor.proto). It focuses on how to write a valid textproto, how grouping references work, and which fields are required.

A Physical Grouping Descriptor specifies the hierarchical structure of physical resources (meshes, pods, superpods, clusters) in a declarative way without requiring explicit ASIC IDs. The actual ASIC IDs are derived at runtime from the Physical System Descriptor (PSD).

---

### Where to look
- Schema: [`tt_metal/fabric/protobuf/physical_grouping_descriptor.proto`](protobuf/physical_grouping_descriptor.proto)
- Example textproto: [`tests/tt_metal/tt_fabric/physical_groupings/3_pod_16x8_bh_galaxy_physical_groupings.textproto`](../../tests/tt_metal/tt_fabric/physical_groupings/3_pod_16x8_bh_galaxy_physical_groupings.textproto)
- C++ API: [`tt_metal/api/tt-metalium/experimental/fabric/physical_grouping_descriptor.hpp`](../../api/tt-metalium/experimental/fabric/physical_grouping_descriptor.hpp)

## Overview

In replacement of the rankfile and rank_bindings file as needed for running tt-run on multi-host systems, we will be introducing a cluster physical groupings file deployed with each machine cluster specifying the valid physical groupings for each cluster of machines. This file is provided by the cluster administrator and is used by FM to understand which subsets of ASICs can be used as candidate physical meshes for a given logical mesh in the MGD.

The Physical Grouping Descriptor defines the hierarchical structure of physical resources (meshes, pods, superpods, clusters) in the cluster. This file uses a **declarative approach** that defines groupings in terms of base units without requiring explicit ASIC IDs. The actual ASIC IDs are derived at runtime from the Physical System Descriptor (PSD).

The groupings file is complementary to the Physical System Descriptor (PSD):
- **PSD**: Flat graph of all ASICs + links
- **Groupings**: Allowed carve-outs (meshes/pods/superpods/clusters) over that flat graph

Files use **protobuf text format** (`.textproto`) with schema validation. The schema enforces validation rules and ensures type safety.

## Quick Example

```protobuf
base_units {
  # Base units are predefined constants - ASIC_LOCATION_1 through ASIC_LOCATION_8
}

groupings {
  name: "trays"
  items: [
    { asic_location: ASIC_LOCATION_1 },
    { asic_location: ASIC_LOCATION_2 },
    # ... through ASIC_LOCATION_8
  ]
}

groupings {
  name: "hosts"
  items: [
    { grouping_ref { grouping_name: "trays" count: 4 } }
  ]
}

groupings {
  name: "meshes"
  items: [
    { grouping_ref { grouping_name: "hosts" count: 1 } }  # Each mesh = 1 host (meshes can have 1 item)
  ]
}

groupings {
  name: "pods"
  items: [
    { grouping_ref { grouping_name: "meshes" count: 2 } }  # Each pod contains 2 meshes (must have at least 2)
  ]
}

groupings {
  name: "superpods"
  items: [
    { grouping_ref { grouping_name: "pods" count: 3 } }  # Each superpod contains 3 pods (must have at least 2)
  ]
}

groupings {
  name: "clusters"
  items: [
    { grouping_ref { grouping_name: "superpods" count: 2 } }  # Each cluster contains 2 superpods (must have at least 2)
  ]
}
```

The actual ASIC IDs are derived at runtime from the PSD, making this file completely hardware-agnostic and reusable across different clusters.

## Schema and Validation

The physical grouping descriptor uses protobuf text format with schema validation. Key features:

- **ASIC Locations as Constants**: ASIC locations 1-8 are predefined as enum constants (`ASIC_LOCATION_1` through `ASIC_LOCATION_8`)
- **Base Units**: Base units are predefined in the schema, not defined as a grouping
- **Type Safety**: Protobuf enforces that each grouping item is either an ASIC location or a grouping reference
- **Required Groupings**: The "meshes" grouping must be defined (enforced by validation)
- **Multiple Definitions**: The same grouping name can be defined multiple times (useful for custom groupings)

See [`tt_metal/fabric/protobuf/physical_grouping_descriptor.proto`](protobuf/physical_grouping_descriptor.proto) for the complete schema definition.

## Groupings Explained

### Base Units

**ASIC Locations**: Predefined constants (`ASIC_LOCATION_1` through `ASIC_LOCATION_8`) representing individual ASIC positions within a tray. These are defined as enum values in the protobuf schema and are always available.

```protobuf
base_units {
  # Base units are predefined constants - no fields needed
  # ASIC_LOCATION_1 through ASIC_LOCATION_8 are available as enum values
}
```

### Physical Groupings

**Trays**: Contains all 8 ASIC locations. Defined using ASIC location enum values.

```protobuf
groupings {
  name: "trays"
  items: [
    { asic_location: ASIC_LOCATION_1 },
    { asic_location: ASIC_LOCATION_2 },
    # ... through ASIC_LOCATION_8
  ]
}
```

**Hosts**: Contains multiple trays. Defined using grouping references.

```protobuf
groupings {
  name: "hosts"
  items: [
    { grouping_ref { grouping_name: "trays" count: 4 } }  # Each host contains 4 trays
  ]
}
```

### Logical Groupings

**Meshes**: The required logical grouping. Can be defined using hosts, trays, or ASIC locations. **Note**: Meshes can have 1 item (e.g., `count: 1`), but all other groupings (pods, superpods, clusters) must have at least 2 items.

```protobuf
groupings {
  name: "meshes"
  items: [
    { grouping_ref { grouping_name: "hosts" count: 1 } }  # Each mesh contains 1 host (meshes can have 1 item)
    # OR { grouping_ref { grouping_name: "hosts" count: 4 } }  # Each mesh contains 4 hosts
    # OR { grouping_ref { grouping_name: "trays" count: 16 } }  # 16 trays per mesh
    # OR { asic_location: ASIC_LOCATION_1 }, { asic_location: ASIC_LOCATION_2 }, ...  # Direct ASIC locations
  ]
}
```

**Subdividing ASIC Locations**: You can use a subset of ASIC locations to create smaller meshes. For example, using only locations 1-4 instead of all 8 creates a mesh with half the ASICs per tray.

```protobuf
groupings {
  name: "meshes"
  items: [
    { asic_location: ASIC_LOCATION_1 },
    { asic_location: ASIC_LOCATION_2 },
    { asic_location: ASIC_LOCATION_3 },
    { asic_location: ASIC_LOCATION_4 }
  ]
  # Only using 4 ASIC locations from each tray instead of all 8
}
```

**Pods**: Contains meshes. Defined using grouping references. **Must have at least 2 items.**

```protobuf
groupings {
  name: "pods"
  items: [
    { grouping_ref { grouping_name: "meshes" count: 2 } }  # Each pod contains 2 meshes
    # OR { grouping_ref { grouping_name: "meshes" count: 3 } }  # Each pod contains 3 meshes
  ]
}
```

**Superpods**: Contains pods and/or meshes. Can mix different grouping types. **Must have at least 2 items.**

```protobuf
groupings {
  name: "superpods"
  items: [
    { grouping_ref { grouping_name: "pods" count: 3 } }  # Each superpod contains 3 pods (must have at least 2)
    # OR { grouping_ref { grouping_name: "meshes" count: 3 } }  # Each superpod contains 3 meshes directly
    # OR both: { grouping_ref { grouping_name: "pods" count: 2 } }, { grouping_ref { grouping_name: "meshes" count: 4 } }  # Mix and match
  ]
}
```

**Clusters**: Contains superpods, pods, and/or meshes. Can mix different grouping types. **Must have at least 2 items.**

```protobuf
groupings {
  name: "clusters"
  items: [
    { grouping_ref { grouping_name: "superpods" count: 2 } }  # Each cluster contains 2 superpods (must have at least 2)
    # OR { grouping_ref { grouping_name: "pods" count: 6 } }  # Each cluster contains 6 pods directly
    # OR { grouping_ref { grouping_name: "meshes" count: 10 } }  # Each cluster contains 10 meshes directly
    # OR mix: { grouping_ref { grouping_name: "superpods" count: 2 } }, { grouping_ref { grouping_name: "pods" count: 2 } }, { grouping_ref { grouping_name: "meshes" count: 3 } }
  ]
}
```

### Custom Groupings

You can define your own custom groupings using ASIC locations. This is useful for creating reusable sub-units like "half trays" or other logical divisions. You can define the same grouping name multiple times with different ASIC location sets.

```protobuf
groupings {
  # Define a custom grouping called "halftray" - first definition
  name: "halftray"
  items: [
    { asic_location: ASIC_LOCATION_1 },
    { asic_location: ASIC_LOCATION_2 },
    { asic_location: ASIC_LOCATION_3 },
    { asic_location: ASIC_LOCATION_4 }
  ]  # Lower half
}

groupings {
  # Same name, different ASIC locations - second definition
  name: "halftray"
  items: [
    { asic_location: ASIC_LOCATION_5 },
    { asic_location: ASIC_LOCATION_6 },
    { asic_location: ASIC_LOCATION_7 },
    { asic_location: ASIC_LOCATION_8 }
  ]  # Upper half
}

# Then use the custom grouping in meshes or other groupings
groupings {
  name: "meshes"
  items: [
    { grouping_ref { grouping_name: "halftray" count: 1 } }  # Each mesh uses 1 half tray (can be either definition)
  ]
}
```

## Specification Format

**ASIC Locations**: Use enum constants (`ASIC_LOCATION_1` through `ASIC_LOCATION_8`).

**Grouping References**: Use `grouping_ref` with `grouping_name` and `count`:
- `{ grouping_ref { grouping_name: "hosts" count: 4 } }` → Each mesh contains 4 hosts
- `{ grouping_ref { grouping_name: "meshes" count: 2 } }` → Each pod contains 2 meshes

**Items List**: Each grouping contains an `items` list where each item is either:
- An ASIC location: `{ asic_location: ASIC_LOCATION_1 }`
- A grouping reference: `{ grouping_ref { grouping_name: "hosts" count: 4 } }`

**Important**: Do not use numbered lists for instances (e.g., `meshes: [0, 1]` or `pods: [0, 1, 2]`). Only use counts to define the structure.

## Validation Rules

1. **Required Groupings**: A grouping with `name == "meshes"` MUST exist
2. **Multiple Definitions**: The same grouping name can appear multiple times (explicitly allowed)
3. **Grouping References**: All `grouping_name` values must reference an existing grouping
4. **Count Validation**:
   - If `name == "meshes"`: `count >= 1` (meshes can have 1 item)
   - Otherwise: `count >= 2` (all other groupings must have at least 2 items)
5. **Grouping Structure**: Each item in `items` must be either `asic_location` or `grouping_ref` (enforced by oneof in schema)

## Complete Example

```protobuf
# Physical Grouping Descriptor for 3-Pod 16x8 Blackhole Galaxy Cluster

base_units {
  # Base units are predefined constants - ASIC_LOCATION_1 through ASIC_LOCATION_8
}

groupings {
  name: "trays"
  items: [
    { asic_location: ASIC_LOCATION_1 },
    { asic_location: ASIC_LOCATION_2 },
    { asic_location: ASIC_LOCATION_3 },
    { asic_location: ASIC_LOCATION_4 },
    { asic_location: ASIC_LOCATION_5 },
    { asic_location: ASIC_LOCATION_6 },
    { asic_location: ASIC_LOCATION_7 },
    { asic_location: ASIC_LOCATION_8 }
  ]
}

groupings {
  name: "hosts"
  items: [
    { grouping_ref { grouping_name: "trays" count: 4 } }
  ]
}

groupings {
  name: "meshes"
  items: [
    { grouping_ref { grouping_name: "hosts" count: 1 } }  # Each mesh = 1 host
  ]
}

groupings {
  name: "pods"
  items: [
    { grouping_ref { grouping_name: "meshes" count: 2 } }  # Each pod contains 2 meshes
  ]
}

groupings {
  name: "superpods"
  items: [
    { grouping_ref { grouping_name: "pods" count: 3 } }  # Each superpod contains 3 pods
  ]
}

groupings {
  name: "clusters"
  items: [
    { grouping_ref { grouping_name: "superpods" count: 2 } }  # Each cluster contains 2 superpods
  ]
}
```

## Examples

See [`tests/tt_metal/tt_fabric/physical_groupings/examples/`](../../tests/tt_metal/tt_fabric/physical_groupings/examples/) for additional example files demonstrating different grouping patterns.

## Key Benefits

- ✅ **Hardware-agnostic**: No ASIC IDs in the file - completely reusable across different clusters
- ✅ **Declarative**: Define structure using counts and base units, not explicit IDs
- ✅ **Flexible**: Can mix and match grouping types, reference multiple levels
- ✅ **Maintainable**: Physical structure (trays, hosts) is more stable than ASIC IDs
- ✅ **Schema-validated**: Protobuf schema enforces structure and type safety
- ✅ **Type-safe**: ASIC locations are enum constants, preventing typos

## File Location

Physical grouping descriptor files are located in: `tests/tt_metal/tt_fabric/physical_groupings/`

- **`3_pod_16x8_bh_galaxy_physical_groupings.textproto`**: Main configuration file for 3-pod 16x8 Blackhole Galaxy cluster
- **`triple_16x8_quad_bh_galaxy_blitz_48_stage_physical_groupings.textproto`**: Blitz pipelined 48-stage configuration
- **`examples/`**: Directory containing example files demonstrating different grouping patterns

## Schema Definition

The protobuf schema is defined in: [`tt_metal/fabric/protobuf/physical_grouping_descriptor.proto`](protobuf/physical_grouping_descriptor.proto)

This schema provides:
- Type definitions for all grouping structures
- Validation rules encoded in the schema structure
- Enum constants for ASIC locations (ASIC_LOCATION_1 through ASIC_LOCATION_8)
