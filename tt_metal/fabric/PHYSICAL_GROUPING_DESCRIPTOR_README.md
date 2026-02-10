# Physical Grouping Descriptor — Quick Use Guide

This guide explains how to define and load a Physical Grouping Descriptor for TT‑Fabric using the schema in [`tt_metal/fabric/protobuf/physical_grouping_descriptor.proto`](protobuf/physical_grouping_descriptor.proto). It focuses on how to write a valid textproto, how grouping references work, and which fields are required.

A Physical Grouping Descriptor specifies the hierarchical structure of physical resources (meshes, pods, superpods, clusters) in a declarative way without requiring explicit ASIC IDs. The actual ASIC IDs are derived at runtime from the Physical System Descriptor (PSD).

---

### Where to look
- Schema: [`tt_metal/fabric/protobuf/physical_grouping_descriptor.proto`](protobuf/physical_grouping_descriptor.proto)
- Example textproto: [`tests/tt_metal/tt_fabric/physical_groupings/3_pod_16x8_bh_galaxy_physical_groupings.textproto`](../../tests/tt_metal/tt_fabric/physical_groupings/3_pod_16x8_bh_galaxy_physical_groupings.textproto)
- C++ API: [`tt_metal/api/tt-metalium/experimental/fabric/physical_grouping_descriptor.hpp`](../../api/tt-metalium/experimental/fabric/physical_grouping_descriptor.hpp)
- Tests: [`tests/tt_metal/tt_fabric/fabric_router/test_physical_grouping_descriptor.cpp`](../../tests/tt_metal/tt_fabric/fabric_router/test_physical_grouping_descriptor.cpp)

## Overview

In replacement of the rankfile and rank_bindings file as needed for running tt-run on multi-host systems, we will be introducing a cluster physical groupings file deployed with each machine cluster specifying the valid physical groupings for each cluster of machines. This file is provided by the cluster administrator and is used by FM to understand which subsets of ASICs can be used as candidate physical meshes for a given logical mesh in the MGD.

The Physical Grouping Descriptor defines the hierarchical structure of physical resources (meshes, pods, superpods, clusters) in the cluster. This file uses a **declarative approach** that defines groupings in terms of ASIC locations and other groupings without requiring explicit ASIC IDs. The actual ASIC IDs are derived at runtime from the Physical System Descriptor (PSD).

The groupings file is complementary to the Physical System Descriptor (PSD):
- **PSD**: Flat graph of all ASICs + links
- **Groupings**: Allowed carve-outs (meshes/pods/superpods/clusters) over that flat graph

Files use **protobuf text format** (`.textproto`) with schema validation. The schema enforces validation rules and ensures type safety.

## Quick Example

```protobuf
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
- **Type Safety**: Protobuf enforces that each grouping item is either an ASIC location or a grouping reference
- **Required Groupings**: The "meshes" grouping must be defined (enforced by validation)
- **Multiple Definitions**: The same grouping name can be defined multiple times (useful for custom groupings)

See [`tt_metal/fabric/protobuf/physical_grouping_descriptor.proto`](protobuf/physical_grouping_descriptor.proto) for the complete schema definition.

## Groupings Explained

### Physical Groupings

**ASIC Locations**: Predefined constants (`ASIC_LOCATION_1` through `ASIC_LOCATION_8`) representing individual ASIC positions within a tray. These are defined as enum values in the protobuf schema and are always available.

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

## MGD to PGD Mapping Algorithm

The `PhysicalGroupingDescriptor` class provides a method `get_valid_groupings_for_mgd()` that intelligently matches logical mesh requirements from a Mesh Graph Descriptor (MGD) to physical groupings defined in the Physical Grouping Descriptor (PGD). This mapping uses a **composition-based matching algorithm** that does not rely on name matching, making it flexible and reusable.

### Overview

The mapping algorithm works in three phases:

1. **Phase 0: Mesh Matching** - Matches mesh instances to "meshes" groupings based on chip count
2. **Phase 1: Composition Analysis** - Analyzes MGD graph structure to determine composition requirements
3. **Phase 2: Composition-Based Matching** - Matches groupings by composition (not by name)

### Key Principles

- **Name Independence**: MGD graph types (e.g., "POD", "CLUSTER") do not need to match PGD grouping names (e.g., "pods", "clusters"). Matching is based purely on composition.
- **Composition-Based**: Groupings are matched based on what they contain (e.g., `{meshes: 2, pods: 1}`), not their names.
- **Bottom-Up Processing**: Lower-level groupings (meshes) are matched first, then higher-level groupings are matched based on their composition.
- **Best Match Selection**: The algorithm prefers exact matches, then selects the closest oversized match (minimal waste) if no exact match exists.
- **ASIC Count Validation**: After matching, validates that each matched grouping has sufficient ASIC count for all its components.

### Algorithm Details

#### Phase 0: Mesh Matching

For each unique mesh type in the MGD:
1. Calculate the required chip count from the mesh's `device_topology`
2. Find the best matching "meshes" grouping that has at least the required number of ASICs
3. Validate that the matched grouping has sufficient capacity

**Example:**
```cpp
// MGD defines a mesh "M0" with device_topology [16, 8] = 128 chips
// PGD has meshes groupings: one with 32 ASICs, one with 128 ASICs
// Result: M0 matches the 128 ASIC meshes grouping (exact match)
```

#### Phase 1: Composition Analysis

For each graph instance in the MGD:
1. Count direct mesh children (e.g., 2 meshes)
2. Count direct graph children by type (e.g., 1 POD, 2 CLUSTER)
3. Convert graph types to grouping names (e.g., "POD" → "pods", "CLUSTER" → "clusters")
4. Build composition map: `{meshes: 2, pods: 1, clusters: 2}`
5. Group graph instances by their unique composition patterns

**Example:**
```cpp
// MGD has a CLUSTER graph instance containing:
//   - 2 meshes (direct children)
//   - 1 POD (direct child)
// Composition requirement: {meshes: 2, pods: 1}
```

#### Phase 2: Composition-Based Matching

For each unique composition requirement:
1. Search **all** available groupings (regardless of name)
2. For each candidate grouping:
   - Analyze its composition (what lower-level groupings it contains)
   - Check if it satisfies the requirement: `actual_count >= required_count` for all types
3. Select best match:
   - **Priority 1**: Exact match (all counts match exactly AND no extra types)
   - **Priority 2**: Closest oversized match (minimal total waste)
4. Assign matched grouping to all graph instances with that composition

**Example:**
```cpp
// Composition requirement: {meshes: 2, pods: 1}
// Candidate 1: "pods" grouping with {meshes: 2} → Doesn't satisfy (missing pods: 1)
// Candidate 2: "clusters" grouping with {meshes: 2, pods: 1} → Exact match! ✓
// Candidate 3: "clusters" grouping with {meshes: 3, pods: 2} → Oversized but valid
// Result: Candidate 2 is selected (exact match)
```

### Matching Examples

#### Example 1: Simple Mesh Matching

**MGD:**
```protobuf
mesh_descriptors {
  name: "M0"
  device_topology { dims: [ 4, 4 ] }  # 16 chips
}
```

**PGD:**
```protobuf
groupings {
  name: "meshes"
  items: [
    { grouping_ref { grouping_name: "trays" count: 2 } }  # 16 ASICs
  ]
}
```

**Result:** M0 matches the "meshes" grouping with 16 ASICs (exact match).

#### Example 2: POD Composition Matching

**MGD:**
```protobuf
graph_descriptors {
  name: "P0"
  type: "POD"
  instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
  instances { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
}
```

**PGD:**
```protobuf
groupings {
  name: "pods"
  items: [
    { grouping_ref { grouping_name: "meshes" count: 2 } }  # Composition: {meshes: 2}
  ]
}
```

**Result:** P0 (requiring `{meshes: 2}`) matches the "pods" grouping. Note: The MGD type "POD" doesn't need to match the PGD name "pods" - matching is based on composition.

#### Example 3: Multiple Graph Instances, Same Composition

**MGD:**
```protobuf
graph_descriptors {
  name: "C0"
  type: "CLUSTER"
  instances { graph { graph_descriptor: "P0" graph_id: 0 } }
  instances { graph { graph_descriptor: "P0" graph_id: 1 } }
}
graph_descriptors {
  name: "C1"
  type: "CLUSTER"
  instances { graph { graph_descriptor: "P0" graph_id: 2 } }
  instances { graph { graph_descriptor: "P0" graph_id: 3 } }
}
```

**PGD:**
```protobuf
groupings {
  name: "clusters"
  items: [
    { grouping_ref { grouping_name: "pods" count: 2 } }  # Composition: {pods: 2}
  ]
}
```

**Result:** Both C0 and C1 (each requiring `{pods: 2}`) map to the same "clusters" grouping. This demonstrates that groupings are reusable templates - multiple MGD instances can map to the same physical grouping.

#### Example 4: Oversized Match (Closest Fit)

**MGD:**
```protobuf
graph_descriptors {
  name: "SP0"
  type: "SUPERPOD"
  instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
  instances { graph { graph_descriptor: "P0" graph_id: 0 } }
  instances { graph { graph_descriptor: "P0" graph_id: 1 } }
}
# Composition requirement: {meshes: 1, pods: 2}
```

**PGD:**
```protobuf
groupings {
  name: "superpods"
  items: [
    { grouping_ref { grouping_name: "meshes" count: 2 } }
    { grouping_ref { grouping_name: "pods" count: 3 } }
  ]
}
# Composition: {meshes: 2, pods: 3}
```

**Result:** SP0 matches the "superpods" grouping even though it's oversized (`{meshes: 2, pods: 3}` vs required `{meshes: 1, pods: 2}`). This is acceptable because it satisfies all requirements and is the closest fit available.

#### Example 5: Mixed Composition

**MGD:**
```protobuf
graph_descriptors {
  name: "SP0"
  type: "SUPERPOD"
  instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
  instances { graph { graph_descriptor: "P0" graph_id: 0 } }
}
# Composition requirement: {meshes: 1, pods: 1}
```

**PGD:**
```protobuf
groupings {
  name: "superpods"
  items: [
    { grouping_ref { grouping_name: "meshes" count: 1 } }
    { grouping_ref { grouping_name: "pods" count: 1 } }
  ]
}
# Composition: {meshes: 1, pods: 1}
```

**Result:** Exact match! SP0 matches the "superpods" grouping with composition `{meshes: 1, pods: 1}`.

### ASIC Count Validation

After all matching is complete, the algorithm validates that each matched grouping has sufficient ASIC count:

1. For meshes: Validates that `matched_grouping.asic_count >= required_chips`
2. For higher-level groupings: Calculates total ASICs required from matched sub-instances and validates that the grouping can accommodate them

**Validation Example:**
```cpp
// POD requires 2 meshes, each with 128 ASICs
// Total required: 2 * 128 = 256 ASICs
// Matched "pods" grouping has asic_count: 256
// Result: Validation passes ✓

// If matched grouping had only 128 ASICs:
// Result: Validation fails with error:
//   "This system is not compatible with the following MGD:
//    Graph instance 'P0' (type 'POD') requires 256 ASICs total
//    from its components, but the matched grouping has only 128 ASICs"
```

### Auto-Creation of Missing Groupings

The `PhysicalGroupingDescriptor` class also provides `auto_create_missing_groupings()` method that can automatically create missing groupings based on MGD requirements:

- **For meshes**: Creates mesh groupings from ASIC locations or trays when no matching grouping exists
- **For higher-level groupings**: Composes groupings from existing lower-level ones
- **Warnings**: Logs warnings when auto-creating groupings, indicating they're not in the groupings file

See the API documentation for details on auto-creation functionality.

### Usage Example

```cpp
#include <tt-metalium/experimental/fabric/physical_grouping_descriptor.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp>

// Load PGD
PhysicalGroupingDescriptor pgd("path/to/physical_groupings.textproto");

// Load MGD
MeshGraphDescriptor mgd("path/to/mesh_graph_descriptor.textproto");

// Get valid groupings for MGD instances
auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd);

// Access matched groupings
for (const auto& [instance_type, instances] : valid_groupings) {
    for (const auto& [instance_name, grouping] : instances) {
        std::cout << "Instance " << instance_name
                  << " (type: " << instance_type << ")"
                  << " matches grouping: " << grouping.name
                  << " with " << grouping.asic_count << " ASICs\n";
    }
}
```

### Test Examples

Comprehensive test examples demonstrating the mapping algorithm can be found in:
- [`tests/tt_metal/tt_fabric/fabric_router/test_physical_grouping_descriptor.cpp`](../../tests/tt_metal/tt_fabric/fabric_router/test_physical_grouping_descriptor.cpp)

Key test cases include:
- `GetValidGroupingsForMGD_TriplePod16x8` - Tests matching a POD with 2 meshes
- `GetValidGroupingsForMGD_ClusterWith2Pods_ExactMatch` - Tests composition-based matching with multiple instances
- `GetValidGroupingsForMGD_SuperpodOversizedMatch_ClosestFit` - Tests closest fit selection
- `GetValidGroupingsForMGD_MixedComposition_OversizedMatch` - Tests mixed composition matching

### Benefits of Composition-Based Matching

- ✅ **Flexible**: MGD types don't need to match PGD names
- ✅ **Reusable**: Multiple MGD instances can map to the same physical grouping
- ✅ **Intelligent**: Automatically finds best matches, handles oversized groupings
- ✅ **Validated**: Ensures sufficient ASIC capacity for all requirements
- ✅ **Maintainable**: Changes to MGD structure don't require PGD name changes

## Examples

See [`tests/tt_metal/tt_fabric/physical_groupings/examples/`](../../tests/tt_metal/tt_fabric/physical_groupings/examples/) for additional example files demonstrating different grouping patterns.

## Key Benefits

- ✅ **Hardware-agnostic**: No ASIC IDs in the file - completely reusable across different clusters
- ✅ **Declarative**: Define structure using counts and ASIC locations, not explicit IDs
- ✅ **Flexible**: Can mix and match grouping types, reference multiple levels
- ✅ **Maintainable**: Physical structure (trays, hosts) is more stable than ASIC IDs
- ✅ **Schema-validated**: Protobuf schema enforces structure and type safety
- ✅ **Type-safe**: ASIC locations are enum constants, preventing typos

## File Location

Physical grouping descriptor files are located in: `tests/tt_metal/tt_fabric/physical_groupings/`

- **`3_pod_16x8_bh_galaxy_physical_groupings.textproto`**: Main configuration file for 3-pod 16x8 Blackhole Galaxy cluster
- **`triple_16x8_quad_bh_galaxy_physical_groupings.textproto`**: Blitz pipelined 48-stage configuration
- **`examples/`**: Directory containing example files demonstrating different grouping patterns

## Schema Definition

The protobuf schema is defined in: [`tt_metal/fabric/protobuf/physical_grouping_descriptor.proto`](protobuf/physical_grouping_descriptor.proto)

This schema provides:
- Type definitions for all grouping structures
- Validation rules encoded in the schema structure
- Enum constants for ASIC locations (ASIC_LOCATION_1 through ASIC_LOCATION_8)
