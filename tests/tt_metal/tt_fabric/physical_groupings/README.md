# Physical Groupings File Specification

## Overview

The Physical Groupings file defines the hierarchical structure of physical resources (meshes, pods, superpods, clusters) in the cluster. This file uses a **declarative approach** that defines groupings in terms of base units without requiring explicit ASIC IDs. The actual ASIC IDs are derived at runtime from the Physical System Descriptor (PSD).

The groupings file is complementary to the Physical System Descriptor (PSD):
- **PSD**: Flat graph of all ASICs + links
- **Groupings**: Allowed carve-outs (meshes/pods/superpods/clusters) over that flat graph

## Quick Example

```yaml
groupings:
  - base_units:
      - asic_location_1
      # ... through asic_location_8

  - trays:
      asic_locations: [asic_location_1, ..., asic_location_8]

  - hosts:
      trays: 4

  - meshes:
      hosts: 1  # Each mesh = 1 host (meshes can have 1 item)

  - pods:
      meshes: 2  # Each pod contains 2 meshes (must have at least 2)

  - superpods:
      pods: 3  # Each superpod contains 3 pods (must have at least 2)

  - clusters:
      superpods: 2  # Each cluster contains 2 superpods (must have at least 2)
```

The actual ASIC IDs are derived at runtime from the PSD, making this file completely hardware-agnostic and reusable across different clusters.

## Groupings Explained

### Base Units

**ASIC Locations** (asic_location_1 through asic_location_8): The fundamental physical units representing individual ASIC positions within a tray.

```yaml
groupings:
  - base_units:
      - asic_location_1
      - asic_location_2
      # ... through asic_location_8
```

### Physical Groupings

**Trays**: Contains all 8 ASIC locations. Defined using base unit names.

```yaml
groupings:
  - trays:
      asic_locations: [asic_location_1, asic_location_2, ..., asic_location_8]
```

**Hosts**: Contains multiple trays. Defined using counts.

```yaml
groupings:
  - hosts:
      trays: 4  # Each host contains 4 trays
```

### Logical Groupings

**Meshes**: The required logical grouping. Can be defined using hosts, trays, or ASIC locations. **Note**: Meshes can have 1 item (e.g., `hosts: 1`), but all other groupings (pods, superpods, clusters) must have at least 2 items.

```yaml
groupings:
  - meshes:
      hosts: 1  # Each mesh contains 1 host (meshes can have 1 item)
      # OR hosts: 4  # Each mesh contains 4 hosts
      # OR trays: 16  # 16 trays per mesh
      # OR asic_locations: [asic_location_1, ..., asic_location_8]  # Base unit names
```

**Subdividing ASIC Locations**: You can use a subset of ASIC locations to create smaller meshes. For example, using only locations 1-4 instead of all 8 creates a mesh with half the ASICs per tray.

```yaml
groupings:
  - meshes:
      asic_locations: [asic_location_1, asic_location_2, asic_location_3, asic_location_4]
      # Only using 4 ASIC locations from each tray instead of all 8
```

**Pods**: Contains meshes. Defined using counts.

```yaml
groupings:
  - pods:
      meshes: 2  # Each pod contains 2 meshes
      # OR meshes: 3  # Each pod contains 3 meshes
```

**Superpods**: Contains pods and/or meshes. Can mix different grouping types. **Must have at least 2 items.**

```yaml
groupings:
  - superpods:
      pods: 3  # Each superpod contains 3 pods (must have at least 2)
      # OR meshes: 3  # Each superpod contains 3 meshes directly
      # OR both: pods: 2, meshes: 4  # Mix and match
```

**Clusters**: Contains superpods, pods, and/or meshes. Can mix different grouping types. **Must have at least 2 items.**

```yaml
groupings:
  - clusters:
      superpods: 2  # Each cluster contains 2 superpods (must have at least 2)
      # OR pods: 6  # Each cluster contains 6 pods directly
      # OR meshes: 10  # Each cluster contains 10 meshes directly
      # OR mix: superpods: 2, pods: 2, meshes: 3
```

### Custom Groupings

You can define your own custom groupings using ASIC locations. This is useful for creating reusable sub-units like "half trays" or other logical divisions. You can define the same grouping name multiple times with different ASIC location sets.

```yaml
groupings:
  # Define a custom grouping called "halftray" - defined twice with different ASIC locations
  # Both represent the same "half tray" concept
  - halftray:
      asic_locations: [asic_location_1, asic_location_2, asic_location_3, asic_location_4]  # Lower half

  - halftray:
      asic_locations: [asic_location_5, asic_location_6, asic_location_7, asic_location_8]  # Upper half

  # Then use the custom grouping in meshes or other groupings
  - meshes:
      halftray: 2  # Each mesh uses 2 half trays (can be either definition)
```

## Specification Format

**Counts**: Use counts to define how many units a grouping contains.

- `hosts: 4` → Each mesh contains 4 hosts
- `meshes: 2` → Each pod contains 2 meshes
- `pods: 3` → Each superpod contains 3 pods

**Base Unit Lists**: For ASIC locations, use the base unit names (not instance IDs).

- `asic_locations: [asic_location_1, asic_location_2, ..., asic_location_8]` → All 8 ASIC locations

**Important**: Do not use numbered lists for instances (e.g., `meshes: [0, 1]` or `pods: [0, 1, 2]`). Only use counts to define the structure.

## File Structure

All groupings are defined in a `groupings:` list at the top level of the file. Each grouping is an item in this list. All groupings must be explicitly defined:

- **base_units** (ASIC locations 1-8) - must be defined
- **trays** - must be defined
- **hosts** - must be defined
- **meshes** - must be defined
- **pods** - must be explicitly defined
- **superpods** - must be explicitly defined
- **clusters** - must be explicitly defined
- **Custom groupings** (like `halftray`) - can be defined as needed

## Complete Example

```yaml
# Physical Groupings File for 3-Pod 16x8 Blackhole Galaxy Cluster

groupings:
  - base_units:
      - asic_location_1
      - asic_location_2
      - asic_location_3
      - asic_location_4
      - asic_location_5
      - asic_location_6
      - asic_location_7
      - asic_location_8

  - trays:
      asic_locations: [asic_location_1, asic_location_2, asic_location_3, asic_location_4, asic_location_5, asic_location_6, asic_location_7, asic_location_8]

  - hosts:
      trays: 4

  - meshes:
      hosts: 4  # Each mesh contains 4 hosts = 16 trays = 128 ASICs (16×8 mesh)

  - pods:
      meshes: 2  # Each pod contains 2 meshes

  - superpods:
      pods: 3  # Each superpod contains 3 pods

  - clusters:
      superpods: 2  # Each cluster contains 2 superpods
```

## Examples

### Example 1: Basic 3-Pod Configuration

**File**: `examples/example_basic_3_pod.yaml`

Standard configuration where each pod contains 2 meshes, each superpod contains 3 pods, and the cluster contains 2 superpods.

```yaml
groupings:
  # ... (base_units, trays, hosts definitions omitted)

  - meshes:
      hosts: 1  # Each mesh = 1 host (meshes can have 1 item)

  - pods:
      meshes: 2  # Pods must have at least 2 items

  - superpods:
      pods: 3  # Superpods must have at least 2 items

  - clusters:
      superpods: 2  # Clusters must have at least 2 items
```

### Example 2: Pod with Multiple Meshes

**File**: `examples/example_pod_multiple_meshes.yaml`

Shows pods containing 2 meshes each instead of 1, allowing for larger pod configurations.

```yaml
groupings:
  # ... (base_units, trays, hosts definitions omitted)

  - meshes:
      hosts: 1  # Each mesh = 1 host (meshes can have 1 item)

  - pods:
      meshes: 2  # Each pod contains 2 meshes (must have at least 2)

  - superpods:
      pods: 2  # Must have at least 2 items
```

### Example 3: Superpod Containing Meshes Directly

**File**: `examples/example_superpod_direct_meshes.yaml`

Demonstrates superpods referencing meshes directly while pods are still defined separately.

```yaml
groupings:
  # ... (base_units, trays, hosts definitions omitted)

  - meshes:
      hosts: 1  # Each mesh = 1 host (meshes can have 1 item)

  - pods:
      meshes: 2  # Pods must have at least 2 items

  - superpods:
      meshes: 3  # Each superpod contains 3 meshes directly (must have at least 2)

  - clusters:
      superpods: 2  # Clusters must have at least 2 items
```

### Example 4: Cluster Containing Pods Directly

**File**: `examples/example_cluster_direct_pods.yaml`

Shows clusters referencing pods directly while superpods are still defined separately.

```yaml
groupings:
  # ... (base_units, trays, hosts definitions omitted)

  - meshes:
      hosts: 4

  - pods:
      meshes: 2

  - superpods:
      pods: 2

  - clusters:
      pods: 6  # Cluster contains 6 pods directly
```

### Example 5: Mixed Counts

**File**: `examples/example_mixed_counts_lists.yaml`

Demonstrates mixing different grouping types within a single grouping, such as a superpod containing both pods and meshes.

```yaml
groupings:
  # ... (base_units, trays, hosts definitions omitted)

  - meshes:
      hosts: 1  # Each mesh = 1 host (meshes can have 1 item)

  - pods:
      meshes: 2  # Pods must have at least 2 items

  - superpods:
      pods: 2  # Superpods must have at least 2 items
      meshes: 4  # Also contains 4 meshes directly

  - clusters:
      superpods: 2  # Clusters must have at least 2 items
      pods: 2  # Also contains 2 pods directly
      meshes: 3  # Also contains 3 meshes directly
```

### Example 6: Mesh Defined by Trays

**File**: `examples/example_mesh_by_trays.yaml`

Shows meshes defined using tray count instead of host count, providing more granular control over mesh size.

```yaml
groupings:
  # ... (base_units, trays, hosts definitions omitted)

  - meshes:
      trays: 16  # Each mesh = 16 trays (instead of hosts)

  - pods:
      meshes: 2

  - superpods:
      pods: 2

  - clusters:
      superpods: 2
```

### Example 7: Mesh Defined by ASIC Locations

**File**: `examples/example_mesh_by_asic_locations.yaml`

Shows meshes defined directly at the ASIC location level using all 8 base unit names.

```yaml
groupings:
  # ... (base_units, trays, hosts definitions omitted)

  - meshes:
      asic_locations: [asic_location_1]  # Each mesh uses 1 ASIC location (smaller mesh example)

  - pods:
      meshes: 2

  - superpods:
      pods: 2

  - clusters:
      superpods: 2
```

### Example 8: Custom Groupings - Half Tray

**File**: `examples/example_mesh_subdivided_asic_locations.yaml`

Demonstrates defining custom groupings (half_tray_lower and half_tray_upper) to represent subsets of ASIC locations, then using them in meshes.

```yaml
groupings:
  # ... (base_units, trays, hosts definitions omitted)

  # Custom grouping: Half tray - defined twice with different ASIC locations
  # Both represent a "half tray" concept
  - halftray:
      asic_locations: [asic_location_1, asic_location_2, asic_location_3, asic_location_4]  # Lower half (locations 1-4)

  - halftray:
      asic_locations: [asic_location_5, asic_location_6, asic_location_7, asic_location_8]  # Upper half (locations 5-8)

  - meshes:
      halftray: 1  # Each mesh uses 1 half tray (can be either lower or upper)

  - pods:
      meshes: 2

  - superpods:
      pods: 2

  - clusters:
      superpods: 2
```

### Example 9: Complex Multi-Level Mixing

**File**: `examples/example_complex_multi_level.yaml`

Shows complex configurations where groupings contain multiple types of lower-level groupings at different levels.

```yaml
groupings:
  # ... (base_units, trays, hosts definitions omitted)

  - meshes:
      hosts: 1  # Each mesh = 1 host (meshes can have 1 item)

  - pods:
      meshes: 2  # Must have at least 2 items

  - superpods:
      pods: 2  # Must have at least 2 items
      meshes: 2  # Also contains 2 meshes directly

  - clusters:
      superpods: 2  # Must have at least 2 items
      pods: 2  # Also contains 2 pods directly
      meshes: 3  # Also contains 3 meshes directly
```

## Key Benefits

- ✅ **Hardware-agnostic**: No ASIC IDs in the file - completely reusable across different clusters
- ✅ **Declarative**: Define structure using counts and base units, not explicit IDs
- ✅ **Flexible**: Can mix and match grouping types, reference multiple levels
- ✅ **Maintainable**: Physical structure (trays, hosts) is more stable than ASIC IDs

## File Location

Physical groupings files are located in: `tests/tt_metal/tt_fabric/physical_groupings/`

- **`3_pod_16x8_bh_galaxy_physical_groupings.yaml`**: Main configuration file for 3-pod 16x8 Blackhole Galaxy cluster
- **`triple_16x8_quad_bh_galaxy_blitz_48_stage_physical_groupings.yaml`**: Blitz pipelined 48-stage configuration
- **`examples/`**: Directory containing example files demonstrating different grouping patterns
