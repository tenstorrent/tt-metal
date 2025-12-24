# MGD Pinning Syntax Proposals

This document proposes syntax options for adding fabric node ID to physical ASIC position pinnings in Mesh Graph Descriptor (MGD) files.

## Current State

Pinnings are currently specified programmatically via the `TopologyMapper` constructor:
```cpp
std::map<MeshId, std::vector<std::pair<AsicPosition, FabricNodeId>>> fixed_asic_position_pinnings;
// AsicPosition = (TrayID, ASICLocation)
// FabricNodeId = (MeshId, ChipId)
```

## Requirements

- Map logical fabric nodes (mesh_id, chip_id) to physical ASIC positions (tray_id, asic_location)
- Pinnings must be a fully separate top-level section (outside mesh_descriptors, graph_descriptors, and instances)
- Human-readable and maintainable
- Compatible with protobuf textproto format
- Use proper message types for FabricNodeId and PhysicalAsicPosition

## Message Type Definitions

The following message types should be defined in the protobuf schema:

```protobuf
// Logical fabric node identifier
message FabricNodeId {
    uint32 mesh_id = 1;    // Mesh identifier
    uint32 chip_id = 2;    // Chip identifier within the mesh
}

// Physical ASIC position in the hardware system
message PhysicalAsicPosition {
    uint32 tray_id = 1;         // Tray identifier
    uint32 asic_location = 2;   // ASIC location within the tray
}
```

## Proposed Solution: Top-Level Pinning Section

Add a top-level `pinnings` section to `MeshGraphDescriptor` that is separate from mesh_descriptors, graph_descriptors, and instances:

```protobuf
message MeshGraphDescriptor {
    // ... existing fields (mesh_descriptors, graph_descriptors, top_level_instance) ...

    message AsicPinning {
        // Logical fabric node to pin
        FabricNodeId fabric_node_id = 1;

        // Physical ASIC position where the fabric node should be pinned
        PhysicalAsicPosition physical_asic_position = 2;
    }

    // Top-level pinnings section (separate from descriptors and instances)
    repeated AsicPinning pinnings = 5;
}
```

**Example MGD file:**
```textproto
# --- Mesh Descriptors ------------------------------------------------------

mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 8, 4 ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 4 policy: RELAXED }
}

# --- Graph Descriptors ------------------------------------------------------

# (No graph descriptors in this example)

# --- Pinnings ---------------------------------------------------------------

# Pin fabric node (mesh 0, chip 0) to ASIC at tray 1, location 1
pinnings {
  fabric_node_id {
    mesh_id: 0
    chip_id: 0
  }
  physical_asic_position {
    tray_id: 1
    asic_location: 1
  }
}

# Pin fabric node (mesh 0, chip 31) to ASIC at tray 4, location 1
pinnings {
  fabric_node_id {
    mesh_id: 0
    chip_id: 31
  }
  physical_asic_position {
    tray_id: 4
    asic_location: 1
  }
}

# --- Instantiation ----------------------------------------------------------

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
```

**Pros:**
- Fully separate section at top level (not nested in descriptors or instances)
- Clear separation of concerns
- Can handle pinnings for multiple mesh instances
- Uses proper message types
- Follows MGD structure pattern (separate sections for descriptors, pinnings, and instantiation)
- Type-safe with explicit mapping direction (FabricNodeId -> PhysicalAsicPosition)

**Cons:**
- Requires proto file modification
- Pinnings are separated from mesh definitions (by design)

## Implementation Notes

1. **Proto file changes**:
   - Add `FabricNodeId` message type with `mesh_id` and `chip_id` fields
   - Add `PhysicalAsicPosition` message type with `tray_id` and `asic_location` fields
   - Add `AsicPinning` message with `fabric_node_id` and `physical_asic_position` fields
   - Add `pinnings` repeated field to `MeshGraphDescriptor` (top-level, separate from descriptors and instances)

2. **Parsing**: Extract pinnings during MGD parsing and convert to `std::map<MeshId, std::vector<std::pair<AsicPosition, FabricNodeId>>>`
   - Note: The C++ structure uses `std::pair<AsicPosition, FabricNodeId>` which maps PhysicalAsicPosition -> FabricNodeId
   - The proto definition maps FabricNodeId -> PhysicalAsicPosition for clarity
   - Conversion logic will need to reverse the mapping during parsing

3. **Validation**:
   - Ensure `fabric_node_id.mesh_id` matches the mesh instance
   - Ensure `fabric_node_id.chip_id` is valid for the mesh size
   - Ensure no duplicate pinnings for the same fabric_node_id
   - Validate `physical_asic_position.tray_id` and `physical_asic_position.asic_location` exist in physical system

4. **Backward compatibility**: Make `pinnings` optional (repeated field defaults to empty)

## Example Implementation Flow

```cpp
// In mesh_graph_descriptor.cpp or similar
std::map<MeshId, std::vector<std::pair<AsicPosition, FabricNodeId>>>
extract_pinnings_from_mgd(const proto::MeshGraphDescriptor& proto) {
    std::map<MeshId, std::vector<std::pair<AsicPosition, FabricNodeId>>> pinnings;

    // Extract pinnings from top-level pinnings section
    for (const auto& pinning : proto.pinnings()) {
        // Extract FabricNodeId from proto
        FabricNodeId fabric_node(
            MeshId{pinning.fabric_node_id().mesh_id()},
            pinning.fabric_node_id().chip_id()
        );

        // Extract PhysicalAsicPosition from proto and convert to AsicPosition
        AsicPosition asic_pos(
            TrayID{pinning.physical_asic_position().tray_id()},
            ASICLocation{pinning.physical_asic_position().asic_location()}
        );

        // Store as pair(AsicPosition, FabricNodeId) for C++ compatibility
        // Group by mesh_id for the return structure
        MeshId mesh_id = fabric_node.mesh_id;
        pinnings[mesh_id].emplace_back(asic_pos, fabric_node);
    }

    return pinnings;
}
```
