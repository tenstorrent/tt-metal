# PR: Control Plane Z-Direction Fix

## Summary

Adds Z-direction port fallback when NESW ports are exhausted during intermesh port assignment. Previously, Z-direction ports were never used unless `assign_z_direction` was explicitly set for a mesh pair—even when NESW ports ran out, assignment would fail instead of falling back to Z. Now the control plane falls back to Z-direction ports when needed, enabling topologies that require more intermesh channels than NESW alone can provide.

Will now show a warning message when using Z direction without setting `assign_z_direction` for a mesh pair.

Also includes galaxy pinnings per-mesh fix and improved error messages.

## Example (MGD)

**Explicit Z direction** (unchanged behavior—Z used from the start):

```textproto
connections {
  nodes { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
  nodes { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
  channels { count: 4 }
  assign_z_direction: true
}
```

**Fallback scenario** (new behavior—Z used when NESW runs out):

```textproto
connections {
  nodes { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
  nodes { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
  channels { count: 4 }
  # assign_z_direction not set
}
```

Previously this would fail if the physical topology had fewer than 4 NESW ports between the meshes. Now the control plane assigns NESW first, then falls back to Z-direction ports when NESW is exhausted, and logs a warning.

## Changes

### Control Plane

- **Z-direction fallback** (main fix): When NESW ports run out during `assign_logical_ports_to_exit_nodes`, fall back to Z-direction ports. Previously Z was never used unless `assign_z_direction` was set; now Z is used as fallback when NESW capacity is exhausted. Logs a warning on first fallback.
- **Galaxy pinnings per-mesh**: Refactored `get_galaxy_fixed_asic_position_pinnings` → `get_galaxy_fixed_asic_position_pinnings_for_mesh`. Pinnings are now applied per mesh based on each mesh's shape (32 chips, non-1D) instead of using total chip count across all meshes.
- **Error messages**: Clearer validation failure in `validate_requested_intermesh_connections` with hint about `assign_z_direction` and reducing `channels.count`.
- **Debug logging**: Added physical location (host, tray, asic) to `get_requested_exit_nodes` logs for diagnostics.
- **Typo fix**: `convert_port_desciptors_to_intermesh_connections` → `convert_port_descriptors_to_intermesh_connections`

### Mesh Graph Descriptor

- **get_switch_chip_count()**: New API to compute chip count for switch instances from `device_topology.dims()` (same logic as meshes).

### Tests

- **Removed N300_2x2 skips**: `TestSplit2x2*` and `TestBigMesh2x4*` no longer skip when cluster type is not N300_2x2.

## Files Modified

- `tt_metal/fabric/control_plane.cpp`
- `tt_metal/api/tt-metalium/experimental/fabric/control_plane.hpp`
- `tt_metal/fabric/mesh_graph_descriptor.cpp`
- `tt_metal/api/tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp`
- `tests/tt_metal/tt_fabric/fabric_router/test_multi_host.cpp`
