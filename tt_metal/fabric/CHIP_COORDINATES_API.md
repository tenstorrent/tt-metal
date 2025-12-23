# Chip Coordinates API

## Overview

The Control Plane now provides an API to export chip coordinates to a YAML file. This is useful for visualization tools like ttnn-visualizer that need to understand the physical topology of the chip mesh.

## API Methods

### `get_chip_coordinates()`

Returns a map of physical chip IDs to their mesh coordinates.

```cpp
std::map<ChipId, std::vector<uint32_t>> get_chip_coordinates() const;
```

**Returns:** A map where:
- Key: Physical chip ID
- Value: A 4D coordinate vector [x, y, z, w] representing the chip's position in the mesh

**Example:**
```cpp
auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
auto chip_coordinates = control_plane.get_chip_coordinates();

for (const auto& [chip_id, coords] : chip_coordinates) {
    std::cout << "Chip " << chip_id << ": [";
    for (size_t i = 0; i < coords.size(); ++i) {
        std::cout << coords[i];
        if (i < coords.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}
```

### `serialize_chip_coordinates_to_file()`

Serializes chip coordinates to a YAML file.

```cpp
void serialize_chip_coordinates_to_file(const std::string& filepath) const;
```

**Parameters:**
- `filepath`: Path where the YAML file will be written

**Output Format:**
```yaml
chips:
  0: [1, 0, 0, 0]
  1: [1, 1, 0, 0]
  2: [2, 1, 0, 0]
  3: [2, 0, 0, 0]
  4: [0, 0, 0, 0]
  5: [0, 1, 0, 0]
  6: [3, 1, 0, 0]
  7: [3, 0, 0, 0]
```

**Example:**
```cpp
auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
control_plane.serialize_chip_coordinates_to_file("/path/to/chip_coordinates.yaml");
```

## Use Cases

1. **Topology Visualization**: Tools can use this information to render the physical layout of chips
2. **Debugging**: Understanding the physical topology helps debug communication patterns
3. **Configuration**: Can be used alongside cluster_description.yaml for complete system configuration

## Coordinate System

Coordinates are provided in a 4D space:
- Dimension 0-1: Primary mesh coordinates (typically row, column in 2D meshes)
- Dimension 2-3: Additional dimensions for hierarchical or 3D+ topologies (padded with zeros if unused)

The coordinate system matches the internal mesh graph representation and is consistent across the entire cluster.
