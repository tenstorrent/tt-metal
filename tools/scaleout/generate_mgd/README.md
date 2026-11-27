# Generate MGD Tool

This tool automatically generates Mesh Graph Descriptors (MGDs) from cabling descriptors for fabric testing.

**Note:** This tool currently only supports creating MGDs for multi-mesh topologies based on the cabling descriptor, where each physical host corresponds to a distinct mesh in the graph.

## Building

```bash
cmake --build build --target generate_mgd
```

## Usage

```bash
./build/tools/scaleout/generate_mgd --cabling-descriptor-path <path> [OPTIONS]
```

### Required Arguments

- `--cabling-descriptor-path PATH` - Path to the cabling descriptor textproto file

### Optional Arguments

- `--output-path PATH` - Path to output MGD file (default: `mesh_graph_descriptor.textproto`)
- `--verbose, -v` - Enable verbose output
- `--help, -h` - Print help message

## Examples

### Generate MGD for 16 N300 ClosetBox cluster

```bash
./build/tools/scaleout/generate_mgd \
    --cabling-descriptor-path tools/tests/scaleout/cabling_descriptors/16_n300_lb_cluster.textproto \
    --output-path tests/tt_metal/tt_fabric/custom_mesh_descriptors/16_n300_lb_closetbox_mgd.textproto \
    --verbose
```

### Generate MGD for Dual T3K

```bash
./build/tools/scaleout/generate_mgd \
    --cabling-descriptor-path tools/tests/scaleout/cabling_descriptors/dual_t3k.textproto \
    --output-path tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_t3k_autogen_mgd.textproto
```

## How It Works

The tool performs the following steps:

1. **Determine Physical Cluster Information** - Extract architecture, system type, and number of compute nodes from the cabling descriptor
2. **Determine Physical Connectivity** - Parse the physical connectivity between compute nodes based on the cabling descriptor
3. **Assign Mesh IDs** - Assign each compute node/host a distinct Mesh ID
4. **Build Mesh Graph** - Construct a graph of the meshes based on the parsed physical connectivity

## Supported Node Types

- N300_LB_DEFAULT, N300_QB_DEFAULT (Wormhole)
- WH_GALAXY, WH_GALAXY_X_TORUS, WH_GALAXY_Y_TORUS, WH_GALAXY_XY_TORUS (Wormhole Galaxy)
- P150_LB, P150_QB_AE_DEFAULT, P300_QB_GE (Blackhole)
- BH_GALAXY, BH_GALAXY_X_TORUS, BH_GALAXY_Y_TORUS, BH_GALAXY_XY_TORUS (Blackhole Galaxy)

To add support for new node types, update the `create_node_type_lookup()` function in `generate_mgd.cpp`.
