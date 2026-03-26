# Generate Rank Bindings Tool

This tool automatically generates rank bindings YAML files by:
1. Running Physical System Descriptor (PSD) discovery via MPI
2. Running topology mapper to map logical meshes to physical ASICs
3. Extracting rank bindings from the mapping result

## Framework Structure

The tool is structured as a framework with placeholder implementations for the core functionality:

### Main Components

1. **`run_psd_discovery()`** - Runs PSD discovery via MPI or loads from file
   - TODO: Implement MPI initialization and discovery
   - Should initialize MetalContext with distributed context
   - Should run PhysicalSystemDescriptor discovery

2. **`run_topology_mapping()`** - Maps logical meshes to physical ASICs
   - TODO: Load mesh graph descriptor (MGD)
   - TODO: Load physical grouping descriptor (PGD)
   - TODO: Build physical multi-mesh graph from PSD
   - TODO: Build logical multi-mesh graph from MGD
   - TODO: Run `map_multi_mesh_to_physical()` solver

3. **`extract_rank_bindings()`** - Extracts rank bindings from mapping result
   - TODO: Group fabric nodes by mesh_id and rank
   - TODO: Convert ASIC IDs to PCIe device IDs using PSD
   - TODO: Build RankBindingConfig entries with TT_VISIBLE_DEVICES

4. **`write_rank_bindings_yaml()`** - Writes the final YAML file
   - ✅ Implemented

## Usage

```bash
generate_rank_bindings \
  --mesh-graph-desc <path_to_mgd.textproto> \
  --physical-grouping-desc <path_to_pgd.textproto> \
  --output <output.yaml> \
  [--psd-path <path_to_psd.yaml>] \
  [--mpi-launcher <mpirun|srun>] \
  [--mpi-args <additional_args>]
```

## Implementation Notes

The framework provides the structure and command-line interface. The actual implementation of:
- MPI-based PSD discovery
- Topology mapping integration
- Rank binding extraction

needs to be completed based on the existing codebase patterns in:
- `tools/scaleout/validation/run_cluster_validation.cpp` (PSD discovery)
- `tt_metal/fabric/topology_mapper.cpp` (topology mapping)
- `tests/tt_metal/tt_fabric/utils/generate_rank_bindings.py` (rank binding generation)

## Output Format

The tool generates a YAML file with the following structure:

```yaml
rank_bindings:
  - rank: 0
    mesh_id: 0
    env_overrides:
      TT_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
  - rank: 1
    mesh_id: 1
    env_overrides:
      TT_VISIBLE_DEVICES: "16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
mesh_graph_desc_path: "path/to/mesh_graph_descriptor.textproto"
```
