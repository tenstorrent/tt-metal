# Generate Cluster Descriptors Script

This script generates cluster descriptor files from multiple hosts using MPI and the UMD topology tool. It supports both simple hostname-based execution and rankfile-based execution with environment variable configuration from rank bindings files.

## Overview

The script runs the topology discovery tool on multiple hosts in parallel using MPI, generating a cluster descriptor file for each host/rank. It can optionally:
- Use a rankfile to specify exact rank-to-host/slot mappings
- Set environment variables (like `TT_VISIBLE_DEVICES`) per rank from a rank bindings YAML file
- Generate a mapping file that maps rank numbers to cluster descriptor file paths

## Prerequisites

1. **MPI**: OpenMPI or compatible MPI implementation with `mpirun`
2. **Topology Tool**: The UMD topology tool (auto-detected from common locations or can be specified)
3. **SSH Access**: Passwordless SSH access to all target hosts
4. **Rank Bindings File** (optional): YAML file specifying rank-to-mesh mappings and environment variables

## Usage

### Basic Syntax

```bash
./scripts/scaleout/generate_cluster_descriptors.sh \
  --mapping-file <output_mapping.yaml> \
  --output-dir <output_directory> \
  --base-name <base_name> \
  [--rankfile <rankfile>] \
  [--hostnames <host1,host2,...>] \
  [--rank-bindings-file <rank_bindings.yaml>] \
  [--topology-tool <path>] \
  [--mpi-launcher <mpirun|mpirun-ulfm>] \
  [--dry-run]
```

### Required Parameters

- `--mapping-file <file>`: Output YAML file path that maps rank numbers to cluster descriptor file paths
- `--output-dir <dir>`: Directory where cluster descriptor files will be saved
- `--base-name <name>`: Base name for cluster descriptor files (e.g., `my_cluster_desc`)

### Optional Parameters

- `--rankfile <file>`: MPI rankfile specifying rank-to-host/slot mappings (required if `--hostnames` not provided)
- `--hostnames <list>`: Comma-separated list of hostnames (required if `--rankfile` not provided)
- `--rank-bindings-file <file>`: YAML file with rank bindings and environment variable overrides
- `--topology-tool <path>`: Path to topology tool (default: auto-detect from `./build/tools/umd/topology`)
- `--mpi-launcher <cmd>`: MPI launcher command (default: `mpirun`)
- `--dry-run`: Show what would be executed without running

## Examples

### Example 1: Simple Multi-Host Setup

Generate cluster descriptors for 3 hosts without rank bindings:

```bash
./scripts/scaleout/generate_cluster_descriptors.sh \
  --hostnames "host1,host2,host3" \
  --mapping-file cluster_mapping.yaml \
  --output-dir ./cluster_descriptors \
  --base-name my_cluster_desc
```

This will create:
- `my_cluster_desc_rank_0.yaml` (on host1)
- `my_cluster_desc_rank_1.yaml` (on host2)
- `my_cluster_desc_rank_2.yaml` (on host3)
- `cluster_mapping.yaml` mapping ranks to files

### Example 2: Using Rankfile with Rank Bindings

Generate cluster descriptors using a rankfile and rank bindings for environment variables:

```bash
./scripts/scaleout/generate_cluster_descriptors.sh \
  --rankfile my_rankfile \
  --mapping-file tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/my_mapping.yaml \
  --output-dir tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/my_cluster_desc \
  --rank-bindings-file tests/tt_metal/distributed/config/my_rank_bindings.yaml \
  --base-name my_cluster_desc
```

**Rankfile format** (`my_rankfile`):
```
rank 0=bh-glx-b08u02 slot=0
rank 1=bh-glx-b08u08 slot=0
rank 2=bh-glx-b09u02 slot=0
rank 3=bh-glx-b09u08 slot=0
```

**Rank bindings file format** (`my_rank_bindings.yaml`):
```yaml
rank_bindings:
  - rank: 0
    mesh_id: 0
    mesh_host_rank: 0
    env_overrides:
      TT_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"
  - rank: 1
    mesh_id: 0
    mesh_host_rank: 1
    env_overrides:
      TT_VISIBLE_DEVICES: "8,9,10,11,12,13,14,15"
  # ... more ranks
mesh_graph_desc_path: "path/to/mesh_graph_descriptor.textproto"
```

This will create files named with hostname and slot:
- `my_cluster_desc_bh-glx-b08u02_slot_0.yaml`
- `my_cluster_desc_bh-glx-b08u08_slot_0.yaml`
- etc.

### Example 3: BH 2x4 Split Configuration

Generate cluster descriptors for a BH 2x4 Split topology:

```bash
./scripts/scaleout/generate_cluster_descriptors.sh \
  --rankfile bh_2x4_split_rankfile \
  --mapping-file tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/bh_2x4_split_mapping.yaml \
  --output-dir tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/bh_2x4_split_cluster_desc \
  --rank-bindings-file tests/tt_metal/distributed/config/bh_2x4_split_rank_bindings.yaml \
  --base-name bh_2x4_split_cluster_desc
```

### Example 4: 32x4 Quad Galaxy Configuration

Generate cluster descriptors for a 32x4 Quad Galaxy topology:

```bash
./scripts/scaleout/generate_cluster_descriptors.sh \
  --rankfile 32x4_quad_galaxy_rankfile \
  --mapping-file tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/32x4_quad_galaxy_cluster_desc_mapping.yaml \
  --output-dir tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/32x4_quad_galaxy_cluster_desc \
  --rank-bindings-file tests/tt_metal/distributed/config/32x4_quad_galaxy_rank_bindings.yaml \
  --base-name 32x4_quad_galaxy_cluster_desc
```

**Rankfile** (`32x4_quad_galaxy_rankfile`):
```
rank 0=bh-glx-b08u02 slot=0
rank 1=bh-glx-b08u08 slot=0
rank 2=bh-glx-b09u02 slot=0
rank 3=bh-glx-b09u08 slot=0
```

## How It Works

1. **Parse Inputs**: The script parses the rankfile (or hostnames) and rank bindings file
2. **Determine Rank Count**:
   - If rank bindings file is provided, uses the number of ranks from that file
   - Otherwise, uses the number of hosts or ranks from the rankfile
3. **Build MPI Command**:
   - Uses `--rankfile` if provided, otherwise uses `--host` with hostname:N format
   - Adds `--oversubscribe` if multiple processes per host
   - Sets `-np` to the total number of ranks
4. **Execute**: Runs MPI command that:
   - Sets environment variables per rank (from rank bindings file)
   - Runs topology tool on each host/rank
   - Generates cluster descriptor files
5. **Generate Mapping**: Creates a YAML mapping file that maps rank N to the cluster descriptor file generated by MPI rank N

## File Naming Conventions

- **With rankfile**: Files are named `{base_name}_{hostname}_slot_{slot}.yaml`
  - Example: `my_cluster_desc_bh-glx-b08u02_slot_0.yaml`

- **Without rankfile**: Files are named `{base_name}_rank_{rank}.yaml`
  - Example: `my_cluster_desc_rank_0.yaml`

## Rank Bindings File Format

The rank bindings file should be a YAML file with the following structure:

```yaml
rank_bindings:
  - rank: 0
    mesh_id: 0
    mesh_host_rank: 0
    env_overrides:
      TT_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"
  - rank: 1
    mesh_id: 0
    mesh_host_rank: 1
    env_overrides:
      TT_VISIBLE_DEVICES: "8,9,10,11,12,13,14,15"
mesh_graph_desc_path: "path/to/mesh_graph_descriptor.textproto"
```

## Rankfile Format

The rankfile should follow OpenMPI rankfile format:

```
rank 0=hostname1 slot=0
rank 1=hostname1 slot=1
rank 2=hostname2 slot=0
rank 3=hostname2 slot=1
```

Comments starting with `#` are supported.

## Output Mapping File Format

The generated mapping file will have the following format:

```yaml
rank_to_cluster_mock_cluster_desc:
  0: "path/to/cluster_desc_rank_0.yaml"
  1: "path/to/cluster_desc_rank_1.yaml"
  2: "path/to/cluster_desc_rank_2.yaml"
  # ... more ranks
```

## Troubleshooting

### MPI Connection Issues

- Ensure passwordless SSH is set up to all hosts
- Check that hosts are reachable: `ssh hostname hostname`
- Verify MPI can connect: `mpirun --host host1,host2 hostname`

### Topology Tool Not Found

- Build the topology tool: `./build_metal.sh --build-tools`
- Or specify explicitly: `--topology-tool /path/to/topology`

### Rank Mismatch Errors

- Ensure the number of ranks in the rankfile matches the rank bindings file
- Check that rank numbers are contiguous starting from 0

### Permission Issues

- Ensure write permissions to the output directory
- Check that the topology tool is executable

## Notes

- The script automatically adds `--oversubscribe` when using multiple processes per host
- Environment variables from rank bindings are set before running the topology tool
- The mapping file ensures rank N from rank bindings maps to the cluster descriptor generated by MPI rank N
- Files are organized into subdirectories when using rankfile-based naming

## See Also

- `tests/tt_metal/tt_fabric/utils/generate_rank_bindings.py` - Script to generate rank bindings files
- `tests/tt_metal/distributed/config/` - Example rank bindings files
- `tt_metal/fabric/mesh_graph_descriptors/` - Mesh graph descriptor files
