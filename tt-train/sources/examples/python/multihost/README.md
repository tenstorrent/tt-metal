# Multihost Training Examples

This directory contains examples for running distributed training across multiple hosts using Tenstorrent hardware.

## Directory Structure

```
multihost/
├── fabric_minimal_example/
│   ├── configurations/
│   │   └── 2loudboxes/
│   ├── example.py
│   └── runner.sh
├── hierarchical_parallel/
│   ├── configurations/
│   │   └── 5loudboxes/
│   ├── trainer.py
│   ├── training.py
│   └── runner.sh
└── pipeline_parallel_training/
    ├── configurations/
    │   └── 5loudboxes/
    ├── trainer.py
    ├── training.py
    └── runner.sh
```

## Configurations Folder Structure

Each example contains a `configurations/` folder that holds hardware-specific setup files. Configurations are organized by hardware setup name (e.g., `2loudboxes`, `5loudboxes`), where the name typically indicates the number of host machines or "loudboxes" in the cluster.

### Configuration Directory Layout

Each configuration directory (e.g., `configurations/2loudboxes/`) contains three required files:

```
configurations/
└── <hardware_setup_name>/
    ├── hosts.txt
    ├── mgd.yaml
    └── rank_bindings.yaml
```

### File Descriptions

#### 1. `hosts.txt`
Defines the list of host machines and the number of MPI slots (processes) available on each host.

**Format:**
```
<hostname> slots=<number>
<hostname> slots=<number>
...
```

**Example (`2loudboxes/hosts.txt`):**
```
metal-wh-01 slots=1
metal-wh-04 slots=1
```

**Example (`5loudboxes/hosts.txt`):**
```
metal-wh-01 slots=1
metal-wh-03 slots=1
metal-wh-04 slots=1
metal-wh-05 slots=1
metal-wh-06 slots=1
```

This file is used by MPI (via `mpirun --hostfile`) to distribute processes across the cluster.

#### 2. `mgd.yaml` (Mesh Graph Descriptor)
Defines the hardware topology and connectivity of the distributed system, including chip specifications, device meshes, and inter-mesh connections.

**Key Sections:**

- **`ChipSpec`**: Specifies the chip architecture (e.g., `wormhole_b0`) and ethernet port configuration
- **`Board`**: Defines board types and their mesh topology (e.g., `[1, 8]` for 1×8 device layout)
- **`Mesh`**: Lists all device meshes with their IDs, device topology, and host topology
- **`RelaxedGraph`**: Defines connectivity between meshes (e.g., `[M0, M1, 2]` means Mesh 0 and Mesh 1 are connected with bandwidth weight 2)

**Example Structure:**
```yaml
ChipSpec: {
  arch: wormhole_b0,
  ethernet_ports: {
    N: 2, E: 2, S: 2, W: 2,
  }
}

Board: [
  { name: lbox, type: Mesh, topology: [1, 8]}
]

Mesh: [
  {
    id: 0,
    board: lbox,
    device_topology: [1, 8],
    host_topology: [1, 1],
  },
  {
    id: 1,
    board: lbox,
    device_topology: [1, 8],
    host_topology: [1, 1],
  }
]

RelaxedGraph: [
  [M0, M1, 2],
]
```

**Mesh ID Assignment:**
- Each mesh has a unique `id` (0, 1, 2, etc.)
- `device_topology: [1, 8]` means 1 row × 8 devices per mesh
- `host_topology: [1, 1]` means 1 host per mesh
- The `RelaxedGraph` defines which meshes can communicate and their connection bandwidth

#### 3. `rank_bindings.yaml`
Maps MPI ranks to mesh IDs and sets global environment variables for the distributed execution.

**Format:**
```yaml
rank_bindings:
  - rank: <mpi_rank>
    mesh_id: <mesh_id>
  - rank: <mpi_rank>
    mesh_id: <mesh_id>
  ...

global_env:
  TT_METAL_HOME: "<path_to_tt_metal>"

mesh_graph_desc_path: "configurations/<config_name>/mgd.yaml"
```

**Example (`2loudboxes/rank_bindings.yaml`):**
```yaml
rank_bindings:
  - rank: 0
    mesh_id: 1
  - rank: 1
    mesh_id: 0

global_env:
  TT_METAL_HOME: "/home/ttuser/git/tt-metal"
mesh_graph_desc_path: "configurations/2loudboxes/mgd.yaml"
```

**Key Fields:**
- **`rank_bindings`**: Maps each MPI rank to a specific mesh ID from `mgd.yaml`
- **`global_env`**: Environment variables set for all processes (typically `TT_METAL_HOME`)
- **`mesh_graph_desc_path`**: Relative path to the corresponding `mgd.yaml` file

## Creating a New Configuration

To create a new configuration for your hardware setup:

1. **Create a new directory** under `configurations/` with a descriptive name:
   ```bash
   mkdir -p configurations/my_setup
   ```

2. **Create `hosts.txt`** listing your hosts:
   ```bash
   cat > configurations/my_setup/hosts.txt << EOF
   hostname1 slots=1
   hostname2 slots=1
   EOF
   ```

3. **Create `mgd.yaml`** describing your hardware topology:
   - Define chip architecture and specifications
   - List all device meshes with their IDs and topologies
   - Specify connectivity in the `RelaxedGraph`

4. **Create `rank_bindings.yaml`** mapping ranks to meshes:
   - Assign each MPI rank to a mesh ID
   - Set necessary environment variables
   - Point to your `mgd.yaml` file

5. **Update the runner script** to use your configuration:
   ```bash
   ./runner.sh --hostfile configurations/my_setup/hosts.txt \
               --rank-bindings configurations/my_setup/rank_bindings.yaml
   ```

## Usage

Each example includes a `runner.sh` script that accepts configuration parameters:

```bash
# Use default configuration
./runner.sh

# Use custom configuration
./runner.sh --hostfile configurations/my_setup/hosts.txt \
            --rank-bindings configurations/my_setup/rank_bindings.yaml
```

The runner script:
1. Syncs code to all machines listed in `hosts.txt`
2. Launches the training script using `ttrun.py` with MPI
3. Uses `rank_bindings.yaml` to assign ranks to specific hardware meshes

## Examples Overview

### 1. Fabric Minimal Example
Located in `fabric_minimal_example/`, demonstrates basic distributed tensor operations using the fabric API with all-reduce synchronization.

### 2. Hierarchical Parallel Training
Located in `hierarchical_parallel/`, implements a 2-tier and 3-tier hierarchical parallelism strategy with workers, aggregators, and optimizers.

### 3. Pipeline Parallel Training
Located in `pipeline_parallel_training/`, demonstrates pipeline parallelism for training large models across multiple devices.
