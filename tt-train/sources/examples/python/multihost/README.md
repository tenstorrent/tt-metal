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
    ├── mgd.textproto
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

#### 2. `mgd.textproto` (Mesh Graph Descriptor)
Defines the hardware topology and connectivity of the distributed system using the textproto format. The YAML format (MGD 1.0) has been deprecated.

**Note:** If you have an existing `mgd.yaml` file, you need to convert it to `mgd.textproto` format. See the conversion guide in [`tt_metal/fabric/MGD_README.md`](../../../../tt_metal/fabric/MGD_README.md) (section: Converting from MGD 1.0 to MGD).

**Key Sections:**

- **`mesh_descriptors`**: Defines reusable mesh templates with architecture, device topology, host topology, and channel configuration
- **`graph_descriptors`**: Defines logical groupings and connectivity across meshes
- **`top_level_instance`**: Specifies the root instance to instantiate

**Example Structure:**
```proto
# --- Meshes ---------------------------------------------------------------

mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 8 ] }
  host_topology   { dims: [ 1, 1 ] }
  channels {
    count: 2
    policy: STRICT
  }
}

mesh_descriptors {
  name: "M1"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 8 ] }
  host_topology   { dims: [ 1, 1 ] }
  channels {
    count: 2
    policy: STRICT
  }
}

# --- Graphs ---------------------------------------------------------------

graph_descriptors {
  name: "G0"
  type: "FABRIC"
  instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
  instances { mesh { mesh_descriptor: "M1" mesh_id: 1 } }

  connections {
    nodes { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
    nodes { mesh { mesh_descriptor: "M1" mesh_id: 1 } }
    channels { count: 2 }
  }
}

# --- Instantiation ----------------------------------------------------------
top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
```

**Mesh ID Assignment:**
- Each mesh descriptor has a unique `name` (e.g., "M0", "M1")
- `device_topology { dims: [ 1, 8 ] }` means 1 row × 8 devices per mesh
- `host_topology { dims: [ 1, 1 ] }` means 1 host per mesh
- `channels.count` specifies the number of ethernet channels per direction
- The `graph_descriptors` with `connections` defines which meshes can communicate and their connection bandwidth

For more examples and detailed documentation, see [`tt_metal/fabric/MGD_README.md`](../../../../tt_metal/fabric/MGD_README.md).

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

mesh_graph_desc_path: "configurations/<config_name>/mgd.textproto"
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
mesh_graph_desc_path: "configurations/2loudboxes/mgd.textproto"
```

**Key Fields:**
- **`rank_bindings`**: Maps each MPI rank to a specific mesh ID from `mgd.textproto`
- **`global_env`**: Environment variables set for all processes (typically `TT_METAL_HOME`)
- **`mesh_graph_desc_path`**: Relative path to the corresponding `mgd.textproto` file

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

3. **Create `mgd.textproto`** describing your hardware topology:
   - Define mesh descriptors with architecture, device topology, and host topology
   - Create graph descriptors to define connectivity between meshes
   - Specify the top-level instance
   - See [`tt_metal/fabric/MGD_README.md`](../../../../tt_metal/fabric/MGD_README.md) for detailed format documentation
   - If you have an existing `mgd.yaml` file, see the conversion guide in the same README

4. **Create `rank_bindings.yaml`** mapping ranks to meshes:
   - Assign each MPI rank to a mesh ID
   - Set necessary environment variables
   - Point to your `mgd.textproto` file

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
