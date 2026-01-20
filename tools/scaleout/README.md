# Cabling Generator

A tool meant to be used in in scale-out of various Tenstorrent systems. Given
deployment specifications (rack locations of systems i.e. from the rack
elevation of a data center) and cabling specification (how to connect a set of
hosts in a certain topology) the cabling guide will generate a cutsheet listing
out each cabling link that a technician will need to attach.
![Cabling Generator Flow Diagram](images/CablingFlow.png)

## Expected Inputs

### Cabling Descriptor

This is where a topology expert will work. With no need to consider how the hosts are arranged physically, they can focus on the ideal way to connect a set of hosts together.

The Cabling Descriptor is designed to support a generic hierarchy structure so that users don't have to duplicate redundant work. At each level of hierarchy (i.e. Pod, SuperPod) the user can define connections to be duplicated at each instance of hierarchy. For example, a user only has to describe the Host->Self connections once as a part of the base node descriptor and those connections will be duplicated across all instances of that Node in the graph.

#### Cabling Hierarchy

There are 3 main parts to how the Cabling Hierarchy works

1. Nodes: the base level of the hierarchy, they describe a host system. A user of the Cabling Generator tool would define a node format as a collection of boards or use some defaults that we have included ourselves. For example, a T3K would best be described as a Node.
Examples can be seen at [Node Code](node/node.cpp).

2. Graphs: Essentially collections of Nodes (to represent a collection of multiple hosts) or other Graphs (to represent some hierarchy). This would be where users would consider groupings of hosts and how connections are duplicated at different levels of groupings.
Examples can be seen at [Graph Example](../tests/scaleout/cabling_descriptors/)

3. Root Instance: A summary of the hierarchical view of the system, also used to assign host_ids to each host. This structure in the hierarchy will enumerate every possible host in the system and where they are in the hierarchy. The host_ids enumerated here will help inform the mapping to real hosts defined in the Deployment Descriptor.
Examples can be seen at the last section of files in [Graph Example](../tests/scaleout/cabling_descriptors/)

### Deployment Descriptor

This is where a person managing a specific data center deployment of a system cluster will work. After installing/setting up the hosts required for the cluster, the technician can fill out a deployment descriptor enumerating the physical location and hostnames of each host in the cluster they wish to connect.
Examples can be seen at [Deployment Example](../tests/scaleout/deployment_descriptors/).

### Putting Them Together

One thing to consider with how the Cabling Generator puts both the Descriptors together is that the Cabling treats hosts indexes array, and the Deployment is basically an array of hosts. This is brought up to point out that order matters in the Deployment Descriptor; you will not get the same cabling guide if you mix up the order of hosts in a Deployment Descriptor

## Outputs

### Cabling Guide

The Cabling Guide is a `.csv` file that will be emitted by the cabling generator. It is a list of all cable connections that are part of the system. The Guide will specify the physical location of the host at each end of a connection, estimated cable length, and expected cable type.

Example:
![Cabling Guide Example](images/CablingGuide_ex.png)

### FSD

A `.textproto` file that will enumerate all the expected hosts, boards, and connected channels in a scaleout system.

### Cluster Descriptor

The Cluster Descriptor is a `.yaml` file (or set of files for multi-host systems) that describes the chip topology, connections, and configuration for the TT-Metal runtime. This is generated from the FSD using the `generate_cluster_descriptor` tool.

For single-host systems, a single YAML file is generated with the cluster configuration. For multi-host systems, one YAML file is generated per host (with suffix `_rank_N.yaml`) plus a mapping file (with suffix `_mapping.yaml`) that maps each rank to its corresponding cluster descriptor file.

## Tools

### run_cabling_generator

Generates a Factory System Descriptor (FSD) and Cabling Guide from Cabling and Deployment descriptors.

**Usage:**
```bash
./build/tools/scaleout/run_cabling_generator --cabling <cabling_descriptor.textproto> --deployment <deployment_descriptor.textproto> [--output <suffix>] [--simple]
```

**Options:**
- `--cabling, -c`: Path to the cabling descriptor file (.textproto) **or directory** containing multiple descriptor files
- `--deployment, -d`: Path to the deployment descriptor file (.textproto)
- `--output, -o`: Optional name suffix for output files
- `--simple, -s`: Generate simple CSV output (hostname-based) instead of detailed location information

**Outputs:**
- `out/scaleout/factory_system_descriptor_<suffix>.textproto` - Factory System Descriptor
- `out/scaleout/cabling_guide_<suffix>.csv` - Cabling guide

#### Merging Multiple Cabling Descriptors

For large systems (e.g., BH Exabox), you can organize cabling into multiple descriptor files and merge them automatically. This is useful when managing different cable batches separately:

- **Intrapod Cabling**: Cables for forming big meshes within a pod
- **Interpod Cabling**: Connections used to build a SuperPod
- **Inter-SuperPod Cabling**: Connections to build a cluster of SuperPods

**Directory-based merging:**
```bash
# Place multiple .textproto files in a directory
./build/tools/scaleout/run_cabling_generator \
    --cabling ./cabling_descriptors/ \
    --deployment deployment.textproto \
    --output merged_system
```

The tool will:
1. Find all `.textproto` files in the specified directory (sorted alphabetically)
2. Merge all graph templates and connections
   - Validates structural compatibility (same node types, board configurations)
   - Allows cross-descriptor connections on different ports for fully-connected graphs
   - Supports torus-compatible merging (X + Y → XY torus)
3. Deduplicate any duplicate connections
4. Validate that no single descriptor defines a port multiple times

**Conflict Detection:**
- **Error**: If endpoint A is connected to endpoint B in one descriptor but to endpoint C in another
- **Warning**: If the same connection appears in multiple descriptors (deduplicated automatically)
- **Warning**: If descriptors have different host counts

### generate_cluster_descriptor

Generates cluster descriptor YAML file(s) from a Factory System Descriptor (FSD). Automatically handles both single-host and multi-host systems.

**Usage:**
```bash
./build/tools/scaleout/generate_cluster_descriptor --fsd <fsd_file.textproto> [--output-dir <dir>] [--base-filename <name>]
```

**Options:**
- `--fsd, -f`: Path to the Factory System Descriptor file (.textproto)
- `--output-dir, -o`: Directory where cluster descriptor files will be written (default: `out/scaleout`)
- `--base-filename, -b`: Base name for generated files (default: `cluster_desc`)

**Outputs:**
- **Single-host:** `<output_dir>/<base_filename>.yaml`
- **Multi-host:**
  - `<output_dir>/<base_filename>_rank_0.yaml`, `_rank_1.yaml`, etc. (one per host)
  - `<output_dir>/<base_filename>_mapping.yaml` (rank-to-file mapping)

**Example:**
```bash
# Generate from an FSD file
./build/tools/scaleout/generate_cluster_descriptor --fsd factory_system_descriptor.textproto

# Custom output location and naming
./build/tools/scaleout/generate_cluster_descriptor --fsd my_system.textproto --output-dir /tmp/cluster --base-filename my_cluster
```

## Notes/Warnings

- Usage examples can be found in the tests directory at [Examples](../tests/scaleout/test_factory_system_descriptor.cpp)

- As stated previously, the ordering of hosts in the deployment descriptor does matter. This means that you will get different cabling setups with different ordering of hosts.
