# 4x Multi-Node BH Loudbox System Cluster Setup

This directory contains configuration files, test utilities, and documentation for bringing up and validating a 4x Blackhole (BH) Loudbox system cluster composed into a 4x8 mesh topology.

## Overview

### System Architecture

The cluster consists of:
- **4 BH Loudbox hosts**: `bh-lb-39`, `bh-lb-37`, `bh-lb-40`, `bh-lb-38` (in ring order)
- **32 P150 boards total**: 8 P150 boards per Loudbox (Tray IDs 1-8)
- **32 ASICs total**: 1 Blackhole ASIC per P150 board

### Physical Topology

Each Loudbox contains 8 P150 boards with internal mesh connections:
- **Vertical connections**: Between lower half (trays 1-4) and upper half (trays 5-8)
- **Ring 1**: Lower half (trays 1-4) on ports 3/4
- **Ring 2**: Upper half (trays 5-8) on ports 3/4
- **External connections**: Inter-loudbox ring: lb-39 → lb-37 → lb-40 → lb-38 → lb-39

### Host Mapping

The system maps MPI ranks to physical hosts as follows (from `rankfile/4x8.txt`):
- **Rank 0** → `bh-lb-39` (L1 in ring)
- **Rank 1** → `bh-lb-37` (L2 in ring)
- **Rank 2** → `bh-lb-40` (L3 in ring)
- **Rank 3** → `bh-lb-38` (L4 in ring)

**Ring topology**: lb-39 → lb-37 → lb-40 → lb-38 → lb-39

### Mesh Graph Descriptors

Define logical mesh topologies for workload execution. Located in `mesh_graph_descriptors/`:

| File | Topology | Description |
|------|----------|-------------|
| `4x8_bh_torus_xy_mesh_graph_descriptor.textproto` | 4x8 Torus-XY | 32 devices, 4x2 hosts, full torus with RING topology |

## Directory Structure

```
tests/scale_out/4x_blaze_loudbox/
├── README.md                          # This file
├── test_sanity.py                     # System sanity test
├── cabling_descriptors/               # Physical cabling topology
├── deployment_descriptors/            # Host deployment information
├── factory_system_descriptors/        # Generated system configuration
│   └── factory_system_descriptor_4x_bh_loudbox.textproto
├── mesh_graph_descriptors/            # Logical mesh topologies
│   └── 4x8_bh_torus_xy_mesh_graph_descriptor.textproto
├── rank_bindings/                     # MPI rank to mesh mappings
│   └── 4x8.yaml
└── rankfile/                          # MPI rankfile configurations
    └── 4x8.txt
```

## Getting Started

### Prerequisites

1. All hosts must be accessible via network
2. The cluster must be properly cabled according to the cabling descriptor
3. Build the project with: `./build_metal.sh --build-tests`

### Validate Cluster Health

Before running tests, validate that all ethernet links are trained and healthy:

```bash
./build/tools/scaleout/run_cluster_validation \
  --hard-fail \
  --factory-descriptor-path tests/scale_out/4x_blaze_loudbox/factory_system_descriptors/factory_system_descriptor_4x_bh_loudbox.textproto
```

### Run Sanity Test

Once the system is healthy, run the 4x8 mesh sanity test:

```bash
tt-run \
  --tcp-interface <your_network_interface> \
  --mesh-graph-descriptor tests/scale_out/4x_blaze_loudbox/mesh_graph_descriptors/4x8_bh_torus_xy_mesh_graph_descriptor.textproto \
  --hosts <ip1>,<ip2>,<ip3>,<ip4> \
  --mpi-args "--map-by rankfile:file=tests/scale_out/4x_blaze_loudbox/rankfile/4x8.txt" \
  --rank-binding tests/scale_out/4x_blaze_loudbox/rank_bindings/4x8.yaml \
  pytest -svv tests/scale_out/4x_blaze_loudbox/test_sanity.py
```

**Replace:**
- `<your_network_interface>` with your network interface (e.g., `enp10s0f1np1`)
- `<ip1>,<ip2>,<ip3>,<ip4>` with actual IPs for: **bh-lb-39, bh-lb-37, bh-lb-40, bh-lb-38** (in ring order)

### Example Command

```bash
tt-run \
  --tcp-interface enp10s0f1np1 \
  --mesh-graph-descriptor tests/scale_out/4x_blaze_loudbox/mesh_graph_descriptors/4x8_bh_torus_xy_mesh_graph_descriptor.textproto \
  --hosts 10.140.20.1,10.140.20.2,10.140.20.3,10.140.20.4 \
  --mpi-args "--map-by rankfile:file=tests/scale_out/4x_blaze_loudbox/rankfile/4x8.txt" \
  --rank-binding tests/scale_out/4x_blaze_loudbox/rank_bindings/4x8.yaml \
  pytest -svv tests/scale_out/4x_blaze_loudbox/test_sanity.py
```

## Test Details

The sanity test (`test_sanity.py`) performs:
1. Creates a 4x8 mesh device with 2D fabric configuration
2. Generates a random tensor sharded across all 32 devices
3. Executes a GELU operation on the distributed tensor
4. Validates output matches expected results with 99.99% correlation

## Troubleshooting

- If cluster validation fails, check physical cabling matches the cabling descriptor
- Ensure all hosts are accessible and network interfaces are configured correctly
- Check that the correct network interface is specified in the tt-run command
- Verify MPI is properly configured on all hosts
- Ensure rank order matches the ring topology: lb-39 → lb-37 → lb-40 → lb-38
