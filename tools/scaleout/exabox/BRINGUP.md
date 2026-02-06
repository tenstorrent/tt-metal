# Cluster Bringup Guide

This guide covers how to add new hardware to an existing Exabox cluster by merging cabling descriptors and validating the expanded configuration.

**Use case:** You've physically installed new Galaxy nodes and need to generate cutsheets for technicians, create cabling descriptors, and validate the cluster.

## Introduction

### Galaxy Chassis

A **Galaxy** is a 6U server housing 32 Blackhole (or Wormhole but in the context of this document we are talking about Blackhole chips exclusively) chips arranged across 4 UBB Trays. Each chip connects to its neighbors via internal mesh links, with external ethernet ports on all four sides for inter-Galaxy connectivity.

<img src="https://github.com/tenstorrent/tutorial-assets/blob/main/media/tt_metal/scaleout/exabox/images/galaxy_chasis.png?raw=true" width="500"/>

*Galaxy chassis: 32 chips across 4 UBB Trays (top: Trays 1 & 3, bottom: Trays 2 & 4). Diamond-shaped ports on the edges are external ethernet connections used for inter-Galaxy cabling.*

### Cluster Hierarchy

```
SuperPod (9 pods)
  └─ Pod (4 Galaxies = 128 chips)
       └─ Galaxy (1 host = 32 Blackhole chips)
```

A **Pod** combines 4 Galaxies (128 chips) into a unified compute mesh using TT-Fabric. A **SuperPod** connects 9 pods for large-scale workloads.

### Supported Topologies

For the full list of officially supported topologies with cable lengths and configurations, see the [Top Level Topologies spreadsheet](https://docs.google.com/spreadsheets/d/1mnZJueW4BKZGNvrtGiCQZa12hxQbg49Aref2FDrRBsQ/edit?gid=0#gid=0).

| Topology | Hosts | Chips | Host Layout | Connectivity | Use Case |
|----------|-------|-------|-------------|--------------|----------|
| **Single Galaxy** | 1 | 32 | Single host | Torus XY | Single-host workloads, development |
| **4×32** | 4 | 128 | 4×1 line | Torus XY | 1D data flow: video gen, decode, ring allreduce |
| **8×16** | 4 | 128 | 2×2 grid | Torus XY | 2D data flow: all-to-all, 2D tensor parallelism |
| **16×8** | 4 | 128 | 1×4 column | Torus XY | 2D data flow: alternative orientation to 8×16 |

All multi-Galaxy topologies use **2D Torus** connectivity (wrap-around on both X and Y), enabling shorter hop counts and multiple routing paths. The **cabling differs** between topologies — each has its own cutsheet, cable lengths, and physical layout.

**Note:** 8×16 and 16×8 both use 4 Galaxies but with different host arrangements and cabling. They are **not** interchangeable — each requires its own cutsheet and cabling. MGD files for all topologies are in `tt_metal/fabric/mesh_graph_descriptors/`.

---

#### Single Galaxy (8×4)

One Galaxy (32 chips) as a self-contained mesh. Used for single-host development and testing.

<img src="https://github.com/tenstorrent/tutorial-assets/blob/main/media/tt_metal/scaleout/exabox/images/8x4_cabling.png?raw=true" width="700"/>

*Cabling web tool view of a single Galaxy (8×4). Shows 1 host (node_0) with 4 trays (T1–T4) and internal torus connections (orange curves) wrapping between trays.*

---

#### 4×32 (Quad Galaxy — Linear)

Four Galaxies arranged in a horizontal line, forming an elongated 32-column × 4-row mesh. Optimized for 1D data flow: pipeline parallelism, ring allreduce, video generation (Wan2.1/2.2), and low-latency decode (Blitz, DeepSeek).

<img src="https://github.com/tenstorrent/tutorial-assets/blob/main/media/tt_metal/scaleout/exabox/images/4x32_topology.png?raw=true" width="700"/>

*Chip-level topology: 4 Galaxy chassis side by side, which can form full 32×4 mesh.*

<img src="https://github.com/tenstorrent/tutorial-assets/blob/main/media/tt_metal/scaleout/exabox/images/4x32_cabling.png?raw=true" width="700"/>

*Cabling web tool view: 4 Galaxies across 2 racks. Blue lines are inter-Galaxy mesh connections, green lines are torus wrap-around connections.*

---

#### 8×16 (Quad Galaxy — 2×2 Grid)

Four Galaxies arranged in a 2×2 grid. Optimized for 2D data flow: all-to-all collectives and 2D tensor parallelism.

<img src="https://github.com/tenstorrent/tutorial-assets/blob/main/media/tt_metal/scaleout/exabox/images/8x16_topology.png?raw=true" width="500"/>

*Chip-level topology: 4 Galaxy chassis in a 2×2 grid arrangement.*

<img src="https://github.com/tenstorrent/tutorial-assets/blob/main/media/tt_metal/scaleout/exabox/images/8x16_cabling.png?raw=true" width="700"/>

*Cabling web tool view: 4 Galaxies across 2 racks with inter-Galaxy cables crossing between racks in a distributed pattern.*

---

#### 16×8 (Quad Galaxy — 1×4 Column)

Four Galaxies stacked vertically in a single column. An alternative orientation to 8×16, using the same 4 Galaxies but with a different cabling pattern and host layout.

<img src="https://github.com/tenstorrent/tutorial-assets/blob/main/media/tt_metal/scaleout/exabox/images/16x8_topology.png?raw=true" width="400"/>

*Host layout: 4 Galaxy chassis stacked vertically (no inter-Galaxy cabling shown — see cabling web tool view below).*

<img src="https://github.com/tenstorrent/tutorial-assets/blob/main/media/tt_metal/scaleout/exabox/images/16x8_cabling.png?raw=true" width="700"/>

*Cabling web tool view: 4 Galaxies across 2 racks with a different inter-Galaxy cable pattern than 8×16.*

---

#### Multi-Pod (4×32 SuperPod)

Multiple pods can be connected into a SuperPod. The image below shows 4 pods in 4×32 topology (16 Galaxies total) in an exploded view — pods are spread apart to make inter-pod cables visible; in a physical deployment they are racked together.

<img src="https://github.com/tenstorrent/tutorial-assets/blob/main/media/tt_metal/scaleout/exabox/images/4_pods_4x32_topology.png" width="700"/>

*Exploded view of 4 pods (16 Galaxies). Each pod is 4 Galaxies in a line. Inter-pod cables are shown between the separated pods for clarity.*

### Current Deployment

The Exabox SuperPod uses **4×32 topology** for all 9 pods, optimized for video generation and low-latency decode workloads.

**Inter-pod connectivity** is workload-driven (not uniform 1:1 between nodes) — daisy-chain connections for decode pipelines with overlay connections for pod-to-pod communication.

For routing details, see [TT-Fabric Architecture](../../../tech_reports/TT-Fabric/TT-Fabric-Architecture.md).

## Prerequisites

Before starting, ensure you have:

- **SSH access** to all cluster hosts (existing and new)
- **NFS access** to shared config directories (e.g., `/data/scaleout_configs/`)
- **tt-metal repository** cloned and built (see [README.md](./README.md#prerequisites))

## Quick Reference

| Item | Location |
|------|----------|
| **Cutsheets (Internal)** | [Top Level Topologies Spreadsheet](https://docs.google.com/spreadsheets/d/1mnZJueW4BKZGNvrtGiCQZa12hxQbg49Aref2FDrRBsQ/edit?gid=0#gid=0) |
| Cabling Web Tool | https://aus2-cablegen.aus2.tenstorrent.com/ |
| 4x32 IntraPod Configs (Current) | `/data/scaleout_configs/4xBH_4x32_intrapod_updated/` |
| Legacy BH GLX Configs | `/data/scaleout_configs/bh_glx_exabox/` |
| Merged Output Location | `/data/scaleout_configs/` (requires sudo) |
| Docker Image (Known Good) | `ghcr.io/tenstorrent/tt-metal/upstream-tests-bh-glx:v0.66.0-dev20260115-28-g6eccf7061a` |

---

## Step 0: Get or Create Cutsheet

A **cutsheet** is a physical cabling instruction document (CSV/Excel) that tells technicians exactly which ports to connect. Before generating cabling descriptors, you need a cutsheet.

### Using a Standard Cutsheet

The [Top Level Topologies spreadsheet](https://docs.google.com/spreadsheets/d/1mnZJueW4BKZGNvrtGiCQZa12hxQbg49Aref2FDrRBsQ/edit?gid=0#gid=0) contains cabling guide CSVs and cabling descriptor TextProtos for all supported topologies and connectivity variants (mesh, torus-x, torus-y, torus-2d).

1. Find your topology and connectivity variant in the spreadsheet
2. Download the **Cabling Guide CSV** — this is the cutsheet for your technician
3. Optionally download the **Cabling Descriptor TextProto** — this can be used directly in Step 1 (skip the web tool import)
4. Provide the cutsheet CSV to your cabling technician

**Example cutsheet format:**

```csv
Source_Node,Source_Port,Dest_Node,Dest_Port,Cable_Length
C1U02,eth0,C1U08,eth4,3m
C1U02,eth1,C2U02,eth5,5m
...
```

The cutsheet specifies physical connections: which port on which node connects to which port on another node, plus cable length.

### Creating a Custom Cutsheet

If you need a non-standard topology:

1. **Option A: Use the cabling web tool** - Generate a cutsheet from the [Cabling Web Tool](https://aus2-cablegen.aus2.tenstorrent.com/)
2. **Option B: Edit manually** - Copy an existing cutsheet from the spreadsheet and modify as needed

**Key topology parameters for custom configurations:**

```
device_topology { dims: [ X, Y ] dim_types: [ RING|LINE, RING|LINE ] }
host_topology   { dims: [ Hx, Hy ] }
```

Where:
- `X × Y` = total chips in the cluster
- `Hx × Hy` = number of hosts (must divide evenly into X × Y)
- RING = torus (wrap-around), LINE = mesh (no wrap-around)

---

## Step 1: Generate Cabling Descriptor from Cutsheet

1. Open the cabling generator web tool:
   ```
   https://aus2-cablegen.aus2.tenstorrent.com/
   ```

2. **Force refresh** the page to ensure you have the latest version:
   - macOS: `Cmd + Shift + R`
   - Linux/Windows: `Ctrl + Shift + R`

3. Import your cutsheet (CSV format from the cabling team)

4. Export as a **cabling descriptor** (`.textproto` format)

5. Additionally export as a **deployment descriptor** (`.textproto` format, can be skipped if not adding new devices but rather connections between the devices that exist in the original deployment descriptor)

6. Save the exported file to a location accessible from your working host, e.g.:
   ```
   /data/<your-username>/new_cabling_descriptor.textproto
   /data/<your-username>/new_deployment_descriptor.textproto
   ```

## Step 2: Merge Configurations

Use the `merge_cluster_configs.py` script to combine the new cabling descriptor with the existing cluster configuration.

### With Existing Deployment Descriptor

If you're adding hosts to an existing deployment, merge both cabling and deployment descriptors:

```bash
./tools/scaleout/cabling_generator/merge_cluster_configs.py \
    --cabling1 /data/scaleout_configs/bh_glx_exabox/cabling_descriptor.textproto \
    --cabling2 /data/<your-username>/new_cabling_descriptor.textproto \
    --deployment1 /data/scaleout_configs/bh_glx_exabox/deployment_descriptor.textproto \
    --deployment2 /data/<your-username>/new_deployment_descriptor.textproto \
    --output-dir merged_output
```

### With Same Deployment (Intra-Cluster Cabling Only)

If you're only adding new cables between existing hosts (no new deployment):

```bash
./tools/scaleout/cabling_generator/merge_cluster_configs.py \
    --cabling1 /data/scaleout_configs/bh_glx_exabox/cabling_descriptor.textproto \
    --cabling2 /data/<your-username>/new_cabling_descriptor.textproto \
    --deployment1 /data/scaleout_configs/bh_glx_exabox/deployment_descriptor.textproto \
    --output-dir merged_output
```

### Output Files

The script generates in `merged_output/`:
- `merged_fsd.textproto` - Factory System Descriptor
- `merged_cabling_descriptor.textproto` - Combined cabling topology
- `merged_deployment_descriptor.textproto` - Combined host deployment

## Step 3: Deploy Merged Configuration

Copy the merged configuration to a shared location accessible by all cluster hosts:

```bash
sudo cp -r merged_output/ /data/scaleout_configs/<your-config-name>/
```

**Important:** The path must be accessible on all cluster nodes (typically via NFS).

Verify the files are in place:

```bash
ls -la /data/scaleout_configs/<your-config-name>/
```

## Step 4: Update Validation Scripts

Before running validation, update the script to use your new configuration files.

Edit the validation script for your topology (4x32 or 8x16):

```bash
# Check current configuration paths
grep "descriptor-path" ./tools/scaleout/exabox/run_validation_4x32.sh
```

Update the `--cabling-descriptor-path` and `--deployment-descriptor-path` arguments to point to your merged configs:

```bash
--cabling-descriptor-path /data/scaleout_configs/<your-config-name>/merged_cabling_descriptor.textproto
--deployment-descriptor-path /data/scaleout_configs/<your-config-name>/merged_deployment_descriptor.textproto
```

Alternatively, if you generated an FSD, update the scripts to use `--factory-descriptor-path` instead.

## Step 5: Run Physical Validation

Run 50 iterations of physical validation to verify hardware stability. Use the merged descriptors from Step 3.

See [Physical Validation](./README.md#physical-validation) in the README for validation script usage and result analysis.

For success criteria and interpreting results, see [Analyzing validation logs](./TROUBLESHOOTING.md#general-debugging-tips) in the Troubleshooting guide.

## Step 6: Run Fabric Tests (Optional)

After physical validation passes, run fabric tests to verify coordinated workloads across the mesh.

See [Fabric Tests](./README.md#fabric-tests) in the README for topology-specific scripts and usage.

## Troubleshooting

### Common Issues
For common issues with validation scripts, hardware failures, and cluster debugging, see the [Troubleshooting Guide](./TROUBLESHOOTING.md).

### Getting Help

- Report issues in `#exabox-infra` Slack channel
- Tag syseng team for hardware issues
- Tag scaleout team for topology/validation issues

## Related Documentation

- [Exabox README](./README.md) - Full hardware qualification workflow
- [Troubleshooting Guide](./TROUBLESHOOTING.md) - Common issues and solutions
- [Cabling Generator](../README.md) - How descriptors and FSD generation work
- [Cluster Validation Tools](../validation/README.md) - Understanding validation output
- [TT-Fabric Architecture](../../../tech_reports/TT-Fabric/TT-Fabric-Architecture.md) - Deep dive into mesh routing and topology concepts
