# Cluster Bringup Guide

This guide covers how to add new hardware to an existing Exabox cluster by merging cabling descriptors and validating the expanded configuration.

**Use case:** You've physically installed and cabled new Galaxy nodes and need to integrate them into the cluster's logical topology.

## Introduction

### Cluster Hierarchy

```
SuperPod (9 pods)
  └─ Pod (4 Galaxies = 128 chips)
       └─ Galaxy (1 host = 32 Blackhole chips)
```

A **Galaxy** is a 6U server containing 32 Blackhole chips arranged in an 8×4 mesh. A **Pod** combines 4 Galaxies (128 chips) into a unified compute mesh using TT-Fabric. A **SuperPod** connects 9 pods for large-scale workloads.

### Pod Topologies

How the 4 Galaxies within a pod are arranged determines the mesh shape:

| Topology | Arrangement | Mesh Shape | Use Case |
|----------|-------------|------------|----------|
| **4x32** | 4×1 line | 32 cols × 4 rows | Workloads with 1D data flow: video generation (Wan2.1/2.2), low-latency decode (Blitz, DeepSeek), ring allreduce |
| **8x16** | 2×2 grid | 16 cols × 8 rows | Workloads with 2D data flow: all-to-all collectives, 2D tensor parallelism |

```
4x32: ┌───┬───┬───┬───┐     8x16: ┌───┬───┐
      │ 0 │ 1 │ 2 │ 3 │           │ 0 │ 1 │
      └───┴───┴───┴───┘           ├───┼───┤
                                  │ 2 │ 3 │
                                  └───┴───┘
```

Within a pod, chips connect via **2D Torus** (wrap-around on both X and Y), enabling shorter hop counts and multiple routing paths.

### Current Deployment

The Exabox SuperPod uses **4x32 topology** for all 9 pods, optimized for:
- **Video Generation** (Wan2.1/2.2) - pipeline parallelism along the elongated X dimension
- **Low-Latency LLM Inference** (Blitz Decode, DeepSeek) - efficient ring communication

**Inter-pod connectivity** is workload-driven (not uniform 1:1 between nodes):
- Daisy-chain loopback connections for decode pipelines
- Overlay connections enabling 9-way all-to-all pod communication
- Cable distribution optimized for target workloads rather than symmetric distribution

For routing details, see [TT-Fabric Architecture](../../../tech_reports/TT-Fabric/TT-Fabric-Architecture.md).

## Prerequisites

Before starting, ensure you have:

- **SSH access** to all cluster hosts (existing and new)
- **NFS access** to shared config directories (e.g., `/data/scaleout_configs/`)
- **tt-metal repository** cloned and built (see [README.md](./README.md#prerequisites))
- **Cutsheet** for the new hardware (provided by the cabling team)

## Quick Reference

| Item | Location |
|------|----------|
| Cabling Web Tool | https://aus2-cablegen.aus2.tenstorrent.com/ |
| 4x32 IntraPod Configs (Current) | `/data/scaleout_configs/4xBH_4x32_intrapod_updated/` |
| Legacy BH GLX Configs | `/data/scaleout_configs/bh_glx_exabox/` |
| Merged Output Location | `/data/scaleout_configs/` (requires sudo) |
| Docker Image (Known Good) | `ghcr.io/tenstorrent/tt-metal/upstream-tests-bh-glx:v0.66.0-dev20260115-28-g6eccf7061a` |

---

## Step 0: Plan Your Topology

Before generating cabling descriptors, decide on your target topology.

### For Standard Configurations

Use an existing topology if your cluster matches a known configuration:

| Cluster Size | Topology Options | MGD File |
|-------------|------------------|----------|
| 4 Galaxies (128 chips) | 8x16 or 4x32 | `16x8_quad_bh_galaxy_torus_xy_graph_descriptor.textproto` or `32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto` |
| 1 Galaxy (32 chips) | Single mesh | `single_bh_galaxy_torus_xy_graph_descriptor.textproto` |

MGD files are located in `tt_metal/fabric/mesh_graph_descriptors/`.

### For Custom Configurations

If you need a non-standard topology:

1. **Define the mesh dimensions**: Determine `device_topology` (total chip grid) and `host_topology` (how hosts map to the grid)
2. **Choose connectivity type**: RING (torus) or LINE (mesh with edges) for each dimension
3. **Consider workload requirements**: Match the topology to your communication patterns
4. **Generate the cabling descriptor**: The cabling web tool will produce cables matching your topology

**Key topology parameters:**

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
