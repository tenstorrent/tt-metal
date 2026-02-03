# Cluster Bringup Guide

This guide covers how to add new hardware to an existing Exabox cluster by merging cabling descriptors and validating the expanded configuration.

**Use case:** You've physically installed and cabled new Galaxy nodes and need to integrate them into the cluster's logical topology.

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
| Existing BH GLX Configs | `/data/scaleout_configs/bh_glx_exabox/` |
| Merged Output Location | `/data/scaleout_configs/` (requires sudo) |
| Docker Image (Known Good) | `ghcr.io/tenstorrent/tt-metal/upstream-tests-bh-glx:v0.66.0-dev20260115-28-g6eccf7061a` |

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

5. Save the exported file to a location accessible from your working host, e.g.:
   ```
   /data/<your-username>/new_cabling_descriptor.textproto
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
    --deployment2 /data/scaleout_configs/bh_glx_exabox/deployment_descriptor.textproto \
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
grep "descriptor-path" ./tools/scaleout/exabox/run_validation.sh
```

Update the `--cabling-descriptor-path` and `--deployment-descriptor-path` arguments to point to your merged configs:

```bash
--cabling-descriptor-path /data/scaleout_configs/<your-config-name>/merged_cabling_descriptor.textproto
--deployment-descriptor-path /data/scaleout_configs/<your-config-name>/merged_deployment_descriptor.textproto
```

Alternatively, if you generated an FSD, update the scripts to use `--fsd-path` instead.

## Step 5: Run Physical Validation

Run 50 iterations of physical validation to verify hardware stability.

### For 4x32 Topology

```bash
./tools/scaleout/exabox/run_validation.sh \
    bh-glx-c05u02,bh-glx-c05u08,bh-glx-c06u02,bh-glx-c06u08 \
    ghcr.io/tenstorrent/tt-metal/upstream-tests-bh-glx:v0.66.0-dev20260115-28-g6eccf7061a
```

**Note:** Replace the host list with all hosts in your expanded cluster (comma-separated, no spaces).

Logs are written to `validation_output/` in your current directory.

## Step 6: Analyze Results

After validation completes, analyze the results to verify the cluster meets the stability threshold.

### Quick Analysis (Shell Script)

```bash
./tools/scaleout/exabox/analyze_validation_results.sh validation_output/
```

### Detailed Analysis (Python Script)

For more detailed analysis with plots:

```bash
python3 ./tools/scaleout/exabox/analyze_validation_results.py \
    validation_output/ \
    --plot \
    --plot-dir ./analysis_output
```

### Success Criteria

| Success Rate | Status | Action |
|-------------|--------|--------|
| 80%+ (40/50) | Pass | Cluster is ready for workloads |
| 70-79% | Marginal | Review failure patterns, may need cable investigation |
| <70% | Fail | Investigate failures before proceeding |

For detailed baseline expectations, see the [Physical Validation Results spreadsheet](https://docs.google.com/spreadsheets/d/1lg6cG0TovYqwJtn6kkm5Fb1erQHH9p-p8NIUL279V0U/edit?pli=1&gid=489670889#gid=489670889).

## Step 7: Run Fabric Tests (Optional)

After physical validation passes, run fabric tests to verify coordinated workloads:

```bash
./tools/scaleout/exabox/run_fabric_tests_4x32.sh \
    bh-glx-c05u02,bh-glx-c05u08,bh-glx-c06u02,bh-glx-c06u08 \
    ghcr.io/tenstorrent/tt-metal/upstream-tests-bh-glx:v0.66.0-dev20260115-28-g6eccf7061a
```

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Missing connections | Cables unseated or wrong FSD | Verify cables, check FSD matches physical topology |
| Merge fails | Incompatible descriptors | Verify both descriptors use same node types |
| Low pass rate | Hardware issues | See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) |
| Permission denied on NFS | Wrong permissions | Use `sudo` or contact infra team |

### Getting Help

- Report issues in `#exabox-infra` Slack channel
- Tag syseng team for hardware issues
- Tag scaleout team for topology/validation issues

See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for detailed troubleshooting steps.

## Related Documentation

- [Exabox README](./README.md) - Full hardware qualification workflow
- [Troubleshooting Guide](./TROUBLESHOOTING.md) - Common issues and solutions
- [Cabling Generator](../README.md) - How descriptors and FSD generation work
- [Cluster Validation Tools](../validation/README.md) - Understanding validation output
