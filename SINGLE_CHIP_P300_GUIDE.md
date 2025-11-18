# Running on a Single P300 Chip - Complete Guide

## Problem

On P300 systems, chips are paired on the same PCB and connected via both PCIe and Ethernet:
- When you set `TT_VISIBLE_DEVICES="0"`, UMD discovers chip 1 via Ethernet and tries to access it remotely → **HANGS**
- Even with `MESH_DEVICE=P150`, both chips may still be initialized by UMD

## Solution: Custom Physical Topology

Disable Ethernet discovery by using a **custom physical topology file** that removes Ethernet connections.

## Files Created

### For Chip 0 Only
`p300_single_chip_0_phys_topology.yaml` - Physical topology with only chip 0, no Ethernet

### For Chip 1 Only
`p300_single_chip_1_phys_topology.yaml` - Physical topology with only chip 1, no Ethernet

## Usage

### Run Test on Chip 0 Only

```bash
cd ~/aperezvicente/tt-metal-apv

# Use custom physical topology (no Ethernet connections)
export TT_METAL_CLUSTER_DESC_PATH="p300_single_chip_0_phys_topology.yaml"

# Use single-chip logical mesh
export TT_MESH_GRAPH_DESC_PATH="tt_metal/fabric/mesh_graph_descriptors/p300_single_chip_mesh_graph_descriptor.yaml"

# Request only chip 0 (no hang because Ethernet is disabled in topology)
export TT_VISIBLE_DEVICES="0"

# Optional: Force P150 mode for safety
export MESH_DEVICE=P150

# Run the test
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1"
```

### Run Test on Chip 1 Only

```bash
# Use custom physical topology for chip 1
export TT_METAL_CLUSTER_DESC_PATH="p300_single_chip_1_phys_topology.yaml"

# Use single-chip logical mesh
export TT_MESH_GRAPH_DESC_PATH="tt_metal/fabric/mesh_graph_descriptors/p300_single_chip_mesh_graph_descriptor.yaml"

# Request only chip 1
export TT_VISIBLE_DEVICES="1"

# Optional: Force P150 mode
export MESH_DEVICE=P150

# Run the test
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1"
```

## How It Works

### Normal P300 Topology (2xp300_phys_topology.yaml)
```yaml
ethernet_connections:
  - - {chip: 0, chan: 2}
    - {chip: 1, chan: 9}
  - - {chip: 0, chan: 3}
    - {chip: 1, chan: 8}
  # ... more connections
```
❌ UMD discovers chip 1 when chip 0 is opened → tries remote access → **hang**

### Custom Single-Chip Topology (p300_single_chip_0_phys_topology.yaml)
```yaml
ethernet_connections: []
chips_with_mmio:
  - 0: 1
```
✅ NO Ethernet connections → UMD only knows about chip 0 → **no hang**

## Environment Variables Summary

| Variable | Purpose | Value |
|----------|---------|-------|
| `TT_METAL_CLUSTER_DESC_PATH` | Physical topology (UMD layer) | `p300_single_chip_0_phys_topology.yaml` |
| `TT_MESH_GRAPH_DESC_PATH` | Logical mesh (Fabric layer) | `tt_metal/fabric/.../p300_single_chip_mesh_graph_descriptor.yaml` |
| `TT_VISIBLE_DEVICES` | Which chip to use | `"0"` or `"1"` |
| `MESH_DEVICE` | Override mesh shape (optional) | `P150` |

## Quick Test Script

```bash
#!/bin/bash
# test_single_chip.sh

CHIP_ID=${1:-0}  # Default to chip 0

echo "Testing on P300 Chip ${CHIP_ID}"

export TT_METAL_CLUSTER_DESC_PATH="p300_single_chip_${CHIP_ID}_phys_topology.yaml"
export TT_MESH_GRAPH_DESC_PATH="tt_metal/fabric/mesh_graph_descriptors/p300_single_chip_mesh_graph_descriptor.yaml"
export TT_VISIBLE_DEVICES="${CHIP_ID}"
export MESH_DEVICE=P150

pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1"
```

Usage:
```bash
chmod +x test_single_chip.sh

# Test on chip 0
./test_single_chip.sh 0

# Test on chip 1
./test_single_chip.sh 1
```

## Running Multiple Tests Concurrently

You can now run independent tests on each chip simultaneously:

**Terminal 1: Chip 0**
```bash
export TT_METAL_CLUSTER_DESC_PATH="p300_single_chip_0_phys_topology.yaml"
export TT_MESH_GRAPH_DESC_PATH="tt_metal/fabric/mesh_graph_descriptors/p300_single_chip_mesh_graph_descriptor.yaml"
export TT_VISIBLE_DEVICES="0"
export TT_METAL_CACHE="~/.cache/tt_metal_chip0"
pytest test1.py
```

**Terminal 2: Chip 1**
```bash
export TT_METAL_CLUSTER_DESC_PATH="p300_single_chip_1_phys_topology.yaml"
export TT_MESH_GRAPH_DESC_PATH="tt_metal/fabric/mesh_graph_descriptors/p300_single_chip_mesh_graph_descriptor.yaml"
export TT_VISIBLE_DEVICES="1"
export TT_METAL_CACHE="~/.cache/tt_metal_chip1"
pytest test2.py
```

## Verification

Test that it works:

```bash
# Should NOT hang
export TT_METAL_CLUSTER_DESC_PATH="p300_single_chip_0_phys_topology.yaml"
export TT_VISIBLE_DEVICES="0"

python -c "
import ttnn
print('Opening device 0...')
device = ttnn.create_device(device_id=0)
print(f'✅ Success! Device {device.id()} is open')
print(f'   MMIO capable: {device.is_mmio_capable()}')
ttnn.close_device(device)
print('✅ Clean exit - no hang!')
"
```

## What Changed vs Original Approach

| Approach | Layer | What It Does | Result |
|----------|-------|--------------|--------|
| **TT_VISIBLE_DEVICES="0"** (original) | UMD | Request chip 0 | ❌ UMD discovers chip 1 via Ethernet → hang |
| **MESH_DEVICE=P150** | Application | Request 1x1 mesh | ⚠️ Helps but UMD still opens both chips |
| **TT_METAL_CLUSTER_DESC_PATH** (new) | UMD | Override physical topology | ✅ **Disables Ethernet discovery** |

The custom cluster descriptor fixes the problem **at the UMD layer** where the issue originates.

## Technical Details

### Why This Works

1. **UMD reads cluster descriptor on startup**
2. **Custom descriptor has `ethernet_connections: []`**
3. **UMD doesn't discover chip 1 at all**
4. **Only chip 0 exists in the system from UMD's perspective**
5. **No remote access, no hang**

### Relationship to Other Descriptors

```
Physical Topology (TT_METAL_CLUSTER_DESC_PATH)
  ↓ Tells UMD which chips exist and how they're connected
  ↓
UMD Layer Opens Chips
  ↓
Logical Mesh (TT_MESH_GRAPH_DESC_PATH)
  ↓ Tells Fabric how to route between chips
  ↓
Application Layer (MESH_DEVICE)
  ↓ Tells your code how many devices to use
```

All three must align for single-chip operation.

## Troubleshooting

### Still seeing both chips?
Check that the environment variable is set correctly:
```bash
echo $TT_METAL_CLUSTER_DESC_PATH
# Should output: p300_single_chip_0_phys_topology.yaml
```

### File not found error?
Use absolute path:
```bash
export TT_METAL_CLUSTER_DESC_PATH="$PWD/p300_single_chip_0_phys_topology.yaml"
```

### Want to use Board 2 (chips 2 and 3)?
Create similar topology files for chips 2 and 3 by copying the pattern from the original `2xp300_phys_topology.yaml` file.

## Summary

✅ **This is the correct solution** to run on a single P300 chip!

- Disables Ethernet at the UMD level
- Prevents remote chip discovery
- No hangs with `TT_VISIBLE_DEVICES="0"`
- True single-chip operation

Use the provided topology files and environment variables to run tests on individual chips without any remote access or initialization conflicts.
