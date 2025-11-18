# Single-Chip Usage on N300/P300 Systems

## The Solution: Custom Mesh Graph Descriptors

To use **only one chip** on N300 or P300 systems, use custom mesh graph descriptors that define a [1, 1] topology.

## Quick Start

### For P300 Systems

```bash
# Set mesh graph descriptor to single-chip mode
export TT_MESH_GRAPH_DESC_PATH="tt_metal/fabric/mesh_graph_descriptors/p300_single_chip_mesh_graph_descriptor.yaml"

# Still set visible devices to both chips on the board (to avoid remote access hang)
export TT_VISIBLE_DEVICES="0,1"

# Run your test - only 1 chip will be exposed logically
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1"
```

### For N300 Systems

```bash
# Set mesh graph descriptor to single-chip mode
export TT_MESH_GRAPH_DESC_PATH="tt_metal/fabric/mesh_graph_descriptors/n300_single_chip_mesh_graph_descriptor.yaml"

# Still set visible devices to both chips on the board
export TT_VISIBLE_DEVICES="0,1"

# Run your test - only 1 chip will be exposed logically
pytest your_test.py
```

## How It Works

### Physical vs Logical Topology

```
PHYSICAL TOPOLOGY (P300 Board):
┌─────────────────────────────────┐
│  Chip 0     Chip 1              │
│  (PCIe)     (PCIe)              │
│    │          │                 │
│    └──────┬───┘                 │
│           │                     │
│       PCIe Bus                  │
└─────────────────────────────────┘

Without custom descriptor:
  TT_VISIBLE_DEVICES="0" → Tries remote access → HANGS ❌

With custom descriptor:
  TT_VISIBLE_DEVICES="0,1" → Logical topology [1, 1] → Uses only 1 chip ✅
```

### What the Descriptor Does

- **Physical Layer:** Both chips are opened via PCIe (no remote access, no hang)
- **Logical Layer:** Only 1 chip is exposed to your application
- **Result:** Your code sees and uses only 1 device, initialization doesn't hang

## Usage in Python Code

```python
import os
import ttnn

# Set before importing ttnn
os.environ["TT_MESH_GRAPH_DESC_PATH"] = "tt_metal/fabric/mesh_graph_descriptors/p300_single_chip_mesh_graph_descriptor.yaml"
os.environ["TT_VISIBLE_DEVICES"] = "0,1"

# Create mesh device - will have only 1 device
mesh_device = ttnn.open_mesh_device(
    mesh_shape=(1, 1),  # 1x1 mesh
    device_ids=[0],     # Only device 0 is exposed
)

print(f"Mesh has {len(mesh_device.get_devices())} device(s)")  # Output: 1

# Use the single device
device = mesh_device.get_devices()[0]
# Your workload here...

ttnn.close_mesh_device(mesh_device)
```

## Usage in Pytest

### Method 1: Environment Variables

```bash
export TT_MESH_GRAPH_DESC_PATH="tt_metal/fabric/mesh_graph_descriptors/p300_single_chip_mesh_graph_descriptor.yaml"
export TT_VISIBLE_DEVICES="0,1"
pytest your_test.py
```

### Method 2: Pytest Fixture

Create a conftest.py:

```python
import pytest
import os

@pytest.fixture(scope="session", autouse=True)
def single_chip_mode():
    """Force single-chip mode for all tests."""
    os.environ["TT_MESH_GRAPH_DESC_PATH"] = "tt_metal/fabric/mesh_graph_descriptors/p300_single_chip_mesh_graph_descriptor.yaml"
    os.environ["TT_VISIBLE_DEVICES"] = "0,1"
    yield
```

### Method 3: Command Line Override

```bash
pytest your_test.py \
  --override-env TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p300_single_chip_mesh_graph_descriptor.yaml \
  --override-env TT_VISIBLE_DEVICES=0,1
```

## Comparison: Different Approaches

| Approach | TT_VISIBLE_DEVICES | Mesh Descriptor | Result | Status |
|----------|-------------------|-----------------|--------|--------|
| **Wrong** | `"0"` | Default (1x2) | Remote access hang | ❌ HANGS |
| **Workaround** | `"0,1"` | Default (1x2) | Both chips init, use only 0 in code | ✅ Works but wasteful |
| **Best** | `"0,1"` | Custom (1x1) | Both init, only 1 exposed logically | ✅ **BEST** |

## Running Multiple Single-Chip Tests Concurrently

You can run independent tests on different boards:

**Terminal 1: Test on Board 1 (Chip 0)**
```bash
export TT_MESH_GRAPH_DESC_PATH="tt_metal/fabric/mesh_graph_descriptors/p300_single_chip_mesh_graph_descriptor.yaml"
export TT_VISIBLE_DEVICES="0,1"
export TT_METAL_CACHE="$HOME/.cache/tt_metal_board1"
pytest test_board1.py
```

**Terminal 2: Test on Board 2 (Chip 2)**
```bash
export TT_MESH_GRAPH_DESC_PATH="tt_metal/fabric/mesh_graph_descriptors/p300_single_chip_mesh_graph_descriptor.yaml"
export TT_VISIBLE_DEVICES="2,3"
export TT_METAL_CACHE="$HOME/.cache/tt_metal_board2"
pytest test_board2.py
```

## Verification Script

Test that single-chip mode works:

```bash
export TT_MESH_GRAPH_DESC_PATH="tt_metal/fabric/mesh_graph_descriptors/p300_single_chip_mesh_graph_descriptor.yaml"
export TT_VISIBLE_DEVICES="0,1"

python -c "
import ttnn
import os

print(f'Mesh descriptor: {os.environ.get(\"TT_MESH_GRAPH_DESC_PATH\")}')
print(f'Visible devices: {os.environ.get(\"TT_VISIBLE_DEVICES\")}')

mesh = ttnn.open_mesh_device(mesh_shape=(1, 1), device_ids=[0])
devices = mesh.get_devices()

print(f'✅ Mesh created with {len(devices)} device(s)')
print(f'   Device IDs: {[d.id() for d in devices]}')

ttnn.close_mesh_device(mesh)
print('✅ Success!')
"
```

Expected output:
```
Mesh descriptor: tt_metal/fabric/mesh_graph_descriptors/p300_single_chip_mesh_graph_descriptor.yaml
Visible devices: 0,1
✅ Mesh created with 1 device(s)
   Device IDs: [0]
✅ Success!
```

## Technical Details

### Mesh Graph Descriptor Format

**Single-chip (1x1):**
```yaml
ChipSpec: {
  arch: blackhole,
  ethernet_ports: {
    N: 0,  # No ethernet connections needed
    E: 0,
    S: 0,
    W: 0,
  }
}

Board: [
  { name: P300_SINGLE,
    type: Mesh,
    topology: [1, 1]}  # 1 row, 1 column = 1 chip
]

Mesh: [
{
  id: 0,
  board: P300_SINGLE,
  device_topology: [1, 1],  # Logical: 1 device
  host_topology: [1, 1],     # Physical: 1 host
}
]
```

**Dual-chip (1x2) - Default:**
```yaml
Board: [
  { name: P300,
    type: Mesh,
    topology: [1, 2]}  # 1 row, 2 columns = 2 chips
]
```

### Why This Works

1. **TT_VISIBLE_DEVICES="0,1"** tells UMD to open both chips via PCIe
2. **Mesh descriptor [1, 1]** tells the mesh layer to expose only 1 logical device
3. Physical initialization succeeds (both chips via PCIe)
4. Logical topology shows only 1 device to your application
5. No remote access, no hangs, clean single-device usage

## Troubleshooting

### Issue: Still seeing 2 devices

**Check:**
```bash
echo $TT_MESH_GRAPH_DESC_PATH
# Should show: tt_metal/fabric/mesh_graph_descriptors/p300_single_chip_mesh_graph_descriptor.yaml
```

**Verify the descriptor exists:**
```bash
cat $TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p300_single_chip_mesh_graph_descriptor.yaml
```

### Issue: Test still hangs

**Make sure both chips on the board are in TT_VISIBLE_DEVICES:**
```bash
# ❌ Wrong - will hang
export TT_VISIBLE_DEVICES="0"

# ✅ Correct
export TT_VISIBLE_DEVICES="0,1"
```

### Issue: "Mesh shape mismatch" error

**Make sure your code requests [1, 1] mesh:**
```python
# ✅ Correct
mesh = ttnn.open_mesh_device(mesh_shape=(1, 1), device_ids=[0])

# ❌ Wrong - requesting 1x2 mesh
mesh = ttnn.open_mesh_device(mesh_shape=(1, 2), device_ids=[0, 1])
```

## Summary

To use **only one chip** on N300/P300:

1. ✅ Use custom mesh graph descriptor (topology [1, 1])
2. ✅ Set `TT_VISIBLE_DEVICES` to both chips on board (avoid remote access)
3. ✅ Your application sees and uses only 1 device
4. ✅ No hangs, clean initialization

**One-liner for pytest:**
```bash
TT_MESH_GRAPH_DESC_PATH="tt_metal/fabric/mesh_graph_descriptors/p300_single_chip_mesh_graph_descriptor.yaml" \
TT_VISIBLE_DEVICES="0,1" \
pytest your_test.py
```
