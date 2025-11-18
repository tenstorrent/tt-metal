# Single-Chip Usage - Quick Start

## The Answer: YES! Use Custom Mesh Graph Descriptors

You CAN use only one chip on N300/P300 systems without hangs or remote access.

## Solution in 3 Lines

```bash
# For P300 systems:
export TT_MESH_GRAPH_DESC_PATH="tt_metal/fabric/mesh_graph_descriptors/p300_single_chip_mesh_graph_descriptor.yaml"
export TT_VISIBLE_DEVICES="0,1"
pytest your_test.py

# For N300 systems:
export TT_MESH_GRAPH_DESC_PATH="tt_metal/fabric/mesh_graph_descriptors/n300_single_chip_mesh_graph_descriptor.yaml"
export TT_VISIBLE_DEVICES="0,1"
pytest your_test.py
```

## How It Works

| Setting | Purpose | Value |
|---------|---------|-------|
| `TT_MESH_GRAPH_DESC_PATH` | Defines **logical topology** | Custom [1, 1] descriptor |
| `TT_VISIBLE_DEVICES` | Sets **physical devices** to open | `"0,1"` (both on board) |
| **Result** | Physical: both chips opened via PCIe<br>Logical: only 1 device exposed | ✅ No hang, 1 device |

## Test It

```bash
cd ~/aperezvicente/tt-metal-apv

# For P300:
TT_MESH_GRAPH_DESC_PATH="tt_metal/fabric/mesh_graph_descriptors/p300_single_chip_mesh_graph_descriptor.yaml" \
TT_VISIBLE_DEVICES="0,1" \
python test_single_chip_mesh.py
```

Expected output:
```
✅ Mesh device created successfully!
   Number of devices in mesh: 1
   Device IDs: [0]

✅ SUCCESS: Only 1 device is exposed as expected!
```

## Use in Your Tests

### Option 1: Environment Variables (Easiest)

```bash
export TT_MESH_GRAPH_DESC_PATH="tt_metal/fabric/mesh_graph_descriptors/p300_single_chip_mesh_graph_descriptor.yaml"
export TT_VISIBLE_DEVICES="0,1"
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1"
```

### Option 2: Python Code

```python
import os
os.environ["TT_MESH_GRAPH_DESC_PATH"] = "tt_metal/fabric/mesh_graph_descriptors/p300_single_chip_mesh_graph_descriptor.yaml"
os.environ["TT_VISIBLE_DEVICES"] = "0,1"

import ttnn
mesh = ttnn.open_mesh_device(mesh_shape=(1, 1), device_ids=[0])
# Only 1 device will be in the mesh!
```

### Option 3: Pytest Fixture (Best for CI)

Create `conftest.py`:
```python
import pytest
import os

@pytest.fixture(scope="session", autouse=True)
def single_chip_config():
    os.environ["TT_MESH_GRAPH_DESC_PATH"] = "tt_metal/fabric/mesh_graph_descriptors/p300_single_chip_mesh_graph_descriptor.yaml"
    os.environ["TT_VISIBLE_DEVICES"] = "0,1"
```

## Why Both Devices in TT_VISIBLE_DEVICES?

**Remember:** On N300/P300, chips share PCIe connection:
- Setting `TT_VISIBLE_DEVICES="0"` → Chip 1 accessed remotely via Ethernet → **HANGS** ❌
- Setting `TT_VISIBLE_DEVICES="0,1"` → Both opened via PCIe → **Works** ✅
- Custom mesh descriptor → Only 1 device exposed logically → **Perfect!** ✅✅✅

## Files Created

- **Mesh Descriptors:**
  - `tt_metal/fabric/mesh_graph_descriptors/p300_single_chip_mesh_graph_descriptor.yaml`
  - `tt_metal/fabric/mesh_graph_descriptors/n300_single_chip_mesh_graph_descriptor.yaml`

- **Documentation:**
  - `SINGLE_CHIP_USAGE_GUIDE.md` - Complete guide
  - `SINGLE_CHIP_QUICK_START.md` - This file

- **Test Scripts:**
  - `test_single_chip_mesh.py` - Verification script

## Multiple Boards

Run different tests on different boards simultaneously:

```bash
# Terminal 1: Board 1 (chips 0,1) - use only chip 0
TT_MESH_GRAPH_DESC_PATH="tt_metal/fabric/mesh_graph_descriptors/p300_single_chip_mesh_graph_descriptor.yaml" \
TT_VISIBLE_DEVICES="0,1" \
TT_METAL_CACHE="~/.cache/tt_metal_board1" \
pytest test1.py

# Terminal 2: Board 2 (chips 2,3) - use only chip 2
TT_MESH_GRAPH_DESC_PATH="tt_metal/fabric/mesh_graph_descriptors/p300_single_chip_mesh_graph_descriptor.yaml" \
TT_VISIBLE_DEVICES="2,3" \
TT_METAL_CACHE="~/.cache/tt_metal_board2" \
pytest test2.py
```

## Summary

✅ **YES, you can use only one chip!**
✅ **No hangs, no remote access**
✅ **Works with both N300 and P300**
✅ **Ready for pytest integration**

📖 See `SINGLE_CHIP_USAGE_GUIDE.md` for complete documentation.
