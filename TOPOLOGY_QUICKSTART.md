# Topology Tools - Quick Start Guide

## ✅ Tools Created for You

I've created 4 new tools to visualize and manage topologies:

### 1. `visualize_topology_3d_enhanced.py` - ⭐ **NEW!** Enhanced 3D with Board Boxes
### 2. `visualize_topology.py` - Interactive 3D & 2D Visualization
### 3. `generate_mesh_descriptor.py` - Auto-generate Mesh Descriptors
### 4. Updated `simple_link_check.py` - Fixed to handle both descriptor formats

## 🚀 Quick Demo (Already Generated!)

I've already run these for your T3K topology:

```bash
# ✅ Generated mesh descriptor
./generate_mesh_descriptor.py --cluster t3k_phys_topology.yaml --shape 2,4 \\
    --output t3k_generated_mesh.yaml --compare

# ✅ Created visualizations
./visualize_topology.py --cluster t3k_phys_topology.yaml \\
    --mesh t3k_generated_mesh.yaml --both

# ✅ Created ENHANCED 3D with board boxes
./visualize_topology_3d_enhanced.py --cluster t3k_phys_topology.yaml \\
    --mesh t3k_generated_mesh.yaml --output topology_3d_with_boards.html
```

**Generated Files**:
- `t3k_generated_mesh.yaml` (237 bytes) - Logical mesh descriptor
- `topology_3d_with_boards.html` (3.5 MB) - ⭐ **Enhanced 3D with Board Boxes!**
- `topology_3d.html` (3.5 MB) - Standard interactive 3D view
- `topology_2d.png` (153 KB) - High-res 2D diagram

## 📊 View Your 3D Topology

### ⭐ Enhanced 3D (Recommended!)

Shows boards as 3D boxes, chips inside boards, and mesh overlay:

```bash
# Copy to your local machine
scp user@remote:/workspace/tt-metal-apv/topology_3d_with_boards.html .
# Then open locally
open topology_3d_with_boards.html
```

**What you'll see:**
- 📦 **Green transparent boxes** = N300 boards
- 🔵 **Blue cubes inside** = Wormhole chips (2 per N300)
- 🔗 **Gray lines** = Ethernet connections
- 🔴 **Red transparent box** = Logical 2×4 mesh overlay

### Standard 3D

Simpler view without board enclosures:

```bash
# Open in browser
firefox topology_3d.html
# or
google-chrome topology_3d.html
```

The 3D views are fully interactive:
- **Rotate**: Left-click and drag
- **Zoom**: Scroll wheel
- **Pan**: Right-click and drag
- **Hover**: See chip details and connections

## 🎯 Your Use Cases

### 1. Display Physical Topology

```bash
# Enhanced 3D with board boxes (BEST!)
./visualize_topology_3d_enhanced.py --cluster t3k_phys_topology.yaml

# Standard 3D interactive
./visualize_topology.py --cluster t3k_phys_topology.yaml --3d

# 2D image
./visualize_topology.py --cluster t3k_phys_topology.yaml \\
    --output-2d my_topology.png
```

### 2. Display Physical + Logical Overlay

```bash
# Enhanced 3D with boards AND mesh (RECOMMENDED!)
./visualize_topology_3d_enhanced.py \\
    --cluster t3k_phys_topology.yaml \\
    --mesh t3k_generated_mesh.yaml

# Standard 2D/3D with mesh overlay
./visualize_topology.py \\
    --cluster t3k_phys_topology.yaml \\
    --mesh t3k_generated_mesh.yaml \\
    --both
```

The **enhanced** visualization shows:
- **📦 Colored transparent boxes**: Physical boards (N300, Galaxy, etc.)
- **🔵 Blue cubes**: Physical chips inside boards
- **🔗 Gray lines**: Ethernet connections with channel numbers
- **🔴 Red transparent box**: Logical mesh boundary overlay

The **standard** visualization shows:
- **Blue boxes/spheres**: Physical chips
- **Gray lines**: Ethernet connections
- **Red dashed outline**: Logical mesh boundary

### 3. Generate Mesh Descriptor for Any Shape

```bash
# For 2x4 mesh (T3K)
./generate_mesh_descriptor.py --cluster my_phys_topology.yaml \\
    --shape 2,4 --output my_mesh.yaml

# Auto-infer from chip count
./generate_mesh_descriptor.py --cluster my_phys_topology.yaml \\
    --auto --output my_mesh.yaml

# With comparison report
./generate_mesh_descriptor.py --cluster my_phys_topology.yaml \\
    --shape 2,4 --output my_mesh.yaml --compare
```

### 4. Test on Hardware (Now Fixed!)

```bash
# The script now correctly uses both descriptors
./simple_link_check.py --yaml t3k_phys_topology.yaml
```

It automatically:
- Sets `TT_METAL_CLUSTER_DESC_PATH` for physical topology
- Sets `TT_MESH_GRAPH_DESC_PATH` for logical mesh
- Tests ethernet connectivity

## 📁 File Formats Explained

You have **2 different descriptor formats**:

### Physical Topology (Cluster Descriptor)
**File**: `t3k_phys_topology.yaml`
**Format**: Old UMD format
**Contains**: Actual hardware - chip IDs, physical connections, PCIe buses
**Used for**: Hardware initialization, actual chip communication
**Env var**: `TT_METAL_CLUSTER_DESC_PATH`

### Logical Mesh (Mesh Graph Descriptor)
**File**: `t3k_generated_mesh.yaml`
**Format**: New fabric format with `ChipSpec`, `Board`, `Mesh`
**Contains**: Logical topology - mesh shape, routing rules
**Used for**: Fabric layer routing, mesh operations
**Env var**: `TT_MESH_GRAPH_DESC_PATH`

## 🔧 How They Work Together

```
Physical Topology (t3k_phys_topology.yaml)
  ↓
  Describes actual hardware connections
  ↓
  Used by UMD layer

Logical Mesh (t3k_generated_mesh.yaml)
  ↓
  Describes logical 2×4 mesh
  ↓
  Used by fabric/routing layer

Both together = Full system configuration!
```

## 💡 Complete Workflow Example

```bash
# 1. Generate physical topology from hardware
./build/tools/umd/topology -f my_cluster.yaml

# 2. Visualize what you have
./visualize_topology.py --cluster my_cluster.yaml --3d

# 3. Generate logical mesh for your desired shape
./generate_mesh_descriptor.py --cluster my_cluster.yaml \\
    --shape 2,4 --output my_mesh.yaml --compare

# 4. Visualize both together
./visualize_topology.py --cluster my_cluster.yaml \\
    --mesh my_mesh.yaml --both

# 5. Test on hardware
./simple_link_check.py --yaml my_cluster.yaml

# 6. Use in your application
export TT_METAL_CLUSTER_DESC_PATH=my_cluster.yaml
export TT_MESH_GRAPH_DESC_PATH=my_mesh.yaml
python your_app.py
```

## 🐛 Troubleshooting

### "MeshGraph: Expecting yaml to define a ChipSpec"
❌ You're using physical topology where logical mesh is expected

✅ Solution: Generate mesh descriptor
```bash
./generate_mesh_descriptor.py --cluster your_cluster.yaml --auto --output mesh.yaml
export TT_MESH_GRAPH_DESC_PATH=mesh.yaml
```

### Want to see what's inside?
```bash
# View physical topology
cat t3k_phys_topology.yaml | grep -A5 "ethernet_connections"

# View logical mesh
cat t3k_generated_mesh.yaml
```

### Need different mesh shape?
```bash
# Change from 2x4 to 4x2 (for example)
./generate_mesh_descriptor.py --cluster t3k_phys_topology.yaml \\
    --shape 4,2 --output t3k_4x2_mesh.yaml
```

## 📚 More Info

See `TOPOLOGY_TOOLS_README.md` for:
- Detailed documentation
- All command options
- File format specifications
- Advanced use cases

## 🎉 What You Have Now

✅ **⭐ Enhanced 3D with boards** (`topology_3d_with_boards.html`) - **BEST!**
✅ **Interactive 3D visualization** (`topology_3d.html`)
✅ **2D diagram** (`topology_2d.png`)
✅ **Generated mesh descriptor** (`t3k_generated_mesh.yaml`)
✅ **Working test script** (`simple_link_check.py`)
✅ **Tools for any topology** (`visualize_topology_3d_enhanced.py`, `visualize_topology.py`, `generate_mesh_descriptor.py`)

**Open `topology_3d_with_boards.html` in a browser to see:**
- Your T3K as 4 N300 boards (green 3D boxes)
- 8 Wormhole chips inside (2 per board, blue cubes)
- All ethernet connections (gray lines)
- Logical 2×4 mesh overlay (red 3D box)

All fully interactive - rotate, zoom, explore! 🚀

See `ENHANCED_3D_VISUALIZATION.md` for detailed guide.
