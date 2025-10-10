# Enhanced 3D Topology Visualization 🎨

## 🎉 New Enhanced Visualizer!

I've created `visualize_topology_3d_enhanced.py` - a much more sophisticated 3D visualization tool that shows:

### ✨ Features

1. **📦 Board Enclosures**
   - Each board (N300, N150, P150, Galaxy) is shown as a 3D box
   - Transparent colored boxes with different colors per board type
   - Proper dimensions based on board type

2. **🔵 Chips Inside Boards**
   - Chips displayed as cubes inside their parent boards
   - Proper spatial layout (N300: 2 side-by-side, Galaxy: 8×4 grid, etc.)
   - Hover to see chip details

3. **🔗 Ethernet Connections**
   - Gray lines connecting chips across boards
   - Shows channel numbers on hover

4. **🔴 Mesh Overlay Box**
   - Red transparent box showing the logical mesh boundary
   - Wireframe outline for clarity
   - Shows which chips are part of the logical mesh

## 🎯 Board Types Supported

| Board Type | Chips | Layout | Color |
|-----------|-------|---------|-------|
| **N150** | 1 | Single chip | Light Blue |
| **N300** | 2 | Side-by-side | Light Green |
| **P150** | 1 | Single chip | Light Coral |
| **P300** | 2 | Side-by-side | Light Salmon |
| **T3K** | 8 | 2×4 grid | Light Cyan |
| **Galaxy** | 32 | 8×4 grid | Light Yellow |

## 🚀 Quick Start

### Generate Enhanced Visualization

```bash
# With mesh overlay (recommended)
./visualize_topology_3d_enhanced.py \\
    --cluster t3k_phys_topology.yaml \\
    --mesh t3k_generated_mesh.yaml

# Without mesh overlay
./visualize_topology_3d_enhanced.py \\
    --cluster t3k_phys_topology.yaml

# Custom output file
./visualize_topology_3d_enhanced.py \\
    --cluster my_cluster.yaml \\
    --mesh my_mesh.yaml \\
    --output my_viz.html
```

## 📊 Your T3K Visualization

**Already generated for you:**
- `topology_3d_with_boards.html` (Enhanced 3D view)

**What you'll see:**
```
┌─────────────────────────────────────────┐
│  Red Mesh Box (Logical 2×4 Mesh)       │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐│
│  │ N300 │  │ N300 │  │ N300 │  │ N300 ││
│  │ 🔵🔵 │  │ 🔵🔵 │  │ 🔵🔵 │  │ 🔵🔵 ││
│  │ 0  1 │  │ 2  3 │  │ 4  5 │  │ 6  7 ││
│  └──────┘  └──────┘  └──────┘  └──────┘│
│     ╲         ╱╲         ╱╲         ╱   │
│      ╲───────╱  ╲───────╱  ╲───────╱    │
│         (Ethernet Links)                │
└─────────────────────────────────────────┘
```

## 🎮 Interactive Controls

Once you open the HTML file in a browser:

### Navigation
- **Rotate**: Left-click + drag
- **Zoom**: Scroll wheel or pinch
- **Pan**: Right-click + drag
- **Reset**: Double-click

### Information
- **Hover over chips**: See chip ID, architecture, board type
- **Hover over links**: See ethernet channel details
- **Hover over boards**: See board type and chip count
- **Hover over mesh box**: See mesh configuration

### Legend
- Click legend items to show/hide:
  - Different board types
  - Mesh overlay
  - Individual elements

## 🎨 Visual Hierarchy

```
Scene
├── Board Boxes (semi-transparent)
│   ├── N300 Board 0 (light green)
│   │   ├── Chip 0 (dark blue cube)
│   │   └── Chip 1 (dark blue cube)
│   ├── N300 Board 1 (light green)
│   │   ├── Chip 2
│   │   └── Chip 3
│   └── ... (more boards)
│
├── Ethernet Links (gray lines)
│   ├── Chip 0 ↔ Chip 3 (channels)
│   ├── Chip 0 ↔ Chip 4 (channels)
│   └── ... (more links)
│
└── Mesh Overlay (red transparent box)
    └── Outlines logical 2×4 mesh
```

## 🔍 Comparison: Standard vs Enhanced

### Standard Visualization (`topology_3d.html`)
- ✅ Simple, fast to load
- ✅ Shows chip positions
- ✅ Shows connections
- ❌ No board grouping
- ❌ Flat visualization

### Enhanced Visualization (`topology_3d_with_boards.html`)
- ✅ **Shows board enclosures**
- ✅ **Hierarchical: Boards → Chips → Mesh**
- ✅ **3D boxes for boards and mesh**
- ✅ **Color-coded by board type**
- ✅ **Proper spatial layout**
- ⚠️  Larger file size (more geometry)

## 📐 Layout Algorithms

### Board Placement
Boards are arranged in a grid with spacing:
```python
spacing = 8 units between boards
cols = ceil(sqrt(num_boards))
row = board_idx // cols
col = board_idx % cols
```

### Chip Placement Within Board

**N300** (2 chips):
```
┌─────────┐
│  🔵  🔵  │  Side-by-side
└─────────┘
```

**T3K** (8 chips):
```
┌───────────────┐
│ 🔵 🔵 🔵 🔵 │  2×4 grid
│ 🔵 🔵 🔵 🔵 │
└───────────────┘
```

**Galaxy** (32 chips):
```
┌─────────────────────────┐
│ 🔵🔵🔵🔵🔵🔵🔵🔵 │  8×4 grid
│ 🔵🔵🔵🔵🔵🔵🔵🔵 │
│ 🔵🔵🔵🔵🔵🔵🔵🔵 │
│ 🔵🔵🔵🔵🔵🔵🔵🔵 │
└─────────────────────────┘
```

## 🎯 Use Cases

### 1. Understand Physical Layout
- See how boards are arranged
- Understand chip distribution across boards
- Visualize actual hardware configuration

### 2. Verify Mesh Configuration
- Red box shows logical mesh boundary
- Verify which chips are in the mesh
- Check mesh spans correct boards

### 3. Debug Connectivity
- See which chips connect to which
- Identify cross-board connections
- Trace ethernet links

### 4. Documentation
- Export high-quality visualization for reports
- Share with team to explain topology
- Use in presentations

## 🔧 Advanced Usage

### Customize Board Colors

Edit `visualize_topology_3d_enhanced.py`:

```python
board_colors = {
    'n150': 'lightblue',
    'n300': 'lightgreen',     # Change this
    'p150': 'lightcoral',
    'galaxy': 'gold',         # Change this
    # Add more board types...
}
```

### Adjust Board Spacing

```python
spacing = 8  # Change this value (line ~145)
```

### Change Mesh Color

```python
color='red',      # Mesh box color (line ~489)
opacity=0.15,     # Transparency
```

## 📊 File Sizes

```bash
topology_3d.html               3.5 MB   (simple)
topology_3d_with_boards.html   ~6 MB    (enhanced, more geometry)
```

The enhanced version is larger because it includes:
- Multiple 3D box meshes for boards
- Wireframe edges for clarity
- Mesh overlay geometry

## 🆚 When to Use Each

### Use Standard (`visualize_topology.py`)
- Quick preview
- Simple topology
- Need small file size
- Presenting to non-technical audience

### Use Enhanced (`visualize_topology_3d_enhanced.py`)
- Complex multi-board systems
- Need to understand physical layout
- Debugging connectivity issues
- Technical documentation
- Training and education

## 🎉 Summary

The enhanced visualizer gives you:
- 📦 **Board-level view**: See N300, Galaxy, etc. as physical entities
- 🔵 **Chip-level detail**: Individual chips inside boards
- 🔗 **Connection tracing**: Ethernet links between chips
- 🔴 **Mesh overlay**: Logical topology on top of physical

**Open `topology_3d_with_boards.html` in your browser to see your T3K with:**
- 4 N300 boards (green transparent boxes)
- 8 chips (2 per board, blue cubes)
- 20 ethernet links (gray lines)
- 2×4 mesh overlay (red transparent box)

All fully interactive! 🚀
