# Interactive Topology UI Guide 🎛️

## 🎉 New UI - Much Better Selection!

I've created a completely new interface that addresses all your concerns:

### ✅ **Easy Element Selection**
- **Larger click targets** on chips and connections
- **Hover highlighting** shows what you're about to click
- **Sidebar panel** with full details of selected element
- **Click anywhere** on a connection line to select it

### ✅ **Uses Physical Coordinates**
- **Shelf, Rack X, Rack Y** from your YAML file
- Shows actual physical layout
- Matches your rack documentation

### ✅ **Selectable Ethernet Connections**
- **Click any connection line** to select it
- See source/destination chips
- Channel numbers displayed
- Intra-board vs inter-board type

## 🎨 UI Layout

```
┌──────────────────────────────────────────────────────────────────┐
│                    Interactive Topology Viewer                    │
├─────────────────────────────────────────────┬────────────────────┤
│                                             │  🎛️ Control Panel  │
│                                             │                    │
│          3D/2D Visualization                │  📊 System Stats   │
│          (Main View Area)                   │  • Chips: 8        │
│                                             │  • Boards: 4       │
│          Click chips or connections         │  • Connections: 20 │
│          to see details →                   │                    │
│                                             │  🔍 Filters        │
│                                             │  ☑ Show Chips      │
│                                             │  ☑ Show Connections│
│                                             │                    │
│                                             │  📌 Selected:      │
│                                             │  Chip 3            │
│                                             │  Arch: wormhole_b0 │
│                                             │  Shelf: 2          │
│                                             │  Rack: (0, 0)      │
│                                             │  ...               │
└─────────────────────────────────────────────┴────────────────────┘
```

## 🖱️ Interaction Guide

### Click on a Chip
```
1. Click any chip (blue square)
2. Sidebar shows:
   ├─ Chip ID
   ├─ Architecture
   ├─ Board Type
   ├─ Board ID
   └─ Physical Location
      ├─ Shelf
      ├─ Rack X
      ├─ Rack Y
      └─ Adapter
```

### Click on an Ethernet Connection
```
1. Click any connection line
2. Sidebar shows:
   ├─ Connection ID
   ├─ Source Chip [channel]
   ├─ Destination Chip [channel]
   ├─ Connection Type (intra/inter-board)
   ├─ Source Board
   └─ Destination Board
```

## 🎮 View Modes

### 3D Rack View (Default)
- **Shows**: Physical rack layout using shelf, rack_x, rack_y
- **X axis**: Rack X position
- **Y axis**: Rack Y position
- **Z axis**: Shelf number
- **Best for**: Understanding physical layout

### 2D Schematic View
- **Shows**: Logical schematic diagram
- **Layout**: Grid based on chip count
- **Best for**: Quick overview, documentation

**Switch views**: Click buttons at top of sidebar

## 📍 Physical Coordinates Explained

Your YAML has this format:
```yaml
chips:
  0: [2, 0, 0, 0]  # [shelf, rack_x, rack_y, adapter]
  1: [2, 1, 0, 0]
  ...
```

**In the 3D view:**
- **Shelf** → Z axis (vertical)
- **Rack X** → X axis (horizontal)
- **Rack Y** → Y axis (depth)
- **Adapter** → Shown in details panel

**Example for Chip 0:**
```
Shelf: 2       → Height in rack
Rack X: 0      → Position in row
Rack Y: 0      → Position in column
Adapter: 0     → Adapter slot
```

## 🎯 Selection Features

### Large Click Targets
```
Old:  🔵 (small, hard to click)
New:  🟦 (larger markers, easier to click)
```

### Hover Feedback
- Hover over element → See preview in tooltip
- Cursor changes to pointer
- Element highlights slightly

### Sidebar Details
All info in one place:
- No need to remember popup coordinates
- Persistent display
- Full context at a glance

## 🔍 Filters

Control what's visible:
- ☑ **Show Chips** - Toggle chip visibility
- ☑ **Show Connections** - Toggle connection lines
- ☑ **Show Labels** - Toggle chip ID labels

## 📊 System Stats

Always visible stats:
- **Chips**: Total chip count
- **Boards**: Total board count
- **Connections**: Total ethernet links

## 🎨 Color Coding

### Chips
- **Blue squares** with white borders
- **White text** showing chip ID
- Larger than previous versions

### Connections
- **Blue** = Intra-board (same board, straight)
- **Gray** = Inter-board (different boards, can be curved)
- **Thicker lines** = easier to click

### Sidebar
- **Dark theme** for comfortable viewing
- **Blue highlights** for labels
- **Colored badges** for chip/connection type

## 🚀 Usage Examples

### Example 1: Find Chip Physical Location
```
1. Click "Chip 3" in visualization
2. Look at sidebar "Physical Location" section
3. See: Shelf: 1, Rack X: 0, Rack Y: 0, Adapter: 0
4. Use for rack documentation
```

### Example 2: Trace Ethernet Connection
```
1. Click connection line between chips
2. Sidebar shows:
   - Source: Chip 0 [ch0]
   - Destination: Chip 3 [ch0]
   - Type: Inter-board
3. Verify both endpoints
```

### Example 3: Compare Physical vs Logical
```
1. Start in "3D Rack View" (physical)
   - See chips at actual shelf/rack positions
2. Switch to "2D Schematic View" (logical)
   - See chips in grid layout
3. Understand mapping between physical and logical
```

## 📁 Files Generated

```
topology_ui.html            ~800 KB    ⭐ NEW: Interactive UI with sidebar
topology_3d_battery.html    ~3.5 MB    v3: Battery grid
topology_3d_interactive.html ~3.5 MB   v2: Click coords
```

## 💡 Pro Tips

### Quick Selection
- **Double-click** an element for instant selection
- **Hover first** to preview before clicking
- **Use 2D view** for quick chip identification

### Details Panel Tricks
- **Scroll** the sidebar for long info
- **Leave selected** to keep reference while exploring
- **Click elsewhere** to clear and select new element

### View Switching
- **Start with 3D** to understand physical layout
- **Switch to 2D** for schematic understanding
- **Use both** for complete picture

### Connection Analysis
- **Click connections** to see endpoints
- **Check "Type"** field (intra vs inter-board)
- **Verify channels** match your expectations

## 🐛 Troubleshooting

### Can't Click Elements

**Solution:**
- Try zooming in closer
- Elements have larger click areas now
- Hover to ensure cursor is over element

### Sidebar Not Updating

**Solution:**
- Make sure JavaScript is enabled
- Try clicking directly on chip/connection
- Refresh browser if needed

### Physical Coordinates Look Wrong

**Check:**
1. Your YAML format: [shelf, rack_x, rack_y, adapter]
2. Z axis (shelf) might be inverted
3. Try 2D view for verification

## 🎉 Summary

The new UI gives you:

✅ **Easy selection** - Larger targets, better feedback
✅ **Physical coordinates** - Uses shelf/rack from YAML
✅ **Selectable connections** - Click any ethernet line
✅ **Detailed sidebar** - All info in one place
✅ **Multiple views** - 3D rack + 2D schematic
✅ **Better UX** - Professional interface

**Much easier to use than previous versions!**

## 🚀 Quick Start

```bash
# Copy to local machine
scp user@remote:/workspace/tt-metal-apv/topology_ui.html .

# Open in browser
open topology_ui.html

# Try these:
# 1. Click a chip → See details in sidebar
# 2. Click a connection line → See connection info
# 3. Switch views using buttons
# 4. Explore your topology!
```

This UI is **production-ready** and **much more user-friendly**! 🎊
