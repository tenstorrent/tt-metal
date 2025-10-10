# Interactive 3D Topology Visualization Guide 🎯

## 🎉 Version 2 Features!

I've created an **improved version** of the 3D visualizer that addresses your feedback:

### ✨ New Features

#### 1. **Click to See Coordinates** 📍
Click any element (chip, board, connection, mesh) to see its exact X,Y,Z coordinates:
- **Popup alert** with coordinates
- **Title bar** updated with coordinates
- **Console log** for developers

#### 2. **Improved Layout Algorithm** 🎨
Boards are now arranged with better spacing to minimize line crossings:
- **Linear layout** for 2-4 boards (your T3K case)
- **2-row layout** for 5-8 boards
- **Grid layout** for larger systems
- **12 unit spacing** between boards (increased from 8)

#### 3. **Curved Ethernet Lines** 🌊
Connections now **lift up and curve** instead of passing through chips:
- **Bezier curves** that arc above the boards
- **Proportional lift** based on distance
- **No more lines through chips!**

## 🚀 Quick Start

### Generate the Improved Visualization

```bash
./visualize_topology_3d_enhanced_v2.py \\
    --cluster t3k_phys_topology.yaml \\
    --mesh t3k_generated_mesh.yaml \\
    --output topology_3d_interactive.html
```

**Already generated for you:**
- `topology_3d_interactive.html` - New improved version! ⭐

## 🎮 Interactive Controls

### Basic Navigation
- **Rotate**: Left-click + drag
- **Zoom**: Scroll wheel
- **Pan**: Right-click + drag
- **Hover**: See element details

### NEW: Coordinate Display
- **Click any element**: Shows coordinates in popup + title
- **Double-click anywhere**: Reset view and clear coordinates
- **Works on**: Chips, boards, ethernet links, mesh boxes

### Example Interactions

**Click a chip:**
```
Alert popup: "Chip 3 - Coordinates: (18.50, 0.00, 1.00)"
Title updates: Shows same info
```

**Click a board:**
```
Alert popup: "Coordinates: (15.50, 1.25, 1.25)"
(Shows board center)
```

**Click an ethernet connection:**
```
Alert popup: "Coordinates: (13.45, 0.50, 2.80)"
(Shows midpoint of curved line)
```

## 🌊 Curved Lines Explained

### Why Curved?
Straight lines passed through chips when connecting distant boards. Now they curve up and over!

### How It Works
```
Chip A ────────► Chip B  ❌ OLD: Straight through chips

              ╱‾‾‾╲
Chip A ──────╱     ╲────► Chip B  ✅ NEW: Curves above
```

**Algorithm:**
1. Calculate distance between chips
2. Create control point at midpoint
3. Lift control point proportional to distance
4. Generate smooth Bezier curve
5. Render as 3D line

**Benefits:**
- ✅ No lines through chips
- ✅ Easy to see which chips connect
- ✅ Clear 3D depth perception
- ✅ Professional appearance

## 📐 Layout Improvements

### Old Layout (v1)
```
Board 0  Board 1
  🔵🔵    🔵🔵     ← 8 units apart
Board 2  Board 3
  🔵🔵    🔵🔵

Issue: Lines still crossed sometimes
```

### New Layout (v2) - Your T3K
```
Board 0      Board 1      Board 2      Board 3
  🔵🔵        🔵🔵          🔵🔵          🔵🔵

← 12 units → ← 12 units → ← 12 units →

Result: Linear layout, maximum separation, curved lines!
```

### For Larger Systems

**5-8 boards:**
```
Board 0  Board 1  Board 2  Board 3
  🔵       🔵       🔵       🔵
                                    ← 2 rows
Board 4  Board 5  Board 6  Board 7
  🔵       🔵       🔵       🔵
```

**9+ boards:**
```
Board 0  Board 1  Board 2  Board 3
Board 4  Board 5  Board 6  Board 7
Board 8  Board 9  Board10  Board11
  ...
```

## 🎯 Use Cases

### 1. **Debug Chip Placement**
Click chips to verify their coordinates match your expectations

### 2. **Trace Connections**
Click an ethernet line to see its path midpoint

### 3. **Document Topology**
Take screenshots with coordinate annotations

### 4. **Verify Layout**
Check that boards are properly spaced

### 5. **Development**
Use coordinates for programmatic layout verification

## 📊 Comparison: v1 vs v2

| Feature | v1 (old) | v2 (new) |
|---------|----------|----------|
| Board layout | Grid | Optimized (linear/2-row/grid) |
| Board spacing | 8 units | 12 units |
| Ethernet lines | Straight | **Curved/lifted** |
| Line crossings | Some | **Minimal** |
| Click coordinates | ❌ | **✅ Yes!** |
| Coordinate display | ❌ | **Popup + title** |
| Layout quality | Good | **Better** |

## 💡 Pro Tips

### 1. Find Chip Coordinates
```
1. Click a chip
2. Note coordinates from popup
3. Use in your configuration files
```

### 2. Measure Distances
```
1. Click chip A, note coordinates
2. Click chip B, note coordinates
3. Calculate: √((x₂-x₁)² + (y₂-y₁)² + (z₂-z₁)²)
```

### 3. Verify Mesh Bounds
```
1. Click chips at mesh corners
2. Verify they're within mesh box
3. Check mesh box center coordinates
```

### 4. Export Coordinates
```
1. Open browser console (F12)
2. Click elements
3. Copy coordinates from console
4. Save to file
```

### 5. Clear Display
```
Double-click anywhere to:
• Reset camera view
• Clear coordinate display
• Start fresh
```

## 🔧 Customization

### Adjust Curve Height

Edit `visualize_topology_3d_enhanced_v2.py` line ~209:

```python
mid_z = max(z1, z2) + dist * lift_factor * 0.15
#                                          ^^^^
# Increase for higher curves: 0.20, 0.25, etc.
# Decrease for flatter curves: 0.10, 0.05, etc.
```

### Adjust Board Spacing

Edit line ~139:

```python
spacing = 12  # Change to 15, 20, etc. for more space
```

### Change Curve Smoothness

Edit line ~203:

```python
def create_curved_line(p1, p2, num_points=50, lift_factor=2.0):
#                                        ^^
# Increase for smoother: 100, 200
# Decrease for performance: 30, 20
```

## 🎨 Visual Examples

### T3K Topology (Your System)

**Before (v1):**
```
🟩━━━━━━━🟩━━━━━━━🟩━━━━━━━🟩
│ 🔵🔵   │ 🔵🔵   │ 🔵🔵   │ 🔵🔵
│  0  1  │  2  3  │  4  5  │  6  7
└────────┴────────┴────────┴────────
  Lines cross through boards ❌
```

**After (v2):**
```
          ╱‾‾╲        ╱‾‾╲
🟩═══════╱    ╲═══🟩═╱    ╲═══🟩═══════🟩
│ 🔵🔵          │ 🔵🔵        │ 🔵🔵      │ 🔵🔵
│  0  1         │  2  3       │  4  5     │  6  7
└───────────────┴────────────┴──────────┴────────
  Lines curve above boards ✅
```

## 📁 Files Generated

```
topology_3d_interactive.html    ~3.5 MB    ⭐ NEW: With click & curves
topology_3d_with_boards.html    ~3.5 MB    OLD: v1
topology_3d.html                ~3.5 MB    OLDER: Simple
topology_2d.png                ~150 KB     2D diagram
```

**Recommendation:** Use `topology_3d_interactive.html` (v2)

## 🐛 Troubleshooting

### Coordinates don't show on click

**Check:**
1. Make sure you're clicking on an element (chip, board, line)
2. Look for popup alert
3. Check browser console (F12) for logged coordinates
4. Verify JavaScript is enabled

### Lines still cross chips

**Solutions:**
1. Increase `lift_factor` in code (line 203)
2. Increase `spacing` between boards (line 139)
3. Try different board count layouts

### Can't see curved lines

**Tips:**
1. Zoom in to see curve detail
2. Rotate view to see 3D lift
3. Hover over line midpoint to confirm it's there

## 🎉 Summary

The new v2 visualizer gives you:

✅ **Click any element** → See X,Y,Z coordinates
✅ **Curved ethernet lines** → No more lines through chips
✅ **Optimized layout** → Better board spacing
✅ **Professional appearance** → Ready for presentations
✅ **Interactive coordinates** → Great for debugging

**Your T3K visualization now shows:**
- 4 N300 boards in a clean line
- 8 chips clearly visible
- Curved ethernet connections that lift above boards
- Red mesh overlay showing logical 2×4
- Click anywhere to see exact coordinates!

Open `topology_3d_interactive.html` in your browser and start clicking! 🚀
