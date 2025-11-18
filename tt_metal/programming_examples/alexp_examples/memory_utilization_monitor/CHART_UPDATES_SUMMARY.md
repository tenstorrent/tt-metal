# Chart View Updates Summary

## Changes Made

### 1. ✅ Removed Device Cap
**Before**: Charts capped at 2 devices
**After**: Shows ALL devices (no limit)

### 2. ✅ Square Grid Layout (NxN Matrix)
**Before**: Mixed layouts (2x1, 3x2, etc.)
**After**: Always square grids for better organization

| Devices | Layout |
|---------|--------|
| 1 | 1x1 |
| 2-4 | 2x2 |
| 5-9 | 3x3 |
| 10-16 | 4x4 |
| 17-25 | 5x5 |

### 3. ✅ Visual Device Separators
Each device chart is now separated by a horizontal line:
```
Device 0 chart...
  ----------------------------------------------------------------
Device 1 chart...
  ----------------------------------------------------------------
Device 2 chart...
```

### 4. ✅ Improved Graph Overlap Indicators
**When DRAM and L1 lines overlap:**
- `=` - Horizontal overlap (both flat)
- `#` - Vertical overlap (both rising/falling vertically)
- `+` - Mixed overlap (diagonal + horizontal/vertical)
- **Color**: Yellow to clearly show both are present

**When lines don't overlap:**
- Green: DRAM line only (`-/\|_`)
- Cyan: L1 line only (`:` for vertical, others shared with DRAM)

### 5. ✅ Updated Legend
Shows what each character means:
```
 DRAM        5.0%    (Green)
 L1 (:)      0.0%    (Cyan - : for vertical)
 Both (=#+)          (Yellow - overlap indicators)
 L1_SMALL: 0.0%      (Text only, not graphed)
 TRACE: 0.0%         (Text only, not graphed)
```

### 6. ✅ Added L1_SMALL and TRACE Display
While not graphed (only DRAM and L1 are graphed), their percentages are shown in the legend for reference.

## Visual Example

```
Chart View - Last 60 seconds (8 devices in 3x3 matrix)
(Press 1 for main view)

Device 0 [WORMHOLE_B0] TEMP  45C  CLK 1200MHz
DRAM[|||||||||||||||||||||||||     75.5%  12.0GB/16.0GB] L1[|||||||    45.2%]
  +------------------------------------------------------------+
100|                                                            | DRAM      75.5%
 80|    /---\                                                  |
 60|   /     \    ----                                         | L1 (:)     0.0%
 40|  /       \--/    \                                        | Both (=#+)
 20| /                 \                                       | L1_SMALL: 0.0%
  0|/                   \---                                   |
    |                                                          | TRACE: 0.0%
  +------------------------------------------------------------+
     60s ago                                               now

  ----------------------------------------------------------------

Device 1 [WORMHOLE_B0] TEMP  48C  CLK 1200MHz
[...]
```

## Key Features

### Overlap Detection
- When DRAM=50% and L1=50% at the same time point
- Shows yellow `=` or `#` or `+` to indicate **both** are at that level
- Prevents one line from "hiding" the other

### Square Grid Benefits
- **Visual balance**: Same number of rows and columns
- **Easier scanning**: Regular pattern
- **Better organization**: Groups logically

### All Devices Shown
- No arbitrary limits
- Scales to 25+ devices (5x5 grid)
- Each device gets equal space

## Why L1 Shows as "----------"

If you see a flat line like:
```
----------------------------------------------------
```

This means **L1 usage has been consistently at that percentage** for the entire 60-second window.

- Flat at 0%: L1 not being used (expected for models using mainly DRAM)
- Flat at 50%: L1 consistently at 50%
- Rising/falling: L1 usage changing over time

## Testing

```bash
./build/programming_examples/tt_smi_umd -w

# Press '2' to see chart view
# Should show all devices in square grid
# Each device separated by horizontal line
# Yellow overlap indicators when DRAM=L1
```

## What This Solves

1. ❌ **Old**: Only 2 devices shown → ✅ **New**: All devices shown
2. ❌ **Old**: Irregular grid (2x1, 3x2) → ✅ **New**: Square grid (2x2, 3x3)
3. ❌ **Old**: Overlapping lines hide each other → ✅ **New**: Yellow overlap indicators
4. ❌ **Old**: Confusing `====` for flat L1 → ✅ **New**: Uses `----` like DRAM
5. ❌ **Old**: No visual separation → ✅ **New**: Lines between device charts

## Related Documentation

- `L1_MEMORY_NOT_TRACKED.md` - Explains why L1 shows low usage
- `TRACE_MEMORY_EXPLAINED.md` - Details on TRACE buffer tracking
- `CHART_VIEW_DESIGN.md` - Original chart design
