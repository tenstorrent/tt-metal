# Depthwise Conv2D Weight Preparation - Face-by-Face Layout

## Overview

This document explains the weight preparation process for depthwise convolution operations in TT-Metal. The goal is to arrange weights in a specific "face-by-face" layout before tilization, so that after tilization, the weights are organized stick-by-stick in memory, which is optimal for the depthwise convolution kernel.

## Input Format

**Input Shape**: `[out_channels, 1, kernel_h, kernel_w]`

For a depthwise convolution with 96 output channels and a 3×3 kernel:
- Input shape: `[96, 1, 3, 3]`
- Each output channel has its own 3×3 kernel
- Total of 96 independent 3×3 kernels

Example weight values (showing first few channels):
```
Channel 0: [1, 1, 1, 1, 1, 1, 1, 1, 1]  (9 values for 3x3 kernel)
Channel 1: [2, 2, 2, 2, 2, 2, 2, 2, 2]
Channel 2: [3, 3, 3, 3, 3, 3, 3, 3, 3]
...
Channel 95: [...]
```

## Output Format

**Output Shape**: `[out_channels, padded_kernel_positions, 1, 1]`

For the same example:
- Output shape: `[96, 32, 1, 1]` (kernel positions padded from 9 to 32)
- Each kernel position becomes a "stick" containing all channel values
- After tilization, this will be in TILE layout: `[1, 1, 32, out_channels]`

## Key Concepts

### 1. Stick
A **stick** represents all channel values for a single kernel position:
- Stick 0: All channels' values at kernel position (0, 0)
- Stick 1: All channels' values at kernel position (0, 1)
- ...
- Stick 8: All channels' values at kernel position (2, 2)

For 96 channels, each stick contains 96 values.

### 2. Face
A **face** is a 16×16 block within a 32×32 tile:
- Face 0 (top-left): rows 0-15, cols 0-15
- Face 1 (top-right): rows 0-15, cols 16-31
- Face 2 (bottom-left): rows 16-31, cols 0-15
- Face 3 (bottom-right): rows 16-31, cols 16-31

### 3. Rows per Stick
Each stick occupies multiple rows in the face-by-face layout:
- **32 channels**: 2 rows per stick (32 ÷ 16 = 2)
- **64 channels**: 4 rows per stick (64 ÷ 16 = 4)
- **96 channels**: 6 rows per stick (96 ÷ 16 = 6)

Formula: `rows_per_stick = ceil(out_channels / 16)`

## Face-by-Face Layout Algorithm

### Step 1: Initialize Output Buffer
Create a buffer of shape `[out_channels, padded_kernel_positions, 1, 1]` initialized to zeros.

### Step 2: Track Current Row Position
Maintain a variable `current_absolute_row` that tracks our position as we fill faces sequentially:
- Faces 0-3 are treated as continuous rows 0-63
- Face 0: rows 0-15
- Face 1: rows 16-31
- Face 2: rows 32-47
- Face 3: rows 48-63

### Step 3: Place Each Stick
For each kernel position (0 to 8 for 3×3 kernel):

1. **Calculate stick start position**: Use `current_absolute_row`

2. **For each row of the stick** (0 to `rows_per_stick - 1`):
   - Calculate absolute row: `absolute_row = current_absolute_row + stick_row`
   - Determine which face: `face_idx = absolute_row / 16`
   - Determine row within face: `row_in_face = absolute_row % 16`

3. **Map face index to tile coordinates**:
   - `face_row_offset = (face_idx / 2) * 16`
   - `face_col_offset = (face_idx % 2) * 16`
   - `target_row = face_row_offset + row_in_face`

4. **Place 16 values in this row**:
   - For column 0-15 in the current face
   - Channel index: `ch = stick_row * 16 + col`
   - Read value from input: `input[ch, 0, kh, kw]`
   - Write to output: `output[target_col, target_row, 0, 0]`
   - Where `target_col = face_col_offset + col`

5. **Move to next stick**: `current_absolute_row += rows_per_stick`

## Example: 96 Channels, 3×3 Kernel

### Before Face-by-Face Layout (after tilization would be wrong)
If we naively placed data row-by-row across the full 32-column width, tilization would NOT produce stick-by-stick layout.

### With Face-by-Face Layout

**Rows per stick**: 96 ÷ 16 = 6 rows

**Stick 0** (kernel position 0,0) - starts at absolute row 0:
```
Row 0  (Face 0): [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] (channels 0-15)
Row 1  (Face 0): [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] (channels 16-31)
Row 2  (Face 0): [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] (channels 32-47)
Row 3  (Face 0): [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] (channels 48-63)
Row 4  (Face 0): [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] (channels 64-79)
Row 5  (Face 0): [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] (channels 80-95)
```

**Stick 1** (kernel position 0,1) - starts at absolute row 6:
```
Row 6  (Face 0): [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2] (channels 0-15)
Row 7  (Face 0): [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2] (channels 16-31)
Row 8  (Face 0): [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2] (channels 32-47)
Row 9  (Face 0): [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2] (channels 48-63)
Row 10 (Face 0): [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2] (channels 64-79)
Row 11 (Face 0): [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2] (channels 80-95)
```

**Stick 2** (kernel position 0,2) - starts at absolute row 12:
```
Row 12 (Face 0): [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3] (channels 0-15)
Row 13 (Face 0): [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3] (channels 16-31)
Row 14 (Face 0): [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3] (channels 32-47)
Row 15 (Face 0): [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3] (channels 48-63)
Row 16 (Face 1): [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3] (channels 64-79)
Row 17 (Face 1): [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3] (channels 80-95)
```
*Note: Stick 2 spans across Face 0 and Face 1!*

**Stick 3** (kernel position 1,0) - starts at absolute row 18:
```
Row 18 (Face 1): [4, 4, 4, ...] (channels 0-15)
Row 19 (Face 1): [4, 4, 4, ...] (channels 16-31)
...continuing in Face 1...
```

### Complete 32×32 Layout Visualization

```
         Cols 0-15 (Face 0)    |    Cols 16-31 (Face 1)
         ==================    |    ===================
Row 0:   1 1 1 .. 1 1 1       |    3 3 3 .. 3 3 3
Row 1:   1 1 1 .. 1 1 1       |    3 3 3 .. 3 3 3
Row 2:   1 1 1 .. 1 1 1       |    4 4 4 .. 4 4 4
Row 3:   1 1 1 .. 1 1 1       |    4 4 4 .. 4 4 4
Row 4:   1 1 1 .. 1 1 1       |    4 4 4 .. 4 4 4
Row 5:   1 1 1 .. 1 1 1       |    4 4 4 .. 4 4 4
Row 6:   2 2 2 .. 2 2 2       |    4 4 4 .. 4 4 4
Row 7:   2 2 2 .. 2 2 2       |    4 4 4 .. 4 4 4
Row 8:   2 2 2 .. 2 2 2       |    5 5 5 .. 5 5 5
Row 9:   2 2 2 .. 2 2 2       |    5 5 5 .. 5 5 5
Row 10:  2 2 2 .. 2 2 2       |    5 5 5 .. 5 5 5
Row 11:  2 2 2 .. 2 2 2       |    5 5 5 .. 5 5 5
Row 12:  3 3 3 .. 3 3 3       |    5 5 5 .. 5 5 5
Row 13:  3 3 3 .. 3 3 3       |    5 5 5 .. 5 5 5
Row 14:  3 3 3 .. 3 3 3       |    6 6 6 .. 6 6 6
Row 15:  3 3 3 .. 3 3 3       |    6 6 6 .. 6 6 6
         ==================    |    ===================
         Cols 0-15 (Face 2)    |    Cols 16-31 (Face 3)
         ==================    |    ===================
Row 16:  6 6 6 .. 6 6 6       |    9 9 9 .. 9 9 9
Row 17:  6 6 6 .. 6 6 6       |    9 9 9 .. 9 9 9
Row 18:  6 6 6 .. 6 6 6       |    9 9 9 .. 9 9 9
Row 19:  6 6 6 .. 6 6 6       |    9 9 9 .. 9 9 9
Row 20:  7 7 7 .. 7 7 7       |    9 9 9 .. 9 9 9
Row 21:  7 7 7 .. 7 7 7       |    9 9 9 .. 9 9 9
Row 22:  7 7 7 .. 7 7 7       |    0 0 0 .. 0 0 0 (padding)
Row 23:  7 7 7 .. 7 7 7       |    0 0 0 .. 0 0 0
Row 24:  7 7 7 .. 7 7 7       |    0 0 0 .. 0 0 0
Row 25:  7 7 7 .. 7 7 7       |    0 0 0 .. 0 0 0
Row 26:  8 8 8 .. 8 8 8       |    0 0 0 .. 0 0 0
Row 27:  8 8 8 .. 8 8 8       |    0 0 0 .. 0 0 0
Row 28:  8 8 8 .. 8 8 8       |    0 0 0 .. 0 0 0
Row 29:  8 8 8 .. 8 8 8       |    0 0 0 .. 0 0 0
Row 30:  8 8 8 .. 8 8 8       |    0 0 0 .. 0 0 0
Row 31:  8 8 8 .. 8 8 8       |    0 0 0 .. 0 0 0
```

## After Tilization

When this face-by-face layout is tilized (converted to TILE format), the resulting memory layout becomes:

```
Shape: [1, 1, 32, 96] in TILE layout

Row 0:  [1, 1, 1, ..., 1, 1, 1]  (96 ones - stick 0, all channels)
Row 1:  [1, 1, 1, ..., 1, 1, 1]  (96 ones - stick 0, all channels)
Row 2:  [2, 2, 2, ..., 2, 2, 2]  (96 twos - stick 1, all channels)
Row 3:  [2, 2, 2, ..., 2, 2, 2]  (96 twos - stick 1, all channels)
Row 4:  [3, 3, 3, ..., 3, 3, 3]  (96 threes - stick 2, all channels)
Row 5:  [3, 3, 3, ..., 3, 3, 3]  (96 threes - stick 2, all channels)
...
Row 16: [9, 9, 9, ..., 9, 9, 9]  (96 nines - stick 8, all channels)
Row 17: [9, 9, 9, ..., 9, 9, 9]  (96 nines - stick 8, all channels)
Row 18-31: [0, 0, 0, ..., 0, 0, 0]  (padding)
```

**This is stick-by-stick layout!** Each pair of rows represents one complete stick (all channel values for one kernel position), which is exactly what the depthwise convolution kernel expects.

## Why This Matters

### Without Face-by-Face Layout
If we simply arranged data row-by-row across the full 32 columns:
- Values would be interleaved incorrectly after tilization
- The kernel would read wrong data
- Performance would be poor

### With Face-by-Face Layout
- After tilization, each stick's data is contiguous in memory
- The depthwise convolution kernel can efficiently read stick-by-stick
- Each kernel position's weights for all channels are grouped together
- Optimal memory access patterns for the hardware

## Implementation Summary

The `convert_conv_weight_tensor_to_2d_depthwise_layout` function:

1. **Input**: `[out_channels, 1, kernel_h, kernel_w]` in ROW_MAJOR layout
2. **Calculates**:
   - `rows_per_stick = ceil(out_channels / 16)`
   - `padded_kernel_positions = ceil(kernel_h × kernel_w / 32) × 32`
3. **Creates**: Output buffer `[out_channels, padded_kernel_positions, 1, 1]`
4. **Fills face-by-face**: Places each stick sequentially across faces
5. **Output**: ROW_MAJOR layout ready for tilization
6. **After tilization**: Becomes TILE layout with stick-by-stick memory organization

## Handling Channels Not Multiple of 32 (e.g., 16 channels)

When the number of channels is not a multiple of 32 (tile size), special handling is required to ensure proper tilization. The key insight is that we need to pad both the channels AND add padding rows between sticks.

### The Problem

For 16 channels:
- `rows_per_stick = ceil(16/16) = 1` (only 1 face row of actual data)
- Without padding, sticks would be placed back-to-back
- After tilization, values would interleave incorrectly

### The Solution

1. **Pad channels to tile size**: 16 → 32, 48 → 64, etc.
2. **Calculate padded rows per stick**: `rows_per_stick_padded = ceil(padded_channels / 16)`
3. **Add padding rows**: After each stick's actual data, fill remaining rows with zeros

### Example: 16 Channels, 3×3 Kernel

**Configuration**:
- `out_channels = 16`
- `padded_out_channels = 32` (rounded up to tile size)
- `data_rows_per_stick = ceil(16/16) = 1` (actual data rows)
- `rows_per_stick_padded = ceil(32/16) = 2` (padded for tilization)

**Before Tilization (Face-by-Face Layout)**:

Each stick occupies 2 rows: 1 row of data + 1 row of padding zeros.

```
         Cols 0-15 (Face 0)    |    Cols 16-31 (Face 1)
         ==================    |    ===================
Row 0:   1 1 1 .. 1 1 1       |    9 9 9 .. 9 9 9      (stick 0 + stick 8)
Row 1:   0 0 0 .. 0 0 0       |    0 0 0 .. 0 0 0      (padding)
Row 2:   2 2 2 .. 2 2 2       |    0 0 0 .. 0 0 0      (stick 1)
Row 3:   0 0 0 .. 0 0 0       |    0 0 0 .. 0 0 0      (padding)
Row 4:   3 3 3 .. 3 3 3       |    0 0 0 .. 0 0 0      (stick 2)
Row 5:   0 0 0 .. 0 0 0       |    0 0 0 .. 0 0 0      (padding)
Row 6:   4 4 4 .. 4 4 4       |    0 0 0 .. 0 0 0      (stick 3)
Row 7:   0 0 0 .. 0 0 0       |    0 0 0 .. 0 0 0      (padding)
Row 8:   5 5 5 .. 5 5 5       |    0 0 0 .. 0 0 0      (stick 4)
Row 9:   0 0 0 .. 0 0 0       |    0 0 0 .. 0 0 0      (padding)
Row 10:  6 6 6 .. 6 6 6       |    0 0 0 .. 0 0 0      (stick 5)
Row 11:  0 0 0 .. 0 0 0       |    0 0 0 .. 0 0 0      (padding)
Row 12:  7 7 7 .. 7 7 7       |    0 0 0 .. 0 0 0      (stick 6)
Row 13:  0 0 0 .. 0 0 0       |    0 0 0 .. 0 0 0      (padding)
Row 14:  8 8 8 .. 8 8 8       |    0 0 0 .. 0 0 0      (stick 7)
Row 15:  0 0 0 .. 0 0 0       |    0 0 0 .. 0 0 0      (padding)
         ==================    |    ===================
Row 16-31: All zeros (padding for kernel positions 9-31)
```

**Key observations**:
- Stick 0 is in Face 0, Row 0 (absolute row 0)
- Stick 8 is in Face 1, Row 0 (absolute row 16) - they appear on same physical row!
- Each stick has 1 row of data followed by 1 row of zeros
- Sticks 1-7 only have data in Face 0, zeros in Face 1 (since kernel only has 9 positions)

**After Tilization**:

```
Shape: [1, 1, 32, 32] in TILE layout (padded to tile size)

Row 0:  [1, 1, 1, ..., 1, 0, 0, ..., 0]  (16 ones + 16 zeros - stick 0)
Row 1:  [2, 2, 2, ..., 2, 0, 0, ..., 0]  (16 twos + 16 zeros - stick 1)
Row 2:  [3, 3, 3, ..., 3, 0, 0, ..., 0]  (16 threes + 16 zeros - stick 2)
Row 3:  [4, 4, 4, ..., 4, 0, 0, ..., 0]  (16 fours + 16 zeros - stick 3)
Row 4:  [5, 5, 5, ..., 5, 0, 0, ..., 0]  (16 fives + 16 zeros - stick 4)
Row 5:  [6, 6, 6, ..., 6, 0, 0, ..., 0]  (16 sixes + 16 zeros - stick 5)
Row 6:  [7, 7, 7, ..., 7, 0, 0, ..., 0]  (16 sevens + 16 zeros - stick 6)
Row 7:  [8, 8, 8, ..., 8, 0, 0, ..., 0]  (16 eights + 16 zeros - stick 7)
Row 8:  [9, 9, 9, ..., 9, 0, 0, ..., 0]  (16 nines + 16 zeros - stick 8)
Row 9-31: [0, 0, 0, ..., 0, 0, 0, ..., 0]  (padding)
```

**This is stick-by-stick layout!** Each row contains all channel values (16 data + 16 padding) for one kernel position.

### Algorithm for Non-Multiple-of-32 Channels

```
1. Calculate padded_out_channels = round_up(out_channels, 32)
2. Calculate data_rows_per_stick = ceil(out_channels / 16)
3. Calculate rows_per_stick_padded = ceil(padded_out_channels / 16)
4. For each kernel_position:
   a. Place data in data_rows_per_stick rows
   b. Padding rows (rows_per_stick_padded - data_rows_per_stick) are left as zeros
   c. Advance current_absolute_row by rows_per_stick_padded
```

### Comparison: 16 vs 32 Channels

| Metric | 16 Channels | 32 Channels |
|--------|-------------|-------------|
| `out_channels` | 16 | 32 |
| `padded_out_channels` | 32 | 32 |
| `data_rows_per_stick` | 1 | 2 |
| `rows_per_stick_padded` | 2 | 2 |
| Padding rows per stick | 1 | 0 |
| Output shape | [32, 32, 1, 1] | [32, 32, 1, 1] |

## Supported Configurations

- **Channel counts**: Any multiple of 16 (16, 32, 48, 64, 80, 96, ...)
- **Channels < 32**: Padded to 32 with zeros for proper tilization
- **Channels between 32 and 64**: Padded to 64, etc.
- **Kernel sizes**: 3×3 (9 positions), 5×5 (25 positions), etc.
- **Kernel position padding**: Padded to nearest multiple of 32

## Code Location

Implementation: `ttnn/cpp/ttnn/operations/conv/conv2d/prepare_conv2d_weights.cpp`
- Function: `convert_conv_weight_tensor_to_2d_depthwise_layout`
- Helper: `conv_2d_depthwise_weight_layout_helper`
