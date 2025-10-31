# Reduce-Scatter Example: Simple 2D Case

## Setup
- **2 devices** (Device 0 and Device 1)
- **Tensor shape**: `[2, 4]` (2 rows, 4 columns)
- **Reduce-scatter along dimension 1** (the column dimension)
- **Operation**: Sum reduction

## Step 1: Initial State (Replicated)

Both devices start with the **same full tensor**:

```
Device 0:                    Device 1:
┌─────────────┐              ┌─────────────┐
│  1   2   3   4 │           │  1   2   3   4 │
│  5   6   7   8 │           │  5   6   7   8 │
└─────────────┘              └─────────────┘
Shape: [2, 4]                Shape: [2, 4]
```

## Step 2: Reduce Phase (Sum Across Devices)

Each device sums its tensor with the corresponding elements from other devices.

Since both devices have the same data:
- Element at [0,0]: Device0(1) + Device1(1) = **2**
- Element at [0,1]: Device0(2) + Device1(2) = **4**
- Element at [0,2]: Device0(3) + Device1(3) = **6**
- Element at [0,3]: Device0(4) + Device1(4) = **8**
- Element at [1,0]: Device0(5) + Device1(5) = **10**
- Element at [1,1]: Device0(6) + Device1(6) = **12**
- Element at [1,2]: Device0(7) + Device1(7) = **14**
- Element at [1,3]: Device0(8) + Device1(8) = **16**

**After reduction, both devices temporarily have:**
```
┌─────────────┐
│  2   4   6   8 │
│ 10  12  14  16 │
└─────────────┘
Shape: [2, 4]
```

## Step 3: Scatter Phase (Partition Along Dimension 1)

The reduced tensor is **partitioned along dimension 1** (columns) and distributed:

- **Device 0** gets columns **0-1** (first half)
- **Device 1** gets columns **2-3** (second half)

```
Device 0:                    Device 1:
┌─────────┐                  ┌─────────┐
│  2   4  │                  │  6   8  │
│ 10  12  │                  │ 14  16  │
└─────────┘                  └─────────┘
Shape: [2, 2]                Shape: [2, 2]
```

## Visual Summary

```
BEFORE (Replicated):
┌─────────────────────────┐  ┌─────────────────────────┐
│ Device 0                │  │ Device 1                │
│ ┌─────────────┐         │  │ ┌─────────────┐         │
│ │  1   2   3   4 │      │  │ │  1   2   3   4 │      │
│ │  5   6   7   8 │      │  │ │  5   6   7   8 │      │
│ └─────────────┘         │  │ └─────────────┘         │
└─────────────────────────┘  └─────────────────────────┘

AFTER Reduce-Scatter:
┌─────────────────────────┐  ┌─────────────────────────┐
│ Device 0                │  │ Device 1                │
│ ┌─────────┐             │  │ ┌─────────┐             │
│ │  2   4  │  ← Sum      │  │ │  6   8  │  ← Sum      │
│ │ 10  12  │     of      │  │ │ 14  16  │     of      │
│ └─────────┘     both    │  │ └─────────┘     both    │
│ (cols 0-1)              │  │ (cols 2-3)              │
└─────────────────────────┘  └─────────────────────────┘
```

---

## Example 2: Different Data on Each Device

### Setup
- **2 devices**
- **Tensor shape**: `[2, 4]`
- **Reduce-scatter along dimension 1**

### Initial State (Replicated, but let's show what happens if data differs)

```
Device 0:                    Device 1:
┌─────────────┐              ┌─────────────┐
│  1   2   3   4 │           │ 10  20  30  40 │
│  5   6   7   8 │           │ 50  60  70  80 │
└─────────────┘              └─────────────┘
```

### After Reduce (Sum)

```
Device 0 & 1 (temporarily):
┌─────────────┐
│ 11  22  33  44 │   ← [1+10, 2+20, 3+30, 4+40]
│ 55  66  77  88 │   ← [5+50, 6+60, 7+70, 8+80]
└─────────────┘
```

### After Scatter (Partition)

```
Device 0:                    Device 1:
┌─────────┐                  ┌─────────┐
│ 11  22  │                  │ 33  44  │
│ 55  66  │                  │ 77  88  │
└─────────┘                  └─────────┘
```

---

## Example 3: 4 Devices, 1D Tensor

### Setup
- **4 devices**
- **Tensor shape**: `[8]` (1D, 8 elements)
- **Reduce-scatter along dimension 0**

### Initial State (Replicated)

```
Device 0: [1, 2, 3, 4, 5, 6, 7, 8]
Device 1: [1, 2, 3, 4, 5, 6, 7, 8]
Device 2: [1, 2, 3, 4, 5, 6, 7, 8]
Device 3: [1, 2, 3, 4, 5, 6, 7, 8]
```

### After Reduce (Sum all 4 devices)

Each element is summed across 4 devices:
- Element 0: 1+1+1+1 = **4**
- Element 1: 2+2+2+2 = **8**
- Element 2: 3+3+3+3 = **12**
- Element 3: 4+4+4+4 = **16**
- Element 4: 5+5+5+5 = **20**
- Element 5: 6+6+6+6 = **24**
- Element 6: 7+7+7+7 = **28**
- Element 7: 8+8+8+8 = **32**

Temporary result: `[4, 8, 12, 16, 20, 24, 28, 32]`

### After Scatter (Partition into 4 chunks)

```
Device 0: [4, 8]        ← elements 0-1
Device 1: [12, 16]      ← elements 2-3
Device 2: [20, 24]      ← elements 4-5
Device 3: [28, 32]      ← elements 6-7
```

---

## Key Points

1. **Reduce**: Sum (or other reduction op) corresponding elements across all devices
   - Each position [i, j, k, ...] gets the sum of that position from all devices

2. **Scatter**: Partition the reduced tensor along the specified dimension
   - If dimension has size N and there are D devices, each device gets N/D elements
   - Partitioning happens in order: Device 0 gets first chunk, Device 1 gets second, etc.

3. **Dimension**: The dimension along which scatter happens determines:
   - Which dimension gets partitioned
   - Other dimensions remain unchanged

4. **Shape Change**:
   - Input shape: `[d0, d1, ..., dk, ..., dn]` where `dk` is the reduction dimension
   - Output shape: `[d0, d1, ..., dk/D, ..., dn]` where D is number of devices
   - `dk` must be divisible by D

---

## Your Test Case Breakdown

In your test:
- **4 devices**
- **Input shape**: `[1, 1, 32, 128]`
- **Reduce-scatter along dimension 3** (the width dimension, size 128)

### Before (Replicated on all 4 devices):
```
Each device: [1, 1, 32, 128]
```

### After Reduce:
Each position is summed across 4 devices:
- Position [0, 0, i, j]: Device0 + Device1 + Device2 + Device3
- Since all devices have the same data initially, each element is multiplied by 4

### After Scatter:
Dimension 3 (size 128) is partitioned into 4 chunks of 32:
- **Device 0**: Gets columns [0:32]   → Shape: `[1, 1, 32, 32]`
- **Device 1**: Gets columns [32:64]  → Shape: `[1, 1, 32, 32]`
- **Device 2**: Gets columns [64:96]  → Shape: `[1, 1, 32, 32]`
- **Device 3**: Gets columns [96:128] → Shape: `[1, 1, 32, 32]`

Each device ends up with 1/4 of the reduced tensor along dimension 3.
