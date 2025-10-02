# Multi-Core Mandelbrot Scaling Guide

## Overview

This document explains how to scale the Mandelbrot implementation from **1 core per device** to **64+ cores per device** for maximum performance.

## Current vs Multi-Core Architecture

### 🔴 Current Single-Core Implementation
```
8 devices × 1 core = 8 total cores
├── Device 0: 1 core (0,0) → 524,288 pixels
├── Device 1: 1 core (0,0) → 524,288 pixels
├── Device 2: 1 core (0,0) → 524,288 pixels
├── Device 3: 1 core (0,0) → 524,288 pixels
├── Device 4: 1 core (0,0) → 524,288 pixels
├── Device 5: 1 core (0,0) → 524,288 pixels
├── Device 6: 1 core (0,0) → 524,288 pixels
└── Device 7: 1 core (0,0) → 524,288 pixels

Utilization: ~1% of available cores per device
```

### 🟢 Multi-Core Implementation Options

#### **Option 1: 4×4 = 16 Cores Per Device**
```
8 devices × 16 cores = 128 total cores (16× speedup)
├── Device 0: 16 cores (4×4 grid) → 32,768 pixels per core
├── Device 1: 16 cores (4×4 grid) → 32,768 pixels per core
├── Device 2: 16 cores (4×4 grid) → 32,768 pixels per core
├── Device 3: 16 cores (4×4 grid) → 32,768 pixels per core
├── Device 4: 16 cores (4×4 grid) → 32,768 pixels per core
├── Device 5: 16 cores (4×4 grid) → 32,768 pixels per core
├── Device 6: 16 cores (4×4 grid) → 32,768 pixels per core
└── Device 7: 16 cores (4×4 grid) → 32,768 pixels per core

Utilization: ~25% of available cores per device
```

#### **Option 2: 8×8 = 64 Cores Per Device (Maximum)**
```
8 devices × 64 cores = 512 total cores (64× speedup)
├── Device 0: 64 cores (8×8 grid) → 8,192 pixels per core
├── Device 1: 64 cores (8×8 grid) → 8,192 pixels per core
├── Device 2: 64 cores (8×8 grid) → 8,192 pixels per core
├── Device 3: 64 cores (8×8 grid) → 8,192 pixels per core
├── Device 4: 64 cores (8×8 grid) → 8,192 pixels per core
├── Device 5: 64 cores (8×8 grid) → 8,192 pixels per core
├── Device 6: 64 cores (8×8 grid) → 8,192 pixels per core
└── Device 7: 64 cores (8×8 grid) → 8,192 pixels per core

Utilization: ~100% of available cores per device
```

## Implementation Changes

### 1. **Core Range Expansion**

**Before:**
```cpp
auto target_tensix_core = CoreRange(CoreCoord{0, 0}); // Single core
```

**After:**
```cpp
// Multi-core grid
uint32_t cores_x = 4; // or 8 for maximum
uint32_t cores_y = 4; // or 8 for maximum
auto all_cores = CoreRange({0, 0}, {cores_x - 1, cores_y - 1});
```

### 2. **Work Distribution**

**Before:**
```cpp
// Single core handles all tiles for the device
SetRuntimeArgs(program, compute, CoreCoord{0, 0}, {
    num_tiles, x_min, x_max, y_min, y_max, device_id
});
```

**After:**
```cpp
// Each core gets a subset of tiles
for (each core in grid) {
    uint32_t core_tiles = total_tiles / num_cores;
    uint32_t core_start_pixel = device_start + (core_id * pixels_per_core);
    uint32_t core_end_pixel = core_start_pixel + pixels_per_core;

    SetRuntimeArgs(program, compute, core, {
        core_tiles, x_min, x_max, y_min, y_max, device_id,
        core_start_pixel, core_end_pixel  // NEW: Core-specific range
    });
}
```

### 3. **Kernel Modifications**

**Extended Runtime Arguments:**
```cpp
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t x_min_bits = get_arg_val<uint32_t>(1);
    uint32_t x_max_bits = get_arg_val<uint32_t>(2);
    uint32_t y_min_bits = get_arg_val<uint32_t>(3);
    uint32_t y_max_bits = get_arg_val<uint32_t>(4);
    uint32_t device_id = get_arg_val<uint32_t>(5);
    uint32_t core_start_pixel = get_arg_val<uint32_t>(6); // NEW
    uint32_t core_end_pixel = get_arg_val<uint32_t>(7);   // NEW

    // Core coordinates for debugging
    uint32_t core_x = get_core_coord_x();
    uint32_t core_y = get_core_coord_y();
}
```

## Performance Analysis

### **Theoretical Speedup**

| **Configuration** | **Total Cores** | **Speedup vs Single** | **Pixels per Core** |
|-------------------|------------------|------------------------|----------------------|
| **Single-Core**   | 8               | 1×                     | 524,288             |
| **4×4 Multi**     | 128             | 16×                    | 32,768              |
| **8×8 Multi**     | 512             | 64×                    | 8,192               |

### **Expected Performance**

For a **2048×2048 Mandelbrot** computation:

- **Single-Core**: ~30-60 seconds
- **4×4 Multi-Core**: ~2-4 seconds (16× faster)
- **8×8 Multi-Core**: ~0.5-1 second (64× faster)

### **Memory Considerations**

- **L1 Memory per Core**: ~1.5MB (Wormhole)
- **Tile Size**: 2KB (32×32 × 2 bytes)
- **Tiles per Core**: Limited by L1 memory (~750 tiles max)
- **Optimal Tile Count**: 32-256 tiles per core for best performance

## Usage Examples

### **Conservative Multi-Core (16 cores per device)**
```cpp
MandelbrotConfig config;
config.cores_per_device = 16; // 4×4 grid
```

### **Maximum Multi-Core (64 cores per device)**
```cpp
MandelbrotConfig config;
config.cores_per_device = 64; // 8×8 grid
```

### **Custom Grid**
```cpp
MandelbrotConfig config;
config.cores_per_device = 36; // 6×6 grid
```

## Build and Run

```bash
# Build multi-core version
./build_multi_core.sh

# Run with different core counts
export MANDELBROT_CORES_PER_DEVICE=16
./build-cmake/programming_examples/mandelbrot_multi_core_mesh

export MANDELBROT_CORES_PER_DEVICE=64
./build-cmake/programming_examples/mandelbrot_multi_core_mesh
```

## Debug Multi-Core Execution

```bash
# Enable debug for all cores
export TT_METAL_DPRINT_CORES="all"
export TT_METAL_DPRINT_ENABLE=1
./build-cmake/programming_examples/mandelbrot_multi_core_mesh > multi_core_debug.log 2>&1

# Analyze core usage
grep "CORE(" multi_core_debug.log | head -20
```

## Benefits of Multi-Core Scaling

### ✅ **Advantages**
1. **Massive Speedup**: 16-64× faster computation
2. **Better Resource Utilization**: Use available hardware efficiently
3. **Scalable**: Easy to adjust core count based on workload
4. **SPMD Simplicity**: Same kernel code on all cores

### ⚠️ **Considerations**
1. **Memory Bandwidth**: More cores = more memory pressure
2. **Synchronization**: Ensure proper work distribution
3. **Load Balancing**: Handle remainder tiles appropriately
4. **Debug Complexity**: More cores = more debug messages

## Conclusion

By scaling from **1 core** to **64 cores per device**, the Mandelbrot implementation can achieve **64× speedup** while utilizing the full computational power of Tenstorrent hardware. This demonstrates the massive parallel processing capabilities available when leveraging multiple Tensix cores! 🚀
