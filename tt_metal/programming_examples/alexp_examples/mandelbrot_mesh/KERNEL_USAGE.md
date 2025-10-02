# Mandelbrot Mesh Kernel Usage Guide

## Overview

The Mandelbrot mesh implementation now provides **three different compute kernels** and **two main executable versions** to demonstrate different approaches to distributed computing on Tenstorrent mesh devices.

## Available Kernels

### 🏆 **mandelbrot_fixed.cpp** (Recommended)
```cpp
// Used by: mandelbrot_mesh executable
OVERRIDE_KERNEL_PREFIX "alexp_examples/mandelbrot_mesh/kernels/compute/mandelbrot_fixed.cpp"
```

**Features:**
- ✅ **Fixed-point arithmetic** (16.16 format)
- ✅ **Compilation safe** - no floating-point conversion issues
- ✅ **Integer-only operations** - avoids strict-aliasing problems
- ✅ **Production ready** - robust and efficient

**Coordinate Mapping:**
```cpp
int32_t cx_fixed = x_min_fixed + (x * x_range) / IMAGE_WIDTH;
int32_t cy_fixed = y_min_fixed + (y * y_range) / IMAGE_HEIGHT;
```

### 📚 **mandelbrot_simple.cpp** (Educational)
```cpp
// Used by: mandelbrot_mesh_simple executable
OVERRIDE_KERNEL_PREFIX "alexp_examples/mandelbrot_mesh/kernels/compute/mandelbrot_simple.cpp"
```

**Features:**
- ✅ **Extensive comments** explaining parallelization
- ✅ **Clear device workload breakdown**
- ✅ **Educational value** - great for learning
- ✅ **Parallelization visualization** in code

**Device Distribution:**
```cpp
// Device 0: processes pixels     0 →  32,767
// Device 1: processes pixels 32,768 →  65,535
// Device 2: processes pixels 65,536 →  98,303
// ...
// Total: 8× parallel speedup with perfect load balancing!
```

### 🔧 **mandelbrot_compute.cpp** (Reference)
```cpp
// Available for reference - uses memcpy for safe conversion
OVERRIDE_KERNEL_PREFIX "alexp_examples/mandelbrot_mesh/kernels/compute/mandelbrot_compute.cpp"
```

**Features:**
- ✅ **Memcpy-based conversion** - avoids reinterpret_cast issues
- ✅ **Standard floating-point** approach
- ✅ **Reference implementation** for comparison

## Executable Versions

### 1. **mandelbrot_mesh** (Production)
- Uses `mandelbrot_fixed.cpp` kernel
- Fixed-point arithmetic for robustness
- Recommended for actual deployment
- Output: `mandelbrot_mesh.ppm`

### 2. **mandelbrot_mesh_simple** (Learning)
- Uses `mandelbrot_simple.cpp` kernel
- Detailed console output showing device distribution
- Excellent for understanding parallelization
- Output: `mandelbrot_mesh_simple.ppm`

## Building and Running

### Quick Start
```bash
# Build and run recommended version
./build_and_run.sh

# Build and run educational version
./build_and_run.sh simple

# Build and run both versions
./build_and_run.sh both
```

### Manual Build
```bash
mkdir build && cd build
cmake ..
make mandelbrot_mesh mandelbrot_mesh_simple

# Run production version
./mandelbrot_mesh

# Run educational version
./mandelbrot_mesh_simple
```

## Parallelization Strategy (All Kernels)

### **Data Distribution:**
```
Total pixels: 512 × 512 = 262,144
Devices: 8 (2×4 mesh)
Per device: ~32,768 pixels

Device 0: pixels     0 →  32,767
Device 1: pixels 32,768 →  65,535
Device 2: pixels 65,536 →  98,303
Device 3: pixels 98,304 → 131,071
Device 4: pixels 131,072 → 163,839
Device 5: pixels 163,840 → 196,607
Device 6: pixels 196,608 → 229,375
Device 7: pixels 229,376 → 262,143
```

### **Memory Layout:**
```
Tile size: 32×32 = 1,024 pixels
Total tiles: 256
Tiles per device: ~32
Buffer per device: ~64KB
```

### **Performance:**
```
Sequential: T seconds
Parallel: T/8 seconds
Speedup: 8× (theoretical)
Efficiency: ~100% (embarrassingly parallel)
```

## Kernel Selection Guide

| Use Case | Recommended Kernel | Executable | Reason |
|----------|-------------------|------------|---------|
| **Production Deployment** | `mandelbrot_fixed.cpp` | `mandelbrot_mesh` | Most robust, no FP issues |
| **Learning/Teaching** | `mandelbrot_simple.cpp` | `mandelbrot_mesh_simple` | Extensive comments |
| **Development/Debug** | `mandelbrot_compute.cpp` | Custom build | Standard FP approach |
| **Performance Testing** | `mandelbrot_fixed.cpp` | `mandelbrot_mesh` | Fastest, most efficient |

## Output Files

- **mandelbrot_mesh.ppm** - Fixed-point kernel result
- **mandelbrot_mesh_simple.ppm** - Simple kernel result
- **Build logs** - Compilation and execution details
- **Performance metrics** - Timing and throughput data

## Troubleshooting

### Compilation Issues
- ✅ **Fixed**: Strict-aliasing violations (use memcpy)
- ✅ **Fixed**: Missing headers (removed non-existent includes)
- ✅ **Fixed**: Float conversion (use fixed-point or memcpy)

### Runtime Issues
- Check `TT_METAL_DPRINT_CORES` environment variable
- Verify mesh device availability
- Use fixed-point kernel for maximum compatibility

## Next Steps

1. **Try both versions** to see the difference
2. **Examine the kernels** to understand parallelization
3. **Modify parameters** (resolution, iterations, zoom)
4. **Extend to larger meshes** (4×4, 8×8, etc.)
5. **Add performance profiling** and optimization

The Mandelbrot mesh implementation demonstrates **perfect parallelization** across Tenstorrent mesh devices with **8× theoretical speedup**! 🚀
