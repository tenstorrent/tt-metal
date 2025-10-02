# Multi-Tensix Distributed Elementwise Add

## Overview

This project implements a **multi-Tensix distributed elementwise addition** that solves the L1 memory pressure issues encountered with large tile counts (18+ tiles) in the single-Tensix version.

## Key Innovation: Multi-Tensix Distribution

### Problem Solved
- **Single-Tensix Limitation**: The original implementation used only 1 Tensix core per device, causing L1 memory pressure with large tile counts
- **Memory Bottleneck**: 80KB L1 memory limit per Tensix core became insufficient for 18+ tiles
- **Failure Boundary**: Consistent failures around 16,400 elements due to buffer underruns

### Solution: Multi-Tensix Parallelization
- **Distribute workload** across multiple Tensix cores per device
- **Reduce L1 memory pressure** by dividing tiles among cores
- **Scale to larger tile counts** without memory constraints

## Architecture

### Multi-Tensix Configuration
```
Device 0: N Tensix cores
Device 1: N Tensix cores
Total: 2N Tensix cores working in parallel
```

### Memory Distribution Per Tensix Core
- **CB0 (A tiles)**: 2-4 tiles per core (vs 12 in single-Tensix)
- **CB1 (B tiles)**: 4-6 tiles per core (vs 10 in single-Tensix)
- **Total per core**: ~32KB (vs 80KB in single-Tensix)
- **L1 Memory Pressure**: **60% reduction per core**

### Workload Distribution
```
Original: 1 core handles 16 tiles → 64KB memory pressure
Multi-Tensix: 4 cores handle 4 tiles each → 16KB memory pressure per core
```

## Usage

```bash
# Build the project
cd /home/tt-metal-apv
cmake --build build-cmake --target multi_tensix_distributed_elementwise_add

# Run with different configurations
./build-cmake/.../multi_tensix_distributed_elementwise_add [num_tiles] [tensix_cores_per_device]

# Examples:
./multi_tensix_distributed_elementwise_add 32 4   # 32 tiles using 4 Tensix cores per device
./multi_tensix_distributed_elementwise_add 64 8   # 64 tiles using 8 Tensix cores per device
./multi_tensix_distributed_elementwise_add 128 16 # 128 tiles using 16 Tensix cores per device
```

## Expected Performance

### Scalability Targets
- **32 tiles**: Expected 100% success with 4 Tensix cores per device
- **64 tiles**: Expected 100% success with 8 Tensix cores per device
- **128+ tiles**: Expected 100% success with 16+ Tensix cores per device

### Memory Efficiency
- **4x reduction** in L1 memory pressure per core
- **Linear scalability** with number of Tensix cores
- **No 16,400 element boundary** limitation

## Technical Details

### Kernel Distribution
Each Tensix core runs:
1. **Reader Kernel (NCRISC)**: Reads assigned A tiles + all B tiles
2. **Compute Kernel (TRISC)**: Performs elementwise addition for assigned tiles
3. **Writer Kernel (BRISC)**: Writes results for assigned tiles

### Synchronization
- **Independent execution**: Each Tensix core operates on disjoint tile ranges
- **No inter-core dependencies**: Eliminates synchronization overhead
- **Mesh-level coordination**: Only at device boundaries

### Advantages Over Single-Tensix
1. **Memory Scalability**: Linear scaling with core count
2. **Performance**: Parallel execution across multiple cores
3. **Reliability**: Eliminates memory pressure failures
4. **Flexibility**: Configurable core count based on workload

## Implementation Notes

- **Tile Assignment**: Round-robin distribution across Tensix cores
- **Buffer Management**: Independent circular buffers per core
- **Error Handling**: Graceful degradation if insufficient cores available
- **Debugging**: Per-core tile range reporting for troubleshooting
