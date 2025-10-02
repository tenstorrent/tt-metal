# Distributed Elementwise Add Example

This example demonstrates distributed elementwise addition using TT-Metalium's distributed programming model on a mesh device configuration.

## Overview

The example performs elementwise addition between:
- **Distributed buffer A**: Sharded across mesh devices (1×2 configuration)
- **Replicated buffer B**: Replicated on all mesh devices
- **Result buffer C**: Distributed output buffer

The operation computes: `C[i] = A[i] + sum(B[0..r_tiles-1])` for each tile.

## Files

### Main Program
- `distributed_elementwise_add.cpp` - Main program implementing distributed elementwise addition

### Kernels
- `kernels/replicated_read.cpp` - Data movement kernel for reading distributed and replicated buffers
- `kernels/replicated_add.cpp` - Compute kernel performing the elementwise addition

### Build Configuration
- `CMakeLists.txt` - Build configuration for the example
- `../CMakeLists.txt` - Parent directory build configuration

## Features

- **Distributed Memory Management**: Uses `MeshBuffer` with sharded and replicated configurations
- **Mesh Device Programming**: Demonstrates 1×2 mesh device programming
- **Custom Kernels**: Includes both data movement and compute kernels
- **Tensor Accessor API**: Uses modern TensorAccessor for buffer access
- **Circular Buffer Management**: Proper CB configuration for multi-input operations

## Usage

### Build from tt_metal root:
```bash
cd /home/tt-metal-apv
cmake -B build-cmake -S . -DTT_METAL_BUILD_PROGRAMMING_EXAMPLES=ON
cmake --build build-cmake --target alexp_distributed_elementwise_add -j$(nproc)
```

### Run the example:
```bash
# Run with default number of tiles
./build-cmake/tt_metal/programming_examples/alexp_examples/e4/elementwise_add/programming_examples/alexp_examples/e4/elementwise_add/alexp_distributed_elementwise_add 4

# Run with custom number of tiles
./build-cmake/tt_metal/programming_examples/alexp_examples/e4/elementwise_add/programming_examples/alexp_examples/e4/elementwise_add/alexp_distributed_elementwise_add 8
```

### Parameters
- `num_tiles`: Total number of tiles to process (must be positive integer)

## Implementation Details

### Buffer Configuration
- **Distributed Buffer A**: Sharded row-major across mesh devices
- **Replicated Buffer B**: Same data replicated on all devices
- **Output Buffer C**: Distributed sharded output

### Kernel Flow
1. **Reader Kernel**: Reads tiles from both distributed and replicated buffers
2. **Compute Kernel**: Performs accumulated addition across replicated tiles
3. **Writer Kernel**: Writes results to distributed output buffer

### Mesh Configuration
- Mesh shape: 1×2 (1 row, 2 columns)
- Tile size: 32×32 elements (4096 bytes)
- Data format: Float32

## Verification

The example includes built-in verification that compares computed results against expected golden values, reporting the number of passed/failed elements.

## Directory Structure

```
elementwise_add/
├── distributed_elementwise_add.cpp    # Main program
├── kernels/
│   ├── replicated_read.cpp            # Data movement kernel
│   └── replicated_add.cpp             # Compute kernel
├── CMakeLists.txt                     # Build configuration
└── README.md                          # This file
```
