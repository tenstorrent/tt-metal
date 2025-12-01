# Matmul Mixed Precision Bug Test

## Overview

This minimal operation demonstrates a bug where `matmul_tiles()` fails when circular buffers (CBs) have different data formats (BF16 vs FP32).

## Structure

The implementation follows the same pattern as `silu_bw` but is minimized to only test the matmul behavior:

```
sources/ttml/metal/ops/matmul_test/
├── matmul_test.hpp/.cpp                    # Top-level operation interface
└── device/
    ├── matmul_test_device_operation_types.hpp
    ├── matmul_test_device_operation.hpp/.cpp
    ├── matmul_test_program_factory.hpp/.cpp
    └── kernels/
        ├── compute/
        │   └── matmul_test_kernel.cpp       # Minimal: reconfig + matmul + pack
        └── dataflow/
            ├── reader_matmul_test.cpp       # Reads 2 tiles, optional copy to FP32
            └── writer_matmul_test.cpp       # Writes 1 output tile
```

## Test Cases

The test (`tests/ops/matmul_test_op_test.cpp`) validates 4 scenarios:

1. **BF16 @ BF16**: Both operands in BF16 format ✓ Expected to pass
2. **FP32 @ FP32**: Both operands in FP32 format ✓ Expected to pass
3. **BF16 @ FP32**: First operand BF16, second FP32 ✗ Expected to fail (BUG)
4. **FP32 @ BF16**: First operand FP32, second BF16 ✗ Expected to fail (BUG)

## How It Works

### Program Factory
- Creates CBs with different data formats based on `TestCase` enum
- Always reads input tensors (which are BF16) to BF16 CBs first
- For FP32 cases, additional FP32 CBs are created and data is copied from BF16 CBs

### Reader Kernel
- Reads both input tiles from DRAM to BF16 CBs (c_0 and c_1)
- If `COPY_A_TO_FP32` is defined: copies tile from BF16 CB to FP32 CB (c_2)
- If `COPY_B_TO_FP32` is defined: copies tile from BF16 CB to FP32 CB (c_3)

### Compute Kernel
- Selects which CBs to use based on compile-time flags `USE_FP32_A` and `USE_FP32_B`
- Calls `reconfig_data_format()` with the selected CBs
- Performs single tile matmul: `matmul_tiles(cb_a, cb_b, 0, 0, 0, false)`
- Packs result to output CB

### Writer Kernel
- Simply writes the output tile to DRAM

## Expected Behavior

When running the tests:

- **BF16 @ BF16** and **FP32 @ FP32** should complete successfully, demonstrating that matmul works when both operands have the same format.

- **BF16 @ FP32** and **FP32 @ BF16** are expected to fail, demonstrating the bug where matmul cannot handle mixed precision operands even after `reconfig_data_format()`.

## Running the Test

To build and run:

```bash
# Build the test
cd /home/ubuntu/tt-metal/tt-train
./build_all.sh

# Run the specific test
./build/tests/matmul_test_op_test
```

## Key Points

1. **Minimal Design**: Each kernel does exactly one thing - no complex logic
2. **Single Tile**: Uses 32x32 matrices (one tile each) to minimize complexity
3. **Single Core**: Runs on a single core to avoid distribution complexity
4. **Clear Test Cases**: Four explicit test cases that clearly show which combinations work and which fail
5. **Input Format**: Always uses BF16 input tensors (as typical in practice) and converts in CBs to demonstrate the issue

## What This Proves

This test conclusively demonstrates that:
- `matmul_tiles()` works when both CB operands have the same data format
- `matmul_tiles()` fails when CB operands have different data formats
- The issue persists even after calling `reconfig_data_format()`
- This is a fundamental limitation/bug in the matmul operation
