## Repository Overview

This is the **tt-metal** repository, containing TT-Metalium (low-level programming model) and TT-NN (high-level neural network operations library) for Tenstorrent AI accelerators. The codebase implements tensor operations and kernels that run on Tenstorrent hardware, which uses a unique architecture based on Tensix cores with tile-based computing (32×32 tiles).

## Key Architecture Concepts

### Hardware Architecture
- **Tensix cores**: Each contains 5 RISC-V CPUs (2 data movement, 1 unpack, 1 math, 1 pack), 1.5MB SRAM (L1), NoC interfaces, FPU (matrix unit), and SFPU (vector unit)
- **Tile-based computing**: Native 32×32 element tiles for efficient matrix operations
- **Network-on-Chip (NoC)**: Two bidirectional NoCs for data movement between cores and DRAM

### Programming Model
- **Three kernel types per Tensix core**: Reader (data input via NoC0), Compute (FPU/SFPU operations), Writer (data output via NoC1)
- **Circular buffers**: Inter-kernel communication mechanism in SRAM

Refer to `METALIUM_GUIDE.md` for detailed architecture explanations.

## Building the Code

### Standard Build
```bash
./build_metal.sh
```

### Device-side kernels
Kernels build at runtime. DO NOT build when changing kernel code.

### Running tests

Never use standard pytest, prefer using `scripts/tt-test.sh` (add --dev flag for debug mode). This gives standard pytest output while also managing device ownership and state via flock, and resets the device after every run.

Python environment is sourced with
`source python_env/bin/activate`

Run the tests with

`scripts/tt-test.sh [--dev] <test_file_path.py>`

You should be attentive of device hangs when running tests, especially for new / just implemented operations and kernels.
Watch the bash output at all times, if pytest console output stops, suspect a hang.

**NEVER run `tt-smi -r` directly.** Device resets are handled automatically by `tt-test.sh` after every run. Manual resets can corrupt another agent's in-flight test session.

## Code Organization

### Key Directories
- `tt_metal/`: Core TT-Metalium implementation (C++ low-level APIs, kernels, hardware abstraction)
- `ttnn/`: TT-NN high-level operations library (Python/C++ APIs)
- `models/`: Model implementations and demos
- `tests/`: Test suites (unit tests, sweep tests, integration tests)

## Kernel Development

There are 2 types of kernels (dataflow and compute). The compute kernels run on the 2 data movement threads (mostly compiled and written seperately as reader and writer), compute kernel is compiled down to 3 compute threads (unpack, math, pack)

### Compute APIs
- **Abstraction layer** for hardware compatibility across generations (Wormhole, Blackhole)

## Documentation and Resources. USE THESE PROACTIVELY.

### Documentation in tt-metal repo

**METALIUM_GUIDE.md**: Comprehensive architecture and programming guide covering hardware fundamentals (Tensix processors, tile-based operations, NoC), programming model (reader/compute/writer kernels, circular buffers), and multi-chip scaling.
Reference when learning core architecture or developing low-level kernels.

**tech_reports/tensor_accessor/tensor_accessor.md**: TensorAccessor utility
Reference for mapping logical tensor indices to physical memory locations across distributed banks
Reference when you need to understand how kernels access tensor data.

**tech_reports/tensor_layouts/tensor_layouts.md**: Explanation of tensor layouts (row-major vs tiled) and memory layouts (interleaved vs sharded) in TT-Metal.
Reference when dealing with tensor creation, configuring data organization, or understanding page-to-bank mapping.

## Worktree Development

Worktrees provide fully isolated development environments with their own C++ build and Python venv.
