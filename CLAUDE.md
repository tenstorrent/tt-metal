# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the **tt-metal** repository, containing TT-Metalium (low-level programming model) and TT-NN (high-level neural network operations library) for Tenstorrent AI accelerators. The codebase implements tensor operations and kernels that run on Tenstorrent hardware, which uses a unique architecture based on Tensix cores with tile-based computing (32×32 tiles).

## Key Architecture Concepts

### Hardware Architecture
- **Tensix cores**: Each contains 5 RISC-V CPUs (2 data movement, 1 unpack, 1 math, 1 pack), 1.5MB SRAM (L1), NoC interfaces, FPU (matrix unit), and SFPU (vector unit)
- **Tile-based computing**: Native 32×32 element tiles for efficient matrix operations
- **No cache hierarchy**: Explicit SRAM management and DMA operations required
- **Network-on-Chip (NoC)**: Two bidirectional NoCs for data movement between cores and DRAM

### Programming Model
- **Three kernel types per Tensix core**: Reader (data input via NoC0), Compute (FPU/SFPU operations), Writer (data output via NoC1)
- **Circular buffers**: Inter-kernel communication mechanism in SRAM
- **SPMD execution**: Single Program Multiple Data across cores (most common pattern)
- **Fast dispatch**: Asynchronous command queuing (enabled by default, critical for performance)

Refer to `METALIUM_GUIDE.md` for detailed architecture explanations.

## Building the Code

### Standard Build
```bash
./build_metal.sh
```

### Build with Tests
```bash
./build_metal.sh --build-tests
```

### Build Programming Examples
```bash
./build_metal.sh --build-programming-examples
```

### Build Types
```bash
# Debug build (includes debug symbols)
./build_metal.sh -b Debug
./build_metal.sh --debug

# Release build (default)
./build_metal.sh -b Release
./build_metal.sh --release

# Development build (optimized with debug info)
./build_metal.sh -b RelWithDebInfo
./build_metal.sh --development
```

### Clean Build Artifacts
```bash
./build_metal.sh --clean
```

### Other Build Options
- `--enable-ccache`: Enable ccache for faster rebuilds
- `--export-compile-commands`: Generate compile_commands.json
- `--build-ttnn-tests`: Build only ttnn tests
- `--build-metal-tests`: Build only metal tests

## Running Tests

### Post-Commit Tests (Required Before Commits)
```bash
./build_metal.sh --build-tests
./tests/scripts/run_tests.sh --tt-arch $ARCH_NAME --pipeline-type post_commit
```

### Python API Tests
```bash
pytest tests/python_api_testing/unit_testing/ -vvv
pytest tests/python_api_testing/sweep_tests/pytests/ -vvv
```

### TTNN Unit Tests
```bash
# Run a single test file
pytest tests/ttnn/unit_tests/operations/data_movement/test_concat.py

# Run all tests in a directory
pytest tests/ttnn/unit_tests/operations/
```

### TTNN Sweep Tests
```bash
# Generate test vectors first
python tests/sweep_framework/sweeps_parameter_generator.py --dump-file

# Run sweep tests (vectors from file, results to local JSON)
python tests/sweep_framework/sweeps_runner.py --vector-source vectors_export --result-dest results_export

# Run specific module
python tests/sweep_framework/sweeps_runner.py --module-name eltwise.unary.relu.relu --vector-source vectors_export --result-dest results_export
```

### C++ Tests (Googletest)
```bash
# Build tests first
./build_metal.sh --build-tests

# Run specific test with gtest filter
./build/test/tt_metal/unit_tests_api --gtest_filter="MeshDispatchFixture.TensixDRAMLoopbackSingleCore"

# Run test with slow dispatch mode
export TT_METAL_SLOW_DISPATCH_MODE=1
./build/test/tt_metal/unit_tests/unit_tests_api --gtest_filter="MeshDeviceSingleCardBufferFixture.TestL1BuffersAllocatedTopDown"
```

### Model Performance Tests
```bash
# Virtual machine
./tests/scripts/run_tests.sh --tt-arch $ARCH_NAME --pipeline-type models_performance_virtual_machine

# Bare metal
./tests/scripts/run_tests.sh --tt-arch $ARCH_NAME --pipeline-type models_performance_bare_metal
```

## Environment Variables

### Essential Variables
```bash
export PYTHONPATH=/path/to/tt-metal  # Required for Python imports
export ARCH_NAME=wormhole_b0         # or grayskull, blackhole
```

### Debugging and Development
```bash
# Enable watcher (validates NoC transactions, on-device assertions)
export TT_METAL_WATCHER=10  # Update every 10 seconds

# Enable debug printing from kernels
export TT_METAL_DPRINT_CORES=(0,0)-(4,4)  # Print from 5x5 grid

# Enable operation logging
export TT_LOGGER_TYPES=Op
export TT_LOGGER_LEVEL=DEBUG

# Make ops blocking with logging (useful for debugging)
export TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false, "enable_logging": true}'

# Enable slow dispatch mode (debugging only, not for production)
export TT_METAL_SLOW_DISPATCH_MODE=1

# Enable RISC-V debug info in ELF files
export TT_METAL_RISCV_DEBUG_INFO=1
```

## Development Workflow

### Virtual Environment Setup
```bash
# Create and activate Python environment
./create_venv.sh
source python_env/bin/activate
```

### Running Programming Examples
```bash
# Verify installation
python3 -m ttnn.examples.usage.run_op_on_device
```

### Device Reset
If device hangs or misbehaves:
```bash
tt-smi -tr 0  # Reset device 0
```

## Debugging Guide

### Host-Side Debugging
```bash
# Build with debug symbols
./build_metal.sh -b Debug

# Run with GDB
gdb --args <generated_binary>
gdb --args python <python_file>
```

### Device-Side Debugging
- **Always develop with Watcher enabled**: `export TT_METAL_WATCHER=10`
- **Use Debug Print API**: Include `debug/dprint.h` in kernels, use `DPRINT << x << ENDL();`
- **Watcher flags**: NoC transaction errors, illegal operations, assertions

### Common Debug Workflows
1. Enable watcher to catch errors early
2. Use DPRINT for kernel debugging
3. Check watcher output if device hangs
4. Use slow dispatch mode only when debugging dispatch issues

## Code Organization

### Key Directories
- `tt_metal/`: Core TT-Metalium implementation (C++ low-level APIs, kernels, hardware abstraction)
- `ttnn/`: TT-NN high-level operations library (Python/C++ APIs)
- `models/`: Model implementations and demos
- `tests/`: Test suites (unit tests, sweep tests, integration tests)
- `tech_reports/`: Architecture documentation and optimization guides
- `tt_metal/programming_examples/`: Example kernels and operations

### Important Files
- `METALIUM_GUIDE.md`: Comprehensive architecture and programming guide
- `README.md`: High-level project overview
- `INSTALLING.md`: Installation instructions
- `CONTRIBUTING.md`: Contribution guidelines and workflows
- `build_metal.sh`: Main build script

## Kernel Development

### Kernel Types
1. **Reader kernels** (`tt_metal/kernels/dataflow/`): Read data from DRAM/other cores via NoC
2. **Compute kernels** (`tt_metal/kernels/compute/`): Execute FPU/SFPU operations
3. **Writer kernels** (`tt_metal/kernels/dataflow/`): Write results to DRAM/other cores

### Circular Buffers
- Use `CBIndex::c_0`, `c_1`, ... for input buffers
- Use `CBIndex::c_16` commonly for output buffers
- Coordinate between kernels using `cb_reserve_back`, `cb_push_back`, `cb_wait_front`, `cb_pop_front`

### Compute APIs
- **Abstraction layer** for hardware compatibility across generations (Grayskull, Wormhole, Blackhole)
- Use provided compute APIs (`add_tiles`, `sin_tile`, etc.) instead of low-level implementations
- Different hardware has different vector widths and instruction sets

### Data Flow
Typical pattern: NoC0 → Unpacker → FPU/SFPU → Packer → NoC1

## Common Pitfalls

1. **Fast vs Slow Dispatch**: Always use fast dispatch (default) for production. Slow dispatch is debugging-only.
2. **Tile Alignment**: Tensors must align to 32×32 tiles (dimensions padded to multiples of 32)
3. **Memory Management**: No automatic caching—explicit SRAM allocation and DMA required
4. **SPMD Work Distribution**: Use `tt::tt_metal::split_work_to_cores()` to distribute work evenly
5. **Kernel Arguments**: Must set runtime args for ALL cores, even if some do no work (set args to no-op)
6. **Security**: Avoid command injection, validate inputs, especially in device-side code

## Documentation and Resources

### Tech Reports (in `tech_reports/`)
- **LLMs**: LLM bring-up and optimization (`tech_reports/LLMs/llms.md`)
- **CNNs**: CNN optimization strategies (`tech_reports/CNNs/cnn_optimizations.md`)
- **Flash Attention**: Attention implementation (`tech_reports/FlashAttention/FlashAttention.md`)
- **Programming Mesh of Devices**: Multi-device programming (`tech_reports/Programming_Mesh_of_Devices/`)
- **Performance Optimizations**: Advanced optimization techniques (`tech_reports/AdvancedPerformanceOptimizationsForModels/`)

### Online Documentation
- API Reference: https://docs.tenstorrent.com/tt-metal/latest/
- TT-NN Reference: https://docs.tenstorrent.com/tt-metal/latest/ttnn/index.html
- TT-Metalium Reference: https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/index.html

### Deep Wiki Integration
For high-level architectural questions, hardware details, design patterns, and best practices, use the Deep Wiki MCP tool with repository path "tenstorrent/tt-metal". Deep Wiki has comprehensive indexing of architecture docs and tech reports.

## Contributing

- All commits must pass post-commit tests before merging
- File issues before starting work on features/bugs
- Follow code review process (see `CONTRIBUTING.md`)
- Use pre-commit hooks (installed via `.pre-commit-config.yaml`)
- Branch naming: Use descriptive names (e.g., `feature/add-conv-op`, `bugfix/fix-l1-allocation`)

## Performance Considerations

- **Tile sizes**: Operations on 32×32 tiles are native and most efficient
- **NoC bandwidth**: Minimize data movement, use sharded memory when beneficial
- **SRAM capacity**: 1.5MB per Tensix—keep intermediate results in SRAM when possible
- **Operator fusion**: Less critical than on CPU/GPU due to abundant on-chip SRAM
- **Tracy Profiler**: Use for performance analysis (enabled by default, disable with `--disable-profiler`)
