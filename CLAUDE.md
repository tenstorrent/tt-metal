# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

TT-Metal is Tenstorrent's software stack for programming their AI accelerators. It provides two main layers:

- **TT-Metalium** (`tt_metal/`): Low-level programming model for kernel development
- **TT-NN** (`ttnn/`): High-level Python & C++ neural network operations library built on TT-Metalium

## Build Commands

```bash
# Standard build
./build_metal.sh

# Build with all tests
./build_metal.sh --build-tests

# Build all targets including examples
./build_metal.sh --build-all

# Debug build
CONFIG=Debug ./build_metal.sh

# Clang-tidy linting
cmake --preset clang-tidy && cmake --build --preset clang-tidy
```

## Environment Setup

```bash
# After building, activate the Python environment
source python_env/bin/activate

# Set PYTHONPATH for model demos
export PYTHONPATH=$(pwd)

# Debug logging
export TT_LOGGER_LEVEL=Debug

# Enable Watcher for device debugging (updates every 10 seconds)
export TT_METAL_WATCHER=10
```

## Running Tests

```bash
# Run post-commit regression tests
./tests/scripts/run_tests.sh --tt-arch $ARCH_NAME --pipeline-type post_commit

# Run specific pytest
pytest tests/path/to/test.py -vvv

# Run gtest C++ tests with filter
./build/test/tt_metal/unit_tests_api --gtest_filter="TestName*"

# Slow dispatch mode tests
export TT_METAL_SLOW_DISPATCH_MODE=1
./build/test/tt_metal/unit_tests/unit_tests_api --gtest_filter="TestName"
```

## Architecture

### Directory Structure

- `tt_metal/` - Core runtime: device APIs, allocators, dispatch system, low-level kernels
- `ttnn/` - High-level op library with Python/C++ bindings
- `tt_metal/impl/` - Runtime implementation (dispatch, buffers, allocators, device management)
- `tt_metal/fabric/` - Multi-chip mesh management and inter-device communication
- `tt_metal/api/` - Public host API headers
- `ttnn/cpp/ttnn/operations/` - C++ operation implementations organized by category
- `models/` - ML model implementations and demos
- `tests/` - Test infrastructure (sweep tests, unit tests, integration tests)
- `tech_reports/` - Technical documentation on data formats, optimizations, hardware specifics

### Key Abstractions

**Tensor System**: Supports multiple layouts (Row-Major, Tiled) and memory configurations (DRAM, L1). Tensors can be sharded across device meshes.

**Device Mesh**: Multi-device support via fabric-based Ethernet networking. Distributed tensors automatically partition across device clusters.

**Dispatch System**: Programs are dispatched to devices via command queues. Supports async execution and program tracing for optimization.

### Execution Flow

Python/C++ code → TTNN Operations → TT-Metal Dispatch → Device Kernels → Hardware

## Code Style Guidelines

- C++20 codebase with heavy headers; minimize compile-time impact
- Prefer compile-time safety, clarity, and simplicity over cleverness
- Avoid macros when templates or constexpr suffice
- Avoid excessive templates, enable_if, SFINAE unless absolutely necessary
- Follow existing clang-tidy profile (bugprone, performance, modernize, readability)
- Use Loguru for Python logging, Tenstorrent logger for C++ logging
- SPDX license headers required on all source files

## Debugging Device Code

```bash
# Enable Watcher for NoC validation and device assertions
export TT_METAL_WATCHER=10

# Enable kernel debug printing from specific cores
export TT_METAL_DPRINT_CORES=(0,0)-(4,4)

# Disable specific Watcher features if timing-sensitive
export TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1
```

Watcher logs are written to `generated/watcher/watcher.log`.

## Hardware Reset

```bash
# Single card reset
tt-smi -r 0

# Multi-card reset (T3000/QuietBox/LoudBox)
tt-smi -r 0,1,2,3
```

## PR Guidelines

- PRs touching C++/kernels/runtime should run: `./build_metal.sh --build-all`
- Documentation-only PRs: prefix title with `[skip ci]`
- Pre-commit hooks are configured; run `pre-commit run --all-files` to check
