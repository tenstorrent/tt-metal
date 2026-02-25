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

## Running Tests


### Testing rules

Never use standard pytest, prefer using `tt-test.sh` (add --dev flag for debug mode). This gives standard pytest output while also managing device ownership and state.

Python environment is sourced with
`source python_env/bin/activate`

Run the tests with

`./tt-test.sh [--dev] <test_file_path.py>`

You should be attentive of device hangs when running tests, especially for new / just implemented operations and kernels.
Watch the bash output at all times, if pytest console output stops, suspect a hang.
If a hang occurs, kill the process with

`pkill -9 -f pytest || true`

After that, reset the device with

`tt-smi -r`


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
- `build_metal.sh`: Main build script

## Kernel Development

### Kernel Types
1. **Reader kernels** (`tt_metal/kernels/dataflow/`): Read data from DRAM/other cores via NoC
2. **Compute kernels** (`tt_metal/kernels/compute/`): Execute FPU/SFPU operations
3. **Writer kernels** (`tt_metal/kernels/dataflow/`): Write results to DRAM/other cores

### Circular Buffers
Circular buffers (CBs): SRAM-based page queues for producer-consumer synchronization between kernels (Reader/Compute/Writer) within and across Tensix cores.
API calls:
- cb_reserve_back(cb_id, num_pages): Producer blocks until space available for writing num_pages
- cb_push_back(cb_id, num_pages): Producer publishes written num_pages to consumer
- cb_wait_front(cb_id, num_pages): Consumer blocks until num_pages available for reading
- cb_pop_front(cb_id, num_pages): Consumer frees processed num_pages
Typical pattern: cb_wait_front → process → cb_pop_front → cb_reserve_back → write → cb_push_back


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

**tech_reports/tensor_sharding/tensor_sharding.md**: Guide to dividing tensors across memory banks using four strategies (height, width, block, ND sharding) for performance optimization
Reference when you need to understand how tensors could be sharded between cores for better data locality.


## Creating New TTNN Operations

Use the `/create-op` skill to create new TTNN operations using the Python-based generic_op infrastructure with `ttnn.generic_op()` and ProgramDescriptor APIs.

Pipeline: analyzer → architect → generic_op_builder → TDD kernel-writer

Key features:
- `architect` combines planning + kernel design into a single agent (determines CB layout, maps helpers, registers TDD stages in `.tdd_state.json`), then `generic_op_builder` runs (reads `.tdd_state.json` to discover stages)
- Kernel implementation uses stage-gated TDD (see `/tdd-kernels` skill) — kernel-writer implements one stage at a time, forbidden from implementing future stages
- Supports interactive (default) and fully automated modes

Operation code: `ttnn/ttnn/operations/{op_name}/`
Tests: `tests/ttnn/unit_tests/operations/{op_name}/`

### Additional Resources
- `.claude/QUICK_START.md` - End-to-end workflow example

## Worktree Development

Worktrees provide fully isolated development environments with their own C++ build and Python venv.

### First thing to do in a new worktree
The hook only creates the git worktree + submodules. You MUST kick off the C++ build yourself as a **background Bash task** immediately on entry:
```bash
MAIN_REPO="$(git rev-parse --show-superproject-working-tree 2>/dev/null || echo "")" && ./build_metal.sh --debug --enable-ccache ${MAIN_REPO:+--cpm-source-cache "$MAIN_REPO/.cpmcache"} && ./create_venv.sh --force && touch .worktree_ready
```
Run this in the background (`run_in_background: true`), then continue working. Check `.worktree_ready` before running tests.

### How it works
- `EnterWorktree` or `isolation: "worktree"` creates the worktree and inits submodules
- Build runs as a background task (~4 min with ccache)
- No env vars needed — CWD auto-detection handles everything
- `tt-test.sh` auto-resolves paths from its own location (works in any worktree)
- Device access is serialized via flock (multiple worktrees share devices safely)

### Running tests in a worktree
```
source python_env/bin/activate && pytest <test_path>
# or
./tt-test.sh [--dev] <test_path>
```

### Manual setup (without Claude Code)
```
.claude/scripts/worktree-setup.sh <path>
```

### Checking build status
- `.worktree_ready` exists → ready to use
- `.worktree_setup.log` → build output (if using worktree-setup.sh)
