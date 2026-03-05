# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TT-Metal is Tenstorrent's neural network library for AI hardware (Wormhole, Blackhole chips):
- **TT-NN**: High-level neural network operations (Python/C++)
- **TT-Metalium**: Low-level kernel programming framework

## Essential Workflow

**ALWAYS required before any command:**
```bash
source python_env/bin/activate
```

**Standard build:**
```bash
./build_metal.sh -c --build-tests
```

**Run tests:**
```bash
pytest tests/python_api_testing/unit_testing/ -vvv
```

## Key Build Commands

- `./build_metal.sh -c` - Standard build (always use `-c` for ccache)
- `./build_metal.sh -c --build-all` - Build everything including examples
- `./build_metal.sh --clean && ./build_metal.sh -c` - Clean build
- `./build_metal.sh -c --debug` - Debug build with symbols

**IMPORTANT**: You do NOT need to rebuild when changing device kernels (files in `device/kernels/` directories). Device kernels are compiled at runtime during kernel creation. Only rebuild when changing:
- Host-side code (program factories, Python bindings, etc.)
- Kernel defines or compile-time configuration

## Testing

- `pytest path/to/test.py -vvv` - Single test
- `pytest -m post_commit` - CI tests
- `TT_LOGGER_LEVEL=Debug pytest <test>` - Debug logging
- `./tests/scripts/run_python_api_unit_tests.sh` - Unit test script

## Code Structure

- `tt_metal/` - Core runtime, device APIs, kernels
- `ttnn/` - High-level operations and Python bindings
- `models/` - Model implementations and demos
- `tests/` - Test suites

## Development Standards

- C++20 codebase, avoid excessive templates
- Conventional commits: `<type>(<scope>): <subject>`
- Common scopes: `ttnn`, `tt_metal`, `model`
- Use `[skip ci]` prefix for docs-only changes
- Lint with: `cmake --preset clang-tidy; cmake --build --preset clang-tidy`

## Documentation Resources

- **DeepWiki**: https://deepwiki.com/tenstorrent/tt-metal
  - Use for deeper technical knowledge about hardware architecture, programming model, and TT-Metal concepts
  - Reference when working with multi-device operations, sharding, or low-level APIs
  - Available via MCP tool: `mcp__deepwiki__ask_question`

- **DeepSeek-V3**: https://deepwiki.com/deepseek-ai/DeepSeek-V3
  - Use for DeepSeek-V3 model implementation details
  - Reference for MoE architecture, routing mechanisms, and design rationale
  - Available via MCP tool: `mcp__deepwiki__ask_question`


- **DeepEP**: https://deepwiki.com/deepseek-ai/DeepEP
  - Use for DeepSeek Expert Parallel (DeepEP) knowledge
  - Reference for expert parallelization strategies and distribution across devices
  - Available via MCP tool: `mcp__deepwiki__ask_question`


## Testing Best Practices

### Pytest Fixtures (conftest.py)

**Always use existing fixtures from `/localdev/mbezulj/tt-metal/conftest.py`** - never redefine them:

- **`mesh_device`** (line 527): Multi-device mesh configurations
  - Automatically opens/closes mesh and submeshes
  - Can parametrize with tuple `(rows, cols)` for 2D grid or `int` for 1D line
  - Auto-detects available devices if not parametrized
  - Handles fabric config, device params
  - Usage: Just add `mesh_device` parameter to test function

- **`device`**: Single device fixture (function or module scope)
  - Use `@pytest.mark.use_module_device` for module-scoped device

- **Specialized mesh fixtures**:
  - `pcie_mesh_device` - PCIe devices
  - `t3k_single_board_mesh_device` - T3K single board
  - `bh_1d_mesh_device` - Blackhole 1D mesh
  - `bh_2d_mesh_device` - Blackhole 2D mesh

### Logging in Tests

**ALWAYS use loguru logger, never print():**
```python
from loguru import logger

# Good
logger.info(f"Tensor shape: {shape}")

# Bad
print(f"Tensor shape: {shape}")
```

### Running Tests

```bash
# Always activate environment first
source python_env/bin/activate

# Run with verbose output
pytest path/to/test.py -vvv -s

# Run specific test function
pytest path/to/test.py::test_function_name -vvv
```

## Device APIs

### Querying Device Information

```python
# Worker (Tensix) cores
worker_grid = device.compute_with_storage_grid_size()  # Returns CoreCoord(x, y)
total_cores = worker_grid.x * worker_grid.y

# DRAM cores
dram_grid = device.dram_grid_size()  # Returns CoreCoord(x, y)
total_dram = dram_grid.x * dram_grid.y

# Mesh device info
mesh_shape = mesh_device.shape  # Returns MeshShape
num_devices = mesh_device.get_num_devices()
device_ids = mesh_device.get_device_ids()
```

## Multi-Device Operations

### Creating Sharded Tensors Across Mesh

To shard a tensor across mesh devices:

```python
import torch
import ttnn

# Create torch tensor
torch_tensor = torch.randn(4096, 8192)

# Create mesh mapper
# dims=(None, -1) means: replicate on mesh dim 0, shard on tensor dim -1 across mesh dim 1
mesh_mapper = ttnn.ShardTensor2dMesh(
    mesh_device,
    mesh_shape=mesh_device.shape,
    dims=(None, -1)  # (mesh_row_dim, mesh_col_dim)
)

# Convert to ttnn tensor with sharding
mesh_tensor = ttnn.from_torch(
    torch_tensor,
    mesh_mapper=mesh_mapper,
    layout=ttnn.TILE_LAYOUT,
    device=mesh_device,
    dtype=ttnn.bfloat16,
)
```

**Sharding dimensions**:
- `dims=(None, -1)`: Replicate across mesh rows, shard tensor's last dim across mesh columns
- `dims=(-1, None)`: Shard tensor's last dim across mesh rows, replicate across mesh columns
- `dims=(-2, -1)`: Shard tensor's 2nd-to-last dim across mesh rows, last dim across mesh columns

### Visualizing Tensors

To visualize tensor distribution across mesh devices:

```python
# Visualize sharded tensor across mesh
ttnn.visualize_tensor(mesh_tensor)

# Prints a nice table showing:
# - Placement configuration
# - Each device's ID and mesh coordinates
# - Data slice held by each device
# - Shape, dtype, and layout per device
```

To inspect tensor properties:
```python
# Properties (no parentheses)
dtype = mesh_tensor.dtype
layout = mesh_tensor.layout

# Methods (with parentheses)
storage_type = mesh_tensor.storage_type()
is_sharded = mesh_tensor.is_sharded()
```

## Fabric Configuration & Collective Operations

### Setting Fabric Configuration

Fabric configurations must be set via `device_params` before opening mesh device:

```python
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING},
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=["device_params"],
)
def test_with_fabric(mesh_device, device_params):
    # mesh_device will be initialized with the fabric config
    ...
```

**Available fabric configs**:
- `ttnn.FabricConfig.FABRIC_1D_RING` - 1D ring topology (wraps around)
- `ttnn.FabricConfig.FABRIC_1D` - 1D line topology (linear)
- `ttnn.FabricConfig.DISABLED` - No fabric (default)

### Querying Available Ethernet Links

Check how many ethernet links are available for CCL operations:

```python
from models.tt_transformers.tt.ccl import get_num_links

# Query links across all axes (returns minimum)
num_links = get_num_links(mesh_device)

# Query links along specific mesh axis
num_links_rows = get_num_links(mesh_device, cluster_axis=0)  # Vertical (North-South)
num_links_cols = get_num_links(mesh_device, cluster_axis=1)  # Horizontal (East-West)
```

### All-Gather Operation

Gather distributed tensor data across all devices:

```python
# Ring topology - each device talks to neighbors in a ring
gathered_tensor = ttnn.all_gather(
    mesh_tensor,
    dim=-1,  # Tensor dimension to gather along
    num_links=1,  # Number of ethernet links to use
    topology=ttnn.Topology.Ring
)

# Linear topology - unidirectional line communication
gathered_tensor = ttnn.all_gather(
    mesh_tensor,
    dim=-1,  # Tensor dimension to gather along
    cluster_axis=1,  # Which mesh dimension to gather along (0=rows, 1=cols)
    num_links=1,  # Number of ethernet links to use
    topology=ttnn.Topology.Linear
)
```

**Parameters**:
- `dim`: Tensor dimension to gather along (e.g., -1 for last dimension)
- `num_links`: Number of ethernet links to use (both topologies support this)
- `cluster_axis`: **(Linear only)** Which mesh dimension forms the line
  - `0` = gather along rows (vertical direction)
  - `1` = gather along columns (horizontal direction)
- `topology`: `ttnn.Topology.Ring` (wraparound) or `ttnn.Topology.Linear` (no wraparound)

**Result**: After all_gather, all devices will have the complete tensor replicated.

### Reduce-Scatter Operation

Reduce data across devices then scatter unique portions to each device:

```python
# Ring topology - reduce then scatter in a ring
scattered_tensor = ttnn.reduce_scatter(
    mesh_tensor,
    dim=-1,  # Tensor dimension to reduce and scatter along
    num_links=1,  # Number of ethernet links to use
    topology=ttnn.Topology.Ring
)

# Linear topology - reduce then scatter along a line
scattered_tensor = ttnn.reduce_scatter(
    mesh_tensor,
    dim=-1,  # Tensor dimension to reduce and scatter along
    cluster_axis=1,  # Which mesh dimension to scatter along (0=rows, 1=cols)
    num_links=1,  # Number of ethernet links to use
    topology=ttnn.Topology.Linear
)
```

**Parameters**: Same as all_gather (dim, num_links, cluster_axis for Linear, topology)

**Result**: After reduce_scatter, each device gets a unique portion of the reduced tensor.

**Key Differences**:
- `all_gather`: Collects data and replicates the full result on every device
- `reduce_scatter`: Reduces (sums) data across devices, then distributes unique portions to each
- For N devices: all_gather increases tensor size by N×, reduce_scatter decreases by N×

## System Health Tool

Check connectivity status between chips before running multi-device tests:

```bash
build/test/tt_metal/tt_fabric/test_system_health
```

**Purpose**:
- Verifies ethernet links between devices
- Checks fabric connectivity health
- Useful for debugging mesh/fabric issues before running tests

Run this tool if fabric tests fail to ensure all chips are properly connected.

## Tracy Profiling Workflow

To profile operations and analyze performance:

### Step 1: Run tests with Tracy profiling

```bash
python -m tracy -r -m pytest test_playground.py -k fabric
```

This captures Tracy profiling data for all fabric tests and generates a CSV report.

### Step 2: Analyze with perf_report tool

```bash
python ../tt-perf-report/src/tt_perf_report/perf_report.py \
    /localdev/mbezulj/tt-metal/generated/profiler/reports/<timestamp>/ops_perf_results_<timestamp>.csv \
    --ignore-signpost --no-advice --no-summary
```

**Flags:**
- `--ignore-signpost`: Include entire file (don't filter by signposts)
- `--no-advice`: Skip performance advice generation
- `--no-summary`: Skip summary generation (show only operation list)

**Output:**
- Crunched list of operations with timing, core usage, and performance metrics
- Signposts appear as markers (🪧) for easy test identification
- Device time, op-to-op gap, core count, FLOP counts per operation

The Tracy signposts added in tests help identify which configuration is running.
