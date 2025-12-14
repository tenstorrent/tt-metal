# API Reference - Tensor Sharding and Profiling

## Tensor Sharding Information

### Get Memory Configuration
```python
# Method call (requires parentheses)
mem_config = tensor.memory_config()
```

### Memory Layout (Property)
```python
# Property access (no parentheses)
layout = mem_config.memory_layout

# Returns: TensorMemoryLayout enum
# Values:
#   - TensorMemoryLayout.INTERLEAVED
#   - TensorMemoryLayout.HEIGHT_SHARDED
#   - TensorMemoryLayout.WIDTH_SHARDED
#   - TensorMemoryLayout.BLOCK_SHARDED
```

### Legacy ShardSpec (Property)
```python
# Property access (no parentheses)
shard_spec = mem_config.shard_spec  # May be None

if shard_spec is not None:
    # Access properties
    grid = shard_spec.grid          # CoreRangeSet
    shape = shard_spec.shape        # Tuple (shard_h, shard_w)
    orientation = shard_spec.orientation  # ShardOrientation enum
```

### ND ShardSpec (Property)
```python
# Property access (no parentheses)
nd_shard_spec = mem_config.nd_shard_spec  # May be None

if nd_shard_spec is not None:
    # Access properties
    grid = nd_shard_spec.grid                    # CoreRangeSet
    shard_shape = nd_shard_spec.shard_shape     # Shape tuple
    orientation = nd_shard_spec.orientation      # ShardOrientation enum
    strategy = nd_shard_spec.shard_distribution_strategy  # Distribution strategy
```

### Core Grid Information
```python
# Get core grid from shard spec
core_grid = shard_spec.grid  # CoreRangeSet object

# Get number of cores
num_cores = core_grid.num_cores()

# Grid is printed as: {[(x=0,y=0) - (x=7,y=3)]}
# This represents a rectangular core range
```

### Complete Example
```python
def print_tensor_sharding_info(tensor):
    """Print detailed sharding information for a tensor."""
    mem_config = tensor.memory_config()

    print(f"Memory Layout: {mem_config.memory_layout}")

    # Check legacy shard spec
    if mem_config.shard_spec is not None:
        shard_spec = mem_config.shard_spec
        print(f"  Shard Spec:")
        print(f"    Grid: {shard_spec.grid}")
        print(f"    Num cores: {shard_spec.grid.num_cores()}")
        print(f"    Shard shape: {shard_spec.shape}")
        print(f"    Orientation: {shard_spec.orientation}")

    # Check ND shard spec
    if mem_config.nd_shard_spec is not None:
        nd_shard_spec = mem_config.nd_shard_spec
        print(f"  ND Shard Spec:")
        print(f"    Grid: {nd_shard_spec.grid}")
        print(f"    Num cores: {nd_shard_spec.grid.num_cores()}")
        print(f"    Shard shape: {nd_shard_spec.shard_shape}")
        print(f"    Orientation: {nd_shard_spec.orientation}")
        print(f"    Distribution: {nd_shard_spec.shard_distribution_strategy}")
```

---

## Device Profiling

### Enable Profiling
```bash
export TT_METAL_DEVICE_PROFILER=1
```

Or in Python:
```python
os.environ["TT_METAL_DEVICE_PROFILER"] = "1"
```

### Profiler Output Location
```
/workspace/generated/profiler/reports/ops_perf_results.csv
```

### Read Profiler Data
```python
import pandas as pd
from pathlib import Path

profiler_path = Path('/workspace/generated/profiler/reports/ops_perf_results.csv')

if profiler_path.exists():
    df = pd.read_csv(profiler_path)

    # Get kernel durations (in nanoseconds)
    durations_ns = df['DEVICE KERNEL DURATION [ns]']

    # Convert to microseconds
    durations_us = durations_ns / 1_000

    # Convert to milliseconds
    durations_ms = durations_ns / 1_000_000
```

### Wait for New Profiler Entries
```python
def wait_for_profiler_data(initial_count, max_wait_seconds=10, poll_interval=0.5):
    """Wait for new profiler entries to appear."""
    profiler_path = Path('/workspace/generated/profiler/reports/ops_perf_results.csv')

    if not profiler_path.exists():
        return False

    start_time = time.time()
    while time.time() - start_time < max_wait_seconds:
        try:
            df = pd.read_csv(profiler_path)
            current_count = len(df)
            if current_count > initial_count:
                return True
        except Exception:
            pass  # File might be locked
        time.sleep(poll_interval)

    return False
```

---

## Creating Sharded Tensors

### Basic Template
```python
def create_sharded_tensor(device, shape, sharding, cores):
    """
    Create a sharded tensor.

    Args:
        device: TTNN device
        shape: Tuple (height, width)
        sharding: "height", "width", "block", or "interleaved"
        cores: Number of cores (8, 16, 32) or None for interleaved

    Returns:
        TTNN tensor
    """
```

### Interleaved Tensor
```python
if sharding == "interleaved":
    t = torch.randn(1, 1, h, w, dtype=torch.bfloat16)
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
```

### Height Sharded Tensor
```python
# Grid examples for height sharding
if cores == 8:
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(
        ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 7)
    )})
elif cores == 16:
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(
        ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 7)
    )})
elif cores == 32:
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(
        ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 7)
    )})

grid_size = grid.bounding_box().grid_size()

# Calculate shard shape
shard_h = max(32, (((h + grid_size.y - 1) // grid_size.y + 31) // 32) * 32)
shard_w = max(32, ((w + 31) // 32) * 32)

mem_config = ttnn.create_sharded_memory_config(
    (shard_h, shard_w),
    grid,
    ttnn.ShardStrategy.HEIGHT,
    ttnn.ShardOrientation.ROW_MAJOR,
    True  # use_height_and_width_as_shard_shape
)
```

### Width Sharded Tensor
```python
# Grid examples for width sharding
if cores == 8:
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(
        ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0)
    )})
elif cores == 16:
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(
        ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1)
    )})
elif cores == 32:
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(
        ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3)
    )})

grid_size = grid.bounding_box().grid_size()

# Calculate shard shape
shard_h = max(32, ((h + 31) // 32) * 32)
shard_w = max(32, (((w + grid_size.x - 1) // grid_size.x + 31) // 32) * 32)

mem_config = ttnn.create_sharded_memory_config(
    (shard_h, shard_w),
    grid,
    ttnn.ShardStrategy.WIDTH,
    ttnn.ShardOrientation.ROW_MAJOR,
    True
)
```

### Block Sharded Tensor (Shape-Aware)
```python
# IMPORTANT: Compute valid grid based on shape
grid_shape = compute_valid_block_grid(shape, cores)
if grid_shape is None:
    raise ValueError(f"Cannot create valid block grid for {shape} with {cores} cores")

grid_h, grid_w = grid_shape

# Create grid (Note: CoreCoord is (x, y) where x=column, y=row)
grid = ttnn.CoreRangeSet({ttnn.CoreRange(
    ttnn.CoreCoord(0, 0),
    ttnn.CoreCoord(grid_w - 1, grid_h - 1)
)})

grid_size = grid.bounding_box().grid_size()

# Calculate shard shape (divide by grid dimensions)
shard_h = max(32, (((h + grid_size.y - 1) // grid_size.y + 31) // 32) * 32)
shard_w = max(32, (((w + grid_size.x - 1) // grid_size.x + 31) // 32) * 32)

mem_config = ttnn.create_sharded_memory_config(
    (shard_h, shard_w),
    grid,
    ttnn.ShardStrategy.BLOCK,
    ttnn.ShardOrientation.ROW_MAJOR,
    True
)
```

### Create Tensor from Torch
```python
t = torch.randn(1, 1, h, w, dtype=torch.bfloat16)
tensor = ttnn.from_torch(
    t,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=mem_config
)
```

---

## Binary Operations

### Set Grid Selection Strategy
```bash
export TT_METAL_BINARY_NG_GRID_STRATEGY=max_ab
```

Or in Python:
```python
os.environ["TT_METAL_BINARY_NG_GRID_STRATEGY"] = "max_ab"
```

### Available Strategies
- `max_abc` - Uses max_ab logic (max of A and B cores)
- `max_ab` - Uses max(A_cores, B_cores)
- `min_ab` - Uses min(A_cores, B_cores)
- `current` - Prefers C, then A, then B, then full device grid
- `a_first` - Prefers A, then B, then C, then full device grid
- `b_first` - Prefers B, then A, then C, then full device grid
- `full_grid` - Always uses full device grid (64 cores: 8x8)
- `half_grid` - Uses half of device grid (32 cores: 4x8)

### Run Binary Operation
```python
# ADD
result = ttnn.add(tensor_a, tensor_b)

# POWER
result = ttnn.pow(tensor_a, tensor_b)

# Always synchronize after operation
ttnn.synchronize_device(device)
```

### Capture C++ Logs (WORKER_GRID)
```python
import os
import sys
import tempfile
import re

# Capture stderr at file descriptor level
stderr_fd = sys.stderr.fileno()
saved_stderr = os.dup(stderr_fd)

with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as tmp_file:
    tmp_name = tmp_file.name
    os.dup2(tmp_file.fileno(), stderr_fd)
    sys.stderr = tmp_file

    # Run operation
    result = ttnn.add(tensor_a, tensor_b)
    ttnn.synchronize_device(device)

    # Restore stderr
    os.dup2(saved_stderr, stderr_fd)
    sys.stderr = sys.__stderr__

# Read captured output
with open(tmp_name, 'r') as f:
    stderr_output = f.read()

# Parse WORKER_GRID log
worker_grid_match = re.search(r'WORKER_GRID:\s*strategy=(\S+)\s*cores=(\d+)', stderr_output)
if worker_grid_match:
    strategy = worker_grid_match.group(1)
    compute_cores = int(worker_grid_match.group(2))
    print(f"Strategy: {strategy}, Compute cores: {compute_cores}")

# Cleanup
os.unlink(tmp_name)
os.close(saved_stderr)
```

---

## Block Sharding Utilities

### Compute Valid Block Grid
```python
def compute_valid_block_grid(shape, cores):
    """
    Compute a valid block sharding grid for the given shape and cores.

    For block sharding with shape (H, W) and N cores, the grid (GH, GW) must satisfy:
    - GH * GW = N
    - H (padded to tile size 32) must be divisible by GH
    - W (padded to tile size 32) must be divisible by GW

    Args:
        shape: Tuple (height, width)
        cores: Number of cores

    Returns:
        Tuple (grid_h, grid_w) or None if no valid grid exists
    """
    h, w = shape

    # Pad to tile size (32)
    h_padded = max(32, ((h + 31) // 32) * 32)
    w_padded = max(32, ((w + 31) // 32) * 32)

    # Find all divisor pairs of cores
    divisor_pairs = []
    for gh in range(1, cores + 1):
        if cores % gh == 0:
            gw = cores // gh
            divisor_pairs.append((gh, gw))

    # Find valid grids (those that divide the padded shape)
    valid_grids = []
    for gh, gw in divisor_pairs:
        if h_padded % gh == 0 and w_padded % gw == 0:
            valid_grids.append((gh, gw))

    if not valid_grids:
        return None

    # Prefer grids that are closer to square for better load balancing
    valid_grids.sort(key=lambda g: abs(g[0] - g[1]))

    return valid_grids[0]
```

### Examples
```python
# Tensor (1, 1024) - very wide, height=1
compute_valid_block_grid((1, 1024), 8)   # → (2, 4)
compute_valid_block_grid((1, 1024), 16)  # → (4, 4)
compute_valid_block_grid((1, 1024), 32)  # → (4, 8)

# Tensor (1024, 1) - very tall, width=1
compute_valid_block_grid((1024, 1), 8)   # → (2, 4)
compute_valid_block_grid((1024, 1), 16)  # → (4, 4)
compute_valid_block_grid((1024, 1), 32)  # → (4, 8)

# Tensor (1024, 1024) - square
compute_valid_block_grid((1024, 1024), 8)   # → (2, 4)
compute_valid_block_grid((1024, 1024), 16)  # → (4, 4)
compute_valid_block_grid((1024, 1024), 32)  # → (4, 8)
```

---

## Common Patterns

### Check if Tensor is Sharded
```python
if tensor.is_sharded():
    print("Tensor is sharded")
else:
    print("Tensor is interleaved")
```

### Get Device Grid Size
```python
compute_grid_size = device.compute_with_storage_grid_size()
num_cores = compute_grid_size.x * compute_grid_size.y
print(f"Device has {num_cores} cores ({compute_grid_size.x}x{compute_grid_size.y})")
```

### Convert to PyTorch
```python
torch_tensor = ttnn.to_torch(ttnn_tensor)
```

### Deallocate Tensor
```python
tensor.deallocate()
```

---

## Error Handling

### Common Errors

**"ValueError: BLOCK needs 16+ cores"**
- Old error when using hardcoded grids
- Fixed by using `compute_valid_block_grid()`

**"Cannot create valid block sharding grid for shape X with Y cores"**
- No valid grid exists for this combination
- Try different core count or shape

**"AttributeError: 'Tensor' object has no attribute 'tensor_spec'"**
- Use `tensor.spec` not `tensor.tensor_spec`
- Or use `tensor.memory_config()` for memory config

---

## Reference Locations

### Code
- `tests/ttnn/benchmarks/binary_ng/example_single_test.py`
- `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp`

### Documentation
- `BLOCK_SHARDING_FIX.md` - Detailed explanation
- `SESSION_EXPORT.md` - Full session documentation
- `QUICK_REFERENCE.md` - Quick commands

**Last Updated**: November 13, 2025
