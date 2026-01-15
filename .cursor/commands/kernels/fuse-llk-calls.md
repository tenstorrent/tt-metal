# Fuse LLK Calls

Fuse multiple LLK (Low-Level Kernel) calls together to improve performance by eliminating intermediate value spilling to circular buffers (CB), reducing register wait/commit overhead, and keeping intermediate values in registers.

## Usage

**Quick Start:**
```bash
python .cursor/commands/kernels/fuse_llk_calls.py <kernel_file>
```

**With output file:**
```bash
python .cursor/commands/kernels/fuse_llk_calls.py <kernel_file> --output <output_file>
```

**Dry run (preview changes):**
```bash
python .cursor/commands/kernels/fuse_llk_calls.py <kernel_file> --dry-run
```

**Examples:**
```bash
# Fuse operations in-place
python .cursor/commands/kernels/fuse_llk_calls.py tt_metal/kernels/compute/eltwise_sfpu.cpp

# Fuse operations to a new file
python .cursor/commands/kernels/fuse_llk_calls.py kernels/compute/my_kernel.cpp --output kernels/compute/my_kernel_fused.cpp

# Preview changes without modifying file
python .cursor/commands/kernels/fuse_llk_calls.py kernels/compute/my_kernel.cpp --dry-run
```

## Overview

This command helps optimize kernel performance by:

1. **Identifying sequences of SFPU/FPU operations** that can be fused together
2. **Removing intermediate synchronization calls** (`tile_regs_commit()`, `tile_regs_wait()`, `cb_pop_front()`, `cb_push_back()`)
3. **Chaining operations together** to keep intermediate values in registers
4. **Preserving final synchronization** operations needed for correctness

## Performance Benefits

Fusing LLK calls can provide significant performance improvements by:

- **Eliminating CB spilling**: Intermediate values stay in registers instead of being written to and read from circular buffers
- **Reducing synchronization overhead**: Fewer `tile_regs_commit()` and `tile_regs_wait()` calls
- **Better register utilization**: Multiple operations can share register space more efficiently
- **Reduced memory traffic**: Less data movement between registers and L1 memory

## How It Works

### Step 1: Identify LLK Operations

The tool searches for patterns indicating LLK operations:

**SFPU Init Patterns:**
- `exp_tile_init()`, `log_tile_init()`, `add_binary_tile_init()`
- `llk_math_eltwise_unary_sfpu_init<...>()`
- `llk_math_eltwise_unary_sfpi_init<...>()`
- `_llk_math_eltwise_unary_sfpu_init_<...>()`

**SFPU Operation Patterns:**
- `exp_tile(0)`, `log_tile(0)`, `add_binary_tile(0, 1, 0)`
- `llk_math_eltwise_unary_sfpu<...>(...)`
- `llk_math_eltwise_unary_sfpi<...>(...)`
- `call_sfpu_operation<...>(...)`

**SFPU Done Patterns:**
- `_llk_math_eltwise_unary_sfpu_done()`

### Step 2: Find Fusion Opportunities

The tool identifies consecutive operations that:
- Are close together (within 5 lines)
- Have only intermediate synchronization calls between them
- Can safely be fused without breaking correctness

### Step 3: Fuse Operations

For each group of operations to fuse:

1. **Collect all operations**: Combine init calls, operation calls, and done calls
2. **Remove intermediate syncs**: Remove `tile_regs_commit()`, `tile_regs_wait()`, intermediate `cb_pop_front()`, `cb_push_back()`
3. **Preserve final syncs**: Keep final `cb_reserve_back()`, `pack_tile()`, `cb_push_back()`, `tile_regs_release()`
4. **Replace code**: Replace the operation sequence with fused version

## Example Transformations

### Before (Unfused)

```cpp
// Compute softplus: log(exp(input) + 1)
exp_tile_init();
exp_tile(0);  // exp(input)

tile_regs_commit();
tile_regs_wait();

add_binary_tile_init();
add_binary_tile(0, 1, 0);  // exp(input) + 1

tile_regs_commit();
tile_regs_wait();

log_tile_init();
log_tile(0);  // log(exp(input) + 1)

tile_regs_commit();
tile_regs_wait();
```

### After (Fused)

```cpp
// Fused LLK operations
exp_tile_init();
exp_tile(0);  // exp(input)
add_binary_tile_init();
add_binary_tile(0, 1, 0);  // exp(input) + 1
log_tile_init();
log_tile(0);  // log(exp(input) + 1)

tile_regs_commit();
tile_regs_wait();
```

## Supported Operations

The tool can fuse:

- **SFPU unary operations**: `exp`, `log`, `sin`, `cos`, `tanh`, `relu`, `gelu`, etc.
- **SFPU binary operations**: `add`, `mul`, `sub`, etc.
- **FPU operations**: Element-wise operations using FPU
- **Chained operations**: Multiple consecutive operations of the same or different types

## Limitations

The tool has some limitations:

1. **Register pressure**: Very long chains may exceed available registers
2. **Data dependencies**: Operations with complex dependencies may not fuse correctly
3. **CB dependencies**: Operations that require specific CB states may not fuse
4. **Loop boundaries**: Operations across loop iterations are not fused

## Best Practices

1. **Review fused code**: Always review the output to ensure correctness
2. **Test thoroughly**: Run tests after fusing to verify behavior
3. **Check register usage**: Monitor register pressure for long fused chains
4. **Use dry-run first**: Preview changes before applying them
5. **Incremental fusion**: Fuse operations incrementally and test between changes

## Troubleshooting

### No Operations Found

If no operations are found:
- Check that the file contains LLK calls (SFPU/FPU operations)
- Verify the file is a kernel compute file (not unpack/pack)
- Check that operations use standard LLK API patterns

### No Fusion Opportunities

If no fusion opportunities are found:
- Operations may be too far apart
- There may be non-sync code between operations
- Operations may have dependencies that prevent fusion

### Incorrect Fused Code

If fused code is incorrect:
- Check register usage - may need to reduce fusion
- Verify data dependencies are preserved
- Ensure final synchronization is correct
- Review the fused code manually

## Related Commands

- Kernel optimization tools
- Performance profiling tools
- Register usage analysis tools

## Implementation Details

The tool uses regex pattern matching to identify:
- LLK operation calls
- Synchronization calls
- Operation boundaries

It then uses a greedy algorithm to:
- Group consecutive operations
- Identify removable synchronization calls
- Generate fused code

The tool preserves:
- Comments
- Code structure
- Final synchronization
- Operation correctness
