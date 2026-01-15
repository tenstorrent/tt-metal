# Fuse LLK Calls

Fuse multiple LLK (Low-Level Kernel) calls together to improve performance by eliminating intermediate value spilling to circular buffers (CB), reducing register wait/commit overhead, and keeping intermediate values in registers.

Investigate every function call deeply and act as a simulator to what instructions/effect each call has.
The job is to minimize number of issued instructions and function calls.

## Overview

This command helps optimize kernel performance by:

1. **Identifying sequences of SFPU/FPU operations** and their inits that can be fused together
2. **Removing intermediate synchronization calls** (`tile_regs_commit()`, `tile_regs_wait()`, `cb_pop_front()`, `cb_push_back()`)
3. **Chaining operations together** to keep intermediate values in registers
4. **Preserving final synchronization** operations needed for correctness

## How It Works

### Step 1: Identify LLK Operations and associated init calls

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

### Step 2: Find Fusion Opportunities

Identify consecutive operations that:
- Are close together
- Have only intermediate synchronization calls between them
- Can safely be fused without breaking correctness
- Create a new temporary document to which copy implementation of used LLKs

### Step 3: Fuse Operations

For each group of operations to fuse:

1. **Collect all operations**: Combine init calls, operation calls, and done calls
2. **Remove intermediate syncs**: Remove `tile_regs_commit()`, `tile_regs_wait()`, intermediate `cb_pop_front()`, `cb_push_back()`
3. **Preserve final syncs**: Keep final `cb_reserve_back()`, `pack_tile()`, `cb_push_back()`, `tile_regs_release()`
4. **Keep original code for easy testing**
