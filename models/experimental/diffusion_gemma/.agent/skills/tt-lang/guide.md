# TT-Lang Practical Guide

This guide preserves the imported practical TT-Lang workflow material. Use [SKILL.md](SKILL.md) as the Claude Code entrypoint and [TTLangSpecification.md](TTLangSpecification.md) as the API source of truth when details conflict.

## External Resources

- [TT-Lang Documentation](https://docs.tenstorrent.com/tt-lang/index.html)
- [TT-Lang Tutorial](https://docs.tenstorrent.com/tt-lang/ttl-tutorial/index.html)
- [TT-Lang Specification](TTLangSpecification.md)
- [Reference Models](extern_references.md)

## Prerequisites

Before writing TT-Lang kernels, confirm your environment:
1. Verify `ttl` and `ttnn` Python packages are available (`import ttl; import ttnn`)
2. Verify hardware access or functional simulator is working
3. If you are having problems connecting to a device, run the smallest local TTNN/TT-Lang smoke test available in the workspace; if hardware access is still unclear, ask the user which machine or simulator path to use

## Key Use Case: Fusing TTNN Operations

A common use case is taking a sequence of TTNN operations and fusing them into a single TT-Lang kernel for better performance. For example:

```python
# Original TTNN program (multiple ops, multiple round trips)
x = ttnn.exp(input)
y = ttnn.add(x, bias)
z = ttnn.relu(y)

# Fused TT-Lang kernel (single kernel, all ops in one compute function)
@ttl.kernel(grid=(1, 1))
def fused_kernel(input, bias, out):
    # ... setup CBs ...
    @ttl.compute()
    def compute():
        with input_dfb.wait() as inp, bias_dfb.wait() as b, out_dfb.reserve() as o:
            # All ops fuse into one compute body
            result = ttl.math.relu(ttl.math.exp(inp) + b)
            o.store(result)
```

**When fusing TTNN ops:**
1. Identify the sequence of ops to fuse
2. Create one DFB per input tensor
3. Chain operations in a single compute function
4. TT-Lang will generate optimized fused code

## TT-Lang Programming Model

### Kernel Structure

Every TT-Lang kernel has exactly three threads that run concurrently:
1. **Compute thread** (`@ttl.compute()`): Math operations on tiles in L1
2. **Reader thread** (`@ttl.datamovement()`): Loads data from DRAM to dataflow buffers
3. **Writer thread** (`@ttl.datamovement()`): Writes data from dataflow buffers to DRAM

These threads synchronize via **dataflow buffers** (DFBs).

### Basic Kernel Template

```python
import ttl

@ttl.kernel(grid=(1, 1))
def add_kernel(lhs, rhs, out):
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), buffer_factor=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with lhs_dfb.wait() as l, rhs_dfb.wait() as r, out_dfb.reserve() as o:
            o.store(l + r)

    @ttl.datamovement()
    def dm_read():
        with lhs_dfb.reserve() as blk:
            tx = ttl.copy(lhs[0, 0], blk)
            tx.wait()
        with rhs_dfb.reserve() as blk:
            tx = ttl.copy(rhs[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()

# Call the kernel directly (no return ttl.Program)
# add_kernel(lhs_tensor, rhs_tensor, out_tensor)
```

### Using Context Managers (Preferred)

The `with` statement automatically handles `pop()` and `push()`:

```python
@ttl.compute()
def compute():
    with input1_dfb.wait() as a, input2_dfb.wait() as b:
        with output_dfb.reserve() as o:
            result = a + b
            o.store(result)
    # pop/push happens automatically at end of with block

@ttl.datamovement()
def dm_read():
    with input1_dfb.reserve() as blk:
        tx = ttl.copy(input1[0, 0], blk)
        tx.wait()
    # push happens automatically
```

### Dataflow Buffer API Reference

```python
# Create a dataflow buffer
dfb = ttl.make_dataflow_buffer_like(
    tensor,           # TTNN tensor to inherit dtype/layout from
    shape=(R, C),     # Block size in tiles (e.g., (2, 2) = 4 tiles per block)
    buffer_factor=2   # Factor of extra blocks in DFB (2 = double buffering) for pipelining
)

# Consumer operations (compute thread consumes data)
blk = dfb.wait()       # Block until data available, returns block
blk.pop()              # Release block back to producer

# Producer operations (datamovement thread produces data)
blk = dfb.reserve()    # Block until space available, returns block
blk.push()             # Signal data is ready for consumer

# Context manager (preferred - auto pop/push)
with dfb.wait() as blk:      # For consumers
    # use blk...
with dfb.reserve() as blk:   # For producers
    # fill blk...

# Block operations
blk.store(expr)             # Store result of expression into block
```

**DFB Shape = Block Size:** The `shape=(R, C)` parameter defines the **block size** in tiles. A block is the unit of data transferred between threads. For tensors larger than one block, use **loops** to iterate over multiple blocks:

Note: buffer factor is a pipeline hint, not a queue depth. Almost all kernels just use 2. You are able to push as many tiles into a CB as you want, it's just a datatype like array or queue, even a buffer_factor=1 dataflow buffer can support hundreds of tiles.

```python
# 128x128 tensor = 4x4 tiles, process in 2x2 blocks (4 iterations)
dfb = ttl.make_dataflow_buffer_like(tensor, shape=(2, 2), buffer_factor=2)

@ttl.datamovement()
def dm_read():
    for row in range(2):      # 2 row-blocks
        for col in range(2):  # 2 col-blocks
            with dfb.reserve() as blk:
                tx = ttl.copy(tensor[row*2:(row+1)*2, col*2:(col+1)*2], blk)
                tx.wait()

@ttl.compute()
def compute():
    for _ in range(4):  # Must match total iterations in dm_read
        with dfb.wait() as blk, out_dfb.reserve() as o:
            o.store(ttl.math.exp(blk))
```

## Available Operations

### Binary Operators

```python
result = a + b      # Element-wise addition
result = a - b      # Element-wise subtraction
result = a * b      # Element-wise multiplication
result = a / b      # Element-wise division
result = a @ b      # Matrix multiplication (equivalent to ttl.math.matmul(a, b))
```

### Binary Functions

```python
result = ttl.math.max(a, b)  # Element-wise maximum
result = ttl.math.min(a, b)  # Element-wise minimum
```

### Unary Functions (ttl.math.*)

```python
result = ttl.math.exp(x)      # Exponential
result = ttl.math.log(x)      # Natural logarithm
result = ttl.math.sqrt(x)     # Square root
result = ttl.math.rsqrt(x)    # Reciprocal square root (1/sqrt(x))
result = ttl.math.recip(x)    # Reciprocal (1/x)
result = ttl.math.tanh(x)     # Hyperbolic tangent
result = ttl.math.sigmoid(x)  # Sigmoid (1/(1+exp(-x)))
result = ttl.math.relu(x)     # ReLU (max(0, x))
result = ttl.math.abs(x)      # Absolute value
result = ttl.math.neg(x)      # Negation (-x)
result = ttl.math.floor(x)    # Floor
result = ttl.math.ceil(x)     # Ceil
result = ttl.math.sign(x)     # Sign (-1, 0, or 1)
result = ttl.math.selu(x, scale, alpha)  # SELU activation
result = ttl.math.fill(x, value)         # Fill block with scalar value (value must be a constant!)
```

### Matrix Multiplication

```python
# Two equivalent ways to do matmul:
result = a @ b                    # @ operator
result = ttl.math.matmul(a, b)   # function call

# Example usage:
with a_dfb.wait() as a_tile, b_dfb.wait() as b_tile, c_dfb.reserve() as c_out:
    c_out.store(a_tile @ b_tile)
```

**Multi-tile matmul:** When CBs hold multiple tiles (e.g., shape=(2, 2)), the compiler generates loops over K dimension and accumulates automatically. The DST register persists across K iterations, enabling proper accumulation. For example, with A[1,2] @ B[2,1] = C[1,1], the K=2 tiles accumulate correctly.

### Power (scalar integer exponent)

```python
# Raises each element to an integer power (top-level, not ttl.math)
result = ttl.power(x, 2)  # x^2
result = ttl.power(x, 3)  # x^3
```

### Transpose

```python
# Transpose tiles (top-level, not ttl.math)
# Takes input block, works with multi-tile CBs
with inp_dfb.wait() as x, out_dfb.reserve() as o:
    o.store(ttl.transpose(x))
```

**Non-square example:** For 4x2 tiles → 2x4 tiles:
```python
inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(4, 2), buffer_factor=2)
out_dfb = ttl.make_dataflow_buffer_like(out, shape=(2, 4), buffer_factor=2)  # Swapped!
```

### Reductions (require scaler tensor)

```python
# Reductions are in ttl.math and need a "scaler" tensor (1x1 DFB of all 1.0s)
# dims=[0] = collapse rows, dims=[1] = collapse columns, dims=[0, 1] = scalar

# Scaler: 32x32 tile of 1.0s in a 1x1 DFB
scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)

with inp_dfb.wait() as i, scaler_dfb.wait() as s, out_dfb.reserve() as o:
    # Scalar reduction (sum/max entire DFB -> single value in output [0,0])
    o.store(ttl.math.reduce_sum(i, s, dims=[0, 1]))
    o.store(ttl.math.reduce_max(i, s, dims=[0, 1]))

    # Collapse rows (reduce along dim 0): (N, M) -> (1, M)
    o.store(ttl.math.reduce_sum(i, s, dims=[0]))

    # Collapse columns (reduce along dim 1): (N, M) -> (N, 1)
    o.store(ttl.math.reduce_sum(i, s, dims=[1]))
```

**Dimension semantics match PyTorch:**
- `dims=[0]` for reduce **collapses rows** (dim 0) - output shape [1, M]
- `dims=[1]` for reduce **collapses columns** (dim 1) - output shape [N, 1]

**Multi-tile reduce:** Reduces across ALL tiles in the input DFB. For example, a 4x1 tile input DFB reduced with `dims=[0, 1]` produces a single scalar value (in a 1x1 output DFB). The reduction sums all elements across all 4 tiles into position [0,0].

### Broadcast

```python
# Broadcast expands a smaller block to match a larger output shape
# dims=[0] = expand dim 0 (rows), dims=[1] = expand dim 1 (cols), dims=[0, 1] = broadcast scalar

with scalar_dfb.wait() as s, out_dfb.reserve() as o:
    # Broadcast 1x1 scalar to fill entire output block
    o.store(ttl.math.broadcast(s, dims=[0, 1]))

with row_dfb.wait() as r, out_dfb.reserve() as o:
    # Broadcast (1,M) row across N rows: dims=[0] expands dim 0
    o.store(ttl.math.broadcast(r, dims=[0]))

with col_dfb.wait() as c, out_dfb.reserve() as o:
    # Broadcast (N,1) column across M columns: dims=[1] expands dim 1
    o.store(ttl.math.broadcast(c, dims=[1]))
```

**Broadcast dimension semantics (match PyTorch):**
- `dims=[0]` for broadcast **expands dim 0** (copies row to all rows) - input (1, M) -> output (N, M)
- `dims=[1]` for broadcast **expands dim 1** (copies column to all columns) - input (N, 1) -> output (N, M)

Note: Reduce and broadcast use matching dims. `dims=[1]` reduce collapses columns to produce (N, 1), `dims=[1]` broadcast expands that column back to (N, M).

### Conditional Select

```python
result = ttl.where(condition, true_val, false_val)
```

### Operation Fusion

Operations chain automatically - no need for store/reload between ops:

```python
@ttl.compute()
def fused_compute():
    with input_dfb.wait() as a, bias_dfb.wait() as b, out_dfb.reserve() as o:
        # All these ops fuse into one efficient compute body
        x = ttl.math.exp(a)
        y = x + b
        z = ttl.math.sigmoid(y)
        result = ttl.math.relu(z)
        o.store(result)
```

**Limitation:** Ops that take DFB arguments (matmul, reduce, transpose, broadcast) cannot be fused with each other. Each must have its own `with` block and store. Broadcast cannot be fused with elementwise ops either.

**When fusion fails:** Use sequential `with` blocks to break the chain - you do NOT need separate kernels:

```python
@ttl.compute()
def compute():
    # CORRECT: Break into two with blocks (still one kernel!)
    with a_dfb.wait() as a, b_dfb.wait() as b, intermediate_dfb.reserve() as inter:
        inter.store(a @ b)

    with intermediate_dfb.wait() as inter, scaler_dfb.wait() as s, out_dfb.reserve() as o:
        o.store(ttl.math.reduce_sum(inter, s, dims=[0, 1]))
```

The compiler fuses 20+ elementwise ops in a single compute function without issues.

## Kernel Design: Minimize DRAM Traffic

**Strive for one fused kernel.** Multiple kernels are fine for incremental development and debugging, but each kernel boundary creates DRAM round-trips. For production:

- **One kernel = one DRAM read + one DRAM write** (ideal)
- **Two kernels = read → compute → write → read → compute → write** (2x DRAM traffic)
- **N kernels = N× DRAM traffic** (avoid)

```python
# BAD: Two kernels = 2x DRAM traffic
@ttl.kernel(grid=(1, 1))
def kernel1(inp, temp):
    # Read inp from DRAM, write temp to DRAM
    ...

@ttl.kernel(grid=(1, 1))
def kernel2(temp, out):
    # Read temp from DRAM, write out to DRAM
    ...

# GOOD: One fused kernel = 1x DRAM traffic
@ttl.kernel(grid=(1, 1))
def fused_kernel(inp, out):
    # Read inp from DRAM once, all compute in L1, write out to DRAM once
    # Use intermediate CBs (L1) instead of intermediate tensors (DRAM)
    ...
```

**Development workflow:** Start with multiple simple kernels to verify correctness, then fuse into one kernel for performance.

## Multi-Tile Processing and Streaming

**Strive to always use `grid="auto"` with streaming loops:**

- **`grid="auto"`** - this automatically selects the grid size at compile time. Hardcoded grids are only for special cases (e.g., pipe topologies that require a fixed core count). Using grid="auto" will enable full core utilization from the get go.
- **Stream with loops** in both compute and datamovement threads to handle arbitrary input sizes through DFBs.
- **Compute tiles_per_core dynamically** from tensor shape and grid size so kernels work on any input size.

Always strive to use the above patterns to ensure your kernels are flexible for any input size and fully utilize the cores available.

The exception: often for debugging or incremental development, it's helpful to start with a single core kernel; that is fine. You can start with a single core to isolate or debug a pattern, but strive to set it up in a way that it will naturally work with multiple cores later.

### IMPORTANT: Match the User's Target Data Size

**If the user provides a specific model config or tensor shape, strive to support that size.** You can simplify to smaller tensors for initial testing and debugging, but the goal is a kernel that works on their actual data. Use loops and streaming to handle large inputs:

```python
TILE_SIZE = 32
GRANULARITY = 4  # tiles per block dimension

@ttl.kernel(grid="auto")
def streaming_kernel(a, b, c, y):
    row_tiles_per_block = GRANULARITY
    col_tiles_per_block = GRANULARITY

    grid_cols, grid_rows = ttl.grid_size(dims=2)

    rows = a.shape[0] // TILE_SIZE // row_tiles_per_block
    cols = a.shape[1] // TILE_SIZE // col_tiles_per_block

    rows_per_core = -(-rows // grid_rows)  # divceil
    cols_per_core = -(-cols // grid_cols)  # divceil

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(row_tiles_per_block, col_tiles_per_block), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(row_tiles_per_block, col_tiles_per_block), buffer_factor=2)
    c_dfb = ttl.make_dataflow_buffer_like(c, shape=(row_tiles_per_block, col_tiles_per_block), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(y, shape=(row_tiles_per_block, col_tiles_per_block), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_col, core_row = ttl.node(dims=2)
        for local_row in range(rows_per_core):
            row = core_row * rows_per_core + local_row
            if row < rows:
                for local_col in range(cols_per_core):
                    col = core_col * cols_per_core + local_col
                    if col < cols:
                        with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk, c_dfb.wait() as c_blk, y_dfb.reserve() as y_blk:
                            y_blk.store(a_blk * b_blk + c_blk)

    @ttl.datamovement()
    def dm_read():
        core_col, core_row = ttl.node(dims=2)
        for local_row in range(rows_per_core):
            row = core_row * rows_per_core + local_row
            if row < rows:
                sr = row * row_tiles_per_block
                er = (row + 1) * row_tiles_per_block
                for local_col in range(cols_per_core):
                    col = core_col * cols_per_core + local_col
                    if col < cols:
                        sc = col * col_tiles_per_block
                        ec = (col + 1) * col_tiles_per_block
                        with a_dfb.reserve() as blk:
                            tx = ttl.copy(a[sr:er, sc:ec], blk); tx.wait()
                        with b_dfb.reserve() as blk:
                            tx = ttl.copy(b[sr:er, sc:ec], blk); tx.wait()
                        with c_dfb.reserve() as blk:
                            tx = ttl.copy(c[sr:er, sc:ec], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        core_col, core_row = ttl.node(dims=2)
        for local_row in range(rows_per_core):
            row = core_row * rows_per_core + local_row
            if row < rows:
                sr = row * row_tiles_per_block
                er = (row + 1) * row_tiles_per_block
                for local_col in range(cols_per_core):
                    col = core_col * cols_per_core + local_col
                    if col < cols:
                        sc = col * col_tiles_per_block
                        ec = (col + 1) * col_tiles_per_block
                        with y_dfb.wait() as blk:
                            tx = ttl.copy(blk, y[sr:er, sc:ec]); tx.wait()
```

From `examples/tutorial/multicore_grid_auto.py`. Key patterns: `grid="auto"`, dynamic `tiles_per_core` via divceil, bounds check with `if row < rows`.

**Key streaming principles:**
1. **DFB size is limited by L1** (~1.5MB per core) - you can't fit huge tensors
2. **Stream blocks through CBs** - read a block, process it, write it, repeat
3. **Loop counts must match** - compute iterations = dm_read iterations = dm_write iterations
4. **DRAM is large but slow** - keep data in L1 as long as possible, stream to avoid DRAM round-trips

## Pipes (Core-to-Core Communication)

Pipes are fully implemented in both the simulator and compiler. They enable core-to-core communication for patterns like gather, scatter, and ring exchanges. Get your kernel working without pipes first, then add them when needed for inter-core communication.

### Pipe API

```python
# Create pipes and wrap in a PipeNet
pipes = [ttl.Pipe((x, 0), ((x + 1) % N, 0)) for x in range(N)]
net = ttl.PipeNet(pipes)

# Send data through pipe (in dm_read on source core, inside a reserve block)
with dfb.reserve() as blk:
    tx = ttl.copy(src[0, 0], blk); tx.wait()
    def send(pipe):
        xf = ttl.copy(blk, pipe); xf.wait()
    net.if_src(send)

# Receive data from pipe (in dm_read on destination core)
with dfb.reserve() as blk:
    def recv(pipe):
        xf = ttl.copy(pipe, blk); xf.wait()
    net.if_dst(recv)
```

### Pipe Debugging Tips

- **Pipes cause hangs** when send/receive don't match - every `ttl.copy(blk, pipe)` needs a corresponding `ttl.copy(pipe, blk)`
- **Start without pipes** - get independent multi-core working first, then add pipes
- **Add pipes incrementally** - test after adding each pipe
- See the CB Threading Rules in the Debugging section for common deadlock causes

### Hardware Limits

- **32 CBs max** per core
- **~1.5MB L1** per core
- **~100MB total SRAM** across chip - utilize as much as possible for throughput
- **Tile size**: 32x32 elements = 2KB (bfloat16)

**Prefer `grid="auto"` with streaming** (shown above) over hardcoded grid sizes. See Reference Examples for complete working kernels.

## Tensor Setup

Tensors must be:
- **Tilized**: `layout=ttnn.TILE_LAYOUT` (32x32 element tiles)
- **Interleaved**: `ttnn.DRAM_MEMORY_CONFIG` or `ttnn.L1_MEMORY_CONFIG`
- **bfloat16**: Standard data type for Tenstorrent hardware

IMPORTANT: torch tensors will NOT work as kernel inputs.

```python
import torch
import ttnn

device = ttnn.open_device(device_id=0)

# Create torch tensor (dimensions must be multiples of 32)
input_torch = torch.randn(64, 64, dtype=torch.bfloat16)
output_torch = torch.zeros(64, 64, dtype=torch.bfloat16)

# Convert to TTNN tensors
input_tensor = ttnn.from_torch(
    input_torch,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,  # or ttnn.L1_MEMORY_CONFIG
)
output_tensor = ttnn.from_torch(
    output_torch,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

# Run kernel
my_kernel(input_tensor, output_tensor)

# Read result back
result = ttnn.to_torch(output_tensor)

ttnn.close_device(device)
```

## Semantic Mapping: Think at the Hardware Level

**TT-Lang is a LOW-LEVEL DSL.** Do not expect a 1:1 mapping from PyTorch ops. When translating:

1. **Missing ops don't mean failure** - If `conv2d` doesn't exist, don't stop. Think about what conv2d *actually does* at the hardware level.

2. **Decompose to primitives** - Most "complex" operations are actually:
   - Simple compute (matmul, elementwise ops)
   - Complex data movement (gathering, reordering tiles)

3. **Data movement is the magic** - TT-Lang gives you full control over which tiles go where via `ttl.copy()` and tensor slicing. If you can describe WHERE data needs to go, you can implement the operation.

### Example: Conv2d

Conv2d seems like a "high-level op" but it's actually **matmul with clever data arrangement**:

```
What conv2d does:
- For each output position, gather a KxK window of input
- Flatten that window into a vector
- Dot product with filter weights

How to implement in TT-Lang:
- Reader kernel: Loop over output positions, DMA the KxK windows into CBs (im2col)
- Compute kernel: Just do matmul (window @ weights)
- Writer kernel: Write results back

The "conv2d" is in the data movement, not in a magic instruction.
```

### Example: Softmax

No `softmax` op? Decompose it: max → shift → exp → sum → divide

```python
# softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
# Numerically stable version with max subtraction

with x_dfb.wait() as x, scaler_dfb.wait() as s:
    # 1. Find max for numerical stability
    with max_dfb.reserve() as mx:
        mx.store(ttl.math.reduce_max(x, s, dims=[0, 1]))

    # 2. Broadcast max back to full size
    with max_dfb.wait() as mxv, bcast_dfb.reserve() as mxb:
        mxb.store(ttl.math.broadcast(mxv, dims=[0, 1]))

    # 3. Compute exp(x - max) and sum
    with bcast_dfb.wait() as max_bcast:
        shifted = x - max_bcast
        exp_shifted = ttl.math.exp(shifted)

        with sum_dfb.reserve() as sm:
            sm.store(ttl.math.reduce_sum(exp_shifted, s, dims=[0, 1]))

        # 4. Broadcast sum and divide
        with sum_dfb.wait() as sumv, sum_bcast_dfb.reserve() as smb:
            smb.store(ttl.math.broadcast(sumv, dims=[0, 1]))

        with sum_bcast_dfb.wait() as sum_bcast, out_dfb.reserve() as o:
            o.store(ttl.math.exp(x - max_bcast) / sum_bcast)
```

### Key Principle

When you are re-writing a high level operation or kernel:
1. **What does this kernel or op do at a HW level?** Think about what's actually happening in the HW when this op runs
2. **What primitives do we have?** matmul, elementwise, DMA with indexing
3. **Build it from primitives.** A naive O(n²) loop that works is better than giving up. The goal is NOT performance! Just correctness.
4. This is not a high level DSL like pytorch or ttnn, it's low level and you have explicit control over all of the HW, memory management, and synchronization. Do not think about direct mappings for high level ops and kernels, think about the best way to represent the kernel in tt-lang at the level it is designed to operate.

Even ops that DO exist may have different semantics (write in place, different numerical behavior). Always test to verify.

IMPORTANT: the test runner will just execute your script as a python file. Don't overthink it. The ttlang-sim and the hw runner will just run the script as python (not pytest!) so just **add a main block**, open device, print/assert tensor values. The sim should have full compatibility with ttnn function for moving tensors, opening device and so on:

Below will work on both hw and sim:
```
if __name__ == "__main__":
   device = ttnn.open_device(device_id=0)
   # call test functions here
   ttnn.close_device(device)
```

## Translation Guide: GPU → TT-Lang

### Concept Mapping

| GPU Concept | TT-Lang Equivalent |
|------------|-------------------|
| Thread block / workgroup | Grid of Tensix cores (`grid=(rows, cols)`) |
| Shared memory | L1 via dataflow buffers |
| Global memory | DRAM with DMA transfers |
| Warp/wave operations | Tile-level operations (32x32) |
| `__syncthreads()` | DFB `wait()`/`push()` synchronization |
| Kernel launch | Direct function call: `my_kernel(a, b, c)` |

### CUDA/Triton → TT-Lang

**Original CUDA pattern:**
```cuda
__global__ void add_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

**TT-Lang equivalent:**
```python
@ttl.kernel(grid=(1, 1))  # Or multicore for large tensors
def add_kernel(a, b, c):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), buffer_factor=2)
    c_dfb = ttl.make_dataflow_buffer_like(c, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with a_dfb.wait() as av, b_dfb.wait() as bv:
            with c_dfb.reserve() as cv:
                result = av + bv  # Operates on entire 32x32 tile
                cv.store(result)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[0, 0], blk)
            tx.wait()
        with b_dfb.reserve() as blk:
            tx = ttl.copy(b[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with c_dfb.wait() as blk:
            tx = ttl.copy(blk, c[0, 0])
            tx.wait()

# Call: add_kernel(a, b, c)
```

### PyTorch → TT-Lang

**Original PyTorch:**
```python
def gelu(x):
    return x * 0.5 * (1 + torch.tanh(0.7978845608 * (x + 0.044715 * x**3)))
```

**TT-Lang equivalent:**
```python
@ttl.kernel(grid=(1, 1))
def gelu_kernel(x, out):
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with x_dfb.wait() as xv:
            with out_dfb.reserve() as o:
                # Decompose GELU into available ops
                x3 = xv * xv * xv
                inner = xv + x3 * 0.044715  # Need scale tensor for constants
                # ... continue decomposition
                o.store(result)

    # ... dm_read, dm_write ...
```

**Note:** For scalar constants like 0.5, create a full tile tensor:
```python
scale_torch = torch.full((32, 32), 0.5, dtype=torch.bfloat16)
scale = ttnn.from_torch(scale_torch, ...)
```

## Using TTNN to Fill Gaps

If an operation isn't available in TT-Lang, you can use TTNN ops for:
- Input preprocessing (reshaping, padding, layout conversion)
- Operations not yet supported in TT-Lang
- Output post-processing

**Example: Using TTNN for padding**

```python
# TT-Lang requires tile-aligned dimensions (multiples of 32)
# Use TTNN to pad inputs that aren't tile-aligned

input_torch = torch.randn(100, 50)  # Not tile-aligned

# Pad to 128x64 (multiples of 32)
padded = ttnn.pad(input_tensor, padding=((0, 28), (0, 14)), value=0.0)

# Run TT-Lang kernel on padded input
my_kernel(padded, output_tensor)

# Slice result back to original size if needed
result = ttnn.slice(output_tensor, [0, 0], [100, 50])
```

**Rule of thumb:**
1. Try to implement in TT-Lang first
2. Use TTNN for preprocessing (padding, reshaping) and postprocessing (slicing)
3. The bulk of computation should be in TT-Lang for fusion benefits


## Iteration Workflow (REQUIRED)

**You MUST test every kernel you write.** The workflow has two phases:

### Phase 1: Iterate with the Functional Simulator (default)

The functional simulator (`ttlang-sim`) is the primary development tool. It catches DFB mismatches, shape errors, type errors, and functional bugs via dynamic analysis. Use it for all iteration.

```
1. Write kernel to file
2. Run:
   # Via run-test.sh:
   run-test.sh /path/to/kernel.py
   # Or directly:
   python /path/to/kernel.py
3. Read log output (or tail -100 /tmp/ttlang_test_output.log if using remote tools)
4. If errors: fix and go to step 2
5. If success: verify numerical output is correct
```

### Phase 2: Validate on Real Hardware

Once the kernel passes in the simulator, do a final hardware run:
```bash
# Via run-test.sh:
run-test.sh --hw /path/to/kernel.py

# Or directly (on a machine with HW access):
python /path/to/kernel.py
```

NOTE: it is possible that the sim and hw diverge which may require you to either use --hw early or iterate on a program that passes in the sim but not on HW. If your program works with the sim but not on HW you can use the same iteration flow from phase 1 to debug (you may need to isolate patterns and iterate). You can also ask the user for guidance, they may care more about HW or sim working.

**When to use `--hw` early:** If the simulator has a bug or is overly conservative for your use case, you can bypass it with `--hw` at any point. But prefer the simulator for iteration since it gives better error diagnostics.

**IMPORTANT:**
- Exit code 0 does NOT mean success - always read the log
- The log can be thousands of lines - use `tail`, `head`, or `grep` to filter (e.g., `tail -100 /tmp/ttlang_test_output.log` or `grep "pattern" /tmp/ttlang_test_output.log`)
- Look for: `AssertionError`, `Exception`, `error:`, `FAIL`, `mismatch`
- Never guess at fixes - always read the actual error message
- **IMPORTANT:** Set a low timeout for faster iteration - tests should execute in under 1 second. Hangs are common (especially with pipes or DFB mismatches) and a low timeout helps detect them quickly.

**Handling Hangs:**
- If a kernel hangs, the most common cause is **DFB mismatch** - every `wait()` needs a corresponding `push()` from producer, every `reserve()` needs a corresponding `pop()` from consumer
- Verify loop counts match between compute and datamovement threads
- Kill zombie processes: `pkill -9 python` (or via `remote-run.sh pkill -9 python` if using remote tools)

## Compiler Errors: Workaround or Exit Early

**Your goal is NOT to debug the compiler.** If you hit an MLIR error or miscompile:

1. **First: Try a workaround**
   - Restructure the kernel differently
   - Use a different op combination
   - Split into multiple simpler kernels
   - Use TTNN for the problematic operation

2. **If no workaround exists: Exit early**
   - Report the error clearly to the user
   - Include the MLIR snippet that fails (from `/tmp/ttlang_initial.mlir` or `/tmp/ttlang_final.mlir`)
   - Describe what you tried
   - Do NOT spend time investigating compiler internals

**Signs of a compiler bug (not your fault):**
- MLIR verification errors
- Assertion failures in passes
- Segfaults during compilation
- Generated code that doesn't match the input semantics

## Low-Level DSL: Test Everything

**This is NOT PyTorch.** TT-Lang is a low-level DSL where you directly control memory management and synchronization. Operations may have unexpected semantics:

- Ops might write in place
- Ops might take dataflow buffers as arguments
- Ops might have different numerical behavior than PyTorch equivalents
- Memory layouts matter (tilized, interleaved, etc.)

**Do not assume PyTorch semantics.** If you're unsure how an op behaves, TEST IT.

### Debug Strategy: Isolate and Print

You cannot print or assert inside kernels. Instead:

1. **Test ops in isolation** - Write a minimal kernel with just one op
2. **Print tensors before/after** - Use `print(ttnn.to_torch(tensor))` after the kernel runs
3. **Compare against expected** - Compute the expected result in PyTorch and compare
4. **Build up incrementally** - Once one op works, add the next

```python
# Example: Testing an op in isolation
@ttl.kernel(grid=(1, 1))
def test_single_op(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with inp_dfb.wait() as x:
            with out_dfb.reserve() as o:
                result = ttl.math.exp(x)  # Test just this one op
                o.store(result)
    # ... dm_read, dm_write ...

# After running:
print("Input:", ttnn.to_torch(inp_tensor))
print("Output:", ttnn.to_torch(out_tensor))
print("Expected:", torch.exp(inp_torch))
```

**Iterate as much as you need.** There is no limit on test runs. If behavior is unexpected, simplify further until you understand what's happening.

## Debugging Tips

1. **Start in isolation**: Test one op at a time before combining
2. **Print tensors**: Always print input/output to verify behavior
3. **Check shapes**: All dimensions must be multiples of 32
4. **Verify DFB balance**: Every `wait()` needs `pop()`, every `reserve()` needs `push()`
5. **Read the log**: Always check `/tmp/ttlang_test_output.log` after each run
6. **Check MLIR**: Use `/tmp/ttlang_initial.mlir` and `/tmp/ttlang_final.mlir` for compiler issues

### Debug Printing (dprint)

See [TTLangSpecification.md, Section 10.2](TTLangSpecification.md#102-debug-printing) for using `print` inside thread functions to inspect tensors, blocks, and dataflow buffer state.

### CB Threading Rules (Deadlock Debugging)

Each DFB has exactly one producer (`reserve`+push) and one consumer (`wait`+pop). The three threads (dm_read, compute, dm_write) all start simultaneously and run until they block.

- **Rule 1: One producer, one consumer per DFB.** A DFB flows between two threads (dm_read->compute or compute->dm_write) or is thread-local (compute->compute).
- **Rule 2: A DFB cannot have two producers.** If dm_read reserves on a DFB, compute CANNOT also reserve on it. Violation causes interleaved data or deadlock.
- **Rule 3: Thread-local accumulators must be initialized in compute, not DM.** The first iteration uses `reserve()` with an initial value; subsequent iterations use `wait()` + `reserve()` self-cycle.
- **Rule 4: Check every DFB appears in exactly two threads (or one if local).** For each DFB, list which threads call `reserve()` (producer) and `wait()` (consumer). If any DFB has `reserve()` in two different threads, that's a bug.

**If a kernel deadlocks**, check for DFBs that have `reserve()` in both dm_read and compute. That's the most common cause.
