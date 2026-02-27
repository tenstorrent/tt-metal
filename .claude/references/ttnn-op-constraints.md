# TTNN Operation Constraints Reference

Known constraint rules for validating TTNN operation specifications. Used by the `spec-op` skill for eager validation during interactive specification.

---

## 1. Tile & Layout Constraints

| Rule | Description |
|------|-------------|
| **Compute requires tiles** | ALL FPU/SFPU operations require TILE_LAYOUT input. Row-major data must be tilized before compute and untilized after. |
| **Tile dimensions** | Tile size is 32x32 elements. Tensor height and width must be padded to multiples of 32 when using TILE_LAYOUT. |
| **BFLOAT8_B/BFLOAT4_B require TILE_LAYOUT** | Block float formats are incompatible with ROW_MAJOR_LAYOUT. Always use TILE_LAYOUT. |
| **ROW_MAJOR width alignment** | Width must be a multiple of `4 / sizeof(dtype)` for ROW_MAJOR on device. For BFLOAT16 (2 bytes): width % 2 == 0. For FLOAT32 (4 bytes): width % 1 == 0. |
| **Non-standard tile sizes** | Some operations (e.g., matmul) support 16x32 tiles with limited functionality. Default to 32x32. |
| **Power-of-2 padding** | Some operations (e.g., bitonic sort) require power-of-2 padding or minimum 2+ tiles. |

## 2. Data Type Constraints

| Rule | Description |
|------|-------------|
| **Floating-point operations** | Most eltwise, matmul, softmax, norm ops support: BFLOAT16, FLOAT32. Some also support BFLOAT8_B. |
| **Block formats (BFLOAT8_B, BFLOAT4_B)** | Must use TILE_LAYOUT. May require typecast to BFLOAT16 for certain operations. |
| **FP32 dest accumulation** | Required for precision with FLOAT32 inputs in some operations (reductions, matmul). Set `fp32_dest_acc_en=true`. |
| **Mixed precision** | Not all dtype combinations are valid for binary ops. Typically both inputs must match or one must be broadcastable scalar. |
| **Numpy conversion** | BFLOAT16 not supported for direct numpy tensor conversion. |

## 3. Memory Layout Constraints

| Rule | Description |
|------|-------------|
| **INTERLEAVED** | Default layout. Data distributed round-robin across DRAM banks. Works in both DRAM and L1. |
| **HEIGHT_SHARDED** | Each core gets horizontal strips. L1 only. Shard height varies, shard width = full tensor width. |
| **WIDTH_SHARDED** | Each core gets vertical strips. L1 only. Shard width varies, shard height = full tensor height. |
| **BLOCK_SHARDED** | 2D grid distribution. L1 only. Both shard dims vary. |
| **Sharding is L1 only** | All sharding types (HEIGHT, WIDTH, BLOCK) require L1 memory. Sharding in DRAM is not supported. |
| **Shard tile alignment** | Shard dimensions must be multiples of TILE_WIDTH (32) and TILE_HEIGHT (32) for tiled tensors. |
| **Shard grid bounds** | Shard grid must fit within the device compute grid. |
| **Input/output matching** | Input and output memory layouts typically must match (both interleaved or both same sharding type). |
| **Separate program factories** | Sharded and interleaved paths usually need separate program factory implementations. |

## 4. L1 Memory Budget

| Rule | Description |
|------|-------------|
| **Per-core limit** | 1.5MB SRAM (L1) per Tensix core. Must fit: kernel code + circular buffers + stack. |
| **CB sizing** | Each CB page = 1 tile (for TILE_LAYOUT) or 1 row (for ROW_MAJOR). More pages = more buffering = more L1. |
| **Double buffering cost** | Double buffering (2 pages per CB) doubles L1 usage but enables producer/consumer overlap. |
| **Strategy threshold** | ~512KB is the common threshold for "small" vs "large" tensor handling. Operations select different strategies based on this. |
| **Circular buffer count** | Each CB consumes L1. Operations with many intermediates (reductions, norms) can exhaust L1. |
| **Large tensor concern** | If input tensor per-core exceeds ~256KB, consider single buffering or DRAM-based strategies. |

## 5. Shape & Broadcasting Constraints

| Rule | Description |
|------|-------------|
| **Broadcasting** | Only dimensions of size 1 can be broadcast. The broadcast dimension must equal 1 in the smaller tensor. |
| **Rank auto-expansion** | Tensors with rank < 4 are auto-expanded to 4D internally (prepending dims of size 1). |
| **Rank > 4** | Gets reshaped to 4D, which may cause shape reconstruction issues. Prefer rank <= 4. |
| **Matmul inner dim** | Last dim of A must equal second-to-last dim of B. Both inputs must have equal rank for non-broadcast matmul. |
| **Reduction keepdim** | `keepdim=False` is unsupported for the last 2 dimensions in some reduction operations. |
| **Batch dim matching** | For batched operations (matmul, binary), batch dimensions must match or be broadcastable. |
| **Stride constraints** | Stride != 1 not supported for some operations (e.g., padded_slice). |

## 6. Incompatible Combinations

Known-invalid configuration combinations:

| Combination | Why Invalid |
|------------|-------------|
| BFLOAT8_B + ROW_MAJOR_LAYOUT | Block formats require TILE_LAYOUT |
| BFLOAT4_B + ROW_MAJOR_LAYOUT | Block formats require TILE_LAYOUT |
| Sharded + DRAM | All sharding types require L1 memory |
| ROW_MAJOR + FPU/SFPU compute (without tilize) | Compute units only operate on tiles |
| WIDTH_SHARDED + KV cache ops | KV cache requires INTERLEAVED |
| Batched matmul + transpose_a/b | Batched matmul forbids transpose flags |
| Batched matmul + bias | Bias incompatible with batched input |

## 7. Circular Buffer Conventions

| Rule | Description |
|------|-------------|
| **Index convention** | 0-7: inputs, 8-15: special (scalers, constants), 16-23: outputs, 24-31: intermediates |
| **Synchronization invariant** | Producer push count MUST EQUAL consumer wait count per CB. Violation causes hangs. |
| **Globally allocated CBs** | Sharded tensors use `.set_globally_allocated_address(*buffer)` — no separate allocation. |
| **Page size matching** | CB page size must match data granularity: tile_size for TILE_LAYOUT, row_size for ROW_MAJOR. |

## 8. Common Auto-Mode Defaults

When no specific requirements are given, these are the safest starting choices:

| Property | Safe Default | Rationale |
|----------|-------------|-----------|
| Dtype | BFLOAT16 | Widest operation support, good precision/performance balance |
| Layout | TILE_LAYOUT | Required for compute, avoids tilize/untilize overhead |
| Memory layout | INTERLEAVED | Simplest, works everywhere, no sharding complexity |
| Buffer type | DRAM | Safest for large tensors, no L1 pressure |
| Buffering | SINGLE | Minimal L1 usage, correct before fast |
| Work unit | tile | Natural granularity for TILE_LAYOUT operations |
| Compute config | Optional parameter, defaults inferred | FP32 accum if FLOAT32 input, else off. Math fidelity HiFi4. |
| Tolerances | rtol=0.01, atol=0.01 | Standard for BFLOAT16 operations |
