# SIGNBIT Implementation Analysis

## Overview
The SIGNBIT operation extracts the sign bit of each element in a tensor. For floating-point inputs, it returns 1.0 if the element is negative and 0.0 otherwise. For INT32 inputs, it performs an arithmetic right-shift by 31 to extract the sign bit directly. This is a unary SFPU operation that uses the standard `eltwise_sfpu.cpp` compute kernel with macro-expanded SFPU function calls.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

## Path Selection: FPU vs SFPU

SIGNBIT is a pure SFPU operation; there is no FPU path. The compute kernel path is resolved via `utils::get_compute_kernel_path(ops_chain[0].type(), input.dtype())` in the program factory (line 155). SIGNBIT falls into the `default` case of that function (line 984), which returns `"eltwise_sfpu.cpp"`. This is the generic SFPU dispatch kernel shared by the majority of unary operations that do not require specialized kernel files. The SFPU function to execute is injected at compile time through the `SFPU_OP_CHAIN_0` macro define, which expands to `signbit_tile_init(); signbit_tile(0);` (or `signbit_tile_int32(0)` for INT32 inputs).

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `num_pages` = total number of tiles in the input tensor |
| **Loop structure** | Outer loop over `per_core_block_cnt` blocks, inner loop over `per_core_block_dim` tiles (always 1) |

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|-------------|
| **Logical shape** | Arbitrary (any rank) |
| **Dimension convention** | NHWC (standard TTNN) |
| **Tensor layout** | TILE_LAYOUT (or ROW_MAJOR) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, or INT32 |

### Output Tensor

| Property | Output Tensor |
|----------|--------------|
| **Logical shape** | Same as input |
| **Dimension convention** | NHWC |
| **Tensor layout** | Same as input |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as input (or specified output dtype) |

### Layout Transformations
No tilize/untilize or reshard operations are performed within the program factory. The input and output share the same layout.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM/L1 (src_buffer) | CB c_0 | `cb_reserve_back(c_0, 1)`, `noc_async_read_page`, `noc_async_read_barrier`, `cb_push_back(c_0, 1)` |
| 2 | Compute | CB c_0 | CB c_2 | `cb_reserve_back(c_2, 1)`, `cb_wait_front(c_0, 1)`, `copy_tile`, SFPU op, `pack_tile`, `cb_pop_front(c_0, 1)`, `cb_push_back(c_2, 1)` |
| 3 | Writer | CB c_2 | DRAM/L1 (dst_buffer) | `cb_wait_front(c_2, 1)`, `noc_async_write_page`, `noc_async_writes_flushed`, `cb_pop_front(c_2, 1)` |

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | src0 | Input staging | 2 tiles | 1 tile | Double | Reader | Compute | Program |
| c_2 | output | Output staging | 2 tiles | 1 tile | Double | Compute | Writer | Program |

Note: CB c_1 (tmp0) is **not** allocated for SIGNBIT. It is only created for HARDSHRINK, CBRT, or LOGIT operations.

## Pipeline Pattern Summary
Both c_0 and c_2 are double-buffered (capacity = 2 tiles, block size = 1 tile). This allows the reader to write the next tile into c_0 while the compute kernel processes the current tile, and similarly the compute kernel can write the next result to c_2 while the writer drains the previous result. This enables a three-stage pipeline overlap between reader, compute, and writer.

## Index Calculations
The program factory uses `TensorAccessor` for both read and write paths. The reader and writer iterate sequentially over page indices from `start_id` to `start_id + num_pages`. The `TensorAccessor` abstracts the mapping from logical page index to physical DRAM/L1 bank address and offset, handling interleaved bank distribution automatically. No complex index remapping is needed for this operation.

## Memory Access Patterns

### Read Pattern
Sequential page-by-page reads. Each core reads a contiguous range of tile pages starting from its assigned `start_id`. Pages are read one at a time via `noc_async_read_page` with a barrier after each read before pushing to the CB.

### Write Pattern
Sequential page-by-page writes. Each core writes tiles one at a time via `noc_async_write_page`. A flush is issued after each write, with a final `noc_async_write_barrier` after the loop completes.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (flattened to 1D assignment) |
| **Grid dimensions** | `compute_with_storage_grid_size` (device-dependent, e.g., 8x8) |
| **Total cores** | Determined by `split_work_to_cores` |
| **Work per core** | `num_pages_per_core_group_1` tiles (group 1) or `num_pages_per_core_group_2` tiles (group 2) |
| **Load balancing** | Two-group split: group 1 gets `ceil(num_pages / num_cores)` tiles, group 2 gets `floor(num_pages / num_cores)` tiles |

Cores are enumerated in column-major order: `core = {i / num_cores_y, i % num_cores_y}`. Two separate compute kernels are created with different `per_core_block_cnt` compile-time arguments to handle the unequal work distribution between core groups.

## Arguments

### Compile-Time Arguments

**Reader kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs(src) | uint32_t[] | Packed tensor accessor parameters for source buffer |

**Writer kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | Output CB index (c_2 = 2) |
| 1+ | TensorAccessorArgs(dst) | uint32_t[] | Packed tensor accessor parameters for destination buffer |

**Compute kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of tile blocks to process on this core |
| 1 | per_core_block_dim | uint32_t | Number of tiles per block (always 1) |

### Runtime Arguments

**Reader kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address |
| 1 | num_pages | uint32_t | Number of pages to read |
| 2 | start_id | uint32_t | Starting page index for this core |

**Writer kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address |
| 1 | num_pages | uint32_t | Number of pages to write |
| 2 | start_id | uint32_t | Starting page index for this core |

**Compute kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar1 | uint32_t | Unused for SIGNBIT (always 0) |
| 1 | packed_scalar2 | uint32_t | Unused for SIGNBIT (always 0) |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM/L1 | CB c_0 | Read tiles sequentially via TensorAccessor |
| compute | RISCV_2 | N/A | CB c_0 | CB c_2 | copy_tile to DST, signbit SFPU op, pack_tile |
| writer | RISCV_1 | NOC1 | CB c_2 | DRAM/L1 | Write tiles sequentially via TensorAccessor |

### Reader Kernel

| Property | Value |
|----------|-------|
| File | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` |
| Assigned cores | all_cores (both core groups) |

**Key Logic:**
- Retrieves `src_addr`, `num_pages`, and `start_id` from runtime arguments
- Constructs a `TensorAccessor` from compile-time `TensorAccessorArgs` and runtime source address
- Loops from `start_id` to `start_id + num_pages`, reading one page per iteration
- Each iteration: `cb_reserve_back(c_0, 1)` -> `noc_async_read_page` -> `noc_async_read_barrier` -> `cb_push_back(c_0, 1)`
- Supports optional `BACKWARDS` define for reverse iteration (not used by SIGNBIT)
- **Synchronization**: Produces into CB c_0 using reserve_back/push_back. The read barrier ensures data is in L1 before signaling the compute kernel.

### Compute Kernel

| Property | Value |
|----------|-------|
| File | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` |
| Assigned cores | core_group_1 and core_group_2 (separate kernel instances with different compile-time args) |

**Key Logic:**
- Calls `init_sfpu(c_0, c_2)` to initialize the SFPU pipeline with input/output CB indices
- Outer loop iterates `per_core_block_cnt` times (number of tiles assigned to this core)
- `cb_reserve_back(c_2, per_core_block_dim)` reserves output space (1 tile)
- Inner loop (1 iteration since `per_core_block_dim = 1`):
  - `tile_regs_acquire()` -- acquires DST register file
  - `cb_wait_front(c_0, 1)` -- waits for input tile from reader
  - `copy_tile(c_0, 0, 0)` -- unpacks tile from CB c_0 into DST register 0
  - `SFPU_OP_CHAIN_0` macro expands to: `signbit_tile_init(); signbit_tile(0);` (or `signbit_tile_int32(0)` for INT32)
  - `tile_regs_commit()` / `tile_regs_wait()` -- synchronization between math and pack pipelines
  - `pack_tile(0, c_2)` -- packs result from DST register 0 into CB c_2
  - `cb_pop_front(c_0, 1)` -- frees input tile in CB c_0
  - `tile_regs_release()` -- releases DST register file
- `cb_push_back(c_2, per_core_block_dim)` publishes output tile to writer
- **Synchronization**: Consumes from CB c_0 (wait_front/pop_front), produces into CB c_2 (reserve_back/push_back). Uses tile_regs_acquire/commit/wait/release for DST register synchronization between unpack-math and math-pack pipelines.

**SFPU Implementation (float path - `calculate_signbit`):**
- Iterates 8 times (ITERATIONS=8) over sub-tile rows in the DST register
- Reads `dst_reg[0]` as a `vFloat` (SIMD vector)
- Uses conditional SFPI: `v_if(val < 0.0f) { val = 1.0f; } v_else { val = 0.0f; } v_endif;`
- Writes result back to `dst_reg[0]` and advances `dst_reg++`
- Note: The code has a TODO comment suggesting a bitwise implementation would be more efficient

**SFPU Implementation (INT32 path - `calculate_signbit_int32`):**
- Iterates 8 times over sub-tile rows
- `TTI_SFPLOAD(LREG0, INT32, ADDR_MOD_3, 0)` -- loads INT32 value from DST
- `TTI_SFPSHFT((-31) & 0xfff, LREG0, LREG0, 1)` -- arithmetic right-shift by 31 bits, extracting the sign bit (0 for positive, 1 for negative due to sign extension)
- `TTI_SFPSTORE(LREG0, INT32, ADDR_MOD_3, 0)` -- stores result back to DST
- Advances `dst_reg++`
- Note: Blackhole variant uses `ADDR_MOD_7` for SFPLOAD instead of `ADDR_MOD_3`

### Writer Kernel

| Property | Value |
|----------|-------|
| File | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` |
| Assigned cores | all_cores (both core groups) |

**Key Logic:**
- Retrieves `dst_addr`, `num_pages`, and `start_id` from runtime arguments
- Output CB index (`c_2`) is a compile-time argument
- Constructs a `TensorAccessor` from compile-time args and runtime destination address
- Loops from `start_id` to `start_id + num_pages`, writing one page per iteration
- Each iteration: `cb_wait_front(c_2, 1)` -> `noc_async_write_page` -> `noc_async_writes_flushed` -> `cb_pop_front(c_2, 1)`
- Final `noc_async_write_barrier` after the loop ensures all writes complete
- Supports `OUT_SHARDED` define for sharded output (just waits for all pages, no writes needed) -- not used in interleaved path
- **Synchronization**: Consumes from CB c_2 using wait_front/pop_front. The flush after each write ensures NoC write ordering.

## Implementation Notes

- **Program factory variants**: Three factories exist: `UnaryProgramFactory` (interleaved, no sub-core grids), `UnarySubCoreGridProgramFactory` (interleaved with sub-core grids), and `UnaryShardedProgramFactory` (sharded input). Selection logic in `select_program_factory`: sharded input -> sharded factory; sub_core_grids present -> sub-core grid factory; otherwise -> default factory. SIGNBIT uses all three depending on tensor configuration.
- **Type-based operation variants**: SIGNBIT supports BFLOAT16/FLOAT32 (float path using conditional SFPI) and INT32 (integer path using arithmetic right-shift). The variant is selected at compile time via the `get_op_init_and_func_default` function which checks `input_dtype`.
- **UnpackToDestFP32 mode**: Enabled when `args.preserve_fp32_precision` is true. Sets `UnpackToDestMode::UnpackToDestFp32` on CB c_0 and c_1 (c_1 unused for SIGNBIT).
- **Broadcast type selection**: N/A -- SIGNBIT is a pure unary operation with no broadcasting.
- **Sharding support and constraints**: Sharded inputs route to `UnaryShardedProgramFactory` (not analyzed in detail here). The interleaved factory analyzed above does not handle sharded tensors.
- **FP32 dest accumulation**: Controlled by `args.fp32_dest_acc_en` in the `ComputeConfig`. When enabled, the DST accumulator operates in FP32 precision.

## External Knowledge Sources

### DeepWiki Queries
1. **Query**: "How does the unary SFPU program factory work in ttnn? What is the structure of unary_program_factory.cpp, how does it select between FPU and SFPU paths, and how are SFPU compute kernels dispatched?"
   **Reason**: Initial architectural understanding of the unary program factory structure and SFPU dispatch mechanism.
   **Key Findings**: Confirmed that `get_compute_kernel_path` selects the compute kernel, that `eltwise_sfpu.cpp` is the default SFPU kernel, that defines are injected via `get_block_defines`, and that the program factory creates separate kernel instances for core groups with different tile counts.

### Documentation References
1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` (lines 583-589)
   **Reason**: To identify the SFPU init/func strings generated for SIGNBIT.
   **Key Information**: SIGNBIT maps to `signbit_tile_init()` / `signbit_tile(idst)` for float, `signbit_tile_int32(idst)` for INT32.

2. **Source**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_signbit.h`
   **Reason**: To understand the actual SFPU implementation of the signbit operation.
   **Key Information**: Float path uses conditional SFPI (`v_if/v_else`); INT32 path uses arithmetic right-shift by 31 bits via `TTI_SFPSHFT`.

3. **Source**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_signbit.h`
   **Reason**: To check for hardware-variant differences.
   **Key Information**: Identical logic to Wormhole, except `ADDR_MOD_7` is used instead of `ADDR_MOD_3` for SFPLOAD in the INT32 path.

4. **Source**: `tt_metal/hw/inc/api/compute/compute_kernel_api.h` (lines 188, 204, 220)
   **Reason**: To verify the compute API wrappers for signbit.
   **Key Information**: `signbit_tile_init()` calls `llk_math_eltwise_unary_sfpu_signbit_init<APPROX>()`, `signbit_tile(idst)` calls `llk_math_eltwise_unary_sfpu_signbit<APPROX>(idst)`.
