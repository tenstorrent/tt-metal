# SFPU Operation Kernel Implementations

Consolidated "Kernel Implementations" sections from all analysis files in `.claude/sfpu-operation-analysis-2/`.

---

## ABS (Unary)

### Reader Kernel

| Property | Value |
|---|---|
| File | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` |
| Type | ReaderDataMovementConfig |
| Assigned Cores | all_cores |

**Key Logic**: Simple sequential page reader. For each page in `[start_id, end_id)`: reserves space in CB c_0, issues a NoC async read, waits for completion, then pushes the page to the consumer (compute kernel). Supports a `BACKWARDS` mode via preprocessor define (not used for ABS).

### Compute Kernel

| Property | Value |
|---|---|
| File | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` |
| Type | ComputeConfig |
| Assigned Cores | core_group_1 and core_group_2 (separate kernel handles) |

**Key Logic**: Generic SFPU eltwise kernel. For each block (tile):
1. `tile_regs_acquire()` - acquire DST register file
2. `cb_wait_front(c_0, 1)` - wait for input tile
3. `copy_tile(c_0, 0, 0)` - unpack tile from CB to DST register 0
4. Execute `SFPU_OP_CHAIN_0` macro which expands to `abs_tile_init(); abs_tile(0);`
5. `tile_regs_commit()` / `tile_regs_wait()` - synchronize math pipeline
6. `pack_tile(0, c_2)` - pack result from DST to output CB
7. `cb_pop_front(c_0, 1)` - release input tile

**Compute Config**:
- `math_fidelity`: HiFi4
- `math_approx_mode`: false (ABS returns false from `get_op_approx_mode`)
- `fp32_dest_acc_en`: configurable via operation attributes

**Defines injected**:
- `SFPU_OP_CHAIN_0_INIT_0` = `abs_tile_init();`
- `SFPU_OP_CHAIN_0_FUNC_0` = `abs_tile(0);`
- `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` = `1` (ABS uses the default compute_kernel_api include)
- `SFPU_OP_CHAIN_0` = `SFPU_OP_CHAIN_0_INIT_0 SFPU_OP_CHAIN_0_FUNC_0`
- One of: `INP_FLOAT32`, `INP_INT32`, `INP_UINT32`, or `INP_FLOAT` depending on input dtype

### Writer Kernel

| Property | Value |
|---|---|
| File | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` |
| Type | WriterDataMovementConfig |
| Assigned Cores | all_cores |

**Key Logic**: Sequential page writer. For each page in `[start_id, end_id)`: waits for compute to push a page into CB c_2, reads the L1 address, issues a NoC async write, flushes, then pops the page. Final `noc_async_write_barrier()` ensures all writes complete.

---

## RELU (Unary)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM/L1 | CB c_0 | Read pages sequentially via TensorAccessor |
| compute | RISCV_2 (unpack+math+pack) | N/A | CB c_0 | CB c_2 | Unpack tile, apply RELU SFPU op, pack tile |
| writer | RISCV_1 | NOC1 | CB c_2 | DRAM/L1 | Write pages sequentially via TensorAccessor |

### Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
- **Key Logic**: Simple sequential page reader. Uses `TensorAccessor` for address resolution. Supports `BACKWARDS` define for reverse iteration (not used by RELU). Reads one page per iteration with `noc_async_read_page` and a per-page barrier.

### Compute Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **Key Logic**: Generic SFPU compute kernel shared by most unary operations. The operation-specific behavior is injected via preprocessor defines:
  - `SFPU_OP_RELU_FAMILY_INCLUDE=1` -- enables the RELU family include header
  - `SFPU_OP_CHAIN_0` -- expands to `SFPU_OP_CHAIN_0_INIT_0 SFPU_OP_CHAIN_0_FUNC_0`
  - `SFPU_OP_CHAIN_0_INIT_0` -- expands to `relu_tile_init();`
  - `SFPU_OP_CHAIN_0_FUNC_0` -- expands to `relu_tile(0);` (or `relu_tile_int32(0);` for INT32)
- **Execution pattern**: For each block, reserves output CB space, then for each tile: acquires registers, waits for input tile in c_0, copies to DST registers, executes the SFPU op chain, commits registers, waits for pack, packs to c_2, pops c_0, releases registers. Pushes the block to c_2 after all tiles in the block are packed.
- **Math configuration**: `math_fidelity = HiFi4`, `math_approx_mode = false` (RELU returns false from `get_op_approx_mode`).

### Writer Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Sequential page writer. Uses `TensorAccessor` for address resolution. Supports `OUT_SHARDED` define for sharded output (not used in interleaved path). Writes one page per iteration with `noc_async_write_page`, flushes after each write, and issues a final barrier.

---

## ADD (Binary SFPU - Legacy)

### Reader Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp`
- **Type**: ReaderDataMovementConfig (runs on BRISC, uses NoC0)
- **Key Logic**:
  - For sharded inputs: Simply calls `cb_reserve_back` / `cb_push_back` to signal that data is already present in the CB (backed by L1 shard buffer).
  - For interleaved inputs: Uses `TensorAccessor` + `noc_async_read_tile` to fetch tiles one at a time from DRAM into the CB. Issues a `noc_async_read_barrier` after each tile pair.
  - For block/width sharded mode: Uses a 2D loop (block_height x block_width) with strided row access (`row_start_tile_id += num_cores_y * block_width`) to handle the sharding layout.
- **Synchronization**: Produces into c_0 and c_1.

### Compute Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp`
- **Type**: ComputeConfig (runs on TRISC cores: unpack, math, pack)
- **Key Logic**:
  - Outer loop iterates `per_core_block_cnt` blocks.
  - For each block: waits for data in cb_inp0 and cb_inp1, reserves output space in cb_out0.
  - Uses `tile_regs_acquire` / `tile_regs_wait` for DST register lifecycle.
  - Copies input A tiles to even DST slots (i*2), input B tiles to odd DST slots (i*2+1).
  - Executes the SFPU operation (for ADD: `add_binary_tile(i*2, i*2+1, i*2)`).
  - Packs result from DST[i*2] to output CB.
  - The `copy_tile_to_dst_init_short_with_dt` calls handle data format switching between the two inputs.
- **Synchronization**: Consumes from c_0/c_1 (or c_3/c_4 if pre-scaling), produces into c_2.

### Writer Kernel

- **File** (interleaved): `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **File** (block/width sharded to interleaved): `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp`
- **Type**: WriterDataMovementConfig (runs on NCRISC, uses NoC1)
- **Key Logic**:
  - For sharded output: Simply calls `cb_wait_front` to acknowledge compute completion. Data is already in the correct L1 location.
  - For interleaved output: Uses `TensorAccessor` + `noc_async_write_page` to write tiles one at a time, with `noc_async_writes_flushed` between tiles and a final `noc_async_write_barrier`.
- **Synchronization**: Consumes from c_2.

---

## ADD (Binary NG)

### FPU Path (BFLOAT16 default)

#### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | BRISC (RISCV_0) | NOC0 | DRAM/L1 (A, B) | CB c_0, CB c_1 | Read A and B tiles via noc_async_read_page |

- **File (tensor-tensor)**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_no_bcast.cpp`
- **File (scalar, reads only A)**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/reader_interleaved_no_bcast.cpp`
- **Key Logic**: Uses TensorAccessor for address resolution. Traverses 6 nested dimension loops matching output tile order. For broadcast variants (row, col, scalar), different reader kernels apply tile duplication (fill_tile_with_first_row, fill_tile_with_first_column, fill_tile_with_first_element).

#### Compute Kernel (FPU, no-broadcast)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | TRISC (RISCV_2) | N/A | CB c_0, CB c_1 | CB c_2 | `add_tiles` via FPU |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_no_bcast.cpp`
- **Key Logic**: For ADD, the define `BINARY_OP` resolves to `add_tiles` and `BINARY_OP_TYPE` to `EltwiseBinaryType::ELWADD`. The kernel calls `binary_op_init_common`, then loops tile-by-tile: `cb_wait_front` on both inputs, `tile_regs_acquire`, `BINARY_OP(cb_lhs, cb_rhs, 0, 0, 0)`, optional `PROCESS_POST_ACTIVATIONS`, `pack_tile`, `cb_push_back`, `cb_pop_front`.

#### Compute Kernel (FPU, scalar)

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_scalar.cpp`
- **Key Logic**: Same as no-broadcast but waits for RHS only once (before the loop). The scalar tile stays resident in CB c_1 and is reused for all LHS tiles. RHS is popped only after the loop ends.

#### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | NCRISC (RISCV_1) | NOC1 | CB c_2 | DRAM/L1 (C) | Write output tiles via noc_async_write_page |

- **File (tensor-tensor)**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp`
- **File (scalar, also fills B)**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/writer_interleaved_scalar.cpp`
- **Key Logic**: In scalar mode, the writer first fills CB c_1 with the packed scalar value using `fill_with_val` / `fill_with_val_bfloat16`, then writes output tiles. In tensor-tensor mode, the writer only handles output.

### SFPU Path (FLOAT32 / INT32 / UINT32 / UINT16)

#### Compute Kernel (SFPU, no-broadcast)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | TRISC (RISCV_2) | N/A | CB c_0, CB c_1 | CB c_2 | `add_binary_tile` or `add_int_tile` via SFPU |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp`
- **Key Logic**: SFPU path copies both input tiles to dest registers using `copy_tile`, then calls `BINARY_SFPU_OP(dst_a, dst_b, dst_out)`. For FLOAT32 ADD, this resolves to `add_binary_tile` with init `add_binary_tile_init()`. For INT32, it resolves to `add_int_tile<DataFormat::Int32>`. Uses `UnpackToDestMode::UnpackToDestFp32` for all non-POWER SFPU ops.

The reader and writer kernels are identical between FPU and SFPU paths.

---

## MUL (Binary SFPU - Legacy)

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM/L1 (src0, src1) | CB c_0, CB c_1 | Read input tiles via TensorAccessor |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp`
- **Key Logic**: Handles both interleaved and sharded inputs via conditional compilation (`IN0_SHARDED`, `IN1_SHARDED`). For sharded inputs, simply reserves and pushes the full shard. For interleaved inputs, reads one tile at a time with `noc_async_read_tile`. Supports a block/width-sharded 2D traversal pattern when `block_or_width_sharded` is set.

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | RISCV_2 (MATH) | N/A | CB c_0, CB c_1 | CB c_2 | SFPU mul_binary_tile |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp`
- **Key Logic**: Per block iteration: waits for `per_core_block_size` tiles on both input CBs, acquires tile registers, copies input A tiles to even DST slots (`i*2`) and input B tiles to odd DST slots (`i*2+1`), then for each tile pair calls `BINOP_INIT` (= `mul_binary_tile_init()`) and `BINARY_SFPU_OP` (= `mul_binary_tile(0, 1, 0)`), packs result from DST[i*2] to output CB. The kernel supports optional pre-processing stages via `SFPU_OP_INIT_PRE_IN0_0` / `SFPU_OP_INIT_PRE_IN1_0` (not used for plain MUL). Also supports fused post-activations via `SFPU_OP_CHAIN_0` and `PACK_RELU`.

**FP32 accumulation**: Enabled when output dtype is FLOAT32, INT32, or UINT32. For MUL (non-POWER), `UnpackToDestMode::UnpackToDestFp32` is set on all input CBs.

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | RISCV_1 | NOC1 | CB c_2 | DRAM/L1 (dst_buffer) | Write output tiles via TensorAccessor |

- **File (standard)**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **File (block-to-interleaved)**: `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp`
- **Key Logic**: Standard path writes one tile at a time with `noc_async_write_page` and flushes after each tile. For sharded output, simply calls `cb_wait_front` to wait for all tiles (output buffer is globally allocated). The block-to-interleaved variant handles the case where compute produces sharded blocks but output must be written to interleaved memory.

---

## MUL (Binary NG)

### Reader Kernel -- Tensor-Tensor, No Broadcast

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_no_bcast.cpp`
- **Key Logic**: Reads A tiles into CB0 and B tiles into CB1 in lockstep. Uses a 6-level nested loop (nD, D, N, C, Ht, Wt) iterating over output tile coordinates. For each output tile, computes the corresponding input A and B tile offsets using stride-based broadcasting. When sharded, simply marks shard data as available via `cb_reserve_back`/`cb_push_back`.

### Reader Kernel -- Scalar Mode (A only)

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/reader_interleaved_no_bcast.cpp`
- **Key Logic**: Same structure as tensor-tensor reader but only reads A tiles into CB0. B is handled by the scalar writer.

### Compute Kernel -- FPU No-Broadcast

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_no_bcast.cpp`
- **Key Logic**: For MUL, the define `BINARY_OP` expands to `mul_tiles`. The loop processes one tile per iteration: waits for LHS and RHS, acquires dest registers, calls `mul_tiles(cb_lhs, cb_rhs, 0, 0, 0)`, applies optional post-activations, packs result to CB2. The FPU path does NOT copy tiles to DST first -- the `mul_tiles` LLK reads directly from source CBs.

### Compute Kernel -- SFPU No-Broadcast

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp`
- **Key Logic**: For MUL, `BINARY_SFPU_INIT` = `mul_binary_tile_init()` and `BINARY_SFPU_OP` = `mul_binary_tile`. The SFPU path must explicitly copy both operand tiles into DST registers (`copy_tile(cb_lhs, i, i*2)` for LHS into even slots, `copy_tile(cb_rhs, i, i*2+1)` for RHS into odd slots), then calls `mul_binary_tile(i*2, i*2+1, i*2)` to multiply them in-place.

### Compute Kernel -- FPU Scalar

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_scalar.cpp`
- **Key Logic**: Same as FPU no-broadcast but the RHS tile (scalar) is waited on once before the loop and popped after the loop ends. Each iteration only waits for a new LHS tile.

### Compute Kernel -- SFPU Scalar

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_scalar.cpp`
- **Key Logic**: Same pattern as SFPU no-broadcast but RHS (scalar tile) is waited once and popped after the loop.

### Writer Kernel -- Tensor-Tensor

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp`
- **Key Logic**: Waits for output tiles in CB2 and writes them to DRAM/L1 using `noc_async_write_page`. Same 6-level nested loop structure as the reader. For sharded output, the `#if !DST_SHARDED` guard skips all write logic.

### Writer Kernel -- Scalar (also writes output)

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/writer_interleaved_scalar.cpp`
- **Key Logic**: First fills one tile in CB1 with the packed scalar value using `fill_with_val_bfloat16` (or `fill_with_val<1024, float>` for float32). Then writes output tiles from CB2 to DRAM/L1 using the same nested loop pattern.
