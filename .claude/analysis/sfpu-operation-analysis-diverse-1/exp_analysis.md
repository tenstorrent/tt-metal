# EXP Implementation Analysis

## Overview
The EXP operation computes the element-wise exponential (e^x) of each element in the input tensor. It is dispatched through the shared unary SFPU program factory infrastructure, which provides a generic three-kernel pipeline (reader, compute, writer) for all unary SFPU operations.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

## Path Selection: FPU vs SFPU

EXP is purely an SFPU operation -- there is no FPU path for it. The unary program factory does not distinguish between FPU and SFPU at the factory level; instead, the compute kernel is always `eltwise_sfpu.cpp` (selected via `get_compute_kernel_path()`, which returns the default `"eltwise_sfpu.cpp"` for EXP). The SFPU-specific behavior is injected through compile-time defines:

1. `SFPU_OP_EXP_INCLUDE=1` -- causes `sfpu_split_includes.h` to include `api/compute/eltwise_unary/exp.h`
2. `SFPU_OP_CHAIN_0` -- expands to `exp_tile_init<1u>(); exp_tile<1u>(0);` (with approx=true by default)

The factory selection logic in `UnaryDeviceOperation::select_program_factory()` chooses between three factories based on tensor properties, not on the operation type:
- **UnaryShardedProgramFactory** -- if input is sharded
- **UnarySubCoreGridProgramFactory** -- if `sub_core_grids` is specified
- **UnaryProgramFactory** -- default (interleaved, full grid)

This analysis covers the default `UnaryProgramFactory` path with interleaved tensors.

## Work Unit Definition
One work unit is **one tile** (32x32 elements). The program factory divides the total number of tiles across cores, and each core processes its assigned tiles one at a time through a tile-serial pipeline.

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|---|---|
| Dimension Convention | N-dimensional (flattened to page count) |
| Tensor Layout | TILE (32x32) or ROW_MAJOR |
| Memory Layout | Interleaved |
| Buffer Type | DRAM (typical) |
| Data Type | BF16 (typical), FP32, INT32, UINT32 supported |

### Output Tensor

| Property | Value |
|---|---|
| Dimension Convention | Same as input |
| Tensor Layout | Same as input |
| Memory Layout | Interleaved |
| Buffer Type | DRAM (typical) |
| Data Type | Same as input (may differ for BITCAST, not applicable to EXP) |

### Layout Transformations
None. EXP preserves the input tensor layout and format. No tilize/untilize or reshard operations are performed.

## Data Flow Pattern

1. **Reader kernel** reads one tile at a time from DRAM into CB c_0 (input circular buffer) in L1
2. **Compute kernel** waits for one tile in CB c_0, copies it to DST register, executes `exp_tile_init()` + `exp_tile()` (SFPU exponential), packs result to CB c_2 (output circular buffer)
3. **Writer kernel** waits for one tile in CB c_2, writes it back to DRAM via NoC
4. Steps 1-3 repeat for all tiles assigned to each core, with double-buffering enabling overlap

## Circular Buffer Configuration

| CB ID | Name/Purpose | Data Format | Page Size | Num Pages | Total Size | Buffering | Producer | Consumer |
|---|---|---|---|---|---|---|---|---|
| c_0 | Input buffer | Same as input dtype | tile_size(input_df) | 2 | 2 * tile_size | Double | Reader | Compute |
| c_2 | Output buffer | Same as output dtype | tile_size(output_df) | 2 | 2 * tile_size | Double | Compute | Writer |

**Note**: CB c_1 (tmp0) is only allocated for HARDSHRINK and LOGIT operations, not for EXP.

## Pipeline Pattern Summary
Both CB c_0 and CB c_2 are configured with 2 pages and the kernel processes 1 page at a time, yielding **double-buffered** operation. This allows the reader to fill page N+1 while compute processes page N, and compute to fill the output while writer drains a previous result.

## Index Calculations
The program factory uses `TensorAccessor` for both reader and writer kernels. The accessor is initialized with compile-time args derived from `TensorAccessorArgs(*buffer)`, which encodes the buffer's memory layout (interleaved bank mapping). At runtime, `noc_async_read_page(i, s, l1_addr)` and `noc_async_write_page(i, s, l1_addr)` translate the linear page index `i` to the correct DRAM bank and offset via the accessor `s`.

Each core receives a `start_id` and `num_pages` count, and iterates linearly from `start_id` to `start_id + num_pages`.

## Memory Access Patterns

### Read Pattern
Sequential tile reads. Each core reads a contiguous range of tile indices `[start_id, start_id + num_pages)` from interleaved DRAM. One tile is read per iteration with a NoC async read followed by a barrier before pushing to CB.

### Write Pattern
Sequential tile writes. Each core writes tiles in the same contiguous index order to interleaved DRAM. One tile is written per iteration: wait for CB front, NoC async write, flush, then pop CB.

## Core Distribution Strategy

| Property | Value |
|---|---|
| Grid Topology | Full compute grid (`compute_with_storage_grid_size`) |
| Work Splitting | `split_work_to_cores()` divides total tiles across available cores |
| Core Group 1 | Gets `num_pages_per_core_group_1` tiles (larger share) |
| Core Group 2 | Gets `num_pages_per_core_group_2` tiles (remainder group, may be empty) |
| Core Ordering | Column-major: `core = {i / num_cores_y, i % num_cores_y}` |
| Load Balancing | At most 1 tile difference between group 1 and group 2 |

Cores are enumerated in column-major order. The `split_work_to_cores` utility creates two core groups to handle the case where tiles do not divide evenly: group 1 gets `ceil(num_pages / num_cores)` tiles, group 2 gets `floor(num_pages / num_cores)` tiles. Separate compute kernel instances are compiled for each group (different `per_core_block_cnt` compile-time arg).

## Arguments

### Compile-Time Arguments

**Reader Kernel**:

| Index | Name | Type | Description |
|---|---|---|---|
| 0+ | TensorAccessorArgs | uint32_t[] | Encodes buffer type, memory layout, and bank mapping for the source buffer |

**Writer Kernel**:

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | cb_id_out | uint32_t | Output circular buffer index (always 2 / c_2) |
| 1+ | TensorAccessorArgs | uint32_t[] | Encodes buffer type, memory layout, and bank mapping for the destination buffer |

**Compute Kernel**:

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | per_core_block_cnt | uint32_t | Number of blocks (tiles) this core must process |
| 1 | per_core_block_dim | uint32_t | Tiles per block (always 1 for this factory) |

### Runtime Arguments

**Reader Kernel** (per core):

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | src_addr | uint32_t | Base address of the input buffer in DRAM |
| 1 | num_pages | uint32_t | Number of pages (tiles) this core reads |
| 2 | start_id | uint32_t | Starting page index for this core |

**Writer Kernel** (per core):

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | dst_addr | uint32_t | Base address of the output buffer in DRAM |
| 1 | num_pages | uint32_t | Number of pages (tiles) this core writes |
| 2 | start_id | uint32_t | Starting page index for this core |

**Compute Kernel** (per core):

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | packed_scalar1 | uint32_t | Unused for EXP (always 0) |
| 1 | packed_scalar2 | uint32_t | Unused for EXP (always 0) |

## Kernel Implementations

| Kernel | File | Assigned Cores | Role |
|---|---|---|---|
| Reader | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | all_cores | Reads tiles from DRAM to CB c_0 |
| Compute | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` | core_group_1, core_group_2 | Executes SFPU exp on each tile |
| Writer | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | all_cores | Writes tiles from CB c_2 to DRAM |

### Reader Kernel

| Property | Value |
|---|---|
| File | `reader_unary_interleaved_start_id.cpp` |
| Assigned Cores | all_cores |

**Key Logic**:
- Iterates from `start_id` to `start_id + num_pages`, reading one tile per iteration
- Uses `TensorAccessor` with compile-time args for address translation from page index to DRAM bank/offset
- Gets CB page size dynamically from `get_local_cb_interface(cb_id_in0).fifo_page_size`, making it layout-agnostic
- Supports optional `BACKWARDS` mode (reverse iteration) via compile-time define (not used for EXP)
- **Synchronization**: Calls `cb_reserve_back(c_0, 1)` before each read to ensure space, `noc_async_read_barrier()` to wait for DMA completion, then `cb_push_back(c_0, 1)` to signal data available to compute

### Compute Kernel

| Property | Value |
|---|---|
| File | `eltwise_sfpu.cpp` |
| Assigned Cores | core_group_1 and core_group_2 (separate kernel instances with different `per_core_block_cnt`) |

**Key Logic**:
- Calls `init_sfpu(c_0, c_2)` once at startup to configure unpack/pack hardware for SFPU mode
- Outer loop iterates `per_core_block_cnt` times (one iteration per block); inner loop iterates `per_core_block_dim` times (always 1 for this factory, so effectively one tile per outer iteration)
- For each tile: acquires DST registers via `tile_regs_acquire()`, waits for input via `cb_wait_front(c_0, 1)`, copies tile from CB c_0 to DST register 0 via `copy_tile(c_0, 0, 0)`
- Executes `SFPU_OP_CHAIN_0` macro which expands to `exp_tile_init<1u>(); exp_tile<1u>(0);` -- the `1u` template argument means `approx=true` (fast approximate mode)
- After SFPU execution: `tile_regs_commit()` hands DST to packer, `tile_regs_wait()` waits for pack readiness, `pack_tile(0, c_2)` packs from DST to output CB
- Pops input tile via `cb_pop_front(c_0, 1)`, releases DST registers via `tile_regs_release()`
- **Synchronization**: `cb_reserve_back(c_2, per_core_block_dim)` at block start; `cb_push_back(c_2, per_core_block_dim)` at block end; `cb_wait_front(c_0, 1)` / `cb_pop_front(c_0, 1)` per tile

### Writer Kernel

| Property | Value |
|---|---|
| File | `writer_unary_interleaved_start_id.cpp` |
| Assigned Cores | all_cores |

**Key Logic**:
- Iterates from `start_id` to `start_id + num_pages`, writing one tile per iteration
- Uses `TensorAccessor` with compile-time args for destination address translation
- Supports `OUT_SHARDED` mode (wait for all tiles at once, no write loop) but this is not used in the interleaved factory
- **Synchronization**: Calls `cb_wait_front(c_2, 1)` to wait for compute output, `noc_async_write_page()` to initiate DMA, `noc_async_writes_flushed()` to ensure write issued, then `cb_pop_front(c_2, 1)` to free the buffer slot. Final `noc_async_write_barrier()` after all tiles

## Implementation Notes

- **Program factory variants**: Three factories -- `UnaryProgramFactory` (interleaved, full grid), `UnarySubCoreGridProgramFactory` (interleaved, sub-core grid), `UnaryShardedProgramFactory` (sharded inputs). Selection is based on input tensor properties: sharded input selects sharded factory, presence of `sub_core_grids` selects sub-core grid factory, otherwise default factory.
- **Type-based operation variants**: EXP supports BF16, FP32, INT32, and UINT32 input types. The data type affects compile defines (`INP_FLOAT32`, `INP_INT32`, `INP_UINT32`, or `INP_FLOAT`) and CB page sizes, but the kernel path is the same for all types. When a float param is present (as with EXP), the `exp_tile_init<approx>()` template is instantiated with the approx flag.
- **UnpackToDestFP32 mode**: Enabled when `args.preserve_fp32_precision` is true, which sets `UnpackToDestMode::UnpackToDestFp32` on CB c_0 and c_1. This causes the unpacker to write FP32 values to the DST register instead of the default reduced precision.
- **Broadcast type selection**: N/A. EXP is a unary operation with no broadcasting.
- **Sharding support and constraints**: Sharded inputs are routed to `UnaryShardedProgramFactory` (not analyzed in depth here). The interleaved factory analyzed above does not support sharded inputs.
- **FP32 dest accumulation**: Controlled by `args.fp32_dest_acc_en`. When true, the DST register accumulates in FP32 precision. Passed directly to `ComputeConfig::fp32_dest_acc_en`.

## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/exp.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h` (macro `SFPU_TEMPLATE_PARAMS_KERNEL_FN`) |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_exp.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. The compute kernel's `SFPU_OP_CHAIN_0` macro expands to `exp_tile_init<1u>(); exp_tile<1u>(0);`, calling the API header functions in `exp.h`.
2. `exp_tile_init<approx=true>()` invokes `SFPU_TEMPLATE_INIT_KERNEL(exponential, sfpu::exp_init, true, true, 0x3F800000, true)`, which calls `llk_math_eltwise_unary_sfpu_init<SfpuType::exponential, true>()` to configure SFPU registers and address modes, then calls `_init_exponential_<true, true, 0x3F800000, true>()` to program constants and macro instruction registers.
3. `exp_tile<approx=true>(0)` invokes `SFPU_TEMPLATE_PARAMS_KERNEL_FN(calculate_exponential, true, true, DST_ACCUM_MODE, false, false, true, 8, 0, VectorMode::RC, scale)`, which calls `_llk_math_eltwise_unary_sfpu_params_<true>()` with a functor bound to `ckernel::sfpu::_calculate_exponential_<true, true, DST_ACCUM_MODE, false, 8, false, true>`.
4. `_llk_math_eltwise_unary_sfpu_params_` sets the DST write address for tile index 0, stalls until SFPU is ready, then iterates over 4 faces (VectorMode::RC), calling `_calculate_exponential_()` once per face and advancing the DEST address by 16 rows between faces.
5. Inside `_calculate_exponential_`, the `FAST_APPROX && APPROXIMATION_MODE && CLAMP_NEGATIVE` branch executes: 8 SFPLOADMACRO calls using Macro Sequence 1 (sanitize/clamp), then 8 SFPLOADMACRO calls using Macro Sequence 0 (compute Schraudolph exp), processing one tile face (16x16 elements, 8 DEST column-pairs).

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default) -- all 4 faces of the 32x32 tile are processed. Each face is 16x16 elements.
- **Operation invocation**: The dispatch function loops 4 times (once per face), calling `_calculate_exponential_()` each iteration. In the default fast-approx-with-clamping path, each invocation processes all 8 column-pairs of one face via 16 SFPLOADMACRO instructions (8 sanitize + 8 compute).
- **DEST address progression**: Between faces, `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` advances the DEST write address by 16 rows (two calls to `math::inc_dst_addr<8>()`). Within a face, the SFPLOADMACRO `dest_reg_addr` operands explicitly address each of the 8 column-pairs at offsets 0, 2, 4, 6, 8, 10, 12, 14 -- no ADDR_MOD auto-increment is used in the CLAMP_NEGATIVE path since each LOADMACRO specifies its DEST offset directly.

### Annotated SFPU Kernel Source

The default EXP invocation resolves to `APPROXIMATION_MODE=true, FAST_APPROX=true, CLAMP_NEGATIVE=true, ITERATIONS=8`. This takes the first branch of `_calculate_exponential_`, which uses raw `TTI_` instructions with SFPLOADMACRO-based macro sequencing. While this uses raw instructions, the CC logic is entirely encapsulated within the SWAP macro instruction (not visible as explicit CC manipulation in the instruction stream), so Style A (inline-commented source) is used.

The init function `_init_exponential_<true, true, 0x3F800000, true>()` programs the constants and macro instructions. The calculate function executes the programmed macros. Both are shown below (Blackhole variant).

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_exp.h

// === Init function (called once per tile via exp_tile_init) ===

template <bool APPROXIMATION_MODE, bool FAST_APPROX, std::uint32_t scale /* 1.0f in FP32 */, bool CLAMP_NEGATIVE = true>
inline void _init_exponential_() // APPROXIMATION_MODE=true, FAST_APPROX=true, scale=0x3F800000 (1.0f), CLAMP_NEGATIVE=true
{
    // Implementation notes, see the original file for more details

    constexpr float LN2_RECIP = 1.4426950408889634f;
    constexpr float A         = 256.0f * LN2_RECIP;        // A = (2^8)/ln(2) = 369.3299...
    constexpr float B_minus_C = 32500.818359375f;           // B = 127*256, C ~= 11.2 error correction
    constexpr float THRESHOLD = -88.5f;                     // Below this, exp(x) underflows to ~0

    constexpr float scale_fp32 = __builtin_bit_cast(float, scale); // = 1.0f when scale=0x3F800000

    constexpr float A_scaled         = A * scale_fp32;       // = A when scale=1.0
    constexpr float THRESHOLD_scaled = THRESHOLD / scale_fp32; // = -88.5 when scale=1.0

    // Load THRESHOLD into LREG[14] (used by SWAP for input clamping)
    TTI_SFPLOADI(0, 0xA, lo16(THRESHOLD_scaled));           // LREG[0] lower 16 bits
    TTI_SFPLOADI(0, 0x8, hi16(THRESHOLD_scaled));           // LREG[0] upper 16 bits
    TTI_SFPCONFIG(0, 14, 0); // Move LREG[0] -> LREG[14] = -88.5

    // Load A into LREG[12] (MAD multiplicand)
    TTI_SFPLOADI(0, 0xA, lo16(A_scaled));
    TTI_SFPLOADI(0, 0x8, hi16(A_scaled));
    TTI_SFPCONFIG(0, 12, 0); // LREG[12] = A = 369.3299...

    // Load (B-C) into LREG[13] (MAD addend)
    TTI_SFPLOADI(0, 0xA, lo16(B_minus_C));
    TTI_SFPLOADI(0, 0x8, hi16(B_minus_C));
    TTI_SFPCONFIG(0, 13, 0); // LREG[13] = B-C = 32500.818...

    // Program Macro Instruction 0 (SWAP) via SFPCONFIG method:
    // SFPSWAP(imm12=0, lreg_src_c=0, lreg_dest=14, instr_mod1=1)
    // Compares loaded value against LREG[14](-88.5), outputs max(input, -88.5)
    TTI_SFPLOADI(0, 0xA, 0x00E1); // Encoded SWAP instruction lower bits
    TTI_SFPLOADI(0, 0x8, 0x9200); // Encoded SWAP instruction upper bits
    TTI_SFPCONFIG(0, 0, 0); // Install into Macro Instruction Register 4 (programmable slot 0)
    TTI_SFPNOP;

    // Program Macro Instruction 1 (MAD) via backdoor load: dest=13 -> Macro Register 5
    TTI_SFPMAD(12, 0, 13, 13, 0); // LREG[dest] = LREG[12](A) * LREG[loaded](y) + LREG[13](B-C)

    // Program Macro Instruction 2 (STOCHRND) via backdoor load: dest=14 -> Macro Register 6
    // instr_mod1=14: FP32 input, unsigned INT16 output, use imm as descale
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 14, 14);

    // Program Macro Instruction 3 (SFPSHFT) via backdoor load: dest=15 -> Macro Register 7
    // Shift left by 15 bits to move INT16 result into FP32 exponent bits
    TTI_SFPSHFT(15, 0, 15, 1); // imm=15, instr_mod1=1 (shift left)

    // Program Macro Sequence Register 1 (sanitize sequence): LD, SWAP, delay, STORE
    TTI_SFPLOADI(0, 0xA, 0x0004); // Slot1(Simple)=SWAP@mux[4],delay=0; Slot2(MAD)=unused
    TTI_SFPLOADI(0, 0x8, 0x1300); // Slot3(Round)=unused; Slot4(Store)=STORE@mux[3],delay=2
    TTI_SFPCONFIG(0, 5, 0); // Install into Macro Sequence Register 1 (dest=5)

    // Program Macro Sequence Register 0 (compute sequence): LD, MAD, delay, ROUND, SHIFT, STORE
    TTI_SFPLOADI(0, 0xA, 0x85DF); // Slot1(Simple)=SHIFT@mux[7],delay=3,staging; Slot2(MAD)=MAD@mux[5],delay=0
    TTI_SFPLOADI(0, 0x8, 0x6316); // Slot3(Round)=ROUND@mux[6],delay=2; Slot4(Store)=STORE@mux[3],delay=4,staging-src
    TTI_SFPCONFIG(0, 4, 0); // Install into Macro Sequence Register 0 (dest=4)

    // Reset LoadMacroConfig[Lane].Misc for all lanes
    TTI_SFPCONFIG(0, 8, 1);
}

// === Calculate function (called once per face via _llk_math_eltwise_unary_sfpu_params_) ===

template <bool APPROXIMATION_MODE, bool SCALE_EN, int ITERATIONS, bool FAST_APPROX, bool SKIP_POSITIVE_CHECK, bool CLAMP_NEGATIVE = true>
void _calculate_exponential_(const std::uint16_t exp_base_scale_factor)
// Resolved: APPROXIMATION_MODE=true, SCALE_EN=false, ITERATIONS=8, FAST_APPROX=true, SKIP_POSITIVE_CHECK=false, CLAMP_NEGATIVE=true
{
    // Phase 1: Sanitize -- clamp all 8 column-pairs of this face to [-88.5, +inf]
    // Each SFPLOADMACRO triggers Macro Sequence 1 (LD, SWAP, STORE) on the target LREG and DEST offset
    // LREG cycles 0,1,2,3,0,1,2,3 to avoid write-port conflicts; NOP after each for SWAP 2-cycle latency
    TTI_SFPLOADMACRO(4, 0, ADDR_MOD_7, 0);  // Seq1, LREG[0], DEST offset 0 (even cols, rows 3:0)
    TTI_SFPNOP;
    TTI_SFPLOADMACRO(5, 0, ADDR_MOD_7, 2);  // Seq1, LREG[1], DEST offset 2 (odd cols, rows 3:0)
    TTI_SFPNOP;
    TTI_SFPLOADMACRO(6, 0, ADDR_MOD_7, 4);  // Seq1, LREG[2], DEST offset 4 (even cols, rows 7:4)
    TTI_SFPNOP;
    TTI_SFPLOADMACRO(7, 0, ADDR_MOD_7, 6);  // Seq1, LREG[3], DEST offset 6 (odd cols, rows 7:4)
    TTI_SFPNOP;
    TTI_SFPLOADMACRO(4, 0, ADDR_MOD_7, 8);  // Seq1, LREG[0], DEST offset 8 (even cols, rows 11:8)
    TTI_SFPNOP;
    TTI_SFPLOADMACRO(5, 0, ADDR_MOD_7, 10); // Seq1, LREG[1], DEST offset 10 (odd cols, rows 11:8)
    TTI_SFPNOP;
    TTI_SFPLOADMACRO(6, 0, ADDR_MOD_7, 12); // Seq1, LREG[2], DEST offset 12 (even cols, rows 15:12)
    TTI_SFPNOP;
    TTI_SFPLOADMACRO(7, 0, ADDR_MOD_7, 14); // Seq1, LREG[3], DEST offset 14 (odd cols, rows 15:12)
    // No NOP needed here: next LOADMACRO is computational (doesn't immediately use SIMPLE unit)

    // Phase 2: Compute -- Schraudolph exp approximation on the sanitized values
    // Each SFPLOADMACRO triggers Macro Sequence 0 (LD, MAD, ROUND, SHIFT, STORE)
    // Pipeline: loads value y from DEST, computes i = A*y + (B-C), rounds to INT16, shifts left by 15
    TTI_SFPLOADMACRO(0, 0, ADDR_MOD_7, 0);  // Seq0, LREG[0], DEST offset 0
    TTI_SFPLOADMACRO(1, 0, ADDR_MOD_7, 2);  // Seq0, LREG[1], DEST offset 2
    TTI_SFPLOADMACRO(2, 0, ADDR_MOD_7, 4);  // Seq0, LREG[2], DEST offset 4
    TTI_SFPLOADMACRO(3, 0, ADDR_MOD_7, 6);  // Seq0, LREG[3], DEST offset 6
    TTI_SFPLOADMACRO(0, 0, ADDR_MOD_7, 8);  // Seq0, LREG[0], DEST offset 8
    TTI_SFPLOADMACRO(1, 0, ADDR_MOD_7, 10); // Seq0, LREG[1], DEST offset 10
    TTI_SFPLOADMACRO(2, 0, ADDR_MOD_7, 12); // Seq0, LREG[2], DEST offset 12
    TTI_SFPLOADMACRO(3, 0, ADDR_MOD_7, 14); // Seq0, LREG[3], DEST offset 14
    TTI_SFPNOP; // Allow final LOADMACRO pipeline to drain before next face iteration
}
```

**Algorithm Summary (Schraudolph Fast Exp with Input Clamping)**:

The algorithm is based on "A Fast, Compact Approximation of the Exponential Function" by Nicol N. Schraudolph. It exploits the IEEE 754 float representation where the bit pattern, read as an integer, is approximately linear in log2(value). Computing `exp(x)` reduces to:

1. **Sanitize**: Clamp input to `[-88.5, +inf]` using SWAP(max) against the threshold constant in LREG[14]. This prevents underflow artifacts for very negative inputs.
2. **Compute**: `i = A * x + (B - C)`, where `A = 256/ln(2)`, `B = 127*256`, `C ~= 11.2` (error-minimizing correction). The MAD result is an FP32 value encoding the approximate integer bit pattern.
3. **Round**: Convert the FP32 MAD result to a 16-bit unsigned integer via stochastic rounding (STOCHRND).
4. **Shift**: Left-shift the 16-bit integer by 15 bits to place it into the exponent field of an FP32 value, yielding the approximate `exp(x)`.

Each tile face (16x16 elements) is processed as 8 column-pairs (each column-pair is 32 SFPU lanes), requiring 8 SFPLOADMACRO sanitize calls + 8 SFPLOADMACRO compute calls = 16 total LOADMACROs per face, 64 per tile.

### SFPU Instructions Used

| Instruction | Description |
|---|---|
| `TTI_SFPLOADI` | Loads a 16-bit immediate into the lower or upper half of LREG[0]. Used to construct 32-bit constants (two loads per constant) and to program macro sequence registers. |
| `TTI_SFPCONFIG` | Configures SFPU state: moves LREG[0] to a target LREG (dest < 16), installs macro instructions (dest 0-3), installs macro sequences (dest 4-5), or sets LoadMacroConfig misc bits (dest 8, mode 1). |
| `TTI_SFPMAD` | Multiply-add: `VD = VA * VB + VC`. When `lreg_dest` is 13, 14, or 15, the instruction is captured into programmable macro instruction registers 5, 6, or 7 via backdoor load instead of executing. In the compute macro, computes `i = A * y + (B-C)`. |
| `TTI_SFP_STOCH_RND` | Stochastic rounding: converts FP32 to integer format. With `lreg_dest=14`, backdoor-loads to macro register 6. `instr_mod1=14` selects FP32 input to unsigned INT16 output mode. |
| `TTI_SFPSHFT` | Bit shift: shifts LREG value by immediate amount. With `lreg_dest=15`, backdoor-loads to macro register 7. `imm=15, instr_mod1=1` shifts left by 15 bits to position the integer result in the FP32 exponent field. |
| `TTI_SFPSWAP` | (Encoded via SFPCONFIG) Compares two values and swaps them so the larger ends up in the target register. Used in the sanitize macro to clamp inputs: `max(input, -88.5)`. |
| `TTI_SFPLOADMACRO` | Triggers execution of a pre-programmed macro sequence on a specified LREG and DEST address. The first operand selects the LREG (bits 1:0) and macro sequence (bit 2: 0 = Seq0/compute, 1 = Seq1/sanitize). The DEST offset selects which column-pair to process. |
| `TTI_SFPNOP` | No-operation. Required after SWAP-based LOADMACROs due to the SWAP unit's 2-cycle latency, and after the final compute LOADMACRO to allow the pipeline to drain. |

### SFPU Register Usage

| Register | Usage |
|---|---|
| **LREG[0-3]** | Working registers. Cycled by SFPLOADMACRO (lreg_dest field) to avoid write-port conflicts. Each LOADMACRO loads a value from DEST into its assigned LREG, processes it through the macro sequence pipeline, and stores the result back to DEST. |
| **LREG[12]** | Constant `A = 369.33...` (= 256/ln2). Used as the multiply operand (`VA`) in the MAD macro instruction. |
| **LREG[13]** | Constant `B - C = 32500.82...`. Used as the add operand (`VC`) in the MAD macro instruction. |
| **LREG[14]** | Constant `-88.5` (clamping threshold). Used as the comparison operand in the SWAP macro instruction. |
| **DEST register** | Source and destination for tile data. The SFPU reads from and writes to DEST at explicit offsets 0-14 (8 column-pairs per face). Each column-pair contains 32 elements (the SFPU lane width). |
| **Macro Instruction Registers 4-7** | Programmed during init: Reg 4 = SWAP, Reg 5 = MAD, Reg 6 = STOCHRND, Reg 7 = SFPSHFT. These are the atomic operations composed by the macro sequences. |
| **Macro Sequence Register 0** | The compute sequence: LD -> MAD -> (delay) -> ROUND -> SHIFT -> STORE. |
| **Macro Sequence Register 1** | The sanitize sequence: LD -> SWAP -> (delay) -> STORE. |

### Address Mode Configuration

The default EXP init (`_llk_math_eltwise_unary_sfpu_init_<SfpuType::exponential>()`) configures ADDR_MOD_7 with all-zero increments:

```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}.set(ADDR_MOD_7);
```

This means ADDR_MOD_7 does **not** auto-increment the DEST address between SFPLOADMACRO calls. Instead, each SFPLOADMACRO explicitly specifies its DEST offset (0, 2, 4, ..., 14) in the `dest_reg_addr` operand. This is required because the sanitize and compute phases visit the same 8 DEST offsets in order, and auto-increment would not return to offset 0 between phases.

**Hardware generation differences**:
- **Blackhole**: SFPLOADMACRO uses `ADDR_MOD_7` (value 7) for the addr_mod operand. The `_llk_math_eltwise_unary_sfpu_start_` function does not call `set_addr_mod_base()`.
- **Wormhole B0**: SFPLOADMACRO uses literal `3` (ADDR_MOD_3) for the addr_mod operand. The `_llk_math_eltwise_unary_sfpu_start_` function additionally calls `math::set_addr_mod_base()` to establish the address mode base pointer. The addr_mod configuration is otherwise identical (all-zero increments for the default exponential SfpuType).

**Note on the non-clamping fast-approx path** (`CLAMP_NEGATIVE=false`, `ITERATIONS=8`): This alternative path reconfigures ADDR_MOD_7 at runtime with `dest.incr = 2` for auto-increment, then uses the replay buffer to execute 8 SFPLOADMACRO + SFPSHFT2 pairs. This path is not the default for EXP but is available when `InputClamping::None` is specified.

## External Knowledge Sources

### DeepWiki Queries
1. **Query**: "How does the unary SFPU program factory work in TTNN? What kernels does it use? How does EXP get dispatched? What is the difference between FPU and SFPU paths?"
   **Reason**: Needed architectural overview of the unary SFPU pipeline before reading source code
   **Key Findings**: Confirmed three-kernel pipeline (reader/compute/writer), EXP uses `eltwise_sfpu.cpp` compute kernel, operation is specified via compile-time defines (`SFPU_OP_CHAIN_0`), factory selection is based on tensor properties not op type

2. [SFPU] **Query**: "How does the EXP (exponential) SFPU kernel work? Trace the call chain from the compute kernel API (exp_tile_init, exp_tile) through LLK dispatch down to the ckernel SFPU implementation. What files are involved?"
   **Reason**: Needed to identify all files in the SFPU abstraction layers and understand the call chain from API to core implementation
   **Key Findings**: Identified the 4-layer abstraction (API header -> LLK macros -> params dispatch -> ckernel SFPU), confirmed the Schraudolph algorithm for fast approx mode, identified multiple implementation variants (_sfpu_exp_ for precise, _calculate_exponential_approx_ for approx, and LOADMACRO-based for fast approx)

3. [SFPU] **Query**: "How is the exponential (exp) SFPU kernel implemented in the LLK layer? Show the call chain from llk_math_eltwise_unary_sfpu_exponential through to ckernel_sfpu_exp.h. What SFPU instructions and approximation modes are used?" (asked to `tenstorrent/tt-llk`)
   **Reason**: Needed LLK-specific details on the SFPU kernel implementation, particularly the LOADMACRO-based fast approximation
   **Key Findings**: Confirmed the macro instruction programming flow (SWAP, MAD, STOCHRND, SFPSHFT), the sanitize-then-compute two-phase approach, the constants (A, B-C, threshold), and the replay buffer variants for non-clamping paths

### Documentation References
1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Needed to understand how EXP maps to compute kernel path and compile defines
   **Key Information**: `get_compute_kernel_path()` returns default `"eltwise_sfpu.cpp"` for EXP; `get_macro_definition()` returns `"SFPU_OP_EXP_INCLUDE"`; `get_block_defines()` generates `SFPU_OP_CHAIN_0` with `exp_tile_init<1u>(); exp_tile<1u>(0);` when approx param is true

2. **Source**: `tt_metal/hw/inc/api/compute/eltwise_unary/exp.h`
   **Reason**: Needed to understand the SFPU exp tile API and its template parameters
   **Key Information**: `exp_tile<approx, fast_and_approx, scale_en, skip_positive_check, input_clamping, iterations>()` -- approx=true enables fast approximate mode; fast_and_approx defaults to true; InputClamping::ClampToNegative is default (clamps inputs below ~-88.5)

3. **Source**: `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
   **Reason**: Needed to verify how `SFPU_OP_EXP_INCLUDE` triggers inclusion of exp.h
   **Key Information**: Conditional `#if SFPU_OP_EXP_INCLUDE` includes `api/compute/eltwise_unary/exp.h`

4. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_device_operation.cpp`
   **Reason**: Needed to understand factory selection logic
   **Key Information**: `select_program_factory()` checks `is_sharded()` then `sub_core_grids.has_value()` then defaults to `UnaryProgramFactory`
