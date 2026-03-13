# Dropout Implementation Analysis

## Overview

The dropout operation randomly zeroes out tensor elements with a given probability and scales the remaining elements by a scale factor (typically `1.0 / (1.0 - prob)`). This is a standard regularization technique used during neural network training. The implementation uses the SFPU vector unit to generate pseudo-random numbers per element and conditionally zero or scale each value.

**Program factory path**: `ttnn/cpp/ttnn/operations/experimental/dropout/device/dropout_program_factory.cpp`

## Path Selection: FPU vs SFPU

This operation has a **single SFPU-only implementation path**. There is no FPU variant. The only factory selection logic is between `DropoutProgramFactory` (single-device) and `DropoutMeshWorkloadFactory` (multi-device with per-device seeds), controlled by the `use_per_device_seed` attribute in `DropoutParams`. Both factories produce identical kernel configurations; the mesh variant simply creates one program per device with a seed offset equal to the device ID. All detailed analysis below applies to both variants since the kernel-level implementation is the same.

## Work Unit Definition

One work unit is **one tile** (32x32 elements). The total number of tiles is computed as `input.physical_volume() / TILE_HW`. Each core processes a contiguous range of tiles, with `per_core_block_cnt` tiles per core and a block size of 1 tile (i.e., tiles are processed one at a time in the compute kernel's inner loop).

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|---|---|
| **Dimension Convention** | N-dimensional (shape preserved) |
| **Tensor Layout** | TILE (32x32) |
| **Memory Layout** | INTERLEAVED (required by validation) |
| **Buffer Type** | DRAM (device storage) |
| **Data Type** | BF16 (must match output) |

### Output Tensor

| Property | Value |
|---|---|
| **Dimension Convention** | Same as input |
| **Tensor Layout** | TILE (32x32) |
| **Memory Layout** | INTERLEAVED (must match input) |
| **Buffer Type** | DRAM (device storage) |
| **Data Type** | Same as input (enforced by validation) |

### Layout Transformations

None. Input and output share the same tile layout and memory layout. No tilize/untilize or reshard operations are performed.

## Data Flow Pattern

1. **Reader** reads one tile at a time from DRAM into CB c_0 using `noc_async_read_tile` with TensorAccessor-based addressing. Each tile is read sequentially from `start_id` to `start_id + num_tiles`.
2. **Compute** waits for one tile in CB c_0, copies it to the DEST register via `copy_tile`, applies the SFPU `dropout_tile` operation (which generates a random number per element, zeroes elements below the probability threshold, and scales surviving elements), then packs the result into CB c_2.
3. **Writer** waits for one tile in CB c_2, writes it to DRAM using `noc_async_write_tile` with TensorAccessor-based addressing, then pops the tile.

## Circular Buffer Configuration

| CB ID | Purpose | Data Format | Page Size | Num Pages | Total Size | Buffering | Producer | Consumer |
|---|---|---|---|---|---|---|---|---|
| `c_0` (CB 0) | Input tiles from reader | Same as input dtype | `single_tile_size_in` | 2 | 2 x tile size | Double-buffered | Reader | Compute |
| `c_2` (CB 2) | Output tiles from compute | Same as output dtype | `single_tile_size_out` | 2 | 2 x tile size | Double-buffered | Compute | Writer |

## Pipeline Pattern Summary

Both CB c_0 and CB c_2 are allocated with 2 pages (capacity = 2, block size = 1), making them **double-buffered**. This allows the reader to fill the next tile while compute processes the current one, and compute to produce the next output tile while the writer drains the current one. The pipeline is: Reader -> CB c_0 -> Compute -> CB c_2 -> Writer.

## Index Calculations

Tile indexing uses TensorAccessor for both reader and writer. The TensorAccessor is constructed from `TensorAccessorArgs` (passed as compile-time args) and the buffer address. The `noc_async_read_tile(i, accessor, l1_addr)` and `noc_async_write_tile(i, accessor, l1_addr)` calls translate a linear tile index `i` into a physical DRAM address, handling bank interleaving internally. Each core receives a `start_id` (cumulative tile offset) and `num_tiles` count, processing tile indices `[start_id, start_id + num_tiles)`.

## Memory Access Patterns

### Read Pattern

Sequential tile reads from DRAM. Each core reads tiles with contiguous linear indices from `start_id` to `start_id + num_tiles - 1`. One tile is read per iteration with a barrier after each read (`noc_async_read_barrier`), ensuring the tile is fully in L1 before it is pushed to the CB. TensorAccessor handles the interleaved bank mapping.

### Write Pattern

Sequential tile writes to DRAM. Each core writes tiles with contiguous linear indices from `start_id` to `start_id + num_tiles - 1`. One tile is written per iteration with a barrier after each write (`noc_async_write_barrier`). TensorAccessor handles the interleaved bank mapping.

## Core Distribution Strategy

| Property | Value |
|---|---|
| **Grid Topology** | Full compute grid (`compute_with_storage_grid_size`) |
| **Work Splitting** | `split_work_to_cores(grid_size, num_tiles)` |
| **Core Iteration Order** | Column-major: `core = {i / num_cores_y, i % num_cores_y}` |
| **Group 1 (primary)** | Cores processing `ceil(num_tiles / num_cores)` tiles each |
| **Group 2 (remainder)** | Cores processing `floor(num_tiles / num_cores)` tiles each (empty if evenly divisible) |
| **Load Balancing** | Groups differ by at most 1 tile per core |

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | `cb_id_in0` | uint32_t | Circular buffer index for input (c_0 = 0) |
| 1+ | TensorAccessorArgs | uint32_t[] | Source buffer tensor accessor parameters (appended by `TensorAccessorArgs`) |

#### Writer Kernel

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | `cb_id_out` | uint32_t | Circular buffer index for output (c_2 = 2) |
| 1+ | TensorAccessorArgs | uint32_t[] | Destination buffer tensor accessor parameters (appended by `TensorAccessorArgs`) |

#### Compute Kernel

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | `per_core_block_cnt` | uint32_t | Number of tile blocks this core processes (differs between group 1 and group 2) |
| 1 | `per_core_block_dim` | uint32_t | Number of tiles per block (always 1) |
| 2 | `prob_int` | uint32_t | Dropout probability as `(double)INT_MAX * prob`, used as PRNG threshold |
| 3 | `uscale` | uint32_t | Scale factor as bit-cast uint32_t of float (typically `1.0 / (1.0 - prob)`) |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | `src_addr` | uint32_t | Source buffer DRAM address |
| 1 | `num_tiles` | uint32_t | Number of tiles this core reads |
| 2 | `start_id` | uint32_t | Starting tile index for this core |

#### Writer Kernel

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | `dst_addr` | uint32_t | Destination buffer DRAM address |
| 1 | `num_tiles` | uint32_t | Number of tiles this core writes |
| 2 | `start_id` | uint32_t | Starting tile index for this core |

#### Compute Kernel

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | `seed` | uint32_t | PRNG seed for dropout random number generation |

## Kernel Implementations

| Kernel | File | Type | Assigned Cores |
|---|---|---|---|
| Reader | `kernels/dataflow/reader_dropout_interleaved_start_id.cpp` | DataMovement (Reader) | `all_cores` |
| Compute | `kernels/compute/dropout_kernel.cpp` | Compute (SFPU) | `core_group_1` and `core_group_2` (separate kernel handles with different `per_core_block_cnt`) |
| Writer | `kernels/dataflow/writer_dropout_interleaved_start_id.cpp` | DataMovement (Writer) | `all_cores` |

### Reader Kernel

| Property | Value |
|---|---|
| **File** | `ttnn/cpp/ttnn/operations/experimental/dropout/device/kernels/dataflow/reader_dropout_interleaved_start_id.cpp` |
| **Assigned Cores** | `all_cores` |

**Key Logic**:
- Iterates from `start_id` to `start_id + num_tiles` (forward direction; `BACKWARDS` define exists for reverse iteration but is not set by this program factory)
- For each tile: reserves 1 page in CB c_0, reads tile from DRAM via `noc_async_read_tile`, waits for read completion with `noc_async_read_barrier`, then pushes to CB c_0
- Uses TensorAccessor constructed from compile-time `TensorAccessorArgs` for address translation
- **Synchronization**: Produces to CB c_0 via `cb_reserve_back` / `cb_push_back`; the compute kernel consumes from CB c_0

### Compute Kernel

| Property | Value |
|---|---|
| **File** | `ttnn/cpp/ttnn/operations/experimental/dropout/device/kernels/compute/dropout_kernel.cpp` |
| **Assigned Cores** | `core_group_1` (with group 1 tile count) and `core_group_2` (with group 2 tile count) |

**Key Logic**:
- Calls `init_sfpu(c_0, c_2)` to initialize SFPU with input/output CB indices
- Calls `dropout_kernel_init(seed)` to seed the PRNG for this core
- Outer loop iterates `per_core_block_cnt` times (one iteration per block); inner loop iterates `per_core_block_dim` times (always 1, so effectively one tile per outer iteration)
- For each tile:
  1. `tile_regs_acquire()` -- acquire DEST register file
  2. `cb_wait_front(c_0, 1)` -- wait for input tile from reader
  3. `copy_tile(c_0, 0, 0)` -- unpack tile from CB c_0 into DEST register 0
  4. `dropout_tile(0, int_probability, int_scale_factor)` -- SFPU operation: for each element, generates a random number via SFPU PRNG; if random < probability, zeros the element; otherwise multiplies by scale factor
  5. `tile_regs_commit()` / `tile_regs_wait()` -- handoff from math to pack pipeline
  6. `pack_tile(0, c_2)` -- pack DEST register 0 into CB c_2
  7. `cb_pop_front(c_0, 1)` -- free input tile
  8. `tile_regs_release()` -- release DEST register file
- Output CB push (`cb_push_back(c_2, per_core_block_dim)`) happens after the inner loop completes (effectively after each tile since block dim is 1)
- **Synchronization**: Consumes from CB c_0 via `cb_wait_front` / `cb_pop_front`; produces to CB c_2 via `cb_reserve_back` / `cb_push_back`

### Writer Kernel

| Property | Value |
|---|---|
| **File** | `ttnn/cpp/ttnn/operations/experimental/dropout/device/kernels/dataflow/writer_dropout_interleaved_start_id.cpp` |
| **Assigned Cores** | `all_cores` |

**Key Logic**:
- Non-sharded path (default; `OUT_SHARDED` define exists but is not set by this program factory): iterates from `start_id` to `start_id + num_tiles`
- For each tile: waits for 1 page in CB c_2 via `cb_wait_front`, reads the L1 address with `get_read_ptr`, writes tile to DRAM via `noc_async_write_tile`, waits for write completion with `noc_async_write_barrier`, then pops from CB c_2
- Uses TensorAccessor constructed from compile-time `TensorAccessorArgs` for address translation
- `OUT_SHARDED` path (not used here): would simply `cb_wait_front` for all tiles at once without writing to DRAM
- **Synchronization**: Consumes from CB c_2 via `cb_wait_front` / `cb_pop_front`

## Implementation Notes

- **Program factory variants**: Two factories exist -- `DropoutProgramFactory` (single-device) and `DropoutMeshWorkloadFactory` (multi-device with per-device seeds offset by device ID). The mesh factory calls `DropoutProgramFactory::create` per device. Selection is based on `args.use_per_device_seed`.
- **Type-based operation variants**: Input and output dtypes must match (enforced by validation). The data format is derived from the tensor dtype via `datatype_to_dataformat_converter`. No type-specific code paths in kernels; the SFPU dropout operates on whatever format is loaded into DEST.
- **UnpackToDestFP32 mode**: Not used. The compute config sets `fp32_dest_acc_en = false`.
- **Broadcast type selection**: N/A. This is a unary element-wise operation with no broadcasting.
- **Sharding support and constraints**: The validation allows sharded inputs but the program factory only creates interleaved reader/writer kernels. The writer kernel has an `OUT_SHARDED` compile-time define path, but it is not activated by this program factory. Effectively, only INTERLEAVED memory layout is supported for this SFPU path.
- **FP32 dest accumulation**: Disabled (`fp32_dest_acc_en = false`). Math fidelity is set to `HiFi4`. Math approx mode is `false`.
- **Program caching**: The seed is excluded from the program hash (`args_without_seed.seed = 0`) since it is a runtime argument and does not affect compiled kernel structure. This allows program reuse across different seeds.
- **PRNG details**: The probability is converted to an integer threshold as `(double)INT_MAX * prob`. The SFPU generates a pseudo-random number per element and compares it against this threshold. The scale factor is passed as a bit-cast float-to-uint32_t.

## SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/dropout.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h` (macro) and `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (params dispatch) |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_dropout.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. The compute kernel calls `dropout_tile(0, int_probability, int_scale_factor)` (API header `dropout.h`).
2. This expands via the `SFPU_UNARY_PARAMS_KERNEL_EXTRA_ARGS` macro to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_dropout<APPROX>, idst, (int)VectorMode::RC, probability, scale_factor)`.
3. `_llk_math_eltwise_unary_sfpu_params_` (in `llk_math_eltwise_unary_sfpu_params.h`) sets the DEST write address, stalls until SFPU is ready, then loops over 4 tile faces in `VectorMode::RC`, calling the SFPU function once per face and advancing the DEST address by `DEST_FACE_WIDTH` (16 rows) between faces.
4. `calculate_dropout<APPROX>(probability, scale)` (in the metal-layer wrapper `ckernel_sfpu_dropout.h`) forwards to `_calculate_dropout_<APPROXIMATION_MODE, 8>(8, probability, scale)`.
5. `_calculate_dropout_` (in the tt_llk-layer `ckernel_sfpu_dropout.h`) executes the raw SFPU instructions that perform the scale, PRNG, comparison, and conditional zeroing per row.

Additionally, `dropout_kernel_init(seed)` expands via `SFPU_ONE_PARAM_KERNEL_INIT` to call `llk_math_eltwise_unary_sfpu_init<SfpuType::dropout, APPROX>(sfpu::dropout_init<APPROX>, seed)`, which:
- Calls `_llk_math_eltwise_unary_sfpu_init_<SfpuType::dropout>()` to configure SFPU registers and address modes
- Then calls `dropout_init<APPROX>(seed)` which calls `_init_dropout_(seed)` to write the seed to the hardware PRNG register

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- all 4 faces of a 32x32 tile are processed (faces 0-3).
- **Operation invocation**: The params dispatch function loops 4 times (once per face). Each iteration calls `calculate_dropout(probability, scale)` which internally loops 8 times (once per row within the 16-row face, processing 2 rows per SFPU SIMD width -- each iteration processes one "row" of the SFPU's 32-wide SIMD lanes). **Note on ITERATIONS=8**: The template default `ITERATIONS=8` processes 8 rows per face invocation. Since each tile face has 16 rows and the SFPU processes 32 elements per lane, 8 iterations x 4 faces = 32 rows = one complete 32x32 tile.
- **DEST address progression**: On Wormhole, the params dispatch (`_llk_math_eltwise_unary_sfpu_params_`) manually advances DEST by issuing `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` twice after each face (incrementing by 16 rows total = one face width). On Blackhole, the equivalent is done via `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which calls `math::inc_dst_addr<8>()` twice. Within each face, `_calculate_dropout_` uses `sfpi::dst_reg++` at the end of each loop iteration to advance the DEST row pointer by 1 row (auto-incremented via the SFPU address mode).

### Annotated SFPU Kernel Source

This kernel uses raw `TT_`/`TTI_` instructions with condition code (CC) manipulation via `SFPIADD` (implicit CC side effect) and `SFPENCC`. This qualifies as **Style B**.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_dropout.h
// (Blackhole version at tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_dropout.h is identical)

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_dropout_(const int iterations, std::uint32_t probability, std::uint32_t scale)
{
    // SFPU microcode

    TT_SFPLOADI(p_sfpu::LREG1, 10, scale & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG1, 8, scale >> 16);
    TT_SFPLOADI(p_sfpu::LREG2, 10, probability & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, 8, probability >> 16);
#pragma GCC unroll 0
    for (int d = 0; d < iterations; d++)
    {
        ////////////////////////
        // Scale samples
        // sfpi::dst_reg[0] = sfpi::dst_reg[0] * s2vFloat16b(scale);
        ///////////////////////
        TTI_SFPLOAD(p_sfpu::LREG0, 0, 3, 0);
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);

        ////////////////////////
        // Instruction SFPMOV generates a uint32_t pseudorandom number
        // when instr_mod1 = 8 and lreg_c =  9.
        // Arguments: (imm12_math, lreg_c, lreg_dest, instr_mod1)
        // Unset sign-bit for easy comparison with probability
        ////////////////////////
        TTI_SFPMOV(0, 9, p_sfpu::LREG3, 8);
        TTI_SFPSETSGN(0, p_sfpu::LREG3, p_sfpu::LREG3, 1);

        ////////////////////////
        // Drop samples
        // v_if (rand < probability)
        //   sfpi::dst_reg[0] = vConst0;
        ///////////////////////
        TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG3, 10);
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSTORE(0, 0, 3, 0);

        sfpi::dst_reg++;
    }
}

inline void _init_dropout_(const std::uint32_t seed)
{
    init_prng_seed(seed);
}
```

**CC State Machine diagram:**

```
_calculate_dropout_ -- CC State Transitions
================================================================

  CC State: ENABLED (CC.En=1, CC.Res=1)     <-- persistent state
       |
       |  SFPLOADI LREG1, lo16(scale)       (no CC effect)
       |  SFPLOADI LREG1, hi16(scale)       (no CC effect)
       |  SFPLOADI LREG2, lo16(probability) (no CC effect)
       |  SFPLOADI LREG2, hi16(probability) (no CC effect)
       |
       v
  -- Per-row loop (iterations=8) ------

  CC State: ENABLED (CC.En=1, CC.Res=1)     <-- all lanes active
       |
       |  SFPLOAD  LREG0 = DEST[addr]       (CC-guarded: all lanes, since CC.Res=1)
       |  SFPMUL   LREG0 = LREG0 * LREG1   (CC-guarded: all lanes; LREG0 = input * scale)
       |                                      (no CC effect)
       |
       |  SFPMOV   LREG3 = RS[9]            (CC-guarded: all lanes; RS[9]=PRNG, InstrMod=8
       |           (InstrMod=8: read from     reads SFPU Status view; advances PRNG)
       |            RS view, VC=9=PRNG)
       |  SFPSETSGN LREG3.sign = Imm12[0]=0 (CC-guarded: all lanes; clears sign bit
       |           (InstrMod=1: sign from      so rand is positive for unsigned compare)
       |            immediate)                (no CC effect)
       |
       v
  +------------------------------------------+
  | SFPIADD  InstrMod=10 (0b1010)           |
  |   InstrMod[1:0]=2: Reg/Reg subtraction  |
  |   InstrMod[2]=0:   CC updated           |
  |   InstrMod[3]=1:   CC inverted          |
  |                                          |
  | LREG3 = LREG2 - LREG3                   |
  |        = probability - rand              |
  | CC.Res = !((probability - rand) < 0)    |
  |        = (rand <= probability)           |
  +-------------------+----------------------+
                      |
                      v
  CC State: ENABLED where rand <= probability
       |
       |  SFPMOV LREG0 = LCONST_0 (=0.0)   (CC-guarded: only lanes where
       |                                      rand <= probability execute;
       |                                      these lanes get LREG0 = 0.0,
       |                                      zeroing the scaled result.
       |                                      Lanes where rand > probability
       |                                      retain LREG0 = input * scale.)
       v
  +------------------------------------------+
  | SFPENCC  InstrMod=0 (0b0000)            |
  |   InstrMod[3]=0:   CC.Res = 1           |
  |   InstrMod[1:0]=0: CC.En kept (=1)      |
  |                                          |
  | CC.Res = 1, CC.En = 1 (unchanged)       |
  +-------------------+----------------------+
                      |
                      v
  CC State: ENABLED (CC.En=1, CC.Res=1)     <-- all lanes re-enabled
       |
       |  SFPSTORE LREG0 -> DEST[addr]       (all lanes: stores scaled value
       |                                      for kept lanes, 0.0 for dropped lanes)
       |  dst_reg++                           (advance DEST row pointer)
       |
       v
  -- End per-row loop / next iteration -----
```

**CC Invariant note**: The dropout kernel relies on CC.En being 1 (predicated execution enabled) as the persistent SFPU state. The kernel never explicitly sets CC.En -- it relies on CC.En = 1 being the pre-existing state when the SFPU begins execution. The `SFPENCC(0, 0, 0, 0)` call at the end of each iteration restores CC.Res to 1 while preserving CC.En = 1, ensuring all lanes are active for the next iteration's SFPLOAD/SFPMUL. This pattern is shared by other raw-instruction SFPU kernels (e.g., `_calculate_lrelu_` uses `SFPSETCC` + `SFPENCC(0,0,0,0)` with the same CC.En=1 assumption).

### SFPU Instructions Used

| Instruction | Encoding | Description |
|---|---|---|
| `SFPLOADI` | MI | Loads a 16-bit immediate into the upper or lower half of an LREG. Used to load 32-bit `scale` and `probability` values across two instructions (LO16_ONLY mode=0xA, HI16_ONLY mode=0x8). |
| `SFPLOAD` | MR | Loads a value from the DEST register file into an LREG. Here, `TTI_SFPLOAD(LREG0, 0, 3, 0)` loads the current DEST row into LREG0 using IMPLIED format and ADDR_MOD_3. |
| `SFPMUL` (alias of SFPMAD) | O4 | Floating-point multiply-add: `RG[VD] = RG[VA] * RG[VB] + RG[VC]`. Here, `LREG0 = LREG0 * LREG1 + LCONST_0(=0.0)`, effectively multiplying the input by the scale factor. Latency: 2 cycles. |
| `SFPMOV` | O2 | Register move with mode selection. InstrMod=0: copies `RG[VC]` to `RG[VD]`. InstrMod=8: copies from the SFPU Status (RS) view; `RS[9]` is the PRNG counter, which advances the PRNG as a side effect of reading. Used both for PRNG generation (InstrMod=8) and for conditional zeroing (InstrMod=0, copying LCONST_0=0.0). |
| `SFPSETSGN` | O2 | Sets the sign bit of `RG[VC]`, storing result in `RG[VD]`. InstrMod=1: sign bit is taken from `Imm12[0]` (here Imm12=0, so sign=0=positive). This clears the sign bit of the random number so it can be compared as an unsigned integer against the positive probability threshold. |
| `SFPIADD` | O2 | 2's complement integer addition/subtraction. InstrMod=10 (0b1010): InstrMod[1:0]=2 selects Reg/Reg subtraction (`LREG2 - LREG3`), InstrMod[2]=0 enables CC update (CC.Res set based on result sign), InstrMod[3]=1 inverts CC.Res. Net effect: CC.Res = (rand <= probability). **This is the CC-modifying instruction** that controls predicated execution. |
| `SFPENCC` | O2 | Directly sets CC.En and CC.Res. InstrMod=0: CC.Res=1, CC.En kept unchanged. Executes on all lanes regardless of current LaneEnabled state. Used here to restore all-lanes-active state after the conditional zeroing. |
| `SFPSTORE` | MR | Stores an LREG value to the DEST register file. `TTI_SFPSTORE(0, 0, 3, 0)` stores LREG0 using IMPLIED format and ADDR_MOD_3. CC-guarded via LaneEnabled. |

### SFPU Register Usage

| Register | Role | Lifetime |
|---|---|---|
| **LREG0** | Working register: holds the loaded DEST value, then the scaled result, then conditionally zeroed | Per-iteration (loaded from DEST, modified, stored back) |
| **LREG1** | Scale factor (FP32 bit pattern loaded via SFPLOADI) | Entire function (loaded once before loop) |
| **LREG2** | Probability threshold (INT32, loaded via SFPLOADI) | Entire function (loaded once before loop, used as source in SFPIADD subtraction) |
| **LREG3** | Random number from PRNG, then result of probability comparison | Per-iteration (generated, sign-cleared, subtracted with probability) |
| **LCONST_0** (RG[9]) | Fixed Constant 1 = 0x0000_0000 (0.0) | Read-only hardware constant; used as the zero value for dropped elements and as the addend in SFPMUL (making it a pure multiply) |
| **DEST register file** | Source of input tile data and destination for output | Addressed via ADDR_MOD_3; auto-incremented by `dst_reg++` between iterations |
| **PRNG (RS[9])** | Hardware pseudo-random number generator | Read via SFPMOV InstrMod=8; advances by 1 step per read; seeded by `init_prng_seed(seed)` during init |
| **CC.En** | Condition Code Enable register | Assumed to be 1 (enabled) as persistent state; never explicitly set by this kernel |
| **CC.Res** | Condition Code Result register | Set per-lane by SFPIADD; reset to 1 by SFPENCC at end of each iteration |

### Address Mode Configuration

The init function `_llk_math_eltwise_unary_sfpu_init_<SfpuType::dropout>()` configures address modes via `eltwise_unary_sfpu_configure_addrmod<SfpuType::dropout>()`. Since `SfpuType::dropout` does not match any of the special-cased types (`topk_local_sort`, `typecast`, `unary_max`, `unary_min`, etc.), only the default `ADDR_MOD_7` is configured:

| ADDR_MOD | srca.incr | srcb.incr | dest.incr | Purpose |
|---|---|---|---|---|
| **ADDR_MOD_7** | 0 | 0 | 0 | Default SFPU address mode -- no auto-increment on any register file address. Used as the base address mode by `math::set_addr_mod_base()`. |

This configuration is **identical across Wormhole and Blackhole** (both `eltwise_unary_sfpu_configure_addrmod` implementations have the same default ADDR_MOD_7 setup and the same special-case logic, neither of which matches `SfpuType::dropout`).

Within the kernel, SFPLOAD and SFPSTORE use `ADDR_MOD_3` (the third argument `sfpu_addr_mode=3`). This selects the address mode register that controls how the DEST register file address is calculated for each load/store. The actual DEST address progression between rows is handled by `sfpi::dst_reg++` at the end of each loop iteration, which increments the DEST row pointer by 1 via the RWC (Register Word Counter) mechanism.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the dropout operation work in TTNN? What is its program factory structure, kernel types, and SFPU implementation details?"
   **Reason**: Initial reconnaissance to understand the operation's architecture before reading source code.
   **Key Findings**: Confirmed dropout is a unary SFPU operation with reader/compute/writer kernels. Two factory variants exist (single-device and mesh). Seed is excluded from program hash. Input must be TILE layout with INTERLEAVED memory.

2. **Query**: "How does split_work_to_cores work in tt-metal? What are core_group_1, core_group_2, and how is work divided among cores?"
   **Reason**: Needed to understand the core distribution strategy used in the program factory.
   **Key Findings**: Returns two core groups -- group 1 gets `ceil(total/cores)` tiles, group 2 gets `floor(total/cores)` tiles. Groups differ by at most 1 work unit. Column-major core ordering.

3. **Query**: "How does the dropout_tile SFPU function work? What does dropout_kernel_init do?"
   **Reason**: Needed to understand the SFPU-level implementation of the dropout operation.
   **Key Findings**: `dropout_kernel_init` seeds the PRNG. `dropout_tile` (internally `_calculate_dropout_`) scales each element by the scale factor, generates a random number via `TTI_SFPMOV`, and zeroes elements where the random number falls below the probability threshold. Implementation is identical across Wormhole and Blackhole architectures.

4. [SFPU] **Query**: "How does the dropout_tile compute API work? What is the call chain from dropout_tile through the LLK layer to the ckernel SFPU implementation?"
   **Reason**: Needed to trace the full abstraction layer call chain from compute API to raw SFPU instructions.
   **Key Findings**: `dropout_tile` -> `SFPU_UNARY_PARAMS_KERNEL_EXTRA_ARGS` macro -> `_llk_math_eltwise_unary_sfpu_params_` -> `calculate_dropout` -> `_calculate_dropout_`. Init path: `dropout_kernel_init` -> `SFPU_ONE_PARAM_KERNEL_INIT` -> `llk_math_eltwise_unary_sfpu_init` + `dropout_init` -> `_init_dropout_` -> `init_prng_seed`.

5. [SFPU] **Query**: "How does the dropout SFPU kernel work in the LLK layer? What SFPU instructions does it use?"
   **Reason**: Needed detailed instruction-level understanding of the `_calculate_dropout_` function before reading source.
   **Key Findings**: Confirmed instructions: SFPLOADI, SFPLOAD, SFPMUL, SFPMOV (InstrMod=8 for PRNG), SFPSETSGN, SFPIADD, SFPENCC, SFPSTORE. PRNG seeded via `init_prng_seed` writing to `PRNG_SEED_Seed_Val_ADDR32` config register.

6. [SFPU] **Query**: "How does SFPI v_if work at the instruction level? How does it enable predicated execution (CC.En)?"
   **Reason**: Needed to understand how CC.En gets set for predicated execution, since the dropout kernel uses raw TTI_ instructions without explicit `v_if`.
   **Key Findings**: `v_if` uses `SFPPUSHC`/`SFPSETCC`/`SFPPOPC` to manage CC stack. `SFPENCC` with specific InstrMod values enables/disables CC.En. The dropout kernel relies on CC.En being persistently 1 rather than explicitly setting it.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/experimental/dropout/device/dropout_device_operation.cpp`
   **Reason**: Understanding validation constraints, factory selection logic, and program hash computation.
   **Key Information**: Factory selected by `use_per_device_seed` flag. Input/output dtypes must match. Seed excluded from hash. INTERLEAVED + TILE layout required for non-sharded tensors.

2. **Source**: `ttnn/cpp/ttnn/operations/experimental/dropout/device/dropout_program_factory.hpp`
   **Reason**: Understanding shared variables structure and factory interfaces.
   **Key Information**: `DropoutSharedVariables` stores kernel handles, core groups, and core count for runtime argument override.

### Confluence References

1. [SFPU] **Page**: Tensix SFPU Instruction Set Architecture (Page ID: 1170505767)
   **Sections consulted**: SFPLOADI, SFPLOAD, SFPMAD/SFPMUL, SFPMOV, SFPSETSGN, SFPIADD, SFPENCC, SFPSTORE, SFPSETCC, SFPCONFIG, Predicated Execution, Condition Code Registers, Local Registers (LREGs), Constant Registers, Register Views (GPR/RS views)
   **Reason**: Needed authoritative ISA specifications for every instruction used in `_calculate_dropout_`, with particular focus on CC manipulation semantics (which instructions set CC.Res vs CC.En), the SFPMOV PRNG access mode (InstrMod=8, RS[9]), and the LaneEnabled formula (`(~CC.En | CC.Res) & ~RowDisable`).
   **Key Findings**:
   - SFPIADD with InstrMod=10 performs Reg/Reg subtraction, sets CC.Res based on result sign, then inverts it. Does NOT set CC.En.
   - SFPENCC(0,0,0,0) sets CC.Res=1 and keeps CC.En unchanged. Executes on all lanes regardless of LaneEnabled.
   - SFPMOV with InstrMod=8 reads from the SFPU Status (RS) register view; RS[9] is the PRNG counter which advances on read.
   - SFPSETSGN with InstrMod=1 takes the sign from Imm12[0] (here 0 = positive).
   - Fixed Constant 1 (RG[9] = LCONST_0) = 0x0000_0000 = 0.0.
   - LaneEnabled = (~CC.En | CC.Res) & ~RowDisable. With CC.En=1, LaneEnabled = CC.Res, enabling per-lane predicated execution.
