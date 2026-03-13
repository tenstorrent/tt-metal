# Full Analysis of the EXP Operation

## 1. Overview

The EXP operation computes `e^x` element-wise on a tensor. It flows through three layers:
1. **Host-side program factory** — sets up kernels, circular buffers, and work distribution
2. **Device-side dataflow kernels** — reader and writer move tiles between DRAM and L1
3. **Device-side compute kernel** — executes the SFPU exponential on each tile

## 2. Program Factory (`unary_program_factory.cpp`)

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

### Entry Point

`UnaryProgramFactory::create()` (line 24) builds the `tt::tt_metal::Program` object.

### Circular Buffers

Three CBs are configured on all active cores:

| CB Index | Role | Size | Data Format |
|---|---|---|---|
| `c_0` | Input | 2 pages | Input dtype (e.g. BFloat16) |
| `c_2` | Output | 2 pages | Output dtype |
| `c_1` | Temp (only for HARDSHRINK/CBRT/LOGIT) | 2 pages | Input dtype |

For EXP, only `c_0` and `c_2` are created (no temp buffer needed).

### Work Distribution (lines 46-49)

```cpp
auto [num_cores, all_cores, core_group_1, core_group_2,
      num_pages_per_core_group_1, num_pages_per_core_group_2] =
    tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_pages);
```

Tiles are evenly divided across available Tensix cores. Two core groups handle the remainder: group 1 gets `N+1` tiles, group 2 gets `N` tiles.

### Kernel Creation

**Reader kernel** (lines 91-95): `reader_unary_interleaved_start_id.cpp`
- Compile-time args: `TensorAccessorArgs` for the source buffer
- Runtime args: `{src_buffer_address, num_pages_per_core, start_page_id}`

**Writer kernel** (lines 97-101): `writer_unary_interleaved_start_id.cpp`
- Compile-time args: `{output_cb_index}` + `TensorAccessorArgs` for destination buffer
- Runtime args: `{dst_buffer_address, num_pages_per_core, start_page_id}`

**Compute kernel** (lines 155-168): Path is resolved via `get_compute_kernel_path()`:
- EXP falls into the `default` case, selecting **`eltwise_sfpu.cpp`**
- Compile-time args: `{per_core_block_cnt, 1}` (1 tile per block)
- Runtime args: `{packed_scalar1, packed_scalar2}` — for EXP with a parameter, `param0` controls approximation mode
- `ComputeConfig`:
  - `math_fidelity = HiFi4`
  - `math_approx_mode = false` (EXP returns `false` from `get_op_approx_mode` at line 777-780)
  - `fp32_dest_acc_en` and `bfp8_pack_precise` come from operation attributes

### Preprocessor Defines (lines 116, 244-247)

`get_block_defines()` generates two critical defines for the compute kernel.

For **parametrized** EXP (with `param0`):
```cpp
#define SFPU_OP_EXP_INCLUDE 1
#define SFPU_OP_CHAIN_0  exp_tile_init<param0>(); exp_tile<param0>(0);
```

For **non-parametrized** EXP:
```cpp
#define SFPU_OP_EXP_INCLUDE 1
#define SFPU_OP_CHAIN_0  exp_tile_init(); exp_tile(0);
```

The parameter `param0` controls whether the approximation mode is used (1) or the precise mode (0).

### Program Caching (lines 218-247)

`override_runtime_arguments()` allows re-use of the compiled program with new buffer addresses, avoiding recompilation when only tensor addresses change.

## 3. Dataflow Kernels

### Reader (`reader_unary_interleaved_start_id.cpp`)

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`

```
for each page [start_id, start_id + num_pages):
    cb_reserve_back(c_0, 1)       // wait for space in input CB
    noc_async_read_page(i, ...)    // DMA read from DRAM to L1
    noc_async_read_barrier()       // wait for DMA complete
    cb_push_back(c_0, 1)          // signal compute kernel: data ready
```

Uses `TensorAccessor` for address calculation (handles interleaved bank mapping).

### Writer (`writer_unary_interleaved_start_id.cpp`)

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

```
for each page [start_id, start_id + num_pages):
    cb_wait_front(c_2, 1)         // wait for compute to produce a result
    noc_async_write_page(i, ...)   // DMA write from L1 to DRAM
    noc_async_writes_flushed()     // ensure write committed
    cb_pop_front(c_2, 1)          // free output CB slot
noc_async_write_barrier()          // final barrier
```

## 4. Compute Kernel (`eltwise_sfpu.cpp`)

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`

The generic SFPU kernel template:

```cpp
void kernel_main() {
    init_sfpu(c_0, c_2);                    // Initialize unpacker & packer
    for (block = 0; block < per_core_block_cnt; block++) {
        cb_reserve_back(c_2, 1);             // Reserve output space
        for (tile = 0; tile < per_core_block_dim; tile++) {
            tile_regs_acquire();              // Lock DST registers
            cb_wait_front(c_0, 1);            // Wait for input tile from reader
            copy_tile(c_0, 0, 0);             // Unpack tile into DST[0]
            SFPU_OP_CHAIN_0                   // <<< exp_tile_init(); exp_tile(0); >>>
            tile_regs_commit();               // Signal pack
            tile_regs_wait();                 // Wait for pack ready
            pack_tile(0, c_2);                // Pack DST[0] -> output CB
            cb_pop_front(c_0, 1);             // Free input CB slot
            tile_regs_release();              // Release DST registers
        }
        cb_push_back(c_2, 1);               // Signal writer: output ready
    }
}
```

The key call chain is:
1. `copy_tile` — unpacks one 32x32 tile from CB `c_0` into DST register 0
2. `exp_tile_init()` / `exp_tile(0)` — runs the SFPU exponential on DST[0]
3. `pack_tile` — packs DST[0] into CB `c_2`

## 5. SFPU Exponential Implementation (`ckernel_sfpu_exp.h`)

**File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h`

The LLK layer implements EXP with two modes plus an optimized FAST_APPROX variant.

### 5.1 Approximation Mode (`APPROXIMATION_MODE = true`)

Based on Schraudolph's "A Fast, Compact Approximation of the Exponential Function". Exploits the linearity of IEEE 754 bit-patterns with log2:

```
i = A * x + (B - C)
```

Where:
- `A = 256 / ln(2) ~ 369.33`
- `B - C ~ 32500.82` (bias minus error-correction constant)

Steps in `_calculate_exponential_body_<true>` (lines 48-85):
1. Multiply input by `1/ln(2)` and convert to 7.3 fixed-point
2. Subtract the FP exponent encoding constant `C23_73`
3. Add IEEE 754 bias (127 << 3)
4. Shift left by 7 bits to place integer part into the FP32 exponent field

The piecewise version `_calculate_exponential_piecewise_` (lines 103-159) adds clamping:
- Input >= 89 -> saturate to infinity
- Input < -42 -> saturate to 0

### 5.2 FAST_APPROX Mode (Macro-based)

For maximum throughput, uses SFPU macro instructions and replay buffers.

**Initialization** (`_init_exponential_<true, true>`, lines 338-639):
1. Loads constants into SFPU local registers: `A` -> LREG[12], `B-C` -> LREG[13], threshold -> LREG[14]
2. Programs macro instruction registers via backdoor loads:
   - Macro 4: `SFPSWAP` — clamps input against -88.5
   - Macro 5: `SFPMAD` — compute `A * x + (B-C)`
   - Macro 6: `SFP_STOCH_RND` — convert FP32 to INT16
   - Macro 7: `SFPSHFT` — shift left 15 bits to place result into FP32 exponent
3. Programs macro sequence registers for two pipelines:
   - Sequence 1: Load -> SWAP -> Store (input sanitization)
   - Sequence 0: Load -> MAD -> ROUND -> SHIFT -> Store (exponential calculation)
4. Records 32-instruction pattern into replay buffer

**Execution** (`_calculate_exponential_<true, ..., true>`, lines 161-332):
- For `CLAMP_NEGATIVE` variant: 16 `SFPLOADMACRO` pairs execute both sanitization (Sequence 1) and computation (Sequence 0) across 16 dest offsets covering one 32x32 tile
- For 8-element replay variant: `lltt::replay(0, 16)` replays the recorded buffer, achieving ~2.5 cycles/element
- For 32-element replay variant: two `lltt::replay(0, 32)` calls, achieving ~2.125 cycles/element

### 5.3 Precise Mode (`APPROXIMATION_MODE = false`)

Uses series expansion in Horner form via `_sfpu_exp_()` (lines 18-46):

1. **Extract exponent**: `exp = exexp(val)`
2. **Clamp mantissa**: if exponent >= 0, set exponent to 126 (forcing range [-1, 0))
3. **Horner polynomial**: `val = val * (val * 0.8373 + 0.8633) + 1.0`
4. **Iterative squaring**: for original exponents >= 0, square the result up to 8 times, narrowing the predicate (`v_and(exp >= 0)`) on each iteration
5. **Negative inputs**: compute exp(|x|) then take reciprocal via `_sfpu_reciprocal_<2>()`

Initialization for precise mode (`_init_exponential_<false>`, line 649) sets up the reciprocal function's constants.

## 6. Data Flow Summary

```
DRAM --NoC0--> [Reader RISC-V]
                    |
                    v
               CB c_0 (L1 SRAM, 2 tiles)
                    |
                    v
              [Compute RISC-V]
              +-- Unpack tile to DST
              +-- SFPU: exp_tile()
              +-- Pack DST to CB c_2
                    |
                    v
               CB c_2 (L1 SRAM, 2 tiles)
                    |
                    v
              [Writer RISC-V] --NoC1--> DRAM
```

All three kernels (reader, compute, writer) run concurrently on the same Tensix core. The circular buffers provide producer-consumer synchronization:
- Reader produces into `c_0`, compute consumes from `c_0`
- Compute produces into `c_2`, writer consumes from `c_2`
- Double-buffering (2 pages per CB) allows overlap between DMA transfers and computation

## 7. Key Files

| File | Role |
|---|---|
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp` | Host-side program construction |
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` | Define generation, kernel path selection |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` | Generic SFPU compute kernel |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | Reader dataflow kernel |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | Writer dataflow kernel |
| `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h` | SFPU exp algorithm (LLK layer) |
