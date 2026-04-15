## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the `tanhshrink` operation.

**Mathematical definition**: `tanhshrink(x) = x - tanh(x)`

### Critical Status: Operation is Currently Non-Functional

Before detailing the SFPU kernel architecture, it is important to note that **TANHSHRINK is currently broken in this codebase**. Three independent issues prevent it from compiling or executing:

1. **Dispatch failure**: `UnaryOpType::TANHSHRINK` is not handled in `get_op_init_and_func_default()` in `unary_op_utils.cpp`. When `UnaryProgramFactory::create` calls `get_block_defines()`, it eventually hits `TT_THROW("unexpected op type {}", op_type)` for TANHSHRINK.

2. **Missing tanh SFPU kernel**: Both dedicated compute kernels call `tanh_tile()` / `tanh_tile_init()`, which are declared in `compute_kernel_api.h` as calling `llk_math_eltwise_unary_sfpu_tanh<>()`. This function has **no definition** anywhere in the codebase -- the tanh SFPU kernel was removed.

3. **Missing binary wrapper**: The SFPU binary subtraction path (`tanhshrink_sfpu_kernel.cpp`) calls `sub_binary_tile()` which calls `llk_math_eltwise_binary_sfpu_binop()` which references `ckernel::sfpu::calculate_sfpu_binary` -- a wrapper function that also has **no definition** (only the internal `_calculate_sfpu_binary_` exists).

Despite these issues, the architectural intent is clear from the existing code and can be fully analyzed.

### Unary Dispatch Summary
- **UnaryOpType**: `TANHSHRINK`
- **Compute kernel**: Two dedicated kernel variants exist (neither uses the standard `eltwise_sfpu.cpp` path):
  - `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_kernel.cpp` (FPU-based subtraction variant)
  - `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_sfpu_kernel.cpp` (SFPU-based subtraction variant)
- **SFPU_OP_CHAIN_0 expansion**: Not applicable -- the dedicated kernels do not use `SFPU_OP_CHAIN_0`. Instead, they directly call `tanh_tile()` for the tanh computation followed by subtraction.

#### Approximation Mode Resolution

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(TANHSHRINK)` -- falls through to `default: return false` |
| Template parameter (tanh_tile) | `false` (default) | Both compute kernels call `tanh_tile_init()` and `tanh_tile(0)` without template arguments, using the default `fast_and_approx = false` |
| Effective SFPU path | Would use the non-approximate tanh path | `tanh_tile<false>()` calls `llk_math_eltwise_unary_sfpu_tanh<false, DST_ACCUM_MODE>(idst)` -- but this function is missing |

### SFPU Abstraction Layers

The tanhshrink operation is a two-phase computation: (1) compute tanh(x) via SFPU, (2) subtract tanh(x) from x. The two kernel variants differ in how phase 2 is done.

#### Phase 1: tanh(x) -- SFPU unary tanh (MISSING)

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/compute_kernel_api.h` (lines 154-180) |
| **LLK Dispatch** | **MISSING** -- `llk_math_eltwise_unary_sfpu_tanh<>()` is called but never defined |
| **Core SFPU Implementation** | **MISSING** -- no `ckernel_sfpu_tanh.h` exists in any platform |
| **Parameters Dispatch** | **MISSING** -- no `llk_math_eltwise_unary_sfpu_tanh.h` exists |

#### Phase 2a: x - tanh(x) -- FPU binary subtraction (tanhshrink_kernel.cpp)

This variant uses the FPU math engine (not SFPU) for the subtraction via `binary_dest_reuse_tiles<ELWSUB, DEST_TO_SRCB>`.

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary.h` (lines 211, 248) |
| **LLK Dispatch** | `llk_math_eltwise_binary<ELWSUB, NONE, ...>()` -- standard FPU binary op |
| **Core Implementation** | FPU matrix unit (not SFPU) |
| **Parameters Dispatch** | This level of abstraction doesn't exist (FPU ops don't use SFPU params dispatch) |

#### Phase 2b: x - tanh(x) -- SFPU binary subtraction (tanhshrink_sfpu_kernel.cpp)

This variant uses the SFPU for subtraction via `sub_binary_tile(0, 1, 0)`.

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` (lines 39, 68) |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_binary.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

### Call Chain

The analysis below focuses on the SFPU-based subtraction variant (`tanhshrink_sfpu_kernel.cpp`), since this is the variant that uses SFPU for its second phase.

**Phase 1 (tanh -- would-be call chain, currently broken)**:
1. `tanh_tile<false>(1)` in compute kernel
2. -> `llk_math_eltwise_unary_sfpu_tanh<false, DST_ACCUM_MODE>(1)` in `compute_kernel_api.h` -- **MISSING function**

**Phase 2 (SFPU subtraction call chain)**:
1. `sub_binary_tile(0, 1, 0)` in compute kernel -- subtracts tile at DST[1] from tile at DST[0], stores result in DST[0]
2. -> `llk_math_eltwise_binary_sfpu_binop<APPROX, BinaryOp::SUB>(0, 1, 0)` in `eltwise_binary_sfpu.h`
3. -> `_llk_math_eltwise_binary_sfpu_params_<APPROX>(calculate_sfpu_binary<APPROX, SUB, 8, false>, 0, 1, 0, VectorMode::RC)` in `llk_math_eltwise_binary_sfpu_binop.h`
4. -> Iterates over 4 faces, calling `_calculate_sfpu_binary_<APPROX, SUB, 8>(0, 1, 0)` per face in `llk_math_eltwise_binary_sfpu_params.h`
5. -> Core loop: loads from `dst_reg[in0_offset]` and `dst_reg[in1_offset]`, computes `in0 - in1`, stores to `dst_reg[out_offset]` in `ckernel_sfpu_binary.h`

### Parameters Dispatch Summary

The binary SFPU subtraction dispatch is handled by `_llk_math_eltwise_binary_sfpu_params_`:

- **Vector mode**: `VectorMode::RC` (default) -- all 4 faces of the tile are processed.
- **Operation invocation**: The core function `_calculate_sfpu_binary_<APPROX, BinaryOp::SUB, 8>` is called once per face (4 times total for RC mode). Each call processes 8 iterations (ITERATIONS=8), covering one full 16x16 face.
- **DEST address progression**: Between faces, `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` is called twice per face boundary (in RC mode), advancing the DEST read/write counter by 8+8=16 sfpi rows per face transition. Within the core function, `dst_reg++` advances 1 sfpi row per iteration. This is the standard binary SFPU progression pattern.
- **Init sequence**: `_llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>()` calls `_init_sfpu_config_reg()`, configures `ADDR_MOD_7` (all increments = 0), and resets RWC counters.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`vFloat`, `dst_reg`), so Style A (inline-commented source) is used.

**Compute Kernel (SFPU variant):**

```cpp
// File: ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_sfpu_kernel.cpp

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;
    init_sfpu(cb_input, cb_output);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(cb_output, per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            cb_wait_front(cb_input, 1);
            tile_regs_acquire();

            // Copy input tile x into DST[1] for tanh computation
            copy_tile_init(cb_input);
            copy_tile(cb_input, 0, 1);          // unpack input tile to DST slot 1

            tanh_tile_init();                    // init SFPU for tanh (MISSING: would configure tanh SFPU state)
            tanh_tile(1);                        // DST[1] = tanh(DST[1]) (MISSING: llk function not defined)

            // Copy original x into DST[0] for subtraction
            copy_tile_init(cb_input);
            copy_tile(cb_input, 0, 0);          // unpack input tile to DST slot 0

            sub_binary_tile_init();              // init SFPU binary subtraction
            sub_binary_tile(0, 1, 0);           // DST[0] = DST[0] - DST[1] = x - tanh(x)

            tile_regs_commit();
            tile_regs_wait();

            pack_tile(0, cb_output);             // pack result from DST[0]

            tile_regs_release();

            cb_pop_front(cb_input, 1);
        }
        cb_push_back(cb_output, per_core_block_dim);
    }
}
```

**Compute Kernel (FPU variant):**

```cpp
// File: ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_kernel.cpp

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;
    init_sfpu(cb_input, cb_output);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(cb_output, per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            cb_wait_front(cb_input, 1);
            tile_regs_acquire();

            // Copy input tile x to DST[0], then compute tanh in-place
            copy_tile_init(cb_input);
            copy_tile(cb_input, 0, 0);          // unpack input tile to DST slot 0

            tanh_tile_init();                    // init SFPU for tanh (MISSING)
            tanh_tile(0);                        // DST[0] = tanh(DST[0]) (MISSING)

            // Use FPU binary with dest reuse: load x from CB to SRCB, subtract DST[0] (tanh(x)) from SRCB
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWSUB, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_input);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWSUB, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                cb_input, 0, 0);                // DST[0] = x (from CB) - tanh(x) (from DST via SRCB)

            tile_regs_commit();
            tile_regs_wait();

            pack_tile(0, cb_output);

            tile_regs_release();

            cb_pop_front(cb_input, 1);
        }
        cb_push_back(cb_output, per_core_block_dim);
    }
}
```

**Core SFPU Binary Subtraction Implementation:**

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h
// (Blackhole variant is identical)

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8> // APPROXIMATION_MODE=false, BINOP=SUB, ITERATIONS=8
inline void _calculate_sfpu_binary_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{
    static constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    for (int d = 0; d < ITERATIONS; d++)                                 // 8 iterations per face
    {
        constexpr std::uint32_t dst_tile_size_sfpi = 32;                 // sfpi rows per tile (stride-2 addressing)
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi]; // SFPLOAD: load 32 elements from DST tile 0
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi]; // SFPLOAD: load 32 elements from DST tile 1
        sfpi::vFloat result = 0.0f;

        if constexpr (BINOP == BinaryOp::SUB)                           // compile-time branch for subtraction
        {
            result = in0 - in1;                                          // SFPMAD: result = in0 * 1.0 + (-in1)
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;     // SFPSTORE: write 32 elements to DST tile 0
        sfpi::dst_reg++;                                                 // advance DEST pointer by 1 sfpi row (= 2 physical rows)
    }
}

template <bool APPROXIMATION_MODE /*unused*/, BinaryOp BINOP>
inline void _sfpu_binary_init_() {}                                      // no-op init
```

### SFPU Instructions Used

The following instructions are emitted by the SFPU binary subtraction kernel (`_calculate_sfpu_binary_` with `BinaryOp::SUB`):

| Instruction | SFPI Abstraction | Description |
|-------------|-----------------|-------------|
| `SFPLOAD` | `sfpi::dst_reg[offset]` (read) | Load 32 elements (2 physical DEST rows) from the input tile at the specified offset into an LREG. Called twice per iteration -- once for `in0` (x) and once for `in1` (tanh(x)). |
| `SFPMAD` | `in0 - in1` | Fused multiply-add computing `in0 * 1.0 + (-in1)`. There is no dedicated float subtract instruction; subtraction is implemented as SFPMAD with InstrMod[1]=1 (addend sign inversion). |
| `SFPSTORE` | `sfpi::dst_reg[offset] = result` (write) | Store 32 elements from an LREG back to the output tile position in DEST. |
| `TTI_SETRWC` | (in params dispatch) | Between-face DEST pointer advancement. Called as `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` twice per face boundary to advance by 16 sfpi rows. |
| `TTI_STALLWAIT` | (in init/done) | `STALL_SFPU` at start ensures SFPU is idle before work; `STALL_CFG`/`WAIT_SFPU` (Wormhole only) at end waits for SFPU completion. |

**Note**: The tanh computation (Phase 1) would use additional SFPU instructions, but since the tanh SFPU kernel is missing from the codebase, those cannot be documented. The hardware has `SFPNONLINEAR` with `TANH_MODE=0x5` (Quasar only), or the software implementation would typically use polynomial/LUT-based approximation.

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0-3** | General-purpose LREGs used by SFPI abstractions. `in0` and `in1` are loaded into LREGs from DEST, the subtraction result is computed in an LREG, and the result is stored back to DEST. The exact LREG assignments are determined by the compiler (SFPI register allocator). |
| **DEST tiles** | Two tiles are resident in DEST simultaneously: DST[0] holds the original input `x`, DST[1] holds `tanh(x)`. The subtraction reads from both and writes the result to DST[0]. Each tile occupies 64 physical DEST rows (32 sfpi rows). |
| **dst_tile_size_sfpi** | Constant = 32, used to compute the base offset for each tile in DEST: `dst_index * 32` gives the sfpi row base address. |
| **Programmable constants** | Not used by the binary subtraction kernel. |

### Address Mode Configuration

The SFPU binary subtraction uses `ADDR_MOD_7` configured in `eltwise_binary_sfpu_configure_addrmod<SfpuType::unused>()`:

```
ADDR_MOD_7:
  srca.incr = 0
  srcb.incr = 0
  dest.incr = 0
```

This configuration is identical on both Wormhole B0 and Blackhole. The `dest.incr = 0` means the hardware does not auto-increment the DEST address between SFPU instructions. Instead, DEST addressing is managed explicitly:

- **Within a face**: `dst_reg++` in the SFPI code advances the DEST pointer by 1 sfpi row (= 2 physical rows) per iteration, processing 32 elements per step. After 8 iterations, one full face (256 elements) is processed.
- **Between faces**: `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` is called twice in the params dispatch loop (for `VectorMode::RC`), advancing the DEST read/write counter by 8+8 = 16 sfpi rows to reach the next face.

The `SfpuType::unused` template parameter for the binary init means the `ADDR_MOD_6` (with `dest.incr = 2`) is NOT configured for the generic binary operation -- that path is only for specific ops like `mul_int32`, `max`, `min`, etc.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path and SFPU_OP_CHAIN_0 defines for TANHSHRINK
   **Key Findings**: TANHSHRINK is not in `get_op_init_and_func_default` (would TT_THROW), `get_op_approx_mode` returns false (default), `get_compute_kernel_path` returns `eltwise_sfpu.cpp` (default)

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_kernel.cpp`
   **Reason**: Read the FPU-variant dedicated compute kernel for tanhshrink
   **Key Findings**: Uses `tanh_tile(0)` then `binary_dest_reuse_tiles<ELWSUB, DEST_TO_SRCB>` for x - tanh(x) via FPU binary

3. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_sfpu_kernel.cpp`
   **Reason**: Read the SFPU-variant dedicated compute kernel for tanhshrink
   **Key Findings**: Copies x to DST[1], computes tanh on DST[1], copies x to DST[0], then `sub_binary_tile(0, 1, 0)` for SFPU subtraction

4. **File**: `tt_metal/hw/inc/api/compute/compute_kernel_api.h`
   **Reason**: Trace tanh_tile() API declaration and template defaults
   **Key Findings**: `tanh_tile<false>()` calls `llk_math_eltwise_unary_sfpu_tanh<false, DST_ACCUM_MODE>(idst)` -- function is declared but never defined

5. **File**: `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h`
   **Reason**: Trace sub_binary_tile() API
   **Key Findings**: `sub_binary_tile(idst0, idst1, odst)` calls `llk_math_eltwise_binary_sfpu_binop<APPROX, BinaryOp::SUB>(idst0, idst1, odst)`

6. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h`
   **Reason**: Trace the LLK dispatch for SFPU binary subtraction
   **Key Findings**: Calls `_llk_math_eltwise_binary_sfpu_params_` with `calculate_sfpu_binary<APPROX, BINOP, 8, is_fp32_dest_acc_en>` -- note: wrapper `calculate_sfpu_binary` is not defined, only `_calculate_sfpu_binary_` exists

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h`
   **Reason**: Read the core SFPU binary implementation
   **Key Findings**: `_calculate_sfpu_binary_<APPROX, SUB, 8>` loads two tiles from DEST, computes `in0 - in1` via SFPMAD, stores result back. Identical on Wormhole B0 and Blackhole.

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary_sfpu_params.h`
   **Reason**: Understand the params dispatch (face iteration, vector mode, DEST progression)
   **Key Findings**: VectorMode::RC iterates 4 faces, calling the SFPU function once per face with SETRWC between faces. Identical on Blackhole.

9. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary_sfpu.h`
   **Reason**: Understand address mode configuration for binary SFPU
   **Key Findings**: `ADDR_MOD_7` with all increments = 0. Init calls `_init_sfpu_config_reg()` and `reset_counters(SET_ABD_F)`.

10. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Reference for SFPU hardware model, instruction semantics, tile geometry
    **Key Findings**: SFPNONLINEAR with TANH_MODE=0x5 exists on hardware. Stride-2 addressing model. SFPMAD is used for both add and subtract (no dedicated subtract instruction).
