## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `HARDSWISH`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `hardswish_tile(0)`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(HARDSWISH)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none (no parameterized variant) | `get_op_init_and_func_default()` returns `hardswish_tile_init()` / `hardswish_tile({idst})` with no template parameter override; the API header passes `APPROX` directly |
| Effective SFPU path | `APPROXIMATION_MODE=false` throughout; the kernel has no `if constexpr` branches on `APPROXIMATION_MODE`, so both paths produce identical code | `calculate_hardswish<APPROXIMATION_MODE, ITERATIONS>` in `ckernel_sfpu_hardswish.h` -- no branching on `APPROXIMATION_MODE` |

**Note:** The `APPROX` macro is JIT-generated as `constexpr bool APPROX = false;` in `chlkc_descriptors.h` (from `genfiles.cpp:394`), based on `get_op_approx_mode()` returning `false`. The `calculate_hardswish` template accepts `APPROXIMATION_MODE` but does not branch on it -- the implementation is identical regardless of approximation mode.

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/hardswish.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardswish.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_hardswish.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (from tt_llk submodule) |

### Call Chain
1. **Compute kernel** (`eltwise_sfpu.cpp`): The `SFPU_OP_CHAIN_0` macro expands to `hardswish_tile_init(); hardswish_tile(0);` which calls the API header functions.
2. **API Header** (`hardswish.h`): `hardswish_tile(idst)` calls `llk_math_eltwise_unary_sfpu_hardswish<APPROX>(idst)` on the MATH RISC-V thread. `hardswish_tile_init()` calls `llk_math_eltwise_unary_sfpu_hardswish_init<APPROX>()`.
3. **LLK Dispatch** (`llk_math_eltwise_unary_sfpu_hardswish.h`): The init function calls `llk_math_eltwise_unary_sfpu_init<SfpuType::hardswish, APPROXIMATE>()` to configure address modes and SFPU state. The tile function calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_hardswish<APPROXIMATE, ITERATIONS>, dst_index, vector_mode)`.
4. **Parameters Dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): Sets up DEST addressing for the target tile, stalls until SFPU is ready, then loops over 4 faces (in `VectorMode::RC` mode) calling `calculate_hardswish()` once per face with `SETRWC`/`inc_dst_face_addr` between faces.
5. **Core SFPU** (`ckernel_sfpu_hardswish.h`): `calculate_hardswish<APPROXIMATION_MODE, ITERATIONS>()` executes the SFPU instruction sequence for 8 iterations (one face), computing `hardswish(x) = x * clamp(x/6 + 0.5, 0, 1)`.

### Parameters Dispatch Summary
- **Vector mode**: `VectorMode::RC` (default) -- all 4 faces of the tile are processed (Face 0 through Face 3).
- **Operation invocation**: The dispatch layer calls `calculate_hardswish()` once per face. In `VectorMode::RC`, a loop runs 4 iterations (one per face). Each call to `calculate_hardswish()` internally loops for `ITERATIONS=8` sfpi rows, covering one full face (8 x 32 = 256 elements).
- **DEST address progression**: Standard DEST progression. On Wormhole, `ADDR_MOD_7` is set with `dest.incr=0` (SFPU manages its own address via `dst_reg++` inside the kernel); between faces, `TTI_SETRWC(CR_D, 8, SET_D)` is called twice (advancing by 16 physical DEST rows = 1 face). On Blackhole, `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice for the same effect.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `v_if`/`v_endif`), so Style A (inline-commented source) applies. The Wormhole and Blackhole implementations are identical.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardswish.h

namespace ckernel::sfpu {

// hardswish(x) = x * min(max(x + 3, 0), 6) / 6
//              = x * hardsigmoid(x)
//              = x * clamp(x/6 + 0.5, 0, 1)
// Piecewise:
//   x <= -3  =>  0
//   x >= 3   =>  x
//   else     =>  x * (x/6 + 0.5)
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_hardswish() { // APPROXIMATION_MODE=false, ITERATIONS=8
    constexpr float one_sixth = 1.0f / 6.0f; // 0.16666667f, loaded as SFPLOADI immediate

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];  // SFPLOAD: load 32 elements from current DEST position
        sfpi::vFloat hsigmoid = x * one_sixth + 0.5f;  // SFPMAD: hsigmoid = x * (1/6) + 0.5

        // Clamp hardsigmoid to [0, 1]
        v_if(hsigmoid < 0.0f) { hsigmoid = 0.0f; }  // SFPPUSHC + SFPSETCC(LT0) -> CC-guarded SFPLOADI(0.0) + SFPPOPC
        v_endif;
        v_if(hsigmoid > sfpi::vConst1) { hsigmoid = sfpi::vConst1; }  // SFPPUSHC + comparison with 1.0 via SFPMAD(hsigmoid - 1.0) + SFPSETCC(GTE0) -> CC-guarded load from const reg (1.0) + SFPPOPC
        v_endif;

        sfpi::dst_reg[0] = x * hsigmoid;  // SFPMAD: result = x * hsigmoid + 0.0, then SFPSTORE to DEST
        sfpi::dst_reg++;  // TTI_INCRWC: advance RWC by SFP_DESTREG_STRIDE=2 physical rows
    }
}

}  // namespace ckernel::sfpu
```

### SFPU Instructions Used

| Instruction | Usage in Kernel | Description |
|-------------|----------------|-------------|
| **SFPLOAD** | `sfpi::dst_reg[0]` read | Loads 32 elements from current DEST position into an LREG. Format conversion from DEST format to FP32. |
| **SFPLOADI** | Loading constants `one_sixth`, `0.5f`, `0.0f` | Loads a 16-bit immediate value into an LREG. Two SFPLOADI instructions needed for a full 32-bit FP32 constant. |
| **SFPMAD** | `x * one_sixth + 0.5f`, `x * hsigmoid`, comparison subtraction | Fused multiply-add: `VD = VA * VB + VC`. Used for the multiply-add to compute hardsigmoid, for the final multiply, and for subtraction in the `> vConst1` comparison (hsigmoid * 1.0 - 1.0). |
| **SFPSETCC** | `hsigmoid < 0.0f`, `hsigmoid > vConst1` | Sets per-lane CC.Res based on comparison. For `< 0.0f`, uses `SFPSETCC_MOD1_LREG_LT0` (sign bit test). For `> 1.0`, uses the result of a subtraction followed by `SFPSETCC_MOD1_LREG_GTE0`. |
| **SFPPUSHC** | `v_if(...)` entry | Pushes current CC state onto the 8-entry CC stack, enabling nested conditional regions. |
| **SFPPOPC** | `v_endif` | Pops CC state from the stack, restoring the previous CC enable/result state. |
| **SFPENCC** | `v_if(...)` entry/exit | Enables or disables CC masking. Part of the `v_if` prologue/epilogue to set up and tear down predicated execution. |
| **SFPSTORE** | `sfpi::dst_reg[0] = ...` write | Stores LREG contents to the current DEST position. Format conversion from FP32 to DEST format. |
| **SFPMOV** | Constant register to LREG | Moves `vConst1` (fixed constant register, value 1.0f at CREG_IDX_1=10) to an LREG for the clamping assignment. |
| **TTI_INCRWC** | `sfpi::dst_reg++` | Increments the Read/Write Counter by `SFP_DESTREG_STRIDE=2`, advancing to the next pair of physical DEST rows. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0-LREG3** | General-purpose working registers used by the SFPI compiler for intermediate values (`x`, `hsigmoid`, `one_sixth`, `0.5f`, comparison results). Exact allocation is determined by the SFPI compiler backend. |
| **DEST rows** | Input tile data loaded via SFPLOAD; output written via SFPSTORE. 32 elements (2 physical rows) per sfpi iteration. |
| **Fixed Const Reg (CREG_IDX_1)** | Hardware constant register at index 10, value `0x3F800000` = 1.0f. Accessed as `sfpi::vConst1` for the upper clamp bound. |
| **CC Stack** | Two independent `v_if`/`v_endif` blocks each push/pop one CC stack entry. Maximum CC stack depth used: 1 (no nesting between the two `v_if` blocks). |

### Address Mode Configuration

The address mode is configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType::hardswish>()`, which is called during `_llk_math_eltwise_unary_sfpu_init_<SfpuType::hardswish>()`.

Since `SfpuType::hardswish` does not match any of the special-case `if constexpr` branches (which handle `topk_local_sort`, `typecast`, `unary_max`, `unary_min`, etc.), only the default `ADDR_MOD_7` is configured:

**Wormhole B0 and Blackhole (identical):**

| ADDR_MOD | srca.incr | srcb.incr | dest.incr |
|----------|-----------|-----------|-----------|
| `ADDR_MOD_7` | 0 | 0 | 0 |

The `dest.incr=0` means the hardware does not auto-increment the DEST address after SFPLOAD/SFPSTORE. Instead, DEST address advancement is handled explicitly by the SFPI abstraction layer: `dst_reg++` emits `TTI_INCRWC(0, SFP_DESTREG_STRIDE, 0, 0)` which advances the RWC by 2 physical DEST rows per iteration. Between faces, the parameters dispatch layer calls `SETRWC` (Wormhole) or `inc_dst_addr<8>()` twice (Blackhole) to advance by 16 physical rows (one face).

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 defines, and approximation mode for HARDSWISH
   **Key Findings**: `get_compute_kernel_path()` returns `"eltwise_sfpu.cpp"` (default). `get_op_init_and_func_default()` returns `"hardswish_tile_init();"` and `"hardswish_tile({idst});"`. `get_op_approx_mode()` returns `false` (default). `get_macro_definition()` returns `"SFPU_OP_HARDSWISH_INCLUDE"`.

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/hardswish.h`
   **Reason**: API header that exposes `hardswish_tile()` and `hardswish_tile_init()` to the compute kernel
   **Key Findings**: `hardswish_tile(idst)` calls `llk_math_eltwise_unary_sfpu_hardswish<APPROX>(idst)`. `hardswish_tile_init()` calls `llk_math_eltwise_unary_sfpu_hardswish_init<APPROX>()`. Both are gated by `#ifdef TRISC_MATH`.

3. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardswish.h`
   **Reason**: LLK dispatch layer bridging API to core SFPU kernel
   **Key Findings**: Init calls `llk_math_eltwise_unary_sfpu_init<SfpuType::hardswish, APPROXIMATE>()`. Tile function calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(calculate_hardswish<APPROXIMATE, ITERATIONS>, dst_index, vector_mode)`. Both WH and BH files are identical.

4. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_hardswish.h`
   **Reason**: Core SFPU implementation -- the target of this analysis
   **Key Findings**: Implements `calculate_hardswish()` using SFPI abstractions. Computes `hardswish(x) = x * clamp(x/6 + 0.5, 0, 1)` with two `v_if` clamp blocks. Both WH and BH files are identical.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch function that handles face iteration and DEST addressing
   **Key Findings**: `_llk_math_eltwise_unary_sfpu_params_()` in `VectorMode::RC` loops 4 faces, calling the SFPU functor once per face, with `SETRWC`/`inc_dst_face_addr` between faces. WH variant uses `TTI_SETRWC` directly; BH variant uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()`.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Init and address mode configuration
   **Key Findings**: `eltwise_unary_sfpu_configure_addrmod<SfpuType::hardswish>()` sets only `ADDR_MOD_7` with `{srca.incr=0, srcb.incr=0, dest.incr=0}`. No special-case branches apply to `hardswish`. Both WH and BH are identical for this op.

7. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: SFPU hardware model reference for instruction semantics, register layout, and addressing
   **Key Findings**: Verified tile geometry (32x32 = 1024 elements, 4 faces), SFPU stride-2 model, SFPMAD semantics (used for add and multiply), Fixed Const register 2 = 1.0f (CREG_IDX_1).

8. **File**: `tt_metal/jit_build/genfiles.cpp`
   **Reason**: Understand how `APPROX` macro is JIT-generated
   **Key Findings**: Line 394: `"constexpr bool APPROX = {};\n"` is emitted into `chlkc_descriptors.h` using `desc.get_hlk_math_approx_mode()`, which for HARDSWISH resolves to `false`.

9. **File**: `runtime/sfpi/include/sfpi.h`
   **Reason**: SFPI abstraction definitions for `vFloat`, `dst_reg`, `v_if`/`v_endif`, `vConst1`
   **Key Findings**: `vConst1` maps to `CREG_IDX_1` (fixed constant register, value 1.0f). `dst_reg++` emits `__builtin_rvtt_ttincrwc(0, SFP_DESTREG_STRIDE, 0, 0)`. `v_if`/`v_endif` use `__vCCCtrl` for CC push/pop/condition management.

10. **File**: `runtime/sfpi/include/sfpi_constants.h`
    **Reason**: SFPU constant register index mapping
    **Key Findings**: `CREG_IDX_0 = 9` (value 0.0f), `CREG_IDX_1 = 10` (value 1.0f), `SFP_DESTREG_STRIDE = 2`.
