## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `SWISH`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `swish_tile_init(); swish_tile(0);`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(SWISH)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | `APPROX` (= `false`, from `math_approx_mode`) | `get_op_init_and_func_default()` returns `swish_tile_init()` / `swish_tile(idst)` with no explicit template parameter; the API header uses `APPROX` which is the JIT-generated `constexpr bool APPROX = false` |
| Effective SFPU path | `APPROXIMATION_MODE=false` in `calculate_swish<false, 8>()`. The kernel has no `if constexpr` branch on `APPROXIMATION_MODE` -- the template parameter is accepted but unused. The same code path runs regardless of the approximation mode value. | `ckernel_sfpu_swish.h` line 34-35: `template <bool APPROXIMATION_MODE, int ITERATIONS = 8>` -- parameter is never tested |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/swish.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. **`swish_tile(idst)`** (API header `swish.h`) calls `llk_math_eltwise_unary_sfpu_swish<APPROX>(idst)` inside the `MATH()` macro, which ensures the call runs on the math RISC-V thread (TRISC_MATH).
2. **`llk_math_eltwise_unary_sfpu_swish<APPROX>(dst_index)`** (LLK dispatch `llk_math_eltwise_unary_sfpu_swish.h`) forwards to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_swish<APPROX, 8>, dst_index, VectorMode::RC)`.
3. **`_llk_math_eltwise_unary_sfpu_params_<APPROX>(...)`** (params dispatch `llk_math_eltwise_unary_sfpu_params.h`) sets the DEST write address, stalls until SFPU is ready, then loops over 4 faces in `VectorMode::RC` mode, calling `calculate_swish<false, 8>()` once per face and advancing the DEST pointer between faces via `TTI_SETRWC` (Wormhole) or `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` (Blackhole).
4. **`calculate_swish<false, 8>()`** (core SFPU `ckernel_sfpu_swish.h`) executes the inner SFPU kernel: 8 iterations per face, processing 32 elements per iteration via the stride-2 addressing model.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default). All 4 faces of the tile are processed (face 0, 1, 2, 3), covering the full 32x32 = 1024 elements.
- **Operation invocation**: The params dispatch loops `for (int face = 0; face < 4; face++)`, calling `calculate_swish<false, 8>()` once per face. Each invocation processes ITERATIONS=8 sfpi rows (= 8 x 32 = 256 elements = one full face).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, SETRWC between faces). On Wormhole, two `TTI_SETRWC` instructions with increment 8 advance by one face stride (16 physical DEST rows = 8 sfpi rows) between faces. On Blackhole, `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice for the same effect.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `v_if`/`v_endif`, `sfpi::abs`), so Style A (inline-commented source) is used.

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h

// swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
// Implementation notes, see the original file for more details

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_swish() { // APPROXIMATION_MODE=false, ITERATIONS=8
    // Polynomial coefficients for sigmoid(t) over [0, 2.5]
    // Fitted to minimize max error at t = 0, 0.5, 1.0, 1.5, 2.0, 2.5
    constexpr float c1 = 0.2533f;
    constexpr float c2 = -0.01479f;
    constexpr float c3 = -0.00747f;

    // Linear segment coefficients for [2.5, 5.0]
    constexpr float lin_slope = 0.0276f;
    constexpr float lin_offset = 0.855f;

    // Breakpoints
    constexpr float bp1 = 2.5f;
    constexpr float bp2 = 5.0f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST row pair into LREG

        // Compute sigmoid(|x|) using degree-3 polynomial for |x| <= 2.5 (Horner's method)
        sfpi::vFloat ax = sfpi::abs(x); // SFPABS: clear sign bit of x
        sfpi::vFloat sig_pos = 0.5f + ax * (c1 + ax * (c2 + ax * c3)); // SFPMAD chain: Horner evaluation then add 0.5

        // Override with linear segment for 2.5 < |x| <= 5.0
        v_if(ax > bp1) { sig_pos = ax * lin_slope + lin_offset; } // SFPPUSHC + SFPMAD(subtract) + SFPSETCC for comparison; guarded SFPMAD for linear eval
        v_endif; // SFPPOPC

        // Saturate to 1.0 for |x| > 5.0
        v_if(ax > bp2) { sig_pos = sfpi::vConst1; } // SFPPUSHC + comparison; guarded SFPLOADI or MOV from constant register (vConst1 = Fixed Const 2 = 1.0)
        v_endif; // SFPPOPC

        // For x < 0: sigmoid(x) = 1 - sigmoid(|x|)
        v_if(x < 0.0f) { sig_pos = sfpi::vConst1 - sig_pos; } // SFPPUSHC + sign-bit test via SFPSETCC; guarded SFPMAD (1.0 * 1.0 + (-sig_pos))
        v_endif; // SFPPOPC

        // swish(x) = x * sigmoid(x)
        sfpi::dst_reg[0] = x * sig_pos; // SFPMAD(x * sig_pos + 0.0) then SFPSTORE: write result back to DEST
        sfpi::dst_reg++; // Advance DEST pointer by 1 sfpi row (= 2 physical DEST rows = 32 elements)
    }
}
```

### SFPU Instructions Used

The following SFPU instructions are emitted by the SFPI compiler from the high-level abstractions used in `calculate_swish`. The kernel itself contains no raw `TTI_`/`TT_` instructions -- all instructions below are inferred from the SFPI-to-instruction lowering rules.

| Instruction | Source Abstraction | Description |
|-------------|-------------------|-------------|
| `SFPLOAD` | `sfpi::dst_reg[0]` (read) | Load 32 elements from the current DEST row pair into an LREG. Format conversion from DEST format to FP32 happens automatically. |
| `SFPSTORE` | `sfpi::dst_reg[0] = ...` (write) | Store 32 elements from an LREG back to the current DEST row pair. Format conversion from FP32 to DEST format. |
| `SFPABS` | `sfpi::abs(x)` | Compute absolute value by clearing the sign bit of each FP32 element. |
| `SFPMAD` | `vFloat * vFloat`, `vFloat + vFloat`, Horner polynomial, float comparisons | Fused multiply-add: `VD = VA * VB + VC`. Used for all floating-point arithmetic including additions (as `a * 1.0 + b`), multiplications (as `a * b + 0.0`), subtractions (as `a * 1.0 + (-b)` via sign inversion), and the comparison-by-subtraction pattern in `v_if`. |
| `SFPLOADI` | Float literal constants (`0.2533f`, `2.5f`, etc.) | Load a 16-bit immediate (bfloat16 or float16) into an LREG. Used to materialize the polynomial coefficients and breakpoint constants. Two `SFPLOADI` instructions needed for full 32-bit FP constants. |
| `SFPSETCC` | `v_if(ax > bp1)`, `v_if(x < 0.0f)` | Set per-lane CC.Res based on a comparison result. For float comparisons, the SFPI compiler first computes `a - b` via SFPMAD, then uses SFPSETCC to test the sign of the result (LT0 or GTE0 mode). |
| `SFPENCC` | `v_if` / `v_endif` internal CC management | Enable or disable per-lane condition code masking. Used internally by the SFPI v_if/v_endif framework. |
| `SFPCOMPC` | Implicit in `v_if` CC management | Complement CC.Res. May be used internally during CC state transitions, though `calculate_swish` has no `v_else` branches. |
| `SFPPUSHC` | `v_if(...)` | Push current CC state onto the per-lane CC stack before evaluating a new condition. Each `v_if` generates one push. |
| `SFPPOPC` | `v_endif` | Pop CC state from the per-lane CC stack, restoring the prior condition. Each `v_endif` generates one pop. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST rows** (via `dst_reg`) | Input/output tile data. Each iteration loads from and stores to the same DEST row pair (`dst_reg[0]`), then advances to the next row pair (`dst_reg++`). |
| **LREGs (LREG0-LREG7)** | General-purpose vector registers used by the SFPI compiler for intermediate values. The compiler allocates LREGs for: `x` (original input), `ax` (absolute value), `sig_pos` (sigmoid approximation), intermediate polynomial terms, and comparison temporaries. Exact allocation is determined by the compiler's register allocator. |
| **Constant registers** | `vConst1` (Fixed Const 2 = 1.0f at CREG index 10) is used for the saturation value and the `1 - sigmoid(|x|)` negation. `vConst0` (Fixed Const 1 = 0.0f at CREG index 9) is implicitly used as the addend in multiply-only SFPMAD instructions (`a * b + 0.0`). |
| **CC stack** | The 8-entry per-lane CC stack is used by three `v_if`/`v_endif` pairs (max depth 1 at any point, no nesting). Each `v_if` pushes one entry, each `v_endif` pops it. |

### Address Mode Configuration

The init function `llk_math_eltwise_unary_sfpu_init<SfpuType::swish, APPROXIMATE>()` calls `eltwise_unary_sfpu_configure_addrmod<SfpuType::swish>()`. Since `SfpuType::swish` does not match any special-case `if constexpr` branch in the address mode configuration, only the default `ADDR_MOD_7` is configured:

| Address Mode | Field | Value | Notes |
|-------------|-------|-------|-------|
| **ADDR_MOD_7** | `srca.incr` | 0 | No SrcA auto-increment |
| | `srcb.incr` | 0 | No SrcB auto-increment |
| | `dest.incr` | 0 | No DEST auto-increment from address mode |

This configuration is **identical on both Wormhole and Blackhole** for `SfpuType::swish`.

DEST address advancement is handled entirely by the SFPI `dst_reg++` abstraction (which compiles to explicit DEST pointer manipulation within the loop) and by the params dispatch layer's `TTI_SETRWC` (Wormhole) / `math::inc_dst_addr<8>()` (Blackhole) calls between faces. The zero-increment ADDR_MOD_7 ensures the hardware auto-increment does not interfere with the software-managed DEST addressing.

No additional address modes (`ADDR_MOD_6`, etc.) are configured for this operation, unlike operations such as `typecast`, `unary_max`, or `topk_local_sort` that require special DEST auto-increment behavior.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 expansion, and approximation mode for SWISH
   **Key Findings**: SWISH uses `SFPU_OP_SWISH_INCLUDE` macro, compute kernel `eltwise_sfpu.cpp`, init=`swish_tile_init()`, func=`swish_tile(idst)`, non-parameterized, `get_op_approx_mode()` returns false (default case)

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/swish.h`
   **Reason**: API header exposing tile-level SFPU call
   **Key Findings**: `swish_tile(idst)` calls `llk_math_eltwise_unary_sfpu_swish<APPROX>(idst)`, `swish_tile_init()` calls `llk_math_eltwise_unary_sfpu_swish_init<APPROX>()`

3. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h`
   **Reason**: LLK dispatch layer bridging API to core SFPU kernel
   **Key Findings**: Forwards to `_llk_math_eltwise_unary_sfpu_params_` with `calculate_swish<APPROXIMATE, ITERATIONS>` as the callable. Identical on both architectures.

4. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h`
   **Reason**: Core SFPU kernel implementation
   **Key Findings**: Uses SFPI abstractions. Implements swish via piecewise sigmoid approximation: degree-3 polynomial for |x|<=2.5, linear interpolation for 2.5<|x|<=5.0, saturation to 1.0 for |x|>5.0. Then applies symmetry (sigmoid(x) = 1 - sigmoid(|x|) for x<0) and multiplies by x. Identical on both architectures. APPROXIMATION_MODE parameter is unused.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch controlling face iteration and DEST addressing
   **Key Findings**: VectorMode::RC processes all 4 faces. WH uses TTI_SETRWC for face advancement, BH uses math::inc_dst_addr<8>() called twice.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Init function and address mode configuration
   **Key Findings**: SfpuType::swish has no special address mode case; only ADDR_MOD_7 (all-zero increments) is configured.

7. **File**: `tt_metal/jit_build/genfiles.cpp`
   **Reason**: Determine how `APPROX` constant is generated
   **Key Findings**: `emit_math_scalar_descriptors()` generates `constexpr bool APPROX = {math_approx_mode};` in `chlkc_descriptors.h`, fed by `get_op_approx_mode()` result.

8. **File**: `runtime/sfpi/include/sfpi.h` and `runtime/sfpi/include/sfpi_lib.h`
   **Reason**: Understand SFPI abstraction-to-instruction mappings
   **Key Findings**: `sfpi::abs()` maps to `SFPABS`, float comparisons use `__builtin_rvtt_sfpxfcmps` (lowers to SFPMAD subtract + SFPSETCC), `v_if`/`v_endif` use SFPPUSHC/SFPPOPC, `vConst1` is Fixed Const 2 (1.0f) at CREG index 10.

9. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: Authoritative SFPU hardware model for instruction semantics, register layout, and addressing
   **Key Findings**: Stride-2 model (32 elements per sfpi row), ITERATIONS=8 per face, SFPMAD used for all float arithmetic, SFPLOAD/SFPSTORE for DEST access, CC stack for conditional execution.
