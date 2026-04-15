## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `ATANH`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `atanh_tile_init(); atanh_tile(0);`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(ATANH)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none (uses default) | `get_op_init_and_func_default()` returns `"atanh_tile_init();"` and `"atanh_tile(0);"` -- no parameterized template argument; `atanh_tile_init()` calls `llk_math_eltwise_unary_sfpu_atanh_init<APPROX>()` where `APPROX` resolves to the `math_approx_mode` compile define = `false` |
| Effective SFPU path | `APPROXIMATION_MODE=false` for both `calculate_atanh` and `atanh_init` | The template parameter is unused in both functions -- `calculate_atanh` and `atanh_init` have no `if constexpr` branches on `APPROXIMATION_MODE`, so the code path is identical regardless |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/atanh.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. **`atanh_tile(idst)`** (API header `atanh.h`) -- guarded by `TRISC_MATH`, calls `llk_math_eltwise_unary_sfpu_atanh<APPROX>(idst)`.
2. **`llk_math_eltwise_unary_sfpu_atanh<APPROXIMATE>(dst_index, vector_mode=RC)`** (LLK dispatch) -- calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_atanh<APPROXIMATE, 8>, dst_index, VectorMode::RC)`.
3. **`_llk_math_eltwise_unary_sfpu_params_`** (parameters dispatch) -- sets DEST write address, stalls for SFPU availability, then loops over 4 faces (RC mode), calling `calculate_atanh<false, 8>()` once per face with `SETRWC`/`inc_dst_addr` between faces.
4. **`calculate_atanh<false, 8>()`** (core SFPU implementation) -- executes 8 iterations per face, each processing 32 elements via SFPI abstractions (dst_reg load/store, exexp, setexp, int32_to_float, Horner polynomial via vFloat MAD chain).

Similarly, `atanh_tile_init()` calls `llk_math_eltwise_unary_sfpu_atanh_init<APPROX>()`, which calls `llk_math_eltwise_unary_sfpu_init<SfpuType::atanh, APPROXIMATE>(sfpu::atanh_init<APPROXIMATE>)`. This first runs `_llk_math_eltwise_unary_sfpu_init_<SfpuType::atanh>()` (configures SFPU config reg, address modes, resets counters), then calls `atanh_init<false>()` which programs the three cubic polynomial coefficients into the SFPU constant registers.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default) -- all 4 faces of the tile are processed (Face 0, 1, 2, 3), covering all 1024 elements.
- **Operation invocation**: In RC mode, the dispatch loops `for (int face = 0; face < 4; face++)`, calling `calculate_atanh<false, 8>()` once per face. Each call processes 8 SFPU iterations (one full face of 256 elements). Between faces, on Wormhole `TTI_SETRWC` advances the DEST pointer by 16 physical rows (two calls of `inc_dst_addr<8>`); on Blackhole `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` does the same.
- **DEST address progression**: Standard DEST progression. The `SfpuType::atanh` does not match any special `if constexpr` branch in `eltwise_unary_sfpu_configure_addrmod`, so only `ADDR_MOD_7` is configured (with `srca.incr=0, srcb.incr=0, dest.incr=0`). Within a face, `dst_reg++` advances 1 sfpi row (= 2 physical DEST rows = 32 elements) per iteration, covering ITERATIONS=8 per face. Between faces, SETRWC advances by face stride (16 physical rows).

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::vInt`, `sfpi::dst_reg`, `sfpi::exexp`, `sfpi::setexp`, etc.), so Style A (inline-commented source code) is used. The Wormhole and Blackhole implementations are identical.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h

namespace ckernel::sfpu {

// atanh(x) = 0.5 * ln((1+x)/(1-x)) = 0.5 * (ln(1+x) - ln(1-x))
// Valid for |x| < 1.
//
// ln(y) is computed via IEEE 754 decomposition:
//   y = 2^e * m, where m in [1, 2)
//   ln(y) = e * ln(2) + P(m)
// where P(m) is a cubic minimax polynomial approximation for ln(m) on [1, 2).
// Coefficients are from the rpow scalar log2 precomputation (Horner form):
//   P(m) = c0 + m * (c1 + m * (c2 + m * c3))
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_atanh() { // APPROXIMATION_MODE=false, ITERATIONS=8
    constexpr float c3 = 0x2.44734p-4f;  // ~0.1416, leading Horner coefficient
    constexpr float ln2 = 0.6931471805599453f;  // natural log of 2

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {  // 8 iterations per face
        sfpi::vFloat x = sfpi::dst_reg[0];  // SFPLOAD: load 32 elements from DEST

        // a = 1 + x, b = 1 - x (both positive for |x| < 1)
        sfpi::vFloat a = x + sfpi::vConst1;   // SFPMAD: x * 1.0 + 1.0
        sfpi::vFloat b = -x + sfpi::vConst1;  // SFPMAD: x * (-1.0) + 1.0

        // ln(a): decompose a = 2^ea * ma, ma in [1, 2)
        sfpi::vInt ea = sfpi::exexp(a);        // SFPEXEXP (debiased): extract exponent of a as signed int
        sfpi::vFloat ma = sfpi::setexp(a, 127); // SFPSETEXP: set exponent to 127 (bias), normalizing mantissa to [1,2)
        // P(ma) via Horner's method: c0 + ma*(c1 + ma*(c2 + ma*c3))
        sfpi::vFloat pa = ma * c3 + sfpi::vConstFloatPrgm2;  // SFPMAD: ma * c3 + c2
        pa = pa * ma + sfpi::vConstFloatPrgm1;                // SFPMAD: pa * ma + c1
        pa = pa * ma + sfpi::vConstFloatPrgm0;                // SFPMAD: pa * ma + c0
        sfpi::vFloat ln_a = sfpi::int32_to_float(ea, 0) * ln2 + pa; // SFPCAST(ea->float) then SFPMAD: float(ea)*ln2 + pa

        // ln(b): decompose b = 2^eb * mb, mb in [1, 2)
        sfpi::vInt eb = sfpi::exexp(b);        // SFPEXEXP (debiased): extract exponent of b
        sfpi::vFloat mb = sfpi::setexp(b, 127); // SFPSETEXP: normalize mantissa
        sfpi::vFloat pb = mb * c3 + sfpi::vConstFloatPrgm2;  // SFPMAD: mb * c3 + c2
        pb = pb * mb + sfpi::vConstFloatPrgm1;                // SFPMAD: pb * mb + c1
        pb = pb * mb + sfpi::vConstFloatPrgm0;                // SFPMAD: pb * mb + c0
        sfpi::vFloat ln_b = sfpi::int32_to_float(eb, 0) * ln2 + pb; // SFPCAST + SFPMAD

        // atanh(x) = 0.5 * (ln(a) - ln(b))
        sfpi::vFloat result = (ln_a - ln_b) * 0.5f;  // SFPMAD (subtract) then SFPMAD (multiply by 0.5)

        sfpi::dst_reg[0] = result;  // SFPSTORE: write 32 elements back to DEST
        sfpi::dst_reg++;            // advance DEST pointer by 1 sfpi row (2 physical rows)
    }
}

template <bool APPROXIMATION_MODE>
inline void atanh_init() { // APPROXIMATION_MODE=false
    // Cubic polynomial coefficients for ln(m) on [1, 2), programmed via SFPCONFIG
    sfpi::vConstFloatPrgm0 = -0x1.952992p+0f;  // c0 ~ -1.5828
    sfpi::vConstFloatPrgm1 = 0x2.4f5388p+0f;   // c1 ~  2.3110
    sfpi::vConstFloatPrgm2 = -0xd.e712ap-4f;   // c2 ~ -0.8691
}

}  // namespace ckernel::sfpu
```

### SFPU Instructions Used

| Instruction | SFPI Abstraction | Usage in Kernel |
|-------------|-----------------|-----------------|
| **SFPLOAD** | `dst_reg[0]` (read) | Loads 32 elements from the current DEST row pair into an LREG for processing |
| **SFPSTORE** | `dst_reg[0] = result` (write) | Stores the computed 32-element result back to the current DEST row pair |
| **SFPMAD** | `vFloat + vFloat`, `vFloat * float` | All floating-point additions and multiplications compile to SFPMAD (fused multiply-add). Used for: computing `a=x+1`, `b=1-x`, Horner polynomial evaluation (6 MADs for two ln() calls), exponent-to-ln conversion (`float(e)*ln2 + poly`), final subtraction and scaling (`(ln_a - ln_b) * 0.5`) |
| **SFPEXEXP** | `sfpi::exexp(v)` | Extracts the debiased exponent field from an IEEE 754 float as a signed integer. Called twice per iteration (once for `a`, once for `b`) |
| **SFPSETEXP** | `sfpi::setexp(v, 127)` | Sets the exponent field to 127 (IEEE bias), effectively normalizing the mantissa to [1, 2). Called twice per iteration |
| **SFPCAST** | `sfpi::int32_to_float(vInt, 0)` | Converts the integer exponent to FP32 using round-to-nearest-even mode (mode=0). Called twice per iteration |
| **SFPCONFIG** | `vConstFloatPrgm{0,1,2} = ...` (in `atanh_init`) | Programs three SFPU constant registers with the cubic minimax polynomial coefficients for ln(m) approximation |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST rows** | Input tile data is read from DEST via `dst_reg[0]` (SFPLOAD) and results are written back via `dst_reg[0] = result` (SFPSTORE). Stride-2 addressing means each access covers 2 physical rows (32 elements). |
| **LREGs (general purpose)** | Multiple LREGs are used as temporaries to hold intermediate values: `x`, `a`, `b`, `ea`, `eb`, `ma`, `mb`, `pa`, `pb`, `ln_a`, `ln_b`, `result`. The compiler allocates these across LREG0-LREG7 as needed. The high register pressure (many live variables per iteration) may cause register spills. |
| **Constant Register PrgmConst2** (`vConstFloatPrgm0`, index `CREG_IDX_PRGM1`) | Holds `c0 = -0x1.952992p+0f` (~-1.5828), the degree-0 coefficient of the ln(m) minimax polynomial |
| **Constant Register PrgmConst3** (`vConstFloatPrgm1`, index `CREG_IDX_PRGM2`) | Holds `c1 = 0x2.4f5388p+0f` (~2.3110), the degree-1 coefficient |
| **Constant Register PrgmConst4** (`vConstFloatPrgm2`, index `CREG_IDX_PRGM3`) | Holds `c2 = -0xd.e712ap-4f` (~-0.8691), the degree-2 coefficient |
| **Fixed Constant Register 2** (`vConst1`) | Provides the value 1.0f, used to compute `a = x + 1` and `b = -x + 1` |

### Address Mode Configuration

The `SfpuType::atanh` does not match any special-cased `if constexpr` branch in `eltwise_unary_sfpu_configure_addrmod`, so only the default address mode is configured:

**`ADDR_MOD_7`** (configured for all unary SFPU ops):
- `srca.incr = 0`
- `srcb.incr = 0`
- `dest.incr = 0`

This means the hardware does not auto-increment the DEST address between SFPU instructions. Instead, DEST address advancement is handled explicitly:
- **Within a face**: `dst_reg++` in the SFPI code emits instructions that advance the SFPU's internal row pointer by 1 sfpi row (2 physical DEST rows) per iteration.
- **Between faces**: The parameters dispatch uses `TTI_SETRWC` (Wormhole) or `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` (Blackhole) to advance by 16 physical rows (1 face height), via two calls of `inc_dst_addr<8>`.

This configuration is identical on both Wormhole B0 and Blackhole.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: To determine compute kernel path, SFPU_OP_CHAIN_0 expansion, and approximation mode for ATANH
   **Key Findings**: ATANH uses `eltwise_sfpu.cpp`, expands to `atanh_tile_init(); atanh_tile(0);`, `get_op_approx_mode` returns false (default case), macro is `SFPU_OP_ATANH_INCLUDE`

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/atanh.h`
   **Reason**: API header defining `atanh_tile()` and `atanh_tile_init()`
   **Key Findings**: `atanh_tile(idst)` calls `llk_math_eltwise_unary_sfpu_atanh<APPROX>(idst)`, `atanh_tile_init()` calls `llk_math_eltwise_unary_sfpu_atanh_init<APPROX>()`

3. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h`
   **Reason**: LLK dispatch layer bridging API to ckernel
   **Key Findings**: Init calls `llk_math_eltwise_unary_sfpu_init<SfpuType::atanh, APPROXIMATE>(sfpu::atanh_init<APPROXIMATE>)`. Compute calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(calculate_atanh<APPROXIMATE, 8>, dst_index, VectorMode::RC)`. WH and BH implementations are identical.

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h`
   **Reason**: Core SFPU implementation of atanh
   **Key Findings**: Computes atanh(x) = 0.5 * (ln(1+x) - ln(1-x)) using IEEE 754 decomposition for ln(). Each ln() uses SFPEXEXP to extract exponent, SFPSETEXP to normalize mantissa to [1,2), a cubic Horner polynomial for ln(mantissa), and SFPCAST for int-to-float conversion of the exponent. WH and BH files are identical.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch layer controlling face iteration and DEST address progression
   **Key Findings**: RC mode loops 4 faces, calling sfpu_func once per face, with TTI_SETRWC between faces to advance DEST pointer by 16 physical rows

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Address mode configuration for SFPU ops
   **Key Findings**: ADDR_MOD_7 set with all increments=0 for standard unary ops. SfpuType::atanh has no special address mode branch.

7. **File**: `runtime/sfpi/include/sfpi_lib.h`
   **Reason**: SFPI intrinsic-to-instruction mappings
   **Key Findings**: `exexp()` maps to `SFPEXEXP` with DEBIAS mode, `setexp(v, imm)` maps to `SFPSETEXP`, `int32_to_float(v, 0)` maps to `SFPCAST` with RNE mode

8. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: SFPU hardware model reference for instruction semantics and register layout
   **Key Findings**: SFPMAD is used for all float add/multiply, SFPEXEXP extracts exponent, SFPSETEXP sets exponent field, SFPCAST converts int-to-float, SFPCONFIG programs constant registers

9. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
   **Reason**: Compute kernel confirming the tile-level dispatch pattern
   **Key Findings**: Standard eltwise_sfpu pattern with `SFPU_OP_CHAIN_0` macro expansion between `copy_tile` and `tile_regs_commit`

10. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`
    **Reason**: To confirm how `math_approx_mode` is resolved and passed to the compute kernel
    **Key Findings**: `math_approx_mode` is set via `get_op_approx_mode()` for all ops in the chain (returns false for ATANH), passed as `.math_approx_mode` in `ComputeConfig`
