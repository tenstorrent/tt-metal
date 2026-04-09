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
| Template parameter (SFPU_OP_CHAIN) | none (uses `APPROX` directly) | `get_op_init_and_func_default()` returns `"atanh_tile_init();"` / `"atanh_tile(0);"` -- no parameterized template argument; the API header uses `<APPROX>` which resolves to the `constexpr bool APPROX` emitted by `genfiles.cpp` from `math_approx_mode` |
| Effective SFPU path | `APPROXIMATION_MODE = false` | The core SFPU function `calculate_atanh<APPROXIMATION_MODE, ITERATIONS>` receives `false` but does not branch on `APPROXIMATION_MODE` -- the same code path executes regardless |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/atanh.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. **`atanh_tile(idst)`** (API header `atanh.h:27`) calls `llk_math_eltwise_unary_sfpu_atanh<APPROX>(idst)` via the `MATH()` macro which gates execution to the math RISC-V thread.
2. **`llk_math_eltwise_unary_sfpu_atanh<APPROXIMATE, 8>(dst_index, VectorMode::RC)`** (LLK dispatch `llk_math_eltwise_unary_sfpu_atanh.h:19-22`) calls the generic `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>()` passing `ckernel::sfpu::calculate_atanh<APPROXIMATE, 8>` as the callable along with `dst_index` and `vector_mode`.
3. **`_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(sfpu_func, dst_index, vector_mode)`** (params dispatch `llk_math_eltwise_unary_sfpu_params.h`) sets the DEST write address, stalls until SFPU is available, then loops over 4 faces (for `VectorMode::RC`) calling `calculate_atanh<false, 8>()` once per face, with `SETRWC` instructions between faces to advance the DEST pointer.
4. **`calculate_atanh<false, 8>()`** (core SFPU `ckernel_sfpu_atanh.h:22-57`) executes 8 iterations per face, processing 32 elements each iteration using SFPI vector intrinsics.

**Init path**: `atanh_tile_init()` -> `llk_math_eltwise_unary_sfpu_atanh_init<APPROX>()` -> `llk_math_eltwise_unary_sfpu_init<SfpuType::atanh, APPROXIMATE>(sfpu::atanh_init<APPROXIMATE>)` -> `_llk_math_eltwise_unary_sfpu_init_<SfpuType::atanh>()` (configures ADDR_MOD, resets counters, inits SFPU config reg) then calls `atanh_init<false>()` which loads the three polynomial coefficients into programmable constant registers.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default) -- all 4 faces of the tile are processed, covering the full 32x32 = 1024 elements.
- **Operation invocation**: The params dispatch loops `for (int face = 0; face < 4; face++)`, calling `calculate_atanh<false, 8>()` once per face. Each invocation processes 8 SFPI iterations x 32 elements = 256 elements = one full face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). On Wormhole, the params dispatch uses `TTI_SETRWC` with `CR_D, 8` twice between faces (advancing 16 physical DEST rows = 1 face stride). On Blackhole, the equivalent `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice.

### Annotated SFPU Kernel Source

This kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::vInt`, `sfpi::dst_reg`, `sfpi::exexp`, `sfpi::setexp`, `sfpi::int32_to_float`), so Style A (inline-commented source code) is used.

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h

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
    constexpr float c3 = 0x2.44734p-4f;  // ~0.1416, top Horner coefficient
    constexpr float ln2 = 0.6931471805599453f; // natural log of 2

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) { // 8 iterations per face, 32 elements each
        sfpi::vFloat x = sfpi::dst_reg[0]; // SFPLOAD: load current DEST row pair into LREG

        // a = 1 + x, b = 1 - x (both positive for |x| < 1)
        sfpi::vFloat a = x + sfpi::vConst1;   // SFPMAD: x * 1.0 + 1.0
        sfpi::vFloat b = -x + sfpi::vConst1;  // SFPMAD: x * (-1.0) + 1.0

        // ln(a): decompose a = 2^ea * ma, ma in [1, 2)
        sfpi::vInt ea = sfpi::exexp(a);         // SFPEXEXP: extract biased exponent of a, debias to signed int
        sfpi::vFloat ma = sfpi::setexp(a, 127); // SFPSETEXP: replace exponent with 127 (bias), giving mantissa in [1,2)
        // P(ma) = c0 + ma*(c1 + ma*(c2 + ma*c3)) -- Horner evaluation
        sfpi::vFloat pa = ma * c3 + sfpi::vConstFloatPrgm2;  // SFPMAD: ma * c3 + c2
        pa = pa * ma + sfpi::vConstFloatPrgm1;                // SFPMAD: pa * ma + c1
        pa = pa * ma + sfpi::vConstFloatPrgm0;                // SFPMAD: pa * ma + c0
        sfpi::vFloat ln_a = sfpi::int32_to_float(ea, 0) * ln2 + pa; // SFPCAST + SFPMAD + SFPMAD

        // ln(b): decompose b = 2^eb * mb, mb in [1, 2)
        sfpi::vInt eb = sfpi::exexp(b);         // SFPEXEXP: extract biased exponent of b, debias to signed int
        sfpi::vFloat mb = sfpi::setexp(b, 127); // SFPSETEXP: replace exponent with 127
        sfpi::vFloat pb = mb * c3 + sfpi::vConstFloatPrgm2;  // SFPMAD: mb * c3 + c2
        pb = pb * mb + sfpi::vConstFloatPrgm1;                // SFPMAD: pb * mb + c1
        pb = pb * mb + sfpi::vConstFloatPrgm0;                // SFPMAD: pb * mb + c0
        sfpi::vFloat ln_b = sfpi::int32_to_float(eb, 0) * ln2 + pb; // SFPCAST + SFPMAD + SFPMAD

        // atanh(x) = 0.5 * (ln(a) - ln(b))
        sfpi::vFloat result = (ln_a - ln_b) * 0.5f; // SFPMAD (subtract) + SFPMAD (scale by 0.5)

        sfpi::dst_reg[0] = result; // SFPSTORE: write result back to DEST row pair
        sfpi::dst_reg++;           // advance DEST pointer by 1 sfpi row (2 physical rows, 32 elements)
    }
}

template <bool APPROXIMATION_MODE>
inline void atanh_init() { // APPROXIMATION_MODE=false
    // Cubic polynomial coefficients for ln(m) on [1, 2)
    sfpi::vConstFloatPrgm0 = -0x1.952992p+0f;  // c0 ~ -1.5828; SFPLOADI to LREG[4]
    sfpi::vConstFloatPrgm1 = 0x2.4f5388p+0f;   // c1 ~  2.3110; SFPLOADI to LREG[5]
    sfpi::vConstFloatPrgm2 = -0xd.e712ap-4f;   // c2 ~ -0.8691; SFPLOADI to LREG[6]
}

}  // namespace ckernel::sfpu
```

### SFPU Instructions Used

| Instruction | SFPI Abstraction | Description | Count per Iteration |
|-------------|-----------------|-------------|---------------------|
| **SFPLOAD** | `dst_reg[0]` (read) | Load 32 elements from current DEST row pair into an LREG | 1 |
| **SFPSTORE** | `dst_reg[0] = result` | Store 32 elements from LREG back to current DEST row pair | 1 |
| **SFPMAD** | `vFloat + vFloat`, `vFloat * scalar + vFloat` | Fused multiply-add: `a * b + c`. Used for all floating-point addition, subtraction, and Horner polynomial evaluation. There is no dedicated float add -- `a + b` compiles to `a * 1.0 + b` via SFPMAD. | ~12 (2 for a/b computation, 6 for two Horner chains, 2 for `e*ln2+p`, 2 for final subtract+scale) |
| **SFPEXEXP** | `sfpi::exexp(v)` | Extract the IEEE 754 exponent from a float, debias it (subtract 127), and return as integer. Used to decompose `a` and `b` into `2^e * m` form. | 2 |
| **SFPSETEXP** | `sfpi::setexp(v, 127)` | Replace the exponent field of a float with the given value (127 = bias, yielding mantissa in [1,2)). Used to isolate the mantissa for polynomial evaluation. | 2 |
| **SFPCAST** | `sfpi::int32_to_float(ea, 0)` | Convert integer exponent to float for the `e * ln(2)` multiplication. Round mode 0 = truncate. | 2 |
| **SFPLOADI** | `vConstFloatPrgmN = ...` (in init) | Load immediate constant into a programmable LREG. Used in `atanh_init()` to set polynomial coefficients c0, c1, c2 into LREG[4], LREG[5], LREG[6]. | 3 (init only) |

### SFPU Register Usage

| Register | Purpose | Details |
|----------|---------|---------|
| **DEST rows** (via `dst_reg`) | Input/output tile data | Each `dst_reg[0]` access reads/writes 32 elements (2 physical DEST rows x 16 elements/row). The pointer auto-increments via `dst_reg++`. |
| **LREGs (temporary)** | Intermediate computation | The SFPI compiler allocates LREGs 0-3 for temporary values (`x`, `a`, `b`, `ea`, `ma`, `pa`, `ln_a`, `eb`, `mb`, `pb`, `ln_b`, `result`). The compiler manages spilling and reuse across the ~12 SFPMAD operations per iteration. With 4 available temporary LREGs and many intermediates, the compiler must carefully schedule register allocation. |
| **LREG[4]** (`vConstFloatPrgm0`) | Polynomial coefficient c0 | Loaded in `atanh_init()` with `-0x1.952992p+0f` (~-1.5828). Used in the final step of both Horner chains. |
| **LREG[5]** (`vConstFloatPrgm1`) | Polynomial coefficient c1 | Loaded in `atanh_init()` with `0x2.4f5388p+0f` (~2.3110). Used in the middle step of both Horner chains. |
| **LREG[6]** (`vConstFloatPrgm2`) | Polynomial coefficient c2 | Loaded in `atanh_init()` with `-0xd.e712ap-4f` (~-0.8691). Used in the first step of both Horner chains. |
| **LREG[7]** (`vConst1`) | Constant 1.0f | Hardware-provided constant. Used to compute `a = x + 1` and `b = -x + 1`. |

### Address Mode Configuration

The SFPU init function `_llk_math_eltwise_unary_sfpu_init_<SfpuType::atanh>()` calls `eltwise_unary_sfpu_configure_addrmod<SfpuType::atanh>()`. Since `SfpuType::atanh` does not match any of the special-cased types (`topk_local_sort`, `typecast`, `unary_max`, `unary_min`, etc.), only the default `ADDR_MOD_7` is configured:

```
ADDR_MOD_7: { srca.incr = 0, srcb.incr = 0, dest.incr = 0 }
```

This is identical for both Wormhole and Blackhole. The zero-increment ADDR_MOD means the hardware auto-increment is not used for DEST addressing -- instead, the SFPI `dst_reg++` abstraction handles address progression within the kernel loop, and `TTI_SETRWC` (Wormhole) / `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` (Blackhole) handles inter-face advancement in the params dispatch.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine the compute kernel path, SFPU_OP_CHAIN_0 expansion, and approximation mode for ATANH
   **Key Findings**: ATANH uses `eltwise_sfpu.cpp`, expands to `atanh_tile_init(); atanh_tile(0);`, macro guard is `SFPU_OP_ATANH_INCLUDE`, `get_op_approx_mode()` returns `false` (default case)

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/atanh.h`
   **Reason**: API header exposing the tile-level SFPU call
   **Key Findings**: `atanh_tile(idst)` calls `llk_math_eltwise_unary_sfpu_atanh<APPROX>(idst)`, `atanh_tile_init()` calls `llk_math_eltwise_unary_sfpu_atanh_init<APPROX>()`

3. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h`
   **Reason**: LLK dispatch layer bridging API to core SFPU function
   **Key Findings**: Uses `_llk_math_eltwise_unary_sfpu_params_` generic dispatch with `calculate_atanh<APPROXIMATE, 8>` as callable, `VectorMode::RC` default, `ITERATIONS=8`

4. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h`
   **Reason**: Core SFPU implementation -- the primary analysis target
   **Key Findings**: SFPI-based kernel using IEEE 754 decomposition for ln() with cubic minimax polynomial approximation. Computes `atanh(x) = 0.5 * (ln(1+x) - ln(1-x))`. Uses `exexp`, `setexp`, `int32_to_float` intrinsics and Horner-form polynomial evaluation. WH and BH implementations are identical.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch controlling per-face iteration and DEST address progression
   **Key Findings**: Standard RC dispatch -- 4 faces, `calculate_atanh` called once per face, `SETRWC`/`inc_dst_addr` between faces. WH uses `TTI_SETRWC` directly; BH uses helper function. Both advance 16 physical DEST rows between faces.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: SFPU init and ADDR_MOD configuration
   **Key Findings**: `SfpuType::atanh` only triggers the default `ADDR_MOD_7` configuration (all zero increments). No special ADDR_MOD branches apply.

7. **File**: `tt_metal/jit_build/genfiles.cpp`
   **Reason**: Understand how `APPROX` compile-time constant is generated
   **Key Findings**: `emit_math_scalar_descriptors` emits `constexpr bool APPROX = {math_approx_mode};` into the generated `chlkc_descriptors.h` header

8. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: Authoritative hardware model for SFPU tile geometry, stride-2 addressing, and instruction semantics
   **Key Findings**: Confirmed ITERATIONS=8 per face, stride-2 model (32 elements per dst_reg access), SFPMAD used for all float adds, SFPLOAD/SFPSTORE for DEST access

9. **File**: `docs/source/tt-metalium/tt_metal/apis/kernel_apis/sfpu/llk.rst`
   **Reason**: SFPI library function documentation
   **Key Findings**: `exexp()` extracts and debiases IEEE 754 exponent; `setexp()` replaces exponent preserving sign/mantissa; `int32_to_float()` converts integer to float with configurable rounding; `vConstFloatPrgm0/1/2` are programmable constant registers
