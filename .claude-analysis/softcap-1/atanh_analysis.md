## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `ATANH`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `atanh_tile(0)`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(ATANH)` in `unary_op_utils.cpp` -- currently returns `false` for all ops (switch has only a `default: return false` case) |
| Template parameter (SFPU_OP_CHAIN) | none | `get_op_init_and_func_default()` -- non-parameterized case: `atanh_tile_init()` / `atanh_tile(idst)` with no explicit template arguments; the `APPROX` compile-time define (set from `math_approx_mode`) is forwarded as the template argument |
| Effective SFPU path | `APPROXIMATION_MODE=false`; however, the `calculate_atanh` implementation does not branch on `APPROXIMATION_MODE` -- the same code path executes regardless of this flag | The template parameter is accepted but unused in `ckernel_sfpu_atanh.h` |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/atanh.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. The compute kernel invokes `atanh_tile(idst)` (from the `SFPU_OP_CHAIN_0` macro expansion).
2. `atanh_tile(idst)` in `atanh.h` calls `llk_math_eltwise_unary_sfpu_atanh<APPROX>(idst)` within the `MATH((...))` wrapper, which gates execution to the math thread.
3. `llk_math_eltwise_unary_sfpu_atanh<APPROXIMATE, 8>(dst_index, VectorMode::RC)` in `llk_math_eltwise_unary_sfpu_atanh.h` calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_atanh<APPROXIMATE, 8>, dst_index, VectorMode::RC)`.
4. `_llk_math_eltwise_unary_sfpu_params_` in `llk_math_eltwise_unary_sfpu_params.h` sets up the DEST write address, stalls until SFPU is ready, then loops over 4 faces calling `calculate_atanh<false, 8>()` once per face with SETRWC/inc_dst_addr between faces.
5. `calculate_atanh<false, 8>()` in `ckernel_sfpu_atanh.h` executes the core SFPU math: 8 iterations per face, processing 32 elements per iteration via SFPI vector operations.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default). All 4 faces of the tile are processed, covering all 1024 elements.
- **Operation invocation**: The params dispatch loops over 4 faces, calling `calculate_atanh<false, 8>()` once per face. Each invocation runs an internal loop of 8 iterations (ITERATIONS=8), processing one sfpi row (32 elements) per iteration for a total of 256 elements per face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, SETRWC between faces). On Wormhole, the params dispatch uses `TTI_SETRWC` directly to advance by 16 physical rows (2x `TTI_SETRWC` with increment 8) between faces. On Blackhole, it calls `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which does `math::inc_dst_addr<8>()` twice. The address mode is `ADDR_MOD_7` on both architectures, configured with all-zero increments (srca=0, srcb=0, dest=0).

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::vInt`, `sfpi::dst_reg`, `sfpi::exexp`, `sfpi::setexp`, `sfpi::int32_to_float`), so Style A (inline-commented source) is used.

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h

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
    constexpr float c3 = 0x2.44734p-4f;  // ~0.1416
    constexpr float ln2 = 0.6931471805599453f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST row pair

        // a = 1 + x, b = 1 - x (both positive for |x| < 1)
        sfpi::vFloat a = x + sfpi::vConst1;   // SFPMAD: x * 1.0 + 1.0
        sfpi::vFloat b = -x + sfpi::vConst1;  // SFPMAD: x * (-1.0) + 1.0

        // ln(a): decompose a = 2^ea * ma, ma in [1, 2)
        sfpi::vInt ea = sfpi::exexp(a);         // SFPEXEXP with DEBIAS: extract biased exponent, subtract 127
        sfpi::vFloat ma = sfpi::setexp(a, 127); // SFPSETEXP_I: force exponent to 127, yielding mantissa in [1, 2)
        // P(ma) = c0 + ma*(c1 + ma*(c2 + ma*c3))  -- Horner evaluation
        sfpi::vFloat pa = ma * c3 + sfpi::vConstFloatPrgm2; // SFPMAD: ma * c3 + c2
        pa = pa * ma + sfpi::vConstFloatPrgm1;              // SFPMAD: pa * ma + c1
        pa = pa * ma + sfpi::vConstFloatPrgm0;              // SFPMAD: pa * ma + c0
        sfpi::vFloat ln_a = sfpi::int32_to_float(ea, 0) * ln2 + pa; // SFPCAST(int->float) then SFPMAD: ea_float * ln2 + pa

        // ln(b): decompose b = 2^eb * mb, mb in [1, 2)
        sfpi::vInt eb = sfpi::exexp(b);         // SFPEXEXP with DEBIAS
        sfpi::vFloat mb = sfpi::setexp(b, 127); // SFPSETEXP_I
        sfpi::vFloat pb = mb * c3 + sfpi::vConstFloatPrgm2; // SFPMAD: mb * c3 + c2
        pb = pb * mb + sfpi::vConstFloatPrgm1;               // SFPMAD: pb * mb + c1
        pb = pb * mb + sfpi::vConstFloatPrgm0;               // SFPMAD: pb * mb + c0
        sfpi::vFloat ln_b = sfpi::int32_to_float(eb, 0) * ln2 + pb; // SFPCAST then SFPMAD

        // atanh(x) = 0.5 * (ln(a) - ln(b))
        sfpi::vFloat result = (ln_a - ln_b) * 0.5f; // SFPMAD (subtract) then SFPMAD (multiply by 0.5)

        sfpi::dst_reg[0] = result; // SFPSTORE: write 32 elements back to current DEST row pair
        sfpi::dst_reg++;           // Advance DEST pointer by 1 sfpi row (2 physical rows, 32 elements)
    }
}

template <bool APPROXIMATION_MODE>
inline void atanh_init() { // APPROXIMATION_MODE=false
    // Cubic polynomial coefficients for ln(m) on [1, 2)
    sfpi::vConstFloatPrgm0 = -0x1.952992p+0f;  // c0 ~ -1.5828; loaded via SFPCONFIG into Prog Const register 2
    sfpi::vConstFloatPrgm1 = 0x2.4f5388p+0f;   // c1 ~  2.3110; loaded via SFPCONFIG into Prog Const register 3
    sfpi::vConstFloatPrgm2 = -0xd.e712ap-4f;   // c2 ~ -0.8691; loaded via SFPCONFIG into Prog Const register 4
}
```

### SFPU Instructions Used

| Instruction | SFPI Abstraction | Description |
|-------------|-----------------|-------------|
| `SFPLOAD` | `sfpi::dst_reg[0]` (read) | Load 32 elements from the current DEST row pair into an LREG |
| `SFPSTORE` | `sfpi::dst_reg[0] = result` (write) | Store 32 elements from an LREG back to the current DEST row pair |
| `SFPMAD` | `vFloat + vFloat`, `vFloat * float + vFloat` | Fused multiply-add; used for all float addition (`a * 1.0 + b`) and multiply-add operations. The Horner polynomial evaluation is a chain of SFPMADs. Also used for the final `(ln_a - ln_b) * 0.5` computation. |
| `SFPEXEXP` | `sfpi::exexp(v)` | Extract the biased exponent from an IEEE 754 float and subtract the bias (127), producing a signed integer exponent. Called with `SFPEXEXP_MOD1_DEBIAS`. |
| `SFPSETEXP` | `sfpi::setexp(v, 127)` | Set the exponent field of a float to a given value (127 = bias, producing a mantissa in [1, 2)). Uses the immediate form `SFPSETEXP_I`. |
| `SFPCAST` | `sfpi::int32_to_float(ea, 0)` | Convert a signed 32-bit integer to FP32. Called with `SFPCAST_MOD1_INT32_TO_FP32_RNE` (round-to-nearest-even, since `round_mode=0`). |
| `SFPCONFIG` | `sfpi::vConstFloatPrgm{0,1,2} = ...` (in `atanh_init`) | Program the SFPU constant registers with the cubic polynomial coefficients for `ln(m)` approximation. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST rows** | Source and destination for tile data. Each iteration reads 32 elements (2 physical rows x 16 cols) via `SFPLOAD` and writes 32 elements back via `SFPSTORE`. The DEST pointer advances by 1 sfpi row per iteration. |
| **LREGs (0-7)** | General-purpose 32-bit registers used as temporaries by the compiler. The SFPI compiler allocates LREGs for `x`, `a`, `b`, `ea`, `ma`, `pa`, `ln_a`, `eb`, `mb`, `pb`, `ln_b`, and `result`. Since there are many live variables, the compiler must spill and reuse LREGs across the computation. Exact LREG allocation is determined by the SFPI compiler backend. |
| **Prog Const 2 (vConstFloatPrgm0)** | Stores c0 = -0x1.952992p+0f (~-1.5828), the degree-0 coefficient of the cubic minimax polynomial for `ln(m)`. |
| **Prog Const 3 (vConstFloatPrgm1)** | Stores c1 = 0x2.4f5388p+0f (~2.3110), the degree-1 coefficient. |
| **Prog Const 4 (vConstFloatPrgm2)** | Stores c2 = -0xd.e712ap-4f (~-0.8691), the degree-2 coefficient. |
| **Fixed Const 2 (vConst1)** | Hardware-fixed constant = 1.0f, used for computing `a = x + 1` and `b = -x + 1`. |

### Address Mode Configuration

The `atanh` operation uses `ADDR_MOD_7` on both Wormhole and Blackhole. This is the default address mode set by `eltwise_unary_sfpu_configure_addrmod<SfpuType::atanh>()` -- since `SfpuType::atanh` does not match any special-cased `if constexpr` branch, only the default `ADDR_MOD_7` is configured.

| Field | Value | Description |
|-------|-------|-------------|
| `srca.incr` | 0 | No auto-increment for SrcA |
| `srcb.incr` | 0 | No auto-increment for SrcB |
| `dest.incr` | 0 | No auto-increment for DEST |

This all-zero configuration means the SFPU does not auto-increment DEST addresses between instructions. Instead, DEST address progression is managed explicitly: within a face, `dst_reg++` (compiled to an SFPU pointer increment) advances by 1 sfpi row per loop iteration; between faces, the params dispatch layer advances the DEST write address by 16 physical rows (via `TTI_SETRWC` on Wormhole, or `math::inc_dst_addr<8>()` x2 on Blackhole).

The configuration is identical on Wormhole and Blackhole.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 expansion, approximation mode for ATANH
   **Key Findings**: ATANH uses default `eltwise_sfpu.cpp` compute kernel; `get_op_approx_mode()` returns false (default); `get_op_init_and_func_default()` maps to `atanh_tile_init()` / `atanh_tile(idst)`; include guard is `SFPU_OP_ATANH_INCLUDE`

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/atanh.h`
   **Reason**: API header defining `atanh_tile()` and `atanh_tile_init()`
   **Key Findings**: `atanh_tile(idst)` calls `llk_math_eltwise_unary_sfpu_atanh<APPROX>(idst)`; `atanh_tile_init()` calls `llk_math_eltwise_unary_sfpu_atanh_init<APPROX>()`

3. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h`
   **Reason**: LLK dispatch layer bridging API to ckernel
   **Key Findings**: Calls `_llk_math_eltwise_unary_sfpu_params_` with `calculate_atanh<APPROXIMATE, 8>` and `VectorMode::RC`; init calls `llk_math_eltwise_unary_sfpu_init<SfpuType::atanh, APPROXIMATE>` with `sfpu::atanh_init<APPROXIMATE>`

4. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h`
   **Reason**: Core SFPU implementation of atanh
   **Key Findings**: Uses IEEE 754 decomposition to compute ln(1+x) and ln(1-x) separately via cubic minimax polynomial in Horner form, then combines as 0.5*(ln(1+x)-ln(1-x)). SFPI-based (vFloat, dst_reg, exexp, setexp, int32_to_float). WH and BH implementations are identical.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch layer controlling face iteration and DEST address progression
   **Key Findings**: VectorMode::RC loops 4 faces, calling sfpu_func once per face. WH uses TTI_SETRWC for face advancement; BH uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()`.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Address mode configuration and SFPU init
   **Key Findings**: `eltwise_unary_sfpu_configure_addrmod<SfpuType::atanh>()` sets ADDR_MOD_7 with all-zero increments (default path). SfpuType::atanh is not special-cased.

7. **File**: `runtime/sfpi/include/sfpi_lib.h`
   **Reason**: SFPI intrinsic-to-instruction mappings
   **Key Findings**: `exexp()` -> `__builtin_rvtt_sfpexexp` (SFPEXEXP), `setexp(v, imm)` -> `__builtin_rvtt_sfpsetexp_i` (SFPSETEXP), `int32_to_float(v, 0)` -> `__builtin_rvtt_sfpcast` with SFPCAST_MOD1_INT32_TO_FP32_RNE

8. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: SFPU architecture reference for instruction semantics, register layout, addressing model
   **Key Findings**: Stride-2 model (dst_reg++ = 2 physical rows = 32 elements), ITERATIONS=8 per face, SFPMAD for all float arithmetic, programmable constant registers Prog Const 2-4 map to vConstFloatPrgm0-2
