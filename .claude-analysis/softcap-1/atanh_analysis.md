## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `ATANH`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `atanh_tile(0)`

#### Approximation Mode Resolution
There are TWO independent controls for approximation.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(ATANH)` in `unary_op_utils.cpp` -- currently returns `false` for all ops (switch has only a `default: return false` case) |
| Template parameter (SFPU_OP_CHAIN) | none | `get_op_init_and_func_default()` -- non-parameterized: `atanh_tile_init()` / `atanh_tile(idst)` with default template args, so `APPROX` is used directly from JIT-generated `constexpr bool APPROX = false` |
| Effective SFPU path | `APPROXIMATION_MODE=false`, but the kernel does not branch on `APPROXIMATION_MODE` at all -- the same code path executes regardless | `calculate_atanh` is not conditioned on `APPROXIMATION_MODE`; the template parameter is accepted but unused |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/atanh.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. **Compute kernel** (`eltwise_sfpu.cpp`): `SFPU_OP_CHAIN_0` expands to `atanh_tile_init(); atanh_tile(0);`.
2. **API header** (`atanh.h`): `atanh_tile(idst)` calls `llk_math_eltwise_unary_sfpu_atanh<APPROX>(idst)` on the MATH thread via the `MATH(...)` macro.
3. **LLK dispatch** (`llk_math_eltwise_unary_sfpu_atanh.h`): `llk_math_eltwise_unary_sfpu_atanh<APPROXIMATE, 8>(dst_index, VectorMode::RC)` calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_atanh<APPROXIMATE, 8>, dst_index, VectorMode::RC)`.
4. **Parameters dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): Iterates over 4 faces in `VectorMode::RC`, calling `calculate_atanh<false, 8>()` once per face, with `SETRWC` between faces to advance the DEST write address.
5. **Core SFPU** (`ckernel_sfpu_atanh.h`): `calculate_atanh<false, 8>()` processes 8 sfpi rows per face using IEEE 754 decomposition and a cubic minimax polynomial to compute `atanh(x) = 0.5 * (ln(1+x) - ln(1-x))`.

For init: `atanh_tile_init()` calls `llk_math_eltwise_unary_sfpu_atanh_init<APPROX>()`, which calls `llk_math_eltwise_unary_sfpu_init<SfpuType::atanh, APPROXIMATE>(sfpu::atanh_init<APPROXIMATE>)`. This configures `ADDR_MOD_7` and programs three constant registers with the cubic polynomial coefficients via `SFPCONFIG`.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default) -- all 4 faces of the tile are processed.
- **Operation invocation**: In the `VectorMode::RC` branch, the params dispatch loops `for (int face = 0; face < 4; face++)`, calling `calculate_atanh<false, 8>()` once per face. Each call processes 8 sfpi rows (ITERATIONS=8), covering one full 16x16 face (256 elements).
- **DEST address progression**: Standard DEST progression. On Wormhole, `ADDR_MOD_7` is used (configured with all zero increments -- `srca=0, srcb=0, dest=0`); within a face, `dst_reg++` in the SFPI kernel advances 1 sfpi row (= 2 physical DEST rows) per iteration, covering 32 elements per step. Between faces, `TTI_SETRWC(CR_D, 8) x2` advances the DEST write counter by 16 physical rows (= 1 face stride). On Blackhole, the same pattern applies via `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which calls `math::inc_dst_addr<8>()` twice.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`vFloat`, `vInt`, `dst_reg`, `exexp`, `setexp`, `int32_to_float`), so Style A (inline-commented source) is used.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h
// (Blackhole implementation is identical)

namespace ckernel::sfpu {

// atanh(x) = 0.5 * ln((1+x)/(1-x)) = 0.5 * (ln(1+x) - ln(1-x))
// Valid for |x| < 1.
//
// ln(y) is computed via IEEE 754 decomposition:
//   y = 2^e * m, where m in [1, 2)
//   ln(y) = e * ln(2) + P(m)
// where P(m) is a cubic minimax polynomial approximation for ln(m) on [1, 2).
// Implementation notes, see the original file for more details
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_atanh() { // APPROXIMATION_MODE=false, ITERATIONS=8
    constexpr float c3 = 0x2.44734p-4f;  // ~0.1416, leading coefficient of cubic
    constexpr float ln2 = 0.6931471805599453f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST row pair

        // a = 1 + x, b = 1 - x (both positive for |x| < 1)
        sfpi::vFloat a = x + sfpi::vConst1;  // SFPMAD: x * 1.0 + 1.0 (vConst1 = Fixed Const 2 = 1.0)
        sfpi::vFloat b = -x + sfpi::vConst1; // SFPMAD: x * (-1.0) + 1.0 (sign-inverted multiply)

        // ln(a): decompose a = 2^ea * ma, ma in [1, 2)
        sfpi::vInt ea = sfpi::exexp(a);       // SFPEXEXP with DEBIAS: extract biased exponent, subtract 127
        sfpi::vFloat ma = sfpi::setexp(a, 127); // SFPSETEXP: force exponent to 127 -> mantissa in [1,2)
        // P(ma) = c0 + ma*(c1 + ma*(c2 + ma*c3)) -- Horner's method via chained SFPMADs
        sfpi::vFloat pa = ma * c3 + sfpi::vConstFloatPrgm2; // SFPMAD: ma * c3 + c2
        pa = pa * ma + sfpi::vConstFloatPrgm1;              // SFPMAD: pa * ma + c1
        pa = pa * ma + sfpi::vConstFloatPrgm0;              // SFPMAD: pa * ma + c0
        sfpi::vFloat ln_a = sfpi::int32_to_float(ea, 0) * ln2 + pa; // SFPCAST(int->float RNE) then SFPMAD

        // ln(b): decompose b = 2^eb * mb, mb in [1, 2)
        sfpi::vInt eb = sfpi::exexp(b);       // SFPEXEXP with DEBIAS
        sfpi::vFloat mb = sfpi::setexp(b, 127); // SFPSETEXP
        sfpi::vFloat pb = mb * c3 + sfpi::vConstFloatPrgm2; // SFPMAD: Horner step 1
        pb = pb * mb + sfpi::vConstFloatPrgm1;              // SFPMAD: Horner step 2
        pb = pb * mb + sfpi::vConstFloatPrgm0;              // SFPMAD: Horner step 3
        sfpi::vFloat ln_b = sfpi::int32_to_float(eb, 0) * ln2 + pb; // SFPCAST + SFPMAD

        // atanh(x) = 0.5 * (ln(a) - ln(b))
        sfpi::vFloat result = (ln_a - ln_b) * 0.5f; // SFPMAD (subtract) then SFPMAD (multiply by 0.5)

        sfpi::dst_reg[0] = result; // SFPSTORE: write 32 elements to current DEST row pair
        sfpi::dst_reg++;           // Advance DEST pointer by 1 sfpi row (2 physical rows, 32 elements)
    }
}

template <bool APPROXIMATION_MODE>
inline void atanh_init() {
    // Programs three constant registers via SFPCONFIG for the cubic polynomial coefficients
    sfpi::vConstFloatPrgm0 = -0x1.952992p+0f;  // c0 ~ -1.5828 -> Prog Const 3 (CREG_IDX_PRGM1)
    sfpi::vConstFloatPrgm1 = 0x2.4f5388p+0f;   // c1 ~  2.3110 -> Prog Const 4 (CREG_IDX_PRGM2)
    sfpi::vConstFloatPrgm2 = -0xd.e712ap-4f;   // c2 ~ -0.8691 -> Prog Const 5 (CREG_IDX_PRGM3)
}

}  // namespace ckernel::sfpu
```

### SFPU Instructions Used

| Instruction | SFPI Abstraction | Description |
|-------------|-----------------|-------------|
| `SFPLOAD` | `dst_reg[0]` (read) | Loads 32 elements from the current DEST row pair into an LREG for processing |
| `SFPSTORE` | `dst_reg[0] = result` (write) | Stores 32 elements from an LREG back to the current DEST row pair |
| `SFPMAD` | `vFloat + vFloat`, `vFloat * scalar`, subtraction | Fused multiply-add: `VD = VA * VB + VC`. Used for all floating-point arithmetic: addition (a*1+b), subtraction (via sign inversion on InstrMod[1]), multiplication (a*b+0), and the Horner polynomial chain |
| `SFPEXEXP` | `sfpi::exexp(v)` | Extracts the 8-bit biased exponent from an IEEE 754 float, subtracts bias (127), returns debiased integer exponent in bits 7:0 |
| `SFPSETEXP` | `sfpi::setexp(v, 127)` | Replaces the exponent field of a float with the given value (127), effectively normalizing the mantissa to [1, 2) |
| `SFPCAST` | `sfpi::int32_to_float(ei, 0)` | Converts a signed 32-bit integer to FP32 using round-to-nearest-even (round_mode=0 -> `SFPCAST_MOD1_INT32_TO_FP32_RNE`) |
| `SFPLOADI` | Immediate float constants (`c3`, `ln2`, `0.5f`) | Loads 16-bit immediate values into LREGs; the compiler may use two SFPLOADI instructions (HI16 + LO16) or one for bfloat16-representable constants |
| `SFPCONFIG` | `vConstFloatPrgm{0,1,2} = value` (in `atanh_init`) | Programs the SFPU's programmable constant registers (Prog Const 3/4/5) with the polynomial coefficients c0, c1, c2 |

### SFPU Register Usage

| Register Type | Registers Used | Purpose |
|---------------|---------------|---------|
| **DEST rows** | Current sfpi row pair (via `dst_reg[0]`) | Source of input `x` and destination of output `result`; accessed sequentially across 8 iterations per face |
| **LREGs (LREG0-LREG7)** | Allocated by SFPI compiler | Temporary storage for intermediate values: `x`, `a`, `b`, `ea`, `ma`, `pa`, `eb`, `mb`, `pb`, `ln_a`, `ln_b`, `result`, and scalar constants `c3`, `ln2`, `0.5f`. The SFPI compiler manages register allocation across these 8 registers. Given the high number of live variables (~13 at peak), the compiler must carefully schedule spills/reloads. |
| **Programmable Constants** | Prog Const 3 (`vConstFloatPrgm0`), Prog Const 4 (`vConstFloatPrgm1`), Prog Const 5 (`vConstFloatPrgm2`) | Loaded during `atanh_init()` with cubic polynomial coefficients: c0=-1.5828, c1=2.3110, c2=-0.8691 |
| **Fixed Constants** | Fixed Const 2 (`vConst1` = 1.0) | Used to compute `1+x` and `1-x` |

### Address Mode Configuration

The SFPU address mode is configured during `_llk_math_eltwise_unary_sfpu_init_<SfpuType::atanh>()` by calling `eltwise_unary_sfpu_configure_addrmod<SfpuType::atanh>()`.

Since `SfpuType::atanh` does not match any of the special cases (`topk_local_sort`, `typecast`, `unary_max`, `unary_min`, etc.), only the default `ADDR_MOD_7` is configured:

```
ADDR_MOD_7: srca.incr = 0, srcb.incr = 0, dest.incr = 0
```

This is the same on both **Wormhole** and **Blackhole** -- the `eltwise_unary_sfpu_configure_addrmod` templates are identical for both architectures regarding the default `ADDR_MOD_7` configuration. The zero-increment configuration means that auto-increment of DEST addressing is NOT used via the address mode; instead, DEST row advancement is handled explicitly by the SFPI `dst_reg++` abstraction within the kernel loop and `SETRWC` instructions between faces in the params dispatch.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 expansion, and approximation mode for ATANH
   **Key Findings**: ATANH uses `eltwise_sfpu.cpp`, expands to `atanh_tile_init(); atanh_tile({idst});`, macro `SFPU_OP_ATANH_INCLUDE`, `get_op_approx_mode()` returns false (default case)

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/atanh.h`
   **Reason**: API header layer -- understand the tile-level API signature and APPROX template propagation
   **Key Findings**: `atanh_tile(idst)` calls `llk_math_eltwise_unary_sfpu_atanh<APPROX>(idst)`, `atanh_tile_init()` calls `llk_math_eltwise_unary_sfpu_atanh_init<APPROX>()`

3. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h`
   **Reason**: LLK dispatch layer -- understand how the tile-level call reaches the core SFPU function
   **Key Findings**: Bridges to `_llk_math_eltwise_unary_sfpu_params_` with `calculate_atanh<APPROXIMATE, 8>` as callable and `VectorMode::RC` as default mode

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h`
   **Reason**: Core SFPU implementation -- the main analysis target
   **Key Findings**: Implements atanh via IEEE 754 decomposition: decomposes both `1+x` and `1-x` into mantissa and exponent, evaluates `ln()` on each using a cubic minimax polynomial in Horner form, then computes `0.5*(ln(1+x)-ln(1-x))`. Uses SFPI abstractions (vFloat, vInt, dst_reg, exexp, setexp, int32_to_float). WH and BH implementations are identical.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch -- understand face iteration and DEST address progression
   **Key Findings**: VectorMode::RC loops over 4 faces, calls sfpu_func per face, uses TTI_SETRWC to advance DEST by face stride between faces

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: SFPU init and address mode configuration
   **Key Findings**: `ADDR_MOD_7` configured with all-zero increments; SfpuType::atanh does not match any special-case addr_mod configuration

7. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: Authoritative SFPU hardware reference for instruction semantics, register layout, and addressing model
   **Key Findings**: Stride-2 addressing, SFPMAD for all float arithmetic, SFPEXEXP/SFPSETEXP for IEEE 754 decomposition, SFPCAST for int-to-float conversion, SFPCONFIG for programming constant registers

8. **File**: `runtime/sfpi/include/sfpi_lib.h`
   **Reason**: Understand SFPI intrinsic-to-instruction mappings
   **Key Findings**: `exexp()` -> `__builtin_rvtt_sfpexexp` (SFPEXEXP with DEBIAS), `setexp()` -> `__builtin_rvtt_sfpsetexp_i` (SFPSETEXP), `int32_to_float()` -> `__builtin_rvtt_sfpcast` (SFPCAST with INT32_TO_FP32_RNE for round_mode=0)

9. **File**: `runtime/sfpi/include/sfpi.h`
   **Reason**: Understand constant register mappings and vConst abstraction
   **Key Findings**: `vConst1` = Fixed Const 2 (CREG_IDX_1 = 10, value 1.0), `vConstFloatPrgm0/1/2` = Prog Const 3/4/5 (CREG_IDX_PRGM1/2/3 = 12/13/14), assignment to vConst emits SFPCONFIG

10. **File**: `tt_metal/jit_build/genfiles.cpp`
    **Reason**: Understand how `APPROX` constexpr bool is JIT-generated
    **Key Findings**: `constexpr bool APPROX = {}` is generated from `desc.get_hlk_math_approx_mode()`, which reflects the `math_approx_mode` field from `ComputeConfig`

11. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
    **Reason**: Understand conditional include mechanism for split SFPU ops
    **Key Findings**: `#if SFPU_OP_ATANH_INCLUDE` guards the include of `atanh.h`, preventing compilation overhead for unused SFPU ops
