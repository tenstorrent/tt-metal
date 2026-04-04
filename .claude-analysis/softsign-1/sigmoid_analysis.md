## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the **sigmoid** unary operation.

### Unary Dispatch Summary
- **UnaryOpType**: `SIGMOID`
- **Compute kernel**: `eltwise_sfpu.cpp` (default path via `get_compute_kernel_path()`)
- **SFPU_OP_CHAIN_0 expansion**: `sigmoid_tile_init<{fast_and_approx}>(); sigmoid_tile<VectorMode::RC, {fast_and_approx}>(0);`

Note: The `SIGMOID` op passes `fast_and_approximate_mode` (0.0 or 1.0) as `param0` from `unary.cpp`. This parameterizes the `fast_and_approx` template argument on `sigmoid_tile_init` and `sigmoid_tile`. In the default mode (`SigmoidMode::ACCURATE`), `fast_and_approx=false`; in `SigmoidMode::FAST_APPROXIMATE`, `fast_and_approx=true`.

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(SIGMOID)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | `fast_and_approx` = 0 (default) or 1 (fast mode) | `sigmoid_tile_init<{param0}u>()` / `sigmoid_tile<VectorMode::RC, {param0}u>(0)` where `param0` comes from `UnaryWithParam{UnaryOpType::SIGMOID, static_cast<float>(fast_and_approximate_mode)}` |
| Effective SFPU path | When `fast_and_approx=false` (APPROXIMATE=false): `calculate_sigmoid<false, is_fp32_dest_acc_en, 8>()` which calls `_sfpu_sigmoid_<is_fp32_dest_acc_en>(val)` using exp + reciprocal. When `fast_and_approx=true` (APPROXIMATE=true): `calculate_sigmoid<true, *, 8>()` which delegates to `calculate_sigmoid_appx<8>()` using 3-entry SFPLUT | `if constexpr (!APPROXIMATION_MODE)` branch in `calculate_sigmoid()` in `ckernel_sfpu_sigmoid.h` (build-expanded) |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/compute_kernel_api.h` (lines 64-86) |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sigmoid.h` (build-generated from metal overlay) |
| **Core SFPU Implementation (tt_llk source)** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_sigmoid.h` (lut2-based accurate path only) |
| **Core SFPU Implementation (build-expanded)** | `build_Debug/libexec/tt-metalium/tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_sigmoid.h` (contains both accurate and appx paths: `calculate_sigmoid`, `_sfpu_sigmoid_`, `sigmoid_init`) |
| **Approximate Path Implementation** | `build_Debug/libexec/tt-metalium/tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_sigmoid_appx.h` (3-entry SFPLUT path: `calculate_sigmoid_appx`, `sigmoid_appx_init`) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. **`sigmoid_tile<VectorMode::RC, fast_and_approx>(idst)`** (API header `compute_kernel_api.h:84`): Wraps the call in the `MATH(...)` macro (active only on TRISC_MATH). Calls `llk_math_eltwise_unary_sfpu_sigmoid<fast_and_approx, DST_ACCUM_MODE>(idst, VectorMode::RC)`.

2. **`llk_math_eltwise_unary_sfpu_sigmoid<APPROXIMATE, is_fp32_dest_acc_en>(dst_index, vector_mode)`** (LLK dispatch `llk_math_eltwise_unary_sfpu_sigmoid.h:18-22`): Forwards to `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(sfpu::calculate_sigmoid<APPROXIMATE, is_fp32_dest_acc_en, 8>, dst_index, vector_mode)`.

3. **`_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(...)`** (params dispatch `llk_math_eltwise_unary_sfpu_params.h:14`): Sets DEST write address, stalls for SFPU readiness, then iterates over faces (4 for VectorMode::RC) calling `sfpu::calculate_sigmoid<APPROXIMATE, is_fp32_dest_acc_en, 8>()` per face, with `TTI_SETRWC` between faces.

4. **`calculate_sigmoid<APPROXIMATION_MODE, is_fp32_dest_acc_en, ITERATIONS>()`** (core SFPU `ckernel_sfpu_sigmoid.h:59-77`): Dispatches to either `_sfpu_sigmoid_<is_fp32_dest_acc_en>(val)` (accurate path) or `calculate_sigmoid_appx<ITERATIONS>()` (approximate path) based on `APPROXIMATION_MODE`.

**Init path**: `sigmoid_tile_init<fast_and_approx>()` -> `llk_math_eltwise_unary_sfpu_sigmoid_init<fast_and_approx>()` -> `llk_math_eltwise_unary_sfpu_init<SfpuType::sigmoid, APPROXIMATE>(sfpu::sigmoid_init<APPROXIMATE>)`. The generic init calls `_llk_math_eltwise_unary_sfpu_init_<SfpuType::sigmoid>()` (configures SFPU config reg, address modes, resets counters), then calls `sigmoid_init<APPROXIMATE>()`. When `APPROXIMATE=false`, this calls `_init_reciprocal_<false, false>()` (WH) or `_init_sfpu_reciprocal_<false>()` (BH) which loads Newton-Raphson polynomial coefficients into programmable constants. When `APPROXIMATE=true`, it calls `sigmoid_appx_init()` which loads 3 LUT coefficients via `TTI_SFPLOADI`.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default) -- processes all 4 faces of the tile.
- **Operation invocation**: For VectorMode::RC, the params dispatch iterates `for (int face = 0; face < 4; face++)`, calling `sfpu::calculate_sigmoid<APPROXIMATE, is_fp32_dest_acc_en, 8>()` once per face. Inside the calculate function, there is a nested loop `for (int d = 0; d < ITERATIONS; d++)` with `ITERATIONS=8`, processing 8 sfpi rows per face (8 iterations x 32 elements = 256 elements = 1 face).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `TTI_SETRWC` between faces). On both Wormhole and Blackhole, the address mode used is `ADDR_MOD_7` with `dest.incr=0` (no auto-increment), since SFPU address advancement is handled by `dst_reg++` in the SFPI abstraction, not by hardware auto-increment.

### Annotated SFPU Kernel Source

The sigmoid operation has **two distinct code paths** determined by `APPROXIMATION_MODE`:
1. **Accurate path** (`APPROXIMATION_MODE=false`): Computes `sigmoid(x) = 1 / (1 + exp(-x))` using SFPI exp + Newton-Raphson reciprocal.
2. **Approximate path** (`APPROXIMATION_MODE=true`): Uses a 3-entry piecewise linear LUT via `SFPLUT` instruction.

Additionally, the **tt_llk source** (`ckernel_sfpu_sigmoid.h` in `tt_metal/third_party/tt_llk/`) contains a third approach using a **6-entry piecewise linear LUT** via `SFPLUTFP32` (`lut2`). This is the `_calculate_sigmoid_` / `_init_sigmoid_` functions. This code is architecture-independent and identical on WH and BH.

#### Accurate Path (build-expanded `ckernel_sfpu_sigmoid.h`)

```cpp
// File: build_Debug/libexec/tt-metalium/tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_sigmoid.h
// (identical on WH and BH)

template <bool is_fp32_acc_to_dest_mode = true>
sfpi_inline sfpi::vFloat _sfpu_sigmoid_(sfpi::vFloat x) { // is_fp32_acc_to_dest_mode depends on DST_ACCUM_MODE
    // Compute sigmoid as:
    // sigmoid(x) = 1 / (1 + exp(-x))

    sfpi::vFloat exp_neg_x;
    // If fp32 then use higher accuracy exp function
    // Otherwise, use exp_21f (~1 ULP on bfloat16)
    if constexpr (is_fp32_acc_to_dest_mode) {
        exp_neg_x = _sfpu_exp_improved_<true>(-x);  // -> _sfpu_exp_f32_accurate_(-x) for fp32
    } else {
        exp_neg_x = _sfpu_exp_21f_<true>(-x);       // bfloat16 path, ~1 ULP accuracy
    }

    sfpi::vFloat denominator = sfpi::vConst1 + exp_neg_x; // 1 + exp(-x), emits SFPMAD (1.0 * 1.0 + exp_neg_x)

    sfpi::vFloat result;
    if constexpr (is_fp32_acc_to_dest_mode) {
        result = _sfpu_reciprocal_<2>(denominator);  // 2 Newton-Raphson iterations for fp32 precision
    } else {
        result = _sfpu_reciprocal_<1>(denominator);  // 1 NR iteration for bfloat16 precision
    }

    return result;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_sigmoid() { // APPROXIMATION_MODE=false for accurate path
    if constexpr (!APPROXIMATION_MODE) {
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];     // SFPLOAD from DEST

            sfpi::vFloat result = _sfpu_sigmoid_<is_fp32_dest_acc_en>(val);

            if constexpr (!is_fp32_dest_acc_en) {
                result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0)); // convert to bfloat16
            }

            sfpi::dst_reg[0] = result;               // SFPSTORE back to DEST
            sfpi::dst_reg++;                          // advance 1 sfpi row = 32 elements
        }
    } else {
        calculate_sigmoid_appx<ITERATIONS>();         // delegate to approximate path
    }
}

template <bool APPROXIMATION_MODE>
inline void sigmoid_init() {
    if constexpr (!APPROXIMATION_MODE) {
        _init_reciprocal_<false, false>();            // WH: loads NR polynomial coefficients into vConstFloatPrgm{0,1,2}
        // BH variant: _init_sfpu_reciprocal_<false>() -- same coefficients
    } else {
        sigmoid_appx_init();                          // loads SFPLUT LUT coefficients
    }
}
```

#### Approximate Path (`ckernel_sfpu_sigmoid_appx.h`)

```cpp
// File: build_Debug/libexec/tt-metalium/tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_sigmoid_appx.h
// (identical on WH and BH)

template <int ITERATIONS = 8>
inline void calculate_sigmoid_appx() {
    vUInt l0 = l_reg[LRegs::LReg0]; // LUT coefficient register 0
    vUInt l1 = l_reg[LRegs::LReg1]; // LUT coefficient register 1
    vUInt l2 = l_reg[LRegs::LReg2]; // LUT coefficient register 2

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat val = dst_reg[0];                     // SFPLOAD from DEST

        dst_reg[0] = lut(val, l0, l1, l2) + 0.5f;   // SFPLUT + SFPMAD (result + 0.5)
        // lut() emits __builtin_rvtt_sfplut with SFPLUT_MOD0_SGN_RETAIN
        // The LUT computes a piecewise linear approximation of sigmoid(|x|) - 0.5
        // on 3 intervals, then adds 0.5 back to get sigmoid(x)

        dst_reg++;                                    // advance 1 sfpi row
    }

    l_reg[LRegs::LReg0] = l0; // restore LUT coefficients
    l_reg[LRegs::LReg1] = l1;
    l_reg[LRegs::LReg2] = l2;
}

inline void sigmoid_appx_init() {
    uint imm0;
    uint imm1;
    uint imm2;
    imm0 = 0x3DFF; // LUT coefficients for 3-piece sigmoid approximation
    imm1 = 0x21D8;
    imm2 = 0xFF10;
    TTI_SFPLOADI(0, 2, imm0); // insmod=2: load unsigned imm16 to LREG0
    TTI_SFPLOADI(1, 2, imm1); // insmod=2: load unsigned imm16 to LREG1
    TTI_SFPLOADI(2, 2, imm2); // insmod=2: load unsigned imm16 to LREG2
}
```

#### tt_llk Source: 6-Entry LUT Path (`ckernel_sfpu_sigmoid.h`)

This is the version in the `tt_llk` submodule source. It uses `lut2` (6-entry `SFPLUTFP32`) for higher accuracy than the 3-entry `SFPLUT` approximate path.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_sigmoid.h
// (identical on WH and BH)

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_sigmoid_(const int iterations)
{
    constexpr int lut_mode = 0; // SFPLUTFP32_MOD0_FP16_6ENTRY_TABLE1
    sfpi::vUInt l0         = sfpi::l_reg[sfpi::LRegs::LReg0]; // slopes A0,A1 packed as FP16 pair
    sfpi::vUInt l1         = sfpi::l_reg[sfpi::LRegs::LReg1]; // slopes A2,A3 packed
    sfpi::vUInt l2         = sfpi::l_reg[sfpi::LRegs::LReg2]; // slopes A4,A5 packed
    sfpi::vUInt l4         = sfpi::l_reg[sfpi::LRegs::LReg4]; // intercepts B0,B1 packed
    sfpi::vUInt l5         = sfpi::l_reg[sfpi::LRegs::LReg5]; // intercepts B2,B3 packed
    sfpi::vUInt l6         = sfpi::l_reg[sfpi::LRegs::LReg6]; // intercepts B4,B5 packed

#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat val = sfpi::dst_reg[0];         // SFPLOAD from DEST

        // lut2 with 6 vUInt regs -> __builtin_rvtt_sfplutfp32_6r
        // emits SFPLUTFP32 with SFPLUTFP32_MOD0_FP16_6ENTRY_TABLE1 | SFPLUTFP32_MOD0_SGN_RETAIN
        // 6-piece piecewise linear: sigmoid(|x|) - 0.5 approximated on 6 intervals
        sfpi::dst_reg[0] = lut2(val, l0, l1, l2, l4, l5, l6, lut_mode) + 0.5f;
        // The + 0.5f is emitted as SFPMAD (result * 1.0 + 0.5)

        sfpi::dst_reg++;
    }

    sfpi::l_reg[sfpi::LRegs::LReg0] = l0; // restore all 6 LUT coefficient registers
    sfpi::l_reg[sfpi::LRegs::LReg1] = l1;
    sfpi::l_reg[sfpi::LRegs::LReg2] = l2;
    sfpi::l_reg[sfpi::LRegs::LReg4] = l4;
    sfpi::l_reg[sfpi::LRegs::LReg5] = l5;
    sfpi::l_reg[sfpi::LRegs::LReg6] = l6;
}

template <bool APPROXIMATION_MODE>
inline void _init_sigmoid_()
{
    // Implementation notes, see the original file for more details
    // Using a 6 piece LUT to calculate and model sigmoid directly
    // x <= 0.5 --> 0.2452x + (-0.0004997)
    // x <= 1.0 --> 0.2173x + 0.0152
    // x <= 1.5 --> 0.1731x + 0.05988
    // x <= 2.0 --> 0.1262x + 0.1298
    // x <= 4.0 --> 0.0485x + 0.2998
    // x >  4.0 --> 0.4998

    // imm0[15:0] = A0=0.2452 = 0x33D9 -- imm0[31:16] = A1=0.2173 = 0x32F4
    _sfpu_load_imm32_(0, 0x32F433D9);
    // imm4[15:0] = B0= -0.0004997  = 0x9018 -- imm4[31:16] = B1= 0.0152 = 0x23c8
    _sfpu_load_imm32_(4, 0x23C89018);

    // imm1[15:0] = A2=0.1731 = 0x318a -- imm1[31:16] = A3=0.1262 = 0x300a
    _sfpu_load_imm32_(1, 0x300A318A);
    // imm5[15:0] = B2=0.05988 = 0x2BAA -- imm5[31:16] = B3=0.1298 = 0x3027
    _sfpu_load_imm32_(5, 0x30272BAA);

    // imm2[15:0] = A4=0.0485 = 0x2A35 -- imm2[31:16] = A5=0.0 = 0x7C00
    _sfpu_load_imm32_(2, 0x7C002A35);
    // imm6[15:0] = B4=0.2998 = 0x34CC -- imm6[31:16] = B5=0.4998 = 0x37ff
    _sfpu_load_imm32_(6, 0x37ff34CC);
}
```

#### Reciprocal Helper (`_sfpu_reciprocal_`)

The accurate sigmoid path relies on Newton-Raphson reciprocal. For reference, the key helper:

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_recip.h
// (BH version is identical)

template <int max_iter = 2>
sfpi_inline sfpi::vFloat _sfpu_reciprocal_(const sfpi::vFloat in)
{
    // Scale input to [1.0, 2.0) range, negate
    sfpi::vFloat negative_x = sfpi::setman(sfpi::vConstNeg1, sfpi::reinterpret<sfpi::vInt>(in));

    // Quadratic initial estimate: y = k2 - k1*x + k0*x^2
    sfpi::vFloat y = sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0 * negative_x; // SFPMAD

    // Scale factor via exponent complement
    sfpi::vInt scale_bits = ~sfpi::reinterpret<sfpi::vUInt>(in); // SFPNOT

    // Continue quadratic estimate
    y = sfpi::vConstFloatPrgm2 + y * negative_x;    // SFPMAD

    // Clear mantissa from scale
    sfpi::vFloat scale = sfpi::setman(sfpi::reinterpret<sfpi::vFloat>(scale_bits), 0); // SFPSETMAN

    // First Newton-Raphson iteration: t = 1.0 - x*y
    sfpi::vFloat t = sfpi::vConst1 + negative_x * y; // SFPMAD
    scale *= 0.5f;                                    // SFPMAD (scale * 0.5 + 0)
    y = y + y * t;                                    // SFPMAD

    if constexpr (max_iter > 1) {
        // Second NR iteration (for fp32 precision)
        t = sfpi::vConst1 + negative_x * y;          // SFPMAD
        y = y + y * t;                                // SFPMAD
    }

    y = y * scale;                                    // SFPMAD
    y = sfpi::setsgn(y, in);                          // SFPSETSGN - restore original sign

    return y;
}

template <bool APPROXIMATION_MODE>
inline void _init_sfpu_reciprocal_()
{
    // Polynomial coefficients for quadratic initial estimate over [1,2)
    sfpi::vConstFloatPrgm0 = 0.3232325017452239990234375f;    // k0
    sfpi::vConstFloatPrgm1 = 1.4545459747314453125f;          // k1
    sfpi::vConstFloatPrgm2 = 2.121212482452392578125f;        // k2
}
```

### SFPU Instructions Used

The instructions used depend on which code path is active:

#### Accurate Path (`APPROXIMATION_MODE=false`)

| Instruction | Source | Description |
|-------------|--------|-------------|
| `SFPLOAD` | `dst_reg[0]` read | Load 32 elements from DEST row into LREG for processing |
| `SFPSTORE` | `dst_reg[0]` write | Store 32 processed elements back to DEST row |
| `SFPMAD` | `vFloat + vFloat`, `vFloat * vFloat`, `vConst1 + exp_neg_x`, NR iterations | Fused multiply-add: used for float addition (a*1.0+b), multiplication (a*b+0.0), and NR iteration steps. The dominant instruction in this path. |
| `SFPNOT` | `~reinterpret<vUInt>(in)` in reciprocal | Bitwise NOT for exponent complement in scale factor computation |
| `SFPSETMAN` | `setman(...)` in reciprocal | Set mantissa field -- used to scale input to [1,2) range and clear mantissa from scale factor |
| `SFPSETSGN` | `setsgn(y, in)` in reciprocal | Set sign field to restore original input sign on reciprocal result |
| `SFPEXEXP` | Inside `_sfpu_exp_21f_` / `_sfpu_exp_f32_accurate_` | Extract exponent for exp(x) range reduction |
| `SFPDIVP2` | Inside `_sfpu_exp_21f_` | Divide by power of 2 (exponent subtract) for exp range reduction |
| `SFPSETEXP` | Inside `_sfpu_exp_21f_` | Set exponent field for exp(x) reconstruction |
| `SFP_STOCH_RND` / `float_to_fp16b` | `reinterpret<vFloat>(float_to_fp16b(result, 0))` | Format conversion to bfloat16 when `is_fp32_dest_acc_en=false` |

#### Approximate Path via SFPLUT (`APPROXIMATION_MODE=true`, appx)

| Instruction | Source | Description |
|-------------|--------|-------------|
| `SFPLOADI` | `TTI_SFPLOADI(reg, 2, imm)` in `sigmoid_appx_init()` | Load 16-bit immediate to LREG (3 coefficients for 3-entry LUT) |
| `SFPLOAD` | `dst_reg[0]` read | Load 32 elements from DEST |
| `SFPLUT` | `lut(val, l0, l1, l2)` -> `__builtin_rvtt_sfplut` | 3-entry piecewise linear LUT lookup on `|val|` using coefficients in LREG0-2 |
| `SFPMAD` | `lut(...) + 0.5f` | Add 0.5 offset to LUT result |
| `SFPSTORE` | `dst_reg[0]` write | Store result back to DEST |

#### tt_llk Source Path via SFPLUTFP32 (6-entry LUT)

| Instruction | Source | Description |
|-------------|--------|-------------|
| `SFPLOADI` | `_sfpu_load_imm32_(reg, val)` calls `TT_SFPLOADI` twice per 32-bit value | Load 32-bit LUT coefficients into 6 LREGs (via two 16-bit loads each: LO16 then HI16) |
| `SFPLOAD` | `dst_reg[0]` read | Load 32 elements from DEST |
| `SFPLUTFP32` | `lut2(val, l0..l6, mode)` -> `__builtin_rvtt_sfplutfp32_6r` | 6-entry piecewise linear LUT with FP16-packed coefficients. Mode `SFPLUTFP32_MOD0_FP16_6ENTRY_TABLE1 | SFPLUTFP32_MOD0_SGN_RETAIN`. Higher accuracy than 3-entry SFPLUT. |
| `SFPMAD` | `lut2(...) + 0.5f` | Add 0.5 offset to LUT result |
| `SFPSTORE` | `dst_reg[0]` write | Store result back to DEST |

### SFPU Register Usage

#### Accurate Path

| Register | Usage |
|----------|-------|
| `LREG0-LREG3` | Temporary working registers for exp() and reciprocal() intermediate values (local `vFloat` variables compiled to LREG allocations) |
| `vConstFloatPrgm0` | Reciprocal quadratic coefficient k0 = 0.3232325 (loaded during `_init_sfpu_reciprocal_`) |
| `vConstFloatPrgm1` | Reciprocal quadratic coefficient k1 = 1.4545460 (loaded during `_init_sfpu_reciprocal_`) |
| `vConstFloatPrgm2` | Reciprocal quadratic coefficient k2 = 2.1212125 (loaded during `_init_sfpu_reciprocal_`) |
| `vConst1` | Fixed constant 1.0 (used in `1 + exp(-x)` and NR iterations) |
| `vConstNeg1` | Fixed constant -1.0 (used in `setman` to create negative scaled input for reciprocal) |
| `vConst0` | Fixed constant 0.0 |
| `dst_reg[0]` | Input/output: reads input value, writes sigmoid result |

#### Approximate Path (SFPLUT, 3-entry)

| Register | Usage |
|----------|-------|
| `LREG0` | LUT coefficient 0 (0x3DFF) |
| `LREG1` | LUT coefficient 1 (0x21D8) |
| `LREG2` | LUT coefficient 2 (0xFF10) |
| `dst_reg[0]` | Input/output |

#### tt_llk Path (SFPLUTFP32, 6-entry)

| Register | Usage |
|----------|-------|
| `LREG0` | Slopes A0=0.2452, A1=0.2173 packed as FP16 pair (0x32F433D9) |
| `LREG1` | Slopes A2=0.1731, A3=0.1262 packed (0x300A318A) |
| `LREG2` | Slopes A4=0.0485, A5=0.0 packed (0x7C002A35) |
| `LREG4` | Intercepts B0=-0.0005, B1=0.0152 packed (0x23C89018) |
| `LREG5` | Intercepts B2=0.0599, B3=0.1298 packed (0x30272BAA) |
| `LREG6` | Intercepts B4=0.2998, B5=0.4998 packed (0x37FF34CC) |
| `dst_reg[0]` | Input/output |

### Address Mode Configuration

The address mode for sigmoid is `ADDR_MOD_7` on both Wormhole and Blackhole, configured identically:

```cpp
// From llk_math_eltwise_unary_sfpu.h (both WH and BH)
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}.set(ADDR_MOD_7);
```

Since `SfpuType::sigmoid` does not match any of the special-case `if constexpr` conditions (topk_local_sort, typecast, etc.), only the default `ADDR_MOD_7` is configured. The `dest.incr = 0` means DEST addressing does not auto-increment via hardware address modes. Instead, DEST address advancement is handled entirely by the SFPI `dst_reg++` abstraction within the kernel loop.

The WH and BH versions are **identical** for `SfpuType::sigmoid`. The only difference between WH and BH `eltwise_unary_sfpu_configure_addrmod` is that BH additionally special-cases `SfpuType::reciprocal` for `ADDR_MOD_6` configuration, which is not relevant for sigmoid.

## Local Knowledge Sources
### Local References
1. **File**: `tt_metal/hw/inc/api/compute/compute_kernel_api.h`
   **Reason**: API header providing `sigmoid_tile()` and `sigmoid_tile_init()` function signatures
   **Key Findings**: `sigmoid_tile<vec_mode, fast_and_approx>(idst)` calls `llk_math_eltwise_unary_sfpu_sigmoid<fast_and_approx, DST_ACCUM_MODE>(idst, vec_mode)`. Default template args: `vec_mode=VectorMode::RC`, `fast_and_approx=false`.

2. **File**: `build_Debug/libexec/tt-metalium/tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sigmoid.h`
   **Reason**: LLK dispatch layer bridging API to ckernel
   **Key Findings**: Init calls `llk_math_eltwise_unary_sfpu_init<SfpuType::sigmoid, APPROXIMATE>(sfpu::sigmoid_init<APPROXIMATE>)`. Tile function calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(sfpu::calculate_sigmoid<APPROXIMATE, is_fp32_dest_acc_en, 8>, ...)`.

3. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_sigmoid.h`
   **Reason**: Core SFPU implementation in tt_llk submodule (6-entry LUT path)
   **Key Findings**: Uses `lut2()` (SFPLUTFP32) with 6 packed FP16 coefficient registers for a 6-piece piecewise linear sigmoid approximation, plus 0.5 offset. Init loads 12 FP16 coefficients (6 slopes + 6 intercepts) via `_sfpu_load_imm32_`.

4. **File**: `build_Debug/libexec/tt-metalium/tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_sigmoid.h`
   **Reason**: Build-expanded core SFPU containing both accurate and approximate paths
   **Key Findings**: `calculate_sigmoid()` dispatches to `_sfpu_sigmoid_()` (accurate: exp + reciprocal) or `calculate_sigmoid_appx()` (3-entry SFPLUT). `sigmoid_init()` dispatches to `_init_reciprocal_()` or `sigmoid_appx_init()`. Contains dead code `_sfpu_sigmoid_legacy_()` that is never called.

5. **File**: `build_Debug/libexec/tt-metalium/tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_sigmoid_appx.h`
   **Reason**: Approximate sigmoid implementation via 3-entry SFPLUT
   **Key Findings**: `calculate_sigmoid_appx()` uses `lut()` (SFPLUT) + 0.5 offset. Init loads 3x16-bit coefficients via `TTI_SFPLOADI`.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_recip.h`
   **Reason**: Newton-Raphson reciprocal used by accurate sigmoid path
   **Key Findings**: `_sfpu_reciprocal_<max_iter>()` implements quadratic initial estimate + NR iterations. `max_iter=2` for fp32, `max_iter=1` for bfloat16. Init loads polynomial coefficients k0, k1, k2 into `vConstFloatPrgm{0,1,2}`.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Address mode configuration for sigmoid SFPU operation
   **Key Findings**: `eltwise_unary_sfpu_configure_addrmod<SfpuType::sigmoid>()` sets `ADDR_MOD_7` with `dest.incr=0`. Identical on WH and BH for sigmoid.

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch function that iterates over faces
   **Key Findings**: `_llk_math_eltwise_unary_sfpu_params_()` dispatches per-face calls with `TTI_SETRWC` between faces. VectorMode::RC processes all 4 faces.

9. **File**: `build_Debug/libexec/tt-metalium/runtime/sfpi/include/sfpi_lib.h`
   **Reason**: SFPI library providing `lut()` and `lut2()` builtins
   **Key Findings**: `lut()` emits `__builtin_rvtt_sfplut` (SFPLUT instruction). `lut2()` with 6 vUInt registers emits `__builtin_rvtt_sfplutfp32_6r` (SFPLUTFP32 instruction with `SFPLUTFP32_MOD0_FP16_6ENTRY_TABLE1`).

10. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_load_config.h`
    **Reason**: Helper for loading 32-bit immediates into LREGs
    **Key Findings**: `_sfpu_load_imm32_(dest, val)` emits two `TT_SFPLOADI` instructions: first with `insmod=10` (LO16) then `insmod=8` (HI16).

11. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: SFPU architecture reference for instruction semantics, register layout, and addressing model
    **Key Findings**: SFPLUT operates on `|RG[3]|` with 3 piecewise linear segments. SFPLUTFP32 supports 6-entry tables with FP16-packed coefficients. SFPMAD is the universal float arithmetic instruction.

12. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp`
    **Reason**: High-level sigmoid function dispatch
    **Key Findings**: `sigmoid()` creates `UnaryWithParam{UnaryOpType::SIGMOID, static_cast<float>(fast_and_approximate_mode)}` where `param0=0.0` for accurate mode, `param0=1.0` for fast approximate mode.
