## SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary

- **UnaryOpType**: `SINH`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `sinh_tile_init(); sinh_tile(0);`

#### Approximation Mode Resolution

There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(SINH)` in `unary_op_utils.cpp` — switch has only `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | `false` (default) | `get_op_init_and_func_default()` returns `sinh_tile_init()` / `sinh_tile(0)` with no explicit template argument; `sinh_tile` calls `llk_math_eltwise_unary_sfpu_sinh<APPROX>` where `APPROX` is set to `math_approx_mode` = `false` at compile time |
| Effective SFPU path | Both template arguments `APPROXIMATION_MODE=false` and `ITERATIONS=8` are resolved; the kernel has no `if constexpr (APPROXIMATION_MODE)` branches, so the approximation mode parameter is present but does not alter the code path — `calculate_sinh<false, 8>()` executes the same logic regardless | `ckernel_sfpu_sinh.h` — `calculate_sinh` body has no `if constexpr (APPROXIMATION_MODE)` guard |

---

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/sinh.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sinh.h` (identical for Blackhole: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sinh.h`) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h` (identical for Blackhole: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h`) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Wormhole); `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Blackhole) |

---

### Call Chain

`sinh_tile(idst)` in the API header calls `llk_math_eltwise_unary_sfpu_sinh<APPROX>(idst)` (wrapped in `MATH(...)` so only the math thread compiles it). `llk_math_eltwise_unary_sfpu_sinh` in the LLK dispatch header calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_sinh<APPROXIMATE, ITERATIONS>, dst_index, vector_mode)`. `_llk_math_eltwise_unary_sfpu_params_` sets the DEST write address, issues a STALLWAIT, then calls `calculate_sinh<false, 8>()` once per face in a face-loop controlled by `VectorMode::RC`. `calculate_sinh` is the core SFPU kernel defined in `ckernel_sfpu_sinh.h`; it contains the per-element loop (`ITERATIONS=8`) that loads from `dst_reg[0]`, computes the result, and stores back.

---

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default, passed as integer). All 4 tile faces are processed in a `for (face=0; face<4; face++)` loop; each iteration calls `calculate_sinh<false,8>()` for one face's worth of 8 sfpi rows.
- **Operation invocation**: `calculate_sinh` is invoked as a callable passed to `_llk_math_eltwise_unary_sfpu_params_`. It is called once per face (4 times total per tile). Within `calculate_sinh`, a `for (d=0; d<ITERATIONS=8; d++)` loop processes 8 sfpi rows, each covering 32 elements (2 physical DEST rows × 16 elements).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). The address-mode slot `ADDR_MOD_7` is configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType::sinh>()` with `dest.incr=0`, `srca.incr=0`, `srcb.incr=0` — no hardware auto-increment. Manual advancement is done inside `calculate_sinh` via `sfpi::dst_reg++` (1 sfpi row = 2 physical DEST rows = 32 elements per step) and between faces via two `math::inc_dst_addr<8>()` calls (`SETRWC`) in `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()`. On Wormhole, `set_addr_mod_base()` / `clear_addr_mod_base()` bookend the SFPU work; on Blackhole those calls are absent from `_llk_math_eltwise_unary_sfpu_params_`.

---

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::vInt`, `dst_reg`, `v_if`/`v_endif`). Style A applies.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h
// (Blackhole file is byte-for-byte identical)

namespace ckernel::sfpu {

// Helper: compute 2^z using exp_21f algorithm (Moroz et al. 2022)
// Input z must be clamped to avoid overflow/underflow before calling.
// Returns 2^z as a vFloat.
template <bool APPROXIMATION_MODE> // APPROXIMATION_MODE=false (unused inside function body)
inline sfpi::vFloat exp_21f(sfpi::vFloat z) {
    // Step 1: Scale by 2^23 to shift fractional bits into integer position
    // sfpi::addexp wraps SFPDIVP2 in ADD mode: z.exp += 23 (multiply z by 2^23 in float)
    z = sfpi::addexp(z, 23);

    // Step 2: Add IEEE 754 bias (0x3F800000 = 1.0f) and reinterpret bits as int
    const sfpi::vFloat bias = sfpi::vFloat(0x3f800000);  // immediate load of 1.0f bits
    // _float_to_int32_positive_ is a bit-reinterpret: treats the FP32 bit pattern as int32
    // Equivalent to sfpi::reinterpret<sfpi::vInt>(z + bias). Defined only within this kernel.
    sfpi::vInt z_int = _float_to_int32_positive_(z + bias);  // z+bias emits SFPMAD; reinterpret is zero-cost

    // Step 3: Decompose integer representation into exponent and mantissa parts
    // sfpi::exexp: SFPEXEXP with DEBIAS mode — extracts biased exponent, subtracts 127
    sfpi::vInt exp_part = sfpi::exexp(sfpi::reinterpret<sfpi::vFloat>(z_int));
    // sfpi::exman9: SFPEXMAN with PAD9 mode — extracts 9 mantissa bits (right-aligned)
    sfpi::vInt man_part = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z_int));

    // Step 4: Polynomial refinement for 2^frac(z) using Moroz et al. coefficients
    sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7f);  // constant scalar load
    // sfpi::int32_to_float: SFPCAST INT32_TO_FP32_RNE (round=0) — converts integer to float
    sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(0xf94ee7) + man_part, 0);  // SFPIADD + SFPCAST
    sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(0x560e) + man_part, 0);    // SFPIADD + SFPCAST

    d2 = d1 * d2;  // SFPMAD: d2 = d1 * d2 + 0.0
    sfpi::vInt frac_int = _float_to_int32_positive_(d2 * d3);  // SFPMAD + bit-reinterpret

    // Step 5: Reconstruct result by combining mantissa fraction with exponent
    // sfpi::setexp (SFPSETEXP): replaces the exponent field in frac_int with (127 + exp_part)
    sfpi::vInt result_int =
        sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(frac_int), 127U + exp_part));

    return sfpi::reinterpret<sfpi::vFloat>(result_int);  // zero-cost type pun
}

// sinh(x) = (exp(x) - exp(-x)) / 2
//         = (2^(x * log2(e)) - 2^(-x * log2(e))) / 2
//
// For small |x| (< 0.5), catastrophic cancellation is avoided by using the
// Taylor approximation sinh(x) ≈ x + x³/6, accurate to < 1 ULP in bfloat16.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8> // APPROXIMATION_MODE=false, ITERATIONS=8
inline void calculate_sinh() {
    constexpr float log2e = 1.4426950408889634f;  // log2(e), loaded as immediate
    const sfpi::vFloat v_log2e = log2e;
    const sfpi::vFloat v_half = 0.5f;
    const sfpi::vFloat v_low_threshold = -127.0f;  // clamp exponent to prevent SFPU underflow
    const sfpi::vFloat v_sixth = 0.16666667f;       // 1/6 for Taylor series

#pragma GCC unroll 0  // prevent GCC from unrolling — keeps code size small
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];  // SFPLOAD: load 32 elements from current DEST row pair

        // Compute z_pos = x * log2(e), so exp(x) = 2^z_pos
        sfpi::vFloat z_pos = x * v_log2e;  // SFPMAD: z_pos = x * log2e + 0.0

        // Clamp z_pos to prevent 2^z_pos underflow when z_pos << -127
        v_if(z_pos < v_low_threshold) { z_pos = v_low_threshold; }  // SFPSETCC + CC-guarded SFPMOV
        v_endif;

        sfpi::vFloat exp_pos = exp_21f<APPROXIMATION_MODE>(z_pos);  // computes 2^z_pos

        // z_neg = -z_pos = -x * log2(e), so exp(-x) = 2^z_neg
        sfpi::vFloat z_neg = -z_pos;  // SFPMAD: negate (multiply by -1.0 + 0.0)

        // Clamp z_neg similarly (handles large positive x where z_neg becomes very negative)
        v_if(z_neg < v_low_threshold) { z_neg = v_low_threshold; }  // SFPSETCC + CC-guarded SFPMOV
        v_endif;

        sfpi::vFloat exp_neg = exp_21f<APPROXIMATION_MODE>(z_neg);  // computes 2^z_neg = exp(-x)

        // Main formula: sinh(x) = (exp(x) - exp(-x)) / 2
        sfpi::vFloat y = (exp_pos - exp_neg) * v_half;  // SFPMAD: (exp_pos - exp_neg) * 0.5

        // Small-x override: replace with Taylor series for |x| < 0.5 to avoid cancellation
        sfpi::vFloat abs_x = sfpi::setsgn(x, 0);  // SFPSETSGN with imm=0: force sign bit to 0 → |x|
        v_if(abs_x < v_half) {  // SFPSETCC: condition (|x| < 0.5), CC-guards the block below
            sfpi::vFloat x_sq = x * x;               // SFPMAD: x^2
            y = x + x_sq * x * v_sixth;              // SFPMAD chain: x + (x^2 * x) * (1/6)
        }
        v_endif;  // SFPENCC: restore all-lanes-enabled

        // Round result to bfloat16 to ensure deterministic output regardless of FP32 precision
        // sfpi::float_to_fp16b: SFP_STOCH_RND with SFPSTOCHRND_MOD1_FP32_TO_FP16B, rounding=0 (NearestEven)
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));

        sfpi::dst_reg[0] = y;  // SFPSTORE: write 32 elements back to current DEST row pair
        sfpi::dst_reg++;       // advance by 1 sfpi row = 2 physical DEST rows = 32 elements
    }
}

template <bool APPROXIMATION_MODE>
inline void sinh_init() {
    // No programmable constants needed — exp_21f uses only computed immediates
}

}  // namespace ckernel::sfpu
```

---

### SFPU Instructions Used

The following SFPU instructions are emitted by `calculate_sinh` and its helper `exp_21f` (via SFPI compiler lowering of the abstractions listed):

| Instruction | Usage in kernel |
|-------------|----------------|
| `SFPLOAD` | `sfpi::dst_reg[0]` — loads 32 elements (2 physical DEST rows) from DEST into an LREG for processing |
| `SFPSTORE` | `sfpi::dst_reg[0] = y` — writes 32 elements back to the same DEST rows |
| `SFPMAD` | Emitted for every `vFloat` multiply (`x * v_log2e`, `x * x`, `d1 * d2`, `d2 * d3`, `(exp_pos - exp_neg) * v_half`) and addition (`z + bias`). There is no dedicated float-add instruction; subtraction `exp_pos - exp_neg` is `SFPMAD(exp_pos, 1.0, -exp_neg)`. All arithmetic on `vFloat` lowers to `SFPMAD`. |
| `SFPIADD` | Emitted for integer additions: `sfpi::vInt(0xf94ee7) + man_part` and `sfpi::vInt(0x560e) + man_part` inside `exp_21f`. Also emitted by `127U + exp_part` in the exponent reconstruction step. |
| `SFPLOADI` | Used to load scalar float constants as immediates into LREGs: `log2e`, `0.5f`, `-127.0f`, `0.16666667f`, and the bias `0x3f800000`. |
| `SFPDIVP2` | Backing instruction for `sfpi::addexp(z, 23)` in `exp_21f` (SFPDIVP2 in ADD mode: adds an integer to the IEEE exponent field). |
| `SFPEXEXP` | Backing instruction for `sfpi::exexp(...)` — extracts the biased exponent from an FP32 register with DEBIAS mode (subtracts 127). |
| `SFPEXMAN` | Backing instruction for `sfpi::exman9(...)` — extracts 9 bits of the significand (PAD9 mode). |
| `SFPSETEXP` | Backing instruction for `sfpi::setexp(...)` — sets the exponent field of an FP32 register to a computed value (`127 + exp_part`). |
| `SFPSETSGN` | Backing instruction for `sfpi::setsgn(x, 0)` — clears the sign bit to compute `|x|`. |
| `SFPCAST` | Backing instruction for `sfpi::int32_to_float(...)` — converts INT32 to FP32 with round=0 (round-nearest-even). Used twice in `exp_21f` to convert integer intermediate values to float for polynomial evaluation. |
| `SFP_STOCH_RND` | Backing instruction for `sfpi::float_to_fp16b(y, 0)` — converts FP32 to BF16 with round=0 (nearest-even). Used once per SFPU iteration to round the output before storing. |
| `SFPSETCC` | Generated by `v_if` comparisons: `z_pos < v_low_threshold`, `z_neg < v_low_threshold`, and `abs_x < v_half`. Sets the per-lane condition code register. |
| `SFPENCC` | Generated by `v_endif` — re-enables all lanes after each conditional block. |
| `SFPMOV` | Generated inside `v_if` for the conditional assignment `z_pos = v_low_threshold` and `z_neg = v_low_threshold` — copies a scalar constant to a lane register only on enabled lanes. |

---

### SFPU Register Usage

- **DEST register**: Each iteration of `calculate_sinh` reads from and writes to `dst_reg[0]`, which addresses the current DEST row pair (2 physical rows × 16 elements = 32 elements per access). `dst_reg++` advances by one sfpi row (stride-2: 2 physical DEST rows) between iterations. After 8 iterations (ITERATIONS=8), one face (256 elements) is processed. The parameters dispatch layer advances to the next face via `SETRWC` (two calls to `math::inc_dst_addr<8>()`).

- **LREGs (LREG0–LREG7)**: The SFPI compiler manages LREG allocation automatically. Variables used include:
  - `x` — input element loaded from DEST
  - `z_pos`, `z_neg` — scaled exponent arguments for `exp_21f`
  - `exp_pos`, `exp_neg` — 2^z_pos and 2^z_neg outputs from `exp_21f`
  - `z_int`, `exp_part`, `man_part`, `frac_int`, `result_int` — integer intermediates inside `exp_21f`
  - `d1`, `d2`, `d3` — polynomial coefficient temporaries in `exp_21f`
  - `abs_x`, `x_sq`, `y` — scratch registers for the Taylor fallback and output assembly

  The SFPI compiler may alias some of these across live ranges. With 8 LREGs available per lane and the function having approximately 10–12 simultaneously live values at peak, the compiler may need to spill to DEST via SFPSTORE/SFPLOAD. The exact LREG assignment is compiler-determined.

- **Condition code register (CC)**: Three separate `v_if`/`v_endif` blocks update CC per SFPU iteration. Each `v_if` sets CC via `SFPSETCC` and restores all-lanes via `SFPENCC` at `v_endif`. The CC stack is not used (no nesting).

- **Programmable constant registers (RL0–RL5)**: Not used. `sinh_init()` does nothing; all constants are loaded as immediates via `SFPLOADI`.

---

### Address Mode Configuration

Both Wormhole and Blackhole configure the same address mode slot for `SfpuType::sinh` (which falls into the default case in `eltwise_unary_sfpu_configure_addrmod`):

| Hardware | Slot | srca.incr | srcb.incr | dest.incr | Notes |
|----------|------|-----------|-----------|-----------|-------|
| Wormhole B0 | `ADDR_MOD_7` | 0 | 0 | 0 | Configured in `llk_math_eltwise_unary_sfpu.h` (WH); no auto-increment |
| Blackhole | `ADDR_MOD_7` | 0 | 0 | 0 | Configured in `llk_math_eltwise_unary_sfpu.h` (BH); identical |

`ADDR_MOD_6` is **not** configured for `sinh` (it is only set for `topk_local_sort`, `typecast`, `unary_max`, `unary_min`, and a few other ops). With `dest.incr=0`, there is no hardware-assisted DEST pointer auto-increment. All DEST pointer advancement is done explicitly:

- **Within a face**: `sfpi::dst_reg++` inside `calculate_sinh` increments the SFPI destination register pointer by 1 sfpi row (= 2 physical DEST rows, due to `SFP_DESTREG_STRIDE=2`).
- **Between faces**: `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice, issuing two `SETRWC` instructions each incrementing by 8, totaling 16 physical rows (= one full face of 16 rows × 16 elements).

On Wormhole, `math::set_addr_mod_base()` and `math::clear_addr_mod_base()` bracket the SFPU operation (establishing and releasing the base address register). On Blackhole, these calls are absent from `_llk_math_eltwise_unary_sfpu_params_` — face advancement uses helper functions directly.

---

## Local Knowledge Sources

### Local References

1. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h`
   **Reason**: Primary SFPU kernel source — contains `exp_21f` helper and `calculate_sinh` main function
   **Key Findings**: Implements the Moroz et al. 2022 exp_21f algorithm for 2^z; dual-path design (full exp for |x|>=0.5, Taylor x+x³/6 for |x|<0.5); outputs rounded to BF16 via SFP_STOCH_RND; APPROXIMATION_MODE template param present but has no branching effect

2. **File**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h`
   **Reason**: Blackhole variant for cross-architecture comparison
   **Key Findings**: Byte-for-byte identical to the Wormhole version — single implementation shared across both architectures

3. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/sinh.h`
   **Reason**: Public compute API header — exposes `sinh_tile()` and `sinh_tile_init()` to user kernels
   **Key Findings**: `sinh_tile(idst)` calls `llk_math_eltwise_unary_sfpu_sinh<APPROX>(idst)` via the `MATH(...)` macro; `APPROX` is set to `math_approx_mode` from the program factory (false for sinh)

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sinh.h`
   **Reason**: LLK dispatch layer — bridges the API to the parameters dispatch and the core kernel
   **Key Findings**: `llk_math_eltwise_unary_sfpu_sinh` calls `_llk_math_eltwise_unary_sfpu_params_` with `calculate_sinh<APPROXIMATE, ITERATIONS=8>` as a callable; default vector_mode is `VectorMode::RC`

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch — implements face-loop iteration and SETRWC-based face address advancement
   **Key Findings**: `VectorMode::RC` drives a `for (face=0; face<4; face++)` loop calling the SFPU callable once per face; face transition uses two `TTI_SETRWC` calls each advancing 8 physical rows; `set_addr_mod_base`/`clear_addr_mod_base` bookend the operation on Wormhole

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Base LLK math unary SFPU — contains `eltwise_unary_sfpu_configure_addrmod` and `_llk_math_eltwise_unary_sfpu_init_`
   **Key Findings**: `SfpuType::sinh` falls into the default case — only `ADDR_MOD_7` (dest.incr=0) is configured; no auto-increment hardware is used for this operation

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Blackhole variant of the base LLK math unary SFPU for address mode comparison
   **Key Findings**: Same `ADDR_MOD_7` configuration for sinh; Blackhole omits `set_addr_mod_base`/`clear_addr_mod_base` from the params dispatch wrapper

8. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determines compute kernel path, SFPU_OP_CHAIN_0 expansion, and approximation mode for sinh
   **Key Findings**: `get_compute_kernel_path(SINH)` returns `eltwise_sfpu.cpp` (default); `get_op_init_and_func_default(SINH)` returns `sinh_tile_init()` / `sinh_tile(0)`; `get_op_approx_mode(SINH)` returns `false`; `SFPU_OP_SINH_INCLUDE` macro is defined to pull in the API header

9. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
   **Reason**: Conditional include guard — confirms how `SFPU_OP_SINH_INCLUDE` gates the sinh API header in the compute kernel
   **Key Findings**: `#if SFPU_OP_SINH_INCLUDE` includes `api/compute/eltwise_unary/sinh.h`; sinh is one of four "Wave 3 generated ops" listed in this file

10. **File**: `runtime/sfpi/include/sfpi_lib.h`
    **Reason**: Source of SFPI intrinsic functions used in the kernel (`addexp`, `exexp`, `exman9`, `setexp`, `setsgn`, `int32_to_float`, `float_to_fp16b`)
    **Key Findings**: Each SFPI function maps to a specific SFPU instruction: `addexp`→`SFPDIVP2(ADD)`, `exexp`→`SFPEXEXP(DEBIAS)`, `exman9`→`SFPEXMAN(PAD9)`, `setexp`→`SFPSETEXP`, `setsgn`→`SFPSETSGN`, `int32_to_float`→`SFPCAST`, `float_to_fp16b`→`SFP_STOCH_RND(FP32_TO_FP16B)`

11. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Hardware model reference — tile geometry, DEST layout, stride-2 addressing, SFPU instruction semantics
    **Key Findings**: Confirmed ITERATIONS=8 per face derivation (FACE_HEIGHT=16 / SFP_DESTREG_STRIDE=2); dst_reg++ = 2 physical rows = 32 elements; SFPMAD is the only float arithmetic instruction; SFPIADD is integer-only
