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
| Template parameter (SFPU_OP_CHAIN) | none (non-parameterized) | `get_op_init_and_func_default()` -- `atanh_tile_init()` / `atanh_tile(idst)` with no explicit template arguments; defaults resolve through `APPROX` compile-time constant |
| Effective SFPU path | `APPROXIMATION_MODE=false`, but the kernel has no `if constexpr` branches on this parameter -- the code path is identical regardless of its value | `calculate_atanh<APPROXIMATION_MODE, ITERATIONS>()` in `ckernel_sfpu_atanh.h` does not branch on `APPROXIMATION_MODE` |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/atanh.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. **Compute kernel** (`eltwise_sfpu.cpp`): The `SFPU_OP_CHAIN_0` macro expands to `atanh_tile(0)`, calling into the API header.
2. **API header** (`atanh.h`): `atanh_tile(idst)` calls `llk_math_eltwise_unary_sfpu_atanh<APPROX>(idst)` on the MATH thread via the `MATH(...)` wrapper.
3. **LLK dispatch** (`llk_math_eltwise_unary_sfpu_atanh.h`): `llk_math_eltwise_unary_sfpu_atanh<APPROXIMATE, ITERATIONS=8>(dst_index, VectorMode::RC)` calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_atanh<APPROXIMATE, 8>, dst_index, VectorMode::RC)`.
4. **Parameters dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): Sets up DEST addressing for `dst_index`, stalls until SFPU is ready, then loops over 4 faces calling `calculate_atanh<false, 8>()` per face, with `SETRWC`/`inc_dst_face_addr` between faces to advance the DEST write pointer.
5. **Core SFPU** (`ckernel_sfpu_atanh.h`): `calculate_atanh<false, 8>()` executes 8 iterations (one per sfpi row within a face), computing `atanh(x) = 0.5 * (ln(1+x) - ln(1-x))` using IEEE 754 decomposition and a cubic minimax polynomial approximation for `ln`.

For initialization: `atanh_tile_init()` calls `llk_math_eltwise_unary_sfpu_atanh_init<APPROX>()`, which calls `llk_math_eltwise_unary_sfpu_init<SfpuType::atanh, APPROXIMATE>(sfpu::atanh_init<APPROXIMATE>)`. This configures the SFPU address mode and then calls `atanh_init<false>()` to load the cubic polynomial coefficients into programmable constant registers.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default) -- processes all 4 faces of the tile (full 32x32 tile coverage).
- **Operation invocation**: The params dispatch calls `calculate_atanh<false, 8>()` once per face in a loop of 4 iterations. Each invocation processes 8 sfpi rows (= 256 elements = one complete face).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC`/`inc_dst_face_addr` between faces). On Wormhole, the WH params dispatch uses `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` twice between faces (each advances by 8 sfpi rows). On Blackhole, the BH params dispatch calls `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which internally calls `math::inc_dst_addr<8>()` twice.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::vInt`, `sfpi::dst_reg`, `sfpi::exexp`, `sfpi::setexp`, `sfpi::int32_to_float`). Style A is used.

The Wormhole and Blackhole implementations are **identical**.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h

namespace ckernel::sfpu {

// Implementation notes, see the original file for more details
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_atanh() { // APPROXIMATION_MODE=false, ITERATIONS=8
    constexpr float c3 = 0x2.44734p-4f;  // ~0.1416, highest-degree coefficient of cubic ln(m) polynomial
    constexpr float ln2 = 0.6931471805599453f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST row pair

        // a = 1 + x, b = 1 - x (both positive for |x| < 1)
        sfpi::vFloat a = x + sfpi::vConst1;   // SFPMAD: x * 1.0 + 1.0
        sfpi::vFloat b = -x + sfpi::vConst1;  // SFPMAD: (-x) * 1.0 + 1.0

        // ln(a): decompose a = 2^ea * ma, ma in [1, 2)
        sfpi::vInt ea = sfpi::exexp(a);        // SFPEXEXP: extract biased exponent of a, debias to signed int
        sfpi::vFloat ma = sfpi::setexp(a, 127); // SFPSETEXP: force exponent to 127 (bias), yielding mantissa in [1,2)
        // P(ma) = c0 + ma*(c1 + ma*(c2 + ma*c3))  -- Horner's method
        sfpi::vFloat pa = ma * c3 + sfpi::vConstFloatPrgm2;  // SFPMAD: ma * c3 + c2
        pa = pa * ma + sfpi::vConstFloatPrgm1;                // SFPMAD: pa * ma + c1
        pa = pa * ma + sfpi::vConstFloatPrgm0;                // SFPMAD: pa * ma + c0
        sfpi::vFloat ln_a = sfpi::int32_to_float(ea, 0) * ln2 + pa; // SFPCAST(ea->float) then SFPMAD: float(ea)*ln2 + pa

        // ln(b): decompose b = 2^eb * mb, mb in [1, 2)
        sfpi::vInt eb = sfpi::exexp(b);        // SFPEXEXP: extract biased exponent of b, debias
        sfpi::vFloat mb = sfpi::setexp(b, 127); // SFPSETEXP: force exponent to 127
        sfpi::vFloat pb = mb * c3 + sfpi::vConstFloatPrgm2;  // SFPMAD: mb * c3 + c2
        pb = pb * mb + sfpi::vConstFloatPrgm1;                // SFPMAD: pb * mb + c1
        pb = pb * mb + sfpi::vConstFloatPrgm0;                // SFPMAD: pb * mb + c0
        sfpi::vFloat ln_b = sfpi::int32_to_float(eb, 0) * ln2 + pb; // SFPCAST + SFPMAD

        // atanh(x) = 0.5 * (ln(a) - ln(b))
        sfpi::vFloat result = (ln_a - ln_b) * 0.5f; // SFPMAD for subtraction, SFPMAD for *0.5

        sfpi::dst_reg[0] = result; // SFPSTORE: write 32 elements back to current DEST row pair
        sfpi::dst_reg++;           // advance DEST pointer by 1 sfpi row (= 2 physical DEST rows = 32 elements)
    }
}

template <bool APPROXIMATION_MODE>
inline void atanh_init() { // APPROXIMATION_MODE=false
    // Cubic polynomial coefficients for ln(m) on [1, 2)
    sfpi::vConstFloatPrgm0 = -0x1.952992p+0f;  // c0 ~ -1.5828, loaded via SFPLOADI to CREG_PRGM1
    sfpi::vConstFloatPrgm1 = 0x2.4f5388p+0f;   // c1 ~  2.3110, loaded via SFPLOADI to CREG_PRGM2
    sfpi::vConstFloatPrgm2 = -0xd.e712ap-4f;   // c2 ~ -0.8691, loaded via SFPLOADI to CREG_PRGM3
}

}  // namespace ckernel::sfpu
```

### SFPU Instructions Used

| Instruction | SFPI Abstraction | Usage in Kernel | Description |
|-------------|-----------------|-----------------|-------------|
| **SFPLOAD** | `sfpi::dst_reg[0]` (read) | Load input `x` from DEST | Loads 32 elements (2 physical DEST rows) from the current sfpi address into an LREG for SFPU processing |
| **SFPSTORE** | `sfpi::dst_reg[0] = result` (write) | Store computed `atanh(x)` to DEST | Writes 32 elements from an LREG back to the current sfpi DEST address |
| **SFPMAD** | `vFloat * vFloat + vFloat`, `vFloat + vFloat`, `vFloat - vFloat` | All arithmetic: additions (`x + 1`), negation+add (`-x + 1`), Horner polynomial steps, multiply by `ln2`, multiply by `0.5`, subtraction (`ln_a - ln_b`) | Fused multiply-add: computes `a * b + c`. Addition is `a * 1.0 + b`. Subtraction is `a * 1.0 + (-b)` or `(-a) * 1.0 + b`. This is the only float arithmetic instruction; there is no dedicated float add. |
| **SFPEXEXP** | `sfpi::exexp(v)` | Extract exponent of `a` and `b` | Extracts the IEEE 754 biased exponent field and debiases it, returning a signed integer representing the true exponent `e` such that `v = 2^e * m` |
| **SFPSETEXP** | `sfpi::setexp(v, 127)` | Normalize mantissa of `a` and `b` to [1, 2) | Sets the exponent field of a float to the given value (127 = bias, meaning exponent 0), effectively extracting the mantissa `m` in [1, 2) |
| **SFPCAST** | `sfpi::int32_to_float(vInt, 0)` | Convert integer exponents `ea`, `eb` to float | Converts a 32-bit signed integer to IEEE 754 float (round mode 0 = stochastic rounding). Used to convert the extracted exponent to float for the `e * ln(2)` computation |
| **SFPLOADI** | `sfpi::vConstFloatPrgmN = ...` (in `atanh_init`) | Load polynomial coefficients c0, c1, c2 into programmable constant registers | Loads an immediate floating-point value into a programmable constant register (CREG). Done once during init, not per-iteration |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST rows** (via `dst_reg`) | Input/output: each iteration reads 32 elements from the current DEST row pair, computes atanh, and writes 32 elements back. `dst_reg++` advances to the next sfpi row. |
| **LREGs (L0-L3)** | Temporary storage for intermediate values during computation. The SFPI compiler allocates these automatically for `vFloat`/`vInt` variables (`x`, `a`, `b`, `ea`, `ma`, `pa`, `ln_a`, `eb`, `mb`, `pb`, `ln_b`, `result`). With many live variables, register pressure is high and the compiler may spill/reload between LREGs and DEST. |
| **CREG_PRGM1** (`vConstFloatPrgm0`) | Polynomial coefficient c0 = -0x1.952992p+0f (~-1.5828). Set once in `atanh_init()`. |
| **CREG_PRGM2** (`vConstFloatPrgm1`) | Polynomial coefficient c1 = 0x2.4f5388p+0f (~2.3110). Set once in `atanh_init()`. |
| **CREG_PRGM3** (`vConstFloatPrgm2`) | Polynomial coefficient c2 = -0xd.e712ap-4f (~-0.8691). Set once in `atanh_init()`. |
| **CREG_1** (`vConst1`) | Hardware constant register holding `1.0f`. Used to compute `1 + x` and `1 - x`. |

### Address Mode Configuration

The address mode is configured in `eltwise_unary_sfpu_configure_addrmod<SfpuType::atanh>()` during initialization. For `SfpuType::atanh`, none of the special `if constexpr` branches apply (atanh is not `topk_local_sort`, `typecast`, `unary_max`, etc.), so only the default `ADDR_MOD_7` is configured.

**Wormhole and Blackhole** (identical configuration):

| ADDR_MOD | srca.incr | srcb.incr | dest.incr | Purpose |
|----------|-----------|-----------|-----------|---------|
| `ADDR_MOD_7` | 0 | 0 | 0 | Default no-auto-increment mode. DEST address progression is handled entirely by `dst_reg++` within the SFPU kernel (per iteration) and `SETRWC`/`inc_dst_face_addr` in the params dispatch (between faces). |

The DEST addressing within the kernel uses the SFPI `dst_reg++` abstraction, which advances the sfpi row pointer by 1 (= 2 physical DEST rows = 32 elements) after each iteration. Between faces, the params dispatch advances by 16 physical rows (2 x `inc_dst_addr<8>()` on BH, or 2 x `TTI_SETRWC(CLR_NONE, CR_D, 8, ...)` on WH).

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 expansion, approximation mode, and include guard for ATANH.
   **Key Findings**: ATANH uses `eltwise_sfpu.cpp`, expands to `atanh_tile(0)`, `get_op_approx_mode` returns `false` (default), include guard is `SFPU_OP_ATANH_INCLUDE`.

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/atanh.h`
   **Reason**: API header exposing `atanh_tile()` and `atanh_tile_init()` to the compute kernel.
   **Key Findings**: Passes `APPROX` template parameter to LLK dispatch. No additional template parameters.

3. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h`
   **Reason**: LLK dispatch layer bridging API to core SFPU.
   **Key Findings**: Init calls `llk_math_eltwise_unary_sfpu_init<SfpuType::atanh>` + `atanh_init<APPROXIMATE>()`. Tile function calls `_llk_math_eltwise_unary_sfpu_params_` with `calculate_atanh<APPROXIMATE, 8>`. Identical on WH and BH.

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h`
   **Reason**: Core SFPU implementation containing `calculate_atanh()` and `atanh_init()`.
   **Key Findings**: Computes atanh(x) = 0.5 * (ln(1+x) - ln(1-x)) using IEEE 754 exponent/mantissa decomposition and a cubic minimax polynomial for ln(m) on [1,2). Uses SFPI abstractions (vFloat, vInt, dst_reg, exexp, setexp, int32_to_float). No condition code usage. No branching on APPROXIMATION_MODE. Identical on WH and BH.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch controlling DEST addressing and per-face invocation.
   **Key Findings**: VectorMode::RC loops over 4 faces, calling the SFPU function once per face, with SETRWC advancing DEST between faces (2 x advance-by-8).

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Address mode configuration for unary SFPU ops.
   **Key Findings**: `eltwise_unary_sfpu_configure_addrmod<SfpuType::atanh>()` only sets ADDR_MOD_7 with all-zero increments. No special addr_mod branches for atanh.

7. **File**: `tt_metal/jit_build/genfiles.cpp`
   **Reason**: Confirm how the `APPROX` compile-time constant is generated from `math_approx_mode`.
   **Key Findings**: `emit_math_scalar_descriptors()` emits `constexpr bool APPROX = {value};` where value comes from `hlk_desc.get_hlk_math_approx_mode()`, which is set from the ComputeConfig's `math_approx_mode` field.

8. **File**: `runtime/sfpi/include/sfpi_lib.h`
   **Reason**: Understand SFPI intrinsic-to-instruction mappings for `exexp`, `setexp`, `int32_to_float`.
   **Key Findings**: `exexp()` maps to `__builtin_rvtt_sfpexexp` (SFPEXEXP with DEBIAS mode). `setexp(v, imm)` maps to `__builtin_rvtt_sfpsetexp_i` (SFPSETEXP). `int32_to_float(v, 0)` maps to `__builtin_rvtt_sfpcast` with stochastic rounding mode.

9. **File**: `runtime/sfpi/include/sfpi.h`
   **Reason**: Understand programmable constant register mappings.
   **Key Findings**: `vConstFloatPrgm0` maps to `CREG_IDX_PRGM1`, `vConstFloatPrgm1` to `CREG_IDX_PRGM2`, `vConstFloatPrgm2` to `CREG_IDX_PRGM3`. `vConst1` maps to `CREG_IDX_1`.

10. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Authoritative reference for SFPU hardware model, tile geometry, DEST layout, stride-2 addressing.
    **Key Findings**: Confirmed ITERATIONS=8 per face, SFP_DESTREG_STRIDE=2, dst_tile_size_sfpi=32, 32 elements per dst_reg access.
