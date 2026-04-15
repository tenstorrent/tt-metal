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
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(ATANH)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none (default) | `get_op_init_and_func_default()` returns `atanh_tile_init()` / `atanh_tile(0)` with no parameterized template argument |
| Effective SFPU path | `APPROXIMATION_MODE=false`, but the kernel does not branch on this parameter -- the same code path executes regardless | `calculate_atanh<APPROXIMATION_MODE, ITERATIONS>` has no `if constexpr` on `APPROXIMATION_MODE` |

**Note**: The `APPROX` constant is emitted as `constexpr bool APPROX = false;` in the generated `chlkc_descriptors.h` file (see `tt_metal/jit_build/genfiles.cpp:394`), based on the `math_approx_mode` field of `ComputeConfig`. Since `get_op_approx_mode(ATANH)` returns `false`, `APPROX=false` is propagated through the API header `atanh.h` into the LLK and SFPU layers. However, the `calculate_atanh` kernel does not contain any branching on `APPROXIMATION_MODE`, so the behavior is identical regardless of the value.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/atanh.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. **Compute kernel** (`eltwise_sfpu.cpp`) invokes `SFPU_OP_CHAIN_0` which expands to `atanh_tile(0)`.
2. **API Header** (`atanh.h`): `atanh_tile(idst)` calls `MATH((llk_math_eltwise_unary_sfpu_atanh<APPROX>(idst)))`.
3. **LLK Dispatch** (`llk_math_eltwise_unary_sfpu_atanh.h`): `llk_math_eltwise_unary_sfpu_atanh<APPROXIMATE, 8>(dst_index, VectorMode::RC)` calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_atanh<APPROXIMATE, 8>, dst_index, VectorMode::RC)`.
4. **Parameters Dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): Sets up the DEST write address, stalls for SFPU readiness, iterates over 4 faces in `VectorMode::RC`, calling `calculate_atanh<false, 8>()` per face, with `SETRWC` (Wormhole) or `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` (Blackhole) between faces.
5. **Core SFPU** (`ckernel_sfpu_atanh.h`): `calculate_atanh<false, 8>()` executes 8 iterations per face, computing `atanh(x) = 0.5 * (ln(1+x) - ln(1-x))` using IEEE 754 exponent decomposition and a cubic minimax polynomial for `ln(m)`.

**Init chain**: `atanh_tile_init()` -> `llk_math_eltwise_unary_sfpu_atanh_init<APPROX>()` -> `llk_math_eltwise_unary_sfpu_init<SfpuType::atanh, APPROXIMATE>(sfpu::atanh_init<APPROXIMATE>)` -> (1) `_llk_math_eltwise_unary_sfpu_init_<SfpuType::atanh>()` which configures SFPU config reg, address modes, and resets counters, then (2) `atanh_init<false>()` which loads the cubic polynomial coefficients into `vConstFloatPrgm0/1/2`.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default) -- all 4 faces of the tile are processed (full 32x32 = 1024 elements).
- **Operation invocation**: The params dispatch loops over 4 faces. For each face, it calls `calculate_atanh<false, 8>()` which internally iterates 8 times (ITERATIONS=8). Total: 4 faces x 8 iterations = 32 sfpi iterations covering all 1024 elements.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces on Wormhole / `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` on Blackhole). On Wormhole, between faces the params dispatch issues two `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` instructions (each advancing by 8 sfpi rows), which together advance past the face boundary. On Blackhole, `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice for the equivalent effect.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::vInt`, `sfpi::dst_reg`, `sfpi::exexp`, `sfpi::setexp`, `sfpi::int32_to_float`). Style A applies.

The Wormhole and Blackhole implementations are identical.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h

namespace ckernel::sfpu {

// Implementation notes, see the original file for more details
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_atanh() { // APPROXIMATION_MODE=false, ITERATIONS=8
    constexpr float c3 = 0x2.44734p-4f;  // ~0.1416
    constexpr float ln2 = 0.6931471805599453f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST row pair

        // a = 1 + x, b = 1 - x (both positive for |x| < 1)
        sfpi::vFloat a = x + sfpi::vConst1; // SFPMAD: x * 1.0 + 1.0
        sfpi::vFloat b = -x + sfpi::vConst1; // SFPMAD: (-x) * 1.0 + 1.0

        // ln(a): decompose a = 2^ea * ma, ma in [1, 2)
        sfpi::vInt ea = sfpi::exexp(a); // SFPEXEXP: extract debiased exponent of a
        sfpi::vFloat ma = sfpi::setexp(a, 127); // SFPSETEXP: set exponent to 127 (bias), giving mantissa in [1,2)
        // P(ma) = c0 + ma*(c1 + ma*(c2 + ma*c3))  -- Horner's method
        sfpi::vFloat pa = ma * c3 + sfpi::vConstFloatPrgm2; // SFPMAD: ma * c3 + c2
        pa = pa * ma + sfpi::vConstFloatPrgm1; // SFPMAD: pa * ma + c1
        pa = pa * ma + sfpi::vConstFloatPrgm0; // SFPMAD: pa * ma + c0
        sfpi::vFloat ln_a = sfpi::int32_to_float(ea, 0) * ln2 + pa; // SFPCAST + SFPMAD: float(ea) * ln2 + pa

        // ln(b): decompose b = 2^eb * mb, mb in [1, 2)
        sfpi::vInt eb = sfpi::exexp(b); // SFPEXEXP: extract debiased exponent of b
        sfpi::vFloat mb = sfpi::setexp(b, 127); // SFPSETEXP: set exponent to 127
        sfpi::vFloat pb = mb * c3 + sfpi::vConstFloatPrgm2; // SFPMAD: mb * c3 + c2
        pb = pb * mb + sfpi::vConstFloatPrgm1; // SFPMAD: pb * mb + c1
        pb = pb * mb + sfpi::vConstFloatPrgm0; // SFPMAD: pb * mb + c0
        sfpi::vFloat ln_b = sfpi::int32_to_float(eb, 0) * ln2 + pb; // SFPCAST + SFPMAD: float(eb) * ln2 + pb

        // atanh(x) = 0.5 * (ln(a) - ln(b))
        sfpi::vFloat result = (ln_a - ln_b) * 0.5f; // SFPMAD (sub) + SFPMAD (mul by 0.5)

        sfpi::dst_reg[0] = result; // SFPSTORE: write result back to current DEST row pair
        sfpi::dst_reg++; // advance DEST pointer by 1 sfpi row (= 2 physical DEST rows = 32 elements)
    }
}

template <bool APPROXIMATION_MODE>
inline void atanh_init() { // APPROXIMATION_MODE=false
    // Cubic polynomial coefficients for ln(m) on [1, 2)
    sfpi::vConstFloatPrgm0 = -0x1.952992p+0f;  // c0 ~ -1.5828, loaded via SFPLOADI to LREG[0]
    sfpi::vConstFloatPrgm1 = 0x2.4f5388p+0f;   // c1 ~  2.3110, loaded via SFPLOADI to LREG[1]
    sfpi::vConstFloatPrgm2 = -0xd.e712ap-4f;   // c2 ~ -0.8691, loaded via SFPLOADI to LREG[2]
}

}  // namespace ckernel::sfpu
```

### SFPU Instructions Used

| Instruction / Intrinsic | SFPU Instruction | Description |
|-------------------------|------------------|-------------|
| `sfpi::dst_reg[0]` (read) | `SFPLOAD` | Loads 32 elements (2 physical rows x 16 elements) from the current DEST row pair into an LREG for SFPU processing |
| `sfpi::dst_reg[0] = result` (write) | `SFPSTORE` | Stores 32 elements from an LREG back to the current DEST row pair |
| `sfpi::dst_reg++` | (address increment) | Advances the SFPU DEST pointer by 1 sfpi row (= 2 physical DEST rows, due to SFP_DESTREG_STRIDE=2) |
| `vFloat + vFloat`, `vFloat * float + vFloat` | `SFPMAD` | Fused multiply-add: `a * b + c`. All vFloat additions are compiled as `SFPMAD(a, 1.0, b)`. Each Horner step is a single SFPMAD |
| `-x` (vFloat negation) | `SFPMAD` | Compiled as `SFPMAD(x, -1.0, 0.0)` or sign-bit manipulation |
| `sfpi::exexp(a)` | `SFPEXEXP` | Extracts the debiased IEEE 754 exponent from a float value as a signed integer. With `SFPEXEXP_MOD1_DEBIAS`, the bias (127) is subtracted, yielding the true exponent `e` |
| `sfpi::setexp(a, 127)` | `SFPSETEXP` | Replaces the exponent field of the float with 127 (the IEEE bias), effectively isolating the mantissa `m` in [1, 2) |
| `sfpi::int32_to_float(ea, 0)` | `SFPCAST` | Converts a 32-bit signed integer to an IEEE 754 float. The second argument `0` selects round-to-nearest-even mode |
| `sfpi::vConstFloatPrgm0/1/2 = ...` | `SFPLOADI` | Loads an immediate floating-point constant into the specified programmable constant LREG (LREG[0], LREG[1], LREG[2]) |
| `sfpi::vConst1` | (hardware constant) | The constant 1.0, available as a built-in SFPU constant (no instruction emitted for loading) |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST rows** (current tile) | Input/output data. Each iteration reads 32 elements from the current DEST row pair, computes atanh, and writes back. 4 faces x 8 iterations = 32 row pairs = full tile |
| **LREG[0]** (`vConstFloatPrgm0`) | Polynomial coefficient c0 = -1.5828. Loaded once during `atanh_init()` |
| **LREG[1]** (`vConstFloatPrgm1`) | Polynomial coefficient c1 = 2.3110. Loaded once during `atanh_init()` |
| **LREG[2]** (`vConstFloatPrgm2`) | Polynomial coefficient c2 = -0.8691. Loaded once during `atanh_init()` |
| **LREG[3]** (`vConst1`) | Hardware constant 1.0. Used for computing `a = 1+x` and `b = 1-x` |
| **Temporary LREGs** | The SFPU compiler allocates temporary LREGs for intermediate values (`x`, `a`, `b`, `ea`, `ma`, `pa`, `ln_a`, `eb`, `mb`, `pb`, `ln_b`, `result`). The exact allocation is compiler-dependent, but the SFPU has 4 general-purpose LREGs (LREG[0]-LREG[3]) plus the dedicated constant registers. Since LREG[0]-LREG[2] are occupied by polynomial coefficients, the compiler must carefully reuse LREG[3] and spill intermediates back to DEST or restructure the computation to minimize live register pressure |

### Address Mode Configuration

The address mode is configured during `_llk_math_eltwise_unary_sfpu_init_<SfpuType::atanh>()` which calls `eltwise_unary_sfpu_configure_addrmod<SfpuType::atanh>()`.

Since `SfpuType::atanh` does NOT match any of the specialized `if constexpr` branches in `eltwise_unary_sfpu_configure_addrmod`, only the default address mode is set:

**Wormhole and Blackhole** (identical):
| ADDR_MOD | srca.incr | srcb.incr | dest.incr | Purpose |
|----------|-----------|-----------|-----------|---------|
| `ADDR_MOD_7` | 0 | 0 | 0 | Default SFPU address mode. All increments are zero because the SFPU kernel manages DEST addressing explicitly via `dst_reg++` (within a face) and `SETRWC`/`inc_dst_addr` (between faces) |

The `atanh` operation does not configure `ADDR_MOD_6` or any other address mode slot. The DEST auto-increment is not used; instead, address progression is handled entirely by the SFPI `dst_reg++` abstraction within each face and by the params dispatch layer between faces.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 expansion, and approximation mode for ATANH
   **Key Findings**: ATANH uses `eltwise_sfpu.cpp`, expands to `atanh_tile_init()` / `atanh_tile(0)`, `get_op_approx_mode` returns false (default case), include guard is `SFPU_OP_ATANH_INCLUDE`

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/atanh.h`
   **Reason**: API header defining the tile-level functions
   **Key Findings**: `atanh_tile(idst)` calls `llk_math_eltwise_unary_sfpu_atanh<APPROX>(idst)`, `atanh_tile_init()` calls `llk_math_eltwise_unary_sfpu_atanh_init<APPROX>()`

3. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h`
   **Reason**: LLK dispatch layer bridging API to core SFPU
   **Key Findings**: Dispatches to `_llk_math_eltwise_unary_sfpu_params_` with `calculate_atanh<APPROXIMATE, 8>` and `VectorMode::RC`. Init calls `llk_math_eltwise_unary_sfpu_init<SfpuType::atanh>` with `atanh_init` callback

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h`
   **Reason**: Core SFPU implementation -- the primary analysis target
   **Key Findings**: Computes `atanh(x) = 0.5 * (ln(1+x) - ln(1-x))` via IEEE 754 exponent decomposition and cubic minimax polynomial for ln(m). Uses SFPI abstractions throughout. No branching on APPROXIMATION_MODE. WH and BH implementations identical

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch layer -- handles per-face iteration and DEST address management
   **Key Findings**: VectorMode::RC loops over 4 faces, calls SFPU function per face, uses TTI_SETRWC for face address advancement on Wormhole

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Init and address mode configuration
   **Key Findings**: `eltwise_unary_sfpu_configure_addrmod<SfpuType::atanh>` only sets ADDR_MOD_7 with all-zero increments (no specialized branch for atanh)

7. **File**: `runtime/sfpi/include/sfpi_lib.h`
   **Reason**: SFPI intrinsic-to-builtin mapping for exexp, setexp, int32_to_float
   **Key Findings**: `exexp()` maps to `__builtin_rvtt_sfpexexp` (SFPEXEXP with DEBIAS), `setexp(v, 127)` maps to `__builtin_rvtt_sfpsetexp_i` (SFPSETEXP), `int32_to_float(v, 0)` maps to `__builtin_rvtt_sfpcast` with RNE mode (SFPCAST)

8. **File**: `tt_metal/jit_build/genfiles.cpp`
   **Reason**: Verify how APPROX is defined for compute kernels
   **Key Findings**: Line 394 emits `constexpr bool APPROX = {math_approx_mode};` into `chlkc_descriptors.h`

9. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: Reference for SFPU addressing model, tile geometry, instruction semantics
   **Key Findings**: Confirmed stride-2 addressing, 8 iterations per face, 32 elements per iteration, SFPMAD for float addition
