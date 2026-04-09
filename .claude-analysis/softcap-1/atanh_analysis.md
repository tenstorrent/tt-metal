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
| Template parameter (SFPU_OP_CHAIN) | none (non-parameterized) | `get_op_init_and_func_default()` returns `atanh_tile_init()` / `atanh_tile({idst})` with no template arguments; the API header `atanh.h` passes `APPROX` (JIT-generated `constexpr bool`, value `false`) |
| Effective SFPU path | `APPROXIMATION_MODE=false`, but the kernel has no `if constexpr` branching on this parameter -- both paths execute identical code | `calculate_atanh<APPROXIMATION_MODE, ITERATIONS>` in `ckernel_sfpu_atanh.h` |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/atanh.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h` (identical on Blackhole) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h` (identical on Blackhole) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Wormhole); `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Blackhole) |

### Call Chain

1. **Compute kernel** (`eltwise_sfpu.cpp`): The `SFPU_OP_CHAIN_0` macro expands to `atanh_tile_init(); atanh_tile(0);` (init called once, tile call per tile).
2. **API header** (`atanh.h`): `atanh_tile(idst)` calls `llk_math_eltwise_unary_sfpu_atanh<APPROX>(idst)` via the `MATH()` macro which routes to the math RISC-V thread.
3. **LLK dispatch** (`llk_math_eltwise_unary_sfpu_atanh.h`): `llk_math_eltwise_unary_sfpu_atanh<APPROXIMATE, ITERATIONS=8>(dst_index, VectorMode::RC)` calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_atanh<APPROXIMATE, ITERATIONS>, dst_index, vector_mode)`.
4. **Parameters dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): Sets DEST write address, stalls until SFPU is ready, then loops over 4 faces (VectorMode::RC), calling `calculate_atanh<false, 8>()` once per face, with `TTI_SETRWC` advancing the DEST address between faces.
5. **Core SFPU** (`ckernel_sfpu_atanh.h`): `calculate_atanh<false, 8>()` processes 8 SFPI iterations (one face), computing `atanh(x) = 0.5 * (ln(1+x) - ln(1-x))` using IEEE 754 exponent decomposition and a cubic minimax polynomial for `ln(m)`.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default) -- all 4 faces of the tile are processed. The dispatch loops `for (int face = 0; face < 4; face++)`, calling the SFPU function once per face.
- **Operation invocation**: `calculate_atanh<false, 8>()` is called 4 times (once per face), each invocation processing 8 SFPI iterations = 256 elements per face.
- **DEST address progression**: Standard DEST progression. On Wormhole, `ADDR_MOD_7` is configured with all increments = 0 (the SFPU kernel manages its own `dst_reg++` within the loop). Between faces, the params dispatch calls `TTI_SETRWC(CLR_NONE, CR_D, 8, ..., SET_D)` twice per face to advance by 16 physical DEST rows (= 1 face). On Blackhole, the equivalent is `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which calls `math::inc_dst_addr<8>()` twice. Within a face, `dst_reg++` advances 1 sfpi row (= 2 physical DEST rows = 32 elements) per iteration, covering all 8 iterations (ITERATIONS=8 per face).

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::vInt`, `sfpi::dst_reg`, `sfpi::exexp`, `sfpi::setexp`, `sfpi::int32_to_float`), so **Style A** (inline-commented source) is used.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h

// atanh(x) = 0.5 * ln((1+x)/(1-x)) = 0.5 * (ln(1+x) - ln(1-x))
// Valid for |x| < 1.
//
// Implementation notes, see the original file for more details
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_atanh() { // APPROXIMATION_MODE=false, ITERATIONS=8
    constexpr float c3 = 0x2.44734p-4f;  // ~0.1416, cubic coefficient for ln(m) minimax polynomial
    constexpr float ln2 = 0.6931471805599453f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST position

        // a = 1 + x, b = 1 - x (both positive for |x| < 1)
        sfpi::vFloat a = x + sfpi::vConst1;   // SFPMAD: x * 1.0 + 1.0
        sfpi::vFloat b = -x + sfpi::vConst1;  // SFPMAD: x * (-1.0) + 1.0

        // ln(a): decompose a = 2^ea * ma, ma in [1, 2)
        sfpi::vInt ea = sfpi::exexp(a);        // SFPEXEXP with DEBIAS: extract biased exponent, subtract 127
        sfpi::vFloat ma = sfpi::setexp(a, 127); // SFPSETEXP: set exponent to 127 (= bias), yielding mantissa in [1, 2)
        // P(ma) = c0 + ma*(c1 + ma*(c2 + ma*c3)) -- Horner form cubic minimax polynomial
        sfpi::vFloat pa = ma * c3 + sfpi::vConstFloatPrgm2;  // SFPMAD: ma * c3 + c2 (c2 ~ -0.8691)
        pa = pa * ma + sfpi::vConstFloatPrgm1;                // SFPMAD: pa * ma + c1 (c1 ~ 2.3110)
        pa = pa * ma + sfpi::vConstFloatPrgm0;                // SFPMAD: pa * ma + c0 (c0 ~ -1.5828)
        sfpi::vFloat ln_a = sfpi::int32_to_float(ea, 0) * ln2 + pa; // SFPCAST(int->fp32) then SFPMAD: ea_f * ln2 + pa

        // ln(b): decompose b = 2^eb * mb, mb in [1, 2)
        sfpi::vInt eb = sfpi::exexp(b);        // SFPEXEXP with DEBIAS
        sfpi::vFloat mb = sfpi::setexp(b, 127); // SFPSETEXP: normalize mantissa to [1, 2)
        sfpi::vFloat pb = mb * c3 + sfpi::vConstFloatPrgm2;  // SFPMAD: Horner step 1
        pb = pb * mb + sfpi::vConstFloatPrgm1;                // SFPMAD: Horner step 2
        pb = pb * mb + sfpi::vConstFloatPrgm0;                // SFPMAD: Horner step 3
        sfpi::vFloat ln_b = sfpi::int32_to_float(eb, 0) * ln2 + pb; // SFPCAST then SFPMAD

        // atanh(x) = 0.5 * (ln(a) - ln(b))
        sfpi::vFloat result = (ln_a - ln_b) * 0.5f; // SFPMAD (subtract) then SFPMAD (multiply by 0.5)

        sfpi::dst_reg[0] = result; // SFPSTORE: write 32 elements back to current DEST position
        sfpi::dst_reg++;           // Advance DEST pointer by 1 sfpi row (= 2 physical DEST rows = 32 elements)
    }
}

template <bool APPROXIMATION_MODE>
inline void atanh_init() {
    // Cubic polynomial coefficients for ln(m) on [1, 2) -- loaded into programmable constant registers
    sfpi::vConstFloatPrgm0 = -0x1.952992p+0f;  // c0 ~ -1.5828 -> SFPCONFIG to Prog Const register
    sfpi::vConstFloatPrgm1 = 0x2.4f5388p+0f;   // c1 ~  2.3110 -> SFPCONFIG to Prog Const register
    sfpi::vConstFloatPrgm2 = -0xd.e712ap-4f;   // c2 ~ -0.8691 -> SFPCONFIG to Prog Const register
}
```

### SFPU Instructions Used

| Instruction | Opcode | Usage in Kernel | Description |
|-------------|--------|-----------------|-------------|
| **SFPLOAD** | 0x70 | `sfpi::dst_reg[0]` (read) | Loads 32 elements from the current DEST row pair into an LREG |
| **SFPSTORE** | 0x72 | `sfpi::dst_reg[0] = result` | Stores 32 elements from an LREG back to the current DEST row pair |
| **SFPMAD** | 0x84 | `+`, `-`, `*` operators on `vFloat` | Fused multiply-add: `VD = VA * VB + VC`. Used for all float arithmetic: addition is `a * 1.0 + b`, subtraction is `a * 1.0 - b` (via sign inversion), multiplication is `a * b + 0.0`. 10 SFPMAD instructions per iteration for the polynomial evaluations, additions, and final scaling |
| **SFPEXEXP** | 0x77 | `sfpi::exexp(a)`, `sfpi::exexp(b)` | Extracts the exponent field of an FP32 value and subtracts the IEEE 754 bias (127), yielding the unbiased integer exponent. Used twice per iteration (once for `a`, once for `b`) |
| **SFPSETEXP** | 0x7C | `sfpi::setexp(a, 127)`, `sfpi::setexp(b, 127)` | Replaces the exponent field of an FP32 value with the given value (127 = bias), effectively normalizing the mantissa to `[1, 2)`. Used twice per iteration |
| **SFPCAST** | 0x90 | `sfpi::int32_to_float(ea, 0)`, `sfpi::int32_to_float(eb, 0)` | Converts a signed 32-bit integer to FP32. The `round_mode=0` parameter selects round-to-nearest-even (RNE). Used twice per iteration to convert the integer exponent to float for the `e * ln(2)` term |
| **SFPCONFIG** | 0x8D | `atanh_init()`: setting `vConstFloatPrgm0/1/2` | Writes to the programmable constant registers (Prog Const 1, 2, 3) to load the polynomial coefficients for ln(m). Called once during initialization |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST rows** | Source and destination for tile data. Each SFPI iteration accesses 2 physical DEST rows (32 elements) via `dst_reg[0]`. The SFPLOAD/SFPSTORE pair reads input `x` and writes back `atanh(x)` in place. |
| **LREGs (LREG0-LREG7)** | General-purpose 32-bit vector registers used as temporaries for all intermediate computations. The compiler allocates them automatically for SFPI code. Variables `x`, `a`, `b`, `ea`, `ma`, `pa`, `ln_a`, `eb`, `mb`, `pb`, `ln_b`, `result` are mapped to LREGs by the compiler. Given the 13 live variables, register spilling to DEST/SrcS is likely needed for some intermediates. |
| **Prog Const 1** (`vConstFloatPrgm0`) | Polynomial coefficient `c0 = -0x1.952992p+0f` (~-1.5828). Set by `atanh_init()`. |
| **Prog Const 2** (`vConstFloatPrgm1`) | Polynomial coefficient `c1 = 0x2.4f5388p+0f` (~2.3110). Set by `atanh_init()`. |
| **Prog Const 3** (`vConstFloatPrgm2`) | Polynomial coefficient `c2 = -0xd.e712ap-4f` (~-0.8691). Set by `atanh_init()`. |
| **Fixed Const 2** (`vConst1`) | Hardware constant `1.0` (FP32 `0x3F800000`). Used for computing `a = x + 1` and `b = -x + 1`. |

### Address Mode Configuration

The address mode for `SfpuType::atanh` is configured in `eltwise_unary_sfpu_configure_addrmod<SfpuType::atanh>()`, which is called during `_llk_math_eltwise_unary_sfpu_init_<SfpuType::atanh>()`.

Since `SfpuType::atanh` does not match any of the special-case `if constexpr` branches (which handle `topk_local_sort`, `typecast`, `unary_max/min`, `signbit`, and on Blackhole also `reciprocal`), only the default address mode is configured:

**Wormhole B0 and Blackhole (identical for atanh):**

| Address Mode | srca.incr | srcb.incr | dest.incr | Purpose |
|-------------|-----------|-----------|-----------|---------|
| `ADDR_MOD_7` | 0 | 0 | 0 | Default SFPU address mode. All auto-increments are disabled because SFPU kernels using SFPI abstractions manage DEST address progression internally via `dst_reg++`. The `SETRWC` instructions in the params dispatch handle inter-face address advancement. |

No additional address modes (`ADDR_MOD_6`, etc.) are configured for this operation. The inter-face DEST address advancement is handled by the params dispatch layer:
- **Wormhole**: Two `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` calls per face (advancing by 8+8=16 physical DEST rows = 1 face)
- **Blackhole**: Two `math::inc_dst_addr<8>()` calls per face (equivalent)

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 expansion, and approximation mode for ATANH
   **Key Findings**: ATANH uses `eltwise_sfpu.cpp`, expands to `atanh_tile_init()` / `atanh_tile(0)`, `get_op_approx_mode()` returns `false` (default case), `SFPU_OP_ATANH_INCLUDE` macro enables the split include

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/atanh.h`
   **Reason**: API header defining `atanh_tile()` and `atanh_tile_init()`
   **Key Findings**: `atanh_tile(idst)` calls `llk_math_eltwise_unary_sfpu_atanh<APPROX>(idst)`, `atanh_tile_init()` calls `llk_math_eltwise_unary_sfpu_atanh_init<APPROX>()`

3. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h`
   **Reason**: LLK dispatch layer connecting API to core SFPU implementation
   **Key Findings**: Init calls `llk_math_eltwise_unary_sfpu_init<SfpuType::atanh, APPROXIMATE>(sfpu::atanh_init<APPROXIMATE>)`, tile function calls `_llk_math_eltwise_unary_sfpu_params_` with `calculate_atanh` as the callable

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h`
   **Reason**: Core SFPU implementation -- the primary analysis target
   **Key Findings**: Implements `atanh(x) = 0.5 * (ln(1+x) - ln(1-x))` using IEEE 754 exponent decomposition (`exexp`/`setexp`) and a cubic minimax polynomial for `ln(m)` on `[1, 2)`. Coefficients loaded via programmable constant registers. No branching on APPROXIMATION_MODE -- single code path.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch -- manages face iteration and DEST addressing
   **Key Findings**: VectorMode::RC processes all 4 faces with `SETRWC` between faces. Standard SFPU dispatch pattern.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Address mode configuration for SFPU operations
   **Key Findings**: `SfpuType::atanh` gets only `ADDR_MOD_7` (all increments = 0). No special-case address modes.

7. **File**: `runtime/sfpi/include/sfpi_lib.h`
   **Reason**: SFPI intrinsic-to-instruction mapping verification
   **Key Findings**: `exexp()` maps to `__builtin_rvtt_sfpexexp` (SFPEXEXP), `setexp()` maps to `__builtin_rvtt_sfpsetexp_i` (SFPSETEXP), `int32_to_float()` maps to `__builtin_rvtt_sfpcast` (SFPCAST)

8. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: Authoritative SFPU hardware model reference
   **Key Findings**: Tile/face geometry, stride-2 addressing model, instruction semantics, register file layout, programmable constant register indices
