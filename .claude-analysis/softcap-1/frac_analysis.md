## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `FRAC`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `frac_tile(0)`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(FRAC)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none | `get_op_init_and_func_default()` returns `frac_tile_init()` / `frac_tile(idst)` with no explicit template argument; the API header `frac.h` passes `APPROX` (which resolves to `false` from ComputeConfig) |
| Effective SFPU path | `APPROXIMATION_MODE=false` in `calculate_frac<false, 8>()` | The kernel does not branch on `APPROXIMATION_MODE` -- the template parameter is accepted but unused, so both `true` and `false` produce identical code paths |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/frac.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_frac.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_frac.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain
1. **Compute kernel** calls `frac_tile(0)` (via `SFPU_OP_CHAIN_0` macro expansion).
2. **API Header** (`frac.h`): `frac_tile(idst)` expands to `MATH((llk_math_eltwise_unary_sfpu_frac<APPROX>(idst)))`.
3. **LLK Dispatch** (`llk_math_eltwise_unary_sfpu_frac.h`): `llk_math_eltwise_unary_sfpu_frac<APPROXIMATE>(dst_index)` calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(calculate_frac<APPROXIMATE, 8>, dst_index, VectorMode::RC)`.
4. **Parameters Dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): `_llk_math_eltwise_unary_sfpu_params_` sets the DEST write address, stalls until SFPU is ready, then loops over 4 faces calling `calculate_frac<false, 8>()` once per face with `SETRWC` between faces.
5. **Core SFPU** (`ckernel_sfpu_frac.h`): `calculate_frac<false, 8>()` processes 8 SFPU iterations (one face) computing `frac(x) = x - trunc(x)` via IEEE 754 mantissa bit masking.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default) -- all 4 faces of the tile are processed.
- **Operation invocation**: The params dispatch calls `calculate_frac<false, 8>()` once per face in a `for (int face = 0; face < 4; face++)` loop. Each invocation processes 8 SFPU iterations (ITERATIONS=8), covering one 16x16 face (256 elements).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). On Wormhole, `TTI_SETRWC` is called directly with stride 8 twice between faces. On Blackhole, `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice between faces. Both achieve the same net advance of 16 physical DEST rows per face boundary.

### Annotated SFPU Kernel Source

This kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::vInt`, `sfpi::vUInt`, `sfpi::dst_reg`, `v_if`/`v_endif`, `sfpi::exexp`, `sfpi::reinterpret`), so Style A (inline-commented source) is used. The Wormhole and Blackhole implementations are identical.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_frac.h

namespace ckernel::sfpu {

// frac(x) = x - trunc(x)
// Implementation notes, see the original file for more details
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_frac() { // APPROXIMATION_MODE=false, ITERATIONS=8
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];  // SFPLOAD: load 32 elements from current DEST row pair

        // Default: frac = 0 for integers (exp >= 23)
        sfpi::vFloat trunc_x = x;  // Copy x; if no v_if branch modifies trunc_x, result = x - x = 0

        // Extract unbiased exponent
        sfpi::vInt exp = sfpi::exexp(x);  // SFPEXEXP with DEBIAS: extracts biased exponent and subtracts 127

        // Case 1: |x| < 1 (exp < 0) -- trunc toward zero gives 0
        v_if(exp < 0) { trunc_x = 0.0f; }  // SFPIADD sets CC for exp<0; guarded SFPLOADI loads 0.0f into trunc_x
        v_endif;

        // Case 2: 0 <= exp < 23 (has fractional bits in float32)
        v_if(exp >= 0 && exp < 23) {  // Compound condition: SFPIADD(exp>=0) AND SFPIADD(exp<23), uses CC stack
            // Create bitmask to zero out fractional mantissa bits.
            // IEEE 754 float32 has 23 mantissa bits. For exponent e,
            // the lowest (23 - e) bits are fractional.
            // mask = 0xFFFFFFFF << (23 - exp)
            sfpi::vUInt shift = sfpi::vUInt(23 - exp);  // SFPIADD: compute 23 - exp (integer subtract)
            sfpi::vInt mask = sfpi::vInt(-1) << shift;   // SFPSHFT: shift 0xFFFFFFFF left by (23-exp), vector shift

            // Apply mask to get trunc(x) (round toward zero)
            sfpi::vInt xi = sfpi::reinterpret<sfpi::vInt>(x);    // No instruction: zero-cost type reinterpret
            trunc_x = sfpi::reinterpret<sfpi::vFloat>(xi & mask); // SFPAND: bitwise AND to mask off fractional bits
        }
        v_endif;

        // frac(x) = x - trunc(x)
        sfpi::dst_reg[0] = x - trunc_x;  // SFPADD (or SFPMAD with sign inversion): float subtract, then SFPSTORE
        sfpi::dst_reg++;  // Advance DEST pointer by 1 sfpi row (= 2 physical rows = 32 elements)
    }
}

}  // namespace ckernel::sfpu
```

### SFPU Instructions Used

| Instruction | SFPI Abstraction | Usage in Kernel |
|-------------|-----------------|-----------------|
| `SFPLOAD` | `sfpi::dst_reg[0]` (read) | Loads 32 elements from the current DEST row pair into an LREG |
| `SFPSTORE` | `sfpi::dst_reg[0] = ...` (write) | Stores the computed `frac(x)` result back to the current DEST row pair |
| `SFPEXEXP` | `sfpi::exexp(x)` | Extracts the unbiased exponent from float32 input (subtracts bias 127) |
| `SFPIADD` | `exp < 0`, `exp >= 0`, `exp < 23`, `23 - exp` | Integer add/subtract used for: (1) comparison via CC side-effect (sets CC.Res for conditional branches), (2) computing `23 - exp` shift amount |
| `SFPSHFT` | `vInt(-1) << shift` | Vector left-shift of `0xFFFFFFFF` by a per-lane shift amount to create the mantissa mask |
| `SFPAND` | `xi & mask` | Bitwise AND of the integer-reinterpreted float with the mask to zero out fractional mantissa bits |
| `SFPADD` / `SFPMAD` | `x - trunc_x` | Float subtraction to compute `frac(x) = x - trunc(x)`. The compiler may emit `SFPADD` or fold into `SFPMAD` with sign inversion on the addend |
| `SFPLOADI` | `trunc_x = 0.0f` | Loads the immediate value 0.0f into an LREG (within the `v_if(exp < 0)` branch) |
| `SFPSETCC` | `v_if(...)` conditions | Sets per-lane CC.Res based on comparison results for predicated execution |
| `SFPENCC` | `v_if` / `v_endif` | Enables/disables condition code masking at the start and end of conditional blocks |
| `SFPPUSHC` | `v_if` (nested/compound) | Pushes CC state onto the per-lane CC stack for the compound `exp >= 0 && exp < 23` condition |
| `SFPPOPC` | `v_endif` | Pops CC state from the stack, restoring the previous predication context |
| `SFPCOMPC` | Implicit in `v_if`/`v_endif` | May be used internally by the compiler for CC complement operations between conditional branches |

### SFPU Register Usage

| Register / Resource | Usage |
|---------------------|-------|
| **DEST row pairs** | Input and output storage. Each iteration reads from and writes back to `dst_reg[0]` (the current DEST row pair, 32 elements). `dst_reg++` advances to the next pair after each iteration. |
| **LREG0-LREG3** (estimated) | General-purpose LREGs used by the compiler for intermediate values: `x`, `trunc_x`, `exp`, `shift`, `mask`, `xi`. The exact LREG allocation is determined by the SFPI compiler's register allocator. |
| **CC register** | Per-lane condition code bits used for predicated execution in the two `v_if` blocks. The first `v_if(exp < 0)` tests the sign of the exponent. The second `v_if(exp >= 0 && exp < 23)` uses a compound condition requiring the CC stack. |
| **CC stack** | Used for the compound boolean `exp >= 0 && exp < 23`. The `v_if` macro pushes the first condition result, evaluates the second condition, and combines them via AND on the CC stack. |

### Address Mode Configuration

The `frac` operation uses `SfpuType::frac` for its init. The `eltwise_unary_sfpu_configure_addrmod<SfpuType::frac>()` function (in `llk_math_eltwise_unary_sfpu.h`) does NOT match any of the special-cased `if constexpr` branches (which check for `topk_local_sort`, `typecast`, `unary_max`, `unary_min`, `signbit`, etc.). Therefore, only the default `ADDR_MOD_7` is configured:

**Wormhole B0 and Blackhole (identical):**

| ADDR_MOD | srca.incr | srcb.incr | dest.incr | Purpose |
|----------|-----------|-----------|-----------|---------|
| `ADDR_MOD_7` | 0 | 0 | 0 | Default SFPU address mode -- no auto-increment. DEST addressing is managed entirely by `dst_reg++` within the SFPU kernel (which advances by `SFP_DESTREG_STRIDE=2` physical rows per iteration) and `SETRWC`/`inc_dst_addr` between faces in the params dispatch. |

The `ADDR_MOD_7` configuration is the same on both Wormhole B0 and Blackhole. No additional address modes (e.g., `ADDR_MOD_6`) are configured for this operation.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 expansion, and approximation mode for FRAC
   **Key Findings**: FRAC uses `eltwise_sfpu.cpp`, expands to `frac_tile(idst)`, `frac_tile_init()`, `math_approx_mode=false` (default case), macro `SFPU_OP_FRAC_INCLUDE` is defined

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/frac.h`
   **Reason**: Trace the API header layer for frac_tile and frac_tile_init
   **Key Findings**: `frac_tile(idst)` calls `llk_math_eltwise_unary_sfpu_frac<APPROX>(idst)`, `frac_tile_init()` calls `llk_math_eltwise_unary_sfpu_frac_init<APPROX>()`

3. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_frac.h`
   **Reason**: Trace the LLK dispatch layer
   **Key Findings**: Bridges to `_llk_math_eltwise_unary_sfpu_params_` with `calculate_frac<APPROXIMATE, 8>` as the SFPU function, `VectorMode::RC` default

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_frac.h`
   **Reason**: Core SFPU kernel implementation
   **Key Findings**: SFPI-based kernel implementing `frac(x) = x - trunc(x)` via IEEE 754 mantissa bit masking. Three cases: (1) |x|<1: frac=x, (2) |x|>=2^23: frac=0, (3) otherwise: mask fractional bits and subtract. Uses `exexp`, `SFPSHFT`, `SFPAND`, `v_if` for conditional execution.

5. **File**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_frac.h`
   **Reason**: Verify Blackhole implementation is identical
   **Key Findings**: Identical to Wormhole implementation

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Understand the params dispatch pattern for VectorMode::RC
   **Key Findings**: Loops over 4 faces, calls SFPU function once per face, uses `TTI_SETRWC` with stride 8 twice between faces

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Verify Blackhole params dispatch
   **Key Findings**: Same structure as Wormhole but uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` helper instead of direct `TTI_SETRWC`

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Understand address mode configuration for SfpuType::frac
   **Key Findings**: `frac` does not match any special-cased SfpuType, so only `ADDR_MOD_7` (dest.incr=0) is configured

9. **File**: `runtime/sfpi/include/sfpi_lib.h`
   **Reason**: Verify SFPI intrinsic mappings (exexp, reinterpret, shift, etc.)
   **Key Findings**: `exexp()` maps to `__builtin_rvtt_sfpexexp` with `SFPEXEXP_MOD1_DEBIAS`, `reinterpret` is a zero-cost type cast, shift maps to `__builtin_rvtt_sfpshft_v`

10. **File**: `runtime/sfpi/include/sfpi.h`
    **Reason**: Verify operator mappings for `<<`, `&`, `-` on vector types
    **Key Findings**: `vInt << vUInt` calls `int_shift(vUInt, true)` -> `__builtin_rvtt_sfpshft_v` (SFPSHFT), `vInt & vInt` calls `int_and(b)` -> `__builtin_rvtt_sfpand` (SFPAND), `vFloat - vFloat` calls `flt_add(-b)` -> `__builtin_rvtt_sfpadd` (SFPADD)

11. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Reference for SFPU addressing model, instruction semantics, tile/face geometry
    **Key Findings**: Confirmed stride-2 model, ITERATIONS=8 per face, 32 elements per iteration, SFPEXEXP/SFPAND/SFPSHFT instruction semantics
