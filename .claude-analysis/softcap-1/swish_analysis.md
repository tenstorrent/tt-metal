## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `SWISH`
- **Compute kernel**: `eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `swish_tile(0)`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(UnaryOpType::SWISH)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none (no parameterized variant) | `get_op_init_and_func_default()` returns `{"swish_tile_init();", "swish_tile({idst});"}` -- no template parameter in the init/func strings |
| Effective SFPU path | `APPROXIMATION_MODE=false`, `ITERATIONS=8` | `swish_tile_init()` calls `llk_math_eltwise_unary_sfpu_swish_init<APPROX>()` where `APPROX` is JIT-compiled as `false` from `math_approx_mode`. The `calculate_swish` function does not branch on `APPROXIMATION_MODE` -- the same code path is taken regardless. |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/swish.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain
1. **Compute kernel** (`eltwise_sfpu.cpp`): The `SFPU_OP_CHAIN_0` macro expands to `swish_tile(0)`.
2. **API Header** (`swish.h`): `swish_tile(idst)` calls `MATH((llk_math_eltwise_unary_sfpu_swish<APPROX>(idst)))`.
3. **LLK Dispatch** (`llk_math_eltwise_unary_sfpu_swish.h`): `llk_math_eltwise_unary_sfpu_swish<APPROXIMATE>(dst_index)` calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_swish<APPROXIMATE, ITERATIONS=8>, dst_index, VectorMode::RC)`.
4. **Parameters Dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): `_llk_math_eltwise_unary_sfpu_params_` sets up DEST addressing, stalls for SFPU readiness, and calls `calculate_swish<false, 8>()` once per face (4 times for `VectorMode::RC`), with `SETRWC` between faces.
5. **Core SFPU** (`ckernel_sfpu_swish.h`): `calculate_swish<false, 8>()` iterates 8 times per face, processing 32 elements per iteration via SFPI `dst_reg` abstractions.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default) -- all 4 faces of the tile are processed. The dispatch loops `for (int face = 0; face < 4; face++)`, calling the SFPU function once per face, then advancing the DEST write pointer by one face stride via `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` twice (on Wormhole) or `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which calls `math::inc_dst_addr<8>()` twice (on Blackhole). Both advance by 16 physical DEST rows = 1 face.
- **Operation invocation**: `calculate_swish<false, 8>()` is called 4 times (once per face). Each invocation runs an internal loop of 8 iterations, processing 32 elements per iteration (2 physical rows x 16 elements/row). Total: 4 x 8 x 32 = 1024 elements = full tile.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). The address mode used is `ADDR_MOD_7` on both Wormhole and Blackhole, configured with all increments set to 0 (`srca.incr=0`, `srcb.incr=0`, `dest.incr=0`). Since the kernel uses SFPI `dst_reg++` for intra-face advancement (which directly manipulates the RWC pointer), the address mode auto-increment is not needed.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `v_if`/`v_endif`, `sfpi::abs`), so Style A (inline-commented source) is used. The Wormhole and Blackhole implementations are identical.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h

namespace ckernel {
namespace sfpu {

// Implementation notes, see the original file for more details

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_swish() { // APPROXIMATION_MODE=false, ITERATIONS=8
    // Polynomial coefficients for sigmoid(t) over [0, 2.5]
    constexpr float c1 = 0.2533f;      // degree-1 coeff
    constexpr float c2 = -0.01479f;    // degree-2 coeff
    constexpr float c3 = -0.00747f;    // degree-3 coeff

    // Linear segment coefficients for [2.5, 5.0]
    constexpr float lin_slope = 0.0276f;
    constexpr float lin_offset = 0.855f;

    // Breakpoints
    constexpr float bp1 = 2.5f;
    constexpr float bp2 = 5.0f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST row pair into LREG

        sfpi::vFloat ax = sfpi::abs(x);    // SFPABS (SFPABS_MOD1_FLOAT): |x|
        // Horner polynomial: sig_pos = 0.5 + ax * (c1 + ax * (c2 + ax * c3))
        // Each float constant -> SFPLOADI (2x for 32-bit FP32)
        // Each * -> SFPMUL, each + -> SFPADD (optimizer may fuse into SFPMAD chain)
        sfpi::vFloat sig_pos = 0.5f + ax * (c1 + ax * (c2 + ax * c3));

        // Piecewise segment 1: linear approximation for 2.5 < |x| <= 5.0
        v_if(ax > bp1) {                   // SFPXCMP (CC_GT against 2.5f) + SFPPUSHC: enable CC for lanes where |x| > 2.5
            sig_pos = ax * lin_slope + lin_offset; // CC-guarded: SFPLOADI + SFPMUL + SFPADD
        }
        v_endif;                            // SFPPOPC: restore CC state

        // Piecewise segment 2: saturation for |x| > 5.0
        v_if(ax > bp2) {                   // SFPXCMP (CC_GT against 5.0f) + SFPPUSHC
            sig_pos = sfpi::vConst1;        // CC-guarded: load constant 1.0 (Fixed Const 2, CREG_IDX_1=10)
        }
        v_endif;                            // SFPPOPC

        // Negative half-plane: sigmoid(x) = 1 - sigmoid(|x|) for x < 0
        v_if(x < 0.0f) {                   // SFPXCMP (CC_LT against 0.0f) + SFPPUSHC
            sig_pos = sfpi::vConst1 - sig_pos; // CC-guarded: SFPMOV (negate sig_pos) + SFPADD with 1.0
        }
        v_endif;                            // SFPPOPC

        // Final multiply: swish(x) = x * sigmoid(x)
        sfpi::dst_reg[0] = x * sig_pos;    // SFPMUL + SFPSTORE: store result to DEST
        sfpi::dst_reg++;                    // Advance RWC by 1 sfpi row (= 2 physical DEST rows = 32 elements)
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

### SFPU Instructions Used

| Instruction | SFPI Abstraction | Description |
|-------------|-----------------|-------------|
| `SFPLOAD` | `sfpi::dst_reg[0]` (read) | Load 32 elements from current DEST row pair into an LREG |
| `SFPSTORE` | `sfpi::dst_reg[0] = ...` (write) | Store 32 elements from LREG back to current DEST row pair |
| `SFPABS` | `sfpi::abs(x)` | Compute absolute value (clears sign bit); uses `SFPABS_MOD1_FLOAT` mode |
| `SFPLOADI` | `vFloat(float_constant)` | Load 16-bit immediate to LREG; emitted twice per 32-bit float constant (high/low halves via `SFPXLOADI` pseudo-instruction). Used for all float literals: `0.5f`, `c1`, `c2`, `c3`, `lin_slope`, `lin_offset`, `bp1`, `bp2`, `0.0f` |
| `SFPMUL` | `vFloat * vFloat` | Floating-point multiply via `__builtin_rvtt_sfpmul`; used for `ax * c3`, `ax * (...)`, `x * sig_pos`, `ax * lin_slope` |
| `SFPADD` | `vFloat + vFloat` | Floating-point add via `__builtin_rvtt_sfpadd`; used for `c2 + (...)`, `c1 + (...)`, `0.5f + (...)`, `(...) + lin_offset`, `vConst1 - sig_pos` (add of negated value) |
| `SFPMOV` | Unary negation `-sig_pos` | Move with sign complement (`SFPMOV_MOD1_COMPSIGN`); used in `vConst1 - sig_pos` which is `vConst1.flt_add(-sig_pos)` |
| `SFPXCMP` | `ax > bp1`, `ax > bp2`, `x < 0.0f` | Scalar float compare-and-set-CC; emitted by `__builtin_rvtt_sfpxfcmps` for each `v_if` condition comparing a vector register against a float scalar |
| `SFPPUSHC` | `v_if(...)` | Push current CC state onto the per-lane CC stack; enables conditional execution for the `v_if` body |
| `SFPPOPC` | `v_endif` | Pop CC state from the per-lane CC stack; restores previous CC state after the `v_if` block |

Note: The SFPI compiler may optimize sequences of `SFPMUL` + `SFPADD` into fused `SFPMAD` instructions. The Horner polynomial evaluation (`0.5 + ax * (c1 + ax * (c2 + ax * c3))`) is a chain of multiply-add operations that is a prime candidate for such fusion.

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST rows** | Input/output tile data. Each iteration processes 2 physical DEST rows (32 elements). SFPLOAD reads from DEST into LREGs; SFPSTORE writes results back. |
| **LREGs (general)** | The SFPI compiler allocates LREGs automatically for intermediate values. Key temporaries: `x` (original input), `ax` (absolute value), `sig_pos` (sigmoid approximation), intermediate polynomial terms, and the final `x * sig_pos` product. Multiple LREGs are in use simultaneously due to the polynomial evaluation requiring several intermediate results. |
| **LREG (constant loads)** | Each float literal (`0.5f`, `0.2533f`, `-0.01479f`, `-0.00747f`, `0.0276f`, `0.855f`, `2.5f`, `5.0f`, `0.0f`) is loaded into an LREG via `SFPLOADI` pairs (two 16-bit immediates to form a 32-bit FP32 value). The compiler manages register allocation and may reuse LREGs after their values are consumed. |
| **Constant registers** | `vConst1` maps to Fixed Const 2 (CREG_IDX_1 = index 10), which holds the value `1.0` (FP32 `0x3F800000`). Accessed directly without an SFPLOADI; used for the saturation branch (`sig_pos = 1.0`) and the negative-half computation (`1.0 - sig_pos`). |
| **CC stack** | Three `v_if`/`v_endif` pairs each push/pop one CC stack entry. Maximum CC stack depth during execution is 1 (no nesting). Each `v_if` uses `SFPXCMP` to set CC.Res per lane, then `SFPPUSHC` to save state; `v_endif` uses `SFPPOPC` to restore. |

### Address Mode Configuration

The SFPU address mode for swish is configured during `_llk_math_eltwise_unary_sfpu_init_<SfpuType::swish>()`, which calls `eltwise_unary_sfpu_configure_addrmod<SfpuType::swish>()`.

Since `SfpuType::swish` does not match any of the special-cased `constexpr if` branches (only `topk_local_sort`, `typecast`, `unary_max`, `unary_min`, `signbit`, etc. have special handling), only the default address mode is configured:

**ADDR_MOD_7** (used on both Wormhole and Blackhole):
```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}
```

All three increments are zero because the SFPI `dst_reg++` abstraction directly manipulates the Read-Write Counter (RWC) to advance through DEST rows -- the hardware auto-increment mechanism provided by ADDR_MOD is not needed. The inter-face advancement is handled by explicit `SETRWC` instructions emitted by the parameters dispatch layer between face invocations.

This configuration is identical across Wormhole and Blackhole hardware.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 expansion, and approximation mode for SWISH
   **Key Findings**: SWISH uses `eltwise_sfpu.cpp`, expands to `swish_tile_init()` / `swish_tile({idst})`, uses `SFPU_OP_SWISH_INCLUDE` macro, and `get_op_approx_mode` returns `false` (default case)

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/swish.h`
   **Reason**: API header exposing `swish_tile()` and `swish_tile_init()` to compute kernels
   **Key Findings**: `swish_tile(idst)` forwards to `llk_math_eltwise_unary_sfpu_swish<APPROX>(idst)` via MATH() macro; `swish_tile_init()` forwards to `llk_math_eltwise_unary_sfpu_swish_init<APPROX>()`

3. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h`
   **Reason**: LLK dispatch layer bridging API to ckernel SFPU implementation
   **Key Findings**: Dispatches to `_llk_math_eltwise_unary_sfpu_params_` with `calculate_swish<APPROXIMATE, 8>` and `VectorMode::RC`. WH and BH implementations are identical.

4. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h`
   **Reason**: Core SFPU implementation of swish
   **Key Findings**: Implements `swish(x) = x * sigmoid(x)` using a piecewise sigmoid approximation: degree-3 polynomial for |x| <= 2.5, linear interpolation for 2.5 < |x| <= 5.0, saturation to 1.0 for |x| > 5.0, with symmetry `sigmoid(x) = 1 - sigmoid(|x|)` for x < 0. Uses pure SFPI abstractions. WH and BH implementations are identical.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch layer controlling face iteration and DEST addressing
   **Key Findings**: For `VectorMode::RC`, iterates 4 faces with `SETRWC`-based advancement between faces. WH version uses inline `TTI_SETRWC` instructions; BH version calls `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()`.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Init function and address mode configuration
   **Key Findings**: `SfpuType::swish` uses default ADDR_MOD_7 with all increments = 0. No special-case address mode is configured for swish.

7. **File**: `runtime/sfpi/include/sfpi.h`
   **Reason**: SFPI C++ abstraction layer defining vFloat operations, comparison operators, and v_if/v_endif CC management
   **Key Findings**: `vFloat + vFloat` -> `SFPADD`, `vFloat * vFloat` -> `SFPMUL`, `vFloat(float)` -> `SFPXLOADI`, comparison -> `SFPXCMP`, `v_if` -> `SFPPUSHC`, `v_endif` -> `SFPPOPC`, unary minus -> `SFPMOV(COMPSIGN)`, `vConst1` -> Fixed Const 2 (1.0)

8. **File**: `runtime/sfpi/include/sfpi_lib.h`
   **Reason**: SFPI library functions including `abs()`
   **Key Findings**: `sfpi::abs(vFloat)` emits `SFPABS` with `SFPABS_MOD1_FLOAT` mode

9. **File**: `tt_metal/jit_build/genfiles.cpp`
   **Reason**: Understand how APPROX compile-time constant is set
   **Key Findings**: `APPROX` is emitted as `constexpr bool APPROX = {value}` in `chlkc_descriptors.h` during JIT build, derived from `math_approx_mode` in compute config

10. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Authoritative reference for SFPU hardware model, tile/face geometry, addressing, and instruction semantics
    **Key Findings**: Confirmed stride-2 model, 8 iterations per face, 32 elements per iteration, ADDR_MOD configuration patterns, CC mechanism details
