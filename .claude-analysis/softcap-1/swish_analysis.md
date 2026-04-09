## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `SWISH`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `swish_tile_init(); swish_tile(0);`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(UnaryOpType::SWISH)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none (uses default) | `get_op_init_and_func_default()` returns `swish_tile_init()` / `swish_tile(0)` with no parameterized template argument |
| Effective SFPU path | `APPROXIMATION_MODE=false`, but the kernel does not branch on `APPROXIMATION_MODE` -- a single code path is always taken regardless | The `calculate_swish` template accepts `APPROXIMATION_MODE` but does not contain any `if constexpr (APPROXIMATION_MODE)` branches |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/swish.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h` (identical on Blackhole: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h`) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h` (identical on Blackhole: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h`) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Blackhole: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`) |

### Call Chain
1. The compute kernel `eltwise_sfpu.cpp` invokes the `SFPU_OP_CHAIN_0` macro, which expands to `swish_tile_init(); swish_tile(0);`.
2. `swish_tile(idst)` (in `swish.h`) calls `llk_math_eltwise_unary_sfpu_swish<APPROX>(idst)` on the math thread via the `MATH(...)` wrapper.
3. `llk_math_eltwise_unary_sfpu_swish<APPROXIMATE>(dst_index)` (in `llk_math_eltwise_unary_sfpu_swish.h`) calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>()` passing `ckernel::sfpu::calculate_swish<APPROXIMATE, 8>` as the callable, along with `dst_index` and `VectorMode::RC`.
4. `_llk_math_eltwise_unary_sfpu_params_()` (in `llk_math_eltwise_unary_sfpu_params.h`) sets up DEST addressing, stalls for SFPU readiness, then loops over 4 faces calling `calculate_swish<false, 8>()` once per face with `SETRWC` advancing between faces.
5. `calculate_swish<false, 8>()` (in `ckernel_sfpu_swish.h`) executes 8 iterations per face, processing 32 elements per iteration (2 physical DEST rows x 16 elements), using SFPI abstractions for all computation.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default). All 4 faces of the tile are processed, covering the full 32x32 = 1024 elements.
- **Operation invocation**: The core `calculate_swish<false, 8>()` function is called 4 times (once per face) inside a `for (int face = 0; face < 4; face++)` loop. Each invocation processes 8 iterations of 32 elements = 256 elements = one complete face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). On Wormhole, the params dispatch uses `TTI_SETRWC` with `p_setrwc::CR_D, 8` twice per face to advance by 16 physical DEST rows (= 1 face). On Blackhole, the equivalent `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice.

### Annotated SFPU Kernel Source

This kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `sfpi::abs`, `v_if`/`v_endif`), so Style A (inline-commented source) is used.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h

namespace ckernel {
namespace sfpu {

// Implementation notes, see the original file for more details

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_swish() { // APPROXIMATION_MODE=false, ITERATIONS=8
    // Polynomial coefficients for sigmoid(t) over [0, 2.5]
    // Fitted to minimize max error at t = 0, 0.5, 1.0, 1.5, 2.0, 2.5
    constexpr float c1 = 0.2533f;
    constexpr float c2 = -0.01479f;
    constexpr float c3 = -0.00747f;

    // Linear segment coefficients for [2.5, 5.0]
    constexpr float lin_slope = 0.0276f;
    constexpr float lin_offset = 0.855f;

    // Breakpoints
    constexpr float bp1 = 2.5f;
    constexpr float bp2 = 5.0f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST rows into LREG

        // Compute sigmoid(|x|) using polynomial for |x| <= 2.5
        sfpi::vFloat ax = sfpi::abs(x); // SFPABS: compute absolute value
        sfpi::vFloat sig_pos = 0.5f + ax * (c1 + ax * (c2 + ax * c3)); // Horner's method: chain of SFPMAD (mul+add via fused multiply-add)

        // Override with linear segment for 2.5 < |x| <= 5.0
        v_if(ax > bp1) { sig_pos = ax * lin_slope + lin_offset; } // SFPPUSHC + SFPMAD(subtract for comparison) + SFPSETCC + guarded SFPMAD(linear)
        v_endif; // SFPPOPC: restore CC state

        // Saturate to 1.0 for |x| > 5.0
        v_if(ax > bp2) { sig_pos = sfpi::vConst1; } // SFPPUSHC + SFPMAD(subtract) + SFPSETCC + guarded SFPLOADI/SFPMOV from const reg
        v_endif; // SFPPOPC

        // For x < 0: sigmoid(x) = 1 - sigmoid(|x|)
        v_if(x < 0.0f) { sig_pos = sfpi::vConst1 - sig_pos; } // SFPPUSHC + SFPSETCC(sign test) + guarded SFPMAD(1.0 - sig_pos)
        v_endif; // SFPPOPC

        // swish(x) = x * sigmoid(x)
        sfpi::dst_reg[0] = x * sig_pos; // SFPMAD(multiply) then SFPSTORE: write 32 elements back to DEST
        sfpi::dst_reg++; // advance DEST address by 1 sfpi row (= 2 physical DEST rows = 32 elements)
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

### SFPU Instructions Used

| Instruction | Description | Usage in swish kernel |
|-------------|-------------|----------------------|
| `SFPLOAD` | Load data from DEST rows into LREG | Implicit in `sfpi::dst_reg[0]` read -- loads 32 elements (2 physical rows) into an LREG for processing |
| `SFPSTORE` | Store data from LREG back to DEST rows | Implicit in `sfpi::dst_reg[0] = ...` write -- stores the computed result back to DEST |
| `SFPABS` | Compute absolute value (clear sign bit) | Used via `sfpi::abs(x)` to compute `|x|` for the piecewise sigmoid approximation |
| `SFPMAD` | Fused multiply-add: VD = VA * VB + VC | Core arithmetic workhorse. Used for: (1) Horner's polynomial evaluation (`c2 + ax * c3`, then `c1 + ax * (...)`, etc.), (2) linear segment `ax * lin_slope + lin_offset`, (3) `vConst1 - sig_pos` (multiply by 1.0 then subtract), (4) final `x * sig_pos` multiply, (5) comparison subtractions for `v_if` conditions (computing `ax - bp1`, `ax - bp2`, `x - 0.0`) |
| `SFPLOADI` | Load 16-bit immediate to LREG | Used to materialize float constants (0.5f, c1, c2, c3, lin_slope, lin_offset, bp1, bp2, 0.0f) into LREGs. Two `SFPLOADI` instructions per 32-bit float constant (hi16 + lo16) |
| `SFPPUSHC` | Push CC state onto stack | Used at the start of each `v_if` block to save the current CC state before the conditional |
| `SFPPOPC` | Pop CC state from stack | Used at `v_endif` to restore the CC state after each conditional block |
| `SFPSETCC` | Set CC.Res based on comparison result | Used to evaluate comparison conditions (`ax > bp1`, `ax > bp2`, `x < 0.0f`) by testing the sign of the subtraction result |
| `SFPENCC` | Enable/disable condition code evaluation | Used at destruction of `__vCCCtrl` objects to restore unconditional execution after the `v_if`/`v_endif` block |
| `SFPCOMPC` | Complement CC.Res | Not directly used in this kernel (no `v_else` branches), but may be emitted by the SFPI compiler as part of CC stack management |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST rows (via `dst_reg[0]`)** | Input/output data. Each iteration loads 32 elements (2 physical DEST rows x 16 elements) for processing and stores the result back. The `dst_reg++` at the end of each iteration advances the DEST read/write pointer by 1 sfpi row (= 2 physical DEST rows). |
| **LREGs (LREG0-LREG7)** | General-purpose registers used to hold intermediate values. The SFPI compiler allocates these automatically for: `x` (input value), `ax` (absolute value), `sig_pos` (sigmoid approximation), and various intermediate products/sums from the polynomial and linear evaluations. Float constants (c1, c2, c3, lin_slope, lin_offset, bp1, bp2, 0.5f) are loaded into LREGs via `SFPLOADI`. |
| **Fixed Const Register (index 10)** | `vConst1` = 1.0 (0x3F800000). Used for sigmoid saturation (`sig_pos = vConst1`) and negative-input correction (`vConst1 - sig_pos`). Accessed directly as a constant register operand without consuming an LREG. |
| **CC Stack (per-lane, 8 entries)** | Used by `v_if`/`v_endif` for predicated execution. Three `v_if` blocks use 1 CC stack level each (non-nested). Each `v_if` pushes onto the stack, and `v_endif` pops. Maximum CC stack depth during execution: 1 (since the three `v_if` blocks are sequential, not nested). |

### Address Mode Configuration

The `swish_tile_init()` call invokes `llk_math_eltwise_unary_sfpu_init<SfpuType::swish, APPROX>()`, which calls `_llk_math_eltwise_unary_sfpu_init_<SfpuType::swish>()`. This in turn calls `eltwise_unary_sfpu_configure_addrmod<SfpuType::swish>()`.

Since `SfpuType::swish` does not match any of the special-cased `SfpuType` values in the `if constexpr` branches (which handle `topk_local_sort`, `typecast`, `unary_max`, `unary_min`, etc.), only the default address mode is configured:

**ADDR_MOD_7** (default for all unary SFPU ops):
```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}
```

This configuration is **identical on both Wormhole and Blackhole**. The `.dest.incr = 0` means the hardware does not auto-increment the DEST address after each SFPU instruction -- instead, the SFPI abstraction handles DEST address progression via `dst_reg++` (which the compiler translates to explicit address manipulation) and the params dispatch layer advances between faces via `SETRWC`.

No `ADDR_MOD_6` is configured for swish (that is reserved for ops like `typecast`, `unary_max`/`unary_min`, and `signbit` which need a `.dest.incr = 2`).

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 expansion, and approximation mode for SWISH
   **Key Findings**: SWISH uses `SFPU_OP_SWISH_INCLUDE` macro, `swish_tile_init()`/`swish_tile(i)` API calls, compute kernel `eltwise_sfpu.cpp`, `get_op_approx_mode` returns `false` (default case)

2. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h`
   **Reason**: Core SFPU implementation -- primary analysis target
   **Key Findings**: Implements swish via piecewise sigmoid approximation: degree-3 polynomial for |x| <= 2.5, linear segment for 2.5 < |x| <= 5.0, saturation to 1.0 for |x| > 5.0, with symmetry correction for x < 0. Uses SFPI abstractions exclusively (no raw TTI instructions). APPROXIMATION_MODE template parameter is accepted but unused (no branching on it).

3. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/swish.h`
   **Reason**: API header layer connecting compute kernel to LLK
   **Key Findings**: `swish_tile(idst)` calls `llk_math_eltwise_unary_sfpu_swish<APPROX>(idst)`, `swish_tile_init()` calls `llk_math_eltwise_unary_sfpu_swish_init<APPROX>()`

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h`
   **Reason**: LLK dispatch layer connecting API to core SFPU
   **Key Findings**: Init calls `llk_math_eltwise_unary_sfpu_init<SfpuType::swish, APPROXIMATE>()`. Tile function calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>()` with `calculate_swish<APPROXIMATE, 8>` and `VectorMode::RC`.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch -- understand face iteration and DEST progression
   **Key Findings**: VectorMode::RC loops over 4 faces, calling the SFPU function once per face. Between faces, uses `TTI_SETRWC(CR_D, 8)` twice to advance DEST address by 16 physical rows (1 face).

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Understand ADDR_MOD configuration and SFPU init
   **Key Findings**: `eltwise_unary_sfpu_configure_addrmod<SfpuType::swish>()` configures only ADDR_MOD_7 with all-zero increments. No ADDR_MOD_6 for swish.

7. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`
   **Reason**: Understand how `math_approx_mode` is passed to the compute kernel
   **Key Findings**: `math_approx_mode` is computed as `std::all_of(op_chain, get_op_approx_mode)` and passed via `ComputeConfig`. For SWISH alone, this is `false`.

8. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: Authoritative reference for SFPU tile geometry, addressing model, and instruction semantics
   **Key Findings**: Confirmed stride-2 model, ITERATIONS=8 per face, 32 elements per iteration, SFPMAD used for all float arithmetic, SFPABS for absolute value, CC mechanism for predicated execution.

9. **File**: `runtime/sfpi/include/sfpi.h`
   **Reason**: Understand SFPI abstraction mappings to hardware instructions
   **Key Findings**: `v_if` macro expands to `SFPPUSHC` + condition evaluation + `SFPSETCC`; `v_endif` destructor triggers `SFPPOPC`; `vConst1` maps to constant register index 10 (Fixed Const 2 = 1.0); `dst_reg` read/write compiles to `SFPLOAD`/`SFPSTORE`.

10. **File**: `runtime/sfpi/include/sfpi_lib.h`
    **Reason**: Verify that `sfpi::abs()` maps to `SFPABS`
    **Key Findings**: `sfpi::abs(vFloat)` calls `__builtin_rvtt_sfpabs(v.get(), SFPABS_MOD1_FLOAT)`, confirming it compiles to the `SFPABS` instruction.
