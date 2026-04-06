## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `HARDTANH`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `hardtanh_tile_init(); hardtanh_tile(0, <min_val_bits>u, <max_val_bits>u);`

The `min_val` and `max_val` parameters are passed as IEEE 754 float bit patterns (bitcast to `uint32_t`) via `std::bit_cast<uint32_t>(min_val)` and `std::bit_cast<uint32_t>(max_val)` in `get_op_init_and_func_parameterized()`. Default values are `min_val = -1.0f` and `max_val = 1.0f`.

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(HARDTANH)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none | `get_op_init_and_func_parameterized()` -- `hardtanh_tile_init()` and `hardtanh_tile(idst, param0, param1)` use no template argument; the API header passes `APPROX` (which is `false`) |
| Effective SFPU path | `APPROXIMATION_MODE=false`, but the kernel does not branch on `APPROXIMATION_MODE` at all | `calculate_hardtanh` template parameter is unused -- the function body has no `if constexpr (APPROXIMATION_MODE)` branches |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/hardtanh.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardtanh.h` (identical on both architectures) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_hardtanh.h` (identical on both architectures) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain
1. The compute kernel `eltwise_sfpu.cpp` invokes the `SFPU_OP_CHAIN_0` macro, which expands to `hardtanh_tile_init(); hardtanh_tile(0, param0, param1);`.
2. `hardtanh_tile_init()` (in `hardtanh.h`) calls `llk_math_eltwise_unary_sfpu_hardtanh_init<APPROX>()`, which invokes `llk_math_eltwise_unary_sfpu_init<SfpuType::hardtanh, APPROX>()` to configure SFPU registers and address modes.
3. `hardtanh_tile(idst, param0, param1)` (in `hardtanh.h`) calls `llk_math_eltwise_unary_sfpu_hardtanh<APPROX>(idst, param0, param1)`.
4. The LLK dispatch function calls `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_hardtanh<APPROX, 8>, dst_index, VectorMode::RC, param0, param1)`.
5. `_llk_math_eltwise_unary_sfpu_params_` sets the DEST write address, stalls for SFPU readiness, then iterates over 4 faces calling `calculate_hardtanh(param0, param1)` once per face, with `SETRWC`/`inc_dst_addr` between faces.
6. `calculate_hardtanh` (in `ckernel_sfpu_hardtanh.h`) is the core SFPU function that performs the actual clamping computation using SFPI abstractions.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed (the full 32x32 tile).
- **Operation invocation**: The `_llk_math_eltwise_unary_sfpu_params_` function loops over 4 faces, calling `calculate_hardtanh(param0, param1)` once per face. Each call processes 8 SFPU iterations (ITERATIONS=8), covering one 16x16 face (256 elements).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC`/`inc_dst_addr` between faces). On Wormhole, the params dispatch uses `TTI_SETRWC(CR_D, 8, SET_D)` twice between faces (advancing by 16 physical DEST rows = 1 face). On Blackhole, the equivalent `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`vFloat`, `dst_reg`, `v_if`/`v_endif`), so Style A (inline-commented source code) is used.

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_hardtanh.h

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_hardtanh(std::uint32_t param0, std::uint32_t param1) { // APPROXIMATION_MODE=false, ITERATIONS=8
    // Reinterpret IEEE 754 bit patterns back to float, then broadcast to vFloat (all SFPU lanes)
    sfpi::vFloat min_val = Converter::as_float(param0); // SFPLOADI to load scalar min_val into LREG
    sfpi::vFloat max_val = Converter::as_float(param1); // SFPLOADI to load scalar max_val into LREG

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST position into LREG

        v_if(val < min_val) { val = min_val; } // SFPXFCMPV(LT) -> SFPPUSHC + SFPENCC + SFPSETCC + guarded SFPMOV
        v_endif;                                // SFPPOPC: restore CC state

        v_if(val > max_val) { val = max_val; } // SFPXFCMPV(GT) -> SFPPUSHC + SFPENCC + SFPSETCC + guarded SFPMOV
        v_endif;                                // SFPPOPC: restore CC state

        sfpi::dst_reg[0] = val; // SFPSTORE: write 32 elements back to current DEST position
        sfpi::dst_reg++;        // Advance DEST pointer by 1 sfpi row (= 2 physical DEST rows = 32 elements)
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

**Instruction generation from SFPI abstractions:**

The `v_if(val < min_val)` construct compiles through these SFPI abstractions:
1. `val < min_val` (both `vFloat`) invokes `operator<(vFloat, vFloat)` which creates `__vCond(__vCondLT, val, min_val)`, calling `__builtin_rvtt_sfpxfcmpv(val, min_val, SFPXCMP_MOD1_CC_LT)`.
2. The `v_if` macro expands to `__cc.cc_push().cc_if().cc_cond(cond)`:
   - `cc_push()` emits `SFPPUSHC` (push current CC state onto the CC stack)
   - `cc_if()` emits the enable-CC sequence via `__builtin_rvtt_sfpxvif()`
   - `cc_cond(cond)` emits the comparison result via `__builtin_rvtt_sfpxcondb()`, which combines with the comparison result to set CC.Res per-lane
3. Inside the guarded block, `val = min_val` is a conditional `SFPMOV` (only executes on lanes where CC is active).
4. `v_endif` triggers `__vCCCtrl` destructor which calls `cc_pop()`, emitting `SFPPOPC` to restore the prior CC state.

The same pattern repeats for `v_if(val > max_val)` with `SFPXCMP_MOD1_CC_GT`.

The SFPI compiler (`__builtin_rvtt_sfpxfcmpv`) lowers the float vector comparison to a sequence that computes `a - b` via `SFPMAD` (subtract), then tests the sign of the result via `SFPSETCC` to determine which lanes satisfy the condition. The exact instruction lowering is compiler-dependent, but the semantic is: compute the difference, set CC based on the sign.

### SFPU Instructions Used

| Instruction | Usage in this kernel |
|-------------|---------------------|
| `SFPLOADI` | Load the scalar `min_val` and `max_val` constants (from `Converter::as_float()`) as immediate values into LREGs. Each 32-bit float requires two `SFPLOADI` instructions (high 16 bits, then low 16 bits). |
| `SFPLOAD` | Load 32 elements from current DEST position into an LREG (`sfpi::dst_reg[0]` read). Executed once per iteration. |
| `SFPSTORE` | Write 32 elements from an LREG back to current DEST position (`sfpi::dst_reg[0] = val`). Executed once per iteration. |
| `SFPMAD` | Used by the compiler-lowered float comparison (`val < min_val`, `val > max_val`) to compute the difference `val - min_val` or `val - max_val` (as `val * 1.0 + (-min_val)`). Also used implicitly when the compiler needs to move float values between LREGs (multiply by 1.0 + 0.0). |
| `SFPSETCC` | Set CC.Res per-lane based on the sign of the comparison result (from `SFPMAD` output). Part of the compiler-generated comparison lowering. |
| `SFPPUSHC` | Push CC state onto the CC stack at each `v_if` entry. Two pushes per iteration (one for each `v_if` block). |
| `SFPPOPC` | Pop CC state from the CC stack at each `v_endif`. Two pops per iteration. |
| `SFPENCC` | Enable CC masking as part of the `v_if` entry sequence (via `cc_if()` -> `__builtin_rvtt_sfpxvif()`). |
| `SFPMOV` | Conditional register copy for `val = min_val` and `val = max_val` inside the CC-guarded blocks. Only executes on lanes where the condition is true. |

### SFPU Register Usage

| Register | Purpose |
|----------|---------|
| **LREG (vFloat val)** | Holds the current tile data loaded from DEST. Modified conditionally to `min_val` or `max_val`, then stored back. |
| **LREG (vFloat min_val)** | Holds the broadcast `min_val` scalar constant. Loaded once before the loop via `SFPLOADI` and reused across all 8 iterations. |
| **LREG (vFloat max_val)** | Holds the broadcast `max_val` scalar constant. Loaded once before the loop via `SFPLOADI` and reused across all 8 iterations. |
| **LREG (temp)** | Used transiently by the compiler for the float comparison result (`val - min_val` or `val - max_val`). |
| **DEST registers** | The tile data in DEST is read via `SFPLOAD`, potentially modified, and written back via `SFPSTORE`. 8 sfpi rows processed per face (32 elements each), 4 faces per tile. |
| **CC register** | Per-lane condition code used by `v_if` blocks. Saved/restored via the CC stack (`SFPPUSHC`/`SFPPOPC`) for each conditional block. |
| **CC stack** | Used to save/restore CC state across the two sequential `v_if` blocks. Each `v_if`/`v_endif` pair pushes and pops one entry. Maximum stack depth during execution: 1. |

### Address Mode Configuration

The address mode is configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType::hardtanh>()` during initialization (`hardtanh_tile_init()`). Since `SfpuType::hardtanh` does not match any of the special `if constexpr` branches (which handle `topk_local_sort`, `typecast`, `unary_max`, `unary_min`, etc.), only the default `ADDR_MOD_7` is configured:

```
ADDR_MOD_7: { .srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0} }
```

This means: no auto-increment of source or destination addresses between SFPU operations. The DEST address advancement is handled explicitly by `dst_reg++` (compiler-generated DEST pointer increment) within the kernel loop, and by `SETRWC`/`inc_dst_addr` between faces in the params dispatch.

This configuration is **identical on both Wormhole and Blackhole** -- both architectures set `ADDR_MOD_7` with all-zero increments for `SfpuType::hardtanh`.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 expansion, and approximation mode for HARDTANH
   **Key Findings**: HARDTANH uses `eltwise_sfpu.cpp`; macro define is `SFPU_OP_HARDTANH_INCLUDE`; `get_op_approx_mode()` returns `false` (default case); `get_op_init_and_func_parameterized()` produces `hardtanh_tile_init()` and `hardtanh_tile(idst, param0_bits, param1_bits)` with min/max passed as IEEE 754 bitcast uint32_t

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
   **Reason**: Confirm `is_parametrized_type()` returns true for HARDTANH
   **Key Findings**: HARDTANH is listed as a parameterized type, meaning it always takes parameters (min_val, max_val)

3. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/hardtanh.h`
   **Reason**: API header defining `hardtanh_tile()` and `hardtanh_tile_init()`
   **Key Findings**: `hardtanh_tile(idst, param0, param1)` calls `llk_math_eltwise_unary_sfpu_hardtanh<APPROX>(idst, param0, param1)` via the `MATH()` macro; `hardtanh_tile_init()` calls `llk_math_eltwise_unary_sfpu_hardtanh_init<APPROX>()`

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardtanh.h`
   **Reason**: LLK dispatch layer bridging API to core SFPU function
   **Key Findings**: Dispatches to `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(calculate_hardtanh<APPROXIMATE, ITERATIONS>, dst_index, VectorMode::RC, param0, param1)`; identical on WH and BH

5. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardtanh.h`
   **Reason**: Core SFPU implementation of hardtanh
   **Key Findings**: Simple clamp kernel using SFPI abstractions; loops 8 iterations per face; uses `v_if(val < min_val)` and `v_if(val > max_val)` for conditional clamping; `APPROXIMATION_MODE` template parameter is unused (no branching on it); identical on WH and BH

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Understand the params dispatch pattern (face iteration, DEST address progression)
   **Key Findings**: For `VectorMode::RC`, iterates over 4 faces calling the SFPU function once per face, advancing DEST address by one face (16 physical rows via two `SETRWC CR_D, 8, SET_D` calls) between faces

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Understand SFPU init and address mode configuration
   **Key Findings**: `eltwise_unary_sfpu_configure_addrmod<SfpuType::hardtanh>()` configures only `ADDR_MOD_7` with all-zero increments (no special address mode overrides for hardtanh)

8. **File**: `runtime/sfpi/include/sfpi.h`
   **Reason**: Understand how SFPI abstractions (`vFloat`, `v_if`, `v_endif`, comparison operators) lower to SFPU instructions
   **Key Findings**: `vFloat < vFloat` calls `__builtin_rvtt_sfpxfcmpv()` which the compiler lowers to SFPMAD+SFPSETCC; `v_if` macro uses `SFPPUSHC`+`SFPENCC`+comparison+guarded ops; `v_endif` triggers `SFPPOPC` via `__vCCCtrl` destructor

9. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: Authoritative reference for SFPU instruction semantics, CC mechanism, and addressing model
   **Key Findings**: SFPMAD is the only float arithmetic instruction (no dedicated add); SFPSETCC sets CC.Res based on sign/zero tests; CC stack supports nested conditionals via SFPPUSHC/SFPPOPC; stride-2 addressing means each dst_reg++ advances 2 physical DEST rows

10. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_converter.h`
    **Reason**: Understand `Converter::as_float()` utility
    **Key Findings**: Simple union-based bitcast from `uint32_t` to `float` -- reinterprets IEEE 754 bit patterns without conversion
