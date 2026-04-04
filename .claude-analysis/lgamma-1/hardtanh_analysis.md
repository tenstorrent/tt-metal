## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `HARDTANH`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `hardtanh_tile(0, 0xBF800000u, 0x3F800000u)` (default params: min=-1.0f, max=1.0f)

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(HARDTANH)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none | `get_op_init_and_func_parameterized()` -- `hardtanh_tile_init()` has no template parameter; the API header calls `llk_math_eltwise_unary_sfpu_hardtanh_init<APPROX>()` where `APPROX` is the compile-time define from `math_approx_mode` |
| Effective SFPU path | `APPROXIMATION_MODE=false` passed to `calculate_hardtanh<false, 8>()` | The `calculate_hardtanh` template has `APPROXIMATION_MODE` but does not branch on it -- the kernel logic is the same regardless of approximation mode |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/hardtanh.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardtanh.h` (identical for both architectures) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_hardtanh.h` (identical for both architectures) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. **`hardtanh_tile(idst, param0, param1)`** (API header, `hardtanh.h`) -- wraps in `MATH((...))` macro for TRISC_MATH processor, calls `llk_math_eltwise_unary_sfpu_hardtanh<APPROX>(idst, param0, param1)`.
2. **`llk_math_eltwise_unary_sfpu_hardtanh<APPROXIMATE, ITERATIONS=8>(dst_index, param0, param1, VectorMode::RC)`** (LLK dispatch) -- forwards to `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>()` with `ckernel::sfpu::calculate_hardtanh<APPROXIMATE, ITERATIONS>` as the callable functor and `param0`, `param1` as forwarded arguments.
3. **`_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(sfpu_func, dst_index, vector_mode, param0, param1)`** (params dispatch, `llk_math_eltwise_unary_sfpu_params.h`) -- sets DEST write address, stalls until SFPU is ready, then loops over 4 faces (in RC mode), calling `sfpu_func(param0, param1)` once per face, with `TTI_SETRWC` between faces to advance the DEST address.
4. **`calculate_hardtanh<false, 8>(param0, param1)`** (core SFPU, `ckernel_sfpu_hardtanh.h`) -- converts params to `vFloat`, then for each of 8 iterations: loads from DEST, clamps to `[min_val, max_val]` using two conditional blocks, stores back, and advances `dst_reg`.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (all 4 faces of the tile are processed).
- **Operation invocation**: The core SFPU function `calculate_hardtanh<APPROXIMATE, 8>` is called once per face (4 times total for RC mode). Each call runs 8 iterations internally, processing one sfpi row (32 elements) per iteration. Total: 4 faces x 8 iterations = 32 sfpi rows = 1024 elements = full tile.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `TTI_SETRWC` between faces). On Wormhole, the params dispatch uses `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` twice between faces (incrementing DEST address by 8+8=16 physical rows = 1 face). On Blackhole, `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice for the same effect.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`vFloat`, `dst_reg`, `v_if`/`v_endif`) -- Style A applies.

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_hardtanh.h

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_hardtanh(std::uint32_t param0, std::uint32_t param1) {
    // APPROXIMATION_MODE=false, ITERATIONS=8 (APPROXIMATION_MODE is unused in this kernel)
    // param0 = min_val as IEEE 754 float bits (bitcast uint32_t)
    // param1 = max_val as IEEE 754 float bits (bitcast uint32_t)
    sfpi::vFloat min_val = Converter::as_float(param0); // Reinterpret uint32_t as float, load into LREG via SFPLOADI
    sfpi::vFloat max_val = Converter::as_float(param1); // Same for max_val

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST row pair into LREG

        v_if(val < min_val) { val = min_val; } // SFPPUSHC + compare (SFPMAD subtract + SFPSETCC LT0) -> CC-guarded SFPMOV + SFPPOPC
        v_endif;

        v_if(val > max_val) { val = max_val; } // SFPPUSHC + compare (SFPMAD subtract + SFPSETCC GTE0) -> CC-guarded SFPMOV + SFPPOPC
        v_endif;

        sfpi::dst_reg[0] = val; // SFPSTORE: store 32 elements from LREG back to DEST row pair
        sfpi::dst_reg++;        // Advance sfpi address by 1 (= 2 physical DEST rows = 32 elements)
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

**Detailed instruction expansion for each `v_if` block:**

The `v_if(val < min_val) { val = min_val; } v_endif;` pattern expands to the following SFPU instruction sequence:

1. `SFPPUSHC` (mode 0: push current CC state onto stack)
2. `SFPENCC` (enable CC with both En and Res set)
3. Comparison `val < min_val`: the compiler emits `__builtin_rvtt_sfpxfcmpv(val, min_val, CC_LT)` which typically generates:
   - `SFPMAD` to compute `val - min_val` (as `val * 1.0 + (-min_val)`)
   - `SFPSETCC` with `LREG_LT0` mode to set CC.Res=1 for lanes where `val - min_val < 0` (i.e., `val < min_val`)
4. `SFPMOV` (CC-guarded): assigns `min_val` to `val` only for lanes where CC is active
5. `SFPPOPC` (mode 0: pop CC state from stack, restoring previous CC)

The second `v_if(val > max_val)` block follows the same pattern but with `CC_GT` comparison semantics (compiler may emit `SFPMAD` for `max_val - val` + `SFPSETCC LT0`, or `val - max_val` + `SFPSETCC GTE0`, depending on how the compiler optimizes the greater-than comparison).

### SFPU Instructions Used

| Instruction | Description | Usage in `calculate_hardtanh` |
|-------------|-------------|-------------------------------|
| `SFPLOADI` | Load 16-bit immediate to LREG | Loading `min_val` and `max_val` constants from the bitcast `uint32_t` parameters (2 SFPLOADI per constant for full 32-bit float) |
| `SFPLOAD` | Load from DEST row to LREG | Reading current tile element vector from DEST into `val` (once per iteration) |
| `SFPMAD` | Fused multiply-add (VD = VA x VB + VC) | Computing subtraction for comparisons: `val - min_val` and `val - max_val` (or their negations). There is no dedicated float subtract -- SFPMAD with sign inversion of the addend is used. |
| `SFPSETCC` | Set CC.Res based on comparison | Setting per-lane condition code after subtraction to determine which lanes need clamping (LT0 mode for less-than, GTE0 for greater-than-or-equal) |
| `SFPPUSHC` | Push CC state onto stack | Saving CC state before each `v_if` conditional block (2 pushes per iteration) |
| `SFPPOPC` | Pop CC state from stack | Restoring CC state after each `v_if`/`v_endif` block (2 pops per iteration) |
| `SFPMOV` | Register copy (CC-guarded) | Conditionally assigning `min_val` or `max_val` to `val` for lanes that fail the range check |
| `SFPSTORE` | Store LREG to DEST row | Writing the clamped result back to DEST (once per iteration) |
| `SFPENCC` | Enable/disable CC | Enabling CC masking at the start of each `v_if` block and disabling it at cleanup |

### SFPU Register Usage

| Register | Purpose |
|----------|---------|
| **LREG (val)** | Holds the current element vector loaded from DEST. Used as both the comparison operand and the output value after clamping. Compiler assigns this to an available LREG (typically LREG0 or LREG1). |
| **LREG (min_val)** | Holds the `min_val` constant (converted from `param0` via `Converter::as_float`). Loaded once before the iteration loop and reused across all 8 iterations. Compiler assigns to a distinct LREG. |
| **LREG (max_val)** | Holds the `max_val` constant (converted from `param1` via `Converter::as_float`). Loaded once before the iteration loop and reused across all 8 iterations. Compiler assigns to a distinct LREG. |
| **LREG (temporary)** | Used by SFPMAD for the subtraction result during comparisons. The compiler may reuse an existing LREG or allocate a new one. |
| **DEST rows** | Source and destination for tile data. Each iteration processes 2 physical DEST rows (32 elements) via stride-2 addressing. |
| **CC stack** | The per-lane condition code stack is used for nested predication. Each `v_if` pushes one entry; `v_endif` pops it. Maximum stack depth during execution is 1 (no nesting between the two `v_if` blocks). |

### Address Mode Configuration

The address mode for `hardtanh` is configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType::hardtanh>()` during initialization.

Since `SfpuType::hardtanh` does not match any of the special-cased `if constexpr` branches (which handle `topk_local_sort`, `typecast`, `unary_max/min`, etc.), only the default `ADDR_MOD_7` is configured:

**Wormhole B0 and Blackhole (identical):**
```
ADDR_MOD_7:
  .srca = {.incr = 0}
  .srcb = {.incr = 0}
  .dest = {.incr = 0}
```

This means the hardware address auto-increment is zero for all register files. DEST address advancement between iterations is handled entirely by the SFPI `dst_reg++` abstraction (which emits inline pointer arithmetic), and between faces by the `TTI_SETRWC` instructions in the params dispatch layer. No additional `ADDR_MOD_6` is configured for this operation.

The Wormhole and Blackhole `eltwise_unary_sfpu_configure_addrmod` functions differ only in that Blackhole also special-cases `SfpuType::reciprocal` for `ADDR_MOD_6` -- this does not affect `hardtanh`.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine the compute kernel path, SFPU_OP_CHAIN_0 expansion, and approximation mode for HARDTANH
   **Key Findings**: HARDTANH uses `eltwise_sfpu.cpp`, init is `hardtanh_tile_init()`, tile func is `hardtanh_tile({idst}, {min_hex}u, {max_hex}u)`, approx mode is `false` (default case)

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
   **Reason**: Check `is_parametrized_type` and function signatures
   **Key Findings**: HARDTANH is a parameterized type; takes min_val and max_val parameters (defaults -1.0f and 1.0f)

3. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/hardtanh.h`
   **Reason**: API header that exposes `hardtanh_tile()` and `hardtanh_tile_init()`
   **Key Findings**: `hardtanh_tile(idst, param0, param1)` calls `llk_math_eltwise_unary_sfpu_hardtanh<APPROX>(idst, param0, param1)`. `hardtanh_tile_init()` calls `llk_math_eltwise_unary_sfpu_hardtanh_init<APPROX>()`

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardtanh.h`
   **Reason**: LLK dispatch layer that bridges API to ckernel
   **Key Findings**: Init calls `llk_math_eltwise_unary_sfpu_init<SfpuType::hardtanh, APPROXIMATE>()`. Tile function calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>()` with `calculate_hardtanh<APPROXIMATE, ITERATIONS>` as callable and `param0, param1` forwarded

5. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardtanh.h`
   **Reason**: Core SFPU implementation -- the actual kernel function
   **Key Findings**: Uses SFPI abstractions (vFloat, dst_reg, v_if/v_endif). Clamps val to [min_val, max_val] with two conditional blocks. 8 iterations per face. WH and BH implementations are identical.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch function that orchestrates SFPU invocation across faces
   **Key Findings**: For VectorMode::RC, loops 4 faces, calls sfpu_func once per face, uses TTI_SETRWC to advance DEST address between faces

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Init and address mode configuration
   **Key Findings**: `ADDR_MOD_7` configured with all-zero increments for hardtanh (no special case). Init also calls `_init_sfpu_config_reg()` and `reset_counters(SET_ABD_F)`.

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_converter.h`
   **Reason**: Understand the `Converter::as_float()` utility used to reinterpret uint32_t as float
   **Key Findings**: Simple union-based bitcast from uint32_t to float

9. **File**: `runtime/sfpi/include/sfpi.h`
   **Reason**: Understand SFPI abstraction semantics: vFloat comparisons, v_if/v_endif CC mechanism, dst_reg addressing
   **Key Findings**: `vFloat < vFloat` emits `__builtin_rvtt_sfpxfcmpv` (compiler generates SFPMAD + SFPSETCC). `v_if` expands to SFPPUSHC + SFPENCC + comparison. `v_endif` destructs `__vCCCtrl` which calls SFPPOPC.

10. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Authoritative reference for SFPU tile geometry, register layout, instruction semantics, and CC mechanism
    **Key Findings**: Verified stride-2 model, 8 iterations per face, SFPMAD is used for float subtraction, SFPSETCC modes for comparison
