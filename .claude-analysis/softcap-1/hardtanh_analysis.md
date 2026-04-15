## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `HARDTANH`
- **Compute kernel**: `eltwise_sfpu.cpp` (default path from `get_compute_kernel_path()` which has no HARDTANH case)
- **SFPU_OP_CHAIN_0 expansion**: **NOT WIRED** -- `get_op_init_and_func_parameterized()` has no `HARDTANH` case (falls through to `default: TT_THROW`) and `get_op_init_and_func_default()` also has no case. The SFPU kernel exists but the full dispatch chain is incomplete.

**Dispatch chain status**: The `UnaryOpType::HARDTANH` enum exists in `unary_op_types.hpp`, the `is_parametrized_type()` returns `true` for it, and the Python/C++ API in `unary.hpp` passes `min_val` and `max_val` as float parameters. However, the critical dispatch functions that generate the `SFPU_OP_CHAIN_0` macro expansion (`get_op_init_and_func_parameterized`, `get_op_init_and_func_default`) do not handle HARDTANH, so the operation will throw at runtime if called through the standard unary path. The SFPU kernel itself (`_calculate_hardtanh_`) is fully implemented and ready for integration.

#### Approximation Mode Resolution
There are TWO independent controls for approximation. You MUST report both.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(HARDTANH)` in `unary_op_utils.cpp` -- falls through to `default: return false` (no HARDTANH case) |
| Template parameter (SFPU_OP_CHAIN) | N/A (not wired) | `get_op_init_and_func()` has no HARDTANH case; the `_calculate_hardtanh_` template takes `APPROXIMATION_MODE` but never uses it in the function body |
| Effective SFPU path | `APPROXIMATION_MODE` is unused -- no `if constexpr` branches on it | The kernel body has no conditional paths based on `APPROXIMATION_MODE`; both true/false produce identical code |

### SFPU Abstraction Layers
List the file path for each abstraction layer. If a layer does not exist for this operation, write "This level of abstraction doesn't exist" instead of a path.

| Layer | File Path |
|-------|-----------|
| **API Header** | This level of abstraction doesn't exist (no `hardtanh_tile()` function in `tt_metal/hw/inc/api/compute/eltwise_unary/`) |
| **LLK Dispatch** | This level of abstraction doesn't exist (no `llk_math_eltwise_unary_sfpu_hardtanh.h` in `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/`) |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h` (WH); `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_hardtanh.h` (BH) -- implementations are identical |
| **Parameters Dispatch** | This level of abstraction doesn't exist (no LLK dispatch layer to call `_llk_math_eltwise_unary_sfpu_params_`); however, the generic dispatch template exists at `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` and would be used once wired |

### Call Chain
The full call chain is **not yet connected** for hardtanh. The intended chain (based on the pattern used by implemented operations like `frac`) would be:

1. `hardtanh_tile(idst, param0, param1, param2)` (API header, does not exist yet) would call `llk_math_eltwise_unary_sfpu_hardtanh<APPROX>(idst, ...)` via the `MATH()` macro.
2. The LLK dispatch function (does not exist yet) would call `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::_calculate_hardtanh_<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, param0, param1, param2)`.
3. `_llk_math_eltwise_unary_sfpu_params_` (generic, exists at `llk_math_eltwise_unary_sfpu_params.h`) would set up DEST addressing, stall for SFPU, loop over faces based on `VectorMode`, and invoke the core SFPU function per face.
4. `_calculate_hardtanh_<APPROXIMATION_MODE, ITERATIONS>` (exists in `ckernel_sfpu_hardtanh.h`) executes the actual SFPU microcode.

### Parameters Dispatch Summary
Since the LLK dispatch and API layers do not exist for hardtanh, this section describes the **expected** dispatch behavior based on the generic `_llk_math_eltwise_unary_sfpu_params_` template that would be used.

- **Vector mode**: The default mode would be `VectorMode::RC`, processing all 4 faces of the tile (full 32x32 = 1024 elements). This is the standard mode for element-wise unary operations.
- **Operation invocation**: The `_llk_math_eltwise_unary_sfpu_params_` function loops over 4 faces, calling `_calculate_hardtanh_` once per face with `ITERATIONS=8`. Between faces, `TTI_SETRWC` advances the DEST write counter by `8+8=16` physical rows (one face stride). The three `uint32_t` parameters (param0, param1, param2) are forwarded as additional arguments.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `TTI_SETRWC` between faces). Address mode `ADDR_MOD_7` is configured with all increments = 0 (srca=0, srcb=0, dest=0) since SFPU addressing is handled by `dst_reg++` within the kernel and `SETRWC` in the dispatch layer, not by the address mode auto-increment.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `v_if`/`v_endif`), so Style A (inline-commented source code) is used.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h
// (Blackhole implementation at tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_hardtanh.h is identical)

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_hardtanh_(const int iterations, std::uint32_t param0, std::uint32_t param1, std::uint32_t param2)
{   // APPROXIMATION_MODE is unused (no conditional branches on it), ITERATIONS is unused (iterations arg used instead)
    // All params are in FP16_B format, pre-negated by the host:
    // param0 = -min_val (negated low threshold)
    // param1 = -(max_val - min_val) (negated range width)
    // param2 = max_val (positive high threshold; see note below)

    sfpi::vFloat p0 = sfpi::s2vFloat16b(param0); // SFPLOADI: load -min_val as FP16_B immediate into LREG
    sfpi::vFloat p1 = sfpi::s2vFloat16b(param1); // SFPLOADI: load -(max_val - min_val) as FP16_B immediate into LREG
    sfpi::vFloat p2 = sfpi::s2vFloat16b(param2); // SFPLOADI: load max_val as FP16_B immediate into LREG
// SFPU microcode
#pragma GCC unroll 0
    for (int d = 0; d < iterations; d++) // 8 iterations per face, 32 elements per iteration
    {
        sfpi::vFloat val = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST row pair

        val += p0; // SFPMAD: val = val * 1.0 + p0 (effectively val = x - min_val)
        v_if (val < 0.0f) // SFPPUSHC + SFPSETCC(CC_LT): enable CC for lanes where x < min_val
        {
            val = 0.0f; // CC-guarded SFPLOADI/SFPMOV: zero out lanes below min_val
        }
        v_endif; // SFPPOPC: restore CC state (all lanes active again)

        val += p1; // SFPMAD: val = val * 1.0 + p1 (shift range so max boundary maps to 0)
        v_if (val >= 0.0f) // SFPPUSHC + SFPSETCC(CC_GTE): enable CC for lanes where x > max_val
        {
            val = 0.0f; // CC-guarded SFPLOADI/SFPMOV: zero out lanes above max_val
        }
        v_endif; // SFPPOPC: restore CC state

        val += p2; // SFPMAD: val = val * 1.0 + p2 (restore to final output value)

        sfpi::dst_reg[0] = val; // SFPSTORE: write 32 elements back to current DEST row pair

        sfpi::dst_reg++; // advance DEST address by 1 sfpi row (= 2 physical rows = 32 elements)
    }
}
```

**Note on param2 and the source code comment**: The source comment says `param2 = -(pos_threshold)`, but mathematical analysis shows the kernel produces correct results only when `param2 = max_val` (positive, not negated). The comment may reflect a documentation error in the kernel, or the terms "neg_threshold" and "pos_threshold" may use a non-obvious convention. The mathematical correctness analysis is:
- For `x < min_val`: after first clamp, val=0; after second addition, val=-(max_val-min_val); second clamp does not trigger (val<0); final: val = -(max_val-min_val) + param2. Correct only if param2 = max_val, yielding min_val.
- For `x > max_val`: first clamp does not trigger; after range shift, val>0; second clamp zeros val; final: val = 0 + param2. Correct only if param2 = max_val, yielding max_val.
- For `min_val <= x <= max_val`: no clamps trigger; final: val = (x - min_val) - (max_val - min_val) + param2 = x - max_val + param2. Correct only if param2 = max_val, yielding x.

### SFPU Instructions Used

| Instruction | SFPI Abstraction | Description |
|-------------|-----------------|-------------|
| `SFPLOADI` | `sfpi::s2vFloat16b(param)` | Load a 16-bit FP16_B immediate scalar value into an LREG. Used 3 times to load the three threshold parameters into vector registers. Also used for the `val = 0.0f` assignments inside `v_if` blocks. |
| `SFPLOAD` | `sfpi::dst_reg[0]` (read) | Load 32 elements (2 physical DEST rows) from the current DEST address into an LREG. Used once per iteration to read input data. |
| `SFPMAD` | `val += p0`, `val += p1`, `val += p2` | Fused multiply-add: `VD = VA * 1.0 + VC`. Since there is no dedicated float add instruction, vFloat addition compiles to SFPMAD with multiplicand = 1.0. Used 3 times per iteration for the threshold additions. |
| `SFPSETCC` | `val < 0.0f`, `val >= 0.0f` | Set per-lane condition code based on floating-point comparison. `val < 0.0f` uses `SFPSETCC_MOD1_LREG_LT0` (sign bit test); `val >= 0.0f` uses `SFPSETCC_MOD1_LREG_GTE0` (inverted sign bit test). The SFPI compiler may emit these via the `SFPXCMP/SFPXFCMPS` pseudo-instructions. |
| `SFPPUSHC` | `v_if(...)` | Push current CC state onto the per-lane CC stack. Creates a new nesting level for conditional execution. Used once per `v_if` block (2 times per iteration). |
| `SFPPOPC` | `v_endif` | Pop CC state from the per-lane CC stack, restoring the previous conditional execution context. Used once per `v_endif` (2 times per iteration). |
| `SFPSTORE` | `sfpi::dst_reg[0] = val` | Store 32 elements from an LREG back to the current DEST address (2 physical DEST rows). Used once per iteration to write the clamped result. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG (p0)** | Holds the first parameter value (`-min_val` in FP16_B format). Loaded once before the loop via `s2vFloat16b(param0)` and reused across all 8 iterations. |
| **LREG (p1)** | Holds the second parameter value (`-(max_val - min_val)` in FP16_B format). Loaded once before the loop and reused. |
| **LREG (p2)** | Holds the third parameter value (`max_val` in FP16_B format). Loaded once before the loop and reused. |
| **LREG (val)** | Temporary register for the current element value. Loaded from DEST each iteration, modified through the three-step clamp algorithm, and stored back to DEST. |
| **DEST register** | Source and destination for tile data. Read via `dst_reg[0]` and written via `dst_reg[0] = val`. The `dst_reg++` advances the DEST read/write pointer by 1 sfpi row (2 physical DEST rows = 32 elements) per iteration. |
| **CC Stack** | The per-lane condition code stack is used by `v_if`/`v_endif` for predicated execution. Two levels of push/pop per iteration (non-nested, sequential). Maximum CC stack depth during execution is 1 (each `v_if`/`v_endif` pair completes before the next begins). |

### Address Mode Configuration

The address mode configuration for hardtanh would use the default path in `eltwise_unary_sfpu_configure_addrmod<SfpuType>()` (once a `SfpuType::hardtanh` is added to the metal SfpuType enum). The default configuration is:

**ADDR_MOD_7** (both Wormhole and Blackhole):
| Field | Value | Description |
|-------|-------|-------------|
| `srca.incr` | 0 | No auto-increment for SrcA |
| `srcb.incr` | 0 | No auto-increment for SrcB |
| `dest.incr` | 0 | No auto-increment for DEST |

This configuration sets all address auto-increments to zero because SFPU DEST addressing is managed explicitly:
- **Within a face**: `dst_reg++` in the kernel loop advances the SFPU DEST pointer by 1 sfpi row per iteration (8 iterations per face).
- **Between faces**: `TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D)` is called twice per face transition (advancing by 8+8=16 physical rows = 1 face stride) by the dispatch layer.

The `ADDR_MOD_6` special configuration (`.dest.incr = 2`) is NOT needed for hardtanh since it is only used for operations that require custom DEST stride patterns (typecast, unary_max/min, signbit, etc.).

The WH and BH configurations are identical for the default path. The only difference in the BH version is that `SfpuType::reciprocal` is added to the `ADDR_MOD_6` conditional, which does not affect hardtanh.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: To determine `get_op_approx_mode()`, `get_op_init_and_func()`, and `get_compute_kernel_path()` for HARDTANH
   **Key Findings**: HARDTANH has no case in any of these functions; `get_op_approx_mode` returns false (default), `get_compute_kernel_path` returns "eltwise_sfpu.cpp" (default), and `get_op_init_and_func` would TT_THROW for HARDTANH

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
   **Reason**: To check `is_parametrized_type()` for HARDTANH
   **Key Findings**: Returns `true` for HARDTANH, confirming it is a parameterized operation (takes min_val, max_val)

3. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`
   **Reason**: To understand the Python/C++ API for hardtanh
   **Key Findings**: `hardtanh(input, min_val=-1.0f, max_val=1.0f)` passes both params as `UnaryWithParam{UnaryOpType::HARDTANH, min_val, max_val}`

4. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h`
   **Reason**: Core SFPU kernel implementation for hardtanh (Wormhole)
   **Key Findings**: Uses SFPI abstractions for a three-step clamping algorithm with predicated execution. Takes 3 pre-computed uint32_t params in FP16_B format. APPROXIMATION_MODE template parameter is unused.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_hardtanh.h`
   **Reason**: Core SFPU kernel implementation for hardtanh (Blackhole)
   **Key Findings**: Identical to the Wormhole implementation

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Generic parameters dispatch template that would be used by hardtanh once wired
   **Key Findings**: Implements the standard VectorMode::RC loop over 4 faces with 8 iterations per face, calling the SFPU function via perfect forwarding of additional arguments

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: SFPU init and address mode configuration
   **Key Findings**: Default ADDR_MOD_7 has all increments = 0; no special SfpuType for hardtanh in the metal SfpuType enum yet

8. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
   **Reason**: Metal build SfpuType enum
   **Key Findings**: Only contains `frac`, `swish`, `atanh`, `sinh`; no `hardtanh` entry yet

9. **File**: `tt_metal/third_party/tt_llk/tests/helpers/include/llk_sfpu_types.h`
   **Reason**: Test/LLK SfpuType enum
   **Key Findings**: Contains `hardtanh` at enum position 1 (after `tanh`), confirming the LLK layer recognizes the operation type

10. **File**: `runtime/sfpi/include/sfpi.h`
    **Reason**: SFPI C++ abstraction layer defining vFloat, v_if/v_endif, dst_reg, and how comparisons compile to SFPU instructions
    **Key Findings**: `v_if(val < 0.0f)` expands to SFPPUSHC + SFPSETCC via `__builtin_rvtt_sfpxfcmps`; `v_endif` expands to SFPPOPC; vFloat addition compiles to SFPMAD(a * 1.0 + b)

11. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: SFPU hardware reference for instruction semantics, register layout, and addressing model
    **Key Findings**: Confirmed stride-2 addressing (32 elements per dst_reg access), SFPMAD is the universal float add/multiply instruction, SFPSETCC modes for sign-bit testing (LT0, GTE0)

12. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel_defs.h`
    **Reason**: Check ActivationType enum for hardtanh
    **Key Findings**: `ActivationType::Hardtanh = 3` exists in the enum, indicating the operation is recognized at the ckernel level
