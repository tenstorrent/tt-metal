## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `HARDTANH`
- **Compute kernel**: `eltwise_sfpu.cpp` (via `get_compute_kernel_path()` default case)
- **SFPU_OP_CHAIN_0 expansion**: **Incomplete dispatch chain** -- `UnaryOpType::HARDTANH` is registered as a parametrized type in `is_parametrized_type()` but has no case in `get_op_init_and_func_parameterized()`. Attempting to generate compute defines at runtime would hit `TT_THROW("unexpected parameterized op type {}", op_type)`. The core SFPU kernel `_calculate_hardtanh_` exists at the tt_llk level but the intermediate API/LLK dispatch layers are not wired up.

**Note on dispatch gap**: The TTNN-level function `ttnn::hardtanh()` (in `unary.hpp`) creates `UnaryWithParam{UnaryOpType::HARDTANH, min_val, max_val}`, but the compute kernel define generation path is missing. The expected tile-level API call (e.g., `hardtanh_tile(idst, param0, param1, param2)`) does not exist. The intended dispatch would use macro `SFPU_UNARY_THREE_PARAM_KERNEL_FN(_calculate_hardtanh_, RC, APPROXIMATE, DST_IDX, PARAM0, PARAM1, PARAM2)` from the macros header, but this wiring is absent.

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(HARDTANH)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none (dispatch not wired) | `get_op_init_and_func_parameterized()` has no HARDTANH case; the template `APPROXIMATION_MODE` would be passed through when wired |
| Effective SFPU path | The `_calculate_hardtanh_` template does not branch on `APPROXIMATION_MODE` -- the parameter is accepted but unused; all code paths are identical regardless of its value | `ckernel_sfpu_hardtanh.h` line 16: template parameter is declared but no `if constexpr (APPROXIMATION_MODE)` exists |

### SFPU Abstraction Layers
List the file path for each abstraction layer. Since the full dispatch chain is not wired, the API Header and LLK Dispatch layers do not exist.

| Layer | File Path |
|-------|-----------|
| **API Header** | This level of abstraction doesn't exist (no `hardtanh_tile()` function in any compute_kernel_api header) |
| **LLK Dispatch** | This level of abstraction doesn't exist (no `llk_math_eltwise_unary_sfpu_hardtanh` function) |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h` (identical for Blackhole: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_hardtanh.h`) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (shared generic params dispatch; Blackhole: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`) |

### Call Chain
The full call chain is not wired in this codebase. The **intended** call chain when complete would be:

1. `hardtanh_tile(idst, param0, param1, param2)` -- (API header, does not exist) would call the LLK dispatch function.
2. LLK dispatch function -- (does not exist) would invoke `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::_calculate_hardtanh_<APPROXIMATE>, dst_index, (int)VectorMode::RC, param0, param1, param2)` using the `SFPU_UNARY_THREE_PARAM_KERNEL_FN` macro pattern.
3. `_llk_math_eltwise_unary_sfpu_params_` -- the generic parameters dispatch function sets up DEST addressing, stalls for SFPU readiness, then calls the SFPU function once per face (4 faces for `VectorMode::RC`), advancing the DEST write pointer between faces via `TTI_SETRWC` (Wormhole) or `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` (Blackhole).
4. `_calculate_hardtanh_<APPROXIMATION_MODE, ITERATIONS>()` -- the core SFPU function processes `ITERATIONS` (default 8) sfpi rows per face, performing the shifted-addition clamping algorithm.

### Parameters Dispatch Summary
The parameters dispatch is handled by the generic `_llk_math_eltwise_unary_sfpu_params_` function, which is shared across all unary SFPU operations. The intended usage for hardtanh would be:

- **Vector mode**: `VectorMode::RC` (all 4 faces processed), which is the standard mode for eltwise unary operations that must process the entire tile.
- **Operation invocation**: The SFPU function is called once per face (4 times total for RC mode). Each invocation processes `ITERATIONS=8` sfpi rows within the face. The three `uint32_t` parameters (param0, param1, param2) are forwarded to the SFPU function via variadic template arguments.
- **DEST address progression**: Standard DEST progression. On Wormhole, `ADDR_MOD_7` is configured with `dest.incr = 0` (the SFPU kernel manages its own `dst_reg++` within the iteration loop). Between faces, `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` is called twice per face to advance by 16 physical DEST rows (= 1 face). On Blackhole, `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice for the same effect. Within the kernel, `dst_reg++` advances 1 sfpi row (= 2 physical DEST rows = 32 elements) per iteration, covering all 8 sfpi rows (256 elements) per face.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `v_if`/`v_endif`), so this follows Style A with inline-commented source code.

The Wormhole B0 and Blackhole implementations are **identical** (byte-for-byte identical files).

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_hardtanh_(const int iterations, std::uint32_t param0, std::uint32_t param1, std::uint32_t param2)
{ // APPROXIMATION_MODE unused (no branch on it), ITERATIONS=8 (default per-face iteration count)
    // All params are in FP16_B format
    // param0 = -(neg_threshold)         i.e. negated lower bound
    // param1 = -(pos_threshold - neg_threshold)  i.e. negated range width
    // param2 = -(pos_threshold)         i.e. negated upper bound

    sfpi::vFloat p0 = sfpi::s2vFloat16b(param0); // broadcast scalar param0 to all SFPU lanes as FP16_B -> vFloat
    sfpi::vFloat p1 = sfpi::s2vFloat16b(param1); // broadcast scalar param1 to all SFPU lanes
    sfpi::vFloat p2 = sfpi::s2vFloat16b(param2); // broadcast scalar param2 to all SFPU lanes
// SFPU microcode
#pragma GCC unroll 0 // prevent loop unrolling to save instruction memory
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat val = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST row pair

        val += p0; // SFPMAD: val = val * 1.0 + p0, i.e. val = x + (-(neg_threshold)) = x - neg_threshold
        v_if (val < 0.0f) // SFPSETCC(LT0): if (x - neg_threshold) < 0, i.e. x < neg_threshold
        {
            val = 0.0f; // SFPLOADI/SFPMOV: set val to 0 on lanes where x is below lower bound
        }
        v_endif; // SFPENCC: restore CC state
        // After this block: val = max(0, x - neg_threshold)

        val += p1; // SFPMAD: val = val + (-(pos_threshold - neg_threshold))
        v_if (val >= 0.0f) // SFPSETCC(GTE0): tests if val >= 0 after shift
        {
            val = 0.0f; // SFPLOADI/SFPMOV: set val to 0 on lanes where x exceeds upper bound
        }
        v_endif; // SFPENCC: restore CC state
        // After this block: if x was in [neg_threshold, pos_threshold], val = x - pos_threshold
        //                    if x < neg_threshold, val = 0 + (-(pos - neg)) = neg_threshold - pos_threshold
        //                    if x > pos_threshold, val = 0

        val += p2; // SFPMAD: val = val + (-(pos_threshold)) = val - pos_threshold
        // Final reconstruction:
        //   x in range:     val = (x - neg_threshold) + (-(pos-neg)) + (-pos) = x - pos + (-pos) ...
        //   Actually: val = (x - neg) + (neg - pos) + (-pos) = x - 2*pos  -- wait, let's trace carefully:
        //   For x in [neg, pos]: val after step1 = x - neg, after step2 = (x-neg)+(neg-pos) = x-pos,
        //                        after step3 = (x-pos)+(-pos) ... This doesn't give x.
        //   Re-reading the params: p2 = -(pos_threshold), so val += p2 means val = val - pos_threshold
        //   Hmm, let me re-trace with actual negated params:
        //   p0 = -neg, p1 = -(pos - neg), p2 = -pos
        //   For x in [neg, pos]:
        //     step1: val = x + (-neg) = x - neg  (>= 0 since x >= neg, so v_if(val<0) is false)
        //     step2: val = (x - neg) + (-(pos - neg)) = x - pos  (<= 0 since x <= pos, so v_if(val>=0) is false)
        //     step3: val = (x - pos) + (-pos) = x - 2*pos  -- this is NOT x
        //   This suggests the param encoding must be different. Looking again at the comment:
        //   "param2 = -(pos_threshold)" -- but for correct output x, we need val = x after all 3 adds.
        //   If we assume the algorithm yields: val = 0 + p2 = -pos for the clamped-high case,
        //   and val = (x-neg) + (neg-pos) + (-pos) for in-range... this doesn't reconstruct to x.
        //   The correct interpretation: p0, p1, p2 are PRE-COMPUTED by the host to make the algebra work.
        //   For hardtanh with bounds [low, high]:
        //     p0 = -low, p1 = -(high - low), p2 = high (not -high)
        //   Then: in-range: (x - low) + (low - high) + high = x. Clamped-low: 0 + (low-high) + high = low.
        //   Clamped-high: (x-low) + 0 + high ... no.
        //   The actual algebra with the clamp-to-zero pattern:
        //     Step 1: val = x + p0. If val < 0 → val = 0 (clamps low).
        //     Step 2: val += p1. If val >= 0 → val = 0 (clamps high).
        //     Step 3: val += p2 (reconstructs final value).

        sfpi::dst_reg[0] = val; // SFPSTORE: write 32 elements back to current DEST row pair

        sfpi::dst_reg++; // advance DEST pointer by 1 sfpi row (= 2 physical rows = 32 elements)
    }
}
```

**Algorithm correctness trace** (with `low = neg_threshold`, `high = pos_threshold`, `p0 = -low`, `p1 = -(high - low)`, `p2 = -high`):

Given the comment in the source says `p2 = -(pos_threshold)`, the three cases evaluate as:

| Case | After step 1 (`+p0`) | Clamp? | After step 2 (`+p1`) | Clamp? | After step 3 (`+p2`) | Result |
|------|----------------------|--------|----------------------|--------|----------------------|--------|
| `x < low` | `x - low < 0` | Yes -> `val=0` | `0 + (low - high) < 0` | No | `(low - high) + (-high) = low - 2*high` | **Not `low`** |
| `low <= x <= high` | `x - low >= 0` | No | `(x - low) + (low - high) = x - high <= 0` | No | `(x - high) + (-high) = x - 2*high` | **Not `x`** |
| `x > high` | `x - low > 0` | No | `(x - low) + (low - high) = x - high > 0` | Yes -> `val=0` | `0 + (-high) = -high` | **Not `high`** |

The literal interpretation of the parameter comments does not yield correct hardtanh results. This indicates the actual parameter encoding passed by the host differs from the comments. The correct encoding that makes this algorithm work is:

- `p0 = -low` (negated lower bound)
- `p1 = low - high` (negated range width, same as `-(high - low)`)
- `p2 = high` (the upper bound itself, **not** negated)

With `p2 = high` (positive):

| Case | After step 1 | Clamp? | After step 2 | Clamp? | After step 3 | Result |
|------|-------------|--------|-------------|--------|-------------|--------|
| `x < low` | `x - low < 0` | Yes -> `0` | `0 + (low - high) < 0` | No | `(low - high) + high = low` | `low` |
| `low <= x <= high` | `x - low >= 0` | No | `(x - low) + (low - high) = x - high <= 0` | No | `(x - high) + high = x` | `x` |
| `x > high` | `x - low > 0` | No | `(x - low) + (low - high) = x - high > 0` | Yes -> `0` | `0 + high = high` | `high` |

This produces correct `hardtanh(x) = clamp(x, low, high)`. The source comment `param2 = -(pos_threshold)` is likely incorrect or describes a different sign convention than what the host actually computes. The algorithm is fundamentally a branch-free shifted-clamping technique.

### SFPU Instructions Used

The kernel uses SFPI C++ abstractions which compile to the following SFPU instructions:

| Instruction | Source Abstraction | Description |
|-------------|-------------------|-------------|
| `SFPLOAD` | `sfpi::dst_reg[0]` (read) | Load 32 elements from current DEST row pair into an LREG |
| `SFPLOADI` | `sfpi::s2vFloat16b(param)`, `0.0f` literal | Load 16-bit immediate (FP16_B scalar) to LREG, broadcast to all lanes |
| `SFPMAD` | `val += p0`, `val += p1`, `val += p2` | Fused multiply-add used as addition: `val = val * 1.0 + p_n`. Three SFPMAD instructions per iteration for the three shift-add steps |
| `SFPSETCC` | `val < 0.0f` (in `v_if`) | Set CC.Res based on sign-bit test (LT0 mode). Also used for `val >= 0.0f` (GTE0 mode) |
| `SFPENCC` | `v_if` / `v_endif` | Enable/disable per-lane condition code masking |
| `SFPPUSHC` | `v_if` (entering conditional) | Push current CC state onto per-lane CC stack |
| `SFPPOPC` | `v_endif` (leaving conditional) | Pop CC state from stack to restore pre-conditional state |
| `SFPMOV` | `val = 0.0f` (inside `v_if`) | Move zero constant into LREG for lanes where condition is true |
| `SFPSTORE` | `sfpi::dst_reg[0] = val` (write) | Store 32 elements from LREG back to current DEST row pair |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST rows** | Input tile data read via `dst_reg[0]` (SFPLOAD), output written back to same location (SFPSTORE). Standard in-place processing. |
| **LREG (val)** | Holds the working value being processed through the three-step clamp algorithm. Loaded from DEST, modified by additions and conditional zeroing, stored back. |
| **LREG (p0)** | Holds `-neg_threshold` broadcast constant, loaded once before the loop via `s2vFloat16b`. Reused across all iterations. |
| **LREG (p1)** | Holds `-(pos_threshold - neg_threshold)` broadcast constant, loaded once before the loop. |
| **LREG (p2)** | Holds the reconstruction offset (per algorithm analysis, likely `pos_threshold`), loaded once before the loop. |
| **CC stack** | Used by `v_if`/`v_endif` to save/restore condition code state. Two `v_if`/`v_endif` blocks per iteration means the CC stack depth reaches 1 at most (non-nested). |

Note: The SFPI compiler assigns specific LREG indices (LREG0-LREG7) to these logical variables. The three parameter constants (`p0`, `p1`, `p2`) are loaded before the loop and persist across all 8 iterations per face invocation. The working value `val` is loaded fresh each iteration from DEST.

### Address Mode Configuration

The address mode is configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType>()` during init. Since `hardtanh` is not in the production `SfpuType` enum, it would use a generic SfpuType (or default configuration). The standard configuration is:

**ADDR_MOD_7** (both Wormhole and Blackhole):
```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}
```

- `dest.incr = 0`: No automatic DEST address increment via the address mode hardware. The SFPU kernel manages its own DEST pointer advancement via `dst_reg++` (which translates to incrementing the SFPU's internal DEST row pointer by `SFP_DESTREG_STRIDE = 2`).
- `srca.incr = 0`, `srcb.incr = 0`: Source register addresses are not incremented (not used by SFPU operations).

There is no special `ADDR_MOD_6` configuration for hardtanh -- it is not in the `typecast`/`unary_max`/`unary_min`/`signbit` set that gets `dest.incr = 2` on ADDR_MOD_6.

This configuration is identical for Wormhole B0 and Blackhole.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
   **Reason**: Determine if HARDTANH is a parametrized type and check dispatch functions
   **Key Findings**: HARDTANH is listed in `is_parametrized_type()` returning true, confirming it takes parameters

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Find `get_op_init_and_func`, `get_op_approx_mode`, and `get_compute_kernel_path` cases for HARDTANH
   **Key Findings**: No HARDTANH case in `get_op_init_and_func_parameterized` (would throw), `get_op_approx_mode` returns false (default), `get_compute_kernel_path` returns `eltwise_sfpu.cpp` (default). The dispatch chain is incomplete.

3. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`
   **Reason**: Understand the TTNN-level hardtanh API
   **Key Findings**: `ttnn::hardtanh(input, min_val=-1.0f, max_val=1.0f)` creates `UnaryWithParam{UnaryOpType::HARDTANH, min_val, max_val}` and dispatches through `unary_impl`

4. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h`
   **Reason**: Read the core SFPU kernel implementation
   **Key Findings**: SFPI-style kernel using shifted-addition clamping with three pre-computed FP16_B parameters. Processes 8 iterations per face. Does not branch on APPROXIMATION_MODE.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_hardtanh.h`
   **Reason**: Compare Blackhole implementation to Wormhole
   **Key Findings**: Byte-for-byte identical to Wormhole implementation

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Understand the parameters dispatch layer (VectorMode handling, DEST advancement)
   **Key Findings**: Generic dispatch for all unary SFPU ops. VectorMode::RC processes all 4 faces. Uses `TTI_SETRWC` to advance DEST between faces on Wormhole.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Compare Blackhole params dispatch to Wormhole
   **Key Findings**: Same logic but uses `_llk_math_eltwise_unary_sfpu_start_`/`_llk_math_eltwise_unary_sfpu_done_` and `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` instead of inline TTI calls

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Understand addr_mod configuration and init sequence
   **Key Findings**: ADDR_MOD_7 set to `dest.incr=0` for all generic SfpuTypes. `_llk_math_eltwise_unary_sfpu_init_` calls `_init_sfpu_config_reg`, `eltwise_unary_sfpu_configure_addrmod`, and `math::reset_counters`.

9. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
   **Reason**: Check if hardtanh has a SfpuType enum entry
   **Key Findings**: No hardtanh entry. Only `unused`, `frac`, `swish`, `atanh`, `sinh` are defined.

10. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h`
    **Reason**: Identify the correct macro pattern for wiring hardtanh dispatch
    **Key Findings**: `SFPU_UNARY_THREE_PARAM_KERNEL_FN` is the appropriate macro for a 3-parameter SFPU function like `_calculate_hardtanh_`.

11. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_clamp.h`
    **Reason**: Compare with clamp (a closely related operation) to understand the parameter passing pattern
    **Key Findings**: Clamp uses a similar shifted-value pattern but with different parameter semantics (direct min/max in FP16_A, offset in FP16_B) and `v_if`/`v_elseif` instead of two sequential `v_if` blocks.

12. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Reference for SFPU instruction semantics, addressing model, and CC mechanism
    **Key Findings**: Confirmed SFPMAD is used for all float additions (no dedicated add), SFPLOAD/SFPSTORE for DEST access, stride-2 addressing model with 8 iterations per face.

13. **File**: `runtime/sfpi/include/sfpi_fp16.h`
    **Reason**: Understand `s2vFloat16b` scalar-to-vector conversion
    **Key Findings**: `s2vFloat16b` is a class that converts a `uint32_t` (containing FP16_B-encoded value) to a broadcast `vFloat` across all SFPU lanes.
