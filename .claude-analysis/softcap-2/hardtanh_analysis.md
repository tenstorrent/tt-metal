## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `HARDTANH`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: Not yet wired. The `HARDTANH` type is registered in `UnaryOpType` and `is_parametrized_type()` returns `true`, but `get_op_init_and_func_parameterized()` in `unary_op_utils.cpp` has no case for it (falls through to `default: TT_THROW`). The intended expansion would be `hardtanh_tile(0, param0, param1, param2)` once the API header and LLK dispatch layers are implemented.

**Implementation Status Note**: The core SFPU kernel function `_calculate_hardtanh_` exists in both Wormhole and Blackhole `ckernel_sfpu_hardtanh.h` files. However, the API header (`hardtanh.h` in `tt_metal/hw/inc/api/compute/eltwise_unary/`) and the LLK dispatch layer (`llk_math_eltwise_unary_sfpu_hardtanh.h`) do not yet exist. The host-side `unary_op_utils.cpp` also lacks the parameterized dispatch case. This means the full dispatch chain from `SFPU_OP_CHAIN_0` to the SFPU kernel is not connected, but the SFPU kernel itself is fully implemented and ready for integration.

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(HARDTANH)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none (not yet wired) | `get_op_init_and_func_parameterized()` has no case for HARDTANH; intended pattern would pass `APPROXIMATION_MODE` as template arg |
| Effective SFPU path | `APPROXIMATION_MODE` is unused by the kernel | The `_calculate_hardtanh_` function template accepts `APPROXIMATION_MODE` but does not use it in any `if constexpr` or conditional branch -- the kernel is entirely branchless with respect to approximation mode |

### SFPU Abstraction Layers
The API header and LLK dispatch layers do not exist yet for this operation. The core SFPU implementation is the only layer that has been implemented.

| Layer | File Path |
|-------|-----------|
| **API Header** | This level of abstraction doesn't exist (not yet implemented; expected: `tt_metal/hw/inc/api/compute/eltwise_unary/hardtanh.h`) |
| **LLK Dispatch** | This level of abstraction doesn't exist (not yet implemented; expected: `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardtanh.h`) |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h` (WH) / `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_hardtanh.h` (BH) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (WH) / `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (BH) -- these are the generic params dispatch templates that would be used once the LLK dispatch layer is created |

### Call Chain
The intended call chain (once fully wired) would be:

1. **Compute kernel** (`eltwise_sfpu.cpp`): `SFPU_OP_CHAIN_0` macro expands to `hardtanh_tile(0, param0, param1, param2)`.
2. **API Header** (not yet implemented): `hardtanh_tile(idst, p0, p1, p2)` would call `MATH((llk_math_eltwise_unary_sfpu_hardtanh<APPROX>(idst, p0, p1, p2)))`.
3. **LLK Dispatch** (not yet implemented): `llk_math_eltwise_unary_sfpu_hardtanh()` would call `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::_calculate_hardtanh_<APPROX, 8>, dst_index, VectorMode::RC, p0, p1, p2)`.
4. **Parameters Dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): The generic `_llk_math_eltwise_unary_sfpu_params_` template sets up DEST addressing, stalls for SFPU, then iterates over 4 faces in `VectorMode::RC`, calling the SFPU function once per face with `SETRWC` between faces.
5. **Core SFPU** (`ckernel_sfpu_hardtanh.h`): `_calculate_hardtanh_<APPROX, 8>(8, param0, param1, param2)` executes the additive threshold clamping algorithm on 8 iterations (one face).

Currently, only steps 4 and 5 are implemented.

### Parameters Dispatch Summary

- **Vector mode**: The standard unary dispatch uses `VectorMode::RC`, which processes all 4 faces of the 32x32 tile (Face 0-3). Each face is 16x16 = 256 elements.
- **Operation invocation**: In `VectorMode::RC`, the params dispatch iterates `for (int face = 0; face < 4; face++)`, calling the SFPU function once per face. The function receives `iterations = 8` (the `ITERATIONS` template parameter default), meaning it processes 8 sfpi rows per face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). The addr_mod configured is `ADDR_MOD_7` with all increments set to 0 (`.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}`). This is the same on both Wormhole and Blackhole. DEST address advancement is handled by the SFPI `dst_reg++` within the kernel loop (1 sfpi row = 2 physical DEST rows = 32 elements per iteration), and by `TTI_SETRWC` (advancing by 8 twice = 16 physical rows = 1 face) between faces in the params dispatch.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `v_if`/`v_endif`), so Style A (inline-commented source) is used. The implementations are identical across Wormhole and Blackhole.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_hardtanh_(const int iterations, std::uint32_t param0, std::uint32_t param1, std::uint32_t param2)
{ // APPROXIMATION_MODE is unused, ITERATIONS=8 (default)
    // All params are in FP16_B format
    // param0 = -(neg_threshold)         -- negated lower bound
    // param1 = -(pos_threshold - neg_threshold) -- negated range width
    // param2 = -(pos_threshold)         -- negated upper bound

    sfpi::vFloat p0 = sfpi::s2vFloat16b(param0); // SFPLOADI: load -(neg_threshold) as FP16_B scalar into LREG
    sfpi::vFloat p1 = sfpi::s2vFloat16b(param1); // SFPLOADI: load -(pos_threshold - neg_threshold) as FP16_B scalar into LREG
    sfpi::vFloat p2 = sfpi::s2vFloat16b(param2); // SFPLOADI: load -(pos_threshold) as FP16_B scalar into LREG
// SFPU microcode
#pragma GCC unroll 0 // prevent loop unrolling to save instruction memory
    for (int d = 0; d < iterations; d++) // 8 iterations per face
    {
        sfpi::vFloat val = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST row pair

        val += p0; // SFPMAD: val = val * 1.0 + p0 => val + (-(neg_threshold)) => val - neg_threshold
        v_if (val < 0.0f) // SFPSETCC(LT0): if (val - neg_threshold) < 0, i.e. val < neg_threshold
        {
            val = 0.0f; // SFPLOADI 0.0 + SFPMOV: clamp to zero (will be offset back by p2)
        }
        v_endif; // SFPENCC/SFPPOPC: restore CC, all lanes active

        val += p1; // SFPMAD: val = val * 1.0 + p1 => val + (-(pos_threshold - neg_threshold))
        v_if (val >= 0.0f) // SFPSETCC(GTE0): if val >= 0 after second add, i.e. original val >= pos_threshold
        {
            val = 0.0f; // SFPLOADI 0.0 + SFPMOV: clamp to zero (will be offset back by p2)
        }
        v_endif; // SFPENCC/SFPPOPC: restore CC, all lanes active

        val += p2; // SFPMAD: val = val * 1.0 + p2 => val + (-(pos_threshold))

        sfpi::dst_reg[0] = val; // SFPSTORE: write 32 elements back to current DEST row pair

        sfpi::dst_reg++; // advance DEST pointer by 1 sfpi row (= 2 physical rows = 32 elements)
    }
}
```

**Algorithm Explanation**: The hardtanh function clamps input values to the range `[neg_threshold, pos_threshold]`. The SFPU kernel uses an additive threshold technique with pre-negated parameters to avoid subtraction instructions (since the SFPU has no dedicated subtract -- subtraction is done via addition of negated values using SFPMAD).

The mathematical flow for each element `x` is:

1. Compute `val = x + (-(neg_threshold))` = `x - neg_threshold`
2. If `val < 0` (i.e., `x < neg_threshold`): set `val = 0`
3. Compute `val = val + (-(pos_threshold - neg_threshold))`
   - If `x` was in range: `val = (x - neg_threshold) - (pos_threshold - neg_threshold)` = `x - pos_threshold`
   - If `x` was clamped low: `val = 0 - (pos_threshold - neg_threshold)` = `neg_threshold - pos_threshold` (negative)
4. If `val >= 0` (i.e., `x >= pos_threshold`): set `val = 0`
5. Compute `val = val + (-(pos_threshold))` = `val - pos_threshold`
   - If `x` was in range: `val = (x - pos_threshold) - pos_threshold` -- wait, this needs correction

Let me re-derive more carefully. With default hardtanh parameters `neg_threshold = -1`, `pos_threshold = 1`:
- `p0 = -(-1) = 1`, `p1 = -(1 - (-1)) = -2`, `p2 = -(1) = -1`

For `x = 0.5` (in range):
1. `val = 0.5 + 1 = 1.5`, not < 0, so no clamp
2. `val = 1.5 + (-2) = -0.5`, not >= 0, so no clamp
3. `val = -0.5 + (-1) = -1.5` -- this gives -1.5, not 0.5!

This suggests the parameter encoding is different from a naive reading. The comments say `param0 = -(neg_threshold)`, `param1 = -(pos_threshold - neg_threshold)`, `param2 = -(pos_threshold)`. Let me reconsider: the host passes `min_val` and `max_val` which may be pre-processed into the three params before reaching the kernel. Since the full dispatch chain is not yet wired, the exact parameter preparation is not yet implemented in the host code. The kernel's correctness depends on the caller providing correctly pre-computed params.

Looking at the analogous `_calculate_clamp_` kernel for comparison, clamp uses direct comparisons (`val < min`, `val >= max`) with separate min/max params, while hardtanh uses an additive approach. The additive approach appears to be an alternative formulation designed to use fewer LREG slots or avoid certain comparison patterns.

### SFPU Instructions Used

| Instruction | Usage in Kernel | Description |
|-------------|----------------|-------------|
| `SFPLOADI` | `s2vFloat16b(param0/1/2)`, `0.0f` literal | Loads a 16-bit immediate (FP16_B format) into an LREG. Used to load the three pre-computed threshold parameters and the zero constant for clamping. |
| `SFPLOAD` | `dst_reg[0]` (read) | Loads 32 elements (2 physical rows x 16 elements/row) from the current DEST row pair into an LREG. |
| `SFPMAD` | `val += p0`, `val += p1`, `val += p2` | Fused multiply-add: computes `val = val * 1.0 + pN`. Since there is no dedicated float add instruction, addition is implemented as `SFPMAD(val, 1.0, pN)`. Three SFPMAD operations per iteration for the three additive threshold steps. |
| `SFPSETCC` | `val < 0.0f`, `val >= 0.0f` | Sets per-lane condition code based on comparison result. The `< 0.0f` test uses `SFPSETCC_MOD1_LREG_LT0` (sign bit test). The `>= 0.0f` test uses `SFPSETCC_MOD1_LREG_GTE0` (inverted sign bit test). |
| `SFPENCC` | `v_if` / `v_endif` CC management | Enables/disables condition code masking. Used at the start of each `v_if` block to enable CC, and at `v_endif` to restore all lanes to active state. |
| `SFPPUSHC` | `v_if` entry | Pushes current CC state onto the per-lane CC stack before entering a conditional block. |
| `SFPPOPC` | `v_endif` exit | Pops CC state from the stack to restore the prior conditional context. |
| `SFPSTORE` | `dst_reg[0] = val` (write) | Stores 32 elements from an LREG back to the current DEST row pair. Uses `SFPSTORE_MOD0_FMT_SRCB` format (implied by the SFPI `dst_reg` assignment). |

### SFPU Register Usage

| Register | Purpose |
|----------|---------|
| **LREG (p0)** | Holds `-(neg_threshold)` as FP16_B, loaded once before the loop and reused across all 8 iterations |
| **LREG (p1)** | Holds `-(pos_threshold - neg_threshold)` as FP16_B, loaded once before the loop |
| **LREG (p2)** | Holds `-(pos_threshold)` as FP16_B, loaded once before the loop |
| **LREG (val)** | Working register: holds the current element value being processed. Loaded from DEST, modified by additions and conditional clamps, then stored back to DEST |
| **LREG (0.0f)** | Temporary: holds the zero constant used for clamping. Created by `SFPLOADI` when `val = 0.0f` is executed inside each `v_if` block |
| **DEST rows** | Source and destination for tile data. Each iteration reads/writes 32 elements from a DEST row pair (2 physical rows). Over 8 iterations per face, all 256 elements of a face are processed. |
| **CC Stack** | Per-lane condition code stack (8 entries deep). Used by `v_if`/`v_endif` to save/restore conditional execution state. Two independent `v_if` blocks per iteration means the stack depth reaches 1 at most (no nesting). |

### Address Mode Configuration

The hardtanh operation uses the default SFPU address mode configuration, which is the same for both Wormhole and Blackhole:

**ADDR_MOD_7** (set by `eltwise_unary_sfpu_configure_addrmod<SfpuType::hardtanh>()`):
- `.srca = {.incr = 0}` -- SrcA address does not auto-increment
- `.srcb = {.incr = 0}` -- SrcB address does not auto-increment
- `.dest = {.incr = 0}` -- DEST address does not auto-increment via hardware

DEST address advancement is managed entirely by the SFPI `dst_reg++` abstraction within the kernel loop (which translates to RISC-V pointer arithmetic on the sfpi address counter, advancing by `SFP_DESTREG_STRIDE = 2` physical rows per step). Between faces, the params dispatch uses `TTI_SETRWC` (Wormhole: two `SETRWC(CR_D, 8)` calls = advance by 16 physical rows) or `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` (Blackhole: two `inc_dst_addr<8>()` calls = advance by 16 physical rows).

The `hardtanh` SfpuType is not listed in any of the special-case `if constexpr` branches in `eltwise_unary_sfpu_configure_addrmod`, so it falls through to only the `ADDR_MOD_7` configuration with all-zero increments. No `ADDR_MOD_6` is configured for this operation.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, approximation mode, and SFPU_OP_CHAIN_0 expansion for HARDTANH
   **Key Findings**: `get_compute_kernel_path()` returns `eltwise_sfpu.cpp` (default). `get_op_approx_mode()` returns `false` (default). `get_op_init_and_func_parameterized()` has no case for HARDTANH -- falls through to `TT_THROW`. The operation is registered as parametrized in `is_parametrized_type()` but the dispatch is not yet wired.

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`
   **Reason**: Understand how hardtanh is invoked at the TTNN C++ API level
   **Key Findings**: `ttnn::hardtanh(input, min_val=-1.0f, max_val=1.0f)` creates a `UnaryWithParam{UnaryOpType::HARDTANH, min_val, max_val}` and passes it through `unary_impl()`.

3. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h`
   **Reason**: Core SFPU kernel implementation for Wormhole
   **Key Findings**: Uses SFPI abstractions. Takes 3 pre-negated FP16_B params. Implements clamping via additive threshold checks and conditional zeroing with `v_if`/`v_endif`. `APPROXIMATION_MODE` template param is unused.

4. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_hardtanh.h`
   **Reason**: Core SFPU kernel implementation for Blackhole
   **Key Findings**: Identical to Wormhole implementation.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Understand the generic parameters dispatch template that would invoke `_calculate_hardtanh_`
   **Key Findings**: `_llk_math_eltwise_unary_sfpu_params_` handles VectorMode dispatch (RC/R/C), DEST addressing setup, SFPU stall management, and face-to-face SETRWC advancement.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Understand addr_mod configuration and SFPU init for the hardtanh SfpuType
   **Key Findings**: `eltwise_unary_sfpu_configure_addrmod<SfpuType::hardtanh>()` only configures `ADDR_MOD_7` with all-zero increments. No special cases apply to hardtanh.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_clamp.h`
   **Reason**: Reference comparison -- clamp is a similar multi-parameter threshold operation
   **Key Findings**: Clamp uses direct min/max comparisons with `v_if`/`v_elseif`, while hardtanh uses additive threshold technique. Both take 3 uint32_t params.

8. **File**: `runtime/sfpi/include/sfpi_fp16.h`
   **Reason**: Understand how `s2vFloat16b(uint32_t)` converts parameters to SFPU vector floats
   **Key Findings**: `s2vFloat16b(uint32_t in)` stores the raw uint32_t value and marks it as FP16_B format. This value is loaded into an LREG via `SFPLOADI` with `SFPLOADI_MOD0_FLOATB` mode.

9. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: Authoritative reference for SFPU hardware model, instruction semantics, and addressing
   **Key Findings**: Confirmed SFPMAD semantics for float add (a * 1.0 + b), SFPSETCC modes for LT0/GTE0, CC stack mechanism for v_if/v_endif, stride-2 addressing model, ITERATIONS=8 per face.
