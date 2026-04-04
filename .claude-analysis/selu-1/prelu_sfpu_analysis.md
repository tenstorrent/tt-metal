## SFPU Kernel Implementation

**STATUS: OPERATION PARTIALLY PRESENT — SFPU KERNEL NUKED, ENUM RETAINED**

The `prelu_sfpu` (Parametric ReLU) operation exists as `UnaryOpType::PRELU_SFPU` in the `UnaryOpType` enum but all SFPU implementation layers (compute API, LLK wrappers, SFPU kernel, SfpuType enum entry, C++ API registration, and `unary_op_utils.cpp` dispatch logic) were removed in commit `db3f683e0a5` ("Batch nuke all SFPU unary eltwise operations"). The enum was intentionally preserved to avoid breaking downstream consumers (matmul/conv/backward ops).

However, the **structurally identical** `_calculate_lrelu_` function (Leaky ReLU) remains in the shared LLK library at `ckernel_sfpu_relu.h`. PReLU and Leaky ReLU are computationally identical at the SFPU kernel level — both multiply `x` by a slope/weight when `x < 0` and pass through `x` unchanged when `x >= 0`. The only difference is at the Python/TTNN level where PReLU's weight is learned vs. Leaky ReLU's slope is fixed.

### Evidence Summary

| Check | Result |
|-------|--------|
| `UnaryOpType::PRELU_SFPU` in `unary_op_types.hpp` | **Present** — line 110 |
| `PRELU_SFPU` case in `get_op_init_and_func_parameterized()` | **Not found** — nuked |
| `PRELU_SFPU` case in `get_macro_definition()` | **Not found** — falls through to default `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` |
| `SfpuType::prelu` in `llk_sfpu_types.h` (metal build) | **Not found** — enum nuked to `{ unused = 0 }` |
| `SfpuType::prelu` in `llk_sfpu_types.h` (LLK test helpers) | **Present** — line 101 |
| `ckernel_sfpu_prelu.h` in `tt_metal/hw/ckernels/` | **Not found** — nuked |
| `ckernel_sfpu_prelu.h` in `tt_metal/third_party/tt_llk/` | **Not found** — was in custom layer, not shared LLK |
| Compute API header for `prelu_tile` / `prelu_tile_init` | **Not found** — nuked |
| `ttnn::prelu_sfpu()` C++ function definition | **Not found** — nuked (only call site in `binary_composite_op.cpp` remains) |
| `_calculate_lrelu_` (structurally identical) in `ckernel_sfpu_relu.h` | **Present** — shared LLK library |
| `docs/sfpu_operations/key_notes/prelu_sfpu_key_notes.md` | **Present** |
| Pre-nuke catalog (`unary_eltwise_sfpu_list.md`) | **Present** — documents `SFPU_OP_PRELU_INCLUDE` macro group |

### Unary Dispatch Summary

- **UnaryOpType**: `PRELU_SFPU` — present in enum, **but dispatch logic nuked**
- **Compute kernel**: `eltwise_sfpu.cpp` (default path via `get_compute_kernel_path()`)
- **SFPU_OP_CHAIN_0 expansion**: Would have been `prelu_tile(0, param0)` — **nuked**
- **Pre-nuke macro group**: `SFPU_OP_PRELU_INCLUDE`

#### Approximation Mode Resolution

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode()` — switch has only `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | N/A | `get_op_init_and_func_parameterized()` — no case for PRELU_SFPU (nuked) |
| Effective SFPU path | N/A | No SFPU implementation exists in current codebase |

### Mathematical Definition

From `docs/sfpu_operations/key_notes/prelu_sfpu_key_notes.md`:

```
f(x) = max(0, x) + weight * min(0, x)
```

Equivalently:
```
f(x) = x              if x >= 0
f(x) = weight * x     if x < 0
```

Where:
- **weight**: learnable parameter, default initialization = 0.25
- **PyTorch reference**: `torch.nn.PReLU`

### SFPU Abstraction Layers

All layers were nuked. The table shows what would have existed based on pre-nuke catalog and structural analysis of similar operations (ELU, CELU, Leaky ReLU):

| Layer | File Path | Status |
|-------|-----------|--------|
| **API Header** | Would have been `api/compute/eltwise_unary/prelu.h` (exposing `prelu_tile` / `prelu_tile_init`) | **Nuked** |
| **LLK Dispatch** | Would have been `llk_math_eltwise_unary_sfpu_prelu.h` or dispatched via macros in `llk_math_eltwise_unary_sfpu_macros.h` | **Nuked** |
| **Core SFPU Implementation** | Would have been `ckernel_sfpu_prelu.h` in custom layer (`tt_metal/hw/ckernels/*/metal/`) | **Nuked** |
| **Parameters Dispatch** | `llk_math_eltwise_unary_sfpu_params.h` (shared infrastructure, still present) | **Present** |

### Call Chain (Pre-Nuke, Reconstructed)

Based on the pre-nuke catalog and the pattern of similar operations (ELU, CELU, Leaky ReLU):

1. `SFPU_OP_CHAIN_0` expands to `prelu_tile(0, param0)` in `eltwise_sfpu.cpp`
2. `prelu_tile()` → declared in compute API header `prelu.h` → calls LLK dispatch function
3. LLK dispatch → calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>()` with `ckernel::sfpu::_calculate_prelu_<APPROXIMATE>` as the functor and `param0` (weight) as the runtime argument
4. `_llk_math_eltwise_unary_sfpu_params_()` → iterates over 4 faces (VectorMode::RC), calling the SFPU kernel function once per face with ITERATIONS=8
5. Core SFPU kernel (`_calculate_prelu_`) → for each of 8 iterations, loads value from DEST, conditionally multiplies by weight if negative, stores back

### Parameters Dispatch Summary (Reconstructed)

Based on the shared `_llk_math_eltwise_unary_sfpu_params_` infrastructure (which is still present) and the pattern of similar single-parameter operations:

- **Vector mode**: `VectorMode::RC` — processes all 4 faces of the tile
- **Operation invocation**: The params dispatch function calls the SFPU kernel functor once per face (4 calls total). Each call processes `ITERATIONS=8` sfpi rows, covering one face (256 elements).
- **DEST address progression**: Standard DEST progression. Within each face, `dst_reg++` advances 1 sfpi row per iteration (8 iterations per face). Between faces, `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` x2 advances to the next face.

### Annotated SFPU Kernel Source — Structurally Identical Reference

Since the `ckernel_sfpu_prelu.h` file was nuked, the structurally identical `_calculate_lrelu_` (Leaky ReLU) function from the shared LLK library is provided as a reference. PReLU and Leaky ReLU are **computationally identical** at the SFPU level — both compute `x * slope` when `x < 0` and leave `x` unchanged when `x >= 0`. The only difference is at the Python level: PReLU's weight is learned via backpropagation, while Leaky ReLU's slope is a fixed hyperparameter.

The `_calculate_lrelu_` function uses **raw TT_/TTI_ instructions** with explicit condition code manipulation:

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h

template <bool APPROXIMATION_MODE>
inline void _calculate_lrelu_(const int iterations, std::uint32_t slope)
{
    TT_SFPLOADI(p_sfpu::LREG2, 10, slope & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, 8, slope >> 16);
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);        // load from dest into lreg[0]
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 0);                                         // condition - if value in LREG0 is negative //will set cc result reg
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG0, 0); // Multiply LREG0 * LREG2 (x * slope)
        TTI_SFPENCC(0, 0, 0, 0);                                                      // clear cc result reg
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);       // store from lreg0 into dest register
        sfpi::dst_reg++;
    }
}
```

#### CC State Machine Diagram

```
┌────────────────────────────────────────────────────────────────┐
│ CC State Machine for _calculate_lrelu_                         │
│ Initial CC state: CLEAR (all lanes enabled)                    │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│ ── Setup (before loop) ──                                      │
│                                                                │
│ TT_SFPLOADI(LREG2, 10, slope_lo)  // Load low 16 bits of      │
│                                    // slope into LREG2         │
│ TT_SFPLOADI(LREG2, 8, slope_hi)   // Load high 16 bits of     │
│                                    // slope into LREG2         │
│                                    // CC: unchanged (CLEAR)    │
│                                                                │
│ ── Per-iteration loop (d = 0..iterations-1) ──                 │
│                                                                │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │ SFPLOAD(LREG0, DEFAULT, ADDR_MOD_3, 0)                  │   │
│ │   Load 32 elements from DEST into LREG0                  │   │
│ │   CC: unchanged (CLEAR)                                  │   │
│ └──────────────────────────────────────────────────────────┘   │
│                         │                                      │
│                         ▼                                      │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │ ★ SFPSETCC(0, LREG0, 0, 0)                              │   │
│ │   Sets CC based on sign of LREG0 values                  │   │
│ │   CC: SET — lanes where LREG0 < 0 are ACTIVE             │   │
│ │          — lanes where LREG0 >= 0 are INACTIVE            │   │
│ └──────────────────────────────────────────────────────────┘   │
│                         │                                      │
│                         ▼                                      │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │ SFPMUL(LREG0, LREG2, LCONST_0, LREG0, 0)  [CC-GUARDED] │   │
│ │   LREG0 = LREG0 * LREG2 + LCONST_0 (= x * slope + 0)  │   │
│ │   Only executes on ACTIVE lanes (x < 0)                  │   │
│ │   Lanes where x >= 0: LREG0 retains original value       │   │
│ │   CC: unchanged (SET)                                    │   │
│ └──────────────────────────────────────────────────────────┘   │
│                         │                                      │
│                         ▼                                      │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │ ★ SFPENCC(0, 0, 0, 0)                                   │   │
│ │   Clears the condition code                              │   │
│ │   CC: CLEAR — all lanes enabled again                    │   │
│ └──────────────────────────────────────────────────────────┘   │
│                         │                                      │
│                         ▼                                      │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │ SFPSTORE(LREG0, DEFAULT, ADDR_MOD_3, 0)                 │   │
│ │   Store all 32 elements from LREG0 back to DEST          │   │
│ │   CC: CLEAR (all lanes written)                          │   │
│ └──────────────────────────────────────────────────────────┘   │
│                         │                                      │
│                         ▼                                      │
│ dst_reg++ (advance to next sfpi row)                           │
│                                                                │
│ ── End loop ──                                                 │
│                                                                │
│ ★ = CC-modifying instruction                                   │
│ [CC-GUARDED] = instruction whose effect is masked by CC        │
├────────────────────────────────────────────────────────────────┤
│ Net effect per iteration:                                      │
│   For lanes where x >= 0: output = x  (unchanged)             │
│   For lanes where x < 0:  output = x * slope                  │
│   This implements PReLU / Leaky ReLU identically.              │
└────────────────────────────────────────────────────────────────┘
```

**Key insight**: The SFPSETCC/SFPENCC pair creates a brief CC-active window where only the SFPMUL instruction is conditionally executed. The SFPMUL uses the multiply-accumulate form `LREG0 * LREG2 + LCONST_0` (where LCONST_0 = 0), effectively computing `x * slope`. Only lanes where LREG0 is negative execute the multiply; positive lanes retain the original value.

### SFPU Instructions Used

Based on the `_calculate_lrelu_` reference (structurally identical to what PReLU would use):

| Instruction | Purpose |
|-------------|---------|
| `TT_SFPLOADI` | Loads an immediate 16-bit value into a local register. Used twice to construct the full 32-bit `slope` (weight) parameter in LREG2 — first the low 16 bits (mode 10), then the high 16 bits (mode 8). |
| `TTI_SFPLOAD` | Loads 32 elements from the DEST register file into a local register (LREG0). Uses ADDR_MOD_3 for address mode. |
| `TTI_SFPSETCC` | Sets the condition code (CC) register based on the sign of values in LREG0. Lanes with negative values become ACTIVE; lanes with non-negative values become INACTIVE. |
| `TTI_SFPMUL` | Multiply-accumulate: `LREG0 = LREG0 * LREG2 + LCONST_0`. CC-guarded: only executes on active lanes (negative values). Inactive lanes (positive values) retain their original value. |
| `TTI_SFPENCC` | Clears the condition code, re-enabling all lanes for subsequent instructions. |
| `TTI_SFPSTORE` | Stores 32 elements from LREG0 back to the DEST register file. Uses ADDR_MOD_3 for address mode. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| `LREG0` | Working register: holds the input value loaded from DEST, receives the result (conditionally multiplied), and is stored back to DEST |
| `LREG2` | Holds the slope/weight parameter. Loaded once before the loop via two `TT_SFPLOADI` instructions (low 16 bits, then high 16 bits). Remains constant across all iterations. |
| `LCONST_0` | Constant zero, used as the addend in the multiply-accumulate (`x * slope + 0 = x * slope`) |
| `dst_reg` | SFPI DEST address pointer, incremented by 1 sfpi row per iteration. Each increment corresponds to 2 physical DEST rows = 32 elements. |

### Address Mode Configuration

The `_calculate_lrelu_` kernel uses `ADDR_MOD_3` for both `SFPLOAD` and `SFPSTORE` instructions. This is a per-instruction address mode selector (not a global configuration).

The standard unary SFPU init function (`_llk_math_eltwise_unary_sfpu_init_<SfpuType::prelu>`) configures:
- `ADDR_MOD_7`: `.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}` — no auto-increment (used by the infrastructure, not the SFPU kernel directly)

The kernel itself uses `ADDR_MOD_3` which is configured elsewhere in the address mode table. The `dst_reg++` call in the SFPI layer handles the actual DEST address progression between iterations (advancing 1 sfpi row = 2 physical DEST rows = 32 elements per iteration).

**Note**: Since `SfpuType::prelu` does not appear in the special-case branches of `eltwise_unary_sfpu_configure_addrmod<>()`, it would use only the default `ADDR_MOD_7` configuration (dest increment = 0). This is the standard configuration for the vast majority of unary SFPU operations.

## Local Knowledge Sources

### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`
   **Reason**: Verify `PRELU_SFPU` exists in UnaryOpType enum
   **Key Findings**: Present at line 110

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Check dispatch logic for PRELU_SFPU (init/func, approx mode, macro definition, compute kernel path)
   **Key Findings**: All PRELU_SFPU-specific cases were nuked. Only MISH, LOGIT, IDENTITY, DROPOUT remain. Default path → `eltwise_sfpu.cpp`, approx mode → `false`, macro → `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE`.

3. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
   **Reason**: Check SfpuType enum for prelu entry
   **Key Findings**: Enum nuked to `{ unused = 0 }` — no prelu entry

4. **File**: `tt_metal/third_party/tt_llk/tests/helpers/include/llk_sfpu_types.h`
   **Reason**: Check pre-nuke SfpuType enum
   **Key Findings**: `SfpuType::prelu` present at line 101

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h`
   **Reason**: Contains `_calculate_lrelu_` — structurally identical to PReLU SFPU kernel
   **Key Findings**: Raw TTI_ instruction-based kernel using SFPSETCC/SFPMUL/SFPENCC pattern for conditional multiply. Uses LREG0 (working), LREG2 (slope), ADDR_MOD_3.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_elu.h`
   **Reason**: ELU is structurally related (conditional branch on sign, different negative path)
   **Key Findings**: Uses SFPI abstractions (`v_if`, `sfpi::dst_reg`), `_calculate_exponential_piecewise_` for negative branch. Shows the SFPI-style alternative to the raw TTI_ approach used by lrelu.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Shared parameters dispatch infrastructure for all unary SFPU ops
   **Key Findings**: `_llk_math_eltwise_unary_sfpu_params_` template handles VectorMode::RC (4 faces), VectorMode::R (2 faces), VectorMode::C (2 faces). Uses SETRWC for face-to-face address advancement.

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Shared SFPU init infrastructure
   **Key Findings**: `_llk_math_eltwise_unary_sfpu_init_<SfpuType>()` configures ADDR_MOD_7 with dest increment 0. Special cases for topk, typecast, max/min operations only.

9. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h`
   **Reason**: Shared macro definitions for SFPU kernel dispatch
   **Key Findings**: PReLU would have used `SFPU_UNARY_ONE_PARAM_KERNEL_FN` macro pattern (one runtime uint32_t parameter for weight).

10. **File**: `docs/sfpu_operations/unary_eltwise_sfpu_list.md`
    **Reason**: Pre-nuke catalog documenting all SFPU operations and their macro groups
    **Key Findings**: PRELU_SFPU documented under `SFPU_OP_PRELU_INCLUDE`, parametrized with weight param.

11. **File**: `docs/sfpu_operations/key_notes/prelu_sfpu_key_notes.md`
    **Reason**: Mathematical definition and PyTorch reference
    **Key Findings**: Formula `max(0, x) + weight * min(0, x)`, weight default init 0.25, PyTorch `torch.nn.PReLU`.

12. **File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_composite_op.cpp`
    **Reason**: Call site for `ttnn::prelu_sfpu()`
    **Key Findings**: `prelu()` delegates to `ttnn::prelu_sfpu(input, weight)` for scalar weight case. The function declaration was nuked — only the call site remains.

13. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
    **Reason**: Compute kernel that PRELU_SFPU routes through
    **Key Findings**: Standard SFPU dispatch pattern: `copy_tile` → `SFPU_OP_CHAIN_0` → `pack_tile`. This is the default compute kernel for operations without a custom kernel path.

14. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_init.h`
    **Reason**: Init function template for unary SFPU operations
    **Key Findings**: Two overloads — one that just calls `_llk_math_eltwise_unary_sfpu_init_<sfpu_op>()`, another that also invokes a custom init callback. PReLU would have used the second form with an init callback (no-op or lightweight, since no polynomial coefficients are needed).
