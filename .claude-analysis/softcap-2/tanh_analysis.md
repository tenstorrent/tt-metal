## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

**IMPORTANT: Deep Nuke Status** -- This branch is a controlled evaluation environment (`vignjatijevic/sfpu-agent-codegen_deeply_nuked_for_rrelu`) where the tanh SFPU implementation has been surgically removed. The following analysis documents the **intended architecture** based on surviving API surfaces, reference patterns, and hardware documentation.

### Unary Dispatch Summary
- **UnaryOpType**: `TANH`
- **Compute kernel**: `eltwise_sfpu.cpp` (default path from `get_compute_kernel_path()` which returns `"eltwise_sfpu.cpp"` for all ops including TANH)
- **SFPU_OP_CHAIN_0 expansion**: `tanh_tile_init(); tanh_tile(0);` (confirmed by test infrastructure in `tests/tt_metal/tt_metal/llk/test_sfpu_compute.cpp:62`)

**Dispatch Status (Nuked)**: The `get_op_init_and_func_default()` switch in `unary_op_utils.cpp` does NOT contain a `case UnaryOpType::TANH` entry -- it was removed during the deep nuke. The `UnaryOpType::TANH` enum value and `REGISTER_UNARY_OPERATION(tanh, TANH)` macro survive in `unary_op_types.hpp:29` and `unary.hpp:110` respectively, but calling `ttnn::tanh()` at runtime will throw `TT_THROW("unexpected op type {}", op_type)` when attempting to generate compute kernel defines. The tile-level API functions `tanh_tile_init()` and `tanh_tile(idst)` survive in `compute_kernel_api.h` but reference deleted LLK functions.

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(UnaryOpType::TANH)` in `unary_op_utils.cpp:73-77` -- switch has only `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | `false` (default) | `tanh_tile<false>(0)` / `tanh_tile_init<false>()` -- the `fast_and_approx` template parameter defaults to `false` in `compute_kernel_api.h:154,177` |
| Effective SFPU path | Software polynomial approximation (accurate mode) | With `fast_and_approx=false`, the historical implementation used a polynomial-based tanh approximation rather than `SFPNONLINEAR` or `SFPLOADMACRO` fast paths |

**Note on `fast_and_approx`**: Unlike most SFPU operations that use the `APPROX` macro, `tanh_tile()` uses its own `fast_and_approx` template parameter that is independent of `math_approx_mode`. The TANH dispatch was never parameterized through `get_op_init_and_func_parameterized()` -- it used the default (non-parameterized) path.

### SFPU Abstraction Layers
All layers below except the API header have been **deleted** in the deep nuke.

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/compute_kernel_api.h` (lines 154-180) -- SURVIVES |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_tanh.h` -- DELETED (deep nuke Phase 1) |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole,quasar}/common/inc/sfpu/ckernel_sfpu_tanh.h` -- DELETED (deep nuke Phase 1, Family 1: Exponential-Composition) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` -- SURVIVES (shared infrastructure, not op-specific) |

### Call Chain
The intended call chain (pre-nuke) follows the standard unary SFPU dispatch pattern:

1. **Compute kernel** (`eltwise_sfpu.cpp`): `SFPU_OP_CHAIN_0` expands to `tanh_tile_init(); tanh_tile(0);`
2. **API header** (`compute_kernel_api.h:178`): `tanh_tile<false>(idst)` calls `MATH((llk_math_eltwise_unary_sfpu_tanh<false, DST_ACCUM_MODE>(idst)));`
3. **LLK dispatch** (DELETED `llk_math_eltwise_unary_sfpu_tanh.h`): Would have called `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::_calculate_tanh_<...>, dst_index, vector_mode)`
4. **Parameters dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): Iterates over 4 faces in VectorMode::RC, calling the SFPU function once per face with ITERATIONS=8
5. **Core SFPU function** (DELETED `ckernel_sfpu_tanh.h`): `_calculate_tanh_<APPROXIMATION_MODE, ITERATIONS>()` -- loops 8 iterations per face, computing tanh on 32 elements per iteration

### Parameters Dispatch Summary
The parameters dispatch uses the shared `_llk_math_eltwise_unary_sfpu_params_` function (which survives the nuke) in `llk_math_eltwise_unary_sfpu_params.h`.

- **Vector mode**: `VectorMode::RC` (default for tanh, processing all 4 faces of the tile)
- **Operation invocation**: The sfpu function (`_calculate_tanh_`) is called once per face (4 times total for RC mode). Each invocation processes 8 iterations (ITERATIONS=8), with each iteration handling 32 elements via the stride-2 addressing model.
- **DEST address progression**: Standard DEST progression. On Wormhole: within a face, `dst_reg++` advances 1 sfpi row (= 2 physical DEST rows) per iteration, covering 32 elements (2 rows x 16 elements/row); between faces, `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` is called twice to advance by 16 physical DEST rows (one face stride). On Blackhole: same pattern but uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which calls `math::inc_dst_addr<8>()` twice.
- **Address mode**: `ADDR_MOD_7` is configured with all-zero increments (`srca.incr=0, srcb.incr=0, dest.incr=0`). SFPU operations manage their own address progression through `dst_reg++` (SFPI abstraction) rather than relying on hardware auto-increment.

### Annotated SFPU Kernel Source

**The core SFPU implementation (`ckernel_sfpu_tanh.h`) has been DELETED** as part of the deep nuke (Phase 1, Family 1: Exponential-Composition). The file was deleted from all three hardware targets (wormhole_b0, blackhole, quasar).

Per the `DEEP_NUKE_MANIFEST.md` (line 40): `ckernel_sfpu_tanh.h` (wh+bh+quasar) contained `_calculate_tanh_` and sigmoid-via-tanh functionality.

**Surviving API surface** (the only code that references tanh SFPU):

```cpp
// File: tt_metal/hw/inc/api/compute/compute_kernel_api.h (lines 154-180)

template <bool fast_and_approx = false>  // fast_and_approx=false when called from SFPU_OP_CHAIN_0
ALWI void tanh_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_tanh_init<fast_and_approx, DST_ACCUM_MODE>()));  // DELETED target
}

template <bool fast_and_approx = false>  // fast_and_approx=false when called from SFPU_OP_CHAIN_0
ALWI void tanh_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_tanh<fast_and_approx, DST_ACCUM_MODE>(idst)));  // DELETED target
}
```

**Reconstructed kernel pattern** (based on the surviving `calculate_swish` which uses sigmoid internally, and the established patterns in surviving SFPU kernels):

The deleted `_calculate_tanh_` function would have followed this general structure, consistent with the SFPI-based kernel style (Style A):

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_tanh.h
// [DELETED - RECONSTRUCTED PATTERN]
//
// tanh(x) can be computed via:
//   tanh(x) = 2 * sigmoid(2x) - 1
// or directly as a polynomial/piecewise approximation.
//
// The historical implementation likely used sigmoid composition or
// a direct polynomial approximation similar to the swish kernel's
// sigmoid approximation but applied to produce tanh output.
//
// Template signature (inferred from API and surviving patterns):
// template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
// inline void _calculate_tanh_(const int iterations) {
//     #pragma GCC unroll 8
//     for (int d = 0; d < iterations; d++) {
//         sfpi::vFloat x = sfpi::dst_reg[0];
//         // ... compute tanh(x) using polynomial approximation ...
//         sfpi::dst_reg[0] = result;
//         sfpi::dst_reg++;
//     }
// }
```

**Hardware-accelerated path (Quasar only)**: On Quasar, `SFPNONLINEAR` with `instr_mod1 = TANH_MODE (0x5)` provides a hardware-accelerated tanh approximation with max 1 ULP error for FP16_B. This instruction has opcode `0x99` and computes tanh in a single cycle. The Quasar kernel would have used:

```cpp
// Quasar-only pattern (DELETED):
// TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);  // load from DEST
// TTI_SFPNONLINEAR(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpnonlinear::TANH_MODE);  // HW tanh
// TTI_SFPSTORE(0, p_sfpu::LREG1, ADDR_MOD_7, 0);  // store to DEST
```

### SFPU Instructions Used

Since the implementation is deleted, the following lists instructions that **would be used** based on the surviving sigmoid/swish approximation pattern and the tanh mathematical relationship:

**Wormhole/Blackhole (software approximation):**

| Instruction | Description | Usage Context |
|-------------|-------------|---------------|
| `SFPLOAD` | Load data from DEST register to LREG | Loading input elements for computation (hidden by `dst_reg[0]` read) |
| `SFPSTORE` | Store LREG data back to DEST register | Writing computed tanh values back (hidden by `dst_reg[0] = ...` assignment) |
| `SFPMAD` | Fused multiply-add (a * b + c) | Polynomial evaluation via Horner's method for tanh approximation. Also used for `vFloat + vFloat` (as `a * 1.0 + b`) and `vFloat * vFloat` (as `a * b + 0.0`) |
| `SFPABS` | Absolute value | Computing `|x|` for symmetric function evaluation (tanh is an odd function) |
| `SFPSETCC` | Set condition code based on comparison | Branch conditions for piecewise approximation segments (e.g., `|x| < threshold`) |
| `SFPENCC` | Enable/disable condition code | Activating/deactivating predicated execution for `v_if`/`v_endif` blocks |
| `SFPCOMPC` | Complement condition code | Implementing `v_else` branches |
| `SFPPUSHC` / `SFPPOPC` | Push/pop condition code stack | Nested conditional branches for multi-segment piecewise approximation |

**Quasar (hardware-accelerated):**

| Instruction | Description | Usage Context |
|-------------|-------------|---------------|
| `SFPLOAD` | Load data from DEST register to LREG | Loading input elements |
| `SFPNONLINEAR` (mode=0x5) | Hardware tanh approximation | Single-instruction tanh computation, max 1 ULP error for FP16_B |
| `SFPSTORE` | Store LREG data back to DEST register | Writing computed results |

### SFPU Register Usage

**Standard register usage pattern** (inferred from surviving SFPU kernels):

| Register | Usage |
|----------|-------|
| `dst_reg[0]` (DEST) | Input/output: reads input value, writes computed tanh result. Maps to current sfpi address in DEST register file. |
| LREG0 | Implicit working register for `dst_reg[0]` access (SFPLOAD target) |
| LREG1-LREG3 | Intermediate computation registers for polynomial evaluation. Used for storing partial results during Horner's method evaluation. |
| Programmable Constants (`vConstFloatPrgm0`-`vConstFloatPrgm2`) | May be used by `_init_tanh_()` to preload polynomial coefficients or key constants (e.g., breakpoint thresholds). Set via `SFPCONFIG` during initialization. |
| `vConst1` (Fixed Const 2 = 1.0) | Used for computing `1.0 - sigmoid(|x|)` in the negative-x branch, or for saturation to +/-1.0 for large |x|. |

### Address Mode Configuration

The address mode for tanh follows the standard unary SFPU pattern configured by `eltwise_unary_sfpu_configure_addrmod()`:

**Wormhole B0 and Blackhole:**

```cpp
// ADDR_MOD_7: all-zero increments (used by all standard SFPU unary ops)
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}.set(ADDR_MOD_7);
```

This is the standard configuration where SFPU address progression is managed by the kernel itself (`dst_reg++` in the SFPI loop) rather than hardware auto-increment. ADDR_MOD_7 is chosen to avoid conflicts with ADDR_MOD_0 and ADDR_MOD_2, which are used by the A2D (Acquire-to-DEST) pipeline that runs concurrently on the unpack thread.

Tanh is not in the special-case list for `ADDR_MOD_6` (which is configured with `dest.incr=2` only for `typecast`, `unary_max/min`, and `signbit` operations). Tanh would use a generic `SfpuType` value (before the nuke, likely `SfpuType::tanh` which has been removed from the enum).

**Quasar:**
The Quasar LLK infrastructure uses a different dispatch model and may configure address modes differently, but the fundamental ADDR_MOD_7 pattern is consistent across hardware generations.

## Local Knowledge Sources
### Local References
1. **File**: `DEEP_NUKE_MANIFEST.md`
   **Reason**: Understanding what was deleted and what survives in this evaluation branch
   **Key Findings**: `ckernel_sfpu_tanh.h` was deleted in Phase 1 (Family 1: Exponential-Composition) from all three hardware targets (wh+bh+quasar). The LLK dispatch, compute API references, and `SfpuType::tanh` enum value were also removed. The dispatch case in `get_op_init_and_func_default` was removed.

2. **File**: `tt_metal/hw/inc/api/compute/compute_kernel_api.h` (lines 154-180)
   **Reason**: Surviving API surface for tanh_tile() and tanh_tile_init()
   **Key Findings**: Both functions use `fast_and_approx` template parameter (default=false) and `DST_ACCUM_MODE`. They call `llk_math_eltwise_unary_sfpu_tanh` and `llk_math_eltwise_unary_sfpu_tanh_init` which no longer exist.

3. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Checking dispatch path for TANH
   **Key Findings**: `get_op_approx_mode()` returns `false` for all ops (default case only). `get_compute_kernel_path()` returns `"eltwise_sfpu.cpp"` for all ops. `get_op_init_and_func_default()` does NOT have a TANH case -- throws at runtime.

4. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` (line 110)
   **Reason**: Checking how TANH is registered
   **Key Findings**: `REGISTER_UNARY_OPERATION(tanh, TANH)` -- simple registration without `fast_and_approximate_mode` parameter (unlike `exp` or `gelu` which use `REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE`).

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Understanding the parameters dispatch pattern shared by all unary SFPU ops
   **Key Findings**: `_llk_math_eltwise_unary_sfpu_params_` iterates over faces based on VectorMode (RC: 4 faces, R: 2 faces, C: 2 faces). Uses `TTI_SETRWC` to advance between faces. Calls the sfpu function once per face.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Understanding init, start, done, and address mode configuration
   **Key Findings**: `eltwise_unary_sfpu_configure_addrmod()` sets ADDR_MOD_7 with all-zero increments for standard ops. `_llk_math_eltwise_unary_sfpu_init_()` initializes SFPU config, address modes, and resets counters.

7. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h`
   **Reason**: Reference for LLK dispatch pattern (surviving op)
   **Key Findings**: Standard pattern: init calls `llk_math_eltwise_unary_sfpu_init<SfpuType::swish, APPROXIMATE>()`, compute calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_swish<APPROXIMATE, ITERATIONS>, dst_index, vector_mode)`.

8. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h`
   **Reason**: Reference for SFPI-based sigmoid/activation kernel pattern
   **Key Findings**: Swish uses piecewise polynomial sigmoid approximation (3 segments: polynomial for |x|<=2.5, linear for 2.5<|x|<=5.0, saturation for |x|>5.0). Uses `sfpi::vFloat`, `sfpi::abs()`, `v_if`/`v_endif` for CC-guarded piecewise logic. Max ~4 ULP error for bfloat16.

9. **File**: `tt_metal/third_party/tt_llk/tt_llk_quasar/common/inc/ckernel_instr_params.h` (line 460-467)
   **Reason**: Verifying SFPNONLINEAR TANH_MODE on Quasar
   **Key Findings**: `p_sfpnonlinear::TANH_MODE = 0x5`. Only exists on Quasar -- Wormhole and Blackhole do not have SFPNONLINEAR.

10. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Authoritative SFPU hardware reference
    **Key Findings**: SFPNONLINEAR (opcode 0x99) supports tanh at InstrMod=5 with max 1 ULP FP16_B error. SFPU uses stride-2 addressing (32 elements per iteration, 8 iterations per face, 4 faces per tile). SFPMAD is the core math instruction for all float arithmetic.

11. **File**: `tests/tt_metal/tt_metal/llk/test_sfpu_compute.cpp` (line 62)
    **Reason**: Confirming the expected SFPU_OP_CHAIN_0 expansion for tanh
    **Key Findings**: `{"tanh", {{"SFPU_OP_CHAIN_0", "tanh_tile_init(); tanh_tile(0);"}}}` -- confirms the tile-level API call pattern.

12. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
    **Reason**: Checking if SfpuType::tanh exists
    **Key Findings**: `SfpuType` enum only contains: `unused`, `frac`, `swish`, `atanh`, `sinh`. The `tanh` entry was removed during the deep nuke.
