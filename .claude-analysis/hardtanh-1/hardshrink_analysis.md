## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel functions that the hardshrink compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `HARDSHRINK`
- **Compute kernel**: `hardshrink_kernel.cpp` (non-FLOAT32 dtypes) / `hardshrink_kernel_sfpu.cpp` (FLOAT32 dtype)
- **SFPU_OP_CHAIN_0 expansion**: **Not used** — HARDSHRINK uses a dedicated compute kernel instead of the standard `eltwise_sfpu.cpp` + `SFPU_OP_CHAIN_0` dispatch pattern. The dedicated kernel directly calls `ltz_tile(0)` and `gtz_tile(0)` inline alongside FPU binary operations.

**Important architectural note**: HARDSHRINK is a **composite** compute kernel that mixes FPU binary operations (`binary_dest_reuse_tiles` with ELWADD/ELWSUB/ELWMUL or SFPU binary ops `add_binary_tile`/`sub_binary_tile`/`mul_binary_tile`) with SFPU comparison operations (`ltz_tile`, `gtz_tile`). It also uses `fill_tile` (SFPU fill). There is no single `_calculate_hardshrink_` SFPU function; instead, the hardshrink math is decomposed into a sequence of primitive tile-level operations composed directly in the compute kernel.

#### Mathematical Definition

`hardshrink(a, λ) = a⋅1(a+λ<0) + a⋅1(a−λ>0)`

This is equivalent to: if `|a| > λ`, output `a`; otherwise output `0`. The `λ` (lambda) parameter defaults to `0.5` and is passed as `packed_scalar1` from the program factory.

#### Approximation Mode Resolution

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(HARDSHRINK)` in `unary_op_utils.cpp` — falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none | HARDSHRINK does not use `SFPU_OP_CHAIN_0`; it uses dedicated kernel files. The `APPROX` macro defined by ComputeConfig propagates to the inline SFPU calls (`ltz_tile`, `gtz_tile`, `fill_tile`) but `_calculate_zero_comp_` is templated on `APPROXIMATION_MODE` which is unused in the comparison branch (no approximation-dependent code path). |
| Effective SFPU path | Approximation mode has no effect | The `apply_zero_comp<less_than_zero>` and `apply_zero_comp<greater_than_zero>` specializations use simple `v_if (v >= 0)` / `v_if (v > 0)` comparisons with no approximation-dependent branches |

### SFPU Abstraction Layers

HARDSHRINK uses **two** different routing variants depending on input dtype. Both share the same SFPU comparison functions but differ in how FPU binary operations are invoked.

#### Variant 1: `hardshrink_kernel.cpp` (non-FLOAT32 — e.g., BFLOAT16)

Uses FPU binary operations via `binary_dest_reuse_tiles` API (hardware FPU math engine, not SFPU).

| Layer | File Path |
|-------|-----------|
| **Compute Kernel** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/hardshrink_kernel.cpp` |
| **SFPU API Header (ltz/gtz)** | Generated `comp.h` (not in source tree — generated at build time) |
| **SFPU API Header (fill)** | Generated `fill.h` (not in source tree — generated at build time) |
| **LLK Dispatch (SFPU params)** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |
| **LLK Macros** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h` |
| **Core SFPU (comparison)** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_comp.h` |
| **Core SFPU (fill)** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_fill.h` |
| **FPU Binary API** | `tt_metal/hw/inc/api/compute/eltwise_binary.h` (`binary_dest_reuse_tiles`) |

#### Variant 2: `hardshrink_kernel_sfpu.cpp` (FLOAT32)

Uses SFPU binary operations via `add_binary_tile`/`sub_binary_tile`/`mul_binary_tile` API.

| Layer | File Path |
|-------|-----------|
| **Compute Kernel** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/hardshrink_kernel_sfpu.cpp` |
| **SFPU API Header (ltz/gtz)** | Generated `comp.h` (not in source tree — generated at build time) |
| **SFPU API Header (fill)** | Generated `fill.h` (not in source tree — generated at build time) |
| **SFPU Binary API** | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` (`add_binary_tile`, `sub_binary_tile`, `mul_binary_tile`) |
| **LLK Dispatch (SFPU params)** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |
| **Core SFPU (comparison)** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_comp.h` |
| **Core SFPU (fill)** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_fill.h` |

### Call Chain

#### ltz_tile(0) dispatch path
1. `ltz_tile(0)` → generated API header (`comp.h`) calls the `SFPU_ZERO_KERNEL` macro
2. `SFPU_ZERO_KERNEL(less_than_zero, RC, false, 0)` → expands to `_llk_math_eltwise_unary_sfpu_params_<false>(ckernel::sfpu::calculate_comp<false, SfpuType::less_than_zero>, 0, (int)VectorMode::RC, 8)`
3. `_llk_math_eltwise_unary_sfpu_params_` → sets DEST write address, stalls SFPU, loops over 4 faces calling `calculate_comp` once per face with `SETRWC` between faces
4. `calculate_comp` → calls `_calculate_zero_comp_<false, SfpuType::less_than_zero, 8>()` (or the older `_calculate_comp_` variant depending on which generated header path is used)
5. `_calculate_zero_comp_` → loops 8 iterations per face, calling `apply_zero_comp<SfpuType::less_than_zero>` on each dst_reg element
6. `apply_zero_comp<less_than_zero>` → `v_if (v >= 0) { v = 0; } v_else { v = 1; } v_endif;`

#### gtz_tile(0) dispatch path
Same as `ltz_tile` but with `SfpuType::greater_than_zero`:
1. `gtz_tile(0)` → `SFPU_ZERO_KERNEL(greater_than_zero, RC, false, 0)`
2. Through the same LLK layers to `apply_zero_comp<SfpuType::greater_than_zero>`
3. `apply_zero_comp<greater_than_zero>` → `v_if (v > 0) { v = 1; } v_else { v = 0; } v_endif;`

#### fill_tile(0, λ) dispatch path
1. `fill_tile(0, λ)` → generated API header (`fill.h`) dispatches to `_calculate_fill_<APPROX, ITERATIONS>(value)`
2. `_calculate_fill_` → loads `value` into a `vFloat`, then loops `ITERATIONS` times writing `fill_val` to `dst_reg[0]`, advancing `dst_reg++` each iteration

### Parameters Dispatch Summary

The SFPU comparison functions (`ltz_tile`, `gtz_tile`) are dispatched through `_llk_math_eltwise_unary_sfpu_params_`:

- **Vector mode**: `VectorMode::RC` — all 4 faces of the tile are processed (full 32×32 tile coverage)
- **Operation invocation**: The params dispatch function calls the SFPU compute functor once per face (4 times total for RC mode). Each invocation processes 8 iterations (ITERATIONS=8), covering one full 16×16 face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration within the SFPU function, `SETRWC` with `CR_D, 8` twice between faces in the params dispatch). On both Wormhole and Blackhole, the default address mode for comparison operations is `ADDR_MOD_7` with `dest.incr = 0` (DEST auto-increment is managed by explicit `dst_reg++` in SFPI code, not by the hardware address mode).

### Annotated Compute Kernel Source — Variant 1 (non-FLOAT32)

The following is the dedicated compute kernel for hardshrink when input dtype is NOT FLOAT32 (e.g., BFLOAT16). It uses FPU binary operations via `binary_dest_reuse_tiles` with DEST reuse patterns.

```cpp
// File: ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/hardshrink_kernel.cpp

void kernel_main() {
    const uint32_t packed_scalar = get_arg_val<uint32_t>(0); // runtime arg: lambda packed as float bits
    const auto lambd = reinterpret_cast<const float*>(&packed_scalar);
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;
    constexpr auto cb_tmp0 = tt::CBIndex::c_1; // intermediate buffer for first term
    init_sfpu(cb_input, cb_output);

    // hardshrink(a, λ) = a⋅1(a+λ<0) + a⋅1(a−λ>0)
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(cb_output, per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            cb_wait_front(cb_input, 1);
            cb_reserve_back(cb_tmp0, 1);
            tile_regs_acquire();

            // === Phase 1: Compute a⋅1(a+λ<0) ===
            fill_tile(0, *lambd);                       // DEST[0] = λ (fill tile 0 with lambda)
            // FPU add: DEST[0] = DEST[0] + cb_input[0] = λ + a = (a+λ)
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_input);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_input, 0, 0);
            ltz_tile(0);                                // SFPU: DEST[0] = 1.0 if (a+λ)<0, else 0.0
            // FPU mul: DEST[0] = DEST[0] * cb_input[0] = 1(a+λ<0) * a
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_input);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_input, 0, 0);

            tile_regs_commit();
            tile_regs_wait();

            pack_tile(0, cb_tmp0);                      // pack first term to tmp CB
            tile_regs_release();

            cb_push_back(cb_tmp0, 1);
            cb_wait_front(cb_tmp0, 1);
            tile_regs_acquire();

            // === Phase 2: Compute a⋅1(a−λ>0) and add first term ===
            fill_tile(0, *lambd);                       // DEST[0] = λ
            // FPU sub: DEST[0] = cb_input[0] - DEST[0] = a - λ = (a−λ)
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWSUB, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_input);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWSUB, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                cb_input, 0, 0);
            gtz_tile(0);                                // SFPU: DEST[0] = 1.0 if (a−λ)>0, else 0.0
            // FPU mul: DEST[0] = DEST[0] * cb_input[0] = 1(a−λ>0) * a
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_input);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_input, 0, 0);

            // FPU add: DEST[0] = DEST[0] + cb_tmp0[0] = a⋅1(a−λ>0) + a⋅1(a+λ<0)
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_tmp0);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_tmp0, 0, 0);

            tile_regs_commit();
            tile_regs_wait();

            pack_tile(0, cb_output);
            tile_regs_release();

            cb_pop_front(cb_input, 1);
            cb_pop_front(cb_tmp0, 1);
        }
        cb_push_back(cb_output, per_core_block_dim);
    }
}
```

### Annotated Compute Kernel Source — Variant 2 (FLOAT32)

The FLOAT32 variant uses SFPU-based binary operations instead of FPU binary ops, and loads `cb_input` tiles explicitly into DEST register slots using `copy_tile`.

```cpp
// File: ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/hardshrink_kernel_sfpu.cpp

void kernel_main() {
    const uint32_t packed_scalar = get_arg_val<uint32_t>(0);
    const auto lambd = reinterpret_cast<const float*>(&packed_scalar);
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;
    constexpr auto cb_tmp0 = tt::CBIndex::c_1;
    init_sfpu(cb_input, cb_output);

    // hardshrink(a, λ) = a⋅1(a+λ<0) + a⋅1(a−λ>0)
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(cb_output, per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            cb_wait_front(cb_input, 1);
            cb_reserve_back(cb_tmp0, 1);
            tile_regs_acquire();

            // === Phase 1: Compute a⋅1(a+λ<0) ===
            fill_tile(0, *lambd);                       // DEST[0] = λ
            copy_tile_to_dst_init_short(cb_input);
            copy_tile(cb_input, 0, 1);                  // DEST[1] = a (input tile into dst slot 1)
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);                   // SFPU add: DEST[0] = DEST[0] + DEST[1] = λ + a
            ltz_tile(0);                                // SFPU: DEST[0] = 1.0 if (a+λ)<0, else 0.0
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);                   // SFPU mul: DEST[0] = DEST[0] * DEST[1] = 1(a+λ<0) * a

            tile_regs_commit();
            tile_regs_wait();

            pack_tile(0, cb_tmp0);
            tile_regs_release();

            cb_push_back(cb_tmp0, 1);
            cb_wait_front(cb_tmp0, 1);
            tile_regs_acquire();

            // === Phase 2: Compute a⋅1(a−λ>0) and add first term ===
            fill_tile(1, *lambd);                       // DEST[1] = λ
            copy_tile_to_dst_init_short(cb_input);
            copy_tile(cb_input, 0, 0);                  // DEST[0] = a
            sub_binary_tile_init();
            sub_binary_tile(0, 1, 0);                   // SFPU sub: DEST[0] = DEST[0] - DEST[1] = a - λ
            gtz_tile(0);                                // SFPU: DEST[0] = 1.0 if (a−λ)>0, else 0.0
            copy_tile_to_dst_init_short(cb_input);
            copy_tile(cb_input, 0, 1);                  // DEST[1] = a (reload input)
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);                   // SFPU mul: DEST[0] = DEST[0] * DEST[1] = 1(a−λ>0) * a
            copy_tile_to_dst_init_short(cb_tmp0);
            copy_tile(cb_tmp0, 0, 1);                   // DEST[1] = first term from tmp
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);                   // SFPU add: DEST[0] = a⋅1(a−λ>0) + a⋅1(a+λ<0)

            tile_regs_commit();
            tile_regs_wait();

            pack_tile(0, cb_output);
            tile_regs_release();

            cb_pop_front(cb_input, 1);
            cb_pop_front(cb_tmp0, 1);
        }
        cb_push_back(cb_output, per_core_block_dim);
    }
}
```

### Annotated SFPU Kernel Source — Comparison Functions

The core SFPU functions used by `ltz_tile` and `gtz_tile`. These use SFPI abstractions (Style A).

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_comp.h
// (Blackhole version at tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_comp.h is identical for these functions)

template <>
inline void apply_zero_comp<SfpuType::less_than_zero>(sfpi::vFloat& v, std::uint32_t /*unused*/)
{
    v_if (v >= ZERO)   // CC set: enabled lanes where v >= 0
    {
        v = ZERO;      // CC-guarded: set to 0.0 where v >= 0 (not less than zero)
    }
    v_else             // CC inverted: enabled lanes where v < 0
    {
        v = ONE;       // CC-guarded: set to 1.0 where v < 0
    }
    v_endif;           // CC restored to all-enabled
}

template <>
inline void apply_zero_comp<SfpuType::greater_than_zero>(sfpi::vFloat& v, std::uint32_t /*unused*/)
{
    v_if (v > ZERO)    // CC set: enabled lanes where v > 0
    {
        v = ONE;       // CC-guarded: set to 1.0 where v > 0
    }
    v_else             // CC inverted: enabled lanes where v <= 0
    {
        v = ZERO;      // CC-guarded: set to 0.0 where v <= 0
    }
    v_endif;           // CC restored to all-enabled
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void _calculate_zero_comp_(std::uint32_t exponent_size_8) // APPROXIMATION_MODE=false, ITERATIONS=8
{
    for (int d = ZERO; d < ITERATIONS; d++) // 8 iterations per face
    {
        sfpi::vFloat v = sfpi::dst_reg[0];       // load 32 elements from DEST
        apply_zero_comp<COMP_MODE>(v, exponent_size_8); // apply comparison (less_than_zero or greater_than_zero)
        sfpi::dst_reg[0] = v;                     // store result back to DEST
        sfpi::dst_reg++;                          // advance to next sfpi row (2 physical DEST rows)
    }
}
```

### Annotated SFPU Kernel Source — Fill Function

The core SFPU function used by `fill_tile`:

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_fill.h

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_fill_(const float value) // APPROXIMATION_MODE=false, ITERATIONS=8
{
    sfpi::vFloat fill_val = value; // broadcast scalar to all 32 SIMD lanes

    for (int d = 0; d < ITERATIONS; d++) // 8 iterations per face
    {
        sfpi::dst_reg[0] = fill_val; // write fill value to 32 elements in DEST
        sfpi::dst_reg++;             // advance to next sfpi row
    }
}
```

### SFPU Instructions Used

The SFPU operations in hardshrink use SFPI abstractions, so the actual hardware instructions are generated by the SFPI compiler. The following are the logical operations and their expected hardware instruction mappings:

| SFPI Operation | Expected Hardware Instruction(s) | Description |
|---|---|---|
| `sfpi::vFloat v = sfpi::dst_reg[0]` | `SFPLOAD` | Load 32 elements from current DEST row pair into LREG |
| `sfpi::dst_reg[0] = v` | `SFPSTORE` | Store 32 elements from LREG back to DEST row pair |
| `sfpi::dst_reg++` | Address counter increment | Advance SFPU address by 1 sfpi row (2 physical DEST rows, 32 elements) |
| `v_if (v >= 0)` | `SFPSETCC` (with `CC_GTE0` or sign-bit test) | Set condition code based on sign of v; enables lanes where v >= 0 |
| `v_if (v > 0)` | `SFPSETCC` + `SFPENCC` / compound CC ops | Set condition code for strict greater-than-zero; requires checking both sign and zero |
| `v_else` | `SFPCOMPC` | Complement (invert) the current condition code mask |
| `v_endif` | `SFPENCC` | Re-enable all lanes (restore CC to all-enabled) |
| `v = ZERO` / `v = ONE` | `SFPLOADI` or `SFPMOV` | Load immediate constant (0.0 or 1.0) into LREG, CC-guarded |
| `sfpi::vFloat fill_val = value` | `SFPLOADI` | Load scalar constant (λ) into LREG, broadcast to all lanes |
| `add_binary_tile(0, 1, 0)` | `SFPMAD` chain | SFPU element-wise add: `a * 1.0 + b` via multiply-accumulate |
| `sub_binary_tile(0, 1, 0)` | `SFPMAD` chain | SFPU element-wise sub: `a * 1.0 + (-b)` via multiply-accumulate with negation |
| `mul_binary_tile(0, 1, 0)` | `SFPMUL` / `SFPMAD` | SFPU element-wise multiply |

### SFPU Register Usage

| Register | Usage |
|---|---|
| **DEST[0]** (tile slot 0) | Primary working register — holds intermediate and final results throughout both phases |
| **DEST[1]** (tile slot 1) | Used in SFPU variant only — holds `a` (input) or `λ` depending on phase, used as second operand for SFPU binary ops |
| **LREGs (L0-L3)** | Implicit in SFPI abstractions — `vFloat` variables map to LREGs during SFPU execution. Used for: `v` (comparison input), `fill_val` (lambda constant), comparison result (0.0 or 1.0) |
| **CB c_0 (input)** | Input tensor tiles — read but never popped until both phases complete for a tile |
| **CB c_1 (tmp0)** | Temporary storage for Phase 1 result (`a⋅1(a+λ<0)`). Required because DEST is overwritten in Phase 2. This is why HARDSHRINK requires `needs_tmp0_cb() == true` |
| **CB c_2 (output)** | Final result tiles |

### Address Mode Configuration

The SFPU comparison operations (`ltz_tile`, `gtz_tile`) and fill operation are initialized with `ADDR_MOD_7`:

**Wormhole B0** (`llk_math_eltwise_unary_sfpu.h`):
```
ADDR_MOD_7: { srca.incr = 0, srcb.incr = 0, dest.incr = 0 }
```

**Blackhole** (`llk_math_eltwise_unary_sfpu.h`):
```
ADDR_MOD_7: { srca.incr = 0, srcb.incr = 0, dest.incr = 0 }
```

Both architectures use identical `ADDR_MOD_7` configuration with zero auto-increment for all registers. The DEST address progression is managed entirely by explicit `dst_reg++` in the SFPI kernel code (each `dst_reg++` advances by `SFP_DESTREG_STRIDE=2` physical DEST rows = 32 elements). Between faces, the params dispatch function uses `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` twice to advance 16 physical rows (one face height).

There is no special `ADDR_MOD_6` configuration for comparison operations — the generic `SfpuType::less_than_zero` / `SfpuType::greater_than_zero` types do not match any of the special `if constexpr` branches in `eltwise_unary_sfpu_configure_addrmod`.

### Circular Buffer and Two-Phase Execution Pattern

HARDSHRINK is notable among unary operations for requiring **CB c_1 as a temporary buffer** and a **two-phase execution pattern** per tile:

1. **Phase 1** (compute first term → pack to c_1):
   - Fill DEST with λ, add input `a`, apply `ltz_tile`, multiply by `a`
   - Result: `a⋅1(a+λ<0)` in DEST → pack to CB c_1
   - Requires: `tile_regs_commit → tile_regs_wait → pack_tile → tile_regs_release`

2. **Phase 2** (compute second term + combine → pack to c_2):
   - Fill DEST with λ, subtract from input `a`, apply `gtz_tile`, multiply by `a`
   - Add Phase 1 result from CB c_1
   - Result: `a⋅1(a+λ<0) + a⋅1(a−λ>0)` in DEST → pack to CB c_2

This two-phase pattern is necessary because:
- Both terms need the original input `a`, but DEST is destructively modified by the comparison operations
- The FPU variant (non-FLOAT32) uses `binary_dest_reuse_tiles` which sources one operand from DEST and one from a CB, so it can reuse `cb_input` for `a` without extra copies
- The SFPU variant (FLOAT32) must explicitly `copy_tile` from CBs into DEST slots since SFPU binary ops operate entirely within DEST

### Key Differences Between Variants

| Aspect | hardshrink_kernel.cpp (non-FP32) | hardshrink_kernel_sfpu.cpp (FP32) |
|---|---|---|
| Binary ops | FPU via `binary_dest_reuse_tiles` | SFPU via `add/sub/mul_binary_tile` |
| Input access | Implicit via DEST_TO_SRCA/SRCB reuse | Explicit `copy_tile(cb_input, 0, slot)` |
| DEST slots used | Only slot 0 | Slots 0 and 1 |
| Input reloads | None (FPU reads from CB directly) | Multiple `copy_tile` calls needed |
| Why FP32 differs | FPU binary ops may not preserve FP32 precision in DEST | SFPU binary ops operate in full FP32 when `fp32_dest_acc_en` is set |

## Local Knowledge Sources

### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/hardshrink_kernel.cpp`
   **Reason**: Primary compute kernel for hardshrink (non-FLOAT32 path)
   **Key Findings**: Two-phase execution with FPU binary ops + SFPU comparison ops, uses CB c_1 as temporary

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/hardshrink_kernel_sfpu.cpp`
   **Reason**: FLOAT32 compute kernel variant
   **Key Findings**: Uses SFPU binary ops instead of FPU, requires explicit copy_tile for input access, uses two DEST slots

3. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_comp.h`
   **Reason**: Core SFPU comparison implementation — contains `apply_zero_comp` specializations and `_calculate_zero_comp_`
   **Key Findings**: `less_than_zero` uses `v_if (v >= 0) { v = 0 } v_else { v = 1 }`, `greater_than_zero` uses `v_if (v > 0) { v = 1 } v_else { v = 0 }`, both pure SFPI

4. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_comp.h`
   **Reason**: Blackhole variant of comparison SFPU implementation
   **Key Findings**: Functionally identical to Wormhole for the `apply_zero_comp` specializations used by hardshrink

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_fill.h`
   **Reason**: Core SFPU fill implementation used by `fill_tile`
   **Key Findings**: Simple loop broadcasting scalar to all DEST rows via `dst_reg[0] = fill_val; dst_reg++`

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: LLK dispatch layer — routes SFPU function calls through VectorMode face loop
   **Key Findings**: RC mode processes 4 faces, SETRWC with CR_D 8 twice between faces, stalls SFPU before and waits after

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: LLK init and address mode configuration
   **Key Findings**: Comparison ops use ADDR_MOD_7 with dest.incr=0; no special ADDR_MOD for ltz/gtz SfpuTypes

8. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h`
   **Reason**: Macro definitions for SFPU dispatch — `SFPU_ZERO_KERNEL` macro
   **Key Findings**: `SFPU_ZERO_KERNEL(OP, MODE, APPROX, DST_IDX)` routes to `ckernel::sfpu::calculate_comp<APPROX, SfpuType::OP>` with 8 iterations

9. **File**: `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h`
   **Reason**: API definitions for SFPU binary tile operations (add/sub/mul_binary_tile)
   **Key Findings**: These are SFPU-based binary ops operating entirely within DEST, dispatching to `llk_math_eltwise_binary_sfpu_binop`

10. **File**: `tt_metal/hw/inc/api/compute/eltwise_binary.h`
    **Reason**: API definitions for FPU binary tile operations (binary_dest_reuse_tiles)
    **Key Findings**: FPU binary ops with DEST reuse patterns — one operand from DEST, one unpacked from CB

11. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
    **Reason**: Dispatch configuration for HARDSHRINK
    **Key Findings**: `get_op_approx_mode` returns false (default), `get_compute_kernel_path` returns "eltwise_sfpu.cpp" (default, but overridden by `unary_ng` path), HARDSHRINK not in `get_op_init_and_func` (uses dedicated kernel)

12. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp`
    **Reason**: Next-gen unary dispatch — determines which compute kernel is used
    **Key Findings**: HARDSHRINK routes to `hardshrink_kernel_sfpu.cpp` for FLOAT32, `hardshrink_kernel.cpp` otherwise

13. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`
    **Reason**: Program factory — sets up CBs and runtime args
    **Key Findings**: HARDSHRINK gets CB c_1 allocated as temporary, lambda packed as `packed_scalar1` runtime arg
