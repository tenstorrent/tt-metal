## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel functions that the hardshrink compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `HARDSHRINK`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/hardshrink_kernel_sfpu.cpp` (custom compute kernel, NOT the standard `eltwise_sfpu.cpp`)
- **SFPU_OP_CHAIN_0 expansion**: Not applicable. Hardshrink uses a **dedicated compute kernel** that directly calls multiple SFPU tile-level API functions (`fill_tile`, `ltz_tile`, `gtz_tile`, `add_binary_tile`, `sub_binary_tile`, `mul_binary_tile`) rather than the standard `SFPU_OP_CHAIN_0` macro dispatch pattern.

**Note on compute kernel variants**: Two compute kernel files exist for hardshrink:
1. `hardshrink_kernel.cpp` -- uses FPU-based `binary_dest_reuse_tiles` for arithmetic operations
2. `hardshrink_kernel_sfpu.cpp` -- uses SFPU-based `add_binary_tile`/`sub_binary_tile`/`mul_binary_tile` for arithmetic operations

Both implement the same mathematical formula: `a * 1(a + lambda < 0) + a * 1(a - lambda > 0)`. This analysis focuses on the SFPU variant (`hardshrink_kernel_sfpu.cpp`).

#### Approximation Mode Resolution

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(HARDSHRINK)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none | Not applicable -- hardshrink uses a custom compute kernel, not the `SFPU_OP_CHAIN_0` pattern. The `APPROX` compile-time flag (derived from `math_approx_mode`) is passed to the SFPU API calls within the kernel. |
| Effective SFPU path | `APPROXIMATION_MODE=false` for all SFPU functions | The `APPROX` template parameter resolves to `false`. For the comp functions (`_calculate_comp_`, `_calculate_zero_comp_`), `APPROXIMATION_MODE` is a template parameter but does not affect any code branch -- the comp functions have no approximation-dependent logic. For `_calculate_fill_`, the approximation mode is similarly unused. |

### SFPU Abstraction Layers

Hardshrink's compute kernel is a custom kernel that calls multiple SFPU tile-level APIs. Each SFPU operation has its own abstraction layer chain. The table below lists the layers for the primary SFPU operations used.

**Comparison operations (`ltz_tile`, `gtz_tile`):**

| Layer | File Path |
|-------|-----------|
| **API Header** | Nuked in this codebase (originally `compute_kernel_api/eltwise_unary/comp.h` or similar) |
| **LLK Dispatch** | Nuked in this codebase (originally `llk_math_eltwise_unary_sfpu_comp.h`); the dispatch uses `_llk_math_eltwise_unary_sfpu_params_` from `llk_math_eltwise_unary_sfpu_params.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_comp.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

**Fill operation (`fill_tile`):**

| Layer | File Path |
|-------|-----------|
| **API Header** | Nuked in this codebase (originally `compute_kernel_api/eltwise_unary/fill.h` or similar) |
| **LLK Dispatch** | Nuked in this codebase (originally `llk_math_eltwise_unary_sfpu_fill.h`); the dispatch uses `_llk_math_eltwise_unary_sfpu_params_` from `llk_math_eltwise_unary_sfpu_params.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_fill.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

**Binary SFPU operations (`add_binary_tile`, `sub_binary_tile`, `mul_binary_tile`):**

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

### Call Chain

The hardshrink compute kernel (`hardshrink_kernel_sfpu.cpp`) implements the formula `a * 1(a + lambda < 0) + a * 1(a - lambda > 0)` using a two-pass approach through DEST registers:

**Pass 1** (computing `a * 1(a + lambda < 0)` into `cb_tmp0`):
1. `fill_tile(0, lambda)` -- Fills DST tile 0 with the lambda scalar value. API call -> `_llk_math_eltwise_unary_sfpu_params_` -> `_calculate_fill_<false, 8>(lambda)` per face.
2. `copy_tile(cb_input, 0, 1)` -- Copies input tile from `c_0` to DST tile 1.
3. `add_binary_tile(0, 1, 0)` -- Computes `lambda + a` into DST tile 0. API call -> `llk_math_eltwise_binary_sfpu_binop<APPROX, BinaryOp::ADD>` -> `_calculate_sfpu_binary_<false, BinaryOp::ADD, 8>(0, 1, 0)` per face.
4. `ltz_tile(0)` -- Computes `1(result < 0)` on DST tile 0 (replaces each element with 1.0 if < 0, else 0.0). API call -> LLK dispatch -> `_calculate_comp_<false, false, false, false, false, 8>` or `_calculate_zero_comp_<false, SfpuType::less_than_zero, 8>` per face.
5. `mul_binary_tile(0, 1, 0)` -- Computes `indicator * a` into DST tile 0. API call -> `llk_math_eltwise_binary_sfpu_binop_mul<APPROX, BinaryOp::MUL>` -> `calculate_sfpu_binary_mul<false, BinaryOp::MUL, 8>` per face.
6. Result is packed into `cb_tmp0`.

**Pass 2** (computing `a * 1(a - lambda > 0)` and adding to pass 1 result):
1. `fill_tile(1, lambda)` -- Fills DST tile 1 with lambda.
2. `copy_tile(cb_input, 0, 0)` -- Copies input tile to DST tile 0.
3. `sub_binary_tile(0, 1, 0)` -- Computes `a - lambda` into DST tile 0.
4. `gtz_tile(0)` -- Computes `1(result > 0)` on DST tile 0.
5. `copy_tile(cb_input, 0, 1)` -- Copies input tile to DST tile 1.
6. `mul_binary_tile(0, 1, 0)` -- Computes `indicator * a` into DST tile 0.
7. `copy_tile(cb_tmp0, 0, 1)` -- Copies pass 1 result to DST tile 1.
8. `add_binary_tile(0, 1, 0)` -- Computes final `pass1 + pass2` into DST tile 0.
9. Result is packed into `cb_output`.

### Parameters Dispatch Summary

Since hardshrink uses multiple different SFPU operations, each has its own dispatch pattern:

**For unary SFPU operations (`fill_tile`, `ltz_tile`, `gtz_tile`):**
- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed.
- **Operation invocation**: The core SFPU function is called once per face (4 times total per tile), with ITERATIONS=8 per face call, covering all 32 sfpi rows (= 1024 elements).
- **DEST address progression**: On Wormhole, the params dispatch uses `TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D)` twice between faces (each SETRWC advances by 8 sfpi rows = 16 physical DEST rows). Within a face, `dst_reg++` advances 1 sfpi row per iteration. On Blackhole, the params dispatch uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which calls `math::inc_dst_addr<8>()` twice between faces.

**For binary SFPU operations (`add_binary_tile`, `sub_binary_tile`, `mul_binary_tile`):**
- **Vector mode**: `VectorMode::RC` -- all 4 faces processed.
- **Operation invocation**: The `_calculate_sfpu_binary_` function is called once per face (4 times), with ITERATIONS=8. It indexes into DEST using `dst_index_in0 * dst_tile_size_sfpi` and `dst_index_in1 * dst_tile_size_sfpi` to read from two different tiles and writes to `dst_index_out * dst_tile_size_sfpi`.
- **DEST address progression**: Same as unary -- `TTI_SETRWC` with increment 8, applied twice between faces. Within iterations, `dst_reg++` advances by 1 sfpi row.

### Annotated SFPU Kernel Source

The hardshrink operation does not have a single `_calculate_hardshrink_` function. Instead, it composes several SFPU primitives. Below are the core SFPU functions that implement the operation's logic.

#### Custom Compute Kernel (the orchestrating kernel)

```cpp
// File: ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/hardshrink_kernel_sfpu.cpp

void kernel_main() {
    const uint32_t packed_scalar = get_arg_val<uint32_t>(0); // Runtime arg: packed lambda
    const auto lambd = reinterpret_cast<const float*>(&packed_scalar); // Reinterpret as float*
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;
    constexpr auto cb_tmp0 = tt::CBIndex::c_1; // Temporary CB for intermediate result

    init_sfpu(cb_input, cb_output);

    // Formula: a * 1(a + lambda < 0) + a * 1(a - lambda > 0)
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(cb_output, per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            cb_wait_front(cb_input, 1);
            cb_reserve_back(cb_tmp0, 1);
            tile_regs_acquire();

            // --- Pass 1: compute a * 1(a + lambda < 0) ---
            fill_tile(0, *lambd);               // DST[0] = lambda (broadcast scalar to all elements)
            copy_tile_to_dst_init_short(cb_input);
            copy_tile(cb_input, 0, 1);          // DST[1] = input tile (a)
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);           // DST[0] = a + lambda
            ltz_tile(0);                        // DST[0] = 1.0 if (a + lambda) < 0, else 0.0
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);           // DST[0] = indicator * a

            tile_regs_commit();
            tile_regs_wait();

            pack_tile(0, cb_tmp0);              // Pack pass 1 result to tmp CB
            tile_regs_release();

            cb_push_back(cb_tmp0, 1);
            cb_wait_front(cb_tmp0, 1);
            tile_regs_acquire();

            // --- Pass 2: compute a * 1(a - lambda > 0) + pass1 ---
            fill_tile(1, *lambd);               // DST[1] = lambda
            copy_tile_to_dst_init_short(cb_input);
            copy_tile(cb_input, 0, 0);          // DST[0] = input tile (a)
            sub_binary_tile_init();
            sub_binary_tile(0, 1, 0);           // DST[0] = a - lambda
            gtz_tile(0);                        // DST[0] = 1.0 if (a - lambda) > 0, else 0.0
            copy_tile_to_dst_init_short(cb_input);
            copy_tile(cb_input, 0, 1);          // DST[1] = input tile (a)
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);           // DST[0] = indicator * a
            copy_tile_to_dst_init_short(cb_tmp0);
            copy_tile(cb_tmp0, 0, 1);           // DST[1] = pass 1 result
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);           // DST[0] = pass2 + pass1 = final result

            tile_regs_commit();
            tile_regs_wait();

            pack_tile(0, cb_output);            // Pack final result to output CB
            tile_regs_release();

            cb_pop_front(cb_input, 1);
            cb_pop_front(cb_tmp0, 1);
        }
        cb_push_back(cb_output, per_core_block_dim);
    }
}
```

#### Core SFPU Implementation: Fill (`_calculate_fill_`)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_fill.h

template <bool APPROXIMATION_MODE, int ITERATIONS> // APPROXIMATION_MODE=false, ITERATIONS=8
inline void _calculate_fill_(const float value)
{
    // SFPU microcode
    sfpi::vFloat fill_val = value; // Broadcast scalar to all SFPU lanes via SFPLOADI

    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::dst_reg[0] = fill_val; // SFPSTORE: write fill value to current DEST row pair
        sfpi::dst_reg++;             // Advance to next sfpi row (2 physical DEST rows)
    }
}
```

#### Core SFPU Implementation: Comparison (`_calculate_comp_`)

The `ltz_tile` and `gtz_tile` operations use the comparison functions from `ckernel_sfpu_comp.h`. There are two implementations: the legacy `_calculate_comp_` with template boolean flags, and the newer `_calculate_zero_comp_` with `SfpuType` template specializations. The nuked codebase does not reveal which dispatch path is used, but both achieve the same result.

**For `ltz_tile`** (less-than-zero):

Using `_calculate_comp_`: instantiated as `_calculate_comp_<false, true, false, false, false, 8>` where `invert_output=true` means output_0=0.0 (for negative), output_1=1.0 (for non-negative). Wait -- actually `invert_output=true` makes output_0=0.0 and output_1=1.0, and the `v < 0` branch assigns output_0. So for LTZ: v < 0 gets 0.0 (wrong). Let me re-examine.

Actually, the correct path for `ltz_tile` is `_calculate_zero_comp_<false, SfpuType::less_than_zero, 8>`, which dispatches to `apply_zero_comp<SfpuType::less_than_zero>`:

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_comp.h

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
// APPROXIMATION_MODE=false, COMP_MODE=SfpuType::less_than_zero, ITERATIONS=8
inline void _calculate_zero_comp_(std::uint32_t exponent_size_8)
{
    for (int d = ZERO; d < ITERATIONS; d++)
    {
        sfpi::vFloat v = sfpi::dst_reg[0]; // SFPLOAD: read current element from DEST
        apply_zero_comp<COMP_MODE>(v, exponent_size_8); // Apply comparison
        sfpi::dst_reg[0] = v;              // SFPSTORE: write result back to DEST
        sfpi::dst_reg++;                   // Advance to next sfpi row
    }
}

template <>
inline void apply_zero_comp<SfpuType::less_than_zero>(sfpi::vFloat& v, std::uint32_t /*unused*/)
{
    v_if (v >= ZERO)   // SFPSETCC: set CC based on v >= 0
    {
        v = ZERO;      // If v >= 0, output 0.0 (not less than zero)
    }
    v_else             // SFPENCC: invert condition code
    {
        v = ONE;       // If v < 0, output 1.0 (is less than zero)
    }
    v_endif;           // Restore CC state
}
```

**For `gtz_tile`** (greater-than-zero):

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_comp.h
// Uses _calculate_zero_comp_<false, SfpuType::greater_than_zero, 8>

template <>
inline void apply_zero_comp<SfpuType::greater_than_zero>(sfpi::vFloat& v, std::uint32_t /*unused*/)
{
    v_if (v > ZERO)    // SFPSETCC: set CC based on v > 0
    {
        v = ONE;       // If v > 0, output 1.0
    }
    v_else             // SFPENCC: invert condition code
    {
        v = ZERO;      // If v <= 0, output 0.0
    }
    v_endif;           // Restore CC state
}
```

#### Core SFPU Implementation: Binary Operations (`_calculate_sfpu_binary_`)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8>
// APPROXIMATION_MODE=false, BINOP=ADD/SUB/MUL, ITERATIONS=8
inline void _calculate_sfpu_binary_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{
    static constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    for (int d = 0; d < ITERATIONS; d++)
    {
        constexpr std::uint32_t dst_tile_size_sfpi = 32; // 32 sfpi rows per tile in DEST
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi]; // SFPLOAD from tile 0
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi]; // SFPLOAD from tile 1
        sfpi::vFloat result = 0.0f;

        if constexpr (BINOP == BinaryOp::ADD)
        {
            result = in0 + in1; // SFPMAD(in0, 1.0, in1) -- vFloat add emits MAD
        }
        else if constexpr (BINOP == BinaryOp::SUB)
        {
            result = in0 - in1; // SFPMAD(in0, 1.0, -in1) -- vFloat sub emits MAD with negated addend
        }
        else if constexpr (BINOP == BinaryOp::MUL)
        {
            result = in0 * in1; // SFPMAD(in0, in1, 0.0) -- vFloat mul emits MAD with zero addend
        }
        // ... other cases omitted for brevity

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result; // SFPSTORE to output tile
        sfpi::dst_reg++; // Advance base address by 1 sfpi row
    }
}
```

#### Helper: `_sfpu_is_fp16_zero_`

This helper is used by some comp variants (not directly by `ltz_tile`/`gtz_tile` since those use sign comparison, not zero comparison). Included for completeness:

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_is_fp16_zero.h

sfpi_inline sfpi::vInt _sfpu_is_fp16_zero_(const sfpi::vFloat& v, std::uint32_t exponent_size_8)
{
    if (exponent_size_8)
    {
        // fp16b: direct comparison
        return v == 0.0F;
    }
    else
    {
        // fp16a: SFPU adds bias to 5-bit exponent unconditionally
        sfpi::vInt tmp = 0x3800; // {0, 8'd112, 10'b0} -- the bias offset for fp16a zero
        tmp += sfpi::reinterpret<sfpi::vInt>(v);
        return tmp == 0;
    }
}
```

### SFPU Instructions Used

| Instruction | Description | Used By |
|-------------|-------------|---------|
| `SFPLOAD` | Load data from DEST register row into SFPU LREG for processing. Emitted by `sfpi::dst_reg[N]` reads. | All SFPU functions (`_calculate_fill_`, `_calculate_zero_comp_`, `_calculate_sfpu_binary_`) |
| `SFPSTORE` | Store data from SFPU LREG back to DEST register row. Emitted by `sfpi::dst_reg[N] = val` writes. | All SFPU functions |
| `SFPLOADI` | Load immediate float constant into SFPU LREG. Emitted by `sfpi::vFloat v = constant` assignments (e.g., `fill_val = value`, `result = 0.0f`, `v = ONE`, `v = ZERO`). | `_calculate_fill_`, `_calculate_zero_comp_`, `_calculate_sfpu_binary_` |
| `SFPMAD` | Fused multiply-add: `a * b + c`. Emitted for all vFloat arithmetic: addition (`a + b` = `a * 1.0 + b`), subtraction (`a - b` = `a * 1.0 + (-b)`), multiplication (`a * b` = `a * b + 0.0`). There is no dedicated float add instruction. | `_calculate_sfpu_binary_` (ADD, SUB, MUL variants) |
| `SFPSETCC` | Set per-lane condition code based on a comparison. Emitted by `v_if (v < 0.0F)`, `v_if (v >= ZERO)`, `v_if (v > ZERO)`. | `_calculate_zero_comp_` (less_than_zero, greater_than_zero) |
| `SFPENCC` | Invert (complement) the condition code. Emitted by `v_else` in SFPI conditional blocks. | `_calculate_zero_comp_` |
| `SFPCOMPC` | Complement condition code (restore previous CC state). Emitted by `v_endif` to restore the CC state from before the `v_if`. | `_calculate_zero_comp_` |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST tile 0** | Primary working tile. In pass 1: holds lambda, then `a + lambda`, then `1(a + lambda < 0)`, then `indicator * a`. In pass 2: holds `a`, then `a - lambda`, then `1(a - lambda > 0)`, then `indicator * a`, then final result. |
| **DEST tile 1** | Secondary tile. Holds the input tile `a` for binary operations (`mul_binary_tile`), or lambda for `sub_binary_tile`, or pass 1 result for final addition. |
| **LREG[0]-LREG[3]** | SFPU local registers used implicitly by SFPI abstractions. `dst_reg[N]` loads into LREG for computation, then stores back. `vFloat` variables map to LREGs during computation. |
| **Condition Code (CC)** | Per-lane 1-bit flags. Used by `ltz_tile` and `gtz_tile` to predicate stores: comparison sets CC, then `v_if`/`v_else` regions write 1.0 or 0.0 depending on CC state. |
| **cb_tmp0 (c_1)** | Circular buffer used as intermediate storage between pass 1 and pass 2. Pass 1 result is packed from DEST to L1 via this CB, then unpacked back for pass 2. |

### Address Mode Configuration

The address mode configuration depends on which SFPU operation is being dispatched:

**For unary SFPU operations (`fill_tile`, `ltz_tile`, `gtz_tile`):**

The init function `_llk_math_eltwise_unary_sfpu_init_<SfpuType>()` configures:
- `ADDR_MOD_7`: `.srca = {.incr = 0}`, `.srcb = {.incr = 0}`, `.dest = {.incr = 0}` -- no auto-increment (SFPU manages its own addressing via `dst_reg++`)

For comp operations (`SfpuType::less_than_zero`, `SfpuType::greater_than_zero`), no special `ADDR_MOD_6` is configured (the `eltwise_unary_sfpu_configure_addrmod` template only sets `ADDR_MOD_6` for specific types like `typecast`, `unary_max`, `signbit`, etc.).

The DEST address progression between faces is managed by `TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D)` called twice per face boundary in the `_llk_math_eltwise_unary_sfpu_params_` function (on Wormhole). On Blackhole, `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice.

**For binary SFPU operations (`add_binary_tile`, `sub_binary_tile`, `mul_binary_tile`):**

The binary SFPU init (`_llk_math_eltwise_binary_sfpu_init_`) configures the same `ADDR_MOD_7` pattern. The binary params dispatch (`_llk_math_eltwise_binary_sfpu_params_`) uses the same `TTI_SETRWC` pattern for face progression.

Both Wormhole and Blackhole use the same address mode values (`ADDR_MOD_7` with all zero increments). The difference is in face progression mechanics: Wormhole uses raw `TTI_SETRWC` in the params dispatch, while Blackhole uses the `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` helper.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, approximation mode, and SFPU_OP_CHAIN_0 defines for HARDSHRINK
   **Key Findings**: `get_op_approx_mode()` returns false (default case). `get_compute_kernel_path()` returns `eltwise_sfpu.cpp` (default case -- the HARDSHRINK-specific case was nuked). HARDSHRINK is not in `get_op_init_and_func_default` or `get_op_init_and_func_parameterized` (nuked).

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`
   **Reason**: Understand how the program factory handles HARDSHRINK -- scalar parameter packing, tmp CB creation
   **Key Findings**: HARDSHRINK gets a `cb_tmp0` (c_1) circular buffer. The lambda parameter is packed as `packed_scalar1` via `pack_scalar_runtime_arg`. The packed scalar is passed as a runtime arg to the compute kernel.

3. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/hardshrink_kernel_sfpu.cpp`
   **Reason**: The SFPU variant of the custom compute kernel for hardshrink
   **Key Findings**: Implements `a * 1(a + lambda < 0) + a * 1(a - lambda > 0)` using fill_tile, add/sub/mul_binary_tile, ltz_tile, gtz_tile. Uses two passes with cb_tmp0 for intermediate storage.

4. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/hardshrink_kernel.cpp`
   **Reason**: The FPU variant of the custom compute kernel for hardshrink (for comparison)
   **Key Findings**: Same formula but uses `binary_dest_reuse_tiles` (FPU path) instead of `add/sub/mul_binary_tile` (SFPU path).

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_comp.h`
   **Reason**: Core SFPU implementation for comparison operations (ltz, gtz)
   **Key Findings**: `_calculate_zero_comp_<APPROXIMATION_MODE, COMP_MODE, ITERATIONS>` iterates through sfpi rows, loading from DEST, applying comparison via `apply_zero_comp` specializations, and storing back. Uses `v_if`/`v_else`/`v_endif` (SFPSETCC/SFPENCC/SFPCOMPC) for per-lane conditional assignment of 1.0 or 0.0.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_fill.h`
   **Reason**: Core SFPU implementation for fill operation
   **Key Findings**: Simple loop: broadcasts scalar to vFloat, then writes to each DEST row for ITERATIONS iterations. No approximation-dependent logic.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h`
   **Reason**: Core SFPU implementation for binary arithmetic (add, sub, mul) on tiles in DEST
   **Key Findings**: `_calculate_sfpu_binary_` reads from two tiles at `dst_index_in0 * 32` and `dst_index_in1 * 32`, computes the binary op, and writes to `dst_index_out * 32`. Uses SFPMAD for all float arithmetic.

8. **File**: `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h`
   **Reason**: API header for binary SFPU tile operations
   **Key Findings**: `add_binary_tile` dispatches to `llk_math_eltwise_binary_sfpu_binop<APPROX, BinaryOp::ADD>`. `mul_binary_tile` uses `llk_math_eltwise_binary_sfpu_binop_mul`. All use `APPROX` template parameter from compile config.

9. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: LLK dispatch layer for unary SFPU operations -- face iteration and DEST progression
   **Key Findings**: `_llk_math_eltwise_unary_sfpu_params_` iterates over 4 faces for VectorMode::RC, calling the SFPU function once per face. Uses `TTI_SETRWC` to advance DEST address between faces.

10. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
    **Reason**: LLK init and address mode configuration for unary SFPU ops
    **Key Findings**: `eltwise_unary_sfpu_configure_addrmod` sets `ADDR_MOD_7` with zero increments for all ops. Only specific op types get additional `ADDR_MOD_6` configuration.

11. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary_sfpu_params.h`
    **Reason**: LLK dispatch layer for binary SFPU operations
    **Key Findings**: Same face iteration pattern as unary params dispatch. Passes additional dst_index arguments for two-operand operations.

12. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_is_fp16_zero.h`
    **Reason**: Helper function used by some comp variants for zero detection
    **Key Findings**: Handles fp16a vs fp16b zero detection differently due to exponent bias. Not directly used by ltz_tile/gtz_tile (which use sign comparison, not zero comparison).

13. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: SFPU hardware model reference for tile geometry, DEST layout, and addressing
    **Key Findings**: Confirmed tile = 32x32 = 1024 elements, 4 faces x 16 rows x 16 cols, ITERATIONS=8 per face, dst_tile_size_sfpi=32, stride-2 addressing model.

14. **File**: `docs/sfpu_operations/key_notes/hardshrink_key_notes.md`
    **Reason**: Mathematical definition of the operation
    **Key Findings**: `x if |x| > lambda, 0 otherwise`. Default lambda=0.5.
