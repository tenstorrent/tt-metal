## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel functions that the hardshrink compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `HARDSHRINK`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/hardshrink_kernel_sfpu.cpp` (custom compute kernel; does NOT use `eltwise_sfpu.cpp` with `SFPU_OP_CHAIN_0`)
- **SFPU_OP_CHAIN_0 expansion**: Not applicable. HARDSHRINK uses a dedicated compute kernel that manually orchestrates multiple SFPU and SFPU-binary tile operations: `fill_tile()`, `ltz_tile()`, `gtz_tile()`, `add_binary_tile()`, `sub_binary_tile()`, `mul_binary_tile()`, and `copy_tile()`.

**Note on compute kernel variants**: Two custom compute kernels exist for HARDSHRINK:
1. `hardshrink_kernel_sfpu.cpp` -- uses `copy_tile()` + standalone `add_binary_tile()` / `sub_binary_tile()` / `mul_binary_tile()` SFPU binary ops.
2. `hardshrink_kernel.cpp` -- uses `binary_dest_reuse_tiles<>()` FPU binary ops with DEST reuse optimization.

Both variants implement the same mathematical algorithm. The `_sfpu` variant is documented below as the primary reference.

#### Mathematical Algorithm
The hardshrink function is defined as:
```
hardshrink(x, lambda) = x   if |x| > lambda
                         0   otherwise
```

This is decomposed algebraically as: `a * 1(a + lambda < 0) + a * 1(a - lambda > 0)`, where `1(...)` is the indicator function. The compute kernel implements this as a two-pass algorithm:

- **Pass 1**: Compute `a * 1(a + lambda < 0)` -- this captures the negative inputs beyond `-lambda`. Store result to `cb_tmp0`.
- **Pass 2**: Compute `a * 1(a - lambda > 0)` -- this captures the positive inputs beyond `+lambda`. Add the Pass 1 result to get the final output.

#### Approximation Mode Resolution
There are TWO independent controls for approximation. You MUST report both.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(HARDSHRINK)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none | Not applicable -- HARDSHRINK uses a custom compute kernel, not `SFPU_OP_CHAIN_0`. The `APPROX` macro is implicitly available via `common.h` and passed through to underlying SFPU calls. |
| Effective SFPU path | Comparison operations use `v_if (v < 0.0F)` / `v_if (v > ZERO)` SFPI abstractions which do not depend on approximation mode. Fill uses direct `dst_reg[0] = fill_val` assignment. Binary ops (add/sub/mul) use direct SFPI arithmetic. | The `APPROXIMATION_MODE` template parameter is accepted but unused by `_calculate_fill_`, `_calculate_comp_`, and `_calculate_sfpu_binary_` for the operations used here. |

### SFPU Abstraction Layers

HARDSHRINK does not follow the standard single-SFPU-function dispatch pattern. Instead, it uses multiple SFPU operations composed in the custom compute kernel. The abstraction layers for each SFPU operation are:

**`ltz_tile(idst)` -- Less Than Zero comparison:**

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/comp.h` [NUKED -- file removed from this repo] |
| **LLK Dispatch** | [NUKED -- `llk_math_eltwise_unary_sfpu_comp.h` removed] |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_comp.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

**`gtz_tile(idst)` -- Greater Than Zero comparison:**

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/comp.h` [NUKED -- file removed from this repo] |
| **LLK Dispatch** | [NUKED -- `llk_math_eltwise_unary_sfpu_comp.h` removed] |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_comp.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

**`fill_tile(idst, value)` -- Fill tile with scalar constant:**

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/fill.h` [NUKED -- file removed from this repo] |
| **LLK Dispatch** | [NUKED -- `llk_math_eltwise_unary_sfpu_fill.h` removed] |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_fill.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

**`add_binary_tile(idst0, idst1, odst)` / `sub_binary_tile(...)` / `mul_binary_tile(...)` -- SFPU binary ops:**

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

### Call Chain

The hardshrink compute kernel (`hardshrink_kernel_sfpu.cpp`) directly calls the following tile-level APIs in sequence. There is no single SFPU dispatch chain; instead, the kernel orchestrates a multi-step algorithm:

**Pass 1 (compute `a * 1(a + lambda < 0)`):**
1. `fill_tile(0, *lambd)` -- Fills DST tile slot 0 with the lambda scalar value. API call goes through `llk_math_eltwise_unary_sfpu_fill<APPROX>(0, *lambd)` -> `_llk_math_eltwise_unary_sfpu_params_()` -> `_calculate_fill_<APPROX, 8>(value)` for each of the 4 faces.
2. `copy_tile_to_dst_init_short(cb_input)` + `copy_tile(cb_input, 0, 1)` -- Unpacks the input tile from CB to DST slot 1 (FPU datacopy, not SFPU).
3. `add_binary_tile_init()` + `add_binary_tile(0, 1, 0)` -- Computes DST[0] = DST[0] + DST[1] = lambda + input = `a + lambda`. Calls `llk_math_eltwise_binary_sfpu_binop<APPROX, BinaryOp::ADD>()` -> `_llk_math_eltwise_binary_sfpu_params_()` -> `_calculate_sfpu_binary_<APPROX, BinaryOp::ADD, 8>()`.
4. `ltz_tile(0)` -- Computes DST[0] = 1.0 if (a + lambda) < 0, else 0.0. Calls through the LLK to `_calculate_comp_<APPROX, false, false, false, false, 8>()` which uses `v_if (v < 0.0F)` to produce a 0/1 mask. Template parameters: `invert_output=false` means output 1.0 when condition is true.
5. `mul_binary_tile_init()` + `mul_binary_tile(0, 1, 0)` -- Computes DST[0] = DST[0] * DST[1] = mask * input = `a * 1(a + lambda < 0)`.
6. Result packed to `cb_tmp0`.

**Pass 2 (compute `a * 1(a - lambda > 0)` and add Pass 1):**
7. `fill_tile(1, *lambd)` -- Fills DST tile slot 1 with lambda.
8. `copy_tile(cb_input, 0, 0)` -- Copies input to DST slot 0.
9. `sub_binary_tile_init()` + `sub_binary_tile(0, 1, 0)` -- Computes DST[0] = DST[0] - DST[1] = input - lambda = `a - lambda`.
10. `gtz_tile(0)` -- Computes DST[0] = 1.0 if (a - lambda) > 0, else 0.0. Calls through LLK to `_calculate_comp_<APPROX, true, false, true, false, 8>()` which decomposes the "greater than zero" test as: flag1 = `1.0 if v >= 0 else 0.0`, flag2 = `1.0 if v != 0 else 0.0`, result = flag1 AND flag2 (bitwise).
11. `copy_tile(cb_input, 0, 1)` + `mul_binary_tile(0, 1, 0)` -- Computes `mask * input`.
12. `copy_tile(cb_tmp0, 0, 1)` + `add_binary_tile(0, 1, 0)` -- Adds Pass 1 result to get final output.

### Parameters Dispatch Summary

Since HARDSHRINK uses multiple SFPU operations, each has its own parameters dispatch:

- **Vector mode**: All SFPU operations (`fill_tile`, `ltz_tile`, `gtz_tile`) use `VectorMode::RC` (all 4 faces). The SFPU binary operations (`add_binary_tile`, `sub_binary_tile`, `mul_binary_tile`) also process all 4 faces via `VectorMode::RC`.
- **Operation invocation**: Each SFPU tile-level call goes through `_llk_math_eltwise_unary_sfpu_params_()` which loops over 4 faces, calling the core SFPU function once per face with `ITERATIONS=8`. Between faces, `TTI_SETRWC` advances the DEST address by 2 face strides (2 x `TTI_SETRWC(..., 8, ...)` = 16 physical rows = 1 face). For binary SFPU ops, `_llk_math_eltwise_binary_sfpu_params_()` similarly processes 4 faces, managing addresses for the 2 input tiles and 1 output tile.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). Address mode `ADDR_MOD_7` is configured with all increments = 0 (the SFPU iteration loop uses `dst_reg++` to advance within each face).

### Annotated SFPU Kernel Source

The hardshrink operation uses three core SFPU functions. Their annotated source is below.

#### 1. `_calculate_fill_` -- Fill tile with scalar value

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_fill.h

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_fill_(const float value) // APPROXIMATION_MODE unused, ITERATIONS=8
{
    // SFPU microcode
    sfpi::vFloat fill_val = value; // SFPLOADI: broadcast scalar float to all 32 SFPU lanes

    for (int d = 0; d < ITERATIONS; d++) // 8 iterations per face
    {
        sfpi::dst_reg[0] = fill_val; // SFPSTORE: write fill_val to current DEST row pair (32 elements)
        sfpi::dst_reg++;             // advance DEST pointer by 1 sfpi row (= 2 physical DEST rows)
    }
}
```

#### 2. `_calculate_comp_` -- Comparison against zero (used by `ltz_tile` and `gtz_tile`)

The `_calculate_comp_` function is a heavily templated comparison kernel. For hardshrink, two instantiations are used:

- **`ltz_tile(0)`**: `_calculate_comp_<APPROX, invert_output=false, check_zero=false, second_check=false, is_less_than_equal_zero=false, ITERATIONS=8>`
  - This enters the `else` branch (not `check_zero`), tests `v < 0.0F`, and since `invert_output=false`: output_0=1.0 (true), output_1=0.0 (false). Since `second_check=false`, result = flag1 directly.

- **`gtz_tile(0)`**: `_calculate_comp_<APPROX, invert_output=true, check_zero=true, second_check=true, is_less_than_equal_zero=false, ITERATIONS=8>`
  - First check: `check_zero=true` tests for fp16 zero. Second check: flag1 from sign test, flag2 from zero test. Since `is_less_than_equal_zero=false`, result = flag1 AND flag2 (bitwise), producing 1.0 when value is strictly greater than zero.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_comp.h

sfpi_inline void _calculate_comp_init_flag_(bool check, sfpi::vFloat& flag1, sfpi::vFloat& flag2, float init)
{
    flag1 = init;
    if (check)
    {
        flag2 = init;
    }
}

template <bool APPROXIMATION_MODE, bool invert_output, bool check_zero, bool second_check, bool is_less_than_equal_zero, int ITERATIONS>
inline void _calculate_comp_(const int iterations, std::uint32_t exponent_size_8)
{
    // Implementation notes, see the original file for more details
    constexpr float output_0 = invert_output ? 0.0f : 1.0f; // value when condition is TRUE
    constexpr float output_1 = invert_output ? 1.0f : 0.0f; // value when condition is FALSE

    for (int d = ZERO; d < iterations; d++) // ITERATIONS=8 per face
    {
        sfpi::vFloat v = sfpi::dst_reg[0]; // SFPLOAD: read current DEST row pair (32 elements)
        sfpi::vFloat flag1, flag2;
        if constexpr (check_zero) // gtz_tile path
        {
            v_if (_sfpu_is_fp16_zero_(v, exponent_size_8)) // test if value is zero
            {
                _calculate_comp_init_flag_(second_check, flag1, flag2, output_0);
            }
            v_else
            {
                _calculate_comp_init_flag_(second_check, flag1, flag2, output_1);
            }
            v_endif;
        }
        else // ltz_tile path
        {
            v_if (v < 0.0F) // SFPSETCC: test sign bit, sets per-lane condition code
            {
                _calculate_comp_init_flag_(second_check, flag1, flag2, output_0); // flag1 = 1.0 if v < 0
            }
            v_else
            {
                _calculate_comp_init_flag_(second_check, flag1, flag2, output_1); // flag1 = 0.0 if v >= 0
            }
            v_endif;
        }

        sfpi::vFloat result;
        if constexpr (second_check) // gtz_tile path: combine two flags
        {
            if constexpr (is_less_than_equal_zero) // NOT taken for gtz
            {
                result = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vUInt>(flag1) | sfpi::reinterpret<sfpi::vUInt>(flag2));
            }
            else // gtz_tile: flag1 (>= 0) AND flag2 (!= 0) => strictly > 0
            {
                result = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vUInt>(flag1) & sfpi::reinterpret<sfpi::vUInt>(flag2));
            }
        }
        else // ltz_tile path: single flag is the result
        {
            result = flag1;
        }

        sfpi::dst_reg[0] = result; // SFPSTORE: write 0.0 or 1.0 mask back to DEST
        sfpi::dst_reg++;           // advance to next sfpi row
    }
}
```

#### 3. `_sfpu_is_fp16_zero_` -- Helper for zero detection (used by `gtz_tile`)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_is_fp16_zero.h

sfpi_inline sfpi::vInt _sfpu_is_fp16_zero_(const sfpi::vFloat& v, std::uint32_t exponent_size_8)
{
    if (exponent_size_8) // fp16b: 8-bit exponent, direct comparison works
    {
        return v == 0.0F;
    }
    else // fp16a: 5-bit exponent converted to 8-bit by SFPU with bias addition
    {
        sfpi::vInt tmp = 0x3800; // {0, 8'd112, 10'b0} -- bias correction
        tmp += sfpi::reinterpret<sfpi::vInt>(v);
        return tmp == 0;
    }
}
```

#### 4. `_calculate_sfpu_binary_` -- Binary SFPU operations (add, sub, mul between DST tiles)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8>
inline void _calculate_sfpu_binary_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{   // APPROXIMATION_MODE unused, ITERATIONS=8
    static constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    for (int d = 0; d < ITERATIONS; d++)
    {
        constexpr std::uint32_t dst_tile_size_sfpi = 32; // 32 sfpi rows per tile
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi]; // SFPLOAD from tile 0
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi]; // SFPLOAD from tile 1
        sfpi::vFloat result = 0.0f;

        if constexpr (BINOP == BinaryOp::ADD) // add_binary_tile path
        {
            result = in0 + in1; // SFPMAD: in0 * 1.0 + in1
        }
        else if constexpr (BINOP == BinaryOp::SUB) // sub_binary_tile path
        {
            result = in0 - in1; // SFPMAD: in0 * 1.0 + (-in1)
        }
        else if constexpr (BINOP == BinaryOp::MUL) // mul_binary_tile path
        {
            result = in0 * in1; // SFPMAD: in0 * in1 + 0.0
        }
        else if constexpr (BINOP == BinaryOp::RSUB)
        {
            result = in1 - in0;
        }
        // DIV, POW, XLOGY branches removed in this codebase

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result; // SFPSTORE to output tile
        sfpi::dst_reg++; // advance all three tile pointers by 1 sfpi row
    }
}
```

### SFPU Instructions Used

| Instruction | Description | Used By |
|-------------|-------------|---------|
| **SFPLOAD** (`dst_reg[N]` read) | Load 32 elements (2 physical DEST rows) from a DEST tile slot into SFPU lane registers (LREGs) | `_calculate_comp_`, `_calculate_sfpu_binary_` |
| **SFPSTORE** (`dst_reg[N] =` write) | Store 32 elements from SFPU lane registers back to a DEST tile slot | `_calculate_comp_`, `_calculate_fill_`, `_calculate_sfpu_binary_` |
| **SFPLOADI** (`vFloat v = constant`) | Load an immediate scalar value into all 32 SFPU lanes (broadcast) | `_calculate_fill_` (fill_val), `_calculate_comp_` (output_0/output_1 constants) |
| **SFPMAD** (`vFloat +`, `-`, `*`) | Fused multiply-add: `a * b + c`. Used for all vFloat arithmetic (add emits `a*1.0+b`, sub emits `a*1.0+(-b)`, mul emits `a*b+0.0`) | `_calculate_sfpu_binary_` (ADD, SUB, MUL), `_sfpu_is_fp16_zero_` (bias add) |
| **SFPSETCC** / **v_if** | Set per-lane condition codes based on a comparison (e.g., `v < 0.0F`, `v == 0.0F`). Subsequent instructions are predicated on these CC flags. | `_calculate_comp_` (sign/zero tests), `_sfpu_is_fp16_zero_` |
| **SFPENCC** / **v_else** | Complement the current per-lane condition codes (flip true/false lanes) | `_calculate_comp_` |
| **SFPCOMPC** / **v_endif** | Restore condition codes to "all lanes active" state after conditional block | `_calculate_comp_` |
| **SFPIADD** | Integer addition. Used in `_sfpu_is_fp16_zero_` for fp16a bias correction (`tmp += reinterpret<vInt>(v)`) | `_sfpu_is_fp16_zero_` |
| **SFPAND** / **SFPOR** (via `reinterpret<vUInt>` bitwise ops) | Bitwise AND/OR on integer-reinterpreted float values. Used to combine comparison flags in `gtz_tile` (flag1 AND flag2 = strict greater-than). | `_calculate_comp_` (second_check path) |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST tile slot 0** | Multi-purpose: filled with lambda (Pass 1 & 2), holds intermediate computation results (a+lambda, a-lambda), holds comparison mask (0/1), holds final result after mul and add |
| **DEST tile slot 1** | Holds input tile data (copied from CB via `copy_tile`). In Pass 2, also used for lambda fill and for loading Pass 1 intermediate from `cb_tmp0` |
| **LREG[0..3]** (SFPU local regs) | Implicitly used by SFPI abstractions. `vFloat` variables (`v`, `flag1`, `flag2`, `result`, `fill_val`, `in0`, `in1`) are mapped to LREGs by the SFPI compiler. Typically LREG0 is used for `dst_reg[0]` loads. |
| **Condition Codes (CC)** | Per-lane 1-bit flags set by `v_if` comparisons. Used to predicate which lanes receive 0.0 vs 1.0 in the comparison operations. |

### Address Mode Configuration

**ADDR_MOD_7** is configured for all SFPU operations in the hardshrink kernel:

```cpp
// From llk_math_eltwise_unary_sfpu.h: eltwise_unary_sfpu_configure_addrmod<sfpu_op>()
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}
    .set(ADDR_MOD_7);
```

All increments are zero because SFPU operations advance DEST addressing via `dst_reg++` in the software loop rather than via hardware auto-increment. Between faces, `TTI_SETRWC` instructions advance the DEST pointer by one face (16 physical rows = 8 sfpi rows with stride-2).

For the binary SFPU operations, `ADDR_MOD_7` is also used with zero increments -- the `_calculate_sfpu_binary_` function indexes into multiple tiles using `dst_index * dst_tile_size_sfpi` offsets, with `dst_reg++` advancing all accesses in lockstep.

This configuration is identical across Wormhole and Blackhole architectures. The `ADDR_MOD_7` choice specifically avoids conflicts with `ADDR_MOD_0` and `ADDR_MOD_2` which are used by the A2D (unpack-to-DEST) datacopy operations that precede the SFPU phase.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Consulted to find `get_compute_kernel_path()`, `get_op_approx_mode()`, and `get_op_init_and_func()` for HARDSHRINK
   **Key Findings**: HARDSHRINK case has been nuked from all switch statements. `get_compute_kernel_path()` returns `"eltwise_sfpu.cpp"` by default, but the original would return the custom kernel path. `get_op_approx_mode()` returns `false` for all ops.

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`
   **Reason**: Consulted to understand how HARDSHRINK is dispatched, including CB allocation and scalar parameter passing
   **Key Findings**: HARDSHRINK allocates an extra `cb_tmp0` (c_1) circular buffer and packs the lambda parameter as `packed_scalar1` via `pack_scalar_runtime_arg()`.

3. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/hardshrink_kernel_sfpu.cpp`
   **Reason**: Primary compute kernel for HARDSHRINK. Contains the full algorithmic flow.
   **Key Findings**: Two-pass algorithm using fill_tile, copy_tile, add/sub/mul_binary_tile, ltz_tile, gtz_tile. Reads lambda parameter via `get_arg_val<uint32_t>(0)` + `reinterpret_cast<const float*>`.

4. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/hardshrink_kernel.cpp`
   **Reason**: Alternative compute kernel using FPU `binary_dest_reuse_tiles` instead of SFPU binary ops.
   **Key Findings**: Same algorithm but uses `binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCA>` and `binary_dest_reuse_tiles<ELWSUB, DEST_TO_SRCB>` for potentially better FPU utilization.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_comp.h`
   **Reason**: Core SFPU implementation for comparison operations (ltz_tile, gtz_tile)
   **Key Findings**: `_calculate_comp_` is a 6-template-parameter function handling all comparison variants. Uses `v_if`/`v_else`/`v_endif` for per-lane conditional assignment of 0.0 or 1.0. `gtz` requires a two-flag approach (sign check AND zero check) with bitwise AND to produce strict greater-than.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_fill.h`
   **Reason**: Core SFPU implementation for fill_tile
   **Key Findings**: Simple broadcast-and-store loop: load scalar to vFloat, iterate 8 times writing to `dst_reg[0]` and incrementing.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h`
   **Reason**: Core SFPU implementation for add/sub/mul_binary_tile
   **Key Findings**: Reads from two tile slots using `dst_index * dst_tile_size_sfpi` offset, performs SFPI arithmetic (+ / - / *), writes result to output tile slot. `dst_reg++` advances all three tile accesses in lockstep.

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_is_fp16_zero.h`
   **Reason**: Helper function used by `gtz_tile` for zero detection
   **Key Findings**: Handles fp16b (direct comparison) and fp16a (bias-corrected comparison) zero detection.

9. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: LLK dispatch layer for unary SFPU operations; configures address modes and provides `_llk_math_eltwise_unary_sfpu_start_`/`_done_` functions.
   **Key Findings**: `ADDR_MOD_7` configured with all increments = 0. `_llk_math_eltwise_unary_sfpu_start_` sets DEST write address and stalls until SFPU is ready.

10. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
    **Reason**: Parameters dispatch for unary SFPU operations; handles face iteration and SETRWC between faces.
    **Key Findings**: `VectorMode::RC` processes all 4 faces with 2x `SETRWC(CR_D, 8)` between faces.

11. **File**: `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h`
    **Reason**: API header for SFPU binary tile operations.
    **Key Findings**: `add_binary_tile()` -> `llk_math_eltwise_binary_sfpu_binop<APPROX, BinaryOp::ADD>()`, similarly for sub and mul.

12. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h`
    **Reason**: LLK dispatch for SFPU binary operations.
    **Key Findings**: Calls `_llk_math_eltwise_binary_sfpu_params_()` with `calculate_sfpu_binary<APPROX, BINOP, 8>` for add/sub, and `calculate_sfpu_binary_mul<APPROX, BinaryOp::MUL, 8>` for mul.

13. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Authoritative reference for SFPU architecture, tile geometry, and addressing model.
    **Key Findings**: Confirmed SFPU stride-2 addressing, 32 elements per sfpi row, 8 iterations per face, ADDR_MOD_7 configuration.

14. **File**: `docs/sfpu_operations/key_notes/hardshrink_key_notes.md`
    **Reason**: Operation definition and parameter documentation.
    **Key Findings**: Formula `x if |x| > lambda, 0 otherwise`. Default lambda=0.5, deterministic operation.
