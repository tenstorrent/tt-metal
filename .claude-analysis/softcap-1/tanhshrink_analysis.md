## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `TANHSHRINK`
- **Compute kernel**: Dedicated kernel -- NOT dispatched via `eltwise_sfpu.cpp` / `SFPU_OP_CHAIN_0`
  - Variant 1: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_kernel.cpp`
  - Variant 2: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_sfpu_kernel.cpp`
- **SFPU_OP_CHAIN_0 expansion**: N/A -- tanhshrink does not use the standard SFPU chain dispatch. Instead, it directly calls `tanh_tile(0)` followed by an explicit subtraction step.

**Critical finding**: TANHSHRINK is **not wired** through the standard `unary_op_utils.cpp` dispatch. Specifically:
1. `get_op_init_and_func_default()` has no case for `UnaryOpType::TANHSHRINK` (would throw at `default` case)
2. `get_compute_kernel_path()` has no case for TANHSHRINK (falls to `default: return "eltwise_sfpu.cpp"`)
3. `get_op_approx_mode()` has no case for TANHSHRINK (falls to `default: return false`)
4. Both dedicated kernel files call `tanh_tile()` which invokes `llk_math_eltwise_unary_sfpu_tanh()` -- a function that has **no implementation** anywhere in this codebase (no `llk_math_eltwise_unary_sfpu_tanh.h`, no `ckernel_sfpu_tanh.h`, tanh is not in the Metal `SfpuType` enum)

As a result, neither kernel variant would compile in the current codebase state. The analysis below documents the **intended design** based on the existing source code, focusing on the SFPU subtraction path (`ckernel_sfpu_binary.h`) which IS fully defined.

#### Approximation Mode Resolution

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode()` in `unary_op_utils.cpp` -- TANHSHRINK falls through to `default: return false` |
| Template parameter (tanh_tile) | `false` (default) | `tanh_tile_init()` and `tanh_tile(0)` called without template argument in both kernel files; template default is `fast_and_approx = false` |
| Template parameter (sub_binary_tile) | `APPROX` (compile-time global) | `sub_binary_tile_init()` passes `APPROX` to binop_init; `sub_binary_tile()` passes `APPROX` to binop |
| Effective SFPU path | Tanh: would use the non-approximate path if `llk_math_eltwise_unary_sfpu_tanh` existed. Binary SUB: uses `BinaryOp::SUB` branch in `_calculate_sfpu_binary_`, which is a simple `in0 - in1` with no approximation-dependent branching | tanh_tile: `fast_and_approx=false`; binary sub: `if constexpr (BINOP == BinaryOp::SUB)` branch at line 41 of `ckernel_sfpu_binary.h` |

### SFPU Abstraction Layers

The tanhshrink operation uses TWO distinct computation paths chained together in a single kernel. The tanh path is **broken** (undefined LLK function), so we document only the subtraction path fully.

**Tanh component** (broken -- `llk_math_eltwise_unary_sfpu_tanh` is undefined):

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/compute_kernel_api.h` (lines 154-180: `tanh_tile_init()`, `tanh_tile()`) |
| **LLK Dispatch** | UNDEFINED -- `llk_math_eltwise_unary_sfpu_tanh` is called but has no implementation |
| **Core SFPU Implementation** | UNDEFINED -- no `ckernel_sfpu_tanh.h` exists |
| **Parameters Dispatch** | UNDEFINED |

**SFPU Binary Subtraction component** (variant 2 only -- `tanhshrink_sfpu_kernel.cpp`):

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` (lines 39-40, 68: `sub_binary_tile()`, `sub_binary_tile_init()`) |
| **LLK Dispatch** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

**FPU Binary Subtraction component** (variant 1 only -- `tanhshrink_kernel.cpp`):

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary.h` (lines 208-257: `binary_dest_reuse_tiles_init()`, `binary_dest_reuse_tiles()`) |
| **LLK Dispatch** | This level of abstraction doesn't exist -- dispatches directly to `llk_math_eltwise_binary` (FPU, not SFPU) |
| **Core SFPU Implementation** | This level of abstraction doesn't exist -- FPU operation, not SFPU |
| **Parameters Dispatch** | This level of abstraction doesn't exist |

### Call Chain

**Variant 1 (`tanhshrink_kernel.cpp`) -- Hybrid FPU+SFPU approach**:
1. `copy_tile(cb_input, 0, 0)` -- copies input tile from CB c_0 to DEST tile slot 0
2. `tanh_tile_init()` -> `llk_math_eltwise_unary_sfpu_tanh_init<false, DST_ACCUM_MODE>()` [UNDEFINED]
3. `tanh_tile(0)` -> `llk_math_eltwise_unary_sfpu_tanh<false, DST_ACCUM_MODE>(0)` [UNDEFINED] -- would compute tanh(x) in-place in DEST[0]
4. `binary_dest_reuse_tiles_init<ELWSUB, DEST_TO_SRCB>(cb_input)` -- reconfigures unpack for DEST-to-SRCB reuse mode
5. `binary_dest_reuse_tiles<ELWSUB, DEST_TO_SRCB>(cb_input, 0, 0)` -- moves DEST[0] (= tanh(x)) to SRCB, unpacks input from cb_input to SRCA, then FPU computes SRCA - SRCB = x - tanh(x), result goes back to DEST[0]

**Variant 2 (`tanhshrink_sfpu_kernel.cpp`) -- Pure SFPU approach**:
1. `copy_tile(cb_input, 0, 1)` -- copies input tile from CB c_0 to DEST tile slot 1
2. `tanh_tile_init()` -> `llk_math_eltwise_unary_sfpu_tanh_init<false, DST_ACCUM_MODE>()` [UNDEFINED]
3. `tanh_tile(1)` -> `llk_math_eltwise_unary_sfpu_tanh<false, DST_ACCUM_MODE>(1)` [UNDEFINED] -- would compute tanh(x) in-place in DEST[1]
4. `copy_tile(cb_input, 0, 0)` -- copies fresh input x into DEST tile slot 0
5. `sub_binary_tile_init()` -> `llk_math_eltwise_binary_sfpu_binop_init<APPROX, BinaryOp::SUB>()` -> calls `_llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>()` + `_sfpu_binary_init_<APPROX, BinaryOp::SUB>()`
6. `sub_binary_tile(0, 1, 0)` -> `llk_math_eltwise_binary_sfpu_binop<APPROX, BinaryOp::SUB>(0, 1, 0)` -> `_llk_math_eltwise_binary_sfpu_params_<APPROX>(_calculate_sfpu_binary_<APPROX, BinaryOp::SUB, 8>, 0, 1, 0, VectorMode::RC)` -- SFPU computes DEST[0] - DEST[1] = x - tanh(x), stores result in DEST[0]

### Parameters Dispatch Summary

**For the SFPU binary subtraction** (variant 2, `sub_binary_tile`):

- **Vector mode**: `VectorMode::RC` (default) -- processes all 4 faces of the tile
- **Operation invocation**: `_llk_math_eltwise_binary_sfpu_params_` calls the SFPU function `_calculate_sfpu_binary_<APPROX, BinaryOp::SUB, 8>` once per face (4 times total for RC mode). Each call iterates 8 times (`ITERATIONS=8`), processing 8 sfpi rows per face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` advances by `inc_dst_addr<8>` twice between faces = 16 physical DEST rows). The binary SFPU uses tile-index-based addressing: each iteration reads from `dst_reg[dst_index_in0 * 32]` and `dst_reg[dst_index_in1 * 32]`, writes to `dst_reg[dst_index_out * 32]`, then advances via `dst_reg++`.

**For the FPU binary subtraction** (variant 1, `binary_dest_reuse_tiles`):
- This is a **FPU** operation (not SFPU). It uses `llk_math_eltwise_binary<ELWSUB>` which operates through the matrix unit, not the SFPU. The DEST-to-SRCB reuse mode moves the current DEST tile to SRCB, then performs SRCA - SRCB elementwise via the FPU.

### Annotated SFPU Kernel Source

The core SFPU operation for tanhshrink is the binary subtraction (`BinaryOp::SUB`). The tanh component has no SFPU implementation in this codebase.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8>
inline void _calculate_sfpu_binary_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{   // APPROXIMATION_MODE=APPROX (compile-time global), BINOP=BinaryOp::SUB, ITERATIONS=8
    static constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    for (int d = 0; d < ITERATIONS; d++)
    {
        constexpr std::uint32_t dst_tile_size_sfpi = 32; // 32 sfpi rows per tile (64 physical / stride 2)
        sfpi::vFloat in0                           = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi]; // SFPLOAD: read 32 elements from DEST tile 0 (x)
        sfpi::vFloat in1                           = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi]; // SFPLOAD: read 32 elements from DEST tile 1 (tanh(x))
        sfpi::vFloat result                        = 0.0f;

        if constexpr (BINOP == BinaryOp::ADD)
        {
            result = in0 + in1;
        }
        else if constexpr (BINOP == BinaryOp::SUB) // This branch is taken for tanhshrink
        {
            result = in0 - in1; // SFPMAD: computes x - tanh(x) = tanhshrink(x)
        }
        else if constexpr (BINOP == BinaryOp::MUL)
        {
            result = in0 * in1;
        }
        else if constexpr (BINOP == BinaryOp::DIV)
        {
            // Reciprocal removed -- Family 3 primitive
        }
        else if constexpr (BINOP == BinaryOp::RSUB)
        {
            result = in1 - in0;
        }
        else if constexpr (BINOP == BinaryOp::POW)
        {
            // Power removed -- depends on exp/log/recip primitives
        }
        else if constexpr (BINOP == BinaryOp::XLOGY)
        {
            // XLOGY removed -- depends on log primitive
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result; // SFPSTORE: write result to DEST tile 0
        sfpi::dst_reg++; // Advance by 1 sfpi row = 2 physical DEST rows = 32 elements
    }
}

template <bool APPROXIMATION_MODE /*unused*/, BinaryOp BINOP>
inline void _sfpu_binary_init_() {} // No-op init for binary SFPU ops
```

**Compute kernel source** (variant 2 -- the pure-SFPU approach):

```cpp
// File: ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_sfpu_kernel.cpp

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;
    init_sfpu(cb_input, cb_output);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(cb_output, per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            cb_wait_front(cb_input, 1);
            tile_regs_acquire();

            // Step 1: Copy input x to DEST tile slot 1
            copy_tile_init(cb_input);
            copy_tile(cb_input, 0, 1);

            // Step 2: Compute tanh(x) in-place in DEST[1] [BROKEN: undefined LLK function]
            tanh_tile_init();
            tanh_tile(1);

            // Step 3: Copy fresh input x to DEST tile slot 0
            copy_tile_init(cb_input);
            copy_tile(cb_input, 0, 0);

            // Step 4: SFPU subtract: DEST[0] - DEST[1] = x - tanh(x) -> DEST[0]
            sub_binary_tile_init();
            sub_binary_tile(0, 1, 0);

            tile_regs_commit();
            tile_regs_wait();

            pack_tile(0, cb_output);

            tile_regs_release();

            cb_pop_front(cb_input, 1);
        }
        cb_push_back(cb_output, per_core_block_dim);
    }
}
```

**Compute kernel source** (variant 1 -- the hybrid FPU+SFPU approach):

```cpp
// File: ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_kernel.cpp

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;
    init_sfpu(cb_input, cb_output);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(cb_output, per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            cb_wait_front(cb_input, 1);
            tile_regs_acquire();

            // Step 1: Copy input x to DEST tile slot 0
            copy_tile_init(cb_input);
            copy_tile(cb_input, 0, 0);

            // Step 2: Compute tanh(x) in-place in DEST[0] [BROKEN: undefined LLK function]
            tanh_tile_init();
            tanh_tile(0);

            // Step 3: FPU subtract using DEST reuse -- moves DEST[0]=tanh(x) to SRCB,
            // unpacks input x from cb_input to SRCA, computes SRCA - SRCB = x - tanh(x)
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWSUB, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_input);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWSUB, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                cb_input, 0, 0);

            tile_regs_commit();
            tile_regs_wait();

            pack_tile(0, cb_output);

            tile_regs_release();

            cb_pop_front(cb_input, 1);
        }
        cb_push_back(cb_output, per_core_block_dim);
    }
}
```

### SFPU Instructions Used

The following analysis covers only the SFPU binary subtraction path (`_calculate_sfpu_binary_<APPROX, BinaryOp::SUB, 8>`). The tanh SFPU path is undefined in this codebase.

| Instruction | Emitted By | Description |
|---|---|---|
| **SFPLOAD** | `sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi]` read | Loads 32 elements from DEST at tile 0's current sfpi row offset into LREG (input x) |
| **SFPLOAD** | `sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi]` read | Loads 32 elements from DEST at tile 1's current sfpi row offset into LREG (tanh(x)) |
| **SFPMAD** | `in0 - in1` (vFloat subtraction) | Computes `x - tanh(x)`. There is no dedicated float subtract instruction; subtraction is implemented as `a * 1.0 + (-b)` via SFPMAD |
| **SFPSTORE** | `sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result` write | Stores the 32-element result back to DEST at tile 0's current sfpi row offset |

Note: Each iteration processes 32 elements (2 physical DEST rows x 16 elements/row due to stride-2 addressing). With ITERATIONS=8 per face and 4 faces, the full tile of 1024 elements is processed.

### SFPU Register Usage

| Register | Usage |
|---|---|
| **DEST tile 0** (sfpi offset 0-31) | Input x (copied from CB), then output x - tanh(x) after subtraction. In variant 1, also temporarily holds tanh(x) before FPU subtract. |
| **DEST tile 1** (sfpi offset 32-63) | Only used in variant 2: holds tanh(x) after tanh_tile(1). Read-only during the subtraction. |
| **LREG (implicit)** | Temporary storage during SFPLOAD/SFPMAD/SFPSTORE. `in0` and `in1` are loaded into local registers, the SFPMAD result is in an LREG, then stored back to DEST. |
| **ADDR_MOD_7** | Set during binary SFPU init with all-zero increments (.srca=0, .srcb=0, .dest=0). This is a non-incrementing address mode since the SFPU kernel manages addressing via `dst_reg++` explicitly. |

### Address Mode Configuration

**Binary SFPU operations** (`sub_binary_tile`):

The address mode is configured in `eltwise_binary_sfpu_configure_addrmod<SfpuType::unused>()` (called from `_llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>()`):

```
ADDR_MOD_7: { .srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0} }
```

This sets all address increments to zero. The SFPU kernel itself handles address progression explicitly:
- **Within a face**: `dst_reg++` advances by 1 sfpi row (= 2 physical DEST rows) per iteration (8 iterations per face)
- **Between faces**: `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` called twice between faces (advances by 8+8 = 16 physical DEST rows = 1 face width)

The `SfpuType::unused` template parameter means ADDR_MOD_6 is NOT additionally configured (that path only activates for mul_int32, mul_uint16, max, min, and their int32/uint32 variants).

This configuration is **identical between Wormhole and Blackhole** -- the `eltwise_binary_sfpu_configure_addrmod` function has the same logic in both architectures.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Checked for TANHSHRINK dispatch configuration (get_compute_kernel_path, get_op_init_and_func_default, get_op_approx_mode)
   **Key Findings**: TANHSHRINK is not present in any switch statement. get_compute_kernel_path defaults to eltwise_sfpu.cpp, get_op_init_and_func_default would throw, get_op_approx_mode defaults to false.

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_kernel.cpp`
   **Reason**: First variant of the dedicated tanhshrink compute kernel
   **Key Findings**: Uses tanh_tile(0) + binary_dest_reuse_tiles<ELWSUB, DEST_TO_SRCB> (FPU subtract). Relies on undefined llk_math_eltwise_unary_sfpu_tanh.

3. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_sfpu_kernel.cpp`
   **Reason**: Second variant using pure-SFPU approach for subtraction
   **Key Findings**: Uses tanh_tile(1) + sub_binary_tile(0,1,0). Copies input twice: once to DEST[1] for tanh, once to DEST[0] for subtraction. Also relies on undefined llk_math_eltwise_unary_sfpu_tanh.

4. **File**: `tt_metal/hw/inc/api/compute/compute_kernel_api.h`
   **Reason**: API definitions for tanh_tile and tanh_tile_init
   **Key Findings**: tanh_tile calls llk_math_eltwise_unary_sfpu_tanh which has NO implementation in the codebase. Template default is fast_and_approx=false.

5. **File**: `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h`
   **Reason**: API definitions for sub_binary_tile and sub_binary_tile_init
   **Key Findings**: sub_binary_tile routes through llk_math_eltwise_binary_sfpu_binop<APPROX, BinaryOp::SUB>.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h`
   **Reason**: Core SFPU implementation of binary operations
   **Key Findings**: _calculate_sfpu_binary_ with BinaryOp::SUB performs `in0 - in1` using SFPI vFloat arithmetic. Uses dst_tile_size_sfpi=32 for tile-based addressing. Identical between WH and BH.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary_sfpu_params.h`
   **Reason**: Parameters dispatch for binary SFPU operations
   **Key Findings**: VectorMode::RC iterates 4 faces with SETRWC between faces. SFPU function is called once per face with dst_index arguments.

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary_sfpu.h`
   **Reason**: Binary SFPU base layer with address mode configuration
   **Key Findings**: Sets ADDR_MOD_7 with all-zero increments. SfpuType::unused does not trigger ADDR_MOD_6 configuration.

9. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
   **Reason**: Metal SfpuType enum to check if tanh is registered
   **Key Findings**: Metal SfpuType only contains: unused, frac, swish, atanh, sinh. Tanh is NOT registered, confirming it was a built-in hardware function whose LLK binding was removed or never implemented.

10. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: SFPU hardware model reference for addressing and instruction semantics
    **Key Findings**: Confirmed stride-2 addressing model, dst_tile_size_sfpi=32, SFPMAD for float add/sub, SFPLOAD/SFPSTORE for DEST access.

11. **File**: `tt_metal/hw/inc/api/compute/eltwise_binary.h`
    **Reason**: API for binary_dest_reuse_tiles used in variant 1
    **Key Findings**: binary_dest_reuse_tiles<ELWSUB, DEST_TO_SRCB> is a FPU operation (not SFPU). It unpacks from CB to SRCA, moves DEST to SRCB, then performs FPU eltwise subtract.
