## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `RSUB`
- **Compute kernel**: **NOT functional via the unary pipeline.** `UnaryOpType::RSUB` exists in the enum (`unary_op_types.hpp:72`) and `UNARY_OP_SCALAR_VARIANT(rsub_sfpu, RSUB)` creates the `rsub_sfpu` API function in `unary.hpp:175`, but the unary dispatch chain (`get_op_init_and_func_parameterized` in `unary_op_utils.cpp`) does not have a case for `RSUB`. Since `rsub_sfpu` passes a scalar parameter, the dispatch hits `get_op_init_and_func_parameterized`, which calls `TT_FATAL(is_parametrized_type(op_type), ...)`, and `is_parametrized_type(RSUB)` returns `false`. This means calling `rsub_sfpu` at runtime would **TT_FATAL** before reaching the program factory.
- **Actual functional path**: `BinaryOpType::RSUB` in the **binary_ng** pipeline (`binary_ng_utils.cpp:152-158`). When `is_sfpu_op()` is true, it maps to `SfpuBinaryOp::RSUB`, which produces the tile API calls `rsub_binary_tile_init()` / `rsub_binary_tile(idst0, idst1, odst)`.
- **SFPU_OP_CHAIN_0 expansion**: N/A for unary pipeline. For binary_ng: `rsub_binary_tile(idst0, idst1, odst)`
- **Non-SFPU fallback** (binary_ng, FPU path): When `is_sfpu_op()` is false, RSUB decomposes to `NEG(lhs)` + `FPU ADD`, avoiding the SFPU entirely.

#### Approximation Mode Resolution

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(RSUB)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | `APPROX` (from ComputeConfig) | `rsub_binary_tile()` passes `APPROX` to `llk_math_eltwise_binary_sfpu_binop<APPROX, BinaryOp::RSUB>` |
| Effective SFPU path | Approximation mode is unused -- `_sfpu_binary_init_` is an empty function, and `_calculate_sfpu_binary_<APPROX, RSUB>` does not branch on APPROXIMATION_MODE | The RSUB branch at line 52-55 of `ckernel_sfpu_binary.h` is a simple `result = in1 - in0` with no approximation-dependent logic |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h` (also identical at `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h`) |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h` (identical at `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary.h`) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary_sfpu_params.h` (identical at `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary_sfpu_params.h`) |
| **Init / SFPU Base** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary_sfpu.h` |
| **Init Wrapper** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_init.h` |

### Call Chain

1. **`rsub_binary_tile(idst0, idst1, odst)`** (API Header `eltwise_binary_sfpu.h:51-53`) -- calls `llk_math_eltwise_binary_sfpu_binop<APPROX, ckernel::BinaryOp::RSUB>(idst0, idst1, odst)` via `MATH((...))` macro to ensure it runs on the math thread.

2. **`llk_math_eltwise_binary_sfpu_binop<APPROX, BinaryOp::RSUB>()`** (LLK Dispatch `llk_math_eltwise_binary_sfpu_binop.h:19-28`) -- passes the core SFPU function `ckernel::sfpu::calculate_sfpu_binary<APPROX, RSUB, 8, false>` as a callable to `_llk_math_eltwise_binary_sfpu_params_<APPROX>()` along with the three tile indices and `VectorMode::RC` (default).

   **NOTE**: There is a naming inconsistency in this codebase. The LLK dispatch references `ckernel::sfpu::calculate_sfpu_binary` (without underscore prefix/suffix) but the function defined in `ckernel_sfpu_binary.h` is `_calculate_sfpu_binary_` (with underscores). Similarly, `sfpu_binary_init` vs `_sfpu_binary_init_`. This means the metal build path would encounter a linker/compiler error for these particular function names. The implementations with underscores contain the actual SFPU logic.

3. **`_llk_math_eltwise_binary_sfpu_params_<APPROX>(sfpu_func, ...)`** (Parameters Dispatch `llk_math_eltwise_binary_sfpu_params.h:14-75`) -- sets up SFPU execution context: validates tile indices, calls `_llk_math_eltwise_binary_sfpu_start_` to configure DEST write addressing and stall for SFPU readiness, then iterates over faces (4 faces for `VectorMode::RC`), calling `sfpu_func(dst_index_in0, dst_index_in1, dst_index_out)` per face, with `TTI_SETRWC` between faces to advance DEST address by 16 rows (2 x `inc_dst_addr<8>`).

4. **`_calculate_sfpu_binary_<APPROX, BinaryOp::RSUB, 8>(dst_index_in0, dst_index_in1, dst_index_out)`** (Core SFPU Implementation `ckernel_sfpu_binary.h:26-68`) -- per-face: iterates 8 times (ITERATIONS=8), loading `in0` from `dst_reg[dst_index_in0 * 32]`, `in1` from `dst_reg[dst_index_in1 * 32]`, computing `result = in1 - in0`, storing to `dst_reg[dst_index_out * 32]`, and advancing `dst_reg++` per iteration.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default for all binary SFPU operations called from `rsub_binary_tile`). All 4 faces of the tile are processed.
- **Operation invocation**: The params dispatch calls the core SFPU function once per face (4 calls total for RC mode). Each call processes 8 iterations within the face (ITERATIONS=8). Between faces, `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` is issued twice, advancing the DEST base address by 16 physical rows (= 1 face height).
- **DEST address progression**: Standard binary SFPU DEST progression. Within a face, `dst_reg++` advances 1 sfpi row (= 2 physical DEST rows, due to `SFP_DESTREG_STRIDE=2`) per iteration, covering 32 elements (2 rows x 16 elements/row). Between faces, `TTI_SETRWC` with CR_D and stride 8 is called twice (8+8=16 physical rows = 1 face). The address mode used is `ADDR_MOD_7` (all zero increments for srca, srcb, dest), consistent across both Wormhole and Blackhole. The `ADDR_MOD_7` configuration is specifically chosen to avoid conflicting with `ADDR_MOD_0` and `ADDR_MOD_2` used by the A2D (unpack-to-DEST) pipeline.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`), so Style A (inline-commented source code) is used.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h
// (Blackhole version at tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary.h is identical)

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8>
inline void _calculate_sfpu_binary_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{   // APPROXIMATION_MODE=false (unused by RSUB branch), BINOP=BinaryOp::RSUB, ITERATIONS=8
    static constexpr float nan = std::numeric_limits<float>::quiet_NaN(); // unused by RSUB branch
    for (int d = 0; d < ITERATIONS; d++) // 8 iterations per face
    {
        constexpr std::uint32_t dst_tile_size_sfpi = 32; // 32 sfpi rows per tile (64 physical / stride 2)
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi]; // Load 32 elements from input tile 0 at current sfpi row
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi]; // Load 32 elements from input tile 1 at current sfpi row
        sfpi::vFloat result = 0.0f; // Initialize result register

        // ... other BINOP branches (ADD, SUB, MUL, DIV) elided via if constexpr ...

        // RSUB branch: reverse subtraction (in1 - in0)
        // else if constexpr (BINOP == BinaryOp::RSUB)
        // {
            result = in1 - in0; // vFloat subtraction -> emits SFPMAD (in1 * 1.0 + (-in0)), computes scalar - x for each lane
        // }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result; // Store 32 elements to output tile at current sfpi row
        sfpi::dst_reg++; // Advance all three tile pointers by 1 sfpi row (= 2 physical DEST rows = 32 elements)
    }
}

template <bool APPROXIMATION_MODE /*unused*/, BinaryOp BINOP>
inline void _sfpu_binary_init_() {} // Empty init -- no LUT, no programmable constants needed for RSUB
```

### SFPU Instructions Used

| Instruction | Description | Source |
|-------------|-------------|--------|
| **SFPLOAD** | Loads 32 elements (2 physical DEST rows) from DEST into an SFPU local register (LREG). Emitted by `sfpi::vFloat in0 = sfpi::dst_reg[...]`. Two SFPLOAD instructions are issued per iteration -- one for `in0` (from input tile 0) and one for `in1` (from input tile 1). | `ckernel_sfpu_binary.h:32-33` |
| **SFPMAD** | Multiply-accumulate: `a * b + c`. This is the core ALU instruction for floating-point arithmetic. The subtraction `in1 - in0` is implemented as a single SFPMAD: `in1 * 1.0 + (-in0)` (the compiler negates `in0` via the sign bit in the SFPMAD encoding). There is no dedicated float subtraction instruction in the SFPU ISA. | `ckernel_sfpu_binary.h:54` (via `result = in1 - in0`) |
| **SFPSTORE** | Stores 32 elements from an SFPU local register back into DEST at the specified address. Emitted by `sfpi::dst_reg[...] = result`. | `ckernel_sfpu_binary.h:65` |
| **TTI_SETRWC** | Set Read/Write Counters -- advances the DEST base address between faces. Called from the parameters dispatch (`llk_math_eltwise_binary_sfpu_params.h`) between face iterations: `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` x 2 = advance by 16 physical rows. | `llk_math_eltwise_binary_sfpu_params.h:66-67` |
| **TTI_STALLWAIT** | Stall math thread until SFPU is ready. Issued once at the start (`_llk_math_eltwise_binary_sfpu_start_`) and once at the end (`_llk_math_eltwise_binary_sfpu_done_`). | `llk_math_eltwise_binary_sfpu.h:56,63` |

### SFPU Register Usage

| Register | Usage | Description |
|----------|-------|-------------|
| **LREG (in0)** | Input operand 0 | Loaded via `sfpi::dst_reg[dst_index_in0 * 32]`. Contains 32 elements from the first input tile's current sfpi row. |
| **LREG (in1)** | Input operand 1 | Loaded via `sfpi::dst_reg[dst_index_in1 * 32]`. Contains 32 elements from the second input tile's current sfpi row. |
| **LREG (result)** | Output accumulator | Initialized to `0.0f`, then set to `in1 - in0`. Written back to DEST via `sfpi::dst_reg[dst_index_out * 32]`. |
| **DEST registers** | Source and destination | Three DEST tile regions are addressed: `dst_index_in0 * 32` (input 0), `dst_index_in1 * 32` (input 1), `dst_index_out * 32` (output). The `dst_reg++` at the end of each iteration advances all three pointers by 1 sfpi row simultaneously. |
| **RWC_DEST** | DEST base address counter | Managed by the parameters dispatch. Reset to 0 at init, then advanced by `TTI_SETRWC` between faces (increments of 8, applied twice per face boundary = 16 physical rows per face). |

### Address Mode Configuration

The binary SFPU init function (`_llk_math_eltwise_binary_sfpu_init_`) configures address modes via `eltwise_binary_sfpu_configure_addrmod<SfpuType::unused>()`. Since `SfpuType::unused` does not match any of the special-cased types (`mul_int32`, `mul_uint16`, `max`, `min`, `max_int32`, `min_int32`, `max_uint32`, `min_uint32`), only `ADDR_MOD_7` is configured:

| Hardware | Address Mode | srca.incr | srcb.incr | dest.incr | Purpose |
|----------|-------------|-----------|-----------|-----------|---------|
| **Wormhole B0** | `ADDR_MOD_7` | 0 | 0 | 0 | All-zero increments. SFPU addressing is managed entirely by `dst_reg++` (SFPI auto-increment) within the kernel and `TTI_SETRWC` between faces, not by hardware address mode auto-increment. |
| **Blackhole** | `ADDR_MOD_7` | 0 | 0 | 0 | Same configuration as Wormhole. |

The `ADDR_MOD_7` slot is deliberately chosen to avoid conflicting with `ADDR_MOD_0` and `ADDR_MOD_2`, which are used by the A2D (unpack-to-DEST) pipeline that may be running concurrently on a different coprocessor thread.

Note: For RSUB specifically, `ADDR_MOD_6` (with `dest.incr = 2`) is NOT configured, because that mode is only set for `mul_int32`, `mul_uint16`, `max`, `min`, and their int32/uint32 variants.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`
   **Reason**: To find `UnaryOpType::RSUB` in the enum
   **Key Findings**: RSUB is enum value 72, exists in the unary type system

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: To check if RSUB has dispatch entries in `get_op_init_and_func`, `get_op_approx_mode`, `get_compute_kernel_path`
   **Key Findings**: RSUB is NOT in any switch case -- `get_op_approx_mode` returns false (default), `get_compute_kernel_path` returns `eltwise_sfpu.cpp` (default), but `get_op_init_and_func_parameterized` would TT_FATAL because `is_parametrized_type(RSUB)` returns false

3. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`
   **Reason**: To understand how `rsub_sfpu` is registered
   **Key Findings**: `UNARY_OP_SCALAR_VARIANT(rsub_sfpu, RSUB)` creates the API function, passes scalar as a parameter to `EltwiseUnaryWithParam`

4. **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp`
   **Reason**: To trace the actual functional RSUB dispatch path
   **Key Findings**: `BinaryOpType::RSUB` maps to `SfpuBinaryOp::RSUB` (SFPU path) or `NEG(lhs) + FPU ADD` (FPU path). SFPU path produces `rsub_binary_tile_init()` / `rsub_binary_tile`

5. **File**: `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h`
   **Reason**: API header defining `rsub_binary_tile()` and `rsub_binary_tile_init()`
   **Key Findings**: `rsub_binary_tile` calls `llk_math_eltwise_binary_sfpu_binop<APPROX, BinaryOp::RSUB>`. Init calls `llk_math_eltwise_binary_sfpu_binop_init<APPROX, BinaryOp::RSUB>`

6. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h`
   **Reason**: LLK dispatch layer bridging API to ckernel
   **Key Findings**: Passes `calculate_sfpu_binary<APPROX, BINOP, 8, false>` to params dispatcher. Note naming mismatch with actual function `_calculate_sfpu_binary_`

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary_sfpu_params.h`
   **Reason**: Parameters dispatch -- face iteration, SETRWC, SFPU start/done
   **Key Findings**: VectorMode::RC iterates 4 faces, 2x SETRWC(CR_D, 8) between faces. Identical for WH and BH

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h`
   **Reason**: Core SFPU implementation containing the `_calculate_sfpu_binary_` function
   **Key Findings**: RSUB branch (line 52-55): `result = in1 - in0`. Uses SFPI abstractions (vFloat, dst_reg). ITERATIONS=8, dst_tile_size_sfpi=32. Init function is empty (no LUT/constants needed)

9. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary.h`
   **Reason**: Check for Blackhole-specific differences
   **Key Findings**: Identical to Wormhole implementation

10. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary_sfpu.h`
    **Reason**: SFPU init, start, done, and address mode configuration
    **Key Findings**: `ADDR_MOD_7` configured with all-zero increments. `ADDR_MOD_6` only for mul_int32/max/min variants. Start stalls for SFPU, sets DEST write address. Done clears DEST address

11. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel_defs.h`
    **Reason**: BinaryOp enum definition
    **Key Findings**: `BinaryOp::RSUB = 4`

12. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: SFPU hardware model reference for tile geometry, DEST layout, stride-2 addressing
    **Key Findings**: Confirmed: dst_tile_size_sfpi=32, ITERATIONS=8 per face, 4 faces per tile, vFloat subtraction emits SFPMAD
