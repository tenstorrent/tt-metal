## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `PRELU_SFPU`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `prelu_tile_init(); prelu_tile(0, param0_as_uint32);`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(PRELU_SFPU)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none (no parameterized template argument) | `get_op_init_and_func()` -- `prelu_tile_init()` and `prelu_tile(idst, param0)` use no template params; the `APPROX` constant is generated at JIT time as `constexpr bool APPROX = false;` in `chlkc_descriptors.h` via `genfiles.cpp:394` |
| Effective SFPU path | `APPROXIMATION_MODE=false` -- however, the `calculate_prelu` function does not branch on `APPROXIMATION_MODE` at all, so both `true` and `false` execute the identical code path | The function body has no `if constexpr(APPROXIMATION_MODE)` branch |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/prelu.h` |
| **LLK Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_prelu.h` |
| **Parameters Dispatch** | Same as LLK Dispatch: `llk_math_eltwise_unary_sfpu_params.h` (the `_llk_math_eltwise_unary_sfpu_params_` function handles both dispatch and face iteration) |

### Call Chain
1. **`prelu_tile(idst, param0)`** (API header, `prelu.h:28`) expands via `SFPU_UNARY_ONE_PARAM_KERNEL_FN(calculate_prelu, RC, APPROX, idst, param0)` to call `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_prelu<APPROX>, idst, (int)VectorMode::RC, param0)`.
2. **`_llk_math_eltwise_unary_sfpu_params_<false>`** (LLK dispatch, `llk_math_eltwise_unary_sfpu_params.h`) sets the DEST write address, stalls for SFPU availability, then loops over 4 faces (VectorMode::RC), calling the SFPU function once per face and advancing the DEST address between faces via `SETRWC`/`inc_dst_addr`.
3. **`calculate_prelu<false>(param0)`** (core SFPU, `ckernel_sfpu_prelu.h`) executes the inner SFPU microcode: for each of 8 iterations per face, loads an element from DEST, conditionally multiplies by the scalar parameter if negative, and stores the result back to DEST.

### Parameters Dispatch Summary
- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed (full tile operation on all 1024 elements).
- **Operation invocation**: The dispatch function calls `calculate_prelu<false>(param0)` once per face, for a total of 4 invocations. Each invocation internally loops 8 iterations (ITERATIONS=8), processing 32 elements per iteration (8 iterations x 32 elements = 256 elements = 1 face).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC`/`inc_dst_addr` between faces). On Wormhole, `ADDR_MOD_7` is configured with `dest.incr=0` (the SFPI `dst_reg++` handles per-iteration advancement internally, and `TTI_SETRWC` with increment 8 twice advances by 16 physical rows = 1 face between faces). On Blackhole, the same `ADDR_MOD_7` with `dest.incr=0` is used, with `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calling `math::inc_dst_addr<8>()` twice between faces.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`vFloat`, `dst_reg`, `v_if`/`v_endif`), so Style A (inline-commented source) is used.

The Wormhole and Blackhole implementations are identical in logic; the only difference is the `#pragma GCC unroll` directive (8 on Wormhole, 0 on Blackhole) and `const` qualifier on the parameter. The Wormhole version is shown below with annotations.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_prelu.h

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_prelu(uint value) { // APPROXIMATION_MODE=false, ITERATIONS=8
    // Reinterpret the uint32 parameter as a float via union-based type punning
    vFloat init = Converter::as_float(value); // SFPLOADI x2 (lo16 + hi16) to load the scalar into an LREG

#pragma GCC unroll 8 // Compiler hint to fully unroll the 8 iterations (BH uses unroll 0 = no unroll)
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat a = dst_reg[0]; // SFPLOAD: load 32 elements from current DEST position into LREG
        v_if(a < 0.0f) { // SFPPUSHC + SFPSETCC(CC_LT): push CC, set CC.Res=1 for lanes where a < 0
            a = a * init;  // SFPMAD (a * init + 0.0): multiply negative elements by the prelu slope
        }
        v_endif; // SFPPOPC: pop CC state, restoring all-lanes-enabled
        dst_reg[0] = a; // SFPSTORE: store 32 elements back to current DEST position
        dst_reg++;       // Advance DEST pointer by 1 sfpi row (= 2 physical DEST rows = 32 elements)
    }
}
```

**Blackhole variant differences** (file: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_prelu.h`):
- `#pragma GCC unroll 0` instead of `#pragma GCC unroll 8` (loop is not unrolled on Blackhole)
- Parameter is `const uint value` instead of `uint value`
- All other logic is identical

### SFPU Instructions Used

| Instruction | Emitted By | Description |
|-------------|-----------|-------------|
| `SFPLOADI` (x2) | `vFloat init = Converter::as_float(value)` | Loads the 32-bit float scalar parameter into an LREG in two steps (lo16, hi16). Executed once before the loop, so it is hoisted outside the per-iteration work. |
| `SFPLOAD` | `vFloat a = dst_reg[0]` | Loads 32 elements (2 physical DEST rows) from the current DEST address into an LREG. Uses IMPLIED format mode. |
| `SFPPUSHC` | `v_if(...)` | Pushes the current CC state (CC.En, CC.Res) onto the per-lane CC stack. This preserves the outer CC state so it can be restored after the conditional block. |
| `SFPSETCC` (CC_LT / `SFPXCMP` with LT mode) | `a < 0.0f` within `v_if` | Sets CC.Res=1 for lanes where the loaded value is less than zero (sign bit test). Lanes with non-negative values have CC.Res=0 and will be masked out for subsequent guarded instructions. |
| `SFPMAD` | `a = a * init` | Fused multiply-add: computes `a * init + 0.0` for the conditional multiplication. Only executes on lanes where CC.Res=1 (i.e., negative input values). This is the core PReLU scaling operation. |
| `SFPPOPC` | `v_endif` (destructor of `__vCCCtrl`) | Pops the CC stack, restoring the previous CC state. After this, all lanes are enabled again for the unconditional SFPSTORE. |
| `SFPSTORE` | `dst_reg[0] = a` | Stores the result (32 elements) from the LREG back to the current DEST address. Uses IMPLIED format mode. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG (general purpose)** | One LREG holds the scalar `init` value (the PReLU slope parameter), loaded once before the loop via `SFPLOADI` x2. Another LREG holds `a` (the tile element loaded from DEST). The SFPI compiler manages LREG allocation automatically. |
| **DEST register** | Input tile data is read from DEST via `SFPLOAD` and results are written back via `SFPSTORE`. The DEST address auto-advances by `dst_reg++` (1 sfpi row = 2 physical rows = 32 elements per iteration). |
| **CC stack** | One level of CC stack depth is used per iteration by the `v_if`/`v_endif` block. The stack is pushed at `v_if` and popped at `v_endif`, so it returns to depth 0 after each iteration. |
| **Constant registers** | Not explicitly programmed by this kernel. The `0.0f` literal in the comparison `a < 0.0f` is handled by the SFPI compiler -- for a less-than-zero test, the compiler can use `SFPSETCC` with `LREG_LT0` mode (sign bit test), which does not require loading 0.0f as a separate constant. |

### Address Mode Configuration

The address mode is configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType::prelu>()` during `_llk_math_eltwise_unary_sfpu_init_()`. Since `SfpuType::prelu` does not match any of the special-cased `if constexpr` branches (topk_local_sort, typecast, unary_max/min, signbit), only the default `ADDR_MOD_7` is configured:

| Hardware | ADDR_MOD | srca.incr | srcb.incr | dest.incr |
|----------|----------|-----------|-----------|-----------|
| Wormhole B0 | `ADDR_MOD_7` | 0 | 0 | 0 |
| Blackhole | `ADDR_MOD_7` | 0 | 0 | 0 |

Both Wormhole and Blackhole configure `ADDR_MOD_7` identically with all increments set to 0. The DEST address advancement is handled entirely by:
- **Within a face**: The SFPI `dst_reg++` abstraction, which internally advances the SFPU's read/write counter by `SFP_DESTREG_STRIDE=2` physical rows per iteration.
- **Between faces**: On Wormhole, two `TTI_SETRWC(..., 8, ...)` calls advance by 16 physical rows (= 1 face). On Blackhole, two `math::inc_dst_addr<8>()` calls achieve the same.

The `ADDR_MOD_7` with `dest.incr=0` means the hardware address mode does NOT auto-increment DEST -- all advancement is explicit, giving the SFPI runtime full control over addressing.

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "How does the SFPI v_if / v_endif mechanism work in SFPU kernels? What SFPU instructions does it emit?"
   **Reason**: Needed to understand the CC instruction sequence generated by the `v_if(a < 0.0f) { ... } v_endif;` pattern used in the prelu kernel.
   **Key Findings**: DeepWiki was unavailable for this repository. The information was obtained directly from the SFPI source code at `runtime/sfpi/include/sfpi.h`. The `v_if` macro creates a `__vCCCtrl` object, calls `cc_push()` (emits `SFPPUSHC`), `cc_if()` (prepares dependency tracking), and `cc_cond()` (emits comparison via `__builtin_rvtt_sfpxcondb`). The `v_endif` macro closes the scope, triggering the `__vCCCtrl` destructor which calls `cc_pop()` (emits `SFPPOPC`).

### Confluence References
No Confluence references were needed for this analysis. The prelu kernel is straightforward and uses well-documented SFPI abstractions.

### Glean References
No Glean references were needed for this analysis.
