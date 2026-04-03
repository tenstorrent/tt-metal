## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `PRELU_SFPU`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `prelu_tile(idst, param0)` where `param0` is the bit-cast `uint32_t` representation of the scalar slope value

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(PRELU_SFPU)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none (uses default `APPROX`) | `get_op_init_and_func()` returns `prelu_tile_init()` / `prelu_tile({idst}, {param0})` -- no explicit template parameter; `APPROX` macro resolves to the `math_approx_mode` value (`false`) |
| Effective SFPU path | `APPROXIMATION_MODE=false` in `calculate_prelu` | The `calculate_prelu` template parameter `APPROXIMATION_MODE` is `false`. However, the kernel does not contain any `if constexpr (APPROXIMATION_MODE)` branches, so the approximation mode has **no effect** on the execution path -- both modes produce identical code. |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/prelu.h` |
| **LLK Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Wormhole) / `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Blackhole) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_prelu.h` (Wormhole) / `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_prelu.h` (Blackhole) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (same file as LLK Dispatch) |

### Call Chain
1. **Compute kernel** (`eltwise_sfpu.cpp`) invokes `SFPU_OP_CHAIN_0` which expands to `prelu_tile(0, param0)`.
2. **API header** (`prelu.h`) defines `prelu_tile(idst, param0)` which calls `MATH(SFPU_UNARY_ONE_PARAM_KERNEL_FN(calculate_prelu, RC, APPROX, idst, param0))`.
3. **Macro expansion** (`llk_math_eltwise_unary_sfpu_macros.h`) expands this to `_llk_math_eltwise_unary_sfpu_params_<false>(ckernel::sfpu::calculate_prelu<false>, idst, (int)VectorMode::RC, param0)`.
4. **LLK dispatch** (`llk_math_eltwise_unary_sfpu_params.h`) sets the DEST write address, stalls until SFPU is ready, then for `VectorMode::RC` loops over all 4 faces calling `calculate_prelu<false>(param0)` once per face, advancing the DEST pointer between faces with `SETRWC`.
5. **Core SFPU function** (`ckernel_sfpu_prelu.h`) executes 8 iterations per face, loading each 32-element chunk from DEST, conditionally multiplying negative elements by the slope, and storing the result back.

### Parameters Dispatch Summary
- **Vector mode**: `VectorMode::RC` -- all 4 faces (the full 32x32 tile) are processed.
- **Operation invocation**: The params dispatch calls `calculate_prelu<false>(param0)` once per face in a `for (face = 0; face < 4; face++)` loop. Each call to `calculate_prelu` internally loops 8 times (ITERATIONS=8), processing one sfpi row (32 elements) per iteration, covering all 256 elements of the face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). On Wormhole, the params dispatch uses `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` twice between faces (advancing by 16 physical DEST rows = 1 face). On Blackhole, the params dispatch calls `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which does `math::inc_dst_addr<8>()` twice.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`vFloat`, `dst_reg`, `v_if`/`v_endif`), so Style A (inline-commented source) is used.

The Wormhole and Blackhole implementations are identical except for the `#pragma GCC unroll` directive (`8` on Wormhole for full unrolling, `0` on Blackhole for no unrolling). Only the Wormhole version is shown below; differences are noted inline.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_prelu.h

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_prelu(uint value) { // APPROXIMATION_MODE=false, ITERATIONS=8
    // Reinterpret the uint32_t param as a float (the PReLU slope)
    vFloat init = Converter::as_float(value); // SFPLOADI: loads scalar slope into an LREG, broadcast across all lanes

#pragma GCC unroll 8 // Blackhole uses #pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat a = dst_reg[0]; // SFPLOAD: load 32 elements (2 physical DEST rows) into an LREG
        v_if(a < 0.0f) { // SFPPUSHC + SFPENCC + SFPSETCC(LT0): enable CC, test sign bit, guarding negative lanes
            a = a * init; // SFPMUL (CC-guarded): multiply negative elements by slope
        }
        v_endif; // SFPPOPC: restore CC state (all lanes enabled)
        dst_reg[0] = a; // SFPSTORE: write result back to DEST (CC-guarded stores write only for active lanes)
        dst_reg++; // INCRWC: advance DEST pointer by 1 sfpi row (= 2 physical DEST rows = 32 elements)
    }
}
```

### SFPU Instructions Used

| Instruction | Source (SFPI abstraction) | Description |
|-------------|--------------------------|-------------|
| **SFPLOADI** | `vFloat init = Converter::as_float(value)` | Load 32-bit immediate (the slope parameter) into an LREG using float format. The scalar is broadcast to all 32 SFPU lanes. |
| **SFPLOAD** | `vFloat a = dst_reg[0]` | Load 32 elements from DEST (2 physical rows at the current DEST pointer) into an LREG, using SRCB format (implied bfloat16-to-FP32 conversion). |
| **SFPPUSHC** | `v_if(...)` | Push current CC state onto the per-lane CC stack (saves prior CC for restoration at `v_endif`). |
| **SFPENCC** | `v_if(...)` | Enable condition code checking. After this, subsequent instructions are predicated on the CC bits. |
| **SFPSETCC** | `a < 0.0f` | Set CC.Res based on the sign bit of the loaded value. For `< 0.0f`, uses `SFPSETCC_MOD1_LREG_LT0` (CC.Res = 1 if value is negative). This is a sign-bit test, not a full floating-point comparison. |
| **SFPMUL** | `a = a * init` | Multiply the input value by the slope. CC-guarded: only executes on lanes where CC.Res = 1 (negative values). Emitted via `__builtin_rvtt_sfpmul`. 2-cycle latency, fully pipelined. |
| **SFPPOPC** | `v_endif` | Pop CC state from the stack, restoring all lanes to active (the `__vCCCtrl` destructor calls `cc_pop()`). |
| **SFPSTORE** | `dst_reg[0] = a` | Store 32 elements from LREG back to DEST at the current pointer, using SRCB format. |
| **INCRWC** | `dst_reg++` | Increment the DEST read/write counter by `SFP_DESTREG_STRIDE` (2 physical rows), advancing to the next sfpi row. Emitted via `__builtin_rvtt_ttincrwc`. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG (init)** | Holds the PReLU slope parameter for the entire kernel execution. Loaded once via `SFPLOADI` before the loop. Persistent across all 8 iterations (and across all 4 face invocations since the params dispatch calls the function fresh each time, but the compiler may hoist it). |
| **LREG (a)** | Temporary register holding the loaded DEST value for the current iteration. Loaded via `SFPLOAD`, conditionally modified by `SFPMUL`, then stored back via `SFPSTORE`. Reused each iteration. |
| **DEST register** | The tile data resides in DEST. Each iteration reads 32 elements (2 physical rows x 16 elements/row) from DEST, conditionally modifies them, and writes them back in-place. The DEST pointer advances by `SFP_DESTREG_STRIDE=2` physical rows per iteration. |
| **CC Stack** | Used by `v_if`/`v_endif` to save/restore CC state. Depth = 1 (single `SFPPUSHC`/`SFPPOPC` pair per iteration). |

### Address Mode Configuration

The `prelu` operation uses `SfpuType::prelu` which does not match any special case in `eltwise_unary_sfpu_configure_addrmod()`. Only the default `ADDR_MOD_7` is configured:

| Field | Value | Meaning |
|-------|-------|---------|
| `srca.incr` | 0 | No SrcA auto-increment |
| `srcb.incr` | 0 | No SrcB auto-increment |
| `dest.incr` | 0 | No DEST auto-increment via address mode |

This is identical on both Wormhole and Blackhole. DEST advancement is handled explicitly by the `dst_reg++` (INCRWC) instruction within the SFPU kernel loop rather than through address mode auto-increment. Between faces, the params dispatch uses `SETRWC` (Wormhole) or `math::inc_dst_addr<8>()` (Blackhole) to advance by one face stride (16 physical DEST rows = 2 SETRWC increments of 8).

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "How does the SFPI v_if/v_endif mechanism work in SFPU kernels? Specifically, what SFPU instructions are emitted by `v_if(a < 0.0f)`, `v_endif`, and what happens when you do `a = a * init` inside a v_if block? What instructions does vFloat multiplication emit (SFPMAD or SFPMUL)?"
   **Reason**: Needed to confirm the exact SFPU instruction mapping for SFPI C++ abstractions used in the prelu kernel (v_if, vFloat multiplication, CC management).
   **Key Findings**: DeepWiki returned "Repository not found" -- the tt-metal repository is not indexed on DeepWiki. All instruction mapping was derived from direct source code analysis of `runtime/sfpi/include/sfpi.h`.

### Confluence References
No Confluence pages were consulted for this analysis. The SFPU ISA reference was not needed because the kernel uses only standard SFPI abstractions with well-understood instruction mappings (SFPLOAD, SFPSTORE, SFPMUL, SFPSETCC, SFPPUSHC/SFPPOPC).

### Glean References
No Glean queries were needed for this analysis.
