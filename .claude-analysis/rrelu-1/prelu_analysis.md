## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `PRELU_SFPU`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `prelu_tile(0, <param0_as_hex_uint32>u)`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(PRELU_SFPU)` in `unary_op_utils.cpp` -- currently returns `false` for all ops (switch has only a `default: return false` case) |
| Template parameter (SFPU_OP_CHAIN) | none (uses `APPROX` compile-time define, which equals `math_approx_mode`) | `get_op_init_and_func()` returns `prelu_tile_init()` / `prelu_tile(idst, param0)` -- the API header `prelu.h` passes `APPROX` directly to the macro, which forwards it as the template argument to `calculate_prelu<APPROX>` |
| Effective SFPU path | `APPROXIMATION_MODE=false`, but since `calculate_prelu` does not branch on `APPROXIMATION_MODE`, the code path is identical regardless | The kernel body has no `if constexpr (APPROXIMATION_MODE)` branch |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/prelu.h` |
| **LLK Dispatch** | This level of abstraction doesn't exist (the API header macro `SFPU_UNARY_ONE_PARAM_KERNEL_FN` calls directly into the params dispatch function) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_prelu.h` (BH) / `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_prelu.h` (WH) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (BH) / `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (WH) |

### Call Chain
1. **`prelu_tile(idst, param0)`** (API header `prelu.h`): Expands via the `SFPU_UNARY_ONE_PARAM_KERNEL_FN` macro to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_prelu<APPROX>, idst, (int)VectorMode::RC, param0)`.
2. **`_llk_math_eltwise_unary_sfpu_params_`** (params dispatch in `llk_math_eltwise_unary_sfpu_params.h`): Sets the DEST write address for the tile, stalls until SFPU is ready, then loops over 4 faces (since `VectorMode::RC`), calling `calculate_prelu<false>(param0)` once per face (8 iterations each), advancing the DEST face address between faces.
3. **`calculate_prelu<false>`** (core SFPU in `ckernel_sfpu_prelu.h`): Reinterprets `param0` as a float (the PReLU slope), then iterates 8 times per face. Each iteration: loads 32 elements from DEST, conditionally multiplies negative elements by the slope, writes the result back, and advances the DEST pointer.

### Parameters Dispatch Summary
- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed (full 32x32 tile = 1024 elements).
- **Operation invocation**: The params function loops `for (int face = 0; face < 4; face++)`, calling `calculate_prelu<false>(param0)` once per face. Each call runs `ITERATIONS=8` loop iterations inside the kernel, processing 8 sfpi rows x 32 elements/row = 256 elements per face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, face stride advancement between faces).
  - **Wormhole**: `ADDR_MOD_7` is set with all increments = 0 (srca=0, srcb=0, dest=0). Between faces, `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` is called twice (advancing by 16 physical DEST rows = 8 sfpi rows per call).
  - **Blackhole**: `ADDR_MOD_7` is set with all increments = 0. Between faces, `math::inc_dst_addr<8>()` is called twice (same net effect: advancing by 16 physical DEST rows total).
  - Within each face, `dst_reg++` in the kernel code issues `__builtin_rvtt_ttincrwc(0, 2, 0, 0)` which advances the DEST pointer by `SFP_DESTREG_STRIDE=2` physical rows per iteration.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`vFloat`, `dst_reg`, `v_if`/`v_endif`), so Style A (inline-commented source code) is used. The Blackhole and Wormhole implementations are identical except for the `#pragma GCC unroll` directive (0 for BH, 8 for WH).

```cpp
// File: tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_prelu.h
// (Wormhole version at tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_prelu.h is identical except #pragma GCC unroll 8)

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_prelu(const uint value) { // APPROXIMATION_MODE=false, ITERATIONS=8
    // SFPU microcode
    vFloat init = Converter::as_float(value); // Reinterpret param0 bits as float -> SFPLOADI (load immediate float into LREG)

#pragma GCC unroll 0 // BH: no unrolling; WH: #pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) { // 8 iterations per face, each processes 32 elements (2 DEST rows)
        vFloat a = dst_reg[0]; // SFPLOAD: load 32 elements from current DEST position into LREG
        v_if(a < 0.0f) { // SFPXFCMPS(a, 0.0, CC_LT) + SFPPUSHC: compare a < 0.0 and set CC, push CC stack
            a = a * init; // SFPMUL: multiply negative elements by slope; CC-guarded (only lanes where a < 0)
        }
        v_endif; // SFPPOPC: pop CC stack, restoring previous condition code state
        dst_reg[0] = a; // SFPSTORE: write 32 elements back to current DEST position
        dst_reg++; // INCRWC: advance DEST pointer by SFP_DESTREG_STRIDE=2 physical rows
    }
}
```

### SFPU Instructions Used

| Instruction | Builtin | Description |
|-------------|---------|-------------|
| **SFPLOADI** | `__builtin_rvtt_sfpxloadi` | Load an immediate scalar value into all 32 SIMD lanes. Used to convert the `param0` uint32 into a vFloat representing the PReLU slope. |
| **SFPLOAD** | `__builtin_rvtt_sfpload` | Load 32 elements from the current DEST register position into an SFPU LREG. Used by `vFloat a = dst_reg[0]`. Format: `SFPLOAD_MOD0_FMT_SRCB` (bfloat16/float16b format). |
| **SFPXFCMPS** | `__builtin_rvtt_sfpxfcmps` | Compare a vector of floats against a scalar float and set the condition code (CC). Used by `a < 0.0f` with `SFPXCMP_MOD1_CC_LT` mode. Each lane's CC bit is set if the element is less than 0. |
| **SFPPUSHC** | `__builtin_rvtt_sfppushc` | Push the current condition code onto the CC stack. Used by `v_if` to save CC state before entering the conditional block. |
| **SFPMUL** | `__builtin_rvtt_sfpmul` | Multiply two vectors element-wise. Used by `a * init` to scale negative elements by the PReLU slope. This instruction is CC-guarded: only lanes where CC is set (a < 0) perform the multiplication. |
| **SFPPOPC** | `__builtin_rvtt_sfppopc` | Pop the condition code stack, restoring the CC state from before `v_if`. Used by `v_endif` to end the conditional block. |
| **SFPSTORE** | `__builtin_rvtt_sfpstore` | Store 32 elements from an SFPU LREG back to the current DEST register position. Used by `dst_reg[0] = a`. Format: `SFPSTORE_MOD0_FMT_SRCB`, address mode: `SFPSTORE_ADDR_MODE_NOINC`. |
| **INCRWC** | `__builtin_rvtt_ttincrwc` | Increment the DEST write counter by `SFP_DESTREG_STRIDE=2` physical DEST rows. Used by `dst_reg++` to advance to the next pair of rows within a face. |

Additionally, the SFPI compiler may emit auxiliary instructions for condition code management:
- **SFPXVIF** (`__builtin_rvtt_sfpxvif`): Internal to the `cc_if()` mechanism, prepares the CC dependency tracking for the `v_if` block.
- **SFPXCONDB** (`__builtin_rvtt_sfpxcondb`): Internal to the `cc_cond()` mechanism, applies the comparison result to the CC enable mask.

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST rows** | Input/output storage. Each iteration reads 2 physical DEST rows (32 elements) and writes back 2 rows. Over 8 iterations per face and 4 faces, all 64 physical DEST rows (1024 elements) of one tile are processed. |
| **LREG (implicit)** | The SFPI compiler allocates LREGs automatically. At minimum: one LREG for `init` (the PReLU slope, loaded once via SFPLOADI and reused across iterations), one LREG for `a` (loaded from DEST each iteration via SFPLOAD, potentially modified by SFPMUL, then stored back). |
| **CC stack** | The condition code stack is used by `v_if`/`v_endif` to manage per-lane conditional execution. `SFPPUSHC` saves the CC state; `SFPPOPC` restores it. Only one level of CC nesting is used by this kernel. |

### Address Mode Configuration

The `prelu_tile_init()` call triggers `llk_math_eltwise_unary_sfpu_init<SfpuType::prelu, APPROX>()`, which calls `eltwise_unary_sfpu_configure_addrmod<SfpuType::prelu>()`.

Since `SfpuType::prelu` does not match any of the special-case `if constexpr` branches (those are for `topk_local_sort`, `reciprocal`, `typecast`, `signbit`, and `unary_max`/`unary_min` variants), only the default `ADDR_MOD_7` is configured:

**Blackhole and Wormhole (identical):**
```
ADDR_MOD_7: { srca.incr = 0, srcb.incr = 0, dest.incr = 0 }
```

This means the hardware auto-increment is disabled for this operation. All DEST address advancement is handled explicitly:
- **Within a face**: `dst_reg++` issues `INCRWC` to advance by `SFP_DESTREG_STRIDE=2` physical DEST rows per iteration.
- **Between faces**:
  - **Wormhole**: Two `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` instructions advance by 16 physical DEST rows total (= 1 face = 256 elements).
  - **Blackhole**: Two `math::inc_dst_addr<8>()` calls achieve the same 16-row advancement.

Note: `ADDR_MOD_7` is deliberately chosen to avoid conflicts with `ADDR_MOD_0` and `ADDR_MOD_2`, which are used by the A2D (unpack-to-DEST) pipeline that runs concurrently.

## External Knowledge Sources
### DeepWiki Queries
1. [SFPU] **Query**: "How do SFPI abstractions like vFloat, dst_reg, v_if, v_endif, and vFloat multiplication map to underlying SFPU instructions?"
   **Reason**: Needed to understand the instruction-level semantics of the SFPI high-level abstractions used by the prelu kernel.
   **Key Findings**: DeepWiki returned a 429 rate limit error. Analysis was performed entirely from source code (`runtime/sfpi/include/sfpi.h`). Key mappings confirmed: `vFloat * vFloat` -> `__builtin_rvtt_sfpmul` (SFPMUL), `dst_reg[0]` read -> `__builtin_rvtt_sfpload` (SFPLOAD), `a < 0.0f` -> `__builtin_rvtt_sfpxfcmps` (SFPXFCMPS), `v_if` -> `SFPPUSHC` + `SFPXVIF` + `SFPXCONDB`, `v_endif` -> `SFPPOPC`, `dst_reg++` -> `INCRWC`.

### Confluence References
No Confluence pages were consulted for this analysis.

### Glean References
No Glean queries were made for this analysis.
