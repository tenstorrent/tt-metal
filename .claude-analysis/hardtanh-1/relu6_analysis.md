## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the RELU6 operation. RELU6 computes `min(max(x, 0), 6)` — clamping input values to the range [0, 6].

### Unary Dispatch Summary
- **UnaryOpType**: `RELU6`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary_ng/device/kernels/compute/eltwise_sfpu.cpp` (also `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` for legacy path)
- **SFPU_OP_CHAIN_0 expansion**: `relu_max_tile_init(); relu_max_tile(0, 0x40c00000u);`
  - The constant `0x40c00000u` is the IEEE 754 float32 bit pattern for `6.0f`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(UnaryOpType)` in `unary_ng_op_utils.cpp` line 194 — returns `false` for all ops (no switch cases, unconditional `return false`) |
| Template parameter (SFPU_OP_CHAIN) | `APPROX` (= `false`, inherited from `math_approx_mode`) | `relu_max_tile()` uses macro `SFPU_UNARY_ONE_PARAM_KERNEL_FN_FLOAT(_relu_max_, RC, APPROX, idst, param0)` — `APPROX` is the compile-time define set to `math_approx_mode` |
| Effective SFPU path | `APPROXIMATION_MODE=false` passed to `_relu_max_<sfpi::vFloat, false, 8, uint32_t>()` — however, this parameter is unused in the kernel; the implementation contains no `if constexpr (APPROXIMATION_MODE)` branches | The `_relu_max_impl_` function in `ckernel_sfpu_relu.h` has `APPROXIMATION_MODE` as a template parameter but never reads it |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/relu.h` — defines `relu_max_tile()`, `relu_max_tile_init()`, `relu_min_tile()`, `relu_tile()`, `leaky_relu_tile()` |
| **LLK Dispatch (Params)** | Wormhole: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` — `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>()` |
| | Blackhole: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |
| **LLK Dispatch (Init)** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_init.h` — `llk_math_eltwise_unary_sfpu_init<SfpuType::relu_max, APPROX>()` |
| **LLK Dispatch (Low-level)** | Wormhole: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h` — `_llk_math_eltwise_unary_sfpu_init_<SfpuType::relu_max>()`, addr_mod config |
| | Blackhole: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu.h` |
| **Core SFPU Implementation** | Wormhole: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h` — `_relu_max_()`, `_relu_max_impl_()` |
| | Blackhole: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_relu.h` |
| **Macros** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h` — `SFPU_UNARY_ONE_PARAM_KERNEL_FN_FLOAT`, `SFPU_UNARY_KERNEL_INIT` |
| **Parameters Dispatch** | Same as "LLK Dispatch (Params)" above — the params dispatch function handles face iteration and invokes the core SFPU function |

### Call Chain
1. **`relu_max_tile(0, 0x40c00000u)`** (API header `relu.h` line 32)
   → Expands via `SFPU_UNARY_ONE_PARAM_KERNEL_FN_FLOAT(_relu_max_, RC, APPROX, 0, 0x40c00000u)` macro

2. **`SFPU_UNARY_ONE_PARAM_KERNEL_FN_FLOAT`** (macros header line 142–144)
   → Expands to: `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::_relu_max_<sfpi::vFloat, APPROX, 8, uint32_t>, 0, (int)VectorMode::RC, 0x40c00000u)`

3. **`_llk_math_eltwise_unary_sfpu_params_<false>()`** (params dispatch, WH line 14)
   → Sets DEST write address, stalls SFPU until MATH completes, then iterates over 4 faces (RC mode), calling the SFPU function once per face with `SETRWC`/`inc_dst_addr` between faces

4. **`ckernel::sfpu::_relu_max_<sfpi::vFloat, false, 8, uint32_t>(0x40c00000u)`** (ckernel_sfpu_relu.h line 77)
   → Converts `uint32_t` threshold to `vFloat` via `Converter::as_float(0x40c00000u)` = `6.0f`, then calls `_relu_max_impl_<sfpi::vFloat, false, 8>(8, vFloat(6.0f))`

5. **`_relu_max_impl_<sfpi::vFloat, false, 8>(8, vFloat(6.0f))`** (ckernel_sfpu_relu.h line 55)
   → Iterates 8 times per face: loads DEST value, clamps to [0, threshold] using two `v_if` conditional blocks, stores result back

### Parameters Dispatch Summary
- **Vector mode**: `VectorMode::RC` — all 4 faces of the tile are processed (full tile coverage: 4 faces × 8 iterations × 32 elements = 1024 elements)
- **Operation invocation**: The params dispatch loops 4 times (once per face), calling `_relu_max_<sfpi::vFloat, false, 8, uint32_t>(0x40c00000u)` each iteration. Each call internally iterates 8 times using `dst_reg++` to advance through the face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC`/`inc_dst_addr<8>` between faces).
  - **Wormhole**: ADDR_MOD_7 configured with `{srca.incr=0, srcb.incr=0, dest.incr=0}`. The `dst_reg++` in the SFPI kernel handles intra-face advancement (1 sfpi row = 2 physical DEST rows = 32 elements per iteration). Between faces, the params dispatch issues `TTI_SETRWC(...CR_D, 8...)` twice (advancing by 16 physical rows = 1 face worth).
  - **Blackhole**: Identical ADDR_MOD_7 configuration. Between faces, `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (Style A). The Wormhole and Blackhole implementations are **identical** — the same `ckernel_sfpu_relu.h` code is used on both architectures (with the only difference being Wormhole uses `ckernel_sfpu_load_config.h` as an additional include, which has no impact on the relu_max logic). Below is the Wormhole version with annotations.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h

sfpi_inline sfpi::vFloat _relu_max_body_(sfpi::vFloat val, sfpi::vFloat threshold)
{
    sfpi::vFloat result = val;
    v_if (result > threshold) // CC-guarded: SFPXCMP sets CC where result > threshold
    {
        result = threshold; // Only lanes where result > threshold get clamped
    }
    v_endif;
    v_if (result < 0.0f) // CC-guarded: SFPXCMP sets CC where result < 0.0
    {
        result = 0.0f; // Only lanes where result < 0 get zeroed
    }
    v_endif;
    return result;
}

template <typename VecType, bool APPROXIMATION_MODE, int ITERATIONS> // APPROXIMATION_MODE=false, ITERATIONS=8
inline void _relu_max_impl_(const int iterations, VecType threshold)
{
    for (int d = 0; d < iterations; d++) // 8 iterations per face
    {
        VecType result = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST row
        v_if (result > threshold) // SFPXCMP(GT): compare result > 6.0, set CC per lane
        {
            result = threshold; // CC-guarded: clamp to 6.0 where result exceeds threshold
        }
        v_endif; // restore CC state
        v_if (result < 0) // SFPXCMP(LT): compare result < 0, set CC per lane
        {
            result = 0; // CC-guarded: clamp to 0.0 where result is negative
        }
        v_endif; // restore CC state
        sfpi::dst_reg[0] = result; // SFPSTORE: store 32 elements back to DEST row
        sfpi::dst_reg++; // advance to next sfpi row (+2 physical DEST rows, +32 elements)
    }
}

template <typename VectorType, bool APPROXIMATION_MODE, int ITERATIONS, typename T>
    // VectorType=sfpi::vFloat, APPROXIMATION_MODE=false, ITERATIONS=8, T=uint32_t
inline void _relu_max_(T threshold) // threshold = 0x40c00000u (IEEE 754 bits for 6.0f)
{
    static_assert(std::is_same_v<VectorType, sfpi::vFloat> || std::is_same_v<VectorType, sfpi::vInt>, "VectorType must be sfpi::vFloat or sfpi::vInt");

    VectorType v_threshold;
    if constexpr (std::is_same_v<T, float>)
    {
        v_threshold = threshold;
    }
    else if constexpr (std::is_same_v<T, std::uint32_t>) // This branch taken: T=uint32_t
    {
        if constexpr (std::is_same_v<VectorType, sfpi::vInt>)
        {
            v_threshold = static_cast<int>(Converter::as_float(threshold));
        }
        else // This branch taken: VectorType=sfpi::vFloat
        {
            v_threshold = Converter::as_float(threshold); // Reinterpret 0x40c00000 as float → 6.0f
        }
    }
    else
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, std::uint32_t>, "Threshold type must be float or uint32_t");
    }

    _relu_max_impl_<VectorType, APPROXIMATION_MODE, ITERATIONS>(ITERATIONS, v_threshold);
}
```

### SFPU Instructions Used

The `_relu_max_impl_` function uses SFPI C++ abstractions that compile down to the following SFPU instructions:

| SFPU Instruction | Source Abstraction | Purpose |
|------------------|--------------------|---------|
| `SFPLOAD` | `sfpi::dst_reg[0]` (read) | Load 32 elements from current DEST row into an LREG |
| `SFPSTORE` | `sfpi::dst_reg[0] = result` (write) | Store 32 elements from LREG back to current DEST row |
| `SFPXCMP` (GT mode) | `result > threshold` | Float compare: sets CC.Res=1 per lane where `result > threshold` |
| `SFPXCMP` (LT mode) | `result < 0` | Float compare: sets CC.Res=1 per lane where `result < 0.0` |
| `SFPLOADI` | `result = threshold` / `result = 0` | Load immediate float value into LREG (for the constant `6.0f` and `0.0f`). The SFPI compiler may optimize constant loading. |
| `SFPMOV` | `result = threshold` (conditional) | CC-guarded move of threshold/zero value into result LREG for clamped lanes |
| `SFPPUSHC` | `v_if` | Push current CC state onto CC stack to enable nested/sequential conditionals |
| `SFPPOPC` | `v_endif` | Pop CC state from stack, restoring previous CC context |
| `SFPENCC` | `v_if` / `v_endif` (CC management) | Enable/disable condition code masking for predicated execution |
| `SFPSETCC` | Generated by SFPI comparison | Set CC.Res based on comparison result (part of `SFPXCMP` expansion) |

**Note:** The SFPI compiler may optimize instruction sequences. The exact instruction stream depends on the compiler version, but the logical operations remain: load → compare+clamp_upper → compare+clamp_lower → store, repeated 8 times per face × 4 faces.

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0** | Holds `result` — the value loaded from DEST, conditionally modified, and stored back |
| **LREG (threshold)** | Holds the threshold value `6.0f` (converted from `0x40c00000u`). The SFPI compiler assigns this to an available LREG. Since `threshold` is a function parameter broadcast to a `vFloat`, it gets loaded once and reused across all 8 iterations. |
| **LREG (zero)** | Holds the constant `0.0f` for the lower-bound clamp. The SFPI compiler may use `SFPLOADI` to materialize this or use a known constant register. |
| **DEST rows** | Input/output: each iteration loads from and stores to the current DEST row addressed by `dst_reg`. The stride-2 model means each `dst_reg[0]` accesses 2 physical rows (32 elements). |
| **CC register** | Used for per-lane predication: the `v_if (result > threshold)` and `v_if (result < 0)` blocks each set CC to select which lanes get modified. |
| **CC stack** | Used by `v_if`/`v_endif` to save and restore CC state between the two conditional blocks. |

### Address Mode Configuration

The address mode for RELU6 (`SfpuType::relu_max`) is configured in the init function `eltwise_unary_sfpu_configure_addrmod<SfpuType::relu_max>()`.

**Both Wormhole and Blackhole** use the same configuration:

| ADDR_MOD | srca.incr | srcb.incr | dest.incr | Purpose |
|----------|-----------|-----------|-----------|---------|
| `ADDR_MOD_7` | 0 | 0 | 0 | Default SFPU addr_mod — no auto-increment on DEST. The SFPI abstraction handles DEST advancement via `dst_reg++` (software-managed). |

- `SfpuType::relu_max` does **not** match any of the special-cased types in `eltwise_unary_sfpu_configure_addrmod()` (which only special-cases `topk_local_sort`, `typecast`, `unary_max/min`, and on Blackhole also `reciprocal`).
- No `ADDR_MOD_6` is configured for this operation.
- The `ADDR_MOD_7` with `dest.incr=0` means the hardware does not auto-increment DEST pointers between SFPU instructions — all advancement is handled by the explicit `dst_reg++` in the SFPI kernel (which emits the appropriate RWC increment).

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp`
   **Reason**: Determine how RELU6 is dispatched — what init/func strings are injected as SFPU_OP_CHAIN macros
   **Key Findings**: RELU6 maps to `relu_max_tile_init()` / `relu_max_tile(idst, 0x40c00000u)` where `0x40c00000u` is `6.0f`

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/relu.h` (exists in main repo at `/localdev/vignjatijevic/tt-metal/tt_metal/hw/inc/api/compute/eltwise_unary/relu.h`)
   **Reason**: Trace the API-level `relu_max_tile()` function to understand macro expansion
   **Key Findings**: `relu_max_tile()` expands via `SFPU_UNARY_ONE_PARAM_KERNEL_FN_FLOAT(_relu_max_, RC, APPROX, idst, param0)`, resolving to `_llk_math_eltwise_unary_sfpu_params_<APPROX>()` with `_relu_max_<sfpi::vFloat, APPROX, 8, uint32_t>` as the callable

3. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h`
   **Reason**: Understand macro expansion from API call to LLK params dispatch
   **Key Findings**: `SFPU_UNARY_ONE_PARAM_KERNEL_FN_FLOAT` instantiates `_relu_max_<sfpi::vFloat, APPROX, 8, uint32_t>` and passes it to `_llk_math_eltwise_unary_sfpu_params_`

4. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Understand the params dispatch layer — how faces are iterated and the SFPU function is invoked
   **Key Findings**: RC mode iterates 4 faces with `SETRWC` between them; stalls SFPU before execution and waits after

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h`
   **Reason**: Core SFPU kernel implementation — the actual compute logic
   **Key Findings**: `_relu_max_impl_` uses pure SFPI abstractions (v_if, dst_reg, vFloat) to implement `min(max(x, 0), threshold)` with two conditional clamp blocks per iteration

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_relu.h`
   **Reason**: Compare Blackhole implementation against Wormhole
   **Key Findings**: Identical implementation — same `_relu_max_impl_`, `_relu_max_`, and `_relu_max_body_` functions

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Understand addr_mod configuration and init sequence
   **Key Findings**: `SfpuType::relu_max` only gets default `ADDR_MOD_7` with `dest.incr=0`; no special addr_mod case

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_converter.h`
   **Reason**: Understand how `uint32_t` threshold parameter is converted to `float`
   **Key Findings**: `Converter::as_float()` uses union-based bit reinterpretation: `0x40c00000u` → `6.0f`

9. **File**: `/localdev/vignjatijevic/tt-metal/runtime/sfpi/include/sfpi.h`
   **Reason**: Understand SFPI instruction mapping for comparison operators and conditional execution
   **Key Findings**: `vFloat > vFloat` emits `SFPXCMP` with `__vCondGT` mode; `v_if`/`v_endif` use `SFPPUSHC`/`SFPPOPC`/`SFPENCC` for CC stack management
