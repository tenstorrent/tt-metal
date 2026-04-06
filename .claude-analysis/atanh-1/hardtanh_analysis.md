## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `HARDTANH`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `hardtanh_tile_init(); hardtanh_tile(0, <min_val_bits>u, <max_val_bits>u);`

The `min_val` and `max_val` parameters are passed as `uint32_t` values obtained via `std::bit_cast<uint32_t>(float)`. Default values are `-1.0f` (min) and `1.0f` (max). The include guard `SFPU_OP_HARDTANH_INCLUDE` is defined, which causes `sfpu_split_includes.h` to include `api/compute/eltwise_unary/hardtanh.h`.

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported here.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(HARDTANH)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none | `get_op_init_and_func_parameterized()` -- the init call is `hardtanh_tile_init()` with no template arguments; `hardtanh_tile(idst, param0, param1)` likewise has no explicit approximation template parameter |
| Effective SFPU path | `APPROXIMATION_MODE=false` at the SFPU kernel level | `hardtanh_tile()` passes `APPROX` (which is `false`) as the template argument to `llk_math_eltwise_unary_sfpu_hardtanh<APPROX>()`. However, `calculate_hardtanh` does not branch on `APPROXIMATION_MODE` -- the code path is identical regardless of its value. |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/hardtanh.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardtanh.h` (identical across architectures) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_hardtanh.h` (identical across architectures) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain
1. **Compute kernel** (`eltwise_sfpu.cpp`): The `SFPU_OP_CHAIN_0` macro expands to `hardtanh_tile_init(); hardtanh_tile(0, <min_bits>u, <max_bits>u);` for each tile.
2. **API Header** (`hardtanh.h`): `hardtanh_tile(idst, param0, param1)` calls `llk_math_eltwise_unary_sfpu_hardtanh<APPROX>(idst, param0, param1)` inside the `MATH(...)` guard (active only on the math RISC-V thread).
3. **LLK Dispatch** (`llk_math_eltwise_unary_sfpu_hardtanh.h`): `llk_math_eltwise_unary_sfpu_hardtanh<APPROXIMATE, ITERATIONS=8>()` invokes `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>()` with the function pointer `ckernel::sfpu::calculate_hardtanh<APPROXIMATE, ITERATIONS>`, passing `dst_index`, `VectorMode::RC`, `param0`, and `param1`.
4. **Parameters Dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): Sets up DEST addressing, stalls for SFPU availability, then in `VectorMode::RC` loops over 4 faces, calling `calculate_hardtanh(param0, param1)` once per face and advancing DEST address by `SETRWC` (2 increments of 8 physical rows = 16 rows = one face) between faces.
5. **Core SFPU Implementation** (`ckernel_sfpu_hardtanh.h`): `calculate_hardtanh()` iterates 8 times per face, loading 32 elements per iteration from `dst_reg[0]`, clamping to `[min_val, max_val]` via two `v_if` conditional blocks, and writing the result back.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed (full tile coverage: 4 faces x 256 elements = 1024 elements).
- **Operation invocation**: The params dispatch calls `calculate_hardtanh(param0, param1)` once per face (4 times total for RC mode). Each invocation of `calculate_hardtanh` processes one face with `ITERATIONS=8` loop iterations.
- **DEST address progression**: On Wormhole, the params dispatch uses `TTI_SETRWC` with CR_D increment of 8 (issued twice per face = 16 physical rows per face). On Blackhole, it calls `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which issues `math::inc_dst_addr<8>()` twice per face. Both achieve the same result: advancing DEST by one face (16 physical rows) between invocations. Within `calculate_hardtanh`, `dst_reg++` advances 1 sfpi row (= 2 physical DEST rows = 32 elements) per iteration, covering 8 iterations x 32 elements = 256 elements per face. Standard DEST progression (ITERATIONS=8 per face, dst_reg++ per iteration, SETRWC between faces).

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `v_if`/`v_endif`), so Style A (inline-commented source code) is used.

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_hardtanh.h

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_hardtanh(std::uint32_t param0, std::uint32_t param1) {
    // APPROXIMATION_MODE=false, ITERATIONS=8
    // param0 and param1 are IEEE 754 float bits passed as uint32_t (bitcast from host)
    sfpi::vFloat min_val = Converter::as_float(param0); // reinterpret uint32_t as float, broadcast to all SFPU lanes
    sfpi::vFloat max_val = Converter::as_float(param1); // same for max_val

#pragma GCC unroll 8 // hint to unroll all 8 iterations for performance
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST position into LREG

        v_if(val < min_val) { val = min_val; } // SFPPUSHC + SFPXFCMPV(LT) + CC-guarded SFPMOV: clamp below min
        v_endif; // SFPPOPC: restore CC state

        v_if(val > max_val) { val = max_val; } // SFPPUSHC + SFPXFCMPV(GT) + CC-guarded SFPMOV: clamp above max
        v_endif; // SFPPOPC: restore CC state

        sfpi::dst_reg[0] = val; // SFPSTORE: write 32 clamped elements back to current DEST position
        sfpi::dst_reg++; // advance DEST pointer by 1 sfpi row (= 2 physical DEST rows = 32 elements)
    }
}
```

**Algorithmic summary**: The hardtanh operation implements element-wise clamping: `output = clamp(input, min_val, max_val)`. For each group of 32 elements, it loads the value from DEST, conditionally replaces it with `min_val` if it is below the minimum, conditionally replaces it with `max_val` if it is above the maximum, and stores the result back. This is equivalent to `max(min_val, min(max_val, x))`.

### SFPU Instructions Used

The following SFPU instructions are emitted by the SFPI compiler from the kernel source code. Since this kernel uses high-level SFPI abstractions, the exact instruction sequence is determined by the SFPI compiler backend, but the logical mapping is:

| SFPU Instruction | Source Construct | Description |
|------------------|-----------------|-------------|
| `SFPLOAD` | `sfpi::dst_reg[0]` (read) | Loads 32 elements (2 physical DEST rows x 16 elements/row) from the current DEST address into an LREG |
| `SFPSTORE` | `sfpi::dst_reg[0] = val` (write) | Stores 32 elements from an LREG back to the current DEST address |
| `SFPLOADI` | `Converter::as_float(param0)` / `Converter::as_float(param1)` | Loads the 32-bit immediate constants (min_val, max_val) into LREGs; may require two SFPLOADI instructions per constant (hi16 + lo16) |
| `SFPPUSHC` | `v_if(...)` expansion | Pushes current CC state onto the CC stack before entering a conditional block |
| `SFPPOPC` | `v_endif` expansion (destructor) | Pops CC state from the CC stack, restoring the previous CC |
| `SFPMAD` | `val < min_val` / `val > max_val` comparison | The float comparison `a < b` is implemented as subtraction `a - b` via SFPMAD, whose result sign determines the comparison outcome |
| `SFPSETCC` | `val < min_val` / `val > max_val` comparison | Sets CC.Res based on the comparison result (LT0 for `<`, GTE0 inverted for `>`) |
| `SFPMOV` | `val = min_val` / `val = max_val` (conditional assign) | CC-guarded register copy: only lanes where the condition is true have their value replaced |

Note: The exact instruction encoding depends on the SFPI compiler's lowering of `__builtin_rvtt_sfpxfcmpv` and `__builtin_rvtt_sfpxvif` built-ins. The `v_if(val < min_val)` construct compiles to the pseudo-instruction `SFPXFCMPV` with `CC_LT` mode, which the compiler expands into the appropriate SFPMAD + SFPSETCC sequence.

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG (val)** | Holds the current 32-element vector loaded from DEST; receives the clamped result before store-back |
| **LREG (min_val)** | Holds the broadcast `min_val` scalar (constant across all iterations), reinterpreted from `param0` via `Converter::as_float` |
| **LREG (max_val)** | Holds the broadcast `max_val` scalar (constant across all iterations), reinterpreted from `param1` via `Converter::as_float` |
| **LREG (temp)** | Used internally by the comparison sequence (SFPMAD subtraction result) for CC evaluation |
| **DEST rows** | Source and destination: 32 elements (2 physical rows) accessed per iteration via `dst_reg[0]` |
| **CC stack** | Used by `v_if`/`v_endif` to save and restore per-lane condition codes; depth = 1 for each `v_if` block (two independent blocks, not nested) |

The exact LREG allocation (which numbered LREG holds which value) is determined by the SFPI compiler's register allocator. The kernel requires at minimum 3 live LREGs simultaneously (val, min_val, max_val) plus temporaries for comparison results.

### Address Mode Configuration

The address mode for `hardtanh` is configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType::hardtanh>()` in `llk_math_eltwise_unary_sfpu.h`. Since `SfpuType::hardtanh` does not match any of the special-cased types (`topk_local_sort`, `typecast`, `reciprocal`, `unary_max/min`), only the default `ADDR_MOD_7` is configured:

**Wormhole B0 and Blackhole** (identical configuration):
```
ADDR_MOD_7:
  .srca = {.incr = 0}
  .srcb = {.incr = 0}
  .dest = {.incr = 0}
```

This zero-increment address mode means the hardware does not auto-increment the DEST address between SFPU instructions. Instead, DEST address progression is managed explicitly:
- **Within a face**: The SFPI `dst_reg++` construct advances the DEST read/write pointer by 1 sfpi row (2 physical DEST rows) per iteration.
- **Between faces**: The params dispatch issues `SETRWC` (Wormhole) or `inc_dst_addr<8>` x2 (Blackhole) to advance by one face (16 physical rows).

The init function also calls `sfpu::_init_sfpu_config_reg()` to configure the SFPU config register and `math::reset_counters(p_setrwc::SET_ABD_F)` to reset the A, B, D, and F counters.

---

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 expansion, and approximation mode for HARDTANH
   **Key Findings**: HARDTANH uses `eltwise_sfpu.cpp`, init is `hardtanh_tile_init()`, tile function is `hardtanh_tile(idst, param0, param1)` with bitcast float params; `get_op_approx_mode` returns `false` (default case)

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
   **Reason**: Verify HARDTANH is a parameterized type and understand the dispatch template structure
   **Key Findings**: `is_parametrized_type(HARDTANH)` returns `true`; params[0] = min_val, params[1] = max_val with defaults -1.0f and 1.0f

3. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/hardtanh.h`
   **Reason**: API header -- the tile-level entry point for the compute kernel
   **Key Findings**: `hardtanh_tile(idst, param0, param1)` calls `llk_math_eltwise_unary_sfpu_hardtanh<APPROX>(idst, param0, param1)` via MATH() guard; `hardtanh_tile_init()` calls `llk_math_eltwise_unary_sfpu_hardtanh_init<APPROX>()`

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardtanh.h`
   **Reason**: LLK dispatch layer -- bridges API to core SFPU function
   **Key Findings**: Calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>()` with `calculate_hardtanh<APPROXIMATE, ITERATIONS=8>` as the callable, VectorMode::RC, and param0/param1

5. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardtanh.h`
   **Reason**: Core SFPU implementation -- the actual kernel function
   **Key Findings**: Simple clamping kernel using SFPI abstractions; loads from dst_reg, applies two v_if guards for min/max clamping, stores back; 8 iterations per face processing 32 elements each; APPROXIMATION_MODE is unused (no branching on it)

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch -- manages DEST addressing and face iteration
   **Key Findings**: For VectorMode::RC, loops 4 faces, calls sfpu_func(args...) per face, uses SETRWC(CR_D, 8, SET_D) x2 per face to advance DEST by 16 physical rows

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Init function and addr_mod configuration
   **Key Findings**: `eltwise_unary_sfpu_configure_addrmod<SfpuType::hardtanh>()` sets only ADDR_MOD_7 with all-zero increments; no special cases for hardtanh

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_converter.h`
   **Reason**: Understand `Converter::as_float()` utility used by the kernel
   **Key Findings**: Simple union-based bitcast from uint32_t to float -- reinterprets IEEE 754 bits as float scalar

9. **File**: `runtime/sfpi/include/sfpi.h`
   **Reason**: Understand SFPI abstraction layer -- `v_if`/`v_endif` macro expansion and vFloat comparison operators
   **Key Findings**: `v_if(cond)` expands to SFPPUSHC + SFPXVIF + SFPXFCMPV (for vFloat comparisons); `v_endif` triggers destructor which calls SFPPOPC; vFloat `<` uses `__builtin_rvtt_sfpxfcmpv` with `SFPXCMP_MOD1_CC_LT`

10. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Authoritative reference for SFPU tile geometry, addressing model, and instruction semantics
    **Key Findings**: Confirmed ITERATIONS=8 per face, dst_reg++ = 2 physical rows, 32 elements per iteration, SFPMAD used for float subtraction in comparisons

11. **File**: `tt_metal/jit_build/genfiles.cpp`
    **Reason**: Understand how the `APPROX` compile-time constant is generated
    **Key Findings**: `APPROX` is emitted as `constexpr bool APPROX = <hlk_math_approx_mode>;` in the generated `chlkc_descriptors.h` file, sourced from `ComputeConfig.math_approx_mode`

12. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
    **Reason**: Understand the include guard mechanism for split SFPU ops
    **Key Findings**: When `SFPU_OP_HARDTANH_INCLUDE` is defined, includes `api/compute/eltwise_unary/hardtanh.h`
