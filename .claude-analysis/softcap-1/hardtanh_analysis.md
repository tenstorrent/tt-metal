# SFPU Analysis: hardtanh

**Operation:** hardtanh
**Math:** `hardtanh(x) = clamp(x, min_val, max_val)`
**Equivalent PyTorch:** `torch.nn.functional.hardtanh`
**Parametrized:** Yes — two float params: `min_val` (default -1.0) and `max_val` (default 1.0)

---

## 1. Math Definition and Algorithm Strategy

`hardtanh` clamps each input element to the closed interval `[min_val, max_val]`:

```
hardtanh(x) = min_val   if x < min_val
              x          if min_val ≤ x ≤ max_val
              max_val    if x > max_val
```

This is a pure arithmetic/comparison operation — no transcendentals, no polynomial approximations. The SFPU implements it via a **three-shift piecewise algorithm** using three pre-encoded FP16_B scalar constants.

### Three-Shift Algorithm

Given `L = min_val`, `H = max_val`, the three SFPU params are:

| Param | Value | Role |
|-------|-------|------|
| `p0`  | `-L` (= `-min_val`) | Shift x up so lower bound maps to 0 |
| `p1`  | `-(H - L)` (= `min_val - max_val`) | Shift down so upper bound maps to 0 |
| `p2`  | `H` (= `max_val`) | Restore correct final position |

Per-element execution trace:

| Step | Operation | x < L | L ≤ x ≤ H | x > H |
|------|-----------|--------|------------|-------|
| 1 | `val = x` | `x` | `x` | `x` |
| 2 | `val += p0` | `x - L < 0` | `x - L ≥ 0` | `x - L > 0` |
| 3 | if `val < 0`: `val = 0` | **0** | `x - L` | `x - L` |
| 4 | `val += p1` | `-(H-L)` | `x - H ≤ 0` | `x - H > 0` |
| 5 | if `val ≥ 0`: `val = 0` | `-(H-L)` | `x - H` (if `x<H`) **or 0** (if `x=H`) | **0** |
| 6 | `val += p2` | `-(H-L)+H = L` ✓ | `x - H + H = x` ✓ | `0 + H = H` ✓ |

All three cases produce the exact clamped result. The `APPROXIMATION_MODE` template parameter has **no effect** — both modes execute identical code because clamp is an exact operation.

**Numerical precision:** Parameter values are packed as BFloat16 (FP16_B, `uint32_t` bit pattern). For default min=-1.0/max=1.0, BFloat16 is exact. For arbitrary float params, BFloat16 rounding of the boundary values can cause ≤1 ULP boundary shift.

**Wave 0 eval results:** 97.72% (257/263 tests passed). The 6 failures were on rank-0 tensor edge cases, not algorithmic errors.

---

## 2. SFPI Kernel Code

**File (Wormhole B0):** `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h`
**File (Blackhole):** `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_hardtanh.h`

Both files are **identical**.

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_hardtanh_(
    const int iterations,
    std::uint32_t param0,
    std::uint32_t param1,
    std::uint32_t param2)
{
    // All params are in FP16_B format
    // param0 = -(neg_threshold)    →  effectively = -min_val
    // param1 = -(pos_threshold - neg_threshold)  →  effectively = -(max_val - min_val)
    // param2 = -(pos_threshold)    →  effectively = max_val  [note: confusing naming; actual value IS max_val]

    sfpi::vFloat p0 = sfpi::s2vFloat16b(param0);
    sfpi::vFloat p1 = sfpi::s2vFloat16b(param1);
    sfpi::vFloat p2 = sfpi::s2vFloat16b(param2);
#pragma GCC unroll 0
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat val = sfpi::dst_reg[0];

        val += p0;              // shift: val = x - min_val
        v_if (val < 0.0f)
        {
            val = 0.0f;         // clamp lower: x < min_val → val = 0
        }
        v_endif;

        val += p1;              // shift: val -= (max_val - min_val)
        v_if (val >= 0.0f)
        {
            val = 0.0f;         // clamp upper: x > max_val → val = 0
        }
        v_endif;

        val += p2;              // restore: val += max_val → final result

        sfpi::dst_reg[0] = val;
        sfpi::dst_reg++;
    }
}
```

### Key SFPI Primitives Used

| Primitive | Purpose |
|-----------|---------|
| `sfpi::s2vFloat16b(paramN)` | Broadcast FP16_B scalar to SFPU vector register |
| `sfpi::dst_reg[0]` | Load/store current element from DST register file |
| `v_if (val < 0.0f)` | SFPU predicated conditional (lower clamp) |
| `v_if (val >= 0.0f)` | SFPU predicated conditional (upper clamp) |
| `v_endif` | End SFPU predicated block |
| `sfpi::dst_reg++` | Advance to next element |
| `#pragma GCC unroll 0` | Disable GCC loop unrolling (SFPU handles pipelining) |

### Loop Structure

- `iterations` is the outer loop count: typically `ITERATIONS = 8` (one face of a 32×32 tile = 8 rows of 4 elements each when processed 1 element at a time).
- Note the template parameter `int ITERATIONS` is **not used inside the loop body** — the actual loop bound comes from the `const int iterations` function argument. This is the standard hardtanh loop form.
- `#pragma GCC unroll 0` disables software unrolling since SFPU has hardware pipelining for sequential element processing.

---

## 3. Parameter Encoding

Parameters are packed as BFloat16 (`uint32_t` with FP16_B bit pattern in the low 16 bits) by the host at program compilation time and baked into `SFPU_OP_CHAIN_0`.

Given `min_val` (L) and `max_val` (H), the host computes:

```
param0_bits = float_to_bf16bits(-L)         // -min_val
param1_bits = float_to_bf16bits(L - H)      // -(max_val - min_val)  [= min_val - max_val]
param2_bits = float_to_bf16bits(H)          // max_val
```

For default L=-1.0, H=1.0:
- `p0` = BF16(-(-1.0)) = BF16(1.0) = `0x3F80`
- `p1` = BF16(-1.0 - 1.0) = BF16(-2.0) = `0xC000`
- `p2` = BF16(1.0) = `0x3F80`

The `s2vFloat16b(param)` call in the kernel broadcasts these stored FP16_B patterns to SFPU vector registers, making them available element-wise in the computation.

**Note on comment naming:** The kernel comment uses "neg_threshold" for `min_val` (negative number) and "pos_threshold" for **negative** `max_val`, which leads to the confusing "param2 = -(pos_threshold)" notation. In practice, `param2` holds `max_val` directly (a positive number for the default case).

---

## 4. LLK Wrapper (nuked — must be reconstructed)

The LLK wrapper file was removed by the operation nuke. The pattern to reconstruct:

**Target path (wormhole_b0):** `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardtanh.h`
**Target path (blackhole):** `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardtanh.h`

Expected structure (based on analogous ops like relu_min/relu_max which take scalar params):

```cpp
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_hardtanh_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::hardtanh, APPROXIMATE>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_hardtanh(
    uint dst_index,
    int vector_mode = (int)VectorMode::RC,
    uint32_t param0 = 0,
    uint32_t param1 = 0,
    uint32_t param2 = 0)
{
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::_calculate_hardtanh_<APPROXIMATE, ITERATIONS>,
        dst_index,
        vector_mode,
        param0,
        param1,
        param2);
}
```

- **SfpuType enum value:** `SfpuType::hardtanh` — must be added back to `llk_sfpu_types.h`
- The three FP16_B-encoded params (param0, param1, param2) flow from the SFPU_OP_CHAIN_0 string into this wrapper via the chain expansion.

---

## 5. Compute API Layer (nuked — must be reconstructed)

**Target path:** `tt_metal/hw/inc/api/compute/eltwise_unary/hardtanh.h`

Expected structure (based on the documented tile API signature `hardtanh_tile(uint32_t idst, uint32_t param0, uint32_t param1)` from the Doxygen RST):

```cpp
// The tile API may expose 2 or 3 params depending on implementation choice:
// - Docs show (idst, param0, param1) suggesting 2 user-visible params
// - SFPU kernel needs 3; the LLK may derive the 3rd from the 2 passed params
// OR the host precomputes all 3 and the tile function takes (idst, p0, p1, p2)

ALWI void hardtanh_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_hardtanh_init<APPROX>()));
}

ALWI void hardtanh_tile(uint32_t idst, uint32_t param0, uint32_t param1, uint32_t param2) {
    MATH((llk_math_eltwise_unary_sfpu_hardtanh<APPROX>(idst, (int)VectorMode::RC, param0, param1, param2)));
}
```

Included conditionally via `sfpu_split_includes.h` when `SFPU_OP_HARDTANH_INCLUDE` is defined.

**Note:** The Doxygen RST at `docs/source/tt-metalium/tt_metal/apis/kernel_apis/compute/hardtanh_tile.rst` documents:
```
.. doxygenfunction:: hardtanh_tile(uint32_t idst, uint32_t param0, uint32_t param1)
```
This shows only 2 uint32 params. The tile API likely takes 2 params (FP16_B of min_val and FP16_B of max_val), then the LLK wrapper computes the 3rd internally, OR the host precomputes all 3 and this is a doc discrepancy. Implementors should verify against the actual pre-nuke tile signature.

---

## 6. Dispatch / Registration Chain

### SfpuType Enum (must restore)

`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` — needs `hardtanh` added:
```cpp
enum class SfpuType {
    unused = 0,
    frac,
    swish,
    atanh,
    sinh,
    hardtanh,  // ← ADD THIS
};
```
Same change required for blackhole: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h`.

### SFPU Split Include Macro (must restore)

`tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` — add:
```cpp
#if SFPU_OP_HARDTANH_INCLUDE
#include "api/compute/eltwise_unary/hardtanh.h"
#endif
```

### UnaryOpType Enum

`ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` — **PRESENT** (not nuked):
```cpp
HARDTANH,  // line 115
```

### Parametrized Type Check

`ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp` — **PRESENT**:
```cpp
case UnaryOpType::HARDTANH: return true;
```

### Compile-Time Define (must restore)

`ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` — `get_macro_definition`:
```cpp
case UnaryOpType::HARDTANH: return "SFPU_OP_HARDTANH_INCLUDE";
```

### Kernel Init/Call Strings (must restore)

`ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` — `get_op_init_and_func_parameterized`:
```cpp
case UnaryOpType::HARDTANH: {
    float min_val = static_cast<float>(params[0]);
    float max_val = static_cast<float>(params[1]);
    // Precompute the three FP16_B-encoded SFPU params
    uint32_t p0 = float_to_bf16bits(-min_val);
    uint32_t p1 = float_to_bf16bits(min_val - max_val);
    uint32_t p2 = float_to_bf16bits(max_val);
    return {"hardtanh_tile_init();",
            fmt::format("hardtanh_tile({}, {}u, {}u, {}u);", idst, p0, p1, p2)};
}
```

### Compute Kernel Path

`ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` — `get_compute_kernel_path`:
```cpp
default: return "eltwise_sfpu.cpp";
```
HARDTANH uses the default `eltwise_sfpu.cpp` path (no special kernel file needed).

### TTNN C++ Registration

`ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` — **PRESENT** (inline function, not nuked):
```cpp
inline Tensor hardtanh(
    const Tensor& input_tensor,
    float min_val = -1.0f,
    float max_val = 1.0f,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return ttnn::detail::unary_impl(
        input_tensor,
        {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::HARDTANH, min_val, max_val}},
        memory_config,
        optional_output_tensor,
        sub_core_grids);
}
```

### Python Nanobind

`ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` — **PRESENT**:
```cpp
ttnn::bind_function<"hardtanh">(mod, doc.c_str(),
    &unary_two_float_5param_to_6param_wrapper<&ttnn::hardtanh>,
    nb::arg("input_tensor"),
    nb::kw_only(),
    nb::arg("min_val") = -1.0f,
    nb::arg("max_val") = 1.0f,
    nb::arg("memory_config") = nb::none(),
    nb::arg("output_tensor") = nb::none());
```

### Python Golden Function (must restore in `ttnn/ttnn/operations/unary.py`)

```python
def _golden_function_hardtanh(input_tensor_a, min_val=-1.0, max_val=1.0, *args, **kwargs):
    return torch.nn.functional.hardtanh(input_tensor_a, min_val, max_val)

ttnn.attach_golden_function(ttnn.hardtanh, golden_function=_golden_function_hardtanh)
```

---

## 7. CB Layout

Uses the standard `eltwise_sfpu.cpp` CB configuration (no intermediate buffer):

| CB Index | Role | Size |
|----------|------|------|
| `c_0` (src0) | Input tiles | `num_input_tiles × tile_size` |
| `c_2` (out0) | Output tiles | `num_output_tiles × tile_size` |

No `c_1` intermediate CB is needed (unlike hardshrink which uses a multi-pass binary approach).

The program factory (`unary_program_factory.cpp`) does **not** need a special case for HARDTANH in the CB creation path — the default CB setup is sufficient.

---

## 8. Supported dtypes

`BFLOAT16`, `BFLOAT8_B`, `FLOAT32`

---

## 9. Test Files

- Sweep test: `tests/sweep_framework/sweeps/eltwise/unary/hardtanh/hardtanh.py`
- Sweep test (sharded): `tests/sweep_framework/sweeps/eltwise/unary/hardtanh/hardtanh_sharded.py`
- Unit test: `tests/ttnn/unit_tests/operations/eltwise/test_activation.py::test_hardtanh`
- Backward test: `tests/ttnn/nightly/unit_tests/operations/eltwise/backward/test_backward_hardtanh.py`

Test uses default min_val=-1.0, max_val=1.0 with PCC ≥ 0.999 threshold.

---

## 10. Nuked vs. Present State

| Layer | File | State |
|-------|------|-------|
| SFPI kernel (wormhole_b0) | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h` | **PRESENT** |
| SFPI kernel (blackhole) | `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_hardtanh.h` | **PRESENT** |
| LLK wrapper (wormhole_b0) | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardtanh.h` | **NUKED** |
| LLK wrapper (blackhole) | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardtanh.h` | **NUKED** |
| `SfpuType::hardtanh` enum | `tt_metal/hw/ckernels/{wh,bh}/metal/llk_api/llk_sfpu_types.h` | **NUKED** |
| Compute API header | `tt_metal/hw/inc/api/compute/eltwise_unary/hardtanh.h` | **NUKED** |
| Split include guard | `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` | **NUKED** |
| `get_macro_definition` case | `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` | **NUKED** |
| `get_op_init_and_func_parameterized` case | `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` | **NUKED** |
| `UnaryOpType::HARDTANH` enum | `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` | **PRESENT** |
| `is_parametrized_type` check | `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp` | **PRESENT** |
| `ttnn::hardtanh` inline fn | `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` | **PRESENT** |
| Nanobind binding | `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` | **PRESENT** |
| Python golden function | `ttnn/ttnn/operations/unary.py` | **NUKED** |

---

## 11. Architecture Summary

```
Python: ttnn.hardtanh(tensor, min_val=-1.0, max_val=1.0)
    ↓
C++ TTNN: hardtanh() → unary_impl({HARDTANH, min_val, max_val})
    ↓
Dispatch: UnaryOpType::HARDTANH
    → get_macro_definition() → "SFPU_OP_HARDTANH_INCLUDE"
    → is_parametrized_type() → true
    → get_op_init_and_func_parameterized() with [min_val, max_val]
      → precomputes p0=bf16(-min_val), p1=bf16(min_val-max_val), p2=bf16(max_val)
      → generates: "hardtanh_tile_init();" + "hardtanh_tile(0, p0, p1, p2);"
    ↓
Compute kernel: eltwise_sfpu.cpp (default path)
    → SFPU_OP_CHAIN_0 expands to: hardtanh_tile_init(); ... hardtanh_tile(0, p0, p1, p2);
    ↓
Compute API: hardtanh_tile(idst, p0, p1, p2)  [hardtanh.h, TRISC_MATH guard]
    → included via sfpu_split_includes.h when SFPU_OP_HARDTANH_INCLUDE is defined
    ↓
LLK Wrapper: llk_math_eltwise_unary_sfpu_hardtanh<APPROX>(dst_index, VectorMode::RC, p0, p1, p2)
    → calls _llk_math_eltwise_unary_sfpu_params_<APPROX>(
            _calculate_hardtanh_<APPROX, 8>, dst_index, VectorMode::RC, p0, p1, p2)
    ↓
SFPI Kernel: _calculate_hardtanh_<APPROX, 8>(iterations=8, param0, param1, param2)
    — loads 3 FP16_B params via s2vFloat16b()
    — 8-element loop over dst_reg:
        val = x + p0              [val = x - min_val]
        if val < 0: val = 0       [clamp: x < min_val → zeroed]
        val += p1                 [val -= (max_val - min_val)]
        if val >= 0: val = 0      [clamp: x > max_val → zeroed]
        val += p2                 [val += max_val → final result]
    — result: clamp(x, min_val, max_val) for all x
```

Wormhole B0 and Blackhole share **identical** SFPI kernel implementations.
