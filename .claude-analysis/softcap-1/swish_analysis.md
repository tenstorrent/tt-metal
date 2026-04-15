# SFPU Analysis: swish

**Operation:** swish
**Math:** `swish(x) = x * sigmoid(x) = x / (1 + exp(-x))`
**Equivalent PyTorch:** `torch.nn.functional.silu`

---

## 1. Math Definition and Approximation Strategy

Swish does not use any hardware exp or sigmoid primitives. Instead, it approximates `sigmoid(|x|)` with a **three-segment piecewise approach**, then reflects for negative inputs:

| Segment | Domain | Formula | Max error |
|---------|--------|---------|-----------|
| Poly (seg 0) | `|x| ≤ 2.5` | `0.5 + t*(0.2533 + t*(-0.01479 + t*(-0.00747)))` | ~0.007 (at t≈2.0) |
| Linear (seg 1) | `2.5 < |x| ≤ 5.0` | `0.0276*t + 0.855` | ~0.017 (at t≈4.0) |
| Saturate (seg 2) | `|x| > 5.0` | `1.0` | ~0.007 (at t=5.0) |

For `x < 0`: `sigmoid(x) = 1 - sigmoid(|x|)`.
Final: `swish(x) = x * sigmoid(x)`.

Overall max ULP error for bfloat16: **~4 ULP** (per kernel comment).
Test suite accepts **≤2 ULP** (bfloat16) and **≤3 ULP** (fp32).

---

## 2. SFPI Kernel Code

**File:** `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h`
(Identical file exists for Blackhole: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h`)

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_swish() {
    constexpr float c1 = 0.2533f;
    constexpr float c2 = -0.01479f;
    constexpr float c3 = -0.00747f;
    constexpr float lin_slope = 0.0276f;
    constexpr float lin_offset = 0.855f;
    constexpr float bp1 = 2.5f;
    constexpr float bp2 = 5.0f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // Polynomial sigmoid(|x|) for |x| <= 2.5
        sfpi::vFloat ax = sfpi::abs(x);
        sfpi::vFloat sig_pos = 0.5f + ax * (c1 + ax * (c2 + ax * c3));

        // Linear segment for 2.5 < |x| <= 5.0
        v_if(ax > bp1) { sig_pos = ax * lin_slope + lin_offset; }
        v_endif;

        // Saturate to 1.0 for |x| > 5.0
        v_if(ax > bp2) { sig_pos = sfpi::vConst1; }
        v_endif;

        // Reflect for negative x
        v_if(x < 0.0f) { sig_pos = sfpi::vConst1 - sig_pos; }
        v_endif;

        // Final result
        sfpi::dst_reg[0] = x * sig_pos;
        sfpi::dst_reg++;
    }
}
```

### Key SFPI primitives used
| Primitive | Purpose |
|-----------|---------|
| `sfpi::dst_reg[0]` | Load input from DST register |
| `sfpi::abs(x)` | Absolute value (SFPU SFPI intrinsic) |
| `v_if` / `v_endif` | SFPU predicated execution |
| `sfpi::vConst1` | Hardware constant 1.0 |
| `sfpi::dst_reg++` | Advance to next element |
| `#pragma GCC unroll 8` | Unroll loop for throughput |

### Loop structure
- `ITERATIONS = 8` (default): processes 8 elements per tile row per call.
- The `APPROXIMATION_MODE` template parameter is accepted but has **no branching on it** — both approximate and exact modes execute the same code path (the piecewise polynomial IS the approximation).

---

## 3. LLK Wrapper

**File (wormhole_b0):** `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h`
**File (blackhole):** `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h`

```cpp
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_swish_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::swish, APPROXIMATE>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_swish(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_swish<APPROXIMATE, ITERATIONS>, dst_index, vector_mode);
}
```

- **SfpuType enum value:** `SfpuType::swish` (defined in `llk_sfpu_types.h`).
- Default `vector_mode = VectorMode::RC` (row+column, processes full tile).

---

## 4. Compute API Layer

**File:** `tt_metal/hw/inc/api/compute/eltwise_unary/swish.h`

```cpp
ALWI void swish_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_swish<APPROX>(idst)));
}

ALWI void swish_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_swish_init<APPROX>()));
}
```

- Guarded by `#ifdef TRISC_MATH` — only compiled for the math RISC-V thread.
- Included conditionally via `sfpu_split_includes.h` when `SFPU_OP_SWISH_INCLUDE` is defined.

---

## 5. Dispatch / Registration Chain

### SfpuType enum
`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`:
```cpp
enum class SfpuType { unused = 0, frac, swish, atanh, sinh, };
```

### UnaryOpType enum
`ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`:
```cpp
SWISH,  // line 126
```

### Compile-time define
`ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`:
```cpp
case UnaryOpType::SWISH: return "SFPU_OP_SWISH_INCLUDE";
```

### Kernel init/call strings
```cpp
case UnaryOpType::SWISH:
    return {"swish_tile_init();", fmt::format("swish_tile({});", idst)};
```

### TTNN C++ registration
`ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` line 163:
```cpp
REGISTER_UNARY_OPERATION(swish, SWISH)
```

### Python nanobind
`ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` line 1824:
```cpp
bind_unary_operation<"swish", &ttnn::swish>(mod, R"doc(\text{swish}(x) = x \times \sigma(x))doc", "", R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
```

### Python golden function
`ttnn/ttnn/operations/unary.py`:
```python
def _golden_function_swish(input_tensor_a, *args, **kwargs):
    return torch.nn.functional.silu(input_tensor_a)

ttnn.attach_golden_function(ttnn.swish, golden_function=_golden_function_swish)
```

---

## 6. Supported dtypes

`BFLOAT16`, `BFLOAT8_B`, `FLOAT32`

---

## 7. Test File

`tests/ttnn/unit_tests/operations/eltwise/test_swish.py`

- Tests both `bfloat16` and `fp32` modes.
- Golden: `torch.nn.functional.silu`.
- ULP thresholds: **2 ULP** (bfloat16), **3 ULP** (fp32).
- Flushes subnormal values to zero on both reference and hardware sides.

---

## 8. Architecture Summary

```
Python: ttnn.swish(tensor)
    ↓
C++ TTNN: REGISTER_UNARY_OPERATION(swish, SWISH)
    ↓
Dispatch: UnaryOpType::SWISH → "SFPU_OP_SWISH_INCLUDE" + "swish_tile_init(); swish_tile(idst);"
    ↓
Compute API: swish_tile(idst) / swish_tile_init()  [swish.h, TRISC_MATH guard]
    ↓
LLK Wrapper: llk_math_eltwise_unary_sfpu_swish<APPROX>(dst_index)  [llk_math_eltwise_unary_sfpu_swish.h]
    ↓
SFPI Kernel: calculate_swish<APPROXIMATION_MODE, ITERATIONS=8>()  [ckernel_sfpu_swish.h]
             — piecewise polynomial sigmoid approximation, no exp/sigmoid HW primitives
             — 3-segment: poly [0,2.5], linear (2.5,5.0], saturate (5.0,∞)
             — x < 0: reflect via 1 - sigmoid(|x|)
             — Final: x * sigmoid(x)
```

Both Wormhole B0 and Blackhole share identical SFPI kernel implementations.
