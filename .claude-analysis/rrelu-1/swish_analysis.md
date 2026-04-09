# SFPU Kernel Analysis: swish

## 1. Operation Identity

| Field | Value |
|---|---|
| **Operation name** | swish |
| **Math definition** | `swish(x) = x * sigmoid(x) = x / (1 + exp(-x))` |
| **PyTorch equivalent** | `torch.nn.functional.silu(x)` |
| **UnaryOpType enum** | `UnaryOpType::SWISH` (in `unary_op_types.hpp` line 126) |
| **SfpuType enum** | `SfpuType::swish` (in `llk_sfpu_types.h` line 10) |
| **Parameterized** | No — not listed in `is_parametrized_type()` |
| **Approx mode** | Returns `false` from `get_op_approx_mode()` |

## 2. File Inventory

### Layer 1 — SFPU Kernel (ckernel)
| File | Path |
|---|---|
| **Wormhole B0** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h` |
| **Blackhole** | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h` |

Both files are **identical** — same piecewise sigmoid approximation, same constants.

### Layer 2 — LLK Math Wrapper
| File | Path |
|---|---|
| **Wormhole B0** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h` |
| **Blackhole** | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h` |

Both identical. Provides `llk_math_eltwise_unary_sfpu_swish_init<APPROXIMATE>()` and `llk_math_eltwise_unary_sfpu_swish<APPROXIMATE, ITERATIONS>(dst_index, vector_mode)`.

### Layer 3 — Compute API Header
| File | Path |
|---|---|
| **API header** | `tt_metal/hw/inc/api/compute/eltwise_unary/swish.h` |

Provides `swish_tile(idst)` and `swish_tile_init()`.

### Layer 4 — Split Include Gate
| File | Path |
|---|---|
| **Gate header** | `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` |

Guarded by `#if SFPU_OP_SWISH_INCLUDE`.

### Layer 5 — SfpuType Enum
| File | Path |
|---|---|
| **Wormhole B0** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` |
| **Blackhole** | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` |

Entry: `swish` (value 2 in the enum ordering).

### Layer 6 — UnaryOpType Enum
| File | Path |
|---|---|
| **Enum** | `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` |

Entry: `SWISH` (line 126).

### Layer 7 — Op Utils (dispatch)
| File | Path |
|---|---|
| **Utils** | `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` |

- `get_macro_definition(SWISH)` → `"SFPU_OP_SWISH_INCLUDE"`
- `get_op_init_and_func_default(SWISH)` → `{"swish_tile_init();", "swish_tile({idst});"}`
- `get_op_approx_mode(SWISH)` → `false` (falls to default)
- `get_compute_kernel_path(SWISH)` → `"eltwise_sfpu.cpp"` (falls to default)

### Layer 8 — unary_ng Utils
| File | Path |
|---|---|
| **NG utils** | `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp` |

Same dispatch: `SWISH` → `{"swish_tile_init();", "swish_tile({idst});"}`.

### Layer 9 — C++ API Registration
| File | Path |
|---|---|
| **Header** | `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` |

`REGISTER_UNARY_OPERATION(swish, SWISH)` at line 163. No parameters. Comment at line 205 confirms swish is a first-class SFPU operation.

### Layer 10 — Nanobind (Python binding)
| File | Path |
|---|---|
| **Binding** | `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` |

```cpp
bind_unary_operation<"swish", &ttnn::swish>(mod, ...)
```

Supported dtypes: BFLOAT16, BFLOAT8_B, FLOAT32.

### Layer 11 — Python Golden Function
| File | Path |
|---|---|
| **Golden** | `ttnn/ttnn/operations/unary.py` |

```python
def _golden_function_swish(input_tensor_a, *args, **kwargs):
    return torch.nn.functional.silu(input_tensor_a)

ttnn.attach_golden_function(ttnn.swish, golden_function=_golden_function_swish)
```

### Test File
| File | Path |
|---|---|
| **Test** | `tests/ttnn/unit_tests/operations/eltwise/test_swish.py` |

Exhaustive bfloat16 bitpattern test with ULP + allclose assertions.

## 3. SFPU Kernel Deep Dive

### Source
`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h`

### Function Signature
```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_swish()
```

### Algorithm
The kernel computes `swish(x) = x * sigmoid(x)` where sigmoid is approximated via a **piecewise hybrid approach** operating on `|x|`:

**Segment 0** — Polynomial (|x| <= 2.5):
```
sigmoid(|x|) ≈ 0.5 + |x| * (0.2533 + |x| * (-0.01479 + |x| * (-0.00747)))
```
A degree-3 polynomial fitted to minimize max error at sample points 0, 0.5, 1.0, 1.5, 2.0, 2.5. Max error ~0.007.

**Segment 1** — Linear (2.5 < |x| <= 5.0):
```
sigmoid(|x|) ≈ 0.0276 * |x| + 0.855
```
Max error ~0.017.

**Segment 2** — Saturation (|x| > 5.0):
```
sigmoid(|x|) = 1.0
```
Max error ~0.007 (at boundary).

**Negation symmetry**: `sigmoid(x) = 1 - sigmoid(|x|)` for x < 0.

**Final**: `swish(x) = x * sigmoid(x)`.

### SFPI Instructions Used

| Instruction / API | Purpose |
|---|---|
| `sfpi::dst_reg[0]` | Read input tile element from DST register |
| `sfpi::abs(x)` | Compute absolute value for symmetric sigmoid |
| `sfpi::vFloat` | Vector float register type |
| `sfpi::vConst1` | Constant 1.0f |
| `v_if / v_endif` | Predicated execution (3 branches for piecewise segments + 1 for sign) |
| `sfpi::dst_reg[0] = ...` | Write result back to DST |
| `sfpi::dst_reg++` | Advance to next tile row |
| Arithmetic: `+`, `*`, `-`, `>`, `<` | Standard SFPI vector operations |

### Key Characteristics

1. **No LUT usage** — Pure arithmetic with predicated branches; no lookup tables.
2. **No hardware exp/sigmoid** — The comment explicitly states these are unavailable; uses polynomial approximation instead.
3. **APPROXIMATION_MODE template parameter** — Present in the signature but **not used** in the function body. The same algorithm runs regardless.
4. **4 predicated branches** (`v_if`/`v_endif`) — One for each piecewise segment transition (>2.5, >5.0) and one for the sign flip (x < 0).
5. **Constants are `constexpr float`** — All 7 constants (c1, c2, c3, lin_slope, lin_offset, bp1, bp2) are compile-time constants.
6. **Horner's method** for the polynomial: `0.5 + ax * (c1 + ax * (c2 + ax * c3))` — 3 multiply-adds, numerically stable.
7. **Iteration count** — Default ITERATIONS=8, matching 8 rows per tile face, with `#pragma GCC unroll 8`.

### Error Characteristics
- Overall max ULP error for bfloat16: ~4 ULP (documented in kernel comments)
- Test assertions: ULP threshold of 2 for bfloat16, 3 for float32
- Allclose: rtol=1.6e-2, atol=1e-2 (bfloat16); rtol=1e-3, atol=1e-4 (float32)

## 4. Data Flow Summary

```
Input tensor (DRAM/L1)
    │
    ▼
Reader kernel → CB c_in0
    │
    ▼
Compute kernel (eltwise_sfpu.cpp):
  SFPU_OP_CHAIN_0 dispatches:
    swish_tile_init()  → llk_math_eltwise_unary_sfpu_swish_init<false>()
                        → llk_math_eltwise_unary_sfpu_init<SfpuType::swish, false>()
    swish_tile(idst)   → llk_math_eltwise_unary_sfpu_swish<false>(idst)
                        → _llk_math_eltwise_unary_sfpu_params_<false>(calculate_swish<false, 8>, idst, RC)
                        → calculate_swish<false, 8>()  ← the SFPU kernel
    │
    ▼
CB c_out0 → Writer kernel → Output tensor (DRAM/L1)
```

## 5. Registration Chain (Bottom-Up)

```
Python: ttnn.swish(tensor)
    │
    ▼
Nanobind: bind_unary_operation<"swish", &ttnn::swish>
    │
    ▼
C++ API: REGISTER_UNARY_OPERATION(swish, SWISH)  [unary.hpp:163]
    │
    ▼
UnaryOpType::SWISH  [unary_op_types.hpp:126]
    │
    ▼
Op Utils dispatch:
  get_macro_definition(SWISH) → "SFPU_OP_SWISH_INCLUDE"
  get_op_init_and_func_default(SWISH) → {"swish_tile_init();", "swish_tile({idst});"}
  get_op_approx_mode(SWISH) → false
  get_compute_kernel_path(SWISH) → "eltwise_sfpu.cpp"
    │
    ▼
Compute API: swish_tile(idst) / swish_tile_init()  [swish.h]
  (guarded by #if SFPU_OP_SWISH_INCLUDE in sfpu_split_includes.h)
    │
    ▼
LLK Wrapper: llk_math_eltwise_unary_sfpu_swish<APPROX>(dst_index)
  calls _llk_math_eltwise_unary_sfpu_params_<APPROX>(calculate_swish<...>, dst_index, RC)
    │
    ▼
SFPU Kernel: calculate_swish<false, 8>()  [ckernel_sfpu_swish.h]
  SfpuType::swish  [llk_sfpu_types.h]
```

## 6. Key Patterns for Reimplementation

1. **No parameters** — Swish is a purely unary op with no runtime parameters. Registered via `REGISTER_UNARY_OPERATION` (not `REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER`).

2. **Split include pattern** — Uses the `SFPU_OP_SWISH_INCLUDE` macro gate. Operations that use this pattern are listed in `sfpu_split_includes.h` and require a corresponding entry in `get_macro_definition()`.

3. **Identical kernels across architectures** — Wormhole B0 and Blackhole have byte-identical ckernel files. The SFPI abstraction layer handles hardware differences.

4. **Piecewise approximation strategy** — Rather than using hardware transcendentals, the kernel approximates sigmoid with polynomial + linear + saturation segments. This is a common pattern for operations that need sigmoid internally.

5. **Predicated execution for branches** — Uses `v_if`/`v_endif` for piecewise selection rather than any form of blending or masking.

6. **Golden function uses PyTorch SiLU** — `torch.nn.functional.silu(x)` is mathematically identical to swish(x) = x * sigmoid(x).

7. **Default compute kernel path** — Uses the standard `eltwise_sfpu.cpp` compute kernel (falls through to default in `get_compute_kernel_path`).
