# SFPU Kernel Analysis: atanh

## 1. Overview

**Operation**: `atanh` (inverse hyperbolic tangent)
**Math definition**: `atanh(x) = 0.5 * ln((1+x)/(1-x))`
**Valid domain**: `|x| < 1` (open interval)
**Parameters**: None (parameterless unary operation)
**Approximation mode**: Template parameter `APPROXIMATION_MODE` exists but is not used to change behavior — the same cubic polynomial path is taken regardless.

## 2. SFPU Kernel Function

**File (Wormhole B0)**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h`
**File (Blackhole)**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h`
**Namespace**: `ckernel::sfpu`
**Identical across architectures**: Yes — Wormhole B0 and Blackhole implementations are byte-identical.

### 2.1 Core Algorithm: `calculate_atanh<APPROXIMATION_MODE, ITERATIONS>()`

The kernel rewrites the formula to avoid division:
```
atanh(x) = 0.5 * (ln(1+x) - ln(1-x))
```

Each `ln(y)` is computed via IEEE 754 floating-point decomposition:
1. Decompose `y = 2^e * m` where `m in [1, 2)` using `sfpi::exexp()` and `sfpi::setexp()`
2. Compute `ln(y) = e * ln(2) + P(m)` where `P(m)` is a cubic minimax polynomial

#### SFPI Instructions Used

| Instruction | Purpose | Count per iteration |
|---|---|---|
| `sfpi::dst_reg[0]` (read) | Load input tile element from DST register | 1 |
| `sfpi::vConst1` | Hardware constant 1.0f for `1+x` and `1-x` | 2 uses |
| `sfpi::exexp(v)` | Extract biased exponent as integer (IEEE 754 decomposition) | 2 (once for `a`, once for `b`) |
| `sfpi::setexp(v, 127)` | Force exponent to 0 (bias 127), extracting mantissa in [1,2) | 2 |
| `sfpi::int32_to_float(v, 0)` | Convert integer exponent to float for `e * ln(2)` | 2 |
| `sfpi::vConstFloatPrgm0/1/2` | Programmable constant registers (polynomial coefficients) | 6 reads total (3 per ln) |
| `sfpi::dst_reg[0]` (write) | Store result back to DST register | 1 |
| `sfpi::dst_reg++` | Advance to next tile row | 1 |

#### Polynomial Coefficients (Horner form)

Loaded once in `atanh_init()` into programmable constant registers:

| Register | Value (hex) | Value (decimal) | Role |
|---|---|---|---|
| `vConstFloatPrgm0` | `-0x1.952992p+0f` | ~-1.5828 | c0 (constant term) |
| `vConstFloatPrgm1` | `0x2.4f5388p+0f` | ~2.3110 | c1 (linear term) |
| `vConstFloatPrgm2` | `-0xd.e712ap-4f` | ~-0.8691 | c2 (quadratic term) |
| local `c3` | `0x2.44734p-4f` | ~0.1416 | c3 (cubic term, not in programmable reg) |
| local `ln2` | `0.6931471805599453f` | ln(2) | Scaling factor for exponent contribution |

The polynomial `P(m) = c0 + m*(c1 + m*(c2 + m*c3))` is evaluated in Horner form for numerical stability:
```
pa = ma * c3 + c2
pa = pa * ma + c1
pa = pa * ma + c0
```

### 2.2 Init Function: `atanh_init<APPROXIMATION_MODE>()`

Loads the three polynomial coefficients into `vConstFloatPrgm0/1/2`. The `c3` coefficient and `ln2` constant are kept as local `constexpr` variables in `calculate_atanh` since all programmable constant registers are consumed by `c0`, `c1`, `c2`.

### 2.3 Iteration Structure

- **Template parameter**: `ITERATIONS = 8` (default)
- **Loop**: `for (int d = 0; d < ITERATIONS; d++)` with `#pragma GCC unroll 8`
- **Each iteration**: Processes one row of a 32x32 tile (32 elements in SIMD via vFloat)
- **Total**: 8 iterations x 32 SIMD lanes = 256 elements = 8 tile rows per call

## 3. LLK Wrapper Layer

**File (Wormhole B0)**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h`
**File (Blackhole)**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h`
**Identical across architectures**: Yes

Two functions in `namespace ckernel`:

```cpp
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_atanh_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::atanh, APPROXIMATE>(sfpu::atanh_init<APPROXIMATE>);
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_atanh(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_atanh<APPROXIMATE, ITERATIONS>, dst_index, vector_mode);
}
```

- The init function passes `SfpuType::atanh` and the `atanh_init` function pointer to the generic `llk_math_eltwise_unary_sfpu_init` template.
- The compute function passes `calculate_atanh` as a function pointer to `_llk_math_eltwise_unary_sfpu_params_`, which handles DST indexing and vector mode dispatch.

## 4. Compute API Layer

**File**: `tt_metal/hw/inc/api/compute/eltwise_unary/atanh.h`

```cpp
ALWI void atanh_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_atanh<APPROX>(idst))); }
ALWI void atanh_tile_init() { MATH((llk_math_eltwise_unary_sfpu_atanh_init<APPROX>())); }
```

- `ALWI` = always-inline attribute
- `MATH(...)` = compile guard ensuring this runs only on the math RISC-V thread (`TRISC_MATH`)
- `APPROX` = global compile-time constant controlling approximation mode
- Guarded by `#ifdef TRISC_MATH` in the include

### Split-Include Mechanism

**File**: `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`

The `atanh.h` header is conditionally included via:
```cpp
#if SFPU_OP_ATANH_INCLUDE
#include "api/compute/eltwise_unary/atanh.h"
#endif
```

The macro `SFPU_OP_ATANH_INCLUDE` is set to `"1"` by the host-side `update_macro_defines()` function when the operation type is `UnaryOpType::ATANH`. This split-include mechanism reduces compile time by only pulling in the SFPU kernel headers needed for the specific operation.

## 5. SfpuType Enum Registration

**File (Wormhole B0)**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
**File (Blackhole)**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h`

```cpp
enum class SfpuType {
    unused = 0,
    frac,
    swish,
    atanh,   // <-- registered here
    sinh,
};
```

Identical across both architectures. The enum value is used in the LLK init to tag the operation type.

## 6. Host-Side Dispatch (UnaryOpType and Op Utils)

### 6.1 UnaryOpType Enum

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`

`ATANH` is registered at line 59 in the `UnaryOpType` enum.

### 6.2 Op Utils — Init/Func Strings and Macro Defines

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`

Three dispatch points:

1. **`get_macro_definition()`** (line 22): Returns `"SFPU_OP_ATANH_INCLUDE"` for `UnaryOpType::ATANH`
2. **`get_op_init_and_func_default()`** (line 51): Returns `{"atanh_tile_init();", "atanh_tile({idst});"}` — the init and per-tile function call strings injected as compute kernel defines
3. **`get_op_approx_mode()`**: Falls through to `default: return false;` — atanh does not use approximation mode

### 6.3 Compute Kernel Path

`get_compute_kernel_path()` returns `"eltwise_sfpu.cpp"` (default path) for atanh. This is the standard unary SFPU compute kernel at `tt_metal/kernels/compute/eltwise_sfpu.cpp`.

## 7. Compute Kernel Entry Point

**File**: `tt_metal/kernels/compute/eltwise_sfpu.cpp`

Standard unary SFPU compute kernel. The `SFPU_OP_CHAIN_0` macro expands to:
```
atanh_tile_init(); atanh_tile(0);
```

The kernel flow per tile:
1. `tile_regs_acquire()` — acquire DST register
2. `cb_wait_front(c_0, 1)` — wait for input tile in CB c_0
3. `copy_tile(c_0, 0, 0)` — unpack tile from CB c_0 to DST[0]
4. `SFPU_OP_CHAIN_0` — execute `atanh_tile_init()` + `atanh_tile(0)` on DST[0]
5. `tile_regs_commit()` / `tile_regs_wait()` — synchronize
6. `pack_tile(0, c_16)` — pack result from DST[0] to CB c_16
7. `cb_pop_front(c_0, 1)` — release input tile

Circular buffers used: **c_0** (input) and **c_16** (output).

## 8. C++ API Registration

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` (line 136)

```cpp
REGISTER_UNARY_OPERATION(atanh, ATANH)
```

This macro generates:
```cpp
inline Tensor atanh(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
```

No parameters — simple unary signature.

## 9. Python Binding

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` (line 1803)

```cpp
bind_unary_operation<"atanh", &ttnn::atanh>(
    mod,
    R"doc(\mathrm{output\_tensor}_i = \text{atanh}(\mathrm{input\_tensor}_i))doc",
    "[supported range -1 to 1]",
    R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
```

Supported dtypes: BFLOAT16, BFLOAT8_B, FLOAT32.

## 10. Golden Function

**File**: `ttnn/ttnn/experimental_loader/golden_functions.py` (line 142)

```python
def _atanh_golden_function(input_tensor, *args, **kwargs):
    import torch
    return torch.atanh(input_tensor)

if hasattr(ttnn, "atanh"):
    ttnn.attach_golden_function(ttnn.atanh, _atanh_golden_function)
```

## 11. Test Coverage

### Unit Test
**File**: `tests/ttnn/unit_tests/operations/eltwise/test_atanh.py`

- Tests both bfloat16 and fp32 paths
- Exhaustive: generates all 65536 bfloat16 bit patterns via `generate_all_bfloat16_bitpatterns()`
- Filters to valid domain `|x| < 1`, replacing out-of-domain values with 0.0
- Flushes subnormals to zero to match hardware behavior
- **Tolerances**:
  - fp32: `rtol=1.6e-2, atol=2e-3`
  - bfloat16: `rtol=1.6e-2, atol=1e-2`
  - ULP check (bfloat16 only, `|expected| > 0.25`): threshold = 4 ULP
- Notes catastrophic cancellation for small `x` due to `ln(1+x) - ln(1-x)` subtracting near-equal values

### Backward Test
**File**: `tests/ttnn/nightly/unit_tests/operations/eltwise/backward/test_backward_atanh.py`

### Sweep Tests
**Files**: `tests/sweep_framework/sweeps/eltwise/unary/atanh/atanh.py`, `atanh_sharded.py`

### LLK Test Infrastructure
- `tt_metal/third_party/tt_llk/tests/python_tests/helpers/llk_params.py`: `Atanh = OpSpec("atanh", MathOpType.SFPU_UNARY)`
- `tt_metal/third_party/tt_llk/tests/python_tests/helpers/golden_generators.py`: Python golden using `math.atanh(x)` with boundary handling for x = +/-1
- `tt_metal/third_party/tt_llk/tests/helpers/include/sfpu_operations.h`: C++ LLK dispatch case for `SfpuType::atanh`
- `tt_metal/third_party/tt_llk/tests/helpers/include/llk_sfpu_types.h`: `atanh` registered in LLK SfpuType enum

## 12. Backward Operation

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward.cpp` (line 949)

`atanh_bw()` computes `grad / (1 - x^2)` which is the derivative of atanh. This is a composite operation using existing ops (square, tanh, multiply), not a separate SFPU kernel.

## 13. Numerical Considerations

### Precision Characteristics
- The cubic minimax polynomial for `ln(m)` on `[1, 2)` provides approximately **10-bit effective precision** (~2-3 decimal digits)
- This is sufficient for bfloat16 (7-bit mantissa) but limited for fp32 (23-bit mantissa)
- The fp32 path uses the same polynomial, so fp32 results are not fp32-accurate — they're bfloat16-quality results stored in fp32 format

### Known Weakness: Catastrophic Cancellation Near Zero
For small `|x|`, both `ln(1+x)` and `ln(1-x)` are close to zero, so their difference `ln(1+x) - ln(1-x)` suffers from catastrophic cancellation. The test file explicitly documents this:
> "the ln-based kernel has reduced precision due to catastrophic cancellation (computing ln(1+x) - ln(1-x) for small x subtracts near-equal values)"

This is why the ULP check is restricted to `|expected| > 0.25`.

### Boundary Behavior
- `x = +/-1`: produces `+/-inf` (division by zero in `ln(1-x)` or `ln(1+x)` when denominator argument hits 0)
- `|x| > 1`: undefined (produces NaN or garbage since `1-x` or `1+x` becomes negative, and `ln` of negative is undefined)

## 14. Instruction Count Estimate (Per Tile Row)

| Category | Count | Instructions |
|---|---|---|
| Load/store | 2 | `dst_reg` read + write |
| Arithmetic (add/sub/mul) | ~14 | 2 additions for a,b; 2x(3 mul + 3 add) for Horner polynomials; 2 mul for `e*ln2`; 2 add for ln terms; 1 sub; 1 mul by 0.5 |
| IEEE 754 decomposition | 4 | 2x `exexp`, 2x `setexp` |
| Int-to-float conversion | 2 | 2x `int32_to_float` |
| Register advance | 1 | `dst_reg++` |
| **Total per row** | **~23** | |
| **Total per tile (8 rows)** | **~184** | |

## 15. Complete File Manifest

### New files (SFPU kernel specific):
| File | Layer | Purpose |
|---|---|---|
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h` | SFPU kernel | Core `calculate_atanh` + `atanh_init` |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h` | SFPU kernel | Identical Blackhole copy |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h` | LLK wrapper | LLK init + compute wrappers |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h` | LLK wrapper | Identical Blackhole copy |
| `tt_metal/hw/inc/api/compute/eltwise_unary/atanh.h` | Compute API | `atanh_tile()` + `atanh_tile_init()` |

### Modified files (atanh wired into existing infrastructure):
| File | What was added |
|---|---|
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` | `atanh` enum value in `SfpuType` |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` | `atanh` enum value in `SfpuType` |
| `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` | `SFPU_OP_ATANH_INCLUDE` guard + include |
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` | `ATANH` in `UnaryOpType` enum |
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` | Dispatch cases for macro define, init/func strings |
| `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` | `REGISTER_UNARY_OPERATION(atanh, ATANH)` |
| `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` | Python binding via `bind_unary_operation` |
| `ttnn/ttnn/experimental_loader/golden_functions.py` | Golden function `torch.atanh` |

## 16. Key Patterns for Reimplementation

1. **No parameters**: Simplest unary pattern — `REGISTER_UNARY_OPERATION` macro, no `get_op_init_and_func_parameterized` branch needed.
2. **Split-include**: Uses `SFPU_OP_ATANH_INCLUDE` guard to conditionally include the compute API header. Must be wired in both `sfpu_split_includes.h` and `get_macro_definition()`.
3. **Programmable constants**: Uses all three `vConstFloatPrgm0/1/2` registers for polynomial coefficients loaded in `atanh_init()`.
4. **IEEE 754 decomposition pattern**: `exexp` + `setexp` + `int32_to_float` is a reusable pattern for any ln-based SFPU kernel.
5. **Architecture parity**: Wormhole B0 and Blackhole files are identical — copy-paste is the pattern.
6. **Approx mode unused**: The `APPROXIMATION_MODE` template parameter exists for interface compatibility but doesn't change behavior.
