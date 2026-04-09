# SFPU Kernel Analysis: atanh

## 1. Operation Overview

**Math definition:** `atanh(x) = 0.5 * ln((1+x)/(1-x))`, valid for |x| < 1.

**Operation type:** Unary eltwise SFPU, no parameters, not parameterized.

**Supported data types:** BFLOAT16, BFLOAT8_B, FLOAT32.

**Approximation mode:** Template parameter `APPROXIMATION_MODE` is accepted but **not used** — the same code path runs regardless.

**Registration pattern:** `REGISTER_UNARY_OPERATION(atanh, ATANH)` — simple unary, no `fast_and_approximate_mode` flag, no float parameter.

---

## 2. Abstraction Layer Map

### Layer 1: UnaryOpType enum
- **File:** `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp:59`
- **Value:** `UnaryOpType::ATANH`

### Layer 2: SfpuType enum
- **File (WH):** `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
- **File (BH):** `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h`
- **Value:** `SfpuType::atanh`

### Layer 3: Host-side dispatch (unary_op_utils.cpp)
- **File:** `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
- **Macro define:** `SFPU_OP_ATANH_INCLUDE` (split-include guard, line 22)
- **Init string:** `"atanh_tile_init();"` (line 51)
- **Func string:** `"atanh_tile({idst});"` (line 51)
- **Approx mode:** `false` (falls through to default, line 75)
- **Compute kernel path:** `"eltwise_sfpu.cpp"` (falls through to default, line 169)

### Layer 4: C++ API registration (unary.hpp)
- **File:** `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp:136`
- **Macro:** `REGISTER_UNARY_OPERATION(atanh, ATANH)`
- **Signature:** `Tensor atanh(const Tensor&, optional<MemoryConfig>, optional<Tensor>, optional<CoreRangeSet>)`

### Layer 5: Python binding (unary_nanobind.cpp)
- **File:** `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp:1803`
- **Call:** `bind_unary_operation<"atanh", &ttnn::atanh>(...)`
- **Python API:** `ttnn.atanh(input_tensor)`

### Layer 6: Compute API header (tile-level API)
- **File:** `tt_metal/hw/inc/api/compute/eltwise_unary/atanh.h`
- **Functions:**
  - `atanh_tile(uint32_t idst)` — calls `llk_math_eltwise_unary_sfpu_atanh<APPROX>(idst)` guarded by `TRISC_MATH`
  - `atanh_tile_init()` — calls `llk_math_eltwise_unary_sfpu_atanh_init<APPROX>()`

### Layer 7: Split-include guard
- **File:** `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h:16-18`
- **Guard:** `#if SFPU_OP_ATANH_INCLUDE` includes `"api/compute/eltwise_unary/atanh.h"`

### Layer 8: LLK wrapper (per-arch)
- **File (WH):** `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h`
- **File (BH):** `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h`
- **Init function:** `llk_math_eltwise_unary_sfpu_atanh_init<APPROXIMATE>()` — calls `llk_math_eltwise_unary_sfpu_init<SfpuType::atanh, APPROXIMATE>(sfpu::atanh_init<APPROXIMATE>)`
- **Compute function:** `llk_math_eltwise_unary_sfpu_atanh<APPROXIMATE, ITERATIONS=8>(uint dst_index, int vector_mode=RC)` — calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_atanh<APPROXIMATE, ITERATIONS>, dst_index, vector_mode)`
- **Identical across WH and BH.**

### Layer 9: SFPU kernel (per-arch)
- **File (WH):** `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h`
- **File (BH):** `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h`
- **Identical across WH and BH.**

---

## 3. SFPU Kernel Deep Dive

### Namespace & Location
- **Namespace:** `ckernel::sfpu`
- **Functions:** `calculate_atanh<APPROXIMATION_MODE, ITERATIONS>()` and `atanh_init<APPROXIMATION_MODE>()`

### Algorithm
The kernel computes `atanh(x) = 0.5 * (ln(1+x) - ln(1-x))` using IEEE 754 float decomposition for the natural logarithm.

**Logarithm sub-algorithm** (used twice per element):
1. Extract exponent: `e = exexp(y)` — gets the biased IEEE 754 exponent
2. Normalize mantissa: `m = setexp(y, 127)` — forces exponent to 0, so `m ∈ [1, 2)`
3. Evaluate cubic minimax polynomial in Horner form: `P(m) = c0 + m*(c1 + m*(c2 + m*c3))`
4. Reconstruct: `ln(y) = e * ln(2) + P(m)`

**Final computation:**
```
result = (ln(1+x) - ln(1-x)) * 0.5
```

### Polynomial Coefficients
The cubic minimax polynomial approximates `ln(m)` for `m ∈ [1, 2)`:

| Coefficient | Hex literal | Approximate value | Storage |
|---|---|---|---|
| c0 | `-0x1.952992p+0f` | -1.5828 | `vConstFloatPrgm0` |
| c1 | `0x2.4f5388p+0f` | +2.3110 | `vConstFloatPrgm1` |
| c2 | `-0xd.e712ap-4f` | -0.8691 | `vConstFloatPrgm2` |
| c3 | `0x2.44734p-4f` | +0.1416 | local `constexpr` |

**Note:** c0, c1, c2 are loaded into programmable SFPU constant registers (`vConstFloatPrgm0/1/2`) during `atanh_init()`. c3 is a compile-time constant used directly in the loop. The constant `ln2 = 0.6931471805599453f` is also a local constexpr.

### SFPI Instructions Used

| Instruction / Intrinsic | Purpose | Count per iteration |
|---|---|---|
| `sfpi::dst_reg[0]` (read) | Load input tile element from DST | 1 |
| `sfpi::vConst1` | Hardware constant 1.0f | 2 (for `1+x` and `1-x`) |
| `sfpi::exexp(v)` | Extract IEEE 754 exponent as integer | 2 |
| `sfpi::setexp(v, 127)` | Set exponent to 127 (normalize mantissa to [1,2)) | 2 |
| `sfpi::int32_to_float(v, 0)` | Convert integer exponent to float | 2 |
| `sfpi::vConstFloatPrgm0/1/2` (read) | Read programmable constant registers | 6 (3 per ln) |
| `vFloat` multiply | SFPU multiply | 8 (4 per ln sub-evaluation) |
| `vFloat` add/subtract | SFPU add/sub | ~9 |
| `sfpi::dst_reg[0]` (write) | Store result back to DST | 1 |
| `sfpi::dst_reg++` | Advance to next tile element | 1 |

**Total SFPU operations per element:** ~34 (high instruction count due to dual logarithm evaluation).

### Loop Structure
```cpp
#pragma GCC unroll 8
for (int d = 0; d < ITERATIONS; d++) { ... }
```
- Default `ITERATIONS = 8` — processes 8 elements per call (one face of a 32x32 tile has 1024 elements; the LLK framework calls this function once per face row)
- The `#pragma GCC unroll 8` fully unrolls the loop at compile time

### Init Function
```cpp
template <bool APPROXIMATION_MODE>
inline void atanh_init() {
    sfpi::vConstFloatPrgm0 = -0x1.952992p+0f;  // c0
    sfpi::vConstFloatPrgm1 = 0x2.4f5388p+0f;   // c1
    sfpi::vConstFloatPrgm2 = -0xd.e712ap-4f;    // c2
}
```
- Uses all 3 programmable constant registers
- `APPROXIMATION_MODE` template parameter is accepted but unused
- Called once before the tile loop begins

---

## 4. Key Design Characteristics

### No Parameters
- `atanh` takes no runtime parameters — it is a pure unary function
- Not listed in `is_parametrized_type()`
- Dispatch goes through `get_op_init_and_func_default()`

### Split-Include Pattern
- Uses `SFPU_OP_ATANH_INCLUDE` guard (not the default `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE`)
- This means the atanh header is conditionally compiled only when needed, reducing kernel binary size

### Arch Portability
- WH and BH implementations are **byte-identical**
- Both architectures share the same ckernel and LLK wrapper files (content duplicated, not symlinked)

### Precision Characteristics
- The cubic polynomial for `ln(m)` provides approximately 10-bit effective precision
- For small |x|, catastrophic cancellation occurs when computing `ln(1+x) - ln(1-x)` since both values are close
- The test file uses relaxed tolerances: `rtol=1.6e-2, atol=2e-3` for fp32, `rtol=1.6e-2, atol=1e-2` for bfloat16
- ULP check (threshold=4) only applied for bfloat16 where `|expected| > 0.25` to avoid the small-value precision issues

### Domain Restrictions
- Valid only for `|x| < 1` (open interval)
- At boundaries (x = ±1), atanh diverges to ±infinity
- Test filters inputs to `|x| < 1.0` replacing out-of-domain values with 0.0

---

## 5. File Inventory

| Layer | File Path | Role |
|---|---|---|
| UnaryOpType | `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` | Enum value `ATANH` |
| SfpuType | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu_types.h` | Enum value `atanh` |
| Host dispatch | `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` | Macro define, init/func strings |
| C++ API | `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` | `REGISTER_UNARY_OPERATION(atanh, ATANH)` |
| Python bind | `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` | `bind_unary_operation<"atanh", &ttnn::atanh>` |
| Compute API | `tt_metal/hw/inc/api/compute/eltwise_unary/atanh.h` | `atanh_tile()` / `atanh_tile_init()` |
| Split include | `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` | `SFPU_OP_ATANH_INCLUDE` guard |
| LLK wrapper (WH) | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h` | LLK init + compute |
| LLK wrapper (BH) | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h` | LLK init + compute |
| SFPU kernel (WH) | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h` | Core algorithm |
| SFPU kernel (BH) | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h` | Core algorithm |
| Test | `tests/ttnn/unit_tests/operations/eltwise/test_atanh.py` | Exhaustive bfloat16 + fp32 test |
| Key notes | `docs/sfpu_operations/key_notes/atanh_key_notes.md` | Operation metadata |

---

## 6. Test Coverage

**Test file:** `tests/ttnn/unit_tests/operations/eltwise/test_atanh.py`

**Strategy:** Exhaustive bfloat16 bitpattern sweep (256x256 = 65536 values) with domain filtering.

**Key test details:**
- Generates all bfloat16 bitpatterns
- Filters to `|x| < 1.0`, replacing out-of-domain with 0.0
- Computes golden reference via `torch.atanh()` in float32
- Flushes subnormals to zero to match hardware behavior
- Tests both bfloat16 and float32 paths

**Tolerances:**
| Dtype | rtol | atol | ULP threshold |
|---|---|---|---|
| bfloat16 | 1.6e-2 | 1e-2 | 4 (only for |expected| > 0.25) |
| float32 | 1.6e-2 | 2e-3 | N/A |

---

## 7. Summary for Implementors

To implement an operation similar to `atanh`, follow this pattern:

1. **SFPU kernel** (`ckernel_sfpu_*.h`): Write `calculate_<op>()` with the computation loop and `<op>_init()` for constant setup. Use `vConstFloatPrgm0/1/2` for polynomial coefficients.
2. **LLK wrapper** (`llk_math_eltwise_unary_sfpu_*.h`): Template wrappers that connect ckernel to LLK framework via `llk_math_eltwise_unary_sfpu_init` and `_llk_math_eltwise_unary_sfpu_params_`.
3. **Compute API** (`atanh.h`): `*_tile()` and `*_tile_init()` functions, guarded by `#ifdef TRISC_MATH`.
4. **Split include** (`sfpu_split_includes.h`): Add `#if SFPU_OP_<NAME>_INCLUDE` guard.
5. **SfpuType** (`llk_sfpu_types.h`): Add enum value for both WH and BH.
6. **Host dispatch** (`unary_op_utils.cpp`): Add macro definition, init/func strings, optional approx mode.
7. **UnaryOpType** (`unary_op_types.hpp`): Add enum value.
8. **C++ API** (`unary.hpp`): Use appropriate `REGISTER_UNARY_OPERATION` macro.
9. **Python binding** (`unary_nanobind.cpp`): Call `bind_unary_operation`.
10. **Test**: Exhaustive bfloat16 bitpattern test with domain-appropriate filtering and tolerance selection.

The key SFPI intrinsics for float decomposition (`exexp`, `setexp`, `int32_to_float`) combined with Horner-form polynomial evaluation and programmable constant registers (`vConstFloatPrgm*`) are the foundational building blocks for transcendental function implementations on this hardware.
