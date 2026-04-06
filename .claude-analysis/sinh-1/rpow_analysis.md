# SFPU Kernel Analysis: rpow

## 1. Operation Identity

| Property | Value |
|----------|-------|
| **Operation Name** | rpow |
| **UnaryOpType Enum** | `RPOW` (in `unary_op_types.hpp:128`) |
| **Math Definition** | `rpow(x, base) = base^x` (scalar base raised to tensor element power) |
| **Parametrized** | Yes - takes `base` as a float parameter, passed as IEEE 754 bits (`uint32_t`) |
| **Hardware Targets** | Wormhole B0, Blackhole (identical kernel code) |

## 2. File Inventory

### SFPU Kernel (Core Algorithm)
| File | Path |
|------|------|
| **Wormhole B0** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rpow.h` |
| **Blackhole** | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rpow.h` |

### LLK Wrapper
| File | Path |
|------|------|
| **Wormhole B0** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rpow.h` |
| **Blackhole** | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rpow.h` |

### CKernel API Header
| File | Path |
|------|------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/rpow.h` |

### Compute Kernel (includes rpow.h)
| File | Path |
|------|------|
| **unary_ng** | `ttnn/cpp/ttnn/operations/eltwise/unary_ng/device/kernels/compute/eltwise_sfpu.cpp` |
| **unary (legacy)** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` |

### Registration & Binding
| File | Path |
|------|------|
| **UnaryOpType enum** | `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp:128` |
| **TTNN API registration** | `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp:168` |
| **Python nanobind** | `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp:1928` |

### LLK API Aggregator (includes the LLK wrapper)
| File | Path |
|------|------|
| **Wormhole B0** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_unary_sfpu_api.h:30` |
| **Blackhole** | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_unary_sfpu_api.h` |

## 3. Abstraction Layer Wiring

### Layer 1: SFPU Kernel Function
**File**: `ckernel_sfpu_rpow.h`
```
Namespace: ckernel::sfpu
Functions:
  - calculate_rpow<APPROXIMATION_MODE, ITERATIONS=8>(uint32_t base_val) -> void
  - rpow_init<APPROXIMATION_MODE>() -> void  [empty body]
```

### Layer 2: LLK Wrapper
**File**: `llk_math_eltwise_unary_sfpu_rpow.h`
```
Namespace: ckernel
Functions:
  - llk_math_eltwise_unary_sfpu_rpow_init<APPROXIMATE>()
      -> calls llk_math_eltwise_unary_sfpu_init<SfpuType::rpow, APPROXIMATE>(rpow_init<APPROXIMATE>)
  - llk_math_eltwise_unary_sfpu_rpow<APPROXIMATE, ITERATIONS=8>(dst_index, base_val, vector_mode=RC)
      -> calls _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(calculate_rpow<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, base_val)
```

**NOTE**: Uses `_llk_math_eltwise_unary_sfpu_params_` (the variadic-parameter variant), NOT `_llk_math_eltwise_unary_sfpu_`. This is because rpow passes an extra runtime argument (`base_val`) to the SFPU function.

### Layer 3: CKernel API Header
**File**: `tt_metal/hw/inc/api/compute/eltwise_unary/rpow.h`
```
Functions:
  - rpow_tile_init() -> MATH((llk_math_eltwise_unary_sfpu_rpow_init<APPROX>()))
  - rpow_tile(uint32_t idst, uint32_t base_val) -> MATH((llk_math_eltwise_unary_sfpu_rpow<APPROX>(idst, base_val)))
```

### Layer 4: Compute Kernel Dispatch
**File**: `eltwise_sfpu.cpp` (both unary and unary_ng)

The compute kernel `#include`s `rpow.h` and dispatches via `SFPU_OP_CHAIN_0` macro, which expands to calls like:
```
rpow_tile_init();
rpow_tile(0, <base_val_as_uint32>);
```

### Layer 5: SfpuType Enum
**File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu_types.h`

**BUG/ANOMALY**: `SfpuType::rpow` is referenced in `llk_math_eltwise_unary_sfpu_rpow.h` but is **NOT present** in the `SfpuType` enum:
```cpp
enum class SfpuType {
    unused = 0,
    hardsigmoid,
    hardtanh,
    hardswish,
    softshrink,
};
```
This means the LLK init call `llk_math_eltwise_unary_sfpu_init<SfpuType::rpow, APPROXIMATE>(...)` will **fail to compile** if instantiated. However, since `rpow_init()` is empty and `SfpuType` is only used as a non-type template parameter (not switched on at runtime), this may compile if the init path is optimized away or if the enum is extended by a not-yet-committed change.

### Layer 6: UnaryOpType Registration
**File**: `unary_op_types.hpp:128` - `RPOW` is in the enum.

### Layer 7: Op Utils Dispatch
**File**: `unary_op_utils.cpp`

**OBSERVATION**: RPOW is **NOT** handled in `get_op_init_and_func_parameterized()` or `get_op_init_and_func_default()`. The switch statements in these functions do not have a `case UnaryOpType::RPOW:` branch. This means RPOW cannot currently be dispatched through the standard `unary_op_utils.cpp` SFPU_OP_CHAIN define generation.

The `update_macro_defines()` and `get_macro_definition()` functions also have no RPOW-specific handling; they would fall through to `"SFPU_OP_COMPUTE_KERNEL_API_INCLUDE"` (the default).

### Layer 8: TTNN C++ API
**File**: `unary.hpp:168`
```cpp
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(rpow, RPOW)
```
This macro creates `ttnn::rpow(input_tensor, float_param, ...)` which constructs a `UnaryWithParam{UnaryOpType::RPOW, param}` and passes it to `unary_impl`.

### Layer 9: Python Nanobind
**File**: `unary_nanobind.cpp:1928`

Uses a custom binding function `bind_unary_composite_rpow` (defined at line 1716) with signature:
```python
ttnn.rpow(input_tensor, base, *, memory_config=None, output_tensor=None)
```
The base parameter is a positional float argument.

## 4. SFPU Algorithm Deep Dive

### Mathematical Basis
```
rpow(x) = base^x = 2^(x * log2(base))
```

The algorithm splits into two phases:
1. **Scalar precomputation** (runs once, on host RISC-V): Compute `log2(base)` from the IEEE 754 representation
2. **Vector computation** (runs per-tile on SFPU): Compute `2^(x * log2_base)` using the exp_21f algorithm

### Phase 1: log2(base) Precomputation (Scalar)

This runs **outside the SFPU iteration loop**, on the RISC-V math thread. It operates entirely on scalar floats:

1. **Decode base**: `Converter::as_float(base_val)` converts IEEE 754 bits back to float
2. **Extract IEEE 754 components**:
   - Extract exponent: `base_exp = ((base_bits >> 23) & 0xFF) - 127`
   - Normalize mantissa to [1,2): `mantissa_bits = (base_bits & 0x007FFFFF) | 0x3F800000`
3. **Polynomial approximation for log2(mantissa)**:
   - 3rd-order Horner form: `series = c0 + m*(c1 + m*(c2 + m*c3))`
   - Coefficients (hex float literals):
     - `c3 = 0x2.44734p-4` (~0.1416)
     - `c2 = -0xd.e712ap-4` (~-0.8689)
     - `c1 = 0x2.4f5388p+0` (~2.3103)
     - `c0 = -0x1.952992p+0` (~-1.5828)
   - Final: `log2_base = base_exp + series * inv_ln2` where `inv_ln2 = 1/ln(2) = 1.4426950408889634`

**Note**: These coefficients are described as matching `_sfpu_unary_power_21f_` from `ckernel_sfpu_unary_power.h`, indicating code reuse from the existing POWER operation's log2 computation.

### Phase 2: 2^z Vector Computation (SFPU, Per-Element)

Uses the **exp_21f algorithm** from Moroz et al. 2022 ("Simple Multiple Precision Algorithms for Exponential Functions"):

```
For each element x in the tile (ITERATIONS=8 means 8 face-rows):
  1. z = x * log2_base                          // Scale by precomputed log2(base)
  2. Clamp: if z < -127, z = -127               // Prevent underflow
  3. z = addexp(z, 23)                           // Multiply by 2^23
  4. z_int = float_to_int32(z + 0x3F800000)      // Add IEEE 754 bias and convert to int
  5. exponent = exexp(reinterpret(z_int))         // Extract integer exponent part
  6. mantissa = exman9(reinterpret(z_int))        // Extract 9-bit mantissa fraction

  // Polynomial refinement for 2^frac(z):
  7. d1 = 0.40196114e-7                           // Magic constant
  8. d2 = int_to_float(0xf94ee7 + mantissa)
  9. d3 = int_to_float(0x560e + mantissa)
  10. d2 = d1 * d2
  11. frac_result = float_to_int32(d2 * d3)

  // Reconstruct: result = mantissa_frac * 2^exponent
  12. result = setexp(frac_result, 127 + exponent)
```

### Phase 3: Special Cases (Conditional Branches)

The kernel handles three domains via scalar `if`/`else if` (compile-time branches based on `base_scalar`):

#### Case 1: base == 0
```
v_if (x > 0)  -> y = 0.0          // 0^positive = 0
v_if (x == 0) -> y = 1.0          // 0^0 = 1 (by convention)
v_if (x < 0)  -> y = +infinity    // 0^negative = infinity
```

#### Case 2: base < 0 (Negative base)
```
x_int = float_to_int16(x)         // Truncate x to integer
x_rounded = int32_to_float(x_int) // Convert back
y = setsgn(y, x_int << 31)        // If x_int is odd, negate result
v_if (x_rounded != x) -> y = NaN  // Non-integer exponent -> NaN
```

#### Case 3: base > 0 (Normal case)
No special handling needed; the exp_21f result is used directly.

### Final Output Conversion
```cpp
y = reinterpret<vFloat>(float_to_fp16b(y, 0));  // Round to bfloat16
dst_reg[0] = y;
dst_reg++;
```
This explicit bfloat16 rounding (`float_to_fp16b`) ensures deterministic rounding behavior regardless of the destination format accumulator setting.

## 5. SFPI Instructions Used

| Instruction | Usage | Purpose |
|-------------|-------|---------|
| `sfpi::dst_reg[0]` | Read/Write | Load input element, store result |
| `sfpi::dst_reg++` | Increment | Advance to next face-row |
| `sfpi::addexp(z, 23)` | Exponent manipulation | Multiply by 2^23 (bit-level scaling) |
| `_float_to_int32_positive_` | Type conversion | Float-to-int truncation (positive values) |
| `sfpi::exexp()` | Exponent extraction | Extract IEEE 754 exponent field |
| `sfpi::exman9()` | Mantissa extraction | Extract 9 MSBs of mantissa |
| `sfpi::int32_to_float()` | Type conversion | Integer to float conversion |
| `sfpi::setexp()` | Exponent manipulation | Set IEEE 754 exponent field |
| `sfpi::reinterpret<>` | Bitcast | Reinterpret vInt/vFloat without conversion |
| `sfpi::float_to_int16()` | Type conversion | Float-to-int16 truncation (for integer check) |
| `sfpi::setsgn()` | Sign manipulation | Set sign bit from integer LSB |
| `sfpi::float_to_fp16b()` | Format conversion | Round float32 to bfloat16 |
| `sfpi::vConst1` | Constant | Load 1.0f |
| `v_if`/`v_endif` | Predication | SIMD conditional execution |
| `Converter::as_float()` | Utility | Reinterpret uint32 as float (scalar) |

## 6. Template Parameters

| Parameter | Values | Effect |
|-----------|--------|--------|
| `APPROXIMATION_MODE` | `true`/`false` | Not used in current implementation (no conditional paths depend on it) |
| `ITERATIONS` | Default `8` | Number of face-rows processed per invocation (standard 32x32 tile = 8 iterations of 4 rows each) |

## 7. Init Function

```cpp
template <bool APPROXIMATION_MODE>
inline void rpow_init() {
    // No programmable constants needed - log2(base) is computed from the parameter
}
```

The init function is **empty** because rpow computes `log2(base)` inline from the runtime parameter rather than preloading any SFPU programmable constants (LREG registers). This is a deliberate design choice: since `base` varies per-call, there's nothing to preload.

## 8. Parameter Passing Mechanism

The `base` parameter flows through the stack as:

```
Python: ttnn.rpow(tensor, base=2.0)
  -> C++ TTNN API: UnaryWithParam{UnaryOpType::RPOW, 2.0f}
     -> SFPU_OP_CHAIN_0 define: "rpow_tile_init(); rpow_tile(0, 0x40000000u);"
        -> CKernel API: rpow_tile(idst, base_val)
           -> LLK: llk_math_eltwise_unary_sfpu_rpow<APPROX>(dst_index, base_val)
              -> _llk_math_eltwise_unary_sfpu_params_(calculate_rpow<APPROX>, dst_index, vector_mode, base_val)
                 -> SFPU kernel: calculate_rpow(base_val)
```

The float parameter is bit-cast to `uint32_t` via `std::bit_cast<uint32_t>(float_val)` at the host C++ level and passed through as `uint32_t` all the way down to the SFPU kernel, where `Converter::as_float(base_val)` converts it back.

## 9. Complexity & Performance Characteristics

### Instruction Count Estimate (per element, normal case)
- 1 multiply (`x * log2_base`)
- 1 comparison + conditional (`z < -127` clamp)
- 1 `addexp` (2^23 scaling)
- 1 add + `float_to_int32` (bias + convert)
- 1 `exexp` + 1 `exman9` (decompose)
- 2 `int32_to_float` (polynomial evaluation)
- 2 multiplies (Horner steps)
- 1 `float_to_int32` (reconstruct mantissa)
- 1 `setexp` (reconstruct)
- 1 `float_to_fp16b` (bfloat16 rounding)

**Estimated ~14 SFPU instructions per element** for the normal (base > 0) path.

### Special-case overhead
- **base == 0**: 3 additional `v_if` comparisons and assignments (compile-time branch, so no overhead when base != 0)
- **base < 0**: `float_to_int16`, `int32_to_float`, `setsgn`, 1 `v_if` comparison (compile-time branch)

Since the special-case branches are scalar `if`/`else if` (not `v_if`), only ONE branch is compiled into each kernel invocation. This is efficient because `base` is constant across all elements.

### Scalar Precomputation Overhead
The `log2(base)` computation (Horner polynomial, bit manipulation) runs **once per kernel invocation**, not per element. This is ~10 scalar operations that are amortized across all 1024 elements in a tile (8 iterations * 128 elements per face-row = 1024 elements).

## 10. Accuracy Considerations

1. **log2(base) approximation**: Uses a 3rd-order polynomial over [1,2) with `inv_ln2` scaling. This gives approximately 20-23 bits of precision for the logarithm.

2. **exp_21f algorithm**: The Moroz et al. approach provides approximately 16-20 bits of precision for the exponential, which is adequate for bfloat16 output (8 bits mantissa).

3. **bfloat16 rounding**: The explicit `float_to_fp16b(y, 0)` call at the end rounds to bfloat16 with round-to-nearest-even semantics, ensuring consistent output regardless of internal accumulator precision.

4. **Underflow clamping**: The `z < -127` clamp prevents the exp_21f algorithm from producing denormalized or invalid results, at the cost of clamping very small results to zero.

## 11. Known Issues / Anomalies

### Issue 1: Missing SfpuType::rpow Enum Entry
`SfpuType::rpow` is referenced in `llk_math_eltwise_unary_sfpu_rpow.h` line 15:
```cpp
llk_math_eltwise_unary_sfpu_init<SfpuType::rpow, APPROXIMATE>(...)
```
But the `SfpuType` enum in `llk_sfpu_types.h` does not contain `rpow`. This will cause a compile error if this code path is instantiated. The enum needs to be extended with `rpow` for the LLK wrapper to compile.

### Issue 2: Missing unary_op_utils.cpp Dispatch
`UnaryOpType::RPOW` is in the enum but has **no dispatch case** in `get_op_init_and_func_parameterized()` in `unary_op_utils.cpp`. The function will hit the `default: TT_THROW(...)` branch. This means the host-side SFPU_OP_CHAIN_0 define generation for RPOW is not wired up in the standard unary program factory path.

### Issue 3: Helper Function in Anonymous Namespace
The `float_to_bits()` helper function is defined in an anonymous namespace inside the ckernel_sfpu_rpow.h header. While functional, this could cause ODR issues if the header is included in multiple translation units. A better pattern would be using `std::bit_cast` (C++20) or a static inline function.

## 12. Key Patterns for Reuse

### Pattern: Scalar Precomputation Outside Loop
The rpow kernel demonstrates a valuable pattern for parametrized operations: compute derived constants (like `log2(base)`) from the runtime parameter ONCE before the tile iteration loop, then use the precomputed value as a vector constant inside the loop. This avoids redundant per-element computation.

### Pattern: exp_21f for 2^x
The core `2^z` computation using `addexp`, `exexp`, `exman9`, and Horner polynomial refinement is a reusable building block for any operation that needs exponential computation. This same algorithm appears in the POWER operation.

### Pattern: IEEE 754 Bit Manipulation for log2
The technique of extracting the IEEE 754 exponent and mantissa fields, then applying a polynomial correction, is a standard pattern for computing logarithms without dedicated log instructions.

### Pattern: bfloat16 Rounding at Output
The explicit `float_to_fp16b(y, 0)` call before writing to `dst_reg` ensures consistent bfloat16 precision regardless of the destination accumulator mode. This is a good practice for operations that may accumulate numerical error through multiple stages.

### Pattern: Parameter Passing via _llk_math_eltwise_unary_sfpu_params_
For parametrized operations, use the `_llk_math_eltwise_unary_sfpu_params_` dispatch (not `_llk_math_eltwise_unary_sfpu_`) to pass runtime arguments through to the SFPU kernel function. The LLK wrapper signature should accept the parameter and forward it.
