# SFPU Kernel Analysis: `atanh`

## 1. Overview

**Operation**: `atanh` (inverse hyperbolic tangent)
**Math Definition**: `atanh(x) = 0.5 * ln((1+x)/(1-x))` for `|x| < 1`
**Type**: Unary element-wise SFPU operation (non-parameterized)
**Program Factory**: Uses the standard `UnaryProgramFactory` — no custom program factory
**Supported Dtypes**: BFLOAT16, BFLOAT8_B, FLOAT32

---

## 2. File Inventory

### Layer 1: SFPU Kernel (`ckernel_sfpu_atanh.h`)
- **Wormhole**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h`
- **Blackhole**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h`
- **Content**: Identical across both architectures (verbatim copy)

### Layer 2: LLK Math Wrapper (`llk_math_eltwise_unary_sfpu_atanh.h`)
- **Wormhole**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h`
- **Blackhole**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h`
- **Content**: Identical across both architectures

### Layer 3: Compute API (`atanh.h`)
- `tt_metal/hw/inc/api/compute/eltwise_unary/atanh.h`

### Layer 4: Split-Include Guard
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
- Macro: `SFPU_OP_ATANH_INCLUDE`

### Layer 5: SfpuType Enum
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h`
- Enum value: `SfpuType::atanh`

### Layer 6: UnaryOpType Enum
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` — `UnaryOpType::ATANH`

### Layer 7: Op Dispatch (`unary_op_utils.cpp`)
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
- Macro definition: `"SFPU_OP_ATANH_INCLUDE"`
- Init/func strings: `"atanh_tile_init();"` / `"atanh_tile({idst});"`

### Layer 8: C++ Registration (`unary.hpp`)
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`
- Registration macro: `REGISTER_UNARY_OPERATION(atanh, ATANH)`

### Layer 9: Python Nanobind (`unary_nanobind.cpp`)
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`
- Binding: `bind_unary_operation<"atanh", &ttnn::atanh>(...)`
- Docstring: `\mathrm{output\_tensor}_i = \text{atanh}(\mathrm{input\_tensor}_i)`
- Supported range note: `[supported range -1 to 1]`

### Layer 10: Golden Function
- `ttnn/ttnn/experimental_loader/golden_functions.py` — `torch.atanh(input_tensor)`

### Layer 11: Backward Operation
- `ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward.hpp` — `atanh_bw()`
- `ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward.cpp` — implements derivative via unary chain: `1/(1 - x^2)`
- `ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward_nanobind.cpp` — `bind_unary_backward_op<"atanh_bw">`
- `ttnn/ttnn/operations/unary_backward.py` — golden function registration

### Tests
- `tests/ttnn/unit_tests/operations/eltwise/test_atanh.py` — exhaustive bfloat16 bitpattern + fp32 test
- `tests/sweep_framework/sweeps/eltwise/unary/atanh/atanh.py` — sweep test
- `tests/sweep_framework/sweeps/eltwise/unary/atanh/atanh_sharded.py` — sharded sweep
- `tests/ttnn/nightly/unit_tests/operations/eltwise/backward/test_backward_atanh.py` — backward test

### LLK Test Infrastructure
- `tt_metal/third_party/tt_llk/tests/helpers/include/sfpu_operations.h` — dispatch via `_init_atanh_<>()` / `_calculate_atanh_<>()`
- `tt_metal/third_party/tt_llk/tests/helpers/include/llk_sfpu_types.h` — `SfpuType::atanh`
- `tt_metal/third_party/tt_llk/tests/python_tests/helpers/golden_generators.py` — `MathOperation.Atanh` golden

---

## 3. SFPU Kernel Deep Dive

### 3.1 Algorithm

The kernel computes `atanh(x) = 0.5 * (ln(1+x) - ln(1-x))` using an IEEE 754 floating-point decomposition approach for the logarithm.

**Logarithm strategy**: For any positive `y`:
1. Decompose `y = 2^e * m` where `m in [1, 2)` using `exexp` and `setexp`
2. Compute `ln(y) = e * ln(2) + P(m)` where `P(m)` is a **cubic minimax polynomial** for `ln(m)` on `[1, 2)`, evaluated in Horner form

The two logarithms (`ln(1+x)` and `ln(1-x)`) are computed independently with this same decomposition, then subtracted and halved.

### 3.2 SFPI Instructions Used

| SFPI Instruction | Usage | Count per iteration |
|---|---|---|
| `sfpi::dst_reg[0]` (load) | Read input tile element | 1 |
| `sfpi::dst_reg[0]` (store) | Write result tile element | 1 |
| `sfpi::dst_reg++` | Advance to next tile row | 1 |
| `sfpi::vConst1` | Literal `1.0f` for `1+x` and `1-x` | 2 |
| `sfpi::exexp(v)` | Extract biased exponent as int (IEEE 754 decomposition) | 2 |
| `sfpi::setexp(v, 127)` | Set exponent to 0 → normalize mantissa to `[1,2)` | 2 |
| `sfpi::int32_to_float(v, 0)` | Convert integer exponent to float for `e * ln(2)` | 2 |
| `sfpi::vConstFloatPrgm0` | Programmable constant register (c0 coefficient) | 2 reads |
| `sfpi::vConstFloatPrgm1` | Programmable constant register (c1 coefficient) | 2 reads |
| `sfpi::vConstFloatPrgm2` | Programmable constant register (c2 coefficient) | 2 reads |
| `vFloat * scalar + vFloat` | Fused multiply-add (Horner steps) | 6 (3 per ln) |
| `vFloat + vFloat` | Addition | 4 |
| `vFloat - vFloat` | Subtraction | 1 |
| `-vFloat + vConst` | Negation + add | 1 |
| `vFloat * scalar` | Scale by 0.5 | 1 |

**Total SFPI operations per tile element**: ~25 instructions (heavy, due to dual ln computation)

### 3.3 Init Function

```cpp
template <bool APPROXIMATION_MODE>
inline void atanh_init() {
    sfpi::vConstFloatPrgm0 = -0x1.952992p+0f;  // c0 ~ -1.5828
    sfpi::vConstFloatPrgm1 = 0x2.4f5388p+0f;   // c1 ~  2.3110
    sfpi::vConstFloatPrgm2 = -0xd.e712ap-4f;    // c2 ~ -0.8691
}
```

Loads 3 programmable constant registers with the cubic polynomial coefficients for `ln(m)` on `[1, 2)`. The 4th coefficient (`c3 = 0x2.44734p-4f ≈ 0.1416`) is a local `constexpr` in the compute loop.

**Note**: Does NOT use `APPROXIMATION_MODE` template parameter — single code path for both modes.

### 3.4 Compute Function

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_atanh() {
    constexpr float c3 = 0x2.44734p-4f;  // ~0.1416
    constexpr float ln2 = 0.6931471805599453f;

    #pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x = dst_reg[0];
        vFloat a = x + vConst1;          // 1 + x
        vFloat b = -x + vConst1;         // 1 - x

        // ln(1+x) via IEEE754 decomposition + cubic polynomial
        vInt ea = exexp(a);
        vFloat ma = setexp(a, 127);
        vFloat pa = ma * c3 + vConstFloatPrgm2;
        pa = pa * ma + vConstFloatPrgm1;
        pa = pa * ma + vConstFloatPrgm0;
        vFloat ln_a = int32_to_float(ea, 0) * ln2 + pa;

        // ln(1-x) via same decomposition
        vInt eb = exexp(b);
        vFloat mb = setexp(b, 127);
        vFloat pb = mb * c3 + vConstFloatPrgm2;
        pb = pb * mb + vConstFloatPrgm1;
        pb = pb * mb + vConstFloatPrgm0;
        vFloat ln_b = int32_to_float(eb, 0) * ln2 + pb;

        dst_reg[0] = (ln_a - ln_b) * 0.5f;
        dst_reg++;
    }
}
```

**Key observations**:
- Default `ITERATIONS = 8` (standard for processing 8 rows of a 32×32 tile per SFPU invocation)
- `#pragma GCC unroll 8` — fully unrolled for maximum throughput
- No branching or conditionals — pure dataflow
- `APPROXIMATION_MODE` is accepted but unused (no fast-path shortcut)
- Both `ln(1+x)` and `ln(1-x)` share identical polynomial coefficients loaded once in `atanh_init()`

### 3.5 Polynomial Coefficients

The cubic minimax polynomial `P(m) = c0 + m*(c1 + m*(c2 + m*c3))` approximates `ln(m)` for `m ∈ [1, 2)`:

| Coefficient | Hex Literal | Decimal Value |
|---|---|---|
| c0 (vConstFloatPrgm0) | `-0x1.952992p+0f` | ≈ -1.5828 |
| c1 (vConstFloatPrgm1) | `0x2.4f5388p+0f` | ≈ 2.3110 |
| c2 (vConstFloatPrgm2) | `-0xd.e712ap-4f` | ≈ -0.8691 |
| c3 (local constexpr) | `0x2.44734p-4f` | ≈ 0.1416 |

These are described as coming from "rpow scalar log2 precomputation." The cubic provides ~10-bit effective precision, which is adequate for bfloat16 (7-8 mantissa bits) but limits fp32 accuracy.

---

## 4. Abstraction Layer Call Chain

```
Python: ttnn.atanh(tensor)
  → C++ nanobind: bind_unary_operation<"atanh", &ttnn::atanh>
    → REGISTER_UNARY_OPERATION(atanh, ATANH)
      → UnaryProgramFactory (shared) with UnaryOpType::ATANH
        → unary_op_utils.cpp:
            get_macro_definition(ATANH) → "SFPU_OP_ATANH_INCLUDE"
            get_op_init_and_func_default(ATANH) →
              init: "atanh_tile_init();"
              func: "atanh_tile({idst});"
          → Compute kernel: eltwise_sfpu.cpp
              #define SFPU_OP_ATANH_INCLUDE → includes atanh.h
              SFPU_OP_CHAIN_0 expands to: atanh_tile_init(); atanh_tile(0);
            → atanh.h:
                atanh_tile_init() → llk_math_eltwise_unary_sfpu_atanh_init<APPROX>()
                  → llk_math_eltwise_unary_sfpu_init<SfpuType::atanh>(sfpu::atanh_init<APPROX>)
                atanh_tile(idst) → llk_math_eltwise_unary_sfpu_atanh<APPROX>(idst)
                  → _llk_math_eltwise_unary_sfpu_params_<APPROX>(sfpu::calculate_atanh<APPROX, 8>, idst, RC)
                    → ckernel_sfpu_atanh.h: calculate_atanh() — the actual SFPU kernel
```

---

## 5. Circular Buffer Usage

Standard unary pattern (via `UnaryProgramFactory`):
- **CB c_0** (input): Reader writes tiles here, compute reads from it
- **CB c_2** (output): Compute writes result tiles here, writer reads from it

No intermediate CBs needed — single in-place SFPU operation on DST register.

---

## 6. Numerical Precision & Known Issues

### 6.1 Catastrophic Cancellation Near Zero
For small `|x|`, `ln(1+x) ≈ ln(1-x)`, so the subtraction `ln(1+x) - ln(1-x)` suffers from catastrophic cancellation. The test acknowledges this:
> "the ln-based kernel has reduced precision due to catastrophic cancellation (computing ln(1+x) - ln(1-x) for small x subtracts near-equal values)"

### 6.2 Test Tolerances
From `tests/ttnn/unit_tests/operations/eltwise/test_atanh.py`:
- **bfloat16**: `rtol=1.6e-2, atol=1e-2`, ULP threshold = 4 (only for `|expected| > 0.25`)
- **fp32**: `rtol=1.6e-2, atol=2e-3` (wider relative tolerance due to cubic polynomial's limited precision)
- fp32 note: "ULP is too fine-grained for the cubic polynomial's ~10-bit effective precision"

### 6.3 Domain
- Valid for `|x| < 1` (strict inequality)
- Test masks out `|x| >= 1` and replaces with `0.0`
- Boundary behavior at `x = ±1` is `±∞` (handled by golden but not by the SFPU kernel which would compute `ln(0)`)

---

## 7. Distinguishing Characteristics

1. **No APPROXIMATION_MODE differentiation** — unlike many SFPU ops, there is only one code path regardless of the template parameter.

2. **Heavy instruction count** — the dual-logarithm approach requires ~25 SFPI operations per element, making this one of the more expensive unary SFPU kernels. Most unary ops use 5-15 instructions.

3. **Uses IEEE 754 decomposition** (`exexp`/`setexp`/`int32_to_float`) — this pattern is shared with other log-based operations. It decomposes floating-point values into exponent and mantissa, then uses polynomial approximation on the mantissa range `[1, 2)`.

4. **Three programmable constant registers** — occupies `vConstFloatPrgm0`, `vConstFloatPrgm1`, `vConstFloatPrgm2` for the cubic coefficients. Only `vConstFloatPrgm3` remains available if chaining.

5. **Custom split-include guard** — uses `SFPU_OP_ATANH_INCLUDE` (not the default `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE`), indicating it was added as a separate compilation unit to reduce compile time.

6. **Identical WH/BH implementations** — both Wormhole and Blackhole use byte-for-byte identical kernel code, with no architecture-specific optimizations.

7. **Has backward op** (`atanh_bw`) — derivative is `1/(1 - x^2)`, implemented as a unary chain: `SQUARE → SUB_UNARY_SFPU(1.0) → NEG → RECIP` applied to input, then multiplied by grad.

---

## 8. Comparison with Related Operations

| Aspect | atanh | asinh / acosh |
|---|---|---|
| Init function | `atanh_init` (standalone) | Shared `_init_inverse_hyperbolic_` |
| Uses `exexp`/`setexp` | Yes (for dual ln) | Typically uses `lut_log` or HW log |
| Programmable constants | 3 (c0, c1, c2) | Varies |
| Instruction count | ~25/elem | ~15-20/elem |
| Cancellation risk | Yes (near zero) | No |

The atanh implementation notably does NOT share the `_init_inverse_hyperbolic_` initialization that `asinh` and `acosh` use, instead having its own completely independent `atanh_init()` that sets up log polynomial coefficients.

---

## 9. Implementation Reproducing Guide

To re-implement `atanh` from scratch, you need:

1. **SFPU kernel** (`ckernel_sfpu_atanh.h`): The `calculate_atanh()` and `atanh_init()` functions in `ckernel::sfpu` namespace. 67 lines total.

2. **LLK wrapper** (`llk_math_eltwise_unary_sfpu_atanh.h`): Boilerplate connecting the ckernel to the LLK dispatch. 24 lines, standard template.

3. **Compute API** (`atanh.h`): `atanh_tile()` and `atanh_tile_init()` functions with Doxygen. 34 lines, standard template.

4. **Registration in 6 locations**:
   - `SfpuType::atanh` in `llk_sfpu_types.h` (both WH and BH)
   - `UnaryOpType::ATANH` in `unary_op_types.hpp`
   - Macro `SFPU_OP_ATANH_INCLUDE` in `sfpu_split_includes.h`
   - Dispatch in `unary_op_utils.cpp` (macro definition + init/func strings)
   - `REGISTER_UNARY_OPERATION(atanh, ATANH)` in `unary.hpp`
   - `bind_unary_operation<"atanh", &ttnn::atanh>` in `unary_nanobind.cpp`

5. **Golden function**: `torch.atanh` in `golden_functions.py`

6. **LLK test integration**: Case in `sfpu_operations.h` dispatching to `_init_atanh_<>()` / `_calculate_atanh_<>()`
