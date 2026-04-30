# Unary SFPU Omnibus Investigation Report — Phase 1

## Scope

Covers ~90 unary SFPU operations across 5 groups:
- **Activations** (27): relu, gelu, sigmoid, silu, softplus, threshold, leaky_relu, hardmish, prelu, hardshrink, softshrink, celu, hardtanh, selu, softsign, xielu
- **Math** (25): exp, log, sqrt, rsqrt, cbrt, reciprocal, abs, sign, trig (sin, cos, tan, asin/acos/atan, sinh/cosh/atanh), special (erf, erfc, erfinv, i0, i1, lgamma, digamma, polygamma)
- **Trig** (11): sin, cos, tan, asin, acos, atan, sinh, cosh, atanh, asinh, acosh
- **Rounding** (4): floor, ceil, round, trunc
- **Bitwise Unary** (1): bitwise_not (binary go to binary group)
- **Predicates** (11): isnan, isinf, isfinite, eqz, nez, ltz, gtz, lez, gez, logical_not, comp variants
- **Scalar** (8): unary_add, unary_sub, unary_mul, unary_div, unary_max, unary_min, rpow, rsub

Common shape: `op_tile_init()` / `op_tile(uint32_t dst_idx, ...)`.

---

## Dimension 1: Device-Side LLK Signatures

### Common Wrapper Pattern

All unary SFPU ops route through `_llk_math_eltwise_unary_sfpu_params_()` macro:

```cpp
template <typename Callable, typename... Args>
inline void _llk_math_eltwise_unary_sfpu_params_(
    Callable&& sfpu_func,
    uint32_t dst_index,
    int vector_mode = (int)VectorMode::RC,
    Args&&... args);
```

Wrapper:
1. Validates `dst_index < max_dest_tiles()`.
2. Acquires DEST via `_llk_math_eltwise_unary_sfpu_start_()`.
3. Iterates faces (R=2, C=2, RC=4) calling SFPU function with forwarded args.
4. Releases DEST via `_llk_math_eltwise_unary_sfpu_done_()`.

`vector_mode` is runtime; loop unrolling count compile-time (faces fixed per mode).

### Signature Families

#### Family 1: Plain (No Approx, No Scalar)
`op_init()` / `op(uint dst_idx, int vector_mode)`. ~19 ops.

Members: bitwise_not, abs, sign, sqrt, cbrt, exp2, expm1, sin, cos, tan, asin, acos, atan, sinh, cosh, asinh, acosh, atanh, erf, erfc, erfinv, i0, i1, sigmoid, silu, relu, relu6, tiled_prod, selu_tile

```cpp
inline void llk_math_eltwise_unary_sfpu_abs_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::abs>();
}
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_abs(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::calculate_abs<APPROXIMATE>, dst_index, vector_mode);
}
```

#### Family 2: With Approx (+ optional Legacy/FP32_DEST_ACC)
~10 ops: exp, gelu, log, log1p, log2, log10, tanh, rsqrt, recip, erfinv

```cpp
template <bool APPROXIMATE, bool FAST_APPROX, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_log_init() {...}
template <bool APPROXIMATE, bool FAST_APPROX, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_log(uint dst_index, int vector_mode = (int)VectorMode::RC) {...}
```

**Variant log_with_base**: extra runtime `base_scale` (uint32_t packed float).

**RSQRT init/exec template mismatch**:
```cpp
template <bool APPROXIMATE, bool legacy_compat>
inline void llk_math_eltwise_unary_sfpu_rsqrt_init() {...}
template <bool APPROXIMATE, bool fp32_dest_acc_en, bool FAST_APPROX, bool legacy_compat>
inline void llk_math_eltwise_unary_sfpu_rsqrt(uint dst_index, int vector_mode) {...}
```
Init has 2 templates, exec has 4. Helper must reconcile.

#### Family 3: With Scalar (Runtime Packed Uint32)
~18 ops: power, leaky_relu, elu, prelu, rpow, rsub, relu_max, relu_min, unary_max/min, fill, heaviside, round, softplus, xielu, hardmish, hardtanh, selu, celu, threshold, clamp

```cpp
template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_power(
    uint dst_index, uint32_t exponent = 0, int vector_mode = (int)VectorMode::RC) {...}
```

**Clamp (two scalars)**:
```cpp
template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_clamp(
    uint dst_index, uint min_val, uint max_val, int vector_mode = (int)VectorMode::RC);
```

#### Family 4: Scalar Binops
`binop_with_scalar` family: unary_add, unary_sub, unary_mul, unary_div.

```cpp
template <bool APPROXIMATE, int binop_mode>
inline void llk_math_eltwise_unary_sfpu_binop_with_scalar(
    uint dst_index, uint32_t scalar, int vector_mode = VectorMode::RC);
```

`binop_mode` enum: ADD=0, SUB=1, MUL=2, DIV=3.

#### Family 5: Mask (Data-Format Dependent)
```cpp
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_mask(
    uint dst_index, DataFormat data_format, int vector_mode);
```

Per lessons §1.4: hardcoded reads from `DataSlot + 1`. Encode in struct.

#### Family 6: Multi-Param Activations
celu, selu, hardtanh, threshold, softplus, xielu — multiple scalar params.

#### Family 7: Predicates
eqz, nez, ltz, gtz, lez, gez, isnan, isinf, isfinite, logical_not + int32 variants. Output 0.0/1.0 fp.

#### Family 8: Non-Trivial Init (LUT-Dependent)
Programs SFPU LUT/state during init: exp, log, tanh, erf/erfc/erfinv, sigmoid, silu, gelu, hardmish, i0, i1, lgamma, digamma, polygamma, sqrt, rsqrt.

**Implication**: chained inits may clobber each other's LUTs. Hoisting unsafe unless declared `clobbers_sfpu_lut = false`.

### Init State Compatibility

| Op A | Op B | Compatible? | Reason |
|---|---|---|---|
| exp_init | log_init | No | Both load polynomial LUT |
| abs_init | sign_init | Yes | Both trivial |
| sigmoid_init | silu_init | No | Both load activation LUT |
| sqrt_init | rsqrt_init | No (conservative) | Shared inverse-sqrt assist |
| relu_init | abs_init | Yes | Trivial |
| power_init | exp_init | No (conservative) | Different LUT regions, risky |
| Plain | LUT op | No | LUT clobbers |

**Safe chains**: CopyTile + abs + sign; CopyTile + relu + relu6; CopyTile + predicates; CopyTile + bitwise.
**Unsafe chains**: any two from {exp, log, tanh, sigmoid, silu, gelu, rsqrt}.

### DEST Batching Limits

- Max tiles per DEST batch: limited by DEST_AUTO_LIMIT (8 default).
- FP32 accumulation: not required for unary (write directly).
- Vector mode RC = 4 face iters; R or C = 2 face iters.
- Per-tile init cost: non-trivial for LUT ops (init per tile required if hoisting unsafe).

---

## Dimension 2: Host-Side Code Generation

### Parameter Encoding (`unary_op_utils.cpp`)

Scalars packed via `std::bit_cast<uint32_t>(param)`:
- Float: `std::bit_cast<uint32_t>(scalar)`
- Int32: `std::bit_cast<uint32_t>(static_cast<int32_t>(p))`
- Reciprocal pre-computed on host for division-like ops

| Op | Host Type | Packing | Kernel Receives |
|---|---|---|---|
| power | float | bit_cast | uint32_t exponent |
| leaky_relu | float | bit_cast | uint32_t scalar |
| relu_max int32 | int32 | bit_cast | uint32_t scalar |
| unary_add | float | bit_cast | uint32_t scalar |
| binop_with_scalar | varies | bit_cast | uint32_t scalar |
| clamp | float×2 | bit_cast | uint min, uint max |
| round | enum | (int) | int rounding_mode |

### Code Generation Layout

Parametrized: `get_op_init_and_func_parameterized(...)` emits init + exec strings per op type.

Default (non-param): `get_op_init_and_func_default(...)` — abs, sign, sqrt, sin, cos, etc.

### Compile-Time Feature Matrix

| Flag | Affects | Classification | Becomes |
|---|---|---|---|
| APPROXIMATE_MODE | LUT selection | Loop-internal | Template `Approx` |
| FP32_DEST_ACC_EN | Accum register width | Loop-internal | Template `Fp32DestAccEn` |
| FAST_APPROX | Secondary approx | Loop-internal | Template `FastApprox` |
| LEGACY_COMPAT | rsqrt back-compat | Loop-internal | Template `Legacy` |
| VECTOR_MODE (R/C/RC) | Face iters | Loop-internal | Template `VectorMode` enum |
| ITERATIONS | SFPU loop unroll | Loop-internal | Template int `ITERATIONS` |
| SFPU_OP_*_INCLUDE | Macro-injection dispatch | Configuration | eltwise_chain composition |

### Init/Exec Template Pairing

| Ops | Init Templates | Exec Templates | Rule |
|---|---|---|---|
| Plain | None | `<APPROX>` | Single template, init zero |
| exp, log, tanh | `<A,FA,FP32>` | `<A,FA,FP32>` | 3-param match |
| rsqrt | `<A,L>` | `<A,FP32,FA,L>` | **MISMATCH** |
| clamp, celu | generic | `<A,ITER>` | Init generic, exec specific |
| power | init_func arg | `<A,FP32>` | Init via function param |

Helper must enforce compatible templates or document the mismatch.

---

## Dimension 3: Usage Patterns

Unary SFPU invoked in:
1. Macro-dispatched `eltwise_sfpu.cpp` (out of scope per lessons §11.1)
2. Dedicated kernels (lgamma, mish, tanhshrink, logit, hardswish, logsigmoid, where_tss, clamp_tss)
3. Eltwise chain kernels (future migration target)

Observed param patterns:
- Approx modes: compile-time template
- Scalars: runtime float bitcast to uint32_t
- Vector modes: mostly template, sometimes runtime
- DEST slots: runtime (typically 0-3)
- Output dtypes: same as input (predicates: fp32 0/1)

---

## Dimension 4: Encapsulation Analysis

### Signature Shape Clusters (8 total)

| Family | Init | Exec | Templates | Runtime | Examples | Init Cost |
|---|---|---|---|---|---|---|
| **S1 Plain** | `op_init()` | `op(dst)` | `<APPROX>` | dst, vector_mode | abs, sin, sqrt | Trivial |
| **S2 Approx** | `op_init<A,F,FP32>()` | `op<A,F,FP32>(dst)` | `<APPROX, FAST_APPROX, FP32_DEST_ACC>` | dst, vector_mode | exp, log, tanh | LUT-dependent |
| **S3 Scalar** | `op_init()` | `op<APPROX>(dst, scalar)` | `<APPROX>` | dst, scalar, vector_mode | power, clamp, leaky_relu | Minimal |
| **S4 Scalar Binop** | `op_init()` | `op<APPROX, MODE>(dst, scalar)` | `<APPROX, BINOP_MODE>` | dst, scalar, vector_mode | unary_add/sub/mul/div | Trivial |
| **S5 Mask** | `op_init()` | `op<APPROX>(dst, fmt)` | `<APPROX>` | dst, data_format, vector_mode | mask | Minimal |
| **S6 Predicates** | `op_init()` | `op<APPROX>(dst[, param])` | `<APPROX>` | dst, [comparand], vector_mode | eqz, isnan, logical_not | Trivial |
| **S7 Bitwise** | `op_init()` | `op(dst, mask)` | None | dst, mask, vector_mode | bitwise_not, shifts | Trivial |
| **S8 Init/Exec Template Mismatch** | `init<A,L>(f)` | `exec<A,FP32,F,L>(dst)` | Imbalanced | Depends | rsqrt | Complex |

### Cross-Iteration State

NONE — DEST acquired+released per-tile, vector mode per-call, scalars immutable, LUT set in init only.

**Implication**: helper does NOT need to own tile loop; per-tile callable per eltwise_chain model.

### Parameter Independence

All independent — none derivable from others. Helper takes them all as runtime args.

### CB Compile-Time Analysis

Most call sites declare input CB as runtime arg (varies per kernel instance).

### Template Parameter Consolidation

```cpp
template <
    typename DerivedOp,
    Dst OutputSlot = Dst::D0,
    Approx Approximation = Approx::Exact,
    Approx FastApprox = Approx::Exact,
    Legacy LegacyMode = Legacy::Off,
    bool Fp32DestAccEn = false,
    RoundingMode Rounding = RoundingMode::None,
    DataFormat MaskFormat = DataFormat::Float16_b
>
struct UnaryOp { ... };
```

---

## Recommendations for CRTP Base & Helper

1. **UnaryOp CRTP**: single base per output slot; template all 8 params with smart defaults.
2. **Scalar runtime field**: ops like power/clamp store scalar in `value` field, NOT template.
3. **Init hoisting precondition**: only `CopyTile + 1 Plain Op` (abs, sign, bitwise_not, predicates); LUT or multi-param disqualifies.
4. **Init mutual exclusion trait**: `clobbers_sfpu_lut_polynomial`, `clobbers_sfpu_lut_activation` — validate chains at compile time.
5. **Predicate tag**: per lessons §1.5 — semantic grouping.
6. **Mask hardcoded slot**: per lessons §1.4 — bake `Slot+1` into struct instantiation.
7. **Vector mode default**: RC (4 faces) is common case.

---

## Open Questions

1. rsqrt template mismatch (init 2, exec 4): does init really not need fp32_dest_acc_en or fast_approx? Verify `sfpu::rsqrt_init` signature.
2. Hardmish/Mish (LUT-dependent): mark as `clobbers_sfpu_lut_polynomial`?
3. Sigmoid vector mode (sometimes param0 not template): macro-injection artifact or real runtime param?
4. WHERE_TSS / CLAMP_TSS: ternary, separate investigation.

---

Generated: 2026-04-30
Investigation Scope: Unary SFPU omnibus (~90 ops across 7 groups)
