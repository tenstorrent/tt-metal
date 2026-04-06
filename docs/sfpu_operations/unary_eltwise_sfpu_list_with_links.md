# SFPU Unary Eltwise Operations -- PyTorch Equivalents and Documentation Links

Complete mapping of all 109 SFPU UnaryOpType operations (plus RReLU) to their PyTorch equivalents with documentation links.

**Source files**:
- Enum: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`
- Dispatch: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
- Parametrized check: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`

---

## Activation Functions

| # | Operation | Category | PyTorch Equivalent | PyTorch Docs Link | Parametrized | Notes |
|---|-----------|----------|-------------------|-------------------|--------------|-------|
| 1 | RELU | Activation | `torch.nn.ReLU` / `torch.nn.functional.relu` | [torch.nn.ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html) | No | Family head of RELU_FAMILY |
| 2 | RELU6 | Activation | `torch.nn.ReLU6` | [torch.nn.ReLU6](https://pytorch.org/docs/stable/generated/torch.nn.ReLU6.html) | No | Equivalent to `torch.clamp(min=0, max=6)` |
| 3 | RELU_MAX | Activation | `torch.clamp(min=0, max=upper)` | [torch.clamp](https://pytorch.org/docs/stable/generated/torch.clamp.html) | Yes | `upper_limit` param |
| 4 | RELU_MIN | Activation | `torch.clamp(min=lower)` | [torch.clamp](https://pytorch.org/docs/stable/generated/torch.clamp.html) | Yes | `lower_limit` param |
| 5 | LEAKY_RELU | Activation | `torch.nn.LeakyReLU` | [torch.nn.LeakyReLU](https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html) | Yes | `negative_slope` param |
| 6 | ELU | Activation | `torch.nn.ELU` | [torch.nn.ELU](https://pytorch.org/docs/stable/generated/torch.nn.ELU.html) | Yes | `alpha` param |
| 7 | SELU | Activation | `torch.nn.SELU` | [torch.nn.SELU](https://pytorch.org/docs/stable/generated/torch.nn.SELU.html) | Yes | `alpha`, `scale` params |
| 8 | CELU | Activation | `torch.nn.CELU` | [torch.nn.CELU](https://pytorch.org/docs/stable/generated/torch.nn.CELU.html) | Yes | `alpha` param |
| 9 | GELU | Activation | `torch.nn.GELU` | [torch.nn.GELU](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html) | Yes | `fast_and_approximate_mode` param |
| 10 | SILU | Activation | `torch.nn.SiLU` | [torch.nn.SiLU](https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html) | No | Also known as Swish |
| 11 | SIGMOID | Activation | `torch.sigmoid` | [torch.sigmoid](https://pytorch.org/docs/stable/generated/torch.sigmoid.html) | Yes | `fast_and_approximate_mode` param |
| 12 | SOFTPLUS | Activation | `torch.nn.Softplus` | [torch.nn.Softplus](https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html) | Yes | `beta`, `threshold` params |
| 13 | SOFTSHRINK | Activation | `torch.nn.Softshrink` | [torch.nn.Softshrink](https://pytorch.org/docs/stable/generated/torch.nn.Softshrink.html) | Yes | `lambda` param |
| 14 | SOFTSIGN | Activation | `torch.nn.Softsign` | [torch.nn.Softsign](https://pytorch.org/docs/stable/generated/torch.nn.Softsign.html) | No | |
| 15 | HARDSIGMOID | Activation | `torch.nn.Hardsigmoid` | [torch.nn.Hardsigmoid](https://pytorch.org/docs/stable/generated/torch.nn.Hardsigmoid.html) | No | |
| 16 | HARDSWISH | Activation | `torch.nn.Hardswish` | [torch.nn.Hardswish](https://pytorch.org/docs/stable/generated/torch.nn.Hardswish.html) | No | Mixed routing: also has `hardswish_kernel.cpp` |
| 17 | HARDTANH | Activation | `torch.nn.Hardtanh` | [torch.nn.Hardtanh](https://pytorch.org/docs/stable/generated/torch.nn.Hardtanh.html) | Yes | `min_val`, `max_val` params |
| 18 | HARDSHRINK | Activation | `torch.nn.Hardshrink` | [torch.nn.Hardshrink](https://pytorch.org/docs/stable/generated/torch.nn.Hardshrink.html) | No | Mixed routing: also has `hardshrink_kernel.cpp` |
| 19 | LOGSIGMOID | Activation | `torch.nn.LogSigmoid` | [torch.nn.LogSigmoid](https://pytorch.org/docs/stable/generated/torch.nn.LogSigmoid.html) | No | Mixed routing: also has `logsigmoid_kernel.cpp` |
| 20 | TANHSHRINK | Activation | `torch.nn.Tanhshrink` | [torch.nn.Tanhshrink](https://pytorch.org/docs/stable/generated/torch.nn.Tanhshrink.html) | No | Mixed routing: also has `tanhshrink_kernel.cpp` |
| 21 | THRESHOLD | Activation | `torch.nn.Threshold` | [torch.nn.Threshold](https://pytorch.org/docs/stable/generated/torch.nn.Threshold.html) | Yes | `threshold`, `value` params |
| 22 | PRELU_SFPU | Activation | `torch.nn.PReLU` | [torch.nn.PReLU](https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html) | Yes | `weight` param |
| 23 | HARDMISH | Activation | Tenstorrent custom | -- | No | No PyTorch equivalent |
| 24 | XIELU | Activation | Tenstorrent custom | -- | Yes | No PyTorch equivalent |

## Math Functions

| # | Operation | Category | PyTorch Equivalent | PyTorch Docs Link | Parametrized | Notes |
|---|-----------|----------|-------------------|-------------------|--------------|-------|
| 25 | EXP | Math | `torch.exp` | [torch.exp](https://pytorch.org/docs/stable/generated/torch.exp.html) | Yes | `fast_and_approximate_mode` param |
| 26 | EXP2 | Math | `torch.exp2` | [torch.exp2](https://pytorch.org/docs/stable/generated/torch.exp2.html) | No | Also `torch.special.exp2` |
| 27 | EXPM1 | Math | `torch.expm1` | [torch.expm1](https://pytorch.org/docs/stable/generated/torch.expm1.html) | No | |
| 28 | LOG | Math | `torch.log` | [torch.log](https://pytorch.org/docs/stable/generated/torch.log.html) | Yes | |
| 29 | LOG2 | Math | `torch.log2` | [torch.log2](https://pytorch.org/docs/stable/generated/torch.log2.html) | Yes | |
| 30 | LOG10 | Math | `torch.log10` | [torch.log10](https://pytorch.org/docs/stable/generated/torch.log10.html) | Yes | |
| 31 | LOG1P | Math | `torch.log1p` | [torch.log1p](https://pytorch.org/docs/stable/generated/torch.log1p.html) | Yes | |
| 32 | SQRT | Math | `torch.sqrt` | [torch.sqrt](https://pytorch.org/docs/stable/generated/torch.sqrt.html) | Yes | |
| 33 | RSQRT | Math | `torch.rsqrt` | [torch.rsqrt](https://pytorch.org/docs/stable/generated/torch.rsqrt.html) | Yes | `fast_and_approximate_mode` param |
| 34 | CBRT | Math | `torch.pow(x, 1/3)` | [torch.pow](https://pytorch.org/docs/stable/generated/torch.pow.html) | No | No dedicated `cbrt` in PyTorch |
| 35 | RECIP | Math | `torch.reciprocal` | [torch.reciprocal](https://pytorch.org/docs/stable/generated/torch.reciprocal.html) | No | |
| 36 | SQUARE | Math | `torch.square` | [torch.square](https://pytorch.org/docs/stable/generated/torch.square.html) | No | |
| 37 | POWER | Math | `torch.pow` | [torch.pow](https://pytorch.org/docs/stable/generated/torch.pow.html) | Yes | `exponent` param |
| 38 | POWER_ITERATIVE | Math | `torch.pow` | [torch.pow](https://pytorch.org/docs/stable/generated/torch.pow.html) | Yes | `exponent` param; iterative variant |
| 39 | RPOW | Math | `scalar ** tensor` | [torch.pow](https://pytorch.org/docs/stable/generated/torch.pow.html) | Yes | `exponent` param; reverse power (`torch.pow(scalar, tensor)`) |
| 40 | ABS | Math | `torch.abs` | [torch.abs](https://pytorch.org/docs/stable/generated/torch.abs.html) | No | |
| 41 | ABS_INT32 | Math | `torch.abs` | [torch.abs](https://pytorch.org/docs/stable/generated/torch.abs.html) | No | Int32 variant |
| 42 | NEG | Math | `torch.neg` | [torch.neg](https://pytorch.org/docs/stable/generated/torch.neg.html) | No | |
| 43 | SIGN | Math | `torch.sign` | [torch.sign](https://pytorch.org/docs/stable/generated/torch.sign.html) | No | |
| 44 | SIGNBIT | Math | `torch.signbit` | [torch.signbit](https://pytorch.org/docs/stable/generated/torch.signbit.html) | No | |
| 45 | HEAVISIDE | Math | `torch.heaviside` | [torch.heaviside](https://pytorch.org/docs/stable/generated/torch.heaviside.html) | Yes | `value` param |

## Trigonometric Functions

| # | Operation | Category | PyTorch Equivalent | PyTorch Docs Link | Parametrized | Notes |
|---|-----------|----------|-------------------|-------------------|--------------|-------|
| 46 | SIN | Trigonometric | `torch.sin` | [torch.sin](https://pytorch.org/docs/stable/generated/torch.sin.html) | No | |
| 47 | COS | Trigonometric | `torch.cos` | [torch.cos](https://pytorch.org/docs/stable/generated/torch.cos.html) | No | |
| 48 | TAN | Trigonometric | `torch.tan` | [torch.tan](https://pytorch.org/docs/stable/generated/torch.tan.html) | No | |
| 49 | ASIN | Trigonometric | `torch.asin` | [torch.asin](https://pytorch.org/docs/stable/generated/torch.asin.html) | No | |
| 50 | ACOS | Trigonometric | `torch.acos` | [torch.acos](https://pytorch.org/docs/stable/generated/torch.acos.html) | No | |
| 51 | ATAN | Trigonometric | `torch.atan` | [torch.atan](https://pytorch.org/docs/stable/generated/torch.atan.html) | No | |
| 52 | SINH | Trigonometric | `torch.sinh` | [torch.sinh](https://pytorch.org/docs/stable/generated/torch.sinh.html) | No | |
| 53 | COSH | Trigonometric | `torch.cosh` | [torch.cosh](https://pytorch.org/docs/stable/generated/torch.cosh.html) | No | |
| 54 | TANH | Trigonometric | `torch.tanh` | [torch.tanh](https://pytorch.org/docs/stable/generated/torch.tanh.html) | Yes | |
| 55 | ASINH | Trigonometric | `torch.asinh` | [torch.asinh](https://pytorch.org/docs/stable/generated/torch.asinh.html) | No | |
| 56 | ACOSH | Trigonometric | `torch.acosh` | [torch.acosh](https://pytorch.org/docs/stable/generated/torch.acosh.html) | No | |
| 57 | ATANH | Trigonometric | `torch.atanh` | [torch.atanh](https://pytorch.org/docs/stable/generated/torch.atanh.html) | No | |

## Error and Special Functions

| # | Operation | Category | PyTorch Equivalent | PyTorch Docs Link | Parametrized | Notes |
|---|-----------|----------|-------------------|-------------------|--------------|-------|
| 58 | ERF | Special | `torch.erf` | [torch.erf](https://pytorch.org/docs/stable/generated/torch.erf.html) | Yes | `fast_and_approximate_mode` param |
| 59 | ERFC | Special | `torch.erfc` | [torch.erfc](https://pytorch.org/docs/stable/generated/torch.erfc.html) | Yes | `fast_and_approximate_mode` param |
| 60 | ERFINV | Special | `torch.erfinv` | [torch.erfinv](https://pytorch.org/docs/stable/generated/torch.erfinv.html) | No | |
| 61 | I0 | Special | `torch.i0` | [torch.special.i0](https://pytorch.org/docs/stable/generated/torch.special.i0.html) | No | Modified Bessel function, first kind, order 0 |
| 62 | I1 | Special | `torch.special.i1` | [torch.special.i1](https://pytorch.org/docs/stable/generated/torch.special.i1.html) | No | Modified Bessel function, first kind, order 1 |
| 63 | LGAMMA | Special | `torch.lgamma` | [torch.lgamma](https://pytorch.org/docs/stable/generated/torch.lgamma.html) | No | Mixed routing: also has `lgamma_kernel.cpp` |

## Rounding

| # | Operation | Category | PyTorch Equivalent | PyTorch Docs Link | Parametrized | Notes |
|---|-----------|----------|-------------------|-------------------|--------------|-------|
| 64 | FLOOR | Rounding | `torch.floor` | [torch.floor](https://pytorch.org/docs/stable/generated/torch.floor.html) | No | |
| 65 | CEIL | Rounding | `torch.ceil` | [torch.ceil](https://pytorch.org/docs/stable/generated/torch.ceil.html) | No | |
| 66 | TRUNC | Rounding | `torch.trunc` | [torch.trunc](https://pytorch.org/docs/stable/generated/torch.trunc.html) | No | |
| 67 | FRAC | Rounding | `torch.frac` | [torch.frac](https://pytorch.org/docs/stable/generated/torch.frac.html) | No | |
| 68 | ROUND | Rounding | `torch.round` | [torch.round](https://pytorch.org/docs/stable/generated/torch.round.html) | Yes | `decimals` param |

## Comparison (Unary with Scalar)

| # | Operation | Category | PyTorch Equivalent | PyTorch Docs Link | Parametrized | Notes |
|---|-----------|----------|-------------------|-------------------|--------------|-------|
| 69 | UNARY_NE | Comparison | `torch.ne(x, scalar)` | [torch.ne](https://pytorch.org/docs/stable/generated/torch.ne.html) | Yes | `scalar` param |
| 70 | UNARY_EQ | Comparison | `torch.eq(x, scalar)` | [torch.eq](https://pytorch.org/docs/stable/generated/torch.eq.html) | Yes | `scalar` param |
| 71 | UNARY_GT | Comparison | `torch.gt(x, scalar)` | [torch.gt](https://pytorch.org/docs/stable/generated/torch.gt.html) | Yes | `scalar` param |
| 72 | UNARY_LT | Comparison | `torch.lt(x, scalar)` | [torch.lt](https://pytorch.org/docs/stable/generated/torch.lt.html) | Yes | `scalar` param |
| 73 | UNARY_GE | Comparison | `torch.ge(x, scalar)` | [torch.ge](https://pytorch.org/docs/stable/generated/torch.ge.html) | Yes | `scalar` param |
| 74 | UNARY_LE | Comparison | `torch.le(x, scalar)` | [torch.le](https://pytorch.org/docs/stable/generated/torch.le.html) | Yes | `scalar` param |
| 75 | GTZ | Comparison | `torch.gt(x, 0)` | [torch.gt](https://pytorch.org/docs/stable/generated/torch.gt.html) | No | Greater than zero |
| 76 | LTZ | Comparison | `torch.lt(x, 0)` | [torch.lt](https://pytorch.org/docs/stable/generated/torch.lt.html) | No | Less than zero |
| 77 | EQZ | Comparison | `torch.eq(x, 0)` | [torch.eq](https://pytorch.org/docs/stable/generated/torch.eq.html) | No | Equal to zero |
| 78 | LEZ | Comparison | `torch.le(x, 0)` | [torch.le](https://pytorch.org/docs/stable/generated/torch.le.html) | No | Less than or equal to zero |
| 79 | GEZ | Comparison | `torch.ge(x, 0)` | [torch.ge](https://pytorch.org/docs/stable/generated/torch.ge.html) | No | Greater than or equal to zero |
| 80 | NEZ | Comparison | `torch.ne(x, 0)` | [torch.ne](https://pytorch.org/docs/stable/generated/torch.ne.html) | No | Not equal to zero |

## Logic

| # | Operation | Category | PyTorch Equivalent | PyTorch Docs Link | Parametrized | Notes |
|---|-----------|----------|-------------------|-------------------|--------------|-------|
| 81 | LOGICAL_NOT_UNARY | Logic | `torch.logical_not` | [torch.logical_not](https://pytorch.org/docs/stable/generated/torch.logical_not.html) | No | |

## Bitwise

| # | Operation | Category | PyTorch Equivalent | PyTorch Docs Link | Parametrized | Notes |
|---|-----------|----------|-------------------|-------------------|--------------|-------|
| 82 | BITWISE_XOR | Bitwise | `torch.bitwise_xor(x, scalar)` | [torch.bitwise_xor](https://pytorch.org/docs/stable/generated/torch.bitwise_xor.html) | Yes | `scalar` param |
| 83 | BITWISE_NOT | Bitwise | `torch.bitwise_not` | [torch.bitwise_not](https://pytorch.org/docs/stable/generated/torch.bitwise_not.html) | No | |
| 84 | BITWISE_AND | Bitwise | `torch.bitwise_and(x, scalar)` | [torch.bitwise_and](https://pytorch.org/docs/stable/generated/torch.bitwise_and.html) | Yes | `scalar` param |
| 85 | BITWISE_OR | Bitwise | `torch.bitwise_or(x, scalar)` | [torch.bitwise_or](https://pytorch.org/docs/stable/generated/torch.bitwise_or.html) | Yes | `scalar` param |
| 86 | RIGHT_SHIFT | Bitwise | `torch.bitwise_right_shift` | [torch.bitwise_right_shift](https://pytorch.org/docs/stable/generated/torch.bitwise_right_shift.html) | Yes | `shift_amount` param |
| 87 | LEFT_SHIFT | Bitwise | `torch.bitwise_left_shift` | [torch.bitwise_left_shift](https://pytorch.org/docs/stable/generated/torch.bitwise_left_shift.html) | Yes | `shift_amount` param |

## Arithmetic with Scalar

| # | Operation | Category | PyTorch Equivalent | PyTorch Docs Link | Parametrized | Notes |
|---|-----------|----------|-------------------|-------------------|--------------|-------|
| 88 | ADD_UNARY_SFPU | Arithmetic | `torch.add(x, scalar)` | [torch.add](https://pytorch.org/docs/stable/generated/torch.add.html) | Yes | `scalar` param |
| 89 | SUB_UNARY_SFPU | Arithmetic | `torch.sub(x, scalar)` | [torch.sub](https://pytorch.org/docs/stable/generated/torch.sub.html) | Yes | `scalar` param |
| 90 | MUL_UNARY_SFPU | Arithmetic | `torch.mul(x, scalar)` | [torch.mul](https://pytorch.org/docs/stable/generated/torch.mul.html) | Yes | `scalar` param |
| 91 | DIV_UNARY_SFPU | Arithmetic | `torch.div(x, scalar)` | [torch.div](https://pytorch.org/docs/stable/generated/torch.div.html) | Yes | `scalar` param |
| 92 | RSUB | Arithmetic | `scalar - x` | [torch.sub](https://pytorch.org/docs/stable/generated/torch.sub.html) | Yes | `scalar` param; reverse subtraction |
| 93 | RDIV | Arithmetic | `scalar / x` | [torch.div](https://pytorch.org/docs/stable/generated/torch.div.html) | Yes | `scalar` param; reverse division |
| 94 | REMAINDER | Arithmetic | `torch.remainder` | [torch.remainder](https://pytorch.org/docs/stable/generated/torch.remainder.html) | Yes | `divisor` param |
| 95 | FMOD | Arithmetic | `torch.fmod` | [torch.fmod](https://pytorch.org/docs/stable/generated/torch.fmod.html) | Yes | `divisor` param |

## Clamp and Where

| # | Operation | Category | PyTorch Equivalent | PyTorch Docs Link | Parametrized | Notes |
|---|-----------|----------|-------------------|-------------------|--------------|-------|
| 96 | CLAMP_TSS | Clamp/Where | `torch.clamp(tensor, scalar, scalar)` | [torch.clamp](https://pytorch.org/docs/stable/generated/torch.clamp.html) | Yes | `min`, `max` params |
| 97 | WHERE_TSS | Clamp/Where | `torch.where(tensor, scalar, scalar)` | [torch.where](https://pytorch.org/docs/stable/generated/torch.where.html) | Yes | Mixed routing: also has `where_tss_kernel.cpp` |
| 98 | MINIMUM | Clamp/Where | `torch.minimum(x, scalar)` | [torch.minimum](https://pytorch.org/docs/stable/generated/torch.minimum.html) | Yes | `scalar` param |
| 99 | MAXIMUM | Clamp/Where | `torch.maximum(x, scalar)` | [torch.maximum](https://pytorch.org/docs/stable/generated/torch.maximum.html) | Yes | `scalar` param |

## Infrastructure / Type

| # | Operation | Category | PyTorch Equivalent | PyTorch Docs Link | Parametrized | Notes |
|---|-----------|----------|-------------------|-------------------|--------------|-------|
| 100 | FILL | Infrastructure | `torch.full_like` / `tensor.fill_` | [torch.full_like](https://pytorch.org/docs/stable/generated/torch.full_like.html) | Yes | `fill_value` param |
| 101 | TYPECAST | Infrastructure | `tensor.to(dtype)` | [torch.Tensor.to](https://pytorch.org/docs/stable/generated/torch.Tensor.to.html) | Yes | `target_dtype` param |
| 102 | BITCAST | Infrastructure | `tensor.view(dtype)` | [torch.Tensor.view](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html) | Yes | `target_dtype` param; reinterprets bits |
| 103 | TILED_PROD | Infrastructure | Tenstorrent internal | -- | No | Internal reduction; no PyTorch equivalent |
| 104 | ALT_COMPLEX_ROTATE90 | Infrastructure | Tenstorrent internal | -- | No | Internal complex rotation; no PyTorch equivalent |
| 105 | DROPOUT | Infrastructure | `torch.nn.Dropout` | [torch.nn.Dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html) | No | **EXCLUDED from nuke** -- infrastructure |
| 106 | ZERO_POINT | Infrastructure | Tenstorrent internal | -- | No | **EXCLUDED from nuke** -- quantization infrastructure |

---

## Excluded from Nuke (Non-SFPU)

These operations are NOT implemented via SFPU and are excluded from the SFPU operation catalog:

| Operation | Reason | PyTorch Equivalent | PyTorch Docs Link |
|-----------|--------|-------------------|-------------------|
| MISH | Custom kernel `mish_kernel.cpp` | `torch.nn.Mish` | [torch.nn.Mish](https://pytorch.org/docs/stable/generated/torch.nn.Mish.html) |
| IDENTITY | Custom kernel `eltwise_identity_kernel.cpp` | `torch.nn.Identity` | [torch.nn.Identity](https://pytorch.org/docs/stable/generated/torch.nn.Identity.html) |
| LOGIT | Custom kernel `logit_kernel.cpp` | `torch.logit` | [torch.logit](https://pytorch.org/docs/stable/generated/torch.logit.html) |

---

## Inspection Checks (Non-SFPU)

These operations appear in the UnaryOpType enum but are inspection/check operations, not SFPU compute:

| Operation | PyTorch Equivalent | PyTorch Docs Link |
|-----------|-------------------|-------------------|
| ISINF | `torch.isinf` | [torch.isinf](https://pytorch.org/docs/stable/generated/torch.isinf.html) |
| ISNAN | `torch.isnan` | [torch.isnan](https://pytorch.org/docs/stable/generated/torch.isnan.html) |
| ISNEGINF | `torch.isneginf` | [torch.isneginf](https://pytorch.org/docs/stable/generated/torch.isneginf.html) |
| ISPOSINF | `torch.isposinf` | [torch.isposinf](https://pytorch.org/docs/stable/generated/torch.isposinf.html) |
| ISFINITE | `torch.isfinite` | [torch.isfinite](https://pytorch.org/docs/stable/generated/torch.isfinite.html) |

---

## New Operation: RReLU

| # | Operation | Category | PyTorch Equivalent | PyTorch Docs Link | Parametrized | Notes |
|---|-----------|----------|-------------------|-------------------|--------------|-------|
| 110 | RReLU | Activation | `torch.nn.RReLU` | [torch.nn.RReLU](https://pytorch.org/docs/stable/generated/torch.nn.RReLU.html) | Yes | **NEW -- to be implemented**; `lower`, `upper` params; randomized leaky ReLU |

RReLU (Randomized Leaky Rectified Linear Unit) applies:
- `f(x) = x` if `x >= 0`
- `f(x) = a * x` if `x < 0`, where `a` is sampled from `Uniform(lower, upper)` during training

During evaluation, `a` is fixed to `(lower + upper) / 2`.

---

## Summary

| Metric | Count |
|--------|-------|
| Total SFPU operations (existing) | 109 |
| Operations with PyTorch equivalent | 101 |
| Tenstorrent custom (no PyTorch equivalent) | 5 (HARDMISH, XIELU, TILED_PROD, ALT_COMPLEX_ROTATE90, ZERO_POINT) |
| Infrastructure / excluded from nuke | 3 (DROPOUT, ZERO_POINT, TILED_PROD) |
| Excluded non-SFPU operations | 3 (MISH, IDENTITY, LOGIT) |
| Mixed routing (SFPU + custom kernel) | 6 (HARDSWISH, HARDSHRINK, TANHSHRINK, LOGSIGMOID, WHERE_TSS, LGAMMA) |
| New operation to add | 1 (RReLU) |
| **Total after RReLU** | **110** |
