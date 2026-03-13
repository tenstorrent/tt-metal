# Top 25 SFPU Operations — Deduplicated by Implementation, Spanning All Complexities

Curated for **implementation diversity**: operations with near-identical SFPU kernel patterns are consolidated into one representative. HARD operations are included to cover all major implementation paradigms (transcendental chains, iterative algorithms, dual-path kernels, PRNG, polynomial series).

Complexity ratings extracted from:
- `docs/unary_sfpu_operations_analysis.md` (112 unary ops)
- `docs/binary_sfpu_operations_analysis.md` (28 native + composites)
- `docs/ternary_sfpu_operations_analysis.md` (5 ternary ops)

> **Note**: ADD and MUL run on the **FPU** for BFloat16 and only hit the SFPU for FP32/Int32/UInt32/UInt16. Included because they are the most invoked element-wise operations.

---

## Removed as Implementation-Similar

These were in the original top 25 but are near-identical in SFPU kernel pattern to another operation that remains:

| Removed | Same Pattern As | Why Redundant |
|---------|----------------|---------------|
| SUB | ADD | Same single SFPU instruction, different opcode |
| NEG | ABS | Both are trivial sign-bit manipulation |
| SQUARE | MUL | `MUL(x, x)` — identical instruction |
| RSQRT | SQRT | Same Newton-Raphson algorithm, different seed/target |
| LEAKY_RELU | RELU | Same conditional pattern, just adds slope parameter |
| SOFTPLUS | EXP + LOG | `log(1+exp(x))` — pure composition of EXP and LOG |
| CLAMP | RELU | Same conditional threshold pattern, two bounds instead of one |
| EQ | MAXIMUM | Same compare + conditional-select SFPU pattern |

---

## Top 25 Table

| Rank | Operation | Arity | Math | Complexity | Implementation Pattern | Primary Use Cases |
|------|-----------|-------|------|------------|----------------------|-------------------|
| 1 | **ADD** | Binary | `a + b` | **EASY** | Single SFPU instruction | Residual connections, bias, skip connections |
| 2 | **MUL** | Binary | `a * b` | **EASY** | Single SFPU instruction | Attention scores, scaling, gating |
| 3 | **RELU** | Unary | `max(0, x)` | **EASY** | Single conditional zero | Most common activation (CNNs, MLPs) |
| 4 | **ABS** | Unary | `\|x\|` | **EASY** | Single instruction (`sfpi::abs`) | L1 loss, gradient magnitude |
| 5 | **MAXIMUM** | Binary | `max(a, b)` | **EASY** | Compare + conditional select | Gradient clipping, element-wise max |
| 6 | **EXP** | Unary | `e^x` | **MEDIUM** | Polynomial approx + range reduction | Softmax, attention weights, probability |
| 7 | **RECIP** | Unary | `1/x` | **MEDIUM** | Newton-Raphson (1-2 iterations) | Normalization denominators, softmax |
| 8 | **SIGMOID** | Unary | `1/(1+e^-x)` | **MEDIUM** | exp + reciprocal composition | Gating (LSTM, GRU), classification |
| 9 | **GELU** | Unary | `x * Phi(x)` | **MEDIUM** | 15th-degree Chebyshev polynomial | Transformer activations (BERT, GPT) |
| 10 | **TANH** | Unary | `tanh(x)` | **MEDIUM** | Polynomial approximation | LSTM, GELU approx, normalization |
| 11 | **SQRT** | Unary | `sqrt(x)` | **MEDIUM** | Newton-Raphson iteration | LayerNorm, BatchNorm, RMSNorm |
| 12 | **LOG** | Unary | `ln(x)` | **MEDIUM** | Minimax approx + normalization | Cross-entropy, log-softmax, KL div |
| 13 | **DIV** | Binary | `a / b` | **MEDIUM** | Iterative reciprocal + mul (FP); multi-step (Int) | Normalization, attention scaling |
| 14 | **SILU** | Unary | `x * sigmoid(x)` | **MEDIUM** | sigmoid composition (exp+recip+mul) | SwiGLU (LLaMA, Mistral, modern LLMs) |
| 15 | **WHERE** | Ternary | `cond ? a : b` | **EASY/MEDIUM** | 3-input conditional select; bcast variants add CB sync | Masking, padding, conditional ops |
| 16 | **TYPECAST** | Unary | Type conversion | **MEDIUM** | Format conversion logic | Mixed-precision training (FP32 <-> BF16) |
| 17 | **DROPOUT** | Unary | `x * mask(p)` | **MEDIUM** | PRNG state management + conditional | Training regularization (ubiquitous) |
| 18 | **LERP** | Ternary | `a + w*(b-a)` | **EASY/MEDIUM** | Same 3-input SFPU as WHERE; bcast adds sync | EMA weight updates, interpolation |
| 19 | **POWER** | Binary | `a^b` | **HARD** | `exp(b * log(a))` — 3 transcendental chain | Learning rate schedules, custom losses |
| 20 | **ADDCMUL** | Ternary | `a + v*b*c` | **HARD** | Dual SFPU/FPU paths, scalar runtime arg, 5 kernel files | Adam/AdamW optimizer step |
| 21 | **LOGADDEXP** | Binary | `log(e^a + e^b)` | **HARD** | EXP pre-A + EXP pre-B + ADD + LOG post (composite) | Log-space arithmetic, numerical stability |
| 22 | **ERFINV** | Unary | `erf^-1(x)` | **HARD** | Winitzki approx: nested sqrt + log composition | Normal distribution sampling, init |
| 23 | **ASIN** | Unary | `arcsin(x)` | **HARD** | Domain [-1,1] polynomial + sqrt composition | Geometric ops, angular computation |
| 24 | **GCD** | Binary | `gcd(a, b)` | **HARD** | Iterative Euclidean algorithm, variable-latency | Integer arithmetic, quantization grids |
| 25 | **I0** | Unary | Bessel `I_0(x)` | **HARD** | 10th-degree polynomial series (Taylor truncation) | Audio/signal processing, window functions |

---

## Grouped by Complexity

### EASY (5 of 25)

Single instruction or trivial logic. Minimal SFPU cycles per element.

| Rank | Operation | Arity | Implementation Pattern | What's Distinct |
|------|-----------|-------|-----------------------|-----------------|
| 1 | **ADD** | Binary | Single SFPU instruction | Canonical arithmetic; represents SUB/RSUB too |
| 2 | **MUL** | Binary | Single SFPU instruction | Represents SQUARE as well |
| 3 | **RELU** | Unary | Single conditional zero | Represents LEAKY_RELU, CLAMP family |
| 4 | **ABS** | Unary | `sfpi::abs(v)` — single instruction | Represents NEG, SIGN (sign-bit ops) |
| 5 | **MAXIMUM** | Binary | Compare + conditional select | Represents MINIMUM, EQ, LT/GT/GE/LE |

### MEDIUM (12 of 25)

Polynomial approximations, Newton-Raphson iterations, multi-step composition, or unique SFPU patterns.

| Rank | Operation | Arity | Implementation Pattern | What's Distinct |
|------|-----------|-------|-----------------------|-----------------|
| 6 | **EXP** | Unary | Polynomial approx + range reduction | Core building block; represents EXP2, EXPM1 |
| 7 | **RECIP** | Unary | Newton-Raphson (1-2 iterations) | Iterative convergence pattern |
| 8 | **SIGMOID** | Unary | exp(-x) + reciprocal composition | Two-function composition; represents LOGSIGMOID |
| 9 | **GELU** | Unary | 15th-degree Chebyshev polynomial | Highest-degree polynomial in the codebase |
| 10 | **TANH** | Unary | Polynomial approximation | Distinct polynomial from GELU; represents TANHSHRINK |
| 11 | **SQRT** | Unary | Newton-Raphson iteration | Same algorithm family as RECIP but different seed; represents RSQRT |
| 12 | **LOG** | Unary | Minimax approximation + exponent normalization | Different approx technique from EXP; represents LOG2, LOG10, LOG1P |
| 13 | **DIV** | Binary | Iterative reciprocal + mul (FP); multi-step division (Int) | Dual FP/Int paths; represents DIV_FLOOR, DIV_TRUNC, FMOD, REMAINDER |
| 14 | **SILU** | Unary | x * sigmoid(x) — full sigmoid embedded | Three-step composition (exp+recip+mul); represents MISH, HARDSWISH |
| 16 | **TYPECAST** | Unary | Format conversion logic | Unique bit-format manipulation; no polynomial or iteration |
| 17 | **DROPOUT** | Unary | PRNG state + conditional zeroing | Only operation using random number generation on SFPU |
| 18 | **LERP** | Ternary | 3-input SFPU (`lerp_tile`) + bcast variants | Ternary interpolation; shared kernel infra with WHERE |

### EASY/MEDIUM (1 of 25)

Complexity varies by variant and broadcast configuration.

| Rank | Operation | Arity | When EASY | When MEDIUM |
|------|-----------|-------|-----------|-------------|
| 15 | **WHERE** | Ternary | TTT no-broadcast: single SFPU call, 58-line kernel | TTS/TST scalar fill, broadcast variants (CB sync, iteration control, LLK row bcast, 7 CBs) |

### HARD (7 of 25)

Transcendental chains, iterative algorithms with data-dependent cost, dual execution paths, or complex multi-step approximations.

| Rank | Operation | Arity | Implementation Pattern | What's Distinct |
|------|-----------|-------|-----------------------|-----------------|
| 19 | **POWER** | Binary | `exp(b * log(a))` — 3 transcendental ops chained | Only binary op requiring log+mul+exp in sequence; represents RPOW, XLOGY |
| 20 | **ADDCMUL** | Ternary | Dual SFPU/FPU execution paths, scalar runtime arg, 5 kernel files | Only op with runtime path selection between SFPU and FPU; represents ADDCDIV |
| 21 | **LOGADDEXP** | Binary | EXP(a) + EXP(b) → ADD → LOG: composite with 3 transcendental ops | Pre-A + Pre-B + Post pattern; represents LOGADDEXP2, HYPOT |
| 22 | **ERFINV** | Unary | Winitzki approximation: `sqrt(-2/(pi*a) - log(1-x^2)/2 + sqrt(...))` | Nested sqrt + log — deepest function composition in the codebase |
| 23 | **ASIN** | Unary | Domain [-1,1] polynomial + sqrt composition | Domain-restricted with sqrt; represents ACOS, ATAN, ASINH, ACOSH, ATANH |
| 24 | **GCD** | Binary | Iterative Euclidean algorithm, variable-latency per element | Only SFPU op with data-dependent iteration count; represents LCM |
| 25 | **I0** | Unary | 10th-degree polynomial series (Taylor truncation) | Highest-coefficient-count polynomial; represents I1 (Bessel functions) |

---

## Implementation Pattern Coverage

This list covers all **10 distinct SFPU implementation paradigms** found across the codebase:

| Pattern | Representative Op | Complexity |
|---------|--------------------|------------|
| Single SFPU instruction | ADD, MUL | EASY |
| Conditional/compare | RELU, ABS, MAXIMUM | EASY |
| Polynomial approximation (range reduction) | EXP, TANH | MEDIUM |
| Chebyshev polynomial (high degree) | GELU | MEDIUM |
| Newton-Raphson iterative convergence | RECIP, SQRT | MEDIUM |
| Minimax approximation + normalization | LOG | MEDIUM |
| Multi-function composition | SIGMOID, SILU, DROPOUT | MEDIUM |
| Transcendental chain (3+ ops) | POWER, LOGADDEXP | HARD |
| Nested composition (sqrt+log depth) | ERFINV, ASIN | HARD |
| Data-dependent iterative algorithm | GCD | HARD |
| High-degree polynomial series | I0 | HARD |
| Dual SFPU/FPU execution paths | ADDCMUL | HARD |
| 3-input SFPU with broadcast variants | WHERE, LERP | EASY/MEDIUM |
| Format conversion | TYPECAST | MEDIUM |
| PRNG + conditional | DROPOUT | MEDIUM |

---

## Complexity Distribution

```
EASY:          5 ops  (20%)  ██████████
MEDIUM:       12 ops  (48%)  ████████████████████████
EASY/MEDIUM:   1 op   ( 4%)  ██
HARD:          7 ops  (28%)  ██████████████
```

Compared to the original list (36% EASY, 56% MEDIUM, 0% HARD), this version better represents the full spectrum of SFPU implementation complexity.

---

## Program Factory Mapping

Each operation maps to one or more program factories that set up kernels, circular buffers, and core grids on device. Operations with multiple factories use different paths depending on tensor layout (interleaved vs sharded), data type, or legacy vs next-gen infrastructure.

### Unary Operations (13 ops)

All unary SFPU ops share a single program factory:

| Operation | Factory |
|-----------|---------|
| RELU, ABS, EXP, RECIP, SIGMOID, GELU, TANH, SQRT, LOG, SILU, ERFINV, ASIN, I0 | `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp` |

### Binary Operations (7 ops)

Binary SFPU ops have two factory paths: the next-gen (`binary_ng`) factory and the legacy SFPU factory. Six of the seven ops exist in both; LOGADDEXP is `binary_ng`-only.

| Operation | `binary_ng` Factory | Legacy SFPU Factory |
|-----------|:-------------------:|:-------------------:|
| ADD | yes | yes |
| MUL | yes | yes |
| MAXIMUM | yes | yes |
| DIV | yes | yes |
| POWER | yes | yes |
| GCD | yes | yes |
| LOGADDEXP | yes | — |

- **binary_ng**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`
- **Legacy SFPU**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp`

### Ternary Operations (3 ops)

All ternary SFPU ops share one factory with variant-specific kernel selection (TTT, TTS, TST broadcast patterns):

| Operation | Factory |
|-----------|---------|
| WHERE, LERP, ADDCMUL | `ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_program_factory.cpp` |

### Special-Case Operations (2 ops)

| Operation | Factory | Notes |
|-----------|---------|-------|
| TYPECAST | `ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_program_factory.cpp` | Interleaved tilized layout |
| TYPECAST | `ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_sharded_program_factory.cpp` | Sharded layout |
| TYPECAST | `ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_rm_chunked_program_factory.cpp` | Row-major chunked layout |
| DROPOUT | `ttnn/cpp/ttnn/operations/experimental/dropout/device/dropout_program_factory.cpp` | Experimental; PRNG-based |

### Factory Count Summary

| Factory | # Ops | Category |
|---------|-------|----------|
| `unary_program_factory.cpp` | 13 | Unary SFPU |
| `binary_ng_program_factory.cpp` | 7 | Binary SFPU (next-gen) |
| `element_wise_multi_core_sfpu_pgm_factory.cpp` | 6 | Binary SFPU (legacy) |
| `ternary_program_factory.cpp` | 3 | Ternary SFPU |
| `typecast_*_program_factory.cpp` (×3) | 1 | Format conversion |
| `dropout_program_factory.cpp` | 1 | Experimental |

**Total**: 7 unique factories covering all 25 operations.

### Unique Factories Found (7)

| Factory | Operations |
|---------|-----------|
| `unary_program_factory.cpp` | RELU, ABS, EXP, RECIP, SIGMOID, GELU, TANH, SQRT, LOG, SILU, ERFINV, ASIN, I0 (13 ops) |
| `binary_ng_program_factory.cpp` | ADD, MUL, MAXIMUM, DIV, POWER, GCD, LOGADDEXP (7 ops) |
| `element_wise_multi_core_sfpu_pgm_factory.cpp` | ADD, MUL, MAXIMUM, DIV, POWER, GCD (6 ops — legacy path) |
| `ternary_program_factory.cpp` | WHERE, LERP, ADDCMUL (3 ops) |
| `typecast_program_factory.cpp` | TYPECAST (interleaved tilized) |
| `typecast_sharded_program_factory.cpp` | TYPECAST (sharded) |
| `typecast_rm_chunked_program_factory.cpp` | TYPECAST (row-major chunked) |
| `dropout_program_factory.cpp` | DROPOUT (experimental) |

---

## Source Documents

| Document | Operations Covered | Complexity Categories |
|----------|-------------------|----------------------|
| `docs/unary_sfpu_operations_analysis.md` | 112 unary SFPU ops | ~25 EASY, ~45 MEDIUM, ~26 HARD |
| `docs/binary_sfpu_operations_analysis.md` | 28 native SFPU + composites | 29 EASY, 14 MEDIUM, 8 HARD |
| `docs/ternary_sfpu_operations_analysis.md` | 5 ternary ops (4 device + 1 composite) | 3 EASY, varies MEDIUM, 2 HARD |
