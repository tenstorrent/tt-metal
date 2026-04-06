# Reference Operation Selection for sinh

## Target Operation
- **Name**: sinh
- **Definition**: sinh(x) = (exp(x) - exp(-x)) / 2
- **Component operations identified**:
  - Negation: compute `-x` to obtain the negated argument for exp(-x)
  - Base-2 exponentiation: compute `2^(x * log2(e))` as a proxy for `exp(x)` and `exp(-x)`
  - Subtraction: `exp(x) - exp(-x)`
  - Scalar multiply: scale result by 0.5
  - Sign symmetry: sinh is an odd function; result sign matches input sign

## Context: Available Implementations

The codebase has been heavily stripped. Only the following SFPU kernels have
complete, working (non-stub) implementations in the worktree:

- **rpow** (`ckernel_sfpu_rpow.h`) — full exp_21f algorithm for base^x
- **cbrt** (`ckernel_sfpu_cbrt.h`) — full magic-constant + Newton iteration
- **hardsigmoid** (`ckernel_sfpu_hardsigmoid.h`) — piecewise linear, no exp
- **hardswish** (`ckernel_sfpu_hardswish.h`) — composite: x * hardsigmoid(x)
- **softshrink** (`ckernel_sfpu_softshrink.h`) — parameterized piecewise linear
- **hardtanh** (`ckernel_sfpu_hardtanh.h`) — parameterized clamp

Stub implementations (body removed — depend on removed exp/log/recip primitives):
- lgamma, softsign, exp2

## Selected References (ranked by relevance)

### 1. rpow
- **Why selected**: `rpow` contains the only complete working implementation of
  the `2^z` exponential function in this codebase, using the `exp_21f` algorithm
  from Moroz et al. 2022. Since `exp(x) = 2^(x * log2(e))`, this algorithm is
  directly reusable to compute both `exp(x)` and `exp(-x)` needed by sinh.
  The implementation shows the exact SFPU instruction sequence: `addexp` for
  scaling by 2^23, bias addition, `exexp`/`exman9` for mantissa/exponent
  extraction, and a Horner-form polynomial for `2^frac(z)`.
- **Relevance**: high — the exp_21f algorithm in `calculate_rpow` is the direct
  building block for computing exp(x) and exp(-x) in sinh

### 2. cbrt
- **Why selected**: `cbrt` demonstrates two patterns essential for sinh: (a) use
  of `sfpi::setsgn(d, a)` to copy the sign of the original input into the result,
  which is directly needed since sinh is an odd function (sign of output = sign of
  input), and (b) use of `sfpi::abs(a)` to work with the magnitude before restoring
  the sign, (c) use of `sfpi::addexp()` for exponent-bit manipulation, and (d)
  a multi-step Newton-Raphson refinement pattern with `vConstFloatPrgm0/1/2`
  constants loaded in the `_init` function.
- **Relevance**: high — setsgn pattern for sign-symmetry and addexp usage directly
  applicable to sinh's implementation structure

### 3. hardsigmoid
- **Why selected**: `hardsigmoid` is the cleanest minimal SFPU kernel in the
  codebase. It shows the canonical structure: no-param init function, a single
  `#pragma GCC unroll 8` loop, loading from `sfpi::dst_reg[0]`, computing a
  result, and storing back. This is the structural template that `calculate_sinh`
  should follow for the outer loop, header style, and namespace.
- **Relevance**: medium — provides the code skeleton and coding conventions that
  sinh should replicate

### 4. hardswish
- **Why selected**: `hardswish` is implemented as a composite operation:
  it computes an intermediate value (`hsigmoid`) and then multiplies the original
  input `x` by that intermediate. This composite pattern — "compute a subexpression
  using the input, then combine with the original input via arithmetic" — directly
  mirrors what sinh must do: compute exp(x), compute exp(-x), subtract, then scale.
  The kernel also shows how to handle a running intermediate (`sfpi::vFloat hsigmoid`)
  within the loop alongside the original value `x`.
- **Relevance**: medium — composite subexpression-then-combine pattern matches sinh

### 5. softshrink
- **Why selected**: `softshrink` is the simplest complete parameterized kernel
  and provides clean reference for: (a) decoding a parameter from IEEE 754 bits
  using `Converter::as_float(param0)`, (b) computing a negated version of a
  constant (`-lambda_val`), (c) performing arithmetic on the decoded value within
  the loop. For sinh, the implementor needs to load constants (log2(e) = ~1.4427,
  scaling factor 0.5) and the softshrink pattern shows the cleanest example of
  constant setup and use.
- **Relevance**: medium — parameter decoding pattern and constant-setup idioms
  for the init/compute function pair
