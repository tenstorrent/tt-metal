# Reference Operation Selection for cbrt

## Target Operation
- **Name**: cbrt
- **Definition**: x^(1/3), cbrt(-x) = -cbrt(x). Uses IEEE 754 bit manipulation for initial estimate, then Newton-Raphson refinement.
- **Component operations identified**: IEEE 754 bit manipulation (reinterpret float as int, shift, subtract from magic constant), Newton-Raphson iteration (multiply, fused multiply-add), sign extraction and restoration (`setsgn`/`abs`), conditional branching for negative inputs, `reinterpret<vFloat>` / `reinterpret<vInt>` for bit-level access, `float_to_fp16b` for bfloat16 output rounding.

## Selected References (ranked by relevance)

### 1. sqrt
- **Why selected**: `sqrt` uses exactly the same algorithmic skeleton as `cbrt`: (1) IEEE 754 integer bit-trick to get an initial approximation (`vConstIntPrgm0 - (bits >> 1)`), (2) one or two Newton-Raphson refinement iterations using SFPI multiply and fused multiply-add, (3) special-case handling for x < 0 via a `v_if` branch, (4) optional `float_to_fp16b` rounding for bfloat16 destinations, (5) `_init_sqrt_` loading magic constants into `vConstIntPrgm0` / `vConstFloatPrgm1` / `vConstFloatPrgm2`. The cbrt implementation needs all of these pieces, adapting the magic constant (0x2A5137A4 for cube-root vs 0x5F1110A0 for square-root) and the Newton-Raphson formula (y_new = y*(4/3 - x*y^3/3) for cbrt vs y*(1.5 - 0.5*x*y^2) for rsqrt). The two-iteration non-approximate path in `_calculate_sqrt_body_` is the closest structural template.
- **Relevance**: high — direct algorithmic template: bit-trick initial estimate + Newton-Raphson + sign handling + bfloat16 rounding

### 2. rsqrt
- **Why selected**: `rsqrt` reuses `_calculate_sqrt_body_<APPROXIMATION_MODE, RECIPROCAL=true>` from `ckernel_sfpu_sqrt.h`. It demonstrates how the same bit-trick initial estimate can be adapted for a different root (reciprocal square root), showing the pattern of switching between `RECIPROCAL` and non-reciprocal via template parameters. For cbrt, a similar "cbrt of absolute value, then restore sign" structure is needed. The `rsqrt_compat_` path in `ckernel_sfpu_rsqrt_compat.h` also shows an alternative Newton-Raphson structure using `setexp` and `exexp` to manipulate the exponent, which is directly applicable to cbrt's exponent-based initial estimate (dividing the biased exponent by 3 instead of 2).
- **Relevance**: high — shows how to adapt the bit-trick + Newton-Raphson pattern for a non-square-root, and the `_reciprocal_compat_` shows exponent manipulation via `setexp`/`exexp` that cbrt can use for the initial approximation

### 3. frac
- **Why selected**: `frac` is the primary example in this worktree of using `sfpi::reinterpret<sfpi::vInt>`, `sfpi::exexp`, bitmask construction from integer arithmetic (`sfpi::vInt(-1) << shift`), and bit-masking (`xi & mask`) to do IEEE 754 float decomposition. The cbrt initial approximation requires exactly this kind of bit-level reinterpretation: reading the float bits as an integer, performing integer arithmetic on them (divide by 3 via right-shift and multiply, subtract from magic constant), then reinterpreting the result back as a float. `frac` provides the cleanest local reference for this SFPI bit-manipulation idiom.
- **Relevance**: high — canonical local worktree example of `reinterpret<vInt>`, `exexp`, and integer bit-masking on float values in the SFPI framework

### 4. atanh
- **Why selected**: `atanh` implements a custom mathematical function using IEEE 754 float decomposition via `sfpi::exexp` (to extract unbiased exponent) and `sfpi::setexp` (to normalize the mantissa to [1,2)), followed by a Horner polynomial evaluation and integer-to-float conversion (`sfpi::int32_to_float`). The `atanh_init()` function shows loading multiple polynomial coefficients into `vConstFloatPrgm0/1/2`. This pattern is relevant because the cbrt Newton-Raphson refinement involves polynomial-like expressions (y*(4/3 - x*y^3/3)) that may benefit from the Horner form, and cbrt's initial estimate uses `exexp`/`setexp` idioms for the exponent division by 3. The `atanh_init` also shows the correct way to initialize programmable constants for a custom SFPU operation.
- **Relevance**: medium — demonstrates `exexp`/`setexp`/`int32_to_float` for custom IEEE 754 math, and the `init()` pattern for loading constants into `vConstFloatPrgm*` registers

### 5. sinh
- **Why selected**: `sinh` handles odd symmetry: `sinh(-x) = -sinh(x)`. The cbrt math definition explicitly states `cbrt(-x) = -cbrt(x)`, the same odd-function property. The `sinh` implementation uses `sfpi::setsgn(x, 0)` to take absolute value and then branches on sign. For cbrt, the negative-input handling is a critical correctness concern — `sqrt` returns NaN for negative inputs while `cbrt` must return a negative result. The `sinh` reference demonstrates how to handle odd-symmetry correctly: extract absolute value, compute on positive input, then restore sign. The `v_if(abs_x < threshold)` conditional branch pattern for a small-x Taylor fallback also illustrates the structure cbrt may need if it uses a small-x polynomial fallback.
- **Relevance**: medium — shows the odd-symmetry sign-handling pattern (`setsgn`, `abs`, sign restoration) that cbrt requires for negative inputs, which is absent from `sqrt` and `rsqrt`
