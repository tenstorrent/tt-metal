# Reference Operation Selection for hardswish

## Target Operation
- **Name**: hardswish
- **Definition**: x * min(max(x + 3, 0), 6) / 6
- **Component operations identified**:
  - Scalar addition: `x + 3`
  - Lower clamp (ReLU-like): `max(..., 0)` via `v_if(result < 0) { result = 0 }`
  - Upper clamp: `min(..., 6)` via `v_if(result > 6) { result = 6 }`
  - Scalar multiply by constant: `/ 6` (equivalently `* one_sixth`)
  - Multiply original input by clamped-and-scaled result: `x * clamp_result`
  - Three-region piecewise linear (x <= -3 maps to 0, x >= 3 maps to x, linear in between)

## Selected References (ranked by relevance)

### 1. hardsigmoid
- **Why selected**: hardswish(x) = x * hardsigmoid(x) by definition. The hardsigmoid kernel in `ckernel_sfpu_hardsigmoid.h` already computes `clamp(x/6 + 0.5, 0, 1)`, which is algebraically identical to `min(max(x + 3, 0), 6) / 6`. The SFPU instruction pattern — add offset, multiply by one_sixth, clamp to [0,1] using two sequential `v_if`/`v_endif` blocks — is verbatim reusable. The hardswish kernel is a minimal extension: preserve original x, compute hardsigmoid, multiply the two together.
- **Relevance**: high — the inner computation of hardswish is the hardsigmoid kernel unchanged; only the final multiply-by-x is new.

### 2. silu
- **Why selected**: silu (SiLU/Swish) is structurally the closest analog at the algorithm level: `silu(x) = x * sigmoid(x)` and `hardswish(x) = x * hardsigmoid(x)`. Both are "gated identity" operations where the input multiplies a squashed version of itself. The silu kernel demonstrates exactly how to preserve the original x value in `sfpi::vFloat`, compute the activation sub-expression, and then perform the final `x * activation` multiply before storing to `dst_reg`. This composite pattern is the primary structural template for hardswish.
- **Relevance**: high — provides the complete composite x*activation(x) template showing register management, intermediate storage, and final multiply.

### 3. hardtanh
- **Why selected**: hardtanh implements double-sided clamping using the exact same `v_if`/`v_endif` SFPU idiom that hardswish requires. The kernel applies `v_if(val < min_val) { val = min_val }` then `v_if(val > max_val) { val = max_val }` — the same two sequential conditional blocks needed in hardswish to enforce `max(0, ...)` and `min(6, ...)`. Although hardtanh uses runtime parameters while hardswish uses fixed constants, the SFPU branching structure is identical.
- **Relevance**: high — supplies the exact two-sided clamping SFPU pattern (v_if lower bound, v_if upper bound) central to hardswish.

### 4. selu
- **Why selected**: selu demonstrates the pattern of loading x, conditionally transforming it in a `v_if` branch, and then unconditionally applying a scalar multiply (`v = v_scale * v`). In hardswish the analogous pattern is: compute the piecewise linear expression (conditional), then unconditionally multiply by x. selu also shows how fixed-constant scalars are loaded via `Converter::as_float` for use in the loop, which informs how to handle the `1/6` constant in hardswish.
- **Relevance**: medium — provides the preserve-then-multiply pattern and constant-loading via Converter::as_float, applicable to hardswish's unconditional final multiply.

### 5. softsign
- **Why selected**: softsign(x) = x / (1 + |x|) = x * recip(1 + |x|) demonstrates the clean two-step pattern: compute a scalar transform of x into an intermediate, then multiply by x. Despite using a reciprocal rather than clamping, the `v * recip` final step is the same structural template as hardswish's `x * clamped_expression`. softsign is also one of the simplest composite operations in the codebase, making it a clear structural guide for an operation of similar complexity.
- **Relevance**: medium — confirms the x-times-transform composite idiom and provides a clean minimal template for a non-transcendental, multiply-terminated operation.
