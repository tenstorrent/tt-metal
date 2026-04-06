# Reference Operation Selection for frac

## Target Operation
- **Name**: frac
- **Definition**: frac(x) = x - floor(x)
- **Component operations identified**:
  - **trunc / floor** — PyTorch `torch.frac()` semantics use `x - trunc(x)` (sign-preserving), so `frac(-1.5) = -0.5`; confirmed by build artifact `ckernel_sfpu_frac.h`
  - **IEEE 754 exponent extraction** — use `sfpi::exexp(v)` to determine whether `x` has fractional bits
  - **Mantissa bit masking** — zero the lower `(23 - E)` mantissa bits via `sfpi::shft` + bitwise AND to produce `trunc(x)`
  - **Float reinterpret** — `sfpi::reinterpret<sfpi::vInt>` / `sfpi::reinterpret<sfpi::vFloat>` to do integer bit operations on float values
  - **Float subtraction** — `v - trunc_val`
  - **Three-way conditional** — `v_if/v_endif` blocks for E < 0 (entire value is fractional), 0 <= E < 23 (mixed), E >= 23 (exact integer, result = 0)

## Selected References (ranked by relevance)

### 1. cbrt
- **Why selected**: `ckernel_sfpu_cbrt.h` is the primary structural reference because it uses `sfpi::exexp(v)` to extract the debiased exponent and then `sfpi::reinterpret<sfpi::vInt>` / `sfpi::reinterpret<sfpi::vFloat>` for IEEE 754 bit manipulation — exactly the same primitives that `frac` requires to extract and mask the integer part of the mantissa. It also uses `sfpi::int32_to_float`, `sfpi::setsgn`, and the standard `#pragma GCC unroll 8` loop over `sfpi::dst_reg`. No `vConstFloatPrgm` init constants are needed (matches `frac` which needs no init).
- **Relevance**: High — provides the exact SFPU intrinsic pattern (`exexp` + `reinterpret` bit ops) that is the core of the `frac` implementation; implementor should follow the `sfpi::exexp` + `sfpi::reinterpret<sfpi::vInt>` / `sfpi::reinterpret<sfpi::vFloat>` idiom from this file

### 2. hardtanh
- **Why selected**: `ckernel_sfpu_hardtanh.h` is the cleanest example of the standard `ckernel_sfpu_*.h` file structure: `#pragma GCC unroll 8` loop, `sfpi::vFloat val = sfpi::dst_reg[0]`, `v_if / v_endif` conditionals, and `sfpi::dst_reg[0] = result; sfpi::dst_reg++` store pattern. Also shows how `Converter::as_float` is used to decode float parameters from `uint32_t` bits — a technique also used in `softshrink` and `rpow` and worth understanding even though `frac` has no parameters.
- **Relevance**: High — directly informs the boilerplate file structure, loop pattern, and `v_if/v_endif` conditional idiom that every `ckernel_sfpu_*.h` file must follow

### 3. hardsigmoid
- **Why selected**: `ckernel_sfpu_hardsigmoid.h` is a parameterless operation with no `vConstFloatPrgm` init, which matches `frac` exactly. It demonstrates how to use `sfpi::vConst1` for comparisons, how to default a `result` variable and then override it inside conditionals, and how to write the `hardsigmoid_init()` stub (no-op init). Structural simplicity makes it easy to follow as a template for the `frac_init()` function.
- **Relevance**: High — informs the no-parameter / no-const-register init pattern and the `result` variable + conditional override pattern used in `calculate_frac()`

### 4. hardswish
- **Why selected**: `ckernel_sfpu_hardswish.h` demonstrates the composite computation pattern where an intermediate value is computed from `x` (here `hsigmoid = x/6 + 0.5`, clamped) and then combined with the original `x` to produce the final result (`x * hsigmoid`). This directly parallels `frac` which computes an intermediate `trunc_val` by masking mantissa bits and then produces `result = v - trunc_val`. The file also shows how two sequential conditional clamp branches interact cleanly in the `v_if/v_endif` model.
- **Relevance**: Medium-High — models the compute-intermediate-then-subtract-from-original pattern; implementor should adapt the intermediate-computation + final-combination structure to produce `trunc_val` and `v - trunc_val`

### 5. softshrink
- **Why selected**: `ckernel_sfpu_softshrink.h` demonstrates a three-case conditional structure with a pre-initialized default result (`sfpi::vFloat result = 0.0f`) that is selectively overridden by `v_if/v_endif` branches for different input ranges. This structural pattern maps closely onto `frac`'s three cases: (1) E >= 23 → result = 0 (default override), (2) E < 0 → result = v (default), (3) 0 <= E < 23 → result = v - trunc_val (inner nested conditional). The pattern of initializing `result` and then conditionally updating it is exactly what `calculate_frac()` uses.
- **Relevance**: Medium-High — informs the three-branch conditional structure with a default result value and selective `v_if/v_endif` overrides; simpler than `cbrt` and thus easier for an implementor to adapt
