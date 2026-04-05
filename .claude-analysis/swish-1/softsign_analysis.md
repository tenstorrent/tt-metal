# Reference Analysis: softsign

## Math Definition
softsign(x) = x / (1 + |x|)

## SFPU Kernel Pattern
- File: `ckernel_sfpu_softsign.h`
- Namespace: `ckernel::sfpu`
- Includes: `ckernel.h`, `ckernel_defs.h`, `sfpu/ckernel_sfpu_recip.h`
- Template: `<bool APPROXIMATION_MODE, int ITERATIONS = 8>`
- Function: `calculate_softsign()`
- Uses `sfpi::abs(v)` for absolute value
- Uses `_sfpu_reciprocal_<2>(denom)` for division via reciprocal
- Has `softsign_init()` that calls `_init_sfpu_reciprocal_<APPROXIMATION_MODE>()`

## Key Insight for Swish
- Shows how to compute x / f(x) pattern using reciprocal
- For swish: x / (1 + exp(-x)) could use similar reciprocal approach
- But x * sigmoid(x) = x * (1 / (1 + exp(-x))) might be more natural
