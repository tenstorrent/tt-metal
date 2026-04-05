# Reference Analysis: hardsigmoid

## Math Definition
hardsigmoid(x) = max(0, min(1, x/6 + 0.5))

## SFPU Kernel Pattern
- File: `ckernel_sfpu_hardsigmoid.h`
- Namespace: `ckernel::sfpu`
- Template: `<bool APPROXIMATION_MODE, int ITERATIONS = 8>`
- Function: `calculate_hardsigmoid()`
- Simple arithmetic with clamping
- Shows v_if/v_endif pattern for conditional logic

## Key Insight for Swish
- Shows sigmoid-like function pattern
- Demonstrates clamping to [0, 1] range
