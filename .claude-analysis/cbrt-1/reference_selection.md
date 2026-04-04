# Reference Selection for cbrt

## Selected References

1. **sqrt** - Square root operation. Same class of root functions; uses SFPU hardware intrinsics; closest mathematical relative (sqrt = x^(1/2), cbrt = x^(1/3)).
2. **rsqrt** - Inverse square root. Demonstrates Newton-Raphson refinement pattern on root operations.
3. **silu** - SiLU (x * sigmoid(x)). Clean, simple unary SFPU example. Shows the standard init/compute/tile pattern.
4. **exp2** - Exp base-2. Another simple non-parametrized SFPU op, demonstrates basic macro include pattern.
5. **power** - Power function. Relevant because cbrt can be expressed as pow(x, 1/3).

## Selection Rationale

cbrt (cube root, x^(1/3)) is a non-parametrized unary operation that operates on the SFPU vector unit. The pre-nuke implementation used a specialized algorithm based on Moroz et al.'s magic constant approach, similar in spirit to the fast inverse square root algorithm. The selected references cover:

- **Mathematical similarity**: sqrt, rsqrt, power are all root/exponent operations
- **Pattern examples**: silu, exp2 demonstrate the clean SFPU operation registration pattern
- **All 5 are non-parametrized or simple** (matching cbrt's non-parametrized nature)

SELECTED_REFERENCES: sqrt, rsqrt, silu, exp2, power
