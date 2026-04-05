# Reference Selection: sinh

## Operation
- **Name**: sinh
- **Math definition**: sinh(x) = (exp(x) - exp(-x)) / 2

## Selected References

| Rank | Operation | Rationale |
|------|-----------|-----------|
| 1 | **cosh** | Sister hyperbolic function with nearly identical `(exp(x) + exp(-x)) / 2` structure -- only the sign differs |
| 2 | **selu** | Uses `exp()` with scaling/shifting, demonstrates SFPU exp + arithmetic patterns |
| 3 | **elu** | Uses `exp(x) - 1` with conditional logic, shows exp-based subtraction patterns |
| 4 | **lgamma** | Complex multi-step SFPU computation, demonstrates chaining multiple SFPU ops |
| 5 | **rpow** | Shows exponential computation patterns and per-tile arithmetic |

## Selection Methodology
- Prioritized operations with similar mathematical structure (exp-based)
- cosh is the closest match: identical formula except for sign
- selu and elu show exp usage patterns in SFPU kernels
- lgamma demonstrates complex multi-step SFPU computations
- rpow shows exponential/power patterns
