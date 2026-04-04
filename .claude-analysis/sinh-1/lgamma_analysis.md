# SFPU Operation Analysis: lgamma

## Overview
- **Operation**: lgamma (log gamma)
- **Math**: lgamma(x) = ln(|Gamma(x)|)
- **Parameters**: None

## Architecture
Uses the LLK dispatch pattern:
1. SFPU kernel: `ckernel_sfpu_lgamma.h`
2. LLK dispatch: `llk_math_eltwise_unary_sfpu_lgamma.h`
3. Compute API: `lgamma.h` calls `llk_math_eltwise_unary_sfpu_lgamma<APPROX>(idst)`
4. Split includes: `SFPU_OP_LGAMMA_INCLUDE`

## Key Takeaway
- Shows the LLK intermediary pattern (alternative to direct macro dispatch)
- Full modern registration stack
