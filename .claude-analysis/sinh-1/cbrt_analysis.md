# SFPU Operation Analysis: cbrt

## Overview
- **Operation**: cbrt (cube root)
- **Math**: cbrt(x) = x^(1/3)
- **Parameters**: None

## Key Patterns
- Uses a different LLK dispatch pattern (llk_math_eltwise_unary_sfpu_cbrt.h)
- Registered in TTNN_ELTWISE_UNARY_CPP_FUNCTIONS list (not separate golden)
- Simple no-parameter operation

## Key Takeaway
- Shows the simpler registration for parameter-free operations
- Uses the TTNN_ELTWISE_UNARY_CPP_FUNCTIONS pattern for golden functions
