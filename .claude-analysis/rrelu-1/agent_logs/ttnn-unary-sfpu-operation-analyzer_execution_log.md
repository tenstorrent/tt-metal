# Execution Log: ttnn-unary-sfpu-operation-analyzer

## Operation: swish
## Date: 2026-04-09

### Summary
Analyzed the SFPU kernel implementation for the `swish` unary operation (`UnaryOpType::SWISH`).

### Key Findings
- **Kernel style**: SFPI-based (Style A) -- uses `vFloat`, `dst_reg`, `v_if`/`v_endif`, `sfpi::abs`
- **Algorithm**: Hybrid 3-segment sigmoid approximation:
  - Segment 0 (|x| <= 2.5): degree-3 polynomial via Horner's method
  - Segment 1 (2.5 < |x| <= 5.0): linear interpolation
  - Segment 2 (|x| > 5.0): saturation to 1.0
  - Negative correction: sigmoid(x) = 1 - sigmoid(|x|) for x < 0
  - Final: swish(x) = x * sigmoid(x)
- **Approximation mode**: `APPROXIMATION_MODE` template parameter exists but is never referenced in the function body
- **Address mode**: `ADDR_MOD_7` with all-zero increments (identical on WH and BH)
- **Architecture**: WH and BH implementations are identical

### Files Produced
- `.claude-analysis/rrelu-1/swish_analysis.md`

### Status: SUCCESS
