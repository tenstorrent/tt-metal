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

---

## Operation: leaky_relu
## Date: 2026-04-09

### Summary
Analyzed the SFPU kernel implementation for the `leaky_relu` unary operation (`UnaryOpType::LEAKY_RELU`). The operation has been fully nuked from this repository clone (Phase 2 deep nuke). Analysis was reconstructed from surviving artifacts.

### Key Findings
- **Kernel style**: B_raw_TTI (raw TTI instructions, not SFPI abstractions) -- classification from logging reference
- **Algorithm**: Piecewise-linear: `x if x >= 0, else negative_slope * x`
  - Test sign via SFPSETCC (LT0 mode)
  - Conditionally multiply negative elements by negative_slope via SFPMUL
  - Simple CC guard pattern: SFPENCC enable -> SFPSETCC -> guarded SFPMUL -> SFPENCC disable
- **Instructions**: SFPLOADI, SFPLOAD, SFPSETCC, SFPMUL, SFPENCC, SFPSTORE
- **Approximation mode**: Not applicable (simple piecewise-linear, no approximate/exact paths)
- **Address mode**: ADDR_MOD_7 with all-zero increments (identical on WH and BH)
- **Parameter**: negative_slope (float bit-cast to uint32_t, default 0.01)
- **Nuked**: All source code deleted. Analysis is a reconstruction.

### Files Produced
- `.claude-analysis/rrelu-1/leaky_relu_analysis.md`

### Status: COMPLETED (source code nuked, analysis reconstructed from artifacts)

---

## Operation: prelu_sfpu
## Date: 2026-04-09

### Summary
Analyzed the SFPU kernel implementation for the `prelu_sfpu` unary operation (`UnaryOpType::PRELU_SFPU`). The operation has been fully nuked from this repository clone (Phase 2 deep nuke, commit `8c0af4489d`). Analysis was reconstructed from documentation, the deep nuke manifest, and surviving structurally-similar operations.

### Key Findings
- **Kernel style**: A_sfpi (SFPI abstractions) -- PRELU_SFPU shared the `_calculate_lrelu_` function with LEAKY_RELU
- **Algorithm**: Piecewise-linear: `max(0, x) + weight * min(0, x)` which simplifies to `x >= 0 ? x : weight * x`
  - Load element from DEST
  - Test sign via `v_if(v < 0.0f)` (SFPSETCC LT0 mode)
  - Conditionally multiply negative elements by weight via SFPMAD
  - Store result back to DEST
- **Instructions**: SFPLOAD, SFPLOADI, SFPSETCC, SFPMAD, SFPENCC, SFPSTORE
- **Approximation mode**: `APPROXIMATION_MODE=false` (default), but unused in kernel body
- **Address mode**: `ADDR_MOD_7` with all-zero increments (identical on WH and BH)
- **Parameter**: weight (float bit-cast to uint32_t, default init 0.25 from PyTorch)
- **Shared kernel**: `_calculate_lrelu_` in `ckernel_sfpu_relu.h` was used by both LEAKY_RELU and PRELU_SFPU
- **Nuked**: All source code deleted. Analysis is a reconstruction.

### Files Produced
- `.claude-analysis/rrelu-1/prelu_sfpu_analysis.md`

### Status: COMPLETED (source code nuked, analysis reconstructed from artifacts)

---

## Operation: hardtanh
## Date: 2026-04-09

### Summary
Analyzed the SFPU kernel implementation for the `hardtanh` unary operation (`UnaryOpType::HARDTANH`). The core SFPU implementation survives in the repository. The API header and LLK dispatch layers have been removed (nuked repo).

### Key Findings
- **Kernel style**: SFPI-based (Style A) -- uses `vFloat`, `dst_reg`, `v_if`/`v_endif`
- **Algorithm**: Implements clamping via shifted arithmetic and sign-bit comparisons against 0, avoiding full-precision threshold comparisons
- **Parameters**: 3 pre-computed FP16_B values representing negated thresholds: `-(neg_threshold)`, `-(pos_threshold - neg_threshold)`, `-(pos_threshold)`
- **APPROXIMATION_MODE**: Template parameter is declared but never referenced in the function body -- has no effect
- **Address mode**: ADDR_MOD_7 with `dest.incr=0` (same for Wormhole and Blackhole)
- **Instructions**: SFPLOAD, SFPLOADI, SFPMAD (x3 for additions), SFPSETCC (x2 for conditionals), SFPENCC/SFPPUSHC/SFPPOPC (CC management), SFPSTORE

### Files Produced
- `.claude-analysis/rrelu-1/hardtanh_analysis.md`

### Status: SUCCESS
