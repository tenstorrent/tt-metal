# Execution Log — ttnn-unary-sfpu-reference-discoverer

## Metadata
- **Operation**: swish
- **Agent**: ttnn-unary-sfpu-reference-discoverer
- **Status**: SUCCESS
- **Date**: 2026-04-06

## Input Interpretation

| Field | Value | Confidence |
|-------|-------|------------|
| operation_name | swish | HIGH — explicitly stated |
| math_definition | x / (1 + exp(-x)) | HIGH — explicitly stated |
| output_folder | .claude-analysis/swish-2/ | HIGH — explicitly stated |
| predecessor | ttnn-unary-sfpu-operation-generator | HIGH |

## Execution Timeline

### Phase 1: Initialize and Read Logging References
- Read `sfpu-reference-discoverer.md` and `common.md` logging references
- Initialized breadcrumbs
- Logged `start` event

### Phase 2: Read Source Files
Files read in this worktree:
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` — op registration
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp` — parameterized vs default ops
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` — full UnaryOpType enum
- All 8 `ckernel_sfpu_*.h` files in wormhole_b0 LLK directory
- All 10 `llk_math_eltwise_unary_sfpu_*.h` files in wormhole_b0 LLK directory
- `tt_metal/hw/inc/api/compute/eltwise_unary/` API headers
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_unary_sfpu_api.h`

### Phase 3: Math Decomposition

swish(x) = x / (1 + exp(-x)) = x * sigmoid(x)

Components identified:
1. `exp(-x)` — exponential of negated input (hardest part, needs exp2 algorithm)
2. `1 + exp(-x)` — scalar addition to produce denominator
3. `1 / (...)` — reciprocal (sigmoid)
4. `x * sigmoid(x)` — final multiply

Key insight: swish is mathematically identical to `silu(x)` already in UnaryOpType::SILU,
but swish needs a custom SFPI kernel not relying on hardware sigmoid_tile LLK.

### Phase 4: Candidate Identification

All 7 available custom ckernel operations in the worktree were evaluated:
- **hardswish** — x * clamp(x/6+0.5, 0, 1): exact x*f(x) template
- **hardsigmoid** — clamp(x/6+0.5, 0, 1): inner sigmoid analog
- **rpow** — base^x = 2^(x*log2(base)): exp2 algorithm for exp(-x)
- **cbrt** — cube root via magic constant + Newton-Raphson
- **softsign** — x/(1+|x|): structural analog (removed stub)
- **softshrink** — piecewise with lambda parameter: parameterized kernel pattern
- **hardtanh** — clamp(x, min, max): simple clamping

### Phase 5: Ranking and Selection

| Rank | Operation | Rationale |
|------|-----------|-----------|
| 1 | hardswish | x*f(x) structural template, directly adapted by replacing hardsigmoid with sigmoid |
| 2 | hardsigmoid | Inner activation component showing how sigmoid sub-function is computed |
| 3 | rpow | Complete exp2 algorithm (exp_21f) needed for exp(-x) in sigmoid computation |
| 4 | softsign | x/(1+|x|) is the closest structural analog to x/(1+exp(-x)) |
| 5 | cbrt | Polynomial approximation pattern with vConstFloatPrgm registers for init()/calculate() |

## Deviations
None.

## Artifacts Created
- `.claude-analysis/swish-2/reference_selection.md`
- `.claude-analysis/swish-2/agent_logs/ttnn-unary-sfpu-reference-discoverer_breadcrumbs.jsonl`
- `.claude-analysis/swish-2/agent_logs/ttnn-unary-sfpu-reference-discoverer_execution_log.md`

## Handoff Notes for ttnn-unary-sfpu-operation-analyzer
The 5 reference operations to analyze are: hardswish, hardsigmoid, rpow, softsign, cbrt.

Key implementation insight for the analyzer: swish = x * sigmoid(x). The implementor
should:
1. Copy the hardswish loop structure
2. Replace the `hardsigmoid` linear approximation with sigmoid(x) computed via:
   - Negate x → compute exp(-x) using rpow's exp2 algorithm (with log2(e) = 1.4426950408)
   - Add 1 → compute reciprocal to get sigmoid
3. Multiply result by original x

The softsign stub's note "depends on recip primitive" informs that reciprocal within
SFPI requires the exp2 bit-manipulation approach from rpow rather than a hardware call.

## Discovery Rationale — SFPU-Specific Sections

### Formula Decomposition Quality
- swish = x / (1 + exp(-x)) cleanly decomposed into 4 sub-operations
- Key observation: exp(-x) is the hardest sub-operation; rpow provides a complete solution

### Reference Quality Assessment
- All 5 references are FROM THE CURRENT WORKTREE — no external sources used
- hardswish and hardsigmoid are pair references (one uses the other) — maximum synergy
- rpow provides the only complete exp implementation in the worktree's custom kernel layer
- softsign's stub + comment guides the implementor to avoid the recip dead-end
- cbrt provides an alternative approximation strategy if needed
