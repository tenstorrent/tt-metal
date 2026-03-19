# Issues Log: smelu

## Configuration
- **Operation**: smelu
- **Math definition**: SmeLU(x, β) = x if x ≥ β; (x + β)² / (4β) if |x| ≤ β; 0 if x < -β
- **Source**: https://arxiv.org/abs/2202.06499
- **Output folder**: `.claude/analysis/smelu-1/`
- **Date**: 2026-03-19

## Phase Timeline
| Phase | Description | Status | Duration | Issues |
|-------|-------------|--------|----------|--------|
| 1 | Reference Discovery | ok | ~99s | 0 |
| 2 | Reference Analysis | ok | ~942s (wall) | 0 |
| 3 | Implementation | ok | ~386s | 0 |
| 4 | Testing & Debugging | ok | ~1124s | 2 (env workarounds, not smelu bugs) |
| 5 | Documentation | ok | ~10s | 0 |

## Issues
| # | Phase | Severity | Description | Resolution |
|---|-------|----------|-------------|------------|
| 1 | 4 | LOW | Root conftest.py import failure (pre-existing) | Self-contained test with --noconftest |
| 2 | 4 | LOW | beta keyword-only arg in nanobind | Changed to ttnn.smelu(tt_input, beta=beta) |
| 3 | 4 | LOW | Stale tt_llk submodule | Updated to commit 59ea0128 |
