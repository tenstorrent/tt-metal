# Quasar spec-as-key — host-dispatch results: OLD vs NEW (all optimizations)

**old** = legacy `ttnn.<op>` · **new** = `ttnn.experimental.quasar.<op>` with ALL optimizations.
Both measured **under graph-capture NO_DISPATCH** (full host path, nothing pushed to device) so the
comparison is apples-to-apples: the same ~50µs capture overhead is embedded in BOTH columns. The
**Δ (µs) is exact** (overhead cancels); the **% is conservative** (overhead inflates the denominator —
on real dispatch the same Δ is a larger %, e.g. transpose Δ+21µs is +30% here vs ~+100% of a 20µs
real-dispatch legacy op). Host dispatch µs/op, warm cache.

## Single device (1-chip)
| op | old | new | Δ | new = % of old |
|----|----:|----:|----:|----:|
| slice | 108.5 | 132.9 | +24.4 | 123% |
| transpose | 70.5 | 91.4 | +20.9 | 130% |
| tilize | 71.9 | 90.8 | +18.9 | 126% |
| fold | 66.5 | 84.7 | +18.2 | 127% |
| untilize | 70.7 | 85.9 | +15.2 | 121% |
| reshard | 75.0 | 89.9 | +14.9 | 120% |
| i2s | 72.9 | 86.8 | +13.9 | 119% |

## Multi device (32-chip, ttsim)
| op | old | new | Δ | new = % of old |
|----|----:|----:|----:|----:|
| slice | 271.5 | 303.2 | +31.7 | 112% |
| fold | 169.8 | 196.1 | +26.2 | 115% |
| transpose | 175.0 | 200.5 | +25.5 | 115% |
| tilize | 182.7 | 203.5 | +20.8 | 111% |
| untilize | 177.5 | 196.2 | +18.6 | 111% |
| i2s | 182.8 | 199.4 | +16.6 | 109% |
| reshard | 187.3 | 203.1 | +15.8 | 108% |

## Conclusion
After all optimizations, the spec path adds a **~constant ~+15–25µs** of host dispatch over legacy.
- Single chip: **+14–24µs (~+20–30% of the NO_DISPATCH measure; ~+80–100% of real-dispatch legacy)**.
- 32-chip: **+16–32µs but only ~+8–15%** — the constant tax is amortized over the per-device command
  write that dominates multi-chip host time, so the relative cost shrinks as the mesh grows.

## Per-optimization contribution (1-chip, NO_DISPATCH, µs saved; noise ±1µs)
| op | skip-validate #48138 | small-vector #48060 | CoreRangeSet | RtaName #48071 | memoize #48252 |
|----|----:|----:|----:|----:|----:|
| transpose | -23 | -1 | -2 | ~0 | ~0 |
| slice | -24 | -10 | -2 | ~0 | ~0 |
| fold | -17 | -3 | -3 | +1 | ~0 |
| tilize | -10 | -5 | -2 | ~0 | ~0 |
| untilize | -4 | ~0 | -1 | ~0 | ~0 |
| reshard | -3 | ~0 | -4 | +4 | ~0 |
| i2s | -3 | +2 | -5 | +4 | ~0 |

skip-validate = dominant win; small-vector helps arg-heavy ops; CoreRangeSet helps sharded;
RtaName/memoize neutral. builder #48250 foundational (not isolatable). galaxy #48314 not
NO_DISPATCH-measurable (needs real multi-chip).
