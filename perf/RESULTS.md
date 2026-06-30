# Quasar spec-as-key — host-dispatch results: CURRENT MAIN vs SPEC-FACTORY (all optimizations)

**main** = the op as it exists in `origin/main` today (legacy / MetalV2 path — zero spec-as-key
optimizations). **spec+opts** = `ttnn.experimental.quasar.<op>` with all optimizations **except memoize**
(#48252 closed — measured 0 host gain). Both measured under graph-capture **NO_DISPATCH** (full host
path, nothing to device) so it's apples-to-apples: the ~50µs capture overhead is in BOTH columns, so the
**Δ (µs) is exact**; the **% is conservative** (overhead inflates the denominator). Host µs/op, warm cache.

Baseline verified: built actual `origin/main` and measured — legacy transpose 73.5µs, MetalV2 quasar
transpose 71.5µs, both match the "main" column (70.5µs) within ~3µs cross-build noise. The baseline is
genuinely unoptimized.

## Single device (1-chip)
| op | main | spec+opts | Δ | spec = % of main |
|----|----:|----:|----:|----:|
| slice | 108.5 | 132.9 | +24.4 | 123% |
| transpose | 70.5 | 91.4 | +20.9 | 130% |
| tilize | 71.9 | 90.8 | +18.9 | 126% |
| fold | 66.5 | 84.7 | +18.2 | 127% |
| untilize | 70.7 | 85.9 | +15.2 | 121% |
| reshard | 75.0 | 89.9 | +14.9 | 120% |
| i2s | 72.9 | 86.8 | +13.9 | 119% |

## Multi device (32-chip, ttsim)
| op | main | spec+opts | Δ | spec = % of main |
|----|----:|----:|----:|----:|
| slice | 271.5 | 303.2 | +31.7 | 112% |
| fold | 169.8 | 196.1 | +26.2 | 115% |
| transpose | 175.0 | 200.5 | +25.5 | 115% |
| tilize | 182.7 | 203.5 | +20.8 | 111% |
| untilize | 177.5 | 196.2 | +18.6 | 111% |
| i2s | 182.8 | 199.4 | +16.6 | 109% |
| reshard | 187.3 | 203.1 | +15.8 | 108% |

## Conclusion
The fully-optimized spec factory adds a **~constant +14–24µs** of host dispatch over current main.
- Single chip: **+14–24µs (~119–130% of main)**.
- 32-chip: **+16–32µs but only ~108–115%** — the constant tax is amortized over the per-device command
  write that dominates multi-chip host time, so the relative cost shrinks as the mesh grows.

## Per-optimization contribution (1-chip, NO_DISPATCH, µs saved off the spec path; noise ±1µs)
| op | skip-validate #48138 | small-vector #48060 | CoreRangeSet #48619 | RtaName #48071 |
|----|----:|----:|----:|----:|
| transpose | -23 | -1 | -2 | ~0 |
| slice | -24 | -10 | -2 | ~0 |
| fold | -17 | -3 | -3 | +1 |
| tilize | -10 | -5 | -2 | ~0 |
| untilize | -4 | ~0 | -1 | ~0 |
| reshard | -3 | ~0 | -4 | +4 |
| i2s | -3 | +2 | -5 | +4 |

skip-validate = dominant win; small-vector helps arg-heavy ops; CoreRangeSet helps sharded; RtaName
neutral (slightly hurts sharded — short arg names). memoize DROPPED (0 value). builder foundational
(not isolatable). galaxy parallel-write (#48314) not NO_DISPATCH-measurable (needs real multi-chip).
