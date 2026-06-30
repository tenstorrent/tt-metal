# Quasar spec-as-key — host-dispatch results: CURRENT MAIN vs SPEC-FACTORY (all optimizations)

**main** = the op as it exists in `origin/main` today (legacy / MetalV2 path — zero spec-as-key
optimizations). **spec+opts** = `ttnn.experimental.quasar.<op>` with all optimizations **except memoize**
(#48252 closed — measured 0 host gain). Both measured under graph-capture **NO_DISPATCH** (full host
path, nothing to device) so it's apples-to-apples: the ~50µs capture overhead is in BOTH columns, so the
**Δ (µs) is exact**; the **% is conservative** (overhead inflates the denominator). Host µs/op, warm cache.

Baseline verified: built actual `origin/main` and measured — legacy transpose 73.5µs, MetalV2 quasar
transpose 71.5µs, both match the "main" column (70.5µs) within ~3µs cross-build noise. The baseline is
genuinely unoptimized.

## Single device (1-chip) — both lenses side by side

`real` = real-dispatch absolutes (n150). `ND` = NO_DISPATCH. **The µs Δ is the same in both** (~+20µs).
The **ratio differs** only because NO_DISPATCH adds a ~50µs graph-capture pedestal to *both* columns:
it cancels in the Δ but pads the ratio denominator, so the same +20µs reads as ~2x on `real`
(base ~20µs) and ~1.3x on `ND` (base ~70µs). **`real %` is the true single-chip slowdown.**

| op | real main | real spec | real % | | ND main | ND spec | ND % | Δ (µs) |
|----|----:|----:|----:|--|----:|----:|----:|----:|
| tilize | 21.8 | 45.5 | **209%** | | 71.9 | 90.8 | 126% | +24 |
| transpose | 19.9 | 40.3 | **202%** | | 70.5 | 91.4 | 130% | +20 |
| fold | 19.2 | 35.8 | **187%** | | 66.5 | 84.7 | 127% | +17 |
| untilize | 19.4 | 35.9 | **185%** | | 70.7 | 85.9 | 121% | +17 |
| i2s | 21.0 | 35.6 | **170%** | | 72.9 | 86.8 | 119% | +15 |
| reshard | 22.6 | 37.3 | **165%** | | 75.0 | 89.9 | 120% | +15 |
| slice | 41.2 | 64.8 | **157%** | | 108.5 | 132.9 | 123% | +24 |

> Why "2x here but 1.3x there": same physical cost (+~20µs), two lenses. `real` has no overhead so the
> ratio is true (~2x on a ~20µs op). `ND`'s pedestal inflates the denominator, deflating the ratio. We use
> `ND` only for 32-chip (real dispatch backpressures the 0.5KHz sim); there the base is big (~175µs) so the
> +20µs is ~+11% under BOTH lenses — they converge.


## Multi device (32-chip, ttsim) — main = actual `origin/main` build
| op | main | spec+opts | Δ | spec = % of main |
|----|----:|----:|----:|----:|
| slice | 273.2 | 303.2 | +30.0 | 111% |
| transpose | 176.8 | 200.5 | +23.7 | 113% |
| tilize | 181.2 | 203.5 | +22.3 | 112% |
| fold | 174.8 | 196.1 | +21.3 | 112% |
| untilize | 176.3 | 196.2 | +19.9 | 111% |
| reshard | 191.6 | 203.1 | +11.5 | 106% |
| i2s | 190.3 | 199.4 | +9.1 | 105% |

(main and spec are different builds, so this is cross-build: the legacy control reads 176.8 on the
main build vs 175.0 on the spec build → ~2µs cross-build offset on top of the sim's ±7µs run-noise.)


## Conclusion
The fully-optimized spec factory adds a **~constant +14–24µs** of host dispatch over current main.
- Single chip: **+14–24µs = ~1.6–2.1x main** (real dispatch; the constant tax is a big % on a small op).
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
