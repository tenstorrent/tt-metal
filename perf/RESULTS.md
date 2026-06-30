# Quasar spec-as-key — host-dispatch results (old vs new)

**old** = legacy `ttnn.<op>` · **new** = `ttnn.experimental.quasar.<op>` with ALL optimizations
(spec migration + run-args builder + small-vector Tables + RtaName + skip-validate + memoize + CoreRangeSet).
Host dispatch µs/op, warm cache.

## Single device (n150, REAL dispatch, amortized, ±0.5%) — definitive
| op | old | new | new = % of old |
|----|----:|----:|----:|
| tilize | 21.8 | 45.5 | 209% |
| transpose | 19.9 | 40.3 | 202% |
| fold | 19.2 | 35.8 | 187% |
| untilize | 19.4 | 35.9 | 185% |
| i2s | 21.0 | 35.6 | 170% |
| reshard | 22.6 | 37.3 | 165% |
| slice | 41.2 | 64.8 | 157% |

**Single-chip conclusion: fully optimized, new is ~1.6–2.1× old (avg ~1.8×).** A host-side
regression; the optimizations reduce it but do not close it.

## Multi device (32-chip, ttsim — INDICATIVE only; no real galaxy, sim is noisy)
| op | old | new | new = % of old |
|----|----:|----:|----:|
| transpose | 132 | 152 | 115% |
| reshard | 78 | 89 | 114% |
| i2s | 74 | 89 | 120% |
| untilize | 118 | 146 | 124% |
| fold | 74 | 93 | 126% |

**Multi-chip conclusion: the regression shrinks to ~1.15–1.26× old** — the spec tax is a roughly
constant absolute cost (paid once per dispatch), amortized over the per-device command write that
dominates Galaxy host time. The bigger the mesh, the smaller the relative cost.

## Per-optimization contribution (1-chip, NO_DISPATCH apples-to-apples, µs saved; noise ±1µs)
| op | skip-validate | small-vector | CoreRangeSet | RtaName | memoize |
|----|----:|----:|----:|----:|----:|
| transpose | -23 | -1 | -2 | ~0 | ~0 |
| slice | -24 | -10 | -2 | ~0 | ~0 |
| fold | -17 | -3 | -3 | +1 | ~0 |
| tilize | -10 | -5 | -2 | ~0 | ~0 |
| untilize | -4 | ~0 | -1 | ~0 | ~0 |
| reshard | -3 | ~0 | -4 | +4 | ~0 |
| i2s | -3 | +2 | -5 | +4 | ~0 |

skip-validate = the dominant win; small-vector helps arg-heavy ops; CoreRangeSet helps sharded;
RtaName/memoize neutral (RtaName slightly hurts sharded — short arg names). builder = foundational
(not isolatable). galaxy parallel-write (#48314) not measurable here (needs real multi-chip).
