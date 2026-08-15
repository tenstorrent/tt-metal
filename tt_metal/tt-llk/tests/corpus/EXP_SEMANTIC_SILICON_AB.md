# Exp fresh semantic C++ silicon A/B

This test-only lane compares the production accurate BF16 Exp implementation with a
compiler-visible semantic implementation. The fresh source loops over typed Dst rows and
calls the existing mathematical 21-bit BF16 Exp helper. It contains no raw TTI, fixed
LREG assignment, replay range, SFPLOADMACRO configuration, or handwritten instruction
schedule. Production remains the default selector.

## Correctness and pre-silicon gates

The paired lane uses Float16_b input/output with destination accumulation disabled and
accurate mode. It reuses both the registered Exp functional domain and the checked-in
`edge_spec` for Exp. `passed_test` requires the existing per-format element tolerance
(`rtol=0.05`, `atol=0.05`, paired NaNs accepted) and PCC greater than `0.99` for
nontrivial signal; PCC alone cannot pass.

| Gate | Result |
|---|---:|
| Wormhole functional, edge, and profiler compile | 6/6 |
| Blackhole functional, edge, and profiler compile | 6/6 |
| Blackhole CRAQ functional and edge A/B | 4/4 |
| Blackhole silicon functional and edge A/B | 4/4 |

## Scoped Blackhole result

Each sample is a fresh serialized pytest process. The metric is `TILE_LOOP
mean(MATH_ISOLATE)` cycles/tile, not whole-kernel throughput.

| Production | Fresh semantic C++ | Delta |
|---:|---:|---:|
| 579.7421875 / 579.7421875 / 579.7421875 | 989.75 / 989.75 / 989.75 | +410.0078125 (+70.72%) |

The complete archive is `/localdev/nkapre/exp-semantic-bh-silicon-20260815`.
It contains unique raw/post CSVs, logs, build headers, ELFs, disassemblies, and `.text`
images for every recaptured process. The SHA256 manifest itself hashes to
`1d0e07ca2ff64f9fe0f9f2c4e13ad21df0fd4a55a6ab80f22f03d37f64c6d5a4`.
The representative `.text` hashes are
`7dcd9e9a560961ac0112623a88ba2adb20ddc991f72db5b33569394d84531d75`
for production and
`e48e4e8a7f73d0d172b768b251d2d26e6a33cff09bfbbb5df2695dacfef9ed17`
for fresh semantic C++.

## Mechanism

Production places invariant coefficients in architectural constant registers and forms
a 16-slot replay capture followed by playback. The fresh final ELF emits the typed
load, exponent/mantissa conversion, polynomial, rounding, and store directly in the
counted row loop; it has no `TTREPLAY` and rematerializes coefficients in the loop.

This is negative evidence for two generic compiler targets: legal loop-invariant SFPU
constant placement and replay extraction from counted typed loops. Any fix must be based
on invariant dataflow, Dst/RWC legality, and fixed-encoding proof rather than recognizing
Exp or its coefficients.
