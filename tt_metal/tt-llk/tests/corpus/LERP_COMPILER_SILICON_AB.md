# Lerp identical-source compiler silicon A/B

## Result

The production typed Lerp body is a stable scoped Blackhole compiler win. Both
variants compile the exact same source and use the same seeded inputs, golden,
format, and destination-accumulation mode. The only causal change is the
test-only compiler option set.

| compiler selector | sample 1 | sample 2 | sample 3 |
|---|---:|---:|---:|
| passes off | 580.984375 | 580.984375 | 580.984375 |
| passes on | 564.984375 | 564.984375 | 564.984375 |

The candidate saves 16 cycles/tile, or 2.7539468%. Each number is the post CSV's
`TILE_LOOP` row and `mean(MATH_ISOLATE)` column from a fresh serialized pytest
process. It is a scoped device-cycle result, not host time or whole-kernel
throughput.

## Correctness and attribution

The paired physical-Blackhole correctness node passed with both compiler
selectors. The same node also passed CRAQ with both selectors. The existing
Float16_b contract requires element tolerance (`rtol=0.05`, `atol=0.05`) and
PCC greater than 0.99 on nontrivial signal.

The exact production source is `ckernel_sfpu_lerp.h`: three typed destination
loads, `in0 + in2 * (in1 - in0)`, BF16 RNE conversion, typed store, and Dst
advance. There is no fresh body, handwritten alternative, fixed LREG, raw TTI,
or kernel-name peephole in this comparison. Because Lerp has no distinct
handwritten implementation, the semantic and production controls are the same
source; a separate handwritten 2x2 device matrix would duplicate this A/B and
cannot add an independent source control.

The control leaves the four opt-in passes disabled. The candidate enables:

```text
-mtt-tensix-optimize-latency-schedule
-mtt-tensix-optimize-dst-iteration-fusion
-mtt-tensix-optimize-replay-hoist
-mtt-tensix-optimize-invariant-loadi
```

The final linked correctness ELF changes from a 14-slot replay capture executed
eight times to a 28-slot fused capture executed four times. Dynamic replay
launches and Dst RWC advances both fall from eight to four. The `.text` SHA-256
changes from `50b1d35c4a89286d7724722613c8501b94b8572e9e880da94d88e8803dcce2d7`
(1240 bytes) to
`2505e7459845b58da720547d1d97cfeaa27c9517bca31d6d559248ef711d7c2a`
(1292 bytes). The profiler ELF `.text` changes from
`970fe16d6cec97dab1bddd09a6b6899b0f0f352bf94e865429d4bd3b8d3e9794`
(1964 bytes) to
`c8eb6449e4fe8aaaeffd064328930515a44b85ddd345adc82853ef3773cbab6f`
(1948 bytes).

The checked-in corpus runner independently collected the exact correctness
node, compiled both selectors with the version pin enforced, classified the
math ELF `.text` as `CHANGED_BINARY`, and reproduced those same hashes and
sizes. Its result is archived under `corpus-compiler-ab-pinned`.

## Provenance

- TT-Metal discriminator source: `42fdced8393d42a44aa71166148622573700375e`
- SFPI-GCC: `e17a4f8fdd733cf523d5d8d4c37c15be41b4433d`
- compiler driver SHA-256: `a6fe054dea8b08e1131a0e233679e1b149ea56c30f10490bb98a7bbbd405f041`
- compiler `cc1plus` SHA-256: `cd68b28f639dba4423564ef23a6abc84798774d93a3ad9e670a88409c7618c2b`
- `sfpu_ternary_test.cpp` SHA-256: `f00eb5dd27f23a3f461026043645060350819ff25948c5497146f44413dfa4df`
- production Lerp header SHA-256: `fabf38ddd838f58380cb6cef359c51a6e35f0a518dafb7fa50ebd974e9107aa7`
- CRAQ simulator SHA-256: `16ca46261895e78b424cdbf531789eb02a8d10bd32ecfcc228362e505f778d6e`

The immutable local archive is
`/localdev/nkapre/lerp-silicon-win-20260815`. It contains compiler A/B ELFs and
disassembly, paired CRAQ and device logs, six raw/post CSV pairs copied
immediately after their processes, per-process profiler ELFs, and aggregate
hashes.
