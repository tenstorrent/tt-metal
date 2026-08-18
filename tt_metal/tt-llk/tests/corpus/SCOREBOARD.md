# BH LLK SFPU Scoreboard — compiler vs handwritten

Provenance: nightly-20260819b (pin #10, cc1plus `2911f0e680e4…`, corrected oracle `32489dda…`),
BH p150, correctness-gated (§1 protocol: paired CRAQ + device correctness before every perf leg),
cycles per the row's recorded marker units. Regenerate: run the nightly and this table's data is
`<evidence>/scoreboard.tsv`. Baselines: `sfpu_device_baseline_p150_v1.tsv` (200 cells, all pin-10).

## Tier A — hand-comparable rows (semantic C++ vs handwritten LLK)

| op | sem OFF | sem ON | hand | causal | vs hand | verdict | active mechanism |
|---|---:|---:|---:|---:|---:|---|---|
| unarymaxmin-max | 24.21 | 15.59 | 27.31 | -35.6% | -42.92% | **WIN** | — |
| unarymaxmin-min | 24.21 | 15.59 | 27.31 | -35.6% | -42.92% | **WIN** | — |
| subint | 33.96 | 29.97 | 45.99 | -11.8% | -34.83% | **WIN** | — |
| addint | 33.96 | 29.97 | 45.98 | -11.7% | -34.82% | **WIN** | — |
| addcmul | 44.63 | 32.62 | 36.62 | -26.9% | -10.90% | **WIN** | — |
| signbit | 27.76 | 21.51 | 22.70 | -22.5% | -5.24% | **WIN** | — |
| typecast | 361.67 | 259.00 | 265.00 | -28.4% | -2.26% | **WIN** | — |
| recip | 523.00 | 457.00 | 466.33 | -12.6% | -2.00% | **WIN** | — |
| reduce-sdpa | — | 832.75 | 839.00 | — | -0.74% | **WIN** | — |
| sdpa | 1289.00 | 1004.00 | 1009.00 | -22.1% | -0.50% | **WIN** | — |
| binary-bcast | — | 608.00 | 608.00 | — | +0.00% | parity | — |
| welford | — | 331.00 | 325.00 | — | +1.85% | LOSS | — |
| sigmoidappx-tree | 123.85 | 29.85 | 27.86 | -75.9% | +7.17% | LOSS | coefficient cross-call hoist (IPA pkg, gated on mop_cfg derivation) |
| where | 312.50 | 174.25 | 159.25 | -44.2% | +9.42% | LOSS | descriptor residency (BJ#4) + template self-programming |
| minmax-min | 28.48 | 19.85 | 17.63 | -30.3% | +12.61% | LOSS | AY drain-aware scheduling (design ready) |
| minmax-max | 28.48 | 19.85 | 17.63 | -30.3% | +12.61% | LOSS | AY drain-aware scheduling (design ready) |
| exp | 131.72 | 83.47 | 72.47 | -36.6% | +15.18% | LOSS | M3 re-entry via mop_cfg derivation (-8u -> ~74) + epoch per-tile setup |
| mul_int32 (production) | — | 53.99 | 35.62 | — | +51.59% | LOSS | IMS placement (BJ#5) + renaming |
| sigmoidappx | 55.85 | 44.85 | 27.86 | -19.7% | +60.98% | LOSS | algorithm choice (SFPLUT superior for this contract) — documented, tree row is the fair comparison |
| mulint32-fresh | 81.97 | 58.36 | 35.62 | -28.8% | +63.84% | LOSS | IMS placement (BJ#5) + renaming |

Tally: 10 WIN, 1 parity, 9 LOSS (of which sigmoid-cubic is a documented algorithm choice and
mul_int32 appears twice: production pinpair + fresh semantic row). Every loss carries a named,
in-flight or designed mechanism. welford's WIN->LOSS is the bisected pristine-ruling cost: the
prior 323 rode an l_reg hand-ism; the clean typed body measures 331 vs 325.

## Tier B — causal-only rows (production typed body; OFF vs ON is the honest axis)

| op | sem OFF | sem ON | causal |
|---|---:|---:|---:|
| abs | 19.73 | 19.36 | -1.9% |
| activations | 60.10 | 56.23 | -6.4% |
| add1 | 19.86 | 19.86 | -0.0% |
| addcdiv | 104.62 | 100.62 | -3.8% |
| binopscalar | 20.38 | 20.00 | -1.8% |
| castfp32tofp16a | 19.73 | 15.86 | -19.6% |
| cbrt | 96.35 | 92.85 | -3.6% |
| celu | 193.62 | 187.86 | -3.0% |
| clamp | 59.87 | 61.23 | +2.3% |
| comp | 27.74 | 27.74 | +0.0% |
| digamma | 419.74 | 415.86 | -0.9% |
| elu | 193.62 | 187.86 | -3.0% |
| erf | 191.86 | 191.86 | -0.0% |
| erfc | 319.73 | 315.86 | -1.2% |
| erfinv | 372.48 | 365.73 | -1.8% |
| exp2 | 107.74 | 83.11 | -22.9% |
| expm1 | 155.72 | 151.85 | -2.5% |
| expm1cw | 175.86 | 148.74 | -15.4% |
| fmod | 371.98 | 368.48 | -0.9% |
| gelu | 356.11 | 336.24 | -5.6% |
| hardmish | 55.86 | 56.99 | +2.0% |
| hardshrink | 44.11 | 40.74 | -7.6% |
| hardtanh | 59.86 | 61.23 | +2.3% |
| heaviside | 59.86 | 57.61 | -3.8% |
| i0 | 163.86 | 144.61 | -11.8% |
| i1 | 471.73 | 443.85 | -5.9% |
| lerp | 72.63 | 58.87 | -18.9% |
| lgamma | 263.73 | 255.86 | -3.0% |
| log | 76.10 | 72.22 | -5.1% |
| log1p | 136.10 | 132.23 | -2.8% |
| logicalnot | 28.96 | 25.70 | -11.2% |
| mish | 264.10 | 252.23 | -4.5% |
| negative | 19.74 | 15.86 | -19.6% |
| polygamma | 355.75 | 351.88 | -1.1% |
| prelu | 27.73 | 24.49 | -11.7% |
| rdiv | 48.23 | 47.86 | -0.8% |
| relu | 51.86 | 49.23 | -5.1% |
| remainder | 415.98 | 412.48 | -0.8% |
| rpow | 400.23 | 392.86 | -1.8% |
| rsqrt | 128.10 | 120.22 | -6.2% |
| selu | 202.24 | 196.86 | -2.7% |
| sigmoid | 156.11 | 144.23 | -7.6% |
| sign | 43.74 | 40.98 | -6.3% |
| silu | 171.86 | 171.86 | -0.0% |
| snakebeta | 168.62 | 153.62 | -8.9% |
| softplus | 179.61 | 175.74 | -2.2% |
| softshrink | 79.74 | 77.62 | -2.7% |
| softsign | 67.86 | 69.36 | +2.2% |
| sqrt | 112.10 | 108.22 | -3.5% |
| sqrtcustom | 111.74 | 110.11 | -1.5% |
| square | 19.86 | 19.86 | -0.0% |
| tanh | 66.35 | 64.97 | -2.1% |
| tanhderivative | 383.73 | 375.86 | -2.1% |
| tanhderivative-lut | 27.84 | 28.34 | +1.8% |
| tanhshrink | 172.11 | 164.23 | -4.6% |
| threshold | 40.24 | 36.87 | -8.4% |
| trigonometry | 387.72 | 383.85 | -1.0% |
| unarycomp | 44.23 | 40.86 | -7.6% |
| xielu | 419.35 | 411.98 | -1.8% |

59 rows measured this run; 5 more (`shift, fill, identity, roundingops, unarypower`) are
weekly-scheduled (unengaged by current passes; deferrals scoreboard-visible, never silent).

## Not yet measured — every exclusion named

- `topk-perf` — multi-result frontier: no perf claim until a sourced impl exists (blocked, AN epoch scoping)
- `addtoprow` — ZERO executable BH perf variant upstream (perf test skips all runnable combos) — upstream fix filed
- `mulint` — machine-readable alias of mul_int32 (same kernel; no double-booking)
- 26 corr-only mapped rows (compile+correctness gated nightly; per-row missing perf vehicle named in AUDIT.md)
- 42 BH headers unmapped ([C]/[D]/[E] classes — no upstream BH test surface / deprecated; re-audited by Lane AZ, one rescued)

## Standing guarantees
- LLK library trees byte-identical to upstream (conf-lint R7); zero markers or trusted annotations anywhere;
  semantic sources are plain typed C++ under tests/; every compiler mechanism proves or refuses by name.
- Every measured row re-verified nightly against the 200-cell baseline (drift/flip/lb gates); measured-but-
  unwired is mechanically impossible (R8); no ON-set flag without a dump-proven fire witness.
