# Issue #52930 finding 4 — `RsqrtCompat(0)`: all measurements

Every number on this page was measured on **Wormhole n300 silicon** via
`tt_metal/tt-llk/.claude/scripts/run_test.sh`, tree `ldjurovic/wrong_sfpu_edge_cases`, 2026-08-17.
Fix committed as `21557a33763`. Companion prose/methodology:
[RESULTS_52930_reciprocal_compat_pole.md](RESULTS_52930_reciprocal_compat_pole.md),
[FIX_PLAN_52930_reciprocal_compat_pole.md](FIX_PLAN_52930_reciprocal_compat_pole.md).

`MATH_ISOLATE` figures are per-tile `TILE_LOOP` cycles. Measurement noise across 3 repeats was
< 0.1 cycles out of ~2250, so every delta below is far outside noise.

## 1. Headline — the four affected paths

| # | Path | Values | Accuracy (max rel err) | `MATH_ISOLATE` | Regression? |
|---|---|---|---|---|---|
| 1 | `rsqrt_compat` / `RsqrtCompat` | changed | `1.470e-03` → `5.012e-06` (**293×**) | **−25.6 … −38.9 %** | none |
| 2 | `recip()` `legacy_compat=true` (public default) | changed | `3.636e-05` → `5.960e-08` (**610×**) | **−43.2 … −52.6 %** | none |
| 3 | SDPA `calculate_recip_first_column<true>` | changed + sign fixed | same as #2 (same instantiation) | −16 cyc/slot *(derived)* | none |
| 4 | Blackhole `sampling_recip_value<true>` | **bit-identical, untouched** | unchanged | unchanged | n/a |

Accuracy is quoted on `Float32→Float32 / dest_acc=Yes`, the only combination that measures the kernel;
elsewhere a bf16 pack quantises to ~2⁻⁸ and both kernels land on the same value.

## 2. Option A vs Option B — why A was chosen (`RsqrtCompat`)

| `dest_acc` / `approx` | baseline | Option B (pole guard) | Δ | Option A (redirect, **shipped**) | Δ |
|---|---|---|---|---|---|
| acc=No apx=No | 2246.97 | 2356.04 | **+4.85 %** | 1567.97 | **-30.22 %** |
| acc=No apx=Yes | 1152.12 | 1248.16 | **+8.34 %** | 704.07 | **-38.89 %** |
| acc=Yes apx=No | 2237.92 | 2340.58 | **+4.59 %** | 1665.27 | **-25.59 %** |
| acc=Yes apx=Yes | 1152.13 | 1248.05 | **+8.33 %** | 703.94 | **-38.90 %** |

| Secondary | baseline | Option B | Option A |
|---|---|---|---|
| `INIT` cycles (once per kernel) | 223–224 | 223–224 | 236–240 |
| `TEXT_SIZE(MATH_ISOLATE)` bytes | 3105–5225 | 3297–5417 | 2869–4173 |
| Bit-exact vs baseline? | — | **yes**, only the 8 pole values | no, every value changes (toward golden) |
| Fixes the pole? | — | yes | yes |
| Fixes the accuracy floor? | — | no | yes |
| Fixes the SDPA sign drop? | — | no | yes |

Option B's cost is exactly the guard: **+96 cycles / 32 SFPU iterations = 3 cycles per iteration**
(`SFPSETCC` + constant load + predicated move). Option A instead deletes a Newton iteration *and* the
`exexp`/`setexp` fix-up the guard was patching.

## 3. Path 1 — `RsqrtCompat` accuracy, per value (`Float32→Float32`, `dest_acc=Yes`)

Hardware bit pattern vs an exact fp64 `1/sqrt(x)`.

| `x` | hw before | rel err before | hw after | rel err after | verdict |
|---|---|---|---|---|---|
| `0.0` (**the pole**) | `0x7EFFFD9E` | — | `0x7F800000` | — | **fixed: 1.7e38 → +inf** |
| `1` | `0x3F7F9FAA` | `1.470e-03` | `0x3F80002A` | `5.007e-06` | **better** |
| `2` | `0x3F3504F3` | `1.711e-08` | `0x3F3504F3` | `1.711e-08` | bit-identical |
| `4` | `0x3EFF9FAA` | `1.470e-03` | `0x3F00002A` | `5.007e-06` | **better** |
| `0.25` | `0x3FFF9FAA` | `1.470e-03` | `0x4000002A` | `5.007e-06` | **better** |
| `0.5` | `0x3FB504F3` | `1.711e-08` | `0x3FB504F3` | `1.711e-08` | bit-identical |
| `3` | `0x3F13CCB1` | `1.416e-05` | `0x3F13CD41` | `7.047e-07` | **better** |
| `7` | `0x3EC184AA` | `2.113e-06` | `0x3EC184AA` | `2.113e-06` | bit-identical |
| `1.5` | `0x3F51060B` | `2.299e-06` | `0x3F51060B` | `2.299e-06` | bit-identical |
| `100` | `0x3DCCCCF3` | `2.846e-06` | `0x3DCCCCF3` | `2.846e-06` | bit-identical |
| `0.001` | `0x41FCF9C8` | `2.569e-05` | `0x41FCFB9C` | `2.539e-06` | **better** |
| `1000` | `0x3D01634D` | `1.073e-03` | `0x3D01870D` | `5.012e-06` | **better** |
| `1e-10` | `0x47C35021` | `2.585e-06` | `0x47C35021` | `2.585e-06` | bit-identical |
| `1e+10` | `0x3727C5C4` | `2.158e-06` | `0x3727C5C3` | `2.067e-06` | ≈ same (1 ulp) |
| `1e-30` | `0x58635F94` | `1.421e-06` | `0x58635FAB` | `1.228e-07` | **better** |
| `1e+30` | `0x26901C0B` | `3.916e-05` | `0x26901D7E` | `1.170e-07` | **better** |
| `1.17549e-38` | `0x5E6BD88D` | `5.394e-01` | `0x5E6BD8D8` | `5.394e-01` | `_sqrt_compat_` seed, **unchanged** |
| `3.40282e+38` | `0x1EEBD88D` | `5.394e-01` | `0x1EEBD8D8` | `5.394e-01` | `_sqrt_compat_` seed, **unchanged** |
| `2.38419e-07` | `0x44FF9FAA` | `1.470e-03` | `0x4500002A` | `5.007e-06` | **better** |
| `0.015625` | `0x40FF9FAA` | `1.470e-03` | `0x4100002A` | `5.007e-06` | **better** |
| `+inf` | `0x00000000` | — | `0x00000000` | — | correct both (+0) |

**Max over the normal range: `1.470e-03` → `5.012e-06` — 293× better, and no value got worse.**

> The two `5.394e-01` rows are **54 % wrong before and after identically**: `_sqrt_compat_`'s
> fast-inverse-sqrt seed running out of range at the fp32 extremes. Not the reciprocal, not fixed here,
> same family as `sqrt_custom(+inf)` in `FIX_PLAN_52930_sqrt_custom_infinity.md`. **Worth its own issue.**

## 4. Path 2 — `recip()` `legacy_compat=true` accuracy (the public default)

126 values spanning 2⁻³⁰…2³⁰, both signs, non-power-of-two mantissas, vs exact fp64 `1/x`.

| variant | max rel err before | max rel err after | mean before | mean after |
|---|---|---|---|---|
| `Float16_b->Float16_b dest_acc=No` | `6.653e-03` | `6.653e-03` | `2.889e-03` | `2.889e-03` |
| `Float16_b->Float16_b dest_acc=Yes` | `6.653e-03` | `6.653e-03` | `2.889e-03` | `2.889e-03` |
| `Float16_b->Float32 dest_acc=No` | `6.653e-03` | `6.653e-03` | `2.889e-03` | `2.889e-03` |
| `Float16_b->Float32 dest_acc=Yes` | `4.595e-03` | `4.587e-03` | `1.544e-03` | `1.529e-03` |
| `Float32->Float16_b dest_acc=No` | `2.734e-03` | `2.734e-03` | `1.042e-03` | `1.042e-03` |
| `Float32->Float16_b dest_acc=Yes` | `2.734e-03` | `2.734e-03` | `1.042e-03` | `1.042e-03` |
| `Float32->Float32 dest_acc=No` | `2.734e-03` | `2.734e-03` | `1.042e-03` | `1.042e-03` |
| `Float32->Float32 dest_acc=Yes` **← measures the kernel** | `3.636e-05` | `5.960e-08` | `1.444e-05` | `2.583e-08` |

Per-value detail, `Float32→Float32 / dest_acc=Yes` (first rows):

| `x` | exact | hw before | rel before | hw after | rel after |
|---|---|---|---|---|---|
| `9.3132e-10` | `1.0737418e+09` | `0x4E7FFD9E` | `3.636e-05` | `0x4E800000` | `0.000e+00` |
| `-9.3132e-10` | `-1.0737418e+09` | `0xCE7FFD9E` | `3.636e-05` | `0xCE800000` | `0.000e+00` |
| `1.2107e-09` | `8.2595525e+08` | `0x4E44EC4F` | `1.788e-08` | `0x4E44EC4F` | `1.788e-08` |
| `-1.2107e-09` | `-8.2595525e+08` | `0xCE44EC4F` | `1.788e-08` | `0xCE44EC4F` | `1.788e-08` |
| `1.5832e-09` | `6.3161284e+08` | `0x4E169652` | `6.950e-06` | `0x4E169696` | `5.960e-08` |
| `-1.5832e-09` | `-6.3161284e+08` | `0xCE169652` | `6.950e-06` | `0xCE169696` | `5.960e-08` |
| `2.9802e-08` | `33554432` | `0x4BFFFD9E` | `3.636e-05` | `0x4C000000` | `0.000e+00` |
| `-2.9802e-08` | `-33554432` | `0xCBFFFD9E` | `3.636e-05` | `0xCC000000` | `0.000e+00` |

> `0x4E7FFD9E` where the exact answer is `0x4E800000`: the legacy kernel is wrong even where the result is
> **exactly representable**. That `…FFD9E` mantissa is the same signature as the `0x7EFFFD9E` the pole
> produced — one Newton iteration converging just short of 2.0 drives both. It is a ~16-bit-accurate
> reciprocal that the public API was defaulting to on fp32 paths.

## 5. Path 2 — `recip()` `legacy_compat=true` perf, all 64 variants

16 format pairs × `approx` × `dest_acc`. **Every one of the 64 improved.**

| variant class | before (min–max) | after (min–max) | delta |
|---|---|---|---|
| Float32 in, acc=No, apx=No | 1216.12–1216.23 | 576.17–576.25 | **-52.62 %** |
| Float32 in, acc=No, apx=Yes | 992.20–992.22 | 544.26–544.32 | **-45.15 %** |
| Float32 in, acc=Yes, apx=No | 1165.70–1165.70 | 653.65–653.66 | **-43.93 %** |
| Float32 in, acc=Yes, apx=Yes | 973.67–973.68 | 525.62–525.63 | **-46.02 %** |
| other in, acc=No, apx=No | 1215.93–1216.02 | 575.98–576.03 | **-52.63 %** |
| other in, acc=No, apx=Yes | 991.97–992.02 | 544.02–544.06 | **-45.16 %** |
| other in, acc=Yes, apx=No | 1184.02–1184.09 | 671.98–672.09 | **-43.25 %** |
| other in, acc=Yes, apx=Yes | 991.98–992.04 | 544.00–544.10 | **-45.16 %** |

Math kernel text: **2945–3301 → 2833–2877 bytes.**

## 6. Path 3 — SDPA `calculate_recip_first_column<true>` *(derived, not directly measured)*

`sfpu_sdpa_test.cpp` has no `MEASURE_PERF_COUNTERS` and there is no perf test for it, so cycles are
divided out of §5's per-tile figure (32 SFPU slots per tile). **Accuracy needs no derivation:** the legacy
body was `_reciprocal_compat_<APPROX ? 2 : 3>`, the identical instantiation measured in §4.

| SDPA precision | before | after | per call (4 slots) | saving |
|---|---|---|---|---|
| `Bf16Dest` (`DST_ACCUM_MODE=No`) | 38.0 cyc/slot | 18.0 cyc/slot | 152 → 72 | −80 |
| `Fp32Dest` / `Fp32E2E` (`DST_ACCUM_MODE=Yes`) | 37.0 cyc/slot | 21.0 cyc/slot | 148 → 84 | −64 |

Plus a semantic fix unique to this path:

| | before | after |
|---|---|---|
| `recip_first_column(-x)` | `+1/x` — **sign silently dropped** | `-1/x` |

The legacy branch called `_reciprocal_compat_` (which returns the magnitude `\|1/x\|`) and, unlike every
other consumer, never restored the sign. The golden generator encoded `RecipLegacy = 1/\|x\|` to match;
it now says `1/x`. Harmless in production only because a softmax denominator is positive.

## 7. Path 4 — Blackhole `sampling_recip_value<true>` — untouched

| Check | Result |
|---|---|
| `ckernel_sfpu_sampling.h` in commit `21557a33763`? | **no** |
| Still calls `_reciprocal_compat_`? | **yes**, line 52 |
| Values / cycles | bit-identical by construction |
| Pole still unguarded there? | **yes** — see §9 |
| Verified how | source + clean Blackhole compile; **no Blackhole silicon on this host** |

## 8. Functional verification (Wormhole silicon, shipped fix)

| Suite | Result |
|---|---|
| `test_sfpu_unary.py -k "RsqrtCompat or Reciprocal or Rsqrt or Sqrt"` | **847 passed, 9 xfailed, 0 xpassed, 0 failed** |
| `test_sfpu_unary.py -k "edges and (RsqrtCompat or Rsqrt or Reciprocal or Sqrt or Erfinv)"` | 37 passed, 11 xfailed, 0 xpassed |
| `test_sfpu_sdpa.py` | 65 passed |
| `test_sfpu_sdpa_fw.py` | 21 passed |
| `test_sfpu_reduce_sdpa.py` / `test_sdpa_reinits.py` / `test_sfpu_sampling.py` | pass |
| Blackhole `test_sfpu_unary.py` / `test_sfpu_sdpa.py` / `test_sfpu_sampling.py` | **compile only** — no BH silicon here |

The 8 `RsqrtCompat` edge combinations that previously XPASSed now **assert**: the entries were removed from
`_EDGE_KNOWN_DIVERGENCES` and `_EDGE_DIVERGENCE_REASON`. The 9 remaining xfails are `Reciprocal`'s
pre-existing `1/NaN` case and the `Sqrt`/`Rsqrt` `-0` cases, all untouched.

## 9. Open items

| # | Item | Why it is not closed here |
|---|---|---|
| 1 | `_reciprocal_compat_` still has the **unguarded pole** for BH sampling | That path documents a bit-identity requirement for *blaze* (`ckernel_sfpu_sampling.h:37`) and constrains callers to `in > 0`, so the pole is unreachable. Owners' call: leave it, or apply the §2 guard there alone (bit-identical across its whole documented domain, 3 cyc/element). |
| 2 | `recip()` default has **no edge test and no perf variant** | The suite drives `MathOperation.Reciprocal` with `legacy_compat=false`. §4/§5 used a temporary harness adapter, since reverted. A permanent `legacy_compat` axis is the real fix. |
| 3 | `rsqrt_compat` is **54 % wrong at the fp32 extremes** | `_sqrt_compat_`'s seed, unchanged by this fix (§3). Needs its own issue. |
| 4 | `RsqrtCompat` not in `SPECIALS_READY_OPS` | `rsqrt_compat(±inf)`/`(NaN)` never driven by the shipped sweep. Probe shows `+inf → +0` on all 8, now correct *by construction*; worth pinning. |
| 5 | **Blackhole is compile-verified only** | No BH silicon on this host; the plan's §8 step 3 wants both arches run. |

## 10. What shipped

Commit `21557a33763`, 10 files.

| File (× both arches unless noted) | Change |
|---|---|
| `llk_sfpu/ckernel_sfpu_rsqrt.h` | `_calculate_rsqrt_compat_iter_`: keeps `_sqrt_compat_`, pairs it with `sfpu_reciprocal_iter`; `rsqrt_init<…,true>` programs `vConstFloatPrgm0..2` |
| `llk_sfpu/ckernel_sfpu_recip.h` | `calculate_reciprocal` no longer branches on `legacy_compat`; `recip_init` always inits |
| `experimental/llk_sfpu/ckernel_sfpu_sdpa.h` | legacy branch collapsed onto `sfpu_reciprocal_iter` (+ pre-commit `std::uint*_t` normalisation) |
| `tt-llk/tests/helpers/include/sfpu_operations.h` | `rsqrt_compat` routed to a real init |
| `tt-llk/tests/python_tests/test_sfpu_unary.py` | `RsqrtCompat` xfail tables removed |
| `tt-llk/tests/python_tests/helpers/golden_generators.py`, `test_sfpu_sdpa.py` | `RecipLegacy` golden `1/\|x\|` → `1/x` |

Not for merge, kept as evidence (untracked): `test_sfpu_wh_recipcompat_numerics.py` (bit-exact record),
`test_sfpu_wh_recip_accuracy.py` (error sweep).
