# Measurements — reciprocal compat pole, accuracy and performance vs `main`

Accuracy and performance of the three kernel paths this branch changes, measured against the
branch point rather than recalled from the original development run. Read-only record: no
code, no test changes.

| | |
|---|---|
| Branch | `ldjurovic/sfpu_52930_reciprocal_compat_pole` |
| Baseline | `origin/main` @ `35ec0aba7a8` — the branch point, so the only kernel delta is this branch's |
| Silicon | Wormhole n300 (UMD chip 0) |
| Blackhole | compile-verified only — no BH hardware on this host |
| Date | 2026-08-20 |

Companion to [MEASUREMENTS_52930_reciprocal_compat_pole.md](MEASUREMENTS_52930_reciprocal_compat_pole.md)
and [RESULTS_52930_reciprocal_compat_pole.md](RESULTS_52930_reciprocal_compat_pole.md), which
record the fix as it was developed. This one re-measures against the current branch point and
adds two things those did not cover: the SDPA reciprocal's numerics, and an instruction-level
account of where the speedup comes from.

---

## 1. What actually changes, and what does not

Six kernel files change (`ckernel_sfpu_recip.h`, `ckernel_sfpu_rsqrt.h`,
`experimental/llk_sfpu/ckernel_sfpu_sdpa.h`, each on Wormhole and Blackhole), but they gate
**three** dispatch paths, all of them the `legacy_compat = true` side:

| # | Path | Before | After | Reached by |
|---|---|---|---|---|
| 1 | `calculate_reciprocal<…, legacy_compat=true>` | `_calculate_reciprocal_compat_` | `_calculate_reciprocal_internal_` | **the public `recip()` API** — `compute/eltwise_unary/recip.h` declares `template <bool legacy_compat = true>` |
| 2 | `calculate_rsqrt<…, legacy_compat=true>` | `_calculate_rsqrt_compat_` | `_sqrt_compat_` + `sfpu_reciprocal_iter` | `RsqrtCompat` |
| 3 | `calculate_recip_first_column<legacy_compat=true>` | `_reciprocal_compat_` | `sfpu_reciprocal_iter` | SDPA `RecipLegacy` |

Everything on the `legacy_compat = false` side is untouched, which gives three controls that
should not move at all: the public `rsqrt()` (its API default *is* `false`), the tt-llk
`Reciprocal` op as the suite ships it, and SDPA `RecipIter`. All three are measured below and
all three are flat.

**A coverage gap worth naming.** Path 1 is the branch's largest user-visible change and the
tt-llk suite does not exercise it: `sfpu_operations.h` calls `calculate_reciprocal` and
`recip_init` without a `legacy_compat` argument, so they default to `false` and the suite
measures the kernel the public API does *not* dispatch. To measure path 1 I patched both trees
identically to pass `true`, matching the API default. That patch is an instrument, reverted
before the correctness sweeps in §6 and not committed. Without it, every number in §3 and the
`Reciprocal` row of §5 would have read "no change".

## 2. Method

Two checkouts of the same repo with isolated build roots (`RUNNER_TEMP`), so neither can serve
the other a stale ELF; baseline and branch run back to back on the same board.

Accuracy is raw hardware bit patterns or relative error against exact fp64, from fixed
stimulus lists held in the probes rather than read from `edge_values()` — this branch edits the
golden and the xfail tables, so a probe that read them would be comparing two different
experiments rather than two kernels. Performance is the CI harness with CI's flags
(`--speed-of-light`, producer then consumer), `MATH_ISOLATE` on the `TILE_LOOP` marker divided
by `tile_cnt × loop_factor` (128), three consumer runs per tree over 196 variants each.

## 3. Accuracy — the public `recip()` default (path 1)

Relative error against exact fp64 `1/x`, over a wide exponent sweep, both signs:

| Variant | max rel, `main` | max rel, branch | mean rel, `main` | mean rel, branch |
|---|---|---|---|---|
| Float16_b→Float16_b acc=No | 6.653e-03 | 6.653e-03 | 2.889e-03 | 2.889e-03 |
| Float16_b→Float16_b acc=Yes | 6.653e-03 | 6.653e-03 | 2.889e-03 | 2.889e-03 |
| Float16_b→Float32 acc=No | 6.653e-03 | 6.653e-03 | 2.889e-03 | 2.889e-03 |
| Float16_b→Float32 acc=Yes | 4.595e-03 | 4.587e-03 | 1.544e-03 | 1.529e-03 |
| Float32→Float16_b acc=No | 2.734e-03 | 2.734e-03 | 1.042e-03 | 1.042e-03 |
| Float32→Float16_b acc=Yes | 2.734e-03 | 2.734e-03 | 1.042e-03 | 1.042e-03 |
| Float32→Float32 acc=No | 2.734e-03 | 2.734e-03 | 1.042e-03 | 1.042e-03 |
| **Float32→Float32 acc=Yes** | **3.636e-05** | **5.960e-08** | **1.444e-05** | **2.583e-08** |

The last row is the only one that measures the kernel — everywhere else a bf16 pack quantises
the answer to about `2^-8` and hides the difference. There the legacy reciprocal is **610×
less accurate** than its replacement on max error and 559× on mean.

## 4. Accuracy — `RsqrtCompat` (path 2)

**The pole, which is issue #52930 finding 4, is fixed on all 8 combinations:**

| | `main` | branch |
|---|---|---|
| `rsqrt_compat(0)` | `0x7EFFFD9E` = 1.7013e38 (`0x7F000000` = 1.7014e38 where bf16-packed) | `0x7F800000` = `+inf` |

Per-value, on `Float32→Float32 dest_acc=Yes` — the pipeline where nothing is narrowed. Exact
reference is fp64 `1/sqrt(x)`:

| x | `main` | branch | rel `main` | rel branch |
|---|---|---|---|---|
| 0 | `0x7EFFFD9E` | `0x7F800000` | — | — |
| 1 | `0x3F7F9FAA` | `0x3F80002A` | 1.470e-03 | 5.007e-06 |
| 2 | `0x3F3504F3` | `0x3F3504F3` | 1.711e-08 | 1.711e-08 |
| 4 | `0x3EFF9FAA` | `0x3F00002A` | 1.470e-03 | 5.007e-06 |
| 0.25 | `0x3FFF9FAA` | `0x4000002A` | 1.470e-03 | 5.007e-06 |
| 0.5 | `0x3FB504F3` | `0x3FB504F3` | 1.711e-08 | 1.711e-08 |
| 3 | `0x3F13CCB1` | `0x3F13CD41` | 1.416e-05 | 7.047e-07 |
| 7 | `0x3EC184AA` | `0x3EC184AA` | 2.113e-06 | 2.113e-06 |
| 1.5 | `0x3F51060B` | `0x3F51060B` | 2.299e-06 | 2.299e-06 |
| 100 | `0x3DCCCCF3` | `0x3DCCCCF3` | 2.846e-06 | 2.846e-06 |
| 0.001 | `0x41FCF9C8` | `0x41FCFB9C` | 2.569e-05 | 2.539e-06 |
| 1000 | `0x3D01634D` | `0x3D01870D` | 1.073e-03 | 5.012e-06 |
| 1e-10 | `0x47C35021` | `0x47C35021` | 2.585e-06 | 2.585e-06 |
| 1e+10 | `0x3727C5C4` | `0x3727C5C3` | 2.158e-06 | 2.067e-06 |
| 1e-30 | `0x58635F94` | `0x58635FAB` | 1.421e-06 | 1.228e-07 |
| 1e+30 | `0x26901C0B` | `0x26901D7E` | 3.916e-05 | 1.170e-07 |
| 1.1754944e-38 | `0x5E6BD88D` | `0x5E6BD8D8` | 5.394e-01 | 5.394e-01 |
| 3.4028235e+38 | `0x1EEBD88D` | `0x1EEBD8D8` | 5.394e-01 | 5.394e-01 |
| 2.3841858e-07 | `0x44FF9FAA` | `0x4500002A` | 1.470e-03 | 5.007e-06 |
| 0.015625 | `0x40FF9FAA` | `0x4100002A` | 1.470e-03 | 5.007e-06 |
| `+inf` | `0x00000000` | `0x00000000` | — | — |

Excluding the two fp32 extremes (n = 17):

- max rel error **1.470e-03 → 5.012e-06, 293× better**
- median rel error 1.416e-05 → 2.539e-06
- **no value regressed** — 0 of 17 are worse on the branch

The two `5.394e-01` rows are the **pre-existing 54 % error at the fp32 extremes**, where the
bf16-magic seed runs out of range. Identical before and after; this branch neither causes nor
fixes it, and it wants its own issue. (`sqrt_custom` has the same defect at the same values.)

## 5. Accuracy — the SDPA reciprocal column (path 3)

This is the change least covered by the existing records, and it is not only an accuracy
change — **`RecipLegacy` was returning `1/|x|`**. `_reciprocal_compat_` returns a magnitude and
the legacy branch of `calculate_recip_first_column` never restored the sign;
`sfpu_reciprocal_iter` ends in `copysgn(y, in)`, so the sign now survives.

Raw hardware output, `RecipLegacy_apxNo`, `fp32_e2e`, first eight written cells:

```
              main                              branch
x=  1.25    hw= 0.8          ok            hw= 0.8          ok
x= -1.36719 hw= 0.7314286    SIGN LOST     hw=-0.7314286    ok
x=  1.48438 hw= 0.6736842    ok            hw= 0.6736842    ok
x= -1.60156 hw= 0.6243901    SIGN LOST     hw=-0.6243902    ok
x=  1.71875 hw= 0.5818118    ok            hw= 0.5818182    ok
x= -1.83594 hw= 0.5446141    SIGN LOST     hw=-0.5446808    ok
x=  1.95312 hw= 0.5116       ok            hw= 0.512        ok
x= -2.07031 hw= 0.4830107    SIGN LOST     hw=-0.4830189    ok
```

Across all 12 `(variant, precision)` combinations, on the sampled cells:

| Variant | sign lost, `main` | sign lost, branch | max rel (sign-ok), `main` | branch |
|---|---|---|---|---|
| RecipLegacy apxNo bf16_dest | 4/8 | **0/8** | 2.686e-03 | 3.143e-03 |
| RecipLegacy apxNo fp32_dest | 4/8 | **0/8** | 2.686e-03 | 3.143e-03 |
| RecipLegacy apxNo fp32_e2e | 4/8 | **0/8** | 7.812e-04 | **6.892e-08** |
| RecipLegacy apxYes bf16_dest | 4/8 | **0/8** | 3.107e-02 | 6.348e-03 |
| RecipLegacy apxYes fp32_dest | 4/8 | **0/8** | 2.725e-02 | 3.143e-03 |
| RecipLegacy apxYes fp32_e2e | 4/8 | **0/8** | 2.795e-02 | 1.021e-04 |
| RecipIter *(control)* × 6 | 0/8 | 0/8 | identical | identical |

Every negative input lost its sign on `main`; none does on the branch. On the fp32-end-to-end
pipeline accuracy improves by about four orders of magnitude (7.812e-04 → 6.892e-08).

Two honest qualifications. The two `bf16_dest`/`fp32_dest` rows at `apxNo` move the *wrong*
way on max relative error, 2.686e-03 → 3.143e-03: both kernels are dominated by the bf16
output grid there and simply land on different sides of it. Note what the branch value is —
3.143e-03, which is exactly `RecipIter`'s figure on the same rows. That is the intended
outcome: `RecipLegacy` now *is* the iterative kernel, so it inherits its numbers rather than
being independently better or worse. And the `RecipIter` control is bit-for-bit identical
across the two trees on all six of its combinations.

## 6. Performance — Wormhole n300, `MATH_ISOLATE` cycles per tile

Three runs per tree. "Separated" means the run ranges do not overlap.

| Op | Variants | min | median | max | Separated |
|---|---|---|---|---|---|
| **Reciprocal** (path 1, public default) | 60 | **−52.70 %** | **−45.22 %** | **−43.29 %** | 60/60 |
| **RsqrtCompat** (path 2) | 4 | **−38.93 %** | **−34.94 %** | **−25.00 %** | 4/4 |
| Rsqrt *(control, `legacy_compat=false`)* | 120 | −0.00 % | +0.00 % | +0.00 % | 0/120 |

Both changed paths get **faster**, which is the unusual part of this branch: the pole fix is
not paid for, it is refunded.

`RsqrtCompat`, every variant:

| approx | dest_acc | `main` | branch | Δ |
|---|---|---|---|---|
| No | No | 2269.23 | 1566.79 | **−30.96 %** |
| No | Yes | 2216.88 | 1662.77 | **−25.00 %** |
| Yes | No | 1150.83 | 702.79 | **−38.93 %** |
| Yes | Yes | 1150.88 | 702.84 | **−38.93 %** |

`Reciprocal`, absolute cycles on the two representative pairs:

| formats | dest_acc | approx | `main` | branch | Δ |
|---|---|---|---|---|---|
| Float16_b→Float16_b | No | No | 1214.92 | 574.81 | −52.69 % |
| Float16_b→Float16_b | No | Yes | 990.87 | 542.81 | −45.22 % |
| Float16_b→Float16_b | Yes | No | 1182.83 | 670.80 | −43.29 % |
| Float16_b→Float16_b | Yes | Yes | 990.83 | 542.76 | −45.22 % |
| Float32→Float32 | No | No | 1215.14 | 574.98 | −52.68 % |
| Float32→Float32 | No | Yes | 991.11 | 543.01 | −45.21 % |
| Float32→Float32 | Yes | No | 1165.62 | 653.65 | −43.92 % |
| Float32→Float32 | Yes | Yes | 973.62 | 525.65 | −46.01 % |

The win is flat across the format matrix — every one of the 16 format pairs lands at −48.9 %
(`dest_acc=No`) or −44.3 % to −45.0 % (`dest_acc=Yes`), so this is the kernel body and not a
format-conversion effect. `L1_TO_L1` follows at −38 % to −52 %; `UNPACK_ISOLATE` is flat
(−0.08 % to +2.10 %), as it must be.

**Not measured: the SDPA reciprocal's cycles.** `perf_sfpu_reduce_sdpa.py` instruments only
`ReduceColumn`, so the reciprocal column has no perf coverage at all. Its kernel body is the
same `sfpu_reciprocal_iter` that path 1 switches to, so the direction is not in doubt, but this
record will not put a number on it.

## 7. Where the cycles go

Instruction counts of the `MATH_ISOLATE` `math.elf`, paired by build hash
(`Float16_b→Float16_b`, `approx=No`, the two `dest_acc` values):

| Kernel | total instructions | SFPU ops |
|---|---|---|
| `reciprocal` (legacy path) | 635 → 535, 624 → 546 | **131 → 44**, 115 → 50 |
| `rsqrt_compat` | 1126 → 817, 1115 → 870 | **626 → 315**, 610 → 363 |

Both roughly halve their SFPU op count. The mnemonic histogram says why — the compat kernels
work by taking the exponent field apart:

```
reciprocal:    sfpexexp 4->0   sfpsetexp 4->0   sfpiadd 4->0
               sfpsetcc 4->0   sfpencc 18->0    sfpnop 16->4
               sfpmad 0->8     sfpsetman 0->4
rsqrt_compat:  sfpexexp 32->0  sfpsetexp 32->0  sfpiadd 34->2
               sfpsetcc 34->2  sfpencc 34->2    sfpnop 144->48
               sfpmad 0->64    sfpsetman 0->32
```

The exponent-difference arithmetic — `SFPEXEXP` / `SFPSETEXP` / `SFPIADD`, with the predicate
machinery `SFPSETCC` / `SFPENCC` around it — disappears entirely, replaced by `SFPMAD` Newton
iterations and a mantissa insert. That arithmetic is also exactly what had no pole guard
(`126 - exexp(0) = 253`), so the defect and the cost were the same code. The `SFPNOP` count
falling by two thirds is the second half of the story: the predicated exponent code stalls in a
way the multiply-add chain does not.

## 8. Suite results

Wormhole n300, `test_sfpu_unary.py`:

| Sweep | `main` | branch |
|---|---|---|
| `-k "Reciprocal or Rsqrt"` | 497 passed, 15 xfailed | **505 passed, 7 xfailed** |
| `-k edges` | 491 passed, 23 xfailed | **499 passed, 15 xfailed** |

0 xpassed and 0 failed in all four runs. Both deltas are +8 passed / −8 xfailed, and both are
the same 8: `RsqrtCompat`'s per-combination xfails for the saturating pole, which the fix
retires. Nothing else moves.

`test_sfpu_sdpa.py`: 65 passed on both trees. That is **not** evidence the SDPA path is
unchanged — the branch updates `SdpaSfpuGolden` in step with the kernel (`RecipLegacy` from
`1/|x|` to `1/x`), so each tree is asserting against its own expectation. §5 is where the
change is visible; this line only says neither tree is internally inconsistent.

## 9. What was not measured

| # | Item | Why |
|---|---|---|
| 1 | SDPA reciprocal cycles | No perf instrumentation for that column (§6). |
| 2 | Blackhole cycle counts | No BH silicon on this host. BH compiles clean — 440 unary and 65 SDPA variants — but a cycle figure would be a guess. |
| 3 | Quasar | Out of scope; untouched. |
| 4 | `_reciprocal_compat_`'s surviving consumer | Blackhole sampling keeps it alive and documents a bit-identity requirement for blaze. Deliberately untouched by this branch, so there is nothing to measure. |
| 5 | End-to-end model impact | These are LLK-level cycle counts. A −45 % `MATH_ISOLATE` on `recip()` is not a −45 % model-level anything. |
