# Measurements — reciprocal compat pole, accuracy and performance vs `main`

Accuracy and performance of the fix on this branch, measured against the branch point.
Read-only record: no code, no test changes.

| | |
|---|---|
| Branch | `ldjurovic/sfpu_52930_reciprocal_compat_pole` |
| Baseline | `origin/main` @ `35ec0aba7a8` — the branch point, so the only kernel delta is this branch's |
| Silicon | Wormhole n300 (UMD chip 0) |
| Blackhole | compile-verified only — no BH hardware on this host |
| Date | 2026-08-20 |

> **This record supersedes the Option A figures.** An earlier revision of this branch shipped
> **Option A** (redirect the compat consumers to `sfpu_reciprocal_iter`). That was withdrawn on
> review — see §8 — and the branch now ships **Option B**, the in-place pole guard the fix plan
> recommended. Everything below measures Option B. `RESULTS_52930_reciprocal_compat_pole.md`
> and `MEASUREMENTS_52930_reciprocal_compat_pole.md` retain the A-vs-B bake-off data, with
> corrected conclusions.

---

## 1. Headline

| | |
|---|---|
| **Accuracy** | **No value changes except the pole.** 16 of 168 recorded values move, all of them `1/±0`, all from a finite to an infinity. Everything else is bit-identical to `main`. |
| **Performance** | **+128 cycles per tile, flat.** `RsqrtCompat` **+5.7 … +11.1 %**, the public `recip()` default **+10.5 … +13.2 %**. `Rsqrt` and every non-legacy path: unchanged. |
| **Compatibility** | **Preserved.** Every `legacy_compat = true` caller keeps its bit pattern on every input it can legally pass. |

The whole change is one `v_if` in `_reciprocal_compat_`, on both arches:

```c
    v_if (sfpi::setsgn(in, 0) == 0.0F)
    {
        out = std::numeric_limits<float>::infinity();
    }
    v_endif;
```

## 2. Scope — what the guard reaches

`_reciprocal_compat_` is the shared primitive, so one guard fixes every consumer of it:

| Consumer | Reached by | Pole before | Pole after |
|---|---|---|---|
| `_calculate_rsqrt_compat_` | `rsqrt_tile<true>` → `RsqrtCompat` | `1.7014e38` | `+inf` |
| `_calculate_reciprocal_compat_` | **the public `recip()` default** (`recip.h` declares `legacy_compat = true`) | `1.7014e38` | `+inf` |
| SDPA `calculate_recip_first_column<true>` | SDPA `RecipLegacy` | `1.7014e38` | `+inf` |
| Blackhole `sampling_recip_value<true>` | blaze sampling | `1.7014e38` | `+inf` |

The last row is the one with a documented bit-identity contract ("must stay bit-identical for
blaze"). It also documents `Callers must pass in > 0`, so the guard cannot fire on any input
that contract permits: bit-identical across its entire legal domain, and the pole it never
reaches is now correct anyway.

**A coverage gap, found here and closed on this branch.** Row 2 is the largest user-visible
change and the tt-llk suite did not exercise it at all: `sfpu_operations.h` called
`calculate_reciprocal` and `recip_init` without a `legacy_compat` argument, so both defaulted to
`false` and the suite only ever built the kernel the public API does *not* dispatch. The §4 and §6
figures for that path were therefore taken with both trees patched identically to pass `true` — an
instrument, reverted before the correctness sweeps in §7 and never committed.

That is now fixed permanently. A `MathOperation.ReciprocalCompat` op covers the default path,
wired exactly the way `RsqrtCompat` covers legacy rsqrt: a `SfpuType::reciprocal_compat` enum
value on both arches, an init and call branch in `sfpu_operations.h` passing
`legacy_compat = true`, and the op registered in `llk_params.py`, `sfpu_domains.py` (same domain
and pole as `Reciprocal`) and `golden_generators.py` (same `_reciprocal` golden). It brings 8 sweep
variants and 8 edge variants.

It is a real regression test rather than a passing formality, and that was checked rather than
assumed: built against `main`'s **unguarded** kernel, its edge variants **fail 8 of 8** — golden
`inf`, kernel `1.7014e38`. Against the guarded kernel all 16 pass. Anyone reverting the guard now
breaks CI.

## 3. Accuracy — `RsqrtCompat`

168 recorded values (21 inputs × 8 format/dest_acc combinations). **Exactly 8 change**, one per
combination, and every one of them is the pole:

| x | `main` | branch | IEEE |
|---|---|---|---|
| `0.0` | `0x7EFFFD9E` = 1.7013e38 (`0x7F000000` = 1.7014e38 where bf16-packed) | `0x7F800000` = `+inf` | `+inf` |

Every other value — `1`, `2`, `4`, `0.25`, `0.5`, `3`, `7`, `1.5`, `100`, `1e±3`, `1e±10`,
`1e±30`, `1.1754944e-38`, `3.4028235e38`, `2.3841858e-07`, `0.015625`, `+inf` — is
**byte-for-byte identical** to `main` on all 8 combinations.

That is the acceptance criterion the fix plan set, met exactly: *"Preserves every existing value
except the pole."*

Accuracy on non-pole input therefore does not improve either. The legacy kernel's error profile
is unchanged and still worse than the modern reciprocal's — max relative error `1.470e-03`
against `5.012e-06` on `Float32→Float32 dest_acc=Yes`. Closing that gap is Option A's business,
not this branch's; see §8.

## 4. Accuracy — the public `recip()` default

Same shape, and the negative pole is measurable here because the probe carries `-0.0`. 16 of the
recorded values change, 8 for each pole:

| x | `main` | branch | IEEE | Where |
|---|---|---|---|---|
| `+0.0` | `0x7EFFFD9E` / `0x7F000000` | `0x7F800000` = `+inf` | `+inf` | all 8 combinations |
| `-0.0` | `0xFEFFFD9E` = −1.7013e38 | `0xFF800000` = **`-inf`** | `-inf` | the unpack-to-dest pipelines, where a real `-0.0` survives to the LREG |
| `-0.0` | `0x7EFFFD9E` / `0x7F000000` | `0x7F800000` = `+inf` | `-inf` | the pipelines that flush `-0.0` to `+0.0` before the kernel sees it |

The middle row is why the guard compares on `setsgn(in, 0)` rather than a bare `in == 0.0F`:
`SFPSETCC`'s contract excludes negative zero, so the bare compare does not fire for `-0.0` and
would leave that pole at 1.7e38. Clearing the sign first is what lets `-0.0` reach the guard, and
the caller-side `v_if (in < 0.0)` then re-signs the magnitude — giving IEEE's `-inf`.

The bottom row is the unpack path flushing the sign before the kernel runs, not a defect in the
guard.

Every non-pole value is bit-identical, and the relative-error table over the wide exponent sweep
is unchanged in every cell:

| Variant | max rel err, `main` | max rel err, branch |
|---|---|---|
| Float16_b→Float16_b acc=No / acc=Yes | 6.653e-03 | 6.653e-03 |
| Float16_b→Float32 acc=No | 6.653e-03 | 6.653e-03 |
| Float16_b→Float32 acc=Yes | 4.595e-03 | 4.595e-03 |
| Float32→Float16_b acc=No / acc=Yes | 2.734e-03 | 2.734e-03 |
| Float32→Float32 acc=No | 2.734e-03 | 2.734e-03 |
| Float32→Float32 acc=Yes | 3.636e-05 | 3.636e-05 |

## 5. Accuracy — SDPA and sampling

SDPA `RecipLegacy` keeps `_reciprocal_compat_`, so its numerics are unchanged apart from the
pole, and its golden is untouched by this branch. In particular **the `1/|x|` sign behaviour is
preserved** — `_reciprocal_compat_` still returns a magnitude and the legacy branch still does
not re-sign it. That is a real defect, and it is deliberately left alone here: fixing it changes
values for every legacy SDPA caller, which is the same compatibility question §8 is about. It
wants its own issue.

Blackhole sampling: compile-verified only (the op is Blackhole-only and this host has none). The
guard cannot fire inside its documented domain, so blaze's contract holds by construction.

## 6. Performance — Wormhole n300, `MATH_ISOLATE` cycles per tile

Three runs per tree. All separated — the fastest branch run is still slower than the slowest
baseline run.

| Op | Variants | min | median | max | Separated |
|---|---|---|---|---|---|
| **`recip()` legacy default** | 60 | +10.53 % | **+11.95 %** | +13.15 % | 60/60 |
| **`RsqrtCompat`** | 4 | +5.68 % | **+8.76 %** | +11.12 % | 4/4 |
| `Rsqrt` *(control, `legacy_compat=false`)* | 120 | −0.00 % | +0.00 % | +0.00 % | 0/120 |

`RsqrtCompat`, every variant:

| approx | dest_acc | `main` | branch | Δ cycles | Δ % |
|---|---|---|---|---|---|
| No | No | 2269.23 | 2398.23 | +129.00 | +5.68 % |
| No | Yes | 2216.88 | 2358.69 | +141.81 | +6.40 % |
| Yes | No | 1150.83 | 1278.83 | +128.00 | +11.12 % |
| Yes | Yes | 1150.88 | 1278.82 | +127.94 | +11.12 % |

`recip()` legacy default, representative (`Float16_b→Float16_b`):

| approx | dest_acc | `main` | branch | Δ cycles | Δ % |
|---|---|---|---|---|---|
| No | No | 1214.92 | 1342.92 | +128.00 | +10.54 % |
| No | Yes | 1182.83 | 1310.83 | +128.00 | +10.82 % |
| Yes | No | 990.87 | 1118.91 | +128.04 | +12.92 % |
| Yes | Yes | 990.83 | 1118.83 | +128.00 | +12.92 % |

**The cost is a constant +128 cycles per tile, not a percentage.** It reads as +5.7 % on the
2269-cycle `RsqrtCompat` variant and +12.9 % on the 991-cycle approximate `recip`, for the same
absolute work. A tile is 32 SFPU vector iterations, so that is **4 cycles per iteration**.

The static instruction count rises by 64 in an 8×-unrolled loop — 8 per copy, against 4 cycles
per iteration measured. About half the added instructions therefore issue in slots the
surrounding code was already stalling in; this does not close as an exact
instructions-times-iterations identity the way the `pow` and `sqrt_custom` guards did, and is
recorded as measured rather than reasoned.

Added by the guard, per unrolled copy: `SFPSETSGN`, `SFPLOADI` ×2 (the compare constant and the
infinity), `SFPSETCC`, `SFPENCC`.

**A cheaper variant exists and was measured.** Dropping the `setsgn` for a bare `in == 0.0F`
costs +7.89…+9.86 % on `recip` and +4.01…+8.34 % on `RsqrtCompat` — roughly 2 points less. It was
**not** taken: it leaves `1/-0` at 1.7e38, so it fixes one pole of the two.

## 7. Suite results

Wormhole n300:

| Sweep | `main` | branch |
|---|---|---|
| `test_sfpu_unary.py -k "Reciprocal or Rsqrt"` | 497 passed, 15 xfailed | **505 passed, 7 xfailed** |
| `test_sfpu_unary.py -k edges` | 491 passed, 23 xfailed | **499 passed, 15 xfailed** |
| `test_sfpu_sdpa.py` | 65 passed | **65 passed** |

0 xpassed and 0 failed throughout. Both unary deltas are the same 8: `RsqrtCompat`'s
per-combination xfails for the saturating pole, which the guard retires.

Blackhole compiles clean — 440 unary variants, 116 SDPA + sampling variants. No BH silicon here,
so no cycle figures.

## 8. Why Option B, when Option A was faster

Option A — redirecting the compat consumers to `sfpu_reciprocal_iter` — is **26–39 % faster on
`RsqrtCompat` and 43–53 % faster on the public `recip()` default**, and improves accuracy by 293×
and 610× respectively. Those figures are real and are kept in
`MEASUREMENTS_52930_reciprocal_compat_pole.md`.

It was withdrawn because it changes numerics for callers who explicitly asked for the old ones.
The fix plan said Option A "is only viable once someone can say who depends on the legacy bit
pattern… **Do not guess**", and the answer, found on review, is eight production normalization
kernels that hard-code `rsqrt_tile<true>`:

| File | Sites |
|---|---|
| `experimental/ccl/dit_fused_distributed_rmsnorm/…/dit_layernorm_fused_compute.cpp` | 2 |
| `normalization/groupnorm/…/{groupnorm, groupnorm_sharded_v2, welford_groupnorm, welford_groupnorm_sharded_v2}.cpp` | 4 |
| `normalization/layernorm_distributed/…/layernorm_post_allgather_welford.cpp` | 1 |
| `experimental/transformer/dit_layernorm_post_all_gather/…/layernorm_post_allgather_welford.cpp` | 1 |
| `experimental/ccl/rms_allgather/…/rms_compute.cpp` | 1 |

and they say why:

> ```
> // legacy rsqrt to match the composite dit_layernorm baseline (it uses
> // rsqrt_tile<true>); the non-legacy default diverges on low-variance rows.
> ```
> — `dit_layernorm_fused_compute.cpp:267-268`, and at :345 `// DV = 1/std (legacy, matches baseline)`

Someone had already compared the two kernels and chosen legacy deliberately. Option A would have
changed those model baselines silently, under a flag named `legacy_compat`. The earlier search
for that dependency looked for the phrase *bit-identical* and so found only the Blackhole
sampling file; these callers express it as *matches baseline*.

Option A remains the right end state — a second reciprocal implementation that is 610× less
accurate is not worth keeping forever. It is a follow-up, gated on migrating and revalidating
those eight callers, not something to land underneath them.

## 9. What was not measured

| # | Item | Why |
|---|---|---|
| 1 | Blackhole cycles | No BH silicon on this host. Compiles clean; the guard is the same source on both arches. |
| 2 | Blackhole sampling numerics | Blackhole-only op, no BH silicon. The guard is unreachable inside its documented `in > 0` domain. |
| 3 | SDPA reciprocal cycles | `perf_sfpu_reduce_sdpa.py` instruments only `ReduceColumn`; that column has no perf coverage. Its body is the same `_reciprocal_compat_`, so the +128 cycles/tile applies, but this record will not put a measured number on it. |
| 4 | Quasar | Out of scope; untouched. |
| 5 | End-to-end model impact | These are LLK-level cycle counts. +128 cycles on a normalization kernel's rsqrt is not +128 cycles on a model. |
