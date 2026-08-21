# `_reciprocal_compat_(1/0)` on Wormhole and Blackhole

Root cause, fix, and the measured accuracy and performance record for finding 4 of
[issue #52930](https://github.com/tenstorrent/tt-metal/issues/52930). The other four findings in
that issue are not addressed here.

| | |
|---|---|
| Kernels changed | `ckernel_sfpu_rsqrt_compat.h`, Wormhole B0 and Blackhole (identical files, one patch) |
| Silicon | Wormhole n300 and Blackhole p100a |
| Performance | measured on Wormhole n300 and Blackhole p100a |
| Quasar | unchanged — no equivalent compat reciprocal |
| SFPI | 7.69.0 (the pinned version) |

---

## 1. What was wrong

`_reciprocal_compat_(0)` returned the maximum finite `1.7e38` where infinity is the answer.

The kernel computes the reciprocal by exponent difference. It first discards the input's
magnitude — `setexp(val, 126)` — refines `1/0.5` with Newton-Raphson, and then rebuilds the
exponent:

```c
sfpi::vInt orig_exp = exexp(in);        // exexp(0.0)      = -127
sfpi::vInt new_exp  = exexp(result);    // exexp(1.99997…) =    0
new_exp -= orig_exp;                    // 0 - (-127)      =  127
new_exp += 126;                         //                 =  253
v_if (new_exp < 0) { result = 0.0F; new_exp = 0; }   // guards overflow only
return setexp(result, new_exp);
```

`exexp(0.0)` is `0 - 127 = -127`, so the difference lands on **253** — an ordinary finite
exponent — where infinity needs 255. The surviving mantissa ≈ 1.99997 rides along and the result
is `1.99997 × 2^(253-127)` = `0x7EFFFD9E` = 1.70135e38, matching the measurement to the bit.

This is not saturation and no clamp is involved. The function's only guard, `v_if (new_exp < 0)`,
covers the *opposite* end — an input so large that the reciprocal underflows. The zero end was
never handled.

The `0x7F000000` the issue recorded is the bf16-rounded view of that value; the single computed
value is `0x7EFFFD9E`, and the fix had to be verified against the fp32 pattern.

Plain `Rsqrt` and the modern `Reciprocal` do not diverge at the same pole. They use
`sfpu_reciprocal_iter`, which builds its scale factor as `~in.Exp` *specifically* so the poles
fall out for free — its own comment says so. A correct implementation already existed in the tree
next to the broken one; §3 is about why this PR does not simply switch to it.

`_reciprocal_compat_(±inf)` is handled by accident rather than by design: `exexp(inf) = 128`, so
`new_exp = 0 - 128 + 126 = -2 < 0`, which the underflow guard catches and returns `0.0`. That end
happens to be right.

## 2. The fix

One `v_if` in the shared primitive, on both arches:

```c
sfpi::vFloat out = setexp(result, new_exp);

v_if (sfpi::setsgn(in, 0) == 0.0F)
{
    out = std::numeric_limits<float>::infinity();
}
v_endif;
return out;
```

Two details are deliberate.

**It is applied after the `setexp`, not alongside the underflow guard.** Writing an infinity into
`result` first and then running `setexp` over it would overwrite the exponent field that makes it
an infinity. The alternative — setting `new_exp = 255` in the same block — works but is easier to
get wrong.

**The compare is on `setsgn(in, 0)`, not a bare `in == 0.0F`.** `SFPSETCC`'s contract excludes
negative zero, and measured, the bare compare does not fire for `-0.0` and leaves that pole at
1.7e38. Clearing the sign first is what lets `-0.0` reach the guard at all; the caller-side
`v_if (in < 0.0)` then re-signs the magnitude, giving IEEE's `-inf`. The bare variant was built
and measured on both arches — it costs exactly one instruction less and fixes one pole of the two,
so it was not taken. §5.3 prices it.

Because the guard sits in the shared primitive, one edit reaches every consumer:

| Consumer | Reached by | Pole before | Pole after |
|---|---|---|---|
| `_calculate_rsqrt_compat_` | `rsqrt_tile<true>` → `RsqrtCompat` | `1.7014e38` | `+inf` |
| `_calculate_reciprocal_compat_` | **the public `recip()` default** (`recip.h` declares `legacy_compat = true`) | `1.7014e38` | `+inf` |
| SDPA `calculate_recip_first_column<true>` | SDPA `RecipLegacy` | `1.7014e38` | `+inf` |
| Blackhole `sampling_recip_value<true>` | blaze sampling | `1.7014e38` | `+inf` |

The last row is the one with a documented bit-identity contract ("must stay bit-identical for
blaze"). It also documents `Callers must pass in > 0`, so the guard cannot fire on any input that
contract permits: bit-identical across its entire legal domain, and the pole it never reaches is
now correct anyway. §6.2 measures that row on Blackhole silicon rather than asserting it.

## 3. Why this shape of fix, and not the faster one

The fix plan offered two options and this branch **originally shipped the other one** —
redirecting the compat consumers to `sfpu_reciprocal_iter`, which fixes the pole and is
**26–39 % faster** on `RsqrtCompat`, **43–53 % faster** on the public `recip()` default, and
293×/610× more accurate. Those figures are real, and they were withdrawn on review.

Option A changes numerics for every `legacy_compat = true` caller — the opposite of what a flag
named `legacy_compat` promises. The plan said Option A "is only viable once someone can say who
depends on the legacy bit pattern… **Do not guess**", and the answer is eight production
normalization kernels that hard-code `rsqrt_tile<true>`:

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
> — `dit_layernorm_fused_compute.cpp:267-268`, and at `:345` `// DV = 1/std (legacy, matches baseline)`

Someone had already compared the two kernels and chosen legacy deliberately. The earlier search
for that dependency looked for the phrase *bit-identical*, found only the Blackhole sampling file,
and concluded the contract was unclaimed; these callers express it as *matches baseline*.

Consolidating on one reciprocal is still the right end state — a second implementation that is
610× less accurate is not worth keeping forever — but it is a follow-up gated on migrating and
revalidating those eight callers, not something to land underneath them.

## 4. Accuracy

Raw hardware bit patterns for a fixed stimulus list, dumped from silicon and diffed byte for byte
against the same tree with the guard reverted. This is what the shipped sweep cannot do: it
compares against golden with tolerances, not against the previous build, so "every other value is
bit-identical" is not a claim it can check.

The stimulus list is the pole (both signs) plus a spread that exercises the exponent-difference
arithmetic across the range: `1`, `2`, `4`, `0.25`, `0.5`, `3`, `7`, `1.5`, `100`, `1e±3`,
`1e±10`, `1e±30`, `1.1754944e-38` (smallest fp32 normal), `3.4028235e38` (largest fp32 finite),
`2.3841858e-07`, `0.015625`, `+inf`.

### 4.1 Wormhole n300

168 recorded values for `RsqrtCompat` (21 inputs × 8 combinations). **Exactly 8 change**, one per
combination, and every one of them is the pole:

| x | before | after | IEEE |
|---|---|---|---|
| `0.0` | `0x7EFFFD9E` = 1.7013e38 (`0x7F000000` where bf16-packed) | `0x7F800000` = `+inf` | `+inf` |

Every other value is byte-for-byte identical on all 8 combinations.

For the public `recip()` default the negative pole is measurable too, because that probe carries
`-0.0`. 16 values change, 8 per pole:

| x | before | after | IEEE | Where |
|---|---|---|---|---|
| `+0.0` | `0x7EFFFD9E` / `0x7F000000` | `0x7F800000` = `+inf` | `+inf` | all 8 combinations |
| `-0.0` | `0xFEFFFD9E` = −1.7013e38 | `0xFF800000` = **`-inf`** | `-inf` | the unpack-to-dest pipelines, where a real `-0.0` survives to the LREG |
| `-0.0` | `0x7EFFFD9E` / `0x7F000000` | `0x7F800000` = `+inf` | `-inf` | the pipelines that flush `-0.0` to `+0.0` before the kernel sees it |

The bottom row is the unpack path flushing the sign before the kernel runs, not a defect in the
guard.

Accuracy on non-pole input therefore does not improve either. The legacy kernel's error profile is
unchanged and still worse than the modern reciprocal's — max relative error `1.470e-03` against
`5.012e-06` on `Float32→Float32 dest_acc=Yes` for `rsqrt_compat`, and the whole relative-error
table over the wide exponent sweep is unchanged in every cell:

| Variant | max rel err, before | max rel err, after |
|---|---|---|
| Float16_b→Float16_b acc=No / acc=Yes | 6.653e-03 | 6.653e-03 |
| Float16_b→Float32 acc=No | 6.653e-03 | 6.653e-03 |
| Float16_b→Float32 acc=Yes | 4.595e-03 | 4.595e-03 |
| Float32→Float16_b acc=No / acc=Yes | 2.734e-03 | 2.734e-03 |
| Float32→Float32 acc=No | 2.734e-03 | 2.734e-03 |
| Float32→Float32 acc=Yes | 3.636e-05 | 3.636e-05 |

Closing that gap is Option A's business, not this branch's; see §3.

### 4.2 Blackhole p100a

Blackhole is no longer compile-verified only. The same method, run on p100a silicon, against the
same tree with only the guard reverted — so the sole delta between the two records is the `v_if`.

Blackhole reaches 5 of the 8 (format pair, `dest_acc`) combinations rather than Wormhole's 8:
`dest_acc=No` supports only `Float32→Float32` there, the same gate the shipped sweep applies. Both
`RsqrtCompat` and `ReciprocalCompat` are driven, so the public `recip()` default is measured
directly rather than through an instrument — see §7.

**220 rows compared (22 inputs × 5 combinations × 2 ops), 20 differ, and every one of them is a
`±0` input. The count of differing rows whose input is not `±0` is zero, and no pole row was left
behind — all 20 moved.**

| Op | Path | `x` | before | after | IEEE |
|---|---|---|---|---|---|
| both | fp32 output | `+0` | `0x7EFFFD9E` | `0x7F800000` = `+inf` | `+inf` |
| both | bf16 output | `+0` | `0x7F000000` | `0x7F800000` = `+inf` | `+inf` |
| `ReciprocalCompat` | `Float32→Float32`, `dest_acc=Yes` | `-0` | `0xFEFFFD9E` | **`0xFF800000` = `-inf`** | `-inf` |
| `ReciprocalCompat` | `Float32→Float16_b`, `dest_acc=Yes` | `-0` | `0xFF000000` | **`0xFF800000` = `-inf`** | `-inf` |
| `ReciprocalCompat` | the other three | `-0` | positive finite | `+inf` | `-inf` |
| `RsqrtCompat` | all five | `-0` | positive finite | `+inf` | — |

The two `-inf` rows are the ones that matter, and they are exactly the unpack-to-dest pipelines —
fp32 input with `dest_acc=Yes`, the only Blackhole combinations where a real `-0.0` survives to the
LREG. They confirm on BH silicon what Wormhole showed: `setsgn(in, 0)` rather than a bare
`in == 0.0F` is what makes the negative pole reachable at all, and once it is reached the
caller-side `v_if (in < 0.0)` re-signs the magnitude to IEEE's answer. On the other three
combinations the unpack flushes `-0.0` to `+0.0` before the kernel runs, so `+inf` there is the
flush showing through and not a defect in the guard — the probe records the delivered input bits,
so this is measured rather than inferred.

`RsqrtCompat(-0)` is `+inf` on every combination because its caller-side re-sign tests
`in < 0.0`, which `-0.0` does not satisfy.

Every other value — `1`, `2`, `4`, `0.25`, `0.5`, `3`, `7`, `1.5`, `100`, `1e±3`, `1e±10`,
`1e±30`, `1.1754944e-38`, `3.4028235e38`, `2.3841858e-07`, `0.015625`, `+inf` — is byte-for-byte
identical on all five combinations and both ops.

## 5. Performance

`perf_eltwise_unary_sfpu.py` with CI's flags (`--speed-of-light`), `MATH_ISOLATE` on the
`TILE_LOOP` marker, cycles per tile. Three runs per tree, from separate clean build roots so
neither tree can serve the other a stale ELF.

`Rsqrt` and `Reciprocal` are carried as controls: both are `legacy_compat = false`, so they reach
`sfpu_reciprocal_iter` and not the patched primitive. They must be flat, and they are.

### 5.1 Wormhole n300

| Op | Variants | min | median | max | Separated |
|---|---|---|---|---|---|
| **`recip()` legacy default** | 60 | +10.53 % | **+11.95 %** | +13.15 % | 60/60 |
| **`RsqrtCompat`** | 4 | +5.68 % | **+8.76 %** | +11.12 % | 4/4 |
| `Rsqrt` *(control)* | 120 | −0.00 % | +0.00 % | +0.00 % | 0/120 |

`RsqrtCompat`, every variant:

| approx | dest_acc | before | after | Δ cycles | Δ % |
|---|---|---|---|---|---|
| No | No | 2269.23 | 2398.23 | +129.00 | +5.68 % |
| No | Yes | 2216.88 | 2358.69 | +141.81 | +6.40 % |
| Yes | No | 1150.83 | 1278.83 | +128.00 | +11.12 % |
| Yes | Yes | 1150.88 | 1278.82 | +127.94 | +11.12 % |

`recip()` legacy default, representative (`Float16_b→Float16_b`):

| approx | dest_acc | before | after | Δ cycles | Δ % |
|---|---|---|---|---|---|
| No | No | 1214.92 | 1342.92 | +128.00 | +10.54 % |
| No | Yes | 1182.83 | 1310.83 | +128.00 | +10.82 % |
| Yes | No | 990.87 | 1118.91 | +128.04 | +12.92 % |
| Yes | Yes | 990.83 | 1118.83 | +128.00 | +12.92 % |

### 5.2 Blackhole p100a

Same method and the same metric on p100a, baseline and branch three runs each. The measurement is
effectively noise-free: the largest run-to-run spread over all 188 measured variants is **0.039
cycles**, and the eight legacy-compat variants repeat to within 0.024 cycles.

| Op | Variants | min | median | max | Separated |
|---|---|---|---|---|---|
| **`ReciprocalCompat`** (the public `recip()` default) | 4 | +10.53 % | **+11.87 %** | +12.92 % | 4/4 |
| **`RsqrtCompat`** | 4 | +5.88 % | **+8.54 %** | +11.12 % | 4/4 |
| `Reciprocal` *(control, `legacy_compat=false`)* | 60 | +0.00 % | +0.00 % | +0.00 % | 0/60 |
| `Rsqrt` *(control, `legacy_compat=false`)* | 120 | +0.00 % | +0.00 % | +0.01 % | 0/120 |

Every variant, `Float16_b→Float16_b`:

| Op | approx | dest_acc | baseline | branch | Δ cycles | Δ % |
|---|---|---|---|---|---|---|
| **ReciprocalCompat** | No | No | 1215.02 | 1343.02 | +128.00 | **+10.53 %** |
| **ReciprocalCompat** | No | Yes | 1183.02 | 1311.02 | +128.00 | **+10.82 %** |
| **ReciprocalCompat** | Yes | No | 991.02 | 1119.02 | +128.00 | **+12.92 %** |
| **ReciprocalCompat** | Yes | Yes | 991.02 | 1119.02 | +128.00 | **+12.92 %** |
| **RsqrtCompat** | No | No | 2175.08 | 2303.08 | +128.00 | **+5.88 %** |
| **RsqrtCompat** | No | Yes | 2143.11 | 2271.11 | +128.00 | **+5.97 %** |
| **RsqrtCompat** | Yes | No | 1151.11 | 1279.08 | +127.97 | **+11.12 %** |
| **RsqrtCompat** | Yes | Yes | 1151.08 | 1279.11 | +128.03 | **+11.12 %** |

**Blackhole pays +128.00 cycles per tile, the same absolute cost as Wormhole**, and the
percentages land in the same bands (WH: +10.53 … +13.15 % and +5.68 … +11.12 %). The two arches
are directly comparable on this kernel because their baselines nearly coincide — 991.02 against
WH's 990.87 on the approximate `recip`, 1215.02 against 1214.92 on the exact one. The guard costs
the same instructions at the same rate on both.

The 180 control variants are flat: the largest absolute movement anywhere among them is **0.0078
cycles**, which is inside the run-to-run spread. `legacy_compat = false` does not reach the patched
primitive, and silicon confirms it on BH as well as WH.

### Where the cycles go

A tile is 32 SFPU vector iterations, so +128.00 cycles per tile is **4 cycles per iteration**, on
both arches. Added by the guard, per unrolled copy: `SFPSETSGN`, `SFPLOADI` ×2 (the compare
constant and the infinity), `SFPSETCC`, `SFPENCC`.

Blackhole makes one thing visible that Wormhole's numbers could not. The *static* cost of the
guard varies by a factor of eight across variants — math kernel text grows by only 32 bytes on
approximate `ReciprocalCompat` but by 256 bytes on exact `RsqrtCompat`:

| Op | approx | dest_acc | `TEXT_SIZE(MATH_ISOLATE)` before → after | Δ bytes | Δ cycles/tile |
|---|---|---|---|---|---|
| ReciprocalCompat | No | No | 2461 → 2605 | +144 | +128.00 |
| ReciprocalCompat | No | Yes | 2473 → 2561 | +88 | +128.00 |
| ReciprocalCompat | Yes | No / Yes | 2429 → 2461 / 2449 → 2481 | +32 | +128.00 |
| RsqrtCompat | No | No / Yes | 3725 → 3981 / 3681 → 3937 | +256 | +128.00 |
| RsqrtCompat | Yes | No / Yes | 2437 → 2637 / 2457 → 2657 | +200 | +127.97 / +128.03 |

**The measured cost is +128.00 in every row regardless.** So the guard's cycle cost is not
proportional to its instruction footprint: where the compiler emits more instructions, the extra
ones issue in slots the surrounding code was already stalling in. That is the same conclusion the
Wormhole record reached from the +64-static-instructions-against-4-cycles mismatch, stated here
with a cleaner demonstration — an 8× spread in added bytes against an identical cycle delta. It is
recorded as measured rather than reasoned, and it does not close as an
instructions-times-iterations identity the way the `pow` and `sqrt_custom` guards did.

**The cost is absolute, not proportional.** The same +128 cycles reads as +5.88 % on the
2175-cycle exact `RsqrtCompat` and +12.92 % on the 991-cycle approximate `recip`, for identical
work. A larger percentage on Blackhole than Wormhole for a given op would mean only a cheaper
baseline, never a more expensive guard.

### 5.3 What the second pole costs — the bare-compare variant, priced

The cheaper guard the fix did not take, measured on Blackhole against the same baseline, three runs,
the 8 legacy-compat variants:

| Guard form | Δ cycles/tile | per iteration | `RsqrtCompat` | `ReciprocalCompat` |
|---|---|---|---|---|
| bare `in == 0.0F` | **+96.00** | 3 cycles | +4.41 … +8.34 % | +7.90 … +9.69 % |
| shipped `setsgn(in, 0)` | **+128.00** | 4 cycles | +5.88 … +11.12 % | +10.53 … +12.92 % |

Per variant, so the difference is visible as a constant rather than a range:

| Op | approx | dest_acc | baseline | bare | shipped |
|---|---|---|---|---|---|
| ReciprocalCompat | No | No | 1215.02 | 1311.02 | 1343.02 |
| ReciprocalCompat | No | Yes | 1183.02 | 1279.02 | 1311.02 |
| ReciprocalCompat | Yes | No / Yes | 991.02 | 1087.02 | 1119.02 |
| RsqrtCompat | No | No | 2175.08 | 2271.08 | 2303.08 |
| RsqrtCompat | No | Yes | 2143.11 | 2239.11 | 2271.11 |
| RsqrtCompat | Yes | No / Yes | 1151.1 | 1247.1 | 1279.1 |

**The `setsgn` costs exactly +32.00 cycles per tile — one instruction, one cycle per iteration,
a single `SFPSETSGN`.** Unlike the guard as a whole, this one *does* close as an exact
instructions-times-iterations identity, on all eight variants, with a run-to-run spread of at most
0.03 cycles.

So the price of the second pole is one instruction. 32 cycles per tile buys `1/-0 = -inf` on the
pipelines that deliver a real `-0.0`; without it that pole stays at 1.7e38. The Wormhole record
could only put "roughly 2 percentage points" on this; Blackhole prices it to the instruction. The
percentage difference between the two forms is 1.5–3.2 points depending on the denominator, which
is what "roughly 2" was approximating.

## 6. Verification

### 6.1 Wormhole n300

| Sweep | before | after |
|---|---|---|
| `test_sfpu_unary.py -k "Reciprocal or Rsqrt"` | 497 passed, 15 xfailed | **505 passed, 7 xfailed** |
| `test_sfpu_unary.py -k edges` | 491 passed, 23 xfailed | **499 passed, 15 xfailed** |
| `test_sfpu_sdpa.py` | 65 passed | **65 passed** |

0 xpassed and 0 failed throughout. Both unary deltas are the same 8: `RsqrtCompat`'s
per-combination xfails for the saturating pole, which the guard retires.

### 6.2 Blackhole p100a

Run on p100a silicon. "Before" is the same tree with only the guard reverted, so the two columns
differ by the `v_if` and nothing else.

| Sweep | before | after |
|---|---|---|
| `test_sfpu_unary.py -k "Reciprocal or Rsqrt"` | **10 failed**, 435 passed, 78 skipped, 5 xfailed | **445 passed**, 78 skipped, 5 xfailed, **0 failed** |
| `test_sfpu_unary.py -k edges` | **10 failed**, 308 passed, 429 skipped, 13 xfailed | **318 passed**, 429 skipped, 13 xfailed, **0 failed** |
| `test_sfpu_sdpa.py` | 65 passed | 65 passed |
| `test_sfpu_sampling.py` | 51 passed, 93 skipped | 51 passed, 93 skipped |
| `test_sfpu_unary.py` (whole file) | — | **5042 passed**, 1607 skipped, 13 xfailed, **0 failed, 0 xpassed** |

0 xpassed throughout, and no xfail count moves in either selection — so nothing was absorbed by a
marker and no stale xfail entry is left behind. Both unary deltas are exactly the same 10 cells
(435 → 445, 308 → 318): 5 combinations × `{RsqrtCompat, ReciprocalCompat}`, which is every
combination Blackhole reaches for the two ops the guard touches.

The 5 remaining xfails in the first row are pre-existing and untouched by this branch: 3 ×
`Reciprocal`'s `1/NaN` and 2 × `Rsqrt`'s `rsqrt(-0)`.

**The regression test is non-vacuous on Blackhole, and that was checked rather than assumed.**
Against the unguarded kernel the edge variants fail **10 of 10** — golden `inf`, kernel
`1.7014e38`, with the golden row reading `[-4194304, inf, 4194304, inf, …]` against a saturating
result. Against the guarded kernel all 10 pass. Wormhole established this for the 8 combinations
it reaches; Blackhole now establishes it independently for the 5 it reaches, so reverting the guard
breaks CI on both arches rather than only on WH.

**Blackhole sampling is measured now, not argued.** `test_sfpu_sampling.py` is the Blackhole-only
op, and `sampling_recip_value<true>` is the one consumer carrying a written "must stay
bit-identical for blaze" contract. The previous record could only compile-verify it and reason
that the guard is unreachable inside its documented `in > 0` domain. It now runs on silicon and is
unchanged before and after — 51 passed / 93 skipped both ways, including every `legacy_compat=True`
variant across both `vector_mode` settings. The contract holds by measurement.

`test_sfpu_sdpa.py` is 65/65 on Blackhole, the same as Wormhole, against per-path tolerances that
were written for the legacy kernel and left unchanged.

## 7. The coverage gap this found, and closed

The largest user-visible change is the public `recip()` default, and the tt-llk suite did not
exercise it at all: `sfpu_operations.h` called `calculate_reciprocal` and `recip_init` without a
`legacy_compat` argument, so both defaulted to `false` and the suite only ever built the kernel
the public API does *not* dispatch.

That is now fixed permanently. A `MathOperation.ReciprocalCompat` op covers the default path,
wired exactly the way `RsqrtCompat` covers legacy rsqrt: a `SfpuType::reciprocal_compat` enum
value on both arches, an init and call branch in `sfpu_operations.h` passing
`legacy_compat = true`, and the op registered in `llk_params.py`, `sfpu_domains.py` (same domain
and pole as `Reciprocal`) and `golden_generators.py` (same `_reciprocal` golden). It brings 8
sweep variants and 8 edge variants.

It is a real regression test rather than a passing formality, and that was checked rather than
assumed: built against the **unguarded** kernel its edge variants fail 8 of 8 on Wormhole —
golden `inf`, kernel `1.7014e38`. Against the guarded kernel all 16 pass. Anyone reverting the
guard now breaks CI.

Before this op existed, the Wormhole figures for that path had to be taken with both trees patched
identically to pass `true` — a temporary instrument, reverted before the correctness sweeps and
never committed. The Blackhole figures in §4.2 and §5.2 needed no such instrument, because the op
is in the tree.

## 8. User-visible behaviour change

| Path | Before | After |
|---|---|---|
| `recip_tile()` — the **public default**, `recip.h` declares `legacy_compat = true` | `recip(0)` = `1.7014e38` | `recip(0)` = `+inf`, `recip(-0)` = `-inf` |
| `rsqrt_tile<true>` / `RsqrtCompat` | `rsqrt(0)` = `1.7014e38` | `rsqrt(0)` = `+inf` |
| SDPA `calculate_recip_first_column<true>` | same pole | same fix |
| Blackhole `sampling_recip_value<true>` | same pole | same fix; guard is unreachable inside its documented `in > 0` domain, so blaze's bit pattern is unchanged |
| Everything on `legacy_compat = false` | — | untouched |

**Open question — `recip(0)` now propagates infinity.** This is the intended fix, but it is worth
confirming downstream. `recip(0)` was the max finite and is now `+inf`, so a multiply that used to
give `0` now gives `NaN`. The concrete case raised in review: a fully-masked softmax row masks
with `-inf` (`softmax.cpp:275`), reciprocates the zero sum unguarded (`:341-354`) and multiplies
it back (`:369`) — previously a row of zeros, now a row of NaN. `moreh_common.hpp`, moe, sampling
and the deepseek gates ride the same default. If a guarded-zero contract is wanted at the op
level, that belongs in the op, not back in the kernel — but it should be a deliberate decision.

## 9. Scope and follow-ups

Worth filing separately:

* **Consolidating on one reciprocal** (Option A, §3) — gated on migrating and revalidating the
  eight `rsqrt_tile<true>` callers.
* **`rsqrt_compat` is 54 % wrong at the fp32 extremes** (`1.17549e-38` and `3.40282e38`), before
  and after this fix identically. That is `_sqrt_compat_`'s fast-inverse-sqrt seed running out of
  range, not the reciprocal — the same family of problem as `sqrt_custom(+inf)`.
* **The SDPA legacy reciprocal also returns `1/|x|`** — the sign is genuinely dropped, because
  `_reciprocal_compat_` returns a magnitude and that one caller never re-signs it. Out of scope
  here; fixed on `ldjurovic/sfpu_sdpa_recip_legacy_sign`.
* **`RsqrtCompat` is not in `SPECIALS_READY_OPS`**, so `rsqrt_compat(±inf)` and `(NaN)` are still
  never driven by the shipped sweep. The probe shows `rsqrt_compat(+inf) = 0x00000000` on every
  combination, which is correct — and now correct by construction rather than by accident. Worth
  pinning.

## 10. What was not measured

An earlier revision of this record listed five gaps. Blackhole silicon closed the first two; the
rest stand.

| # | Item | Status |
|---|---|---|
| 1 | Blackhole cycles | **Closed.** §5.2 — three runs per tree on p100a, +128.00 cycles/tile, 180 controls flat. |
| 2 | Blackhole sampling numerics | **Closed.** §6.2 — `test_sfpu_sampling.py` runs on p100a and is unchanged before and after, so blaze's bit-identity contract holds by measurement rather than by construction. |
| 3 | SDPA reciprocal cycles | **Still open.** `perf_sfpu_reduce_sdpa.py` instruments only `ReduceColumn`, so `calculate_recip_first_column` has no perf vehicle on either arch. Its body is the same `_reciprocal_compat_`, so the +128 cycles/tile applies and its loop is 4 SFPU slots at stride 2 rather than 32 at stride 1 — but this record will not put a measured number on it. |
| 4 | Quasar | Out of scope; untouched, and it has no equivalent compat reciprocal. |
| 5 | End-to-end model impact | These are LLK-level cycle counts. +128 cycles on a normalization kernel's rsqrt is not +128 cycles on a model, and the eight `rsqrt_tile<true>` callers in §3 are where that would be felt. Not measured here. |

Two further notes on scope, so the Blackhole coverage is not read as wider than it is.

**Blackhole reaches 5 of the 8 (format pair, `dest_acc`) combinations**, not 8: `dest_acc=No`
supports only `Float32→Float32` there. Every BH figure above is over those 5, and the Wormhole
figures over 8. Where the two records give different counts — 20 differing probe rows against 16,
10 fixed edge cells against 8 — that is the combination count, not a behavioural difference.

**The Blackhole perf sweep gives the legacy `recip()` default 4 variants, not Wormhole's 60.**
`ReciprocalCompat` is not in `perf_eltwise_unary_sfpu.py`'s `_FULL_FORMAT_MATRIX_OPS`, so it is
swept on the representative `Float16_b→Float16_b` pair only, while the Wormhole figures came from
instrumenting `Reciprocal`, which does carry the full 16-pair matrix. The guard's cost is a
constant that does not depend on the format pair — +128.00 on every variant of both ops, and the
`Reciprocal` control is flat across all 60 of its format pairs — so this narrows the evidence
rather than weakening the claim. Widening it would mean adding the op to that frozenset, which is a
CI-cost decision and not this PR's.
