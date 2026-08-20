# Result — `RsqrtCompat(0)`: both fix options built and measured

> **Correction, 2026-08-20 — this document's conclusion was reversed on review.**
> It concludes that Option A wins and is in the tree. **The branch ships Option B.** The
> performance and accuracy measurements below are sound and are kept as the A-vs-B bake-off
> record; what was wrong is §5, the answer to "who depends on the legacy bit pattern". That
> search looked for the phrase *bit-identical* and found only the Blackhole sampling file. It
> missed eight production normalization kernels that hard-code `rsqrt_tile<true>` and say so as
> *matches baseline* — including `dit_layernorm_fused_compute.cpp:267`, which states outright
> that "the non-legacy default diverges on low-variance rows". Option A would have changed those
> baselines silently. See `MEASUREMENTS_52930_reciprocal_compat_pole_vs_main.md` §8 for the full
> list and the reasoning, and for the shipped Option B figures.
>
> Read every "Option A is in the tree" statement below as "Option A was measured and withdrawn".

**Plan:** [FIX_PLAN_52930_reciprocal_compat_pole.md](FIX_PLAN_52930_reciprocal_compat_pole.md) (finding 4 of
[ISSUE_52930_INVESTIGATION.md](ISSUE_52930_INVESTIGATION.md)).
**Question asked:** build both options, check each works functionally, and keep the one with the lower
performance penalty, measured as `MATH_ISOLATE` cycles in `perf_eltwise_unary_sfpu.py`.

**Answer in one line, as originally written:** both fix the pole; **Option A is not a penalty at all — it is
26–39 % faster on `RsqrtCompat` and 43–53 % faster on the public `recip()` default**, where Option B costs
4.6–8.3 %. Accuracy improves on every path that changed and regresses on no value anywhere — the legacy
reciprocal was 610× less accurate than its replacement on the fp32 pipeline (§4).

**Answer as it stands after review:** the performance and accuracy comparison holds, and **Option B ships
anyway**, because the plan's blocking question was answered wrongly here. §5 named one dependent and concluded
Option A touched none; there are eight more, and Option A touches all of them. Option B's measured cost on the
shipped branch is **+5.7 … +11.1 % on `RsqrtCompat` and +10.5 … +13.2 % on the public `recip()` default** — a
flat +128 cycles per tile — which is higher than the 4.6–8.3 % quoted below because the shipped guard also
handles `-0.0`. Bit-exactness is preserved everywhere except the two poles.

| | Wormhole n300, silicon | |
|---|---|---|
| Tree | `tt-metal` @ `ldjurovic/wrong_sfpu_edge_cases`, `26c61ff80e9` | |
| Runner | `tt_metal/tt-llk/.claude/scripts/run_test.sh` throughout | |
| Date | 2026-08-17 | |

---

## 1. Baseline, re-established first

`test_sfpu_wh_issue52930_probe.py -k "unary_edge_values and RsqrtCompat"` — 8/8 diverge, values exactly as
the investigation recorded: `0x7EFFFD9E` on the fp32 pipeline, `0x7F000000` (its bf16 rounding) elsewhere.

Because "every other value bit-identical" is an acceptance criterion the shipped sweep cannot check (it
compares against golden with tolerances, not against the previous build),
`tests/python_tests/test_sfpu_wh_recipcompat_numerics.py` was written for this run: a fixed 21-value stimulus
list dumped as raw hardware bit patterns on all 8 (format pair, `dest_acc`) combinations, so before/after
diffs byte for byte. **Records rather than asserts; not for merge**, same status as the #52930 probe.

## 2. Both options, as built

**Option B — pole guard in place** (`tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_rsqrt_compat.h`,
identical files, one patch):

```c
    // Set newly denormalized exponent to result exponent field
    sfpi::vFloat out = setexp(result, new_exp);

    v_if (in == 0.0F)
    {
        out = std::numeric_limits<float>::infinity();
    }
    v_endif;

    return out;
```

Applied *after* the `setexp`, as §5 of the plan recommends — writing `result = inf` before it would have the
exponent field overwritten.

**Option A — redirect the compat consumers to `sfpu_reciprocal_iter`.** `_reciprocal_compat_` lives in the
tt-llk lib and `sfpu_reciprocal_iter` lives one layer up in metal, so the redirect has to happen at the metal
layer, not inside the lib:

| File (× both arches) | Change |
|---|---|
| `llk_sfpu/ckernel_sfpu_rsqrt.h` | new `_calculate_rsqrt_compat_iter_`: keeps `_sqrt_compat_` (the actual point of "compat") and pairs it with `sfpu_reciprocal_iter`. `rsqrt_init<…, true>` now programs the polynomial constants |
| `llk_sfpu/ckernel_sfpu_recip.h` | `calculate_reciprocal<…, legacy_compat>` no longer branches; `recip_init` always calls `sfpu_reciprocal_init` |
| `experimental/llk_sfpu/ckernel_sfpu_sdpa.h` | `calculate_recip_first_column<legacy_compat>` collapsed onto the iter branch |
| `tests/helpers/include/sfpu_operations.h` | `rsqrt_compat` no longer routes through the bare `unused` init — the compat path is no longer self-contained |

Two things the plan did not anticipate, both found by building it:

* **The compat path stopped being init-free.** `sfpu_reciprocal_iter` reads its polynomial from
  `vConstFloatPrgm0..2`, and `rsqrt_init<…, legacy_compat = true>` deliberately programmed nothing. The
  init contract has to change with the kernel, in the harness as well as in production.
* **The legacy sign flip had to go.** `_reciprocal_compat_` returns the *magnitude* `|1/x|` and each caller
  restores the sign afterwards; `sfpu_reciprocal_iter` already ends in `copysgn(y, in)`. Keeping the caller's
  `v_if (in < 0.0) { out = -out; }` on top of that would invert every negative result.

## 3. Functional verification

**Option B — bit-exact, as promised.** Diffing the 168-row numerics record against baseline: exactly 8 lines
changed, all of them the `x = 0.0` row, all `0x7F000000`/`0x7EFFFD9E` → `0x7F800000`. Every other value
identical, including the ~0.15 % `rsqrt_compat(2.384e-07) = 2044.9895` the plan puts out of scope.

**Option A — pole fixed, and the accuracy figure fixed with it.** Numerics change everywhere, and they move
toward the golden:

| `x` | golden | baseline | Option A |
|---|---|---|---|
| `0.0` | `0x7F800000` | `0x7EFFFD9E` | `0x7F800000` ✅ |
| `1.0` | `0x3F800000` | `0x3F7F9FAA` (0.99902) | `0x3F80002A` (1.0000005) |
| `4.0` | `0x3F000000` | `0x3EFF9FAA` (0.49951) | `0x3F00002A` |
| `2.384185791015625e-07` | `0x45000000` (2048.0) | `0x44FF9FAA` (2044.9895) | `0x4500002A` (2048.0005) |

The residual `…002A` is `_sqrt_compat_`'s own error, which Option A deliberately keeps.

Shipped suites, Wormhole silicon, Option A in the tree:

```
test_sfpu_unary.py  -k "RsqrtCompat or Reciprocal"       170 passed, 6 xfailed, 8 XPASS
test_sfpu_unary.py  -k "RsqrtCompat or Reciprocal or Rsqrt or Sqrt"
                       (after removing the xfail tables) 847 passed, 9 xfailed, 0 xpassed, 0 failed
test_sfpu_sdpa.py                                        65 passed
test_sfpu_sdpa_fw.py                                     21 passed
test_sfpu_reduce_sdpa.py / test_sdpa_reinits.py /
test_sfpu_sampling.py                                    pass
```

The 8 XPASS were exactly the 8 `RsqrtCompat` edge combinations — the fix — and the 6 remaining XFAIL are
`Reciprocal`'s pre-existing `1/NaN` entry, untouched. Blackhole compiles clean for
`test_sfpu_unary.py`, `test_sfpu_sdpa.py` and `test_sfpu_sampling.py`; no Blackhole silicon on this host, so
those are compile-only.

**One genuine test change Option A forces.** `test_sfpu_sdpa.py`'s `RecipLegacy` variants failed at first —
9 of 65 — on *sign*, not magnitude. The golden generator encoded `RecipLegacy = 1/|x|`, because
`calculate_recip_first_column`'s legacy branch called `_reciprocal_compat_` and, unlike the other consumers,
never restored the sign. So **the legacy SDPA reciprocal silently dropped the sign of negative inputs** — a
second latent defect in the same kernel, invisible in production only because a softmax denominator is
positive. Option A fixes it; the golden and its comment were updated to `1/x` to match.

## 4. Performance and accuracy, path by path

Four call sites reach the legacy reciprocal. Two were measured when the fix was written; the other two are
measured below. **Nothing regressed on any of them** — accuracy improves or is unchanged everywhere, and
every path that changed got faster.

| Path | Accuracy | Perf (`MATH_ISOLATE`) |
|---|---|---|
| `rsqrt_compat` / `RsqrtCompat` | max rel err **`1.47e-3` → `5.01e-6`**, 293× better at the fp32 end (§4.1) | **−25.6 … −38.9 %** (§4.1) |
| `recip()` with `legacy_compat = true` (the public default) | max rel err **`3.64e-5` → `5.96e-8`**, 610× better (§4.2) | **−43.2 … −52.6 %**, all 64 variants (§4.2) |
| SDPA `calculate_recip_first_column<true>` | same kernel instantiation as the row above — same numbers, plus the sign fix (§4.3) | **−16 cycles per SFPU slot**, derived (§4.3) |
| Blackhole `sampling_recip_value<true>` | **bit-identical — not touched** (§4.4) | unchanged (§4.4) |

### 4.1 `RsqrtCompat` — `perf_eltwise_unary_sfpu.py -k RsqrtCompat`

Per-tile `TILE_LOOP` cycles, `Float16_b → Float16_b`, `iterations = 32`, `loop_factor = 16`. Three runs each;
run-to-run spread was under 0.1 cycles out of ~2247, so every delta below is far outside the noise.

| `dest_acc` / `approx` | baseline | Option B | Option A |
|---|---|---|---|
| No / No | 2246.97 | 2356.04 (**+4.85 %**) | 1567.97 (**−30.22 %**) |
| No / Yes | 1152.12 | 1248.16 (**+8.34 %**) | 704.07 (**−38.89 %**) |
| Yes / No | 2237.92 | 2340.58 (**+4.59 %**) | 1665.27 (**−25.59 %**) |
| Yes / Yes | 1152.13 | 1248.05 (**+8.33 %**) | 703.94 (**−38.90 %**) |

Secondary figures:

| | baseline | Option B | Option A |
|---|---|---|---|
| `INIT` (once per kernel) | 223–224 | 223–224 | 236–240 (+13–16) |
| `TEXT_SIZE(MATH_ISOLATE)` | 3105–5225 | 3297–5417 (+192) | 2869–4173 (−236 … −1236) |

Option B's cost is exactly what the guard is: **+96 cycles per tile / 32 SFPU iterations = 3 cycles per
iteration**, an `SFPSETCC` + constant load + predicated move. The plan's risk table guessed this "should be
noise on a kernel already doing 3 Newton iterations"; measured, it is 4.6–8.3 %, which is not noise.

Option A's win is structural: it drops a whole Newton iteration (3 → 2) *and* the `exexp`/`setexp` exponent
fix-up, which is what the pole guard was patching in the first place. Option A's only cost is +13–16 cycles
of one-time `INIT` to program three constants, amortised over the first tile.

**So the perf question the plan framed as "which fix is cheaper" has a stronger answer than expected:
Option A is not a fix with a smaller penalty, it is a fix that pays for itself several times over.**

Accuracy on this path, computed from the bit-exact record in §3 against an exact fp64 `1/sqrt(x)`, on
`Float32→Float32 / dest_acc=Yes`:

| `x` | rel err before | rel err after |
|---|---|---|
| `1.0`, `4.0`, `0.25`, `0.015625`, `2.384e-07` | `1.470e-03` | `5.007e-06` |
| `1000` | `1.073e-03` | `5.012e-06` |
| `1e+30` | `3.916e-05` | `1.170e-07` |
| `3.0` | `1.416e-05` | `7.047e-07` |
| `2.0`, `0.5`, `7.0`, `1.5`, `100`, `1e-10` | `1.7e-08` … `2.8e-06` | **unchanged** |

**Max over the normal range: `1.470e-03` → `5.012e-06`, 293× better, and not one value got worse.** On the
bf16-packed combinations both kernels round to the same bf16, so the improvement is invisible there.

One caveat this measurement exposes, unrelated to the fix and **unchanged by it**: at the fp32 extremes
(`1.17549e-38`, the smallest normal, and `3.40282e+38`) `rsqrt_compat` is **54 % wrong**, before and after
identically. That is `_sqrt_compat_`'s fast-inverse-sqrt seed running out of range, not the reciprocal — the
same family of problem as `sqrt_custom(+inf)` in
[FIX_PLAN_52930_sqrt_custom_infinity.md](FIX_PLAN_52930_sqrt_custom_infinity.md), and worth filing separately.

### 4.2 `recip()` with `legacy_compat = true` — the public compute-API default

This is the path §7 of the plan flagged as the largest blast radius: `recip.h:17,37` defaults
`legacy_compat = true`, and the tt-llk suite drives `MathOperation.Reciprocal` with `false`, so the shipped
default was never measured on either axis.

Measuring it needed a vehicle, because the harness only reaches the non-legacy path. `_calculate_reciprocal_compat_`
still exists in `ckernel_sfpu_rsqrt_compat.h` (untouched by the fix), so **both kernels are callable from the
committed tree** — no baseline checkout needed. A temporary adapter in `sfpu_operations.h` pointed
`SfpuType::reciprocal` at the legacy kernel for the BEFORE runs and was reverted afterwards; the AFTER runs
use the shipped dispatch unmodified. Identical stimulus, identical harness, same silicon session.

**Accuracy** — `test_sfpu_wh_recip_accuracy.py`, 126 values spanning 2^-30 … 2^30 with both signs and
non-power-of-two mantissas, relative to an exact fp64 `1/x`:

| variant | before, max | after, max | before, mean | after, mean |
|---|---|---|---|---|
| `Float32→Float32` `dest_acc=Yes` | **`3.636e-05`** | **`5.960e-08`** | `1.444e-05` | `2.583e-08` |
| `Float16_b→Float32` `dest_acc=Yes` | `4.595e-03` | `4.587e-03` | `1.544e-03` | `1.529e-03` |
| the other six combinations | `2.734e-03` / `6.653e-03` | identical | identical | identical |

The fp32 end-to-end row is the only one that measures the kernel; everywhere else a bf16 pack quantises to
~2^-8 and both kernels land on the same bf16. There, the legacy kernel is **610× less accurate**, and it is
wrong even where the answer is exactly representable:

```
x = 9.3132e-10 (2^-30)   exact = 1.0737418e+09 = 0x4E800000
                before  = 1.0737028e+09 = 0x4E7FFD9E   rel = 3.64e-05
                after   = 1.0737418e+09 = 0x4E800000   rel = 0
```

That `0x…7FFD9E` mantissa is the same signature as the `0x7EFFFD9E` the pole produced — the legacy kernel's
Newton iteration converging just short of 2.0 is what drives both the pole value and this error floor. It
is a ~16-bit-accurate reciprocal being used on an fp32 path.

**Perf** — `perf_eltwise_unary_sfpu.py -k Reciprocal`, all 64 variants (16 format pairs × approx × dest_acc),
per-tile `TILE_LOOP` cycles:

| variant class | before | after | delta |
|---|---|---|---|
| `dest_acc=No`, `approx=No` | 1215.9–1216.2 | 576.0–576.3 | **−52.6 %** |
| `dest_acc=No`, `approx=Yes` | 992.0–992.2 | 544.0–544.3 | **−45.2 %** |
| `dest_acc=Yes`, `approx=No` | 1184.0–1184.1 | 672.0–672.1 | **−43.2 %** |
| `dest_acc=Yes`, `approx=Yes` | 992.0–992.0 | 544.0–544.1 | **−45.2 %** |
| `Float32` in, `dest_acc=Yes` | 1165.7 / 973.7 | 653.7 / 525.6 | **−43.9 % / −46.0 %** |

All 64 variants improved; the spread within each class is under 0.4 cycles. Math kernel text shrinks from
2945–3301 to 2833–2877 bytes.

### 4.3 SDPA `calculate_recip_first_column<true>`

`sfpu_sdpa_test.cpp` has no `MEASURE_PERF_COUNTERS` instrumentation and there is no perf test for it, so this
path has no direct `MATH_ISOLATE` vehicle. It does not need one for accuracy: its legacy body was
`_reciprocal_compat_<APPROX ? 2 : 3>` plus the optional bf16 narrowing — **the same instantiation, with the
same template argument, as `_calculate_reciprocal_compat_`** measured in §4.2 — and its new body is the same
`sfpu_reciprocal_iter<0/1/2>` selection as `_calculate_reciprocal_internal_`. The §4.2 accuracy table is
therefore this path's accuracy table, exactly.

Two differences from §4.2, both in this path's favour:

* It never had the caller-side sign restoration, so on top of the accuracy gain it stops returning `1/|x|`
  for negative inputs (§3).
* Its loop is 4 SFPU slots at stride 2, not 32 at stride 1.

Per-slot cost, divided out of the §4.2 measurement (**derived, not directly measured**):

| SDPA precision | before | after | per `calculate_recip_first_column` call (4 slots) |
|---|---|---|---|
| `Bf16Dest` (`DST_ACCUM_MODE=No`) | 38.0 cyc/slot | 18.0 cyc/slot | 152 → 72 |
| `Fp32Dest` / `Fp32E2E` (`DST_ACCUM_MODE=Yes`) | 37.0 cyc/slot | 21.0 cyc/slot | 148 → 84 |

Corroboration that nothing regressed in situ: `test_sfpu_sdpa.py` is 65/65 green against per-path tolerances
that were written for the legacy kernel and left unchanged, and `test_sfpu_sdpa_fw.py`,
`test_sfpu_reduce_sdpa.py` and `test_sdpa_reinits.py` all pass.

### 4.4 Blackhole `sampling_recip_value<true>` — untouched, and verified so

Commit `21557a33763` contains no `ckernel_sfpu_sampling.h`, and the file still reads
`return ckernel::sfpu::_reciprocal_compat_<APPROX ? 2 : 3>(in);` at line 52. So this path is bit-identical by
construction — the same instruction sequence, the same values, the same cycles, and the same unguarded pole.
It is the one consumer keeping `_reciprocal_compat_` alive, for the reason in §5. No Blackhole silicon on
this host, so this is source-level verification plus a clean compile, not a hardware run.

## 5. The blocking question the plan raised, answered — INCORRECTLY

> **This section is the error.** Its conclusion — that no path Option A touches has a legacy
> dependency — is false. It searched for the phrase *bit-identical*, which appears in exactly one
> file, and concluded from a one-file hit that the contract was unclaimed elsewhere. Eight
> production normalization kernels hard-code `rsqrt_tile<true>` and record the dependency in
> different words ("matches baseline", "the non-legacy default diverges on low-variance rows").
> The section is kept verbatim below because the Blackhole sampling analysis in it is correct and
> still applies; only its scope conclusion is wrong. The full caller list is in
> `MEASUREMENTS_52930_reciprocal_compat_pole_vs_main.md` §8.

The plan says Option A is "only viable once someone can say who depends on the legacy bit pattern", and
that it is blocked on that. Searching the tree, exactly one place states the dependency, and it is explicit:

> ```
>  * SFPU sequence on the legacy path -- which must stay bit-identical for blaze.
>  * @tparam legacy_compat: Use blaze's bit-identical reciprocal, values = <true/false>
> ```
> — `tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sampling.h:37,39`

That file also documents the sign divergence as deliberate ("the legacy path must stay bit-identical for
blaze, so the divergence is documented rather than fixed") and constrains its callers to `in > 0`.

**Option A would not have touched it** (and Option B, as shipped, guards it — the guard lives inside `_reciprocal_compat_` itself, and is unreachable within sampling's documented `in > 0` domain, so blaze's bit pattern is unchanged on every input it may legally pass).

The original text follows.

**Option A as shipped does not touch it.** `sampling_recip_value<true>` still calls `_reciprocal_compat_`,
byte for byte, so blaze's contract holds. The consequence to be explicit about: `_reciprocal_compat_`
survives with the unguarded pole, for that one Blackhole sampling consumer. Two ways to close that, both
cheap, and the choice is the kernel owners':

1. **Leave it.** Sampling's own contract already excludes the pole (`Callers must pass in > 0`), so the
   defect is unreachable there.
2. **Apply Option B's guard on top of Option A**, for that consumer alone. It changes one input — `0` — which
   sampling's contract already forbids, so it is bit-identical across blaze's entire documented input
   domain, at 3 cycles per element on that path.

A vendored snapshot at
`models/demos/deepseek_v3_b1/kernel_includes/…/ckernel_sfpu_sampling.h` carries the same legacy call and was
deliberately left alone.

## 6. What is in the tree

Option A, plus the test-table removals the plan's §7 asks for.

| File | Change |
|---|---|
| `hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_rsqrt.h` | `_calculate_rsqrt_compat_iter_`; `rsqrt_init<…, true>` programs the constants |
| `hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_recip.h` | `calculate_reciprocal` no longer branches on `legacy_compat`; `recip_init` always inits |
| `hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sdpa.h` | legacy branch collapsed onto `sfpu_reciprocal_iter` |
| `tt-llk/tests/helpers/include/sfpu_operations.h` | `rsqrt_compat` gets a real init |
| `tt-llk/tests/python_tests/test_sfpu_unary.py` | `RsqrtCompat` removed from `_EDGE_KNOWN_DIVERGENCES` and `_EDGE_DIVERGENCE_REASON`; the "STILL OPEN" block loses its `rsqrt at 0` paragraph and gains a CLOSED note |
| `tt-llk/tests/python_tests/helpers/golden_generators.py`, `test_sfpu_sdpa.py` | `RecipLegacy` golden is `1/x`, not `1/\|x\|` |

`_reciprocal_compat_` and `_calculate_reciprocal_compat_` are now dead except for the Blackhole sampling
path; `_sqrt_compat_` and `_calculate_sqrt_compat_` are untouched and still live.

Not for merge, kept as evidence: `tests/python_tests/test_sfpu_wh_recipcompat_numerics.py`.

## 7. Still open, deliberately

* **The plan's §4 step 0 is now answered with measurements, not derivation** (see §4.2): the public `recip()`
  defaults to `legacy_compat = true` (`recip.h:17,37`), and that path was both broken at the pole and 610×
  less accurate and ~2× slower than the non-default one it sat next to. Option A fixes all three. **The
  coverage gap that let it hide is now CLOSED:** the tt-llk suite used to drive
  `MathOperation.Reciprocal` with `legacy_compat = false` only, so the shipped default had no edge test and
  no perf variant, and the measurements in this document needed a temporary harness adapter that was
  reverted. A permanent `MathOperation.ReciprocalCompat` op now covers the default path, wired the same way
  `RsqrtCompat` covers legacy rsqrt. Verified to be a real regression test rather than a passing formality:
  run against `main`'s unguarded kernel its edge variants fail 8 of 8, and against the guarded kernel all 16
  pass.
* **`_reciprocal_compat_` is now dead code except for one Blackhole caller** (§4.4, §5). If the sampling
  owners confirm the pole is unreachable under their `in > 0` contract, the function and
  `_calculate_reciprocal_compat_` can be deleted outright; that is the end state Option A was aiming at.
* **`rsqrt_compat` is 54 % wrong at the fp32 extremes**, before and after this fix identically — see §4.1.
  `_sqrt_compat_`'s seed, not the reciprocal. Worth its own issue.
* **`RsqrtCompat` is not in `SPECIALS_READY_OPS`**, so `rsqrt_compat(±inf)` and `(NaN)` are still never
  driven by the shipped sweep. The numerics probe shows `rsqrt_compat(+inf) = 0x00000000` on all 8, which is
  correct and now correct *by construction* rather than by accident — worth pinning.
* **Blackhole is compile-verified only.** No Blackhole silicon on this host; the plan's step 3 wants both
  arches run.
