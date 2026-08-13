# PR #52938 description — paste into the PR body

> Copilot's sixth comment: *"The PR body only points to issue #49739 and PR #52416, so it does not
> explain this PR's own changes."* This is a replacement body. Everything below is measured on a
> Blackhole p300a unless stated otherwise; the Wormhole figures are a Wormhole n300.
>
> **Read the blocker before reviewing.** The suites have since run on Wormhole for the first time and this
> PR's enrolment turns that arch red — 50 variants, one cause, fix outlined below and not in this diff.

---

## What this PR does

Extends SFPU edge-case coverage from the phase-2 baseline. The headline: **cat B (IEEE specials) goes
from 5 unary ops to 67 of 97**, plus all 5 scalar binops, and **cat E closes**.

Issue: [#49739](https://github.com/tenstorrent/tt-metal/issues/49739) · Previous phase:
[#52416](https://github.com/tenstorrent/tt-metal/pull/52416)

### Coverage added

| Area | Before | After |
|---|---|---|
| Cat B — unary ops injecting `±inf` / `NaN` / signed zeros | 5 | **67** of 97 |
| Cat B — scalar binops | 0 | **5** of 5 |
| Cat E — unary shift amount | one fixed value (3) | **full axis**, in range and out |
| Ternary operand-C poles | unreachable | driven |
| Broad unary profile in CI | **no job on any arch** | runs in llk-e2e's non-coverage groups |

### The framework defects this found

Most of the work was not per-op. Four shared defects accounted for nearly every divergence:

1. **torch's fp32 → bfloat16 cast canonicalises every NaN to `0xFFFF`, sign bit set.** A NaN crossing
   a 16-bit Dest came back negative whatever its true sign was, which is the whole of "`Neg(NaN)` is
   mangled at `dest_acc=No`". `cast_to_dest_dtype` models the Dest write as the truncation it is.
2. **The goldens exported the host libm's arbitrary sign for a *generated* NaN**, which IEEE leaves
   unspecified. 24 of the 97 were affected; found only because enrolling four ops regressed `Acosh`,
   `Cos`, `Sin` and `Exp` — ops the change was not about.
3. **`math.*` raises on a non-finite input** rather than returning NaN. Five ops (`sin`, `cos`, then
   `acos`, `asin`, `tan`), one defect, found three separate times.
4. **The goldens modelled IEEE comparisons; the SFPU does not implement them.** `SFPGT`, `SFPLE` and
   `SFPSWAP` document a total order `-NaN < -Inf < ... < +Inf < +NaN`, so 7 ops that looked like
   kernel divergences were golden bugs. See below.

### Two new gates

- **`negative_zero_delivered()`** — strictly narrower than `specials_safe()`. Several triples carry
  `±inf` and `NaN` intact while flattening `-0.0` on the datacopy path, so the `-0` probe is no longer
  sent where it cannot arrive. Without it `Rsqrt` failed for a datum the kernel never received.
- **`sfpu_total_order_key()`** — models the ISA's comparison order so the comparison family enrols as
  passes rather than xfails.

### Kernel divergences recorded (non-strict xfails)

Derived from the delivery rules rather than listed, so they cannot drift from the format axis:

| Op | Behaviour | Scope |
|---|---|---|
| `Reciprocal` | `1/NaN → +0`; NaN not propagated | every combination delivering a NaN |
| `Sqrt` | `sqrt(-0) → NaN` | unpack-to-dest only |
| `Rsqrt` | `rsqrt(-0) → NaN` | unpack-to-dest only |

### CI

The broad unary profile ran in **no automated job on any architecture** — every LLK python job either
excludes `nightly` or runs with coverage, under which `BROAD_SWEEP_OPS` is skipped wholesale. Every
gain above was therefore unguarded. `llk_e2e_tests.yaml` gains non-coverage companion groups
(`split_group` 6–10), targeted at `test_sfpu_unary.py` rather than the whole directory — `BROAD_SWEEP_OPS`
lives only there, so pointing them at `.` would re-run every other suite a second time per arch per night
for no added coverage. Cost: `wh_n150_civ2` 190 → 315 min, `bh_p150b_civ2` 275 → 400, against an 1800 min
per-SKU budget. **The timeouts are a reserved ceiling, not a measurement, and want one nightly's data to
tune.**

The `BROAD_SWEEP_OPS` skip cited tt-llk#1435, which is circular — that issue is about test *ordering*,
and its one mention of coverage is an observation of this skip's own effect. Citation removed; no
recorded rationale replaced it, because none exists.

## What is deliberately **not** in this PR

- **30 unary ops remain outside cat B**, and none is per-op work. 23 wait on a question about what an
  approximation kernel should do outside its series' range (the `Log` saturation, which turned out to
  be a 23-op family); 2 on whether `SFPSETCC` is usable with a `NaN` operand; 1 on `RsqrtCompat`'s
  pole; 3 have no golden; 1 is skipped on tt-llk#1120.
- **Cat F** — 11 kernels with no `MathOperation` entry. Each needs a new harness *shape*, not a
  dispatch entry: `quant`'s three entry points each take three Dest indices.
- **A Wormhole comparator fix** — see the blocker below. The 50 Wormhole failures this PR's enrolment
  exposes are one shared cause with one shared fix, and it is deliberately a separate change so this
  PR's diff stays about coverage.

## ⚠ Blocker: this PR turns Wormhole red, and the fix is not in it

The suites have now run on a **Wormhole n300** (previously this section said nothing was verified there).
Two of the three things it was uncertain about came back clean, and one did not:

- ✅ `specials_safe()`'s 7-cell matrix, re-measured with the original instrument — 250 variants, 85
  failing, the recorded figures to the variant. It stays un-arch-keyed.
- ✅ **The total order holds on Wormhole**, all 7 goldens 8/8, so they need **no** arch-keying. The
  premise for doubting it was also wrong: Wormhole has no `SFPGT`/`SFPLE`, but `WormholeB0/…/SFPSWAP.md`
  documents the same `SignMagIsSmaller()` total order.
- ❌ **50 variants fail on Wormhole** — 49 in the unary edge sweep, 1 (`ScalarRsub`) in the scalar suite —
  across 10 ops (`Cos`, `Fmod`, `GeluAppx`, `Hardmish`, `Mish`, `Rsqrt`, `Silu`, `Sin`, `Softsign`,
  `Tan`), all one cause: the **sign of a NaN the kernel generates**. `SFPMAD.md` guarantees the canonical
  `0x7fc00000` on Blackhole and says the sign "might or might not be set" on Wormhole; the conversion
  that makes it observable is documented too, and the ISA flags it itself ("NaN becomes infinity (this is
  a potentially surprising behaviour)"). So the kernels are in spec and the **golden** is asserting a sign
  the ISA declines to promise.

**Why it is this PR's problem.** `SPECIALS_READY_OPS` is empty on `main`, so nothing injects a NaN there
and Wormhole is green today. This PR's enrolment is what starts sending the probe. Both Wormhole e2e paths
then see failures and stop at the first one (`-x`): the non-coverage groups (`split_group` 6–10) hit all
10 ops, and the coverage groups still hit 6 of them, since `Fmod`, `GeluAppx`, `Hardmish`, `Mish`,
`Softsign` and `Tan` are standard-profile and the broad-profile skip does not reach them.

**The fix, in one sentence:** where a golden `NaN` becomes `±inf` through a Dest write or a pack, accept
either infinity, keeping the sign assertion only for the ops that *move* the sign bit (`Neg`, `Abs`,
`Identity` — `SFPABS`'s summary says "-NaN is left as -NaN rather than becoming +NaN"). Do **not**
arch-key the measured sign: it is explicitly unspecified, so a table of it would be a permanent
plausible-looking claim with an ISA sentence against it. Full record and both rejected alternatives:
`WORMHOLE_MEASUREMENT_RESULTS.md` §4.

## Verification

Blackhole p300a, all four suites green, `0 xpassed`. These figures are post-review-round: the 12 review
comments were addressed and re-verified, which moved unary from `5030 / 21 xfailed` to `5027 / 18` —
`Signbit`'s six xfails were deleted rather than kept, since the `-0.0` probe they recorded is no longer
sent where it cannot arrive, and the shift sweep dropped six redundant variants.

| Suite | Result |
|---|---|
| `test_sfpu_unary.py` | 5027 passed · 1601 skipped · 18 xfailed |
| `test_sfpu_binary.py` | 739 passed · 531 skipped · 36 xfailed · 0 xpassed |
| `test_sfpu_binop_scalar.py` | 68 passed · 72 skipped |
| `test_sfpu_ternary.py` | 39 passed · 25 skipped |
| `test_sfpu_domains.py` | 108 passed (host-side) |

Wormhole n300 — the first run of these suites on that arch, same two-phase flow:

| Suite | Result |
|---|---|
| `test_sfpu_unary.py` | 6034 passed · 533 skipped · **49 failed** · 30 xfailed · **6 xpassed** |
| `test_sfpu_binary.py` | 865 passed · 392 skipped · 33 xfailed · **16 xpassed** · 0 failed |
| `test_sfpu_binop_scalar.py` | 67 passed · 72 skipped · **1 failed** |
| `test_sfpu_ternary.py` | 39 passed · 25 skipped |

The failures are the blocker above. **The 22 XPASS are not this PR's** — they are two pre-existing
Wormhole-only arch gates, `_APPROX_EXP_ACCURACY_XFAIL` (6) and `negative_zero_golden` (16), each XPASSing
its entire content on the arch it was written for, so each currently asserts nothing on either arch.
Recorded rather than fixed here: `WORMHOLE_MEASUREMENT_RESULTS.md` §6 and the plan's §9. Approximate exp's
overshoot above `x = 8` measures ~1% mean / 3.5% peak on this board where the gate records 5.7% / 6.75%.

Note the skip counts rather than the pass counts when comparing the two arches: Wormhole skips 533 unary
variants where Blackhole skips 1601, because `_skip_bh_unless_fp32` collapses the whole `dest_acc=No` row
there. Wormhole exercises more of the format matrix, which is why it found this.

Every golden change was diffed against a baseline across **all** ops, not just the ones being enrolled
— that is what caught defect 2 above. Finite behaviour is bit-identical wherever a golden was rewritten
for NaN semantics.
