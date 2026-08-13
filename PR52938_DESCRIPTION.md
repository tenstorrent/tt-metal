# PR #52938 description — paste into the PR body

> Copilot's sixth comment: *"The PR body only points to issue #49739 and PR #52416, so it does not
> explain this PR's own changes."* This is a replacement body. Everything below is measured on a
> Blackhole p300a unless stated otherwise.

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
(`split_group` 6–10). Cost: `wh_n150_civ2` 190 → 380 min, `bh_p150b_civ2` 275 → 550, against an 1800
min per-SKU budget. **The timeouts are copied from the instrumented groups and want one nightly's data
to tune.**

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
- **Nothing is verified on Wormhole.** In particular the total order is documented for Blackhole only
  — Wormhole has no `SFPGT`/`SFPLE` — so those 7 goldens may need arch-keying.

## Verification

Blackhole p300a, all four suites green, `0 xpassed`:

| Suite | Result |
|---|---|
| `test_sfpu_unary.py` | 5030 passed · 1601 skipped · 21 xfailed |
| `test_sfpu_binary.py` | 739 passed · 531 skipped · 36 xfailed · 0 xpassed |
| `test_sfpu_binop_scalar.py` | 68 passed · 72 skipped |
| `test_sfpu_ternary.py` | 39 passed · 25 skipped |
| `test_sfpu_domains.py` | 107 passed (host-side) |

Every golden change was diffed against a baseline across **all** ops, not just the ones being enrolled
— that is what caught defect 2 above. Finite behaviour is bit-identical wherever a golden was rewritten
for NaN semantics.
