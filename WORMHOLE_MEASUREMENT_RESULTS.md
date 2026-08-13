# Wormhole re-measurement — results

**What this closes:** the *"Wormhole re-measurement — `specials_safe()`'s four unreachable rows, and
whether the total order holds there"* item, which was
[SFPU_EDGE_CASE_EXPANSION_PLAN.md](SFPU_EDGE_CASE_EXPANSION_PLAN.md) §1 item 3 and §4 up to revision 16.
It is deleted from the plan as of revision 17, whose §4 now carries the work this run *created* instead.
It also supplies the data for [KERNEL_OWNER_QUESTIONS.md](KERNEL_OWNER_QUESTIONS.md) §3's two open
Wormhole items — both now withdrawn — and adds §5.10 to
[SFPU_EDGE_CASE_COVERAGE.md](SFPU_EDGE_CASE_COVERAGE.md).

**Hardware:** Wormhole n300 (UMD chip 0, board `010001461…`), silicon, not simulator.
**Tree:** `tt-metal` branch `ldjurovic/sfpu_edge_cases_phase_3` @ `14ebb6b5815` — the commit that
enrolled the seven total-order ops. SFPI 7.68.0 (the pinned version), `tt-exalens==0.3.29`.
**Runner:** `.claude/scripts/run_test.sh run --arch wormhole` throughout; pytest was never called
directly. **Date:** 2026-08-13.

---

## 1. Verdicts

| # | Question the plan asked | Answer |
|---|---|---|
| 1 | Do `specials_safe()`'s 7 cells hold on Wormhole, or must the table be arch-keyed? | **Hold exactly. Do not arch-key.** 250 variants, 85 failing — the recorded figures to the variant |
| 2 | Does the SFPU total order hold on Wormhole, where the ISA documents no `SFPGT`/`SFPLE`? | **It holds.** All 7 enrolled ops pass 8/8; the other 3 comparison ops agree too. **The goldens do not need arch-keying** |
| 3 | *(not asked — found by running it)* | **49 of 752 edge variants fail on Wormhole, 10 ops, one cause: the sign of a generated NaN — and the ISA documents it, one sentence per arch.** §4 |
| 4 | *(not asked — found by running it)* | **Both Wormhole-only arch gates XPASS on Wormhole**: approximate-exp all 6 cells (the overshoot is ~1% mean, not the recorded 5.7%) and signed-zero all 16. §6 |

Nothing in this suite had ever run on Wormhole. Items 3 and 4 are what that first run found, and they are
the argument for running both arches: three of the tree's claims about Wormhole were written from a
Wormhole measurement nobody had repeated, and two of them no longer hold.

**The plan's own loop closed on item 3 at step 2 again.** *Measure, then read the ISA, then ask a human* —
the measurement located `SFPMAD.md`, which says the emitted NaN is canonical `0x7fc00000` on Blackhole
and sign-unspecified on Wormhole. So this is **golden work, not a kernel divergence and not a question
for an owner**, and it is the third time on this issue that reading the ISA before filing changed the
answer.

---

## 2. `specials_safe()` confirmed — 250 variants, 85 failing

The instrument the recorded table came from: the five `isinf`/`isposinf`/`isneginf`/`isnan`/`isfinite`
predicates over the full 5×5 float format matrix × both `dest_acc`, **no skips** — not gated by
`specials_safe()`, `_skip_bh_unless_fp32()`, or the shipped test's bf16+`dest_acc=Yes` skip.
Reproduced as `test_sfpu_wh_specials_measure.py`.

```
250 collected · 165 passed · 85 failed        (audit §6 records "250 variants, 85 failing")
```

Aggregated to the 50-cell matrix, **all 7 cells the table calls safe pass all 5 predicates, and no cell
the table calls safe fails any**:

| `dest_acc` | Safe `input -> output` | Wormhole |
|---|---|---|
| `No` | `Float32->Float32`, `Float32->Float16_b`, `Float16_b->Float32`, `Float16_b->Float16_b` | 5/5 each ✅ |
| `Yes` | `Float32->Float32`, `Float32->Float16`, `Float32->Float16_b` | 5/5 each ✅ |

Both breakers reproduce with the same shape:

- **A `Float16` input never preserves specials** — 0/5 predicates on all 5 outputs at both `dest_acc`,
  10/10 cells. As an *output* it fails too, except from a 32-bit input at `dest_acc=Yes`;
  `Float32->Float16` at `dest_acc=No` fails all five.
- **A 16-bit input with `dest_acc=Yes`** — `Float16_b` there fails `isinf`/`isneginf`/`isnan` and
  passes `isposinf`/`isfinite`: `+inf` survives, `-inf` and `NaN` do not. 2/5, exactly as recorded.

22 further cells pass all 5 predicates and are **excluded statically anyway** — every one has a
block-float input or output, where a passing predicate is vacuous (neither side ever saw a NaN). One
detail the audit does not record, and it argues *for* the static exclusion rather than against it: a
**`Bfp8_b` input at `dest_acc=No` genuinely fails** `isinf`/`isneginf`/`isnan` — 2/5 on four of its five
outputs, with only `Bfp8_b->Float16` passing — while a `Bfp4_b` input passes 5/5 on every output. So among
the excluded rows, some are excluded from a failure and some from a vacuous pass.

**Consequence for the plan:** §4's conditional — *"if it becomes arch-keyed,
`test_specials_safe_matches_measured_matrix` has to be parametrized by arch too"* — does not fire.
The one-verdict-per-cell pin stays correct.

---

## 3. The total order holds on Wormhole

`test_eltwise_unary_sfpu_edges` on Wormhole, the seven ops the plan flagged as
"Blackhole-documented and Wormhole-unverified", plus the three that agreed with IEEE anyway:

| Op | Wormhole edge variants | Op | Wormhole edge variants |
|---|---|---|---|
| `Clamp` | 8 passed, 0 failed | `UnaryGt` | 8 passed, 0 failed |
| `Hardsigmoid` | 8 passed, 0 failed | `UnaryMin` | 8 passed, 0 failed |
| `Hardtanh` | 8 passed, 0 failed | `UnaryLt` / `UnaryLe` / `UnaryMax` | 8 passed each |
| `ReluMax` | 8 passed, 0 failed | | |
| `UnaryGe` | 8 passed, 0 failed | **XPASS across the whole run** | **0** |

A direct probe of the comparison family at `Float32->Float32`, both `dest_acc`
(`test_sfpu_wh_order_probe.py`), over `+inf / -inf / NaN / +0 / -0`. Every value is what the
total order predicts, and identical to the Blackhole table in KERNEL_OWNER_QUESTIONS §3:

| Op | `NaN` → hardware | golden | Under `-NaN < -Inf < … < +Inf < +NaN` |
|---|---|---|---|
| `UnaryGt` (`x > 0.5`) | `1.0` | `1.0` | `+NaN` is greater — agrees |
| `UnaryGe` (`x >= 0.5`) | `1.0` | `1.0` | agrees |
| `UnaryLt` / `UnaryLe` | `0.0` | `0.0` | `+NaN` is not less — agrees |
| `UnaryMax(x, 0)` | `NaN` | `NaN` | `+NaN` is the max — agrees |
| `UnaryMin(x, 0)` | `0.0` | `0.0` | the other operand — agrees |
| `Clamp` / `Hardtanh` / `Hardsigmoid` | `1.0` | `1.0` | the upper-bound constant — agrees |
| `ReluMax` | `5.0` | `5.0` | agrees |

**The seven goldens stand on both arches.**

### 3.1 Correction: the total order is *not* Blackhole-only

The plan and KERNEL_OWNER_QUESTIONS both say *"Wormhole has no total order, because it has no
`SFPGT`/`SFPLE`"*. The first half is right and the second does not follow.
`WormholeB0/TensixTile/TensixCoprocessor/SFPSWAP.md` carries the same `SignMagIsSmaller()` model as
Blackhole's, with the same comment word for word:

> This treats C and D as sign-magnitude integers and determines whether C is less than D. If C and D are
> instead FP32, this still determines whether C is less than D, **using the total order where
> `-NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN`**.

So **the total order is specified on Wormhole**, for `SFPSWAP`. What Wormhole lacks is the two
compare-and-set-flag instructions — and that is where the honest limit of this run sits.

**What is not established: which instruction each of these seven kernels actually uses on Wormhole.**
The sources are two-vector compares — `_relu_max_body_` is `v_if (result > threshold)` and
`_calculate_clamp_` has the same shape, which is why the Blackhole write-up attributes them to `SFPGT` —
and `SFPGT` does not exist on Wormhole. sfpi lowers `operator>` through a compiler builtin
(`__builtin_rvtt_sfpxfcmpv`) whose expansion lives in the sfpi backend rather than in the headers, so
what it becomes on Wormhole was **not** verified here: `SFPSWAP`, or a subtract plus `SFPSETCC`, are both
plausible and they have different contracts. If it is the latter, the agreement rests on the sign of the
NaN that subtract emits — the very thing `SFPMAD.md` leaves unspecified on Wormhole (§4.1).

Do not read this section as "the seven are ISA-guaranteed on Wormhole". Read it as: the order exists on
Wormhole, the goldens are **measured** green there, and no arch-keying is needed today. Turning that into
a guarantee is one disassembly away — build a `relu_max` and a `unary_gt` kernel and read the opcode —
and that is the follow-up for anyone who needs the promise rather than the observation.

### `Sign` and `Heaviside` — Wormhole answers Q3's item 2 the same way

They go through `SFPSETCC`, whose contract excludes a `NaN` operand on **both** arches. Measured on
Wormhole, and identical to the Blackhole record:

| Op | probe | hardware | golden |
|---|---|---|---|
| `Sign` | `NaN` | `1.0` | `0.0` |
| `Heaviside` | `NaN` | `1.0` | `0.5` |
| `Sign` | `-0` at `dest_acc=Yes` | `-1.0` | `0.0` |

The first two are the out-of-contract case the question is about; the third is the recorded signed-zero
xfail, and it reproduces on Wormhole (2 xfailed for each op, **0 xpassed**). So Q3's Wormhole item is
answered as data: *Wormhole behaves exactly as Blackhole here*, which means the open question is a
contract question for the owners, not a per-arch measurement gap.

---

## 4. New finding — 49 Wormhole edge failures, one cause

`test_eltwise_unary_sfpu_edges`, first run on Wormhole:

```
752 selected · 475 passed · 198 skipped · 49 failed · 30 xfailed · 0 xpassed
```

The 49 failures are **10 ops** — `Cos`, `Fmod`, `GeluAppx`, `Hardmish`, `Mish`, `Rsqrt`, `Silu`, `Sin`,
`Softsign`, `Tan` — and they are one cause, not ten. Characterised by re-driving the identical stimulus
and recording the distinct disagreeing triples (`test_sfpu_wh_edge_diag.py`):

| Op | probe | golden | Wormhole |
|---|---|---|---|
| `Cos`, `Sin` | `+inf`, `NaN` | `+inf` | **`-inf`** |
| `Silu`, `Mish`, `Softsign`, `GeluAppx`, `Hardmish`, `Fmod` | `-inf` | `+inf` | **`-inf`** |
| `Tan` | `-inf` | `+inf` | **`-inf`** |
| `Tan` | `NaN` | `+inf` | **`0.0`** |
| `Rsqrt` | `NaN` | `+inf` | **`-inf`** |

Every one of these ops **generates** a NaN from the probe, and the failing cells are exactly the ones
where a NaN cannot survive as a NaN. Of the 8 (format pair × `dest_acc`) cells, 6 carry specials at all —
`specials_safe()` rules out `Float16_b` inputs at `dest_acc=Yes`, which is where the 198 skips come
from — and of those 6, **5 fail and 1 passes**:

| Cell | NaN's fate | Outcome |
|---|---|---|
| all four format pairs at `dest_acc=No` | 16-bit Dest → substituted `±inf` | **fail** |
| `Float32->Float16_b` at `dest_acc=Yes` | fp32 Dest, 16-bit output pack → substituted `±inf` | **fail** |
| `Float32->Float32` at `dest_acc=Yes` | stays a NaN all the way to L1 | **pass** |

The one passing cell passes because `passed_test`'s both-NaN clause accepts a NaN of either sign — which
is also what makes it the only place the sign can be *read*.

**The mechanism, read off the bit patterns rather than inferred.** At `Float32->Float32`,
`dest_acc=Yes` the NaN reaches L1 intact, so its sign can be read:

| Op | `+inf` → | `-inf` → | `NaN` → |
|---|---|---|---|
| `Cos`, `Sin`, `Softsign`, `Tan`, `Fmod` | `0x7FC00001` | **`0xFFC00001`** | `0x7FC00001` |
| `Silu`, `Mish`, `Hardmish`, `GeluAppx` | (finite) | **`0xFFC00001`** | `0x7FC00001` |
| `Rsqrt` | (0.0) | `0x7FC00000` | **`0xFFF00001`** |

So a Wormhole-generated NaN **can carry a set sign bit**, and which sign depends on the input and on
the datapath. `convert_nan_to_inf` reads that sign deliberately — it models the pack path, which
rewrites exponent and mantissa and leaves the sign alone — so on a 16-bit Dest the arbitrary sign
becomes a visible `+inf`/`-inf` disagreement.

**This lands on a named assumption, and the assumption is Blackhole-measured.**
`UnarySFPUGolden._NAN_SIGN_TRANSPARENT_OPS`'s comment records the reasoning exactly:

> For every other op a NaN result is an invalid-operation default, and IEEE 754 leaves the sign of that
> NaN unspecified. **The SFPU emits a positive one**; torch inherits the host libm and picks either …
> Canonicalise, so the golden asserts the sign only where the sign means something.

The canonicalisation is right; its premise is Blackhole-only, and it is **Blackhole-only by
specification** — see §4.1. On Wormhole the SFPU does not always emit a positive NaN, so canonicalising
to `+NaN` makes the golden assert `+inf` where the hardware packs `-inf`.

**It is an arch difference, not a coverage artefact.** Two of the five failing cells are reachable on
Blackhole and are recorded green there — `Float32->Float32` at `dest_acc=No` (the only cell
`_skip_bh_unless_fp32` allows) and `Float32->Float16_b` at `dest_acc=Yes`. Same variant, same golden,
opposite outcome. (Blackhole was not re-run here — no Blackhole card on this host — so that half rests
on the plan's recorded baseline, `5030 passed · 21 xfailed · 0 xpassed`.)

### 4.1 The ISA documents this, one sentence per architecture

Checked before filing anything, per the plan's rule, and it settles both the cause and the repair.
`SFPMAD.md` is where each arch pins down the bit pattern of an emitted NaN, and the two pages say
different things. (Wormhole also states the mantissa half generally, for the whole vector unit, in
`FloatBitPatterns.md`; Blackhole has no equivalent page, so `SFPMAD` is the citation on both sides.)

| Arch | `SFPMAD.md`, "if a NaN is emitted" |
|---|---|
| Blackhole | "it is always **the canonical NaN with bit pattern `0x7fc00000`**" |
| Wormhole | "the least significant bit of the mantissa is guaranteed to be set; other bits of the mantissa might or might not be set, and **the sign bit might or might not be set**" |

So the golden's premise is a *documented guarantee* on Blackhole and *explicitly unspecified* on
Wormhole. `WormholeB0/…/FloatBitPatterns.md` repeats the mantissa half as a property of the vector unit
generally: "the least significant bit of the mantissa will always be set, and the remaining bits may or
may not be set."

The second half — why an unspecified sign becomes an observable `±inf` — is documented at every point
where a NaN meets a narrower format, and each mention carries a warning:

- **`SFPSTORE`** (LReg → Dst), on both arches: converting to FP16, "NaN is also converted to infinity,
  so software is advised to avoid NaN inputs for this conversion"; converting to BF16, "the mantissa
  truncation can turn _some_ NaN values into infinity". Blackhole's sentence then adds the clause
  Wormhole's lacks: "**albeit canonical NaNs produced by arithmetic instructions do not suffer any
  truncation**". Both functional models keep the sign verbatim (`Sign = x & 0x80000000`).
- **The packer's format conversion** (`WormholeB0/…/Packers/FormatConversion.md`): in the early
  conversion, "**Rounding:** … If the exponent is 8 bits wide, **NaN becomes infinity (this is a
  potentially surprising behaviour)**"; "**Truncation:** NaN can become infinity, if that is the outcome
  of truncating mantissa bits". In the late conversion, an exponent narrowing from 8 bits to 5 "NaN
  becomes infinity, as types with a 5-bit exponent are treated as not having NaN".
- **`Dst.md`**: a 16-bit FP16 Dst does not support NaN at all — "the representation of infinity differs
  from IEEE 754, and NaN is not supported".

That is the whole mechanism, and the two halves compose exactly as measured: Blackhole's canonical
`0x7fc00000` has mantissa bit 22 set, so BF16 truncation keeps it a NaN and nothing is observable;
Wormhole's may have its set bits anywhere below that, so truncation can leave mantissa zero — an
infinity — and the sign bit that "might or might not be set" rides through untouched.

`convert_nan_to_inf`'s docstring — "it models the pack path, which does not synthesise a value: it
leaves the sign bit alone and rewrites the exponent/mantissa" — is confirmed by those functional
models. The model is right. What is unsound on Wormhole is only the *sign it is fed*.

### 4.2 So the repair is not a coin flip

Two repairs looked defensible before reading the ISA. It picks one:

1. **Arch-key the canonicalisation** — keep asserting a sign, but assert the sign Wormhole produces.
   **The ISA rules this out**: the sign "might or might not be set" is not a fact to tabulate, and
   `Cos(+inf)` already yields a positive NaN at a 32-bit Dest and a sign-set one at a 16-bit Dest.
   Encoding it would be exactly the plan's warned-against "permanent, plausible-looking lie about the
   hardware", with an ISA sentence contradicting it.
2. **Stop asserting the sign of a substituted infinity** ✅ — where a golden `NaN` is turned into `±inf`
   by a Dest write or a pack, accept either infinity, and keep the sign assertion only for
   `_NAN_SIGN_TRANSPARENT_OPS`. That carve-out is itself ISA-backed: `SFPABS`'s summary says "-NaN is
   left as -NaN rather than becoming +NaN", and `Neg` compiles to a sign-bit flip (`-val` on a
   `vFloat` in `_calculate_negative_`), so for those ops the sign is a datum the kernel *moved*, not
   one it invented.

This is a comparator/cast change in the shared machinery, so §8's diff-the-whole-op-set rule applies to
it with full force — and per §2.3 it should be driven across the whole op set in one sweep, on both
arches, rather than op by op.

**The 10 ops are not buggy and need no xfail.** Nothing here is a kernel divergence: the kernels emit a
NaN the ISA permits, and the conversion that makes its sign visible is documented and flagged
"potentially surprising" by the ISA itself. Writing 49 xfails for it would record a hardware defect
that does not exist.

**One residue that this does not explain:** `Tan(NaN) -> 0.0` on the 16-bit-Dest path — a finite zero,
not a substituted infinity, so neither the `SFPMAD` sentence nor the pack-path substitution covers it.
`Tan` at `Float32->Float16_b`, `dest_acc=Yes` gives a NaN (`0xFFC00001`) for the same probe, so the
`0.0` belongs to the 16-bit-Dest path specifically. It wants its own measurement before anyone decides
whether it is golden work too.

---

## 5. Wormhole baseline for the four suites

The plan's §7 baseline is a Blackhole p300a. This is the Wormhole n300 counterpart, same runner, same
two-phase flow, no `--maxfail`.

| Suite | Wormhole n300 | Blackhole p300a (recorded) |
|---|---|---|
| `test_sfpu_unary.py` | 6034 passed · 533 skipped · **49 failed** · 30 xfailed · **6 xpassed** | 5030 passed · 1601 skipped · 21 xfailed · 0 xpassed |
| `test_sfpu_binary.py` | 865 passed · 392 skipped · 0 failed · 33 xfailed · **16 xpassed** | 739 passed · 531 skipped · 36 xfailed · 0 xpassed |
| `test_sfpu_ternary.py` | 39 passed · 25 skipped · 0 failed | 39 passed · 25 skipped |
| `test_sfpu_binop_scalar.py` | 67 passed · 72 skipped · **1 failed** | 68 passed · 72 skipped |

Three things in that table are not the Blackhole numbers with a different card in them.

**The 49 unary failures are §4**, and **the 1 scalar failure is §4 as well** —
`ScalarRsub` at `Float16_b->Float16_b`, `dest_acc=No`, diverging `golden=+inf` against `hw=-inf` on the
`NaN` probe while the `+inf` and `-inf` probes agree. `rsub` computes `scalar - x`, so `2 - NaN` is the
kernel's own generated NaN meeting a 16-bit Dest: the same cause, the same signature, in a second suite.
Whatever fixes §4 has to fix this variant too, and the acceptance criterion should say so.

**The counts are not comparable across arches, and the difference is not coverage.** Wormhole runs 6652
unary variants where Blackhole runs 6652 too, but Blackhole *skips* far more of them
(1601 vs 533): `_skip_bh_unless_fp32` collapses the whole `dest_acc=No` row to `Float32->Float32` there.
So Wormhole exercises more of the format matrix, which is the other reason this run found something.

**The 6 and 16 xpassed are the subject of §6.**

---

## 6. Two arch gates XPASS on the arch they were written for

`0 xpassed` is the plan's tripwire — *"if a run reports XPASS again, something the tables call
arch-specific has changed, and that is worth more than most deliberate work"*. It fired twice, and in
both cases the xfail is one that only applies **on Wormhole**, so these are not Blackhole-style
"the other arch is better" results. They are gates asserting nothing on either arch.

### 6.1 The approximate-exp gate — 6 XPASS, and the overshoot is ~5× smaller than recorded

All six are `Exp`, `approx_mode=Yes`, and they are `_APPROX_EXP_ACCURACY_XFAIL`'s entire content — all
three gated cells at both tile shapes:

```
Float16 -> Float16_b  dest_acc=No    XPASS   ([64,64] and [128,256])
Float16 -> Float16_b  dest_acc=Yes   XPASS
Float32 -> Float16_b  dest_acc=Yes   XPASS
```

The gate records *"a systematic ~5.7% overshoot (peak 6.75%) once approximate exp's argument passes ~8,
measured on Wormhole"*. Measured now on a Wormhole n300 (`test_sfpu_wh_approx_exp.py`), signed relative
error over the elements with `x > 8`:

| Cell | mode | `x_max` | n(x>8) | mean rel | max rel | frac > 5% |
|---|---|---|---|---|---|---|
| `Float16->Float16_b` `No` | approx | 9.98 | 425 | **+0.75%** | +3.37% | 0 |
| `Float16->Float16_b` `Yes` | approx | 9.98 | 425 | **+1.03%** | +3.51% | 0 |
| `Float32->Float16_b` `Yes` | approx | 15.98 | 261 | **+1.05%** | +3.49% | 0 |
| `Float16_b->Float16_b` `Yes` (control, not gated) | approx | 15.50 | 261 | +1.10% | +3.37% | 0 |

**The direction reproduces and the magnitude does not.** Approximate exp does overshoot above 8 — the
mean error is positive at every cell — but at ~1% mean and ~3.5% peak, not ~5.7% mean and 6.75% peak. It
sits inside the default 5% rtol with room to spare, and **not one element of any tile breaches 5%**,
which is why all six XPASS. Note the control: `Float16_b->Float16_b` at `dest_acc=Yes` is deliberately
*not* in the gate and behaves identically to the cells that are, so whatever the gate discriminates on
is no longer discriminating.

Three explanations were eliminated first, so nobody re-checks them:

- **Not the stimulus.** The overshoot region is still exercised: 425 and 261 elements above 8 per tile
  (6.4–10.4%), `x_max` 9.98 / 15.98, and `_APPROX_ACCURACY_MAX[Exp]` is 16.0 — well clear of ~8.
- **Not a loosened tolerance.** `CUSTOM_TOLERANCES` has no `Exp` entry, and `passed_test` requires
  `torch.all(is_valid)`: one element over 5% fails the variant.
- **Not a softened golden.** `_exp` is plain `torch.exp`.

So either the kernel's approximate-exp path has changed since the gate was measured, or the overshoot
varies across Wormhole boards. The recorded measurement does not name the card it was taken on — which
is the first thing whatever replaces it should fix.

**A by-product: the accurate path over (16, 80] is now measured on Wormhole**, which `_APPROX_ACCURACY_MAX`'s
comment flags as *"NOT YET MEASURED … it still wants a run of the Exp/Exp2 broad sweep at
ApproximationMode.No"*. That run has now happened — `Exp` 132 passed / `Exp2` 138 passed / 0 failed on
Wormhole — and the probe's `approx_mode=No` rows put the error at **+0.00%** above 8 out to `x_max = 79.97`
for the 32-bit-input cells (≤0.8% for a `Float16` input, which is the format's own rounding). The restored
`high=80` domain is sound on Wormhole.

### 6.2 The signed-zero gate — 16 XPASS, and the arch reading cannot survive it

All sixteen are `edge_class:negative_zero_golden`: four ops (`SfpuElwdiv`, `SfpuXlogy`, `SfpuBinaryFmod`,
`SfpuBinaryRemainder`) × four `(format, dest_acc)` cells. That is, again, the gate's entire content.

`_WORMHOLE_ONLY_EDGE_CLASSES` exists because *"measured on a Blackhole p150b, the negative-zero class
XPASSed on **all 16** cells it is claimed for"*, read as: `SFPMAD` flushes a negative zero to positive
zero on Wormhole and preserves it on Blackhole, so Blackhole should *assert* the sign. The ISA does say
that, and Blackhole's page does list "improved edge-case handling of NaNs and of negative zero".

**But the same 16 now XPASS on Wormhole, where the xfail is the one that applies** — and that reading
cannot explain a gate that XPASSes on both arches. The likely explanation is already written down as a
trap in the audit: *"`passed_test()` compares with `torch.isclose`, a both-NaN clause and PCC, and
`-0.0 == +0.0` under every one of them"*. If the comparator cannot see a zero's sign, these variants pass
regardless of what the hardware does, on either arch — and the Blackhole XPASS was evidence about the
comparator rather than about Blackhole.

That is a hypothesis, not a measurement, and it is worth exactly one cheap experiment rather than an
argument: compare that class's output **bitwise** on Wormhole. Two outcomes, both useful — if hardware
returns `+0.0` where the golden says `-0.0`, the divergence is real and invisible, so the class needs the
bitwise comparator the audit already asks for and the arch gate is spurious; if hardware returns `-0.0`,
Wormhole is not flushing after all and the gate's premise is wrong on its own terms.

Until then, **do not read `_WORMHOLE_ONLY_EDGE_CLASSES` as verified.** It was derived from an XPASS, and
the same XPASS has now appeared on the arch it excludes.

---

## 7. How to reproduce

Three temporary instruments were written for this and are **not for merge** — each records rather than
asserts, because most of what they drive is expected to diverge. They live in
`tests/python_tests/` in the tree above:

| File | What it measures |
|---|---|
| `test_sfpu_wh_specials_measure.py` | The 250-variant predicate matrix, no skips (§2) |
| `test_sfpu_wh_order_probe.py` | The comparison family and the generated-NaN bit patterns (§3, §4) |
| `test_sfpu_wh_edge_diag.py` | The disagreeing `(input, golden, hardware)` triples behind the 49 failures (§4) |
| `test_sfpu_wh_approx_exp.py` | Approximate exp's signed relative error above `x = 8`, with an ungated control cell (§6.1) |

```bash
cd tt_metal/tt-llk
# environment: the venv run_test.sh expects is not created by setup_testing_env.sh
python3 -m venv tests/.venv && tests/.venv/bin/pip install -r tests/requirements.txt
bash tests/setup_testing_env.sh                      # SFPI 7.68.0 into tests/sfpi

# host-side first, no device
cd tests/python_tests && ../.venv/bin/python -m pytest test_sfpu_domains.py -q --noconftest
# 107 passed

# the three measurements
cd ../..
bash .claude/scripts/run_test.sh run --worktree $PWD --arch wormhole \
    --test test_sfpu_wh_specials_measure.py --maxfail 0
WH_PROBE_OUT=/tmp/probe.jsonl bash .claude/scripts/run_test.sh run --worktree $PWD \
    --arch wormhole --test test_sfpu_wh_order_probe.py --maxfail 0
WH_DIAG_OUT=/tmp/diag.jsonl bash .claude/scripts/run_test.sh run --worktree $PWD \
    --arch wormhole --test test_sfpu_wh_edge_diag.py --maxfail 0
WH_EXP_OUT=/tmp/exp.jsonl bash .claude/scripts/run_test.sh run --worktree $PWD \
    --arch wormhole --test test_sfpu_wh_approx_exp.py --maxfail 0

# the sweep itself, then the four-suite baseline
bash .claude/scripts/run_test.sh run --worktree $PWD --arch wormhole \
    --test test_sfpu_unary.py --k test_eltwise_unary_sfpu_edges --maxfail 0
for t in test_sfpu_unary.py test_sfpu_binary.py test_sfpu_ternary.py test_sfpu_binop_scalar.py; do
    bash .claude/scripts/run_test.sh run --worktree $PWD --arch wormhole \
        --test "$t" --maxfail 0 --timeout 900
done
```

The unary suite takes ~21 min of silicon time on top of its compile; the other three are minutes. Run them
**sequentially** — `TestConfig` deletes `/tmp/tt-llk-build` at session setup, so a second pytest session
destroys the first one's build tree (§8 of the plan).

Two things that cost time here and are worth adding to the plan's §8:

- **`run_test.sh count` ignores `--k`.** It reports the whole file's collection, so the edge sweep looks
  like 6652 variants when the filter selects 752. Only the `run`/`simulate` paths honour the filter —
  size a `--maxfail` off the run's own `deselected` line, not off `count`.
- **A failing variant's log makes the outcome unparsable by the obvious grep.** pytest writes
  `<nodeid> FAILED`, but loguru dumps the golden and result tensors *between* the two, and those lines
  carry their own `| ERROR |` prefix. A naïve scan attributes `ERROR` to the pending nodeid and invents
  outcomes — it reported 128 errors for a run that had 49 failures. Parse sequentially and anchor the
  outcome token to end-of-line.
