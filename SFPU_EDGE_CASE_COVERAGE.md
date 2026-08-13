# SFPU LLK Edge-Case Test-Coverage Audit

**Issue:** [tenstorrent/tt-metal#49739 — [LLK] SFPU testing edge cases](https://github.com/tenstorrent/tt-metal/issues/49739)
**Plan for what is left:** [SFPU_EDGE_CASE_EXPANSION_PLAN.md](SFPU_EDGE_CASE_EXPANSION_PLAN.md)
**Audited:** 2026-07-23 · **Regenerated from code:** 2026-08-13 (revision 15)
**Wormhole measurement:** [WORMHOLE_MEASUREMENT_RESULTS.md](WORMHOLE_MEASUREMENT_RESULTS.md)
**Scope:** All SFPU LLK kernels in `tt-metal/tt_metal/tt-llk`, audited through the tt-llk Python test
infra (`tests/python_tests/`). Wormhole B0 and Blackhole share essentially the same SFPU kernel set
(BH adds only `topk_xl`), so this audit treats them together and notes arch-specific gaps inline.
Quasar has its own suite under `quasar/` and is **out of scope** — an op driven only from `quasar/`
counts as untested here.

> ### Revision 15 — the suite has run on Wormhole, and one family failed there
>
> Two claims that had never been exercised on Wormhole are now measured, and both hold: **`specials_safe()`'s
> 7 cells** (250 variants, 85 failing — the §6 figures to the variant, so the table stays un-arch-keyed) and
> **the total order** (all seven revision-12 ops pass 8/8). One correction falls out of it: Wormhole's
> `SFPSWAP.md` *does* document the total order, so §5.8's "Blackhole-only" caveat was too strong.
>
> **Three findings came out of the same run**, and all three are about claims that were written from a single
> arch:
>
> - **§5.10** — 49 of 752 edge variants fail on Wormhole across 10 ops, plus 1 in the scalar suite: one
>   cause, and the ISA documents it, an emitted NaN's sign being canonical-positive on Blackhole by
>   specification and explicitly unspecified on Wormhole. Golden work, not a kernel divergence; do not file
>   it as one.
> - **§5.11** — the approximate-exp arch gate XPASSes on Wormhole, all 6 cells; the overshoot measures ~1%
>   mean where the gate records 5.7%.
> - **§5.12** — the signed-zero arch gate XPASSes on Wormhole, all 16 cells, which the arch reading that
>   created it cannot explain.
>
> The two gates now assert nothing on either arch. Neither breaks CI — they are non-strict — which is why
> they need naming.
>
> *Revision 14 below ran concurrently with this, not before it: the review round was
> verified on Blackhole while the Wormhole measurement was in flight, which is why §5.10's
> failures are reported against ops this PR had just enrolled.*
>
> ### Revision 14 — the review round, verified on Blackhole
>
> Twelve review comments addressed. The one that mattered most: this document *measured* that the
> datacopy path converts `-0.0` to `+0.0` and then kept sending the probe there anyway.
> `negative_zero_delivered()` now gates cat D's zero knee as well as cat B's specials, so
> **`Signbit`'s six xfails are deleted rather than kept** — they could never XPASS, which makes them
> the shape of thing that masks a regression instead of recording one.
>
> Verified on a Blackhole p300a: the three BH-reachable `Signbit` variants flipped from XFAIL to PASS,
> unary went 21 → 18 xfailed, and no suite reports an XPASS. `xlogy`'s Float32 tolerance tightened
> from 0.6 to 0.14 and the binary suite is unchanged at 739/531/36.

>
> ### Revision 13 — a review caught the shift rule, and this document had it wrong too
>
> Copilot's review of #52938 found that `calculate_left_shift` and `calculate_right_shift` do **not**
> share an out-of-range rule: left zeroes, right clamps the amount to 31 and shifts anyway, so a
> negative operand gives `-1`. §2.3 asserted the shared rule and is corrected, as are the goldens.
>
> The correction is currently unreachable — negative int32 operands cannot be delivered at all, which
> was re-tested rather than assumed — so what changed is that the golden is right where it was wrong,
> and §2.3 now says how much the assertion actually covers rather than implying it covers both signs.
>
> ### Revision 12 — the seven ops the ISA freed are enrolled
>
> The total order is now modelled (`sfpu_total_order_key` and its `min`/`max`/`clamp`/`relu_max`
> helpers), and `Clamp`, `Hardsigmoid`, `Hardtanh`, `ReluMax`, `UnaryGe`, `UnaryGt` and `UnaryMin`
> pass as ordinary tests rather than xfails — **67 of the 97 unary ops are enrolled**. The
> op→instruction mapping was confirmed against the kernels first: `_relu_max_body_` and
> `_calculate_clamp_` are two-vector compares, and `Hardsigmoid` turned out to *be*
> `_relu_max_body_(x/6 + 0.5, 1.0)`. Over 8000 finite inputs the rewritten goldens are bit-identical
> to the ones they replace; only the NaN answers moved.
>
> Every cat-B op that remains is now waiting on someone else — §5.6's two questions, or a harness.
>
> ### Revision 11 — the ISA answers the `NaN` comparison question, and reverses it
>
> `tt-isa-documentation` specifies a total order for FP32 — `-NaN < -Inf < ... < -0 < +0 < ... < +Inf
> < +NaN` — on `SFPGT`, `SFPLE` and `SFPSWAP`, all routed through `SignMagIsSmaller()`. So §5.8's
> measurement is documented behaviour and **the goldens are the wrong party**: they model IEEE's
> unordered comparisons, which the SFPU does not implement. Seven of those nine ops become golden work
> rather than kernel divergences; recording them as xfails would have been a permanent lie about
> documented hardware.
>
> Two things the ISA did not close, both now scoped: `SFPSETCC` explicitly excludes a `NaN` operand
> (so `Sign` and `Heaviside` stay open), and the total order looked **Blackhole-only** because Wormhole has
> no `SFPGT`/`SFPLE` — revision 13 corrects that (`SFPSWAP` specifies the order on Wormhole as well) and
> measures all seven green there. §5.6 keeps the two questions that remain, and records why the ISA cannot
> settle the approximation-contract one.
>
> ### Revision 10 — cat E closed, the scalar binops enrolled, and the CI hole shut
>
> Three of the plan's items are gone, and one of them was quietly undermining all the others.
>
> * **CI ran none of this.** Every LLK python job either excluded the `nightly` marker or ran with
>   coverage, which skips the broad profile wholesale — so the largest part of the sweep executed in
>   no job on any architecture, and every gain in this document was unguarded. `llk-e2e` now has
>   non-coverage companion groups. §2.8.
> * **Cat E is closed.** The unary shift amount sweeps its full axis, including the out-of-range half
>   the fixed shift of 3 could never reach. §2.3.
> * **The five scalar binops are enrolled**, and the edge test that had been a comment is live. Their
>   golden was modelling neither Dest nor the pack path, which only showed once a NaN was driven
>   through it.
>
> ### Revision 9 — cat B goes from 9 ops to 60, and the rest reduces to two questions
>
> The remaining tail was driven all at once — all 84 unenrolled ops with a golden, over the full
> specials set, on every Blackhole-reachable triple — instead of op by op. That was the right call only
> because revision 8 had finished fixing the shared machinery; before that, each tranche's framework
> defect would have been misattributed to whichever op was in hand.
>
> * **48 ops agreed with their goldens everywhere** and are enrolled with no change.
> * **7 golden defects total, none of them per-op.** Three shared patterns: `math.*` raising on a
>   non-finite input (`sin`, `cos`, `acos`, `asin`, `tan` — one defect found three separate times), a
>   finite-input guard answering false for `NaN` (`Square`'s overflow test, `Hardshrink`'s band test),
>   and taking torch at face value where torch is not the mathematics (`I0`, `I1` at `±inf`).
> * **The 32 remaining divergences are two kernel behaviours, not 32 decisions.** §5.9: 23 ops where an
>   approximation kernel saturates or NaNs an input outside its series' range — the `Log` finding was
>   never about `Log`. §5.8: 9 ops where SFPU comparisons rank `NaN` above every finite value, derived
>   from the exact pass/fail split across the six unary comparison ops.
> * Both are written up for owners in `KERNEL_OWNER_QUESTIONS.md`. Between them they decide 31 of the
>   37 ops still outside.
>
> ### Revision 8 — the second cat-B tranche, and the golden defect it was really about
>
> `Neg`, `Reciprocal`, `Sqrt` and `Rsqrt` are enrolled and green on Blackhole (p300a), taking
> `SPECIALS_READY_OPS` from 5 ops to 9. `Log` is the only one of the tranche still out, and only
> because §5.6's question about its input saturation is unanswered.
>
> * **The blocking defect was in the test framework, not in four goldens.** torch's fp32 → bfloat16
>   cast canonicalises every NaN to `0xFFFF`, sign bit set. That is the whole of "`Neg(NaN)` is
>   mangled at `dest_acc=No`" — and it was also silently wrong for 24 other ops, found only because
>   enrolling four ops regressed `Acosh`, `Cos`, `Sin` and `Exp`. §5.7.
> * **A second, independent sign bug in the same area:** the goldens were exporting the host libm's
>   arbitrary choice of sign for a *generated* NaN, which IEEE leaves unspecified. §5.7.
> * **Two rows left the blocking list without any work**, because the comparator cannot see a zero's
>   sign at all. §5.2.
> * **`-0.0` scoping is now enforced**, not advised: `negative_zero_delivered()` keeps the probe off
>   the pipelines that flatten it, which is a strictly narrower gate than `specials_safe()`. §5.2.
> * **Three genuine kernel divergences recorded**, derived from the delivery rules rather than listed:
>   `1/NaN → +0`, `sqrt(-0) → NaN`, `rsqrt(-0) → NaN`. §2.1.
>
> ### Revision 7 — cat B is live, and Blackhole answered three open questions
>
> The plan's items 1–4 have been implemented and **verified on Blackhole silicon** (p150b). What
> changed here:
>
> * **Cat B is no longer switched off.** Five ops (`Identity`, `Abs`, `Exp`, `Sin`, `Cos`) now inject
>   `±inf` / `NaN` / signed zeros and are green; five more are measured and deferred (§2.1). §4's
>   tables carry a cat-B column per op.
> * **Ternary operand-C poles are driven.** `addcdiv` and `snake_beta` at `c → 0` and `lerp`'s weight
>   boundaries — previously unreachable by construction — §4.6.
> * **Three open questions closed by measurement**, all in §5: the `SFPMAD` signed-zero prediction
>   (confirmed, now arch-gated), whether `-0.0` reaches DEST (it does not, on the datacopy path), and
>   whether approximate `exp`'s accuracy limit is generational (it is — Wormhole only).
> * **One new kernel finding:** `Log` saturates a non-finite input to the format maximum, so no
>   non-finite value survives it (§5.5).
>
> ### Revision 6 — the tables are now generated, not overlaid
>
> Revisions 2–5 layered "override" notes on top of a body written on 2026-07-23, so reading any row
> meant reconciling it against up to four later sections. That is gone. **§4 is regenerated directly
> from the code** — every op, its registered domain, its actual probe values, and which tests drive
> it — and the stale body it replaces has been deleted rather than annotated. §2 is the list of what
> is still untested, which is what the previous revisions made hardest to extract.
>
> The four phase summaries and the PR-3 plan that used to sit alongside this file are **deleted**:
> their content was verified present in the code (counts, tables, guards, gates all match), so the
> code is now the record. §7 says how to re-derive everything here.

---

## How to read this document

For every SFPU op we record whether the test infra **deliberately drives and asserts** each class of
edge, using the six categories the work was organised around:

| Cat | Edge class | Mechanism that closes it |
|---|---|---|
| **A** | Domain boundaries — poles and branch cuts (`1/0`, `log 0`, `asin(±1)`, `acosh 1`) | `_OP_SINGULARITIES` → `boundary_probes()` → `edge_spec()` |
| **B** | IEEE specials — `±inf`, `NaN`, `+0.0`, `-0.0` | `FLOAT_SPECIALS`, gated by `specials_safe()` **and** `SPECIALS_READY_OPS` |
| **C** | Integer extremes — `INT32_MIN/MAX`, `UINT32_MAX`, `0`, `-1` | `integer_specials()` delivered as a raw `src_A_override` |
| **D** | Op-specific discrete edges — knees, thresholds, exact rounding ties | `_OP_EDGE_POINTS` |
| **E** | Shift-amount limits for the **unary** shift ops | blocked on a C++ template parameter |
| **F** | Kernels with no `MathOperation` entry at all | a new harness per kernel |

Symbols in §4: **✅** the edge sweep drives this op; **⬜** it does not (with the reason);
**⚠️** the op diverges from its golden at a driven edge (§5).

In the **Cat B** column specifically, **🟡 §5.8** and **🟡 §5.9** are not "unknown" — they name the
single kernel behaviour holding that op out, which is why the 37 unenrolled ops are a shorter list than
they look. The two markers now mean different things:

- **🟡 §5.9** — waiting on §5.6's approximation-contract question. 23 ops, blocked on an owner.
- **🟡 §5.8** — only `Sign` and `Heaviside` still carry this marker. The other seven ops that used to
  are now ✅: the ISA settles them and their goldens model the total order. These two compare through
  `SFPSETCC`, whose contract excludes a `NaN` operand.

---

## 1. Coverage at a glance

All figures re-derived from the tree on 2026-08-12 (see §7 for the commands).

| | Count |
|---|---|
| Unique `MathOperation` members | **182** |
| Ops with `SFPU_UNARY` dispatch | 118 — of which **97** have a registered domain and are swept |
| ↳ swept unary ops | 31 broad + 63 standard + 3 perf-only (`TopK*` stages) |
| ↳ with ≥1 deliberate edge value (cat A and/or D) | **50** (§4.1) |
| ↳ smooth everywhere, so cat B is their *only* edge | **47** (§4.2) |
| Unary ops **outside** the registry — in neither sweep | **21** (§4.3): 5 predicates, 3 threshold, 4 int max/min, 2 unary shift, `Typecast`, `Relu`, 4 perf-only int, `SfpuSwiGLU` |
| Binary SFPU ops (float + shift) | 43 — 11 with a registered domain, 5 with a driven pole |
| Binary integer / ternary / scalar / reduce / FPU-binary ops | 5 / 5 / 5 / 3 / 3 |
| `_OP_SINGULARITIES` entries | **21** — +2 for the ternary operand-C poles |
| `_OP_EDGE_POINTS` entries | 43, plus `_OP_OPERAND_EDGE_POINTS` for `lerp`'s operand-C knees |
| `SPECIALS_READY_OPS` (cat B opt-in) | **67 of 97 unary**, plus all **5 scalar binops**; all 30 unary still outside wait on §5.6's two questions or on a harness — none is work this suite can simply do |
| `(format, dest_acc)` triples that can carry specials | 7 cells of 50, **re-confirmed on Wormhole** (250 variants, 85 failing); **3 of those 7 reachable and confirmed on Blackhole**. Carrying a `-0.0` is a strictly narrower gate — see §5.2 |
| Ops diverging from their golden at a driven edge | 12 over 46 cells, of which **16 cells now arch-gated to Wormhole**. Down from 13 over 52: `Signbit`'s 6 were a stimulus limitation and are deleted, not xfailed (§5.2) |
| Host-side guards over the gates and metadata | 107 tests (`test_sfpu_domains.py`) |

**Category status:** A ✅ closed for every op that has a boundary, unary **and ternary** · B 🟡 live
for 67 of the 97 unary ops plus all 5 scalar binops; the 30 still outside all wait on §5.6's two
questions or on a harness · C ✅ closed for the 5 ops whose kernels claim the
full int32 range · D ✅ closed for all 43 knees, plus `lerp`'s weight boundaries · E ✅ closed, unary
and binary · F ⬜ 11 kernels untouched.

**So five of the six categories are closed or bounded**, and what is left is one large build (F) and
two questions someone else has to answer.

**Blackhole status (p300a, 2026-08-13).** All four suites pass, through the two-phase
compile-producer / compile-consumer flow that CI uses:

| Suite | Result |
|---|---|
| `test_sfpu_unary.py` | 5027 passed · 1601 skipped · 18 xfailed · **0 xpassed** · 0 failed |
| `test_sfpu_binary.py` | 739 passed · 531 skipped · 36 xfailed · **0 xpassed** · 0 failed |
| `test_sfpu_ternary.py` | 39 passed · 25 skipped · 0 failed |
| `test_sfpu_binop_scalar.py` | 58 passed · 62 skipped · 0 failed |

**Wormhole status (n300, 2026-08-13)** — the first run of these suites on Wormhole, same flow. Not green,
and the three differences are findings rather than noise:

| Suite | Result |
|---|---|
| `test_sfpu_unary.py` | 6034 passed · 533 skipped · **49 failed** (§5.10) · 30 xfailed · **6 xpassed** (§5.11) |
| `test_sfpu_binary.py` | 865 passed · 392 skipped · 33 xfailed · **16 xpassed** (§5.12) · 0 failed |
| `test_sfpu_ternary.py` | 39 passed · 25 skipped · 0 failed |
| `test_sfpu_binop_scalar.py` | 67 passed · 72 skipped · **1 failed** — `ScalarRsub`, and it is §5.10's cause in a second suite |

Wormhole skips 533 unary variants where Blackhole skips 1600, because `_skip_bh_unless_fp32` collapses the
whole `dest_acc=No` row there — so Wormhole exercises more of the format matrix, which is part of why this
run found anything. Full record: [WORMHOLE_MEASUREMENT_RESULTS.md](WORMHOLE_MEASUREMENT_RESULTS.md).

The binary suite's **0 xpassed is the point**: before the signed-zero class was arch-gated it reported
16, and those 16 cells are now 16 ordinary passes — i.e. assertions rather than tolerated divergences.
The same happened to approximate `exp` in the unary suite (4 XPASS → 4 pass). Everything below marked
"measured" without an arch named is Wormhole, carried forward.

---

## 2. What is still NOT tested

Ordered by how much coverage each item is worth. This is the list to work from; §1 of the plan
sequences it.

### 2.1 Cat B — IEEE specials for the other 30 unary ops

**No longer zero.** Nine ops — `Identity`, `Abs`, `Exp`, `Sin`, `Cos`, and now `Neg`, `Reciprocal`,
`Sqrt` and `Rsqrt` — inject `±inf`, `NaN` and signed zeros through `SPECIALS_READY_OPS` and are green
on Blackhole across every specials-safe triple the sweep reaches. The mechanism is proven end to end
on silicon; what remains is per-op work.

**The second tranche is in, and it was one framework defect wearing four disguises.** Every divergence
that had been booked against those goldens — `Neg(NaN) → +inf` chief among them — traced to torch's
fp32 → bfloat16 cast, which canonicalises every NaN to `0xFFFF`, sign bit set, whatever sign it
started with. That is why the defect appeared only at `dest_acc=No`: it takes a 16-bit Dest for the
pack path to substitute a *signed* infinity and make the invented sign visible. `Neg` is simply the
one op whose NaN is genuinely negative, so it was the op the artefact disagreed with. See §5.7.

Three kernel divergences survived the fix and are now recorded as non-strict xfails, derived from the
delivery rules rather than listed (`_cat_b_divergences` in `test_sfpu_unary.py`):

| Op | Probe | Golden | Hardware | Scope |
|---|---|---|---|---|
| `Reciprocal` | `NaN` | `NaN` | `+0` | every combination that delivers a NaN — 6 |
| `Sqrt` | `-0` | `-0` | `NaN` | unpack-to-dest only — 2 |
| `Rsqrt` | `-0` | `-inf` | `NaN` | unpack-to-dest only — 2 |

`Log` is the only op left out of the tranche, and not for a golden reason: it saturates its input
(§5.5), which no ISA text prescribes, so §5.6's question has to be answered before it can be enrolled.

**37 unary ops remain outside**, and 31 of them are held there by the two kernel behaviours in §5.8 and
§5.9 rather than by anything op-specific — so the remaining cat-B work is two answers, not 31
investigations. The other 6 are the three `TopK*` stages with no golden entry (§2.5), `ReluMin`
(skipped on tt-llk#1120), `RsqrtCompat` (already fully xfailed), and `I1`.

`I1` is worth reading closely, because it is the case that keeps the two gates honest: its golden **was**
wrong and has been fixed, but it stays out of `SPECIALS_READY_OPS` because its *kernel* saturates to
`±1.1547668e37`. Fixing a golden is not a reason to enrol an op — if it were, a kernel divergence would
be laundered into a golden that agrees with it, which is the failure mode this whole gate exists to
prevent.

47 of the 97 unary ops are smooth everywhere (§4.2), and for those cat B is their entire edge story.

The five predicates (`Isinf`, `Isposinf`, `Isneginf`, `Isnan`, `Isfinite`) still inject specials via
`test_eltwise_unary_sfpu_isinf_isnan`, which is also the instrument that measured the safe matrix in
§6.

### 2.2 ~~Ternary operand-C edges~~ ✅ closed

`OperandSpecs` gained `spec_C` (defaulting to a copy of `spec_B`, so all five consumers keep working),
`Operand` gained `C`, and `_OP_OPERAND_EDGE_POINTS` carries per-operand knees. `addcdiv` and
`snake_beta` now have a registered `Operand.C` singularity at 0 and `lerp` has its weight boundaries;
`test_sfpu_ternary_edges` drives them. 9 passed / 7 skipped on Blackhole. See §4.6.

What is **not** driven here, deliberately: the `0/0` indeterminate form at the pole. Holding the
numerator off zero makes the variant assert the pole (4064 of 4096 elements, all `±inf`) instead of
tolerating a case already recorded against `div`, `fmod`, `remainder` and `xlogy`. If it is ever worth
driving here it wants its own variant and its own xfail.

### 2.3 ~~Cat E — the unary shift amount~~ ✅ closed

`LeftShift` and `RightShift` sweep the full amount axis, so cat E has no remaining gap on either
the unary or the binary side. `SHIFT_AMOUNT` is no longer a fixed `constexpr`: the
`SFPU_SHIFT_AMOUNT` template parameter emits it as a macro and `sfpu_operations.h` selects on
`#ifdef`, so the sweep can vary it while every other unary test keeps the original 3.

Both suites now read `sfpu_domains.SHIFT_EDGE_AMOUNTS` — `{0, 1, 2, 7, 15, 16, 30, 31, 32, 33, 40,
63, 100, 1000, −1, −5, −32, −1000}` — rather than each holding a copy, so "interesting shift"
cannot come to mean two different things.

**The half that was completely untested is the out-of-range half**, which a fixed shift of 3 could
never reach. It is now asserted for the unary path — but read the next paragraph before trusting how
much it asserts.

**The two unary shifts do not share an out-of-range rule.** Revisions before 13 said here that "the
kernel defines any amount outside `[0, 31]` as producing 0", which is true for the binary shifts and
for the unary *left* shift and false for the unary *right* shift:

| | out-of-range amount |
|---|---|
| binary shifts | 0, both signs |
| `calculate_left_shift` | `out_of_range ? vInt(0) : (v << amt)` → 0 |
| `calculate_right_shift` | `eff = (amt >= 32) ? 31u` → clamps and shifts, so a **negative operand gives -1** |

The goldens now state and implement each kernel's own rule rather than one shared one.

Three constraints the fixed amount had been hiding, all recorded in the test:

- **The stimulus has to depend on the amount.** A left shift is the only one that can leave int32,
  so values are filtered per variant against two bounds — the result must fit, and must not be
  `INT32_MIN`, which Dst stores as sign-magnitude and cannot represent. At a shift of 31 only 0
  survives, so that variant skips with a reason rather than passing vacuously.
- **Positive-only stimuli**, for the reason §4.3's int sweep already gave: sign-magnitude Dst does
  not round-trip a negative the way two's complement would. Driving negatives makes every
  `RightShift` variant except a shift of 0 disagree, **in range as well as out** — which is what
  identifies it as delivery rather than arithmetic, and would otherwise have been recorded as a
  kernel divergence.
- **So the `RightShift` half of this is weaker than it looks.** The `-1` saturation is modelled but
  cannot be probed while negatives cannot be delivered, so the assertion covers only the positive
  half, where the two kernels' rules coincide at 0. Re-measure the day a negative int32 operand can
  be delivered.

### 2.4 Ops with no WH/BH correctness test at all

**Driven only from the out-of-scope Quasar suite (6 ops):**
`SfpuSwiGLU`, `SfpuElwmulInt`, `SfpuGeInt`, `SfpuGtInt`, `SfpuLeInt`, `SfpuLtInt`.
The four int comparisons and the int multiply have a golden (`_gt_int`, `_mul`, …) but nothing on
Wormhole or Blackhole calls them.

> Two ops that *look* untested are not, and the reason is worth knowing before re-auditing: an op
> can be driven under an **alias**. `MathOperation` has two — `SfpuWhere`/`TTNNWhere` and
> `LogicalNot`/`LogicalNotUnary` — and in both cases the test names the second spelling
> (`test_sfpu_ternary.py` and `test_sfpu_unary.py` respectively). A grep for the canonical name finds
> nothing. Any tooling over this audit has to resolve aliases; §7's inventory does.

**Perf-only — a perf test exists, no functional golden or assert (7 ops):**
`AddInt32`, `SubInt32`, `AbsInt32`, `BitwiseNot` (all in `perf_eltwise_unary_sfpu_int32.py`; the file
records the reason — the int32-unary functional sweep is blocked by the fast-tilize gap,
tt-llk#495), plus `TopKLocalSort`, `TopKMerge`, `TopKRebuild` (whole-op `topk` is tested, the three
stages are not).

### 2.5 Cat F — kernels with no `MathOperation` entry (11)

A header exists; nothing in the Python infra can reach them. Confirmed still absent:
`welfords`, `dropout`, `quant`, `cumsum`, `reshuffle_rows`, `int_sum`, `tiled_prod`,
`copy_dest_values`, `generalized_moe_gate_topk`, `max_pool_indices`, `rand`.

`generic_moe_gate_topk` has come **off** this list — `test_sfpu_generic_moe_gate_topk.py` and
`sources/sfpu_generic_moe_gate_topk_test.cpp` both exist.

### 2.6 Integer edges that remain out of scope *by kernel design*

Not a test gap, and worth not re-filing: 12 of the 17 int binary ops document a **narrower valid
range** than the format's, so the extremes are outside what the kernel promises.
`_INT_BINARY_STIMULI` records each one — `div`/`fmod` below 2²⁴ for an exact int→fp32 reciprocal,
`mul` below ~46340 so the product stays under 2³¹, `lcm` assuming |a|,|b| < 2¹⁵, `max`/`min`
non-negative so signed and unsigned agree. Cat C covers the 5 that do claim the full range
(`SfpuBitwiseAnd/Or/Xor`, `SfpuEqInt`, `SfpuNeInt`).

`INT32_MIN` itself is excluded everywhere and that is hardware: sign-magnitude Dst reads
`0x80000000` as "negative zero" and cannot round-trip it. It has a dedicated documenting xfail
(`test_sfpu_binary_int_shift_int32_min_unsupported`), and `INT32_MIN + 1` stands in for it.

Still genuinely open on the integer side: `gcd`/`lcm` with `0` and negatives, and `INT32_MIN` for
`abs_int32`.

### 2.7 Format / overflow extremes

No float op is driven at its format ceiling, at a denormal, or through an overflow-to-`inf`
transition. `clip_to_format()` exists to keep probes *inside* the representable range, so the
mechanism currently prevents this rather than enabling it. Untouched since the original audit.

### 2.8 Verification gaps, not coverage gaps

- **Blackhole.** The reduce xfail and the scalar presubmit/nightly split were measured on p100a; the
  three edge sweeps and `specials_safe()`'s matrix were **not**. Two parts are arch-sensitive by
  construction: the safe matrix (unpack paths differ, and it is a measurement rather than a
  derivation) and the shift xfail, whose whole purpose is the Blackhole path.
- **~~Wormhole: nothing in this suite had been run there~~ ✅ closed, and it was not free.** The suite has
  now run on a Wormhole n300: the safe matrix re-measured and confirmed, the total order confirmed, and
  **49 of 752 edge variants failed** — one 10-op family, §5.10. The lesson is the general one: an
  unexercised arch is not a documentation gap, it is an unmeasured claim. What is *still* Wormhole-unmeasured
  after this: the ternary and scalar suites' arch-specific claims beyond their headline counts, and
  `Tan(NaN) -> 0.0` on the 16-bit-Dest path (§5.10).
- **~~The accurate exp path over (16, 80].~~ ✅ closed on Wormhole, still open on Blackhole.** The registry
  carries the range bound and `_APPROX_ACCURACY_MAX` the approximation bound, applied only in
  `ApproximationMode.Yes`. The accurate path over that region has now been driven on a Wormhole n300 —
  `Exp` 132 passed, `Exp2` 138 passed, 0 failed, with the measured error **+0.00%** above 8 out to
  `x = 79.97` — so the restored `high=80` is sound there. See §5.11.
- **Whether `-0.0` reaches DEST** on the non-unpack-to-dest path — see §5.2. Three ops' xfail
  reasons depend on the answer.
- **`WITH_COVERAGE` builds** and **Bfp4_b output formats** (`Float16 -> Bfp4_b` fails 100% at
  `dest_acc=No` on Wormhole and is unexplained; every neighbouring cell is clean).
- **~~CI runs none of this~~ ✅ closed.** The broad unary profile used to run in no automated job on
  any arch — every LLK pytest job either excluded `nightly` or ran `--coverage`, under which the broad
  profile is skipped wholesale, so every gain recorded in this document was unguarded. `llk-e2e` now
  carries non-coverage companion groups (`llk_e2e_*_nocov`, `split_group` 6–10) that run the same
  tests without instrumentation. Their timeouts were copied from the instrumented groups and want one
  nightly's data to tune — plan §6.
- **Whether the coverage skip is justified at all.** The `BROAD_SWEEP_OPS` skip cited tt-llk#1435,
  which is about test *ordering*; its one mention of coverage is an observation of the skip's own
  effect. The citation was circular and is gone, but no recorded rationale replaced it — the exclusion
  is presumably cost under instrumentation, and nobody has written that down.

---

## 3. The four systemic findings from the original audit

| # | Finding | Status |
|---|---|---|
| 1 | Unary float sweep is positive-only (`uniform(0.1, 1.1)`, no `spec_A`) | ✅ **fixed.** The sweep defaults `spec_A` to the op's registered signed domain, bounded by the narrowest format in the pipeline (`for_op_pipeline` + `exclude_undefined`), and a missing registry entry is a hard `KeyError`. 31 ops gained their `x<0` branch |
| 2 | Binary / ternary / scalar suites never import `sfpu_domains.py` | 🟡 **closed for binary and for the ternary pole.** 11 of 43 binary ops have a registered domain; the other 32 keep the format default. Ternary now reaches the registry for the operand that matters — `OperandSpecs.spec_C` and `Operand.C` exist, and `addcdiv` / `snake_beta` carry a registered pole (§4.6). Still open: no ternary op has a registered *domain*, and scalar has the plumbing but nothing to read |
| 3 | IEEE specials injected for exactly one op family | 🟡 **enabled for 6 families of op, gated per op.** No longer "measured and switched off": `SPECIALS_READY_OPS` holds `Identity`, `Abs`, `Exp`, `Sin`, `Cos` alongside the five predicates, all green on Blackhole. The safe `(format, dest_acc)` surface is data (§6) pinned by 107 host-side tests. The remaining 87 ops are gated on their *goldens*, not on the pipeline — §2.1 |
| 4 | Integer sign/extreme edges structurally excluded (`_get_integer_bounds` returns `min+1`) | 🟡 **closed where the kernels allow it.** Extremes go through a raw `src_A_override`; `test_sfpu_binary_int_extremes` drives `{INT32_MIN+1, -1, 0, 1, INT32_MAX}²` over the 5 ops that claim the full range. The other 12 are out of scope by kernel design — §2.6 |

**None of #2–#4 reaches a plain "fixed", and for the same reason each time:** the mechanism was the
easy part and something outside the test infra bounds how far it can go. For #2 it is the absence of
registered ternary and scalar domains, for #3 the goldens, for #4 the kernels' own documented ranges.
#3 is the one that moved most in revision 7 — from zero ops injecting specials to six families — and
it moved by fixing goldens, which is exactly what the pattern predicted.

---

## 4. Per-op coverage

Generated from the code — see §7. Every one of the 182 `MathOperation` members appears in exactly one
table below.

`broad` = the full format matrix including block floats and both approximation modes; `standard` =
Float16_b + Float32, `ApproximationMode.No`. Cat A probe values are shown as the singular point and
the side the op is **defined** on (`abo` = above, `bel` = below, `bot` = both); the probe offset
itself is format-relative, so `Reciprocal`'s `0.0` becomes `±0.015625` in Float16_b and `±0.25` in
Bfp4_b, and with `dest_acc=No` a 32-bit probe is stepped by a bfloat16 ULP so the 16-bit DEST cannot
truncate it back onto the boundary.

The three unary tables partition on **why** an op is where it is: §4.1 has a boundary or a knee and is
driven; §4.2 has neither, so the edge sweep skips it and cat B is all that is left; §4.3 has no
registered domain at all, so neither sweep reaches it and coverage depends on a dedicated test.

#### 4.1 Unary ops with a deliberate edge driven — cat A and/or cat D (50 ops)

| Op | Kernel | Random sweep | Cat A boundary (side defined on) | Cat D knees / ties | Cat B | Edge sweep | Other test | ⚠️ |
|---|---|---|---|---|---|---|---|---|
| `Acos` | `acos` | standard | -1.0 (abo); 1.0 (bel) | — | ✅ driven | ✅ | — |  |
| `Acosh` | `acosh` | broad | 1.0 (abo) | — | ✅ driven | ✅ | — |  |
| `Asin` | `asin` | standard | -1.0 (abo); 1.0 (bel) | — | ✅ driven | ✅ | — |  |
| `Atanh` | `atanh` | broad | -1.0 (abo); 1.0 (bel) | — | ✅ driven | ✅ | — |  |
| `Ceil` | `ceil` | broad | — | `-2, -1, 0, 1, 2` | ✅ driven | ✅ | — |  |
| `Celu` | `celu` | broad | — | `0, -0` | ✅ driven | ✅ | — |  |
| `Clamp` | `clamp` | standard | — | `-1, 1` | ✅ driven | ✅ | — |  |
| `Elu` | `elu` | broad | — | `0, -0` | ✅ driven | ✅ | — |  |
| `EqualZero` | `equal_zero` | standard | — | `0, -0` | ✅ driven | ✅ | — |  |
| `Erfinv` | `erfinv` | standard | -1.0 (abo); 1.0 (bel) | — | 🟡 §5.9 | ✅ | — | ⚠️ |
| `Floor` | `floor` | broad | — | `-2, -1, 0, 1, 2` | ✅ driven | ✅ | — |  |
| `Frac` | `frac` | broad | — | `-1.5, -1, 1, 1.5` | 🟡 §5.9 | ✅ | — |  |
| `GreaterThanEqualZero` | `greater_than_equal_zero` | standard | — | `0, -0` | ✅ driven | ✅ | — |  |
| `GreaterThanZero` | `greater_than_zero` | standard | — | `0, -0` | ✅ driven | ✅ | — |  |
| `Hardmish` | `hardmish` | standard | — | `-2, 0` | ✅ driven | ✅ | — |  |
| `Hardshrink` | `hardshrink` | standard | — | `-0.5, 0.5` | ✅ driven | ✅ | — |  |
| `Hardsigmoid` | `hardsigmoid` | broad | — | `-3, 3` | ✅ driven | ✅ | — |  |
| `Hardtanh` | `hardtanh` | standard | — | `-1, 1` | ✅ driven | ✅ | — |  |
| `Heaviside` | `heaviside` | standard | — | `0, -0` | 🟡 §5.8 | ✅ | — | ⚠️ |
| `LessThanEqualZero` | `less_than_equal_zero` | standard | — | `0, -0` | ✅ driven | ✅ | — |  |
| `LessThanZero` | `less_than_zero` | standard | — | `0, -0` | ✅ driven | ✅ | — |  |
| `Log` | `log` | broad | 0.0 (abo) | — | 🟡 §5.9 | ✅ | — |  |
| `Log1p` | `log1p` | broad | -1.0 (abo) | — | ✅ driven | ✅ | — |  |
| `LogWithBase` | `log_with_base` | standard | 0.0 (abo) | — | 🟡 §5.9 | ✅ | — |  |
| `Lrelu` | `lrelu` | standard | — | `0, -0` | ✅ driven | ✅ | — |  |
| `NotEqualZero` | `not_equal_zero` | standard | — | `0, -0` | ✅ driven | ✅ | — |  |
| `Prelu` | `prelu` | standard | — | `0, -0` | ✅ driven | ✅ | — |  |
| `Rdiv` | `rdiv` | standard | 0.0 (bot) | — | 🟡 §5.9 | ✅ | — |  |
| `Reciprocal` | `reciprocal` | broad | 0.0 (bot) | — | ✅ driven | ✅ | — | ⚠️ `1/NaN` xfail |
| `ReluMax` | `relu_max` | broad | — | `0, 5` | ✅ driven | ✅ | — |  |
| `ReluMin` | `relu_min` | broad | — | `5` | ⬜ | ✅ | — |  |
| `Round` | `round` | standard | — | `-2.5, -1.5, -0.5, 0.5, 1.5, 2.5` | ✅ driven | ✅ | — |  |
| `Rsqrt` | `rsqrt` | broad | 0.0 (abo) | — | ✅ driven | ✅ | — | ⚠️ `rsqrt(-0)` xfail |
| `RsqrtCompat` | `rsqrt_compat` | standard | 0.0 (abo) | — | ⬜ | ✅ | — | ⚠️ |
| `Selu` | `selu` | standard | — | `0, -0` | ✅ driven | ✅ | — |  |
| `Sign` | `sign` | standard | — | `0, -0` | 🟡 §5.8 | ✅ | — | ⚠️ |
| `Signbit` | `signbit` | standard | — | `0, -0` | ✅ driven | ✅ | — | ⚠️ |
| `Softplus` | `softplus` | standard | — | `20` | ✅ driven | ✅ | — |  |
| `Softshrink` | `softshrink` | standard | — | `-0.5, 0.5` | ✅ driven | ✅ | — |  |
| `Sqrt` | `sqrt` | broad | 0.0 (abo) | — | ✅ driven | ✅ | — | ⚠️ `sqrt(-0)` xfail |
| `SqrtCustom` | `sqrt_custom` | standard | 0.0 (abo) | — | 🟡 §5.9 | ✅ | — |  |
| `Threshold` | `threshold` | broad | — | `5` | ✅ driven | ✅ | — |  |
| `Trunc` | `trunc` | broad | — | `-1, 0, 1` | ✅ driven | ✅ | — |  |
| `UnaryGe` | `unary_ge` | standard | — | `0.5` | ✅ driven | ✅ | — |  |
| `UnaryGt` | `unary_gt` | standard | — | `0.5` | ✅ driven | ✅ | — |  |
| `UnaryLe` | `unary_le` | standard | — | `0.5` | ✅ driven | ✅ | — |  |
| `UnaryLt` | `unary_lt` | standard | — | `0.5` | ✅ driven | ✅ | — |  |
| `UnaryMax` | `unary_max` | standard | — | `0, -0` | ✅ driven | ✅ | — |  |
| `UnaryMin` | `unary_min` | standard | — | `0, -0` | ✅ driven | ✅ | — |  |
| `Xielu` | `xielu` | standard | — | `0, -0` | ✅ driven | ✅ | — |  |

#### 4.2 Unary ops smooth everywhere — cat B is their **entire** edge story (47 ops)

**27 of the 47 now run**, which is the largest single change in this revision: an op here has no knee
and no pole, so before cat B reached it the edge sweep skipped it outright and its only coverage was
the random sweep. The remaining 20 are all 🟡 — held by §5.8 or §5.9, not by anything op-specific.

| Op | Kernel | Random sweep | Registered domain | Cat B | Edge sweep | Other test |
|---|---|---|---|---|---|---|
| `Abs` | `abs` | broad | yes | ✅ driven | ✅ cat B only | — |
| `Add1` | `add1` | standard | yes | ✅ driven | ✅ cat B only | — |
| `Asinh` | `asinh` | broad | yes | ✅ driven | ✅ cat B only | — |
| `Atan` | `atan` | standard | yes | ✅ driven | ✅ cat B only | — |
| `CastFp32ToFp16a` | `cast_fp32_to_fp16a` | standard | yes | 🟡 §5.9 | ⬜ skips | — |
| `Cbrt` | `cbrt` | standard | yes | ✅ driven | ✅ cat B only | — |
| `Cos` | `cosine` | broad | yes | ✅ driven | ✅ cat B only | — |
| `Cosh` | `cosh` | standard | yes | ✅ driven | ✅ cat B only | — |
| `Digamma` | `digamma` | standard | yes | 🟡 §5.9 | ⬜ skips | — |
| `Erf` | `erf` | standard | yes | 🟡 §5.9 | ⬜ skips | — |
| `Erfc` | `erfc` | standard | yes | 🟡 §5.9 | ⬜ skips | — |
| `Exp` | `exponential` | broad | yes | ✅ driven | ✅ cat B only | — |
| `Exp2` | `exp2` | broad | yes | ✅ driven | ✅ cat B only | — |
| `ExpWithBase` | `exp_with_base` | standard | yes | ✅ driven | ✅ cat B only | — |
| `Expm1` | `expm1` | standard | yes | ✅ driven | ✅ cat B only | — |
| `Expm1Cw` | `expm1_cw` | standard | yes | 🟡 §5.9 | ⬜ skips | — |
| `Fill` | `fill` | broad | yes | ✅ driven | ✅ cat B only | — |
| `Fmod` | `fmod` | standard | yes | ✅ driven | ✅ cat B only | — |
| `Gelu` | `gelu` | broad | yes | 🟡 §5.9 | ⬜ skips | — |
| `GeluAppx` | `gelu_appx` | standard | yes | ✅ driven | ✅ cat B only | — |
| `GeluDerivative` | `gelu_derivative` | standard | yes | 🟡 §5.9 | ⬜ skips | — |
| `GeluTanh` | `gelu_tanh` | broad | yes | ✅ driven | ✅ cat B only | — |
| `I0` | `i0` | standard | yes | ✅ driven | ✅ cat B only | — |
| `I1` | `i1` | standard | yes | 🟡 §5.9 | ⬜ skips | — |
| `Identity` | `identity` | standard | yes | ✅ driven | ✅ cat B only | — |
| `Lgamma` | `lgamma` | standard | yes | 🟡 §5.9 | ⬜ skips | — |
| `Mish` | `mish` | standard | yes | ✅ driven | ✅ cat B only | — |
| `Neg` | `negative` | broad | yes | ✅ driven | ✅ cat B only | — |
| `Polygamma` | `polygamma` | standard | yes | 🟡 §5.9 | ⬜ skips | — |
| `Remainder` | `remainder` | standard | yes | ✅ driven | ✅ cat B only | — |
| `Rpow` | `rpow` | standard | yes | 🟡 §5.9 | ⬜ skips | — |
| `Sigmoid` | `sigmoid` | standard | yes | 🟡 §5.9 | ⬜ skips | — |
| `SigmoidAppx` | `sigmoid_appx` | standard | yes | 🟡 §5.9 | ⬜ skips | — |
| `Silu` | `silu` | broad | yes | ✅ driven | ✅ cat B only | — |
| `Sin` | `sine` | broad | yes | ✅ driven | ✅ cat B only | — |
| `Sinh` | `sinh` | standard | yes | ✅ driven | ✅ cat B only | — |
| `Softsign` | `softsign` | standard | yes | ✅ driven | ✅ cat B only | — |
| `Square` | `square` | broad | yes | ✅ driven | ✅ cat B only | — |
| `Tan` | `tan` | standard | yes | ✅ driven | ✅ cat B only | — |
| `Tanh` | `tanh` | broad | yes | 🟡 §5.9 | ⬜ skips | — |
| `TanhDerivative` | `tanh_derivative` | standard | yes | 🟡 §5.9 | ⬜ skips | — |
| `TanhDerivativeLut` | `tanh_derivative_lut` | standard | yes | 🟡 §5.9 | ⬜ skips | — |
| `Tanhshrink` | `tanhshrink` | broad | yes | ✅ driven | ✅ cat B only | — |
| `TopKLocalSort` | `topk_local_sort` | **perf-only** | yes | ⬜ | ⬜ skips | — |
| `TopKMerge` | `topk_merge` | **perf-only** | yes | ⬜ | ⬜ skips | — |
| `TopKRebuild` | `topk_rebuild` | **perf-only** | yes | ⬜ | ⬜ skips | — |
| `UnaryPower` | `power` | standard | yes | 🟡 §5.9 | ⬜ skips | — |

#### 4.3 Unary ops outside `_OP_DOMAIN_REGISTRY` — not in either sweep (21 ops)

These have no registered domain, so `sfpu_unary_ops()` excludes them from the broad/standard
sweeps **and** from the edge sweep. Each is either covered by a dedicated test, deliberately
unreachable, or genuinely uncovered — the last column says which.

| Op | Kernel | Cat D knees | Dedicated test | Status |
|---|---|---|---|---|
| `AbsInt32` | `abs_int32` | — | **none (WH/BH)** | ⬜ **perf-only** (tt-llk#495) — §2.4 |
| `AddInt32` | `add_int32` | — | **none (WH/BH)** | ⬜ **perf-only** (tt-llk#495) — §2.4 |
| `BitwiseNot` | `bitwise_not` | — | **none (WH/BH)** | ⬜ **perf-only** — §2.4 |
| `Isfinite` | `isfinite` | — | `unary` | ✅ cat B — as `Isinf` |
| `Isinf` | `isinf` | — | `unary` | ✅ cat B — with its four siblings, the only ops injecting `±inf`/`NaN` today |
| `Isnan` | `isnan` | — | `unary` | ✅ cat B — as `Isinf` |
| `Isneginf` | `isneginf` | — | `unary` | ✅ cat B — as `Isinf` |
| `Isposinf` | `isposinf` | — | `unary` | ✅ cat B — as `Isinf` |
| `LeftShift` | `left_shift` | — | `unary` | ✅ cat E — full shift axis via `SFPU_SHIFT_AMOUNT`, in range and out (§2.3) |
| `LogicalNot` | `logical_not_unary` | `0, -0` | `unary` | ✅ cat D — exact threshold forced by `test_eltwise_unary_sfpu_threshold`, which names it `LogicalNotUnary` |
| `Relu` | `relu` | — | `plot` | ➖ unreachable by design — applied by the packer (`STACC_RELU`), not a `SfpuType`. The only reference is a plotting script |
| `RightShift` | `right_shift` | — | `unary` | ✅ cat E — as `LeftShift` |
| `SfpuSwiGLU` | `swiglu` | — | **none (WH/BH)** | ⬜ **Quasar-only** — §2.4 |
| `SubInt32` | `sub_int32` | — | **none (WH/BH)** | ⬜ **perf-only** (tt-llk#495) — §2.4 |
| `Typecast` | `typecast` | — | `eltwise_unary_typecast` | 🟡 value coverage only; no special or format-extreme injection |
| `UnaryEq` | `unary_eq` | `0.5` | `unary` | ✅ cat D — as `LogicalNot` |
| `UnaryMaxInt32` | `unary_max_int32` | `1000` | `unary` | ✅ cat D — comparison tie driven by `test_eltwise_unary_sfpu_int` |
| `UnaryMaxUint32` | `unary_max_uint32` | `1000` | `unary` | ✅ cat D — as `UnaryMaxInt32` |
| `UnaryMinInt32` | `unary_min_int32` | `1000` | `unary` | ✅ cat D — as `UnaryMaxInt32` |
| `UnaryMinUint32` | `unary_min_uint32` | `1000` | `unary` | ✅ cat D — as `UnaryMaxInt32` |
| `UnaryNe` | `unary_ne` | `0.5` | `unary` | ✅ cat D — as `LogicalNot` |

#### 4.4 Binary (float + shift) SFPU ops (43 ops)

| Op | Kernel | Registered domain | Cat A pole | Edge sweep | Driven by |
|---|---|---|---|---|---|
| `SfpuAddTopRow` | `ADD_TOP_ROW` | yes | — | ⬜ | `binary` |
| `SfpuAtan2` | `ATAN2` | no (format default) | — | ⬜ | `binary` |
| `SfpuBinaryFmod` | `FMOD` | no (format default) | B=0.0 (bot) | ✅ | `binary` |
| `SfpuBinaryMax` | `MAX` | no (format default) | — | ⬜ | `binary` |
| `SfpuBinaryMin` | `MIN` | no (format default) | — | ⬜ | `binary` |
| `SfpuBinaryRemainder` | `REMAINDER` | no (format default) | B=0.0 (bot) | ✅ | `binary` |
| `SfpuBitwiseAnd` | `BITWISE_AND` | no (format default) | — | ⬜ | `binary` |
| `SfpuBitwiseOr` | `BITWISE_OR` | no (format default) | — | ⬜ | `binary` |
| `SfpuBitwiseXor` | `BITWISE_XOR` | no (format default) | — | ⬜ | `binary` |
| `SfpuDivInt32` | `DIV_INT32` | no (format default) | — | ⬜ | `binary` |
| `SfpuDivInt32Floor` | `DIV_INT32_FLOOR` | no (format default) | — | ⬜ | `binary` |
| `SfpuElwEq` | `EQ` | no (format default) | — | ⬜ | `binary` |
| `SfpuElwGe` | `GE` | no (format default) | — | ⬜ | `binary` |
| `SfpuElwGt` | `GT` | no (format default) | — | ⬜ | `binary` |
| `SfpuElwLe` | `LE` | no (format default) | — | ⬜ | `binary` |
| `SfpuElwLeftShift` | `LSHFT` | yes | — | ⬜ | `binary` |
| `SfpuElwLogicalRightShift` | `LOGICAL_RSHFT` | yes | — | ⬜ | `binary` |
| `SfpuElwLt` | `LT` | no (format default) | — | ⬜ | `binary` |
| `SfpuElwNe` | `NE` | no (format default) | — | ⬜ | `binary` |
| `SfpuElwRightShift` | `RSHFT` | yes | — | ⬜ | `binary` |
| `SfpuElwadd` | `ADD` | yes | — | ⬜ | `binary` |
| `SfpuElwdiv` | `DIV` | yes | B=0.0 (bot) | ✅ | `binary` |
| `SfpuElwmul` | `MUL` | yes | — | ⬜ | `binary` |
| `SfpuElwpow` | `POW` | yes | A=0.0 (abo) | ✅ | `binary`, `zz_measure_tol` |
| `SfpuElwrsub` | `RSUB` | yes | — | ⬜ | `binary` |
| `SfpuElwsub` | `SUB` | yes | — | ⬜ | `binary` |
| `SfpuEqInt` | `EQ_INT` | no (format default) | — | ⬜ | `binary` |
| `SfpuFmodInt32` | `FMOD_INT32` | no (format default) | — | ⬜ | `binary` |
| `SfpuGcd` | `GCD` | no (format default) | — | ⬜ | `binary` |
| `SfpuIsclose` | `ISCLOSE` | no (format default) | — | ⬜ | `binary` |
| `SfpuLcm` | `LCM` | no (format default) | — | ⬜ | `binary` |
| `SfpuLogsigmoid` | `LOGSIGMOID` | no (format default) | — | ⬜ | `binary` |
| `SfpuMask` | `MASK` | no (format default) | — | ⬜ | `binary` |
| `SfpuMaxInt32` | `MAX_INT32` | no (format default) | — | ⬜ | `binary` |
| `SfpuMaxUint32` | `MAX_UINT32` | no (format default) | — | ⬜ | `binary` |
| `SfpuMinInt32` | `MIN_INT32` | no (format default) | — | ⬜ | `binary` |
| `SfpuMinUint32` | `MIN_UINT32` | no (format default) | — | ⬜ | `binary` |
| `SfpuMulInt32` | `MUL_INT32` | no (format default) | — | ⬜ | `binary` |
| `SfpuNeInt` | `NE_INT` | no (format default) | — | ⬜ | `binary` |
| `SfpuRemainderInt32` | `REMAINDER_INT32` | no (format default) | — | ⬜ | `binary` |
| `SfpuRemainderUint32` | `REMAINDER_UINT32` | no (format default) | — | ⬜ | `binary` |
| `SfpuRsubInt32` | `RSUB_INT32` | no (format default) | — | ⬜ | `binary` |
| `SfpuXlogy` | `XLOGY` | yes | B=0.0 (abo) | ✅ | `binary`, `zz_measure_tol` |

#### 4.5 Binary integer SFPU ops (5 ops)

| Op | Kernel | Registered domain | Cat A pole | Edge sweep | Driven by |
|---|---|---|---|---|---|
| `SfpuElwmulInt` | `MUL` | no (format default) | — | ⬜ | **none (WH/BH)** |
| `SfpuGeInt` | `GE_INT` | no (format default) | — | ⬜ | **none (WH/BH)** |
| `SfpuGtInt` | `GT_INT` | no (format default) | — | ⬜ | **none (WH/BH)** |
| `SfpuLeInt` | `LE_INT` | no (format default) | — | ⬜ | **none (WH/BH)** |
| `SfpuLtInt` | `LT_INT` | no (format default) | — | ⬜ | **none (WH/BH)** |

#### 4.6 Ternary ops (5 ops)

| Op | Kernel | Registered domain | Cat A pole | Edge sweep | Driven by |
|---|---|---|---|---|---|
| `SfpuAddcdiv` | `addcdiv` | no (format default) | C=0.0 (bot) | ✅ | `ternary` |
| `SfpuAddcmul` | `addcmul` | no (format default) | — | ⬜ | `ternary` |
| `SfpuLerp` | `lerp` | no (format default) | — | ✅ | `ternary` |
| `SfpuSnakeBeta` | `snake_beta` | no (format default) | C=0.0 (bot) | ✅ | `ternary` |
| `SfpuWhere` | `where` | no (format default) | — | ⬜ | `ternary` |

Operand-C edges, driven by `test_sfpu_ternary_edges` (9 passed / 7 skipped on Blackhole):

| Op | Formula | Operand-C probe | Source |
|---|---|---|---|
| `SfpuAddcdiv` | `a + value * b / c` | `-0.015625, 0.0, 0.015625` | `_OP_SINGULARITIES` C = (0.0, BOTH) |
| `SfpuSnakeBeta` | `a + sin(b*a)^2 / c` | `-0.015625, 0.0, 0.015625` | `_OP_SINGULARITIES` C = (0.0, BOTH) |
| `SfpuLerp` | `a + c * (b - a)` | `-1.0, 0.0, 1.0, 2.0` | `_OP_OPERAND_EDGE_POINTS` C |
| `SfpuAddcmul` | `a + value * b * c` | none | a multiply has no pole; `edge_spec` returns `None` |

The probe offset is format-relative, so the pole probes become `+/-0.25` in Bfp4_b. `c` is
zero for 4064 of 4096 elements (custom() zero-fills each face), so the pole is driven hard
rather than sampled. The numerator is held off zero for the two dividing ops so the variant
asserts the pole instead of the `0/0` indeterminate form -- see the coverage note in
`test_sfpu_ternary.py`.

#### 4.7 Scalar-binop ops (5 ops)

All five are `x (+|-|*|/) c` for a compile-time `c`, so they are smooth in `x`: no pole, no knee, and
`edge_spec()` returns `None` unless specials are on. **Cat B is their entire edge story**, and all five
are now enrolled — the tensor-operand edge sweep runs them where the pipeline delivers specials.

Two of the eight (format, `dest_acc`) pairs survive both gates, and they are complementary rather than
redundant: `Float32`/`dest_acc=Yes` is unpack-to-dest so a real `-0.0` arrives, and
`Float16_b`/`dest_acc=No` is the datacopy path where it does not. The other six are excluded by
`_skip_unsupported` (Float32 needs a 32-bit Dest, Float16_b cannot use one).

| Op | Kernel | Registered domain | Cat A pole | Cat B | Edge sweep | Driven by |
|---|---|---|---|---|---|---|
| `ScalarAdd` | `ADD` | no (format default) | — | ✅ driven | ✅ cat B only | `binop_scalar` |
| `ScalarDiv` | `DIV` | no (format default) | — | ✅ driven | ✅ cat B only | `binop_scalar` |
| `ScalarMul` | `MUL` | no (format default) | — | ✅ driven | ✅ cat B only | `binop_scalar` |
| `ScalarRsub` | `RSUB` | no (format default) | — | ✅ driven | ✅ cat B only | `binop_scalar` |
| `ScalarSub` | `SUB` | no (format default) | — | ✅ driven | ✅ cat B only | `binop_scalar` |

`ScalarDiv` has no reachable divide-by-zero: the host inverts the divisor at compile time and the
kernel only multiplies, so `d` never reaches the device. That is a property of the dispatch, not an
untested edge.

Still out of scope, both needing a per-op tolerance first (the default bf16 tolerance is only
meaningful while the result stays in range): `|scalar| > 8`, and `±tiny` / `±large` on the tensor
operand.

#### 4.8 Reduce ops (3 ops)

| Op | Kernel | Registered domain | Cat A pole | Edge sweep | Driven by |
|---|---|---|---|---|---|
| `ReduceColumn` | `REDUCE_COL` | yes | — | ⬜ | `reduce`, `reduce`, `reduce_sdpa` |
| `ReduceRow` | `REDUCE_ROW` | yes | — | ⬜ | `reduce`, `reduce` |
| `ReduceScalar` | `REDUCE_SCALAR` | yes | — | ⬜ | `reduce` |

#### 4.9 FPU binary (eltwise) ops (3 ops)

| Op | Kernel | Registered domain | Cat A pole | Edge sweep | Driven by |
|---|---|---|---|---|---|
| `Elwadd` | `ELWADD` | yes | — | ⬜ | `deepseek_moe_gate`, `eltwise_binary`, `generalized_moe_gate` +4 |
| `Elwmul` | `ELWMUL` | yes | — | ⬜ | `deepseek_moe_gate`, `eltwise_bcast_col_custom`, `eltwise_binary` +3 |
| `Elwsub` | `ELWSUB` | yes | — | ⬜ | `deepseek_moe_gate`, `eltwise_bcast_col_custom`, `eltwise_binary` +5 |

---

## 5. What driving the edges found

Ten ops over 42 `(op, format, dest_acc)` cells disagree with their golden at the newly driven points
— 5 unary ops over 20 cells and 5 binary over 22. All are recorded as **non-strict xfails**, so the
case still executes and reports XPASS if the behaviour changes. Every one is cross-checked against
[tt-isa-documentation](https://github.com/tenstorrent/tt-isa-documentation), which splits them
cleanly. **This split is the practically important part:** half of these are specified hardware
behaviour and chasing them would be wasted effort.

### 5.0 Blackhole settled three of these by measurement

Recorded first because it changes how the rest of §5 should be read. Measured on a p150b:

| Question | Answer | Consequence |
|---|---|---|
| Does the `SFPMAD` signed-zero group XPASS on Blackhole, as its ISA page predicts? | **Yes — all 16 cells**, and nothing else XPASSed | The `negative_zero_golden` class is now **arch-gated to Wormhole**, so Blackhole *asserts* the sign of a zero result |
| Does `-0.0` actually reach DEST on the datacopy path? | **No** | §5.2's inference is confirmed by three unrelated ops; `Signbit`'s six xfails have since been **deleted**, the probe being gated out instead |
| Is approximate `exp`'s 5% rtol overshoot generational? | **Yes — Wormhole only.** Both reachable combinations XPASSed on Blackhole | `_APPROX_EXP_ACCURACY_XFAIL` is now arch-gated too, so Blackhole asserts approximate exp's accuracy |

All three were *predictions on record* that the non-strict-xfail convention existed to settle, and all
three resolved in favour of the prediction. That is the convention paying for itself: a skip would have
left every one of them unanswerable.

**Two of those three have since been undercut by running the same suites on Wormhole**, and rows 1 and 3
should now be read together with §5.12 and §5.11. Both gates XPASS on Wormhole as well, which no
"Blackhole is better here" reading survives: an XPASS on *both* arches means the xfail is not measuring
what it was thought to measure. The convention still paid for itself — the second XPASS is only visible
*because* these are non-strict xfails rather than skips — but the conclusion drawn from the first one was
half of the picture, taken on one arch.

### 5.1 Documented — the ISA is the authority, not a bug list

**The sign of a zero *result* is lost on Wormhole, by specification.** `div(0, -x)`,
`fmod`/`remainder` with a negative divisor, and `xlogy(0, tiny)` all return `+0.0` where IEEE gives
`-0.0`. All are built on `SFPMAD`:

> Wormhole — "If the output (before rounding) is denormal or negative zero, it'll be flushed to
> **positive** zero." · Blackhole — "…flushed to **sign-preserved** zero."

Blackhole's `SFPMAD` page lists *"improved edge-case handling of NaNs and of negative zero"* among
its upgrades. So this is a documented Wormhole limitation that Blackhole is documented to fix, and
these xfails are a **testable prediction** there: they should XPASS. If they do not, the
documentation and the hardware disagree.

**`sign(-0.0)` and `heaviside(-0.0)` sit outside the contract of the primitive they use.** `SFPSETCC`
is specified only *"provided that `VC` is neither negative zero nor any kind of NaN"* — identically on
both arches, so unlike the `SFPMAD` group this is **not** generational. Confirmed: these still xfail on
Blackhole while the `SFPMAD` group XPASSed there.

### 5.2 The signed-zero group is a *delivery* question, not three separate findings

The three signed-zero ops partition **exactly** on `unpack_to_dest`, which the driver sets to
`(input.is_32_bit() and dest_acc == Yes)` — the only path where the datum skips SrcA and the
datacopy:

| Op | Diverges on | `unpack_to_dest` there |
|---|---|---|
| `Signbit` | 6 of 8 combinations | **False** on all 6 |
| `Sign` | 2 of 8 | **True** on both |
| `Heaviside` | the same 2 | **True** on both |

`Signbit`'s set is the exact complement of `Sign`'s, and `Sign`'s and `Heaviside`'s sets are
identical. One cause explains all three: **`-0.0` only reaches the LREG on the unpack-to-dest path.**
Neither `calculate_sign` nor `calculate_heaviside` guards `|v| != 0` on its `v_if(v < 0.0F)`, so a
real `-0.0` would make them diverge on all 8; passing on 6 says the LREG holds `+0.0` there. `Signbit`
reads the sign bit directly, so it returns 0 on those 6 — and correctly returns 1 on the 2 where the
datum arrives intact. A genuinely broken sign-bit read would fail on all 8.

Consequences, both recorded in the suite's reason strings:

- **`Signbit`'s 6 entries could never XPASS, and are now gone.** They recorded a *stimulus*
  limitation, not a kernel defect — no kernel change can make an input arrive. Keeping them would have
  left six standing non-strict xfails that can never fire, which masks a regression rather than
  recording one, so `negative_zero_delivered()` now gates the `-0.0` knee off the datacopy pipelines
  and the entries were deleted. A collection-time assertion pins that they stay deleted. The earlier
  reading of this as "a kernel-contract bug" was wrong.
- **`Sign` and `Heaviside` passing on those same 6 is vacuous** — the golden's answers for `-0.0`
  (0 and 0.5) coincide with the hardware's for `+0.0`, so the case agrees without testing what it
  names.

The partition is asserted at collection (`_assert_signed_zero_partition_valid`), because the
explanation rests on it and a reason string is prose no run checks.

**Now measured, and the inference was right.** Enabling cat B for `Reciprocal`, `Rsqrt` and `Sqrt`
probed the same question from a completely different direction — three ops with nothing to do with sign
predicates — and gave the same answer:

| Probe | `dest_acc=No` (datacopy path) | `dest_acc=Yes` (unpack-to-dest) |
|---|---|---|
| `1 / -0` | `+inf` — i.e. `-0` was seen as `+0` | — |
| `rsqrt(-0)` | `+inf` — same | `NaN` — a distinct answer, so a real `-0` arrived |
| `sqrt(-0)` | `+0` — same | `NaN` — same |

So `-0.0` genuinely is **not delivered** on the datacopy path. That measurement is now acted on
rather than just recorded: the `-0.0` knee is gated out of the pipelines that cannot deliver it, which
deleted `Signbit`'s six xfails outright and turned `Sign`'s and `Heaviside`'s six vacuous passes there
into passes that no longer claim to test a signed zero.

**That scoping is now enforced rather than advised.** `negative_zero_delivered(input_format, dest_acc)`
gates the `-0` probe out of the cat-B injection wherever the datacopy path would substitute `+0.0`,
and it is deliberately a *second* gate rather than a tightening of `specials_safe()`: several triples
carry `±inf` and `NaN` intact while flattening `-0.0`, so one predicate cannot answer both questions.
Without it, `Rsqrt` at `dest_acc=No` failed for a probe that never arrived — an xfail there would have
blamed the kernel for the stimulus, which is the exact mistake `Signbit`'s entries document.

**A related non-finding, worth recording so it is not rediscovered.** The *sign of a zero result* —
`Neg(+0) → -0`, `Reciprocal(-inf) → -0` — cannot fail a test at all. `passed_test()` judges by
`torch.isclose`, a both-NaN clause and PCC, and `-0.0 == +0.0` under all three. Those rows spent a
revision on the blocking list before anyone checked whether a failing test could exist for them.
Asserting a zero's sign needs a bitwise comparator, which is a suite-wide change, not a per-op one.

### 5.3 Still open — not explained by the ISA

| Finding | Ops |
|---|---|
| **`0/0` and `x%0` return `inf`, not `nan`.** `SFPMAD` says NaN/±Inf inputs follow "the usual IEEE754 rules", which makes `0 × inf` a NaN — so this is the kernels' own reciprocal composition, not the multiply. Specifically the indeterminate form: the finite poles agree exactly and every ±inf lines up | `div`, `fmod`, `remainder`, `xlogy` |
| **`0**0` returns 0** where C, torch and the golden give 1. `pow` evaluates `exp(b·ln a)`, so a composition artifact | `pow` |
| **`RsqrtCompat(0)` saturates to `1.7014118e38`** (`0x7F000000`) instead of `inf`, on all 8 combinations — while plain `Rsqrt` over the same probe does **not** diverge. Two implementations of one function disagreeing at their shared pole, with nothing in the ISA prescribing either answer | `RsqrtCompat` |
| **`Erfinv(±1)` saturates** rather than returning ±inf, on the fp32-dest combinations only — tolerance-shaped rather than semantic | `erfinv` |

### 5.5 New: `Log` saturates its input, so no non-finite value survives it

Found by enabling cat B. On Blackhole, with a Float32 input:

| Probe | Golden | Hardware |
|---|---|---|
| `+inf` | `+inf` | **88.5** |
| `-inf` | `NaN` | **84.3** |
| `NaN` | `NaN` | **89.1** |
| `-0` | `-inf` | **-92.5** |

All finite, and all near `ln(FLT_MAX) = 88.7`. The kernel clamps its input to the format maximum and
takes the log of *that*, so a non-finite input cannot produce a non-finite output. This is a kernel
behaviour rather than a golden one and it is the largest cat-B finding so far.

### 5.7 New: the golden invented NaN signs, twice, for two different reasons

Not a hardware finding — a test-framework one, and the largest single cause of cat-B divergence so far.
It only becomes visible when a NaN crosses a **16-bit Dest**, because that is where the pack path
substitutes an infinity *of the NaN's own sign*; at `dest_acc=Yes` the NaN stays a NaN and the
comparator's both-NaN clause hides the sign entirely.

1. **The cast destroyed the sign.** `torch`'s fp32 → bfloat16 cast maps every NaN to `0xFFFF`, sign bit
   set, whatever it started as — while `.to(float16)` preserves it correctly, so the defect hid on
   three quarters of the format axis. Hardware does nothing of the kind: a 16-bit Dest holds the top
   half of the fp32 pattern, so the sign survives verbatim. `cast_to_dest_dtype` models it as the
   truncation it is. The cast runs in **two** places per call — the Dest write, and the store into the
   result buffer, whose dtype follows `input_format` through `tilize_block` and is not always the Dest
   dtype. Repairing only the first appears to work and changes nothing on the pipelines where the two
   differ.
2. **libm invented a sign the golden then asserted.** IEEE 754 leaves the sign of a NaN produced by an
   invalid operation unspecified, and torch inherits the host libm, which picks inconsistently:
   `cos(inf)`, `acosh(0.5)`, `rsqrt(-1)` and `acos(2)` give `0xFFC00000` while `sqrt(-1)` and `log(-1)`
   give `0x7FC00000`. The SFPU emits a positive one. 24 of the 97 unary goldens were exporting libm's
   choice; `UnarySFPUGolden._NAN_SIGN_TRANSPARENT_OPS` now canonicalises all of them except the three
   ops that genuinely *move* the sign bit (`Neg` flips it, `Abs` clears it, `Identity` passes it on).

**The rule is confirmed on silicon in both directions**, which matters because a sign convention is
easy to get exactly backwards and still pass half the cases. `Neg(NaN)` packs to `-inf` — its NaN is
genuinely negative, and the variant only goes green with the sign preserved. `Sqrt(NaN)` packs to
`+inf` on the same pipeline — its NaN is positive, and that variant only goes green with the sign
*not* flipped. One measurement alone would have been consistent with "always negate".

Both defects were found the same way: enrolling four ops regressed **`Acosh`, `Cos`, `Sin` and `Exp`**,
none of which the change was about. A golden defect that only shows on ops you were not editing is the
argument for diffing the whole op set against a baseline before and after, rather than checking the
ops in hand.

**One silent fix fell out of it.** `Signbit(NaN)` returned `1.0` at `dest_acc=No` — reading a sign the
cast had invented. Nothing was failing, because `Signbit` is not enrolled for specials; it was waiting
to.

### 5.8 ~~New: SFPU comparisons rank `NaN` above every finite value~~ ✅ closed — the ISA specifies it

IEEE 754 makes every ordered comparison with a `NaN` operand false. The SFPU behaves as though `NaN`
were larger than everything — which is what an unsigned magnitude comparison gives, since a `NaN` has an
all-ones exponent and a set mantissa and so outranks any finite bit pattern.

**Derived from the pass/fail split, not inferred from one case.** The six unary comparison ops divide
exactly along the predicted line, and the ones that agree are as informative as the ones that do not:

| Op | Expression | Golden | Hardware | Consistent with "NaN is greatest"? |
|---|---|---|---|---|
| `UnaryLt` | `x < 0.5` | `0.0` | `0.0` | ✅ false either way — passes |
| `UnaryLe` | `x <= 0.5` | `0.0` | `0.0` | ✅ false either way — passes |
| `UnaryMax` | `max(x, 0.0)` | `NaN` | `NaN` | ✅ keeps the NaN — passes |
| `UnaryGt` | `x > 0.5` | `0.0` | **`1.0`** | ✅ true only under this rule |
| `UnaryGe` | `x >= 0.5` | `0.0` | **`1.0`** | ✅ same |
| `UnaryMin` | `min(x, 0.0)` | `NaN` | **`0.0`** | ✅ takes the *other* operand |

Six more ops follow from the same rule, each returning its upper bound where IEEE gives `NaN`:
`Clamp` → `1.0` (`CLAMP_MAX`), `Hardtanh` → `1.0`, `Hardsigmoid` → `1.0`, `ReluMax` → `5.0`
(`RELU_MAX_THRESHOLD`), `Sign` → `1.0` (the "not `<0`, not `==0`" branch), `Heaviside` → `1.0`
(the `x > 0` branch). Every value is that op's own dispatch constant, which is what makes the
explanation checkable rather than plausible.

**The ISA specifies this, and it makes the golden the wrong party.** `SFPGT`, `SFPLE` and `SFPSWAP`
each document a total order for FP32 — `-NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN` — and all
three route through `SignMagIsSmaller()`, which "treats C and D as sign-magnitude integers". The
comparison is a bit-pattern compare remapped to two's complement, not an IEEE compare, so a `+NaN`
outranking every finite value is by design
(`tt-isa-documentation BlackholeA0/TensixTile/TensixCoprocessor/{SFPGT,SFPLE,SFPSWAP}.md`).

**Seven of the nine are now enrolled**, with goldens that model the total order
(`sfpu_total_order_key` and its `min`/`max`/`clamp`/`relu_max` helpers) and pass as ordinary tests.
Recording them as kernel divergences — which is what this suite was about to do — would have written
seven permanent, plausible-looking lies about documented hardware.

The mapping was confirmed against the kernels before the goldens changed rather than assumed from
behaviour: `_relu_max_body_` is `v_if (result > threshold)`, a two-vector compare and therefore
`SFPGT`, and `_calculate_clamp_` has the same shape. `Hardsigmoid` turned out to *be*
`_relu_max_body_(x * (1/6) + 0.5, 1.0)` — it shares the kernel helper outright, which is why its
golden now shares one too, and why it diverged in exactly the same way.

Only the NaN answers moved: over 8000 finite inputs the rewritten goldens are bit-identical to the
ones they replace.

Two caveats keep the other two ops out, and one of them is architectural:

- **`Sign` and `Heaviside` compare against zero**, which is `SFPSETCC`, and its contract is explicitly
  conditioned: *"Provided that `VC` is neither negative zero nor any kind of NaN"*. `NaN` is outside it,
  so their `1.0` is unspecified rather than documented — consistent with an `int32` test on a positive
  NaN's bit pattern, but not guaranteed. Same shape as the `-0.0` caveat §5.2 already rests on.
- ~~**The total order is Blackhole-only.**~~ **Corrected, and then measured.** `WormholeB0/.../VectorUnit.md`
  indeed has no `SFPGT` and no `SFPLE`, but the order does not live in those two instructions:
  `WormholeB0/.../SFPSWAP.md` carries the same `SignMagIsSmaller()` and the same
  `-NaN < -Inf < … < +Inf < +NaN` comment, so **the order is specified on Wormhole too**. What these
  kernels lower to there is a separate and still-open question — the sources are two-vector compares,
  which is why the mapping above says `SFPGT`, and sfpi's expansion of `operator>` lives in the compiler
  backend rather than the headers. So all seven are **measured green on a Wormhole n300, 8/8 each,
  `0 xpassed`** — no arch-keying — but measured rather than guaranteed there.
  `Sign` and `Heaviside` also behave identically on Wormhole, so the `SFPSETCC` caveat above is one
  contract question rather than two measurements.

See §5.6 for what remains.

### 5.9 The `Log` saturation is a whole-family behaviour, not one op

§5.5 recorded `Log` clamping its input to the format maximum. Driving the full unary set shows **22
further ops** doing the same kind of thing — every one a polynomial or LUT approximation evaluated
outside the range its series covers. They split two ways, and neither is IEEE:

- **Saturating to the asymptote or a magic constant:** `LogWithBase` (`127.9` / `121.6` / `128.5`),
  `Digamma` (`89.09` at `NaN`; `≈ -337920` at `±0`), `I1` (`±1.1547668e37`), `Erf` (`1.0`), `Erfc`
  (`2.94e-12`), `Tanh` (`1.0`), `Sigmoid` / `TanhDerivative` / `Rdiv` / `Polygamma` (`0.0`), `Gelu` and
  `GeluDerivative` (`0.0` / `1.0`), `Lgamma` (`-0.00051` at `±0`), `UnaryPower` / `Rpow` /
  `CastFp32ToFp16a` (`+inf`).
- **Returning `NaN` where a value is defined** — the same failure from the other side: `Frac` at `±inf`,
  `SigmoidAppx` at `±inf`, `TanhDerivativeLut`, `Expm1Cw` at `+inf`, `Lgamma` at `±inf`, `SqrtCustom`
  (which manages `NaN` at `+inf` *and* `+inf` at `-inf`, both backwards), and `Erfinv` at `±1`.

**`LogWithBase` is the evidence that the cause is shared.** Its results are `Log`'s multiplied by the
dispatch scale `1/ln(2) ≈ 1.4427` — `88.7 × 1.4427 = 128.0`. Same clamp, seen through a scale factor.

So §5.6's first question is not about `Log`: it is a contract question about approximation kernels, and
one answer settles 23 held-out ops.

### 5.10 New: the sign of a generated NaN is Blackhole-guaranteed and Wormhole-unspecified

The first Wormhole run of the edge sweep — `475 passed · 198 skipped · 49 failed · 30 xfailed · 0 xpassed`
— fails **49 variants across 10 ops**: `Cos`, `Fmod`, `GeluAppx`, `Hardmish`, `Mish`, `Rsqrt`, `Silu`,
`Sin`, `Softsign`, `Tan`. Every one of them *generates* a NaN from the probe, the divergence is always
`golden=+inf` against `hw=-inf`, and every failing cell is one where a NaN cannot survive as a NaN: the
four format pairs at `dest_acc=No` (16-bit Dest) plus `Float32->Float16_b` at `dest_acc=Yes` (16-bit
output pack). Where the NaN does survive — `Float32->Float32` at `dest_acc=Yes` — the cells pass, because
`passed_test`'s both-NaN clause accepts either sign.

Read off the bit patterns at that surviving cell, a Wormhole-emitted NaN is `0x7FC00001` for a `+inf`
input and **`0xFFC00001`** for a `-inf` one. The ISA specifies exactly this difference, one sentence per
arch, in `SFPMAD.md` — "if a NaN is emitted":

| Blackhole | "it is always **the canonical NaN with bit pattern `0x7fc00000`**" |
|---|---|
| **Wormhole** | "the least significant bit of the mantissa is guaranteed to be set; other bits of the mantissa might or might not be set, and **the sign bit might or might not be set**" |

So `UnarySFPUGolden`'s canonicalisation — resting on the recorded sentence *"The SFPU emits a positive
one"* — is a documented guarantee on Blackhole and unspecified on Wormhole. The other half, why an
unspecified sign becomes an observable `±inf`, is documented too and the ISA flags it: the packer's early
format conversion says "if the exponent is 8 bits wide, **NaN becomes infinity (this is a potentially
surprising behaviour)**", and `SFPSTORE`'s note says "software is advised to avoid NaN inputs for this
conversion" — with Blackhole alone carrying "albeit canonical NaNs produced by arithmetic instructions do
not suffer any truncation". Both functional models keep the sign bit verbatim, which is what
`convert_nan_to_inf` already models correctly; only the sign it is *fed* is unsound on Wormhole.

**This is golden work, not a kernel divergence, and it needs no xfail.** The fix is in the shared
machinery: where a golden `NaN` is turned into `±inf` by a Dest write or a pack, accept either infinity,
and keep asserting the sign only for `_NAN_SIGN_TRANSPARENT_OPS` — a carve-out the ISA backs directly,
since `SFPABS`'s summary is "-NaN is left as -NaN rather than becoming +NaN" and `Neg` is a sign-bit flip.
Do **not** arch-key the measured sign instead: a bit the ISA says "might or might not be set" is not a
fact, and `Cos(+inf)` already gives a positive NaN at a 32-bit Dest and a sign-set one at 16-bit.
Tracked as [the plan's §4](SFPU_EDGE_CASE_EXPANSION_PLAN.md).

**One residue is not explained by any of it:** `Tan(NaN) -> 0.0` on the 16-bit-Dest path — a finite zero
rather than a substituted infinity. The same op at `Float32->Float16_b`/`dest_acc=Yes` returns a NaN for
the same probe, so it belongs to that path specifically and wants its own measurement.

**Why it was invisible until now.** `SPECIALS_READY_OPS` is empty on `main`, so no NaN is injected there;
the enrolment that makes these probes exist is on the branch. Two of the five failing cells are reachable
on Blackhole and are green there, so this is an arch difference rather than a coverage artefact.

### 5.11 New: the approximate-exp Wormhole gate XPASSes on Wormhole

`_APPROX_EXP_ACCURACY_XFAIL` records *"a systematic ~5.7% overshoot (peak 6.75%) once approximate exp's
argument passes ~8, measured on Wormhole"*, and `_APPROX_EXP_XFAIL_IS_WORMHOLE_ONLY` narrows it to Wormhole
because Blackhole XPASSed all four reachable cells. On a Wormhole n300 **all 6 marked variants XPASS** —
the gate's entire content, three cells at both tile shapes — so it now asserts nothing on either arch.

Measured over the elements with `x > 8` (`test_sfpu_wh_approx_exp.py`): mean signed relative error
**+0.75% to +1.05%**, peak **+3.5%**, and **no element of any tile above 5%**. So the direction reproduces
and the magnitude does not — the overshoot is real and roughly five times smaller than recorded, which puts
it inside the default 5% rtol. `Float16_b->Float16_b` at `dest_acc=Yes`, deliberately *not* gated, behaves
identically to the cells that are.

Three explanations are eliminated: the stimulus still reaches the region (425 and 261 elements above 8 per
tile, `x_max` 9.98 / 15.98, `_APPROX_ACCURACY_MAX[Exp]` = 16.0); no tolerance was loosened
(`CUSTOM_TOLERANCES` has no `Exp` entry and `passed_test` requires `torch.all(is_valid)`); and the golden is
plain `torch.exp`. Either the kernel's approximate path changed since the gate was written, or the overshoot
varies by board — and the recorded measurement does not name its card, which is what makes it unsettleable
from one host. Plan §9.1.

**One open item closes here.** §2.8's *"the accurate exp path over (16, 80] has never been isolated on
hardware"* — the Wormhole broad sweep ran `Exp` (132 passed) and `Exp2` (138 passed) at
`ApproximationMode.No` with 0 failures, and the probe measures **+0.00%** error above 8 out to `x = 79.97`
on the 32-bit-input cells. Sound on Wormhole; Blackhole still wants the same run.

### 5.12 New: the signed-zero arch gate XPASSes on Wormhole too

`_WORMHOLE_ONLY_EDGE_CLASSES` holds one class, `negative_zero_golden`, on the strength of *"measured on a
Blackhole p150b, the negative-zero class XPASSed on **all 16** cells it is claimed for"* — read as the ISA's
arch difference, `SFPMAD` flushing a negative zero on Wormhole and preserving it on Blackhole. On a Wormhole
n300 **the same 16 XPASS**: `SfpuElwdiv`, `SfpuXlogy`, `SfpuBinaryFmod` and `SfpuBinaryRemainder` at all four
`(format, dest_acc)` cells, which is again the gate's entire content.

**A gate that XPASSes on both arches cannot mean "the other arch is better".** The likelier reading is
already recorded as a trap in §5.2's neighbourhood: `passed_test` compares with `torch.isclose`, a both-NaN
clause and PCC, and `-0.0 == +0.0` under all three. If the comparator cannot see a zero's sign, these
variants pass whatever the hardware does — and the Blackhole XPASS was evidence about the comparator, not
about Blackhole.

That is a hypothesis, and one cheap experiment settles it: compare the class **bitwise** on Wormhole. If
hardware returns `+0.0` where the golden says `-0.0`, the divergence is real but invisible and the class
needs a bitwise comparator (the suite-wide change §5.2 already asks for), with the arch gate spurious. If
hardware returns `-0.0`, Wormhole is not flushing and the gate's premise is wrong on its own terms. Until
then `_WORMHOLE_ONLY_EDGE_CLASSES` is unverified. Plan §9.2.

### 5.6 What to raise with kernel owners

Written up in full, with measured tables and a reproduce command, in `KERNEL_OWNER_QUESTIONS.md`.
**Two remain**, plus one narrow one the ISA raised rather than settled:

1. **Approximation kernels do not propagate non-finite inputs** (§5.5, §5.9) — 23 ops. Is the input
   clamp intended, and should it be documented? The ISA cannot settle this and it is worth knowing
   why: it specifies the *primitives* only within stated ranges — `SFPARECIP` gives accuracy bounds
   for `0 ≤ x < 2` and suggests following up with Newton-Raphson, `SFPLUTFP32` documents no handling
   for `NaN`/`±inf` — so the out-of-range behaviour of a composition built on them is an LLK/API
   decision by construction.
2. **`RsqrtCompat(0)` saturates to `1.7014118e38`** where plain `Rsqrt` does not — 1 op. The ISA
   narrows it: `SFPARECIP` saturates to `0x7f800000` (`+inf`) for an input below `2^-126`, so
   `0x7F000000` (`2^127`) is not a value the instruction produces. The constant is a software clamp
   added above the primitive, and `Rsqrt`'s `+inf` is what the hardware itself would give. The
   question is therefore *why the clamp was added*, not which one the hardware does.

**The `NaN` comparison question is answered and acted on** — see §5.8. `SFPGT`, `SFPLE` and `SFPSWAP`
document a total order in which `+NaN` is the largest FP32 value, so seven of those nine ops were
golden work rather than a question, and are now enrolled. Two remain for an owner:

3. **Is `SFPSETCC` usable with a `NaN` operand?** Its contract is conditioned — *"Provided that `VC` is
   neither negative zero nor any kind of NaN"* — which leaves `Sign` and `Heaviside` returning `1.0`
   at `NaN` by an unspecified route, even though it is consistent with an `int32` test on a positive
   NaN's bit pattern. **Both arches, one question:** Wormhole has since been measured and returns the
   same `1.0`, so this is a contract question rather than a per-arch measurement.
4. ~~**What is the intended `NaN` comparison behaviour on Wormhole?**~~ **Withdrawn — measured, and the
   premise was wrong twice over.** `SFPSWAP` specifies the same total order on Wormhole, and all seven
   goldens pass there 8/8 with `0 xpassed`. Nothing for an owner.

Two questions are **withdrawn** and should not be re-filed. `signbit`: §5.2's measurement shows the probe
is not delivered on those six combinations, so there is no kernel contract to question. The
**generated-NaN sign on Wormhole** (§5.10): 49 failing variants that read as a kernel divergence and are
documented behaviour — `SFPMAD.md` leaves the sign unspecified on Wormhole and guarantees canonical
`0x7fc00000` on Blackhole. It is golden work, not a kernel question.

### 5.4 Two smaller results

- **The bitwise kernels need the two's-complement pack path** for negative operands.
  `(INT32_MIN+1) & -1` returned `-1`. Nothing had established this, because
  `test_sfpu_binary_bitwise` draws from the positive-only default and had never fed them a negative.
- **Both Blackhole guards are non-strict xfails rather than skips**, so a kernel fix reports XPASS
  instead of leaving the case green by omission indefinitely.

---

## 6. Cat B — where specials can be injected (measured)

Not enabled (§2.1), but the matrix is data rather than guesswork. Measured by driving the five
isinf/isnan predicates over the full 5×5 format matrix × both `dest_acc` with no skips — 250
variants, 85 failing. Two independent breakers:

- **A `Float16` (e5m10) anywhere.** As an *input* it never preserves specials — all 5 predicates fail
  on all 5 outputs at both `dest_acc`, 10/10 cells. As an *output* it fails too, unless a 32-bit input
  is paired with `dest_acc=Yes`; `Float32 -> Float16` at `dest_acc=No` fails all five, which is the
  exact pair Blackhole already guards.
- **A 16-bit input with `dest_acc=Yes`.** `Float16_b` there fails `isinf`/`isneginf`/`isnan` while
  `isposinf`/`isfinite` pass — **`+inf` survives, `-inf` and `NaN` do not.**

A third constraint is applied statically rather than measured: block-float and MX *inputs* cannot
carry specials at all (`quantize_input_to_unpack_format` destroys NaN for Bfp8_b and Bfp4_b), so a
predicate passing there is vacuous — golden and hardware agree there is no NaN because neither ever
saw one. Those rows are excluded rather than trusted. Block-float *outputs* are excluded on the
golden's behalf: an `inf` inside a block whose shared exponent is finite is not a value the format can
express, so neither the lattice nor the tolerance criterion means anything for it.

**Safe surface — 7 cells of 50** (5 formats × 5 × both `dest_acc`; revisions before 12 said "of 40"):

| `dest_acc` | Safe `input -> output` |
|---|---|
| `No` | `Float32->Float32`, `Float32->Float16_b`, `Float16_b->Float32`, `Float16_b->Float16_b` |
| `Yes` | `Float32->Float32`, `Float32->Float16`, `Float32->Float16_b` |

Measured on Wormhole, and **re-measured there on 2026-08-13 with the same instrument: confirmed.** 250
variants, 85 failing again, and aggregating to the 50-cell matrix every one of the 7 safe cells passes all
5 predicates while no safe cell fails any. Both breakers reproduce in shape, `Float16_b` at
`dest_acc=Yes` included (2/5: `+inf` survives, `-inf` and `NaN` do not). One detail this section did not
record: among the statically-excluded block-float rows, a **`Bfp8_b` input at `dest_acc=No` genuinely
fails** `isinf`/`isneginf`/`isnan` (2/5 on four of its five outputs; only `Bfp8_b->Float16` passes), while
a `Bfp4_b` input passes 5/5 everywhere — so the static exclusion is covering a failure in one case and a
vacuous pass in the other.

The whole matrix is written out longhand in `test_sfpu_domains.py` (7 accepted cells of 50) so it cannot
be rewritten without a test changing outcome — including a guard for the `DestAccumulation` truthiness
trap, where both enum members are truthy and `bool(member)` would silently flip whole rows.

**Blackhole: 3 of the 7 confirmed, and the other 4 are unreachable there by construction** — not by
omission. `_skip_bh_unless_fp32` allows only `Float32->Float32` at `dest_acc=No`, which collapses that
row's four triples to one, and the edge sweep's format axis is `Float16_b`/`Float32`, so
`Float32->Float16` at `dest_acc=Yes` is never collected. The three reachable cells
(`Float32->Float32` at both `dest_acc`, `Float32->Float16_b` at `dest_acc=Yes`) **do** carry specials
on Blackhole — all nine enrolled cat-B ops pass there, modulo the three recorded kernel xfails in §2.1.

One caveat the three cells do **not** cover: carrying `±inf` and `NaN` is not the same as carrying a
`-0.0`, and only the two `dest_acc=Yes` cells do the latter. `negative_zero_delivered()` is the second
gate; see §5.2.

The table is therefore **not** arch-keyed, and — now that the Wormhole re-measurement has been done and
agrees — should stay that way. `test_specials_safe_matches_measured_matrix` keeps its one verdict per cell
rather than being parametrized by arch.

---

## 7. How to regenerate this document

Nothing here needs hardware. §1's figures and §4's tables come from the same inventory:

```bash
cd tt_metal/tt-llk/tests/python_tests
python3 -c "
import sys; sys.path.insert(0,'.')
from helpers.sfpu_domains import (_OP_SINGULARITIES, _OP_EDGE_POINTS, _OP_DOMAIN_REGISTRY,
                                  sfpu_unary_ops, edge_spec, SPECIALS_READY_OPS)
from helpers.llk_params import MathOperation, DataFormat as F
u = sorted(sfpu_unary_ops(), key=lambda o: o.name)
e = [o for o in u if edge_spec(o, F.Float32, F.Float32) is not None]
print('singularities', len(_OP_SINGULARITIES), '| edge points', len(_OP_EDGE_POINTS))
print('unary', len(u), '| with an edge', len(e), '| smooth', len(u) - len(e))
print('specials-ready', len(SPECIALS_READY_OPS))
"
# expect: 21 / 43 / 97 / 50 / 47 / 65
# The last number is len(SPECIALS_READY_OPS), which counts the 5 scalar binops as well
# as the 60 enrolled *unary* ops -- the scalar family is not in sfpu_unary_ops(), so it
# does not appear in any of the other five figures. Do not read 65 as a unary count.
# (Revisions before 12 said 19 / ... / 0 here, which contradicted §1's own table: the
#  singularity count gained the two ternary operand-C poles, and specials-ready has not
#  been 0 since revision 7.)
python3 -m pytest test_sfpu_domains.py -q --noconftest   # expect 107 passed
```

Per-op rows are keyed on `MathOperation` and read `_OP_DOMAIN_REGISTRY`, `_OP_SINGULARITIES`,
`_OP_EDGE_POINTS` and the sweep op lists in `test_sfpu_unary.py`. An op enrols in the edge sweep **by
being in the registry**, not by being listed in a test, so a new op appears in §4 automatically —
regenerate rather than editing rows by hand.

The `pytest --noconftest` above is needed because `conftest.py` imports `helpers/device.py`, which
imports `tt-exalens`; `tests/requirements.txt` pins `0.3.29` and later releases moved
`CallstackEntry` and `ElfFile`, so a drifted venv fails at collection with what looks like a broken
checkout. Host-side tests do not need the device at all.
