# SFPU Edge-Case Coverage — Plan for What Is Left

**Companion to:** [SFPU_EDGE_CASE_COVERAGE.md](SFPU_EDGE_CASE_COVERAGE.md) — the per-op audit
**Issue:** [tenstorrent/tt-metal#49739 — [LLK] SFPU testing edge cases](https://github.com/tenstorrent/tt-metal/issues/49739)
**Repo:** `tt-metal/tt_metal/tt-llk/tests/python_tests/`
**Scope:** Wormhole B0 and Blackhole. Quasar keeps its own inline stimulus definitions under
`quasar/` and is tracked separately.

**Revision 10 — 2026-08-12.** Items 1–4 of revision 9's backlog have been **implemented and verified
on Blackhole silicon** (p150b). This revision records what that closed, what it *found*, and
re-scopes what is left. Revision 9's §3–§6 are replaced rather than annotated.

Two results are worth reading even if you read nothing else:

1. **The `SFPMAD` signed-zero prediction was correct.** It was a testable prediction on record — the
   ISA documents flush-to-positive-zero on Wormhole and flush-to-sign-preserved-zero on Blackhole —
   and the negative-zero edge class **XPASSed on all 16 cells** it is claimed for, with nothing else
   XPASSing. Those cells are now arch-gated, so Blackhole *asserts* the sign of a zero result.
2. **The `-0.0` delivery question is answered, and the answer is "not delivered".** At `dest_acc=No`,
   `Reciprocal`, `Rsqrt` and `Sqrt` all treat `-0` exactly as `+0`; at `dest_acc=Yes` with a 32-bit
   input they do not. That is the `unpack_to_dest` split, measured independently of the
   signbit/sign/heaviside partition it had been inferred from. See §4.3.

Where a mechanism landed, the code is the record. Everything asserted here was measured against the
tree at `ldjurovic/sfpu_edge_cases_phase_3`; §7 lists the checks so the next revision can re-run them
rather than trust this sentence.

---

## 1. What is already done

**Phases 0–4** ([#52172](https://github.com/tenstorrent/tt-metal/pull/52172),
[#52416](https://github.com/tenstorrent/tt-metal/pull/52416)) pointed all four families at the per-op
domains in `_OP_DOMAIN_REGISTRY` instead of a positive-only `uniform(0.1, 1.1)`, then added the shared
edge metadata, one builder (`edge_spec()` / `edge_pair_values()`) and three thin sweeps over the
existing drivers.

**This revision** closed four more items:

| Item | What landed | Verified |
|---|---|---|
| 4 — per-op tolerances | `BINARY_CUSTOM_TOLERANCES`; `pow` widened to A≤8 B≤4 with rtol 0.15, `xlogy` x doubled to 8 with atol 0.6 | Blackhole, both ops green |
| 3 — ternary operand C | `OperandSpecs.spec_C`, `Operand.C`, `_OP_OPERAND_EDGE_POINTS`, `test_sfpu_ternary_edges` | Blackhole, 9 passed / 7 skipped |
| 1 — cat B goldens | 5 ops enrolled in `SPECIALS_READY_OPS`; `_sin`/`_cos` goldens fixed; next 5 measured and deferred | Blackhole, 5 ops green on every safe triple |
| 2 — Blackhole verification | signed-zero class arch-gated after 16 XPASS; `specials_safe()` partially confirmed; `-0.0` delivery settled | Blackhole, four suites |

Plus one **pre-existing bug** found by running the edge sweep through the two-phase flow for the first
time: `_classify_edge_pair` called `get_golden_generator`, which the harness replaces with a
`DummyGoldenGenerator` during `--compile-producer`, and that stub has no `ops` mapping. The whole
binary edge sweep raised `AttributeError` under the flow **CI actually uses** — it had only ever
worked when pytest was invoked directly. Fixed by instantiating `BinarySFPUGolden` directly, the same
way `helpers/compressed_utils.py` already does and for the same reason.

Category status: **A** closed for every op that has a boundary · **B** open but no longer zero —
5 ops of 97, with the mechanism proven end to end · **C** closed for the 5 ops whose kernels claim
the full int32 range · **D** closed for all 43 knees, plus lerp's weight boundaries on operand C ·
**E** blocked on C++ · **F** untouched.

The lesson from revision 9 held again, and is now three for three: this plan keeps assuming the
*stimulus* is the hard part, and the binding constraint keeps turning out to be the **golden**. Of
the ten cat-B ops attempted, two needed a golden fix before they could run at all and three more are
deferred *because their goldens are wrong*, not because the kernels are.

---

## 2. The remaining work, ordered by value

| # | Item | Blocked on | Size | Where |
|---|---|---|---|---|
| 1 | **Cat B, next tranche** — 5 ops measured, goldens need fixing first | per-op golden semantics | medium, divisible | §3 |
| 2 | **`Log` saturates non-finite inputs** — new finding, needs a kernel owner | judgement, not code | one question | §4.2 |
| 3 | **Cat B, the long tail** — the remaining ~87 ops | as item 1, repeated | large | §3 |
| 4 | **Cat E** — unary shift amount | C++ `constexpr` → `TemplateParameter` | cross-language | §6 |
| 5 | **Cat F** — new harnesses for 11 kernels with no enum entry | new C++ source + golden each | large, per kernel | §6 |
| 6 | **Scalar tensor-operand edges** | cat B reaching the scalar ops | thin wrapper | §6 |
| 7 | **Wormhole re-measurement** | Wormhole hardware | one sweep | §4.4 |
| 8 | **CI does not run any of this** | scheduling decision, not code | one workflow change | §8 |

Item 1 is the only one that unblocks others. Items 2 and 8 are the cheapest things on the list and
both are "ask someone", not "write code".

---

## 3. Cat B — where it stands and what the next tranche is

### 3.1 What landed

Cat B went from **zero ops** to five, and more importantly from "wired but never executed" to
"executed on silicon". `SPECIALS_READY_OPS` now holds `Identity`, `Abs`, `Exp`, `Sin`, `Cos`, each
with its reason, and each green on Blackhole across every specials-safe triple the sweep reaches.

Two of those needed a golden fix to run at all: `_sin` and `_cos` called `math.sin` / `math.cos`,
which **raise** `ValueError("math domain error")` on a non-finite input rather than returning one.
The golden carried a comment saying the input was "never not finite" — true until cat B, which is
precisely the assumption cat B exists to break. Both now route through `torch`, which is IEEE-correct.

The ordering from revision 9 paid off exactly as intended: the trivially-defined ops went first, they
passed, and that established the *mechanism* was sound before any op with an interesting answer was
attempted. When the next five diverged, there was no ambiguity about whether the harness or the
semantics was at fault.

### 3.2 The next tranche — five ops, measured, and mostly blocked on the golden

`_SPECIALS_NEXT_TRANCHE` records these with the full per-probe measurement in a comment. Measured on
Blackhole with a Float32 input, which is the only specials-carrying input format reachable there:

| Op | Probe | Golden | Hardware | Whose bug |
|---|---|---|---|---|
| `Neg` | `NaN` | `+inf` | `-inf` | **golden** — it mangles NaN at `dest_acc=No` |
| `Neg` | `+0` | `+0` | `-0` | **golden** — the hardware is IEEE-correct here |
| `Reciprocal` | `NaN` | `+inf` / `NaN` | `+0` | kernel — NaN is not propagated |
| `Reciprocal` | `-inf` | `+0` | `-0` | **golden** — hardware IEEE-correct |
| `Sqrt` | `-0` | `+0` | `NaN` | kernel |
| `Rsqrt` | `-0` | `-inf` | `NaN` | kernel |
| `Log` | `±inf`, `NaN` | `±inf` / `NaN` | `88.5`, `84.3`, `89.1` | kernel — see §4.2 |

They are deliberately **not** enrolled. Enrolling an op whose golden is wrong records a kernel xfail
for a test-side defect, which is worse than no coverage: it launders a golden bug into a permanent
"known hardware divergence". Three of the five need the golden fixed first.

**The plan for each, in order:**

1. **Fix the golden's NaN handling through the 16-bit dest path.** `Neg(NaN) → +inf` and
   `Log(NaN) → +inf` are the same defect seen twice, and both appear only at `dest_acc=No`. This is
   one fix in the golden's quantization path, not two op fixes, and it is the prerequisite for
   everything else in this table.
2. **Decide whether the golden should model signed zero at all.** `Neg(+0) → -0` and
   `Reciprocal(-inf) → -0` are cases where the *hardware* is right. Either the golden learns signed
   zero, or these probes are excluded with a recorded reason — but "xfail the hardware" is not an
   option when the hardware is the correct one.
3. **Then enrol `Neg` and `Reciprocal`**, and record what remains (`Reciprocal(NaN) → +0`) as a
   genuine kernel xfail.
4. **`Sqrt(-0)` and `Rsqrt(-0)` are kernel divergences** and can be enrolled as soon as step 1 lands:
   IEEE says `sqrt(-0) = -0` and `rsqrt(-0) = -inf`, the golden agrees, and the hardware returns
   `NaN`. Record as non-strict xfails, `dest_acc=Yes` only — at `dest_acc=No` the probe is not
   delivered (§4.3), so those cells would be vacuous.
5. **`Log` last**, because it is not really a specials question — see §4.2.

### 3.3 The long tail

87 unary ops remain outside `SPECIALS_READY_OPS`, and **47 of them are smooth everywhere**, so cat B
is still their entire edge story. Nothing about that changed; what changed is that the path is now
walked rather than theoretical, and the cost per op is known: check the golden host-side first (one
Python snippet, no hardware), fix it if wrong, enrol, run, record.

The measured rate from this tranche: of 10 ops attempted, 5 were already correct, 2 needed a small
golden fix, 3 need a real golden decision. Budget roughly half the ops as free and half as needing
golden work.

---

## 4. Blackhole verification — what it settled

### 4.1 The `SFPMAD` signed-zero prediction: confirmed

The highest-information result available, and it came back in favour of the ISA. Wormhole's `SFPMAD`
page documents flush-to-**positive** zero; Blackhole's documents flush-to-**sign-preserved** zero and
lists negative-zero handling among its upgrades. The xfails were left non-strict precisely so this
could report XPASS.

Measured on a p150b: **16 XPASS, and every one of them the `negative_zero_golden` class** — all four
ops (`div`, `xlogy`, `fmod`, `remainder`) at both `dest_acc` values. Nothing else XPASSed.

So `_WORMHOLE_ONLY_EDGE_CLASSES` now arch-gates that class: on Blackhole the sign of a zero result is
**asserted**, and a regression there fails instead of quietly returning to XFAIL. That is a real
coverage gain, and it is the outcome the whole non-strict-xfail convention was designed to produce.

The indeterminate-form classes (`both_zero`, `nan_golden`) are deliberately *not* gated — they are the
kernels' own reciprocal composition, unexplained by the ISA, and they still diverge on Blackhole.

### 4.2 New finding: `Log` saturates its input, so no non-finite value survives

`Log(+inf)` returns **88.5**, `Log(-inf)` returns **84.3**, `Log(NaN)` returns **89.1** — all finite,
all near `ln(FLT_MAX) = 88.7`. The kernel clamps its input to the format maximum and takes the log of
*that*, so a non-finite input cannot produce a non-finite output. `Log(-0)` likewise returns `-92.5`
rather than `-inf`.

This is a kernel behaviour, not a golden one, and it is the largest cat-B finding so far. It belongs
with `RsqrtCompat(0)` as a question for kernel owners: **is input saturation intended for `log`, and
if so should it be documented?** Until that is answered there is no way to know whether the right
test outcome is a pass, an xfail, or a bug report — which is why `Log` is last in §3.2's order.

### 4.3 `-0.0` delivery: settled, and it is not delivered

The coverage audit inferred, from the way `signbit` / `sign` / `heaviside` partition exactly on
`unpack_to_dest`, that `-0.0` only reaches the LREG on the unpack-to-dest path. Cat B measured that
directly and independently:

- At **`dest_acc=No`**: `1/-0 → +inf` (not `-inf`), `rsqrt(-0) → +inf` (not `-inf`),
  `sqrt(-0) → +0`. All three ops treat `-0` *exactly* as `+0`.
- At **`dest_acc=Yes`** with a 32-bit input: they do not — `sqrt(-0) → NaN`, `rsqrt(-0) → NaN`,
  distinct answers that could only come from a real `-0`.

That is the `unpack_to_dest` split, confirmed by three ops that have nothing to do with sign
predicates. **Consequence:** `Signbit`'s six xfails are confirmed as a stimulus limitation that can
never XPASS, `Sign`'s and `Heaviside`'s other six passes are confirmed vacuous, and any future `-0`
probe should be scoped to the unpack-to-dest combinations rather than driven everywhere.

### 4.4 `specials_safe()`: partially confirmed, and the rest is unverifiable *on Blackhole*

Of the 7 triples the table accepts, only **3** are reachable on Blackhole, and the two reasons are
both structural rather than oversights:

- `_skip_bh_unless_fp32` allows only `Float32->Float32` at `dest_acc=No`, which collapses that row's
  four triples to one.
- The edge sweep's format axis is `Float16_b` / `Float32`, so `Float32->Float16` at `dest_acc=Yes` is
  not collected at all.

The three reachable cells (`Float32->Float32` at both `dest_acc`, `Float32->Float16_b` at
`dest_acc=Yes`) **do** carry specials on Blackhole — the five enrolled ops pass there. The remaining
four involve a `Float16_b` input at `dest_acc=No` or a `Float16` output, which Blackhole's own
architecture guards exclude.

So the table is **not** arch-keyed, and should not be until someone can run the rows that differ. What
is needed is a Wormhole re-measurement of the same predicate sweep (item 7), not more Blackhole time.

### 4.5 The accurate exp path over (16, 80]

Revision 9 flagged this as restored-but-unmeasured: `_APPROX_ACCURACY_MAX` narrows the exp family only
for `ApproximationMode.Yes`, so the accurate path got its range-bound domain back. The broad unary
sweep exercises `Exp` and `Exp2` at `ApproximationMode.No` over exactly that region, and it is green
on Blackhole. The region is no longer unmeasured.

---

## 5. Item 3 — ternary operand C: what landed

`OperandSpecs` grew `spec_C`, defaulting to a copy of `spec_B` exactly as `spec_B` defaults to a copy
of `spec_A`, so all five existing consumers keep working unchanged. `Operand` grew `C`, and
`spec_for(operand)` replaced the `spec_A if ... else spec_B` chains that could not express a third
operand.

With that in place the metadata carries the pole like any other family:

| Op | Formula | Registered edge |
|---|---|---|
| `addcdiv` | `a + value * b / c` | `_OP_SINGULARITIES` C = `(0.0, BOTH)` |
| `snake_beta` | `a + sin(b*a)² / c` | `_OP_SINGULARITIES` C = `(0.0, BOTH)` |
| `lerp` | `a + c * (b - a)` | `_OP_OPERAND_EDGE_POINTS` C = `(-1, 0, 1, 2)` |
| `addcmul` | `a + value * b * c` | none — a multiply has no pole, so `edge_spec` is `None` |

`lerp` needed a second table: `_OP_EDGE_POINTS` describes the op's own input, which for a ternary op
is the wrong operand. Rather than nest every one of the 43 existing entries under an operand key to
express one op's knees, per-operand knees live in `_OP_OPERAND_EDGE_POINTS` and `op_edge_points()`
takes an operand.

**One design decision worth knowing.** Driving `c = 0` with an unconstrained numerator mixes two
questions: what the kernel does at the pole (4064 of 4096 elements, all of which should be `±inf`) and
what it does at `0/0` (a handful of elements, golden `NaN`, hardware `inf`). Measured on Blackhole,
the only failures were the second kind — and that is the *same* indeterminate form already recorded
against `div`, `fmod`, `remainder` and `xlogy`. Holding the numerator off zero turns a tolerated xfail
into a real assertion about the pole and loses nothing not already covered elsewhere. If `0/0` is ever
worth driving here it wants its own variant and its own xfail, the way the binary suite splits classes.

Result on Blackhole: **9 passed, 7 skipped** (`addcmul` has no edge; `Float32` at `dest_acc=No` is
unsupported).

---

## 6. The independent items

### 6.1 Cat E — the unary shift amount (unchanged)

`SHIFT_AMOUNT` is a `constexpr std::uint32_t SHIFT_AMOUNT = 3u` inside `call_unary_sfpu_operation`
(`helpers/include/sfpu_operations.h`), paired with `_int_shift_amount` on the golden side. Sweeping it
needs a new `TemplateParameter` plus matching golden plumbing — cross-language, not test wiring. The
Python side is written and reusable: `_SHIFT_EDGE_AMOUNTS` already covers
`{0..31, 32, 33, 40, 63, 100, 1000, −1, −5, −32, −1000}`, `_shift_reference` is the golden, and
`_build_paired_tile_override` is the delivery.

### 6.2 Cat F — the 11 kernels with no enum entry (unchanged)

Confirmed still absent from `MathOperation`: `welfords`, `dropout`, `quant`, `cumsum`,
`reshuffle_rows`, `int_sum`, `tiled_prod`, `copy_dest_values`, `generalized_moe_gate_topk`,
`max_pool_indices`, `rand`. Each needs a new C++ source and golden.

| Priority | Kernels | Why |
|---|---|---|
| High | `welfords`, `int_sum`, `cumsum`, `tiled_prod` | Reduction family — four kernels share one harness cost |
| High | `quant` | Used in production quantization, no correctness test at all |
| Medium | `dropout`, `rand` | RNG; need a distribution-level assert |
| Medium | `reshuffle_rows`, `copy_dest_values`, `max_pool_indices` | Data-movement / index |
| Medium | `TopKLocalSort` / `Merge` / `Rebuild` | Have enum entries but are perf-only |
| Medium | `AddInt32`, `SubInt32`, `AbsInt32`, `BitwiseNot` | Perf-only, blocked by the fast-tilize gap (tt-llk#495) |

### 6.3 Scalar tensor-operand edges

Still correctly deferred, and now for a sharper reason. The five scalar ops are `x (+|-|*|/) c` for a
compile-time `c`, so they are smooth in `x` and cat A and cat D contribute nothing — their only edge is
cat B. None of them is in `SPECIALS_READY_OPS` yet, so a wrapper would still skip every variant it
collects. It goes live when cat B reaches the scalar goldens, as one of §3's commits rather than its
own piece of work. The `spec_A` hook and the sketch are already in place.

### 6.4 Per-op tolerances (item 4) — done, with the measurement

Recorded here because the *numbers* are the deliverable and they belong somewhere durable.
`BINARY_CUSTOM_TOLERANCES` in `test_sfpu_binary.py` carries the full table; the finding behind it:

- **`pow`'s relative error is flat in the operands**, not growing with them: 10.0% at `b·ln a = 3.30`,
  13.4% at 8.32, 10.3% at 11.09. What had been capping the domain at 3 was the fixed 5% rtol, not the
  op. With rtol 0.15 the domain widens to A≤8, B≤4 — 2.5× the old argument reach. A≤16 was rejected
  for a *different* reason: it drives `|a**b|` to 6.2e4, within a factor of 1.06 of Float16's ceiling,
  which would make it an overflow test wearing an accuracy test's clothes.
- **`xlogy`'s absolute error is exactly linear in x** — 0.25 / 0.50 / 1.00 / 2.00 at x ≤ 4 / 8 / 16 /
  32 in Float16_b — which confirms the documented model and is why no fixed atol could hold. With
  atol 0.6, x doubles to 8. Most of the Float16_b figure is *output quantization* rather than kernel
  error: at `|golden| ~ 72` a bfloat16 ULP is already 0.5. The Float32 column is 4× smaller.

---

## 7. How to re-verify this document

Every factual claim above is checkable from the tree without hardware. Re-run these before the next
revision rather than trusting the prose:

```bash
cd tt_metal/tt-llk/tests/python_tests
# metadata counts, op inventory, per-op edge coverage
python3 -c "
import sys; sys.path.insert(0,'.')
from helpers.sfpu_domains import (_OP_SINGULARITIES, _OP_EDGE_POINTS, sfpu_unary_ops,
                                  edge_spec, SPECIALS_READY_OPS)
from helpers.llk_params import DataFormat as F
u = sorted(sfpu_unary_ops(), key=lambda o: o.name)
e = [o for o in u if edge_spec(o, F.Float32, F.Float32) is not None]
print(len(_OP_SINGULARITIES), len(_OP_EDGE_POINTS), len(u), len(e), len(SPECIALS_READY_OPS))
"
# expect: 21 43 97 50 5
python3 -m pytest test_sfpu_domains.py -q --noconftest   # expect 107 passed
```

The per-op tables in the coverage audit are generated from the same inventory; regenerate them the
same way rather than editing rows by hand.

**On hardware.** Never call `pytest` directly — use the repo's runner, which serialises silicon access
and cleans up stale state:

```bash
cd tt_metal/tt-llk
.claude/scripts/run_test.sh run --worktree $PWD --arch blackhole \
    --test test_sfpu_ternary.py --k test_sfpu_ternary_edges
```

Expected on a Blackhole p150b, all four suites green through the two-phase flow:

| Suite | Result |
|---|---|
| `test_sfpu_unary.py` | 4932 passed · 1666 skipped · 14 xfailed |
| `test_sfpu_binary.py` | 739 passed · 531 skipped · 36 xfailed · **0 xpassed** |
| `test_sfpu_ternary.py` | 39 passed · 25 skipped |
| `test_sfpu_binop_scalar.py` | 58 passed · 62 skipped |

**A non-zero xpassed count is a signal, not noise.** Both arch-gates in this tree were derived from
one: 16 XPASS in the binary suite became the signed-zero gate, 4 in the unary suite became the
approximate-exp gate. If a future run reports XPASS again, something the tables call arch-specific has
changed.

**Environment.** `tests/requirements.txt` pins `tt-exalens==0.3.29`, and `run_test.sh` expects a venv at
`tests/.venv`. A venv carrying an *older* exalens fails at `conftest` import with a missing-symbol
`ImportError` (`CallstackEntry`, `ElfFile` — both added in later releases), which reads like a broken
checkout rather than a stale venv. Host-side tests need neither and can be run with
`pytest --noconftest`.

---

## 8. What bounds the value of all of it

**None of this coverage is guarded by CI.** The broad unary profile runs in **no automated job on any
architecture**: every LLK pytest job either excludes `nightly` (pr-gate smoke, bit-exact) or runs
`--coverage`, under which the broad profile is skipped wholesale. That leaves the large majority of
the sweep's parametrizations running nowhere, and it predates all three PRs.

Either `llk-e2e` needs a non-coverage companion group, or the broad profile has to stop being
coverage-gated. **Worth filing before item 2 rather than after** — arch verification added to a suite
no job runs is worth strictly less than the same work against a suite that is actually scheduled.

**One citation to check first.** The live skip reason in `test_sfpu_unary.py` attributes the
coverage-gating to [tt-llk#1435](https://github.com/tenstorrent/tt-llk/issues/1435). That issue is
open, but its title is about `test_eltwise_unary_sfpu.py` failing on a mismatch when it runs after
`test_eltwise_binary` — test *ordering*, not coverage. Either the citation is wrong and has
propagated into the source, or the issue has been repurposed in its comments. Resolve it before
filing anything that cites it, since the skip reason points readers there.

**Two questions for kernel owners, independent of any item above.** Neither has been filed. Both are
divergences the ISA does not explain, cheap for an owner to adjudicate and expensive for a test to keep
guessing about:

1. **`Log` saturates a non-finite input** to the format maximum, so `log(+inf)` returns 88.5,
   `log(-inf)` 84.3 and `log(NaN)` 89.1 — all near `ln(FLT_MAX)`, none non-finite (§4.2). Is that
   intended, and should it be documented? `Log` cannot be enrolled in cat B until this is settled,
   because there is no way to know whether the right outcome is a pass, an xfail or a bug report.
2. **`RsqrtCompat(0)` saturates to `1.7014118e38`** (`0x7F000000`) instead of returning `inf`, on all
   8 combinations — while plain `Rsqrt` over the same probe does not diverge. Two implementations of
   one function disagreeing at their shared pole, with nothing in the ISA prescribing either answer.

**One question is withdrawn.** `signbit(-0.0)` was previously the sharpest item on this list. §4.3's
measurement shows the probe is not *delivered* on the six combinations where it diverges, so there is
no kernel contract to question — it is a stimulus limitation, and the reason strings now say so.

---

## 9. Traps to know before starting

Every one of these has already cost time once.

- **A constant derived from another by a prose rule will drift.** `_exp_with_base_spec` is documented
  as double `_exp_spec`'s; two branches moved the two halves independently and nothing failed, they
  just stopped agreeing. There is now a host-side test asserting the relation
  (`test_exp_with_base_argument_ceiling_matches_exp_in_both_modes`) — add the same kind of assertion
  for any new derived constant. A docstring is not a constraint.
- **`exclude_intervals()` is not stimulus-neutral.** It always rewrites its result into the
  `intervals` form, and that sampler consumes **two** `torch.rand` draws per element where the plain
  `low`/`high` path consumes one. So `uniform(1, 8)` and `intervals=[(1, 8)]` are the same
  distribution and different numbers at the same seed, and **declaring a new hole in
  `_SFPU_UNDEFINED_RANGES` re-rolls that op's entire stimulus set** even when the subtraction removes
  nothing. Keep edge metadata off this path.
- **Do not route edge specs through `for_op_pipeline`.** Its `_tighter_spec` measures a domain with
  `_spec_span`, which falls back to `spec.high - spec.low` — `None - None` for a values-list spec.
  Nothing hits it today because the paths are separate; the obvious-looking unification raises
  `TypeError`.
- **`StimuliSpec.custom` cannot carry integer extremes.** `CustomStrategy.generate_face` clamps
  through `_get_integer_bounds`, which returns `info.min + 1`, so a spec asking for `INT32_MIN`
  silently yields `INT32_MIN + 1`. Integer edges go through `src_A_override` as a raw tensor —
  `_build_paired_tile_override` is the shared helper.
- **Enum members are not their values.** `DestAccumulation` and `ApproximationMode` both wrap
  `True`/`False`, so `bool(DestAccumulation.No)` is `True`. `_two_state_flag` normalises both and
  rejects anything else; the next `if dest_acc:` written by hand will be wrong in the same way.
- **A probe must survive the datapath, not just the format.** With `dest_acc=No` the DEST holds 16
  bits whatever the input format is, so an fp32 probe one fp32 ULP above a pole of 1.0 is truncated
  back onto the pole. `probe_beside()` decides per boundary *and per side*, because the step down
  from 1.0 crosses a binade and survives while the step up does not.
- **`format_ulp` returns a lower bound for block-float formats**, because the real step is set by the
  exponent shared across the 16-element block. Safe direction, but a block-float probe *pair* cannot
  be assumed distinct.
- **`TestConfig` calls `shutil.rmtree()` on the fixed path `/tmp/tt-llk-build` at session setup.** Any
  second pytest session on the same host — including a one-op `-k` run started to triage something —
  deletes the build tree out from under a running sweep. The victim reports `ld: cannot open output
  file`, which in a log reads exactly like a real kernel bug. This produced two phantom failures
  during Phase 0 and will recur, because this work is triage-heavy by nature. Worth fixing
  separately: key the artefact root by session, or take the existing
  `/tmp/tt-llk-build-shared.lock` around the rmtree.
- **The pinned test environment drifts, and the direction matters.** `tests/requirements.txt` pins
  `tt-exalens==0.3.29`; a venv carrying an **older** one fails at `conftest` import with a
  missing-symbol `ImportError` (`CallstackEntry`, `ElfFile` — both *added* in later releases), which
  reads like a broken checkout rather than a stale venv. It is easy to misdiagnose this as "the symbol
  moved in a newer release" and start writing shims; check the installed version against the pin first.
  `run_test.sh` also expects a venv at `tests/.venv`, which `setup_testing_env.sh` does **not** create
  — that script installs SFPI and pre-commit only. Host-side tests need neither and run under
  `pytest --noconftest`.
- **A golden reached at stimulus-build time cannot come from `get_golden_generator`.** The harness
  swaps in a `DummyGoldenGenerator` during `--compile-producer`, and that stub has only `__call__` — no
  `ops` mapping, no attributes. `_classify_edge_pair` used the proxy and so raised `AttributeError`
  under the two-phase flow, which meant the entire binary edge sweep could not run the way CI runs it,
  and nobody noticed because it had only ever been invoked directly. Instantiate the golden class
  directly when you need it before the device exists; `helpers/compressed_utils.py` documents the same
  workaround for the matmul golden.
- **A `math.*` function in a golden is a latent cat-B failure.** `math.sin` / `math.cos` *raise*
  `ValueError("math domain error")` on a non-finite input rather than returning NaN, so a golden using
  them turns a special-value probe into a test error. `_sin` and `_cos` both carried a comment
  asserting the input was "never not finite" — accurate until cat B. Prefer the `_torch_unary` helper,
  which is IEEE-correct and already applies the format-aware NaN rule.
