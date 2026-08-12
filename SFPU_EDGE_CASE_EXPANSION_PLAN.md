# SFPU Edge-Case Coverage — Plan for What Is Left

**Companion to:** [SFPU_EDGE_CASE_COVERAGE.md](SFPU_EDGE_CASE_COVERAGE.md) — the per-op audit
**Issue:** [tenstorrent/tt-metal#49739 — [LLK] SFPU testing edge cases](https://github.com/tenstorrent/tt-metal/issues/49739)
**Repo:** `tt-metal/tt_metal/tt-llk/tests/python_tests/`
**Scope:** Wormhole B0 and Blackhole. Quasar keeps its own inline stimulus definitions under
`quasar/` and is tracked separately.

**Revision 9 — 2026-08-12.** Phases 0–4 are **merged and verified against the code**, so they have
been removed from this document rather than left as ticked boxes; the four phase summaries and the
PR-3 plan that recorded them are deleted, and what survived of the PR-3 plan is folded in below.
This document is now only the work that is **not** done.

Where a mechanism landed, the code is the record. Everything asserted here was re-checked against
the tree at `ldjurovic/sfpu_edge_cases_phase_3` (main + the #52416 review follow-ups); the checks
are listed in §7 so the next revision can re-run them rather than trust this sentence.

---

## 1. What is already done (one paragraph, no detail)

Phases 0–1 ([#52172](https://github.com/tenstorrent/tt-metal/pull/52172)) pointed all four families
at the per-op domains in `_OP_DOMAIN_REGISTRY` instead of a positive-only `uniform(0.1, 1.1)`.
Phases 2–4 ([#52416](https://github.com/tenstorrent/tt-metal/pull/52416)) added the shared edge
metadata (`_OP_SINGULARITIES`, `_OP_EDGE_POINTS`, `FLOAT_SPECIALS`, `integer_specials()`,
`format_ulp()`), one builder (`edge_spec()` / `edge_pair_values()`) and three thin sweeps over the
existing drivers. **Verified in code:** 19 singularity entries, 43 edge-point entries, 97 unary
SFPU ops of which 94 are swept and 50 carry at least one deliberate edge value, `SPECIALS_READY_OPS`
empty, `specials_safe()` pinned by 107 host-side tests. The categories from the original plan stand
as: **A closed** for every op that has a boundary, **C closed** for the 5 ops whose kernels claim the
full int32 range, **D closed** for all 43 knees, **B measured and off**, **E blocked on C++**,
**F untouched**.

The one lesson worth carrying: this plan repeatedly assumed the *stimulus* was the hard part, and in
every case the binding constraint was the **golden** or the **kernel's own documented range**. Budget
the same three-way triage — golden correctness at the new values, comparison correctness in the new
regime, genuine kernel limits never previously measured — for every item below.

---

## 2. The remaining work, ordered by value

| # | Item | Blocked on | Size | Where |
|---|---|---|---|---|
| 1 | **Cat B goldens** — model non-finite *inputs* | per-op golden semantics | large, but divisible | §3 |
| 2 | **Blackhole verification** of the edge sweeps and `specials_safe()` | hardware access | one sweep | §4 |
| 3 | **Ternary edges** (`c → 0`, `lerp` weights) | `OperandSpecs` has no third operand | one data-model change + thin wrapper | §5 |
| 4 | **Per-op tolerances** for `xlogy` and `pow` | nothing | small | §6 |
| 5 | **Cat E** — unary shift amount | C++ `constexpr` → `TemplateParameter` | cross-language | §6 |
| 6 | **Cat F** — new harnesses for 11 kernels with no enum entry | new C++ source + golden each | large, per kernel | §6 |
| 7 | **Scalar tensor-operand edges** | cat B (item 1) | thin wrapper, zero value until then | §6 |
| 8 | **CI does not run any of this** | scheduling decision, not code | one workflow change | §8 |

Items 1 and 2 are the only ones that unblock others. Item 1 is half the remaining coverage; item 2
decides whether the table everything downstream reads is even correct on the second architecture.

---

## 3. Item 1 — cat B goldens (the largest remaining item)

### Why it is the whole story for half the op list

47 of the 97 unary SFPU ops are smooth everywhere — no knee, no pole — so `edge_spec()` returns
`None` for them and cat A and cat D contribute *nothing*. IEEE specials are their **only** deliberate
edge. Until the goldens model non-finite inputs, those 47 ops have no edge coverage at all beyond a
wider random sweep.

### Why it is not a sprinkle of xfails

The stimulus side is finished and the gating is already two-sided:

- `specials_safe(input, output, dest_acc)` — does the *pipeline* deliver a special intact. Measured;
  7 of 40 triples pass; pinned by `test_sfpu_domains.py`.
- `SPECIALS_READY_OPS` — does the *golden* define a result for one. Currently **empty**, which is
  byte-identical to the old global `False`.

Both must pass. Turning everything on regardless gives **272 failures out of 564 variants** — and
they are not the format matrix, which gates correctly. They are goldens that return `inf` where IEEE
says `nan`, and so on. 270 xfails would be a monument, not coverage.

### The plan

**One op per commit, and the op joins the gate as its last step.** For each op:

1. Decide the result at `+inf`, `-inf`, `NaN`, `+0.0`, `-0.0`. The authority is IEEE-754 where it
   speaks, then torch, then the kernel's own docstring — and where those disagree, record the
   disagreement rather than picking silently.
2. Extend the golden in `golden_generators.py` to produce it.
3. Add the op to `SPECIALS_READY_OPS` **with its reason string**.
4. Run the unary edge sweep for that op over the 7 safe triples and record whatever the hardware
   then disagrees about, as a non-strict xfail with an ISA cross-check.

**Suggested first ten**, chosen for being smooth (so cat B is their only edge), widely used, and
having an unambiguous IEEE answer:

| Order | Ops | Why first |
|---|---|---|
| 1 | `Identity`, `Neg`, `Abs` | Trivially defined at every special; they validate the *mechanism* rather than the semantics, so a failure here is the harness, not the golden |
| 2 | `Sqrt`, `Rsqrt`, `Reciprocal` | Already have cat-A poles driven, so the cat-B result composes with a known-good probe; `1/±inf = ±0` and `sqrt(-inf) = nan` are unambiguous |
| 3 | `Exp`, `Log` | `exp(-inf) = 0`, `exp(+inf) = +inf`, `log(0) = -inf`, `log(-x) = nan`; both are the base of several composed kernels, so their specials propagate |
| 4 | `Sin`, `Cos` | `sin(±inf) = nan` — the one place a LUT-based kernel is most likely to disagree, which makes it high-information |

Stop after the first group if the mechanism misbehaves; the point of ordering by triviality is that
the first three ops distinguish "the harness is wrong" from "the golden is wrong".

**Acceptance per commit:** the op is in `SPECIALS_READY_OPS` with a reason; the unary edge sweep runs
it on all 7 safe triples with no unexplained failure; every divergence is either an ISA-cross-checked
non-strict xfail or a fixed golden.

**Explicitly not in scope:** making the sweep green in one go, and block-float outputs — an `inf`
inside a block whose shared exponent is finite is not a value the format can express, so
`specials_safe()` excludes those rows on the golden's behalf and should keep doing so.

---

## 4. Item 2 — Blackhole verification

Stays high because **`specials_safe()`'s table is a measurement, not a derivation.** Its rules came
from driving the isinf/isnan predicates over the full 5×5 format matrix with no skips, on Wormhole.
The unpack paths differ on Blackhole, so the table may be wrong there in either direction — and every
cat-B decision in §3 reads it.

Three things to measure, in one sweep:

1. **Re-run the predicate matrix** on Blackhole and either confirm the 7 safe triples or make
   `specials_safe()` arch-keyed. `test_sfpu_domains.py::test_specials_safe_matches_measured_matrix`
   is where the verdict is pinned, so an arch-keyed table means parametrizing that test too.
2. **Resolve the `SFPMAD` signed-zero xfails**, which are a *testable prediction*: Blackhole's ISA
   page documents flush-to-sign-preserved zero where Wormhole documents flush-to-positive zero.
   - **XPASS** → the ISA reading is confirmed, and those cells can be narrowed to Wormhole-only,
     which is a real coverage gain on Blackhole.
   - **Still FAIL** → the documentation and the hardware disagree, which is worth more than the rest
     of this item combined.
3. **Report the shift and reduce xfails**, whose whole purpose is the Blackhole path; on Wormhole
   they are a deliberate no-op, so their reasons are currently unfalsified.

Two measurements added by the #52416 review follow-ups need a first run on **either** arch, and this
is the cheapest place to get them:

- **The accurate exp path over (16, 80].** The registry now carries the range bound (80/100/160) and
  `_APPROX_ACCURACY_MAX` the approximation bound (16/23/32), applied only for
  `ApproximationMode.Yes`. The accurate path over that region has never been isolated on hardware —
  it restores the domain #52172 shipped rather than inventing one, but it is unverified. If it does
  drift, the fix is a mode-conditional `custom_rtol`, **not** a re-narrowed registry entry, because
  the narrowing is what took the exponent-overflow region away from the accurate path in the first
  place.
- **Whether `-0.0` reaches DEST.** Drive datacopy with `custom(values=[0.0, -0.0])` and read the DEST
  sign bit on a `(Float16_b, *, dest_acc=No)` variant. See §5.2 of the coverage audit for why this
  decides how three ops' xfails should read.

**Acceptance:** the three edge sweeps run on Blackhole with every non-strict xfail resolved to XPASS
or FAIL and each outcome recorded; `specials_safe()` confirmed or arch-keyed; the two follow-up
measurements above recorded either way.

---

## 5. Item 3 — ternary edges (one data-model change)

The targets are `addcdiv` / `snake_beta` with `c → 0` — today `c` is pinned to `uniform(1, 2)`, so
the pole is *deliberately* unreachable — and `lerp` with weight `0`, `1` and `> 1`.

The blocker is that **`OperandSpecs` carries only `spec_A` and `spec_B`**, so there is nowhere to
register a third operand's singularity. `_ternary_default_specs` already works around it by reusing
B for C, with a comment saying so.

**Recommended:** add `spec_C: Optional[StimuliSpec] = None`, defaulting to a copy of `spec_B` in
`__post_init__`, exactly as `spec_B` already defaults to a copy of `spec_A`. Backward compatible with
all five consumers including `accuracy/accuracy_harness.py`. The alternative — a mapping keyed by
`Operand` — is cleaner in the abstract but rewrites every `.spec_A` / `.spec_B` access across five
consumers for one extra operand.

Then: `_OP_SINGULARITIES` gains `Operand.C` entries for the divisors of `addcdiv` and `snake_beta`,
`edge_spec(..., operand=Operand.C)` resolves through them, and a thin `test_sfpu_ternary_edges`
wrapper drives them. `_run_sfpu_ternary` **already accepts `spec_A` / `spec_B` / `spec_C`**, so the
wrapper itself is small.

Unlike the scalar wrapper, this one has real cat-A edges to drive the moment it exists — `c → 0` is a
genuine pole, not scaffolding.

**Acceptance:** `Operand.C` is expressible end to end (registry → `edge_spec` → driver); the accuracy
harness reports the same per-op `signed_ulp_error` as before; `test_sfpu_ternary_edges` drives `c = 0`
and the `lerp` weight boundaries; `edge_counterparts()` clips the other two operands to their
registered domains as it already does for binary.

---

## 6. Items 4–7 — the independent ones

### 6.1 Per-op tolerances for `xlogy` and `pow` (item 4, and the only one blocked on nothing)

Both are capped in the registry because their error outruns a fixed tolerance, with the measured
numbers already in the comments: `pow`'s tracks `b·ln a` into the shared exp approximation
(`3·ln3 = 3.30` clean, `4.83` → 4.9% off, `8.05` → 6.1% off against a 5% rtol), and `xlogy`'s
*absolute* error scales with `x` against a fixed atol. `CUSTOM_TOLERANCES` in `test_sfpu_unary.py` is
already the pattern to follow.

This also gates the scalar-axis widening (`|scalar| > 8`, `±tiny` / `±large` on the tensor operand),
which is only meaningful once the result can leave the range where the default bf16 tolerance says
anything. Start here if you want a small, self-contained commit.

### 6.2 Cat E — the unary shift amount (item 5)

`SHIFT_AMOUNT` is a `constexpr std::uint32_t SHIFT_AMOUNT = 3u` inside `call_unary_sfpu_operation`
(`helpers/include/sfpu_operations.h`), paired with `_int_shift_amount` on the golden side. Sweeping it
needs a new `TemplateParameter` plus matching golden plumbing — cross-language, not test wiring. The
Python side genuinely is written and reusable: `_SHIFT_EDGE_AMOUNTS` already covers
`{0..31, 32, 33, 40, 63, 100, 1000, −1, −5, −32, −1000}`, `_shift_reference` is the golden, and
`_build_paired_tile_override` is the delivery.

### 6.3 Cat F — the 11 kernels with no enum entry (item 6)

Confirmed still absent from `MathOperation`: `welfords`, `dropout`, `quant`, `cumsum`,
`reshuffle_rows`, `int_sum`, `tiled_prod`, `copy_dest_values`, `generalized_moe_gate_topk`,
`max_pool_indices`, `rand`. Each needs a new C++ source and golden, so none is reachable by the
shared mechanism.

`generic_moe_gate_topk` has come **off** this list: `test_sfpu_generic_moe_gate_topk.py` and
`sources/sfpu_generic_moe_gate_topk_test.cpp` both exist now.

| Priority | Kernels | Why |
|---|---|---|
| High | `welfords`, `int_sum`, `cumsum`, `tiled_prod` | Reduction family — reuse the reduce harness scaffolding, so four kernels share one harness cost |
| High | `quant` | Used in production quantization and has no correctness test at all |
| Medium | `dropout`, `rand` | RNG kernels; need a distribution-level assert, not an element-wise golden |
| Medium | `reshuffle_rows`, `copy_dest_values`, `max_pool_indices` | Data-movement / index kernels |
| Medium | `TopKLocalSort` / `TopKMerge` / `TopKRebuild` | Have enum entries but are perf-only; whole-op `topk` is tested, the stages are not |
| Medium | `AddInt32`, `SubInt32`, `AbsInt32`, `BitwiseNot` | Perf-only, blocked by the fast-tilize gap (tt-llk#495) — see the coverage audit's untested list |

### 6.4 Scalar tensor-operand edges (item 7 — do not start this one)

The `spec_A` hook exists on `_run_sfpu_binop_scalar`. The wrapper was **removed** in #52416's review
follow-ups and should stay removed until cat B opens: all five ops are `x (+|-|*|/) c` for a
compile-time `c`, so they are smooth in `x`, cat A and cat D contribute nothing, and a wrapper today
collects 20 nightly variants and skips all 20. The sketch and the reasoning sit in a comment where
the test was. It goes live on the same trigger as §3 — one of item 1's commits, not its own.

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
# expect: 19 43 97 50 0
python3 -m pytest test_sfpu_domains.py -q --noconftest   # expect 107 passed
```

The per-op tables in the coverage audit are generated from the same inventory; regenerate them the
same way rather than editing rows by hand.

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
"still open" divergences the ISA does not explain, cheap for an owner to adjudicate and expensive for
a test to keep guessing about:

1. **`signbit(-0.0)` returns 0** on the 6 combinations where `unpack_to_dest` is false — but see the
   coverage audit §5.2: the partition says the probe is probably not *delivered* there, which makes
   this a stimulus limitation rather than the kernel-contract bug it was first read as. Settle the
   delivery measurement (§4) before filing.
2. **`RsqrtCompat(0)` saturates to `1.7014118e38`** (`0x7F000000`) instead of returning `inf`, on all
   8 combinations — while plain `Rsqrt` over the same probe does not diverge. Two implementations of
   one function disagreeing at their shared pole, with nothing in the ISA prescribing either answer.
   This one is unaffected by the delivery question and can be filed now.

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
- **The pinned test environment drifts.** `tests/requirements.txt` pins `tt-exalens==0.3.29`; a venv
  carrying a different one fails at `conftest` import with a missing-symbol `ImportError`
  (`CallstackEntry`, `ElfFile` have both moved in later releases), which looks like a broken checkout
  rather than a stale venv. Check the pin before debugging the tree. Host-side tests can be run
  around it with `pytest --noconftest`.
