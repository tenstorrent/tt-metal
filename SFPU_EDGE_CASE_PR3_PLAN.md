# SFPU Edge Cases — what PR 3 should do

**Companion to:** [SFPU_EDGE_CASE_EXPANSION_PLAN.md](SFPU_EDGE_CASE_EXPANSION_PLAN.md) (the master
plan; §12 is the backlog this document sequences)
**Issue:** [tenstorrent/tt-metal#49739](https://github.com/tenstorrent/tt-metal/issues/49739)
**Written:** 2026-08-11 against `ldjurovic/sfpu_edge_cases_2` @ `2d4e3f2`
**Revised:** 2026-08-11 against `f6d752e`, after §0 and half of §1 landed in #52416

## Where the quest stands

| PR | Phases | State |
|---|---|---|
| [#52172](https://github.com/tenstorrent/tt-metal/pull/52172) | 0–1 — point the sweeps at the per-op registered domains | **merged** |
| [#52416](https://github.com/tenstorrent/tt-metal/pull/52416) | 2–4 — metadata, one builder, four sweeps | open; rebased onto post-#52172 main, and now carries the whole of §0 plus three of §1's items |
| **PR 3 (this document)** | **Blackhole verification, and ternary edges** | **not started** |
| PR 4+ | Phase 5 — cat B goldens, incrementally | not started |

The first revision of this document planned PR 3 as "finish Phase 4 plus the arch verification".
Most of it turned out to be cheaper to do inside #52416 than to hand to a follow-up, so **PR 3 is
now two items**: the one that needs hardware, and the one that needs a data-model change.

---

## §0. The rebase — done, with one collision this document did not predict

Recorded rather than planned, because the parts worth keeping are the ones that were surprises.

`sfpu_edge_cases_2` forked from `sfpu_edge_cases_1` before that branch's last ten commits, so
once #52172 merged this branch needed a rebase onto main. **Eight of its twenty commits dropped
out as already-upstream** — in every case main carried the same idea in a later form, so the
resolution was consistently "take main":

| Superseded on this branch | What main has instead |
|---|---|
| Bfp8_b judged on its lattice alone | lattice **or** tolerance, whichever passes, with a short-circuit |
| `narrowest_range_format()` + `for_op()` | `for_op_pipeline()` |
| hand-written 63-entry `STANDARD_SWEEP_OPS` | the registry complement of `BROAD_SWEEP_OPS` |
| a second Bfp4_b op list | Bfp4_b as a *format axis* over the whole broad op list |
| one flat `_NON_SFPU_UNARY_OPS` | per-family sets composed into it |

The two collisions this document called in advance both resolved as predicted — this branch's
fp8-ceiling superset (the `_E4M3_FORMATS` / `_E5M2_AND_FLOAT16` tiers and the corrected
`_FORMAT_MANTISSA_BITS`) and its declared-set version, with `_INT_ONLY_REGISTERED_OPS` deleted
rather than merged. One refinement: main's collection-time totality check against `_SFPU_BINARY_OPS`
is *complementary* to this branch's driver-side `_classify_stimuli_source()`, not replaced by it,
so it was kept and re-expressed in this branch's sets.

### The collision that was not predicted

`_exp_with_base_spec` computes `exp(0.5*x)`, and its docstring derives its bound as **double
`_exp_spec`'s**. #52172 rewrote it 32 → 160 against the old range-based ceiling of 80. This branch
moves `_exp_spec` to 16 for accuracy. Neither change conflicts textually, and the result was exp
capped at an argument of 16 while `exp_with_base` ran to 80 — the region this branch measured at
11–13% off golden — and the recalibration commit's own claim that the two share a ceiling became
false. Applying the docstring's rule to the new ceiling gives 32, which is also the pre-#52172
value.

**This is the general hazard, and it is worth carrying into §5:** a constant derived from another
by a rule stated only in prose drifts silently when the source moves. Nothing fails; the two just
stop agreeing. Same shape as the transposed-fp8 pair, and the reason `_FORMAT_MAX_MAGNITUDE` now
spreads `MX_FORMAT_MAX_NORMAL` rather than restating it.

### Acceptance, corrected

The first revision asked for "collection unchanged at 9436". That assumed the rebase was a no-op
merge, and it is not: **main widened two tests this branch also touches** (`eq_ne` 16 → 36,
`float_comparison` 32 → 72), so the number is **9496**, and the +60 is main's rather than the
rebase's. Anyone re-running this should expect 9496, or **9516** with §1's scalar wrapper included.

What did hold: no test lost (verified per-test, pre-rebase against post-), and registry stimuli
byte-identical across every `(op, format)` entry — resolved spec and drawn-tensor hash.

---

## §1. Landed in #52416 rather than deferred

Three of the four items the first revision assigned to PR 3 were cheaper to land with the rebase.

**Per-op cat-B opt-in.** `_EDGE_SWEEP_SPECIALS` is gone; `SPECIALS_READY_OPS` replaces it, in
`sfpu_domains` rather than in the unary suite because it is the golden-side half of a pair —
`specials_safe()` says the pipeline delivers specials intact, `SPECIALS_READY_OPS` says the golden
has an answer for them, and both edge sweeps need both. It starts empty, which is exactly the old
global `False`. **This is what makes Phase 5 tractable**, so PR 4+ can now proceed one op at a
time; see §3.1.

**`specials_safe()` hardening.** Confirmed live rather than theoretical: `DestAccumulation` is an
`Enum` whose members wrap `True`/`False`, so `bool(DestAccumulation.No)` is `True`. It now accepts
either a bool or the member and rejects anything else.

**Scalar tensor-operand edges — mechanism only, and the framing was wrong.** The first revision
called this "small, genuinely just pending", implying coverage was waiting on a knob. The knob was
genuinely missing and is now there (`spec_A` on `_run_sfpu_binop_scalar`, consumed by
`test_sfpu_binop_scalar_edges`). But **every variant skips, and will until cat B opens**: all five
ops are `x (+|-|*|/) c` for a compile-time `c`, so they are smooth in `x` — no pole, no knee, and
`edge_values()` returns nothing from cat A or cat D. The only edges the tensor operand has are the
cat-B specials. Budget this as scaffolding, not as a coverage gain.

**The two doc slips** the first revision listed are already correct in this branch's copy of
`SFPU_EDGE_CASES_SUMMARY.md`, which has since been removed from `sfpu_edge_cases_2` entirely —
the file lives here on the docs branch only, so there is no longer a second copy to keep in sync.

---

## §2. What PR 3 is now

### §2.1 Verify Phases 2–4 on Blackhole (still first, and now the only blocker)

Unchanged from the first revision, and it stays first for the same reason: **`specials_safe()`'s
table is a measurement, not a derivation.** Its rules were established by driving the isinf/isnan
predicates over the full 5×5 format matrix with no skips, on Wormhole. The unpack paths differ on
Blackhole, so the table may be wrong there in either direction — and every later cat-B decision
reads it. Re-run the same measurement and either confirm the table or make it arch-keyed.

The two converted xfails exist purely for the Blackhole path; on Wormhole they are a deliberate
no-op, so their reasons are currently unfalsified.

**The highest-information part is a testable prediction.** The `SFPMAD` signed-zero xfails
(`div(0, -x)`, `fmod`/`remainder` with a negative divisor, `xlogy(0, tiny)`) rest on Blackhole's
ISA page documenting flush-to-**sign-preserved** zero where Wormhole documents flush-to-**positive**
zero. They are non-strict xfails precisely so they can XPASS there:

- If they **XPASS**, the ISA reading is confirmed and those five cells can be narrowed to
  Wormhole-only, which is a real coverage gain on Blackhole.
- If they **still fail**, the documentation and the hardware disagree, which is worth more than
  the rest of this PR combined.

Either outcome is worth having, and it costs one sweep on a p100a.

**Acceptance:** the three edge sweeps run on Blackhole with every non-strict xfail resolved to
XPASS or FAIL and each outcome recorded; `specials_safe()` either confirmed or arch-keyed; the
shift and reduce xfails reported.

### §2.2 Ternary edges (needs a data-model change first)

Unchanged. The targets are `addcdiv` / `snake_beta` with `c → 0` — today `c` is pinned to
`uniform(1, 2)`, so the pole is *deliberately* unreachable — and `lerp` with weight `0`, `1` and
`> 1`.

The blocker is that **`OperandSpecs` carries only `spec_A` and `spec_B`**, so there is nowhere to
register a third operand's singularity. `_ternary_default_specs` already works around it by reusing
B for C, with a comment saying so. Two ways out:

- **Add `spec_C: Optional[StimuliSpec] = None`, defaulting to a copy of `spec_B`.** Backward
  compatible with all five consumers including `accuracy/accuracy_harness.py`, and mirrors how
  `spec_B` already defaults to a copy of `spec_A`. Recommended.
- Generalise to a mapping keyed by `Operand`. Cleaner in the abstract, but it rewrites every
  `.spec_A` / `.spec_B` access across five consumers for one extra operand.

Then `_OP_SINGULARITIES` needs `Operand.C` entries for the divisor of `addcdiv` and `snake_beta`,
and `edge_spec(..., operand=Operand.C)` has to resolve through them.

`_run_sfpu_ternary` **already accepts `spec_A` / `spec_B` / `spec_C`**, so once the data model can
express a third operand the wrapper itself is thin. Unlike the scalar wrapper above, this one has
real cat-A edges to drive the moment it exists — `c → 0` is a genuine pole.

**Acceptance:** `Operand.C` is expressible end to end (registry → `edge_spec` → driver), the
accuracy harness still reports the same per-op `signed_ulp_error`, and a `test_sfpu_ternary_edges`
wrapper drives `c = 0` and the `lerp` weight boundaries.

---

## §3. Deferred to PR 4 and beyond, with what unblocks each

### 3.1 Cat B — goldens that model non-finite inputs (the largest item by far)

The stimulus side is **finished**, and as of #52416 so is the sequencing: `specials_safe()` says
where specials may be injected, `SPECIALS_READY_OPS` says which ops have a golden that can receive
them, and an op joins the second set on its own. Turning everything on at once still gives **272
failures out of 564**, because the torch-backed goldens do not define a result for non-finite
*inputs* — they return `inf` where the answer is `nan`, and so on.

The right unit of progress is "define the non-finite behaviour of the ten highest-value ops", not
"make the sweep green". **47 of the 97 unary ops are smooth everywhere** — no knee, no pole — so
cat B is the *entire* edge story for half the op list. Each op is now its own small, reviewable
commit: decide the result at `+inf`, `-inf`, `NaN`, `+0.0` and `-0.0`; extend the golden; add the
op to `SPECIALS_READY_OPS` with its reason; record whatever the hardware then disagrees about.

Note that the scalar wrapper from §1 goes live on the same trigger, so its five ops come along for
free once their goldens are done.

### 3.2 Category E — the unary shift amount

`SHIFT_AMOUNT` is a `constexpr std::uint32_t SHIFT_AMOUNT = 3u` inside
`call_unary_sfpu_operation` (`helpers/include/sfpu_operations.h:728`), paired with
`_int_shift_amount` on the golden side. Sweeping it needs a new `TemplateParameter` plus matching
golden plumbing — cross-language, not test wiring. The builder and `_shift_reference` genuinely are
written and reusable once the parameter exists, and `_SHIFT_EDGE_AMOUNTS` already covers
`{0..31, 32, 33, 40, 63, 100, 1000, −1, −5, −32, −1000}`.

### 3.3 Per-op tolerances for `xlogy` and `pow`

Both are capped in the registry because their error outruns a fixed tolerance, with the measured
numbers in the comments: `pow`'s tracks `b·ln a` into the shared exp approximation (`3·ln3 = 3.30`
clean, `4.83` → 4.9% off, `8.05` → 6.1% off against a 5% rtol), and `xlogy`'s *absolute* error
scales with `x` against a fixed atol. Their edges cannot be pushed further until the tolerance is
per-op. `CUSTOM_TOLERANCES` in `test_sfpu_unary.py` is already the pattern to follow.

**This also gates the scalar axis widening** that §1 explicitly left out: `|scalar| > 8` and
`±tiny` / `±large` on the tensor operand are only meaningful once the result can leave the range
where the default bf16 tolerance says anything.

### 3.4 Category F — genuinely new harnesses

Unchanged from the master plan §8. High priority five: `welfords`, `int_sum`, `cumsum`,
`tiled_prod`, `quant`. `generic_moe_gate_topk` is nearly done. Each needs a new C++ source and
golden, so none of it is reachable by the shared mechanism.

---

## §4. Two questions for kernel owners, independent of any PR

Both are "still open" divergences that the ISA does not explain, and both are cheap for an owner to
adjudicate and expensive for a test to keep guessing about. **Neither has been filed** — worth
doing regardless of PR 3's timing.

1. **`signbit(-0.0)` returns 0**, where the kernel's own docstring promises 1 ("logical-shift the
   fp32 bit pattern right by 31 … incl. `-0.0`"). Unlike `sign` and `heaviside`, this op claims to
   read the sign bit *directly*, so either the claim or the implementation is wrong. `-0.0` is
   confirmed to reach the device, verified host-side. This reads as a kernel-contract bug rather
   than a hardware one.
2. **`RsqrtCompat(0)` saturates to `1.7014118e38`** (`0x7F000000`) instead of returning `inf`, on
   all 8 combinations — while plain `Rsqrt` over the same probe does not diverge. Two
   implementations of one function disagreeing at their shared pole, with nothing in the ISA
   prescribing either answer.

---

## §5. Traps to know before starting

Every one of these has already cost time once.

- **A constant derived from another by a prose rule will drift.** `_exp_with_base_spec` is
  documented as double `_exp_spec`'s; two branches moved the two halves independently and nothing
  failed, they just stopped agreeing (§0). If a value is derived, derive it in code or assert the
  relation — a docstring is not a constraint.
- **`exclude_intervals()` is not stimulus-neutral.** It always rewrites its result into the
  `intervals` form, and that sampler consumes **two** `torch.rand` draws per element where the
  plain `low`/`high` path consumes one. So `uniform(1, 8)` and `intervals=[(1, 8)]` are the same
  distribution and different numbers at the same seed, and **declaring a new hole in
  `_SFPU_UNDEFINED_RANGES` re-rolls that op's entire stimulus set** even when the subtraction
  removes nothing. Any change that adds a hole is signing up for that op's triage; keep edge
  metadata off this path.
- **Do not route edge specs through `for_op_pipeline`.** Its `_tighter_spec` measures a domain with
  `_spec_span`, which falls back to `spec.high - spec.low` — `None - None` for a values-list spec.
  Nothing hits it today because the paths are separate; the obvious-looking unification raises
  `TypeError`.
- **`StimuliSpec.custom` cannot carry integer extremes.** `CustomStrategy.generate_face` clamps
  through `_get_integer_bounds`, which returns `info.min + 1`, so a spec asking for `INT32_MIN`
  silently yields `INT32_MIN + 1`. Integer edges go through `src_A_override` as a raw tensor —
  `_build_int_extremes_src` and `_build_shift_edge_case_src` are the pattern.
- **Enum members are not their values.** `DestAccumulation` wraps `True`/`False`, so
  `bool(DestAccumulation.No)` is `True`. `specials_safe()` is now hardened against this, but the
  enum is passed around widely and the next `if dest_acc:` will be wrong in the same way.
- **`format_ulp` returns a lower bound for block-float formats**, because the real step is set by
  the exponent shared across the 16-element block. Safe direction, but a block-float probe *pair*
  cannot be assumed distinct.
- **`TestConfig` calls `shutil.rmtree()` on the fixed path `/tmp/tt-llk-build` at session setup.**
  Any second pytest session on the same host — including a one-op `-k` run started to triage
  something — deletes the build tree out from under a running sweep. The victim reports
  `ld: cannot open output file`, which in a log reads exactly like a real kernel bug. This produced
  two phantom failures during Phase 0 and will recur, because this work is triage-heavy by nature.
  Worth fixing separately: key the artefact root by session, or take the existing
  `/tmp/tt-llk-build-shared.lock` around the rmtree.
- **The pinned test environment drifts.** `tests/requirements.txt` pins `tt-exalens==0.3.29`; a
  venv carrying a different one fails at `conftest` import with a missing-symbol `ImportError`
  (`CallstackEntry`, `ElfFile`), which looks like a broken checkout rather than a stale venv.
  Check the pin before debugging the tree.

---

## §6. The one thing that bounds the value of all of it

**None of this coverage is guarded by CI.** The broad unary profile runs in **no automated job on
any architecture**: every LLK pytest job either excludes `nightly` (pr-gate smoke, bit-exact) or
runs `--coverage`, under which the broad profile is skipped wholesale. That leaves the large
majority of the sweep's parametrizations running nowhere.

This predates all three PRs, and it means the coverage they add can regress silently. Either
`llk-e2e` needs a non-coverage companion group, or the broad profile has to stop being
coverage-gated. **It is worth filing before PR 3 rather than after** — a PR that adds arch
verification to a suite no job runs is worth strictly less than the same PR against a suite that is
actually scheduled.

**One citation to check first.** Both this document's earlier revision and the live skip reason in
`test_sfpu_unary.py` attribute the coverage-gating to
[tt-llk#1435](https://github.com/tenstorrent/tt-llk/issues/1435). That issue is open, but its title
is *"when test_eltwise_unary_sfpu.py is ran after test_eltwise_binary, it fails on missmatch"* —
test ordering, not coverage. Either the citation is wrong and has propagated into the source, or
the issue has been repurposed in its comments. Worth resolving before filing anything that cites
it, since the skip reason in the suite points readers there.
