# SFPU Edge Cases — what PR 3 should do

**Companion to:** [SFPU_EDGE_CASE_EXPANSION_PLAN.md](SFPU_EDGE_CASE_EXPANSION_PLAN.md) (the master
plan; §12 is the backlog this document sequences)
**Issue:** [tenstorrent/tt-metal#49739](https://github.com/tenstorrent/tt-metal/issues/49739)
**Written:** 2026-08-11, against `ldjurovic/sfpu_edge_cases_2` @ `2d4e3f2`

## Where the quest stands

| PR | Phases | State |
|---|---|---|
| [#52172](https://github.com/tenstorrent/tt-metal/pull/52172) | 0–1 — point the sweeps at the per-op registered domains | in merge queue |
| [#52416](https://github.com/tenstorrent/tt-metal/pull/52416) | 2–4 — metadata, one builder, three sweeps | open; unary + binary + int done, ternary + scalar not |
| **PR 3 (this document)** | **finish Phase 4, and verify Phases 2–4 on Blackhole** | **not started** |
| PR 4+ | Phase 5 — cat B goldens, incrementally | not started |

Phase 4 planned four family wrappers. Two shipped. **PR 3's job is the other two plus the
arch verification that everything already shipped is currently missing** — after which the edge
mechanism is complete and the remaining work is all golden-side.

---

## §0. Prerequisite — the rebase, and two collisions to resolve deliberately

`sfpu_edge_cases_2` forked from `sfpu_edge_cases_1` **before that branch's last ten commits**, so
once #52172 merges this branch needs a rebase onto main. It will not be clean, and two of the
conflicts are the same fix made twice on both branches. **Keep this branch's version in both
cases** — they are supersets, not alternatives:

| Collision | Why this branch wins |
|---|---|
| **fp8 range ceilings** (`helpers/sfpu_domains.py`) | Both branches fix the transposed `MxFp8R`/`MxFp8P` entries. This branch *also* corrects `_FORMAT_MANTISSA_BITS` (e5m2 has 2 bits, e4m3 has 3) and routes `Fp8_e4m3` through named tiers (`_E4M3_FORMATS` / `_E5M2_AND_FLOAT16`) in all four format-sensitive builders. #52172's fix only adds the max-magnitude entry, so there `Fp8_e4m3` still lands in the *wide* default tier of `_exp_spec` / `_exp2_spec` / `_square_spec` |
| **binary declared-set hole** (`test_sfpu_binary.py`) | Both close it. This branch records a per-op *reason* per classification (`_REGISTERED_DEFAULT_STIMULI_OPS` is a `Dict`, not a set), declares the ops that bypass the shared driver (`_OPS_NOT_USING_SHARED_DRIVER`), and returns the routing decision from the same call that validates it (`_classify_stimuli_source`) so classification and routing cannot drift apart. It also raises `KeyError` rather than asserting, so it survives `python -O`. #52172's `_INT_ONLY_REGISTERED_OPS` should be **deleted**, not merged |

Also arriving from #52172 and worth a post-rebase smoke check, since Phases 2–4 sit on top of all
of it: the vectorised `_bfp_block_aware_compare`, the non-finite lattice fix, Bfp8_b quantization
before the binary golden, the registry-derived `STANDARD_SWEEP_OPS` (this branch still carries the
63-entry hand-written list), and the perf sweep's registry drive plus its bounded format axis.

**Acceptance:** collection across the five SFPU suites is unchanged at 9436, and the registry A/B
harness used in #52416 (1265 `(op, format)` entries plus a drawn-tensor hash) still reports
byte-identical stimuli.

---

## §1. Recommended scope for PR 3, and why in this order

Three items, in this sequence. The sequencing is not arbitrary — **item 1 can invalidate data that
items 2–3 and all of Phase 5 are built on**, so it goes first even though it writes the least code.

### 1. Verify Phases 2–4 on Blackhole (do this first)

Everything in #52416 except the reduce xfail and the scalar split was measured on Wormhole n150
only, and two parts are arch-sensitive *by construction*:

- **`specials_safe()`'s table is a measurement, not a derivation.** Its rules were established by
  driving the isinf/isnan predicates over the full 5×5 format matrix with no skips, on Wormhole.
  The unpack paths differ on Blackhole, so the table may be wrong there in either direction — and
  every later cat-B decision reads it. Re-run the same measurement and either confirm the table or
  make it arch-keyed.
- **The two converted xfails exist purely for the Blackhole path.** On Wormhole they are a
  deliberate no-op, so their reasons are currently unfalsified.

**The highest-information part is a testable prediction.** The `SFPMAD` signed-zero xfails
(`div(0, -x)`, `fmod`/`remainder` with a negative divisor, `xlogy(0, tiny)`) rest on Blackhole's
ISA page documenting flush-to-**sign-preserved** zero where Wormhole documents flush-to-**positive**
zero. They are non-strict xfails precisely so they can XPASS there. So:

- If they **XPASS** on Blackhole, the ISA reading is confirmed and those five cells can be narrowed
  to Wormhole-only, which is a real coverage gain on Blackhole.
- If they **still fail**, the ISA documentation and the hardware disagree, which is a finding worth
  more than the rest of this PR combined.

Either outcome is worth having, and it costs one sweep on a p100a.

**Acceptance:** the two edge sweeps run on Blackhole with every non-strict xfail resolved to XPASS
or FAIL and each outcome recorded; `specials_safe()` either confirmed or arch-keyed; the shift and
reduce xfails reported.

### 2. Scalar tensor-operand edges (small, genuinely just pending)

`_run_sfpu_binop_scalar` has no `spec_A` knob — the *scalar* axis is already swept
(`{0, 1, 2, −2, 8, 0.25}`, split presubmit/nightly in #52416), but the **tensor** operand is stuck
on the default. Add the parameter and a thin `test_sfpu_binop_scalar_edges` wrapper that passes
`edge_spec(...)` through it, exactly as the unary wrapper does.

**Explicitly not in scope:** widening the scalar axis to `±large` / `±tiny`. That needs a per-op
tolerance first (§2.4) — inputs are `uniform(-1, 1)` and `|scalar| ≤ 8` is what keeps every op's
result inside the range where the default bf16 tolerance means anything.

**Acceptance:** the wrapper collects, and any divergence is recorded as a non-strict xfail with a
reason, in the same shape as the unary and binary sweeps.

### 3. Ternary edges (needs a data-model change first)

The targets are `addcdiv` / `snake_beta` with `c → 0` — today `c` is pinned to `uniform(1, 2)`, so
the pole is *deliberately* unreachable — and `lerp` with weight `0`, `1` and `> 1`.

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
express a third operand the wrapper itself is thin.

**Acceptance:** `Operand.C` is expressible end to end (registry → `edge_spec` → driver), the
accuracy harness still reports the same per-op `signed_ulp_error`, and a
`test_sfpu_ternary_edges` wrapper drives `c = 0` and the `lerp` weight boundaries.

### Also land in PR 3, cheaply

- **A per-op cat-B opt-in list.** Phase 5 is going to be incremental by nature, and
  `_EDGE_SWEEP_SPECIALS` is a single global bool — so today the only options are "no specials" and
  "272 failures". Replace it with a declared set (`_SPECIALS_READY_OPS` or similar): an op enters
  only once its golden models non-finite inputs, and the sweep injects specials for that op alone.
  That converts Phase 5 from one impossible commit into a series of one-op commits, and it costs a
  few lines *now* versus a rework later. This is the single highest-leverage thing PR 3 can do for
  PR 4+.
- **Harden `specials_safe()` against the enum-truthiness trap.** It takes `dest_acc: bool`, and
  `bool(DestAccumulation.No)` is `True` — the member is truthy even though its `.value` is `False`.
  The one live call site correctly passes `dest_acc == DestAccumulation.Yes`, but
  `specials_safe_formats()` forwards the argument unchecked and has no callers yet. **Its first
  callers will be the ternary and scalar wrappers above**, i.e. this PR. Either accept
  `DestAccumulation` directly or assert the argument is not an enum.
- **Fix the two doc slips** corrected in this revision: the divergence count is **ten ops / 42
  cells**, not nine (verified: 5 unary ops over 20 cells, 5 binary over 22), and
  `SFPU_EDGE_CASES_SUMMARY.md`'s "Everything above is Wormhole" is stale now that the reduce xfail
  and scalar split were measured on p100a. That file is checked into `sfpu_edge_cases_2` as well as
  the docs branch, so it needs the same two corrections in both places.

---

## §2. Deferred to PR 4 and beyond, with what unblocks each

### 2.1 Cat B — goldens that model non-finite inputs (the largest item by far)

The stimulus side is **finished**: `specials_safe()` says where specials may be injected, and the
switch exists. Turning it on gives **272 failures out of 564**, because the torch-backed goldens do
not define a result for non-finite *inputs* — they return `inf` where the answer is `nan`, and so
on.

This is not a sprinkle of xfails, and it matters more than its position in the backlog suggests:
**47 of the 97 unary ops are smooth everywhere** — no knee, no pole — so cat B is the *entire*
edge story for half the op list. Until the goldens model it, those 47 ops have no deliberate edge
coverage at all.

The right unit of progress is "define the non-finite behaviour of the ten highest-value ops", not
"make the sweep green". With the per-op opt-in list from §1 in place, each op is its own small,
reviewable commit: decide the result at `+inf`, `-inf`, `NaN`, `+0.0` and `-0.0`; extend the
golden; add the op to the ready set; record whatever the hardware then disagrees about.

### 2.2 Category E — the unary shift amount

`SHIFT_AMOUNT` is a `constexpr std::uint32_t SHIFT_AMOUNT = 3u` inside
`call_unary_sfpu_operation` (`helpers/include/sfpu_operations.h:728`), paired with
`_int_shift_amount` on the golden side. Sweeping it needs a new `TemplateParameter` plus matching
golden plumbing — cross-language, not test wiring. The builder and `_shift_reference` genuinely are
written and reusable once the parameter exists, and `_SHIFT_EDGE_AMOUNTS` already covers
`{0..31, 32, 33, 40, 63, 100, 1000, −1, −5, −32, −1000}`.

### 2.3 Per-op tolerances for `xlogy` and `pow`

Both are capped in the registry because their error outruns a fixed tolerance, with the measured
numbers in the comments: `pow`'s tracks `b·ln a` into the shared exp approximation (`3·ln3 = 3.30`
clean, `4.83` → 4.9% off, `8.05` → 6.1% off against a 5% rtol), and `xlogy`'s *absolute* error
scales with `x` against a fixed atol. Their edges cannot be pushed further until the tolerance is
per-op. Independent of everything else here — `CUSTOM_TOLERANCES` in `test_sfpu_unary.py` is
already the pattern to follow.

### 2.4 Category F — genuinely new harnesses

Unchanged from the master plan §8. High priority five: `welfords`, `int_sum`, `cumsum`,
`tiled_prod`, `quant`. `generic_moe_gate_topk` is nearly done. Each needs a new C++ source and
golden, so none of it is reachable by the shared mechanism.

---

## §3. Two questions for kernel owners, independent of any PR

Both are "still open" divergences that the ISA does not explain, and both are cheap for an owner to
adjudicate and expensive for a test to keep guessing about:

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

## §4. Traps to know before starting

Every one of these has already cost time once.

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

---

## §5. The one thing that bounds the value of all of it

**None of this coverage is guarded by CI.** The broad unary profile runs in **no automated job on
any architecture**: every LLK pytest job either excludes `nightly` (pr-gate smoke, bit-exact) or
runs `--coverage`, under which the broad profile is skipped wholesale
([tt-llk#1435](https://github.com/tenstorrent/tt-llk/issues/1435)). That leaves the large majority
of the sweep's parametrizations running nowhere.

This predates all three PRs, and it means the coverage they add can regress silently. Either
`llk-e2e` needs a non-coverage companion group, or the broad profile has to stop being
coverage-gated. **It is worth filing before PR 3 rather than after** — a PR that adds arch
verification to a suite no job runs is worth strictly less than the same PR against a suite that is
actually scheduled.
