# SFPU Edge-Case Coverage — Minimal-Code Expansion Plan

**Companion to:** [SFPU_EDGE_CASE_COVERAGE.md](SFPU_EDGE_CASE_COVERAGE.md)
**Issue:** [tenstorrent/tt-metal#49739 — [LLK] SFPU testing edge cases](https://github.com/tenstorrent/tt-metal/issues/49739)
**Goal:** close the edge-case gaps catalogued in the coverage audit while adding the **least possible code** — no per-op edge tests, no duplicated stimulus lists.
**Repo:** `tt-metal/tt_metal/tt-llk/tests/python_tests/`
**Scope:** Wormhole B0 and Blackhole. Quasar and its `quasar/` suite are **out of scope for
this plan** — it keeps its own inline stimulus definitions and is tracked separately.

**Revision 2 — 2026-08-05.** Rewritten against the current tt-llk test infra. The first
revision (2026-07-23) was written against a tree that has since been restructured by
[#50602 *Unify SFPU test dispatch*](https://github.com/tenstorrent/tt-metal/pull/50602),
which merged two drivers away, and by the Phase 0 work on branch
`ldjurovic/sfpu_edge_cases_1`, which landed part of this plan and changed what the later
phases have to do. §2 lists the infra deltas; §10 tracks status per phase. What changed
substantively versus revision 1:

- **Phase 0 has been implemented and is much larger than one reroute.** Widening the unary
  domain broke the Bfp8_b golden and the Bfp8_b comparison, and exposed three registry
  bounds that were range-correct but accuracy-wrong. See §3.
- **Phase 3 (`compare_special` flag) is dropped** — `passed_test` is already NaN-aware on
  every path, and `torch.isclose` already accepts matching signed infinities. Nothing to add.
- **The binary driver has no `spec_B`.** Per-operand edge injection now goes through
  `_paired_two_tile_spec()` or `src_A_override`, not `Operand.B`. See §6a.
- **`StimuliSpec.custom` silently clamps integers to `INT_MIN + 1`**, so cat C cannot be
  delivered through the spec system at all. See §6c — this is a new hard blocker.
- **IEEE specials cannot be injected on every (format, dest_acc) pair**: bf16→fp32 dest
  unpack destroys `-inf`/`NaN`. See §7a. Any family sweep that ignores this produces a wall
  of false failures.
- **A new Phase 1 exists**: rerouting binary/ternary/scalar onto the registry, the direct
  analogue of Phase 0 for the other three families. Revision 1 folded this into phase 4 and
  underestimated it, because those three drivers do not accept per-operand specs at all yet.

---

## 1. Guiding principle: one mechanism per gap-*category*, not per op

The 394 rows in the coverage audit look like 394 problems, but every `No` / `Excluded` row
falls into one of **six edge categories**, and each category is closed by **one shared
mechanism** that all ~150 ops reuse through the *existing* drivers:

| # | Gap category (from the audit) | Shared mechanism that closes it |
|---|-------------------------------|---------------------------------|
| A | Domain boundaries (`recip`/`div`/`log` at 0, `asin`/`acos`/`atanh`/`erfinv` at ±1, `acosh` at 1, `sqrt`/`rsqrt` ≤0, `pow` base≤0, `xlogy` y≤0) | **Auto-derive boundary probes from the data already in `_SFPU_UNDEFINED_RANGES`** |
| B | IEEE specials (±inf, NaN, +0.0, −0.0) into float ops | **One shared `FLOAT_SPECIALS` list injected by format class, gated on the (format, dest_acc) pairs that preserve them** |
| C | Integer extremes (INT32_MIN/MAX, UINT32_MAX, 0, −1, overflow) | **One shared `INT32_SPECIALS` / `UINT32_SPECIALS` list — delivered as a raw override tensor, not a `StimuliSpec` (see §6c)** |
| D | Op-specific discrete edges (knees, thresholds, exact-tie rounding) | **A small `_OP_EDGE_POINTS` table — only for points that aren't already a domain boundary** |
| E | Shift-amount limits for the **unary** shift ops, and the Blackhole shift/reduce arch-skips | **Reuse the existing binary shift edge-case builder; unblock the two tracked HW bugs** |
| F | Entirely-untested kernels (`welfords`, `dropout`, `quant`, …) and perf-only ops (`TopK*` stages) | **Genuinely new harness per kernel — explicitly out of the cheap path (§9)** |

Categories **A–D** are closed by **one new metadata block + one stimulus builder + ~4 thin
test wrappers**, a few hundred lines total that scale to every op. Category **E** is a small
refactor plus two HW-bug dependencies. Category **F** is the only place real new code is
unavoidable, and we scope/prioritize rather than write it here.

**But categories A–D are all downstream of a precondition that revision 1 treated as
free:** a driver can only inject an edge value if it *runs the op over a domain wide enough
that the golden and the comparison agree there*. Phase 0 proved that precondition is not
free — see §3.

---

## 2. Infra deltas since revision 1 (what the plan is now written against)

| Change | Where | Effect on this plan |
|---|---|---|
| `test_sfpu_binary_bcast.py` **deleted**, merged into `test_sfpu_binary.py` (`test_sfpu_binary_bcast`, `sfpu_binary(broadcast_type=…)`) | #50602 | §8 needs **4** family wrappers, not 5; binary's wrapper covers broadcast for free |
| `test_ttnn_where.py` **deleted**, merged into `test_sfpu_ternary.py` (`test_ttnn_where`, `test_ttnn_where_mcw`); `sources/ttnn_where_test.cpp` removed | #50602 | audit citations for the `where` rows move to `test_sfpu_ternary.py` |
| `sfpu_binary()` reads **both operands out of `buffer_A`** (tile0 = in0, tile1 = in1) and exposes `spec_A` + `src_A_override`, **no `spec_B`** | `test_sfpu_binary.py:159` | rewrites cat A/B delivery for binary — §6a |
| `_paired_two_tile_spec(a_face, b_face)` — per-operand faces inside one spec | `test_sfpu_binary.py:80` | **this is the per-operand mechanism**; already used by mask / isclose / eq_ne |
| `passed_test` is NaN-aware on the generic path, the Bfp8_b path and all three block-aware compares | `helpers/utils.py:521`, `:628` | **Phase 3 of revision 1 is obsolete** |
| `_OP_DOMAIN_REGISTRY` values may now be **callables** (`Reciprocal: _reciprocal_spec`), i.e. format-sensitive domains | `helpers/sfpu_domains.py:202` | `edge_spec()` must resolve through `for_op()`, never index the registry directly |
| `narrowest_range_format()` added — domain is bounded by the narrowest float format in the pipeline, not the input format | `helpers/sfpu_domains.py:71` | edge probes must be range-checked the same way |
| Shift edge test grew a third op (`SfpuElwLogicalRightShift`) and a `_SHIFT_EDGE_OPS` list; the Blackhole skip is an **inline `pytest.skip` in the test body**, not a `@skip_for_blackhole` decorator | `test_sfpu_binary.py:650`, `:746` | rewrites cat E's action item |
| Dedicated `test_sfpu_binary_int_shift_int32_min_unsupported` xfail test | `test_sfpu_binary.py:774` | the xfail convention cited by §7 is now its own test, not an annotation |
| `accuracy/accuracy_harness.py` is a **second consumer of `for_op()`**, measuring per-op `signed_ulp_error` | `accuracy/accuracy_harness.py:189` | any registry bound change now moves the accuracy sweep too — a coupling to state in the commit |

Two audit findings are **unchanged** and still the load-bearing ones:

- **Finding #2** — `test_sfpu_binary.py`, `test_sfpu_ternary.py`, `test_sfpu_binop_scalar.py`
  still do not import `sfpu_domains.py`. Confirmed: the registry holds **115 ops**, of which
  `ALL_MATHOPS` (31) + `DOMAIN_MATHOPS` (63) consume 94. The remaining 21 include
  `SfpuElwadd/sub/mul/div/pow/rsub`, `SfpuXlogy` and all three shift ops — **registered
  domains that no test reads.**
- **Finding #4** — `_get_integer_bounds` still returns `info.min + 1`
  (`helpers/stimuli_generator/utils.py:45`), so INT32_MIN is unreachable through any spec.

---

## 3. Phase 0 — run the unary sweep over each op's real domain ✅ **DONE** (`ldjurovic/sfpu_edge_cases_1`)

**Revision 1 scoped this as "≈0 new lines: stop the positive-only default".** That was the
right target and the wrong cost estimate. Rerouting `spec_A` is 5 lines; making the reroute
*pass* took four more commits, because the audit's finding #1 was hiding three other defects
behind it. A test suite that only ever sees `uniform(0.1, 1.1)` cannot distinguish a correct
golden from a wrong one, a correct comparison from a wrong one, or an accurate kernel from an
inaccurate one — the inputs are too benign to separate them. Widening the domain separates
all three at once.

So **Phase 0 is now defined as: reroute the unary sweep onto the registry, and fix
everything the wider domain exposes.** Nine parts, all landed — 0a–0g plus the two closing
items 0h and 0i (§3.2):

| | Part | Commit |
|---|---|---|
| 0a | Default `eltwise_unary_sfpu`'s `spec_A` to `exclude_undefined(mathop, for_op(mathop, domain_format).spec_A)` instead of `default_spec_for_format`'s positive-only `uniform(0.1, 1.1)` | `f8590d5` |
| 0b | Register the 5 sweep ops that had **no registry entry** (`Tanhshrink`, `Floor`, `Ceil`, `Trunc`, `Frac`), and let `for_op` raise `KeyError` on a miss rather than falling back — a silent fallback would restore the very default being removed | `f8590d5` |
| 0c | Select the domain by `narrowest_range_format(input, output)`, not the input format: the sweep pairs every input with every output, and the **output** is often the binding constraint (exp over (−100, 80) peaks at ~5.5e34 — fine into Float32, saturates a Float16 output, where HW returns `nan` against an `inf` golden) | `f8590d5` |
| 0d | **Golden fix:** `UnarySFPUGolden` inlined a partial copy of `quantize_input_to_unpack_format` covering Bfp2_b/Bfp4_b/MX but **not Bfp8_b**, so for Bfp8_b inputs the golden ran on values the hardware never saw. Smooth ops absorbed it in tolerance; `floor`/`ceil`/`trunc`/`frac` turn a sub-ULP step across an integer into a full 1.0 error | `14f2133` |
| 0e | **Comparison fix:** `passed_test` routed Bfp4_b/Bfp2_b to `_bfp_block_aware_compare` but let Bfp8_b fall through to a flat `isclose`, though it is equally a block format (7 magnitude bits, exponent shared across 16 elements). Routing it to the lattice check *alone* was wrong in the other direction — 7 bits make a lattice step *tighter* than the SFPU's own approximation error. **Accept either criterion**: quantization dominates far below the block max, approximation error dominates near it | `14f2133`, `a7c0312` |
| 0f | **Recalibrate three registry bounds** that were chosen for representable range and had never actually been executed: `exp` high 80 → 16 (relative condition number equals the argument, so error grows linearly with x; at 80 the largest outputs land 11–13% off against a 5% rtol), `exp2` → 23 (= 16/ln2), `reciprocal` → format-sensitive (a 1000:1 ratio inside a 16-element block quantizes the smallest elements to zero → golden `inf`; hold block-float inputs to 10:1) | `8841e51` |
| 0g | **Record the residual real accuracy limit as an xfail** rather than loosening a tolerance or shrinking the domain away from where an approximation is most worth testing: approximate `exp` overshoots by a systematic ~5.7% (peak 6.75%) above an argument of ~8. Added via `request.node.add_marker` so the case still runs and reports XPASS if the kernel tightens; `strict=False`, matching the INT32_MIN shift xfail convention | `507e229` |

**Result of 0a–0g:** `test_sfpu_unary.py` on Wormhole — **5108 passed, 334 skipped, 6 xfailed**.
31 ops gained their negative branch, piecewise knees and saturation tails; 31 previously
dead registry entries became live.

**The general lesson for phases 1–5.** Every later phase widens stimuli further, so budget
for the same three-way fallout each time: *golden* correctness at the new values, *comparison*
correctness in the new regime, and *genuine* kernel accuracy limits that were simply never
measured. Revision 1 gave this one line (§7 "golden readiness"); it is closer to half the work.

### 3.1 Done (on `ldjurovic/sfpu_edge_cases_1`)

0a–0g above. Five commits, ~150 lines, all verified on **Wormhole**.

### 3.2 ✅ The two closing items (were open, now done)

| | Item | Outcome |
|---|---|---|
| **0h** | **Verify on Blackhole.** 0f recalibrated `exp`/`exp2`/`reciprocal` against *measured Wormhole error* and 0e's Bfp8_b either-criterion compare is tuned to measured WH approximation error, so neither number was known to be arch-independent. If BH had wanted different bounds, the registry entries would have become arch-sensitive — a change to the *shape* of 0f, not just its constants. | **Green, no arch-sensitivity needed.** `test_sfpu_unary.py` on Blackhole (p300a): **4270 passed, 1174 skipped, 4 xfailed, 0 failed** (14 min). The recalibrated domains and the Bfp8_b compare hold on BH unchanged, so 0f's constants stay arch-independent and no registry entry needs an arch axis. |
| **0i** | **Merge `ALL_MATHOPS` into `DOMAIN_MATHOPS`**, collapsing `test_eltwise_unary_sfpu_float` and `test_eltwise_unary_sfpu_domain` into one test. | **Done, and provably behaviour-preserving** — see §3.3. |

#### Why BH skips so much more than WH, and why that is not a 0h failure

WH reports 5108 passed / 334 skipped; BH reports 4270 / 1174. The gap is entirely the two
architecture guards, not lost coverage from Phase 0:

- `_skip_bh_unsupported_float_combo` — on BH at `dest_acc=No`, neither a Float16 input nor
  `Float32->Float16` is supported.
- `_skip_bh_unless_fp32` — on BH at `dest_acc=No`, only `Float32->Float32` is supported.

BH also reports **4** xfails against WH's 6, for the same reason: two of the three
`_APPROX_EXP_ACCURACY_XFAIL` combinations have a Float16 input at `dest_acc=No` and are
skipped on BH before the xfail can apply. Nothing regressed — the approximate-`exp` accuracy
limit 0g recorded reproduces on BH wherever it is reachable.

#### One finding that is *not* a Phase 0 item: the build dir is not run-isolated

The first BH run reported 2 failures (`Atanh`, `Tanhshrink`, both at `[128, 256]` with
`dest_acc=Yes`). Both were `ld: cannot open output file`, not numerical, and neither
reproduced. Cause: `TestConfig` does
`shutil.rmtree(ARTEFACTS_DIR)` at session setup ("always have a fresh build when compiling")
against a **fixed** path, `/tmp/tt-llk-build`. Any second pytest session on the same host —
even a `-k` run of one op — deletes the build tree out from under a running sweep, and the
victim surfaces it as a link failure attributed to whichever variant was linking.

Worth filing separately (key the artefact root by session, or take the existing
`/tmp/tt-llk-build-shared.lock` for the rmtree). It is called out here because it is a live
trap for Phases 1–4: those phases mean many more triage re-runs, and this failure mode looks
exactly like a real kernel bug in the log.

### 3.3 How 0i was verified

The merge is **stimulus-identical**, not merely equivalent-looking, and that is checkable
rather than asserted:

- **Same case set.** Collection is 5448 tests before and after, and the merged test carries
  5368 of them — 4864 broad + 504 standard, matching the two old tests exactly
  (`4864 + 504 = 5368`). No case gained, lost or duplicated.
- **Same stimuli.** The old `_domain` test passed `spec_A=exclude_undefined(op, for_op(op,
  formats.input_format).spec_A)` explicitly; the merged test lets the driver default apply,
  which resolves the domain through `narrowest_range_format(input, output)`. For the standard
  profile's four format pairs those are the *same* format — Float16_b and Float32 both share
  bfloat16's exponent range, so `narrowest_range_format` ties and resolves to its first
  argument, the input. Verified directly for all four pairs. The broad profile already used
  the driver default, so it is untouched by construction.
- **Same result on hardware.** Blackhole, merged: **4270 passed, 1174 skipped, 4 xfailed,
  0 failed** — identical to the 0h baseline above, case for case.

What the merge actually changes, beyond deleting a test: the two lists are now *coverage
profiles* (`BROAD_SWEEP_OPS` / `STANDARD_SWEEP_OPS`) rather than "ops with a domain" versus
"ops without", which is what they had silently become. A module-level
`_assert_sweep_profiles_disjoint()` now fails at **collection** if an op lands in both
profiles or has no registered domain. Exhaustiveness is still unchecked and deliberately
documented as such: `_OP_DOMAIN_REGISTRY` mixes unary, binary and reduce entries, so there is
no authoritative "every unary op" list to check against. Recording an op's arity in the
registry would close it, and that is Phase 1 work.

**Explicitly out of Phase 0** (so the boundary is unambiguous):

- Rerouting binary / ternary / scalar onto the registry → **Phase 1** (§4). Phase 0 is
  unary-only by definition.
- Injecting boundary probes, IEEE specials or integer extremes → **Phases 2–4**. Phase 0
  only widens the *random domain*; it lands *near* knees, never *on* them.
- The approximate-`exp` accuracy bug itself. 0g records it as an xfail; whether the kernel
  gets fixed is a kernel-side question tracked separately, and the xfail is `strict=False`
  so a fix surfaces as XPASS.
- Per-session build-artefact isolation (above) — a test-infra bug this work surfaced, not a
  coverage item.

**Phase 0 is closed:** 0h green on Blackhole with no arch-specific bounds required, and 0i
merged with identical hardware results on both arches.

---

## 4. Phase 1 — the same reroute for binary / ternary / scalar ⬜ **NOT STARTED**

This is audit finding #2, and it is now the single largest cheap win left. Revision 1 folded
it into phase 4; it deserves its own phase because the three drivers cannot currently express
what it needs.

**The gap.** Ten binary ops (`SfpuElwadd/sub/mul/div/pow/rsub`, `SfpuXlogy`, three shifts)
have registered domains — including the `_SFPU_UNDEFINED_RANGES` holes that *are* the cat-A
boundaries (`SfpuElwdiv` divisor `(−1e-6, 1e-6)`, `SfpuXlogy` B `(−inf, 1e-6)`, `SfpuElwpow`
A `(−inf, 1e-6)`) — and no test reads them. Meanwhile ternary hard-codes
`uniform(−1, 1)` / `uniform(1, 2)` (`test_sfpu_ternary.py:47`) and scalar binops hard-code a
single scalar per op (`_SCALAR_BITS`).

**Work, in dependency order:**

1. **Binary** — default `sfpu_binary`'s `spec_A` to the registry the way `eltwise_unary_sfpu`
   now does, via `_paired_two_tile_spec(for_op(op, fmt).spec_A, for_op(op, fmt).spec_B)` so
   the two operands get their *own* domains. `OperandSpecs` already carries `spec_B` and
   deep-copies `spec_A` into it when omitted (`sfpu_domains.py:51`), so the per-operand data
   exists today. Expect the Phase 0 fallout pattern: div/pow/xlogy over signed domains will
   surface golden gaps.
2. **Ternary** — `_run_sfpu_ternary` takes **no spec parameters at all**. Add
   `spec_A`/`spec_B`/`spec_C`, defaulting to the registry, keeping the current hard-coded
   specs as the fallback only for ops with no entry. This is a prerequisite for *any* ternary
   edge work (§8), so it lands here even if the default does not change for every op.
3. **Scalar binop** — parametrize the scalar instead of fixing it: `_SCALAR_BITS` becomes a
   swept axis. Note `ScalarDiv` inverts the divisor **on the host** at compile time, so a
   `d = 0` case is not reachable on device and should be recorded as N/A, not xfailed.
4. **Un-comment the float comparison ops.** `SfpuElwLt/Gt/Le/Ge` are currently commented out
   of `test_sfpu_binary_float` because independent random draws produce near-ties that diverge
   from the total-order golden. That is not a reason to drop them — it is the cat-D tie edge
   asking to be driven deliberately, with `_paired_two_tile_spec` giving exact ties and
   exact-±1-ULP pairs. Same shape as the existing `_eq_ne_stimuli_spec` fix.

**Estimated footprint:** ~40 lines of driver plumbing + the fallout triage.

---

## 5. Phase 2 — one edge-metadata block in `sfpu_domains.py` ⬜ **NOT STARTED**

Add a single **source of truth** for edge values, most of it **derived** from data already present.

### 5a. Derive boundary probes from the existing undefined ranges (cat A) — no new per-op data
The finite edge of each hole in `_SFPU_UNDEFINED_RANGES` is exactly the boundary we want to
probe (just-inside = defined, at/just-outside = the special result). One helper turns the
*existing* registry into probe points:

```python
def boundary_probes(op, eps=1e-6):
    """Return values straddling each undefined boundary of `op`, derived
    from _SFPU_UNDEFINED_RANGES (no new per-op data)."""
    probes = []
    for operand, holes in _SFPU_UNDEFINED_RANGES.get(op, {}).items():
        for lo, hi in holes:
            if math.isfinite(lo): probes += [(operand, lo - eps), (operand, lo)]      # just inside / at
            if math.isfinite(hi): probes += [(operand, hi),      (operand, hi + eps)] # at / just outside
    return probes
```
This covers `reciprocal`(0), `log`/`sqrt`/`rsqrt`(0), `atanh`/`erfinv`(±1), `acosh`(1),
`div`(divisor 0), `xlogy`(y 0), `pow`(base 0) — **all from data that already exists.**

> `eps` is a *format-relative* quantity, not the fixed `1e-6` the registry happens to use.
> At a boundary of 1.0 in Float16_b, `1.0 - 1e-6` **is** `1.0` — the "just inside" probe
> collapses onto the "at" probe and the pair tests one point twice. Derive it from the format's
> ULP at the boundary magnitude rather than hard-coding it.

### 5b. Shared special-value lists by format class (cats B, C)
```python
FLOAT_SPECIALS  = [float("inf"), float("-inf"), float("nan"), 0.0, -0.0]
INT32_SPECIALS  = [INT32_MIN, INT32_MAX, 0, -1, 1]        # INT32_MIN needs the §6c override path
UINT32_SPECIALS = [0, 1, UINT32_MAX]
def format_specials(fmt):
    if fmt.is_integer():
        return UINT32_SPECIALS if fmt == DataFormat.UInt32 else INT32_SPECIALS
    return FLOAT_SPECIALS
```

### 5c. Small op-specific discrete-edge table (cat D) — only what isn't a boundary
Keep this deliberately tiny; most entries are *shared* across families:
```python
_OP_EDGE_POINTS = {
    # comparison-to-zero & sign: the interesting point is exactly 0 / -0.0
    **{op: [0.0, -0.0] for op in (EqualZero, NotEqualZero, LessThanZero, GreaterThanZero,
                                  LessThanEqualZero, GreaterThanEqualZero, Sign, Heaviside)},
    # unary threshold comparisons fire at 0.5
    **{op: [0.5] for op in (UnaryGt, UnaryLt, UnaryGe, UnaryLe)},
    # piecewise knees / clamp bounds
    Clamp: [-1.0, 1.0], Hardtanh: [-1.0, 1.0],
    Softshrink: [-0.5, 0.5], Hardshrink: [-0.5, 0.5],
    Hardsigmoid: [-3.0, 3.0], Hardmish: [-2.0, 0.0],
    # exact-tie rounding (round-half-to-even) and integer knees
    Round: [-2.5, -0.5, 0.5, 1.5, 2.5], Floor: [-1.0, 0.0, 1.0, 2.0],
    Ceil: [-1.0, 0.0, 1.0, 2.0], Trunc: [-1.0, 0.0, 1.0], Frac: [-1.5, 1.5],
    # max/min ties
    **{op: [0.0] for op in (UnaryMax, UnaryMin)},
}
```
~25 lines that cover every cat-D row in the audit.

> Phase 0 already registered signed `[-10, 10]` domains for `Floor`/`Ceil`/`Trunc`/`Frac`,
> chosen so the random sweep lands *near* several integer knees. `_OP_EDGE_POINTS` is what
> lands *on* them. Keep both: the domain finds unexpected knees, the table pins the known ones.

---

## 6. Phase 3 — one `edge_spec()` builder, and how each driver delivers it ⬜ **NOT STARTED**

### 6a. `edge_spec(op, fmt)` and per-operand delivery

```python
def edge_spec(op, fmt, operand=Operand.A):
    """StimuliSpec.custom() combining domain boundaries + op knees + format specials,
    clipped to what `fmt` can represent."""
    vals  = [v for (o, v) in boundary_probes(op, eps=ulp_at(fmt, ...)) if o == operand]
    vals += _OP_EDGE_POINTS.get(op, [])
    vals += format_specials(fmt)
    return StimuliSpec.custom(values=dedup(vals))   # custom = head-of-face, zero-filled remainder
```

`custom` already exists (`spec.py:252`) and does exactly the placement we need. A face is far
bigger than these value lists, so remainder-zero-fill is harmless — and `0.0` / `−0.0` are
themselves useful probes. (`custom` is **per-face only**: `generate_full_tensor` raises, so the
values repeat in every face. That is fine, and `custom_faces` at `spec.py:257` is available
when different faces need different values.)

**Delivery differs per family, and this is the part revision 1 got wrong.** Revision 1 said
"`edge_spec` on `Operand.B` injects the divisor-zero probes". There is no `Operand.B` knob in
the unified binary driver:

| Family | Driver | How the edge values get in |
|---|---|---|
| Unary | `eltwise_unary_sfpu(..., spec_A=)` | `spec_A=edge_spec(op, fmt)` — works today |
| Binary | `sfpu_binary(..., spec_A=, src_A_override=)` | **`_paired_two_tile_spec(faceA, faceB)`** — both operands live in `buffer_A` (tile0 = in0, tile1 = in1). Build `faceA` from `edge_spec(op, fmt, Operand.A)` and `faceB` from `Operand.B`, so position *p* pairs as `(edge_A[p], edge_B[p])` |
| Ternary | `_run_sfpu_ternary` | **needs the Phase 1 spec parameters first** — currently no way in |
| Scalar | `_run_sfpu_binop_scalar` | Phase 1 makes the scalar an axis; `spec_A` still needs adding |

For binary cat-A/cat-D coverage the *cartesian product* of interesting A-values × B-values
matters more than element-wise pairing (divisor 0 against a positive, a negative and a zero
numerator are three different cases). `_build_shift_edge_case_src`
(`test_sfpu_binary.py:721`) already does exactly this — `itertools.product` over
`_SHIFT_EDGE_VALUES × _SHIFT_EDGE_AMOUNTS`, written into a two-tile tensor and fed through
`src_A_override`. **Generalize that builder** rather than writing a second one:
`_build_edge_pair_src(op, fmt)` over `edge_values(op, fmt, A) × edge_values(op, fmt, B)`.

### 6b. Phase 3 of revision 1 (`compare_special` flag) — **dropped, already satisfied**
`passed_test` already ORs `torch.isnan(golden) & torch.isnan(res)` into `is_valid` on the
generic path (`utils.py:628`), the Bfp8_b path, and inside `_bfp_block_aware_compare` /
`_mxint_block_aware_compare` / `_mxfp_block_aware_compare`. `torch.isclose` already returns
`True` for two `+inf` or two `−inf` and `False` for mismatched signs, which is precisely the
`both_inf` term revision 1 proposed adding. **No comparison change is needed.** (Ternary
additionally has its own `torch_equal_nan` for exact compares, `test_sfpu_ternary.py:40`.)

### 6c. New blocker: `StimuliSpec.custom` cannot carry integer extremes
`CustomStrategy.generate_face` clamps integer values into `_get_integer_bounds`
(`strategies/structured.py:80`), which returns `info.min + 1` for signed formats. So
`StimuliSpec.custom(values=[INT32_MIN, ...])` **silently yields `INT32_MIN + 1`** — the edge
is dropped with no error, which is the worst possible failure mode for an edge test.

Cat C therefore cannot go through the spec system as revision 1 assumed. Two options:

- **Preferred:** deliver integer edges as a **raw override tensor**, bypassing the strategies
  entirely. Both existing INT32-extreme tests already do this —
  `_build_shift_edge_case_src` → `src_A_override` (`test_sfpu_binary.py:721`) and
  `_run_int32_reduce`'s injection (`test_sfpu_reduce.py:466`). There is no new mechanism to
  invent, only one to reuse.
- **Alternative:** add an opt-in `allow_int_extremes=True` to the custom strategy. Cheaper at
  the call site, but it weakens a guard that exists because sign-magnitude Dst genuinely
  cannot represent `INT32_MIN` — most callers *want* the clamp. Prefer the override path and
  leave the guard alone.

Either way, **record in the audit that `INT32_MIN` reaching Dst is a HW limitation, not a
test gap**, in the ops where that is true (the shift and reduce xfails already say so).

---

## 7. Where specials can and cannot be injected (read before Phase 4)

### 7a. The (format, dest_acc) matrix does not preserve specials uniformly
`test_eltwise_unary_sfpu_isinf_isnan` — the one test that injects ±inf/NaN today — **skips
Float16_b input with `dest_acc=Yes`**, because bf16→fp32 dest unpack does not preserve
`-inf`/`NaN` and mangles `is_neg`/`is_nan` (`test_sfpu_unary.py:568`). That is a property of
the unpack path, not of any op.

Consequently a cat-B family sweep must **not** be a plain product over
`formats × dest_acc`. Either restrict special injection to the pairs known to preserve them
(fp32 input, or 16-bit input with `dest_acc=No`), or skip the rest with that reason. Getting
this wrong turns Phase 4 into a wall of failures that all share one root cause and hide the
real findings underneath. Establishing the preserving set — per arch, since the unpack paths
differ — should be the **first** task of Phase 4, ahead of any op sweep.

### 7b. Golden readiness (the one real per-op cost — kept small)
Injecting specials only helps if the golden defines the expected result:
- **Most goldens are torch-based** and already produce the correct `inf`/`nan`/`0`
  (`torch.reciprocal(0)=inf`, `torch.log(0)=-inf`). With `passed_test` already NaN-aware
  (§6b) these **just work** — no golden change.
- **A few goldens explicitly don't model non-finite** (`xlogy`, `addcmul` under dest_acc, some
  int paths). `pytest.mark.xfail(reason=...)` the specific special until the golden is
  extended — one line, not a rewrite.
- **`INT32_MIN`** is a sign-magnitude-Dst HW limitation; reuse the existing xfail convention
  (`test_sfpu_binary_int_shift_int32_min_unsupported`). Don't fight the HW.
- **Block-float outputs need a third look.** Phase 0 showed Bfp8_b is judged by
  *either* the lattice or the tolerance criterion. An `inf`/`NaN` golden inside a block whose
  shared exponent is finite is not a value the format can express at all, so neither
  criterion is meaningful. Expect to exclude block-float outputs from cat-B injection rather
  than to make the comparison handle it.

Rule of thumb: **default to injecting the edge; xfail the handful the golden can't yet
express; exclude the (format, dest_acc) pairs the hardware can't carry.**

---

## 8. Phase 4 — one thin edge test per family ⬜ **NOT STARTED**

No new driver, no new C++ source. Each is a ~15-line `@parametrize` wrapper
(`helpers/param_config.py:315`) over the **existing** driver, iterating the family's op list
with `spec_A=edge_spec(...)`. **Four** functions cover the whole matrix (revision 1 said five;
#50602 merged two drivers away and the binary wrapper now covers broadcast for free):

```python
@parametrize(mathop=ALL_MATHOPS + DOMAIN_MATHOPS, formats=SPECIAL_SAFE_FORMATS, dest_acc=[...])
def test_eltwise_unary_sfpu_edges(mathop, formats, dest_acc):
    eltwise_unary_sfpu("sources/eltwise_unary_sfpu_test.cpp", formats, dest_acc,
                       ApproximationMode.No, mathop, FastMode.No, [64, 64],
                       spec_A=edge_spec(mathop, formats.input_format))
```
…and the analogous `test_sfpu_binary_edges` (via `_paired_two_tile_spec` /
`_build_edge_pair_src`), `test_sfpu_ternary_edges`, `test_sfpu_binop_scalar_edges`. Because
`edge_spec` is keyed off the op, **adding a new op to the enum auto-enrolls it in edge
testing** — zero incremental code per future op.

- **Binary** div/pow/xlogy/fmod/remainder: pair the divisor-zero / base-zero / y≤0 probes
  against positive, negative and zero counterparts.
- **Ternary** `addcdiv`/`snake_beta`: inject `c=0` and tiny/negative `c` (today `c` is pinned
  to `uniform(1, 2)`, i.e. the pole is deliberately unreachable); `lerp`: weight `0`, `1`, `>1`.
- **Scalar** binops: sweep the scalar over `{0, ±large, ±tiny}` instead of the single
  hard-coded value.

### Category E (shift + arch): reuse, don't rewrite
- **Unary shift ops** (`LeftShift`/`RightShift`) still use a *fixed* shift of 3 with positive
  inputs. Generalize `_build_shift_edge_case_src` / `_SHIFT_EDGE_AMOUNTS` (already covering
  `{0..31, 32, 33, 40, 63, 100, 1000, −1, −5, −32, −1000}`) to the unary driver — builder and
  golden `_shift_reference` are written; only the wiring differs. Note the binary side now
  covers **three** ops (`_SHIFT_EDGE_OPS` gained `SfpuElwLogicalRightShift`).
- **Blackhole arch-skips.** Revision 1's action was "delete the `@skip_for_blackhole` once the
  bug closes". Two corrections: (a) the shift skip is now an **inline `pytest.skip` inside the
  test body** citing `docs/SFPU_INT32_SHIFT.md`, and the BH shift kernels are described there
  as unmigrated TTI microcode whose predicated out-of-range/sign handling breaks under
  `INT32_2S_COMP` — a kernel port, not a one-line unskip; (b) `test_int32_reduce_extreme`
  still carries a real `@skip_for_blackhole` (`test_sfpu_reduce.py:566`) on
  tt-metal#44750. Both are **external dependencies of this plan**, not tasks in it — track
  them, and convert each skip to an `xfail` so a fix surfaces as XPASS instead of staying
  silently skipped.

### Optional depth (only where cheap)
For 16-bit formats, `spec_A=StimuliSpec.ulp_sweep(low, high)` (`spec.py:391`) gives exhaustive
per-ULP coverage of a boundary neighbourhood in one line — worth adding for a few high-value
ops (`reciprocal`, `log`, `sqrt`). The natural home is now `accuracy/`, which already sweeps
`for_op` domains and reports `signed_ulp_error` per op/format
(`accuracy/accuracy_harness.py:189`) rather than pass/fail.

---

## 9. Out of the cheap path — genuinely new harnesses (prioritize, don't inline)

These need a new C++ source + golden and cannot be done by the shared mechanism. Verified
still untested: none of `welfords`, `dropout`, `quant`, `cumsum`, `reshuffle_rows`,
`int_sum`, `tiled_prod`, `copy_dest_values`, `max_pool_indices`, `rand` has a
`MathOperation` enum entry (`helpers/llk_params.py`).

| Kernel | Status | Suggested priority |
|--------|--------|--------------------|
| `welfords`, `int_sum`, `cumsum`, `tiled_prod` | reduction-family, no test at all | High — reuse reduce harness scaffolding |
| `quant` | no correctness test | High — used in production quantization |
| `dropout`, `rand` | RNG kernels, need statistical golden | Medium — distribution-level assert, not element-wise |
| `reshuffle_rows`, `copy_dest_values`, `max_pool_indices` | data-movement/index, no test | Medium |
| `generic_moe_gate_topk` (experimental) | **harness in progress** — `test_sfpu_generic_moe_gate_topk.py` + `sources/sfpu_generic_moe_gate_topk_test.cpp` exist as local work, not yet on a branch | Low, but nearly done |
| `TopKLocalSort` / `Merge` / `Rebuild` | perf-only; whole-op `topk` is tested | Medium — add stage-level correctness |

---

## 10. Effort / footprint estimate and status

| Phase | What | New/changed LOC | Ops covered | Status |
|-------|------|-----------------|-------------|--------|
| 0a–0g | Unary reroute + Bfp8_b golden/compare + domain recalibration + accuracy xfail | ~150 across 5 commits | 31 unary ops gain negative branch / knees / tails | ✅ done |
| 0h | Blackhole verification run + triage (0f/0e constants were WH measurements) | 0 — no code needed | same 31, on a second arch | ✅ green, no arch axis needed |
| 0i | Merge the two unary sweeps into one test; lists become coverage profiles (`a5bd408`) | **+235 / −163, one test deleted** | all 94, deduplicated | ✅ done, verified identical |
| 1 | Reroute binary / ternary / scalar onto the registry; add ternary + scalar spec params | ~40 + fallout | 10 binary ops with dormant domains, all ternary, all scalar | ⬜ |
| 2 | Edge metadata (`boundary_probes` + specials + `_OP_EDGE_POINTS`) | ~60 | all | ⬜ |
| 3 | `edge_spec()` + `_build_edge_pair_src` generalization (no compare change needed) | ~40 | all | ⬜ |
| 4 | 4 thin per-family edge tests + unary shift reuse + special-safe format matrix | ~90 | ~150 ops, auto-enrols future ops | ⬜ |
| 5 | golden `xfail` annotations | ~1 line each | the unmodelled few | ⬜ |
| F | new harnesses for untested kernels | large, per-kernel | the 10 untested + TopK stages | ⬜ (moe-gate in progress) |

**Bottom line:** ~230 lines of shared infrastructure closes categories A–E across all ~150
SFPU ops, and every future op auto-enrols by virtue of being in the enum. Phase 0 spent ~150
of those on 31 ops, but two thirds of that spend was the shared golden/comparison correctness
that phases 1–5 no longer have to pay for — and 0i gave one nightly test and its duplicated
skip paths back.

**One dependency Phase 1 inherits.** 0i's `_assert_sweep_profiles_disjoint()` cannot check
*exhaustiveness*, because `_OP_DOMAIN_REGISTRY` mixes unary, binary and reduce entries and
there is no authoritative "every unary op" list to compare against — so an op in neither
profile still goes untested silently. Recording an op's arity in the registry closes it, and
Phase 1 has to touch the registry's shape anyway.

---

## 11. Suggested sequencing

1. ~~**Phase 0**~~ — **closed** on `ldjurovic/sfpu_edge_cases_1` (§3). 0h came back green on
   Blackhole with no arch-specific bounds required, which is what unblocks the rest: 0f's
   domain constants are arch-independent, so Phases 1–5 can treat the registry as
   single-valued rather than per-arch.
2. **Phase 1** next, not Phase 2. It is the same reroute on three more drivers, it converts 10
   dormant registry entries into coverage, and — critically — the ternary/scalar spec
   parameters it adds are a hard prerequisite for Phases 3–4. Expect Phase-0-shaped fallout.
3. **Phase 2–3** with a couple of pilot ops (`reciprocal`, `div`, `round`) to validate the
   probe derivation and the paired-operand delivery before scaling out.
4. **Phase 4** — establish the special-safe `(format, dest_acc)` matrix per arch **first**
   (§7a), then enable the family sweeps and triage into real-bug vs golden-gap (`xfail`).
5. **Phase 5** — extend goldens for the highest-value `xfail`ed specials.
6. **Category E** — fold unary shift into the shift edge builder; convert both Blackhole skips
   to xfails and track tt-metal#44750 / `docs/SFPU_INT32_SHIFT.md` as external dependencies.
7. **Category F** — finish the `generic_moe_gate_topk` harness, then schedule the rest by §9.
