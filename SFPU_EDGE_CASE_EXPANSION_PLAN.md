# SFPU Edge-Case Coverage — Minimal-Code Expansion Plan

**Companion to:** [SFPU_EDGE_CASE_COVERAGE.md](SFPU_EDGE_CASE_COVERAGE.md)
**Issue:** [tenstorrent/tt-metal#49739 — [LLK] SFPU testing edge cases](https://github.com/tenstorrent/tt-metal/issues/49739)
**Goal:** close the edge-case gaps catalogued in the coverage audit while adding the **least possible code** — no per-op edge tests, no duplicated stimulus lists.
**Repo:** `tt-metal/tt_metal/tt-llk/tests/python_tests/`
**Scope:** Wormhole B0 and Blackhole. Quasar and its `quasar/` suite are **out of scope for
this plan** — it keeps its own inline stimulus definitions and is tracked separately.

**Revision 7 — 2026-08-07. Phases 2, 3 and 4 are DONE. The remaining work starts at Phase 5.**

Everything through Phase 4 has landed on `ldjurovic/sfpu_edge_cases_2` and is green on Wormhole
n150. What is left is in §12, which is the section to read if you only read one.

Four of this plan's recommendations were **corrected by measurement** rather than merely
executed. Each is marked inline, and they share a shape worth stating once: this plan
consistently assumed the *stimulus* was the hard part, and in every case the binding constraint
turned out to be the **golden** or the **kernel's own documented range**.

| § | What the plan said | What measuring it showed |
|---|---|---|
| §4a | cat A is derivable from `_SFPU_UNDEFINED_RANGES`, "no new per-op data" | A hole is a *guard band*, not a boundary. Needed a new `_OP_SINGULARITIES` table, and adding holes instead would have re-rolled seven ops' stimuli |
| §6b | "default to injecting the edge; xfail the handful the golden can't express" | The handful is **half the op list** — 272 of 564 fail. Cat B is Phase 5 work, not a stimulus change |
| §5c | cat C is one shared `INT32_SPECIALS` list over the int ops | The delivery analysis was right, but 12 of 17 int ops document a *narrower valid range*, so the extremes are out of scope by kernel design. Cat C covers 5 ops |
| §7 (cat E) | unary shift is "builder and golden written; only the wiring differs" | `SHIFT_AMOUNT` is a C++ `constexpr` paired with a golden constant. Needs a template parameter, not wiring |

**Revision 6 — 2026-08-07. Phase 2 done.**
Phase 2 is now **done** on `ldjurovic/sfpu_edge_cases_2` — see §4. Phase 3 and Phase 4 are
still open, but §5 and §7 have been corrected against what building Phase 2 established;
the corrections are marked inline.

**Revision 5 — 2026-08-05.**
The two completed phases have been removed from here — see
[SFPU_PHASE0_SUMMARY.md](SFPU_PHASE0_SUMMARY.md) (detail in
[SFPU_EDGE_CASE_PHASE0_REPORT.md](SFPU_EDGE_CASE_PHASE0_REPORT.md)) and
[SFPU_PHASE1_SUMMARY.md](SFPU_PHASE1_SUMMARY.md) for what they were. Phase numbering is left
unchanged so that references in commits and PRs keep resolving: §4 is still "Phase 2", §5
"Phase 3", §7 "Phase 4".

The plan was originally written against a tree since restructured by
[#50602 *Unify SFPU test dispatch*](https://github.com/tenstorrent/tt-metal/pull/50602), and
the completed phases invalidated several of its assumptions. Five corrections carried forward,
because they change what the remaining phases have to do:

- **Phase 3 as originally written (a `compare_special` flag) is dropped** — `passed_test` is
  already NaN-aware on every path, and `torch.isclose` already accepts matching signed
  infinities. Nothing to add. See §5b.
- **Per-operand delivery in the binary driver is not two independent stimulus streams.** Both
  operands live in `buffer_A`, so `spec_B` is interleaved into `spec_A`'s `face_specs` by
  `_pair_operand_specs()`, which needs the buffer geometry to do it. There is no `Operand.B`
  knob. See §5a.
- **`StimuliSpec.custom` silently clamps integers to `INT_MIN + 1`**, so category C cannot be
  delivered through the spec system at all. See §5c — a hard blocker.
- **IEEE specials cannot be injected on every (format, dest_acc) pair**: bf16→fp32 dest unpack
  destroys `-inf`/`NaN`. See §6a. Any family sweep that ignores this produces a wall of false
  failures.
- **`xlogy` needs a per-op tolerance before its edges can be pushed further.** Its *absolute*
  error scales with x against a fixed atol, which is why x is capped at 4 in the registry
  rather than at the format's range. Phase 5 material — see §6b.

---

## 1. Guiding principle: one mechanism per gap-*category*, not per op

The 394 rows in the coverage audit look like 394 problems, but every `No` / `Excluded` row
falls into one of **six edge categories**, and each category is closed by **one shared
mechanism** that all ~150 ops reuse through the *existing* drivers:

| # | Gap category (from the audit) | Shared mechanism that closes it |
|---|-------------------------------|---------------------------------|
| A | Domain boundaries (`recip`/`div`/`log` at 0, `asin`/`acos`/`atanh`/`erfinv` at ±1, `acosh` at 1, `sqrt`/`rsqrt` ≤0, `pow` base≤0, `xlogy` y≤0) | **Auto-derive boundary probes from the data already in `_SFPU_UNDEFINED_RANGES`** |
| B | IEEE specials (±inf, NaN, +0.0, −0.0) into float ops | **One shared `FLOAT_SPECIALS` list injected by format class, gated on the (format, dest_acc) pairs that preserve them** |
| C | Integer extremes (INT32_MIN/MAX, UINT32_MAX, 0, −1, overflow) | **One shared `INT32_SPECIALS` / `UINT32_SPECIALS` list — delivered as a raw override tensor, not a `StimuliSpec` (see §5c)** |
| D | Op-specific discrete edges (knees, thresholds, exact-tie rounding) | **A small `_OP_EDGE_POINTS` table — only for points that aren't already a domain boundary** |
| E | Shift-amount limits for the **unary** shift ops, and the Blackhole shift/reduce arch-skips | **Reuse the existing binary shift edge-case builder; unblock the two tracked HW bugs** |
| F | Entirely-untested kernels (`welfords`, `dropout`, `quant`, …) and perf-only ops (`TopK*` stages) | **Genuinely new harness per kernel — explicitly out of the cheap path (§8)** |

Categories **A–D** are closed by **one new metadata block + one stimulus builder + ~4 thin
test wrappers**, a few hundred lines total that scale to every op. Category **E** is a small
refactor plus two HW-bug dependencies. Category **F** is the only place real new code is
unavoidable, and we scope/prioritize rather than write it here.

**Budget for fallout, not just for the mechanism.** Categories A–D all rest on a precondition
the original plan treated as free: a driver can only inject an edge value if the golden and the
comparison already agree over the widened domain. Phase 0 disproved that — rerouting the unary
sweep onto its real domains was 5 lines, and making it pass took four more commits, because
benign stimuli cannot distinguish a correct golden from a wrong one, a correct comparison from
a wrong one, or an accurate kernel from an inaccurate one. Every phase below widens stimuli
further, so budget the same three-way triage each time: **golden** correctness at the new
values, **comparison** correctness in the new regime, and **genuine** kernel accuracy limits
that were simply never measured. On Phase 0 that was roughly half the work.

---

## 2. The code this plan is written against

| Change | Where | Effect on this plan |
|---|---|---|
| `test_sfpu_binary_bcast.py` **deleted**, merged into `test_sfpu_binary.py` (`test_sfpu_binary_bcast`, `sfpu_binary(broadcast_type=…)`) | #50602 | §7 needs **4** family wrappers, not 5; binary's wrapper covers broadcast for free |
| `test_ttnn_where.py` **deleted**, merged into `test_sfpu_ternary.py` (`test_ttnn_where`, `test_ttnn_where_mcw`); `sources/ttnn_where_test.cpp` removed | #50602 | audit citations for the `where` rows move to `test_sfpu_ternary.py` |
| `sfpu_binary()` reads **both operands out of `buffer_A`** (tile0 = in0, tile1 = in1) and exposes `spec_A`, `spec_B` and `src_A_override` | `test_sfpu_binary.py:313` | cat A/B delivery for binary goes through these — §5a |
| `_pair_operand_specs(spec_A, spec_B, input_dimensions)` — interleaves the two operands' faces into one spec, **cycled across every tile pair in the buffer** | `test_sfpu_binary.py:83` | **this is the per-operand mechanism**; used by mask / isclose / eq_ne / the float comparisons. It needs the buffer geometry, so per-operand crafted stimuli return a `(spec_A, spec_B)` pair and let the driver place them |
| The float elementwise binary ops draw from the registry via two declared sets, `_REGISTRY_DOMAIN_OPS` and `_UNREGISTERED_BINARY_OPS`, cross-checked at import by `_assert_domain_sets_consistent()` | `test_sfpu_binary.py:139`, `:191` | registering a domain for a binary op surfaces as a **collection-time failure**, not a silent stimulus change — so Phase 2's registry edits cannot quietly reroute a binary op |
| The binary and ternary drivers now **seed** their stimuli (`StimuliSpec(..., seed=0)`); previously they redrew on every run | `test_sfpu_binary.py:113`, `:301` | triage is only trustworthy with a fixed draw — a flaky variant is indistinguishable from a real finding, and Phases 2–4 are triage-heavy by nature |
| `calculate_mask` only supports **one tile pair per dest block**, so `test_sfpu_binary_mask` runs on a `[64, 32]` buffer (1 block, 1 pair) | `test_sfpu_binary.py:619` | a cat-A/B sweep must not widen the mask test's buffer; the kernel hard-codes `dst_reg[0]`/`dst_reg[32]` and ignores forwarded dst indices |
| `passed_test` is NaN-aware on the generic path, the Bfp8_b path and all three block-aware compares | `helpers/utils.py:521`, `:628` | **Phase 3 of revision 1 is obsolete** |
| `_OP_DOMAIN_REGISTRY` values may now be **callables** (`Reciprocal: _reciprocal_spec`), i.e. format-sensitive domains | `helpers/sfpu_domains.py:202` | `edge_spec()` must resolve through `for_op()`, never index the registry directly |
| `narrowest_range_format()` added — domain is bounded by the narrowest float format in the pipeline, not the input format | `helpers/sfpu_domains.py:71` | edge probes must be range-checked the same way |
| Shift edge test grew a third op (`SfpuElwLogicalRightShift`) and a `_SHIFT_EDGE_OPS` list; the Blackhole skip is an **inline `pytest.skip` in the test body**, not a `@skip_for_blackhole` decorator | `test_sfpu_binary.py:872`, `:974` | rewrites cat E's action item |
| Dedicated `test_sfpu_binary_int_shift_int32_min_unsupported` xfail test | `test_sfpu_binary.py:1005` | the xfail convention cited by §6 is now its own test, not an annotation |
| `accuracy/accuracy_harness.py` is a **second consumer of `for_op()`**, measuring per-op `signed_ulp_error` | `accuracy/accuracy_harness.py:189` | any registry bound change now moves the accuracy sweep too — a coupling to state in the commit |

Where the two load-bearing audit findings now stand:

- **Finding #2** — **closed for the binary family, partly open elsewhere.** The 7 float
  elementwise ops (`SfpuElwadd/sub/mul/div/pow/rsub`, `SfpuXlogy`) now read their registered
  domain, and ternary/scalar have the plumbing but no registry entries to read. What is left is
  a *coverage* gap rather than a wiring one: **32 of the 43 binary ops have no registry entry at
  all** and keep the format default (they are enumerated in `_UNREGISTERED_BINARY_OPS`), and the
  three shift ops are deliberately excluded because their float domains would collapse to
  `{-1, 0, 1}` on Int32. Registering domains for those is a prerequisite for cat-A boundary
  probes on them, since §4a derives probes from registry data.
- **Finding #4** — **unchanged, and now documented at the point of use.**
  `_get_integer_bounds` still returns `info.min + 1`
  (`helpers/stimuli_generator/utils.py:45`), so INT32_MIN is unreachable through any spec.
  Phase 2's `integer_specials()` lists `INT_MIN + 1` alongside `INT_MIN` for that reason: a
  test that receives `INT_MIN + 1` where it asked for `INT_MIN` has hit the silent clamp, not
  found a bug.

One more piece of load-bearing infrastructure, established by building Phase 2 and not
previously recorded anywhere:

- **`exclude_intervals()` is not stimulus-neutral.** It *always* rewrites its result into the
  `intervals` form, and `_sample_uniform_intervals` consumes **two** `torch.rand` draws per
  element where the plain `low`/`high` path consumes **one**. `uniform(1, 8)` and
  `intervals=[(1, 8)]` are therefore the same distribution and different numbers at the same
  seed. Consequence: **declaring a new hole in `_SFPU_UNDEFINED_RANGES` re-draws that op's
  entire stimulus set**, even when the subtraction removes nothing. Any phase that wants to
  add a hole is signing up for that op's triage, and edge metadata must stay off this path
  (§4a).

---

## 4. Phase 2 — one edge-metadata block in `sfpu_domains.py` ✅ **DONE**

A single **source of truth** for edge values, added at the end of `sfpu_domains.py` below a
section banner explaining why it is a separate block and not an extension of the registry.
Two of the three sub-items landed as designed; 4a did not, and the reason it did not is the
most useful thing Phase 2 established.

### 4a. Boundary probes (cat A) ✅ — but **not** derivable from existing data alone

The plan's premise was "no new per-op data": the finite edge of each hole in
`_SFPU_UNDEFINED_RANGES` *is* the boundary, so one helper over the existing table gives the
probes for free. Building it showed that to be **half true, and the wrong half is
load-bearing.**

- **A hole is a guard band, not a boundary.** `Reciprocal`'s is `(-1e-6, 1e-6)`. Its finite
  edges are `±1e-6`; the point the test actually wants — exactly `0` — is *inside* the hole
  and the derivation never produces it. Same for `log`/`pow`/`xlogy` (`1e-6`) and
  `atanh`/`erfinv`/`log1p` (`±1 ∓ 1e-6`). Deriving probes from guard bands probes the guard
  band.
- **The guard band and the true boundary land in different binades**, so the format-relative
  `eps` the plan correctly insists on comes out *different* for each. At a boundary of 1.0 in
  Float16_b the band edge `0.999999` has half the ULP that `1.0` has, and emitting both
  sources produced a third probe (`-0.9921865`) that is neither the boundary nor a clean step
  from it. Observed while testing, not reasoned about.
- **Seven ops have a singularity and no hole at all**, because their registered domain
  already avoids it: `Rdiv` (`2.0/x`, pole at 0), `SqrtCustom`, `RsqrtCompat`, `LogWithBase`,
  `Asin`, `Acos`, plus `Lgamma`/`Digamma`/`Polygamma`. For these the derivation returns an
  empty list — silently, for exactly the ops most worth probing.

**And the obvious fix is not available.** Adding the missing holes to
`_SFPU_UNDEFINED_RANGES` is not stimulus-neutral: `exclude_intervals()` *always* rewrites its
result into the `intervals` form, and the interval sampler consumes **two** `torch.rand`
draws per element where the plain `low`/`high` sampler consumes **one**. So
`uniform(1, 8)` and `intervals=[(1, 8)]` are the same distribution and different numbers at
the same seed — verified empirically. Declaring a hole for an op re-draws that op's entire
stimulus set even when the subtraction removes nothing, which would silently re-roll seven
ops' inputs under the heading "adding metadata".

So Phase 2 adds **`_OP_SINGULARITIES`** — exact singular points, per op, per operand — as new
data alongside the existing table, and `boundary_probes()` prefers it, falling back to the
guard-band derivation only for an operand with no singularity entry (which preserves the
plan's "a new hole yields probes for free" property for future ops). `format_ulp(fmt,
magnitude)` supplies the format-relative `eps` from a new `_FORMAT_MANTISSA_BITS` table, and
`_dedup_representable()` drops probes the format cannot tell apart.

Result, `Reciprocal` in Float16_b: `[-0.015625, 0.0, 0.015625]`. In Bfp4_b:
`[-0.25, 0.0, 0.25]`. The boundary is now `0`, and the step scales with the format.

> Held back deliberately: `Lgamma`, `Digamma` and `Polygamma` have poles at 0 and the
> negative integers, but their kernels are polynomial/LUT fits that only claim accuracy well
> inside a positive domain. A probe at their boundary tests a value the kernel never
> promised, which yields a failure that is neither a bug nor fixable. The omission is
> commented in the table so it does not read as an oversight.

> `format_ulp` returns a **lower bound** for block-float formats: the real step is set by the
> exponent shared across the 16-element block, not by the element's own magnitude. That is
> the safe direction (too fine wastes a value; too coarse walks past the boundary) but it
> means a block-float probe pair cannot be *assumed* distinct.

### 4b. Shared special-value lists by format class (cats B, C) ✅

`FLOAT_SPECIALS` as planned. The integer half is **derived from the format's width and
signedness** rather than hard-coded to 32 bits, so `Int16`/`Int8`/`UInt8` get their own
extremes instead of int32's silently clamped down:

```python
integer_specials(Int32)  -> (-2147483648, -2147483647, -1, 0, 1, 2147483647)
integer_specials(Int8)   -> (-128, -127, -1, 0, 1, 127)
integer_specials(UInt32) -> (0, 1, 4294967295)
```

`INT_MIN + 1` is included next to `INT_MIN` on purpose: it is the value
`_get_integer_bounds` clamps to, so a test that gets `INT_MIN + 1` where it asked for
`INT_MIN` has hit §5c's silent clamp rather than found a bug. The docstring says so and
points at the override path.

### 4c. Op-specific discrete-edge table (cat D) ✅ — larger than planned, and verified

43 entries. Every constant was checked against the golden that owns it rather than taken
from the plan, which found the plan's table **correct where it spoke but incomplete**:
`_UNARY_COMP_THRESHOLD = 0.5` ✓, `_UNARY_MAX_MIN_VALUE = 0.0` ✓, `_HARDSHRINK_LAMBDA = 0.5`
✓, softshrink `lambd=0.5` ✓, clamp/hardtanh `[-1, 1]` ✓, hardsigmoid `[-3, 3]` ✓, hardmish
`[-2, 0]` ✓. Missing from the plan and now present:

| Op | Knee | Owner |
|---|---|---|
| `Threshold` | 5.0 | `_threshold(t=5, v=10)` |
| `ReluMax` | 0.0, 5.0 | `_relu_max(threshold=5)` |
| `ReluMin` | 5.0 | `_relu_min(threshold=5)` |
| `Softplus` | 20.0 | `_SOFTPLUS_THRESHOLD` |
| `Lrelu`, `Prelu`, `Elu`, `Celu`, `Selu`, `Xielu`, `Signbit` | 0.0 / −0.0 | piecewise at zero |
| `UnaryEq`, `UnaryNe`, `LogicalNot` | 0.5 / 0.0 | `_UNARY_COMP_THRESHOLD` |
| `UnaryMaxInt32`, `UnaryMinInt32`, `UnaryMaxUint32`, `UnaryMinUint32` | 1000 | `_int_maxmin_scalar` |

`Relu` is deliberately **excluded** even though its knee is at 0: relu is applied by the
packer (`STACC_RELU`) and is not a member of `SfpuType`, so no SFPU probe can reach it.

Every one of these is a **dispatch constant shared with the golden**, and the golden is the
authority — a coupling the plan did not mention. The table names the owning attribute per
entry, because changing `_UNARY_COMP_THRESHOLD` would otherwise leave this table probing a
point that is no longer a threshold: full coverage on paper, nothing tested in fact.

> The registry already carries signed `[-10, 10]` domains for `Floor`/`Ceil`/`Trunc`/`Frac`,
> chosen so the random sweep lands *near* several integer knees. `_OP_EDGE_POINTS` is what
> lands *on* them. Keep both: the domain finds unexpected knees, the table pins the known ones.

### 4d. Coverage this metadata reaches

**50 of the 97 unary SFPU ops now have at least one edge value** from cat A or cat D. The
other 47 are smooth everywhere with no knee and no pole (`sin`, `cos`, `tanh`, `gelu`, `erf`,
`exp`, …). For those, **cat B is the entire edge story** — which raises the priority of the
special-safe `(format, dest_acc)` matrix (§6a) from "do it first to avoid noise" to "it is
half the remaining coverage".

### 4e. Also fixed here: the binary suite's declared-set hole

`_assert_domain_sets_consistent()` claimed to partition the ops the binary suite drives into
"reads the registry" and "keeps the format default". It did not: four ops — `SfpuAddTopRow`
and the three shift ops — were in **neither** set while having registry entries, so the check
passed while saying nothing about them. That is the silent-drift failure mode the assertion
exists to prevent, and Phase 3 was about to add a third set to the same file and inherit it.

Now three declared sets plus one for ops that do not use the shared driver at all, each with
a recorded reason, and a `_classify_stimuli_source()` guard in `sfpu_binary()` that raises
for an unclassified op. The guard is in the **driver**, not at collection: the set of ops the
suite drives is only known once pytest has expanded the parametrize lists, so a
collection-time assertion cannot check totality without duplicating that set. All four
previously-undeclared ops now have a home.

---

## 5. Phase 3 — one `edge_spec()` builder, and how each driver delivers it ✅ **DONE**

Shipped as `edge_values()` / `edge_spec()` / `clip_to_format()` in `sfpu_domains.py`, plus
`edge_pair_values()` / `edge_counterparts()` for the binary side. Four deviations from the
sketch below, three of them corrections it already anticipated and one found by running it:

- **Takes the input/output pair, not one format** (see 5a).
- **Returns `None`, not an empty spec**, for the 47 ops with no knee and no pole, so the caller
  skips with a reason instead of driving a zero-value stimulus.
- **Raises on integer + specials** rather than silently clamping `INT_MIN` (see 5c).
- **`-0.0` was being deduped away against `+0.0`.** Found by inspecting the output for `Sign`,
  which came out as `[0.0]`. Both are zero ULPs apart, so the numeric dedup discarded one — and
  for `signbit`/`sign`/`heaviside`/`reciprocal` the difference between the two zeros *is* the
  probe. Signed zeros are now keyed by sign. **This bug accounts for four of the nine findings**,
  so the sweeps would have been blind to them.

For the binary side, `edge_pair_values()` crosses both operands' edges, and whichever operand has
no edge of its own contributes `edge_counterparts()` — a `(-2, -1, 0, 1, 2)` spread clipped to
that operand's registered domain, so `pow`'s base-zero probe cannot be paired with an exponent
outside its registered `[0, 3]`. Without it the product would be empty for `div` and the
divisor-zero probe would never run.

### 5a. `edge_spec(...)` and per-operand delivery

**Corrected signature — it needs the input/output pair, not one format.** The sketch below
took a single `fmt`, but the unary driver resolves its own default through
`for_op_pipeline(op, input_format, output_format)` precisely because the sweep pairs every
input with every output and the **output** is often the binding constraint. A caller passing
`spec_A=edge_spec(...)` **bypasses that resolution entirely** — `eltwise_unary_sfpu` only
resolves when `spec_A is None`. So a probe near a format's ceiling would be injected unclipped
into a Float16 or MxFp4 output and overflow. `edge_spec` must clip against
`narrowest_range_format(input_format, output_format)` itself.

Two Phase 2 helpers replace the sketch's placeholders: `boundary_probes(op, operand, fmt)`
already takes the operand and does its own format-relative `eps` and dedup, and
`op_edge_points(op)` reads the cat-D table.

```python
def edge_spec(op, input_format, output_format=None, operand=Operand.A):
    """StimuliSpec.custom() combining domain boundaries + op knees + format specials,
    clipped to what the narrowest format in the pipeline can represent."""
    fmt  = narrowest_range_format(input_format, output_format)
    vals  = list(boundary_probes(op, operand, fmt))
    vals += op_edge_points(op)
    vals += format_specials(fmt)          # gate per §6a before including the specials
    return StimuliSpec.custom(values=clip_to_format(dedup(vals), fmt))
```

> **Do not route edge specs through `for_op_pipeline`.** Its `_tighter_spec` measures a
> domain with `_spec_span`, which falls back to `spec.high - spec.low` — `None - None` for a
> values-list spec. Nothing hits this today because the two paths are separate; the
> obvious-looking simplification of unifying them raises `TypeError`. Keep them separate, or
> teach `_spec_span` about `values` first.

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
| Binary | `sfpu_binary(..., spec_A=, spec_B=, src_A_override=)` | pass `spec_A=edge_spec(op, …, Operand.A)` and `spec_B=edge_spec(op, …, Operand.B)`; the driver interleaves them via `_pair_operand_specs()` so position *p* pairs as `(edge_A[p], edge_B[p])`. Both operands live in `buffer_A` (tile0 = in0, tile1 = in1), so do **not** try to place them yourself. **Caveat:** `_pair_operand_specs` does `replace(spec_A, face_specs=…)`, which silently discards any `face_specs` the caller already set — so per-operand pairing and `custom_faces` cannot currently be combined on one spec, which is exactly what a binary cat-A sweep wants. Decide the composition rule before writing `_build_edge_pair_src` |
| Ternary | `_run_sfpu_ternary(..., spec_A=, spec_B=, spec_C=)` | pass the three specs directly; `c` is the one that matters — its default `uniform(1, 2)` keeps the pole unreachable |
| Scalar | `_run_sfpu_binop_scalar(..., scalar=)` | the scalar is already a swept axis over `_SCALARS = (0.0, 1.0, 2.0, −2.0, 8.0, 0.25)` (`test_sfpu_binop_scalar.py:40`); a `spec_A` knob for the tensor operand still needs adding |

For binary cat-A/cat-D coverage the *cartesian product* of interesting A-values × B-values
matters more than element-wise pairing (divisor 0 against a positive, a negative and a zero
numerator are three different cases). `_build_shift_edge_case_src`
already does exactly this — `itertools.product` over
`_SHIFT_EDGE_VALUES × _SHIFT_EDGE_AMOUNTS` (`test_sfpu_binary.py:943`), written into a two-tile tensor and fed through
`src_A_override`. **Generalize that builder** rather than writing a second one:
`_build_edge_pair_src(op, fmt)` over `edge_values(op, fmt, A) × edge_values(op, fmt, B)`.

### 5b. Phase 3 of revision 1 (`compare_special` flag) — **dropped, already satisfied**
`passed_test` already ORs `torch.isnan(golden) & torch.isnan(res)` into `is_valid` on the
generic path (`utils.py:628`), the Bfp8_b path, and inside `_bfp_block_aware_compare` /
`_mxint_block_aware_compare` / `_mxfp_block_aware_compare`. `torch.isclose` already returns
`True` for two `+inf` or two `−inf` and `False` for mismatched signs, which is precisely the
`both_inf` term revision 1 proposed adding. **No comparison change is needed.** (Ternary
additionally has its own `torch_equal_nan` for exact compares, `test_sfpu_ternary.py:40`.)

### 5c. New blocker: `StimuliSpec.custom` cannot carry integer extremes
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

## 6. Where specials can and cannot be injected (read before Phase 4)

### 6a. The (format, dest_acc) matrix does not preserve specials uniformly
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

### 6b. Golden readiness (the one real per-op cost — kept small)
Injecting specials only helps if the golden defines the expected result:
- **Most goldens are torch-based** and already produce the correct `inf`/`nan`/`0`
  (`torch.reciprocal(0)=inf`, `torch.log(0)=-inf`). With `passed_test` already NaN-aware
  (§5b) these **just work** — no golden change.
- **A few goldens explicitly don't model non-finite** (`xlogy`, `addcmul` under dest_acc, some
  int paths). `pytest.mark.xfail(reason=...)` the specific special until the golden is
  extended — one line, not a rewrite.
- **`INT32_MIN`** is a sign-magnitude-Dst HW limitation; reuse the existing xfail convention
  (`test_sfpu_binary_int_shift_int32_min_unsupported`). Don't fight the HW.
- **Two ops are bounded by *tolerance*, not by the golden or the format**, so their edges
  cannot simply be widened: `pow`'s error tracks the product `b·ln a` handed to the shared exp
  approximation (`3·ln3 = 3.30` clean, `4.83` → 4.9% off, `8.05` → 6.1% off against a 5% rtol),
  and `xlogy`'s *absolute* error scales with x against a fixed atol. Both are capped in the
  registry (at 3 and 4 respectively) with the measured numbers in the comments. Pushing either
  further is a **per-op tolerance** change, which is Phase 5 work, not a stimulus change.
- **Block-float outputs need a third look.** Bfp8_b is judged by *either* the lattice or the
  tolerance criterion. An `inf`/`NaN` golden inside a block whose
  shared exponent is finite is not a value the format can express at all, so neither
  criterion is meaningful. Expect to exclude block-float outputs from cat-B injection rather
  than to make the comparison handle it.

Rule of thumb: **default to injecting the edge; xfail the handful the golden can't yet
express; exclude the (format, dest_acc) pairs the hardware can't carry.**

---

## 7. Phase 4 — one thin edge test per family 🟡 **DONE for unary and binary**

Two of the four planned wrappers exist and are green:

| Test | Cases | Result |
|---|---|---|
| `test_eltwise_unary_sfpu_edges` | 752 (94 ops × 4 pairs × 2 dest_acc) | 372 passed, 360 skipped, 20 xfailed |
| `test_sfpu_binary_edges` | 40 (5 ops × 4 pairs × 2 dest_acc) | 13 passed, 10 skipped, 22 xfailed |
| `test_sfpu_binary_int_extremes` | 5 | 5 passed |

The 360 unary skips are the 44 ops that are smooth everywhere with no knee and no pole — for
those the random sweep already covers everything an edge probe could add, so the test skips with
that reason rather than driving a meaningless variant.

**The format axis is the standard profile, not the broad one**, which the plan did not specify:
an edge probe is a fixed value, so the block-float and approximation-mode axes vary nothing about
it. What *does* vary is whether specials can be injected, and `specials_safe()` decides that per
`(input, output, dest_acc)`.

**Ternary and scalar wrappers are NOT done** — see §12 for why each is blocked rather than
merely pending.

Two corrections the sweep forced, both recorded in §4a and §6b: cat A needed per-singularity
defined-side data, and cat B is off by default.

No new driver, no new C++ source. Each is a ~15-line `@parametrize` wrapper
(`helpers/param_config.py:315`) over the **existing** driver, iterating the family's op list
with `spec_A=edge_spec(...)`. **Four** functions cover the whole matrix (revision 1 said five;
#50602 merged two drivers away and the binary wrapper now covers broadcast for free):

```python
@parametrize(mathop=sorted(sfpu_unary_ops()), formats=SPECIAL_SAFE_FORMATS, dest_acc=[...])
def test_eltwise_unary_sfpu_edges(mathop, formats, dest_acc):
    eltwise_unary_sfpu("sources/eltwise_unary_sfpu_test.cpp", formats, dest_acc,
                       ApproximationMode.No, mathop, FastMode.No, [64, 64],
                       spec_A=edge_spec(mathop, formats.input_format))
```
(`sfpu_unary_ops()` — `helpers/sfpu_domains.py:897` — is the registry's own answer to "which
ops are unary", so the edge sweep enumerates the same set the float sweep does rather than a
second hand-maintained list. The two sweep profiles the unary test uses, `BROAD_SWEEP_OPS` /
`STANDARD_SWEEP_OPS`, are about *stimulus breadth*, which an edge sweep does not vary.)

…and the analogous `test_sfpu_binary_edges` (via `spec_A`/`spec_B` or
`_build_edge_pair_src`), `test_sfpu_ternary_edges`, `test_sfpu_binop_scalar_edges`. Because
`edge_spec` is keyed off the op, **adding a new op to the enum auto-enrolls it in edge
testing** — zero incremental code per future op.

- **Binary** div/pow/xlogy/fmod/remainder: pair the divisor-zero / base-zero / y≤0 probes
  against positive, negative and zero counterparts.
- **Ternary** `addcdiv`/`snake_beta`: inject `c=0` and tiny/negative `c` (today `c` is pinned
  to `uniform(1, 2)`, i.e. the pole is deliberately unreachable); `lerp`: weight `0`, `1`, `>1`.
- **Scalar** binops: the *scalar* axis already exists, but it is deliberately bounded at
  `|scalar| ≤ 8` so that `uniform(-1, 1)` inputs stay inside the range where the default bf16
  tolerance is meaningful. What is left here is the **tensor** operand — add the `spec_A` knob
  and inject `edge_spec` through it. Extending the axis to `±large` / `±tiny` needs a per-op
  tolerance first (Phase 5), not a wider list.

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

## 8. Out of the cheap path — genuinely new harnesses (prioritize, don't inline)

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

## 9. Effort / footprint estimate and status

| Phase | What | New/changed LOC | Ops covered | Status |
|-------|------|-----------------|-------------|--------|
| 2 | Edge metadata (`format_ulp` + `_OP_SINGULARITIES` + `boundary_probes` + specials + `_OP_EDGE_POINTS`), plus the binary declared-set fix | ~320 (est. ~60) | cat A on 15 unary + 4 binary-B, cat D on 43; **50/97 unary ops have an edge value** | ✅ |
| 3 | `edge_spec()` + `edge_pair_values()` + `edge_counterparts()` + `clip_to_format()` (no compare change needed) | ~110 (est. ~40) | all | ✅ |
| 4 | 2 of 4 per-family edge tests + the measured special-safe format matrix | ~320 (est. ~90) | 94 unary + 5 binary + 5 int; auto-enrols future ops | 🟡 unary + binary done; ternary and scalar blocked (§12) |
| 5 | golden work for cat B, and `xfail` annotations | **much larger than "~1 line each"** — 272 of 564 specials variants fail | the unmodelled *half* | ⬜ |
| F | new harnesses for untested kernels | large, per-kernel | the 10 untested + TopK stages | ⬜ (moe-gate in progress) |

Phase 2 came in at ~4.5× the estimate, and the overrun is all in the two places the estimate
assumed were free: cat A needed a **new** per-op table (`_OP_SINGULARITIES`) because the
existing one records guard bands rather than boundaries (§4a), and every cat-D constant had to
be read out of the golden that owns it rather than copied from this document — which is how
the seven missing knees were found (§4c). Both are data, not mechanism, so they do not change
Phase 3's or Phase 4's estimate.

**Bottom line, revised:** ~750 lines closed categories A, C and D across every op that has one,
plus the measured cat-B surface. The estimate of ~230 was out by 3×, and every line of the
overrun was **data or triage**, not mechanism — the "one mechanism per gap-category" principle in
§1 held up completely. What it under-priced was that a mechanism is worthless until something
downstream of it (a golden, a kernel's claimed range) agrees.

---

## 10. Suggested sequencing — superseded

Phases 2, 3 and 4 are done; steps 1–2 below are complete. §12 replaces this section for
everything that remains.

1. ~~**Phase 2–3** with pilot ops to validate the probe derivation~~ — **done.** Validated by
   inspecting output for `reciprocal`, `log`, `sqrt`, `log1p`, `atanh`, `acosh`, `erfinv`,
   `rdiv`, `asin` and the binary ops in both Float16_b and Bfp4_b; that is what surfaced the
   guard-band and binade problems in §4a.
2. ~~**Phase 4** — establish the special-safe matrix **first**, then enable the sweeps~~ —
   **done, and doing it in that order was right.** The matrix cost 250 probe variants and
   immediately explained 85 of them; enabling the sweeps first would have produced a wall of
   failures sharing one root cause. It also revealed that block-float rows had to be *excluded*
   rather than trusted, which measuring after the fact would have hidden.
3. **Phase 5** — extend goldens for cat B, and give `xlogy` the per-op tolerance that caps its x
   at 4. Now the largest remaining item, not a footnote — see §12.
4. **Category E** — unary shift needs a C++ template parameter (§7). Both Blackhole skips are
   **already converted** to xfails.
5. **Category F** — finish the `generic_moe_gate_topk` harness, then schedule the rest by §8.

---

## 12. What is left, and why each item is blocked rather than merely pending

Ordered by value. Nothing here is a thin wrapper away; each is blocked on something specific.

### 1. Cat B — goldens that model non-finite inputs (was "Phase 5, ~1 line each")

**The single largest remaining item, and the plan's biggest mis-estimate.** The stimulus side is
finished: `specials_safe()` says exactly where specials can be injected, and
`_EDGE_SWEEP_SPECIALS = True` turns it on. Flipping it today gives **272 failures out of 564**,
because the torch-backed goldens do not define a result for non-finite *inputs* — they return
`inf` where the answer is `nan`, and so on.

This matters beyond the edge sweep: **47 of the 97 unary ops are smooth everywhere** with no
knee and no pole, so cat B is the *entire* edge story for half the op list (§4d). Until the
goldens model it, those 47 ops have no deliberate edge coverage at all.

Not a sprinkle of xfails. It is per-op golden work, and the right unit of progress is probably
"pick the ten highest-value ops and define their non-finite behaviour", not "make the sweep
green".

### 2. Ternary edges — blocked on a data-model change

`addcdiv`/`snake_beta` with `c → 0` (today `c` is pinned to `uniform(1, 2)`, so the pole is
deliberately unreachable) and `lerp` with weight `0`/`1`/`>1`. Blocked because **`OperandSpecs`
carries only `spec_A` and `spec_B`** — there is no third operand to register a singularity for.
`_ternary_default_specs` already works around this by reusing B for C, with a comment saying so.
Growing `OperandSpecs` touches five consumers including the accuracy harness, so it is a
deliberate change rather than an incremental one.

### 3. Scalar tensor-operand edges — small, genuinely just pending

`_run_sfpu_binop_scalar` needs one `spec_A=` parameter and a thin wrapper. The *scalar* axis
already sweeps `{0, 1, 2, −2, 8, 0.25}`; the tensor operand has no knob. Widening the scalar axis
to `±large`/`±tiny` is **not** part of this — that needs a per-op tolerance first.

### 4. Category E — unary shift needs a C++ change

`SHIFT_AMOUNT` is `constexpr std::uint32_t SHIFT_AMOUNT = 3u` inside
`call_unary_sfpu_operation`, paired with `_int_shift_amount` on the golden side. Sweeping it needs
a new `TemplateParameter` plus matching golden plumbing — cross-language, not test wiring. The
builder and `_shift_reference` genuinely are written and reusable once the parameter exists.

### 5. Per-op tolerances (`xlogy`, `pow`)

`xlogy`'s x is capped at 4 and `pow`'s operands at 3 because their error outruns a fixed
tolerance — `xlogy`'s absolute error scales with x against a fixed atol; `pow`'s tracks
`b·ln a` into the shared exp approximation. Both need a per-op tolerance before their edges can
be pushed further. Independent of everything above.

### 6. Category F — new harnesses

Unchanged from §8. `welfords`, `int_sum`, `cumsum`, `tiled_prod`, `quant` are the high-priority
five; `generic_moe_gate_topk` is nearly done.

### 7. Not phase work, but it bounds the value of all of it

- **Blackhole.** Nothing in Phases 2–4 has been run there. Two parts are arch-sensitive by
  construction: `specials_safe()`'s table (the unpack paths differ) and the two converted xfails,
  whose whole purpose is the Blackhole path. Separately, the `SFPMAD` signed-zero xfails are a
  **testable prediction**: Blackhole's ISA documents sign-preserved zero, so they should XPASS.
- **CI.** The broad unary profile runs in **no automated job on any arch** — every LLK pytest job
  either excludes `nightly` or runs `--coverage`, under which the broad profile is skipped
  wholesale (tt-llk#1435). This predates the branch but means none of this coverage is currently
  guarded. `llk-e2e` needs a non-coverage companion group, or the broad profile must stop being
  coverage-gated.
- **The nine recorded divergences.** §0.6 of the coverage audit splits them into documented ISA
  behaviour and genuine open questions. `signbit(-0.0)` and `RsqrtCompat(0)` are the two worth
  raising with kernel owners.
- **`Float16 -> Bfp4_b`.** The dropped commit's failing cell is still unexplained: either that
  pair needs a guard on Wormhole or the pack path has a bug. Re-adding Bfp4_b as an output format
  depends on which.

---

## 11. Known trap in the test infra (not a phase, but it will bite)

`TestConfig` calls `shutil.rmtree(ARTEFACTS_DIR)` at session setup — "always have a fresh build
when compiling" — against the **fixed** path `/tmp/tt-llk-build`. Any second pytest session on
the same host, including a one-op `-k` run started to triage something, deletes the build tree
out from under a running sweep. The victim reports `ld: cannot open output file` attributed to
whichever variant happened to be linking, which in a log reads exactly like a real kernel bug.

This produced two phantom failures during Phase 0's Blackhole verification before being
identified. Phases 2–4 are triage-heavy by nature — widen stimuli, run, investigate, re-run —
so it will recur. Worth filing and fixing separately: key the artefact root by session, or take
the existing `/tmp/tt-llk-build-shared.lock` around the rmtree.
