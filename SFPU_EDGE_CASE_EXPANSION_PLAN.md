# SFPU Edge-Case Coverage — Minimal-Code Expansion Plan

**Companion to:** [SFPU_EDGE_CASE_COVERAGE.md](SFPU_EDGE_CASE_COVERAGE.md)
**Issue:** [tenstorrent/tt-metal#49739 — [LLK] SFPU testing edge cases](https://github.com/tenstorrent/tt-metal/issues/49739)
**Goal:** close the edge-case gaps catalogued in the coverage audit while adding the **least possible code** — no per-op edge tests, no duplicated stimulus lists.
**Repo:** `tt-metal/tt_metal/tt-llk/tests/python_tests/`
**Scope:** Wormhole B0 and Blackhole. Quasar and its `quasar/` suite are **out of scope for
this plan** — it keeps its own inline stimulus definitions and is tracked separately.

**Revision 5 — 2026-08-05. This document covers the REMAINING work, which starts at Phase 2.**
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
- **Finding #4** — **unchanged.** `_get_integer_bounds` still returns `info.min + 1`
  (`helpers/stimuli_generator/utils.py:45`), so INT32_MIN is unreachable through any spec.

---

## 4. Phase 2 — one edge-metadata block in `sfpu_domains.py` ⬜ **NOT STARTED**

Add a single **source of truth** for edge values, most of it **derived** from data already present.

### 4a. Derive boundary probes from the existing undefined ranges (cat A) — no new per-op data
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

### 4b. Shared special-value lists by format class (cats B, C)
```python
FLOAT_SPECIALS  = [float("inf"), float("-inf"), float("nan"), 0.0, -0.0]
INT32_SPECIALS  = [INT32_MIN, INT32_MAX, 0, -1, 1]        # INT32_MIN needs the §5c override path
UINT32_SPECIALS = [0, 1, UINT32_MAX]
def format_specials(fmt):
    if fmt.is_integer():
        return UINT32_SPECIALS if fmt == DataFormat.UInt32 else INT32_SPECIALS
    return FLOAT_SPECIALS
```

### 4c. Small op-specific discrete-edge table (cat D) — only what isn't a boundary
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

> The registry already carries signed `[-10, 10]` domains for `Floor`/`Ceil`/`Trunc`/`Frac`,
> chosen so the random sweep lands *near* several integer knees. `_OP_EDGE_POINTS` is what
> lands *on* them. Keep both: the domain finds unexpected knees, the table pins the known ones.

---

## 5. Phase 3 — one `edge_spec()` builder, and how each driver delivers it ⬜ **NOT STARTED**

### 5a. `edge_spec(op, fmt)` and per-operand delivery

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
| Binary | `sfpu_binary(..., spec_A=, spec_B=, src_A_override=)` | pass `spec_A=edge_spec(op, fmt, Operand.A)` and `spec_B=edge_spec(op, fmt, Operand.B)`; the driver interleaves them via `_pair_operand_specs()` so position *p* pairs as `(edge_A[p], edge_B[p])`. Both operands live in `buffer_A` (tile0 = in0, tile1 = in1), so do **not** try to place them yourself |
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

## 7. Phase 4 — one thin edge test per family ⬜ **NOT STARTED**

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
| 2 | Edge metadata (`boundary_probes` + specials + `_OP_EDGE_POINTS`) | ~60 | all | ⬜ |
| 3 | `edge_spec()` + `_build_edge_pair_src` generalization (no compare change needed) | ~40 | all | ⬜ |
| 4 | 4 thin per-family edge tests + unary shift reuse + special-safe format matrix | ~90 | ~150 ops, auto-enrols future ops | ⬜ |
| 5 | golden `xfail` annotations | ~1 line each | the unmodelled few | ⬜ |
| F | new harnesses for untested kernels | large, per-kernel | the 10 untested + TopK stages | ⬜ (moe-gate in progress) |

**Bottom line:** ~230 lines of shared infrastructure closes categories A–E across all ~150
SFPU ops, and every future op auto-enrols by virtue of being in the enum. The completed phases
already paid for the shared golden/comparison correctness and for the per-operand delivery
plumbing in all four drivers, so the phases below do not have to.

---

## 10. Suggested sequencing

The registry is **single-valued** — the recalibrated domain constants proved arch-independent,
so nothing below needs a per-arch axis. All four drivers already accept per-operand specs, so
every step below is stimulus data plus a thin wrapper.

1. **Phase 2–3** with a couple of pilot ops (`reciprocal`, `div`, `round`) to validate the
   probe derivation and the paired-operand delivery before scaling out.
2. **Phase 4** — establish the special-safe `(format, dest_acc)` matrix per arch **first**
   (§6a), then enable the family sweeps and triage into real-bug vs golden-gap (`xfail`).
3. **Phase 5** — extend goldens for the highest-value `xfail`ed specials, and give `xlogy` the
   per-op tolerance that currently caps its x at 4.
4. **Category E** — fold unary shift into the shift edge builder; convert both Blackhole skips
   to xfails and track tt-metal#44750 / `docs/SFPU_INT32_SHIFT.md` as external dependencies.
5. **Category F** — finish the `generic_moe_gate_topk` harness, then schedule the rest by §8.

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
