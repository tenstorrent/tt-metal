# SFPU Edge-Case Coverage — Minimal-Code Expansion Plan

**Companion to:** [SFPU_EDGE_CASE_COVERAGE.md](SFPU_EDGE_CASE_COVERAGE.md)
**Issue:** [tenstorrent/tt-metal#49739 — [LLK] SFPU testing edge cases](https://github.com/tenstorrent/tt-metal/issues/49739)
**Goal:** close the edge-case gaps catalogued in the coverage audit while adding the **least possible code** — no per-op edge tests, no duplicated stimulus lists.
**Repo:** `tt-metal/tt_metal/tt-llk/tests/python_tests/`
**Scope:** Wormhole B0 and Blackhole. Quasar and its `quasar/` suite are **out of scope for
this plan** — it keeps its own inline stimulus definitions and is tracked separately.

**Revision 4 — 2026-08-05. This document covers the REMAINING work.**
Phases 0 and 1 are complete; Phase 0 has been removed from here — see
[SFPU_PHASE0_SUMMARY.md](SFPU_PHASE0_SUMMARY.md) for what it was and
[SFPU_EDGE_CASE_PHASE0_REPORT.md](SFPU_EDGE_CASE_PHASE0_REPORT.md) for the detail. Phase
numbering is left unchanged (the next phase is still "Phase 1") so that references in commits
and PRs keep resolving.

The plan was originally written against a tree since restructured by
[#50602 *Unify SFPU test dispatch*](https://github.com/tenstorrent/tt-metal/pull/50602), and
Phase 0 invalidated several of its assumptions. Four corrections carried forward, because they
change what the remaining phases have to do:

- **Phase 3 as originally written (a `compare_special` flag) is dropped** — `passed_test` is
  already NaN-aware on every path, and `torch.isclose` already accepts matching signed
  infinities. Nothing to add. See §5b.
- **The binary driver has no `spec_B`.** Per-operand edge injection goes through
  `_paired_two_tile_spec()` or `src_A_override`, not `Operand.B`. See §5a.
- **`StimuliSpec.custom` silently clamps integers to `INT_MIN + 1`**, so category C cannot be
  delivered through the spec system at all. See §5c — a hard blocker.
- **IEEE specials cannot be injected on every (format, dest_acc) pair**: bf16→fp32 dest unpack
  destroys `-inf`/`NaN`. See §6a. Any family sweep that ignores this produces a wall of false
  failures.

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
| `sfpu_binary()` reads **both operands out of `buffer_A`** (tile0 = in0, tile1 = in1) and exposes `spec_A` + `src_A_override`, **no `spec_B`** | `test_sfpu_binary.py:159` | rewrites cat A/B delivery for binary — §5a |
| `_paired_two_tile_spec(a_face, b_face)` — per-operand faces inside one spec | `test_sfpu_binary.py:80` | **this is the per-operand mechanism**; already used by mask / isclose / eq_ne |
| `passed_test` is NaN-aware on the generic path, the Bfp8_b path and all three block-aware compares | `helpers/utils.py:521`, `:628` | **Phase 3 of revision 1 is obsolete** |
| `_OP_DOMAIN_REGISTRY` values may now be **callables** (`Reciprocal: _reciprocal_spec`), i.e. format-sensitive domains | `helpers/sfpu_domains.py:202` | `edge_spec()` must resolve through `for_op()`, never index the registry directly |
| `narrowest_range_format()` added — domain is bounded by the narrowest float format in the pipeline, not the input format | `helpers/sfpu_domains.py:71` | edge probes must be range-checked the same way |
| Shift edge test grew a third op (`SfpuElwLogicalRightShift`) and a `_SHIFT_EDGE_OPS` list; the Blackhole skip is an **inline `pytest.skip` in the test body**, not a `@skip_for_blackhole` decorator | `test_sfpu_binary.py:650`, `:746` | rewrites cat E's action item |
| Dedicated `test_sfpu_binary_int_shift_int32_min_unsupported` xfail test | `test_sfpu_binary.py:774` | the xfail convention cited by §6 is now its own test, not an annotation |
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

## 3. Phase 1 — the same reroute for binary / ternary / scalar ✅ **DONE**

Commits `b0f4ae8` (binary) and `1291c7a` (ternary, scalar, comparison ops, arity) on
`ldjurovic/sfpu_edge_cases_1`. Blackhole, all four suites: **4990 passed, 1700 skipped,
7 xfailed**.

Audit finding #2 is closed for the binary family and the plumbing exists for the rest.

| | Item | Outcome |
|---|---|---|
| 1 | **Binary** — default `spec_A`/`spec_B` to the registry | ✅ 7 float elementwise ops rerouted. add/sub/mul/rsub/div go from **0% to 50% negative operands**, and div's divisor now spans both sides of the pole it is registered to avoid. Deliberately **float-only**: `SfpuElwadd`/`SfpuElwsub` and the shifts also run on Int32, where the registry's `uniform(-1, 1)` collapses to `{-1, 0, 1}` and would gut int coverage. 32 of 43 ops have no registry entry and keep the format default — both sets are now declared and cross-checked at import. |
| 2 | **Ternary** — add `spec_A`/`spec_B`/`spec_C` | ✅ plumbing only, as scoped: no ternary op has a registry entry, so behaviour is unchanged. What it unblocks is the `c → 0` pole for `addcdiv`/`snake_beta`, previously unreachable because the divisor was pinned to `uniform(1, 2)`. |
| 3 | **Scalar binop** — sweep the scalar | ✅ axis over `{0, 1, 2, −2, 8, 0.25}` instead of one hard-coded 2.0. `ScalarDiv` skips `scalar=0` because the host inverts the divisor at compile time, so a device divide-by-zero does not exist for that op — recorded as N/A, not xfailed. |
| 4 | **Re-enable the float comparison ops** | ✅ `SfpuElwLt/Gt/Le/Ge` gain `test_sfpu_binary_float_comparison`, driving `a < b` / `a == b` / `a > b` in equal thirds with ±1.0 gaps. They previously had **no LLK-level correctness test at all**. The exact-tie third is the point: it is the only input where lt/gt and le/ge disagree. They stay out of the *random* sweep for the original (sound) near-tie reason. |
| 5 | **Record op arity in the registry** | ✅ `_NON_SFPU_UNARY_OPS` + `_UNARY_OPS_NOT_SWEPT` + `sfpu_unary_ops()`. 115 entries partition into 18 non-unary and 97 unary, of which 94 are swept and 3 exempt. The unary sweep's exhaustiveness is now a **collection-time error**, which Phase 0 could not check. |

### 3.1 Three defects the reroute exposed

The Phase 0 pattern repeated exactly as §1 predicts — the mechanism was small, the fallout was most of the work.

- **Per-operand pairing only ever filled the first tile pair.** `face_specs` is applied
  *positionally and is not cycled* (`generator.py:124`), so the fixed 8-entry list the old
  `_paired_two_tile_spec` returned covered tiles 0–1 and left every later pair with operand
  0's distribution on **both** sides. On the `[256, 128]` buffer that is 1 of 16 pairs
  genuinely paired: measured on the eq/ne stimuli, pair 0 came out 50% equal as intended and
  pairs 1–15 came out **100% equal**. So `mask`, `isclose` and `eq`/`ne` were each testing
  one sixteenth of what they appeared to. Pairing now needs the buffer geometry, so it moved
  into the driver and the crafted builders return a `(spec_A, spec_B)` pair.
- **`calculate_mask` only works for one tile pair per dest block.** Fixing the pairing turned
  this from invisible to 12 failures in 16 pairs. The kernel hard-codes its operands — data at
  `dst_reg[0]`, mask at `dst_reg[32]`, result in place — and ignores the forwarded dst indices,
  as `calculate_mask_binary`'s own comment states. The kernel loops `tile += 2` over the block,
  so the mask is applied to tile 0 repeatedly while tiles 2/4/6 are packed out as *unmasked*
  datacopy output against a golden that masks every pair. Exactly the non-block-leading pairs
  failed. The mask test now uses a `[64, 32]` buffer: 1 block, 1 pair, so every pair the kernel
  drives is the one the adapter supports.
- **`pow` and `xlogy` needed accuracy-bounded domains** (0f-style; neither had ever executed).
  `a**b` is `exp(b·ln a)`, so the error tracks the *product* handed to the shared exp
  approximation, not either operand: `3·ln3 = 3.30` clean, `4.83` → 4.9% off, `8.05` → 6.1%
  off against a 5% rtol. Both capped at 3. For `xlogy` the *absolute* error scales with x
  against a fixed atol, so x is capped at 4. Measured numbers are in the registry comments.

### 3.2 The binary suite was never deterministic

Neither `sfpu_binary` nor `default_spec_for_format` set a seed, so stimuli were redrawn on
every run — two xlogy variants failed a full-suite run and then reproduced in neither a
targeted rerun nor five other seeds. `eltwise_unary_sfpu` has always seeded; the binary and
ternary drivers now do too. **Worth knowing for the phases below:** a flaky variant is
indistinguishable from a real finding, so triage is only trustworthy once the draw is fixed.

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
| Binary | `sfpu_binary(..., spec_A=, src_A_override=)` | **`_paired_two_tile_spec(faceA, faceB)`** — both operands live in `buffer_A` (tile0 = in0, tile1 = in1). Build `faceA` from `edge_spec(op, fmt, Operand.A)` and `faceB` from `Operand.B`, so position *p* pairs as `(edge_A[p], edge_B[p])` |
| Ternary | `_run_sfpu_ternary(..., spec_A=, spec_B=, spec_C=)` | pass the three specs directly (Phase 1 added them); `c` is the one that matters — its default keeps the pole unreachable |
| Scalar | `_run_sfpu_binop_scalar(..., scalar=)` | the scalar is already an axis (Phase 1); a `spec_A` knob for the tensor operand still needs adding |

For binary cat-A/cat-D coverage the *cartesian product* of interesting A-values × B-values
matters more than element-wise pairing (divisor 0 against a positive, a negative and a zero
numerator are three different cases). `_build_shift_edge_case_src`
(`test_sfpu_binary.py:721`) already does exactly this — `itertools.product` over
`_SHIFT_EDGE_VALUES × _SHIFT_EDGE_AMOUNTS`, written into a two-tile tensor and fed through
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
| 1 | Reroute binary onto the registry; ternary + scalar spec params; comparison ops; arity check | ~330 across 2 commits | 7 binary ops rerouted, all ternary/scalar plumbed, 4 comparison ops gain their first LLK test | ✅ done |
| 2 | Edge metadata (`boundary_probes` + specials + `_OP_EDGE_POINTS`) | ~60 | all | ⬜ |
| 3 | `edge_spec()` + `_build_edge_pair_src` generalization (no compare change needed) | ~40 | all | ⬜ |
| 4 | 4 thin per-family edge tests + unary shift reuse + special-safe format matrix | ~90 | ~150 ops, auto-enrols future ops | ⬜ |
| 5 | golden `xfail` annotations | ~1 line each | the unmodelled few | ⬜ |
| F | new harnesses for untested kernels | large, per-kernel | the 10 untested + TopK stages | ⬜ (moe-gate in progress) |

**Bottom line:** ~230 lines of shared infrastructure closes categories A–E across all ~150
SFPU ops, and every future op auto-enrols by virtue of being in the enum. Phase 0 already paid
for the shared golden/comparison correctness, so the phases below do not have to.

---

## 10. Suggested sequencing

Phases 0 and 1 are done. Phase 0 left the registry **single-valued** — its recalibrated domain constants
proved arch-independent, so nothing below needs a per-arch axis.

1. ~~**Phase 1**~~ — **done** (§3). It leaves the binary driver with the `spec_B` knob and
   ternary/scalar with the spec parameters that Phases 3–4 need. Two things it hands forward:
   `xlogy`'s absolute error scales with x against a fixed atol (a per-op tolerance, Phase 5
   material), and 32 binary ops still have no registered domain at all.
2. **Phase 2–3** with a couple of pilot ops (`reciprocal`, `div`, `round`) to validate the
   probe derivation and the paired-operand delivery before scaling out.
4. **Phase 4** — establish the special-safe `(format, dest_acc)` matrix per arch **first**
   (§6a), then enable the family sweeps and triage into real-bug vs golden-gap (`xfail`).
4. **Phase 5** — extend goldens for the highest-value `xfail`ed specials.
5. **Category E** — fold unary shift into the shift edge builder; convert both Blackhole skips
   to xfails and track tt-metal#44750 / `docs/SFPU_INT32_SHIFT.md` as external dependencies.
6. **Category F** — finish the `generic_moe_gate_topk` harness, then schedule the rest by §8.

---

## 11. Known trap in the test infra (not a phase, but it will bite)

`TestConfig` calls `shutil.rmtree(ARTEFACTS_DIR)` at session setup — "always have a fresh build
when compiling" — against the **fixed** path `/tmp/tt-llk-build`. Any second pytest session on
the same host, including a one-op `-k` run started to triage something, deletes the build tree
out from under a running sweep. The victim reports `ld: cannot open output file` attributed to
whichever variant happened to be linking, which in a log reads exactly like a real kernel bug.

This produced two phantom failures during Phase 0's Blackhole verification before being
identified. Phases 1–4 are triage-heavy by nature — widen stimuli, run, investigate, re-run —
so it will recur. Worth filing and fixing separately: key the artefact root by session, or take
the existing `/tmp/tt-llk-build-shared.lock` around the rmtree.
