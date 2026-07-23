# SFPU Edge-Case Coverage — Minimal-Code Expansion Plan

**Companion to:** [SFPU_EDGE_CASE_COVERAGE.md](SFPU_EDGE_CASE_COVERAGE.md)
**Goal:** close the edge-case gaps catalogued in the coverage audit while adding the **least possible code** — no per-op edge tests, no duplicated stimulus lists.
**Repo:** `tt-metal/tt_metal/tt-llk/tests/python_tests/`

---

## 1. Guiding principle: one mechanism per gap-*category*, not per op

The 394 rows in the coverage audit look like 394 problems, but every `No` / `Excluded` row falls into one of **six edge categories**, and each category is closed by **one shared mechanism** that all ~150 ops reuse through the *existing* drivers:

| # | Gap category (from the audit) | Shared mechanism that closes it |
|---|-------------------------------|---------------------------------|
| A | Domain boundaries (`recip`/`div`/`log` at 0, `asin`/`acos`/`atanh`/`erfinv` at ±1, `acosh` at 1, `sqrt`/`rsqrt` ≤0, `pow` base≤0, `xlogy` y≤0) | **Auto-derive boundary probes from the data already in `_SFPU_UNDEFINED_RANGES`** |
| B | IEEE specials (±inf, NaN, +0.0, −0.0) into float ops | **One shared `FLOAT_SPECIALS` list injected by format class** |
| C | Integer extremes (INT32_MIN/MAX, UINT32_MAX, 0, −1, overflow) | **One shared `INT32_SPECIALS` / `UINT32_SPECIALS` list injected by format class** |
| D | Op-specific discrete edges (knees, thresholds, exact-tie rounding) | **A small `_OP_EDGE_POINTS` table — only for points that aren't already a domain boundary** |
| E | Shift-amount limits for the **unary** shift ops, and Blackhole shift/reduce arch-skips | **Reuse the existing binary shift edge-case builder; delete `@skip_for_blackhole` when the HW bug closes** |
| F | Entirely-untested kernels (`welfords`, `dropout`, `quant`, …) and perf-only ops (`TopK*` stages) | **Genuinely new harness per kernel — explicitly out of the cheap path (§7)** |

Categories **A–D** are the bulk of the audit and are closed by **one new metadata block + one stimulus builder + one comparison flag + ~5 thin test wrappers** — a few hundred lines total that scale to every op. Category **E** is a small refactor. Category **F** is the only place real new code is unavoidable, and we scope/prioritize rather than write it here.

---

## 2. What already exists (build on it, don't rebuild)

| Existing asset | Location | We reuse it for |
|----------------|----------|-----------------|
| Per-op safe domains + **undefined-region registry** | `helpers/sfpu_domains.py` (`_OP_DOMAIN_REGISTRY`, `_SFPU_UNDEFINED_RANGES`) | Boundary probes (cat A) come **for free** — the finite edge of every hole *is* the boundary to test |
| `StimuliSpec.custom(values=[...])` — explicit values at head of each face, zero-filled remainder | `stimuli_generator/spec.py:254`, `strategies/structured.py:60` | Injecting a small fixed set of edge values (cats A–D) |
| `StimuliSpec.uniform(intervals=[...])` | `spec.py` | Two-sided "just inside the boundary" sweeps |
| `_isinf_isnan_stimuli_spec()` — interleaves ±inf/NaN with a finite ramp | `test_sfpu_unary.py:502` | **Template** for cat B special injection |
| Shift edge-case builder — cartesian product of edge values × edge amounts | `test_sfpu_binary.py` (`_SHIFT_EDGE_VALUES/_AMOUNTS`, `_build_shift_edge_case_src`, `_shift_reference`) | **Template** + direct reuse for cat E |
| Inline NaN-aware compare `is_close | (isnan&isnan)` | `test_sfpu_unary.py:792` | Promote to a `compare_special` flag in `passed_test` (cat B) |
| `int32_reduce_extreme` INT32_MIN/MAX injection | `test_sfpu_reduce.py` | **Template** for cat C |
| `@parametrize(...)` op-sweep helper | `helpers/param_config.py:314` | The thin per-family edge test wrappers |
| `ULP_SWEEP` exhaustive fp16 enumeration | `spec.py:391` | Optional depth for 16-bit formats (§6) |

---

## 3. Phase 0 — the free win (≈0 new lines): stop the positive-only default

**Highest ROI item in the whole plan.** Finding #1 of the audit: `test_eltwise_unary_sfpu_float` (the `ALL_MATHOPS` list) passes no `spec_A`, so ~30 ops (`abs, neg, celu, elu, hardsigmoid, floor, ceil, trunc, frac, relu_max, threshold, exp, log, sqrt, rsqrt, square, tanh, silu, gelu, sin, cos, atanh, asinh, acosh, …`) are **only ever fed `uniform(0.1, 1.1)`** and never exercise their `x<0` branch.

Every one of these ops already has a signed entry in `_OP_DOMAIN_REGISTRY`. The fix is a **reroute, not new code**:

```python
# eltwise_unary_sfpu(): default spec_A to the op's real signed domain instead of positive-only
if spec_A is None:
    spec_A = exclude_undefined(mathop, for_op(mathop, formats.input_format).spec_A)
```

This single change gives ~30 ops their negative-branch / knee / saturation-tail coverage for free, and lets us later **merge `ALL_MATHOPS` into `DOMAIN_MATHOPS`** (deleting the split entirely). Net effect: negative code.

> Gotcha to verify: a handful of `ALL_MATHOPS` ops (`Fill`, `Identity`) are domain-agnostic and some fast-mode ops need their FastMode sweep preserved — keep the `SUPPORTED_FAST_MODE_OPS` product; only the `spec_A` source changes.

---

## 4. Phase 1 — one edge-metadata block in `sfpu_domains.py`

Add a single **source of truth** for edge values, most of it **derived** from data already present.

### 4a. Derive boundary probes from the existing undefined ranges (cat A) — no new per-op data
The finite edge of each hole in `_SFPU_UNDEFINED_RANGES` is exactly the boundary we want to probe (just-inside = defined, at/just-outside = the special result). One helper turns the *existing* registry into probe points:

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
This covers `reciprocal`(0), `log`/`sqrt`/`rsqrt`(0), `atanh`/`erfinv`(±1), `acosh`(1), `div`(divisor 0), `xlogy`(y 0), `pow`(base 0) — **all from data that already exists.**

### 4b. Shared special-value lists by format class (cats B, C) — a few constants, shared by all ops
```python
FLOAT_SPECIALS  = [float("inf"), float("-inf"), float("nan"), 0.0, -0.0]
INT32_SPECIALS  = [INT32_MIN, INT32_MAX, 0, -1, 1]        # INT32_MIN currently blocked by sign-mag Dst — xfail, see §5
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

---

## 5. Phase 2 — one `edge_spec()` builder + Phase 3 special-aware compare

### 5a. `edge_spec(op, fmt)` — one function feeds every driver
```python
def edge_spec(op, fmt, operand=Operand.A):
    """StimuliSpec.custom() combining domain boundaries + op knees + format specials,
    clipped to lie within the op's safe domain where a real value is required."""
    vals = [v for (o, v) in boundary_probes(op) if o == operand]
    vals += _OP_EDGE_POINTS.get(op, [])
    vals += format_specials(fmt)
    return StimuliSpec.custom(values=dedup(vals))   # custom = head-of-face, zero-filled remainder
```
`custom` already exists and does exactly the placement we need. A tile is far bigger than these value lists, so remainder-zero-fill is harmless (and 0.0/−0.0 are themselves useful probes).

### 5b. Special-aware comparison — promote the existing inline pattern
`passed_test` currently treats NaN as a mismatch. `test_sfpu_unary.py:792` already has the fix inline; lift it into a flag (~6 lines):
```python
def passed_test(..., compare_special=False):
    ...
    if compare_special:
        both_nan = torch.isnan(golden) & torch.isnan(res)
        both_inf = torch.isinf(golden) & torch.isinf(res) & (torch.sign(golden) == torch.sign(res))
        is_valid = is_close | both_nan | both_inf
```
Edge tests pass `compare_special=True`; everything else is unchanged.

---

## 6. Phase 4 — one thin edge test per family (reuses existing drivers)

No new driver, no new C++ source. Each is a ~15-line `@parametrize` wrapper over the **existing** driver, iterating the family's full op list with `spec_A=edge_spec(...)`. Five functions cover the whole matrix:

```python
@parametrize(mathop=get_sfpu_unary_operations(), formats=[...], dest_acc=[No, Yes])
def test_eltwise_unary_sfpu_edges(mathop, formats, dest_acc):
    eltwise_unary_sfpu("sources/eltwise_unary_sfpu_test.cpp", formats, dest_acc,
                       ApproximationMode.No, mathop, FastMode.No, [64, 64],
                       spec_A=edge_spec(mathop, formats.input_format),
                       compare_special=True)
```
…and the analogous `test_sfpu_binary_edges`, `test_sfpu_ternary_edges`, `test_sfpu_binop_scalar_edges`. Because `edge_spec` is keyed off the op, **adding a new op to the enum auto-enrolls it in edge testing** — zero incremental code per future op.

- **Binary** div/pow/xlogy/fmod/remainder: `edge_spec` on `Operand.B` injects the divisor-zero / base-zero / y≤0 probes the current positive-only default avoids.
- **Ternary** `addcdiv`/`snake_beta`: inject `c=0` and tiny/negative `c`; `lerp`: inject weight `0, 1, >1`.
- **Scalar** binops: sweep the scalar constant over `{0, ±large, ±tiny}` instead of the single hard-coded `2.0` (parametrize the existing `_SCALAR_BITS`).

### Category E (shift + arch): reuse, don't rewrite
- **Unary shift ops** (`LeftShift`/`RightShift`) currently use a *fixed* shift of 3. Generalize the existing `_build_shift_edge_case_src` / `_SHIFT_EDGE_AMOUNTS` (already covering `{0..31,32,>32,negative}`) to the unary driver — the builder and golden `_shift_reference` are already written; only the wiring differs.
- **Blackhole arch-skips** (shift edge test, `int32_reduce_extreme`): tracked to HW bugs (SFPU_INT32_SHIFT.md, tt-metal#44750). Low-code action = **delete the `@skip_for_blackhole` once the bug closes**; until then, leave a `xfail(reason=...)` so regressions surface.

### Optional depth (only where cheap)
For 16-bit formats, `spec_A=StimuliSpec.ulp_sweep(low, high)` gives exhaustive per-ULP coverage of a boundary neighborhood with one line — worth adding for a few high-value ops (`reciprocal`, `log`, `sqrt`) but not required.

---

## 7. Golden readiness (the one real per-op cost — kept small)

Injecting specials only helps if the golden defines the expected result. Reality from the audit:
- **Most goldens are torch-based** and already produce the correct `inf`/`nan`/`0` (e.g. `torch.reciprocal(0)=inf`, `torch.log(0)=-inf`). With `compare_special=True` these **just work** — no golden change.
- **A few goldens explicitly don't model non-finite** (audit notes: `xlogy`, `addcmul` under dest_acc, some int paths). For those, `pytest.mark.xfail(reason=...)` the specific special until the golden is extended — a one-line annotation, not a rewrite.
- **INT32_MIN** is a known sign-magnitude-Dst limitation (already documented via the shift `xfail`). Reuse the same `xfail` convention; don't fight the HW.

Rule of thumb: **default to injecting the edge + `compare_special`; xfail the handful the golden can't yet express.** This keeps per-op work to at most one annotation.

---

## 8. Out of the cheap path — genuinely new harnesses (prioritize, don't inline)

These need a new C++ source + golden and cannot be done by the shared mechanism. Listed for planning, not to be written under the "minimal code" banner:

| Kernel | Status | Suggested priority |
|--------|--------|--------------------|
| `welfords`, `int_sum`, `cumsum`, `tiled_prod` | reduction-family, no test at all | High — reuse reduce harness scaffolding |
| `quant` | no correctness test | High — used in production quantization |
| `dropout`, `rand` | RNG kernels, need statistical golden | Medium — needs a distribution-level assert, not element-wise |
| `reshuffle_rows`, `copy_dest_values`, `max_pool_indices` | data-movement/index, no test | Medium |
| `generalized_moe_gate_topk` (experimental) | experimental, no test | Low |
| `TopKLocalSort` / `Merge` / `Rebuild` | perf-only; whole-op `topk` is tested | Medium — add stage-level correctness |
| `swiglu` | Quasar-only | Low — port existing Quasar test to WH/BH |

---

## 9. Effort / footprint estimate

| Phase | What | New/changed LOC (order of magnitude) | Ops covered |
|-------|------|--------------------------------------|-------------|
| 0 | Reroute positive-only default | **~5 (net negative after merge)** | ~30 unary ops gain negative branch |
| 1 | Edge metadata (`boundary_probes` + specials + `_OP_EDGE_POINTS`) | ~60 | all |
| 2–3 | `edge_spec()` + `compare_special` flag | ~30 | all |
| 4 | 5 thin per-family edge tests + shift reuse | ~90 | ~150 ops, auto-enrolls future ops |
| 5 | golden `xfail` annotations | ~1 line each, handful of ops | the unmodeled few |
| **A–E total** | **the whole cheap path** | **≈200 lines, once** | **every op in the audit** |
| F | new harnesses for untested kernels | large, per-kernel | the 11 untested + TopK stages |

**Bottom line:** ~200 lines of shared infrastructure closes categories A–E across all ~150 SFPU ops, and every future op auto-enrolls by virtue of being in the enum. Only the 11 genuinely-untested kernels (category F) require real per-kernel test code, and those are scoped and prioritized separately.

---

## 10. Suggested sequencing

1. **Phase 0** first — biggest coverage jump, negative net code, no new metadata. Ship it standalone.
2. **Phases 1–3** — land the metadata + builder + compare flag with a couple of pilot ops (`reciprocal`, `div`, `round`) to validate the golden/`compare_special` path.
3. **Phase 4** — enable the family sweeps; triage the resulting failures into real-bug vs golden-gap (`xfail`).
4. **Phase 5** — extend goldens for the highest-value `xfail`ed specials.
5. **Category E** — fold unary shift into the shift edge builder; file/track the Blackhole un-skips.
6. **Category F** — schedule the untested-kernel harnesses by the priority table in §8.
