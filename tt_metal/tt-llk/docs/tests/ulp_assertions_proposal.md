# Proposal: ULP-bounded assertions in the tt-llk test harness

Status: proposal / RFC.
Scope: `tests/python_tests/` (the pytest harness). No kernel or C++ change.

## 1. The problem

Every float test in tt-llk is gated by `passed_test()` in
[`tests/python_tests/helpers/utils.py`](../../tests/python_tests/helpers/utils.py), which
applies two checks:

1. `torch.isclose(golden, result, rtol, atol)` with a **per-format** tolerance from the
   `tolerances` table (`atol=0.05, rtol=0.05` for every one of Float32, Float16, Float16_b), and
2. `PCC > 0.99` over the whole tensor.

Neither is a usable accuracy gate for eltwise SFPU work:

* **PCC is a shape metric.** On a smooth monotone curve sampled over a wide domain, PCC is
  dominated by the large-magnitude points and stays above 0.99 through error levels that
  would be unacceptable in any consumer. It cannot distinguish a 1-ULP kernel from a
  30-ULP kernel.
* **A flat `atol/rtol` is format-blind and magnitude-blind.** `atol=0.05` on bf16 is ~6 ULP
  at 1.0 and ~0.02 ULP at 512. The same number means two different things at two ends of
  the same sweep, so a tolerance loose enough to pass the tail is blind in the middle.
* **It forces per-op tolerance widening that then hides regressions.** The current
  `CUSTOM_TOLERANCES` in
  [`tests/python_tests/test_eltwise_unary_sfpu.py`](../../tests/python_tests/test_eltwise_unary_sfpu.py)
  is exactly this: `SigmoidAppx` and `GeluAppx` carry `atol=0.13` because a coarse
  3-segment LUT peaks near the knees. Once that number is in place the test can no longer
  see a 10x accuracy regression anywhere else in the domain, and it cannot see the
  accuracy *improvement* either, which is the thing the current retuning push is producing.

What we actually want to assert is the thing the accuracy sweep already plots: *the
hardware result is within N representable steps of the correctly-rounded reference, at
every point*.

## 2. How the rest of tt-metal does it

The prior art is all on the ttnn side and it is mature. Four layers, worth copying in the
same order. (The `../../../../` links below resolve when tt-llk is checked out inside
tt-metal; from the standalone tt-llk repo, read them as paths relative to the tt-metal
root.)

### 2.1 The metric — `ulp()` / `comp_ulp()`

[`models/common/utility_functions.py:568`](../../../../models/common/utility_functions.py)

```python
def ulp(x):            # Goldberg's definition: nextafter(|x|) - |x|, with a finfo.max fixup
def comp_ulp(golden, calculated, ulp_threshold, allow_nonfinite=False):
    ...
    ulp_value = ulp(golden.type(calculated.dtype))   # measure in the *output* format
    ulp_tensor = torch.abs(calculated - golden) / ulp_value
    return torch.max(ulp_tensor) <= ulp_threshold, f"Max ULP Delta: {...}"
```

Details worth stealing:

* ULP is taken **in the calculated tensor's dtype**, even when the golden is higher
  precision. Measuring bf16 error against an fp32 ULP would silently divide by 65536.
* Non-finite values are checked for *positional* agreement first, then zeroed out before
  the arithmetic — NaN/Inf never enter the division.
* The failure message names the worst element and prints the whole formula
  (`|calculated X - golden Y| / ULP(golden) Z`), which is what makes a ULP failure
  actionable instead of just red.

### 2.2 The assertion — `assert_with_ulp()`

[`tests/ttnn/utils_for_testing.py:218`](../../../../tests/ttnn/utils_for_testing.py)

A thin keyword-only wrapper over `comp_ulp` that adds:

* **Keyword-only arguments**, deliberately: swapping golden and actual changes the metric.
* A **shape assert** before the compare.
* A `maximum_meaningful_ulp_threshold` table — `2**mantissa_bits` per dtype (128 for bf16,
  2^23 for fp32) — and a warning if a caller passes a threshold above it. Beyond that
  point the two values differ by more than an order of magnitude and ULP has stopped being
  the right metric.
* dtype mapping for block formats: `bfloat8_b` is measured **as bfloat16**, not as the
  float32 that `to_torch()` would hand back. Same resolution, correct ULP size.

### 2.3 The robust variants

* [`ulp_distance()` — `tests/ttnn/utils_for_testing.py:318`](../../../../tests/ttnn/utils_for_testing.py)
  Integer ULP distance in bit space: map sign-magnitude to a monotonic int
  (`negatives -> all_ones - bits`, `positives -> bits + sign_bit`), subtract, take the
  absolute value. Two properties `comp_ulp` does not have: it is **integer-valued across
  power-of-2 boundaries** (where `ulp(golden)` differs above and below, so the fractional
  metric reports 0.5/2.0 ULP for one representable step), and it can treat `+0`/`-0` as
  0 ULP apart, which matters because our pack paths canonicalise `-0.0`.
* [`measure_ulp_with_near_zero_atol()` — `:354`](../../../../tests/ttnn/utils_for_testing.py)
  Hybrid policy for cancellation-prone outputs: elements above `1%` of the tensor's
  dynamic range are gated on ULP; everything below is gated on a scaled `atol` instead.
  Also reports a **distribution** (mean / p95 / p99 / max), not only the max.
* [`ulp_distance_bf16_daz()` — `tests/ttnn/unit_tests/operations/eltwise/eltwise_test_utils.py:235`](../../../../tests/ttnn/unit_tests/operations/eltwise/eltwise_test_utils.py)
  A bf16 ULP distance computed over a **DAZ+FTZ value-order index**, i.e. an ordering in
  which every subnormal collapses onto zero. This is the model that matches the SFPU, and
  without it a flushed result sitting next to a subnormal golden reads as a large ULP
  error that is really a 0-ULP agreement under the hardware's own number system.

### 2.4 The policy layer — the per-op declared contract

[`ttnn/ttnn/decorators.py:24`](../../../../ttnn/ttnn/decorators.py)

```python
@dataclasses.dataclass(frozen=True)
class GoldenComparisonConfig:
    method: str                      # "ulp" | "allclose" | "skip"
    scope: str = "degenerate"        # apply always, or only where PCC is degenerate
    ulp_threshold: float | None = None
    rtol: float = 1e-5
    atol: float = 1e-4
    equal_nan: bool = True
    nonfinite: str = "strict"        # "strict" | "mask"
    mask: object | None = None       # compare only selected elements
```

The golden function for an op attaches this to its output tensor
(`set_golden_comparison_config(result, method="ulp", scope="all", ulp_threshold=3)` — see
[`ttnn/ttnn/operations/unary.py`](../../../../ttnn/ttnn/operations/unary.py) lines 248, 279,
294, 391), and `compare_tensors_using_pcc` honours it: ULP replaces PCC where the op
declares a ULP contract, PCC stays everywhere else. Individual unit tests do the same
thing by hand with a per-op/per-point threshold parameter — `run_activation_unary_test(..., ulp=2)`,
`test_tanh_bw_ulp.py`'s per-input `max_expected_ulp`.

**This is the shape to copy:** the ULP budget is a *declared property of the op* living
next to the op's other metadata, not a magic number in a test body. That is what makes it
reviewable and what makes an improvement visible (you lower the number in the same PR that
improves the kernel).

There is no reusable C++ prior art — the ULP mentions in `tests/tt_metal/**/*.cpp` are all
comments.

## 3. What tt-llk already has

Most of the machinery exists; it is just not wired to an assert.

| Piece | Where | State |
|---|---|---|
| `local_ulp(golden, out_fmt)`, `compute_pointwise_metrics()` → `signed_ulp_error` | [`helpers/accuracy_metrics.py`](../../tests/python_tests/helpers/accuracy_metrics.py) | Done. Same `nextafter` definition as ttnn's `ulp()`. |
| Per-format mantissa widths, analytic `format_ulp(fmt, magnitude)` | [`helpers/sfpu_domains.py:1265,1299`](../../tests/python_tests/helpers/sfpu_domains.py) | Done. |
| ULP-lattice compares for the block formats (`_bfp_block_aware_compare`, `_mxint_…`, `_mxfp_…`) | [`helpers/utils.py`](../../tests/python_tests/helpers/utils.py) | Done, and already the sole gate for MX. |
| Per-op domain/edge/specials registry to hang a budget off | [`helpers/sfpu_domains.py`](../../tests/python_tests/helpers/sfpu_domains.py) | Done. |
| ULP sweep + plots + CSV/parquet per op | [`accuracy/`](../../tests/python_tests/accuracy/), [`docs/tests/sfpu_accuracy_plots_metrics.md`](sfpu_accuracy_plots_metrics.md) | Done — but explicitly *"Sanity-assert only (no ULP threshold gating)"*. |
| A ULP gate in the functional tests | — | **Missing. This proposal.** |

Two facts about the existing harness make the gate cheap:

1. **`passed_test()` already casts both tensors to the output format's torch dtype**
   (`golden_tensor.type(format_dict[output_data_format])`), so golden and result are
   already bit-comparable in the same format. Integer ULP distance drops straight in.
2. **The golden is already a correctly-rounded, datapath-modelled reference.**
   `UnarySFPUGolden` quantizes the input to what unpack delivers, models the 16-bit Dest
   truncation, evaluates the op in Python `float` (i.e. float64), then applies the *same
   two roundings the hardware applies* (`cast_to_dest_dtype` to the Dest format, then to
   the output format). So "0 ULP" is an achievable, meaningful outcome, and the ULP
   distance is a true integer count of representable steps — no half-ULP handicap for the
   golden's own rounding.

Constraint: tt-llk's test venv has `torch`, `numpy`, `pandas`, `mpmath` (transitively) but
**no `ttnn` and no `models.common`** — see `tests/requirements.txt`. The ttnn helpers must
be **ported**, not imported. They are ~60 lines total.

## 4. Proposal

Three layers, each independently shippable. Layer 1 alone already buys the assertion.

### Layer 0 — `helpers/ulp.py`, the metric

New module, no device dependency, unit-testable on the host.

```python
# helpers/ulp.py
_ULP_DTYPES = {torch.bfloat16: (torch.int16, torch.int32, 0x8000,     0xFFFF),
               torch.float16:  (torch.int16, torch.int32, 0x8000,     0xFFFF),
               torch.float32:  (torch.int32, torch.int64, 0x80000000, 0xFFFFFFFF)}

def ulp_distance(golden: torch.Tensor, result: torch.Tensor, *,
                 flush_subnormals: bool = True) -> torch.Tensor:
    """Integer count of representable steps between `golden` and `result`.

    Both tensors must already be in the output format's dtype (passed_test casts them).
    +0 and -0 are 0 apart: the pack path canonicalises -0.0, so the 1-step gap the bit
    mapping would report is an artefact (see llk-signed-zero-lost-on-unpack).
    With flush_subnormals, every subnormal is collapsed onto zero first, which is the
    SFPU's own number system (FTZ) -- otherwise a legitimately flushed result reads as a
    multi-step error against a subnormal golden.
    """
```

Implementation: the monotonic sign-magnitude→int mapping from
`utils_for_testing.ulp_distance`, plus the DAZ collapse from `ulp_distance_bf16_daz`,
generalised over the dtype table. Plus two reporting helpers:

```python
def ulp_stats(dist, finite_mask) -> dict   # max, mean, p95, p99, exact_frac, worst_index
def ulp_failure_message(golden, result, dist, fmt) -> str
    # "max 7 ULP @ [1043]: result 1.0546875 vs golden 1.109375 (1 ULP = 7.8125e-03, Float16_b)"
```

Why integer bit-distance for the **gate** and not the fractional `|err|/ulp(golden)` the
accuracy sweep already reports: across a power-of-2 boundary the fractional metric returns
0.5 or 2.0 for a single representable step, so a budget of "1 ULP" is not a well-defined
predicate. Keep the fractional form for plots and CSVs (it is the better *diagnostic*), use
the integer form for the pass/fail line.

### Layer 1 — a ULP gate in `passed_test()`

`passed_test` has **295 call sites across 116 files**, so the change has to be
default-off and additive. One new keyword:

```python
def passed_test(golden_tensor, res_tensor, output_data_format=DataFormat.Float16_b,
                ..., max_ulp: int | None = None, near_zero_atol: float | None = None):
```

Behaviour when `max_ulp is not None` (float formats only):

1. Non-finite positions must agree exactly (reuse the existing NaN handling and the
   `nan_sign_is_unspecified` / `specials_after_nan_sign_gate` rules in `sfpu_domains.py` —
   do not re-derive them).
2. `dist = ulp_distance(golden, result)` over the finite lanes.
3. `ok = dist <= max_ulp`. Optionally `ok |= |err| <= near_zero_atol` for the
   cancellation-prone lanes (Layer 1b, see §5).
4. On failure, log `ulp_failure_message` **and** the existing coloured tile dump.
5. **Skip the PCC check.** A ULP budget is strictly stronger than `PCC > 0.99` on an
   eltwise tensor; running both only adds a second, weaker way to fail. (This mirrors the
   MX path, which already returns on its lattice verdict without consulting PCC.)

Everything else — the block-float lattice branches, the tolerance branches, the tile
printer — is untouched. `max_ulp=None` is bit-for-bit today's behaviour.

For `Bfp8_b` and the MX formats the ULP request should be **rejected or redirected**, not
silently applied: their spacing is set by a shared block exponent, so a per-element ULP
count against the fp32 view is meaningless. Either keep the existing lattice compare
(recommended — it is already a ULP-shaped criterion) or, following ttnn, measure Bfp8_b in
bf16 space. Raise on `max_ulp` + an MX format so nobody thinks they have a gate they don't.

### Layer 2 — the per-op budget registry

Copy ttnn's declared-contract model, in tt-llk's own idiom (a table in `sfpu_domains.py`
alongside `_OP_DOMAIN_REGISTRY` / `_APPROX_ACCURACY_MAX`, or a new
`helpers/sfpu_accuracy_budget.py`):

```python
@dataclass(frozen=True)
class AccuracyContract:
    metric: str = "ulp"              # "ulp" | "tolerance"
    max_ulp: int | None = None
    atol: float | None = None        # fallback / near-zero floor
    rtol: float | None = None
    near_zero_atol: float | None = None

# Budget is a property of (op, approx_mode, output format, dest_acc) -- approx mode alone
# moves the error by orders of magnitude, and the fp32 path exposes LUT segment joins that
# a bf16 output rounds away entirely (see bf16-sweep-blind-to-fp32-lut-joins).
_SFPU_ACCURACY_BUDGET: Dict[MathOperation, Dict[BudgetKey, AccuracyContract]] = {
    MathOperation.Abs:         {DEFAULT: AccuracyContract(max_ulp=0)},
    MathOperation.Neg:         {DEFAULT: AccuracyContract(max_ulp=0)},
    MathOperation.Square:      {DEFAULT: AccuracyContract(max_ulp=1)},
    MathOperation.Exp:         {(APPROX_NO, DataFormat.Float32): AccuracyContract(max_ulp=4),
                                (APPROX_YES, ANY):              AccuracyContract(max_ulp=64)},
    MathOperation.SigmoidAppx: {DEFAULT: AccuracyContract(max_ulp=16)},   # was atol=0.13
    ...
}

def accuracy_contract(op, *, output_format, approx_mode, dest_acc) -> AccuracyContract
```

The drivers then look it up exactly where `CUSTOM_TOLERANCES.get(mathop, ...)` sits today
in `test_eltwise_unary_sfpu.py` — two lines changed per driver, and the numbers move out of
the test file into the registry next to the op's domain and edge points.

An op with no entry falls back to `AccuracyContract(metric="tolerance")`, i.e. today's
behaviour, so enrolment is per-op and incremental.

### Layer 3 — seeding the numbers from the sweep we already run

Do not hand-guess budgets. The accuracy suite already writes per-op parquet/CSV with a
`signed_ulp_error` column for every (op, format, approx, fast, dest) variant. Add a small
script — `accuracy/emit_budget.py` — that reads `accuracy/_csv_output/<arch>/` and prints
registry entries at `ceil(p99.9_ulp * headroom)`, with the measured max and exact-match
fraction in a trailing comment:

```python
MathOperation.Tanh: {(APPROX_NO, DataFormat.Float32): AccuracyContract(max_ulp=6)},
#   wh: max 4.1 ULP, p99.9 3.8, 61% exact, 2048 pts, 2026-xx-xx
```

That closes the loop the current push needs: *measure → budget → gate*, and a kernel
improvement lands as a visible decrease in a checked-in number.

## 5. Decisions and traps to get right

* **Near-zero blow-up.** Where the golden crosses zero (`log` near 1, `tanhshrink`,
  `expm1` near 0, `sin` near π), `ulp(golden)` collapses and any absolute error explodes
  into a huge ULP count. Handle it the way `measure_ulp_with_near_zero_atol` does — a
  format-scaled `atol` floor for lanes below ~1% of the tensor's dynamic range — and set
  `near_zero_atol` per op in the registry. Do **not** paper over it by raising `max_ulp`;
  that reopens the hole the whole proposal is closing.
* **Denormals.** The SFPU flushes. Use the DAZ ordering by default (`flush_subnormals=True`).
  This is also why the fp16 path needs care: `UnarySFPUGolden` already models the FP16
  flush-to-zero explicitly, so the golden and the DAZ metric agree.
* **Signed zero and NaN sign.** Already solved in this harness and the solutions must be
  reused, not reinvented: `-0.0` dies on unpack (`llk-signed-zero-lost-on-unpack`), `-NaN`
  folds to the other operand on WH, and `specials_safe()` / `nan_sign_is_unspecified()`
  encode where a sign may be asserted at all. The ULP gate must consult the same
  predicates, or four known-good ops start failing on their edge sweeps.
* **A bf16 gate is blind to what the fp32 path shows.** Sub-ULP downward segment steps at
  LUT joins are invisible in bf16 and large in fp32. Budgets keyed only on the op, with a
  single number across formats, would set the bf16 number and never gate the interesting
  path — hence the format in the budget key, and hence enrolling the fp32 output format
  first.
* **Threshold sanity.** Port ttnn's `2**mantissa_bits` warning (128 for bf16, 1024 for
  fp16, 2^23 for fp32). A budget above it means "wrong by an order of magnitude" and the
  op should be on the tolerance metric, not the ULP one.
* **Budgets are architecture-dependent.** WH and BH SFPUs differ in available instructions
  and therefore in kernel (`wh-sfpu-vs-bh-exp-kernel-gaps`). Key the registry on
  `ChipArchitecture` where the measurements differ, defaulting to the shared entry.
* **Determinism.** The gate must be a max over a fixed, seeded stimulus set, not over a
  random draw per run, or the budget has to absorb sampling noise. The sweeps are already
  `torch.manual_seed(0)`; keep it, and prefer the deterministic ramp used by the accuracy
  harness over a random distribution for any op whose budget is tight (≤ 2 ULP).

## 6. Suggested sequencing

| Phase | Work | Verification |
|---|---|---|
| P0 | `helpers/ulp.py` + host-only unit tests (pow-2 boundaries, ±0, subnormal flush, NaN/Inf, all three dtypes) | `pytest helpers/` — no device |
| P1 | `passed_test(max_ulp=...)`, default `None` | Full suite unchanged; a deliberate ±1-ULP perturbation must fail |
| P2 | Registry + `accuracy_contract()`; enrol the exactly-rounded ops (`Abs`, `Neg`, `Identity`, `Relu`, `Square`, `Floor`/`Ceil`/`Trunc`) at `max_ulp=0..1` | These are the flakiness canaries: if a 0-ULP budget is stable across a week of nightlies, the metric is sound |
| P3 | `emit_budget.py`; enrol the transcendentals on Float32 then Float16_b | Budgets derived from the sweep, reviewed in the PR |
| P4 | Binary/ternary SFPU drivers; `--ulp-report` flag that prints max/p99 ULP per test **even on pass** | Gives a regression signal before a budget is crossed |

`--ulp-report` is worth having early and cheap to add: it turns every existing functional
test into an accuracy datapoint without changing a single verdict.

## 7. Open questions

1. **Where does the budget live** — in `sfpu_domains.py` next to the domains (one registry
   per op, easy to review) or in a separate `sfpu_accuracy_budget.py` (keeps a 2500-line
   file from growing)? Preference: separate module, imported by `sfpu_domains`.
2. **Does the ULP gate replace PCC for eltwise, or run beside it?** Recommendation:
   replace, as §4 Layer 1 step 5 argues. Needs sign-off since it changes what a green test
   means.
3. **Do we gate `Bfp8_b` in bf16 ULP space** (ttnn's choice) or keep the existing block
   lattice compare? Recommendation: keep the lattice compare; it is already the stronger,
   block-aware criterion.
4. **Should budgets be per-arch from the start**, or single-valued with per-arch overrides
   added as measurements diverge? Recommendation: single value + override, to keep the
   table readable.
