# SFPU LLK Edge-Case Test-Coverage Audit

**Issue:** [tenstorrent/tt-metal#49739 — [LLK] SFPU testing edge cases](https://github.com/tenstorrent/tt-metal/issues/49739)
**Plan for what is left:** [SFPU_EDGE_CASE_EXPANSION_PLAN.md](SFPU_EDGE_CASE_EXPANSION_PLAN.md)
**Audited:** 2026-07-23 · **Regenerated from code:** 2026-08-12 (revision 6)
**Scope:** All SFPU LLK kernels in `tt-metal/tt_metal/tt-llk`, audited through the tt-llk Python test
infra (`tests/python_tests/`). Wormhole B0 and Blackhole share essentially the same SFPU kernel set
(BH adds only `topk_xl`), so this audit treats them together and notes arch-specific gaps inline.
Quasar has its own suite under `quasar/` and is **out of scope** — an op driven only from `quasar/`
counts as untested here.

> ### Revision 6 — the tables are now generated, not overlaid
>
> Revisions 2–5 layered "override" notes on top of a body written on 2026-07-23, so reading any row
> meant reconciling it against up to four later sections. That is gone. **§4 is regenerated directly
> from the code** — every op, its registered domain, its actual probe values, and which tests drive
> it — and the stale body it replaces has been deleted rather than annotated. §2 is the list of what
> is still untested, which is what the previous revisions made hardest to extract.
>
> The four phase summaries and the PR-3 plan that used to sit alongside this file are **deleted**:
> their content was verified present in the code (counts, tables, guards, gates all match), so the
> code is now the record. §7 says how to re-derive everything here.

---

## How to read this document

For every SFPU op we record whether the test infra **deliberately drives and asserts** each class of
edge, using the six categories the work was organised around:

| Cat | Edge class | Mechanism that closes it |
|---|---|---|
| **A** | Domain boundaries — poles and branch cuts (`1/0`, `log 0`, `asin(±1)`, `acosh 1`) | `_OP_SINGULARITIES` → `boundary_probes()` → `edge_spec()` |
| **B** | IEEE specials — `±inf`, `NaN`, `+0.0`, `-0.0` | `FLOAT_SPECIALS`, gated by `specials_safe()` **and** `SPECIALS_READY_OPS` |
| **C** | Integer extremes — `INT32_MIN/MAX`, `UINT32_MAX`, `0`, `-1` | `integer_specials()` delivered as a raw `src_A_override` |
| **D** | Op-specific discrete edges — knees, thresholds, exact rounding ties | `_OP_EDGE_POINTS` |
| **E** | Shift-amount limits for the **unary** shift ops | blocked on a C++ template parameter |
| **F** | Kernels with no `MathOperation` entry at all | a new harness per kernel |

Symbols in §4: **✅** the edge sweep drives this op; **⬜** it does not (with the reason);
**⚠️** the op diverges from its golden at a driven edge (§5).

---

## 1. Coverage at a glance

All figures re-derived from the tree on 2026-08-12 (see §7 for the commands).

| | Count |
|---|---|
| Unique `MathOperation` members | **182** |
| Ops with `SFPU_UNARY` dispatch | 118 — of which **97** have a registered domain and are swept |
| ↳ swept unary ops | 31 broad + 63 standard + 3 perf-only (`TopK*` stages) |
| ↳ with ≥1 deliberate edge value (cat A and/or D) | **50** (§4.1) |
| ↳ smooth everywhere, so cat B is their *only* edge | **47** (§4.2) |
| Unary ops **outside** the registry — in neither sweep | **21** (§4.3): 5 predicates, 3 threshold, 4 int max/min, 2 unary shift, `Typecast`, `Relu`, 4 perf-only int, `SfpuSwiGLU` |
| Binary SFPU ops (float + shift) | 43 — 11 with a registered domain, 5 with a driven pole |
| Binary integer / ternary / scalar / reduce / FPU-binary ops | 5 / 5 / 5 / 3 / 3 |
| `_OP_SINGULARITIES` entries | 19 |
| `_OP_EDGE_POINTS` entries | 43 |
| `SPECIALS_READY_OPS` (cat B opt-in) | **0** — cat B is wired and switched off |
| `(format, dest_acc)` triples that can carry specials | 7 of 40, measured on Wormhole |
| Ops diverging from their golden at a driven edge | **10**, over 42 `(op, format, dest_acc)` cells |
| Host-side guards over the gates and metadata | 107 tests (`test_sfpu_domains.py`) |

**Category status:** A ✅ closed for every op that has a boundary · B 🟡 measured, off · C ✅ closed
for the 5 ops whose kernels claim the full int32 range · D ✅ closed for all 43 knees ·
E ⬜ blocked on C++ · F ⬜ 11 kernels untouched.

---

## 2. What is still NOT tested

Ordered by how much coverage each item is worth. This is the list to work from; §2 of the plan
sequences it.

### 2.1 Cat B — IEEE specials, for every op (largest gap)

Wired end to end and **deliberately switched off**: `SPECIALS_READY_OPS` is empty, so no op injects
`±inf` / `NaN` / signed zeros today. The blocker is not the stimulus and not the format matrix — it is
that torch-backed goldens do not define a result for non-finite *inputs*. Turning it on regardless
gives **272 failures out of 564 variants**.

This is the entire edge story for the **47 smooth ops in §4.2** — they have no knee and no pole, so
`edge_spec()` returns `None` and they skip the edge sweep. Half the unary op list has no deliberate
edge coverage until this is done.

The only ops that inject specials today are the five predicates
(`Isinf`, `Isposinf`, `Isneginf`, `Isnan`, `Isfinite`) via `test_eltwise_unary_sfpu_isinf_isnan`,
which is also the instrument that measured the safe matrix in §6.

### 2.2 Ternary operand-C edges

`SfpuAddcdiv` and `SfpuSnakeBeta` divide by `c`, and `c` is pinned to `uniform(1, 2)` — the pole is
**deliberately unreachable**. `SfpuLerp`'s weight boundaries (`0`, `1`, `> 1`) are equally undriven.
No ternary op has a registered domain or a singularity entry. Blocked on `OperandSpecs` carrying only
`spec_A` and `spec_B`.

### 2.3 Cat E — the unary shift amount

`LeftShift` and `RightShift` run at a **fixed shift of 3** with small positive inputs.
`SHIFT_AMOUNT` is a C++ `constexpr` paired with a golden constant, so sweeping it needs a
`TemplateParameter`, not test wiring. The binary side is fully covered by contrast
(`_SHIFT_EDGE_AMOUNTS` drives `{0..31, 32, 33, 40, 63, 100, 1000, −1, −5, −32, −1000}`).

### 2.4 Ops with no WH/BH correctness test at all

**Driven only from the out-of-scope Quasar suite (6 ops):**
`SfpuSwiGLU`, `SfpuElwmulInt`, `SfpuGeInt`, `SfpuGtInt`, `SfpuLeInt`, `SfpuLtInt`.
The four int comparisons and the int multiply have a golden (`_gt_int`, `_mul`, …) but nothing on
Wormhole or Blackhole calls them.

> Two ops that *look* untested are not, and the reason is worth knowing before re-auditing: an op
> can be driven under an **alias**. `MathOperation` has two — `SfpuWhere`/`TTNNWhere` and
> `LogicalNot`/`LogicalNotUnary` — and in both cases the test names the second spelling
> (`test_sfpu_ternary.py` and `test_sfpu_unary.py` respectively). A grep for the canonical name finds
> nothing. Any tooling over this audit has to resolve aliases; §7's inventory does.

**Perf-only — a perf test exists, no functional golden or assert (7 ops):**
`AddInt32`, `SubInt32`, `AbsInt32`, `BitwiseNot` (all in `perf_eltwise_unary_sfpu_int32.py`; the file
records the reason — the int32-unary functional sweep is blocked by the fast-tilize gap,
tt-llk#495), plus `TopKLocalSort`, `TopKMerge`, `TopKRebuild` (whole-op `topk` is tested, the three
stages are not).

### 2.5 Cat F — kernels with no `MathOperation` entry (11)

A header exists; nothing in the Python infra can reach them. Confirmed still absent:
`welfords`, `dropout`, `quant`, `cumsum`, `reshuffle_rows`, `int_sum`, `tiled_prod`,
`copy_dest_values`, `generalized_moe_gate_topk`, `max_pool_indices`, `rand`.

`generic_moe_gate_topk` has come **off** this list — `test_sfpu_generic_moe_gate_topk.py` and
`sources/sfpu_generic_moe_gate_topk_test.cpp` both exist.

### 2.6 Integer edges that remain out of scope *by kernel design*

Not a test gap, and worth not re-filing: 12 of the 17 int binary ops document a **narrower valid
range** than the format's, so the extremes are outside what the kernel promises.
`_INT_BINARY_STIMULI` records each one — `div`/`fmod` below 2²⁴ for an exact int→fp32 reciprocal,
`mul` below ~46340 so the product stays under 2³¹, `lcm` assuming |a|,|b| < 2¹⁵, `max`/`min`
non-negative so signed and unsigned agree. Cat C covers the 5 that do claim the full range
(`SfpuBitwiseAnd/Or/Xor`, `SfpuEqInt`, `SfpuNeInt`).

`INT32_MIN` itself is excluded everywhere and that is hardware: sign-magnitude Dst reads
`0x80000000` as "negative zero" and cannot round-trip it. It has a dedicated documenting xfail
(`test_sfpu_binary_int_shift_int32_min_unsupported`), and `INT32_MIN + 1` stands in for it.

Still genuinely open on the integer side: `gcd`/`lcm` with `0` and negatives, and `INT32_MIN` for
`abs_int32`.

### 2.7 Format / overflow extremes

No float op is driven at its format ceiling, at a denormal, or through an overflow-to-`inf`
transition. `clip_to_format()` exists to keep probes *inside* the representable range, so the
mechanism currently prevents this rather than enabling it. Untouched since the original audit.

### 2.8 Verification gaps, not coverage gaps

- **Blackhole.** The reduce xfail and the scalar presubmit/nightly split were measured on p100a; the
  three edge sweeps and `specials_safe()`'s matrix were **not**. Two parts are arch-sensitive by
  construction: the safe matrix (unpack paths differ, and it is a measurement rather than a
  derivation) and the shift xfail, whose whole purpose is the Blackhole path.
- **The accurate exp path over (16, 80].** The registry now carries the range bound and
  `_APPROX_ACCURACY_MAX` the approximation bound, applied only in `ApproximationMode.Yes`. The
  accurate path over that region has never been isolated on hardware.
- **Whether `-0.0` reaches DEST** on the non-unpack-to-dest path — see §5.2. Three ops' xfail
  reasons depend on the answer.
- **`WITH_COVERAGE` builds** and **Bfp4_b output formats** (`Float16 -> Bfp4_b` fails 100% at
  `dest_acc=No` on Wormhole and is unexplained; every neighbouring cell is clean).
- **CI runs none of this.** The broad unary profile runs in no automated job on any arch — every LLK
  pytest job either excludes `nightly` or runs `--coverage`, under which the broad profile is skipped
  wholesale. So these gains are currently unguarded. See the plan §8.

---

## 3. The four systemic findings from the original audit

| # | Finding | Status |
|---|---|---|
| 1 | Unary float sweep is positive-only (`uniform(0.1, 1.1)`, no `spec_A`) | ✅ **fixed.** The sweep defaults `spec_A` to the op's registered signed domain, bounded by the narrowest format in the pipeline (`for_op_pipeline` + `exclude_undefined`), and a missing registry entry is a hard `KeyError`. 31 ops gained their `x<0` branch |
| 2 | Binary / ternary / scalar suites never import `sfpu_domains.py` | 🟡 **closed for binary.** 11 of 43 binary ops now have a registered domain; the other 32 keep the format default. Ternary and scalar have the per-operand plumbing but **no registry entries at all** — see §2.2 |
| 3 | IEEE specials injected for exactly one op family | 🟡 **measured, deliberately not enabled.** The safe `(format, dest_acc)` surface is now data (§6) rather than assumption, pinned by 107 host-side tests. Injection is off because the *goldens* fail, not the pipeline — §2.1 |
| 4 | Integer sign/extreme edges structurally excluded (`_get_integer_bounds` returns `min+1`) | 🟡 **closed where the kernels allow it.** Extremes go through a raw `src_A_override`; `test_sfpu_binary_int_extremes` drives `{INT32_MIN+1, -1, 0, 1, INT32_MAX}²` over the 5 ops that claim the full range. The other 12 are out of scope by kernel design — §2.6 |

**#3 and #4 both moved, but neither to a plain "fixed", and for the same reason:** the mechanism was
the easy part and something outside the test infra bounds how far it can go. For #3 it is the
goldens; for #4 it is the kernels' own documented ranges.

---

## 4. Per-op coverage

Generated from the code — see §7. Every one of the 182 `MathOperation` members appears in exactly one
table below.

`broad` = the full format matrix including block floats and both approximation modes; `standard` =
Float16_b + Float32, `ApproximationMode.No`. Cat A probe values are shown as the singular point and
the side the op is **defined** on (`abo` = above, `bel` = below, `bot` = both); the probe offset
itself is format-relative, so `Reciprocal`'s `0.0` becomes `±0.015625` in Float16_b and `±0.25` in
Bfp4_b, and with `dest_acc=No` a 32-bit probe is stepped by a bfloat16 ULP so the 16-bit DEST cannot
truncate it back onto the boundary.

The three unary tables partition on **why** an op is where it is: §4.1 has a boundary or a knee and is
driven; §4.2 has neither, so the edge sweep skips it and cat B is all that is left; §4.3 has no
registered domain at all, so neither sweep reaches it and coverage depends on a dedicated test.

#### 4.1 Unary ops with a deliberate edge driven — cat A and/or cat D (50 ops)

| Op | Kernel | Random sweep | Cat A boundary (side defined on) | Cat D knees / ties | Edge sweep | Other test | ⚠️ |
|---|---|---|---|---|---|---|---|
| `Acos` | `acos` | standard | -1.0 (abo); 1.0 (bel) | — | ✅ | — |  |
| `Acosh` | `acosh` | broad | 1.0 (abo) | — | ✅ | — |  |
| `Asin` | `asin` | standard | -1.0 (abo); 1.0 (bel) | — | ✅ | — |  |
| `Atanh` | `atanh` | broad | -1.0 (abo); 1.0 (bel) | — | ✅ | — |  |
| `Ceil` | `ceil` | broad | — | `-2, -1, 0, 1, 2` | ✅ | — |  |
| `Celu` | `celu` | broad | — | `0, -0` | ✅ | — |  |
| `Clamp` | `clamp` | standard | — | `-1, 1` | ✅ | — |  |
| `Elu` | `elu` | broad | — | `0, -0` | ✅ | — |  |
| `EqualZero` | `equal_zero` | standard | — | `0, -0` | ✅ | — |  |
| `Erfinv` | `erfinv` | standard | -1.0 (abo); 1.0 (bel) | — | ✅ | — | ⚠️ |
| `Floor` | `floor` | broad | — | `-2, -1, 0, 1, 2` | ✅ | — |  |
| `Frac` | `frac` | broad | — | `-1.5, -1, 1, 1.5` | ✅ | — |  |
| `GreaterThanEqualZero` | `greater_than_equal_zero` | standard | — | `0, -0` | ✅ | — |  |
| `GreaterThanZero` | `greater_than_zero` | standard | — | `0, -0` | ✅ | — |  |
| `Hardmish` | `hardmish` | standard | — | `-2, 0` | ✅ | — |  |
| `Hardshrink` | `hardshrink` | standard | — | `-0.5, 0.5` | ✅ | — |  |
| `Hardsigmoid` | `hardsigmoid` | broad | — | `-3, 3` | ✅ | — |  |
| `Hardtanh` | `hardtanh` | standard | — | `-1, 1` | ✅ | — |  |
| `Heaviside` | `heaviside` | standard | — | `0, -0` | ✅ | — | ⚠️ |
| `LessThanEqualZero` | `less_than_equal_zero` | standard | — | `0, -0` | ✅ | — |  |
| `LessThanZero` | `less_than_zero` | standard | — | `0, -0` | ✅ | — |  |
| `Log` | `log` | broad | 0.0 (abo) | — | ✅ | — |  |
| `Log1p` | `log1p` | broad | -1.0 (abo) | — | ✅ | — |  |
| `LogWithBase` | `log_with_base` | standard | 0.0 (abo) | — | ✅ | — |  |
| `Lrelu` | `lrelu` | standard | — | `0, -0` | ✅ | — |  |
| `NotEqualZero` | `not_equal_zero` | standard | — | `0, -0` | ✅ | — |  |
| `Prelu` | `prelu` | standard | — | `0, -0` | ✅ | — |  |
| `Rdiv` | `rdiv` | standard | 0.0 (bot) | — | ✅ | — |  |
| `Reciprocal` | `reciprocal` | broad | 0.0 (bot) | — | ✅ | — |  |
| `ReluMax` | `relu_max` | broad | — | `0, 5` | ✅ | — |  |
| `ReluMin` | `relu_min` | broad | — | `5` | ✅ | — |  |
| `Round` | `round` | standard | — | `-2.5, -1.5, -0.5, 0.5, 1.5, 2.5` | ✅ | — |  |
| `Rsqrt` | `rsqrt` | broad | 0.0 (abo) | — | ✅ | — |  |
| `RsqrtCompat` | `rsqrt_compat` | standard | 0.0 (abo) | — | ✅ | — | ⚠️ |
| `Selu` | `selu` | standard | — | `0, -0` | ✅ | — |  |
| `Sign` | `sign` | standard | — | `0, -0` | ✅ | — | ⚠️ |
| `Signbit` | `signbit` | standard | — | `0, -0` | ✅ | — | ⚠️ |
| `Softplus` | `softplus` | standard | — | `20` | ✅ | — |  |
| `Softshrink` | `softshrink` | standard | — | `-0.5, 0.5` | ✅ | — |  |
| `Sqrt` | `sqrt` | broad | 0.0 (abo) | — | ✅ | — |  |
| `SqrtCustom` | `sqrt_custom` | standard | 0.0 (abo) | — | ✅ | — |  |
| `Threshold` | `threshold` | broad | — | `5` | ✅ | — |  |
| `Trunc` | `trunc` | broad | — | `-1, 0, 1` | ✅ | — |  |
| `UnaryGe` | `unary_ge` | standard | — | `0.5` | ✅ | — |  |
| `UnaryGt` | `unary_gt` | standard | — | `0.5` | ✅ | — |  |
| `UnaryLe` | `unary_le` | standard | — | `0.5` | ✅ | — |  |
| `UnaryLt` | `unary_lt` | standard | — | `0.5` | ✅ | — |  |
| `UnaryMax` | `unary_max` | standard | — | `0, -0` | ✅ | — |  |
| `UnaryMin` | `unary_min` | standard | — | `0, -0` | ✅ | — |  |
| `Xielu` | `xielu` | standard | — | `0, -0` | ✅ | — |  |

#### 4.2 Unary ops smooth everywhere — cat B is their **entire** edge story (47 ops)

| Op | Kernel | Random sweep | Registered domain | Edge sweep | Other test |
|---|---|---|---|---|---|
| `Abs` | `abs` | broad | yes | ⬜ skips | — |
| `Add1` | `add1` | standard | yes | ⬜ skips | — |
| `Asinh` | `asinh` | broad | yes | ⬜ skips | — |
| `Atan` | `atan` | standard | yes | ⬜ skips | — |
| `CastFp32ToFp16a` | `cast_fp32_to_fp16a` | standard | yes | ⬜ skips | — |
| `Cbrt` | `cbrt` | standard | yes | ⬜ skips | — |
| `Cos` | `cosine` | broad | yes | ⬜ skips | — |
| `Cosh` | `cosh` | standard | yes | ⬜ skips | — |
| `Digamma` | `digamma` | standard | yes | ⬜ skips | — |
| `Erf` | `erf` | standard | yes | ⬜ skips | — |
| `Erfc` | `erfc` | standard | yes | ⬜ skips | — |
| `Exp` | `exponential` | broad | yes | ⬜ skips | — |
| `Exp2` | `exp2` | broad | yes | ⬜ skips | — |
| `ExpWithBase` | `exp_with_base` | standard | yes | ⬜ skips | — |
| `Expm1` | `expm1` | standard | yes | ⬜ skips | — |
| `Expm1Cw` | `expm1_cw` | standard | yes | ⬜ skips | — |
| `Fill` | `fill` | broad | yes | ⬜ skips | — |
| `Fmod` | `fmod` | standard | yes | ⬜ skips | — |
| `Gelu` | `gelu` | broad | yes | ⬜ skips | — |
| `GeluAppx` | `gelu_appx` | standard | yes | ⬜ skips | — |
| `GeluDerivative` | `gelu_derivative` | standard | yes | ⬜ skips | — |
| `GeluTanh` | `gelu_tanh` | broad | yes | ⬜ skips | — |
| `I0` | `i0` | standard | yes | ⬜ skips | — |
| `I1` | `i1` | standard | yes | ⬜ skips | — |
| `Identity` | `identity` | standard | yes | ⬜ skips | — |
| `Lgamma` | `lgamma` | standard | yes | ⬜ skips | — |
| `Mish` | `mish` | standard | yes | ⬜ skips | — |
| `Neg` | `negative` | broad | yes | ⬜ skips | — |
| `Polygamma` | `polygamma` | standard | yes | ⬜ skips | — |
| `Remainder` | `remainder` | standard | yes | ⬜ skips | — |
| `Rpow` | `rpow` | standard | yes | ⬜ skips | — |
| `Sigmoid` | `sigmoid` | standard | yes | ⬜ skips | — |
| `SigmoidAppx` | `sigmoid_appx` | standard | yes | ⬜ skips | — |
| `Silu` | `silu` | broad | yes | ⬜ skips | — |
| `Sin` | `sine` | broad | yes | ⬜ skips | — |
| `Sinh` | `sinh` | standard | yes | ⬜ skips | — |
| `Softsign` | `softsign` | standard | yes | ⬜ skips | — |
| `Square` | `square` | broad | yes | ⬜ skips | — |
| `Tan` | `tan` | standard | yes | ⬜ skips | — |
| `Tanh` | `tanh` | broad | yes | ⬜ skips | — |
| `TanhDerivative` | `tanh_derivative` | standard | yes | ⬜ skips | — |
| `TanhDerivativeLut` | `tanh_derivative_lut` | standard | yes | ⬜ skips | — |
| `Tanhshrink` | `tanhshrink` | broad | yes | ⬜ skips | — |
| `TopKLocalSort` | `topk_local_sort` | **perf-only** | yes | ⬜ skips | — |
| `TopKMerge` | `topk_merge` | **perf-only** | yes | ⬜ skips | — |
| `TopKRebuild` | `topk_rebuild` | **perf-only** | yes | ⬜ skips | — |
| `UnaryPower` | `power` | standard | yes | ⬜ skips | — |

#### 4.3 Unary ops outside `_OP_DOMAIN_REGISTRY` — not in either sweep (21 ops)

These have no registered domain, so `sfpu_unary_ops()` excludes them from the broad/standard
sweeps **and** from the edge sweep. Each is either covered by a dedicated test, deliberately
unreachable, or genuinely uncovered — the last column says which.

| Op | Kernel | Cat D knees | Dedicated test | Status |
|---|---|---|---|---|
| `AbsInt32` | `abs_int32` | — | **none (WH/BH)** | ⬜ **perf-only** (tt-llk#495) — §2.4 |
| `AddInt32` | `add_int32` | — | **none (WH/BH)** | ⬜ **perf-only** (tt-llk#495) — §2.4 |
| `BitwiseNot` | `bitwise_not` | — | **none (WH/BH)** | ⬜ **perf-only** — §2.4 |
| `Isfinite` | `isfinite` | — | `unary` | ✅ cat B — as `Isinf` |
| `Isinf` | `isinf` | — | `unary` | ✅ cat B — with its four siblings, the only ops injecting `±inf`/`NaN` today |
| `Isnan` | `isnan` | — | `unary` | ✅ cat B — as `Isinf` |
| `Isneginf` | `isneginf` | — | `unary` | ✅ cat B — as `Isinf` |
| `Isposinf` | `isposinf` | — | `unary` | ✅ cat B — as `Isinf` |
| `LeftShift` | `left_shift` | — | `unary` | 🟡 **cat E open** — fixed shift of 3, positive inputs; needs a C++ `TemplateParameter` (§2.3) |
| `LogicalNot` | `logical_not_unary` | `0, -0` | `unary` | ✅ cat D — exact threshold forced by `test_eltwise_unary_sfpu_threshold`, which names it `LogicalNotUnary` |
| `Relu` | `relu` | — | `plot` | ➖ unreachable by design — applied by the packer (`STACC_RELU`), not a `SfpuType`. The only reference is a plotting script |
| `RightShift` | `right_shift` | — | `unary` | 🟡 **cat E open** — as `LeftShift` |
| `SfpuSwiGLU` | `swiglu` | — | **none (WH/BH)** | ⬜ **Quasar-only** — §2.4 |
| `SubInt32` | `sub_int32` | — | **none (WH/BH)** | ⬜ **perf-only** (tt-llk#495) — §2.4 |
| `Typecast` | `typecast` | — | `eltwise_unary_typecast` | 🟡 value coverage only; no special or format-extreme injection |
| `UnaryEq` | `unary_eq` | `0.5` | `unary` | ✅ cat D — as `LogicalNot` |
| `UnaryMaxInt32` | `unary_max_int32` | `1000` | `unary` | ✅ cat D — comparison tie driven by `test_eltwise_unary_sfpu_int` |
| `UnaryMaxUint32` | `unary_max_uint32` | `1000` | `unary` | ✅ cat D — as `UnaryMaxInt32` |
| `UnaryMinInt32` | `unary_min_int32` | `1000` | `unary` | ✅ cat D — as `UnaryMaxInt32` |
| `UnaryMinUint32` | `unary_min_uint32` | `1000` | `unary` | ✅ cat D — as `UnaryMaxInt32` |
| `UnaryNe` | `unary_ne` | `0.5` | `unary` | ✅ cat D — as `LogicalNot` |

#### 4.4 Binary (float + shift) SFPU ops (43 ops)

| Op | Kernel | Registered domain | Cat A pole | Edge sweep | Driven by |
|---|---|---|---|---|---|
| `SfpuAddTopRow` | `ADD_TOP_ROW` | yes | — | ⬜ | `binary` |
| `SfpuAtan2` | `ATAN2` | no (format default) | — | ⬜ | `binary` |
| `SfpuBinaryFmod` | `FMOD` | no (format default) | B=0.0 (bot) | ✅ | `binary` |
| `SfpuBinaryMax` | `MAX` | no (format default) | — | ⬜ | `binary` |
| `SfpuBinaryMin` | `MIN` | no (format default) | — | ⬜ | `binary` |
| `SfpuBinaryRemainder` | `REMAINDER` | no (format default) | B=0.0 (bot) | ✅ | `binary` |
| `SfpuBitwiseAnd` | `BITWISE_AND` | no (format default) | — | ⬜ | `binary` |
| `SfpuBitwiseOr` | `BITWISE_OR` | no (format default) | — | ⬜ | `binary` |
| `SfpuBitwiseXor` | `BITWISE_XOR` | no (format default) | — | ⬜ | `binary` |
| `SfpuDivInt32` | `DIV_INT32` | no (format default) | — | ⬜ | `binary` |
| `SfpuDivInt32Floor` | `DIV_INT32_FLOOR` | no (format default) | — | ⬜ | `binary` |
| `SfpuElwEq` | `EQ` | no (format default) | — | ⬜ | `binary` |
| `SfpuElwGe` | `GE` | no (format default) | — | ⬜ | `binary` |
| `SfpuElwGt` | `GT` | no (format default) | — | ⬜ | `binary` |
| `SfpuElwLe` | `LE` | no (format default) | — | ⬜ | `binary` |
| `SfpuElwLeftShift` | `LSHFT` | yes | — | ⬜ | `binary` |
| `SfpuElwLogicalRightShift` | `LOGICAL_RSHFT` | yes | — | ⬜ | `binary` |
| `SfpuElwLt` | `LT` | no (format default) | — | ⬜ | `binary` |
| `SfpuElwNe` | `NE` | no (format default) | — | ⬜ | `binary` |
| `SfpuElwRightShift` | `RSHFT` | yes | — | ⬜ | `binary` |
| `SfpuElwadd` | `ADD` | yes | — | ⬜ | `binary` |
| `SfpuElwdiv` | `DIV` | yes | B=0.0 (bot) | ✅ | `binary` |
| `SfpuElwmul` | `MUL` | yes | — | ⬜ | `binary` |
| `SfpuElwpow` | `POW` | yes | A=0.0 (abo) | ✅ | `binary` |
| `SfpuElwrsub` | `RSUB` | yes | — | ⬜ | `binary` |
| `SfpuElwsub` | `SUB` | yes | — | ⬜ | `binary` |
| `SfpuEqInt` | `EQ_INT` | no (format default) | — | ⬜ | `binary` |
| `SfpuFmodInt32` | `FMOD_INT32` | no (format default) | — | ⬜ | `binary` |
| `SfpuGcd` | `GCD` | no (format default) | — | ⬜ | `binary` |
| `SfpuIsclose` | `ISCLOSE` | no (format default) | — | ⬜ | `binary` |
| `SfpuLcm` | `LCM` | no (format default) | — | ⬜ | `binary` |
| `SfpuLogsigmoid` | `LOGSIGMOID` | no (format default) | — | ⬜ | `binary` |
| `SfpuMask` | `MASK` | no (format default) | — | ⬜ | `binary` |
| `SfpuMaxInt32` | `MAX_INT32` | no (format default) | — | ⬜ | `binary` |
| `SfpuMaxUint32` | `MAX_UINT32` | no (format default) | — | ⬜ | `binary` |
| `SfpuMinInt32` | `MIN_INT32` | no (format default) | — | ⬜ | `binary` |
| `SfpuMinUint32` | `MIN_UINT32` | no (format default) | — | ⬜ | `binary` |
| `SfpuMulInt32` | `MUL_INT32` | no (format default) | — | ⬜ | `binary` |
| `SfpuNeInt` | `NE_INT` | no (format default) | — | ⬜ | `binary` |
| `SfpuRemainderInt32` | `REMAINDER_INT32` | no (format default) | — | ⬜ | `binary` |
| `SfpuRemainderUint32` | `REMAINDER_UINT32` | no (format default) | — | ⬜ | `binary` |
| `SfpuRsubInt32` | `RSUB_INT32` | no (format default) | — | ⬜ | `binary` |
| `SfpuXlogy` | `XLOGY` | yes | B=0.0 (abo) | ✅ | `binary` |

#### 4.5 Binary integer SFPU ops (5 ops)

| Op | Kernel | Registered domain | Cat A pole | Edge sweep | Driven by |
|---|---|---|---|---|---|
| `SfpuElwmulInt` | `MUL` | no (format default) | — | ⬜ | **none (WH/BH)** |
| `SfpuGeInt` | `GE_INT` | no (format default) | — | ⬜ | **none (WH/BH)** |
| `SfpuGtInt` | `GT_INT` | no (format default) | — | ⬜ | **none (WH/BH)** |
| `SfpuLeInt` | `LE_INT` | no (format default) | — | ⬜ | **none (WH/BH)** |
| `SfpuLtInt` | `LT_INT` | no (format default) | — | ⬜ | **none (WH/BH)** |

#### 4.6 Ternary ops (5 ops)

| Op | Kernel | Registered domain | Cat A pole | Edge sweep | Driven by |
|---|---|---|---|---|---|
| `SfpuAddcdiv` | `addcdiv` | no (format default) | — | ⬜ | `ternary` |
| `SfpuAddcmul` | `addcmul` | no (format default) | — | ⬜ | `ternary` |
| `SfpuLerp` | `lerp` | no (format default) | — | ⬜ | `ternary` |
| `SfpuSnakeBeta` | `snake_beta` | no (format default) | — | ⬜ | `ternary` |
| `SfpuWhere` | `where` | no (format default) | — | ⬜ | `ternary` |

#### 4.7 Scalar-binop ops (5 ops)

| Op | Kernel | Registered domain | Cat A pole | Edge sweep | Driven by |
|---|---|---|---|---|---|
| `ScalarAdd` | `ADD` | no (format default) | — | ⬜ | `binop_scalar` |
| `ScalarDiv` | `DIV` | no (format default) | — | ⬜ | `binop_scalar` |
| `ScalarMul` | `MUL` | no (format default) | — | ⬜ | `binop_scalar` |
| `ScalarRsub` | `RSUB` | no (format default) | — | ⬜ | `binop_scalar` |
| `ScalarSub` | `SUB` | no (format default) | — | ⬜ | `binop_scalar` |

#### 4.8 Reduce ops (3 ops)

| Op | Kernel | Registered domain | Cat A pole | Edge sweep | Driven by |
|---|---|---|---|---|---|
| `ReduceColumn` | `REDUCE_COL` | yes | — | ⬜ | `reduce`, `reduce`, `reduce_sdpa` |
| `ReduceRow` | `REDUCE_ROW` | yes | — | ⬜ | `reduce`, `reduce` |
| `ReduceScalar` | `REDUCE_SCALAR` | yes | — | ⬜ | `reduce` |

#### 4.9 FPU binary (eltwise) ops (3 ops)

| Op | Kernel | Registered domain | Cat A pole | Edge sweep | Driven by |
|---|---|---|---|---|---|
| `Elwadd` | `ELWADD` | yes | — | ⬜ | `deepseek_moe_gate`, `eltwise_binary`, `generalized_moe_gate` +4 |
| `Elwmul` | `ELWMUL` | yes | — | ⬜ | `deepseek_moe_gate`, `eltwise_bcast_col_custom`, `eltwise_binary` +3 |
| `Elwsub` | `ELWSUB` | yes | — | ⬜ | `deepseek_moe_gate`, `eltwise_bcast_col_custom`, `eltwise_binary` +5 |

---

## 5. What driving the edges found

Ten ops over 42 `(op, format, dest_acc)` cells disagree with their golden at the newly driven points
— 5 unary ops over 20 cells and 5 binary over 22. All are recorded as **non-strict xfails**, so the
case still executes and reports XPASS if the behaviour changes. Every one is cross-checked against
[tt-isa-documentation](https://github.com/tenstorrent/tt-isa-documentation), which splits them
cleanly. **This split is the practically important part:** half of these are specified hardware
behaviour and chasing them would be wasted effort.

### 5.1 Documented — the ISA is the authority, not a bug list

**The sign of a zero *result* is lost on Wormhole, by specification.** `div(0, -x)`,
`fmod`/`remainder` with a negative divisor, and `xlogy(0, tiny)` all return `+0.0` where IEEE gives
`-0.0`. All are built on `SFPMAD`:

> Wormhole — "If the output (before rounding) is denormal or negative zero, it'll be flushed to
> **positive** zero." · Blackhole — "…flushed to **sign-preserved** zero."

Blackhole's `SFPMAD` page lists *"improved edge-case handling of NaNs and of negative zero"* among
its upgrades. So this is a documented Wormhole limitation that Blackhole is documented to fix, and
these xfails are a **testable prediction** there: they should XPASS. If they do not, the
documentation and the hardware disagree.

**`sign(-0.0)` and `heaviside(-0.0)` sit outside the contract of the primitive they use.** `SFPSETCC`
is specified only *"provided that `VC` is neither negative zero nor any kind of NaN"* — identically on
both arches, so unlike the `SFPMAD` group this is **not** generational.

### 5.2 The signed-zero group is a *delivery* question, not three separate findings

The three signed-zero ops partition **exactly** on `unpack_to_dest`, which the driver sets to
`(input.is_32_bit() and dest_acc == Yes)` — the only path where the datum skips SrcA and the
datacopy:

| Op | Diverges on | `unpack_to_dest` there |
|---|---|---|
| `Signbit` | 6 of 8 combinations | **False** on all 6 |
| `Sign` | 2 of 8 | **True** on both |
| `Heaviside` | the same 2 | **True** on both |

`Signbit`'s set is the exact complement of `Sign`'s, and `Sign`'s and `Heaviside`'s sets are
identical. One cause explains all three: **`-0.0` only reaches the LREG on the unpack-to-dest path.**
Neither `calculate_sign` nor `calculate_heaviside` guards `|v| != 0` on its `v_if(v < 0.0F)`, so a
real `-0.0` would make them diverge on all 8; passing on 6 says the LREG holds `+0.0` there. `Signbit`
reads the sign bit directly, so it returns 0 on those 6 — and correctly returns 1 on the 2 where the
datum arrives intact. A genuinely broken sign-bit read would fail on all 8.

Consequences, both recorded in the suite's reason strings:

- **`Signbit`'s 6 entries can never XPASS.** They record a *stimulus* limitation, not a kernel defect;
  no kernel change can make an input arrive. The earlier reading of this as "a kernel-contract bug"
  was wrong.
- **`Sign` and `Heaviside` passing on those same 6 is vacuous** — the golden's answers for `-0.0`
  (0 and 0.5) coincide with the hardware's for `+0.0`, so the case agrees without testing what it
  names.

The partition is asserted at collection (`_assert_signed_zero_partition_valid`), because the
explanation rests on it and a reason string is prose no run checks. **Not directly measured:** drive
datacopy with `custom(values=[0.0, -0.0])` and read the DEST sign bit on a `(Float16_b, *, No)`
variant. The probe is deliberately left in place until then — dropping it on an unmeasured hypothesis
would silently lose real coverage if the hypothesis is wrong.

### 5.3 Still open — not explained by the ISA

| Finding | Ops |
|---|---|
| **`0/0` and `x%0` return `inf`, not `nan`.** `SFPMAD` says NaN/±Inf inputs follow "the usual IEEE754 rules", which makes `0 × inf` a NaN — so this is the kernels' own reciprocal composition, not the multiply. Specifically the indeterminate form: the finite poles agree exactly and every ±inf lines up | `div`, `fmod`, `remainder`, `xlogy` |
| **`0**0` returns 0** where C, torch and the golden give 1. `pow` evaluates `exp(b·ln a)`, so a composition artifact | `pow` |
| **`RsqrtCompat(0)` saturates to `1.7014118e38`** (`0x7F000000`) instead of `inf`, on all 8 combinations — while plain `Rsqrt` over the same probe does **not** diverge. Two implementations of one function disagreeing at their shared pole, with nothing in the ISA prescribing either answer | `RsqrtCompat` |
| **`Erfinv(±1)` saturates** rather than returning ±inf, on the fp32-dest combinations only — tolerance-shaped rather than semantic | `erfinv` |

`RsqrtCompat(0)` is the one worth filing with kernel owners now; the `signbit` question should wait on
the delivery measurement in §5.2.

### 5.4 Two smaller results

- **The bitwise kernels need the two's-complement pack path** for negative operands.
  `(INT32_MIN+1) & -1` returned `-1`. Nothing had established this, because
  `test_sfpu_binary_bitwise` draws from the positive-only default and had never fed them a negative.
- **Both Blackhole guards are non-strict xfails rather than skips**, so a kernel fix reports XPASS
  instead of leaving the case green by omission indefinitely.

---

## 6. Cat B — where specials can be injected (measured)

Not enabled (§2.1), but the matrix is data rather than guesswork. Measured by driving the five
isinf/isnan predicates over the full 5×5 format matrix × both `dest_acc` with no skips — 250
variants, 85 failing. Two independent breakers:

- **A `Float16` (e5m10) anywhere.** As an *input* it never preserves specials — all 5 predicates fail
  on all 5 outputs at both `dest_acc`, 10/10 cells. As an *output* it fails too, unless a 32-bit input
  is paired with `dest_acc=Yes`; `Float32 -> Float16` at `dest_acc=No` fails all five, which is the
  exact pair Blackhole already guards.
- **A 16-bit input with `dest_acc=Yes`.** `Float16_b` there fails `isinf`/`isneginf`/`isnan` while
  `isposinf`/`isfinite` pass — **`+inf` survives, `-inf` and `NaN` do not.**

A third constraint is applied statically rather than measured: block-float and MX *inputs* cannot
carry specials at all (`quantize_input_to_unpack_format` destroys NaN for Bfp8_b and Bfp4_b), so a
predicate passing there is vacuous — golden and hardware agree there is no NaN because neither ever
saw one. Those rows are excluded rather than trusted. Block-float *outputs* are excluded on the
golden's behalf: an `inf` inside a block whose shared exponent is finite is not a value the format can
express, so neither the lattice nor the tolerance criterion means anything for it.

**Safe surface — 7 of 40 triples:**

| `dest_acc` | Safe `input -> output` |
|---|---|
| `No` | `Float32->Float32`, `Float32->Float16_b`, `Float16_b->Float32`, `Float16_b->Float16_b` |
| `Yes` | `Float32->Float32`, `Float32->Float16`, `Float32->Float16_b` |

Measured on Wormhole; **unverified on Blackhole**, where the unpack paths differ. The whole matrix is
written out longhand in `test_sfpu_domains.py` (7 accepted cells of 50) so it cannot be rewritten
without a test changing outcome — including a guard for the `DestAccumulation` truthiness trap, where
both enum members are truthy and `bool(member)` would silently flip whole rows.

---

## 7. How to regenerate this document

Nothing here needs hardware. §1's figures and §4's tables come from the same inventory:

```bash
cd tt_metal/tt-llk/tests/python_tests
python3 -c "
import sys; sys.path.insert(0,'.')
from helpers.sfpu_domains import (_OP_SINGULARITIES, _OP_EDGE_POINTS, _OP_DOMAIN_REGISTRY,
                                  sfpu_unary_ops, edge_spec, SPECIALS_READY_OPS)
from helpers.llk_params import MathOperation, DataFormat as F
u = sorted(sfpu_unary_ops(), key=lambda o: o.name)
e = [o for o in u if edge_spec(o, F.Float32, F.Float32) is not None]
print('singularities', len(_OP_SINGULARITIES), '| edge points', len(_OP_EDGE_POINTS))
print('unary', len(u), '| with an edge', len(e), '| smooth', len(u) - len(e))
print('specials-ready', len(SPECIALS_READY_OPS))
"
# expect: 19 / 43 / 97 / 50 / 47 / 0
python3 -m pytest test_sfpu_domains.py -q --noconftest   # expect 107 passed
```

Per-op rows are keyed on `MathOperation` and read `_OP_DOMAIN_REGISTRY`, `_OP_SINGULARITIES`,
`_OP_EDGE_POINTS` and the sweep op lists in `test_sfpu_unary.py`. An op enrols in the edge sweep **by
being in the registry**, not by being listed in a test, so a new op appears in §4 automatically —
regenerate rather than editing rows by hand.

The `pytest --noconftest` above is needed because `conftest.py` imports `helpers/device.py`, which
imports `tt-exalens`; `tests/requirements.txt` pins `0.3.29` and later releases moved
`CallstackEntry` and `ElfFile`, so a drifted venv fails at collection with what looks like a broken
checkout. Host-side tests do not need the device at all.
