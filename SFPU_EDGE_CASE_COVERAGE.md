# SFPU LLK Edge-Case Test-Coverage Audit

**Issue:** [tenstorrent/tt-metal#49739 — [LLK] SFPU testing edge cases](https://github.com/tenstorrent/tt-metal/issues/49739)
**Plan for what is left:** [SFPU_EDGE_CASE_EXPANSION_PLAN.md](SFPU_EDGE_CASE_EXPANSION_PLAN.md)
**Audited:** 2026-07-23 · **Regenerated from code:** 2026-08-19, against
`ldjurovic/sfpu_edge_cases_phase_3` @ `caf8701f973`
**Wormhole measurement:** [WORMHOLE_MEASUREMENT_RESULTS.md](WORMHOLE_MEASUREMENT_RESULTS.md)
**Scope:** every SFPU op reachable from the tt-llk Python test infra
(`tt_metal/tt-llk/tests/python_tests/`), which is where the suites, the registries and all the gates
live. Wormhole B0 and Blackhole share essentially the same SFPU op set (BH adds only `topk_xl`), so this
audit treats them together and notes arch-specific gaps inline. Quasar has its own suite under
`quasar/` and is **out of scope** — an op driven only from `quasar/` counts as untested here.

> **The op kernels are not in `tt-llk`.** `helpers/test_config.py` compiles each test with
> `-I../../hw/ckernels/<arch>/metal/llk_api/llk_sfpu`, and `helpers/include/sfpu_operations.h` — the
> header mapping a `MathOperation` onto a kernel call — resolves 86 of its includes there. So the
> arithmetic under test lives in **`tt_metal/hw/ckernels/<arch>/metal/llk_api/llk_sfpu`** (101 kernel
> headers per arch), while `tt_metal/tt-llk` supplies the LLK plumbing those kernels are driven through
> (`llk_math_eltwise_{unary,binary,ternary}_sfpu.h`, `llk_math_welfords_sfpu.h`) plus 31 SFPU kernels of
> its own in `common/inc/sfpu` per arch, plus 1 Wormhole and 9 Blackhole `experimental/` ones. Both trees
> are in scope. §2.5's cat-F list is mostly `llk_sfpu` kernels.

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

In the **Cat B** column specifically, **🟡 §5.9** and **🟡 §5.6** are not "unknown" — they name the
single kernel behaviour holding that op out, which is why the 30 unenrolled ops are a shorter list than
they look:

- **🟡 §5.9** — waiting on §5.6's approximation-contract question (Q1). 23 ops, blocked on an owner.
- **🟡 §5.6** — `Sign` and `Heaviside`, which compare through `SFPSETCC`, whose contract excludes a
  `NaN` operand. That is §5.6's third question.

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
| Binary SFPU ops (float + shift) | 43 — 11 with a registered domain, 6 with a driven pole; **18 of the 22 float rows now reach the edge sweep** (was 5) |
| Binary integer / ternary / scalar / reduce / FPU-binary ops | 5 / 5 / 5 / 3 / 3 — the 5 binary-integer members are Quasar spellings of kernels covered under `SfpuElw*` at `Int32` (§4.5, §2.4) |
| `_OP_SINGULARITIES` entries | **22** — +2 for the ternary operand-C poles, +1 for `SfpuAtan2`'s `B = 0` branch point |
| `_OP_EDGE_POINTS` entries | 43, plus `_OP_OPERAND_EDGE_POINTS` for `lerp`'s operand-C knees |
| `SPECIALS_READY_OPS` (cat B opt-in, unary + scalar) | **67 of 97 unary**, plus all **5 scalar binops**; all 30 unary still outside wait on §5.6's two questions or on a harness — none is work this suite can simply do |
| `BINARY_SPECIALS_READY_OPS` (cat B opt-in, binary) | **12 of the 21 float ops** reaching the shared driver. The other 9 are recorded in `_BINARY_SPECIALS_NOT_READY` under **five** causes, not nine investigations — 6 on §5.6's Q1, `SfpuMask` on Q3, `SfpuIsclose` on one read-back, `SfpuLogsigmoid` out by construction (§4.4) |
| Cat B for the reduce family | **6 classes × 4 pools × 2 ops × 2 formats = 96 variants** (`test_float_reduce_specials`), where a special reaches the output *through the fold* (§4.8) |
| `(format, dest_acc)` triples that can carry specials | 7 cells of 50, **re-confirmed on Wormhole** (250 variants, 85 failing); **3 of those 7 reachable and confirmed on Blackhole**. Carrying a `-0.0` is a strictly narrower gate — `negative_zero_delivered()` |
| Ops diverging from their golden at a driven edge | **8 ops over 46 cells** — 7 unary over 24 (`Sign` 2, `Heaviside` 2, `RsqrtCompat` 8, `Erfinv` 2, `Reciprocal` 6, `Sqrt` 2, `Rsqrt` 2) and 1 binary over 22 (`SfpuElwpow`'s `0**0` 6, plus the 16 arch-gated signed-zero cells) |
| Host-side guards over the gates and metadata | **126** tests (`test_sfpu_domains.py`) |

**Category status:** A ✅ closed for every op that has a boundary — unary, **binary** (`SfpuAtan2`'s
branch point was the last unregistered one) **and ternary** · B 🟡 live for 67 of the 97 unary ops, all
5 scalar binops, **12 of the 21 float binary ops and the reduce family**; the 30 unary and 9 binary
still outside wait on §5.6's questions, on one read-back, or on a harness · C ✅ closed for the 5 ops
whose kernels claim the full int32 range, **plus the 4 ordered Int32 comparisons**
· D ✅ closed for all 43 knees, plus `lerp`'s weight boundaries **and the Int32 comparison tie** · E ✅
closed, unary and binary · F ⬜ **14** kernels untouched (§2.5).

**So five of the six categories are closed or bounded**, and what is left is one large build (F) and
two questions someone else has to answer.

**One family is deliberately still ⬜ and should not be read as an oversight:** §4.9's FPU eltwise
(`Elwadd`/`Elwmul`/`Elwsub`). It runs on the FPU rather than the SFPU, so `specials_safe()` — measured
on the unpack→Dest path — does not apply to it and has to be re-measured for SrcA/SrcB before any
golden work is worth doing. Plan §5 has the design and the measured `Elwmul` fidelity defect.

**Suite status — both arches green at `caf8701f973` (2026-08-19).** Measured through the two-phase
compile-producer / compile-consumer flow that CI uses. **Zero failures, zero timeouts**, Wormhole
measured suite by suite including the `bcast` tests.

| Suite | Wormhole n300 |
|---|---|
| `test_sfpu_unary.py` | 6044 passed · 573 skipped · 23 xfailed · **6 xpassed** (§5.11) · 0 failed |
| `test_sfpu_binary.py` | 1009 passed · 844 skipped · 9 xfailed · **16 xpassed** (§5.12) · 0 failed |
| `test_sfpu_reduce.py` | 1074 passed · 548 skipped · 0 failed |
| `test_sfpu_binop_scalar.py` | 67 passed · 73 skipped · 0 failed |
| `test_sfpu_ternary.py` | 39 passed · 25 skipped · 0 failed |
| `test_sfpu_domains.py` (host) | 126 passed |

**Blackhole is green at the same head.** Per-suite counts were not captured, so they are deliberately
not tabulated rather than guessed. Two things are Wormhole-gated by construction and therefore assert
*more* on Blackhole, which is where they are first checked: `generated_nan_sign_is_asserted()`
(Blackhole specifies the canonical NaN) and the retracted `both_zero`/`nan_golden` cells, asserted there
in full including the sign.

**The 6 and 16 xpassed are the only non-green signal**, and they are not noise: they are the two
pre-existing Wormhole-only arch gates (§5.11, §5.12), each XPASSing its whole content on the arch it was
written for, so each currently asserts nothing on either arch. Plan §9.

**The unary xfail count is what the code predicts**, which is worth checking rather than assuming:
23 = 14 static (`Sign` 2, `Heaviside` 2, `RsqrtCompat` 8, `Erfinv` 2) + `Reciprocal` 6 + `Sqrt` 2 +
`Rsqrt` 1 — `Rsqrt`'s `Float32->Float16_b` at `dest_acc=Yes` is the single entry the NaN-sign gate
suppresses, so that a gated-off probe cannot leave a non-strict xfail that can never fire.

**A card wedge exists on this suite and a reset clears it.** A `test_sfpu_binary.py` run has twice
produced a cascade of `TENSIX TIMED OUT` failures with a frozen `Brisc Counter` — one hang, with every
variant after it failing. `tt-smi -r` clears it and the re-run is clean. Seen on two boards, transient,
not filed. Two other ways to get a fake red look identical: a consumer run against a missing ELF fails
*every* variant with **zero** timeouts (`TestConfig` rmtree's `/tmp/tt-llk-build` at session setup — see
plan §8), and a stale `ttexalens` fails the whole suite at collection.

Wormhole skips fewer unary variants than Blackhole, because `_skip_bh_unless_fp32` collapses the whole
`dest_acc=No` row there — so Wormhole exercises more of the format matrix, which is part of why the
first Wormhole run found anything at all.

---

## 2. What is still NOT tested

Ordered by how much coverage each item is worth. This is the list to work from; §1 of the plan
sequences it.

### 2.1 Cat B — IEEE specials for the other 30 unary ops

**No longer zero — 67 of the 97 unary ops are enrolled**, plus all 5 scalar binops, so
`SPECIALS_READY_OPS` has 72 entries. They inject `±inf`, `NaN` and signed zeros across every
specials-safe triple the sweep reaches. The mechanism is proven end to end on silicon: the first nine
(`Identity`, `Abs`, `Exp`, `Sin`, `Cos`, `Neg`, `Reciprocal`, `Sqrt`, `Rsqrt`) were verified on
Blackhole, and the tranche of 48 that followed needed no golden change
at all. What remains is the 30 below, and none of it is per-op work this suite can simply do.

**The second tranche is in, and it was one framework defect wearing four disguises.** Every divergence
that had been booked against those goldens — `Neg(NaN) → +inf` chief among them — traced to torch's
fp32 → bfloat16 cast, which canonicalises every NaN to `0xFFFF`, sign bit set, whatever sign it
started with. That is why the defect appeared only at `dest_acc=No`: it takes a 16-bit Dest for the
pack path to substitute a *signed* infinity and make the invented sign visible. `Neg` is simply the
one op whose NaN is genuinely negative, so it was the op the artefact disagreed with.

Three kernel divergences survived the fix and are now recorded as non-strict xfails, derived from the
delivery rules rather than listed (`_cat_b_divergences` in `test_sfpu_unary.py`):

| Op | Probe | Golden | Hardware | Scope |
|---|---|---|---|---|
| `Reciprocal` | `NaN` | `NaN` | `+0` | every combination that delivers a NaN — 6 |
| `Sqrt` | `-0` | `-0` | `NaN` | unpack-to-dest only — 2 |
| `Rsqrt` | `-0` | `-inf` | `NaN` | unpack-to-dest only — 2 |

`Log` is the only op left out of the tranche, and not for a golden reason: it saturates its input
(§5.5), which no ISA text prescribes, so §5.6's question has to be answered before it can be enrolled.

**30 unary ops remain outside**, and 25 of them are held there by the two kernel behaviours in §5.6 and
§5.9 rather than by anything op-specific — so the remaining cat-B work is two answers, not 25
investigations. The split is exact and re-derivable (`sfpu_unary_ops() - SPECIALS_READY_OPS`): **23 on
§5.9**, **2 on §5.6 Q3** (`Sign`, `Heaviside`), and **5 others** — the three `TopK*` stages with no golden
entry (§2.5), `ReluMin` (skipped on tt-llk#1120) and `RsqrtCompat` (already fully xfailed).

`I1` is worth reading closely, because it is the case that keeps the two gates honest: its golden **was**
wrong and has been fixed, but it stays out of `SPECIALS_READY_OPS` because its *kernel* saturates to
`±1.1547668e37`. Fixing a golden is not a reason to enrol an op — if it were, a kernel divergence would
be laundered into a golden that agrees with it, which is the failure mode this whole gate exists to
prevent.

47 of the 97 unary ops are smooth everywhere (§4.2), and for those cat B is their entire edge story.

The five predicates (`Isinf`, `Isposinf`, `Isneginf`, `Isnan`, `Isfinite`) still inject specials via
`test_eltwise_unary_sfpu_isinf_isnan`, which is also the instrument that measured the safe matrix in
§6.

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
>
> **There is a third and harder kind of alias.** The two above are alias *pairs inside one enum*,
> which §7's inventory resolves. The five
> `SFPU_BINARY_INT` members alias **across a `MathOpType` boundary**: `SfpuGtInt` and `SfpuElwGt` are
> different members with different dispatch types that reach the *same kernel* on WH/BH, one via
> `MathOpType.SFPU_BINARY_INT` (Quasar-only) and the other via `SFPU_BINARY` at `DataFormat.Int32`. No
> name comparison finds that — the resolution has to go through the arch's `BinaryOp` enum and the
> dispatch header. §4.5 has the corrected table and the guard that now pins it.

**Perf-only — a perf test exists, no functional golden or assert (7 ops):**
`AddInt32`, `SubInt32`, `AbsInt32`, `BitwiseNot` (all in `perf_eltwise_unary_sfpu_int32.py`; the file
records the reason — the int32-unary functional sweep is blocked by the fast-tilize gap,
tt-llk#495), plus `TopKLocalSort`, `TopKMerge`, `TopKRebuild` (whole-op `topk` is tested, the three
stages are not).

### 2.5 Cat F — kernels with no `MathOperation` entry (14)

A kernel header exists and implements an op; nothing in the Python infra can reach it, because there is
no `MathOperation` member to dispatch. **Re-derived from the kernel trees rather than carried forward,
and the list both grew and shrank.**

In `tt_metal/hw/ckernels/<arch>/metal/llk_api/llk_sfpu` — 11:
`dropout`, `quant`, `cumsum`, `reshuffle_rows`, `int_sum`, `tiled_prod`, `copy_dest_values`,
`max_pool_indices`, `rand`, **`alt_complex_rotate90`**, **`bitwise`** (the unary
bitwise-and/or/xor-with-a-scalar kernel, distinct from the `binary_bitwise` and `bitwise_not` that
`SfpuBitwiseAnd/Or/Xor` and `BitwiseNot` reach).

In `tt_metal/tt-llk/tt_llk_<arch>/common/inc/sfpu` — 3:
`welfords` (via `llk_lib/llk_math_welfords_sfpu.h`), and two Blackhole `experimental/` kernels,
**`sparse_k_filter`** and **`zero_pad`**.

Two entries that look like gaps and are not:

- **`generalized_moe_gate_topk` is covered**: `sources/generalized_moe_gate_test.cpp` and `test_generalized_moe_gate.py` both exist and
  drive `experimental/ckernel_sfpu_generalized_moe_gate_topk_single_face.h`. The two names are one
  character apart and the earlier audit retired the wrong one.
- **Four kernels were never on the list.** `alt_complex_rotate90`, `bitwise`, `sparse_k_filter` and
  `zero_pad` have no `MathOperation` entry and no test source that reaches them.

`quant` is on the list for a different reason from the rest: it *is* driven, but only from
`sfpu_operations_quasar.h` / `test_eltwise_binary_sfpu_quasar.py`, and Quasar is out of scope — see
**Scope** at the top.

Not cat F, and worth recording so they are not re-filed: `conversions`, `piecewise_rational`,
`binary_pow`, `cdf`, `polyval`, `converter`, `load_config`, `is_fp16_zero` are **helpers** pulled in
transitively by kernels that are dispatched, not ops in their own right. `ema` and `reduce` have
dedicated tests (`test_sfpu_ema.py`, `test_sfpu_reduce.py`). `tt-llk`'s own `ckernel_sfpu_mul_int.h` is
dead rather than untested — `sfpu_operations.h` includes Metal's `llk_sfpu/ckernel_sfpu_mul_int32.h`
instead.

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
- **~~Wormhole: nothing in this suite had been run there~~ ✅ closed, and it was not free.** The suite has
  now run on a Wormhole n300: the safe matrix re-measured and confirmed, the total order confirmed, and
  **49 of 752 edge variants failed** — one 10-op family, §5.10. The lesson is the general one: an
  unexercised arch is not a documentation gap, it is an unmeasured claim. What is *still* Wormhole-unmeasured
  after this: the ternary and scalar suites' arch-specific claims beyond their headline counts, and
  `Tan(NaN) -> 0.0` on the 16-bit-Dest path (§5.10).
- **~~The accurate exp path over (16, 80].~~ ✅ closed on Wormhole, still open on Blackhole.** The registry
  carries the range bound and `_APPROX_ACCURACY_MAX` the approximation bound, applied only in
  `ApproximationMode.Yes`. The accurate path over that region has now been driven on a Wormhole n300 —
  `Exp` 132 passed, `Exp2` 138 passed, 0 failed, with the measured error **+0.00%** above 8 out to
  `x = 79.97` — so the restored `high=80` is sound there. See §5.11.
- **Whether `-0.0` reaches DEST** on the non-unpack-to-dest path. Measured three ways and answered
  **no** — `negative_zero_delivered()` encodes it, and the `-0` probe is no longer sent where it cannot
  arrive. What is still unmeasured is a *direct* read of the DEST sign bit rather than an inference from
  three ops' divergence partition.
- **`WITH_COVERAGE` builds** and **Bfp4_b output formats** (`Float16 -> Bfp4_b` fails 100% at
  `dest_acc=No` on Wormhole and is unexplained; every neighbouring cell is clean).
- **~~CI runs none of this~~ — ⚠️ retracted; the finding was wrong and the code now says so.**
  The claim was that every LLK pytest job either excluded the `nightly` marker or ran
  `--coverage` (under which `_skip_coverage_unsupported` drops the broad profile wholesale), so the
  broad unary sweep ran in no automated job on any arch. **Both halves fail on inspection:**
  `.github/workflows/llk-e2e.yaml` passes `pytest-markers: 'not perf and not quasar and not accuracy'`
  — `nightly` is *not* excluded — and no group in `tests/pipeline_reorg/llk_e2e_tests.yaml` passes
  `--coverage` on either pytest leg (all four carry `coverage: false`, and `inputs.coverage` gates
  nothing but the artifact upload). `WITH_COVERAGE` is set solely by that CLI option, so
  `_skip_coverage_unsupported` was already a no-op in this workflow and **the broad profile already ran
  nightly on both arches.** The gains recorded in this document were guarded all along.
  The five `llk_e2e_*_nocov` groups per arch (`split_group` 6–10) added on that premise were therefore a
  **duplicate rather than new coverage**, and are **withdrawn** —
  `tests/pipeline_reorg/llk_e2e_tests.yaml` is byte-identical to `main`. What remains is wall-clock
  inside the existing groups' `timeout: 38`, which is unmeasured; a number to watch, not work to do.
- **Whether the `BROAD_SWEEP_OPS` coverage skip is justified at all.** Still open, and now the *only*
  live question in this area. It cited tt-llk#1435, which is about test *ordering* — an accuracy issue
  possibly caused by BIT11 leaking between tests — and its one mention of coverage is an observation of
  the skip's own effect. The circular citation is gone and nothing replaced it. The suggestion on review
  is to re-run without the skip and either delete it or debug what scrambles the values, rather than
  route around it.
### 2.9 No ternary op has a registered *domain*, and scalar reads none

The ternary family reaches the registry for the operand that matters — `OperandSpecs.spec_C`, `Operand.C`,
and `addcdiv`/`snake_beta`'s registered poles (§4.6) — but **no ternary op has a registered domain**, so
all three operands otherwise draw the format default. The scalar family has the plumbing and nothing to
read: `ScalarBinopGolden` is wired for a domain that no entry supplies.

Neither is a hole in the *edge* coverage — cat A and cat D are driven where they exist, and cat B is
enrolled for all five scalar ops — but both mean the **random** sweep for those families samples a
generic range rather than the op's own. Registering domains is the fix, and it is small per op.


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

| Op | Kernel | Random sweep | Cat A boundary (side defined on) | Cat D knees / ties | Cat B | Edge sweep | Other test | ⚠️ |
|---|---|---|---|---|---|---|---|---|
| `Acos` | `acos` | standard | -1.0 (abo); 1.0 (bel) | — | ✅ driven | ✅ | — |  |
| `Acosh` | `acosh` | broad | 1.0 (abo) | — | ✅ driven | ✅ | — |  |
| `Asin` | `asin` | standard | -1.0 (abo); 1.0 (bel) | — | ✅ driven | ✅ | — |  |
| `Atanh` | `atanh` | broad | -1.0 (abo); 1.0 (bel) | — | ✅ driven | ✅ | — |  |
| `Ceil` | `ceil` | broad | — | `-2, -1, 0, 1, 2` | ✅ driven | ✅ | — |  |
| `Celu` | `celu` | broad | — | `0, -0` | ✅ driven | ✅ | — |  |
| `Clamp` | `clamp` | standard | — | `-1, 1` | ✅ driven | ✅ | — |  |
| `Elu` | `elu` | broad | — | `0, -0` | ✅ driven | ✅ | — |  |
| `EqualZero` | `equal_zero` | standard | — | `0, -0` | ✅ driven | ✅ | — |  |
| `Erfinv` | `erfinv` | standard | -1.0 (abo); 1.0 (bel) | — | 🟡 §5.9 | ✅ | — | ⚠️ |
| `Floor` | `floor` | broad | — | `-2, -1, 0, 1, 2` | ✅ driven | ✅ | — |  |
| `Frac` | `frac` | broad | — | `-1.5, -1, 1, 1.5` | 🟡 §5.9 | ✅ | — |  |
| `GreaterThanEqualZero` | `greater_than_equal_zero` | standard | — | `0, -0` | ✅ driven | ✅ | — |  |
| `GreaterThanZero` | `greater_than_zero` | standard | — | `0, -0` | ✅ driven | ✅ | — |  |
| `Hardmish` | `hardmish` | standard | — | `-2, 0` | ✅ driven | ✅ | — |  |
| `Hardshrink` | `hardshrink` | standard | — | `-0.5, 0.5` | ✅ driven | ✅ | — |  |
| `Hardsigmoid` | `hardsigmoid` | broad | — | `-3, 3` | ✅ driven | ✅ | — |  |
| `Hardtanh` | `hardtanh` | standard | — | `-1, 1` | ✅ driven | ✅ | — |  |
| `Heaviside` | `heaviside` | standard | — | `0, -0` | 🟡 §5.6 | ✅ | — | ⚠️ |
| `LessThanEqualZero` | `less_than_equal_zero` | standard | — | `0, -0` | ✅ driven | ✅ | — |  |
| `LessThanZero` | `less_than_zero` | standard | — | `0, -0` | ✅ driven | ✅ | — |  |
| `Log` | `log` | broad | 0.0 (abo) | — | 🟡 §5.9 | ✅ | — |  |
| `Log1p` | `log1p` | broad | -1.0 (abo) | — | ✅ driven | ✅ | — |  |
| `LogWithBase` | `log_with_base` | standard | 0.0 (abo) | — | 🟡 §5.9 | ✅ | — |  |
| `Lrelu` | `lrelu` | standard | — | `0, -0` | ✅ driven | ✅ | — |  |
| `NotEqualZero` | `not_equal_zero` | standard | — | `0, -0` | ✅ driven | ✅ | — |  |
| `Prelu` | `prelu` | standard | — | `0, -0` | ✅ driven | ✅ | — |  |
| `Rdiv` | `rdiv` | standard | 0.0 (bot) | — | 🟡 §5.9 | ✅ | — |  |
| `Reciprocal` | `reciprocal` | broad | 0.0 (bot) | — | ✅ driven | ✅ | — | ⚠️ `1/NaN` xfail |
| `ReluMax` | `relu_max` | broad | — | `0, 5` | ✅ driven | ✅ | — |  |
| `ReluMin` | `relu_min` | broad | — | `5` | ⬜ | ✅ | — |  |
| `Round` | `round` | standard | — | `-2.5, -1.5, -0.5, 0.5, 1.5, 2.5` | ✅ driven | ✅ | — |  |
| `Rsqrt` | `rsqrt` | broad | 0.0 (abo) | — | ✅ driven | ✅ | — | ⚠️ `rsqrt(-0)` xfail |
| `RsqrtCompat` | `rsqrt_compat` | standard | 0.0 (abo) | — | ⬜ | ✅ | — | ⚠️ |
| `Selu` | `selu` | standard | — | `0, -0` | ✅ driven | ✅ | — |  |
| `Sign` | `sign` | standard | — | `0, -0` | 🟡 §5.6 | ✅ | — | ⚠️ |
| `Signbit` | `signbit` | standard | — | `0, -0` | ✅ driven | ✅ | — | ⚠️ |
| `Softplus` | `softplus` | standard | — | `20` | ✅ driven | ✅ | — |  |
| `Softshrink` | `softshrink` | standard | — | `-0.5, 0.5` | ✅ driven | ✅ | — |  |
| `Sqrt` | `sqrt` | broad | 0.0 (abo) | — | ✅ driven | ✅ | — | ⚠️ `sqrt(-0)` xfail |
| `SqrtCustom` | `sqrt_custom` | standard | 0.0 (abo) | — | 🟡 §5.9 | ✅ | — |  |
| `Threshold` | `threshold` | broad | — | `5` | ✅ driven | ✅ | — |  |
| `Trunc` | `trunc` | broad | — | `-1, 0, 1` | ✅ driven | ✅ | — |  |
| `UnaryGe` | `unary_ge` | standard | — | `0.5` | ✅ driven | ✅ | — |  |
| `UnaryGt` | `unary_gt` | standard | — | `0.5` | ✅ driven | ✅ | — |  |
| `UnaryLe` | `unary_le` | standard | — | `0.5` | ✅ driven | ✅ | — |  |
| `UnaryLt` | `unary_lt` | standard | — | `0.5` | ✅ driven | ✅ | — |  |
| `UnaryMax` | `unary_max` | standard | — | `0, -0` | ✅ driven | ✅ | — |  |
| `UnaryMin` | `unary_min` | standard | — | `0, -0` | ✅ driven | ✅ | — |  |
| `Xielu` | `xielu` | standard | — | `0, -0` | ✅ driven | ✅ | — |  |

#### 4.2 Unary ops smooth everywhere — cat B is their **entire** edge story (47 ops)

**27 of the 47 run.** An op here has no knee and no pole, so until cat B reached it the edge sweep
skipped it outright and its only coverage was the random sweep. Of the remaining 20, **17 are 🟡 §5.9** —
held by one kernel behaviour, not by anything op-specific — and **3 are ⬜**, the `TopK*` stages, which
are perf-only and have no golden at all (§2.4).

| Op | Kernel | Random sweep | Registered domain | Cat B | Edge sweep | Other test |
|---|---|---|---|---|---|---|
| `Abs` | `abs` | broad | yes | ✅ driven | ✅ cat B only | — |
| `Add1` | `add1` | standard | yes | ✅ driven | ✅ cat B only | — |
| `Asinh` | `asinh` | broad | yes | ✅ driven | ✅ cat B only | — |
| `Atan` | `atan` | standard | yes | ✅ driven | ✅ cat B only | — |
| `CastFp32ToFp16a` | `cast_fp32_to_fp16a` | standard | yes | 🟡 §5.9 | ⬜ skips | — |
| `Cbrt` | `cbrt` | standard | yes | ✅ driven | ✅ cat B only | — |
| `Cos` | `cosine` | broad | yes | ✅ driven | ✅ cat B only | — |
| `Cosh` | `cosh` | standard | yes | ✅ driven | ✅ cat B only | — |
| `Digamma` | `digamma` | standard | yes | 🟡 §5.9 | ⬜ skips | — |
| `Erf` | `erf` | standard | yes | 🟡 §5.9 | ⬜ skips | — |
| `Erfc` | `erfc` | standard | yes | 🟡 §5.9 | ⬜ skips | — |
| `Exp` | `exponential` | broad | yes | ✅ driven | ✅ cat B only | — |
| `Exp2` | `exp2` | broad | yes | ✅ driven | ✅ cat B only | — |
| `ExpWithBase` | `exp_with_base` | standard | yes | ✅ driven | ✅ cat B only | — |
| `Expm1` | `expm1` | standard | yes | ✅ driven | ✅ cat B only | — |
| `Expm1Cw` | `expm1_cw` | standard | yes | 🟡 §5.9 | ⬜ skips | — |
| `Fill` | `fill` | broad | yes | ✅ driven | ✅ cat B only | — |
| `Fmod` | `fmod` | standard | yes | ✅ driven | ✅ cat B only | — |
| `Gelu` | `gelu` | broad | yes | 🟡 §5.9 | ⬜ skips | — |
| `GeluAppx` | `gelu_appx` | standard | yes | ✅ driven | ✅ cat B only | — |
| `GeluDerivative` | `gelu_derivative` | standard | yes | 🟡 §5.9 | ⬜ skips | — |
| `GeluTanh` | `gelu_tanh` | broad | yes | ✅ driven | ✅ cat B only | — |
| `I0` | `i0` | standard | yes | ✅ driven | ✅ cat B only | — |
| `I1` | `i1` | standard | yes | 🟡 §5.9 | ⬜ skips | — |
| `Identity` | `identity` | standard | yes | ✅ driven | ✅ cat B only | — |
| `Lgamma` | `lgamma` | standard | yes | 🟡 §5.9 | ⬜ skips | — |
| `Mish` | `mish` | standard | yes | ✅ driven | ✅ cat B only | — |
| `Neg` | `negative` | broad | yes | ✅ driven | ✅ cat B only | — |
| `Polygamma` | `polygamma` | standard | yes | 🟡 §5.9 | ⬜ skips | — |
| `Remainder` | `remainder` | standard | yes | ✅ driven | ✅ cat B only | — |
| `Rpow` | `rpow` | standard | yes | 🟡 §5.9 | ⬜ skips | — |
| `Sigmoid` | `sigmoid` | standard | yes | 🟡 §5.9 | ⬜ skips | — |
| `SigmoidAppx` | `sigmoid_appx` | standard | yes | 🟡 §5.9 | ⬜ skips | — |
| `Silu` | `silu` | broad | yes | ✅ driven | ✅ cat B only | — |
| `Sin` | `sine` | broad | yes | ✅ driven | ✅ cat B only | — |
| `Sinh` | `sinh` | standard | yes | ✅ driven | ✅ cat B only | — |
| `Softsign` | `softsign` | standard | yes | ✅ driven | ✅ cat B only | — |
| `Square` | `square` | broad | yes | ✅ driven | ✅ cat B only | — |
| `Tan` | `tan` | standard | yes | ✅ driven | ✅ cat B only | — |
| `Tanh` | `tanh` | broad | yes | 🟡 §5.9 | ⬜ skips | — |
| `TanhDerivative` | `tanh_derivative` | standard | yes | 🟡 §5.9 | ⬜ skips | — |
| `TanhDerivativeLut` | `tanh_derivative_lut` | standard | yes | 🟡 §5.9 | ⬜ skips | — |
| `Tanhshrink` | `tanhshrink` | broad | yes | ✅ driven | ✅ cat B only | — |
| `TopKLocalSort` | `topk_local_sort` | **perf-only** | yes | ⬜ | ⬜ skips | — |
| `TopKMerge` | `topk_merge` | **perf-only** | yes | ⬜ | ⬜ skips | — |
| `TopKRebuild` | `topk_rebuild` | **perf-only** | yes | ⬜ | ⬜ skips | — |
| `UnaryPower` | `power` | standard | yes | 🟡 §5.9 | ⬜ skips | — |

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
| `LeftShift` | `left_shift` | — | `unary` | ✅ cat E — full shift axis via `SFPU_SHIFT_AMOUNT`, in range and out |
| `LogicalNot` | `logical_not_unary` | `0, -0` | `unary` | ✅ cat D — exact threshold forced by `test_eltwise_unary_sfpu_threshold`, which names it `LogicalNotUnary` |
| `Relu` | `relu` | — | `plot` | ➖ unreachable by design — applied by the packer (`STACC_RELU`, `cpack_common.h`), and `sfpu_operations.h` has no `SfpuType::relu` branch. On WH/BH the only reference is a plotting script; the Quasar suite does drive it |
| `RightShift` | `right_shift` | — | `unary` | ✅ cat E — as `LeftShift` |
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

| Op | Kernel | Registered domain | Cat A pole | Cat B | Edge sweep | Driven by |
|---|---|---|---|---|---|---|
| `SfpuAddTopRow` | `ADD_TOP_ROW` | yes | — | — | ⬜ | `binary` |
| `SfpuAtan2` | `ATAN2` | no (format default) | B=0.0 (bot) | 🟡 §5.6 Q1 | ✅ | `binary`, `binary_edges` |
| `SfpuBinaryFmod` | `FMOD` | no (format default) | B=0.0 (bot) | 🟡 §5.6 Q1 | ✅ | `binary` |
| `SfpuBinaryMax` | `MAX` | no (format default) | — | ✅ driven | ✅ | `binary` |
| `SfpuBinaryMin` | `MIN` | no (format default) | — | ✅ driven | ✅ | `binary` |
| `SfpuBinaryRemainder` | `REMAINDER` | no (format default) | B=0.0 (bot) | 🟡 §5.6 Q1 | ✅ | `binary` |
| `SfpuBitwiseAnd` | `BITWISE_AND` | no (format default) | — | ➖ int axis | ⬜ | `binary` |
| `SfpuBitwiseOr` | `BITWISE_OR` | no (format default) | — | ➖ int axis | ⬜ | `binary` |
| `SfpuBitwiseXor` | `BITWISE_XOR` | no (format default) | — | ➖ int axis | ⬜ | `binary` |
| `SfpuDivInt32` | `DIV_INT32` | no (format default) | — | ➖ int axis | ⬜ | `binary` |
| `SfpuDivInt32Floor` | `DIV_INT32_FLOOR` | no (format default) | — | ➖ int axis | ⬜ | `binary` |
| `SfpuElwEq` | `EQ` | no (format default) | — | ✅ driven | ✅ | `binary` |
| `SfpuElwGe` | `GE` | no (format default) | — | ✅ driven | ✅ | `binary` |
| `SfpuElwGt` | `GT` | no (format default) | — | ✅ driven | ✅ | `binary` |
| `SfpuElwLe` | `LE` | no (format default) | — | ✅ driven | ✅ | `binary` |
| `SfpuElwLeftShift` | `LSHFT` | yes | — | ➖ int axis | ⬜ | `binary` |
| `SfpuElwLogicalRightShift` | `LOGICAL_RSHFT` | yes | — | ➖ int axis | ⬜ | `binary` |
| `SfpuElwLt` | `LT` | no (format default) | — | ✅ driven | ✅ | `binary` |
| `SfpuElwNe` | `NE` | no (format default) | — | ✅ driven | ✅ | `binary` |
| `SfpuElwRightShift` | `RSHFT` | yes | — | ➖ int axis | ⬜ | `binary` |
| `SfpuElwadd` | `ADD` | yes | — | ✅ driven | ✅ | `binary` |
| `SfpuElwdiv` | `DIV` | yes | B=0.0 (bot) | 🟡 §5.6 Q1 | ✅ | `binary` |
| `SfpuElwmul` | `MUL` | yes | — | ✅ driven | ✅ | `binary` |
| `SfpuElwpow` | `POW` | yes | A=0.0 (abo) | 🟡 §5.6 Q1 | ✅ | `binary`, `zz_measure_tol` |
| `SfpuElwrsub` | `RSUB` | yes | — | ✅ driven | ✅ | `binary` |
| `SfpuElwsub` | `SUB` | yes | — | ✅ driven | ✅ | `binary` |
| `SfpuEqInt` | `EQ_INT` | no (format default) | — | ➖ int axis | ⬜ | `binary` |
| `SfpuFmodInt32` | `FMOD_INT32` | no (format default) | — | ➖ int axis | ⬜ | `binary` |
| `SfpuGcd` | `GCD` | no (format default) | — | ➖ int axis | ⬜ | `binary` |
| `SfpuIsclose` | `ISCLOSE` | no (format default) | — | 🟡 read-back | ⬜ | `binary` |
| `SfpuLcm` | `LCM` | no (format default) | — | ➖ int axis | ⬜ | `binary` |
| `SfpuLogsigmoid` | `LOGSIGMOID` | no (format default) | — | ➖ unary in B | ⬜ | `binary` |
| `SfpuMask` | `MASK` | no (format default) | — | 🟡 §5.6 Q3 | ⬜ | `binary` |
| `SfpuMaxInt32` | `MAX_INT32` | no (format default) | — | ➖ int axis | ⬜ | `binary` |
| `SfpuMaxUint32` | `MAX_UINT32` | no (format default) | — | ➖ int axis | ⬜ | `binary` |
| `SfpuMinInt32` | `MIN_INT32` | no (format default) | — | ➖ int axis | ⬜ | `binary` |
| `SfpuMinUint32` | `MIN_UINT32` | no (format default) | — | ➖ int axis | ⬜ | `binary` |
| `SfpuMulInt32` | `MUL_INT32` | no (format default) | — | ➖ int axis | ⬜ | `binary` |
| `SfpuNeInt` | `NE_INT` | no (format default) | — | ➖ int axis | ⬜ | `binary` |
| `SfpuRemainderInt32` | `REMAINDER_INT32` | no (format default) | — | ➖ int axis | ⬜ | `binary` |
| `SfpuRemainderUint32` | `REMAINDER_UINT32` | no (format default) | — | ➖ int axis | ⬜ | `binary` |
| `SfpuRsubInt32` | `RSUB_INT32` | no (format default) | — | ➖ int axis | ⬜ | `binary` |
| `SfpuXlogy` | `XLOGY` | yes | B=0.0 (abo) | 🟡 §5.6 Q1 | ✅ | `binary`, `zz_measure_tol` |

**Reading the two new columns.** `Cat B` is `BINARY_SPECIALS_READY_OPS`, the golden-side gate — the
pipeline-side one is still `specials_safe()`, and both must pass. **12 ops are enrolled and green on
Wormhole across all 6 safe cells.** The markers on the rest name the single thing holding each out, the
same convention §4.1/§4.2 use:

- **🟡 §5.6 Q1** — 6 ops, all compositions over a primitive the ISA specifies only inside a stated
  range (`div` and `fmod`/`remainder` via a reciprocal, `xlogy` via a log, `pow` via `exp(b·ln a)`,
  `atan2` via a ratio plus a format-specific polynomial). This is the *binary half of the same question*
  that holds 23 unary ops out — one answer decides all 29.
- **🟡 §5.6 Q3** — `SfpuMask` alone. `calculate_mask` is `v_if(mask == 0)`, which lowers to `SFPSETCC`,
  whose contract is conditioned *"provided that VC is neither negative zero nor any kind of NaN"*. The
  same sentence holds `Sign` and `Heaviside` out on the unary side.
- **🟡 read-back** — `SfpuIsclose` alone, and the narrowest item in the document.
  `ckernel_sfpu_isclose.h` documents torch.isclose semantics *including* `EQUAL_NAN=false` and
  bit-inspects for `±Inf` against NaN, so both sides claim to agree and do not. One per-cell read-back
  says which is wrong. **Do not adjust the golden first** — that is how a kernel divergence gets
  laundered into a golden that agrees with it.
- **➖ unary in B** — `SfpuLogsigmoid` is out *by construction*, not pending. The kernel reads `in1` only
  on its `x > 4` branch and the golden ignores operand B outright (verified: `logsigmoid(1, y)` is
  constant in `y`), so a special injected into B is not a stimulus for anything.
- **➖ int axis** — the 21 int-typed rows. `specials_safe()` returns False for an integer input, so there
  is no IEEE special to inject; their edges are cat C and cat E, covered by
  `test_sfpu_binary_int_extremes` and the shift sweeps.

`Edge sweep` flips to ✅ for an op that gains **either** a registered singularity **or** a cat-B entry —
`_BINARY_EDGE_OPS` is `ops_with_singularity() | BINARY_SPECIALS_READY_OPS`, so 18 of the 22 float rows
are collected where 5 were. The three ⬜ float rows are exactly the ops with no pole *and* no cat B.

#### 4.5 Binary integer SFPU ops (5 ops)

> **These five are a *naming artifact*, not a coverage gap.** They read as driven by "none (WH/BH)",
> which is true of the **enum members** and false of the **kernels**. All five kernels are covered on Wormhole and Blackhole under a different spelling, and the
> alias crosses a `MathOpType` boundary — which is why §2.4's alias warning did not catch it.
>
> - The WH/BH `BinaryOp` enum (`tt_llk_<arch>/common/inc/ckernel_defs.h`) has **no** `GT_INT` /
>   `LT_INT` / `LE_INT` / `GE_INT`; those four names exist only in the Quasar enum. So these members
>   carry `MathOpType.SFPU_BINARY_INT`, which only `sfpu_operations_quasar.h` implements, and are
>   **unreachable** on WH/BH rather than untested.
> - Meanwhile `sfpu_operations.h` routes `BinaryOp::LT/GT/LE/GE` to `calculate_binary_comp_int32`
>   whenever `MATH_FORMAT == Int32`, and `test_sfpu_binary_int` drives `SfpuElwLt/Gt/Le/Ge` at `Int32`
>   on both arches. **That is the same kernel.**
> - `SfpuElwmulInt`'s `cpp_enum_value` is `MUL`, reaching `_mul_int32_` on Quasar; on WH/BH the same
>   kernel is `MUL_INT32`, driven by `test_sfpu_binary_int_uniform` as `SfpuMulInt32`.
>
> Pinned by `test_quasar_int_binary_members_alias_covered_kernels`, which parses the arch header rather
> than mirroring it — so if one of these ever *becomes* dispatchable, or the alias stops being driven,
> that fails instead of this paragraph going quietly stale.

| Op | Kernel | Registered domain | WH/BH reachability | Kernel covered on WH/BH via |
|---|---|---|---|---|
| `SfpuElwmulInt` | `MUL` (int) | no (format default) | ➖ Quasar-only member | `SfpuMulInt32` (`MUL_INT32`) — `binary_int_uniform` |
| `SfpuGeInt` | `GE_INT` | no (format default) | ➖ not in the WH/BH `BinaryOp` enum | `SfpuElwGe` at `Int32` — `binary_int`, `binary_int_comparison_*`, `binary_int_extremes` |
| `SfpuGtInt` | `GT_INT` | no (format default) | ➖ not in the WH/BH `BinaryOp` enum | `SfpuElwGt` at `Int32` — as above |
| `SfpuLeInt` | `LE_INT` | no (format default) | ➖ not in the WH/BH `BinaryOp` enum | `SfpuElwLe` at `Int32` — as above |
| `SfpuLtInt` | `LT_INT` | no (format default) | ➖ not in the WH/BH `BinaryOp` enum | `SfpuElwLt` at `Int32` — as above |

**What *was* genuinely missing, and is now covered.** The Int32 comparison kernel was being driven
entirely on `generate_stimuli`'s integer default — `uniform(0, INT32_MAX // 2 - 1)`, which is
**positive-only and tie-free**. Measured: a 1024-element draw contained **0 negatives and 0 ties**. That
is finding #1 of the original audit surviving on the integer axis, and it left three holes:

| Gap | Why it mattered | Closed by |
|---|---|---|
| the exact-equality input | `a == b` is the **only** input on which `lt`/`gt` disagree with `le`/`ge`; a comparator with its tie inverted passed the whole integer sweep | `test_sfpu_binary_int_comparison_ties` (reuses the float sweep's three-way builder) |
| operands crossing zero | the kernel normalises by computing `a - b` and folding the sign — sign is the entire mechanism, and it was never exercised | `test_sfpu_binary_int_comparison_across_zero`, `twos_complement=True`; **verified delivered**, 1024 of 2048 lanes negative, so the assertion is not vacuous |
| cat C at the int32 extremes | `INT32_MAX - (INT32_MIN + 1)` does not fit in int32, so whether the sign-fold survives overflow is exactly what "exact on the full range" has to mean. These kernels document no sub-range, so §2.6's exclusion does not apply to them | the 4 ordered comparisons added to `_INT_EXTREME_OPS` |

All green on Wormhole. The kernel is correct at every one of these; the point is that now it is
*asserted* rather than assumed.

#### 4.6 Ternary ops (5 ops)

| Op | Kernel | Registered domain | Cat A pole | Edge sweep | Driven by |
|---|---|---|---|---|---|
| `SfpuAddcdiv` | `addcdiv` | no (format default) | C=0.0 (bot) | ✅ | `ternary` |
| `SfpuAddcmul` | `addcmul` | no (format default) | — | ⬜ | `ternary` |
| `SfpuLerp` | `lerp` | no (format default) | — | ✅ | `ternary` |
| `SfpuSnakeBeta` | `snake_beta` | no (format default) | C=0.0 (bot) | ✅ | `ternary` |
| `SfpuWhere` | `where` | no (format default) | — | ⬜ | `ternary` |

Operand-C edges, driven by `test_sfpu_ternary_edges` (9 passed / 7 skipped on Blackhole):

| Op | Formula | Operand-C probe | Source |
|---|---|---|---|
| `SfpuAddcdiv` | `a + value * b / c` | `-0.015625, 0.0, 0.015625` | `_OP_SINGULARITIES` C = (0.0, BOTH) |
| `SfpuSnakeBeta` | `a + sin(b*a)^2 / c` | `-0.015625, 0.0, 0.015625` | `_OP_SINGULARITIES` C = (0.0, BOTH) |
| `SfpuLerp` | `a + c * (b - a)` | `-1.0, 0.0, 1.0, 2.0` | `_OP_OPERAND_EDGE_POINTS` C |
| `SfpuAddcmul` | `a + value * b * c` | none | a multiply has no pole; `edge_spec` returns `None` |

The probe offset is format-relative, so the pole probes become `+/-0.25` in Bfp4_b. `c` is
zero for 4064 of 4096 elements (custom() zero-fills each face), so the pole is driven hard
rather than sampled. The numerator is held off zero for the two dividing ops so the variant
asserts the pole instead of the `0/0` indeterminate form -- see the coverage note in
`test_sfpu_ternary.py`.

#### 4.7 Scalar-binop ops (5 ops)

All five are `x (+|-|*|/) c` for a compile-time `c`, so they are smooth in `x`: no pole, no knee, and
`edge_spec()` returns `None` unless specials are on. **Cat B is their entire edge story**, and all five
are now enrolled — the tensor-operand edge sweep runs them where the pipeline delivers specials.

Two of the eight (format, `dest_acc`) pairs survive both gates, and they are complementary rather than
redundant: `Float32`/`dest_acc=Yes` is unpack-to-dest so a real `-0.0` arrives, and
`Float16_b`/`dest_acc=No` is the datacopy path where it does not. The other six are excluded by
`_skip_unsupported` (Float32 needs a 32-bit Dest, Float16_b cannot use one).

| Op | Kernel | Registered domain | Cat A pole | Cat B | Edge sweep | Driven by |
|---|---|---|---|---|---|---|
| `ScalarAdd` | `ADD` | no (format default) | — | ✅ driven | ✅ cat B only | `binop_scalar` |
| `ScalarDiv` | `DIV` | no (format default) | — | ✅ driven | ✅ cat B only | `binop_scalar` |
| `ScalarMul` | `MUL` | no (format default) | — | ✅ driven | ✅ cat B only | `binop_scalar` |
| `ScalarRsub` | `RSUB` | no (format default) | — | ✅ driven | ✅ cat B only | `binop_scalar` |
| `ScalarSub` | `SUB` | no (format default) | — | ✅ driven | ✅ cat B only | `binop_scalar` |

`ScalarDiv` has no reachable divide-by-zero: the host inverts the divisor at compile time and the
kernel only multiplies, so `d` never reaches the device. That is a property of the dispatch, not an
untested edge.

Still out of scope, both needing a per-op tolerance first (the default bf16 tolerance is only
meaningful while the result stays in range): `|scalar| > 8`, and `±tiny` / `±large` on the tensor
operand.

#### 4.8 Reduce ops (3 ops)

| Op | Kernel | Registered domain | Cat A pole | Cat B | Cat C | Driven by |
|---|---|---|---|---|---|---|
| `ReduceColumn` | `REDUCE_COL` | yes | — | ✅ 6 classes × 4 pools × 2 formats | ✅ `int32_reduce_extreme` | `reduce`, `float_reduce_specials`, `reduce_sdpa` |
| `ReduceRow` | `REDUCE_ROW` | yes | — | ✅ as `ReduceColumn` | ✅ `int32_reduce_extreme` | `reduce`, `float_reduce_specials` |
| `ReduceScalar` | `REDUCE_SCALAR` | yes | — | ⬜ **different driver** | ⬜ | `reduce` |

**Why the `Edge sweep` column is gone for this family rather than ticked.** All three carry a plain
`uniform(-1, 1)` domain with no singularity and no knee, so `edge_spec()` returns `None` and **no edge
sweep can reach them however it is pointed** — that is a property of the registry, and cat B does
not change it. What they have instead is cat B, and it behaves unlike cat B anywhere else here: a
reduction *propagates* its special to the single output element, so one poisoned lane is not one probe
among 4096, it is the whole answer. The pool is a *parameter* rather than part of the op, so
"the identity of this pool" is not something `_OP_EDGE_POINTS` can hold; the classes are parametrized in
the test file, following `test_int32_reduce_extreme`'s shape.

The six classes, and what each asserts that no element-wise sweep can:

| Class | Stimulus | What it pins |
|---|---|---|
| `pos_inf` / `neg_inf` | one lane, 31 finite | absorption for `Max`/`Sum`; **transparency** for `Min` — the asymmetry is the point |
| `both_inf` | `+inf` and `-inf` in one column | `Sum` must be NaN; `Max`/`Min` must still answer `±inf` |
| `nan` | one lane | the total-order case. `Min` over a column holding `+NaN` must return the **finite** minimum, where `torch.min` propagates |
| `all_inf` | every lane `+inf` | the degenerate fold, where the pool identity is the only other operand |
| `signed_zero` | every lane `-0.0` | IEEE keeps `-0` under addition; Wormhole's SFPMAD flushes it and Blackhole preserves it |

Injection is down **one column** rather than scattered, so a single reduced lane carries the special and
the other 31 stay asserted — a scattered injection leaves every lane poisoned and the variant can then
only report "something in this tensor diverges".

**96 variants, all green on Wormhole**, after two golden corrections that the class list exposed: the
reduce path returned from `UnarySFPUGolden.__call__` *before* the Dest/pack modelling, and
`Max`/`Min` were folding with `torch.max`/`torch.min` instead of the comparator the kernel uses.

**Cat C for this family was already closed** by `test_int32_reduce_extreme` and is not re-filed.
`ReduceScalar` stays outside: it is driven from `test_reduce.py` / `test_mul_reduce_scalar.py` rather
than `test_sfpu_reduce.py`, so covering it means a second driver. **That is an open item, recorded here
so it is a decision rather than an omission.**

#### 4.9 FPU binary (eltwise) ops (3 ops)

| Op | Kernel | Registered domain | Cat A pole | Edge sweep | Driven by |
|---|---|---|---|---|---|
| `Elwadd` | `ELWADD` | yes | — | ⬜ | `deepseek_moe_gate`, `eltwise_binary`, `generalized_moe_gate` +4 |
| `Elwmul` | `ELWMUL` | yes | — | ⬜ | `deepseek_moe_gate`, `eltwise_bcast_col_custom`, `eltwise_binary` +3 |
| `Elwsub` | `ELWSUB` | yes | — | ⬜ | `deepseek_moe_gate`, `eltwise_bcast_col_custom`, `eltwise_binary` +5 |

---

## 5. Open divergences, and what each waits on

**8 ops over 46 cells** disagree with their golden at a driven point — 7 unary over 24 (`Sign` 2,
`Heaviside` 2, `RsqrtCompat` 8, `Erfinv` 2, `Reciprocal` 6, `Sqrt` 2, `Rsqrt` 2) and 1 binary over 22
(`SfpuElwpow`'s `0**0`, 6 cells, plus the 16 signed-zero cells that are arch-gated rather than
divergent). §1 carries the same figures.

Two things to know before quoting them. A unary cell is an `(op, input, output, dest_acc)` variant; a
**binary** cell is `(op, edge_class, input, output, dest_acc)`, because the binary sweep splits each
op's poles into classes and marks them separately — so the two halves are not counted on the same axis.
And the three cat-B unary sets are *derived*, not listed: `_cat_b_divergences()` computes them from the
combinations that actually deliver the probe, so re-derive rather than trusting this sentence.

All are recorded as **non-strict xfails**, so the case still executes and reports XPASS if the
behaviour changes. Every one is cross-checked against
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
both arches, so unlike the `SFPMAD` group this is **not** generational. Confirmed: these still xfail on
Blackhole while the `SFPMAD` group XPASSed there.

### 5.3 Still open — not explained by the ISA

| Finding | Ops |
|---|---|
| **`0**0` returns 0** where C, torch and the golden give 1. `pow` evaluates `exp(b·ln a)`, so a composition artifact. **Survives the retraction above** — both operands and the result are finite, no NaN anywhere near it, which is what `_class_generates_a_nan()` keeps it out of that gate for | `pow` |
| **`RsqrtCompat(0)` saturates to `1.7014118e38`** (`0x7F000000`) instead of `inf`, on all 8 combinations — while plain `Rsqrt` over the same probe does **not** diverge. Two implementations of one function disagreeing at their shared pole, with nothing in the ISA prescribing either answer | `RsqrtCompat` |
| **`Erfinv(±1)` saturates** rather than returning ±inf, on the fp32-dest combinations only — tolerance-shaped rather than semantic | `erfinv` |

### 5.5 `Log` saturates its input, so no non-finite value survives it

Found by enabling cat B. On Blackhole, with a Float32 input:

| Probe | Golden | Hardware |
|---|---|---|
| `+inf` | `+inf` | **88.5** |
| `-inf` | `NaN` | **84.3** |
| `NaN` | `NaN` | **89.1** |
| `-0` | `-inf` | **-92.5** |

All finite, and all near `ln(FLT_MAX) = 88.7`. The kernel clamps its input to the format maximum and
takes the log of *that*, so a non-finite input cannot produce a non-finite output. This is a kernel
behaviour rather than a golden one and it is the largest cat-B finding so far.

### 5.9 The `Log` saturation is a whole-family behaviour, not one op

§5.5 recorded `Log` clamping its input to the format maximum. Driving the full unary set shows **22
further ops** doing the same kind of thing — every one a polynomial or LUT approximation evaluated
outside the range its series covers. They split two ways, and neither is IEEE:

- **Saturating to the asymptote or a magic constant:** `LogWithBase` (`127.9` / `121.6` / `128.5`),
  `Digamma` (`89.09` at `NaN`; `≈ -337920` at `±0`), `I1` (`±1.1547668e37`), `Erf` (`1.0`), `Erfc`
  (`2.94e-12`), `Tanh` (`1.0`), `Sigmoid` / `TanhDerivative` / `Rdiv` / `Polygamma` (`0.0`), `Gelu` and
  `GeluDerivative` (`0.0` / `1.0`), `Lgamma` (`-0.00051` at `±0`), `UnaryPower` / `Rpow` /
  `CastFp32ToFp16a` (`+inf`).
- **Returning `NaN` where a value is defined** — the same failure from the other side: `Frac` at `±inf`,
  `SigmoidAppx` at `±inf`, `TanhDerivativeLut`, `Expm1Cw` at `+inf`, `Lgamma` at `±inf`, `SqrtCustom`
  (which manages `NaN` at `+inf` *and* `+inf` at `-inf`, both backwards), and `Erfinv` at `±1`.

**`LogWithBase` is the evidence that the cause is shared.** Its results are `Log`'s multiplied by the
dispatch scale `1/ln(2) ≈ 1.4427` — `88.7 × 1.4427 = 128.0`. Same clamp, seen through a scale factor.

So §5.6's first question is not about `Log`: it is a contract question about approximation kernels, and
one answer settles 23 held-out ops.

### 5.10 The generated-NaN sign — gated on Wormhole; the comparator repair is still open

`SFPMAD.md` pins the emitted NaN bit pattern differently per arch:

| Arch | "if a NaN is emitted" |
|---|---|
| Blackhole | "it is always **the canonical NaN with bit pattern `0x7fc00000`**" |
| Wormhole | "the least significant bit of the mantissa is guaranteed to be set; other bits of the mantissa might or might not be set, and **the sign bit might or might not be set**" |

The sign is invisible while the NaN stays a NaN — `passed_test`'s both-NaN clause accepts either — and
becomes an observable `+inf`/`-inf` disagreement the moment the result crosses a 16-bit Dest or pack,
where `convert_nan_to_inf` substitutes a *signed* infinity. So the kernels are in spec and the **golden**
was asserting a sign the ISA declines to promise.

**Current state: gated, not repaired.** `nan_survives_to_l1()` mirrors `UnarySFPUGolden`'s own
preservation rule, `GENERATED_NAN_SIGN_OPS` is the measured set of ops that *invent* a NaN rather than
forward one, and `specials_after_nan_sign_gate()` turns cat B off where both hold **and** the arch is
Wormhole. A skip and not an xfail, because "might or might not be set" makes an xfail a coin-flip gate.
Blackhole keeps every one of these variants. The binary family has the same gate without an op argument
(`generated_nan_sign_is_asserted()`), and the reduce family splits it by pool — `Sum`/`Average` emit
through SFPMAD, `Max`/`Min` select a lane, so only the first pair is relaxed.

**What is left, and it is the open part.** The gate stops the golden asserting what it cannot know; it
does not restore the assertion in the weaker form the ISA *does* support — **an infinity of either**
**sign**. That is a change to `convert_nan_to_inf`'s contract rather than to the gate, and it has to keep
the sign assertion for the ops that genuinely *move* the sign bit (`Neg`, `Abs`, `Identity` — `SFPABS`'s
summary says *"-NaN is left as -NaN rather than becoming +NaN"*). **Until it lands, every gated cell is a**
**coverage loss rather than a closed item** — 50 unary cells + 1 scalar, pinned by
`test_nan_sign_gate_matches_the_measured_wormhole_failures` so the reach cannot widen unnoticed. Plan §4.

`GENERATED_NAN_SIGN_OPS` is also a *measured* set rather than a derived one, which is its own open
question: `GeluTanh` and `Xielu` build a NaN through SFPMAD in the same `inf + (-inf)` shape as the gated
`GeluAppx`, but read back raw they come out sign-clear and so are currently out. Plan §10.3.

### 5.11 The approximate-exp Wormhole gate XPASSes on Wormhole

`_APPROX_EXP_ACCURACY_XFAIL` records *"a systematic ~5.7% overshoot (peak 6.75%) once approximate exp's
argument passes ~8, measured on Wormhole"*, and `_APPROX_EXP_XFAIL_IS_WORMHOLE_ONLY` narrows it to Wormhole
because Blackhole XPASSed all four reachable cells. On a Wormhole n300 **all 6 marked variants XPASS** —
the gate's entire content, three cells at both tile shapes — so it now asserts nothing on either arch.

> **Re-confirmed 2026-08-17** at `26c61ff80e9` on the same n300: 6 XPASS, unchanged, and they are the
> only XPASSes in the unary suite (the edge sweep reports 0). The finding stands exactly as written, and
> the gate is still in the code untouched — `_APPROX_EXP_XFAIL_IS_WORMHOLE_ONLY = True`, with a comment
> that still reads *"the limit is real on Wormhole"*. Plan §9.1 is unstarted.

Measured over the elements with `x > 8` (`test_sfpu_wh_approx_exp.py`): mean signed relative error
**+0.75% to +1.05%**, peak **+3.5%**, and **no element of any tile above 5%**. So the direction reproduces
and the magnitude does not — the overshoot is real and roughly five times smaller than recorded, which puts
it inside the default 5% rtol. `Float16_b->Float16_b` at `dest_acc=Yes`, deliberately *not* gated, behaves
identically to the cells that are.

Three explanations are eliminated: the stimulus still reaches the region (425 and 261 elements above 8 per
tile, `x_max` 9.98 / 15.98, `_APPROX_ACCURACY_MAX[Exp]` = 16.0); no tolerance was loosened
(`CUSTOM_TOLERANCES` has no `Exp` entry and `passed_test` requires `torch.all(is_valid)`); and the golden is
plain `torch.exp`. Either the kernel's approximate path changed since the gate was written, or the overshoot
varies by board — and the recorded measurement does not name its card, which is what makes it unsettleable
from one host. Plan §9.1.

**One open item closes here.** §2.8's *"the accurate exp path over (16, 80] has never been isolated on
hardware"* — the Wormhole broad sweep ran `Exp` (132 passed) and `Exp2` (138 passed) at
`ApproximationMode.No` with 0 failures, and the probe measures **+0.00%** error above 8 out to `x = 79.97`
on the 32-bit-input cells. Sound on Wormhole; Blackhole still wants the same run.

### 5.12 The signed-zero arch gate XPASSes on Wormhole too

`_WORMHOLE_ONLY_EDGE_CLASSES` holds one class, `negative_zero_golden`, on the strength of *"measured on a
Blackhole p150b, the negative-zero class XPASSed on **all 16** cells it is claimed for"* — read as the ISA's
arch difference, `SFPMAD` flushing a negative zero on Wormhole and preserving it on Blackhole. On a Wormhole
n300 **the same 16 XPASS**: `SfpuElwdiv`, `SfpuXlogy`, `SfpuBinaryFmod` and `SfpuBinaryRemainder` at all four
`(format, dest_acc)` cells, which is again the gate's entire content.

> **Re-confirmed 2026-08-17** at `26c61ff80e9`: the binary edge sweep reports `50 passed · 64 skipped ·
> 30 xfailed · 16 xpassed`, the same 16. `_WORMHOLE_ONLY_EDGE_CLASSES` is unchanged in the code and
> still holds only `negative_zero_golden`. Plan §9.2 — the bitwise comparison that would settle whether
> this is a comparator blind spot or a wrong arch premise — is unstarted.

**A gate that XPASSes on both arches cannot mean "the other arch is better".** The likelier reading is
already recorded as a trap in plan §8: `passed_test` compares with `torch.isclose`, a both-NaN
clause and PCC, and `-0.0 == +0.0` under all three. If the comparator cannot see a zero's sign, these
variants pass whatever the hardware does — and the Blackhole XPASS was evidence about the comparator, not
about Blackhole.

That is a hypothesis, and one cheap experiment settles it: compare the class **bitwise** on Wormhole. If
hardware returns `+0.0` where the golden says `-0.0`, the divergence is real but invisible and the class
needs a bitwise comparator — a suite-wide change — with the arch gate spurious. If
hardware returns `-0.0`, Wormhole is not flushing and the gate's premise is wrong on its own terms. Until
then `_WORMHOLE_ONLY_EDGE_CLASSES` is unverified. Plan §9.2.

### 5.6 What to raise with kernel owners

Written up in full, with measured tables and a reproduce command, in `KERNEL_OWNER_QUESTIONS.md`.
**Two remain**, plus one narrow one the ISA raised rather than settled:

1. **Approximation kernels do not propagate non-finite inputs** (§5.5, §5.9) — **29 ops as of
   17: the original 23 unary, plus 6 binary compositions** (`SfpuElwdiv`, `SfpuXlogy`, `SfpuElwpow`,
   `SfpuBinaryFmod`, `SfpuBinaryRemainder`, `SfpuAtan2`). Is the input clamp intended, and should it be
   documented? The ISA cannot settle this and it is worth knowing why: it specifies the *primitives*
   only within stated ranges — `SFPARECIP` gives accuracy bounds for `0 ≤ x < 2` and suggests following
   up with Newton-Raphson, `SFPLUTFP32` documents no handling for `NaN`/`±inf` — so the out-of-range
   behaviour of a composition built on them is an LLK/API decision by construction.
   **This one answer now decides 29 ops rather than 23**, which raises its value and does not change
   its shape. Note what it is *no longer* about: the `0/0` and `x%0` items that used to be filed here
   for these same four ops are retracted (§5.3) — the question is about a non-finite **input**, not
   about an indeterminate form over finite operands.
2. **`RsqrtCompat(0)` saturates to `1.7014118e38`** where plain `Rsqrt` does not — 1 op. The ISA
   narrows it: `SFPARECIP` saturates to `0x7f800000` (`+inf`) for an input below `2^-126`, so
   `0x7F000000` (`2^127`) is not a value the instruction produces. The constant is a software clamp
   added above the primitive, and `Rsqrt`'s `+inf` is what the hardware itself would give. The
   question is therefore *why the clamp was added*, not which one the hardware does.

**The `NaN` comparison question is answered and acted on.** `SFPGT`, `SFPLE` and `SFPSWAP`
document a total order in which `+NaN` is the largest FP32 value, so seven of those nine ops were
golden work rather than a question, and are now enrolled. Two remain for an owner:

3. **Is `SFPSETCC` usable with a `NaN` operand?** Its contract is conditioned — *"Provided that `VC` is
   neither negative zero nor any kind of NaN"* — which leaves `Sign` and `Heaviside` returning `1.0`
   at `NaN` by an unspecified route, even though it is consistent with an `int32` test on a positive
   NaN's bit pattern. **Both arches, one question:** Wormhole has since been measured and returns the
   same `1.0`, so this is a contract question rather than a per-arch measurement.
   **A third op belongs to it: `SfpuMask`.** `calculate_mask` is `v_if(mask == 0)`, the same
   compare-against-zero lowering to the same instruction, so the mask operand hits the same excluded
   case. Three ops on one sentence of contract.
4. ~~**What is the intended `NaN` comparison behaviour on Wormhole?**~~ **Withdrawn — measured, and the
   premise was wrong twice over.** `SFPSWAP` specifies the same total order on Wormhole, and all seven
   goldens pass there 8/8 with `0 xpassed`. Nothing for an owner.

Two questions are **withdrawn** and should not be re-filed. `signbit`: the delivery measurement shows the probe
is not delivered on those six combinations, so there is no kernel contract to question. The
**generated-NaN sign on Wormhole** (§5.10): 49 failing variants that read as a kernel divergence and are
documented behaviour — `SFPMAD.md` leaves the sign unspecified on Wormhole and guarantees canonical
`0x7fc00000` on Blackhole. It is golden work, not a kernel question.

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

**Safe surface — 7 cells of 50** (5 formats × 5 × both `dest_acc`):

| `dest_acc` | Safe `input -> output` |
|---|---|
| `No` | `Float32->Float32`, `Float32->Float16_b`, `Float16_b->Float32`, `Float16_b->Float16_b` |
| `Yes` | `Float32->Float32`, `Float32->Float16`, `Float32->Float16_b` |

Measured on Wormhole, and **re-measured there on 2026-08-13 with the same instrument: confirmed.** 250
variants, 85 failing again, and aggregating to the 50-cell matrix every one of the 7 safe cells passes all
5 predicates while no safe cell fails any. Both breakers reproduce in shape, `Float16_b` at
`dest_acc=Yes` included (2/5: `+inf` survives, `-inf` and `NaN` do not). One detail this section did not
record: among the statically-excluded block-float rows, a **`Bfp8_b` input at `dest_acc=No` genuinely
fails** `isinf`/`isneginf`/`isnan` (2/5 on four of its five outputs; only `Bfp8_b->Float16` passes), while
a `Bfp4_b` input passes 5/5 everywhere — so the static exclusion is covering a failure in one case and a
vacuous pass in the other.

The whole matrix is written out longhand in `test_sfpu_domains.py` (7 accepted cells of 50) so it cannot
be rewritten without a test changing outcome — including a guard for the `DestAccumulation` truthiness
trap, where both enum members are truthy and `bool(member)` would silently flip whole rows.

**Blackhole: 3 of the 7 confirmed, and the other 4 are unreachable there by construction** — not by
omission. `_skip_bh_unless_fp32` allows only `Float32->Float32` at `dest_acc=No`, which collapses that
row's four triples to one, and the edge sweep's format axis is `Float16_b`/`Float32`, so
`Float32->Float16` at `dest_acc=Yes` is never collected. The three reachable cells
(`Float32->Float32` at both `dest_acc`, `Float32->Float16_b` at `dest_acc=Yes`) **do** carry specials
on Blackhole — all nine enrolled cat-B ops pass there, modulo the three recorded kernel xfails in §2.1.

One caveat the three cells do **not** cover: carrying `±inf` and `NaN` is not the same as carrying a
`-0.0`, and only the two `dest_acc=Yes` cells do the latter. `negative_zero_delivered()` is the second
gate.

The table is therefore **not** arch-keyed, and — now that the Wormhole re-measurement has been done and
agrees — should stay that way. `test_specials_safe_matches_measured_matrix` keeps its one verdict per cell
rather than being parametrized by arch.

---

## 7. How to regenerate this document

Nothing here needs hardware. §1's figures and §4's tables come from the same inventory:

```bash
cd tt_metal/tt-llk/tests/python_tests
python3 -c "
import sys; sys.path.insert(0,'.')
from helpers.sfpu_domains import (_OP_SINGULARITIES, _OP_EDGE_POINTS, _OP_DOMAIN_REGISTRY,
                                  sfpu_unary_ops, edge_spec, SPECIALS_READY_OPS,
                                  BINARY_SPECIALS_READY_OPS, _BINARY_SPECIALS_NOT_READY)
from helpers.llk_params import MathOperation, DataFormat as F
u = sorted(sfpu_unary_ops(), key=lambda o: o.name)
e = [o for o in u if edge_spec(o, F.Float32, F.Float32) is not None]
print('singularities', len(_OP_SINGULARITIES), '| edge points', len(_OP_EDGE_POINTS))
print('unary', len(u), '| with an edge', len(e), '| smooth', len(u) - len(e))
print('specials-ready', len(SPECIALS_READY_OPS))
print('binary cat B ready', len(BINARY_SPECIALS_READY_OPS),
      '| deferred', len(_BINARY_SPECIALS_NOT_READY))
"
# expect: 22 / 43 / 97 / 50 / 47 / 72 / 12 / 9
# (The 22nd singularity is SfpuAtan2's B = 0 branch point. Re-run rather than trusting the number.)
# The last number is len(SPECIALS_READY_OPS), which counts the 5 scalar binops as well
# as the 67 enrolled *unary* ops -- the scalar family is not in sfpu_unary_ops(), so it
# does not appear in any of the other five figures. Do not read 72 as a unary count.
python3 -m pytest test_sfpu_domains.py -q --noconftest   # expect 126 passed
```

Per-op rows are keyed on `MathOperation` and read `_OP_DOMAIN_REGISTRY`, `_OP_SINGULARITIES`,
`_OP_EDGE_POINTS` and the sweep op lists in `test_sfpu_unary.py`. An op enrols in the edge sweep **by
being in the registry**, not by being listed in a test, so a new op appears in §4 automatically —
regenerate rather than editing rows by hand.

The `pytest --noconftest` above is needed because `conftest.py` imports `helpers/device.py`, which
imports `tt-exalens`; `tests/requirements.txt` pins `0.3.31` and **earlier** releases lack
`CallstackEntry` and `ElfFile`, so a drifted venv fails at collection with what looks like a broken
checkout. Host-side tests do not need the device at all. This bites in practice rather than in theory: a
system interpreter carrying an older `ttexalens` fails every suite, device and host alike, and the fix is
a venv built from `tests/requirements.txt` — plus `PYTHONNOUSERSITE=1` if `~/.local/lib` holds a
shadowing `numpy`.
