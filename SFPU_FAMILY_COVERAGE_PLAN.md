# SFPU Edge-Case Coverage — The Four Families The Sweeps Never Reach

**Companion to:** [SFPU_EDGE_CASE_COVERAGE.md](SFPU_EDGE_CASE_COVERAGE.md) (revision 16, the per-op audit)
and [SFPU_EDGE_CASE_EXPANSION_PLAN.md](SFPU_EDGE_CASE_EXPANSION_PLAN.md) (revision 17, the unary/cat-F plan)
**Issue:** [tenstorrent/tt-metal#49739 — [LLK] SFPU testing edge cases](https://github.com/tenstorrent/tt-metal/issues/49739)
**Written:** 2026-08-17, against `ldjurovic/sfpu_edge_cases_phase_3` @ `26c61ff80e9`
**Scope:** the four audit sections whose **entire** `Edge sweep` column is ⬜ —
§4.4 binary float/shift (43 ops), §4.5 binary integer (5 ops), §4.8 reduce (3 ops),
§4.9 FPU binary (3 ops). Wormhole B0 and Blackhole; Quasar out of scope.

**Why these four are one document and not four.** The existing expansion plan covers the *unary*
tail plus cat F, and everything in it is either blocked on a kernel owner or is a per-kernel build.
These four families are blocked on neither — and they share one cause. The unary and scalar suites
were rebuilt around three gates (`specials_safe`, `SPECIALS_READY_OPS`, `negative_zero_delivered`)
and one golden contract (model Dest, then model the pack path). **None of the four families below
imports any of it.** So this is not four separate coverage projects; it is one contract applied three
more times, plus one audit correction.

Every claim below marked *measured* was re-derived host-side against the tree at `26c61ff80e9`
today, with no device. The reproduce commands are in §6.

---

## 0. What is actually missing, in one table

| § | Family | Ops | What the audit says | What is *really* missing |
|---|---|---|---|---|
| **4.4** | Binary SFPU (float + shift) | 43 (22 float, 21 int-typed) | 11 registered, 5 with a driven pole | **Cat B does not exist in this suite at all.** 16 of the 22 float ops also have no cat-A/cat-D metadata, so nothing collects them |
| **4.5** | Binary integer | 5 | "none (WH/BH)" | **Mostly a naming artifact** — the kernels *are* covered under the `SfpuElw*` spelling. The real gap is that the Int32 comparisons are driven positive-only and tie-free |
| **4.8** | Reduce | 3 | registered domain, no edge sweep | Float specials in an **accumulating** op; the golden bypasses Dest and the pack path; Max/Min ignore the documented total order. (Cat C is already done — do not re-file it) |
| **4.9** | FPU binary | 3 | registered domain, no edge sweep | A **different datapath** (SrcA/SrcB, not Dest), so `specials_safe()` does not apply and has to be re-measured; and the fidelity model corrupts non-finites |

**Ordering.** Do §4.5 first — it is an hour of reading and it may *remove* work rather than add it.
Then §4.4, which is the largest coverage gain and where the shared golden contract gets written for
the second time. Then §4.8, which reuses that contract. Then §4.9, which needs its own hardware
measurement before any golden work is worth doing.

**One prerequisite is shared by 4.4, 4.8 and 4.9 and should be written once (§1).**

---

## 1. The shared prerequisite: a golden that models Dest and the pack path

Three of the four families need the same change, and `ScalarBinopGolden` is the finished template for
it — [golden_generators.py:5086-5101](tt_metal/tt-llk/tests/python_tests/helpers/golden_generators.py#L5086-L5101),
two lines of code and a paragraph of comment:

```python
result = cast_to_dest_dtype(result, format_dict[data_format]).flatten()
if dest_acc == DestAccumulation.No:
    result = convert_nan_to_inf(result)
```

**Why it is load-bearing and not hygiene.** `dest_acc` decides the Dest width. A 32-bit Dest holds a
NaN; a 16-bit one does not, and the packer substitutes an infinity **of the NaN's own sign** on the
way out (SFPSTORE: *"NaN is also converted to infinity"*). A golden that skips this asserts `NaN`
where hardware returns `±inf` — and `cast_to_dest_dtype` rather than `.to()` because torch's bfloat16
cast forces every NaN's sign bit to 1, which would then decide the substituted infinity's sign by
accident. Both halves of that were found the hard way (audit §5.7): one framework defect wearing four
disguises, silently wrong for 24 ops.

**Measured — where the contract is honoured today, and where it is not:**

| Golden | Models Dest width | Models pack path | Status |
|---|---|---|---|
| `UnarySFPUGolden` | ✅ (`__call__`, dst_format derivation) | ✅ `convert_nan_to_inf` | done |
| `ScalarBinopGolden` | ✅ | ✅ | done (revision 10) |
| `BinarySFPUGolden` | ⬜ **takes no `dest_acc` argument at all** | ⬜ | §2 |
| `UnarySFPUGolden`'s **reduce branch** | ⬜ returns before the derivation | ⬜ | §4 |
| `EltwiseBinaryGolden` (FPU) | ⬜ | ⬜ | §5 |

The reduce case is the subtle one: `ReduceColumn`/`ReduceRow` *do* go through `UnarySFPUGolden`, but
`__call__` returns at
[golden_generators.py:2457-2458](tt_metal/tt-llk/tests/python_tests/helpers/golden_generators.py#L2457-L2458)
— **before** the `dst_format` derivation, before `cast_to_dest_dtype`, before `convert_nan_to_inf`. So
"the unary golden already models this" is true and does not help.

**The second shared item: the documented total order.** Revision 12 established that the SFPU
implements a *total* order over FP32 — `-NaN < -Inf < … < -0 < +0 < … < +Inf < +NaN` — specified on
`SFPGT`, `SFPLE` and `SFPSWAP` via `SignMagIsSmaller()`, on **both** arches. `sfpu_total_order_key`,
`sfpu_min`, `sfpu_max`, `sfpu_clamp` and `sfpu_relu_max` model it
([golden_generators.py:247-311](tt_metal/tt-llk/tests/python_tests/helpers/golden_generators.py#L247-L311))
and seven unary ops were enrolled as plain passes rather than xfails because of it.

**Measured: the binary and reduce goldens still model IEEE's unordered comparisons instead.** See §2.2
and §4.2. This is the single highest-value item in the document, because it is a fix that already
exists and is simply not wired in — and because recording these as xfails instead would write
permanent, plausible-looking lies about documented hardware. That mistake was made once already
(audit revision 11) and caught only because someone read the ISA before filing.

**Acceptance for §1:** each of the three goldens takes `dest_acc`, routes through
`cast_to_dest_dtype` + `convert_nan_to_inf`, and the comparison-shaped goldens route through
`sfpu_total_order_key`. Verified host-side per §6 *before* any hardware time is spent — that check is
what caught `_sin`/`_cos` raising on a non-finite input before a single variant was compiled.

---

## 2. §4.4 — Binary SFPU: cat B, and the 16 ops nothing collects

**43 rows. 22 are float-typed; 21 of those reach the shared `sfpu_binary()` driver** (`SfpuAddTopRow`
builds its own stimuli, golden and `TestConfig`, so it is not on this path). The other 21 rows are
int-typed and belong to §3's discussion, not here.

### 2.1 Two distinct gaps, and only one of them is cat B

**Gap A — nothing collects 16 of the 21.** `test_sfpu_binary_edges`
([test_sfpu_binary.py:1279-1287](tt_metal/tt-llk/tests/python_tests/test_sfpu_binary.py#L1279-L1287))
derives its op list as `ops_with_singularity() & _CLASSIFIED_STIMULI_OPS - _INT_DRIVEN_BINARY_OPS`,
which resolves to exactly five ops: `SfpuElwdiv`, `SfpuXlogy`, `SfpuElwpow`, `SfpuBinaryFmod`,
`SfpuBinaryRemainder`. That derivation is correct and should stay — an op joins by gaining registry
metadata, not by being added to a list. **So the fix is to register the metadata that is missing**, and
several of these ops have a real edge that no `_OP_SINGULARITIES` / `_OP_EDGE_POINTS` entry expresses:

| Op | Unregistered edge | Category |
|---|---|---|
| `SfpuAtan2` | the branch cut: `B = 0` with `A < 0` → `±π`, and the sign of the zero picks the branch | cat A, operand B |
| `SfpuBinaryMax` / `SfpuBinaryMin` | the tie `a == b` — the only input that distinguishes a correct comparator from an inverted one | cat D |
| `SfpuElwEq` / `Ne` / `Lt` / `Gt` / `Le` / `Ge` | the tie again. `Le`/`Ge` differ from `Lt`/`Gt` **only** at `a == b` | cat D |
| `SfpuMask` | `mask == 0` exactly, and `mask == -0.0` | cat D, operand B |
| `SfpuIsclose` | the tolerance boundary `|a-b| == atol + rtol·|b|`, either side | cat D |
| `SfpuLogsigmoid` | `x = 0` (the `log(2)` knee) and the large-negative asymptote | cat D |

The comparison ties are partly covered today by crafted paired stimuli
(`_eq_ne_stimuli_specs`, `_comparison_stimuli_specs`), which is why this is a metadata gap rather than
a coverage hole for those six — but the crafted specs live in the test file and the edge sweep cannot
see them, so `Le`/`Ge` vs `Lt`/`Gt` is asserted in one place and invisible in the other.

**Gap B — cat B does not exist in this suite.** `test_sfpu_binary.py` imports `integer_specials` and
nothing else from `sfpu_domains`'s specials machinery: no `FLOAT_SPECIALS`, no `specials_safe`, no
`SPECIALS_READY_OPS`, no `negative_zero_delivered`. Not one `±inf`, `NaN` or `-0.0` is deliberately
driven through any of the 43 binary ops. Given that 47 of the 97 unary ops are smooth everywhere and
cat B is their *entire* edge story, the same is true of most of these 21 — `add`, `sub`, `mul`, `max`,
`min` and the six comparisons have no pole and no knee beyond the tie.

**The stimulus machinery already exists and needs no new code.**
`edge_pair_values(op, in_fmt, out_fmt, specials=True, dest_acc=…)`
([sfpu_domains.py:2660-2705](tt_metal/tt-llk/tests/python_tests/helpers/sfpu_domains.py#L2660-L2705))
already takes the Cartesian product of both operands' edge values *including* specials, and
`_build_edge_pair_src` already turns a pair list into the two-tile override the driver wants. What is
missing is the gate that says `specials=True`, and the golden work behind it.

### 2.2 Measured: the goldens do not raise, but seven of them are wrong

**All 21 float goldens return a value at every one of the 25 `(special, special)` pairs — none
raises.** That is a genuinely different starting position from the unary tranche, where `math.sin`,
`math.cos`, `math.acos`, `math.asin` and `math.tan` all raised `ValueError` on a non-finite input.
There is no `math.*` audit to do here.

**But the six comparison goldens and `_min` model IEEE's unordered comparison rather than the
documented total order.** Measured against `sfpu_total_order_key` over
`(NaN,1) (1,NaN) (NaN,NaN) (NaN,+inf) (-inf,NaN)`:

| Golden | What it returns | What the total order gives | |
|---|---|---|---|
| `_gt` | `0 0 0 0 0` | `1 0 0 1 0` | ⚠️ diverges |
| `_ge` | `0 0 0 0 0` | `1 0 1 1 0` | ⚠️ diverges |
| `_lt` | `0 0 0 0 0` | `0 1 0 0 1` | ⚠️ diverges |
| `_le` | `0 0 0 0 0` | `0 1 1 0 1` | ⚠️ diverges |
| `_eq` | `0 0 0 0 0` | `0 0 1 0 0` | ⚠️ diverges |
| `_ne` | `1 1 1 1 1` | `1 1 0 1 1` | ⚠️ diverges |
| `_min` | `nan nan nan nan nan` | `1 1 nan +inf -inf` | ⚠️ diverges |
| `_max` | `nan nan nan nan nan` | `nan nan nan nan nan` | agrees **on a positive NaN only** |

`_max` is the case to read closely, because it is the trap. It agrees on `+NaN` by accident —
`torch.maximum` propagates NaN and `+NaN` also happens to be the total-order maximum — and diverges
the moment the sign flips: **measured, `sfpu_max(-NaN, 1.0) = 1.0` where `torch.maximum` gives `NaN`**.
An enrolment that probes only `+NaN` would record `_max` as correct and leave a wrong golden in the
tree with a passing test over it.

This is the same finding revision 11 made on the unary side, on the same nine-op shape, and it
resolved *in favour of the ISA*: the goldens were the wrong party. Expect the same here. The seven ops
above become plain passes, not xfails.

**Two more measured golden facts, both needing a decision rather than a fix:**

- `_pow(1.0, NaN)` returns `1.0` (torch and IEEE 754, which define `1**anything = 1`). The kernel
  evaluates `exp(b·ln a)`, i.e. `exp(NaN·0)`, which cannot give 1. `0**0` already diverges for exactly
  this reason and is xfailed. **Measure before writing either answer** — the audit's own rule.
- `_xlogy`'s docstring
  ([golden_generators.py:3798-3803](tt_metal/tt-llk/tests/python_tests/helpers/golden_generators.py#L3798-L3803))
  states outright that *"non-finite edge cases across formats/dest_acc are not consistently modelled,
  so xlogy is exercised with strictly-positive stimuli"*. That is a recorded promise this work
  breaks; the comment has to be retired with evidence, not deleted.

### 2.3 The cells, measured

The binary edge sweep's format axis is `{Float16_b, Float32}` → 4 pairs × 2 `dest_acc` = 8 cells.
Measured against the existing gates:

| Gate | Cells | Which |
|---|---|---|
| `specials_safe()` | **6 of 8** | `F16_b→F16_b/No`, `F16_b→F32/No`, `F32→F16_b/No`, `F32→F16_b/Yes`, `F32→F32/No`, `F32→F32/Yes` |
| `negative_zero_delivered()` | **2 of 6** | `F32→F16_b/Yes`, `F32→F32/Yes` (unpack-to-dest only) |
| `nan_survives_to_l1()` | **1 of 6** | `F32→F32/Yes` — the only cell where a NaN reaches the comparator still a NaN |

`sfpu_binary()` uses `unpack_to_dest=formats.input_format.is_32_bit()`
([test_sfpu_binary.py:647](tt_metal/tt-llk/tests/python_tests/test_sfpu_binary.py#L647)), the same
routing the unary driver uses, so **`specials_safe()` and `negative_zero_delivered()` apply unchanged
here** — no re-measurement needed. That is not true for the FPU family; see §5.

**One arch trap specific to this driver.**
[test_sfpu_binary.py:609-613](tt_metal/tt-llk/tests/python_tests/test_sfpu_binary.py#L609-L613)
silently forces `dest_acc = DestAccumulation.Yes` on Blackhole for `Float16`/`Float32` inputs
(*"ONLY Blackhole needs this for some reason"*). So on Blackhole the `dest_acc=No` row does not exist
for the cells that matter, and a gate computed from the *parameter* rather than from the *effective*
value will disagree with what ran. Compute the gate after that reassignment, or move the reassignment
above it.

### 2.4 Implementation steps

1. **Golden contract (§1).** Give `BinarySFPUGolden.__call__` a `dest_acc` parameter, thread it from
   `sfpu_binary()`, and route the result through `cast_to_dest_dtype` + `convert_nan_to_inf`. The
   call site is
   [test_sfpu_binary.py:592-606](tt_metal/tt-llk/tests/python_tests/test_sfpu_binary.py#L592-L606) and
   currently passes `golden_format` only. Note this is a **shared** golden — `EltwiseBinaryGolden` is
   its base class and the FPU family uses it too — so §5's work must not fork it.
2. **Total order.** Rewrite `_lt/_gt/_le/_ge/_eq/_ne/_min/_max` through `sfpu_total_order_key` /
   `sfpu_min` / `sfpu_max`. Confirm the op→instruction mapping against the kernels *first*, the way
   revision 12 did: `calculate_binary_comp_fp32_*`
   ([ckernel_sfpu_binary_comp.h](tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_comp.h))
   is a two-vector compare, and `binary_max_min` is programmed via `SFPSWAP` — both routed through
   `SignMagIsSmaller()`. Then verify the rewritten goldens are **bit-identical** to the ones they
   replace over the existing finite stimuli (revision 12 did this over 8000 inputs); only the NaN and
   signed-zero answers may move.
3. **Register the missing cat-A/cat-D metadata** from §2.1's table, per operand, using
   `_OP_SINGULARITIES` for `atan2`'s branch cut and `_OP_OPERAND_EDGE_POINTS` for the per-operand
   knees (the mechanism ternary operand-C already uses). The 16 uncollected ops join
   `test_sfpu_binary_edges` automatically as they gain entries — that is what the derived op list
   buys.
4. **Add the gate**, mirroring the unary and scalar sweeps exactly:
   ```python
   specials = mathop in BINARY_SPECIALS_READY_OPS and specials_safe(
       formats.input_format, formats.output_format, effective_dest_acc)
   specials = specials_after_nan_sign_gate(mathop, ..., on_wormhole=...)
   ```
   `BINARY_SPECIALS_READY_OPS` is a new dict in `sfpu_domains.py` keyed by op with a **reason string**,
   populated one tranche at a time. Do not reuse `SPECIALS_READY_OPS`: the two families' membership is
   independent and a shared dict would enrol a binary op because its unary namesake was ready.
5. **Extend the edge-class taxonomy.** `_classify_edge_pair`
   ([test_sfpu_binary.py:1207](tt_metal/tt-llk/tests/python_tests/test_sfpu_binary.py#L1207)) classifies
   by *what the golden says the answer is* — a good design that breaks here, because
   `_EDGE_CLASS_NAN` currently means "`x % 0`" and a NaN *input* would land in the same bucket. The
   causes are unrelated and one xfail must not cover both. Add a class for a non-finite **input**
   (`specials_in`), keeping the existing four as they are, so each cause fails on its own evidence.
   That is the same reasoning the four existing classes were split on.
6. **Drive the whole category at once, not op by op** (expansion plan §2.3). The machinery is
   trustworthy now, so a single sweep over all 21 ops × 6 cells × the classes is strictly better than
   21 commits: a cause that shows up in seven ops at once is invisible one op at a time — which is
   exactly how §2.2's total-order finding was found.
7. **Sign-of-a-generated-NaN.** The binary ops that *build* a NaN through `SFPMAD` rather than
   forwarding one need their own `GENERATED_NAN_SIGN_OPS` measurement on Wormhole, where the sign is
   explicitly unspecified. Do **not** extend the existing frozenset by shape argument — that set is
   documented as membership-by-observed-disagreement, and `GeluTanh`/`Xielu` are on record as having
   the right shape and the right sign anyway.

**Acceptance.** Every enrolled op is in `BINARY_SPECIALS_READY_OPS` with a reason; the sweep runs it
on all 6 safe cells with no unexplained failure; each surviving divergence is either an
ISA-cross-checked non-strict xfail with a per-class reason or a fixed golden; the audit's §4.4 table
regenerates with a populated `Cat B` column; host-side guards in `test_sfpu_domains.py` pin the new
gate and the new op set.

**Size.** The largest item here. Realistic shape: one commit for §1's contract, one for the total
order (with the bit-identity check), one for the metadata, one for the gate plus the first tranche,
then tranches. Roughly 21 ops × 6 cells × ≤5 classes as an upper bound on new variants, most of which
skip because the op has no pair in that class.

---

## 3. §4.5 — Binary integer: read this before building anything

**The audit's "none (WH/BH)" is right about the enum members and wrong about the kernels.** Measured
from the code:

- **The WH/BH `BinaryOp` enum has no `GT_INT` / `LT_INT` / `LE_INT` / `GE_INT`.** Those four names
  exist only in the Quasar enum. WH/BH stop at `FMOD_INT32 = 42`
  ([ckernel_defs.h:295](tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/ckernel_defs.h#L295)).
- **But `sfpu_operations.h` already dispatches the *float-spelled* comparisons to the integer
  kernel.** At
  [sfpu_operations.h:1897-1911](tt_metal/tt-llk/tests/helpers/include/sfpu_operations.h#L1897-L1911),
  `BinaryOp::LT/GT/LE/GE/EQ/NE` route to `calculate_binary_comp_int32` whenever
  `MATH_FORMAT == Int32`.
- **And the suite already drives that path.** `test_sfpu_binary_int`
  ([test_sfpu_binary.py:933-961](tt_metal/tt-llk/tests/python_tests/test_sfpu_binary.py#L933-L961))
  parametrizes `SfpuElwLt/Gt/Le/Ge` at `DataFormat.Int32`, `dest_acc=Yes`, on both WH and BH.
- **`SfpuElwmulInt` is the same story.** Its `cpp_enum_value` is `"MUL"`; on Quasar that reaches
  `_mul_int32_`. On WH/BH the same kernel is reached as `SfpuMulInt32` (`MUL_INT32`), driven by
  `test_sfpu_binary_int_uniform`.

**So all five kernels are covered on WH/BH; the five `MathOperation` members are Quasar spellings of
them.** This is precisely the alias hazard the audit already warns about in §2.4 for
`SfpuWhere`/`TTNNWhere` and `LogicalNot`/`LogicalNotUnary` — *"an op can be driven under an alias… any
tooling over this audit has to resolve aliases"*. §4.5 is a fifth and sixth instance of it that the
inventory did not resolve, because the alias here is across a `MathOpType` boundary
(`SFPU_BINARY_INT` vs `SFPU_BINARY`) rather than within one enum.

### 3.1 What to do instead of five harnesses

1. **Correct the audit.** §4.5's five rows become "kernel covered on WH/BH under the `SfpuElw*`
   spelling at `Int32`; the enum member is Quasar-only". §2.4's alias warning gains the
   `MathOpType`-crossing case. §1's *"Binary integer / ternary / scalar / reduce / FPU-binary ops |
   5 / 5 / 5 / 3 / 3"* row keeps its count but stops implying five uncovered ops.
2. **Pin it with a guard, so it cannot silently stop being true.** A host-side test in
   `test_sfpu_domains.py` asserting the alias pairs — `(SfpuGtInt, SfpuElwGt)` and the rest — resolve
   to the same kernel entry point, and that the `SFPU_BINARY_INT` members are unreachable from the
   WH/BH `BinaryOp` enum. Prose in an audit is not checked by anything; this is.
3. **Then build the coverage that is actually missing**, below. It is real, and it is on the ops that
   *are* driven.

### 3.2 The real gap: the Int32 comparisons are driven positive-only and tie-free

**Measured, and it is finding #1 of the original audit surviving on the integer axis.** The default
integer stimuli spec is
[generator.py:251-256](tt_metal/tt-llk/tests/python_tests/helpers/stimuli_generator/generator.py#L251-L256):

```python
v1_type_max = torch.iinfo(dtype).max // 2
return StimuliSpec.uniform(low=0.0, high=float(v1_type_max - 1))
```

— `uniform(0, 2**30 - 1)`. **Positive-only, and only the lower half of the positive range.** A draw of
1024 Int32 elements produced `min 1350247, max 1071860489`, **0 negatives, 0 ties**. So today's
`test_sfpu_binary_int` coverage of `SfpuElwLt/Gt/Le/Ge` at Int32:

- never crosses zero, on a kernel whose whole method is to normalise via `a - b` and fold the sign
  bit — the one place where operand sign is the entire mechanism;
- never hits `a == b`, so **`Le`/`Ge` are indistinguishable from `Lt`/`Gt`** under it. A comparator
  with the tie inverted passes this test.

Both are fixable with machinery that already exists in the file: `_eq_ne_stimuli_specs()` was written
to make ~50% of positions compare equal for exactly this reason, and `test_sfpu_binary_rsub_int32`
shows the `twos_complement=True` route for negative Int32 operands.

**Three additions, in order of value:**

1. **The tie.** Drive the Int32 ordered comparisons with paired stimuli, reusing
   `_eq_ne_stimuli_specs()`. Cheapest and highest value: it is the input that distinguishes four ops
   from each other.
2. **Cat C.** Add the four ordered Int32 comparisons to `_INT_EXTREME_OPS`
   ([test_sfpu_binary.py:1493-1499](tt_metal/tt-llk/tests/python_tests/test_sfpu_binary.py#L1493-L1499)),
   which today holds only the three bitwise ops and `eq_int`/`ne_int`. `test_sfpu_binary_int_extremes`
   already walks `{INT32_MIN+1, -1, 0, 1, INT32_MAX}²`. This is where the `a - b` normalisation either
   overflows or is proven not to: `INT32_MAX - (INT32_MIN+1)` does not fit in int32, and the kernel's
   sign-fold has to handle it. The comparisons are **exact** on the full range (no reciprocal, no
   product), so §2.6's "narrower valid range by kernel design" exclusion does not apply to them — this
   is a coverage gap, not a documented limitation.
3. **Negatives.** Measure whether the Int32 comparisons need `twos_complement=True` the way
   `SfpuRsubInt32` does, or whether the sign-magnitude Dst round-trip constrains them the way it
   constrains the shifts (audit §2.3: driving negatives made every `RightShift` variant disagree, and
   that identified *delivery* rather than arithmetic). **Measure, do not assume** — that distinction is
   exactly what stopped a stimulus limitation being filed as a kernel divergence.

**Acceptance.** The alias claim is pinned by a host-side test; the four Int32 ordered comparisons are
driven at the tie, at the int32 extremes, and across zero with the encoding question resolved by
measurement; §4.5's rows say what is actually true.

**Size.** Small. Steps 1 and 2 are a parametrize change plus a list edit each.

---

## 4. §4.8 — Reduce: specials in an accumulating op

Three ops, and the smallest family here by op count and the least like the others in kind. A
reduction has an **identity** and an **absorbing** element per pool, and one poisoned lane changes the
whole output — so a special is not one probe among 4096, it is the answer.

### 4.1 Current state, measured

- All three are in `_OP_DOMAIN_REGISTRY` with a plain `uniform(-1, 1)`, **no singularity and no knee**.
  Verified: `edge_spec(ReduceColumn, Float32, Float32)` returns `None`. So no edge sweep can reach
  them today even if one were pointed at the family — the registry has nothing to give it.
- `test_sfpu_reduce` drives only `ReduceColumn` and `ReduceRow`; `get_supported_reduce_axioms`
  ([test_sfpu_reduce.py:71](tt_metal/tt-llk/tests/python_tests/test_sfpu_reduce.py#L71)) never returns
  `ReduceScalar`, which is driven from `test_reduce.py` / `test_mul_reduce_scalar.py` instead. Any
  edge work here has to decide whether it covers two ops or three; the scalar pool is a different
  driver.
- **Cat C is already done and must not be re-filed.** `test_int32_reduce_extreme`
  ([test_sfpu_reduce.py:585](tt_metal/tt-llk/tests/python_tests/test_sfpu_reduce.py#L585)) injects
  `INT32_MIN`-class values with a non-strict arch xfail. It is also the right *shape* to copy: an
  injected-value parametrization over a base range, with the arch gate as a runtime marker.

### 4.2 Two measured blockers

1. **The reduce golden models neither Dest nor the pack path** — §1's early return at
   [golden_generators.py:2457-2458](tt_metal/tt-llk/tests/python_tests/helpers/golden_generators.py#L2457-L2458).
2. **Max/Min ignore the total order.** Measured over a 32×32 column containing one `+inf`, one
   `-inf` and one `NaN`, `Float32`/`dest_acc=Yes`:

   | Pool | Golden gives | Expected from the comparator |
   |---|---|---|
   | `Max` | `nan` | `nan` — agrees for `+NaN` only, as in §2.2 |
   | `Min` | `nan` | the finite minimum: `+NaN` is the total-order *maximum* |
   | `Sum` | `nan` | `nan` (`+inf + -inf`) — agrees, IEEE |
   | `Average` | `nan` | `nan` — agrees |

   `_reduce_columns`/`_reduce_rows` fold with `torch.amin`/`amax`. The kernel's Max/Min path is
   `SFPSWAP(VEC_MIN_MAX)` — sign-magnitude, `SignMagIsSmaller()`, the documented total order — and
   `use_int32_twos_complement`'s own docstring
   ([test_sfpu_reduce.py:89-113](tt_metal/tt-llk/tests/python_tests/test_sfpu_reduce.py#L89-L113))
   already says so for the integer path. So Max/Min need `sfpu_min`/`sfpu_max` folding, and Sum/Average
   keep torch. **The two pools split**, which is worth saying explicitly because a single "fix the
   reduce golden for specials" commit would get one of them wrong.

### 4.3 The edge classes worth driving

These are reduce-specific and have no analogue in the unary or binary sweeps, which is the argument
for building this at all rather than treating three ops as a rounding error:

| Class | Stimulus | What it asserts |
|---|---|---|
| one `+inf` lane | rest finite | absorption for Max/Sum; transparency for Min |
| one `-inf` lane | rest finite | the mirror; and Min's absorption |
| both `±inf` | one lane each | `Sum` → `NaN`; `Max`/`Min` still finite-free answers |
| one `NaN` lane | rest finite | propagation vs the total order — §4.2's split |
| all lanes `±inf` | whole column | the degenerate reduce, where the pool identity is the only other operand |
| `±0` only | whole column | `Sum(-0, -0) = -0` under IEEE, and SFPMAD flushes negative zero to `+0` on Wormhole while Blackhole preserves it — **the arch-gated class**, exactly like the binary suite's `_EDGE_CLASS_NEGATIVE_ZERO` |
| special **vs the pad value** | sub-tile extent | `get_reduce_pad_value` ([test_sfpu_reduce.py:123](tt_metal/tt-llk/tests/python_tests/test_sfpu_reduce.py#L123)) fills the unreduced rows with the pool identity. A real `±inf` datum against an identity pad is a direct collision, and the sub-tile column reduce is the only place it happens |

The last one is the class that only exists here, and the one most likely to find something.

### 4.4 Implementation steps

1. Fix the golden per §1 and §4.2 — Dest + pack path for all pools, total-order folding for Max/Min
   only. Verify host-side first.
2. Register the reduce family's edges. The pool is a *parameter*, not part of the op, so
   `_OP_EDGE_POINTS[ReduceColumn]` cannot express "the identity of this pool" — the class table above
   is pool-keyed and belongs in the test file as a parametrization over injected values, following
   `test_int32_reduce_extreme`'s shape rather than the `edge_spec()` route. Say so in a comment; the
   next reader will otherwise try to register it and find the registry cannot hold it.
3. Gate with `specials_safe()` — the reduce driver uses `unpack_to_dest=True`
   ([test_sfpu_reduce.py:437](tt_metal/tt-llk/tests/python_tests/test_sfpu_reduce.py#L437)), so the
   measured matrix applies — plus `negative_zero_delivered()` for the `±0` class.
4. Arch-gate the `±0` class to Wormhole, non-strict, so Blackhole *asserts* the preserved sign rather
   than tolerating a divergence. The binary suite's `_WORMHOLE_ONLY_EDGE_CLASSES` is the precedent and
   the reason it is worth doing: 16 tolerated cells became 16 assertions.
5. Decide `ReduceScalar` explicitly — in scope with its own driver, or declared out with a reason.
   Silence here is how it ended up outside `test_sfpu_reduce` unnoticed.

**Acceptance.** Each class runs on the safe cells for each pool; the Max/Min total-order split is
asserted rather than assumed; the `±0` class XFAILs on Wormhole and PASSes on Blackhole (a non-strict
xfail that always XPASSes is a gate that has gone quiet — audit §5.11/§5.12); `ReduceScalar`'s status
is written down.

**Size.** Medium. Three ops, but four pools × seven classes, and the golden split is genuine work.

---

## 5. §4.9 — FPU binary: a different datapath, so measure before modelling

`Elwadd`, `Elwmul`, `Elwsub`. These run on the **FPU**, not the SFPU, and that single fact is what
makes this family last rather than first.

### 5.1 `specials_safe()` does not apply here, and reusing it would be a silent error

The measured 7-of-50 matrix was established by driving the five `isinf`/`isnan` **SFPU** predicates
over the format matrix — i.e. it measures the unpack→Dest and datacopy paths that the SFPU reads its
operands from. The FPU reads `SrcA`/`SrcB`, which are a different register file with a different
width and a different conversion on the way in (`SrcFormatModel`'s TF32-ish 19-bit form). Whether a
`NaN` or a `-0.0` survives *that* conversion is an unmeasured claim, and the audit's own lesson
applies: *"an unexercised arch is not a documentation gap, it is an unmeasured claim."* The same holds
for an unexercised datapath.

**So step one is a measurement, not a golden.** Build the FPU equivalent of the predicate instrument:
drive `Elwadd(x, 0.0)` and `Elwmul(x, 1.0)` over `FLOAT_SPECIALS` across the format matrix and both
`dest_acc`, read the result back raw, and record which cells preserve `+inf`, `-inf`, `NaN` and
`-0.0`. Land it as `fpu_specials_safe()` in `sfpu_domains.py` with the measurement written up beside
it and pinned cell-by-cell in `test_sfpu_domains.py`, the way `specials_safe()` is. **Do not
arch-key it before the measurement says to**, and do not merge it into `specials_safe()` — two
datapaths that happen to agree today should still be two functions, because the next format added
will not agree in both.

Three sub-questions that measurement should answer while the instrument is up:

- Does `SrcA`'s conversion preserve a `NaN` at all, or does it arrive as a large finite?
- Is the broadcast path different? `BroadcastType.Scalar/Row/Column` replicates one operand across a
  tile through `BroadcastGolden`, which is a third delivery route.
- Does `Transpose.Yes` change the answer? It moves the datum through the unpack transposer first.

### 5.2 Measured: the fidelity model corrupts non-finites, and it splits the family

`SrcFormatModel._fp32_to_tf32` (and its `_fp16b_`, `_bfp8b_` siblings) unconditionally ORs in the
implied leading mantissa 1 and **never special-cases `exp == 0xFF`**, so a non-finite operand comes out
of `_apply_fidelity_masking` as an ordinary large number. Measured through
`EltwiseBinaryGolden._compute_eltwise` on `Float32`, operands
`a = [+inf, -inf, NaN, 0, 2]`, `b = [2, 2, 2, +inf, 3]`:

| Op | Fidelity | Golden result | Correct |
|---|---|---|---|
| `Elwmul` | LoFi | `[inf, -inf, **inf**, inf, 6]` | `NaN·2` must be `NaN`, not `+inf` |
| `Elwmul` | HiFi2 | `[**nan, nan**, nan, nan, 6]` | `inf·2 = +inf`, `-inf·2 = -inf` — both destroyed |
| `Elwmul` | HiFi4 | `[**nan, nan**, nan, nan, 6]` | same |
| `Elwadd` | LoFi / HiFi2 / HiFi4 | `[inf, -inf, nan, inf, 5]` | ✅ correct at every fidelity |
| `Elwsub` | any | correct at every fidelity | ✅ |

The cause is visible in the code: `_compute_eltwise`
([golden_generators.py:3373-3385](tt_metal/tt-llk/tests/python_tests/helpers/golden_generators.py#L3373-L3385))
applies fidelity masking **only for `Elwmul`**, and accumulates `result += phase_result` across
phases — so at HiFi2+ a corrupted phase turns into `inf + (-inf)` and everything becomes `NaN`.

**This splits the family cleanly, which is the useful part:** `Elwadd` and `Elwsub` can be enrolled as
soon as §5.1's matrix exists, with no golden change at all. `Elwmul` needs `SrcFormatModel` to pass
non-finites through unmodified (and a decision about what the *hardware* does across fidelity phases,
which is an ISA read, not a guess).

### 5.3 The edge classes only the FPU has

| Class | Why it is new |
|---|---|
| exact cancellation `a + (-a)` | The result's zero sign is an FPU question. SFPMAD's flush-to-positive-zero rule is an **SFPU** rule and does not transfer — read the FPU's own ISA page before predicting either answer |
| `inf + (-inf)`, `inf - inf` | The only ops in the whole audit where two infinities meet in one arithmetic op |
| overflow to `inf` at the format ceiling | **§2.7 of the audit, still untouched everywhere.** The FPU family is the natural place to close it: `clip_to_format()` exists to keep probes *inside* the range and is not on this path, so the mechanism does not fight you here. Measured: the golden already gives `3.4e38 + 3.4e38 = inf` correctly |
| denormals | `_apply_ftz` ([golden_generators.py:95](tt_metal/tt-llk/tests/python_tests/helpers/golden_generators.py#L95)) models flush-to-zero. Also §2.7, also untouched, also reachable here |

Closing §2.7 for three ops is worth more than it looks: it is currently listed as *"untouched since
the original audit"* for the entire op set.

### 5.4 Implementation steps

1. Measure `fpu_specials_safe()` (§5.1) and pin it. **Blocked on hardware** — one Wormhole and one
   Blackhole board.
2. Enrol `Elwadd` and `Elwsub` over the measured matrix, with the §1 golden contract applied to
   `EltwiseBinaryGolden`. Careful: `BinarySFPUGolden` **subclasses** it, so this change touches §2's
   family too — do it in whichever order lands first and re-run both suites.
3. Fix `SrcFormatModel`'s non-finite handling, then enrol `Elwmul` per fidelity. Its fidelity axis is
   4 values wide, so this is the largest cell count in the family.
4. Add the cancellation / two-infinity / overflow / denormal classes, and close audit §2.7 for these
   three ops with the measurement written up.
5. Decide whether the broadcast and transpose axes carry specials, from step 1's answer. If they do
   not, record *that* — an excluded axis with a measured reason is coverage information; an
   unmentioned one is a hole.

**Acceptance.** `fpu_specials_safe()` exists, is measured on both arches, and is pinned per cell;
`Elwadd`/`Elwsub` enrolled with no unexplained failure; `Elwmul` enrolled per fidelity or held out
with a reason naming the ISA question; §2.7 marked closed for the FPU family and still open elsewhere.

**Size.** Medium, and the only item here **blocked on hardware access** before design can finish.

---

## 6. How to verify, before spending any hardware time

Everything in §1–§5's golden analysis was produced host-side, with no device, in under a minute. Repeat
it before and after each change. Note `helpers.device` currently fails to import against the installed
`ttexalens` (the plan pins `tt-exalens==0.3.29`), so **import the goldens directly** rather than
importing a test module:

```bash
cd tt_metal/tt-llk/tests/python_tests

# (a) Binary goldens over the full 5x5 specials product -- does any raise, and what does it answer?
python3 -c "
import sys, math; sys.path.insert(0,'.')
import torch
from helpers.golden_generators import BinarySFPUGolden, sfpu_total_order_key, sfpu_min, sfpu_max
from helpers.llk_params import MathOperation as M
g = BinarySFPUGolden(); NAN=float('nan'); INF=float('inf')
SP=[INF,-INF,NAN,0.0,-0.0]
for op in [M.SfpuElwGt, M.SfpuBinaryMin, M.SfpuBinaryMax, M.SfpuElwpow]:
    fn=g.ops[op]
    print(op.name, [float(fn(torch.tensor(a),torch.tensor(b))) for a in SP for b in SP])
print('sfpu_max(-nan,1) =', sfpu_max(-NAN,1.0), 'vs torch', float(torch.maximum(torch.tensor(-NAN),torch.tensor(1.0))))
"

# (b) The FPU fidelity path -- does masking survive a non-finite?
python3 -c "
import sys; sys.path.insert(0,'.')
import torch
from helpers.golden_generators import EltwiseBinaryGolden
from helpers.llk_params import MathOperation as M, DataFormat as F, MathFidelity as MF
g=EltwiseBinaryGolden()
a=torch.tensor([float('inf'),float('-inf'),float('nan'),0.0,2.0]); b=torch.tensor([2.,2.,2.,float('inf'),3.])
for fid in [MF.LoFi, MF.HiFi2, MF.HiFi4]:
    for op in [M.Elwmul, M.Elwadd]:
        print(op.name, fid.name, [float(x) for x in g._compute_eltwise(op,a.clone(),b.clone(),F.Float32,fid,keep_float32=True)])
"

# (c) The cells each family can reach
python3 -c "
import sys; sys.path.insert(0,'.')
from helpers.sfpu_domains import specials_safe, negative_zero_delivered, nan_survives_to_l1
from helpers.llk_params import DataFormat as F, DestAccumulation as D
for a in (F.Float16_b,F.Float32):
  for b in (F.Float16_b,F.Float32):
    for da in (D.No,D.Yes):
      if specials_safe(a,b,da):
        print(f'{a.name:10s}->{b.name:10s} dest_acc={da.name:3s} -0={negative_zero_delivered(a,da)} nan_to_L1={nan_survives_to_l1(a,b,da)}')
"
# expect: 6 cells; -0 on the two dest_acc=Yes; nan_to_L1 on Float32->Float32/Yes only
```

Then the existing host-side gate suite, which must stay green throughout:

```bash
python3 -m pytest test_sfpu_domains.py -q     # 111 tests today; add to it, do not replace
```

**On hardware:** never call `pytest` directly — use the repo's runner, which serialises silicon
access, and go through the two-phase compile-producer / compile-consumer flow that CI uses.
`_classify_edge_pair` already carries a scar from this: it instantiates `BinarySFPUGolden` directly
because the harness swaps in a `DummyGoldenGenerator` during `--compile-producer`, and any new
stimulus-build-time golden call needs the same treatment or the whole sweep is unrunnable under CI's
flow while working fine when pytest is invoked by hand.

---

## 7. Traps, all of them earned

1. **Do not enrol an op on a guess.** A reason string written to make a variant green becomes a
   permanent, plausible-looking claim about the hardware, and nobody re-derives one once it is
   written.
2. **Measure, then read the ISA, then ask a human.** The unary side wrote up a nine-op kernel
   divergence with a measured table; the ISA said the *goldens* were wrong; seven ops were enrolled
   as plain passes instead of xfailed forever. §2.2 and §4.2 are the same shape. Skipping the middle
   step here would write a fresh set of permanent lies.
3. **Fixing a golden is not a reason to enrol an op.** `I1` is the case to read: its golden was wrong
   and is fixed, and it stays out because its *kernel* saturates. Keeping those two decisions apart is
   what stops a kernel divergence being laundered into a golden that agrees with it.
4. **Probe both NaN signs.** §2.2's `_max` agrees on `+NaN` and diverges on `-NaN`. A one-sided probe
   certifies a broken golden.
5. **Run both `dest_acc` values and read the `No` row.** Every golden defect found so far has lived
   there — it is the only path where a NaN becomes a *signed* infinity and a wrong sign is visible at
   all. At `dest_acc=Yes` the comparator's both-NaN clause accepts anything.
6. **Diff the whole op set against a baseline, not just the ops you touched.** §1's changes are to
   *shared* goldens — `EltwiseBinaryGolden` is `BinarySFPUGolden`'s base class, and `UnarySFPUGolden`
   serves the reduce family. A change for one family lands on the others.
7. **A non-strict xfail that always XPASSes is not coverage.** Two arch gates in this tree XPASS on
   Wormhole and assert nothing on either arch (audit §5.11, §5.12). Any gate added here needs a cell
   where it actually fires, or it should not be added.
8. **Count cells on one axis.** A unary cell is `(op, input, output, dest_acc)`; a binary cell is
   `(op, edge_class, input, output, dest_acc)`. The audit records a revision where the binary count was
   quoted as the total — do not reintroduce it.
9. **Blackhole rewrites `dest_acc` under you** in `sfpu_binary()` (§2.3). Gate on the effective value.
10. **Do not add CI groups.** The five `llk_e2e_*_nocov` groups per arch are a duplicate rather than
    new coverage and are under discussion (audit §2.8, plan §6). `llk-e2e` passes
    `not perf and not quasar and not accuracy`, so `nightly` is **not** excluded and a new
    `@pytest.mark.nightly` variant is guarded the moment it lands.

---

## 8. What this document does not cover

- **Cat F** — the 14 kernels with no `MathOperation` entry. Expansion plan §5; unrelated to these
  four families.
- **§5.6's two kernel-owner questions** — the approximation contract (23 unary ops) and
  `RsqrtCompat(0)`. Nothing here is blocked on them, and nothing here answers them.
- **The generated-NaN-sign golden fix on Wormhole** — expansion plan §4, still the most urgent item in
  that document. §2.4 step 7 depends on the same `convert_nan_to_inf` contract change, so the two
  should be sequenced: fix the comparator once, then enrol the binary ops that generate NaNs.
- **The ternary and scalar families** — §4.6 and §4.7 are closed or bounded, and the ternary suite
  declines cat B deliberately with a recorded reason.
- **Quasar.** Out of scope throughout, per the audit's scope statement.

---

# IMPLEMENTATION RECORD — 2026-08-17

Implemented on `ldjurovic/sfpu_edge_cases_phase_3` on top of `26c61ff80e9`, verified on a **Wormhole
n150**. **§5 (FPU binary — `Elwadd`/`Elwmul`/`Elwsub`) was descoped at the user's direction: SFPU ops
only.** `SrcFormatModel`, `_apply_fidelity_masking` and `_compute_eltwise` are untouched, verified by
assertion; the measured `Elwmul` fidelity defect recorded in §5.2 above is therefore **still open and
still unfixed**.

## Measured outcome

| Suite | Before (`26c61ff80e9`) | After | Change |
|---|---|---|---|
| `test_sfpu_binary.py` (`-k "not bcast"`) | 313 p · 128 s · 33 xf · 16 XP · 0 F | **445 p · 552 s · 9 xf · 16 XP · 0 F** | **+132 passing assertions, −24 xfails** |
| ↳ `test_sfpu_binary_edges` alone | 50 p · 64 s · 30 xf · 16 XP | **170 p · 488 s · 6 xf · 16 XP** | +120 |
| `test_sfpu_reduce.py` | 978 p · 548 s · 0 F | **1074 p · 548 s · 0 F** | **+96 (cat B, new)** |
| `test_sfpu_domains.py` (host-side) | 111 p | **120 p** | +9 guards |

The 16 XPASSes are the signed-zero arch gate (§5.12) and are **deliberately preserved** — that finding
is untouched.

## What was built

**§1 — the shared golden contract.** `BinarySFPUGolden.__call__` gained `dest_acc` + `output_format`
and now models the Dest width and the pack path (`cast_to_dest_dtype` → `convert_nan_to_inf`), asking
`sfpu_domains.nan_survives_to_l1()` rather than restating the rule. The reduce path — which returned
from `UnarySFPUGolden.__call__` *before* that modelling — gained `_model_reduce_dest_and_pack`. Both
are pinned against `nan_survives_to_l1` by a host-side guard.

**§2/§4.4 — cat B now exists in the binary suite.** `BINARY_SPECIALS_READY_OPS` (12 ops enrolled with
reasons) plus `_BINARY_SPECIALS_NOT_READY` (9 deferred, five causes), a new `specials_in` edge class
so a non-finite *operand* is not filed alongside `x % 0`, and `_BINARY_EDGE_OPS` widened to
`ops_with_singularity() | BINARY_SPECIALS_READY_OPS` so the 16 float ops with no pole are collectable
at all. Totality is asserted: an op in neither dict fails at collection.

**§3/§4.5 — the audit was wrong, and the real gap was elsewhere.** Alias guards pin, against the arch
`ckernel_defs.h`, that the five `SFPU_BINARY_INT` members are unreachable on WH/BH while their kernels
are covered under the `SfpuElw*` spelling at `Int32`. Then the coverage that was actually missing:
`test_sfpu_binary_int_comparison_ties` (the exact-equality input, the only one distinguishing `Le`/`Ge`
from `Lt`/`Gt`), `test_sfpu_binary_int_comparison_across_zero` (negatives via `twos_complement=True` —
verified as *delivered*, 1024 of 2048 lanes negative, not vacuous), and the four ordered comparisons
added to `_INT_EXTREME_OPS` so `a - b` overflow at the int32 extremes is driven. All green.

**§4/§4.8 — cat B for reduce.** `test_float_reduce_specials`: 6 classes × 4 pools × 2 ops × 2 formats
= 96 variants, all green. Injection is down one column so a *single* reduced lane carries the special
and the other 31 stay asserted.

**Also:** `SfpuAtan2` gained its registered branch point at `B = 0` (cat A) — the only place its answer
is discontinuous in `x`, and unreachable from a positive-only random draw.

## Three findings, in order of how much they change the audit

**1. `_EDGE_CLASS_BOTH_ZERO` / `_EDGE_CLASS_NAN` for `div`/`xlogy`/`fmod`/`remainder` are RETRACTED —
they were the pack path, not the kernel.** The audit records them as *"STILL OPEN — not explained by
the ISA"* and escalates them to kernel owners. They are neither. Measured once the golden modelled the
Dest write and the pack: all four **PASS** on every cell where a NaN reaches L1 (`Float32→Float32` and
`Float16_b→Float32`, both at `dest_acc=Yes`) and diverge only where `nan_survives_to_l1()` is False. A
divergence appearing exactly on the cells that cannot carry the datum is a statement about the
pipeline. 24 xfails deleted.

What remains on the narrowing cells is the *sign* of the substituted infinity, i.e. of a generated NaN
— canonical-positive on Blackhole by specification, explicitly unspecified on Wormhole. The golden no
longer exports the host libm's arbitrary choice for it (`_canonicalise_generated_nan`); that choice was
the *entire* reason `xlogy(0,0)` and `div(0,0)` disagreed with each other. New gate
`generated_nan_sign_is_asserted()`.

**2. Those 24 cells are recovered as assertions, not withdrawn.** The audit's §5.10 asks for "a golden
that accepts either infinity" and records it as unwritten. It is written, scoped to exactly the lanes
where the ISA declines to specify a sign (`unspecified_nonfinite_sign` in the binary driver, per-lane
in the reduce test). Magnitude, finiteness and every finite lane stay checked — so this is the one item
where the outcome is better than a skip: **24 tolerated divergences became 24 checked assertions.**

**3. The comparison family splits, and the ISA page alone gets it wrong.** All eight of
`lt/gt/le/ge/eq/ne/max/min` route through `SFPSWAP`, whose page specifies `SignMagIsSmaller()` and the
total order — so the ISA predicts the total order for all eight. **For the six comparisons that
prediction is wrong**, because the kernel wraps the swap in an explicit NaN rejection
(`SFPIADD(inf, |a|+|b|, CC_GTE0)`, commented *"rejects NaN"*), making them IEEE-unordered by
construction. `binary_max_min` is a bare `SFPSWAP` with no guard, so those two *do* follow the order.

I got this wrong first, modelling all eight on the total order; hardware failed the six comparisons on
4 cells each and passed `max`/`min` everywhere. **The rule this establishes: read the kernel to learn
which sequence an op uses, then the ISA to learn what that sequence does.** Revision 12's total-order
change to the *unary* comparisons stands — different kernels, no such guard. The same reading applies
to the reduce `Max`/`Min` fold (bare `SFPSWAP`, no guard → total order), which is where `Min` over a
column containing `+NaN` must return the **finite** minimum where `torch.min` propagates.

## A harness trap worth knowing before adding any `runtime()` axis

`conftest._collapse_runtime_only_variants` keeps **one item per compile key** for
`--compile-producer`, and that item builds the ELF every value of the runtime axis shares. A
`pytest.skip()` in the test body for that representative therefore leaves the other values running
against a binary that was never built — which presents as **`TENSIX TIMED OUT`**, not as a skip. Hit
twice: once with a gated class as representative (26 of 40 ELFs unbuilt, 48 variants timed out) and
again once cat B made `ordinary` empty for the ops with no pole. Fixed generally by compiling with the
unfiltered pair list when `TestConfig.BUILD_MODE == BuildMode.PRODUCE`; the ELF depends only on the
compile-time axes. **No fixed class is a safe representative** — do not re-derive that requirement from
the ordering.

## What the coverage audit must now say

- **§4.4** — `Cat B` column: 12 of the 22 float ops enrolled; `SfpuAtan2` gains a registered `B=0`
  branch point. `Edge sweep` flips to ✅ for the 12 enrolled plus atan2.
- **§4.5** — the five rows are a **naming artifact**, not a coverage gap; kernels covered under the
  `SfpuElw*` spelling at `Int32`. §2.4's alias warning gains the `MathOpType`-crossing case.
- **§4.8** — `Edge sweep` ✅ for `ReduceColumn`/`ReduceRow` via `test_float_reduce_specials`.
  `ReduceScalar` remains outside (different driver) — still an explicit open item.
- **§5 / §5.6 Q1** — remove `div`, `xlogy`, `fmod`, `remainder` `both_zero`/`nan_golden` from the
  divergence list and from the owner questions. The "12 ops over 70 cells" figure moves.
- **§5.10** — the "golden that accepts either infinity" item is **done** for the binary and reduce
  families and remains open for the unary sweep.
- **§2.1 / §3 finding #3** — cat B is no longer unary+scalar only; the binary and reduce families are
  enrolled, so finding #3's framing needs widening.
