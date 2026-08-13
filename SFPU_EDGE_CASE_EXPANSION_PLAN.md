# SFPU Edge-Case Coverage — What Is Left To Do

**What is already covered:** [SFPU_EDGE_CASE_COVERAGE.md](SFPU_EDGE_CASE_COVERAGE.md) — the per-op
audit, and the record of every finding to date
**Issue:** [tenstorrent/tt-metal#49739 — [LLK] SFPU testing edge cases](https://github.com/tenstorrent/tt-metal/issues/49739)
**Repo:** `tt-metal/tt_metal/tt-llk/tests/python_tests/`
**Scope:** Wormhole B0 and Blackhole. Quasar keeps its own inline stimulus definitions under
`quasar/` and is tracked separately.

**Revision 17 — 2026-08-13.** This document contains **only work that is not done**. Completed items
are deleted rather than ticked off, because their results live in two places that cannot drift from the
code: the code itself, and the coverage audit. For what a finished item established, read the audit —
§5 has the findings, §4 has the per-op state.

Deleted in this revision because it is finished: **the Wormhole re-measurement**. The suite has now run
on a Wormhole n300, and both halves of that item came back clean —
[WORMHOLE_MEASUREMENT_RESULTS.md](WORMHOLE_MEASUREMENT_RESULTS.md) has the full record.
`specials_safe()`'s 7 cells are confirmed (250 variants, 85 failing — the recorded figures to the
variant), so the table stays un-arch-keyed and `test_specials_safe_matches_measured_matrix` keeps its
one verdict per cell; and the total order holds, so the seven enrolled goldens need no arch-keying.
Earlier revisions deleted cat E, the scalar tensor-operand edges, the CI gap and the cat-B tranches the
same way. What survived from all of it is not history but advice, and it is in §8.

**But the same run found something, and it is now the most urgent item here.** Driving the edge sweep on
Wormhole for the first time fails **49 of 752 variants across 10 ops**, all one cause: the sign of a
NaN the kernel generates. `SFPMAD.md` documents that sign as canonical-positive on Blackhole and
**explicitly unspecified on Wormhole**, so it is golden work — see §4, which now holds that instead of
the finished re-measurement. It matters on a schedule rather than eventually: `SPECIALS_READY_OPS` is
empty on `main`, so no NaN is injected there today, and **this branch's enrolment is what turns the
Wormhole e2e groups red.**

**Of the rest, everything unblocked is done and what remains is genuinely other people's.** 67 of the 97
unary ops are enrolled. Of the 30 still outside, **all 30 wait on the two questions in §3 or on a
harness** — there is no cat-B op left that this document could simply fix. Beyond §4, what is actionable
here is cat F (large, unblocked) and one CI number to tune after the next nightly.

---

## 1. The work, ordered by value

| # | Item | Blocked on | Size | Where |
|---|---|---|---|---|
| 1 | **The generated-NaN sign on Wormhole** — 49 red edge variants across 10 ops, one shared comparator/cast fix | nothing | one shared change + a both-arch sweep | §4 |
| 2 | **Two questions for kernel owners** — approximation contract, `RsqrtCompat(0)` (+ `SFPSETCC`/`NaN`) | judgement, not code | drafted, need sending | §3 |
| 3 | **Cat B, the rest** — 26 ops blocked on item 2's answers | see §2 | small, once answered | §2 |
| 4 | **Re-derive the two Wormhole arch gates** — each XPASSes on Wormhole, so each asserts nothing on either arch | a second Wormhole board (9.1); one bitwise compare (9.2) | small each | §9 |
| 5 | **Cat F** — harnesses for 11 kernels with no enum entry | new C++ source + golden each | large, per kernel | §5 |
| 6 | **Tune the new non-coverage CI groups' timeouts** | one nightly run's data | one YAML edit | §6 |

**Start with item 1**, and not because it is the most interesting: it is the only item with a deadline.
The 49 failures are invisible on `main` — `SPECIALS_READY_OPS` is empty there, so nothing injects a NaN —
and they appear the moment this branch lands. Both Wormhole e2e paths see them, so the group stops at the
first one (`-x`):

- the **non-coverage** groups (`split_group` 6–10) run all 10 ops;
- the **coverage** groups still run 6 of them — `Fmod`, `GeluAppx`, `Hardmish`, `Mish`, `Softsign`, `Tan`
  are standard-profile, so `_skip_coverage_unsupported`'s broad-profile skip does not reach them. Only
  `Cos`, `Rsqrt`, `Silu`, `Sin` are hidden there.

Then item 2: it is two emails and it decides 26 of the 30 remaining cat-B ops. Item 6 bounds the value of
everything else — coverage that no job runs can regress silently — and item 5 is the only large build
left. Item 4 is small and easy to skip, which is exactly why it is worth naming: an arch gate that always
XPASSes is coverage that has already gone quiet.

**And check the ISA before filing anything else.** The third question on this list was written as a
kernel divergence, drafted with a measured table, and turned out to be documented behaviour with the
*golden* at fault; seven ops were enrolled instead of xfailed. The measurement was still what located
the ISA page, so the order that worked was: **measure, then read the ISA, then ask a human** — and
skipping the middle step would have written seven permanent, plausible-looking lies about the
hardware.

---

## 2. Cat B — IEEE specials for the other 30 ops

**67 of the 97 unary ops are enrolled and green**, plus all 5 scalar binops. The gate is two-sided and per op, so an op joins on its own:

- `specials_safe(input, output, dest_acc)` — does the *pipeline* deliver a special intact. Measured;
  7 cells of 50; pinned by `test_sfpu_domains.py`; **re-measured on Wormhole and confirmed**, so it is
  correctly not arch-keyed.
- `SPECIALS_READY_OPS` — does the *golden* define a result for one, carrying the reason.

There is now a third gate, and it is narrower than either: `negative_zero_delivered(input, dest_acc)`.
A `-0.0` only survives on the unpack-to-dest path — a 32-bit input at `dest_acc=Yes` — and everywhere
else the datacopy hands the kernel `+0.0`. The `-0` probe is no longer sent where it cannot arrive,
because an xfail for an undelivered datum blames the kernel for the stimulus. That is what scopes
`Sqrt`'s and `Rsqrt`'s xfails, and it is the same reasoning `Signbit`'s six entries already carried.

Both of the first two must pass. What follows is per-op golden work, not stimulus work.

### 2.1 The 30 that are left, and what each is waiting for

Every one has been driven over the full specials set on every Blackhole-reachable triple — and now on
every Wormhole-reachable one too — so this is a measured list, not a to-do list of unknowns. **All 30
that remain wait on someone else**; the seven the ISA settled have been done.

| Waiting on | Ops | Which |
|---|---|---|
| **§3 Q1** — approximation contract | 23 | `CastFp32ToFp16a`, `Digamma`, `Erf`, `Erfc`, `Erfinv`, `Expm1Cw`, `Frac`, `Gelu`, `GeluDerivative`, `I1`, `Lgamma`, `Log`, `LogWithBase`, `Polygamma`, `Rdiv`, `Rpow`, `Sigmoid`, `SigmoidAppx`, `SqrtCustom`, `Tanh`, `TanhDerivative`, `TanhDerivativeLut`, `UnaryPower` |
| **§3 Q3** — `SFPSETCC` and `NaN` (the Wormhole half is now measured, and agrees) | 2 | `Sign`, `Heaviside` |
| §5 — no `MathOperation` golden at all | 3 | `TopKLocalSort`, `TopKMerge`, `TopKRebuild` (perf-only) |
| tt-llk#1120 | 1 | `ReluMin` (skipped outright) |
| §3 Q2 | 1 | `RsqrtCompat` |

**The seven the ISA settled are done and enrolled** — `Clamp`, `Hardsigmoid`, `Hardtanh`, `ReluMax`,
`UnaryGe`, `UnaryGt`, `UnaryMin`. `sfpu_total_order_key` and its `min`/`max`/`clamp`/`relu_max`
helpers model the documented order, and the seven pass as ordinary tests rather than xfails. The
mapping was confirmed against the kernels before the goldens changed, not assumed: `_relu_max_body_`
is `v_if (result > threshold)` — a two-vector compare, so `SFPGT`, so the total order — and
`_calculate_clamp_` has the same shape. `Hardsigmoid` turned out to *be* `_relu_max_body_(x/6 + 0.5,
1.0)`, sharing the kernel helper outright, which is why the golden now shares one too.

**The model now holds on Wormhole too, measured.** All seven pass 8/8 edge variants there, and a direct
probe over `+inf / -inf / NaN / ±0` reproduces the Blackhole table value for value — so **no
arch-keying**. Two corrections to what this section used to say:

- *"Wormhole has no `SFPGT`/`SFPLE` at all"* is true, but *"so Wormhole has no total order"* does not
  follow. `WormholeB0/…/SFPSWAP.md` carries the same `SignMagIsSmaller()` and the same
  `-NaN < -Inf < … < +Inf < +NaN` comment, so **the order is specified on Wormhole too**.
- What is *not* established is which instruction these kernels use there. The sources are two-vector
  compares (`_relu_max_body_`, `_calculate_clamp_`), which is why the Blackhole mapping says `SFPGT` —
  and `SFPGT` does not exist on Wormhole. sfpi lowers `operator>` through a compiler builtin whose
  expansion is in the sfpi backend, not the headers. So on Wormhole the seven are **measured** green,
  not ISA-guaranteed; a disassembly of a built `relu_max` / `unary_gt` would close that gap.

`Sign` and `Heaviside` stay out on both arches: they compare against zero via `SFPSETCC`, whose contract
explicitly excludes a `NaN` operand. Wormhole answers identically — `NaN -> 1.0` against a golden
`0.0` / `0.5`, plus the same `-0` divergence at `dest_acc=Yes`, 2 xfailed each and **0 xpassed** — so §3's
third question is one contract question rather than two measurements.

**Neither blocked group needs per-op work — each needs one answer**, and the measured tables are in
[KERNEL_OWNER_QUESTIONS.md](KERNEL_OWNER_QUESTIONS.md). Once answered, a group enrols together: as
plain passes if the goldens should model the behaviour, with one shared xfail reason if it is a
documented kernel contract, or as one bug report if it is not intended.

**Do not enrol any of them on a guess.** A reason string written to make a variant green becomes a
permanent, plausible-looking claim about the hardware, and nobody re-derives one once it is written.

**`I1` is the case to read before starting.** Its golden *was* wrong — torch returns `NaN` at `±inf`
where `I1` is odd and unbounded, so `I1(±inf) = ±inf` — and it has been fixed. It still stays out,
because its *kernel* saturates to `±1.1547668e37`, which is Q1. **Fixing a golden is not a reason to
enrol an op.** Keeping those two decisions apart is exactly what stops a kernel divergence being
laundered into a golden that agrees with it.

**Acceptance per op:** it is in `SPECIALS_READY_OPS` with a reason; the unary edge sweep runs it on
every safe triple with no unexplained failure; each divergence is either an ISA-cross-checked
non-strict xfail or a fixed golden.

### 2.2 Check the golden before spending hardware time

The cheapest step is host-side and needs no device. This is what caught `_sin` / `_cos` raising
`ValueError` on a non-finite input before a single variant was compiled:

```bash
cd tt_metal/tt-llk/tests/python_tests
python3 -c "
import sys, torch; sys.path.insert(0,'.')
from helpers.golden_generators import UnarySFPUGolden
from helpers.llk_params import MathOperation as M, DataFormat as F, DestAccumulation as D
g = UnarySFPUGolden()
t = torch.zeros(1024, dtype=torch.float32)
for i, v in enumerate([float('inf'), float('-inf'), float('nan'), 0.0, -0.0]): t[i] = v
for op in [M.Neg, M.Sqrt]:                      # <- the ops you are about to enrol
    for dest_acc in [D.No, D.Yes]:              # <- BOTH: they are different code paths
        out = torch.as_tensor(g(op, t.clone(), F.Float32, dest_acc, F.Float32, [32,32])).flatten()
        print(op.name, dest_acc.name, [float(out[i]) for i in range(5)])
"
```

An op that raises here is a golden fix. An op that returns the wrong value here is a golden fix. Only
an op that returns the *right* value is ready for hardware.

**Run both `dest_acc` values, and read the `No` row carefully.** That is the 16-bit-Dest path, the only
one where a NaN is replaced by a *signed* infinity — so it is the only place a wrong NaN sign is
visible at all. At `dest_acc=Yes` the NaN stays a NaN and the comparator's both-NaN clause accepts
anything. **Every golden defect found so far has lived in the `No` row**, and a check that ran only
`Yes` would have passed all of them straight onto hardware.

### 2.3 Drive the whole category at once, not op by op

This is method, not history: it applies to cat F in §5 as much as it did to cat B and cat E.

Per-op work is right only while the *shared machinery* is still suspect — a defect in the golden
framework will otherwise be misattributed to whichever op is in hand, which happened twice. Once the
machinery is trustworthy, driving the whole category in one sweep is strictly better: it costs a single
run, and it is the only way the families become visible at all. **A cause that shows up in 9 or 23 ops
at once is invisible when you look at one op at a time** — both of §3's blocking questions were found
that way, and both had previously been written up as one-op curiosities.

The corresponding trap is in §8: a sweep that changes shared machinery must be diffed against a
baseline across *every* op, not just the ones being enrolled.

---

## 3. Two questions for kernel owners

**Both are drafted and ready to send:** [KERNEL_OWNER_QUESTIONS.md](KERNEL_OWNER_QUESTIONS.md) has each
written up with its measured table, a reproduce command, the relevant ISA text, and what would settle
it. What is left is the part a test cannot do — putting them in front of the owners. **Neither has been
filed.** Between them they decide 24 of the 37 ops still outside cat B; two more (`Sign`, `Heaviside`)
ride along on the tail of the third question the ISA answered.

| # | Question | Ops it decides |
|---|---|---|
| 1 | What should an approximation kernel do with an input outside its series' range? | **23** |
| 2 | Why does `RsqrtCompat` saturate at the pole where `Rsqrt` does not? | 1 |

1. **Approximation kernels do not propagate non-finite inputs.** `Log` is the clearest case —
   `log(+inf)` returns `88.5`, `log(-inf)` `84.3`, `log(NaN)` `89.1`, all near `ln(FLT_MAX) = 88.7`, so
   the kernel appears to clamp its input to the format maximum and take the log of that. **It is not
   about `Log`:** 22 further ops do the same thing, either saturating to an asymptote or returning
   `NaN` where a value is defined. `LogWithBase` returns exactly `Log`'s values scaled by the dispatch
   constant `1/ln(2)`, which is the evidence the cause is shared rather than per-op. The ISA cannot
   settle it: it specifies the primitives only within stated ranges (`SFPARECIP`'s accuracy bounds run
   to `0 ≤ x < 2`, `SFPLUTFP32` documents nothing for `NaN`/`±inf`), so what a composition built on
   them does outside those ranges is an LLK/API decision by construction.
2. **`RsqrtCompat(0)` saturates to `1.7014118e38`** (`0x7F000000`) instead of returning `inf`, on all 8
   combinations — while plain `Rsqrt` over the same probe does not diverge. The ISA narrows this:
   `SFPARECIP` saturates to `0x7f800000` (`+inf`) for an input below `2^-126`, so `2^127` is not a
   value the instruction produces. The constant is a software clamp above the primitive and `Rsqrt`'s
   `+inf` is what the hardware itself gives, which makes the question *why the clamp was added*.

**The third question — are SFPU comparisons defined for a `NaN` operand? — is answered**, by
`SFPGT`/`SFPLE`/`SFPSWAP`'s documented total order. See §2.1: it turned 7 ops into golden work. What
survives of it for an owner is narrow and no longer arch-shaped: `SFPSETCC` excludes a `NaN` operand by
contract on **both** arches, which leaves `Sign` and `Heaviside` — and Wormhole has now been measured to
behave exactly as Blackhole does on them, so it is one contract question rather than two measurements.

**The `signbit(-0.0)` question that used to head this list is withdrawn.** It was read as a
kernel-contract bug; the delivery measurement shows the probe never arrives on the six combinations
where it diverges, so there is no kernel contract to question. Do not re-file it.

---

## 4. The generated-NaN sign on Wormhole — 49 red variants, one fix

Measured on a Wormhole n300; the full record, with bit patterns and ISA quotations, is
[WORMHOLE_MEASUREMENT_RESULTS.md](WORMHOLE_MEASUREMENT_RESULTS.md) §4.

```
test_eltwise_unary_sfpu_edges, Wormhole:  475 passed · 198 skipped · 49 failed · 30 xfailed · 0 xpassed
```

**Ten ops, one cause.** `Cos`, `Fmod`, `GeluAppx`, `Hardmish`, `Mish`, `Rsqrt`, `Silu`, `Sin`,
`Softsign`, `Tan` — every one of them *generates* a NaN from the probe, and every failing cell is one
where a NaN cannot survive as a NaN: the four format pairs at `dest_acc=No` (16-bit Dest) plus
`Float32->Float16_b` at `dest_acc=Yes` (16-bit output pack). The divergence is always the same shape,
`golden=+inf` against `hw=-inf`.

**The ISA settles it, one sentence per arch.** `SFPMAD.md` — "if a NaN is emitted":

| Blackhole | "it is always **the canonical NaN with bit pattern `0x7fc00000`**" |
|---|---|
| **Wormhole** | "the LSB of the mantissa is guaranteed to be set; other bits of the mantissa might or might not be set, and **the sign bit might or might not be set**" |

`UnarySFPUGolden`'s canonicalisation rests on the sentence *"The SFPU emits a positive one"*, which is a
documented Blackhole guarantee and explicitly unspecified on Wormhole. The conversion that makes the
sign observable is documented too, and flagged: the packer's early conversion says "if the exponent is 8
bits wide, NaN becomes infinity (**this is a potentially surprising behaviour**)", and `SFPSTORE`'s note
adds "software is advised to avoid NaN inputs for this conversion" — with Blackhole alone carrying the
clause "albeit canonical NaNs produced by arithmetic instructions do not suffer any truncation".

**So the fix is in the comparator, not in ten goldens and not in an xfail.** Where a golden `NaN` is
turned into `±inf` by a Dest write or a pack, accept **either** infinity; keep asserting the sign only
for `_NAN_SIGN_TRANSPARENT_OPS` (`Neg`, `Abs`, `Identity`), where the kernel *moves* the sign bit — which
the ISA backs directly: `SFPABS`'s summary is "-NaN is left as -NaN rather than becoming +NaN".

**Do not arch-key the sign instead.** It is the tempting shape — measure Wormhole's sign per op and
tabulate it — and the ISA contradicts it in advance: a bit that "might or might not be set" is not a
fact, and `Cos(+inf)` already emits a positive NaN at a 32-bit Dest and a sign-set one at a 16-bit Dest.
That table would be §8's permanent, plausible-looking lie, with an ISA sentence against it.

**Acceptance.** The 49 go green on Wormhole with no new xfail; the Blackhole baseline is unchanged
variant for variant (this touches shared machinery, so §8's diff-the-whole-op-set rule is not optional
here); `0 xpassed` on both arches. Drive it as one sweep per §2.3, not op by op.

**One residue it does not cover:** `Tan(NaN) -> 0.0` on the 16-bit-Dest path — a finite zero, not a
substituted infinity, so neither the `SFPMAD` sentence nor the pack path explains it. `Tan` at
`Float32->Float16_b`/`dest_acc=Yes` returns a NaN for the same probe, so the `0.0` belongs to the
16-bit-Dest path specifically. Measure it before deciding whether it is golden work too.

---

## 5. Cat F — the 11 kernels with no `MathOperation` entry

A header exists; nothing in the Python infra can reach them. Confirmed still absent: `welfords`,
`dropout`, `quant`, `cumsum`, `reshuffle_rows`, `int_sum`, `tiled_prod`, `copy_dest_values`,
`generalized_moe_gate_topk`, `max_pool_indices`, `rand`. Each needs a new C++ source and golden, so none
is reachable by the shared mechanism.

**Where the headers actually are, and why it does not save you anything.** Only `welfords`,
`generalized_moe_gate_topk` and `rand` live under `tt_llk_<arch>/`; the other eight are in the metal
CKernels layer at `tt_metal/hw/ckernels/<arch>/metal/llk_api/llk_sfpu/`. That looks like it should
block them, and it does not — `TestConfig.INCLUDES` puts that directory on the compile line for every
LLK test, so a harness can include them today. Check that before concluding a kernel is out of reach.

**What the harness cost actually is.** Not the include path and not the golden — it is the *call
shape*. `call_unary_sfpu_operation` dispatches on one `dst_index`, and these kernels do not fit it.
`quant` is the clearest case: `calculate_quant_int32`, `calculate_requant_int32` and
`calculate_dequant_int32` each take **three** dest indices (two inputs and an output), so driving them
needs a source that unpacks two operands into separate Dest slots and packs from a third — a harness
shape the unary suite does not have, rather than another entry in an existing dispatch. Budget per
kernel accordingly, and expect the first one to pay for the shape that the rest can then share.

| Priority | Kernels | Why |
|---|---|---|
| High | `welfords`, `int_sum`, `cumsum`, `tiled_prod` | Reduction family — four kernels share one harness cost |
| High | `quant` | Used in production quantization, no correctness test at all |
| Medium | `dropout`, `rand` | RNG; need a distribution-level assert, not element-wise |
| Medium | `reshuffle_rows`, `copy_dest_values`, `max_pool_indices` | Data-movement / index |
| Medium | `TopKLocalSort` / `Merge` / `Rebuild` | Have enum entries but are perf-only; whole-op `topk` is tested |
| Medium | `AddInt32`, `SubInt32`, `AbsInt32`, `BitwiseNot` | Perf-only, blocked by the fast-tilize gap (tt-llk#495) |

---

## 6. Tune the new non-coverage CI groups' timeouts

The broad unary profile used to run in no automated job at all: every LLK python job either
excluded the `nightly` marker or ran with coverage, and coverage skips `BROAD_SWEEP_OPS`
wholesale. `llk_e2e_tests.yaml` now carries non-coverage companion groups (`split_group` 6–10,
`llk_e2e_*_nocov`) that run the same tests without `--coverage`, so it executes.

**What is left is one number per group.** Their timeouts were copied from the instrumented
groups — 38 min on `wh_n150_civ2`, 55 on `bh_p150b_civ2` — which is a starting point, not a
measurement: a non-coverage run has more variants to execute but no instrumentation overhead.
After the first nightly, read the actual durations and set them. Budget is not the constraint
(`verify_time_budget.py` allows 1800 min per SKU for team `llk`; this took `wh_n150_civ2` to
380 and `bh_p150b_civ2` to 550), so err high until there is data.

**Do not re-cite tt-llk#1435 for the coverage skip.** That issue is about test *ordering*, and
its one mention of coverage is an observation of the skip's effect rather than a reason for it —
the citation was circular and has been removed. The actual rationale for excluding the broad
profile from the instrumented run is recorded nowhere; if you find out, write it down.

---

## 7. How to verify your work

**Host-side, no device.** The metadata and gates are pinned by tests; run these before touching
hardware:

```bash
cd tt_metal/tt-llk/tests/python_tests
python3 -m pytest test_sfpu_domains.py -q --noconftest      # 107 passed
python3 -c "
import sys; sys.path.insert(0,'.')
from helpers.sfpu_domains import (_OP_SINGULARITIES, _OP_EDGE_POINTS, sfpu_unary_ops,
                                  edge_spec, SPECIALS_READY_OPS)
from helpers.llk_params import DataFormat as F
u = sorted(sfpu_unary_ops(), key=lambda o: o.name)
e = [o for o in u if edge_spec(o, F.Float32, F.Float32) is not None]
print(len(_OP_SINGULARITIES), len(_OP_EDGE_POINTS), len(u), len(e), len(SPECIALS_READY_OPS))
"
# 21 43 97 50 72 — the last number counts the 5 scalar binops too, so it runs
#                  ahead of the 67 *unary* ops enrolled; bump it as ops join
```

**On hardware.** Never call `pytest` directly — use the repo's runner, which serialises silicon access
and cleans up stale state:

```bash
cd tt_metal/tt-llk
.claude/scripts/run_test.sh run --worktree $PWD --arch blackhole \
    --test test_sfpu_unary.py --k test_eltwise_unary_sfpu_edges
```

Baselines for all four suites through the two-phase flow. Blackhole is green; Wormhole is green apart
from §4's 49, which are the same cause in every suite that injects a NaN:

| Suite | Blackhole p300a | Wormhole n300 |
|---|---|---|
| `test_sfpu_unary.py` | 5027 passed · 1601 skipped · 18 xfailed · **0 xpassed** | 6034 passed · 533 skipped · **49 failed** · 30 xfailed · **6 xpassed** |
| `test_sfpu_binary.py` | 739 passed · 531 skipped · 36 xfailed · **0 xpassed** | 865 passed · 392 skipped · 33 xfailed · **16 xpassed** |
| `test_sfpu_ternary.py` | 39 passed · 25 skipped | 39 passed · 25 skipped |
| `test_sfpu_binop_scalar.py` | 68 passed · 72 skipped | 67 passed · 72 skipped · **1 failed** |

The 49 and the 1 are both §4 — the scalar failure is `ScalarRsub` at `Float16_b->Float16_b`/`dest_acc=No`,
diverging `+inf` against `-inf` on the `NaN` probe, which is the same cause reaching a second suite. The
6 and 16 xpassed are §9. The edge sweep alone, which is the part §4 is about:
`475 passed · 198 skipped · 49 failed · 30 xfailed`.

Note the skip counts, because they explain why this run found things: Wormhole collects the same 6652
unary variants but skips 533 where Blackhole skips 1601, since `_skip_bh_unless_fp32` collapses the whole
`dest_acc=No` row there. **Run both arches from now on** — every arch-keyed claim in this tree was written
when only one of them had ever been exercised, and the first Wormhole run turned up a 10-op family and two
dead arch gates in an afternoon.

**The Blackhole column is current; the Wormhole column predates the review round.** Its unary figures
were taken before `Signbit`'s six xfails were deleted and before the shift sweep dropped six redundant
variants, so expect its `30 xfailed` and its collected count to move on the next Wormhole run. The 49
failures are unaffected — they are §4's NaN-sign family, which none of those commits touched.


**A non-zero `xpassed` count is a signal, not noise.** Both arch-gates in this tree were derived from
one: 16 XPASS in the binary suite became the signed-zero gate, 4 in the unary suite became the
approximate-exp gate. If a run reports XPASS again, something the tables call arch-specific has
changed, and that is worth more than most deliberate work.

**It has just fired again — 6 XPASS on Wormhole, and they are the approximate-exp gate itself.** See §9.

**Diff the whole op set against a baseline, not just the ops you touched.** Any change to a shared
golden reaches every op that can produce the value it touches. Stash the change, run the sweep, unstash,
run it again, and compare per-variant outcomes; a `PASSED -> FAILED` on an op you never edited is the
signal that the fix was aimed one level too low. Both defects in this section's first two entries were
found exactly this way — a four-op enrolment regressed `Acosh`, `Cos`, `Sin` and `Exp`, none of which
the change was about — and neither would have appeared in a run of the ops being enrolled.

**Environment.** `tests/requirements.txt` pins `tt-exalens==0.3.29` and `run_test.sh` expects a venv at
`tests/.venv`, which `setup_testing_env.sh` does **not** create — that script installs SFPI and
pre-commit only. From a bare checkout the whole setup is three commands:

```bash
cd tt_metal/tt-llk
python3 -m venv tests/.venv && tests/.venv/bin/pip install -r tests/requirements.txt
bash tests/setup_testing_env.sh          # SFPI, pinned in tests/sfpi-version (7.68.0), into tests/sfpi
```

A system SFPI (e.g. `/opt/tenstorrent/sfpi`) is **not** a substitute: `TestConfig.TOOL_PATH` is hardcoded
to `tests/sfpi/compiler/bin`, and the version there was 7.35.3 against the tree's pinned 7.68.0.

---

## 8. Traps to know before starting

Every one of these has already cost time once.

- **A property of "the SFPU" is usually a property of *one* SFPU.** `SFPMAD.md` guarantees a canonical
  `0x7fc00000` NaN on Blackhole and guarantees nothing about the sign on Wormhole; `SFPGT`/`SFPLE` exist
  on one arch and not the other, while `SFPSWAP`'s total order exists on both. Before writing a fact
  about the hardware into a golden, open **both** arch directories in tt-isa-documentation — the pages
  have the same names and different sentences, and diffing the two is how §4 was diagnosed in minutes
  rather than by bisecting ten ops.
- **`run_test.sh count` ignores `--k`.** It reports the whole file's collection, so the edge sweep looks
  like 6652 variants when the filter selects 752. Only `run`/`simulate` honour the filter — size a
  `--maxfail` off the run's own `deselected` line, not off `count`.
- **A failing variant's log defeats the obvious grep.** pytest writes `<nodeid> FAILED`, but loguru dumps
  the golden and result tensors *between* the two, and those lines carry their own `| ERROR |` prefix. A
  naïve scan attributes `ERROR` to the pending nodeid and invents outcomes — it reported 128 errors for a
  run that had 49 failures and 30 xfails. Parse sequentially, anchor the outcome token to end of line,
  and reconcile the totals against pytest's own summary line before believing any per-op table.

- **Check the golden before blaming the kernel.** Of the seven divergences booked against the second
  tranche, four were the golden's and one of those was really the *test framework's*, shared by 24 ops.
  An xfail written the wrong way round is worse than no test: it is a permanent, plausible-looking lie
  about the hardware.
- **Check that the comparator can see the divergence before designing around it.** `passed_test()`
  compares with `torch.isclose`, a both-NaN clause and PCC, and `-0.0 == +0.0` under every one of them.
  Two rows sat on the blocking list for a revision because nobody checked whether a failing test could
  even exist for them. The corollary: a probe whose whole point is a zero's sign needs a bitwise
  comparator first, and that is a suite-wide change.
- **A probe that cannot be delivered is not a test.** `specials_safe()` answers whether a pipeline
  carries non-finites, which is *not* the same question as whether it carries a `-0.0`; several triples
  do the first and not the second. Sending the probe anyway costs an xfail per variant that blames the
  kernel for a datum it never received. `negative_zero_delivered()` is the second gate.
- **`torch`'s fp32 → bfloat16 cast destroys a NaN's sign.** Every NaN becomes `0xFFFF`, sign bit set,
  whatever it started as — while `.to(float16)` preserves the sign correctly, so the bug hides on
  three quarters of the format axis. It is invisible until a NaN crosses a 16-bit Dest, where the pack
  path substitutes a *signed* infinity and turns the invented sign into a `+inf`/`-inf` mismatch.
  `cast_to_dest_dtype` models the Dest write as the bit truncation it actually is. Note the cast runs
  in two places per call — the Dest write *and* the store into the result buffer, whose dtype follows
  `input_format` through `tilize_block` and is not always the Dest dtype. Fixing only the first one
  looks like it works and silently does nothing on the pipelines where the two differ.
- **The sign of a *generated* NaN is not a fact about anything.** IEEE 754 leaves it unspecified for
  an invalid operation, and torch inherits the host libm, which picks inconsistently: `cos(inf)`,
  `acosh(0.5)`, `rsqrt(-1)` and `acos(2)` give `0xFFC00000` while `sqrt(-1)` and `log(-1)` give
  `0x7FC00000`. The SFPU emits a positive one. A golden that exports libm's choice will disagree with
  hardware on whichever ops libm happened to sign negatively — 24 of the 97, found only because they
  regressed a hardware run. Canonicalise; assert a NaN's sign only for the ops that *move* the sign bit
  (`Neg`, `Abs`, `Identity`).
- **A `math.*` call in a golden is a latent cat-B failure.** `math.sin` / `math.cos` *raise*
  `ValueError("math domain error")` on a non-finite input rather than returning NaN, so a golden using
  them turns a special-value probe into a test error. Both carried a comment asserting the input was
  "never not finite" — accurate until cat B. Prefer the `_torch_unary` helper, which is IEEE-correct
  and applies the format-aware NaN rule.
- **A golden reached at stimulus-build time cannot come from `get_golden_generator`.** The harness swaps
  in a `DummyGoldenGenerator` during `--compile-producer`, and that stub has only `__call__` — no `ops`
  mapping, no attributes. This made the whole binary edge sweep unrunnable under the flow CI uses,
  undetected because it had only ever been invoked directly. Instantiate the golden class directly when
  you need it before the device exists.
- **A constant derived from another by a prose rule will drift.** `_exp_with_base_spec` is documented as
  double `_exp_spec`'s; two branches moved the two halves independently and nothing failed, they just
  stopped agreeing. There is now a host-side test asserting the relation — add the same kind of
  assertion for any new derived constant. A docstring is not a constraint.
- **`exclude_intervals()` is not stimulus-neutral.** It always rewrites its result into the `intervals`
  form, and that sampler consumes **two** `torch.rand` draws per element where the plain `low`/`high`
  path consumes one. So `uniform(1, 8)` and `intervals=[(1, 8)]` are the same distribution and
  different numbers at the same seed, and **declaring a new hole in `_SFPU_UNDEFINED_RANGES` re-rolls
  that op's entire stimulus set** even when the subtraction removes nothing. Keep edge metadata off
  this path.
- **Do not route edge specs through `for_op_pipeline`.** Its `_tighter_spec` measures a domain with
  `_spec_span`, which falls back to `spec.high - spec.low` — `None - None` for a values-list spec.
  Nothing hits it today because the paths are separate; the obvious-looking unification raises
  `TypeError`.
- **`StimuliSpec.custom` cannot carry integer extremes.** `CustomStrategy.generate_face` clamps through
  `_get_integer_bounds`, which returns `info.min + 1`, so a spec asking for `INT32_MIN` silently yields
  `INT32_MIN + 1`. Integer edges go through `src_A_override` as a raw tensor —
  `_build_paired_tile_override` is the shared helper.
- **Enum members are not their values.** `DestAccumulation` and `ApproximationMode` both wrap
  `True`/`False`, so `bool(DestAccumulation.No)` is `True`. `_two_state_flag` normalises both and
  rejects anything else; the next `if dest_acc:` written by hand will be wrong in the same way.
- **A probe must survive the datapath, not just the format.** With `dest_acc=No` the DEST holds 16 bits
  whatever the input format is, so an fp32 probe one fp32 ULP above a pole of 1.0 is truncated back
  onto the pole. `probe_beside()` decides per boundary *and per side*, because the step down from 1.0
  crosses a binade and survives while the step up does not.
- **`format_ulp` returns a lower bound for block-float formats**, because the real step is set by the
  exponent shared across the 16-element block. Safe direction, but a block-float probe *pair* cannot be
  assumed distinct.
- **`TestConfig` calls `shutil.rmtree()` on the fixed path `/tmp/tt-llk-build` at session setup.** Any
  second pytest session on the same host deletes the build tree out from under a running sweep. The
  victim reports `ld: cannot open output file`, which in a log reads exactly like a real kernel bug —
  and it lands on whichever variants happened to be linking, so the failures look scattered and
  unrelated to anything you changed. **`pytest --collect-only` counts**: it runs session setup, so a
  collection check started to answer a quick question while a sweep is running will corrupt it. Wait,
  or use a plain `python -c` import, which does not. Worth fixing separately: key the artefact
  root by session, or take the existing `/tmp/tt-llk-build-shared.lock` around the rmtree.
- **The pinned test environment drifts, and the direction matters.** A venv carrying an **older**
  exalens than the pin fails at `conftest` import with a missing-symbol `ImportError`
  (`CallstackEntry`, `ElfFile` — both *added* in later releases), which reads like a broken checkout. It
  is easy to misdiagnose as "the symbol moved in a newer release" and start writing shims; check the
  installed version against the pin first.

---

## 9. Re-derive both Wormhole arch gates — each XPASSes on Wormhole

Two gates, one shape of problem: each is a **Wormhole-only** xfail whose entire content XPASSed on a
Wormhole n300, so it now asserts nothing on either arch. Neither is urgent the way §4 is — a non-strict
xfail keeps CI green — and both are the kind of signal the plan says is worth more than most deliberate
work. Full data in [WORMHOLE_MEASUREMENT_RESULTS.md](WORMHOLE_MEASUREMENT_RESULTS.md) §6.

### 9.1 Approximate exp — 6 XPASS, and the overshoot is ~5× smaller than recorded

The unary suite's first Wormhole run reports **6 XPASS**, and they are not incidental: they are
`_APPROX_EXP_ACCURACY_XFAIL`'s entire content — all three gated cells, at both tile shapes.

```
Exp · approx_mode=Yes · fast_mode=No
  Float16 -> Float16_b   dest_acc=No    XPASS  ([64,64] and [128,256])
  Float16 -> Float16_b   dest_acc=Yes   XPASS
  Float32 -> Float16_b   dest_acc=Yes   XPASS
```

That gate records a **systematic ~5.7% overshoot (peak 6.75%) once approximate exp's argument passes
~8, measured on Wormhole**, and `_APPROX_EXP_XFAIL_IS_WORMHOLE_ONLY` narrows it to Wormhole because
Blackhole XPASSed it. So the arch it is *for* now XPASSes it too, which leaves the gate asserting
nothing on either arch.

**The direction reproduces; the magnitude does not.** Measured over the elements with `x > 8`
(`test_sfpu_wh_approx_exp.py`): mean relative error **+0.75% to +1.05%**, peak **+3.5%**, and **not one
element of any tile above 5%**. The gate expects ~5.7% mean and 6.75% peak. The overshoot is real and
about five times smaller than recorded, which puts it inside the default rtol.

**Three explanations were checked and eliminated before writing this down**, so nobody re-checks them:

- **Not the stimulus.** The overshoot region is still being exercised: at these cells the drawn tile
  reaches `x_max ≈ 9.98` (`Float16` input) and `≈ 15.98` (`Float32` input), with **6.4–10.4% of elements
  above 8**. `_APPROX_ACCURACY_MAX[Exp]` is 16.0, well clear of the ~8 threshold.
- **Not a loosened tolerance.** `CUSTOM_TOLERANCES` has no `Exp` entry, so the default 5% rtol applies,
  and `passed_test` requires `torch.all(is_valid)` — *every* element within tolerance. A systematic 5.7%
  overshoot on 6% of a tile cannot pass that.
- **Not a softened golden.** `_exp` is plain `torch.exp`; it does not model the approximation.

So on this Wormhole n300 the approximation holds 5% rtol where the recorded measurement found it
breaching. Either the kernel's approximate-exp path has changed since the gate was written, or the
overshoot varies across Wormhole boards — the recorded measurement does not name the card it was taken
on, which is itself worth fixing in whatever replaces it.

**What to do:** repeat the error measurement on a second Wormhole board. If ~1% holds there too,
**delete the gate rather than leaving it** — a non-strict xfail that always XPASSes is a tolerated
divergence for a divergence that is not there, and it hides a real regression if accuracy ever does drift.
If 5.7% reproduces on another board, the gate needs the *board* in its reason string, not just the arch.
Either way, name the card next time: the recorded measurement does not, which is why this cannot be
settled from one host.

**Also closed by the same probe:** `_APPROX_ACCURACY_MAX`'s comment flags the accurate path over (16, 80]
as *"NOT YET MEASURED … it still wants a run of the Exp/Exp2 broad sweep at ApproximationMode.No"*. That
run has happened on Wormhole — `Exp` 132 passed, `Exp2` 138 passed, 0 failed — and the probe puts the
`approx_mode=No` error at **+0.00%** above 8 out to `x = 79.97` on the 32-bit-input cells. The restored
`high=80` domain is sound on Wormhole; Blackhole still wants the same run.

### 9.2 The signed-zero class — 16 XPASS, and the arch reading cannot survive it

The binary suite reports **16 XPASS**, and they are `_WORMHOLE_ONLY_EDGE_CLASSES`'s entire content: the
`negative_zero_golden` class for `SfpuElwdiv`, `SfpuXlogy`, `SfpuBinaryFmod` and `SfpuBinaryRemainder`,
at all four `(format, dest_acc)` cells.

That gate was derived from *"measured on a Blackhole p150b, the negative-zero class XPASSed on **all 16**
cells"*, read as an arch difference: `SFPMAD` flushes a negative zero on Wormhole and preserves it on
Blackhole. **A gate that XPASSes on both arches cannot mean that.** The likelier reading is already a trap
in §8 — `passed_test` compares with `torch.isclose`, a both-NaN clause and PCC, and `-0.0 == +0.0` under
all three — in which case these variants pass whatever the hardware does, and the Blackhole XPASS was
evidence about the comparator rather than about Blackhole.

**One cheap experiment settles it:** compare that class's output **bitwise** on Wormhole. If hardware gives
`+0.0` where the golden says `-0.0`, the divergence is real but invisible — the class needs the bitwise
comparator §8 already asks for, and the arch gate is spurious. If hardware gives `-0.0`, Wormhole is not
flushing and the gate's premise is wrong on its own terms. Until then, do not treat
`_WORMHOLE_ONLY_EDGE_CLASSES` as verified.

**Do not fold either of these into §4.** They share an arch and nothing else: §4 is a documented ISA
difference with a known fix; these are undocumented disagreements with recorded measurements.
