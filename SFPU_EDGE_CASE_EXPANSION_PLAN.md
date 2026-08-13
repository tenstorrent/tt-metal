# SFPU Edge-Case Coverage — What Is Left To Do

**What is already covered:** [SFPU_EDGE_CASE_COVERAGE.md](SFPU_EDGE_CASE_COVERAGE.md) — the per-op
audit, and the record of every finding to date
**Issue:** [tenstorrent/tt-metal#49739 — [LLK] SFPU testing edge cases](https://github.com/tenstorrent/tt-metal/issues/49739)
**Repo:** `tt-metal/tt_metal/tt-llk/tests/python_tests/`
**Scope:** Wormhole B0 and Blackhole. Quasar keeps its own inline stimulus definitions under
`quasar/` and is tracked separately.

**Revision 14 — 2026-08-13.** This document contains **only work that is not done**. Completed items
are deleted rather than ticked off, because their results live in two places that cannot drift from the
code: the code itself, and the coverage audit. For what a finished item established, read the audit —
§5 has the findings, §4 has the per-op state.

Deleted in this revision because they are finished: **cat E** (the unary shift amount now sweeps its
full axis), **the scalar tensor-operand edges** (the five scalar binops are enrolled and the wrapper
that was a comment is a live test), and **the CI gap** (the broad unary profile now runs in
non-coverage companion groups instead of nowhere). Earlier revisions deleted the cat-B tranches the
same way; 60 of the 97 unary ops are enrolled and green on Blackhole. What survived from all of it is
not history but advice, and it is in §8.

**What is left divides cleanly in three, and only one part is ordinary work.** Of the 37 unary ops
still outside cat B, 33 are waiting on the three questions in §3 — judgement calls that belong to
kernel owners, not to this document. One item needs hardware nobody here has. That leaves cat F, which
is large but unblocked, and one number to tune after the next nightly.

---

## 1. The work, ordered by value

| # | Item | Blocked on | Size | Where |
|---|---|---|---|---|
| 1 | **Three questions for kernel owners** — approximation contract, `NaN` comparisons, `RsqrtCompat(0)` | judgement, not code | drafted, need sending | §3 |
| 2 | **Cat B, the last 37 ops** — 33 of them blocked on item 1's answers | see §2 | small, once answered | §2 |
| 3 | **Wormhole re-measurement** — `specials_safe()`'s four unreachable rows | Wormhole hardware | one sweep | §4 |
| 4 | **Cat F** — harnesses for 11 kernels with no enum entry | new C++ source + golden each | large, per kernel | §5 |
| 5 | **Tune the new non-coverage CI groups' timeouts** | one nightly run's data | one YAML edit | §6 |

**Start with item 1.** Those questions looked like two one-op curiosities when they were written.
Driving the full unary set showed they are two *kernel behaviours* that between them decide **32 of the
37 ops still outside cat B**, so they are no longer the cheap item on the list — they are the one that
unblocks almost everything else. Nothing else here is blocked on anything but hardware and effort.

---

## 2. Cat B — IEEE specials for the other 37 ops

**60 of the 97 unary ops are enrolled and green.** The gate is two-sided and per op, so an op joins on its own:

- `specials_safe(input, output, dest_acc)` — does the *pipeline* deliver a special intact. Measured;
  7 cells of 50; pinned by `test_sfpu_domains.py`.
- `SPECIALS_READY_OPS` — does the *golden* define a result for one, carrying the reason.

There is now a third gate, and it is narrower than either: `negative_zero_delivered(input, dest_acc)`.
A `-0.0` only survives on the unpack-to-dest path — a 32-bit input at `dest_acc=Yes` — and everywhere
else the datacopy hands the kernel `+0.0`. The `-0` probe is no longer sent where it cannot arrive,
because an xfail for an undelivered datum blames the kernel for the stimulus. That is what scopes
`Sqrt`'s and `Rsqrt`'s xfails, and it is the same reasoning `Signbit`'s six entries already carried.

Both of the first two must pass. What follows is per-op golden work, not stimulus work.

### 2.1 The 37 that are left, and what each is waiting for

Every one has been driven over the full specials set on every Blackhole-reachable triple, so this is a
measured list, not a to-do list of unknowns. **32 of the 37 are two kernel behaviours.**

| Waiting on | Ops | Which |
|---|---|---|
| **§3 Q1** — approximation contract | 23 | `CastFp32ToFp16a`, `Digamma`, `Erf`, `Erfc`, `Erfinv`, `Expm1Cw`, `Frac`, `Gelu`, `GeluDerivative`, `I1`, `Lgamma`, `Log`, `LogWithBase`, `Polygamma`, `Rdiv`, `Rpow`, `Sigmoid`, `SigmoidAppx`, `SqrtCustom`, `Tanh`, `TanhDerivative`, `TanhDerivativeLut`, `UnaryPower` |
| **§3 Q3** — `NaN` comparison ordering | 9 | `Clamp`, `Hardsigmoid`, `Hardtanh`, `Heaviside`, `ReluMax`, `Sign`, `UnaryGe`, `UnaryGt`, `UnaryMin` |
| §5 — no `MathOperation` golden at all | 3 | `TopKLocalSort`, `TopKMerge`, `TopKRebuild` (perf-only) |
| tt-llk#1120 | 1 | `ReluMin` (skipped outright) |
| Already fully xfailed at its pole | 1 | `RsqrtCompat` (§3 Q2) |

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

## 3. Three questions for kernel owners

**All three are drafted and ready to send:** [KERNEL_OWNER_QUESTIONS.md](KERNEL_OWNER_QUESTIONS.md) has
each one written up with its measured table, a reproduce command, and what would settle it. What is left
is the part a test cannot do — putting them in front of the owners. **None has been filed.**

Each is a divergence the ISA does not explain, cheap for an owner to adjudicate and expensive for a test
to keep guessing about. Between them they decide 33 of the 37 ops still outside cat B:

| # | Question | Ops it decides |
|---|---|---|
| 1 | What should an approximation kernel do with an input outside its series' range? | **23** |
| 2 | Why does `RsqrtCompat` saturate at the pole where `Rsqrt` does not? | 1 |
| 3 | Are SFPU comparisons defined for a `NaN` operand? | **9** |

1. **Approximation kernels do not propagate non-finite inputs.** `Log` is the clearest case —
   `log(+inf)` returns `88.5`, `log(-inf)` `84.3`, `log(NaN)` `89.1`, all near `ln(FLT_MAX) = 88.7`, so
   the kernel appears to clamp its input to the format maximum and take the log of that. **It is not
   about `Log`:** 22 further ops do the same thing, either saturating to an asymptote or returning
   `NaN` where a value is defined. `LogWithBase` returns exactly `Log`'s values scaled by the dispatch
   constant `1/ln(2)`, which is the evidence the cause is shared rather than per-op.
2. **`RsqrtCompat(0)` saturates to `1.7014118e38`** (`0x7F000000`) instead of returning `inf`, on all 8
   combinations — while plain `Rsqrt` over the same probe does not diverge. Two implementations of one
   function disagreeing at their shared pole, with nothing in the ISA prescribing either answer.
3. **SFPU comparisons rank `NaN` above every finite value**, where IEEE makes every ordered comparison
   with a `NaN` false. Derived rather than guessed: the six unary comparison ops split exactly along
   the predicted line — `UnaryLt`, `UnaryLe` and `UnaryMax` agree with their goldens while `UnaryGt`,
   `UnaryGe` and `UnaryMin` do not — and `Clamp`, `Hardtanh`, `Hardsigmoid`, `ReluMax`, `Sign` and
   `Heaviside` each return their own dispatch constant at `NaN`. The ISA already declines to specify
   `SFPSETCC` for a negative-zero operand; `NaN` looks like the same gap.

**The `signbit(-0.0)` question that used to head this list is withdrawn.** It was read as a
kernel-contract bug; the delivery measurement shows the probe never arrives on the six combinations
where it diverges, so there is no kernel contract to question. Do not re-file it.

---

## 4. Wormhole re-measurement of `specials_safe()`

The table is a *measurement*, not a derivation, and every cat-B decision in §2 reads it. It was taken
on Wormhole, and **only 3 of its 7 triples are reachable on Blackhole** — `_skip_bh_unless_fp32` allows
just `Float32->Float32` at `dest_acc=No`, and the edge sweep's format axis omits `Float32->Float16`.
Those 3 are confirmed. The other 4 involve a `Float16_b` input at `dest_acc=No` or a `Float16` output,
which Blackhole's own architecture guards exclude.

So the table is deliberately **not** arch-keyed. What would settle it is re-running the original
instrument on Wormhole — the five `isinf` / `isposinf` / `isneginf` / `isnan` / `isfinite` predicates
over the full 5×5 format matrix × both `dest_acc` with no skips, 250 variants — and either confirming
the 7 cells or making the table arch-keyed.

If it becomes arch-keyed, `test_sfpu_domains.py::test_specials_safe_matches_measured_matrix` has to be
parametrized by arch too; it currently pins one verdict per cell.

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
# 21 43 97 50 65 — the last number counts the 5 scalar binops too, so it runs
#                  ahead of the 60 *unary* ops enrolled; bump it as ops join
```

**On hardware.** Never call `pytest` directly — use the repo's runner, which serialises silicon access
and cleans up stale state:

```bash
cd tt_metal/tt-llk
.claude/scripts/run_test.sh run --worktree $PWD --arch blackhole \
    --test test_sfpu_unary.py --k test_eltwise_unary_sfpu_edges
```

Current baseline on a Blackhole p300a, all four suites green through the two-phase flow:

| Suite | Result |
|---|---|
| `test_sfpu_unary.py` | 5030 passed · 1601 skipped · 21 xfailed |
| `test_sfpu_binary.py` | 739 passed · 531 skipped · 36 xfailed · **0 xpassed** |
| `test_sfpu_ternary.py` | 39 passed · 25 skipped |
| `test_sfpu_binop_scalar.py` | 68 passed · 72 skipped |

**A non-zero `xpassed` count is a signal, not noise.** Both arch-gates in this tree were derived from
one: 16 XPASS in the binary suite became the signed-zero gate, 4 in the unary suite became the
approximate-exp gate. If a run reports XPASS again, something the tables call arch-specific has
changed, and that is worth more than most deliberate work.

**Diff the whole op set against a baseline, not just the ops you touched.** Any change to a shared
golden reaches every op that can produce the value it touches. Stash the change, run the sweep, unstash,
run it again, and compare per-variant outcomes; a `PASSED -> FAILED` on an op you never edited is the
signal that the fix was aimed one level too low. Both defects in this section's first two entries were
found exactly this way — a four-op enrolment regressed `Acosh`, `Cos`, `Sin` and `Exp`, none of which
the change was about — and neither would have appeared in a run of the ops being enrolled.

**Environment.** `tests/requirements.txt` pins `tt-exalens==0.3.29` and `run_test.sh` expects a venv at
`tests/.venv`, which `setup_testing_env.sh` does **not** create — that script installs SFPI and
pre-commit only.

---

## 8. Traps to know before starting

Every one of these has already cost time once.

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
