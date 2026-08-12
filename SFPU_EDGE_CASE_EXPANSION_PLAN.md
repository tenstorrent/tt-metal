# SFPU Edge-Case Coverage — What Is Left To Do

**What is already covered:** [SFPU_EDGE_CASE_COVERAGE.md](SFPU_EDGE_CASE_COVERAGE.md) — the per-op
audit, and the record of every finding to date
**Issue:** [tenstorrent/tt-metal#49739 — [LLK] SFPU testing edge cases](https://github.com/tenstorrent/tt-metal/issues/49739)
**Repo:** `tt-metal/tt_metal/tt-llk/tests/python_tests/`
**Scope:** Wormhole B0 and Blackhole. Quasar keeps its own inline stimulus definitions under
`quasar/` and is tracked separately.

**Revision 11 — 2026-08-12.** This document now contains **only work that is not done**. Completed
phases and items have been removed rather than ticked off, because their results live in two places
that cannot drift from the code: the code itself, and the coverage audit.

Removed in this revision because it is finished: phases 0–4, the per-op tolerances for `pow` and
`xlogy`, the ternary operand-C poles, the first cat-B tranche, and the Blackhole verification of all of
it. If you need to know what a completed item established, the coverage audit §5 has the findings and
§4 has the per-op state.

**One thing worth carrying forward from all of it.** Every completed item assumed the *stimulus* was
the hard part, and in every case the binding constraint turned out to be the **golden**. The most
recent tranche is the clearest case: of ten cat-B ops attempted, five were already correct, two needed
a golden fix to run at all, and three are still blocked because their goldens are wrong. Budget for
that, not for the mechanism.

---

## 1. The work, ordered by value

| # | Item | Blocked on | Size | Where |
|---|---|---|---|---|
| 1 | **Cat B, next tranche** — 5 ops measured, 3 blocked on one golden fix | golden semantics | medium | §2 |
| 2 | **Two questions for kernel owners** — `Log` input saturation, `RsqrtCompat(0)` | judgement, not code | two emails | §3 |
| 3 | **Cat B, the long tail** — the remaining 87 unary ops | as item 1, repeated | large, divisible | §2.3 |
| 4 | **Wormhole re-measurement** — `specials_safe()`'s four unreachable rows | Wormhole hardware | one sweep | §4 |
| 5 | **Cat E** — the unary shift amount | C++ `constexpr` → `TemplateParameter` | cross-language | §5 |
| 6 | **Cat F** — harnesses for 11 kernels with no enum entry | new C++ source + golden each | large, per kernel | §6 |
| 7 | **Scalar tensor-operand edges** | cat B reaching the scalar goldens | thin wrapper | §7 |
| 8 | **CI runs none of this** | a scheduling decision | one workflow change | §8 |

**Start with item 2.** It is two questions to two people, it costs nothing, and one of them (`Log`)
blocks an op in item 1. **Item 1 is the only item that unblocks other work.** Item 8 bounds the value
of everything else on the list.

---

## 2. Cat B — IEEE specials for the other 92 ops

Five ops are enrolled and green (`Identity`, `Abs`, `Exp`, `Sin`, `Cos`). The gate is two-sided and per
op, so an op joins on its own:

- `specials_safe(input, output, dest_acc)` — does the *pipeline* deliver a special intact. Measured;
  7 of 40 triples; pinned by `test_sfpu_domains.py`.
- `SPECIALS_READY_OPS` — does the *golden* define a result for one, carrying the reason.

Both must pass. What follows is per-op golden work, not stimulus work.

### 2.1 The next five, and the one fix that unblocks three of them

`_SPECIALS_NEXT_TRANCHE` in `sfpu_domains.py` carries this table in a comment. Measured on Blackhole
with a Float32 input, the only specials-carrying input format reachable there:

| Op | Probe | Golden | Hardware | Whose bug |
|---|---|---|---|---|
| `Neg` | `NaN` | `+inf` | `-inf` | **golden** — NaN mangled at `dest_acc=No` |
| `Neg` | `+0` | `+0` | `-0` | **golden** — the hardware is IEEE-correct |
| `Reciprocal` | `NaN` | `+inf` / `NaN` | `+0` | kernel — NaN not propagated |
| `Reciprocal` | `-inf` | `+0` | `-0` | **golden** — hardware IEEE-correct |
| `Sqrt` | `-0` | `+0` | `NaN` | kernel |
| `Rsqrt` | `-0` | `-inf` | `NaN` | kernel |
| `Log` | `±inf`, `NaN` | `±inf` / `NaN` | `88.5`, `84.3`, `89.1` | kernel — §3 |

None is enrolled, deliberately. **Enrolling an op whose golden is wrong launders a test bug into a
permanent "known hardware divergence"** — the xfail reason blames the kernel for the golden's error,
and nobody re-derives a reason string once it is written.

**Do them in this order.** The dependency is real: step 1 unblocks three rows.

1. **Fix the golden's NaN handling through the 16-bit dest path.** `Neg(NaN) → +inf` and
   `Log(NaN) → +inf` are one defect seen twice, and both appear only at `dest_acc=No`. This is a single
   fix in the golden's quantization path and the prerequisite for everything below.
2. **Decide whether the golden models signed zero at all.** `Neg(+0) → -0` and
   `Reciprocal(-inf) → -0` are cases where the *hardware* is right. Either the golden learns signed
   zero, or those probes are excluded with a recorded reason — but "xfail the hardware" is not
   available when the hardware is the correct party.
3. **Enrol `Neg` and `Reciprocal`**, recording `Reciprocal(NaN) → +0` as a genuine kernel xfail.
4. **Enrol `Sqrt` and `Rsqrt`.** Both are clean kernel divergences: IEEE says `sqrt(-0) = -0` and
   `rsqrt(-0) = -inf`, the golden agrees, the hardware returns `NaN`. Non-strict xfails, and
   **`dest_acc=Yes` only** — at `dest_acc=No` the `-0` probe is not delivered at all (coverage audit
   §5.2), so those cells would be vacuous.
5. **`Log` last**, and only after §3 answers whether its input saturation is intended.

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
    out = torch.as_tensor(g(op, t.clone(), F.Float32, D.Yes, F.Float32, [32,32])).flatten()
    print(op.name, [float(out[i]) for i in range(5)])
"
```

An op that raises here is a golden fix. An op that returns the wrong value here is a golden fix. Only
an op that returns the *right* value is ready for hardware.

### 2.3 The long tail

87 unary ops remain outside the gate, and **47 of them are smooth everywhere** — no knee, no pole — so
`edge_spec()` returns `None` for them and cat B is their *entire* edge story. That is the single
largest remaining coverage gap in the suite.

Measured rate from the first tranche, for estimating: **half the ops are free** (the golden already
routes through torch and torch is IEEE-correct), about a fifth need a small mechanical golden fix, and
the rest need a real decision about semantics. Each op is one small reviewable commit.

---

## 3. Two questions for kernel owners

Neither has been filed. Both are divergences the ISA does not explain, cheap for an owner to adjudicate
and expensive for a test to keep guessing about.

1. **`Log` saturates a non-finite input.** `log(+inf)` returns `88.5`, `log(-inf)` `84.3`, `log(NaN)`
   `89.1`, `log(-0)` `-92.5` — all finite, all near `ln(FLT_MAX) = 88.7`. The kernel appears to clamp
   its input to the format maximum and take the log of that, so no non-finite input survives it. **Is
   that intended, and should it be documented?** `Log` cannot be enrolled in cat B until this is
   answered, because there is no way to know whether the right outcome is a pass, an xfail or a bug
   report.
2. **`RsqrtCompat(0)` saturates to `1.7014118e38`** (`0x7F000000`) instead of returning `inf`, on all 8
   combinations — while plain `Rsqrt` over the same probe does not diverge. Two implementations of one
   function disagreeing at their shared pole, with nothing in the ISA prescribing either answer.

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

## 5. Cat E — the unary shift amount

`LeftShift` and `RightShift` still run at a **fixed shift of 3** with small positive inputs.
`SHIFT_AMOUNT` is a `constexpr std::uint32_t SHIFT_AMOUNT = 3u` inside `call_unary_sfpu_operation`
(`helpers/include/sfpu_operations.h`), paired with `_int_shift_amount` on the golden side. Sweeping it
needs a new `TemplateParameter` plus matching golden plumbing — cross-language, not test wiring.

The Python side is already written and reusable: `_SHIFT_EDGE_AMOUNTS` covers
`{0..31, 32, 33, 40, 63, 100, 1000, −1, −5, −32, −1000}`, `_shift_reference` is the golden, and
`_build_paired_tile_override` is the delivery. The binary shift ops are fully covered by contrast, so
this is the last gap in cat E.

---

## 6. Cat F — the 11 kernels with no `MathOperation` entry

A header exists; nothing in the Python infra can reach them. Confirmed still absent: `welfords`,
`dropout`, `quant`, `cumsum`, `reshuffle_rows`, `int_sum`, `tiled_prod`, `copy_dest_values`,
`generalized_moe_gate_topk`, `max_pool_indices`, `rand`. Each needs a new C++ source and golden, so none
is reachable by the shared mechanism.

| Priority | Kernels | Why |
|---|---|---|
| High | `welfords`, `int_sum`, `cumsum`, `tiled_prod` | Reduction family — four kernels share one harness cost |
| High | `quant` | Used in production quantization, no correctness test at all |
| Medium | `dropout`, `rand` | RNG; need a distribution-level assert, not element-wise |
| Medium | `reshuffle_rows`, `copy_dest_values`, `max_pool_indices` | Data-movement / index |
| Medium | `TopKLocalSort` / `Merge` / `Rebuild` | Have enum entries but are perf-only; whole-op `topk` is tested |
| Medium | `AddInt32`, `SubInt32`, `AbsInt32`, `BitwiseNot` | Perf-only, blocked by the fast-tilize gap (tt-llk#495) |

---

## 7. Scalar tensor-operand edges

Do **not** start this one on its own. All five scalar ops are `x (+|-|*|/) c` for a compile-time `c`, so
they are smooth in `x`: cat A and cat D contribute nothing and their only edge is cat B. None is in
`SPECIALS_READY_OPS`, so a wrapper today collects 20 nightly variants and skips all 20.

It goes live as one of §2's commits, when cat B reaches the scalar goldens. The `spec_A` hook on
`_run_sfpu_binop_scalar` is already in place and the wrapper sketch sits in a comment where the test
was.

Widening the *scalar* axis beyond `|scalar| ≤ 8`, and `±tiny` / `±large` on the tensor operand, is a
separate matter and needs a per-op tolerance on the scalar suite first — the same pattern
`BINARY_CUSTOM_TOLERANCES` uses for `pow` and `xlogy`.

---

## 8. CI runs none of this

**The broad unary profile runs in no automated job on any architecture.** Every LLK pytest job either
excludes `nightly` (pr-gate smoke, bit-exact) or runs `--coverage`, under which the broad profile is
skipped wholesale. That leaves the large majority of the sweep's parametrizations running nowhere, and
it predates all of this work.

Either `llk-e2e` needs a non-coverage companion group, or the broad profile has to stop being
coverage-gated. **This bounds the value of every other item on the list**: coverage that no job runs can
regress silently, and the two arch-gates added recently are exactly the kind of thing that regresses
quietly.

**One citation to check first.** The live skip reason in `test_sfpu_unary.py` attributes the
coverage-gating to [tt-llk#1435](https://github.com/tenstorrent/tt-llk/issues/1435). That issue is
open, but its title is about `test_eltwise_unary_sfpu.py` failing on a mismatch when it runs after
`test_eltwise_binary` — test *ordering*, not coverage. Either the citation is wrong and has propagated
into the source, or the issue has been repurposed in its comments. Resolve it before filing anything
that cites it, since the skip reason points readers there.

---

## 9. How to verify your work

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
# 21 43 97 50 5 — bump the last number as ops join SPECIALS_READY_OPS
```

**On hardware.** Never call `pytest` directly — use the repo's runner, which serialises silicon access
and cleans up stale state:

```bash
cd tt_metal/tt-llk
.claude/scripts/run_test.sh run --worktree $PWD --arch blackhole \
    --test test_sfpu_unary.py --k test_eltwise_unary_sfpu_edges
```

Current baseline on a Blackhole p150b, all four suites green through the two-phase flow:

| Suite | Result |
|---|---|
| `test_sfpu_unary.py` | 4932 passed · 1666 skipped · 14 xfailed |
| `test_sfpu_binary.py` | 739 passed · 531 skipped · 36 xfailed · **0 xpassed** |
| `test_sfpu_ternary.py` | 39 passed · 25 skipped |
| `test_sfpu_binop_scalar.py` | 58 passed · 62 skipped |

**A non-zero `xpassed` count is a signal, not noise.** Both arch-gates in this tree were derived from
one: 16 XPASS in the binary suite became the signed-zero gate, 4 in the unary suite became the
approximate-exp gate. If a run reports XPASS again, something the tables call arch-specific has
changed, and that is worth more than most deliberate work.

**Environment.** `tests/requirements.txt` pins `tt-exalens==0.3.29` and `run_test.sh` expects a venv at
`tests/.venv`, which `setup_testing_env.sh` does **not** create — that script installs SFPI and
pre-commit only.

---

## 10. Traps to know before starting

Every one of these has already cost time once.

- **Check the golden before blaming the kernel.** Three of the five ops in §2.1 diverge because the
  *golden* is wrong, and in two of those the hardware is the IEEE-correct party. An xfail written the
  wrong way round is worse than no test: it is a permanent, plausible-looking lie about the hardware.
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
  second pytest session on the same host — including a one-op `-k` run started to triage something —
  deletes the build tree out from under a running sweep. The victim reports `ld: cannot open output
  file`, which in a log reads exactly like a real kernel bug. Worth fixing separately: key the artefact
  root by session, or take the existing `/tmp/tt-llk-build-shared.lock` around the rmtree.
- **The pinned test environment drifts, and the direction matters.** A venv carrying an **older**
  exalens than the pin fails at `conftest` import with a missing-symbol `ImportError`
  (`CallstackEntry`, `ElfFile` — both *added* in later releases), which reads like a broken checkout. It
  is easy to misdiagnose as "the symbol moved in a newer release" and start writing shims; check the
  installed version against the pin first.
