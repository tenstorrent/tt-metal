---
name: tt-dit-performance
description: >-
  Use when someone wants a tt_dit model (or any diffusion/DiT/VAE/encoder
  component) to run **faster** on Tenstorrent hardware — and the answer requires a
  profile-driven optimization campaign rather than a one-line answer. Covers:
  "make it faster", latency/device-time targets, "why is this slow", "am I leaving
  performance on the table", underused chips or unexploited parallelism/sharding,
  and deciding *whether and in what order* to pull a perf lever — dtype/bf16-vs-fp32,
  math fidelity, blockings and config sweeps, SDPA chunking, fusion, layout,
  CCL/compute overlap, trace capture. Also use when a perf change misbehaved: a
  speedup vanished after another change, a lever gave less than expected, or a
  knob's speed-vs-quality tradeoff is unclear. Enforces lever ordering (parallelism
  first, trace last), dtype-matches-reference as correctness not tuning, evidence
  that a change actually engaged, and a commit + journal per iteration. Not for
  correctness bringup, generic software/SQL/GPU perf, or non-Tenstorrent work.
---

# TT-DiT Optimization

```
profile → hypothesize → implement → correctness gate → measure → commit → journal → repeat
```

Every iteration lands as a commit and a journal entry, so the trajectory is
inspectable, any point recoverable, and a fresh agent can resume cold.

## Preflight

| Step | Detail |
|---|---|
| Read `../shared/device-hangs.md` | Every run timeout-gated, every kill followed by a reset. Sweeps are the most reliable way to find a hang |
| State the contract | This skill commits every iteration to a dedicated branch. Check the project's `CLAUDE.md` for conflicting rules and quote any conflict. Never push unless told to |
| Pin the target | Mesh shape, input shape, component. A `(2,4)` number is not comparable to a `(4,8)` one; the trend table does not normalize |
| Read the journal | `Failed attempts` and `Hangs / resets` exist so you don't repeat them |
| Confirm correctness is green | If not, that is `tt-dit-add-model` first |

## Baseline

One profile at the production shape via `tt-dit-benchmark-profile`, recording
commit SHA, mesh shape, input shape, warm-window method, per-op ranking, gap
distribution and bound class. **Fixed for the campaign** — if you must
re-baseline, say so and restate the trend table from the new zero.

## Ordering

Match levers to the **measured bound class** — that is the difference between a
loop that converges and one that grinds. Full catalogue in `optimization-levers.md`:

| # | Lever | Typical |
|---|---|---|
| — | **Invariant: dtype matches the reference exactly** | not a lever |
| 1 | Parallelism you don't have yet | 10–30× |
| 2 | Kernel research — does the op or knob already exist | research, not code |
| 3 | Layout round-trips | 1.5–2× |
| 4 | Math fidelity (distinct from dtype) | up to 2× |
| 5 | Fusion and folding — **after 1 and 2** | 1.2–1.5× |
| 6 | Blocking / config sweeps — **last among tuning** | 1.5–2× |
| 7 | Trace — **absolutely last** | large where it applies |

Start at (1). Trace is last precisely *because* it is a guaranteed win where it
applies — it will still be there after everything else, and chasing it first
ships a model that dispatches efficiently and computes slowly. An agent that
started there because the headline said 97% burned a session and needed a
retraction.

Dtype is a correctness contract: fp32 where the reference is bf16 is a bug to
fix, not a lever to pull; deviating *below* the reference is a last resort.

## Per iteration

| Step | Requirement |
|---|---|
| Hypothesize | One line, grounded in a profile row. Names every knob, predicts which dominates. If you can't point at the row, you're guessing — go profile |
| Implement | Only what the hypothesis requires. No opportunistic refactors — they contaminate the measurement and make the commit unbisectable |
| Correctness gate | The component's existing gate, at the production shape. **Before** measuring, so you can never build on a change that was already wrong |
| Prove it executed | See below — a speedup is not evidence the change took effect |
| Measure | Warm device time, **same window method as baseline** |
| Commit | `opt(<scope>): <hypothesis> — <metric> (<Δ%> vs best)` |
| Journal | Before the next iteration starts. Not batched |

| Rule | |
|---|---|
| Failed trials stay as commits | Label `forensic —` or `revert —` so `git log --grep` recovers the shape of the search |
| A quality regression **aborts** the iteration | Keep the commit for forensics, roll `best` back, stop that line. A faster wrong answer is not a result |
| Single knob by default | Bundle only when parts wouldn't win in isolation (one change unlocking L1 budget another needs), small enough to bisect in one follow-up |
| Unproven changes behind an env flag | Defaulting to current behaviour — an A/B becomes one env var, not a rebuild |

## No speedup is accepted on wall clock alone

A change that did **not** take effect will happily produce a faster number from
noise — untraced whole-model wall clock has ~3× variance. Every iteration must
show positive evidence in the profile that the intended thing happened:

| Change | Evidence it actually engaged |
|---|---|
| Fused an op | The unfused ops are **gone** from the warm profile, and the fused op is present |
| Fidelity or dtype | `MATH FIDELITY` column changed on the intended rows |
| Blocking | No blocking-fallback warnings in the log; the tuned entry was hit |
| Parallelism | `CORE COUNT` or device count moved as intended — factors did not silently degenerate to 1 |
| Env-flagged optimization | The flag is genuinely on in the measured run |
| Trace | The op-to-op gap collapsed, not just wall clock |

Record the evidence in the journal next to the number. An unexplained speedup is
a measurement to re-check, not a win to bank.

## Guard the visible output, not only the component metric

PCC against the reference is the right per-iteration gate and does not drift.
But a chain of individually-passing optimizations can still degrade what a user
sees, and the two artifacts whole-tensor PCC hides best — **tile seams and
temporal flicker** — are exactly what parallelism and normalization changes
produce.

**Freeze a baseline output** (image, video, audio) at campaign start. Re-render
and compare at campaign end, and any time a change touches sharding,
normalization or precision, against the artifact rubric in
`../tt-dit-add-model/testing-and-accuracy.md`. Where a pipeline test exists, its
VBench and CLIP gates are the automated form. Campaign-level, not
per-iteration — too slow to run every time, too important to skip.

## Convergence

| Condition | Action |
|---|---|
| Quality below the component's bar | **Abort.** Keep commit, roll back best |
| Beats best by ≥ 2% | Update best, reset stall, continue |
| Any of the last 5 beat best by ≥ 2% | Continue |
| Stall < 10 iterations | Continue |
| Otherwise | Report status and ask |

Escape hatches, sliding 5-iteration window:

| Trigger | Action |
|---|---|
| 3 flat trials in one parameter sweep | **Next lever family**, regardless of stall counter — extending a flat sweep is the most common way to burn a session |
| 3 reverts | Change approach |
| 3 regressions without revert | Re-profile and re-classify; you may have misread the bound |

Success is the target crossed — absolute (`best ≤ 3 s`), relative, roofline or
utilization. Prefer **absolute** when the user gave a product number; use
**utilization** when the baseline is low-utilization, since a 30% speedup at 25%
FLOPs has not fixed the op.

No hard iteration cap. Journal and commits are current after every iteration, so
an interrupt costs nothing.

**Multi-round campaigns** — work that spans sessions and needs a durable
checkpoint, ledgers and stop gates — are driven by `../tt-dit-loop/`. This
skill supplies the lever ordering and per-iteration discipline it calls into.

## The stall prompt

```
Stalled at iteration <N>. Best <best> (baseline <b>, Δ <pct>%). Target <t> — gap <X>×.
Last 5 trials: <commit, hypothesis, metric, quality>
Untried levers for the measured bound class: <ranked, with expected magnitude>
Journal: <path>
  1. Continue  2. Switch lever family  3. Narrow to <dominant op>  4. Accept current best
```

Always include the untried-lever list — it turns "I'm stuck" into a ten-second
decision.

## Honesty about the target

State the number and the gap plainly:

> 768P/5s: 941 s → 10.9 s (encode 6.4 s, decode 4.6 s). Against the 3 s target, a
> 3.6× gap. Remaining: decoder SDPA at 40% of its layer, untouched; encoder
> Concat and BinaryNg at deeper levels, never profiled per-level.

Do not round toward the goal, do not present a projection as a measurement, do
not describe identified-but-unimplemented work as done. An unmet target with an
honest number and a ranked list of what remains is a good handoff. An overstated
number stops the work.

## Done

Target met, **or** the catalogue is exhausted for the measured bound class with
the gap stated and evidenced. Every landed optimization justified by a profile.
Correctness green at the final commit. Journal carries the trend table,
per-iteration contribution, and what remains untried.
