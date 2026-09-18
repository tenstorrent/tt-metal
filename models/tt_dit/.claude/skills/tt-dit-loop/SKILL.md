---
name: tt-dit-loop
description: >-
  Use for multi-round, multi-session work on models/tt_dit models (bringup or
  performance) where one exchange won't finish it. Trigger whenever the ask is
  open-ended and goal-driven rather than a single question: bring a model or
  all its components up until every gate is green, push a latency toward a
  target, run unattended for hours or days, resume or check the state of work
  already in progress, or decide whether an in-flight loop should keep going,
  change approach, or stop. Also trigger on the problems this solves even when
  no loop is mentioned: a state or STATE.md file grown too large to load,
  losing track of what was already tried, or re-investigating the same op or
  source across sessions. Assume this skill when the scope spans many rounds,
  even if the user never says "campaign" or "loop". Delegate single-phase asks
  instead (profile one layer, debug one PCC, find one op, one benchmark
  number). Not the built-in `loop` skill, which just reruns a prompt on a
  timer.
---

# TT-DiT Loop

> Not the built-in `loop` skill. That one reruns a prompt on an interval. This
> one runs measured rounds against a fixed baseline and stops when a gate fires.

Runs a **campaign** — the multi-round unit of work. The campaign, **not the
session, is the durable unit.** It must be fully
recoverable from the run root — checkpoint, ledgers, artifacts — with no
conversation history.

```
Phase 0 inputs → 0.5 prior art → 1 fixed baseline → 2 plan → 3 rounds until a stop gate
```

## Contract

- **Do the setup yourself.** Never ask the user to run the baseline, profile or
  planning steps separately.
- Ask only if the model, mesh, or production shape cannot be inferred from the
  branch or configs.
- Only Phase 1 sits outside the loop. Deciding whether a gap remains, gathering
  evidence, patching and revalidating all happen **inside** a round.
- Every round ends with a commit and a consistent checkpoint, so an interrupt at
  any point is recoverable.

Read `../shared/device-hangs.md` before the first device run. Rounds run sweeps,
and sweeps are the most reliable way to find a hang.

## Run layout

```
<campaign-root>/                 # branch-local, never upstreamed
  CAMPAIGN.md                    # bounded checkpoint — ALWAYS read, regenerated
  plan.md                        # goal, acceptance criteria, path boundaries
  lineage.jsonl                  # one object per round — jq it, never read whole
  ledgers/{attempts,optimizations,source-ideas,amendments}.md
  artifacts/round-<N>/           # gitignored: ops CSV, .tracy, logs, frozen output
```

`CAMPAIGN.md`, `plan.md` and `ledgers/` are committed. `artifacts/` is not —
profiles are large and regenerable.

Formats: `checkpoint.md` · `ledgers.md` · `plan-template.md`.

## Phase 0 — inputs and run root

Resolve model, mesh shape, branch, and the production shape(s) this campaign
targets. Create the run root. Record all of it in the `CAMPAIGN.md` header.

## Phase 0.5 — prior-art gate

**Mandatory before choosing any source path, and before any kernel work.** A
round that skips this is invalid.

```bash
git log --all --oneline -- models/tt_dit/models/<family>/
gh pr list --repo tenstorrent/tt-metal --search "<model> <op>" --state all --limit 30
```

Also read `../tt-dit-benchmark-profile/existing-fast-paths.md` and
`../shared/reference-models.md`. Record in `ledgers/source-ideas.md` what you
searched and what you **did not** find — a negative result stops the next round
repeating the search.

## Phase 1 — fixed baseline gate

**Once, before the loop. Immutable afterwards.** Via `tt-dit-benchmark-profile`:

| Fix | Detail |
|---|---|
| Warm device time at the production shape | Must pass the validity gate in `../tt-dit-benchmark-profile/benchmark-and-profile.md` |
| Frozen reference output | Decoded image/video/audio, for the artifact-rubric check later |
| Green quality gate | Per `../tt-dit-add-model/testing-and-accuracy.md` |
| The exact command, mesh, input shape, warm-window method, commit SHA | Recorded verbatim |

Changing the shape, mesh or target after seeing results **invalidates the
campaign**. If it must change, re-baseline explicitly and record an amendment
saying why — never silently.

## Phase 2 — plan

Write `plan.md` from `plan-template.md`: goal, acceptance criteria, and the path
boundaries that bound what the loop may do unattended.

## Phase 3 — the round

| # | Step | Detail |
|---|---|---|
| 1 | **Orient** | Read `CAMPAIGN.md` **only**. Need history? `jq` the lineage or grep a ledger — never read a ledger whole |
| 1a | **Priority rule** | A **failing gate outranks everything in `Pending work`**. Never stack perf work on a red correctness gate — the measurement is meaningless and the regression compounds |
| 2 | **Gap decision** | Target met within 2%? → stop-gate check. Else continue |
| 3 | **Evidence** | Profile at layer scope via `tt-dit-benchmark-profile`. Warm window established, op-to-op gap median **and** mean reported. Artifacts to `artifacts/round-<N>/` |
| 4 | **Kernel evidence** *(conditional)* | Invoke `tt-dit-kernel-research` **only** when the target op is top-3 in the warm window *and* `../tt-dit-benchmark-profile/existing-fast-paths.md` shows no fast path already engaged |
| 5 | **Patch** | One hypothesis, grounded in a profile row, delegated to the phase skill. This skill owns the loop, not the technique |
| 6 | **Revalidate** | Quality gate at the production shape **before** measuring · validity gate · "prove it engaged" (`../tt-dit-performance/SKILL.md`) |
| 7 | **Record** | Lineage row + attempt row always; optimization row only if correct **and** measurably better; source-idea row if a source was consulted. Regenerate `CAMPAIGN.md`. Commit all with the code change |

**Two-round throttle.** After two consecutive rounds under the 2% threshold, the
next round must be **research, not an edit** — Phase 0.5 again plus
`../tt-dit-benchmark-profile/existing-fast-paths.md` — before another patch is
attempted. This is what stops the loop grinding one knob.

## Stop gates

Run unattended until **exactly one** fires:

| Gate | Condition |
|---|---|
| **Target met** | `best` crosses the target, gates green, evidence in the ledger |
| **Tied within noise** | Within 2% of target after a repeat run |
| **Blocker** | Missing hardware, unmerged dependency, or a correctness bug outside campaign scope |
| **Ceiling proven** | Profile shows the hot path at a device or algorithmic limit, remaining levers named and ruled out |
| **Quality abort** | Gate regression — immediate. Keep the commit for forensics, roll `best` back |

Anything else — including a stall — is a reason to change approach, not to stop.

## Final report

Fixed-baseline table · post-campaign table · exact commands · artifact paths ·
changed files · gates run · **an honest statement of whether the target was
reached, with the gap if not.**

Do not round toward the goal, present a projection as a measurement, or describe
identified-but-unimplemented work as done. An unmet target with an accurate
number and a ranked list of untried levers is a good outcome; an overstated one
stops the work.

## Recovery

After an interrupt, a context exhaustion, or a device hang:

1. `../shared/device-hangs.md` — kill the stale holder, reset, verify.
2. Read `CAMPAIGN.md`. Its `Pending work` head is the next action.
3. If the last round committed but the checkpoint looks stale, regenerate it
   from `lineage.jsonl` — the lineage is the source of truth for loop state.
4. If a round was interrupted mid-flight, its artifacts exist but no lineage row
   does. Discard the partial round and re-run it; do not infer a result.
