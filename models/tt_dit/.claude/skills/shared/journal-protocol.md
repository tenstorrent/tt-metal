# The journal

One markdown file per campaign (`STATE.md`, or `models/<Model>_STATE.md`),
append-only, newest at the bottom.

Campaigns outlive a context window. Everything known at iteration 40 — levers
tried, measurements that turned out wrong, which device shape ships — is gone at
the next compaction unless written down. It is also the only defence against the
most expensive failure mode in perf work: **acting on a measurement that was
wrong and never noticing.** Recording what was measured *and how* makes a bad
measurement findable later; recording only conclusions does not.

## Sections

| Section | Holds |
|---|---|
| **Scope** | What is in, what is explicitly out |
| **Plan** | Path to the plan doc. Re-read every iteration |
| **Branch** | Where work lands, what it was cut from, what it must never touch |
| **References in priority order** | Reference impl (pinned commit) → upstream PRs → raw checkpoint |
| **Current milestone** | Where you are |
| **Gate evidence** | What passes, with numbers and the command |
| **Measurements recorded** | Every number, with shape and mesh |
| **Amendments** | Below |
| **Hangs / resets** | What hung, what fixed it |
| **Failed attempts** | What was tried, didn't work, and why |
| **Next step** | One concrete action for a cold start |

`Failed attempts` and `Hangs / resets` earn their place: without them successive
agents re-run the same dead ends and re-hang the same shapes.

**Every recorded measurement carries** command · mesh shape · input shape ·
warm-window method · device vs wall time · commit SHA. A number without those is
not a measurement — if you cannot supply them, "incidental timing, not a
measurement" is the honest entry.

## Amendments

**When a measurement contradicts the plan, append a dated, numbered amendment
with the evidence. Do not silently diverge and do not edit the plan in place.**
The plan is a hypothesis; measurements are facts, and the disagreement itself
marks which parts of the plan were guesses.

```markdown
## Amendment <N> (YYYY-MM-DD) — <one-line finding>
<What was assumed. What was measured. The evidence. What changes.>
```

## Retractions are first-class

When a later measurement shows an earlier amendment wrong, write a **retraction
amendment**. Do not delete the original — that it was believed, and for how
long, is part of the record.

| A retraction states | |
|---|---|
| What the original claimed | Quoted |
| Why it was wrong | The flaw in the method, not just the number |
| The correct reading | With the evidence |
| **The method note** | The rule that would have caught it — this is the valuable part |

Worked example: amendment 49 quoted `tt-perf-report`'s "running with tracing
could save 47463439 µs (97.1% of overall time)" and made trace the top priority.
Amendment 51 retracted it — the report had analysed the whole CSV including
weight upload; median op-to-op gap was 0.6 µs against a mean of 18425.9 µs, and
on a warm 300-op window the gap share was 16.2%. Device time was the bottleneck
all along. That method note is now a rule in
`../tt-dit-benchmark-profile/reading-profiles.md` — a wrong number became a
guardrail.

## Discipline

Write the entry **before** advancing, not batched at the end: an agent that runs
out of context mid-iteration leaves a complete record if it journals as it goes.
Anything reported in chat also lands here.
