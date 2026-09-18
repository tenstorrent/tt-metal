# Amendments and retractions

The entry format for recording that a measurement contradicted what you
believed. Used by every skill; the campaign loop that owns *when* to write them
and where they accumulate is `../tt-dit-loop/`.

## Every recorded measurement carries

Command · mesh shape · input shape · warm-window method · device vs wall time ·
commit SHA.

A number without those is not a measurement. If you cannot supply them,
"incidental timing, not a measurement" is the honest entry.

## Amendments

**When a measurement contradicts the plan, append a dated, numbered amendment
with the evidence. Do not silently diverge and do not edit the plan in place.**
The plan is a hypothesis; measurements are facts, and the disagreement itself
marks which parts of the plan were guesses.

```markdown
## Amendment <N> (YYYY-MM-DD) — <one-line finding>
<What was assumed. What was measured. The evidence. What changes.>
```

Numbering is monotonic and continues across file rollovers — an amendment number
is a permanent citation, so `am. 76` must mean one thing forever.

## Retractions are first-class

When a later measurement shows an earlier amendment wrong, write a **retraction
amendment**. Do not delete or edit the original: that it was believed, and for
how long, is part of the record.

| A retraction states | |
|---|---|
| What the original claimed | Quoted |
| Why it was wrong | The flaw in the method, not just the number |
| The correct reading | With the evidence |
| **The method note** | The rule that would have caught it — this is the valuable part |

**Worked example.** Amendment 49 quoted `tt-perf-report`'s "running with tracing
could save 47463439 µs (97.1% of overall time)" and made trace the top priority.
Amendment 51 retracted it — the report had analysed the whole CSV including
weight upload; the median op-to-op gap was 0.6 µs against a mean of 18425.9 µs,
and on a warm 300-op window the gap share was 16.2%. Device time was the
bottleneck all along.

That method note is now a rule in
`../tt-dit-benchmark-profile/reading-profiles.md`. A wrong number became a
guardrail — which is the point of keeping retractions rather than quietly
fixing the original.

## Where they live

In a campaign, amendments accumulate in `ledgers/amendments.md` and only the
latest appears in `CAMPAIGN.md` — see `../tt-dit-loop/ledgers.md`. Outside a
campaign, append them to whatever state file the work already keeps.

Either way: **append-only, and never summarised away.** A predecessor journal
that reached 4972 lines had to be compacted to stay readable, and the forensic
record went with it. Roll over to a new file instead; the campaign layout exists
so that never has to happen again.
