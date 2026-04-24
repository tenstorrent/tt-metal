# Review Loop

Escalating review-to-done gate. Invoke from any skill that ends in a
code-review step (tt:optimizer, skill-creator, future skills). Replaces
ad-hoc "run review and call it done" patterns.

## Protocol

| Cycle | Bar | Action |
|---|---|---|
| 1 | MUST-FIX | Fix every MUST-FIX. Re-invoke `tt:code-review`. |
| 2 | MUST-FIX + SHOULD-FIX | Second review surfacing MUST-FIX means the author's quality model is off — not a specific-line disagreement. Raise the bar: fix both severities. Add a short "quality-escalation" entry to the findings note naming what was missed on cycle 1 and why. Re-invoke. |
| 3 | — | **Abort.** Hand to developer: findings note, all three review outputs, the fix diff, one paragraph on the unresolved disagreement. Do not silently run a fourth cycle. |

0 MUST-FIX on any cycle → done.

## State tracked

- `review_cycle` — integer, 1..3.
- `bar` — set per table.

## Caller contract

- Calling skill invokes this loop in its final phase.
- Calling skill must not emit a "success" or "done" summary until the loop
  returns 0 MUST-FIX or aborts.
- On abort, the calling skill surfaces the abort to the developer — it is
  not a silent failure.

## Rationale

Second-review MUST-FIX signals a systemic quality gap, not a line-level
bug. Escalating to SHOULD-FIX widens the fix scope to address the gap.
The cycle-3 abort prevents indefinite oscillation when author and
reviewer disagree about a specific finding.
