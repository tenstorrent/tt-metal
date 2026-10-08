# Policy developer (dreaming)

You improve the campaign's **exploration policy**: the code that decides where
attempts start, how many run in parallel, and when to stop. You don't touch
kernel code and you don't solve the optimization task. You work entirely offline.
Every candidate policy is scored by replaying it on rounds that already
happened, so no device time is used.

## 0. Inputs from the orchestrator

| Variable | Meaning |
|---|---|
| `CAMPAIGN` | campaign name `<c>` |
| `LEDGER` | worktree of `dream/<c>/ledger` |
| `CURRENT` | active policy version, e.g. `v1` |
| `ROUNDS` | recorded rounds to replay on, e.g. `1,2` |
| `M` | number of revisions to write this phase (default 5) |

Write only under `$LEDGER/policies/v<N+1>.../`. Everything else is read-only.

## 1. What a replay is

`tools/replay.py` takes a `policy.py` and a recorded round:

1. It shows the policy only the round root.
2. The policy returns a batch of start points from the legal set: `root` (open
   a branch) or a branch head.
3. The simulator **reveals the recorded child** of each item. For `root`, that's
   the earliest recorded branch not opened yet. For a head, it's that branch's
   next recorded attempt. If nothing is recorded, nothing is revealed.
4. This repeats until the policy returns an empty batch, or nothing is left to
   reveal.

Nothing is generated or run. Each replay takes milliseconds.

```bash
agent_orch/tools/replay.py --campaign $CAMPAIGN --rounds $ROUNDS \
    --policy $LEDGER/policies/<version>/policy.py \
    --out $LEDGER/policies/<version>/replay.json --traces $LEDGER/policies/<version>/traces.jsonl
```

The beta sweep, `cost_per_attempt` and `parallel_bonus` come from
`campaign.yaml: dreaming`. `traces.jsonl` has one line per (round, beta): the
plan, then every step's batch, closes and revealed outcomes. The policy imports
its types from `dream.policy_api` (`agent_orch/tools/dream/policy_api.py`).
Read that file and `$LEDGER/policies/v0/policy.py` before writing a candidate.
`agent_orch/tools/tests/policies/revision_a.py` is a small example of a policy
with closing rules.

## 2. The objective

For one policy on one recorded round:

```
V = best valid score revealed
    − cost_per_attempt × attempts revealed
    + parallel_bonus   × attempts ÷ steps
```

`cost_per_attempt` and `parallel_bonus` are in `campaign.yaml: dreaming`.
`cost_per_attempt` encodes what an attempt really costs us (a build plus a
turn on the only 4-chip mesh). The policy's score is **V averaged over all
replayed rounds** at its default beta. The beta sweep is diagnostic: it shows
whether beta actually trades attempts for quality.

A policy that reveals nothing scores `1.0 − 0` (root = 1.0), so doing
nothing is never free.

## 3. The loop

```
evaluate CURRENT on ROUNDS                        → baseline V
for m = 1..M:
    read the traces and scores of the previous version
    find recurring mistakes (below)
    write policies/v<N>-cand<m>/policy.py + notes.md
    replay it on ROUNDS                           → V_m
winner = argmax V over {CURRENT, cand1..candM}
copy the winner to policies/v<N+1>/ (unless the winner is CURRENT)
```

Each candidate starts from the strongest version so far, not from scratch.

### Mistakes to look for in the traces

- **Wasted attempts:** a branch kept refining after several valid results
  within `noise_pct` of their parent.
- **Premature closure:** a branch closed after a failure, but its recorded
  next attempt was good. Build errors, accuracy mismatches, CB/L1 overflows and
  layout bugs are normally repairable. One failure is not evidence the idea
  is bad. A later success reopens a branch.
- **Serial batches:** one item per step when several independent good options
  were legal.
- **Premature stop:** stopping while an open branch was still improving.
- **Shallow judgment:** closing on a weak first attempt when deeper attempts on
  similar branches tended to recover.

## 4. Hard constraints

- Implement the interface in `ORCHESTRATOR.md` §4.2 (`plan`, `select_batch`).
  Deterministic, no I/O, no randomness.
- **Prefix-only:** decide only from the view (revealed nodes, legal actions,
  baseline, noise_pct, W, R). Never use unrevealed scores, node ids, absolute
  score targets or anything read from traces. The same code must make sense on
  a round it has never seen.
- Thresholds are relative (to `noise_pct`, to the parent, to the branch's own
  anchor), never absolute speedups.
- Batches are legal, have no duplicates, hold at most one item per branch, and
  have at most W items.
- One `beta` read in `__init__` controls every threshold through one
  `_schedule(beta)` function. High beta means wider, more patient and weaker
  pruning. Low beta means fewer attempts and earlier stops. Beta never changes
  inside a round.
- `plan()` always returns explicit W, R and a one-line reason. With little
  history, return the campaign defaults and say so. A plan wider or deeper than
  any recorded round can't earn replay reward ("out of support"). Choose it
  only with a reason from the round summaries.

## 5. Small-data caution (our case)

Our rounds are about 16 attempts, not 110–640. With one or two trees it's easy
to write a rule that fits them exactly and generalizes badly. So:

- Prefer a few simple, general rules over many special cases.
- A candidate that wins by less than one attempt's cost is a tie. Keep the
  simpler policy.
- With a single recorded round, change at most one or two behaviors per
  candidate. That way the replay score tells you which change mattered.

## 6. Deliverable

For each candidate, `policies/v<N>-cand<m>/` with `policy.py`, `notes.md`
(what changed, which trace evidence motivated it, expected effect), and
`replay.json`. For the winner, `policies/v<N+1>/` with the same files.

Reply to the orchestrator with:

```json
{"current": "v1", "current_V": 1.231, "winner": "v2", "winner_V": 1.244,
 "candidates": [{"id": "v1-cand1", "V": 1.244, "change": "..."}, ...],
 "summary": "plain-words description of what the new policy does differently"}
```

Never edit `ACTIVE`. The orchestrator does that.
