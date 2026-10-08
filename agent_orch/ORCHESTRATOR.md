# Orchestrator

You run one Dream-RSI campaign on one tt-metal op. You are the hands of the
loop, not its brain:

- **The policy** (`policy.py` on the ledger) decides where attempts start, how
  many run in parallel and when to stop.
- **Workers** decide what code to write.
- **You** execute the policy's decisions, spawn workers, check their work,
  record everything, show the human what's happening, and trigger dreaming
  between rounds.

Read `README.md`, `WORKER.md` and `campaigns/<c>/campaign.yaml` before
starting. You never write op code and you never override the policy (§6).

Run on the device machine (bh-qbge-12). Workers build and evaluate there.

## 1. Vocabulary

| Term | Meaning | Our default |
|---|---|---|
| Round `t` | One online run of one policy version. Produces tree `r<tt>-*` | |
| Step `k` | One batch inside a round: policy picks ≤ W start points, one worker each | |
| W | Max parallel workers per step | 4 |
| R | Max attempts per branch in a round | 4 |
| Node | One attempt = one commit + `attempts/<node_id>/` | |
| Branch | `dream/<c>/b/r<tt>-b<bb>`: chain of nodes off the round root, head = frontier | |

Round 1 at our defaults is 4 × 4 = 16 attempts. A step takes as long as its
slowest worker (agent work + build + eval, roughly 30–45 min), so a round is
about 2–3 hours. The paper's rounds were 110–640 attempts. Ours are small
because every attempt costs a build and a turn on the only 4-chip mesh.

## 2. Campaign setup (once)

All tools live in `agent_orch/tools/`. Machine settings are in
`tools/dream_env.sh`, including `DREAM_HOME=/localdev/$USER/dream`. Below,
`T=agent_orch/tools`, `L=$DREAM_HOME/<c>/ledger`, and commands run from the
main checkout.

```bash
$T/setup_campaign.sh --campaign <c>          # root tag, ledger (policy v0), eval checkout + first build
$T/eval_attempt.sh --campaign <c> --baseline # 3 profiled runs of the root -> $L/baseline.json
git -C $L add -A && git -C $L commit -m "[dream:<c>] ledger: baseline"
```

`setup_campaign.sh` requires `agent_orch/campaigns/<c>/campaign.yaml` to be
committed at HEAD; that commit becomes `dream/<c>/root`. The first build of
the eval checkout takes about 10 minutes. `baseline.json` holds per-shape
medians and the run-to-run spread (`noise_pct`). Every score is relative to it.

## 3. Running a round

```bash
# start
$T/eval_attempt.sh --campaign <c> --measure dream/<c>/root --label r<tt>_root   # prints drift per shape
#   any "DRIFT" line -> stop and tell the human: the machine changed, scores aren't comparable
$T/policy_step.py --campaign <c> --round <t> --plan [--root <ref>]   # writes $L/rounds/r<tt>/manifest.json

# each step
$T/policy_step.py --campaign <c> --round <t> --next   # logs the decision, prints {stop, items:[{node, parent, worktree, prepare}]}
#   stop == true -> the round is over
#   for each item: run its "prepare" command (prepare_worker.sh), then spawn a worker (§3.1)
#   wait for all workers of the step
$T/verify_node.py --campaign <c> --node <node> --record   # for each item (§3.2)
$T/history.py --campaign <c>                               # history.md + tree.html (§5)
git -C $L add -A && git -C $L commit -m "[dream:<c>] r<tt> step <k>"

# end
write $L/rounds/r<tt>/summary.md, copy history.md to $L/snapshots/history_r<tt>.md, commit the ledger
publish tree.html, report to the human (§5)
ask the human before dreaming and before the next round (unless they gave a multi-round budget)
```

`policy_step.py` builds the policy's view from git (the `dream/<c>/n/r<tt>-*`
tags and their `score.json`) and the ledger (closed branches, overrides,
lost attempts). The policy sees exactly what the replay simulator will show it
later, which is what makes the round replayable. `--next` refuses to run while
a node from the previous step has neither a commit nor a `--record`ed result.

### 3.1 Spawning a worker

`prepare_worker.sh` (the item's `prepare` command) sets up the worktree and
writes a skeleton `node.json`:

- **`parent: root`** creates branch `dream/<c>/b/r<tt>-b<bb>` and its
  worktree at the round root. The node is `r<tt>-b<bb>-a01`.
- **`parent: <head>`** reuses the branch's worktree, checks it sits exactly on
  the parent, and cleans leftovers of a lost attempt. The node is the next
  `a<nn>`.

Launch each worker with `run_worker.sh`. It starts a headless Claude Code
session on the device machine inside the worker's worktree, so the worker's
edits, git and evaluations all run where the device is:

```bash
ssh <device machine> $T/run_worker.sh --campaign <c> --node <node> --parent <parent>   # run in the background
```

The transcript goes to `$DREAM_HOME/<c>/logs/worker_<node>.jsonl`, and the last
line printed is the worker's report. The prompt has only:

1. "Follow `agent_orch/WORKER.md` exactly."
2. The inputs from WORKER.md §0: `CAMPAIGN`, `NODE_ID`, `PARENT` (`root` or a
   node id), `WORKTREE`, `HISTORY` (`$DREAM_HOME/<c>/history.md`).

If the orchestrator itself runs on the device machine, plain subagents work
too. Give them the same two items.

Don't add advice about what to try. The paper found that injecting directional
guidance into workers' prompts made discovery worse (Dream-RSI §5.1). Workers
read history and decide for themselves.

Spawn all workers of a step in parallel, in one message. Device evaluations
queue on the eval tool's lock, so they never collide.

### 3.2 Verifying a node

Run `verify_node.py --record` for every item of the step, including workers
that died. It checks:

- the tag `dream/<c>/n/<node>` exists and its git parent is the expected parent
- `node.json` and `eval/score.json` agree on `valid`, `fail_class` and `score`
- the commit touches only `allowed_paths` plus its own node directory

With `--record`, a failing node gets an `override` line in `decisions.jsonl`,
which `policy_step.py` and `replay.py` apply on top of `score.json`. The
worker's commit is never rewritten. A worker that never committed gets a
`lost` line instead. The attempt still cost budget, and the branch head stays
where it was.

## 4. The policy

### 4.1 v0 (round 1): parallel refine

- `plan`: W = 4, R = 4 (`campaign.yaml: policy_defaults`).
- `select_batch`, step 1: open W branches from the root.
- `select_batch`, later steps: refine every branch whose head has fewer than
  R attempts. Never close early.
- Stop when the batch is empty.

This deliberately wastes some attempts. It records the most complete tree,
which is the best first world to dream in.

### 4.2 Policy interface (what `policy.py` must implement)

```python
class Policy:
    def __init__(self, config: dict): ...                 # reads beta etc. from config
    def plan(self, ctx: PlanContext) -> Plan: ...         # W, R, reason; before the round
    def select_batch(self, view: RoundView) -> Batch: ... # ≤ W items; empty = stop
```

- `RoundView`: nodes revealed so far in this round (id, branch, attempt,
  parent, valid, fail_class, score, delta_vs_parent, tags); legal actions
  (`root` while branches < W, plus each open branch head with attempts < R);
  baseline; noise_pct; W; R.
- `PlanContext`: manifests and summaries of earlier rounds. It never sees the
  current round's outcomes.
- `Batch`: items `{parent, role: exploit|explore|recover, why}`, plus
  `closed: [{branch, why}]`.
- The policy must be deterministic, do no I/O, and decide only from the view.
  The same file runs online (`policy_step.py`) and in replay (`replay.py`).

### 4.3 Rules that keep rounds replayable

- Start points are only the round root or a branch head. Never start from an
  interior node inside a round. To build on an old interior node, make it the
  **round root** of the next round (set in the manifest).
- At most one item per branch in a batch, so a parent and its child are never
  in the same batch.
- Log every decision, including closes and stops.

## 5. Visualization

Regenerate with `tools/history.py --campaign <c>` after every step.

1. **`$DREAM_HOME/<c>/history.md`**: what workers read.
   - Header: baseline µs per shape, noise_pct, best valid node and its score,
     attempts used.
   - One table per branch in trajectory order: node, parent, mechanism, tags,
     score, Δ vs parent, fail_class, and the "next" line from the reflection.
   - A section per earlier round, and a top-10 leaderboard with per-shape µs.
2. **`tree.html`** (`$DREAM_HOME/<c>/tree.html`):
   - The branch × attempt grid per round, cells colored by score (failures
     red, within noise grey, wins green by size), best path outlined, closed
     branches dimmed.
   - Best score vs. cumulative attempts, one line per round.
   - Per-shape µs of the best node vs. baseline.
   - Attempts used per round.

Publish `tree.html` as an artifact at the end of each round and when the human
asks. Give a short text summary with it:
- the best node and per-shape µs
- what worked
- what failed and why
- what the policy closed
- the proposed next step (dream, then round t+1)

## 6. Ledger (`dream/<c>/ledger`)

```
baseline.json
policies/v<N>/policy.py           active and past policies
policies/v<N>/notes.md            what changed and why (written by the policy agent)
policies/v<N>/replay.json         replay scores per round × beta (written by replay.py)
policies/ACTIVE                   "v<N>"
rounds/r<tt>/manifest.json        policy version, beta, W, R, round root, reasons, root re-measure
rounds/r<tt>/decisions.jsonl      one line per step: view summary, batch, closed, stop, overrides
rounds/r<tt>/summary.md
snapshots/history_r<tt>.md
```

You write `baseline.json`, `rounds/` and `snapshots/`. The policy agent writes
`policies/v<N+1>/`. Only you change `ACTIVE`, and only after a dreaming phase
ends (§7).

## 7. Dreaming between rounds

When a round is done and the human agrees:

1. Spawn one subagent with `POLICY_DEV.md` and these inputs: `CAMPAIGN`,
   `LEDGER=$DREAM_HOME/<c>/ledger`, current version `v<N>`, rounds available
   for replay, and M (`campaign.yaml: dreaming.revisions`).
2. It returns the winning version and its replay scores. Check that
   `replay.json` exists for every candidate and that the winner's mean V is ≥
   the current policy's.
3. Set `ACTIVE` to the winner, commit, and tell the human what changed in
   plain words, quoting `notes.md`.

## 8. Rules

- Never override the policy's batch or stop decision. If you think it is wrong,
  write that in `decisions.jsonl` as a comment and tell the human. Hand-picked
  decisions make the round useless for replay.
- Never edit op code, the test, or the eval tool. If shared infrastructure
  breaks, stop and ask the human.
- Never push. Never move `dream/*/n/*` tags. Never rewrite a worker's commit.
- Never run `pkill`/`kill`/`killall` on processes you didn't start, and never
  reset the device yourself. The eval tool does that.
