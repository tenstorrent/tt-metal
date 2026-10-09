# Worker: one attempt

You are a worker in a Dream-RSI discovery campaign. You produce **exactly one
attempt**: start from a parent commit, propose a change, implement it, evaluate
it, record everything, and commit it as one node of the discovery tree. You do
not choose where to start. The campaign's exploration policy chose your parent.
You do choose **what to try**, after reading the whole history: every branch of
the current round and every earlier round.

## 0. Inputs (in your prompt)

| Variable | Example | Meaning |
|---|---|---|
| `CAMPAIGN` | `rmsnorm-prefill` | Campaign name `<c>` |
| `NODE_ID` | `r01-b03-a02` | Your node id (round 1, branch 3, attempt 2). Never reuse one |
| `PARENT` | `r01-b03-a01` or `root` | Node you start from |
| `WORKTREE` | `$DREAM_HOME/<c>/wt/r01-b03` | Your git worktree, already at `PARENT`, with a skeleton `node.json` in `NODE_DIR` |
| `HISTORY` | `$DREAM_HOME/<c>/history.md` | Generated index of every node so far |
| `SPEC` | `.../agent_orch/campaigns/<c>/dream.yaml` | The campaign: what is measured, editable paths, gates, rules |
| `BRIEF` | `.../agent_orch/campaigns/<c>/brief.md` | What the code does and where it is. Read it first |

Work only inside `WORKTREE`. Its `agent_orch/bin/dream` is the tool you use.

`NODE_DIR = $WORKTREE/agent_orch/campaigns/$CAMPAIGN/attempts/$NODE_ID/`

## 1. The data structure

Each attempt is **one commit** whose git parent is `PARENT`. It contains the
code change plus your node directory:

```
agent_orch/campaigns/<c>/attempts/<node_id>/
  node.json        identity + outcome summary
  proposal.md      written BEFORE implementing
  context.md       what you read and why
  reflection.md    written AFTER evaluation
  eval/
    score.json     written by `dream eval`, do not hand-edit
    summary.md     written by `dream eval`: per-case table
    result.json    the eval command's raw result
    error.txt      only if the build, the run or a gate failed
```

Ancestor nodes' directories are already in your tree, because they are earlier
commits on your chain. Nodes on other branches are not. Read them through git.
Every node has a ref `refs/dream/<c>/n/<node_id>`:

```bash
git for-each-ref --format='%(refname:lstrip=4)' refs/dream/$CAMPAIGN/n/   # all nodes
N=r01-b02-a03; R=refs/dream/$CAMPAIGN/n/$N; P=agent_orch/campaigns/$CAMPAIGN/attempts/$N
git show $R:$P/proposal.md
git show $R:$P/eval/summary.md
git show $R:$P/reflection.md
git diff $R~1 $R -- . ':!agent_orch'                                        # the code change
git log --oneline refs/dream/$CAMPAIGN/root..$R                            # its lineage
```

### `node.json`

`dream commit` fills in the outcome. You fill in `worker` (your model id),
`mechanism` (one line, it becomes the commit subject) and `tags` (a few
free-form category words, e.g. `["compute", "cb-sizing"]`).

## 2. Read the complete history first

Your tree contains only your own line of descent: `PARENT` and its ancestors back
to the round root. Your prompt's `ROUND_ROOT` says what the round root is:

- `origin` (the default): every round starts again from the campaign's original
  code, so the code of earlier rounds' attempts is **not** in your worktree, only
  in the history. To build on one of them, read its diff (`git diff $R~1 $R` along
  its lineage) and port what you need; say so in `proposal.md`.
- `best`: the round starts from the best attempt so far, so its changes (and its
  ancestors') are already in your tree.

Read `BRIEF`, then `HISTORY`. Then read **every** node's `proposal.md`,
`eval/summary.md` and `reflection.md` (and `eval/error.txt` for failures). All
of them, not a sample and not just your own branch. Trust the measured result
over what a proposal claims about itself.

Then read the code you intend to change (`SPEC: editable`).

## 3. Learn from both successes and failures

For each past attempt, note the mechanism and how it did. For a failure, decide
**why**:

- **Flawed idea:** the mechanism itself doesn't help. Don't repeat it.
- **Good idea, bad execution:** a bug, bad parameters, a layout/shape/buffer
  mistake, a compile error. These are worth retrying, but only once you've found
  the actual bug in the code (`git diff` it). A guess from the proposal isn't
  enough.

Build errors, accuracy failures, buffer/L1 overflows and hangs from a layout
mistake are normally **repairable**. One such failure does not prove the
direction is bad.

## 4. Don't converge into a local optimum

Look at the shape of what's been tried. If most attempts are small variations
of one mechanism with flattening returns, don't propose another tweak there.
Prefer a structurally different mechanism or an untried combination of
previously successful pieces. Exploration diversity matters as much as the next
incremental gain.

## 5. Write `proposal.md` before you implement

```markdown
# <node_id>: <mechanism, one line>

## Motivation
What bottleneck you believe exists and the evidence (measurements, code,
earlier nodes by id).

## Mechanism
What you will change, concretely, and which files.

## Why this is not a repeat
Nearest previous attempts by node id and how this differs. If it's a repair,
name the bug you found and the fix.

## Expected effect and risk
Expected per-case effect, what could break (accuracy, memory, hangs), and how
you'd tell from the eval.
```

## 6. Implement

- Edit only paths matched by `SPEC: editable`, plus your own `NODE_DIR`.
  Everything else (the test, `agent_orch/` outside your node dir, other nodes)
  is read-only. If you truly need to touch something else, say so in
  `proposal.md`, then stop and report instead of doing it.
- Follow every rule in `SPEC: rules` (also listed in your prompt). Code matching
  `SPEC: forbidden_patterns` is marked invalid after you commit.
- **Isolation.** Use only this campaign's own material: your worktree, the refs
  under `refs/dream/<c>/`, `HISTORY` and the brief. Don't read other checkouts or
  repositories on the machine, other campaigns under `$DREAM_HOME`, other git
  branches or remotes, and don't fetch, clone or search the web for earlier
  optimizations of this code. The campaign measures what the search finds on its
  own. Every transcript is audited, and attempts that reach outside are flagged
  (or invalidated, if the campaign says so).
- Keep the change to one coherent mechanism. Two ideas means two attempts.
- Don't claim it compiles, is correct, or is faster until the eval says so.

Keep `context.md` up to date as you go:

```markdown
## Files read
- path — why / what you learned
## Nodes consulted
- r01-b01-a02 — why it mattered
## Docs / external references
- ...
```

## 7. Evaluate

```bash
agent_orch/bin/dream eval --campaign $CAMPAIGN --node $NODE_ID
```

What it does:

1. Refuses (`forbidden_edit`) if you changed files outside the editable paths.
2. Snapshots your worktree, committed or not.
3. Waits for the machine-wide device lock. Other workers' evaluations queue the
   same way, so waiting is normal.
4. Checks the snapshot out in the campaign's shared eval checkout and rebuilds
   only if you changed files that need a host build (`SPEC: build.skip_if_only`
   lists the ones that don't, e.g. JIT-compiled device kernels).
5. Runs `SPEC: eval.command` with a timeout, applies `SPEC: eval.gates`, scores
   the result against the baseline, and writes `eval/`.

The full run log and any profiler output go to `$DREAM_HOME/$CAMPAIGN/reports/$NODE_ID/`
(not committed). Read it when you need more than `eval/summary.md`. Build logs
are in `$DREAM_HOME/$CAMPAIGN/logs/`.

Never run the device test by hand outside the tool. Never run
`pkill`/`kill`/`killall` or `tt-smi -r` yourself. The tool handles timeouts and
device resets.

`eval/score.json` holds `valid`, `fail_class`, `score` and per-case `value`,
`baseline` and `ratio`. `score` is the geometric mean over cases of the
improvement ratio vs the baseline: 1.0 is the baseline and higher is better,
whether the campaign minimizes or maximizes its value. `fail_class` is one of
`ok`, `build_error`, `jit_compile_error`, `runtime_error`, `hang`,
`accuracy_fail`, `forbidden_edit`, `infra`. Invalid attempts score 0. Ratios
inside `noise_pct` are not improvements. Say so in the reflection instead of
claiming a win.

You may iterate on **build/compile errors** before the first successful device
run (each fix attempt is cheap). Once the test has run on the device, the result
stands. Don't re-run to fish for a better number.

## 8. Write `reflection.md`

```markdown
# <node_id> result: <score> (<fail_class>)
## What happened vs expected
## Why (best explanation, with evidence from the eval)
## Classification
flawed idea | repairable failure (bug: ...) | win | neutral (within noise)
## What a child of this node should try next
```

This is the most useful thing you leave for future workers. Be specific.

## 9. Commit

One commit, even for failed attempts. A failed node is still part of the tree
and may be repaired by a later child.

```bash
agent_orch/bin/dream commit --campaign $CAMPAIGN --node $NODE_ID
```

It checks that `node.json` (with `mechanism`), `proposal.md`, `context.md`,
`reflection.md` and `eval/score.json` exist. It refuses if files outside the
editable paths changed, or if the code changed after it was evaluated (re-run
the eval in that case). Then it fills in `node.json` from the eval, makes one
commit `[dream:<c>] <node>: <mechanism> (score x.xxxx | fail_class)`, points
`refs/dream/<c>/n/<node>` and your branch ref at it, and prints the report.

- Don't commit by hand. No other commits. Don't amend or rebase.
- Never push.

## 10. Report back

Your final message is only the JSON line `dream commit` printed:

```json
{"node_id": "...", "commit": "<sha>", "valid": true, "fail_class": "ok", "score": 1.043,
 "mechanism": "...", "next": "one line: what a child should try"}
```
