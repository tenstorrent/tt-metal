# agent_orch: Dream-RSI style op optimization, stored in git

These are the instructions for running a Dream-RSI discovery loop (Zheng et al.,
"Dream-RSI: Recursive Self-Improvement through Evolving Worlds") on a tt-metal
op. We change one thing from the paper: **the discovery tree lives in git**.
Every attempt is a commit and every exploration branch is a git branch. The
per-attempt metadata (proposal, files read, scores, reflection) is committed
next to the code it describes.

Walkthrough of the loop and the dreaming phase, with examples on our op:
https://claude.ai/artifact/WHuavi6DpP5TLkN2o7xnWn

## The loop

1. **Round 1 starts with a hand-written policy (v0).** It opens W branches
   and refines each one R times. This produces the first recorded tree.
2. **Dreaming.** The policy-development agent (fixed LLM, fixed prompt)
   revises the policy code and replays every revision on every recorded tree.
   The best-scoring version becomes v1. The current policy is one of the
   candidates, so if nothing beats it, it stays.
3. **Round 2 runs on v1** and records a new tree.
4. **Dreaming again** on all recorded trees gives v2, and so on.

Only the policy code changes between rounds. Workers, the evaluator and the
policy-development agent stay fixed.

## Who reads what

| File | Read by | Job |
|---|---|---|
| `ORCHESTRATOR.md` | orchestrator (one per campaign) | Runs rounds: asks the policy for the next batch, spawns workers, verifies their commits, keeps the ledger, shows the human progress, triggers dreaming |
| `WORKER.md` | each worker subagent | One attempt: read all history, propose, implement, evaluate, reflect, commit |
| `POLICY_DEV.md` | policy-development agent | Dreaming: revise `policy.py`, replay it on recorded trees, keep the best |
| `campaigns/<c>/campaign.yaml` | all | Target op, test, metric, accuracy gate, allowed paths, costs |

## Paper → git mapping

| Dream-RSI | Here |
|---|---|
| Root `r` | Campaign root commit, tag `dream/<c>/root` |
| Node (one generate + evaluate attempt) | One commit: code change + `campaigns/<c>/attempts/<node_id>/` |
| Primary parent | The commit's git parent |
| Saved workspace | The commit's tree (`git worktree add … <tag>`) |
| Score, diagnostics, `error.txt` | `attempts/<node_id>/eval/score.json`, `eval/error.txt` |
| `proposal.md` | `attempts/<node_id>/proposal.md` |
| Branch (chain of refinements from the root) | git branch `dream/<c>/b/r<tt>-b<bb>`; its head is the frontier |
| Eligible start points = root + leaves | Round root + heads of open `dream/<c>/b/r<tt>-*` branches |
| Outer iteration `t`, tree `T_t` | Round `t`; all nodes `r<tt>-*` |
| Decision round `k` (one batch) | Step `k` of a round |
| Exploration policy `π_t` | `policies/v<N>/policy.py` on the ledger branch |
| Replay simulator | `agent_orch/tools/replay.py` over recorded rounds |

Node ids are `r<round>-b<branch>-a<attempt>`. For example, `r01-b03-a02` is the
second attempt on branch 3 of round 1, and `a01` is the first attempt off the
root.

## Refs

```
dream/<c>/root                 tag     campaign root (contains campaign.yaml)
dream/<c>/n/<node_id>          tag     one per attempt, immutable
dream/<c>/b/r<tt>-b<bb>        branch  one per exploration branch, head = frontier
dream/<c>/ledger               branch  orchestrator + policy agent only: policies, round manifests, decisions
```

Nothing under `dream/` is pushed unless a human asks.

## Where things live

- **git (source of truth):** code, `attempts/<node_id>/` metadata, the ledger
  (policies, replay results, decisions).
- **`$DREAM_HOME/<c>/` on the device machine:** worker worktrees, full
  profiler output per node, build logs, and the generated `history.md` and
  `tree.html`. These are large or regenerable, so they stay out of git.
  `DREAM_HOME=/localdev/$USER/dream`.

## Tools (`agent_orch/tools/`)

| Tool | Used by | Does |
|---|---|---|
| `tools/setup_campaign.sh` | orchestrator | Root tag, ledger with policy v0, eval checkout + first build |
| `tools/eval_attempt.sh` | worker, orchestrator | Snapshot the worktree, take the device lock, build if needed, run the profiled test, write `eval/`; also `--baseline` / `--measure` |
| `tools/prepare_worker.sh` | orchestrator | Create or reuse a branch worktree for one node, write a skeleton `node.json` |
| `tools/commit_node.py` | worker | Validate the node, commit + tag it, print the report |
| `tools/verify_node.py` | orchestrator | Check a returned node, record overrides / lost attempts |
| `tools/policy_step.py` | orchestrator | Load the active policy, rebuild the revealed tree from git tags, return the next batch (or stop) |
| `tools/replay.py` | policy agent | Replay a `policy.py` on recorded rounds over a beta sweep, write scores + traces |
| `tools/history.py` | orchestrator | Generate `history.md` and `tree.html` from git + the ledger |
| `tools/tests/run_tests.sh` | anyone | Replays the walkthrough example and checks V = 1.23 / 1.24 / 1.11 |
