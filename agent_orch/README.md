# agent_orch: Dream-RSI campaigns for tt-metal ops

Point it at a tt-metal op and a test that measures it. It runs a search in which
Claude Code agents propose, implement and measure optimizations, and an
exploration policy decides where to spend the attempts. Between rounds the
policy improves itself by replaying the recorded rounds ("dreaming", after
Zheng et al., *Dream-RSI: Recursive Self-Improvement through Evolving Worlds*).
You get a live report page and, at the end, a branch with the best change.

## Quick start

In Claude Code: `/dream`. Install it once with `agent_orch/bin/dream install-skill`, which links
`agent_orch/skill/dream/` into `~/.claude/skills/`. The skill walks you through the steps below,
publishes the report as a live page, and keeps it updated while your session is open.

By hand:

```bash
agent_orch/bin/dream init my-op --machine bh-qbge-15:48486   # scaffold agent_orch/campaigns/my-op/
$EDITOR agent_orch/campaigns/my-op/{dream.yaml,brief.md}    # what to optimize, how to measure it
git add tests/.../test_my_op.py && git commit                # the campaign starts from HEAD
agent_orch/bin/dream policies                                # pick search.policy
agent_orch/bin/dream check my-op     # set up the machine, build, measure the baseline (no model cost)
agent_orch/bin/dream start my-op     # runs in the background on the machine
agent_orch/bin/dream watch my-op     # one line per update; keeps ./dream-my-op.html current
agent_orch/bin/dream fetch my-op     # at the end: brings the dream/my-op/best branch here
```

`stop`, `resume`, `status`, `report`, `finalize`, `export-policy` and `delete` do
what they say; `dream --help` lists them. Commands run on the campaign's machine:
locally, or over ssh when `machine:` names another host (for an IRD reservation,
`host:port` with the reservation's ssh port; your ssh key must be authorized there).

## 1. The evaluation

You write the test. The campaign runs `eval.command` in a checkout of each
attempt, and the command writes one JSON object to `$DREAM_RESULT`:

```json
{"valid": true,
 "cases": {"h3584": {"value": 11.19, "pcc": 0.9999985},
           "h7168": {"value": 15.84, "pcc": 0.9999983}},
 "error": null}
```

- `valid`: the test's own correctness verdict. `false` makes the attempt invalid.
- `cases`: one entry per shape or configuration; `value` is the number being
  optimized. Other numbers are free-form and can be checked by `eval.gates`.
  A single case may use a top-level `"value"` instead.
- Optional `fail_class` hint (`jit_compile_error`, `runtime_error`, ...) and `error` text.

The score of an attempt is the geometric mean over cases of its improvement
ratio vs the baseline (`baseline/value` when minimizing): 1.0 is the baseline
and higher is better. The baseline is the median of `eval.baseline_runs` runs of
the unchanged code, and their spread sets the noise band.

**For a tt-metal op you normally don't write any of this.** The adapter
`agent_orch/adapters/ttmetal_op_perf.py` runs a pytest test under Tracy, one
case per pytest parameter set, and reports the op's per-chip mean
`DEVICE KERNEL DURATION` in µs:

```yaml
eval:
  command: >-
    python agent_orch/adapters/ttmetal_op_perf.py --test tests/.../test_my_op.py
    --op-code MyOpDeviceOperation --warmup 3 --measured 10
    --extra 'pcc=PCC: ([0-9.eE+-]+)'
  gates: ["pcc >= 0.99999"]
```

The test only has to call the op `warmup + measured` times per case and print
its accuracy, either with `--extra name=REGEX` as above or as
`DREAM_METRIC pcc=0.9999985 max_abs=0.02` lines. A failing pytest marks the
attempt invalid. See `campaigns/rmsnorm-prefill/` for a complete example.

## 2. The spec (`agent_orch/campaigns/<name>/dream.yaml`)

Required: `name`, `editable`, `eval.command`. Everything else has a default
(`tools/dream/campaign.py: DEFAULTS`).

| Field | Default | Meaning |
|---|---|---|
| `machine` | `local` | `local`, `host` or `host:port`; where builds, tests and agents run |
| `repo` | this checkout's path | The repo path on the machine |
| `editable` | (required) | fnmatch globs workers may change |
| `brief` | none | Markdown every worker reads first: what the code does, where it is |
| `rules`, `forbidden_patterns` | `[]` | Told to workers / regexes that invalidate an attempt |
| `eval.direction`, `unit`, `gates`, `timeout_s` | `minimize`, `""`, `[]`, 900 | |
| `eval.baseline_runs`, `min_noise_pct`, `drift_check` | 3, 1.0, true | Drift: the root is re-measured each round; a shift beyond noise blocks the campaign |
| `build.command`, `skip_if_only`, `check_file` | `./build_metal.sh --release --enable-ccache`, `*/kernels/*`, `ttnn/ttnn/_ttnn.so` | Rebuild only when a changed file isn't JIT-compiled |
| `budget.max_attempts`, `max_hours`, `max_usd` | 60, 8, 300 | **Hard limits.** No new attempt starts past one; running workers are cut at the time limit |
| `search.policy`, `W`, `R`, `max_rounds` | `fresh`, 4, 4, 6 | Starting policy (`dream policies`), parallel workers per step, depth, rounds |
| `search.round_root` | `origin` | `origin`: every round starts from the campaign root (as in the paper); `best`: from the best attempt so far |
| `dreaming.enabled`, `revisions`, `cost_per_attempt`, `parallel_bonus` | true, 5, 0.005, 0.01 | Policy improvement between rounds |
| `models.worker`, `policy_dev`, `summary` | `opus`, `opus`, `sonnet` | |
| `isolation.enabled`, `on_violation` | `true`, `flag` | Workers see only this campaign (below); `invalidate` also scores audited attempts 0 |

## 3. What a campaign does

`dream start` launches the driver (`tools/dream/driver.py`) on the machine. It
is a deterministic loop, not an agent:

1. **Round t** starts a new tree. With `round_root: origin` (the default, as in
   the Dream-RSI paper) its root is the campaign root, so rounds are independent
   searches that differ only in the policy, and earlier rounds reach workers as
   history (they may port ideas from them). With `round_root: best` it starts from
   the best attempt so far, so improvements compound across rounds.
   The root is re-measured; drift beyond the noise band blocks the campaign.
2. **Each step**, the active policy picks up to W start points (open a new
   branch from the round root, which is always allowed, or refine a branch's newest
   attempt) and may close branches. One worker
   (a headless Claude Code session following `WORKER.md`) runs per start point,
   in parallel; evaluations queue on a machine-wide device lock. The driver
   verifies every commit, regenerates `history.md` and the report, commits the
   ledger. A worker that never commits is retried once, then its branch is closed.
3. **End of round:** a short session writes the round summary and the
   "worked / dead ends / open leads" lists for the report.
4. **Dreaming:** a policy-development session (`POLICY_DEV.md`) writes candidate
   policies and replays them on every recorded round. The driver re-runs the
   replay itself and adopts the winner only if it scores at least as well.
5. Repeat until `max_rounds` or a budget limit, then **finalize**: the best
   node's code change, squashed onto the commit you started from, becomes the
   branch `dream/<name>/best`.

Everything is resumable: `dream stop` / `dream resume` (or a crash) continue
from the last finished worker.

## 4. Where things live

**git** (source of truth), all under `refs/dream/<name>/`, invisible to `git branch` and `git tag`:

```
base          your HEAD when the campaign was created
root          base + agent_orch/campaigns/<name>/ (spec, brief)
n/<node>      one commit per attempt: the code change + attempts/<node>/ (proposal, eval, reflection)
b/r01-b03     tip of each exploration branch
ledger        policies, baseline, round manifests, decisions, costs, summaries
```
plus the one visible branch `dream/<name>/best`. `dream fetch` copies all of them
from the machine. Read a node with `git show refs/dream/<name>/n/<node>:<path>`.

**`$DREAM_HOME/<name>/` on the machine** (default `/localdev/$USER/dream`):
`ctl/` (the tools, pinned at the campaign root), `eval/` (the one build + test
checkout), `wt/` (worker worktrees), `ledger/`, `reports/` (full test output per
attempt), `logs/` (driver, builds, every session transcript), `report/index.html`.

### Isolation

A campaign should measure what its search finds on its own, not what an agent
can dig up from earlier work on the same machine. With `isolation.enabled` (the
default):

- **Its own repo.** The campaign lives in `$DREAM_HOME/<name>/repo.git` on the
  machine: the start commit and its history plus `refs/dream/<name>/*`, with no
  remotes, branches, tags or other campaigns (objects are shared with your repo
  through git alternates, so it costs no disk). Worker worktrees, the eval
  checkout and the ledger all come from it, so `git log --all` in a worktree
  shows nothing else. `dream fetch` copies the results into your repo.
- **A rule.** Every worker is told to use only the campaign's own material.
- **An audit.** After each attempt the driver scans the worker's transcript for
  reads of your checkout (other than `python_env`) or other campaigns, git
  fetch/clone/remote, curl/wget/gh and web tools. Hits are flagged with ⚑ in the
  report and `history.md`; with `on_violation: invalidate` the attempt scores 0.

Workers still run with full permissions on a shared machine, so this blocks the
easy paths and makes the rest visible; it is not a sandbox. Keep briefs free of
pointers to earlier results unless you want the campaign to start from them.

## 5. The report

One page per campaign, the same layout for every campaign, regenerated after
every step (`report/template.html`, filled by `tools/dream/report.py`): status
and budget, the best result per case with its lineage and branch, the discovery
grid, best score vs attempts, what worked / dead ends / open leads, every round,
the policy versions with their replay scores, and the spec and costs. The
`/dream` skill publishes it as a live artifact.

## 6. Policies

A policy (`policy.py`, interface in `tools/dream/policy_api.py`) decides where
attempts start, how many run in parallel and when to stop; never what code to
write. Policies are about search, not about one op, so learned ones are shared:

```bash
agent_orch/bin/dream policies                                        # the library
agent_orch/bin/dream export-policy my-op --version v3 --as my-op-v3 \
    --description "what it does differently"                          # add yours; then commit it
```

`fresh` (= `parallel-refine`) explores exhaustively and records the most
complete first tree; a learned policy prunes harder and spends fewer attempts.

## 7. Files

| Path | What |
|---|---|
| `bin/dream` | CLI entry point |
| `skill/dream/SKILL.md` | The `/dream` Claude Code skill (`dream install-skill`) |
| `WORKER.md`, `POLICY_DEV.md` | Instructions for worker and policy-development sessions |
| `adapters/ttmetal_op_perf.py` | Eval adapter for tt-metal op tests |
| `campaigns/<name>/` | Spec + brief of each campaign |
| `policies/library/` | Shared exploration policies |
| `report/template.html` | The report page |
| `tools/dream/` | `cli` (commands), `driver` (the loop), `evaluate` + `scoring` (the eval contract), `gitops` (refs, worktrees, commits), `steps` (policy online), `replay` + `policy_api` + `tree` (dreaming), `agents` (Claude sessions), `history`, `report`, `policy_lib` |
| `tools/tests/` | `run_tests.sh` (replay fixture), `regress_export.sh` (the 58-attempt rmsnorm campaign, exact replay of v0-v5), `e2e/run_e2e.sh` (a whole campaign offline with a stub `claude`) |
