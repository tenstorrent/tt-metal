# Data Flow Optimization Subagent

Open-ended hypothesis loop over kernel/op code changes: barrier placement,
CB sizing, NOC batching, kernel fusion, sharding, reuse. One workspace per
hypothesis. Runs multiple workspaces in parallel when the main optimizer
was invoked with `parallelism > 1`.

## When this subagent runs

Dispatched by `tt:optimizer` SKILL.md when:
- mode is `dataflow-optimize` (user-declared or auto-picked), AND
- a baseline profile exists, AND
- the baseline profile identifies a clear top op with a classifiable
  bottleneck tag (see `skills/profiler/interpretation.md`).

## Dispatch pattern

The main optimizer dispatches one instance per workspace via the Agent
tool. If parallelism = 1, one instance runs in the current workspace. If
parallelism > 1, N instances run concurrently, one per spawned workspace.
Each instance owns its own branch and its own trend-file section.

For parallel runs: all N Agent calls go in the **same assistant message**
so they run concurrently. Sequential dispatch forfeits the parallelism.

## Inputs per instance

- Target: unit test path + op identifier.
- Goal: per `convergence.md`.
- Baseline metric, baseline commit SHA, bottleneck tag from profile.
- Workspace root — the absolute path this instance must operate inside.
  The instance does NOT touch any other workspace.
- Hypothesis family (optional, when parallelism > 1): a label like
  "reader-batching" or "cb-depth-tuning" so parallel instances do not
  converge on the same idea. Main optimizer assigns one family per letter
  (`-a`, `-b`, `-c`).
- Research pointer: `tt:learn` note on the target op, written during the
  Prepare phase.

## Procedure

### 1. Orient

Read the baseline profile note. Read the `tt:learn` research note. Read
the current kernel / op source for the target. Read `common-wrong-turns.md`
before proposing the first hypothesis — entries there cost prior sessions
full iterations to discover. Note:
- The op-level bound class from the profile (overhead / bandwidth / compute
  / near-peak — see `skills/profiler/interpretation.md`). **Match
  hypotheses to the class.** Compute-bound levers applied to an
  overhead-bound op waste iterations.
- Which processor is the bottleneck (from the per-RISC tag).
- Which CB depths and sharding are currently in use.
- What immediate adjacent patterns (barriers, NOC calls, tile loops) exist.

If a hypothesis family was assigned, constrain exploration to that family
for the first N iterations — diversity across workspaces matters more than
depth in any one.

### 2. Hypothesize

Propose one concrete change. Write the hypothesis in one line, grounded
in the profile evidence:

> "BRISC is 2.1× TRISC1 on matmul_tile — batching 4 reads before the
> single barrier should overlap NOC with compute."

Do NOT propose multiple changes at once. One change per iteration.
Coordinated pairs are allowed only when the second edit is physically
required by the first (e.g., raising `in0_block_w` forces a smaller
`per_core_M` to fit L1, or a CB depth change requires a reader-loop
restructure to consume the new depth). Note the coordination explicitly
in the commit message so attribution stays traceable.

### 3. Implement

Edit the source. Keep the edit tight — only what the hypothesis requires.
Do not refactor surrounding code. Do not fix unrelated issues.

If a matmul fails with a kernel-level assertion (`num_blocks_x` mismatch,
CB overflow, dim assertion), add `print(pc)` before the `ttnn.linear` call
and re-run. The realized `per_core_M/N`, `out_subblock_h/w`,
`out_block_h/w` values root-cause fast — see `common-wrong-turns.md`.

### 4. Build

Invoke `tt:run` to rebuild. Kernel-only changes in `tt_metal/kernels/` skip
the rebuild (JIT-compiled at runtime) — `tt:run` knows this.

### 5. Run + profile

Invoke `tt:run` to execute the unit test (PCC check happens inside).
Invoke `tt:profiler` for device timing.

### 6. Record

- Commit: `opt(<scope>): <one-line hypothesis> — <metric> (<Δ%> vs best)`.
- Append row to the shared `trend-<scope>.md` (one file across all
  workspaces — include the workspace letter in the `WS` column) with
  utilization columns (`DRAM %`, `FLOPs %`, `Bound`, `Cores`) from the
  profile note.
- Emit one-line Claude output including the workspace letter and
  utilization snapshot.

### 7. Evaluate convergence

Apply rules from `convergence.md`:
- PCC < 0.999 → abort this workspace only. Other workspaces continue.
- Goal met → declare success, notify main, main may stop sibling
  workspaces.
- Stall → main handles the developer prompt; individual subagents do not.

### 8. Iterate

After a trial, the next hypothesis can be informed by all prior trials
(this instance's and sibling instances'). Read the trend file at each
iteration — sibling evidence is signal, not noise.

If the last 3 trials in this workspace all regressed (metric worse than
best), pause and rethink the direction instead of generating another
incremental variation. Write the rethink to the findings note so the
developer can see the reasoning shift.

## Interaction with sibling workspaces

- **Device access**: serialized by tt-device-mcp. Tests from different
  workspaces queue naturally. No extra coordination needed.
- **ccache**: shared. Parallel builds from sibling workspaces warm each
  other's caches.
- **Trend file**: single shared file in `~/.tt-agent/notes/`. Each
  workspace's rows are tagged with its letter. Writers append atomically;
  if a conflict occurs, the later writer re-reads and appends.
- **No cross-workspace code copying**: each hypothesis develops
  independently. If a sibling discovers a winning pattern, the main
  optimizer's post-session review (via `tt:code-review`) surfaces it —
  this subagent does not ingest it mid-flight.

## Output contract

On goal-met or main-requested stop:
- Winning commit SHA.
- Summary of hypotheses tried and which were productive.
- Untried hypothesis directions (useful for the developer to continue
  manually).

On PCC abort:
- Failing commit SHA + diff vs previous-best.
- Keep the failing commit; do not revert.
