# Iteration Subagent

Runs one hypothesis per iteration, per workspace. Dispatched by
`tt:optimizer/SKILL.md` after Baseline, in one of two modes.

## Modes

| Mode | Edits | Rebuild per trial | Hypothesis source |
|---|---|---|---|
| `parameter-search` | Test / config / call-site args | No (Python only) | Enumerated config space |
| `dataflow-optimize` | Op source, kernel code | Yes | Open-ended, grounded in profile |

For `parameter-search`, use when the op has an enumerable config space
(block sizes, shard specs, CB depths, fidelity). For `dataflow-optimize`,
use for open-ended code changes (barriers, CB sizing, NOC batching,
sharding, fusion).

## Inputs

- Target: unit-test path (+ optional `-k` filter) and op identifier.
- Goal: per `convergence.md`.
- Baseline metric + commit SHA + bottleneck tag from the baseline profile.
- Workspace root — this instance operates only inside this path.
- **parameter-search**: enumerated config space (developer-supplied or
  derived from `tt:learn`).
- **dataflow-optimize** parallel only: a hypothesis family label (`-a`,
  `-b`, `-c`) to diversify parallel workspaces.

## Procedure

### 1. Orient

- Read the baseline profile note, the Prepare note, and `tt:learn` notes.
- Read `playbook.md` — entries there cost prior sessions full iterations.
- Identify the op-level bound class (overhead / bandwidth / compute /
  near-peak) from `skills/profiler/interpretation.md`. **Match levers to
  the class** — compute-bound levers on an overhead-bound op waste
  iterations.
- Note which processor is the bottleneck (per-RISC tag), current CB
  depths, current sharding.

### 2. Hypothesize

One concrete change, grounded in profile evidence. Write the hypothesis
in one line:

> "BRISC is 2.1× TRISC1 on matmul_tile — batching 4 reads before the
> single barrier should overlap NOC with compute."

**One variable per iteration.** Coordinated pairs are allowed only when
the second change is physically required by the first (e.g., raising
`in0_block_w` forces smaller `per_core_M` to fit L1). Note the coordination
in the commit message so attribution stays traceable.

### 3. Implement

Edit the source (dataflow) or test/config (parameter-search). Keep the
edit tight — only what the hypothesis requires. Do not refactor or fix
unrelated issues.

**Comment hygiene:** see `playbook.md` § "Comments explain the current
invariant only". Re-read every touched comment block at end of edit.

On matmul kernel-level assertions, apply the `print(pc)` recipe per
`playbook.md` § "Print the realized program config on matmul crashes".

### 4. Build

Invoke `tt:run` to rebuild. For when rebuild is skipped (kernel-only
edits), see `knowledge/recipes/tt-metal/build.md` § Kernel iteration.
parameter-search rarely needs a rebuild after the first trial.

### 5. Run + profile

Invoke `tt:run` to execute the unit test (PCC check runs inside). Invoke
`tt:profiler` for device timing.

### 6. Record

- Commit: `opt(<scope>): <one-line hypothesis> — <metric> (<Δ%> vs best)`.
- Append row to `trend-<scope>.md` (format in `convergence.md`) with
  utilization columns: `DRAM %`, `FLOPs %`, `Abs TFLOPs`, `Bound`, `Cores`.
  For parallel workspaces include the workspace letter in the `WS` column.
- Emit one line to the developer in real time:

  ```
  Iter <n> [<ws>] <sha>: <metric> (baseline <B>, Δbest <X%>, best@iter <m>) · <FLOPs%>F / <DRAM%>D / <Bound>
  ```

  Example:
  ```
  Iter 7 [a] abc1234: 8.2ms (baseline 12.1ms, Δbest -3%, best@iter 5) · 44%F / 18%D / overhead
  ```

### 7. Evaluate

Apply `convergence.md`:
- PCC < 0.999 → abort this workspace only. Others continue.
- Goal met → declare success; main may stop sibling workspaces.
- Stall → main handles the developer prompt; subagents do not.

### 8. Next iteration

Next hypothesis can be informed by all prior trials (self + siblings).
Read the trend file at each iteration — sibling evidence is signal.

If 3 consecutive trials in this workspace regressed, pause and rethink
direction before another incremental variation. Record the rethink in the
findings note.

## Forensic commits

Keep failed trials as commits — do not revert. Label via the commit
subject's first word:

- `opt(<scope>): forensic — <what was tried>` for kept-but-failed trials
- `opt(<scope>): revert — <...>` for explicit reverts on the same branch

The trend-file "Forensic failures" table is regenerated from `git log`
by filtering on these prefixes.

## Parallel workspace interaction

- **Device access** serialized by tt-device-mcp. No extra coordination.
- **ccache** shared. Parallel builds warm each other's caches.
- **Trend file** shared — each workspace's rows tagged with its letter.
  Writers append; conflicts resolved by re-read-then-append.
- **No cross-workspace code copying** mid-flight. If a sibling finds a
  winning pattern, `tt:code-review` surfaces it post-session.

## Output contract

On goal-met or main-requested stop:
- Winning commit SHA.
- Hypotheses tried, which were productive.
- Untried directions (for manual continuation).

On PCC abort:
- Failing commit SHA + diff vs previous-best.
- Commit kept; do not revert.
