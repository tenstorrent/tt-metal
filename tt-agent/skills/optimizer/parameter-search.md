# Parameter Search Subagent

Structured sweep over a discrete parameter space. One workspace, one build,
many trials with different config values. Use when the target op has an
explicit config struct (block sizes, shard specs, CB depths, fidelity,
dtype) and the search is over valid combinations of those fields.

## When this subagent runs

Dispatched by `tt:optimizer` SKILL.md when:
- mode is `parameter-search` (user-declared or auto-picked), AND
- a baseline profile exists, AND
- the target has an enumerable config space.

Never use this for kernel code edits — use `dataflow-optimize.md`.

## Inputs from caller

- Target: unit test path + op identifier.
- Goal: per `convergence.md`.
- Config space: a list of parameter names and their candidate values. Either
  the developer supplied it, or `tt:learn` was invoked to identify the
  config struct and its valid range. If neither, ask the developer before
  starting — do not invent.
- Baseline metric and commit SHA.

## Procedure

### 1. Plan the sweep

Build the candidate list. Strategies in order of preference:

1. **Developer-supplied grid**: use it verbatim.
2. **Coarse-to-fine**: start with endpoints + a few interior values per
   dimension. If a region shows promise, refine.
3. **Priority-ordered heuristic**: order candidates by expected impact
   (e.g., for matmul: block sizes first, then shard strategy, then fidelity).

Surface the planned sweep size up front. If it exceeds ~30 trials, warn the
developer and ask whether to proceed or narrow.

### 2. Per-trial loop

For each candidate config:

1. Edit the target test (or a test-local config override) to apply the new
   parameter values. Parameter search rarely needs to touch kernel source
   code — it changes call-site args or a config struct in the test file.
2. Run `tt:run` to execute the unit test — verify it passes correctness
   (PCC check in the test itself, if present).
3. Invoke `tt:profiler` for the device timing.
4. Read PCC — if below `pcc_abort` (0.999), trigger PCC-abort per
   `convergence.md`. Keep the commit.
5. Commit: `opt(<scope>): <param>=<value> — <metric> (<Δ%> vs best)`.
6. Append row to `trend-<scope>.md`.
7. Emit one-line Claude output.
8. Evaluate convergence per `convergence.md`.

### 3. Correctness handling

If a candidate produces incorrect output (PCC < 0.999), record the commit
for forensics, skip the rest of that candidate's refinement, and continue
with the next candidate. A single invalid config does not abort the sweep —
only a drop from a previously-passing state does.

### 4. Sweep completion

The sweep ends when:
- Goal met (success) → hand back to main.
- All candidates tried (no hypotheses left) → report to main; main asks the
  developer whether to refine, widen, or stop.
- Stall per `convergence.md` → ask developer (main handles the prompt).

## Build reuse

Parameter changes live in Python (test / config / call-site). No C++
rebuild needed per trial. Build once at the start of the subagent's run;
if a candidate changes a compile-time constant (rare in this mode),
incremental rebuild via `tt:run` handles it.

## Output contract

Each trial produces:
- One commit on the session's branch.
- One profile note (via `tt:profiler`).
- One row in the trend file.

Final return to caller:
- Best config values.
- Winning commit SHA.
- Remaining untried candidates (if any) — useful for the developer's
  narrowing decision.
