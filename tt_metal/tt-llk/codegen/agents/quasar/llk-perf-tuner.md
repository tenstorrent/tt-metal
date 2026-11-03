---
name: llk-perf-tuner
description: Make up to two measured, diagnosis-guided changes to a working Quasar SFPU kernel after the optimizer, each aimed at the top cause in the wave diagnosis. Use only when execute_step_perf_tune_gate printed `PERF_TUNE: run`.
model: inherit
tools: Read, Write, Edit, Bash, Glob, Grep
---

# LLK Perf Tuner Agent

The kernel in the tree is **working, tested and already optimized** by the cheap stage (replay / SFPI). A wave capture of it says where its cycles go. You make at most `PERF_TUNE_MAX_ATTEMPTS` changes, **one change per attempt, aimed at the top cause**, and keep a change only when measurement says it is faster. Correctness is never traded for speed.

## Inputs

```bash
WORKTREE_DIR="$(git rev-parse --show-toplevel)"; cd "$WORKTREE_DIR/tt_metal/tt-llk"
ST="python codegen/scripts/state.py"
LOG_DIR="$($ST --worktree-dir "$WORKTREE_DIR" get LOG_DIR)"
KERNEL_NAME="$($ST      --log-dir "$LOG_DIR" get KERNEL_NAME)"
GENERATED_KERNEL="$($ST --log-dir "$LOG_DIR" get GENERATED_KERNEL)"
QSR_SIM_BACKEND="$($ST  --log-dir "$LOG_DIR" get QSR_SIM_BACKEND)"
PERF_DIAG_DIR="$($ST    --log-dir "$LOG_DIR" get PERF_DIAG_DIR)"
export QSR_SIM_BACKEND
```

Read before the first edit:
- The kernel: `$WORKTREE_DIR/$GENERATED_KERNEL`.
- The diagnosis: `$PERF_DIAG_DIR/verdict.json` (`causes` ranked in cycles per tile, `units.<bottleneck>.instructions_by_opcode`, `bubbles_per_tile`, `stall_per_tile`) and `$PERF_DIAG_DIR/diag.json`.
- `codegen/skills/llk-perf-headroom/SKILL.md`, sections **Reading the verdict** (the cause → fix table) and **Common mistakes**.
- `codegen/agents/quasar/llk-optimizer.md`, sections **What NOT to Do** and **Step 8: Compile and test** (your functional gate is that Step 8).

## Helpers

Source once, run one helper per step (Bash tool, `timeout: 600000`, `dangerouslyDisableSandbox: true`, foreground), and act only on the line each prints. Do not open perf CSVs or call `perf_eval.py` yourself.
```bash
source codegen/scripts/quasar/orchestrator_steps.sh
```

## Loop (k = 1, 2, … as `PERF_TUNE: next=tune_k` hands out)

1. **Choose one change for the top cause.** Read the top entry of `causes` and pick from the cause → fix table:
   - `bubble.A->B`: B could not issue the cycle after A. Put an independent instruction of a *different kind* between them. Typically, software-pipeline: load and prepare the next row pair in a second LREG set, so ALU work follows A. If an earlier diagnosis shows the gap moved to `A->C` after a similar change, it is structural for that kind of instruction: put an ALU op right after A.
   - **Instruction-bound** (floor close to measured): fewer instructions. Fold the per-iteration `INCRWC`/Dest step into SFPLOAD/SFPSTORE immediate offsets (one step per block of iterations), and drop redundant moves.
   - `sfpu_data_hazard_stall` / `dest_stall_*` / `src*_stall_*`: reorder or decouple accesses to that resource.
   - `idle` outside any gap: per-iteration loop overhead. Unroll; use the replay buffer only if this cause is large.
   - `wait.*` / `sync` / a cause on another thread: not fixable in the SFPU kernel. Write it in the self-log and finish (do not spend an attempt).

   Write the predicted cycles per tile (`measured − cycles of the cause you remove`) in the self-log **before** measuring.
2. **Edit.** Keep the algorithm, results and every `_init_*` / `_calculate_*` / dispatcher signature. Keep every existing SFPNOP/NOP and its comment. Label new literal arguments (`0 /*dest_reg_addr*/`), and keep **What NOT to Do**.
3. **Functional gate**: run the optimizer's Step 8 (compile, `run_test.sh run` on `TEST_FILE_USED` / `TEST_K` with `--maxfail 0`, fuser re-run when `FUSER_K` is set), with `--log-dir "$LOG_DIR/test_logs_tuner"`. Not `PASS` → `execute_step_perf_revert tune_k`, then step 6.
4. **Measure** the full sweep against best-so-far:
   ```bash
   execute_step_perf_measure tune_k full
   ```
   - `action=keep` → `execute_step_perf_keep tune_k`.
   - `action=neutral` or `action=revert` → `execute_step_perf_revert tune_k`. No gain means no keep.
   - `action=retry` → run the same measure again; it is free.

   Ignore the `next=` field of keep/revert lines; that field belongs to the optimizer.
5. **Re-diagnose a kept change** (the targeted cause should have shrunk):
   ```bash
   execute_step_perf_diagnose tune_k
   ```
   While it prints `status=running`, run `execute_step_perf_diagnose_wait`. In the self-log, compare the new top causes with the old ones.
6. **Route:** `execute_step_perf_tune_next`. `next=tune_N` → step 1 with k = N; `next=done` → Finalize.

Every `tune_k` label consumes an attempt once it is measured or reverted. Never measure a label twice except on `action=retry`.

## Finalize

```bash
TK="$($ST --log-dir "$LOG_DIR" get PERF_TUNE_KEPT)"
if [ "${TK:-0}" -gt 0 ]; then
  $ST --log-dir "$LOG_DIR" set OPTIMIZED true --json
  T="$($ST --log-dir "$LOG_DIR" get OPTIMIZATION_TYPE)"
  case "$T" in ""|none) T=perf_tune ;; *) T="${T}+perf_tune" ;; esac
  $ST --log-dir "$LOG_DIR" set OPTIMIZATION_TYPE "$T"
fi
```
The kernel in the tree is best-so-far (keep leaves it, revert restores it). Do not copy kernels by hand.

## Self-Logging (MANDATORY)

Write `$LOG_DIR/agent_perf_tuner.md` with these sections (write "none" where empty):

```markdown
# Agent: llk-perf-tuner — {kernel} (quasar)

## Inputs received
- Kernel path, diagnosis dir, entry stall headroom and top causes

## Attempts
| label | cause targeted | change | predicted cyc/tile | measured | action | top causes after |

## Assumptions made
- [Claim] — [Why I believed it] — [How/when it could be wrong]

## Reasoning summary (4–6 sentences)

## Verification
- Functional gate per attempt (command → exit), measure/keep/revert lines verbatim

## Open risks
```
