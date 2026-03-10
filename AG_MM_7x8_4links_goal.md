# Goal: Fused kernel 7×8 grid with num_links = 4

## Objective
Improve the fused AllGather+Matmul kernel so that on the **7×8 grid** it can use **num_links = 4** (like the baseline non-fused path). Llama 70B uses 7×8 only; today fused is limited to num_links = 1 on 7×8, so AG bandwidth is low and baseline (4-link ring AG) is faster. Enabling 4 links on 7×8 in the fused op would remove this limitation and improve Llama prefill perf.

## Why it's limited today
- Fused op uses **`grid_x % num_links == 0`** (with `force_transpose=True`) for link/core layout.
- For 7×8, `grid_x = 7` (prime) → only num_links = 1 or 7; in practice num_links = 1.
- Baseline uses ring_all_gather with 4 links and does not tie link count to the 7×8 matmul grid.

## Constraints
- No regression on 8×8 or other grids; existing tests and PCC must pass.
- Changes in fused op (program factory / kernel) only; Llama model stays on 7×8.

## Hard rules
- **Never skip rebuild** after kernel-related source changes.
- **Never test multiple unrelated changes** in one experiment (one change, one experiment).
- **Never keep rejected experiments** in the branch (revert or drop before next try).
- **Always prove** whether num_links=4 is actually active (e.g. profiler / logs / assertions).
- **After each change or experiment,** update the status file with one short line in the **Experiment log**: what changed, what didn’t work or error (if any), result. Keep a running note of every experiment so we know what was tried.

## Preferred implementation order
1. **Understand exact layout restriction** — Document where link count is tied to x-grid divisibility (done in Subtask 1).
2. **Find smallest place to decouple** — Change the minimal code so link count is not forced by grid_x divisibility for 7×8.
3. **Gate the new path** — Apply the change only for **7×8 && num_links==4** (no impact on 8×8 or other configs).
4. **Rebuild tt-metal** — Full rebuild after any kernel/program-factory change.
5. **Run profiler** — Use `scripts/run_profiler_sweep.sh`; `prefill.csv` gives both prefill and fused-op duration. Start with **8k ISL only**.
6. **Compare with baseline** — Fused-op duration vs non-fused baseline (e.g. 8k: AG 530.09 µs + MM 1321.09 µs = **1851.18 µs**); confirm fused is in the ballpark or better when num_links=4 is active.

## Rebuild and profiler commands

**Rebuild tt-metal** (after any kernel-related source change; build must not fail). From repo root (e.g. `teja`), `tt-metal` is the working folder:

```bash
cd tt-metal   # or cd teja/tt-metal if you are in the parent of teja
./build_metal.sh
./create_venv.sh
```

Only proceed to profiler after a successful build.

**Before every run** — if devices are stuck or stale from a previous run, reset them:

```bash
tt-smi -glx_reset
```

**Before running profiler** — go to the working folder and activate the venv:

```bash
cd tt-metal   # or cd teja/tt-metal if you are in the parent of teja
source python_env/bin/activate
```

**Profiler run** (fused op path; env: `USE_FUSED_AG_MM=1`). From `tt-metal` with venv active. If devices are stuck or stale, run `tt-smi -glx_reset` before this:

```bash
SKIP_PREFILL_WARMUP=1 USE_FUSED_AG_MM=1 ./scripts/run_profiler_sweep.sh --prompt-lengths 8k --run-name ag_mm_fused_8k
```

- Output folder name (e.g. `ag_mm_fused_8k`) can be changed as needed.
- Generates `prefill.csv` in the run output folder (e.g. `profiler_sweep_results/ag_mm_fused_8k/8k/prefill.csv`).
- In `prefill.csv`, check **AG+MM fused op duration** (e.g. `AllGatherMinimalMatmulMatmulAsyncOp`), cores, and that num_links=4 is reflected in the run.

## Subtasks (in order)
1. **Analyze** — In the fused op (program factory + kernel), document where and why `grid_x % num_links` is used and how link/core layout is done. Identify all code paths that need to change for 7×8 with 4 links.
2. **Propose** — Propose a concrete change (e.g. use grid_y for 7×8, fix force_transpose=False path, or new layout) so 7×8 can use num_links = 4.
3. **Implement** — Implement the change with minimal diff; list files and line ranges.
4. **Test** — Rebuild tt-metal; run profiler (e.g. `run_profiler_sweep.sh` for 8k); use `prefill.csv` for fused-op duration; compare with baseline (8k: 530.09 + 1321.09 = 1851.18 µs); **prove** num_links
