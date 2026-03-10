# Status: 7×8 num_links = 4

**Process:** Follow hard rules and preferred implementation order in `AG_MM_7x8_4links_goal.md`. Gate to 7×8 && num_links==4 only. Rebuild after kernel changes; one change per experiment; prove num_links=4 is active. Validate with 8k ISL first (baseline: 530.09 + 1321.09 = 1851.18 µs). Rebuild: `cd tt-metal && ./build_metal.sh && ./create_venv.sh`. Profiler: `SKIP_PREFILL_WARMUP=1 USE_FUSED_AG_MM=1 ./scripts/run_profiler_sweep.sh --prompt-lengths 8k --run-name ag_mm_fused_8k` → check `prefill.csv` for fused op duration and cores.

## Subtask 1 — Analysis (done)

### Where the limit comes from

**File:** `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/all_gather_minimal_matmul_async_program_factory.cpp`

1. **Divisibility check (lines 420-422)**
   - Condition: `(transpose_core_grid ? grid_size.x : grid_size.y) % num_links == 0`
   - Message: "The number of in0 rows must be a multiple of num_links"
   - So the axis used is **grid_size.x** when `transpose_core_grid` is true, else **grid_size.y**.

2. **transpose_core_grid (line 292)**
   - `transpose_core_grid = force_transpose ? true : (M > N)`
   - For Llama FF2, M > N (e.g. 8192 > 2048) → **transpose_core_grid is always true** (whether force_transpose is true or false).
   - So for Llama we always use **grid_size.x** in the divisibility check → 7×8 gives grid_x=7 → 7 % 4 ≠ 0 → only num_links = 1 or 7; in practice num_links = 1.

3. **Mux placement (lines 432-441)**
   - `num_mux_cores = num_links * 2` (forward + backward per link).
   - Mux cores are on the **top row** of the device grid: `CoreCoord(mux_x_index, full_grid_size.y - 1)`.
   - `mux_x_index = (num_workers_per_link * (link + 1)) - (1 - dir)`; then wrapped if `mux_x_index >= full_grid_size.x`.
   - So mux placement uses **full_grid_size** (device grid), and **num_workers_per_link** is tied to the in0 parallel axis (grid_size.x when transpose).

4. **Why force_transpose=False doesn’t help Llama**
   - With force_transpose=False, transpose_core_grid = (M > N) = true for Llama → divisibility still on grid_size.x.
   - The force_transpose=False path that uses grid_y is only when **M ≤ N** (e.g. 4k4k4k). So for Llama (M > N) we never use grid_y in the current logic.

### What would need to change for 7×8 with num_links = 4

- **Option A — Special case 7×8:** When `grid_size.x == 7` and num_links == 4, use **grid_size.y** (8) for the divisibility check and for the axis that drives link/worker layout (so 8 % 4 == 0). Mux placement may need to use the same axis so that mux_x_index and worker layout are consistent with an “in0 parallel axis” of length 8 (grid_y) instead of 7 (grid_x).
- **Option B — Relax divisibility:** Allow num_links that don’t divide the in0 parallel axis and change the layout so that some links get fewer workers (e.g. 2 links with 4 workers, 2 links with 3 workers). This would require non-uniform worker counts and careful mux/core mapping.
- **Option C — Fix force_transpose=False path for M > N:** Change the formula so that when force_transpose=False we use grid_y for divisibility even when M > N (e.g. `transpose_core_grid` only from force_transpose, not from M > N). Then Llama could call with force_transpose=False and get grid_y=8 % 4 = 0. That path currently segfaults on 7×8; the segfault would need to be fixed first (likely mux/core bounds or NOC usage).

### Code touch points (for any option)

- **Program factory:** Lines 292 (transpose_core_grid), 420-422 (TT_FATAL divisibility), 423-426 (num_mux_cores, full_grid_size check), 432-441 (mux core indices), and all uses of `in0_parallel_axis_cores` / `in1_parallel_axis_cores` (296, 300, 350-356, 852-872, 1062-1063, 1118-1119) so that the “in0” axis used for link layout can be grid_y for 7×8 when we want 4 links.
- **Kernels:** Any use of core indices that assume in0 axis = grid_x; need to stay consistent with the program factory’s chosen axis for 7×8.

---

## Current status

- **Subtask 1 (Analyze):** Done (above).
- **Subtask 2 (Propose):** Done — chose Fallback 2 (padding approach): pad `in0_link_axis_cores` to 8 for 7×8+4-links so 8 % 4 == 0.
- **Subtask 3 (Implement):** Done — see changes below.
- **Subtask 4 (Test):** Next — rebuild tt-metal, run profiler with 8k ISL.

## Subtask 2 — Proposal (done)

**Approach: Fallback 2 — Padding**

For 7×8 + num_links=4, pad the link axis count from 7 to 8 so divisibility check passes (8 % 4 == 0).

- Introduce `in0_link_axis_cores` (separate from `in0_parallel_axis_cores`).
- For 7×8 + num_links=4, set `in0_link_axis_cores = 8`.
- Use `in0_link_axis_cores` for divisibility check and worker-to-link assignment.
- Keep `in0_parallel_axis_cores = 7` for tile layout (unchanged).
- Compute `num_workers_per_link_effective = 8 / 4 = 2` for mux placement.
- Link 3 has only 1 real worker (x=6), but mux placement and routing are consistent.

**Files to edit:**
1. `all_gather_minimal_matmul_async_program_factory.cpp` — add padding logic
2. `llama_ccl.py` — update `line_all_gather_matmul` to pass num_links=4 for 7×8

## Blockers / notes

- force_transpose=False path segfaults on 7×8 (likely NOC or core bounds); would need debugging before relying on it for 4-link 7×8.

## Subtask 3 — Implementation (done)

### Files changed

1. **`ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/all_gather_minimal_matmul_async_program_factory.cpp`**
   - Added `use_padded_link_axis` flag gated to `grid_size.x == 7 && grid_size.y == 8 && num_links == 4 && transpose_core_grid`
   - Added `in0_link_axis_cores` (padded to 8 when flag is true)
   - Added `num_workers_per_link_effective = in0_link_axis_cores / num_links` (= 2 for 7×8+4-links)
   - Updated divisibility check to use `in0_link_axis_cores`
   - Updated mux placement loops to use `num_workers_per_link_effective`
   - Updated worker-to-link assignment to use `num_workers_per_link_effective`
   - Added log_info message when 7×8+4-links path is taken

2. **`models/demos/llama3_70b_galaxy/tt/llama_ccl.py`**
   - Added `use_padded_link_axis` flag in `line_all_gather_matmul`
   - When flag is true: `effective_num_links = 4`, `num_workers_per_link = 8 // 4 = 2`
   - Existing fallback logic unchanged for other grids

### How to verify 4-link path is active

1. Run with `TT_METAL_LOGGER_LEVEL=INFO` and grep for: `AG+MM: 7x8+4-links special case`
2. Check profiler output: fused op duration should be ~1800-2000 µs (vs ~2100+ µs with 1-link)

## Next step

Subtask 4: Rebuild tt-metal and run profiler.

```bash
cd tt-metal
./build_metal.sh
./create_venv.sh
source python_env/bin/activate
tt-smi -glx_reset
SKIP_PREFILL_WARMUP=1 USE_FUSED_AG_MM=1 ./scripts/run_profiler_sweep.sh --prompt-lengths 8k --run-name ag_mm_fused_4link_8k
```

Check `profiler_sweep_results/ag_mm_fused_4link_8k/8k/prefill.csv` for fused op duration.


---

## Experiment log

| When | What changed | Error | Result |
|------|--------------|-------|--------|
| 2026-03-09 | Fallback 2: Pad in0_link_axis_cores=8 for 7x8+4-links | None | Build OK. First run: 2097.87 us (4-link NOT active - model passed num_links=1 default) |
| 2026-03-09 | Fix: Auto-enable 4-links for 7x8 grid (remove num_links==4 check in llama_ccl.py) | CoreRangeSet overlap | Mux cores at (1,9) duplicated due to wrapping with full_grid_size.x=7 |
| 2026-03-09 | Fix: Overflow mux cores to row 8 instead of wrapping | Hang | Mux cores on row 8 caused communication hang during prefetcher setup |
| 2026-03-09 | Try 7 links instead (7 % 7 == 0) | TT_FATAL: not enough cores | 7 links requires 14 mux cores but full_grid_size.y=10 only allows 10 |
| 2026-03-09 | Option A: Use y-axis (grid_y=8) for link grouping, mux at column 6 | Illegal NOC usage | Mux cores at x=6 conflict with worker cores at same column |
| 2026-03-09 | Option A v2: Place mux at rows 8-9 (outside worker grid) | Hang | 4-link path activated, mux placement OK, but hangs during prefetcher setup |

## Conclusion: 4-link not feasible for 7×8 with current fused op design

**Root cause:** The fused AllGather+Matmul op has fundamental constraints that prevent multi-link operation on 7×8 grid:

1. **Divisibility constraint:** `in0_parallel_axis_cores % num_links == 0`
   - For 7×8 with `transpose_core_grid=true`, `in0_parallel_axis_cores = 7`
   - Divisors of 7 are only 1 and 7

2. **Mux core constraint:** `full_grid_size.y >= num_links * 2`
   - Device has `full_grid_size = (7, 10)`
   - 7 links requires 14 mux cores, but only 10 rows available
   - Maximum links = 10 / 2 = 5, but 7 % 5 ≠ 0

3. **Padding approach failed:**
   - Padding `in0_link_axis_cores` to 8 for divisibility causes mux placement conflicts
   - With `full_grid_size.x = 7`, mux positions wrap and overlap
   - Overflowing to a second row causes communication hangs

**Options for future work:**
- Redesign mux placement to support non-contiguous positions
- Allow uneven worker distribution per link (some links with fewer workers)
- Use a different AllGather topology that doesn't require as many mux cores

**Current state:** Reverted to 1-link operation for 7×8 grid. The fused op works but doesn't provide the bandwidth benefit of multi-link.

## Resume instructions

```bash
cd tt-metal
source python_env/bin/activate
tt-smi -glx_reset
SKIP_PREFILL_WARMUP=1 USE_FUSED_AG_MM=1 ./scripts/run_profiler_sweep.sh --prompt-lengths 8k --run-name ag_mm_fused_4link_v2
```

Check `profiler_sweep_results/ag_mm_fused_4link_v2/8k/prefill.csv` for fused op duration.
- Baseline: 1851.18 µs (AG 530.09 + MM 1321.09)
- Previous fused (1-link): 2097.87 µs
- Expected with 4-link: ~1800-2000 µs or better
