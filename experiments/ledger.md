# Experiment Ledger

This file tracks all experiments for the fused AG+MM 7×8 num_links=4 optimization.

---

## Experiment 2026-03-09-01

### Hypothesis
Pad `in0_link_axis_cores` from 7 to 8 so divisibility check passes (8 % 4 == 0). Keep `in0_parallel_axis_cores = 7` for tile layout.

### Files Changed
- `all_gather_minimal_matmul_async_program_factory.cpp`: Added `use_padded_link_axis` flag, `in0_link_axis_cores = 8` for 7×8+4-links
- `llama_ccl.py`: Added logic to set `effective_num_links = 4` for 7×8 grid

### Rebuild
- pass

### Runtime
- fail (4-link path not taken)

### Profiler
- prompt length: 8k
- fused op duration: 2097.87 µs
- num_links=4 active: **NO** (model passed num_links=1 default, condition checked `num_links == 4` which was false)
- baseline comparison: same as 1-link baseline

### Verdict
4-link not active

### Notes
- Root cause: `llama_mlp.py` calls `line_all_gather_matmul()` without passing `num_links`, defaults to 1
- The condition `num_links == 4` in `llama_ccl.py` was never true

### Next Step
Fix by removing `num_links == 4` check - auto-enable 4 links for 7×8 grid

---

## Experiment 2026-03-09-02

### Hypothesis
Auto-enable 4-links for 7×8 grid by removing the `num_links == 4` check in `llama_ccl.py`.

### Files Changed
- `llama_ccl.py`: Changed condition to `core_grid.x == 7 and core_grid.y == 8 and force_transpose` (removed `num_links == 4`)

### Rebuild
- pass

### Runtime
- fail (CoreRangeSet overlap)

### Profiler
- prompt length: 8k
- fused op duration: N/A (crashed)
- num_links=4 active: **YES** (log message appeared before crash)
- baseline comparison: N/A

### Verdict
runtime failure

### Notes
- Error: "Cannot create CoreRangeSet with specified core ranges because core ranges [(x=1,y=9) - (x=1,y=9)] and [(x=1,y=9) - (x=1,y=9)] overlap!"
- Root cause: `full_grid_size.x = 7`, so mux positions 7 and 8 wrap to 0 and 1, causing overlap with mux_id=0

### Next Step
Fix mux placement to handle overflow to second row

---

## Experiment 2026-03-09-03

### Hypothesis
Overflow mux cores to row 8 instead of wrapping within row 9.

### Files Changed
- `all_gather_minimal_matmul_async_program_factory.cpp`: Added `mux_y_offset` logic to place overflow mux cores at `y = full_grid_size.y - 2`

### Rebuild
- pass

### Runtime
- fail (hang)

### Profiler
- prompt length: 8k
- fused op duration: N/A (hung)
- num_links=4 active: **YES** (log message appeared)
- baseline comparison: N/A

### Verdict
runtime failure

### Notes
- Hung at "Running TtLlamaPrefetcherSetup"
- Mux cores at row 8 (y=8) may conflict with something or communication is broken

### Next Step
Try different approach - use 7 links instead (7 % 7 == 0)

---

## Experiment 2026-03-09-04

### Hypothesis
Use 7 links instead of 4 (since 7 % 7 == 0 satisfies divisibility).

### Files Changed
- `llama_ccl.py`: Set `effective_num_links = 7` for 7×8 grid

### Rebuild
- pass

### Runtime
- fail (TT_FATAL)

### Profiler
- prompt length: 8k
- fused op duration: N/A (crashed)
- num_links=4 active: N/A (using 7 links)
- baseline comparison: N/A

### Verdict
runtime failure

### Notes
- Error: "TT_FATAL: The are not enough cores for the number of mux cores requested"
- 7 links × 2 directions = 14 mux cores needed, but `full_grid_size.y = 10` only allows 10

### Next Step
Try Option A: Use grid_y (8) for link grouping, place mux along y-axis

---

## Experiment 2026-03-09-05

### Hypothesis
Option A: Use `grid_y = 8` for link grouping (8 % 4 == 0). Place mux cores along y-axis at `x = full_grid_size.x - 1 = 6`.

### Files Changed
- `all_gather_minimal_matmul_async_program_factory.cpp`: Added `use_y_axis_for_links` flag, mux placement at `CoreCoord(full_grid_size.x - 1, mux_index)`
- `llama_ccl.py`: Set `effective_num_links = 4`, `num_workers_per_link = 2` for 7×8 grid

### Rebuild
- pass

### Runtime
- fail (Illegal NOC usage)

### Profiler
- prompt length: 8k
- fused op duration: N/A (crashed)
- num_links=4 active: **YES** (log message appeared)
- baseline comparison: N/A

### Verdict
runtime failure

### Notes
- Error: "Illegal NOC usage: data movement kernels on logical core (x=6,y=1) cannot use the same NOC"
- Mux cores at column 6 conflict with worker cores at same column

### Next Step
Place mux cores at rows 8-9 (outside worker grid) instead of column 6

---

## Experiment 2026-03-09-06

### Hypothesis
Option A v2: Place mux cores at rows 8-9 (outside 7×8 worker grid). Use consecutive positions: (0,9), (1,9), ..., (6,9), (0,8).

### Files Changed
- `all_gather_minimal_matmul_async_program_factory.cpp`: Changed mux placement to `CoreCoord(mux_id % full_grid_size.x, (mux_id < full_grid_size.x) ? 9 : 8)`
- Worker-to-mux mapping updated to use `mux_id` directly

### Rebuild
- pass

### Runtime
- fail (hang)

### Profiler
- prompt length: 8k
- fused op duration: N/A (hung)
- num_links=4 active: **YES** (32 log messages appeared, one per device)
- baseline comparison: N/A

### Verdict
runtime failure

### Notes
- Hung at "Running TtLlamaPrefetcherSetup"
- Mux placement is correct (no overlap), but worker-to-mux communication is broken
- Likely issue: workers using wrong axis for link assignment (still using in0_idx instead of in1_idx)

### Next Step
Debug worker-to-mux mapping - ensure workers connect based on y-coordinate (in1_idx) not x-coordinate (in0_idx)

---

## Template for New Experiments

### Experiment YYYY-MM-DD-XX

### Hypothesis
...

### Files Changed
- ...

### Rebuild
- pass / fail

### Runtime
- pass / fail / hang

### Profiler
- prompt length:
- fused op duration:
- num_links=4 active:
- baseline comparison:

### Verdict
build failure / runtime failure / 4-link not active / active but slower / active and faster

### Notes
- ...

### Next Step
- ...
