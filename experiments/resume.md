# Resume State

**Last updated:** 2026-03-09
**Task:** Fused AG+MM 7×8 num_links=4 optimization

## Current State

- **Last experiment:** 2026-03-09-06 (Option A v2: mux at rows 8-9)
- **Result:** runtime failure (hang during prefetcher setup)
- **Code state:** REVERTED to baseline (no special 7x8+4links code active)
- **Device state:** Unknown (may need reset)

## What Was Tried

| # | Approach | Result | Root Cause |
|---|----------|--------|------------|
| 1 | Pad in0_link_axis_cores 7→8 | path not taken | Python condition wrong |
| 2 | Fix Python condition | overlap error | full_grid_size.x=7 wraps |
| 3 | Overflow mux to row 8 | hang | worker-mux comm broken |
| 4 | Use 7 links instead | not enough cores | 14 mux > 10 available |
| 5 | Option A: y-axis grouping, mux at x=6 | NOC conflict | mux on worker column |
| 6 | Option A v2: mux at rows 8-9 | hang | worker-mux mapping wrong? |

## Key Constraints Discovered

- `full_grid_size = (7, 10)` — only 7 x-positions, 10 y-positions
- Worker grid is 7×8 (columns 0-6, rows 0-7)
- Storage rows are 8-9 (outside worker grid)
- 4 links × 2 directions = 8 mux cores needed
- Divisors of 7: only 1 and 7
- Divisors of 8: 1, 2, 4, 8 ← this is why we want y-axis grouping

## Hypothesis Queue (for autonomous mode)

Priority order - agent picks the first untried one:

| # | Hypothesis | Status | Rationale |
|---|------------|--------|-----------|
| 1 | Fix worker-to-mux mapping for y-axis | **NEXT** | Workers use `in0_idx` but should use `in1_idx` for y-axis grouping |
| 2 | Try 2 links instead of 4 | pending | Simpler: 8%2=0, only 4 mux cores needed |
| 3 | Fix termination master core logic | pending | Hang may be due to wrong termination signaling |
| 4 | Use different mux placement pattern | pending | Current (0,9)..(6,9),(0,8) may have comm issues |
| 5 | Add explicit synchronization barriers | pending | Hang could be race condition |

### Hypothesis Details

**H1: Fix worker-to-mux mapping for y-axis grouping**
- Workers may still be using `in0_idx` (x-coordinate) for link assignment
- Should use `in1_idx` (y-coordinate) when `use_y_axis_for_links=true`
- Files: `all_gather_minimal_matmul_async_program_factory.cpp` lines ~980-1044
- Look for: `link = in0_idx / num_workers_per_link_effective`
- Change to: `link = (use_y_axis_for_links ? in1_idx : in0_idx) / num_workers_per_link_effective`

**H2: Try 2 links instead of 4**
- 7 % 2 != 0, but 8 % 2 == 0 (y-axis)
- Only 4 mux cores needed (2 links × 2 directions)
- Easier to debug, may reveal if issue is link-count specific
- Files: `llama_ccl.py` - change `effective_num_links = 2`

**H3: Fix termination master core logic**
- Hang may be due to incorrect termination signaling
- Check if termination master is correctly assigned for y-axis case
- Files: `all_gather_minimal_matmul_async_program_factory.cpp`
- Look for: termination_master, is_termination_master

**H4: Use different mux placement pattern**
- Current: (0,9), (1,9), ..., (6,9), (0,8)
- Alternative: (0,8), (1,8), (2,8), (3,8), (0,9), (1,9), (2,9), (3,9)
- May have better communication locality

**H5: Add explicit synchronization barriers**
- Hang could be a race condition between workers and mux
- Add barrier before mux communication starts

## Before Next Experiment

1. Reset devices: `tt-smi -glx_reset && sleep 60`
2. Verify code is clean: `git status` in tt-metal
3. Read ledger.md for full experiment history
4. Choose ONE hypothesis to test

## Files to Modify (for any 7x8+4links change)

- `ttnn/cpp/.../all_gather_minimal_matmul_async_program_factory.cpp`
- `models/demos/llama3_70b_galaxy/tt/llama_ccl.py`
