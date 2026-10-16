# r01-b04-a04: use all 10 Blackhole worker rows: 11x10 grid with expert groups of 10/5/2 rows (was 11x8 with 8/4/2)

## Motivation
The op runs on a fixed 11x8 rectangle, but this device's compute grid is 11x10. In the parent's
profile (`reports/r01-b04-a03/.logs/profile_log_device.csv`) physical rows y=10 and y=11 (logical
rows 8 and 9) run no op kernels: they show 704 zone rows, against 13574 for each of rows 2..9, and
those 704 come from the test's other small ops. So 22 of 110 worker cores (20%) sit idle for the
whole call.

Since r01-b01-a01 the op splits its rows into expert-parallel groups that share nothing except
DRAM, and the per-group pipeline time follows the per-core tile-row units (per_core_M summed over
the group's experts). I simulated the test's routing (seed 0, `/tmp/r01b04a04/sim.py`) with the
kernels' exact cost model (`expert_cost`, `GroupScheduler`, `pick_group_rows`). Max per-core units
per group:

| shape | tiles/expert | 8 rows today | 10 rows |
|---|---|---|---|
| gpt-oss | 8,9,9,8 | 4-row groups, pcM 2/3 -> 5 | 5-row groups, pcM 2/2 -> **4** |
| kimi-k3 | 28 x 3-4 | 4 groups -> 14 | 5 groups -> **12** |
| kimi-k2.7 | 12 x 3-5 | 4 groups -> 7 | 5 groups -> **6** |
| dsv3, glm, dsv4-flash, dsv4-pro, mistral, minimax | | 6/6/5/6/3/3 | same (the extra group(s) take a share of experts or idle) |

gpt-oss is the one compute-bound shape (~186 GB/s total DRAM, groups balanced to <1%; r01-b03-a03
and r01-b04-a03 reflections). Its 9-tile experts on 4-row groups leave one row in four idle (rows
3,3,3,0), which is exactly the waste the r01-b04-a03 reflection flagged. On 5-row groups they run
at per_core_M 2 (2,2,2,2,1).

## Mechanism
- `device/unified_routed_expert_ffn_types.hpp`: new `kCoreGridYMax = 10`. `kCoreGridY = 8` stays as
  the minimum grid and as the exported `UNIFIED_ROUTED_EXPERT_CORE_GRID`. That export only feeds
  moe_fused_swiglu's grid in the model. Its N blocking depends on GRID_X alone, and changing the
  export would change another op.
- `device/unified_routed_expert_ffn_program_factory.cpp`: `GRID_Y = 10` when the device has 10
  rows, else 8, which is today's behaviour. `kMaxChunkMTiles = 4 * GRID_Y`, so per_core_M_max stays 4
  and the L1 fit is unchanged. The per-group-size topology loop already iterates
  `{GRID_Y, GRID_Y/2, GRID_Y/4}`, which gives {10, 5, 2}. Pass `UNIFIED_RE_GRID_Y` as a define to
  the reader, writer and compute kernels.
- `device/kernels/adaptive_chunk.hpp`: `kGridY = UNIFIED_RE_GRID_Y` (default 8). The kernels already
  derive everything from kGridY: `pick_group_rows` halves rows 10 -> 5 -> 2, `group_rows_index` maps
  rows >= kGridY / >= kGridY/2 / else to topology sets 0/1/2, and `kMaxGroups = kGridY/2 = 5` sizes
  the scheduler.
- Sender placement stays collision-free: the weight sender is row group_base + gx % rows and the x
  sender is column (gy+1) % 11. A collision needs gx - gx % rows == group_base + 1, and that is
  impossible for rows 10/5/2 (group_base is a multiple of rows). The host TT_FATAL checks it anyway.

TILE x, biases, activations, the 0-token and experts_per_chip > 1 paths all go through the same
group code, so nothing is specialised to the benchmark.

## Why this is not a repeat
Every earlier node kept the 11x8 rectangle. The row-group nodes (r01-b01-a01, r01-b02-a0x) chose
the group size and expert deal *inside* 8 rows. r01-b02-a02 noted that gpt-oss and the few-expert
shapes lack rows, and suggested splitting an expert's N across groups. This node adds rows instead
and changes no per-group code path. It is not a DM-timing, ring-depth or K-block change (all
ruled out on this lineage).

## Expected effect and risk
- gpt-oss: 5 -> 4 units, roughly -15-20% time (compute-bound). kimi-k3 and kimi-k2.7: up to -14%
  if compute/latency-bound. Less if DRAM-bound: kimi-k2.7 already moves ~434 GB/s total, so maybe
  -2-5%. Other shapes are expected neutral, with a small risk of slowdown from more concurrent
  DRAM streams early in the call (the extra groups finish their single expert early).
- Expected score ~1.49-1.52.
- Numerics are expected bit-identical: every output tile keeps its K-block order, and only the
  row-to-core mapping changes.
- Risks: a multicast rectangle or semaphore count mistake on the 10-row topology (hang). A missed
  hard-coded 8 in a kernel (PCC collapse or hang).
