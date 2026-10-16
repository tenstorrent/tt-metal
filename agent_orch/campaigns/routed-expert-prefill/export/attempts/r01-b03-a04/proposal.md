# r01-b03-a04: span the device's 2 spare rows: 10-row group layouts (5 groups of 2 rows, 2 groups of 5 rows) the runtime picker takes only when they strictly beat the 8-row ones

## Motivation
The op runs on a fixed 11x8 rectangle. This chip's compute grid is 11x10. The profiler CSV of
r01-b03-a03 lists worker cores at physical y = 2..11 (10 rows) on 11 columns, so 22 Tensix cores
(20%) are never used. Since r01-b01-a01 the grid is split at runtime into expert-parallel row groups
(8x1, 4x2, 2x4 rows x groups). That schedule is now granular in whole experts per group, and two
shapes lose to that granularity:

- **gpt-oss** (4 experts of 8-9 tile-rows, bias, SwiGLU-OAI). It is the clear compute/latency
  outlier: ~186 GB/s total DRAM, 32-39 cycles per tile-mult, groups balanced to <1%
  (r01-b01-a02/a03, r01-b03-a03 reflections). It runs 2 groups of 4 rows. A 9-tile expert there
  is per_core_M = 3 on rows 3,3,3,0, so one row in four idles. With 5-row groups every expert is
  per_core_M = 2 (rows 2,2,2,2,1). The critical group's M-work drops from 3+2 to 2+2 tile-rows.
- **kimi-k3** (28 experts of 3-4 tile-rows, SiTU-GLU). It runs 4 groups of 2 rows, 7 experts each,
  at 377 GB/s, well below the ~430 GB/s the big shapes reach. Per-expert time is ~211 us per group.
  With 5 groups of 2 rows it is 6/6/6/5/5 experts, so the critical group does 6 instead of 7.

Simulating the kernels' own cost model (`expert_cost` + list scheduling) on the test's routing
(seed 0, /tmp/rep_b03a04/sim.py) gives these makespans (8-row best -> best incl. 10-row layouts):
gpt-oss 7 -> 6 (5x2), kimi-k3 21 -> 18 (2x5), kimi-k2.7 10 -> 9 (2x5). dsv3, glm, mistral, minimax,
dsv4-flash and dsv4-pro tie, so they keep today's layout exactly.

## Mechanism
- Factory: when the device grid has >= 10 rows, the program spans 11 x 10 cores
  (`grid_rows` = 10). Otherwise it stays 11 x 8 and nothing changes. The exported
  `kCoreGridX/Y` (11x8) stay as they are, because moe_fused_swiglu / the hybrid op read them.
  The chunk base (per_core_M_max * 8) is unchanged. Extra rows get the same per-row args (M-row NoC
  table, x sender), and the counts broadcast rectangle covers all rows. The per-group-size weight
  multicast topologies gain a 4th set for 5-row groups. A set whose group would extend past the grid
  (8- or 4-row groups for rows 8-9) is zero-filled, because those cores are idle under those
  layouts. `URE_GRID_ROWS` is passed to all three kernels as a define.
- `adaptive_chunk.hpp`: a fixed table of group layouts {rows, groups, topology set}: (8,1), (4,2),
  (2,4), and with 10 rows also (5,2), (2,5). `pick_group_layout` keeps the existing strict-< makespan
  rule over that list in that order, so a 10-row layout is taken only when it strictly beats every
  8-row one. On ties today's schedule is kept bit-for-bit. A core whose group index is >= the
  layout's group count (rows 8-9 under an 8-row layout) is never assigned an expert, so it takes the
  existing uniform skip path. The GroupScheduler holds up to 5 groups.
- Reader / writer / compute: select rows, groups and the topology set from the layout instead of
  from the group size alone. Compute's UNPACK sends the layout index to MATH/PACK (same mailbox).

## Why this is not a repeat
Every node so far kept the 11x8 rectangle, and the row-group nodes (b01-a01, b02-a01/a02) only
changed how those 8 rows split. Nothing has used the spare rows. It isn't the rejected R=1 / 1-row
groups (b02-a01/a02): groups stay >= 2 rows, and the fixed x sender / rotating x owner never
collides with a weight sender. It isn't the deeper-ring, read-ahead, boundary-feed or pacing
families (b03-a01, b04-a01, b04-a03, b03-a03, b01-a03): no in-pipeline timing changes.

## Expected effect and risk
- gpt-oss: ~15% faster (the M-proportional part of a compute-bound group drops 20%). kimi-k3: up to
  ~12%, until it hits the DRAM ceiling. kimi-k2.7: about neutral (DRAM-bound at 434 GB/s; the
  critical group loses one cost unit). Other shapes: unchanged layout, so they should be within noise.
  Expected geomean ~+2.5-3% over the parent.
- Numerics bit-identical: each output tile keeps its K order on whichever core owns it.
- Risks: a hang or wrong rows from a layout or topology mismatch between kernels, or from a
  multicast rectangle that is wrong for 5-row groups (eval `hang` / `accuracy_fail`). DRAM
  contention from 5 concurrent groups could eat the kimi-k3 gain. A TT_FATAL from the host
  sender-collision checks (I checked by hand that rows=5 has none).
