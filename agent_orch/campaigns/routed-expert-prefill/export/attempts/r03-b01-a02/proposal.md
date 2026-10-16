# r03-b01-a02: repair r03-b01-a01's kernel-config-buffer overflow by planning once per core: the reader runs plan_launch and publishes the plan in a new L1 scratch CB that the writer and compute (UNPACK) read instead of planning themselves

## Motivation
r03-b01-a01 (combination of r02-b02-a04's mixed-height phase-B planner with r02-b03-a04's PACK-thread
pipelined activation) never ran on the h7168 shapes: `Program size (71168..72752) too large for kernel config
buffer (70656)`. The per-core binary set (reader + writer + 3 TRISCs) overshoots by 512-2096 B.
`plan_launch` is compiled into three of the five binaries (reader at -Os, writer at -Os, compute UNPACK at
-O3). Standalone size of plan_launch + its helpers with the kernel compiler (-mcpu=tt-bh, grid 10):
- -Os: ~3.6 KB .text (reader, writer)
- -O3: 9.2-11.2 KB .text (TRISC0), depending on experts_per_chip
So two of the three planner copies are ~13-14 KB of the core's kernel config buffer, ~6x the overshoot.

## Mechanism
- `adaptive_chunk.hpp`: `plan_page_bytes`, `store_plan`, `load_plan`. Page = [layout0 u32][layout1 u32]
  [owner[0][E] u8][owner[1][E] u8], padded to 32 B.
- Program factory: new 1-page CB `CB_PLAN_SCRATCH` (c_21, 32-64 B), passed as the named compile-time arg
  `cb_plan_scratch` to reader, writer and compute.
- Reader: after `plan_launch`, reserve / store_plan / push the CB (the reader keeps its own local copy).
- Writer: wait_front on the CB and `load_plan` instead of building `tiles_of` and running `plan_launch`.
  Its `start` page read now overlaps the reader's planning.
- Compute: wait_front on the CB, UNPACK `load_plan` instead of `plan_launch`.
The plan is the same object the three RISCs computed independently before (deterministic function of the
same L1 counts), so numerics and schedule are unchanged. Every other code path (TILE x, bias, all activations,
0 tokens, any experts_per_chip) goes through the same plan hand-off.

## Why this is not a repeat
- r03-b02-a01 made the same combination fit by trimming the planner (uniform layouts scheduled once,
  counts cache). Its margin is unknown and it still carries three planner copies. This node removes two of
  the copies outright (~13 KB headroom), the repair r03-b01-a01's reflection ranked #1.
- No earlier node shares the plan between RISCs. r02-b01-a02 folded the `start` page into the counts
  multicast; that is a different hand-off.

## Expected effect and risk
- All 9 shapes run. Expected per shape: r02-b02-a04's dsv3/glm and the pipeline gains elsewhere, i.e.
  ~1.575-1.585 (r03-b02-a01 is the same combination plus a prologue trim, 1.5887). Planning latency on the
  critical path is unchanged (the reader planned anyway, and everything waits on the reader's data).
- Risks: host rebuild (new CB + named args); a hang if a consumer waits on a CB that is never pushed
  (the reader pushes it unconditionally right after planning); stale plan if CB read pointers were wrong
  (would show as PCC failure / hang).
