# r06-b03-a01: NoC-1 weight reads go through each DRAM bank's nearest NoC port (each BH bank has 3 NoC tiles; today NoC 1 always uses tile "port 1"), cutting the hot DRAM-column NoC-1 links, and the reader/writer split goes back toward 50/50 to use the freed NoC-1 capacity

## Motivation
- r05-b02-a01 (best, 1.6667) showed the weight stream is **NoC-link-bound, not DRAM-bound**. Moving ~10% of the
  weight bytes from NoC 1 to NoC 0 gave GU -3..-14% and DN -2..-13% on every DRAM-bound shape. Its link model
  (/tmp/r05b02a01/noc*.py) matched the measured direction.
- Why NoC 1 is hot: a NoC-1 read response leaves the DRAM tile going **north (-y) along the DRAM column first**.
  Today every NoC-1 read of bank `c` uses the soc descriptor's fixed `worker_endpoint[1]` tile, which is UMD port 1
  of every bank: physical rows {1, 10, 4, 7} in both DRAM columns (x = 0, 9). A reader in grid row y gets its
  response from row 1 by wrapping north 1 -> 0 -> 11 -> ... -> y (up to 11 hops on the column), and from row 4/7
  by wrapping all the way round for every row below them. Most column links carry most of the 4 banks' traffic.
- Every BH DRAM bank has **three** NoC tiles (blackhole_140_arch.yaml `dram:` rows: {0,1,11}, {2,10,3}, {9,4,8},
  {5,7,6} on x = 0 and x = 9). All three reach the same GDDR controller (MPFE ports P1/P2/P3 = D0/D1/D2,
  `tensor_prefetcher.hpp`), and in the hardware-default NOC2AXI NIU mode a NoC read at a GDDR address on any of
  them is forwarded to DRAM (`hw/inc/experimental/drisc_mode.h`). Only DRISC firmware (BH FW >= 19.12, which turns
  the non-endpoint NIUs into stream mode) changes that. This machine runs FW bundle 19.7.0 (run.log), so no DRISC.
- Link model with the per-row nearest tile for NoC 1 (`/tmp/r06b03a01w/m/ep*.py`, extends r05-b02-a01's noc4.py,
  aggregate 440 GB/s, weight multicasts included):

  | layout | today (s0 = 0.6) GU / DN hottest link | nearest-port NoC 1, s0 = 0.5-0.55 |
  |---|---|---|
  | 5 x 2 rows | 54 / 56 GB/s | 45-49 / 53 |
  | 3+3+2+2 | 54 / 59 | 50 / 58 |
  | 4 x 2 (mistral) | 63 / 63 | 50 / 55 |
  | 4+3+3 | 54 / 63 | 49 / 62 |

  With nearest ports NoC 1's own read load drops ~35% (52 -> 35 GB/s at s0 0.6), so NoC 0 becomes the hot side
  at today's 60/40 split. The split has to move back toward 50/50 to cash it in: the max link falls ~10-17% in
  gate/up; down is held by NoC 0's down multicast, so DN moves less.

## Mechanism
1. **Host (program factory).** When it is safe, build per core a NoC-1 DRAM port remap: for each of the 8 banks,
   the bank's tile whose physical row is the fewest -y hops above the core's row ((row_tile - row_core) mod 12).
   - Safe = Blackhole, 8 DRAM channels (no DRAM harvesting), NoC translation in the expected layout (bank `c`
     port `p` at translated (17 + c/4, 12 + 3(c%4) + p), checked against `virtual_core_from_logical_core((c, 1),
     DRAM)`), worker row in [2, 11], no DRISC (`!IsTensorPrefetcherSupported(mesh)`) and no
     TT_METAL_STREAMING_PROFILER. Otherwise no remap (today's behaviour); `DS_NO_DRAM_EP_REMAP=1` forces it off.
   - Passed as 8 writer runtime args `(default_xy << 16) | alt_xy` appended after the phase-A sems, gated by the
     writer define `RE_DRAM_EP_REMAP`.
2. **Writer (BRISC, NoC 1).** Both NoC-1 weight read sites (UP_SPLIT `up` rows, DOWN_SPLIT down tail tiles) go
   through `WeightRuns::read` with a remap: the run's address comes from the accessor as today, its coordinate
   field is matched against the 8 default NoC-1 endpoints and replaced by this core's alt tile (no match = no
   change). Same bytes, same L1 slots, same barrier and handshakes.
3. **Split back toward 50/50** (only when the remap is on, otherwise r05-b02-a01's 60/40 stays):
   `up_head_rows = (w + 8) / 16` (1 of 16 -> s0 0.53) and `d_head_tiles = ceil(N_d / 2)`.

Files: `device/unified_routed_expert_ffn_program_factory.cpp`, `device/kernels/weight_runs.hpp`,
`device/kernels/dataflow/unified_routed_expert_ffn_writer.cpp`. NoC 0 is untouched (the soc descriptor requires
NoC-0 DRAM endpoints to stay on the CMFW telemetry tile, SYS-1419).

## Why this is not a repeat
- r05-b02-a01 changed which NoC carries which bytes; nobody has changed **which DRAM tile** a read targets. This
  is the first change of the response *path* on a NoC rather than the byte split between NoCs.
- r02-b02-a02 (request VCs) changed arbitration; r05-b03-a01 (x owner columns) and r05-b04-a01 (multicast wrap)
  changed worker-side placement. None touches the DRAM endpoints.
- r03-b03-a02 / r05-b02-a01's notes listed "NoC-endpoint spreading" as untried; r05-b02-a01 flagged the NoC-0
  hazard, which this avoids by touching NoC 1 only.

## Expected effect and risk
- If the NoC-1 column links are what still caps delivery: DRAM-bound shapes (kimi-k3, dsv4-pro, kimi-k2.7, dsv3,
  glm, dsv4-flash, minimax-m3) -2..-6% chip, mostly in GU; mistral / gpt-oss small. Geomean +1.5..+4%.
- If DRAM-side throughput (~490-512 GB/s) is the next cap: smaller, ~+1%.
- If the remap silently never matches, the 50/50 split alone would cost ~-2% (r05-b02-a01 in reverse). That
  would show as GU/DN back at r04-b02-a01's level.
- Accuracy: bit-identical (same GDDR address, same bytes). A wrong port would read garbage -> accuracy_fail,
  or a hang if a tile did not answer.
- Kernel size: writer +~100-200 B (an 8-entry compare per run). Reader unchanged.
- How to tell: per-row GU / DN (`/tmp/r04b01a02/rows.py`) and mean / max core end vs r05-b02-a01.
