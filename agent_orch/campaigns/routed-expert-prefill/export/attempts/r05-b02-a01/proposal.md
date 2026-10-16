# r05-b02-a01: rebalance the weight DRAM reads ~60/40 toward NoC 0 (the reader also reads the first 3 of 16 `up` K-rows per gate/up block; every down K-row is split at a tile boundary, ~60% of its tiles on the reader's NoC 0) to unload the NoC 1 response links in the DRAM columns

## Motivation
8 of 9 shapes are DRAM/delivery-bound at ~420-450 GB/s steady state (r03-b04-a03: mistral GU ~450 GB/s;
r03-b03-a03: dsv4-pro 417, k2.7 406, dsv3 381 GB/s whole-op). The BH DRAM microbenchmark reaches 490-512 GB/s
(tests/scripts/test_moreh_microbenchmark.py, test_dram_read_all_core bounds). The tech report
tech_reports/Saturating_DRAM_bandwidth says why interleaved / round-robin reads fall short of that: DRAM
response routes overlap on the NoC and congest it. Our weights are ND-sharded one K-row per shard,
round-robin over the 8 banks, so every sender reads from every bank.

I built a link-load model of the weight-read responses on the 17x12 BH torus (/tmp/r05b02a01/noc*.py). It uses
the BH soc descriptor's DRAM endpoints: NoC 0 endpoints at (0|9, {11,2,9,5}), NoC 1 endpoints at
(0|9, {1,10,4,7}). Routing is NoC 0 = east then south, NoC 1 = north then west. Senders are this op's staggered
weight senders (row group_base + gx % rows). The model shows:
- **NoC 1 is the hot NoC.** NoC 1 responses go north along the DRAM column (x = 0 or 9) first. So
  the four banks of a column share the same column links. At 440 GB/s with today's 50/50 split
  (gate on NoC 0, up on NoC 1; down rows half / half):
  - the hottest NoC 1 link carries 65-79 GB/s: (0,10)->(0,9) and (9,10)->(9,9), for 5x2 / 4x2 / 3+3+2+2 / 4+3+3;
  - the hottest NoC 0 link carries only 45 GB/s.
- **The same ~60-65 GB/s NoC 1 peak appears in two independent measured regimes.**
  - A lone 2-row group pulling ~250 GB/s (r02-b01-a01): 57-63 GB/s on its hottest NoC 1 link.
  - All groups at ~440 GB/s: 65 GB/s.
  - That is consistent with a NoC 1 link cap of roughly that size setting both ceilings.
- **It explains two earlier regressions.**
  - r03-b03-a02 put all gate/up reads on NoC 0 (-5..-11%, "aggregate weight bandwidth fell"): the model puts NoC 0 at 90.
  - r02-b04-a02 put all down reads on NoC 1 (-10..-22% delivery): the model puts NoC 1 at ~130.
- **The model optimum is s0 ≈ 0.6 of the weight bytes on NoC 0** in both phases.
  - Gate/up: max link 66 -> 54 GB/s (-18%).
  - Down, 2-row groups, including the down multicast on NoC 0: 66 -> 56.
  - If the hot NoC 1 links are what holds the aggregate at ~440 GB/s, this lets it rise toward the DRAM limit.

## Mechanism
Every byte goes into the same L1 slot as today. Only which RISC (hence NoC) issues the read for which bytes
changes.
1. **Gate/up (UP_SPLIT + IN1_WRITER_MCAST, the only path in use).**
   - The reader (NCRISC, NoC 0) issues its gate block reads plus the first `up_head_rows` K-rows of the `up`
     block into the up slot. That is `round(0.2 * in0_block_w_gu)` = 3 of 16, so NoC 0 carries 19/32 = 59% of
     gate/up bytes.
   - It does this both in the look-ahead (`try_gu_lookahead`) and in the in-loop issue. The existing NoC-0
     read barrier covers both, before `mcast_go`.
   - The writer (BRISC, NoC 1) reads up rows [up_head_rows, w) only.
   - The handshake is unchanged: the writer multicasts after `mcast_go` and its own up barrier, and the reader
     pushes after `up_done`.
2. **Down (DOWN_SPLIT).** The rows split (reader rows [0, w/2), writer the rest) becomes a tile split inside
   every K-row:
   - the reader reads tiles [col0, col0 + d_head) of every K-row of the block, where
     `d_head = round(0.6 * per_core_N_d)` (dsv3 13/21, glm 11/18, mistral 7/12, k3 7/11, gpt-oss 5/9);
   - the writer reads the remaining tiles of every row, at the matching L1 offset.

   Same down_go / down_done handshake. The split now needs only per_core_N_d >= 2 instead of in0_block_w_d >= 2.
3. Files:
   - `unified_routed_expert_ffn_program_factory.cpp`: computes up_head_rows and d_head_tiles. d_head_tiles
     replaces down_split_k in reader CT 30 / writer CT 25; up_head_rows goes in as a define `RE_UP_HEAD_ROWS`.
   - `kernels/dataflow/unified_routed_expert_ffn_reader.cpp` and `_writer.cpp`: the read loops.
   - Host change, so the eval rebuilds.

## Why this is not a repeat
- r03-b03-a02 moved ALL gate/up reads to NoC 0, and r02-b04-a02 moved ALL down reads to NoC 1. Both lost, and the
  lesson recorded was "keep every stream split across both NoCs". Nobody has tried an unequal split. In the model
  the 50/50 split is not the balance point, because the two NoCs' response routes from the DRAM columns are not
  symmetric.
- r02-b02-a02 (request VCs) changes arbitration, not the links' load.
- r03-b03-a02's next #4 and r02-b03-a01's next #3 suggested NoC-endpoint / sender-row spreading. That is left
  untried. Alternative endpoints carry a known hazard on NoC 0 (SYS-1419 comment in the soc descriptor).

## Expected effect and risk
- **If the model is right:** the DRAM-bound shapes (kimi-k2.7, dsv4-pro, kimi-k3, dsv3, glm, dsv4-flash, minimax-m3)
  get -3..-10% as their steady-state aggregate rises from ~440 toward ~490 GB/s. mistral / gpt-oss move less,
  because they are closer to compute-bound.
- **If DRAM itself, not the NoC, is the cap:** neutral.
- **If NoC 0 congestion is worse than modelled:** the multicasts on NoC 0 (x along rows, down weights down columns,
  activated along rows) could push the down phase slightly slower on tall groups (5-row groups: the model has the
  down multicast as NoC 0's hottest link). Then down gets slower and gate/up faster.
- **Numerics:** bit-identical. Same bytes into the same addresses, and every slot is complete before its
  multicast / push.
- **Hang risk:** low. No handshake changes; zero-length reads (phantom last column) issue nothing.
- **Kernel size:** about +0.1-0.3 KB reader, ~0 writer. The h7168 programs have ~0.9 KB of headroom on this lineage.
- **How to read the eval:**
  - chip µs and mean core end on the DRAM-bound shapes;
  - per-row GU / DN (rows.py) to see which phase moved;
  - if gate/up improves and down regresses, the down split should go back toward 0.5 or be made group-height
    dependent.
