## Files read
- agent_orch/WORKER.md, campaigns/routed-expert-prefill/campaign.yaml: the task, rules, allowed paths.
- device/kernels/dataflow/unified_routed_expert_ffn_reader.cpp: the gate/up loop.
  - The rotating x owner `(x_blk + kb) % GRID_X` issues the kb+2 stick reads at the top of an iteration.
  - The weight sender's gate read is followed by a GLOBAL `noc_read.async_read_barrier()`, which also waits for those x reads.
  - The owner then waits cb_x_tl, the row acks, multicasts and write-barriers inside the same sequential iteration.
- device/kernels/dataflow/unified_routed_expert_ffn_writer.cpp: the UP_SPLIT + IN1_WRITER_MCAST chain (wait up_go -> read up -> mcast_go -> receivers' ready -> mcast -> valid). The column multicast waits for the reader's gate barrier.
- device/unified_routed_expert_ffn_program_factory.cpp:
  - per-layout topology: weight sender row = group_base + gx % rows;
  - the UP_SPLIT / DOWN_SPLIT / IN1_WRITER_MCAST notes: weight reads are issue-bound, ~43 cycles of RISC command-buffer time per transaction.
- device/kernels/weight_runs.hpp: one NoC transaction per K-row of a core's ND shard (16 per gate block). The x path issues ~96 stick reads per block at pcM 3.
- device/kernels/compute/fused_swiglu.cpp: owner tests in matmul_phase_fused_gu, and the UNPACK -> MATH/PACK mailbox (count, group rows).
- device/kernels/adaptive_chunk.hpp: layout tables, layout_group.
- tests/.../test_routed_expert_prefill.py: weights are DRAM ND-sharded bf4, x is row-major bf16.

## Nodes consulted
All 36 nodes (proposal + reflection), dumped to /tmp/r03b02a02/hist.md. The ones that shaped this attempt:
- r03-b02-a01 (parent, 1.5887): its zones give GU per block ~2-2.7x the ideal matmul time on mistral / gpt-oss / mm3.
- r01-b02-a02: x sender on a weight-sender core cost gpt-oss 25% (TILE path rotation).
- r01-b01-a02 / r01-b02-a03: the row-shared tilize rotates the owner over all 11 columns. r01-b02-a03 estimated the sender collision at "~1/11 of blocks" and proposed (next #4) keeping the owner off sender columns. It is in fact every block, for one row per group.
- r02-b04-a01, r02-b01-a02, r03-b04-a01: the per-expert fixed cost F that does not scale with per_core_M on the few-expert shapes.
- r01-b03-a01, r01-b04-a01, r01-b04-a03, r02-b02-a02/a03, r02-b04-a02: DM read timing / ring depth / VC / down-stream moves were all neutral or negative, so this attempt moves no read earlier and changes no buffering.
- r03-b01-a01: kernel binary size limit (70656 B) on h7168. This change adds only a few instructions.

## Docs / external references
- None beyond the code. Tools:
  - /tmp/r03b02a02/grid.py: per-core GU/DN/ACT/end grid from profile_log_device.csv.
  - /tmp/r03b02a02/own.cpp: brute-force check of x_owner_col for grid_x 1..16, rows 2..10, every row.
- /tmp/r03b02a02/sz: local LTO compile of parent vs node compute kernels (kcl.sh plus a `-r` relocatable link) for binary-size comparison. First eval's outputs are saved in /tmp/r03b02a02/eval1.
- Consulted r03-b01-a02 (committed during this attempt): it repaired the same overflow by planning once per core.
