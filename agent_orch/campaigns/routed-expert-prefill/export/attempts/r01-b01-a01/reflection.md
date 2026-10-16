# r01-b01-a01 result: 1.4376 (ok)

## What happened vs expected
All 9 shapes are valid. PCC, max_abs and mean_abs match the baseline to every printed digit,
which confirms the expected bit-identical numerics. Speedups per shape: dsv3 1.41, kimi-k2.7 1.49,
kimi-k3 1.73, glm 1.42, mistral 1.31, dsv4-flash 1.49, dsv4-pro 1.40, minimax-m3 1.24,
gpt-oss 1.50. Every shape is far outside the ±1% noise band.

Against the MAC-row model in proposal.md:
- The 3-4-tile shapes (kimi-k3, kimi-k2.7, dsv4-pro, dsv4-flash) gained 1.4-1.73x. The model
  predicted ~2x, so these are now bounded by something other than MAC rows.
- The 5-6-tile shapes (dsv3, glm) gained 1.41x even though the MAC rows per core barely change
  (8 -> 6 units). So overlapping G=4 independent pipelines (and 4x the weight-stream senders) is
  worth about as much as the MAC reduction.
- The 4-expert shapes (mistral, minimax-m3) gained least (1.24-1.31x). There R=2 puts exactly one
  expert in each of the 4 groups at per_core_M=3, so each group runs a single serial expert
  pipeline with no cross-expert overlap.
- gpt-oss picked R=4 (two groups, 2 experts each, per_core_M=2-3) and gained 1.50x.

## Why (best explanation, with profiler evidence)
Effective bf4 weight bandwidth is now 330-390 GB/s on the big shapes (dsv3 349, kimi-k2.7 382,
kimi-k3 350, dsv4-pro 390, glm 341). The baseline ran at about 200-250 GB/s. That is close to
practical BH DRAM bandwidth (~450 GB/s achievable, 512 GB/s peak), so the large shapes are now
mostly DRAM-streaming-bound. The exceptions are mistral (283), minimax-m3 (331) and especially
gpt-oss (146 GB/s). These have only 4 experts, so their per-expert serial pipeline latency
(fill/drain, x tilize and stick reads, down phase) is still exposed.
Per-zone profiling wasn't available (only kernel-level zones exist), so the split between
DRAM-bound and latency-bound is inferred from bandwidth, not measured.

## Classification
win

## What a child of this node should try next
- **4-expert shapes / gpt-oss:** the remaining gap is latency within one expert's pipeline. Two
  options: a cost model that also considers splitting an expert's K or N across groups when there
  are fewer experts than groups, or R=1 (each M-row independent, no weight multicast). Another
  option is to overlap expert e+1's gate/up weight prefetch with expert e's down phase by deeper
  in1 buffering. L1 has headroom at per_core_M <= 3, because the CBs are sized for
  per_core_M_max=4.
- **Large shapes (near DRAM bound):** look at DRAM bank locality and requestor balance. With R=2
  there are 44 sender cores (reader gate + writer up/down) on 8 DRAM banks, so check that ND-shard
  bank rotation interacts well. The remaining compute-side lever is the redundant per-row tilize
  of x (11 cores per row each tilize the same block).
- Tune the cost model's per-chunk overhead (currently 1 unit) and consider letting the
  4-expert cases use R=4 (2 experts per group, overlap) instead of R=2. The model picked R=2 for
  mistral/minimax (makespan 4 vs 6), but measured overlap benefits suggest the two may be close.
- Keep in mind: group selection, ownership and topology all live in
  `kernels/adaptive_chunk.hpp` (`pick_group_rows`, `expert_cost`, `GroupScheduler`). The host
  passes the 3 topologies (rows 8/4/2) as per-core runtime args. Reader topology sets come after
  the M-row NoC table and writer sets start at arg 17.
