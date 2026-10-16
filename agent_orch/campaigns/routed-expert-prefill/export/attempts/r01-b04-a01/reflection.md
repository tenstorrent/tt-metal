# r01-b04-a01 result: 0.9330 (ok)

## What happened vs expected
Expected 10-25% from overlapping the next weight block's DRAM read with the current block's
multicast on the sender RISCs. Got a uniform **~5-10% slowdown** on all eight bias-free shapes
(DS-V3 0.931, Kimi-K2.7 0.930, K3 0.945, GLM 0.912, Mistral 0.904, V4-flash 0.924, V4-pro 0.947,
M3 0.905) and gpt-oss unchanged (1.0045, within the 1% noise). PCC / max_abs / mean_abs are
bit-identical to the baseline on every shape, so the protocol is correct (no race, no hang); the
loss is purely timing.

## Why (best explanation)
* The read->multicast serialisation on the sender RISC is **not** the bottleneck. If it were, the
  overlap could only help. Making the read earlier instead slowed every weight-streaming shape.
* The most likely cause is contention on the critical path. The prefetched DRAM read responses
  (up on NoC 1 into the writer's core, down on NoC 0 into the reader's core) now land while that
  same sender core is multicasting the current block to the 7 receivers, and the polling loops
  hammer L1 (cache-invalidated semaphore reads) / stream registers. The multicast is what the 7
  receivers' compute waits on. Anything that delays it costs the whole column, while an earlier
  read buys nothing because the next block cannot be multicast before the receivers free their slot
  anyway (2-slot ring, receivers' ready gates the multicast).
* gpt-oss being flat fits this. Half its experts have 9 tile-rows → per_core_M = 2, and its
  subblocks are 1x3, so it is compute-bound and the DM timing change does not show.
* Implication for the bottleneck: per-tile-mult time is ~2x the LoFi math rate while DRAM delivers
  only ~250 GB/s, and neither earlier reads nor (by this result) faster senders help. So the limit
  is most likely the lock-step coupling across the grid. Each K-block needs x from the row sender
  and weights from the column sender, with only 2 slots of slack, so the slowest core sets the pace.
  Compute-side fixed per-block overhead could also be the limit: per K-block there is a tilize
  init/uninit, a matmul re-init, pack reconfigs, and a packer drain (STALLWAIT) before each L1_ACC
  re-accumulate.

## Classification
flawed idea (measured regression, accuracy intact). Do not retry the read/multicast overlap as-is.

## What a child of this node should try next
Start from root, not from this node. Directions this result points at:
1. Compute per-K-block overhead: the row-major path re-inits tilize + matmul + pack format every
   K-block, and every block ends in a packer STALLWAIT drain. Wider in0_block_w_gu at per_core_M=1
   (the L1 is sized for per_core_M_max=4, but every benchmark expert runs per_core_M 1-2) would
   halve those fixed costs and the per-block handshakes. That trades the unused M headroom for
   K-block width.
2. gpt-oss specifically: experts with 257-272 tokens run per_core_M=2 on 9 tile-rows (8 rows ×
   2 = 16 slots, 7 wasted). The 1x3 subblocks (per_core_N=9, DST 8) are a separate cost there.
   Either fix should give that shape a large isolated gain.
3. Before more DM work, add a few DeviceZoneScopedN zones (sender vs receiver, wait vs compute)
   in a measurement node to find out which wait actually dominates.
