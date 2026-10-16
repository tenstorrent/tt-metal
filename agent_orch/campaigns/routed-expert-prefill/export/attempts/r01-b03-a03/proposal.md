# r01-b03-a03: on top of r01-b01-a02's row-group code, weight-sender writers feed the next unit's first gate/up block while waiting to drain this unit's output

## Motivation
This branch's own line (root + read-ahead, then x batching) sits at 0.96. Three nodes there (b03-a01,
b03-a02, b04-a01) and b04-a02 showed that per-K-block delivery and overhead tweaks on the root
schedule don't pay. The proven lever is the row-group schedule (r01-b01-a01/a02, r01-b02-a01/a02,
1.40-1.47). So this node first replaces the parent's kernels/factory with **r01-b01-a02's code
(best node, 1.4716)** byte-for-byte. Everything below is measured relative to that node.

Where b01-a02 still loses time, from its profile (`reports/r01-b01-a02/.logs`) and byte accounting:
- Counting x reads and output writes as well as weights, the big shapes already move 410-434 GB/s
  (dsv3 234 MB in 563 us, kimi-k2.7 335 MB in 771 us, dsv4-pro 474 MB in 1149 us). A p150b board in
  this repo measured a DRAM ceiling of 367 GB/s for matmuls (models/experimental/voxtral_tts), so
  those shapes are near the floor.
- The shapes with many small experts or short K run well below that: kimi-k3 377 GB/s
  (28 experts, 7 per 2-row group, K=112), dsv4-flash 378 (K=128), mistral 360, minimax 378. kimi-k3
  runs ~206 us per expert per group, against ~190 us for its DRAM share and ~122 us of compute.
  So about 16 us per expert is fixed per-expert cost. Groups start together and run experts of
  about the same size, so their expert boundaries roughly coincide and DRAM idles at them.
- One concrete boundary bubble (named in b01-a02's reflection, item 3): on each group-column's weight
  sender, the writer (BRISC) runs `up feed(u) -> down feed(u) -> drain cb_out(u)`. The drain blocks
  until the compute has finished unit u (expert chunk) entirely. Only after that does it read
  `up` block 0 of unit u+1 from DRAM and multicast gate+up block 0. Every unit therefore starts with
  the compute of the whole group waiting for one up read (55-83 KB) plus one gate+up multicast
  (110-166 KB), roughly 3-6 us. The reader has long since read gate block 0 and reserved the slot:
  it reaches the next unit right after its last down multicast, about 2 down blocks before the
  compute ends.

## Mechanism
Writer kernel only (`kernels/dataflow/unified_routed_expert_ffn_writer.cpp`). No host change, no
reader or compute change:
1. The per-K-block up feed body moves unchanged into a lambda `feed_up_block(up_acc, kb)`.
   up_seq / mc_seq still advance once per call, so slots and handshakes are identical.
2. Before the drain of unit u, sender cores compute the next unit this group runs: the next chunk of
   the same expert, else the first owned expert after it. They find it by replaying the
   deterministic group schedule on a copy of the scheduler.
3. The drain's `cb_out.wait_front` becomes a poll. While cb_out has no subblock yet, if the reader
   has already granted the next unit's block 0 (`up_go > up_seq`, non-blocking), the writer
   runs `feed_up_block(next, 0)`: up read, up_done, wait for mcast_go and the receivers' in1_ready,
   multicast, valid, mcast_done. The next unit's feed loop then starts at block 1.
Deadlock freedom: the reader grants up_go(0) only after finishing its down loop, and before any wait
on its own compute (bias reserve aside). mcast_go needs only that reader's gate read. in1_ready needs
the receivers' readers to reach the next unit's kb=0 acks, which happen before any x or compute wait.
None of these waits depends on this core's compute or drain. With FUSE_BIAS the reader blocks on the
single-buffered bias CB until compute pops the previous expert's biases, so up_go is not granted
during the drain. The feed then simply doesn't trigger and the old order runs, which is correct but
gives no gain at gpt-oss expert boundaries. Only block 0 is fed early: the reader grants block 1 only
after block 0's x has arrived, which needs the next unit to have started.

## Why this is not a repeat
- r01-b03-a01 and r01-b04-a01 moved block k+1's read before block k's multicast inside one unit's
  steady-state K-loop, on the root schedule. This node doesn't touch the in-loop order. It only
  moves the first block of the next unit into the previous unit's drain wait, where the compute
  is not waiting on any weights.
- No node has touched expert/unit boundaries. b01-a02's reflection item 3 proposed exactly this and
  nobody tried it.
- Taking b01-a02's code wholesale is a combination of a proven piece (not new itself); the new
  mechanism is only the writer change, so the comparison is against b01-a02.

## Expected effect and risk
- Bit-identical numerics: same data, same slots, same math. Any PCC change would mean a slot or
  ordering bug.
- About 3-6 us saved per unit boundary per group: kimi-k3 (6 boundaries per group) ~1.5-2.5%,
  kimi-k2.7 / dsv4-pro (2) ~1%, dsv3 / glm / dsv4-flash (1) ~0.5-1%, mistral / minimax (1 expert per
  group) and gpt-oss (bias) ~0. Expected score ~1.48-1.49 against b01-a02's 1.4716, mostly within
  noise except kimi-k3.
- Risks: a hang if the deadlock argument above is wrong (eval `hang`). Also a stall of the
  compute's output copy pass while the writer is inside the feed; it is bounded, and the feed was on
  the critical path anyway.
