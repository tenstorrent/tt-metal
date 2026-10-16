# r01-b01-a03 result: 1.0363 (ok), a large regression from the parent's 1.4716

## What happened vs expected
All 9 shapes are valid and bit-identical to the parent and baseline (PCC/max_abs/mean_abs equal to
every digit). The pacing did what it was built to do: per-group TRISC1 end times
(`reports/r01-b01-a03/.logs/profile_log_device.csv`, calls 3/7/12) now agree within ~1-3%
(dsv3 756/747/752/749, mistral 232/225/225/229, kimi-k3 2088/2063/2065/2121). In the parent the
spread was 3-15% (dsv3 471/567/570/485). But EVERY group got much slower, the parent's slowest
group included, so the makespan rose instead of moving to the mean:

| shape | parent µs | this µs | change |
|---|---|---|---|
| dsv3 | 563 | 788 | +40% |
| kimi-k2.7 | 772 | 1133 | +47% |
| kimi-k3 | 1479 | 2114 | +43% |
| glm | 487 | 724 | +49% |
| mistral | 186 | 282 | +51% |
| dsv4-flash | 341 | 528 | +55% |
| dsv4-pro | 1149 | 1486 | +29% |
| minimax-m3 | 378 | 495 | +31% |
| gpt-oss | 365 | 493 | +35% |

gpt-oss had only ~1% imbalance in the parent, so pacing should have been nearly idle there. It still lost
35%. That means a fixed per-block cost on the critical path, not the throttling itself. Roughly
3-4 µs per weight K-block (mistral: +96 µs over 25 blocks per group; gpt-oss +128 µs over 34).

## Why (best explanation; no per-zone markers, so not proven)
1. **Publisher on the critical path (most likely).** Each group's publisher is its column-0 core,
   which is a weight sender and also an x/activated multicast source. Every block it issues a
   NON-POSTED 87-destination multicast. Its existing `noc.async_write_barrier()` calls (row-shared x
   owner multicast, activated-sender loopback) wait for ALL outstanding non-posted acks, so they
   now also wait for 87 acks of the pace write. That adds latency to a core that gates its whole
   column's weights and, through the x/act lock-step, its rows. Every group pays it every block.
2. **Convoy effect.** With slack = max(2 blocks, 3%), 2-4 noisy pipelines are held almost in
   lock-step, so each group tends to run at the slowest group's instantaneous pace. That could
   explain some of the loss on top of (1), but not ~35% on gpt-oss.
3. Wait-bound timeouts (64k cycles, ~48 µs) firing on every block would have cost far more than
   observed, so a broken board is unlikely. A few stale-slot timeouts at startup can't be ruled
   out.
The data does not show whether the late group speeds up when the leaders yield (the original
hypothesis). The overhead confounds it: the spread collapsed, but at a much lower aggregate
throughput.

## Classification
repairable failure in implementation, doubtful idea. Balancing works mechanically (spread ~1%).
The implementation adds a large per-block cost, and nothing here shows a throughput benefit.

## What a child of this node should try next
- Don't build on this node. Children should start from r01-b01-a02 (1.4716).
- If pacing is retried: publish from a core/RISC that is NOT a weight sender or multicast source
  (e.g. the writer/BRISC of a receiver core, or a dedicated non-sender core). Use posted writes, or
  keep the publish out of any later `async_write_barrier` scope (separate NoC). Publish per expert
  or every 4 blocks, not every block. Use a much larger slack (≥ one expert, or engage only in the
  last ~20% of a group's work). Then the cost is near zero and the effect on the tail can be
  measured. First check, in one measurement node, whether a lone tail group actually speeds up.
- Better levers per this node's analysis: total DRAM traffic (weights + bf16 x + bf8 out) is
  already ~400-420 GB/s on dsv3/kimi-k2.7/dsv4-pro/glm, so those shapes have little headroom
  except the tail imbalance. gpt-oss (~178 GB/s, 2x off both DRAM and LoFi compute) is the clear
  compute/latency outlier: bias adds, SwiGLU-OAI over 27 tiles per expert, and the serial down-phase
  delivery chain (sender reads block kb+1 only after all receivers acked kb) are the suspects.
- Small, low-risk items noticed: every reader AND writer (176 RISCs) reads the same `start` DRAM
  page at kernel start. Folding it into the counts broadcast would trim the prologue.
