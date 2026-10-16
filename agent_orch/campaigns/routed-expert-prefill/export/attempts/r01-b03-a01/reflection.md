# r01-b03-a01 result: 0.9416 (ok)

## What happened vs expected
Expected 1.1-1.3x on the per_core_M=1 shapes from overlapping the next weight block's DRAM read with
the current block's multicast. Got a uniform REGRESSION: every shape is slower, by 3-9%: mistral 0.910,
dsv4-flash 0.915, glm 0.924, minimax 0.930, dsv3 0.937, kimi-k2.7 0.943, kimi-k3 0.967, dsv4-pro 0.970,
gpt-oss 0.982 (gpt-oss is the only shape with M=2 experts, i.e. the most compute-bound). PCC / max_abs /
mean_abs are bit-identical to baseline on all 9 shapes, so the read-ahead slot addressing and the
handshakes are correct. It just costs time.

## Why (best explanation)
The pattern argues against my delivery-latency model. Shapes with less compute per block (mistral,
dsv4-flash: K=128) lost the most, and the compute-heaviest shape (gpt-oss M=2) lost the least. So
whatever I added sits on the critical path with a roughly fixed cost per K-block:
- The writer now ISSUES block kb+1's 16 `up` reads (an ND-shard row each) right before it multicasts
  block kb. Issuing costs RISC time (a cmd-buffer wait per request), and on NoC 1 the read requests and
  read responses share the NIU with the 110 KB multicast. So the multicast of block kb, which the
  receivers are waiting for, starts later. On the reader, block kb+1's reads are issued before the
  down multicast and the down_go grant comes earlier, which has the same effect in the down phase.
  In the down phase I also moved the receivers'-ready wait to just before the multicast.
- If the receivers were already being fed fast enough (compute- or multicast-bound rather than
  DRAM-latency-bound), the read-ahead gains nothing and the delay to the multicast is pure loss. The
  ~30 us/expert gap between M=1 (104 us) and the ~70 us marginal compute in the single-expert perf
  table is therefore NOT DRAM read latency that a 1-block read-ahead can hide. More likely it is
  multicast/handshake latency per K-block, per-expert phase transitions (gate/up drain -> activation ->
  rotating activated mcast -> down), or the compute's own per-block overhead (packer drain,
  L1-acc reconfig, the partials copy pass).
- I can't separate the gate/up part from the down part: both changed in one run, and there are no
  per-zone markers in the profile.

## Classification
flawed idea. The serial read->mcast chain on the sender is not the binding constraint at M=1, and
reordering reads ahead of the multicast delays the critical multicast. (Implementation correct.)

## What a child of this node should try next
- Don't retry read-ahead in this form. If you revisit it, issue the next block's reads AFTER the
  multicast has been issued (before the flush/valid), never before it. Also change only one phase per
  attempt so the result can be attributed.
- Find the real bottleneck first. A cheap probe is DeviceZoneScopedN zones around the compute's
  gate/up phase, activation, and down phase, and around the sender's mcast wait. Accept one lost
  attempt to measure it.
- Structural levers that don't depend on the latency model:
  (a) Idle M-rows. At 3-6 tile-rows per expert, 2-5 of the 8 rows in every column do no MACs. Run two
      experts at once on 4+4 row groups when both have <= 4 tile-rows (kimi-k3, dsv4-pro, most of
      kimi-k2.7).
  (b) Fewer K-blocks. A wider in0_block_w_gu, or more weight-CB depth using the L1 freed by a smaller
      per_core_M_max, would cut the per-block handshake count if handshake latency is the cost.
  (c) The per-expert phase transition. Overlap the next expert's gate/up with this expert's down.
