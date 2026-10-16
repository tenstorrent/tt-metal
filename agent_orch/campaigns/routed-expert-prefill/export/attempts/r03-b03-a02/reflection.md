# r03-b03-a02 result: 1.4983 (ok). Regression: -4.9% vs the parent r03-b03-a01 (1.5751)

## What happened vs expected
- All 9 shapes are valid. PCC, max_abs and mean_abs are bit-identical to the parent and the baseline.
- No hang, and no JIT error on the first run. The host rebuild (factory `up_mode` change) built first try.

I expected the few-expert shapes to gain 5-10% and the DRAM-bound ones to stay flat. Instead every shape
except kimi-k3 got slower, and the DRAM-bound shapes lost the most (chip-mean µs; GB/s is a rough
weight-bytes estimate):

| shape | parent | this | Δ | ~weight GB/s parent -> this |
|---|---|---|---|---|
| dsv3 | 531.1 | 592.4 | **+11.5%** | 373 -> 334 |
| kimi-k2.7 | 737.0 | 801.2 | **+8.7%** | 403 -> 371 |
| glm | 463.4 | 503.8 | **+8.7%** | 367 -> 337 |
| minimax-m3 | 374.3 | 394.9 | +5.5% | 339 -> 322 |
| mistral | 189.7 | 199.1 | +4.9% | 298 -> 284 |
| dsv4-pro | 1082.1 | 1134.9 | +4.9% | 412 -> 393 |
| dsv4-flash | 326.2 | 333.0 | +2.1% | 346 -> 339 |
| gpt-oss | 270.0 | 274.5 | +1.6% | 207 -> 204 |
| kimi-k3 | 1385.8 | 1368.9 | -1.2% (noise) | 375 -> 380 |

## Why (profiler evidence)
New this node: profiler-only wall-clock counters per DM wait site. One TS_DATA record carries two
counters (`RE-R-*` on NCRISC, `RE-W-*` on BRISC). Parser: `/tmp/rep-routed-r03b03a02/dm.py <profile_log_device.csv>`.
The values below are µs per core per call, mean of the 10 measured calls. "Sender" means the group's
weight-sender row; dsv3 runs 28 gate/up blocks per core per call.

| | dsv3 | mistral | mm3 | gpt-oss |
|---|---|---|---|---|
| reader GU loop total (sender) | 329 | 90 | 198 | 111 |
| read ISSUE, gate+up = 32 reads per block | 44 (1.6/blk) | 18 (2.3/blk) | 33 (2.8/blk) | 12 (1.9/blk) |
| read barrier after issue + x mcast | 112 (4.0/blk) | 39 (4.8/blk) | 90 (7.5/blk) | 41 (6.8/blk) |
| owner x multicast (tl + ready + mcast) | 16 | 4 | 10 | 7 |
| sender waiting x valid | 106 (3.8/blk) | 24 (3.0/blk) | 45 | 47 |
| writer: wait for reader's reads (`mcast_go`) | 193 | 76 | 165 | 112 |
| writer: in1_ready wait | 16 | 4 | 5 | 3 |
| writer: gate+up multicast, issue + flush + sem | 35 (1.2/blk) | 16 (2.0/blk) | 34 (2.8/blk) | 17 (2.9/blk) |
| receiver: x valid wait / weight valid wait | 202 / 100 | 61 / 19 | 113 / 43 | 75 / 24 |

1. **The writer now spends 60-100% of its gate/up time waiting for the reader's reads.** It waits on
   `mcast_go`. The multicast itself is only 1.2-2.9 µs per block. The reader's 32-read DRAM stream is
   the critical chain, and it got SLOWER than the old two-RISC split:
   - issue plus barrier is 5.6 µs per 110 KB block on dsv3, ~20 GB/s per sender core;
   - the DRAM-bound shapes' aggregate fell ~8-10% (dsv3 373 -> 334 GB/s).
   - So the old UP_SPLIT was not only issue-rate parallelism. **Putting all gate/up weight read
     responses on NoC 0 costs aggregate weight bandwidth.** This is the same signature as
     r02-b04-a02 (all down reads on one NoC: -10..-22% delivery).
   - Both NoCs' response paths are needed to reach ~440 GB/s. Every weight read stream must stay split
     across NoC 0 and NoC 1.
2. **The multicast was never the big serial term.** It is 1.2-2.9 µs per block (dsv3, 110 KB to 1-4
   receivers). Even a perfect overlap of up-read and multicast is worth at most ~10% of the gate/up
   block on the 4-group shapes, and nothing on the DRAM-bound ones. The writer serialization I targeted
   is real but small.
3. **The x-owner reorder did not remove the row coupling.**
   - Sender-row cores still wait 3-4 µs per block for x after their own reads land (`xwait`, dsv3 106 µs).
   - The owner's multicast still needs `in0_ready` from all 10 row cores, which ack only when they start
     block kb, after their kb-1 reads and x(kb-1). The chain moved; it was not cut.
   - Receiver rows spend most of gate/up waiting: x 202 + weights 100 of 348 µs on dsv3. That is DM
     starvation, not compute.
   - I can't separate the reorder's own effect from the NoC-0 loss in one run. The reorder alone is
     probably neutral to slightly positive, since it only shortens the owner's wait on its own reads.
4. Read issue cost is visible: ~1.6-2.8 µs per block for 32 reads, i.e. ~50-80 cycles per read with the
   ND-shard address math. With UP_SPLIT it is half that per RISC, in parallel.

## Classification
flawed idea (as built), with one useful part:
- **Flawed:** the reader-reads-gate+up half (all gate/up weight reads on NoC 0) cuts aggregate weight
  bandwidth on every shape.
- **Not worth much:** the writer's read/multicast serialization is real, but the multicast is only
  ~1.2-2.9 µs per block.
- **Inconclusive:** the x-owner reorder.
- **Useful:** the per-site DM counters (`re_prof.hpp`, `RE_PROF_*`). They are zero-cost outside profiler
  builds; in profiler builds they add ~1.5 KB to NCRISC and ~0 to BRISC.

## What a child of this node should try next
1. **Start from the parent's code (r03-b03-a01), or revert this node's factory `up_mode` change.** That
   one line restores UP_SPLIT and the old writer read path; the writer code still supports both modes.
   Keep `re_prof.hpp` and the counters to get the same breakdown on the good configuration. They are
   bit-identical and essentially free.
2. **Never put a whole weight stream on one NoC.** Gate/up (and down) must keep their reads split across
   NoC 0 and NoC 1. This node and r02-b04-a02 both lost ~8-20% doing otherwise.
3. If a read/multicast overlap is retried, keep the reads on the writer's NoC 1 and interleave them with
   the multicast (issue up(kb+1) between the 16 KB bursts of multicast(kb)). Never move them to the
   reader. Expected value is small (multicast ≤ 3 µs per block).
4. **The receivers' wait is the real slack.** On dsv3, receiver rows spend ~87% of gate/up waiting for
   x / weights; the sender row ~65% in read barrier + x wait. Senders' reads run at ~20 GB/s per core
   while the chip moves ~370-400 GB/s, so this is the DRAM / NoC share, not latency.
   - The remaining lever on DRAM-bound shapes is NoC response-path efficiency.
   - Example: spread DRAM bank endpoints across the 3 NoC endpoints per bank, or give each RISC's reads
     the opposite NoC from that column's multicast.
   - Measure per-NoC link use first.
5. Counter layout for later use: reader `RE-R-GU-TOT-RES, -SLOT-ISSUE, -XRM-XOWN, -RBAR-XWAIT,
   -WWAIT-DNTOT, RE-R-DN-SLOT-RDY, -RBAR-MC, -ACT-ACTV, -WV` and writer `RE-W-GO-RDY, -MC-DGO, -DRD-OUT`.
   Each record is hi<<32 | lo in cycles at 1.35 GHz, summed over the call.
