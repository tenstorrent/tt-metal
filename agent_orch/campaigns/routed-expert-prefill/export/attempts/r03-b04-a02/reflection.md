# r03-b04-a02 result: 1.5761 (ok). Neutral vs the parent r03-b04-a01 (1.5813, -0.3%, inside the ±1% noise)

## What happened vs expected
- All 9 shapes are valid. PCC, max_abs and mean_abs are bit-identical to the parent and the baseline on every printed digit.
  - That covers the transaction-id tagged x reads, the trid-scoped gate barrier and the deferred / polled cb_x_rm push. They deliver exactly the same data.
  - No JIT error and no hang on the first and only run. Kernel-only change, so no host build.
- I expected +0.5..+2% on the shapes whose gate/up stream paces the group. Nothing moved outside noise.

Per shape vs the parent (µs, chip mean; median / max core = TRISC1 kernel end, mean of the 10 measured calls, /tmp/r03b04a02/zones.py):

| shape | parent | this | Δ | median core parent -> this |
|---|---|---|---|---|
| dsv3 | 520.2 | 534.1 | +2.7% | 495 -> 504 |
| kimi-k2.7 | 741.1 | 737.9 | -0.4% | 738 -> 735 |
| kimi-k3 | 1378.8 | 1372.4 | -0.5% | 1299 -> 1287 |
| glm | 454.3 | 458.1 | +0.8% | 437 -> 433 |
| mistral | 196.0 | 194.5 | -0.8% | 170 -> 168 |
| dsv4-flash | 320.7 | 317.3 | -1.1% | 298 -> 301 |
| dsv4-pro | 1072.5 | 1087.4 | +1.4% | 1067 -> 1083 |
| minimax-m3 | 369.1 | 374.1 | +1.4% | 342 -> 348 |
| gpt-oss | 273.4 | 271.9 | -0.5% | 242 -> 242 |

The medians moved by -12..+16 µs in both directions. The largest move is dsv4-pro +16, which is inside the 1064-1086 spread that identical dsv4-pro code has shown across earlier nodes. dsv3's +2.7% is the known max-core lottery: its median moved only +9.

## Why (profiler evidence: two new NCRISC zones, both recorded)
Analysis is in /tmp/r03b04a02/zones.py and period.py on reports/r03-b04-a02/.logs/profile_log_device.csv.

**1. The x look-ahead read was never late. RE-XW (the residual blocking x wait) is ~0.**
- RE-XW fires only when an x block has NOT landed by the end of iteration j-1. That happened once per call on gpt-oss (1.7 µs) and never on the other 8 shapes.
- So with one extra block period, every x read (96 KB at pcM 3, ~1 KB sticks) completes, almost always within the non-blocking polls.
- The medians did not move either, so the old deadline (the end of iteration j-2, before the weight push) was also met nearly always.
- The x stream is not what makes a row or group slow. That rules out r02-b04-a03's "row's own x stream" suspect for the group-tail lottery, at least the DRAM-read part of it.

**2. The gate sender's DRAM read is short and is not the pacing resource. RE-GW is the gate-only barrier wait after issuing the 16 gate K-row reads (~55 KB).**

| shape | gate wait p10 / p50 / p90 (µs) | gate/up block period p10 / p50 / p90 (µs, RE-GW start to start on a sender) |
|---|---|---|
| mistral | 0.5 / 2.1 / 5.5 | 6.7 / 10.5 / 15.4 |
| minimax-m3 | 0.5 / 1.7 / 5.3 | 11.3 / 15.8 / 20.3 |
| dsv4-flash | 0.5 / 2.4 / 6.2 | 5.1 / 8.9 / 14.9 |
| dsv3 / glm | 0.5 / 2.5 / 7.5 | 5 / 9 / 18 |
| kimi-k2.7 | 0.5 / 2.2 / 6.7 | 4.8 / 10.0 / 18.3 |
| kimi-k3 | 0.6 / 1.6 / 5.7 | 7.6 / 12.2 / 20.2 |
| dsv4-pro | 0.6 / 1.9 / 6.6 | 7.5 / 15.6 / 25.4 |
| gpt-oss | 0.0 / 3.0 / 10.5 | 6.6 / 14.4 / 38.7 |

- A sender's own DRAM read occupies only ~15-25% of the block period. 55 KB in ~2 µs is ~25 GB/s per RISC.
- The median block period is 9-16 µs. The LoFi compute per block is ~6.8 µs on mistral (pcM 3, 576 tile-mults), less on the pcM-2 shapes.
- So per block ~3-8 µs is spent neither in the sender's DRAM read nor in matmul.
- The per-RISC DRAM read is not the per-group stream limit. Given r02-b01-a01 / r02-b03-a01 (a lone group pulls ~250 GB/s), the limit has to be in:
  - the writer's serial chain on the sender: `up` read → mcast_go → in1_ready (all receivers) → 110 KB NoC-1 multicast → valid;
  - the receivers' ready latency;
  - the x owner's tilize + row multicast;
  - or per-block compute overhead above the ideal (L1_ACC packer drain, tilize on the owner).

## Classification
neutral (within noise); flawed idea. The x look-ahead read, and its sharing of the gate barrier, was not on the critical path. The implementation is correct and bit-identical. The zones are the useful result:
- they rule out the x-read suspect and the sender gate-read latency;
- they give the first direct measurement of the gate/up block cadence on the senders.

## What a child of this node should try next
1. **Start from r03-b02-a01 (best, 1.5887), not from here.** This node's reader change is harmless but buys nothing.
   - The RE-GW zone, and keeping the gate barrier trid-scoped, are free, so a measurement node may keep them.
   - To drop the change, revert unified_routed_expert_ffn_reader.cpp to the parent.
2. **Locate the missing 3-8 µs per gate/up block with zones on the other actors**, using one shape (mistral or minimax-m3; one expert per group, so few zones).
   - (a) Writer (BRISC) on a sender: the `up` read barrier, the `mcast_go` wait, the `in1_ready` wait and the multicast + flush. BRISC zones should record like NCRISC ones. If `in1_ready` dominates, the receivers (compute) pace the group. If the multicast dominates, it is NoC-1 bandwidth.
   - (b) Reader on the x owner: the `cb_x_tl` wait (owner tilize) and the `in0_ready` wait before the row x multicast.
   - (c) TRISC_1: one zone per gate/up K-block around the matmul loop + `drain_packer_before_reaccumulate`, to get the true compute time per block against the ~6.8 µs ideal.

   Only then pick a mechanism. Ruled out so far for the gate/up cadence:
   - sender DRAM-read latency (this node);
   - x read latency (this node);
   - ring depth (r01-b04-a03);
   - request VCs (r02-b02-a02);
   - weight read-ahead (r01-b03-a01, r01-b04-a01);
   - pacing (r01-b01-a03).
3. If (a) shows the writer's serial `up read → multicast` chain is the limit, the cheap test is to split it: let the reader also multicast `up` after `up_done` (UP_SPLIT's original form) while the writer reads ahead. Alternatively, put gate and up on two different sender rows of a group (top row gate, bottom row up), which doubles the issuing NIUs per column. This is r02-b03-a01's next #3, still untried.
4. Don't retry x-read scheduling (this node, r01-b03-a02, r01-b01-a04).
