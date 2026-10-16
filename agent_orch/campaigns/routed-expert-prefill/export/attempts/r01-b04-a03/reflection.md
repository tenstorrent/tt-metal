# r01-b04-a03 result: 1.4622 (ok); neutral vs its real base r01-b01-a02 (1.4716)

## What happened vs expected
The node ports r01-b01-a02's op code, then deepens the gate/up input rings (cb_in0_x, cb_in1_gate,
cb_in1_up) and the down rings (cb_in1_down, cb_in0_down_full) from 2 to 3 slots wherever L1 has
room after the unchanged fit. All 9 shapes are valid and bit-identical to the baseline (PCC,
max_abs and mean_abs equal to every printed digit), so the writer's slot indexing and the
multicast slot addresses are correct at depth 3. No hang.

Against r01-b01-a02 (same code apart from ring depth; noise ±1%):

| shape | b01-a02 µs | this µs | Δ |
|---|---|---|---|
| dsv3 | 563.1 | 561.9 | -0.2% |
| kimi-k2.7 | 771.5 | 767.6 | -0.5% |
| kimi-k3 | 1478.8 | 1477.7 | -0.1% |
| glm | 486.9 | 488.2 | +0.3% |
| mistral | 185.9 | 193.2 | +3.9% (slower) |
| dsv4-flash | 340.7 | 337.6 | -0.9% |
| dsv4-pro | 1148.9 | 1153.2 | +0.4% |
| minimax-m3 | 378.2 | 385.6 | +2.0% (slower) |
| gpt-oss | 364.5 | 368.5 | +1.1% |

I expected +1-4% on the jitter-bound shapes. I got nothing outside noise except two slowdowns.
The slower shapes (mistral, minimax) are the two with one expert per group. Their runs vary a lot
from run to run (mistral measured 186-200 µs across b01-a01/b01-a02/b02-a02, which share the same
schedule), so those two may be noise rather than an effect.

## Why (best explanation, with profiler evidence)
Per-core TRISC1 (math) kernel-end times from `reports/<node>/.logs/profile_log_device.csv`, averaged
over the 10 measured calls, as min / median / max µs. The script is
/tmp/r01b04a03/spread.py: 13 calls per shape, warmups dropped.

| shape | r01-b01-a02 | this node |
|---|---|---|
| dsv3 | 489/526/562 | 477/516/561 |
| kimi-k2.7 | 691/743/770 | 691/745/766 |
| glm | 424/459/486 | 420/457/487 |
| mistral | 153/166/184 | 151/167/192 |
| dsv4-flash | 295/319/339 | 293/313/336 |
| minimax-m3 | 322/352/377 | 322/351/384 |
| gpt-oss | 355/359/363 | 355/361/367 |

- The **median core did not speed up**, except about 2% on dsv3 and dsv4-flash. So with 2 slots
  the per-core steady state was not throttled by the row/column lock-step. Two blocks of slack
  were already enough, and the cores run at their own rate: ~31 cycles per tile-mult, bounded by
  the bf4 srcA unpack per mult plus the shared DRAM.
- The **slowest group did not catch up**. The max-minus-median spread (the 6-14% group imbalance
  r01-b01-a02 found) is unchanged or slightly worse. So that imbalance is not short-term jitter a
  deeper FIFO could absorb. One group is slower for the whole call. That points to a persistent
  bandwidth share, e.g. DRAM bank / NoC-path position of that group's senders and x owners,
  rather than transient stalls.
- At these sizes, the per-expert phase transition (gate/up -> down) and fill/drain don't get
  shorter from an extra slot either. The reader's down-phase iteration is still gated by the
  activated multicast of block 0, which needs compute to finish gate/up + activation.
- I don't know which shapes actually got 3 slots (the factory doesn't log it). By my fitter
  replica, dsv4-pro and minimax-m3 likely stayed at 2/2. Their deltas are noise anyway.

## Classification
neutral (within noise; mistral/minimax regressions are probably run-to-run variance of the
1-expert-per-group shapes). Ring depth / pipeline slack is not a lever on the row-group schedule.
The implementation is correct and cheap, but not worth keeping on its own merit.

## What a child of this node should try next
Treat this node as r01-b01-a02 for practical purposes. It ports that code into branch b04, so
children here can build on the best schedule. If you want the cleaner base, drop the 3-slot change
(revert the factory's gu_slots/d_slots and the writer IN1_*_SLOTS defines).
1. **Persistent per-group slowness.** One group is slow for the whole call, so find which one and
   why. Map each core's math-end time to (group, column, physical x/y) over the 10 calls. Check
   whether the slow group is always the same rows (NoC distance to the DRAM columns at physical
   x=0/9, or the x-owner / weight-sender placement inside the group). If it is position-bound,
   the fix is placement: a different row->group mapping, or staggering which group gets the
   larger/smaller experts. Don't add more buffering.
2. **gpt-oss is the outlier at 186 GB/s total DRAM.** Its 9-tile experts on 4-row groups leave one
   row in four idle (per_core_M 3 on rows 3,3,3,0). Split such an expert's N (or the down K)
   across the two groups, or use non-uniform group heights.
3. **DRAM-bound big shapes** (kimi-k2.7 434, dsv3 416, glm 413, dsv4-pro 412 GB/s total) have at
   most ~10% left. Don't spend attempts on them unless the attempt cuts bytes or improves DRAM
   efficiency (bank/NoC placement).
4. Things now ruled out on this lineage: read-ahead (b03-a01, b04-a01), wider K-blocks
   (accuracy, b04-a02), x batching (b03-a02), deeper rings (this node), and taller matmul
   subblocks. The BH matmul LLK unpacks the non-reused operand once per tile-mult for any rt/ct
   (llk_unpack_AB_matmul.h, reuse_a = ct >= rt), so subblock shape doesn't cut per-mult unpack.
