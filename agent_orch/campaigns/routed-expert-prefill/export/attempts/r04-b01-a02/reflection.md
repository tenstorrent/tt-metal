# r04-b01-a02 result: 1.5915 (ok). Neutral-to-negative: -0.6% vs the parent r04-b01-a01 (1.6011, inside the ±1% noise) and -2.2% vs r04-b02-a01 (1.6275), whose reader this node ports. The activated look-ahead did not shorten the down phase, and the port did not keep r04-b02-a01's down gain

## What happened vs expected
- **Run 1 (not scored): runtime_error on kimi-k2.7 only.** `Program size (70816) too large for kernel config buffer
  (70656)`, 160 B over, at program creation. The other 8 shapes ran and were bit-identical. My local size
  estimate (~0.7 KB margin) was wrong by ~0.9 KB: local -O2 object sizes do not track the JIT's LTO build
  closely. Repair (see proposal addendum): one shared out-of-line `try_act` step for the receiver ack and the
  sender multicast, plus the launch planner built at -Os on the writer (BRISC) only. That is ~-0.96 KB locally.
- **Run 2 (scored):**
  - All 9 shapes are valid. PCC / max_abs / mean_abs are bit-identical to the parent and the baseline, so the
    parity valid flags, the early acks / multicasts and the next-slot addressing deliver the right blocks.
  - No hang. kimi-k2.7 fits now.
- I expected DN -10..-30% on the shapes with the largest activated wait, plus r04-b02-a01's ~+1%.
  I got neither.

Chip µs and MATH zones (mean core, /tmp/r04b04/zones2.py), parent (r04-b01-a01) / r04-b02-a01 / this node:

| shape | chip µs | mean core end | mean GU | mean DN |
|---|---|---|---|---|
| dsv3 | 513.5 / 512.8 / 524.7 | 493 / 482 / 494 | 328.6 / 327.6 / 333.8 | 145.6 / 135.9 / 141.7 |
| kimi-k2.7 | 729.2 / 717.3 / 738.2 | 720 / 708 / 725 | 471.2 / 476.6 / 486.0 | 225.8 / 207.5 / 214.3 |
| kimi-k3 | 1360.0 / 1342.8 / 1349.9 | 1283 / 1244 / 1253 | 726.2 / 727.0 / 737.9 | 532.1 / 492.6 / 489.5 |
| glm | 450.8 / 449.2 / 451.0 | 430 / 421 / 423 | 284.3 / 284.1 / 282.3 | 127.9 / 118.2 / 121.4 |
| mistral | 195.6 / 183.8 / 192.9 | 168 / 160 / 167 | 108.7 / 106.0 / 106.3 | 49.6 / 44.6 / 51.6 |
| dsv4-flash | 320.3 / 308.7 / 322.6 | 298 / 287 / 300 | 185.1 / 181.8 / 183.0 | 103.3 / 95.5 / 106.6 |
| dsv4-pro | 1067.0 / 1056.4 / 1069.9 | 1054 / 1044 / 1054 | 685.5 / 693.5 / 687.1 | 355.4 / 337.0 / 354.1 |
| minimax-m3 | 368.6 / 372.9 / 366.0 | 346 / 343 / 340 | 227.8 / 228.6 / 228.8 | 113.1 / 109.2 / 106.4 |
| gpt-oss | 259.7 / 255.8 / 269.6 | 236 / 227 / 242 | 156.0 / 151.1 / 158.3 | 74.9 / 70.8 / 78.3 |

## Why (profiler evidence)
1. **DN is back at the parent's level on 7 of 9 shapes.** It is not at r04-b02-a01's level, even though this node
   carries r04-b02-a01's down weight look-ahead verbatim, spin re-check included.
   - DN vs r04-b02-a01: +3..+11 µs on dsv3, k2.7, glm, mistral, dsv4-flash, dsv4-pro, gpt-oss.
   - Only kimi-k3 (-3) and minimax-m3 (-3) are below it.
   - Per row (/tmp/r04b01a02/rows.py) the same holds on every row group: mistral DN 45-57 µs (r04-b02-a01:
     41-47), dsv4-flash 100-116 (91-101), gpt-oss 66-86 (66-80).
   - So the activated stream running ahead did not remove DN time. It most likely cost some of the weight
     look-ahead's gain.
2. **The likely reason is that the activated wait in r04-b03-a01's tally was not an activated-handshake cost.**
   - The compute checks activated first, so any block whose down weights are late also counts as activated wait.
   - On these shapes DN is DRAM-share-bound (mistral: 4 groups x 4.7 MB down weights in ~45 µs ≈ 420 GB/s).
   - Delivering activated(kb+1) earlier therefore cannot shorten a block period set by the down-weight stream.
   - This is the same lesson as r04-b04-a01 in gate/up: early handshakes only reorder who waits when the paced
     stream is the weights.
3. **Why it is slightly worse than r04-b02-a01, not just equal.** This run can't separate three candidates:
   - (a) More concurrent multicasts on the reader's NoC. Each early activated multicast is a row-wide linked
     INCL_SRC multicast that holds its path reserved until the valid write. It now overlaps the column
     weight multicasts and the DOWN_SPLIT reads, where before it followed them in a fixed order.
   - (b) This node switched off r04-b02-a01's gate/up look-ahead to make room. Its reflection read the
     look-ahead's GU +2..+5 / DN -x on DRAM-bound shapes as a conserved GU+DN shift. GU here is +1..+9 µs vs
     r04-b02-a01 on dsv3 / k2.7 / k3 / gpt-oss, so that is not the explanation either.
   - (c) Run-to-run spread. r04-b02-a01's own +1.07% was just outside the ±1% band, so part of its DN gain may be
     lottery. Its DN drops were consistent across 9 shapes, though.
4. **The other changes are not on the critical path.**
   - First-GU-start is unchanged (6.5 vs 6.4 µs on dsv3, 22.0 vs 21.9 on k3), so the writer's -Os planner costs
     nothing visible.
   - Dropping the RE-HW counters and the CFGSHIFTMASK MOP is timing-neutral, as both were measured before.

## Classification
flawed idea (measured: no DN gain from decoupling the activated stream). The implementation is correct and
bit-identical, and the run-1 overflow was a repairable size failure. The combination with r04-b02-a01's reader
lost that node's DN gain: this node is ~parent level, 2.2% under r04-b02-a01.

## What a child of this node should try next
1. **Don't build on this node. Start from r04-b02-a01** (best, 1.6275). If a child wants this node's size savings
   on that lineage, take only these:
   - `adaptive_chunk.hpp`'s writer-only -Os planner pragma: ~0.8 KB, no visible cost;
   - the RE-HW counters / MOP removal: they only exist on branch b01's lineage anyway.
2. **Don't retry handshake decoupling in either phase:** activated look-ahead (this node), gate/up early acks
   (r04-b04-a01), x owner / trids (r03-b02-a02, r03-b04-a02). Both phases' block periods are set by the
   weight stream's DRAM share, and the compute's "x / activated wait" tallies (r04-b03-a01) include weight
   lateness, because the row stream is waited first.
3. If the activated question is ever revisited, measure before building:
   - wait for the down weights FIRST in the compute (or tally both waits independently, each from block start);
   - check whether the weight share alone explains DN. My prediction is that it does.
4. **Size budget on the h7168 programs (hard data):**
   - run 1 of this node = 70816 B on kimi-k2.7 (limit 70656);
   - its reader at -O2 local = 20136 B, vs r04-b02-a01's reader 18324 B and the root's 16352 B;
   - compute TRISC0/1/2 local 19500 / 13220 / 14992 B (parent 20176 / 14680 / 14992).

   Local -O2 sizes are good for deltas within ~±0.3 KB, but not for absolute margins. The real fix for this
   lineage is r03-b01-a02's plan-once-per-core (~12 KB).
5. Tools: /tmp/r04b01a02/build_o2.sh (JIT-like -O2 reader/writer compile, `RSRCP=<file>` for an alternative reader
   source), /tmp/r04b01a02/sz/kc.sh (3-TRISC compile), /tmp/r04b01a02/rows.py (per-row end / GU / DN).
