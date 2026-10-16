# r06-b04-a01 result: 1.6931 (ok). Win: +1.6% over the parent r05-b02-a01 (1.6667), outside the ±1% noise band; new campaign best. Bit-identical outputs

## What happened vs expected
- All 9 shapes are valid on the first and only run. Host rebuild OK, no JIT error, no hang, no kernel-config
  overflow. The writer's -Os planner paid for the reader's growth.
- PCC, max_abs and mean_abs equal the parent and the baseline on every printed digit. So the moved senders in
  3+-row groups, the new x-owner sets and the writer's down multicast deliver the same bytes to the same slots.
- Expected: DN -5..-15% and +1..+3% geomean. Got: DN -7..-14% per mean core on 8 of 9 shapes and +1.6% geomean.
  Part of the DN gain reappeared as GU on the DRAM-bound shapes.

Per mean busy core, parent -> this node (/tmp/r04b01a02/rows.py + /tmp/r06b04/agg.py on reports/{r05-b02-a01,r06-b04-a01}):

| shape | chip µs | Δ chip | DN | GU | mean end | max row end |
|---|---|---|---|---|---|---|
| dsv3 | 504.9 -> 494.4 | -2.1% | 122 -> 112 (-8%) | 317 -> 330 | 459 -> 462 | 503 -> 490 |
| kimi-k2.7 | 701.4 -> 695.4 | -0.9% | 180 -> 168 (-7%) | 473 -> 484 | 676 -> 675 | 698 -> 694 |
| kimi-k3 | 1243.6 -> 1234.0 | -0.8% | 445 -> 413 (-7%) | 625 -> 638 | 1096 -> 1076 | 1216 -> 1214 |
| glm | 436.7 -> 442.0 | +1.2% | 107 -> 109 | 271 -> 285 | 398 -> 413 | 435 -> 440 |
| mistral | 190.1 -> 180.2 | **-5.2%** | 40 -> 37 (-8%) | 108 -> 106 | 160 -> 154 | 177 -> 172 |
| dsv4-flash | 298.4 -> 294.7 | -1.2% | 94 -> 86 (-9%) | 165 -> 170 | 270 -> 266 | 294 -> 284 |
| dsv4-pro | 993.2 -> 985.2 | -0.8% | 296 -> 271 (-8.4%) | 649 -> 665 | 958 -> 949 | 990 -> 984 |
| minimax-m3 | 361.9 -> 347.8 | **-3.9%** | 95 -> 88 (-7%) | 222 -> 221 | 325 -> 315 | 359 -> 329 |
| gpt-oss | 261.0 -> 260.3 | -0.3% | 73 -> 63 (-14%) | 151 -> 153 | 232 -> 222 | 247 -> 247 |

## Why (profiler evidence)
1. **The wrapped down multicasts were costing DN time on every layout.**
   - mistral and minimax-m3 run the 5x2 layout on this lineage (rows y10-11 idle). Their sender placement is
     unchanged here: every 2-row group keeps `base + c % 2`. The only change on those shapes is that
     odd-column (bottom-row) senders multicast down on the writer's NoC 1 instead of wrapping on the reader's
     NoC 0.
   - That change alone gave DN -7..-8% and chip -3.9 / -5.2%. They are near compute-bound, so the gain did not
     leak into GU.
   - So the wrap penalty r05-b04-a01 inferred from wait tallies is real and causal. It also confirms that the
     move to the writer adds no hazard.
   - Bottom senders also gain overlap: the reader's head read of block kb+1 now runs during the writer's
     multicast of kb.
2. **On the DRAM-bound shapes the saving is partly conserved.**
   - GU grew +11..+16 µs per mean core on dsv3, k2.7, k3, glm and dsv4-pro, while DN fell 10-32 µs. This is the
     same conserved GU+DN signature as r02-b02-a03 / r03-b04-a03 / r04-b03-a02: down blocks now finish faster,
     and the down stream takes DRAM/NoC share from the groups still in gate/up.
   - Net mean-core end: -1..-20 µs on k2.7, k3, dsv4-pro and dsv4-flash; +3 on dsv3; +15 on glm. glm is the one
     chip loss (+1.2%, within its spread).
3. **The 3+-row groups (phase B of dsv3/glm/dsv4-pro/k2.7, 4+3+3 / 5+5) now use corner senders only.** dsv4-pro
   DN -8.4% and k2.7 -7%. The link model predicted the biggest change on 5+5: down column links 75 -> 35 GB/s.
   I can't separate placement from NoC choice on those shapes in this run.
4. The DN zone is lock-step along a row (activated stream), so per-column DN is the same on both parities before
   and after (/tmp/r06b04/cols.py). The wrap cost showed up as a longer DN for the whole row, not as a
   per-column difference.

## Classification
win (+1.6% geomean, bit-identical, host+kernel). It removes the down-phase half of r05-b04-a01's wrap
diagnosis. That hung node's direction is now a measured gain.

## What a child of this node should try next
1. **Do the same for gate/up, the other half of the wrap diagnosis.**
   - Top-row senders still multicast gate/up on the writer's NoC 1 (-y), so it wraps through the whole column.
     That is half of every 2-row layout's columns, all even columns. r05-b04-a01 measured gate/up wrap waits
     1.4-4x.
   - Moving them to the reader's NoC 0 (the existing `!writer_mcasts_in1` code, made a runtime choice on
     `my_mt == 0`) serializes the reader's 60% gate+up-head read behind its own multicast flush. That is the
     cost IN1_WRITER_MCAST was built to avoid.
   - Options:
     - (a) Try it as-is and measure. The wrapped multicast contends with up to 5 other wrapped multicasts per
       column, so the flush may get much shorter.
     - (b) For top senders only, swap the read split: the writer reads more of the block on NoC 1 so the
       reader's serialized part is small. That costs NoC balance on half the senders.
   - Watch kernel size: the reader's multicast path is compiled out today.
2. **Combine with r05-b01-a01** (plan once per core + idle-row tie pass, +1.1% on mm3/mistral). That lineage
   frees several KB of kernel-config space.
   - Its 3+3+2+2 layout for mistral/mm3 has 3-row groups, which this node's corner placement covers. Its tie
     pass alone won -4..-6% on those shapes, and this node's DN gain should stack.
   - Port order: this node's 5-file diff is small; adaptive_chunk.hpp conflicts only in the x_owner region and
     the pragma block.
3. **The conserved GU growth on the DRAM-bound shapes is the next limit.** Down now finishes faster and takes
   share from groups still in gate/up. A planner-side rebalance of the row groups (r05-b02-a01 next #1: deal
   extra experts to the faster lower groups, put dsv3/glm's phase-B 4-row group at the bottom) is still open
   and independent of this node.
4. **Size:** this node is about +0.4 KB vs the parent (local: reader +420, writer -364 with the -Os planner,
   TRISC0 +368 from an inlining flip, TRISC1 -60, TRISC2 +8), and it fit. Assume only a few hundred bytes of
   headroom left on h7168 without plan-once.
5. **TILE path note:** on odd-height groups (3+3+2+2, 4+3+3, 5+5) the bottom row's fixed TILE x sender is now
   also that column's weight sender. It is functionally the original (0,0) arrangement and only slower on the
   unbenchmarked TILE path. The host TT_FATAL still guards 2-row groups.
6. Don't retry: moving sender rows per phase (r05-b04-a01 hang), whole-stream NoC moves (r02-b04-a02,
   r03-b03-a02).
7. Tools: /tmp/r06b04/nocw2.py (link model with corner placement + per-position NoC choice),
   /tmp/r06b04/chk2.py (placement / x-owner consistency over all layouts), /tmp/r06b04/agg.py and cols.py
   (mean GU/DN/end per shape, per-column DN), /tmp/r06b04/sz/kc.sh + kcb.sh (TRISC size, new vs base).
