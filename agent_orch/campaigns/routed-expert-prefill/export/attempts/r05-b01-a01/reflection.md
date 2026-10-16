# r05-b01-a01 result: 1.6457 (ok). +1.1% over the parent r04-b02-a01 (1.6275), just outside the ±1% noise band; new campaign best. The whole gain is minimax-m3 and mistral, the two shapes whose plan changed

## What happened vs expected
- All 9 shapes are valid on the first and only run, including the host rebuild for the new CB. PCC, max_abs and
  mean_abs are bit-identical to the parent and the baseline on every printed digit.
  - So the shared plan page carries the same plan the three RISCs used to compute.
  - The 3+3+2+2 deal for minimax-m3 / mistral produces the same outputs.
- No hang, no JIT error, no kernel-config-buffer overflow. The writer and compute UNPACK no longer contain a planner.
- Expected: minimax-m3 -3..-5%, mistral ~0, others flat, geomean +0.5..+0.8%.
  Got: minimax-m3 -5.9%, mistral -4.1%, others within their usual spread.

Chip µs vs parent, with mean core end and first-GU-start (/tmp/r04b04/zones2.py, mean over the 10 measured calls):

| shape | chip µs | Δ | mean core end | first-GU-start |
|---|---|---|---|---|
| dsv3 | 512.8 -> 510.3 | -0.5% | 482 -> 479 | 6.3 -> 6.7 |
| kimi-k2.7 | 717.3 -> 729.2 | +1.7% | 708 -> 721 | 7.8 -> 8.2 |
| kimi-k3 | 1342.8 -> 1327.6 | -1.1% | 1244 -> 1233 | 21.9 -> 23.4 |
| glm | 449.2 -> 443.9 | -1.2% | 421 -> 420 | 6.4 -> 6.6 |
| mistral | 183.8 -> **176.2** | **-4.1%** | 160 -> 156 | 3.7 -> 6.1 |
| dsv4-flash | 308.7 -> 313.1 | +1.4% | 287 -> 292 | 8.8 -> 9.3 |
| dsv4-pro | 1056.4 -> 1063.6 | +0.7% | 1044 -> 1054 | 7.5 -> 7.9 |
| minimax-m3 | 372.9 -> **351.0** | **-5.9%** | 343 -> **319** | 3.7 -> 6.1 |
| gpt-oss | 255.8 -> 254.0 | -0.7% | 227 -> 223 | 4.3 -> 4.7 |

## Why (profiler evidence)
1. **The idle-row tie pass is the mechanism.**
   - minimax-m3:
     - The mean core end fell 24 µs. The two 3-row groups (6-tile experts at per_core_M 2) end at ~300-345 µs per
       call, and GU fell 228.6 -> 209.4 µs per mean core.
     - The chip time follows the mean, and the per-call max fell 372 -> 351.
     - This reproduces r04-b03-a03 (-4.6% there) without that node's -Os planner penalty.
   - mistral:
     - It also gained this time (-4.1%; r04-b03-a03 measured 0). Its mean core end fell 160 -> 156 µs, and
       max-row-median 178.7 -> 174.1.
     - The rest is its ±5% lottery.
   - Per-row ends (/tmp/r05b01a01/rows.py): rows y10-11 now work. In 3-4 of 10 calls the 2-row pcM-3 group at
     y10-11 is the slowest by a wide margin: mistral 198-205 vs ~160, minimax-m3 385 / 419 vs ~330. That is the
     same single-group lottery the parent had on its 2-row groups.
2. **Plan-once is latency-neutral, as in r03-b01-a02.**
   - First-GU-start moved +0.2..+0.5 µs on the 7 unchanged shapes, and +1.5 µs on kimi-k3, the 28-expert plan.
     Compute now waits for the reader's plan page instead of planning in parallel. Overlapping the `start` read
     with planning did not fully pay for that.
   - mistral / minimax-m3's +2.4 µs comes with their new 3-row layout (x owners of 3-row groups), not the hand-off.
   - All of this is ≤0.2% per shape.
3. **The 7 unchanged shapes moved within their known spreads.**
   - Their plans and code paths are identical. Only the plan hand-off changed.
   - kimi-k2.7 +1.7% has GU 476.6 -> 483.2 per mean core, inside the 468-498 range identical GU code has shown
     across nodes (r04-b04-a02). dsv4-flash +1.4% and dsv4-pro +0.7% are similar.
   - glm, kimi-k3, dsv3 and gpt-oss moved the other way by similar amounts.
   - Geomean over those 7 ≈ 0.

## Classification
win (+1.1%, just outside noise), mechanistically attributable to minimax-m3 / mistral's new layout. Bit-identical.
Plan-once frees the writer's and UNPACK's planner copies (~4-5 KB each at -O2). The h7168 programs now have
several KB of kernel-config headroom on the best lineage, where the parent had a few hundred bytes.

## What a child of this node should try next
1. **Spend the headroom.** Nodes that overflowed or had to drop pieces on this lineage can now fit. Candidates:
   - r04-b04-a02's 4-slot activated ring + run-ahead, gated to kimi-k3-like shapes (k3 DN -8.6%, chip -1.2%);
   - r04-b02-a02's pump with a non-blocking act send (k3 / mistral DN gains, without the h7168 loss).

   Check sizes with the JIT ELFs in the eval cache, not local -O2 builds.
2. **The 2-row pcM-3 group (y10-11) on mistral / minimax-m3 is now the tail in ~1/3 of calls.**
   - Measure before acting: per-block NCRISC timestamps on that group only.
   - Alternatively deal the larger 5-tile expert to the earliest rows. The order of the 2-row groups in the layout
     is free.
3. **The rest of the lottery is per-group and per-call, not positional or expert-specific.**
   - Analysis on the parent (/tmp/r05b01a01/outl.py, cores.py): the per-call mean of group ends is stable (mistral
     158-174) while the max varies 168-204.
   - Sometimes it is one column of one group (mistral call 1: column x3 of y8-9, DN 107 vs 37 µs).
   - It persists even when that group runs alone at the end, so it is not DRAM share. It is the biggest remaining
     waste on mistral / minimax-m3 / gpt-oss / kimi-k3 / dsv4-flash (10-20% vs their DRAM floor).
   - A measurement node with per-block DN/GU handshake timestamps on 1-expert shapes would locate it.
4. Don't retry: an -Os planner on any RISC that gates first work (r04-b03-a03), DM handshake decoupling on
   DRAM-bound shapes, request VCs, deeper rings alone.
