# r08-b04-a01 result: 1.7812 (ok). Win: +2.4% over the parent r07-b02-a01 (1.7387), outside the ±1% noise band; new campaign best. Bit-identical outputs

## What happened vs expected
- One host build (factory: CB c_21 + `RE_PACE_CB` define) and one device run. No JIT error, no
  kernel-config overflow, no hang. The local estimate held: reader +964 B, TRISC1 -1020 B (RE_HWPROF opt-in).
- All 9 shapes are valid. PCC / max_abs / mean_abs equal the parent's on every printed digit.
- The mechanism did exactly what it was built to do: on every paced shape the row groups now run in
  lockstep, and max row end ≈ mean row end. Expected +5..+7% geomean if the aggregate stream rate held.
  Got +2.4%, because the aggregate rate fell by 0-15% on several shapes (mean end rose), most of all on
  kimi-k3.

Chip µs, parent -> this node, with mean busy-row end / max row end (/tmp/r07b02/ana.py on both reports):

| shape | chip µs | Δ | mean end | max row end | GU / DN per mean core |
|---|---|---|---|---|---|
| dsv3 | 510.1 -> 478.5 | **-6.2%** | 454 -> 469 | 509 -> 477 | 324/111 -> 323/126 |
| kimi-k2.7 | 700.7 -> 675.5 | **-3.6%** | 672 -> 671 | 699 -> 673 | 495/154 -> 457/186 |
| kimi-k3 | 1212.3 -> 1307.4 | **+7.8%** | 1070 -> 1236 | 1212 -> 1306 | 647/398 -> 838/372 |
| glm | 441.0 -> 419.9 | **-4.8%** | 394 -> 409 | 440 -> 418 | 277/98 -> 280/110 |
| mistral | 167.1 -> 160.5 | -4.0% | 146 -> 154 | 167 -> 159 | 102/34 -> 105/40 |
| dsv4-flash | 287.0 -> 286.9 | 0.0% | 263 -> 281 | 287 -> 285 | 169/84 -> 191/80 |
| dsv4-pro | 997.3 -> 956.1 | **-4.1%** | 951 -> 950 | 996 -> 954 | 683/256 -> 655/281 |
| minimax-m3 | 333.7 -> 316.9 | **-5.0%** | 295 -> 309 | 333 -> 316 | 209/81 -> 214/91 |
| gpt-oss | 229.2 -> 227.3 | -0.8% | 217 -> 218 | 229 -> 227 | 150/61 -> 149/64 |

## Why (profiler evidence)
Per-row zones, mean of the 10 measured calls (/tmp/r06b01a01/zones.py on both reports):
1. **The gradient is gone, and so is the tail.**
   - dsv3 phase A, top -> bottom group, end of the first expert: parent 373 / 340 / 288 / 215 / 203 µs;
     now 295 / 293 / 300 / 304 / 290. Phase B therefore starts on all rows at ~295-306 and ends at
     ~465-475 on every row (parent: 509 on y2-5, 382 on y9-11).
   - mistral GU: parent 122 / 113 / 93 / 83; now 113 / 106 / 100 / 100.
   - kimi-k2.7 and dsv4-pro: max row end within 2-4 µs of the mean.
   So the share gradient was a share effect: holding the leaders gives the top groups their bandwidth.
2. **But lockstep costs aggregate rate, and it costs most where groups run many experts.** The mean row
   end rose on 6 of 9 shapes (dsv3 +15, glm +15, mistral +8, mm3 +14, dsv4-flash +18, kimi-k3 +166 µs).
   - kimi-k3 shows the cause: every group now runs each expert in the same window, with GU ~155 µs and
     DN ~65 µs per expert on every row. The parent's groups were out of phase: some groups' DN (activated
     stream + down reads, much lighter on DRAM per µs) overlapped other groups' GU, so DRAM stayed busy.
     In lockstep all groups are in GU together (DRAM-saturated, slower GU: 838 vs 647 µs per mean core)
     and then in DN together (DRAM under-used), so the aggregate rate drops ~10%.
   - On top of that kimi-k3's deal is 6/6/6/5/5, and the absolute-progress board makes the 5-expert
     groups wait for the 6-expert ones: they finish at ~1140 and the 6th expert runs on 3 groups to ~1300.
     The parent's top groups (6 experts) ended at 1212.
   - dsv4-flash (2 experts per group) is the same effect in small: the spread closed (287 -> 285 max) but
     the mean rose by as much (263 -> 281).
   - Single-expert shapes and phase A of two-phase plans have one GU -> DN transition per group, so the
     lockstep costs only one aligned ACT/DN window, and the tail cut wins clearly.
3. kimi-k2.7 / dsv4-pro: GU fell (495 -> 457, 683 -> 655 per mean core) and DN rose (154 -> 186,
   256 -> 281): the conserved GU+DN pattern again, with the tail removed on top.
4. The pacing overhead itself is not visible: gpt-oss (its slow groups are compute-bound pcM-4 groups, so
   pacing only holds the others) is -0.8%, mean end 217 -> 218.

## Classification
win (+2.4% geomean, bit-identical, host+kernel). The first closed-loop fix of the row gradient: the tail
collapses on 8 of 9 shapes. One measured flaw: the lockstep it imposes kills the GU/DN phase mixing
between groups, which costs aggregate rate on many-expert groups (kimi-k3 +7.8%, dsv4-flash flat).

## What a child of this node should try next
1. **Let groups drift in phase while still bounding the lead (fix kimi-k3, dsv4-flash).** Options, one
   per node, cheapest first:
   - (a) Pace only the first expert of each group (or only units of the first chunk): the gradient is
     mostly a first-expert, synchronized-start effect (r06-b01-a01 note 3), and later experts drift
     apart naturally. In `Pacer::start`, stop gating once this group has started its 2nd expert (pass
     the expert index, or mark the group done after its first expert with `publish(kPaceDone)`). Keeps
     the single-expert / phase-A wins exactly and gives kimi-k3 / dsv4-flash back their phase mixing.
   - (b) Larger slack (half an expert's weight tiles instead of one gate/up block) so a group can be in
     DN while another is in GU.
   - (c) Disable pacing when the paced phase has > 2 experts per group (kimi-k3 only in the benchmark).
   Expected: kimi-k3 back to ~1200 or better, dsv4-flash -3..-6%, others unchanged: about +1..+1.5%.
2. **Fractional progress for uneven deals** (kimi-k3 6/6/6/5/5): the first build of this node compared
   done_g / total_g (proposal point 4) but cost +1.2 KB reader. With plan-once (r07-b01-a01) the reader
   has room; per-group totals can also be computed once and passed in the plan page.
3. **Combine with r07-b01-a01 (plan once + idle-row tie pass)** if a sibling's port wins: mistral / mm3
   on 3+3+2+2 have no idle group, and pacing then also covers the 2-row-vs-3-row imbalance r07-b01-a01
   measured on mm3. Plan-once also frees ~8 KB for (2).
4. **Now the plan can assume equal group speeds again.** With the gradient removed, the r07-b02-a01
   next #1 / r07-b04-a01 rebalance ideas (move work to the bottom groups) are obsolete under pacing:
   don't stack them on this node.
5. Don't retry: static stagger / tilt / idle-top placement (rotate the loser), r01-b01-a03's per-block
   multicast board (cost). Never issue the board read or atomic inside an open linked multicast.
6. Tools: /tmp/r08b04/sz/b.sh (reader local size, XDEF="-DRE_UP_HEAD_ROWS=3 -DRE_PACE_CB=21" RWM=1),
   /tmp/r08b04/kc.sh + tA/tB (TRISC1 with/without -DRE_HWPROF), /tmp/r06b01a01/zones.py (per-row zones),
   /tmp/r07b02/ana.py (mean end / max row end / GU / DN). `DS_NO_PACE=1` restores the parent behaviour.
