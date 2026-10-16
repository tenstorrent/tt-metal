# Round r08 summary: routed-expert-prefill

- **Policy:** `v2` (one-step rounds: W=4, R=1), beta 0.6. This is the fourth round on v2. Dreaming after r07 produced no `v3` (driver log 06:26:11, "keeping v2").
- **W / R:** W=4 branches, R=1 attempt per branch (max_steps 30). Step 1 ran 4 workers from the root. Step 2 returned an empty batch with `stop: true`.
- **Round root:** `dream/routed-expert-prefill/n/r07-b02-a01` @ `a13919e548e`, score **1.7387**.
  - This is r07's best node: wrap-free gate/up weight multicasts (top-row senders multicast gate/up on the reader's NoC 0). Kernel-only, bit-identical.
  - The pre-round drift check measured geomean +0.01%, worst shape mistral -0.34% (noise band 1.0%). As in earlier rounds it re-measures the campaign root (baseline code at `c54a00a7697`), not the round root.
- **Wall time:** 06:27:07 → 06:58:20 (+02:00), 31 min (r07: 30 min, r06: 1 h 19 min). b03 and b04 were done by 06:48:34. b02 finished at 06:53:21. b01 committed at 06:56:15 but its session did not exit and was stopped at 06:58:20 (see Ops notes).
- **Attempts:** 4 committed, 4 valid, 0 runtime_error, 0 accuracy_fail, 0 hang, 0 lost.
  - Each node had one eval and one device run. No in-attempt repairs (r07 had two).
  - All 4 scored outputs are bit-identical to the root and the baseline (PCC / max_abs / mean_abs on every printed digit).
- **Branches closed:** none. With R=1 every branch ends after its one attempt.
- **Round gain:** 1.7387 → 1.7812 (+2.4%), outside the ±1% noise band. Two nodes beat the root outside noise (b04-a01 +2.45%, b02-a01 +2.36%). They are 0.08% apart, so they are tied within noise. Two were neutral (b03-a01 +0.5%, b01-a01 +0.06%). No node lost on geomean.

## Best node: `r08-b04-a01`, score 1.7812

Mechanism: closed-loop cross-group pacing of the weight stream.
- Each row group's weight senders hold their next block's DRAM reads while the group is more than one gate/up block of weight tiles ahead of any unfinished group.
- Progress lives on a per-group board on core (0,0): one atomic + one 32-B read per block. The board is never touched inside an open linked multicast.
- Pacing covers the first phase only.
- Host+kernel: the factory adds CB c_21 and the `RE_PACE_CB` define. Reader +964 B locally, paid for by making the `RE_HWPROF` MATH counters opt-in (TRISC1 -1020 B). `DS_NO_PACE=1` restores the parent behaviour.

Lineage: r07-b02-a01 (root) → b04-a01. One run.

| shape | baseline µs | round root µs | best µs | speedup vs baseline | Δ vs round root |
|---|---|---|---|---|---|
| deepseek-v3-h7168-i2048 | 801.27 | 510.11 | 478.45 | 1.675 | -6.2% |
| kimi-k2-7-h7168-i2048 | 1161.99 | 700.66 | 675.48 | 1.720 | -3.6% |
| kimi-k3-latent-h3584-i3072 | 2570.42 | 1212.29 | 1307.42 | 1.966 | +7.8% |
| glm-5-3-h6144-i2048 | 704.06 | 440.99 | 419.92 | 1.677 | -4.8% |
| mistral-small-4-h4096-i2048 | 261.79 | 167.13 | 160.47 | 1.631 | -4.0% |
| deepseek-v4-flash-h4096-i2048 | 525.15 | 287.02 | 286.92 | 1.830 | -0.0% |
| deepseek-v4-pro-h7168-i3072 | 1604.12 | 997.34 | 956.10 | 1.678 | -4.1% |
| minimax-m3-h6144-i3072 | 477.77 | 333.69 | 316.88 | 1.508 | -5.0% |
| gpt-oss-120b-h2880-i2880 | 572.05 | 229.24 | 227.32 | 2.516 | -0.8% |
| **geomean** | | | | **1.781** | |

- **The row gradient was a bandwidth-share effect, and pacing removes it.** Example, dsv3 phase A end of the first expert, top → bottom group: 373 / 340 / 288 / 215 / 203 µs → 295 / 293 / 300 / 304 / 290. Phase B then starts on all rows at ~295-306 and ends at ~465-475 on every row (root: 509 on y2-5, 382 on y9-11). On kimi-k2.7 and dsv4-pro the max row end is within 2-4 µs of the mean.
- **Lockstep costs aggregate rate.** Mean busy-row end rose on 6 of 9 shapes (dsv3 +15, glm +15, mistral +8, mm3 +14, dsv4-flash +18, kimi-k3 +166 µs). On many-expert groups, all groups now run gate/up together (DRAM-saturated) and then down together (DRAM under-used). In the root, groups drifted out of phase, so one group's down overlapped another's gate/up.
  - kimi-k3: GU per mean core 647 → 838 µs. Its 6/6/6/5/5 deal also makes the 5-expert groups wait for the 6-expert ones under an absolute-progress board. So the 6th expert runs on 3 groups to ~1300 µs (+7.8%).
  - dsv4-flash (2 experts per group): spread closed (max 287 → 285), mean rose by the same amount (263 → 281). Net 0.
- kimi-k2.7 / dsv4-pro show the conserved GU+DN signature again: GU 495 → 457 and 683 → 655, DN 154 → 186 and 256 → 281 per mean core, with the tail removed on top. That makes six nodes with this signature.
- Pacing overhead is not visible: gpt-oss, whose slowest groups are compute-bound, is -0.8%, and its mean end moved 217 → 218.

Per-shape bests from other nodes this round:

| shape | best µs | node | vs round root |
|---|---|---|---|
| kimi-k2.7 | 674.76 | b01-a01 | -3.7% |
| kimi-k3 | 1177.56 | b01-a01 | -2.9% |
| mistral | 151.63 | b02-a01 | -9.3% |
| dsv4-flash | 286.42 | b02-a01 | -0.2% |
| dsv4-pro | 952.15 | b01-a01 | -4.5% |
| minimax-m3 | 305.06 | b02-a01 | -8.6% |

- The best node holds the round's per-shape bests only on dsv3, glm and gpt-oss (3 of 9).
- b04-a01's dsv3 478.45 is a new campaign best (previous: r06-b04-a01's 494.38). b01-a01's kimi-k3 1177.56 is a new campaign best (previous: r07-b04-a01's 1192.24).
- The per-shape minimum over the four nodes has a geomean of 1.8225, +2.3% over the best node. That number only bounds stacking. b02 and b04 touch different shapes' plans, but b01's and b04's gains overlap on k2.7 / dsv4-pro.

## Score grid (branch x attempt)

| branch | a01 | branch best | Δ vs root | outcome |
|---|---|---|---|---|
| b01 | 1.7397 | 1.7397 (a01) | +0.06% | neutral; 3 shapes won, 2 lost (flawed sub-rule) |
| b02 | 1.7797 | 1.7797 (a01) | +2.36% | win, outside noise (port / combination) |
| b03 | 1.7478 | 1.7478 (a01) | +0.52% | neutral; 5 shapes won, kimi-k3 lost 9.2% |
| b04 | **1.7812** | 1.7812 (a01) | +2.45% | win, round best (tied with b02 within noise) |

Bold marks the round best. There was one step:

| step | workers | running best | set by | step-best Δ | outside noise? |
|---|---|---|---|---|---|
| s1 | 4 | 1.7812 | b04-a01 | +0.0425 | yes (+2.4%) |
| s2 | 0 | - | stop | | |

Cross-branch code traffic that the parent links don't show:
- b02-a01 ports r07-b01-a01 (plan once per core + idle-row tie pass), an r07 sibling of the root. The diff applied with git's 3-way merge and no conflict. This is the same combination as r07-b01-a01 itself (which ported r05-b01-a01), now on the next root.
- b01-a01 implements r07-b04-a01's next #1 (bottom tie-break only when every group is busy) plus a new top-free phase-B rule. It extends r07-b04-a01's planner replay (`/tmp/r08b01/pt.cpp`).
- b03-a01 and b04-a01 both implement the dynamic share cap named by r07-b04-a01 next #3 and r07-b03-a01 next #3. They were built independently, with different boards, slack and scope (see Policy observations).
- Workers saw sibling results mid-step. b02's next #1 cites r08-b03-a01's per-shape numbers (b03 committed at 06:47, b02 at 06:53). b01's next #3 points at "b04 in this round".

## What worked

1. **Cross-group weight-stream pacing** (b04-a01, +2.45%; b03-a01, +0.5%).
   - It is the first closed-loop fix of the row gradient. On 7-8 of 9 shapes, every row group ends within a few µs of the others (b03: max − min row end under 12 µs on 7 of 9 shapes, from 50-330 µs).
   - It confirms that the gradient is a share effect. b03 measured the top group's first gate/up falling by 24% (dsv3) and 43% (k2.7). Static fixes (start-up stagger r06-b02-a01, NoC tilt r06-b01-a01, idle-top r07-b04-a01) could not get this.
   - It wins wherever each group runs 1-2 experts per phase: dsv3 -6%, glm -4..-5%, k2.7 -3..-4%, dsv4-pro -2..-4%, mm3 -2..-5%.
2. **Plan once per core + idle-row tie pass on the wrap-free root** (b02-a01, +2.36%).
   - mistral -9.3% and minimax-m3 -8.6%, against -9.8% / -0.6% when the same code went onto r07-b01-a01's wrapped root. The two mechanisms add up. After the wrap fix, the bottom groups are the fast ones. 3+3+2+2 then puts the light per_core_M-2 groups at the top, where bandwidth is short, and the per_core_M-3 2-row groups at the bottom, where bandwidth is plentiful.
   - mm3 per-row ends: 333 / 311 / 280 / 265 / idle → 303 / 280 / 280 / 266. Delivered traffic rose ~390 → ~430 GB/s (mistral) and ~420 → ~460 GB/s (mm3).
   - The other 7 shapes stayed within -0.9..+0.2%. Plan-once delays the first GU start from 3.6 to 6.0 µs, which is not visible in chip time.
   - Kernel-config headroom is back: reader +1.2 KB, writer -3.6 KB, and UNPACK loses its planner copy.
   - It needed only one device run with no repairs. The cost line shows $11.46.
3. **Bottom tie-break when every group is busy, and top-free phase B where it costs no more per expert** (b01-a01, the winning half).
   - kimi-k2.7 -3.7%, dsv4-pro -4.5%: phase B runs on 4+3+3's lower two groups and the top 4 rows stop after phase A. dsv4-pro's delivered weight rate rose ~448 → ~468 GB/s, so the root's dsv4-pro was not at a DRAM ceiling.
   - kimi-k3 -2.9%: the conditional flip reproduces r07-b04-a01's 5/5/6/6/6 gain. The `active >= groups` guard removes r07-b04-a01's mistral / mm3 loss (mistral 167.1 → 166.9, mm3 333.7 → 331.9).
   - Every plan change matched the replay prediction. Only the 5 targeted shapes moved.

## What failed and why

| node | score | outcome | reason (from its reflection) |
|---|---|---|---|
| b01-a01 | 1.7397 | neutral (+0.06%), one sub-rule flawed | The top-free phase B was made a requirement. For dsv3 / glm every top-free layout needs per_core_M 3 on 2-row groups. The planner saw that cost (t_b 509 vs 346) and dropped to single-phase 1/1/2/2/2. The middle group y6-7's second expert (2-row, pcM 3, started while 4-5 groups stream) ran 237 µs against an estimated ~190. dsv3 +5.2%, glm +6.2%, which cancelled the k2.7 / dsv4-pro / k3 wins. Repair (next #1): make top-free a preference, taken only when its t_b is ≤ the top-using layout's. Expected ~1.76. |
| b03-a01 | 1.7478 | neutral (+0.5%) | Pacing every block of every phase with a 2-block slack forces whole-phase GU/DN lockstep. kimi-k3: per-expert GU ~145 µs on every group (root 100-140), period ~225 µs (root ~170-210), +9.2%. dsv4-flash +2.3%, gpt-oss +2.7% (spread only). mm3 / mistral gained less than modelled (-2.1% / -0.4%, modelled -8..-15%). With rows y10-11 idle on the root's 2x5 layout, freeing bandwidth helped the top group little, and the lower groups were held in down waiting for the top group's gate/up tail. |
| b04-a01 | 1.7812 | win, with the same flaw on kimi-k3 | It won overall, but the same lockstep cost hit kimi-k3 (+7.8%). The absolute-progress board also made the 5-expert groups of a 6/6/6/5/5 deal wait for the 6-expert ones. b04 tried fractional progress (done_g / total_g) in its first build, but it cost +1.2 KB on the reader and was dropped. |

b03 vs b04, same idea with different details. Score gap 2.0%, mostly mistral (-0.4 vs -4.0%), mm3 (-2.1 vs -5.0%), dsv4-pro (-2.3 vs -4.1%), kimi-k3 (+9.2 vs +7.8%), dsv4-flash (+2.3 vs 0.0%):

| | b03-a01 | b04-a01 |
|---|---|---|
| who is held | column-0 weight sender of each group | all weight senders of the group |
| board | per-peer NoC-atomic board in the 16th program semaphore slot | per-group progress board on core (0,0) in a new CB |
| slack | 2 blocks behind any other group | one gate/up block of weight tiles ahead of any unfinished group |
| scope | every phase | first phase only |

Neither reflection isolates which difference mattered.

Newly established negatives (in addition to the r01-r07 lists):
- **Pacing every block of a multi-expert phase with a small slack** (b03 kimi-k3 +9.2%, b04 kimi-k3 +7.8%). Lockstep kills the GU/DN phase mixing between groups.
- **Single-phase bottom-heavy deals for dsv3 / glm** (b01-a01: +5..+6%). Top-free phase-B layouts that raise the leftovers to per_core_M 3 are worse than using the top group.
- **Absolute-progress pacing on uneven expert deals** (b04 kimi-k3). Groups with fewer experts are held for the larger ones.
- **The planner's per-expert cost under contention.** A pcM-3 expert on a 2-row group, or a pcM-1 phase-B expert on a 3-row group, runs at a share-bound rate while 4-5 groups stream (b01: 237 µs vs ~190 modelled; 190-290 µs vs ~110 alone). The planner model is optimistic for both.

Open levers named by the reflections:
- **Let paced groups drift in phase while bounding the lead** (b04 next #1, b03 next #1-#2; the convergent next #1 of the round).
  - (a) Pace only the first expert (or first chunk) of each phase. Both nodes name it as the cheapest. b03 gives the exact change (`limit = num_blocks_gu + num_blocks_d`). Expected: kimi-k3 back to ~1200, dsv4-flash -3..-6%, others unchanged, +1..+2%.
  - (b) A larger or phase-aware slack (half an expert's weight tiles, or one expert's blocks after the first).
  - (c) Disable pacing when a phase has > 2 experts per group.
- **Fractional progress for uneven deals** (b04 next #2): done_g / total_g, with per-group totals in the plan page. This needs plan-once's headroom.
- **Combine b02-a01 with b04-a01** (b02 next #1, b04 next #3, b03 next #3). They are tied for the round best and target different shapes: b02 mistral / mm3 via the plan, b04 the 1-2-expert phases via pacing. b02 frees ~3-8 KB that pacing and fractional progress need. b03 noted that pacing helps the top group more when no group is idle, which b02 provides on mistral / mm3.
- **b01 repair: top-free phase B as a preference** (b01 next #1, expected ~1.76 on the root). b04 next #4 says this is obsolete under pacing ("don't stack them on this node"). b02 next #2 says combine it if it wins. The reflections disagree.
- **k2.7 / dsv4-pro: 4+2+2+2 phase B on the y8-11 groups** (b01 next #2), a cost-model tweak.
- **Top-group NoC split tilt for mm3 / mistral's y2-4 group** (b02 next #3). It is untested, and r06-b01-a01's tilt result was ambiguous.

## Policy observations (input for dreaming)

- **v2's fourth round, same cost profile, the largest attempt count with two wins outside noise.**

  | round | attempts | wall time | worker cost | gain |
  |---|---|---|---|---|
  | r08 | 4 | 31 min | $37.01 | +2.4% |
  | r07 | 4 | 30 min | $21.45 | +2.7% |
  | r06 | 4 | 1 h 19 min (≈23 min of work) | $30.37 | +1.6% |
  | r05 | 4 | 45 min | $49.45 | +2.4% |
  | r04 | 9 | 1 h 20 min | $71.93 | +1.07% |
  | r03 | 16 | 1 h 48 min | $118.48 | +1.2% outside noise |

  - 2 of 4 attempts beat the root outside noise (r07: 2 of 4). None lost on geomean, the second round in a row.
- **Duplicated idea: b03 and b04 both built cross-group pacing.** That is half the round on one mechanism. The r07 summary flagged duplication risk, but it predicted the planner rebalance. Only b01 took the planner, and two workers converged on the dynamic share cap (r07-b04-a01 next #3 / r07-b03-a01 next #3).
  - The duplicate was not fully wasted. Two independent builds of one idea agree on the mechanism: the gradient collapses, k3 loses to lockstep, dsv3 gains ~6% in both. The 2% gap between them is a free design-variant signal, though neither node isolates which variant choice caused it.
  - But it was one slot not spent on a fourth lever. Proposal input (third time: r06, r07, r08): give each step-1 worker the levers the other slots are taking, or dedupe by mechanism tags at batch time. Both nodes carry `pacing`, `dram-fairness`, `row-gradient`.
- **The off-lineage port was chosen by a worker again, and won again.** b02-a01 ported r07-b01-a01 onto the new root: +2.36%, one run, no conflict. It is the third round in a row (r07 b01, r08 b02) where a near-mechanical port of a sibling's outside-noise win paid ≥1.6%.
- **The same pattern is set up for r09, more sharply.** Re-rooting on b04-a01 drops b02-a01, which is tied with it within noise (0.08% apart) and gains on different shapes. b04 next #3, b02 next #1 and b03 next #3 all name the combination. Proposal input, repeated from r06 and r07: seed one step-1 slot with the port of every off-lineage, outside-noise win, or have the driver apply the port mechanically and measure it. Also consider the root choice itself. When the top two are within noise and touch disjoint shapes, picking one by a 0.08% margin is a coin flip that discards the other's gain until someone ports it.
- **No wasted attempts outright.**
  - **b01-a01** is neutral, but its two parts are cleanly attributable, because they hit disjoint shapes. It has a measured one-comparison repair (expected ~1.76) and a replay check. Its value under pacing is disputed by b04 next #4.
  - **b03-a01** is the duplicate above.
- **No closes, so no premature closes.** R=1 removes the close decision.
- **Workers read sibling results mid-step.** b02's next #1 builds on b03's committed numbers, and b01's next #3 defers dsv3 / glm to b04's pacing. Every step-1 worker sees the same root, but slower workers end with more information. This is a channel the policy does not control. Dreaming could use it deliberately, e.g. a late-starting combiner slot.
- **Conserved GU+DN on DRAM-bound shapes, sixth node** (b04-a01 on k2.7 / dsv4-pro, after r02-b02-a03, r03-b04-a03, r04-b03-a02, r06-b04-a01, r07-b02-a01).
- **Noise.** With r07's wrap fix and now pacing, per-row end spread on 7 of 9 shapes is ≤12 µs. Several r08 verdicts rest on 2-5% per-shape moves that are well above that spread, so the ±1% geomean gate looks adequate on the paced lineage. The tie between b02 and b04 (0.08%) is the case where it decides nothing.
- **Support collapse:** as in r05-r07, R=1 records no depth. r08 adds four depth-1 nodes.
- **Yield:**
  - 4 of 4 valid on the first and only run (r07: 2 of 4 needed a repaired second eval).
  - 2 wins outside noise: b04-a01 (+2.45%), b02-a01 (+2.36%).
  - 2 neutral: b03-a01 (+0.5%), b01-a01 (+0.06%).
  - 0 losses.
- **Diminishing returns:**

  | round | gain | attempts |
  |---|---|---|
  | r01 | +50% over the baseline | 16 |
  | r02 | +4.6% | 16 |
  | r03 | +2.6% by score (+1.2% outside noise) | 16 |
  | r04 | +1.07% | 9 |
  | r05 | +2.4% | 4 |
  | r06 | +1.6% | 4 |
  | r07 | +2.7% | 4 |
  | r08 | +2.4% | 4 |

  The trend is flat at ~2-3% per v2 round. r05-r07 gained from NoC geometry. r08 gained from the two things the r07 reflections pointed at: a dynamic share cap (pacing) and stacking the off-lineage plan-once / tie-pass win. The open levers now are pacing scope (k3 / dsv4-flash) and the b02 + b04 combination.

## Ops notes

- **Pre-round:**
  - r07 completed at 06:22:18. Dreaming ran and produced no `v3` (06:26:11).
  - The drift check built (`build_r08_root_1.log`, 06:26:24) and measured the campaign root 06:26:25-06:27:07: geomean +0.01%, worst mistral -0.34%. As noted in r07, it checks out `c54a00a7697` and compares with `baseline.json`, so it detects machine drift only.
  - Overhead between r07 completion and r08 planning was 4.8 min (r07: 6.3 min).
- **Step duration.** The barrier is synchronous, so the slowest worker sets the step's length. Times are local (+02:00); run windows are from each report's `run.log`.

| worker | evals | build done / run window | commit | exit | note |
|---|---|---|---|---|---|
| b03-a01 | 1 | 06:43:22 / 06:43:23-06:44:09 | 06:47:13 | 06:47:22, rc=0 | |
| b01-a01 | 1 | 06:45:25 / 06:45:26-06:46:11 | 06:56:15 | 06:58:20, rc=143 | "committed but the session is still running; stopping leftover processes" |
| b04-a01 | 1 | 06:46:37 / 06:46:38-06:47:23 | 06:48:29 | 06:48:34, rc=0 | |
| b02-a01 | 1 | 06:51:53 / 06:51:54-06:52:39 | 06:53:13 | 06:53:21, rc=0 | |

  - Step 1 ran 31 min (06:27:07 → 06:58:20). The driver logged all four results at 06:58:20.
  - b01's final result message was at 06:56:18, but the process lingered ~2 min until the wrapper stopped it (rc=143). That set the step's end; the step could have closed at ~06:56:20. The wrapper handled it correctly and the tag is present. b01 also spent ~10 min between its eval (06:46:11) and its commit on profiler analysis and the reflection.
  - The four device runs did not overlap (serialized on the device lock). Each test run took ~45 s after its build.
- **Device:** no hang, no reset, no JIT error, no kernel-config overflow. Five device runs this round: the drift check plus one per node.
- **Builds:** every node has a build log (`build_r08-b0{1..4}-a01.log`). The shared eval checkout switches between branches, so every eval rebuilds, including b01's, which is kernel-only.
  - Host code changed in b02-a01 (factory: plan CB), b03-a01 (pacing board setup, host linearity check) and b04-a01 (factory: CB c_21 + `RE_PACE_CB`). b01-a01 is kernel-only.
- **verify.log:** 4/4 ok, no issues.
- **Worker exits:** b02, b03, b04 rc=0, b01 rc=143 after commit (above). Tags present on all four, no attempts lost.
- **Reported worker cost** totals $37.01: b01 $7.11, b02 $11.46, b03 $8.73, b04 $9.71 (r07: $21.45).
  - b02's `[cost ...]` line says 5 turns, 0 min. Its session returned an intermediate result ("Waiting for the eval to finish.", 94 turns / 1474 s, $10.99) while the eval ran in the background, then continued on the task notification (5 turns / 36 s). The cost line shows the last segment's turns and minutes; its $11.46 appears cumulative. r07 had the same reporting artifact on b01.
- **Eval / tooling notes:**
  - Kernel sizes were again measured with local -O2 compiles, not from the JIT cache. All three size-sensitive nodes (b02, b03, b04) fit on the first run, unlike r07's b03.
  - b03 and b04 both rely on the `RE_HWPROF` opt-in from r07-b03-a01 for TRISC1 room. b04's fractional-progress variant (+1.2 KB reader) did not fit without plan-once.
  - Pacing kill switch: `DS_NO_PACE=1` on both b03 and b04.
  - Analysis scripts again live only in `/tmp`; no r08 node committed any:
    - `/tmp/r08b01/`: `jobs.py` (per-row job intervals GU start | GU end | DN end), `pt.cpp` (planner replay) with `base.txt` / `new.txt`, `tr/kc.sh` (TRISC0 size), `basek/`;
    - `/tmp/r08b02/`: `pt` (planner replay), `cc/b.sh` (reader/writer compile with `cb_plan_scratch`), `sim/` (fluid DRAM-share model fitted to per-row ends; it reproduced r07-b04-a01's kimi-k3 flip), `eval1.log`;
    - `/tmp/r08b04/`: `sz/b.sh` (reader size, `XDEF="-DRE_UP_HEAD_ROWS=3 -DRE_PACE_CB=21" RWM=1`), `kc.sh` + `tA/` / `tB/` (TRISC1 with / without `RE_HWPROF`).
  - Older `/tmp` scripts still in use: `/tmp/r06b01a01/ends.py` and `zones.py` (b01, b02, b03, b04), `/tmp/r06b02/gu.py` (b02), `/tmp/r07b02/ana.py` and `sz/b.sh` (b01, b04).
- **driver.log** again carries, after the r08 results, a fragment of r07's drift-check lines and manifest, an unlabeled v2 policy-objective printout (V = 1.6342 at beta 0.6; per-round V for r01-r07, r07 V=1.7587), then the r08 drift check and the r08 manifest. These are out of order relative to the timestamps.
- **`logs/summary_r08.jsonl`** is the summary-writing session's transcript, not a worker result log.
- **Timestamps:** `decisions.jsonl` and `manifest.json` are in UTC (04:27:07Z). Worker JSONL timestamps are UTC too. The driver log, run_*.out files, eval logs and git commit times use local time (+02:00).
