# Round r07 summary: routed-expert-prefill

- **Policy:** `v2` (one-step rounds: W=4, R=1), beta 0.6. This is the third round on v2. Dreaming after r06 produced no `v3` (driver log 05:47:17, "keeping v2").
- **W / R:** W=4 branches, R=1 attempt per branch (max_steps 30). Step 1 ran 4 workers from the root. Step 2 returned an empty batch with `stop: true`.
- **Round root:** `dream/routed-expert-prefill/n/r06-b04-a01` @ `a7291294edd`, score **1.6931**.
  - This is r06's best node: wrap-free down-weight multicasts (corner senders, bottom-row senders multicast down on the writer's NoC 1). Host+kernel, bit-identical.
  - The pre-round drift check measured geomean -0.04%, worst shape gpt-oss -0.29% (noise band 1.0%). It re-measures the campaign root (`dream/routed-expert-prefill/root`, the baseline code at `c54a00a7697`), not the round root (see Ops notes).
- **Wall time:** 05:48:14 → 06:18:13 (+02:00), 30 min (r06: 1 h 19 min, r05: 45 min). b01 and b04 were done by 05:54:15. b02 and b03 each needed a second eval (see below).
- **Attempts:** 4 committed, 4 valid, 0 runtime_error, 0 accuracy_fail, 0 hang, 0 lost.
  - Two of the four nodes failed their first device run and were repaired inside the attempt before the scored run. b02-a01's first run hung. b03-a01's first run was a runtime_error (kernel config buffer overflow by 128 B on kimi-k2.7). The ledger records only the scored runs, so neither failure appears in the fail_class counts.
  - All 4 scored outputs are bit-identical to the parent and the baseline (PCC / max_abs / mean_abs on every printed digit).
- **Branches closed:** none. With R=1 every branch ends after its one attempt.
- **Round gain:** 1.6931 → 1.7387 (+2.7%), outside the ±1% noise band. Two nodes beat the root outside noise (b02-a01 +2.7%, b01-a01 +1.6%). Two were neutral (b03-a01 +0.7%, b04-a01 +0.6%). No node lost.

## Best node: `r07-b02-a01`, score 1.7387

Mechanism: wrap-free gate/up weight multicasts.
- A top-row sender's gate/up multicast now runs on its reader's NoC 0 (+y) instead of the writer's NoC 1 (-y). So no gate/up multicast wraps around the column torus through the other groups' rows.
- The top sender's DRAM read is issued before the receivers' acks. The next block's look-ahead read is issued only after the valid-sem multicast that closes the linked multicast (run 1's hang, see What failed).
- Every multicast of one kind goes through one out-of-line helper. Local -O2 sizes: reader -44 B, writer -276 B vs the parent.
- Kernel-only (reader + writer). It completes r05-b04-a01's torus-wrap diagnosis: r06-b04-a01 fixed the down half, this node the gate/up half.

Lineage: r06-b04-a01 (root) → b02-a01. Scored on run 2; run 1 hung.

| shape | baseline µs | round root µs | best µs | speedup vs baseline | Δ vs round root |
|---|---|---|---|---|---|
| deepseek-v3-h7168-i2048 | 801.27 | 494.38 | 510.11 | 1.571 | +3.2% |
| kimi-k2-7-h7168-i2048 | 1161.99 | 695.40 | 700.66 | 1.658 | +0.8% |
| kimi-k3-latent-h3584-i3072 | 2570.42 | 1233.96 | 1212.29 | 2.120 | -1.8% |
| glm-5-3-h6144-i2048 | 704.06 | 441.99 | 440.99 | 1.597 | -0.2% |
| mistral-small-4-h4096-i2048 | 261.79 | 180.19 | 167.13 | 1.566 | -7.2% |
| deepseek-v4-flash-h4096-i2048 | 525.15 | 294.73 | 287.02 | 1.830 | -2.6% |
| deepseek-v4-pro-h7168-i3072 | 1604.12 | 985.23 | 997.34 | 1.608 | +1.2% |
| minimax-m3-h6144-i3072 | 477.77 | 347.78 | 333.69 | 1.432 | -4.1% |
| gpt-oss-120b-h2880-i2880 | 572.05 | 260.30 | 229.24 | 2.495 | -11.9% |
| **geomean** | | | | **1.739** | |

- **The main effect is the collapse of the per-call group-end spread, not a steady GU saving.** Per-row kernel end std over the 10 measured calls, parent → this node: dsv3 ±14-32 → ±4-5 µs, kimi-k2.7 ±19 → ±4-5, mistral ±10-19 → ±1-3, gpt-oss ±7-26 → ±2-6, dsv4-pro ±11-20 → ±9-11.
  - The reflection's reading: a wrapped NoC-1 multicast holds a ~12-hop path through the other groups' rows. Whether it collides with another group's stream changes from call to call, so a different group is the tail in each call.
  - This is the "max-core lottery" that reflections since r01 blamed for noise on the 1-expert-per-group shapes. Much of it was this.
  - gpt-oss (-11.9%) shows it most clearly: mean GU / DN per busy core barely moved (153 → 150, 63 → 61), and the max row end fell 31 µs.
- **DN fell on the DRAM-bound shapes with no DN code change** (per mean core: k2.7 -14, dsv4-pro -15, glm -11, k3 -15 µs). Down blocks cross NoC-1 column links that no longer carry the other groups' wrapped gate/up multicasts.
- **The losses are the top groups, now deterministically.** dsv3's phase-B 4-row group y2-5 ended 490 → 509 µs while y9-11 went 428 → 387. kimi-k2.7 y2-6 694 → 699, dsv4-pro y2-6 984 → 996. Those shapes' chip time is the top group's end, hence dsv3 +3.2%, dsv4-pro +1.2%, k2.7 +0.8%.
  - Suspected cause, not isolated: top senders' gate/up multicast now shares the top group's NoC-0 column links with the NoC-0 DRAM read responses.
  - The GU+DN conservation signature appears again: k2.7 GU +12 and dsv4-pro GU +18 µs per mean core, against their DN savings.

Per-shape bests from other nodes this round:

| shape | best µs | node | vs round root |
|---|---|---|---|
| dsv3 | 498.48 | b04-a01 | +0.8% |
| kimi-k2.7 | 681.34 | b03-a01 | -2.0% |
| kimi-k3 | 1192.24 | b04-a01 | -3.4% |
| glm | 431.49 | b04-a01 | -2.4% |
| mistral | 162.56 | b01-a01 | -9.8% |
| dsv4-pro | 980.81 | b03-a01 | -0.4% |

- The best node holds the round's per-shape bests only on dsv4-flash, minimax-m3 and gpt-oss (3 of 9).
- No r07 node beat the root on dsv3. The campaign-best dsv3 is still r06-b04-a01's 494.38 µs.
- kimi-k3's -3.4% on b04-a01 has a matching plan change (5/5/6/6/6 deal). glm, k2.7 and dsv4-pro on b03 / b04 are within those shapes' spread, per their reflections.

## Score grid (branch x attempt)

| branch | a01 | branch best | Δ vs root | outcome |
|---|---|---|---|---|
| b01 | 1.7196 | 1.7196 (a01) | +1.6% | win, outside noise (port / combination) |
| b02 | **1.7387** | 1.7387 (a01) | +2.7% | win, round best (scored on run 2; run 1 hung) |
| b03 | 1.7051 | 1.7051 (a01) | +0.7% | neutral (scored on eval 2; eval 1 overflowed) |
| b04 | 1.7036 | 1.7036 (a01) | +0.6% | neutral, one sub-idea won and one lost |

Bold marks the round best. There was one step:

| step | workers | running best | set by | step-best Δ | outside noise? |
|---|---|---|---|---|---|
| s1 | 4 | 1.7387 | b02-a01 | +0.0456 | yes (+2.7%) |
| s2 | 0 | - | stop | | |

Cross-branch code traffic that the parent links don't show:
- b01-a01 ports r05-b01-a01 (plan once per core + idle-row tie pass), an r05 sibling of r05-b02-a01 that is not on the root's lineage. r05-b01-a01's 5-file diff applied with one conflict, the planner pragma block in `adaptive_chunk.hpp`.
- b02-a01 implements r06-b04-a01's next #1 (gate/up half of the wrap fix). It reuses r05-b02-a01's size script (`/tmp/r05b02a01/b.sh`) and r04-b01-a02's `rows.py`.
- b03-a01 repairs r06-b03-a01 (an r06 sibling of the root), following that node's next #2: NoC-1 reads only through {default port, free port}, never the NoC-0 endpoint tile. It also changed the split step (0.594 → 0.5625 NoC 0), so it is not the pure isolation test that next #2 asked for.
- b04-a01 implements the planner rebalance named by r05-b02-a01 next #1, r06-b01-a01 next #2, r06-b02-a01 next #3 and r06-b04-a01 next #3. It uses r06-b01-a01's `ends.py` and its own planner replay `/tmp/r07b04/pt.cpp`.

## What worked

1. **Wrap-free gate/up multicasts** (b02-a01, +2.7%).
   - It is the second half of a diagnosis that has now paid twice: r06-b04-a01 (down, +1.6%) and this node (gate/up, +2.7%).
   - The gain came from removing call-to-call tail variance more than from mean GU. The predicted mechanism (GU -3..-10% per mean core) did not appear. The geomean did.
   - With the per-row ends now stable to ±4-5 µs, the row gradient (top groups slow) is a deterministic target for the planner.
2. **Plan once per core + idle-row tie pass, ported onto the round root** (b01-a01, +1.6%).
   - The cheapest node of the campaign: 4.5 min of worker time, $3.04, one eval.
   - The tie pass fires on mistral and minimax-m3 as designed: rows y10-11 now work, and the 3-row groups end 30-60 µs earlier than the parent's 2-row groups.
   - mistral -9.8%: the parent's critical group (y2-3 at 172 µs) is gone, and the mean row end fell ~155 → ~142 µs.
   - minimax-m3 only -0.6%: the 2-row groups (5-tile experts at per_core_M 3) became the new tail at 311-315 µs, against 264-285 for the 3-row groups. The planner's `group_cost` ~ per_core_M ties a 2-row pcM-3 group with a 3-row pcM-2 group, but the 2-row group is slower in practice. gpt-oss's 3+3+2+2 shows the same signature (y8-11 245 vs y2-7 198-210 µs).
   - It removes the writer's and UNPACK's planner copies (~4-5 KB each per r05-b01-a01 / r03-b01-a02), so the h7168 programs should have several KB of kernel-config headroom again. Not measured this round (see Ops notes).
3. **Uneven deal toward the lower groups works on kimi-k3** (b04-a01, the k3 half).
   - 5/5/6/6/6 instead of 6/6/6/5/5: the top groups ended ~95 µs earlier, the bottom groups ~180-190 µs later, and the max moved to y6-7 (1146 → 1192 µs). The group-end spread fell from 320 to 117 µs. Chip time -3.4%.
4. **NoC-1 reads through each bank's free tile are safe and harmless to GU** (b03-a01).
   - It answers r06-b03-a01's open question. That node's +10..+17% GU loss came from port concentration at the NoC-0 endpoint tiles (plus its larger split step), not from the free-tile path.
   - The table-toggle remap costs nothing per tile: the host passes 4 words of (remapped - default), and the writer adds / negates them around each NoC-1 weight read loop. Writer +108 B locally.
   - `re_hwprof` MATH counters are now opt-in (`RE_HWPROF`): -1.0 KB text / -132 B data on TRISC1 in profiled builds.

## What failed and why

| node | score | outcome | reason (from its reflection) |
|---|---|---|---|
| b02-a01 run 1 | not scored | hang (repaired) | The reader issued block kb+1's look-ahead DRAM reads between the linked gate/up data multicasts and the unlinked valid-sem multicast that closes the link, on the same NoC 0. BH deadlocks on any non-multicast transaction while a linked multicast is open (`debug_sanitize_check_linked_transactions` in `tt_metal/hw/inc/internal/debug/sanitize.h`). Watcher is off in the eval, so it showed as a hang on dsv3's first call (900 s timeout, device reset). Repair: issue the look-ahead after the valid sem. No other change. |
| b03-a01 eval 1 | not scored | runtime_error (repaired) | r06-b03-a01's remap code as-is (8 runtime words, an 8-way coordinate compare per weight read) made kimi-k2.7's program 70784 B against a 70656 B kernel config buffer. The local -O2 non-LTO delta (+180 B) did not predict the JIT LTO delta. Repaired with the zero-per-tile table toggle and the opt-in counters. |
| b03-a01 | 1.7051 | neutral (+0.7%) | The NoC-1 relief moved bandwidth between row groups more than it raised the total. Top rows got 3-10% faster on some shapes (k2.7 y2-6 -15..-28 µs, glm y6-7 -28), lower rows got slower (k2.7 y8-10 +9..+16, gpt-oss y5-7 +8..+10). The link model predicted the hottest link 54 → ~49-50 GB/s (-7..-10%). Delivered weight bandwidth rose ~1% (dsv4-pro 455 GB/s, k3 ~425 GB/s). Per-port load is ~30 GB/s everywhere, so port load is not the limit either. The two remaining readings: a DRAM-side ceiling at ~455-475 GB/s of 512 peak, or request arbitration at the DRAM tiles. |
| b04-a01 | 1.7036 | neutral (+0.6%), one sub-idea flawed | One tie-break rule (break load ties toward the bottom group) made two plan changes. kimi-k3 gained 3.4% (above). On mistral / minimax-m3 it left the top group y2-3 idle instead of the bottom one, and lost 3.3% / 2.5%. The topmost *active* group (y4-5) became the slow one, slower than y2-3 had been (mistral 175 vs 171, mm3 348 vs 326), and y6-7 slowed too (mistral 145 → 174). The gradient follows relative position among the streaming groups, not absolute grid row. |

Newly established negatives (in addition to the r01-r06 lists):
- **Any NoC read, unicast write or atomic issued while a linked multicast is open on the same NoC** (b02-a01 run 1, hang). r01-b03-a01's advice "issue the next read after the multicast, before the flush/valid" is unsafe whenever the multicast is linked, and every weight / x / activated multicast in this op is linked.
- **Idle groups on top of the grid** (b04-a01). An idle group at the top buys nothing, because the next active group becomes the slow one.
- **Static placement or offset fixes for the 1-expert-per-group gradient** (b04-a01 with r06-b02-a01's start-up stagger). Both reflections say the cure has to be dynamic (a run-time share cap or priority).
- **More byte placement between NoCs and DRAM tiles** (b03-a01 next #2). r05-b02-a01 won once. r06-b01-a01, r06-b03-a01 and b03-a01 each moved bandwidth between rows without raising the total. Four NoC-placement link-model predictions of 7-15% delivered 0-2%.
- **A noinline-lambda refactor of the writer's three multicasts** (b03-a01: +176 B). b02-a01's out-of-line helper per multicast kind went the other way (-276 B on the writer); the difference is the form of the helper.

Open levers named by the reflections:
- **Planner rebalance toward the bottom groups, now with deterministic targets** (b02-a01 next #1, b01-a01 next #3, b03-a01 next #3, b04-a01 next #1-#2).
  - dsv3 / glm: phase B's 4-row group at the bottom ({3,3,4} layout entry) or the largest leftover dealt to a lower group. b02 estimates dsv3 -5..-10%.
  - kimi-k2.7 / dsv4-pro (phase B 5+5): fewer tile-rows to the top 5-row group.
  - kimi-k3: b04-a01's 5/5/6/6/6 deal, kept. Further: give y6-7 the smaller experts (b04 estimates <1%).
  - b04-a01 next #1 is the concrete form: flip the tie-break to the bottom group only when every group is busy, keep the parent's lowest-index tie-break otherwise. Expected +0.4..+0.6% over r06-b04-a01, and it stacks with NoC-side changes.
- **Combine b02-a01 with b01-a01** (b02-a01 next #2; b01-a01 next #1 is b02's mechanism). Conflicts are small: b01 touches the planner, the factory and `fused_swiglu.cpp`; b02 only the reader and writer. Plan-once also frees kernel-config space for the planner work.
- **3+3+2+2 imbalance on minimax-m3 / gpt-oss** (b01-a01 next #2): cost a group as roughly max(bytes-bound, pcM-bound) so a 2-row pcM-3 group is costed above a 3-row pcM-2 group, or try 4+2+2+2, or put the 2-row groups at the top (y2-5).
- **Why the top groups lose more after b02-a01** (b02-a01 next #3): e.g. return only the TOP group's top senders to the writer's NoC 1. Judge by per-row ends on dsv3 / k2.7 / dsv4-pro. It would trade back some tail variance.
- **Overlap the top senders' next read with their multicast safely** (b02-a01 next #4): read before the multicast (after the in1_ready wait; r01-b03-a01 measured that ordering as harmful at M=1 on the old root), or multicast unlinked with a write barrier before the valid sem. Measure first with NCRISC zones around the sender's read barrier.
- **A dynamic share cap or priority for the leading groups** (b04-a01 next #3, b03-a01 next #3). r06-b02-a01 next #2's in-flight cap (`gu_lookahead` off) is the cheapest form, still untried.
- **Keep b03-a01's infrastructure** (zero-cost remap toggle, opt-in counters) as headroom. b03 says a NoC split sweep is now one knob, but that it mostly moves the row gradient.

## Policy observations (input for dreaming)

- **v2's third round paid the most of the three, at the lowest cost.**

  | round | attempts | wall time | worker cost | gain |
  |---|---|---|---|---|
  | r07 | 4 | 30 min | $21.45 | +2.7% |
  | r06 | 4 | 1 h 19 min (≈23 min of work) | $30.37 | +1.6% |
  | r05 | 4 | 45 min | $49.45 | +2.4% |
  | r04 | 9 | 1 h 20 min | $71.93 | +1.07% |
  | r03 | 16 | 1 h 48 min | $118.48 | +1.2% outside noise |

  - 2 of 4 attempts beat the root outside noise (r06: 1 of 4, r05: 2 of 4). None lost outside noise, the first such round.
- **No duplicated ideas this round.** The four workers took four distinct levers, and each maps one-to-one to an item on r06's open-lever list: wrap-free gate/up (b02), combine r05-b01-a01 (b01), planner rebalance (b04), remap without the NoC-0 endpoint (b03). r06's three-workers-on-one-effect pattern did not repeat. The policy did not change between r06 and r07, so this came from the workers' own choices, not from a dedupe mechanism.
- **The off-lineage port happened without a policy change, and paid.** r06's summary proposed reserving a step-1 slot to port a non-lineage win. No v3 was produced, but b01-a01 chose the port anyway: +1.6%, the cheapest node of the campaign (4.5 min, $3.04).
- **The same pattern is set up again for r08.** Re-rooting on b02-a01 drops b01-a01's +1.6% (not on b02's lineage) and b04-a01's k3 gain. Both b02 next #2 and b01 next #1 name the combination. Without a reserved slot, r08 depends on a worker choosing it again. Proposal input: make the port of an off-lineage, outside-noise win a seeded slot. b01 shows a port can be near-mechanical; the driver could also apply it without a worker, then measure.
- **Duplication risk for r08: the planner rebalance is now the convergent next #1.** It is b02 next #1, b04 next #1-#2 and b01 next #2-#3, with b03 next #3 pointing the same way. Four r08 workers reading these reflections can plausibly pick variants of it. Proposal input (repeated from r06): give each worker the levers the other slots are taking, or dedupe by mechanism tags at batch time (`planner` / `row-groups` / `row-gradient`).
- **In-attempt failures are invisible in the ledger.** b02-a01's hang and b03-a01's kernel-config overflow were repaired before the scored run, so the round records 0 hang / 0 runtime_error. Both cost wall time: b02's hang held the device lock for ~15 min (900 s timeout + reset), and b03's second eval queued behind it. These two workers set the step's length (06:16:29 and 06:18:12). Proposal input: record a per-attempt eval count and first-run fail_class in node.json, so dreaming can see repair cost and repeated failure classes (r03-b01-a01, r05-b04-a01 and now b02 / b03 were all size or protocol failures).
- **Wasted or weakly informative attempts: none outright.**
  - **b03-a01** is neutral, but it closes r06-b03-a01's open question (cause (b), port concentration) and gives a measured stop signal on NoC byte placement. It changed two knobs again (the remap and the split step), though the GU result is now clean enough to attribute.
  - **b04-a01** bundled two plan changes under one tie-break rule. They hit disjoint shapes (k3 vs mistral / mm3), so both effects are attributable: one win and one refuted sub-idea. Its next #1 is a one-comparison fix.
- **No closes, so no premature closes.** R=1 removes the close decision.
- **The noise picture changed.** b02-a01 shows most of the "max-core lottery" on mistral / mm3 / gpt-oss / dsv3 came from wrapped multicast collisions. Per-row std fell from ±15-30 to ±4-5 µs. Two consequences for dreaming:
  - Verdicts on earlier lineages that relied on those shapes' moves (several "neutral" / "lottery" calls in r04-r06) were made at a per-shape noise well above the ±1% baseline band.
  - On the new root the per-shape noise should be closer to the band. A re-measured per-shape noise on the round root would tell whether ±1% is still the right gate.
- **Conserved GU+DN on DRAM-bound shapes again.** b02-a01's k2.7 / dsv4-pro GU growth against DN savings is the fifth node with that signature (after r02-b02-a03, r03-b04-a03, r04-b03-a02, r06-b04-a01).
- **Support collapse:** as in r05 and r06, R=1 records no depth. r07 adds four depth-1 nodes.
- **Yield:**
  - 4 of 4 valid as scored (r06: 4 of 4), with 2 of 4 needing a repaired second eval.
  - 2 wins outside noise: b02-a01 (+2.7%), b01-a01 (+1.6%).
  - 2 neutral: b03-a01 (+0.7%), b04-a01 (+0.6%).
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

  r05, r06 and r07 all gained from NoC-geometry fixes: the NoC-1 response-link ceiling, then the down-multicast wrap, then the gate/up wrap. b03-a01's result says NoC byte placement is now exhausted. The reflections point at the planner (row gradient, 2-row vs 3-row group cost) and at stacking the round's two wins.

## Ops notes

- **Pre-round:**
  - r06 completed at 05:41:58. Dreaming ran and produced no `v3` (05:47:17).
  - The drift check built (`build_r07_root_1.log`, 05:47) and measured the campaign root 05:47:17-05:48:13: geomean -0.04%, worst gpt-oss -0.29%. It checks out `c54a00a7697` (the baseline code) and compares with `baseline.json`, per `drift_ok` in `run_campaign.sh`. So it detects machine drift; it does not re-validate the round root's score. The r06 summary's wording ("the pre-round drift check" under the round root) could be read as the latter.
  - Overhead between r06 completion and r07 planning was 6.3 min.
- **Step duration.** The barrier is synchronous, so the slowest worker sets the step's length. Times are local (+02:00).

| worker | evals | eval windows | exit | note |
|---|---|---|---|---|
| b01-a01 | 1 | 05:50:42-05:51:49 | 05:52:45, rc=0 | |
| b04-a01 | 1 | 05:52:27-05:53:34 | 05:54:15, rc=0 | |
| b03-a01 | 2 | ~05:55:07-05:56:17 (runtime_error); 2nd queued on the device lock, done 06:15:12 | 06:16:29, rc=0 | waited behind b02's hung run |
| b02-a01 | 2 | 05:58:25-06:13:50 (hang, 900 s timeout + device reset); 06:15:24-06:16:25 | 06:18:12, rc=0 | repair took ~1.5 min after the hang was reported |

  - Step 1 ran 30 min (05:48:14 → 06:18:13). The driver logged all four results at 06:18:12-13.
  - No idle-worker stall like r06's b04-a01. b02 waited on its eval with `until grep -q "^EXIT" <eval log>` loops (bounded by `timeout 590`), not `pgrep -f`.
- **Device:** one hang (b02-a01 run 1) with an automatic reset. The device ran clean afterwards (b03 eval 2, b02 eval 2). No JIT errors. Seven device runs this round besides the drift check: 1 + 1 + 2 + 2 for the nodes. Each non-hung eval took ~1 min including the build.
- **Builds:** every node has a build log (`build_r07-b0{1..4}-a01.log`); b02 and b03 rebuilt twice, so their logs are the second build's. The shared eval checkout switches between branches, so every eval rebuilds.
  - Host code changed in b01-a01 (program factory: plan CB) and b03-a01 (factory: remap words, split formulas). b02-a01 and b04-a01 are kernel-only.
- **verify.log:** 4/4 ok, no issues.
- **Worker exits:** all four rc=0, tags present, no attempts lost.
- **Eval / tooling notes:**
  - **eval summary.md misaligns µs on a failed case.** In b03-a01's runtime_error report, kimi-k2.7 (FAILED) shows 1215.9 µs (k3's time), k3 shows 432.0 (glm's), and so on down to gpt-oss with "-". The µs column shifts up one row after the failed shape; the PCC columns are correct. It only affects invalid runs, but a worker reading that table would draw wrong per-shape conclusions.
  - **Kernel sizes can't be attributed from the shared jit cache** while other evals run (b01-a01 could not measure its size). b02 and b03 measured locally instead. b03 found local -O2 non-LTO deltas don't equal the JIT LTO delta (+180 B local → 128 B over the buffer).
  - **Watcher is off in the eval**, so b02's sanitizer-detectable linked-transaction violation surfaced as a 15-min hang instead of an error.
  - Analysis scripts again live only in `/tmp`; no r07 node committed any:
    - `/tmp/r07b02/`: `ana.py` (mean end / max row end / GU / DN / even-odd column GU, several CSVs side by side), `m/gufix.py` (link model with per-position gate/up NoC choice), `sz/b.sh`;
    - `/tmp/r07b03/`: `model.py` (link model with x = 9 NoC-0 endpoints and free tiles), `kc.sh` + `tr/` (TRISC size with or without `RE_HWPROF`), `agg.py`, `rows_*.txt`;
    - `/tmp/r07b04/pt.cpp` (planner replay; `g++ -std=c++17 -I<kernels dir>`).
  - Older `/tmp` scripts still in use: `/tmp/r06b02/ends.py` (b01), `/tmp/r06b01a01/ends.py` and `pt.cpp` (b04, b02 next), `/tmp/r04b01a02/rows.py` (b02, b03), `/tmp/r05b02a01/b.sh` / `b2.sh` (b02, b03), `/tmp/r06b04/agg.py` / `cols.py` (b02).
- **Reported worker cost** totals $21.45: b01 $3.04, b02 $7.94, b03 $8.38, b04 $2.09 (r06: $30.37).
  - b01's `[cost ...]` line says 8 turns, 0 min. Its log has two sessions: 36 turns / 211 s, then 8 turns / 53 s. The cost line shows the last session's turns and minutes. b02, b03 and b04 are single sessions and their lines match.
- **driver.log** ends, after the r07 results, with a fragment of r06's manifest and an unlabeled v2 policy-objective printout (V = 1.6135 at beta 0.6; per-round V for r01-r06 only, with r01 replayed as 11 attempts / 4 steps).
- **`logs/summary_r07.jsonl`** is the summary-writing session's transcript (as `summary_r06.jsonl` was for r06), not a worker result log.
- **Timestamps:** `decisions.jsonl` and `manifest.json` are in UTC (03:48:14Z). Worker JSONL timestamps are UTC too. The driver log, run_*.out files and eval logs use local time (+02:00).
