# Round r04 summary: routed-expert-prefill

- **Policy:** `v1` (frontier prune + collapse close + recover role), beta 0.6. This is the first round on a non-default policy. Dreaming after r03 promoted `v0-cand3` unchanged as `v1` (activated 01:55:01). Its r01-r03 replay gave V 1.5306 vs v0's 1.5204.
- **W / R:** W=4 branches, R=4 attempts per branch (max_steps 30). 3 decision steps ran batches and step 4 returned an empty batch with `stop: true`.
- **Round root:** `dream/routed-expert-prefill/n/r03-b02-a03` @ `e7b91709fcf`, score **1.6103**.
  - This is r03's profiler-only measurement node. Its code is r03-b02-a02 plus MATH perf counters that compile out of eval builds.
  - The pre-round drift check measured geomean -0.07%, worst shape gpt-oss -0.76% (noise band 1.0%).
- **Wall time:** 01:55:56 → 03:15:31 (+02:00), 1 h 20 min (r03: 1 h 48 min)
- **Attempts:** 9 committed, 9 valid, 0 runtime_error, 0 accuracy_fail, 0 lost. 7 of the 16-attempt budget were not used.
  - Four attempts had a kernel-config-buffer overflow on their first device run. Each was repaired and re-run before scoring.
- **Branches closed by the frontier prune:** b01, b02 and b04 after step 2, b03 after step 3. All four had a head more than 1.1% (1.1x noise) below the round's best, 1.6275.
- **Round gain:** 1.6103 → 1.6275 (+1.07%), just outside the ±1% noise band. Every gain came from step 1 (b02-a01). Steps 2 and 3 set no new best.

## Best node: `r04-b02-a01`, score 1.6275

Mechanism: weight senders start block kb+1's DRAM reads as soon as their own ring slot is free, in both phases.
- **Gate/up:** the reader's gate read and the writer's up read no longer wait for the sender row's x block kb.
- **Down:** r03-b04-a04's down look-ahead is ported, with one change: the sender re-checks for a free slot while it spins on `act_valid`.
- Kernel-only (reader only, +1.3 KB), bit-identical to the baseline.

Lineage: r03-b02-a03 (root) → b02-a01. It also ports r03-b04-a04's down look-ahead, the functional best of r03.

| shape | baseline µs | round root µs | best µs | speedup vs baseline | Δ vs round root |
|---|---|---|---|---|---|
| deepseek-v3-h7168-i2048 | 801.27 | 513.98 | 512.82 | 1.562 | -0.2% |
| kimi-k2-7-h7168-i2048 | 1161.99 | 729.53 | 717.28 | 1.620 | -1.7% |
| kimi-k3-latent-h3584-i3072 | 2570.42 | 1365.81 | 1342.79 | 1.914 | -1.7% |
| glm-5-3-h6144-i2048 | 704.06 | 451.23 | 449.24 | 1.567 | -0.4% |
| mistral-small-4-h4096-i2048 | 261.79 | 185.90 | 183.76 | 1.425 | -1.2% |
| deepseek-v4-flash-h4096-i2048 | 525.15 | 316.95 | 308.69 | 1.701 | -2.6% |
| deepseek-v4-pro-h7168-i3072 | 1604.12 | 1070.94 | 1056.41 | 1.518 | -1.4% |
| minimax-m3-h6144-i3072 | 477.77 | 365.46 | 372.86 | 1.281 | +2.0% |
| gpt-oss-120b-h2880-i2880 | 572.05 | 261.82 | 255.75 | 2.237 | -2.3% |
| **geomean** | | | | **1.627** | |

- minimax-m3's +2.0% is its usual ±5% group-tail lottery: its mean core end moved 344 → 343 µs.
- Mean-core RE-DN fell on all 9 shapes. That includes the 4-expert shapes, where r03-b04-a04's down look-ahead had not fired: mistral -5.3, dsv4-flash -10.7, gpt-oss -6.2, mm3 -3.6 µs.
- The crit core also moved down 7-23 µs on k2.7, k3, dsv4-flash, dsv4-pro and gpt-oss. The op time followed the mean core this round.

Per-shape bests from other nodes this round:

| shape | best µs | node |
|---|---|---|
| kimi-k3 | 1314.56 | b03-a02 |
| glm | 448.13 | b04-a01 |
| mistral | 178.67 | b02-a02 |
| minimax-m3 | 351.98 | b03-a03 |

The best node itself holds the round's per-shape bests for dsv3, kimi-k2.7, dsv4-flash, dsv4-pro and gpt-oss.
- kimi-k3 (b03-a02) and minimax-m3 (b03-a03) come from mechanisms with a measured effect on that shape.
- The glm and mistral bests are within those shapes' noise.

## Score grid (branch x attempt)

| branch | a01 | a02 | a03 | a04 | branch best | closed |
|---|---|---|---|---|---|---|
| b01 | 1.6011 | 1.5915 | - | - | 1.6011 (a01) | after step 2 (head 1.5915) |
| b02 | **1.6275** | 1.6093 | - | - | 1.6275 (a01) | after step 2 (head 1.6093) |
| b03 | 1.5945 | 1.6119 | 1.6066 | - | 1.6119 (a02) | after step 3 (head 1.6066) |
| b04 | 1.6002 | 1.6037 | - | - | 1.6037 (a02) | after step 2 (head 1.6037) |

Bold marks the round's running best when it was set. The running best after each step was:

| step | workers | running best | set by | step-best Δ | outside noise? |
|---|---|---|---|---|---|
| s1 | 4 | 1.6275 | b02-a01 | +0.0172 | just (+1.07%) |
| s2 | 4 | 1.6275 | no new best | 0 | |
| s3 | 1 | 1.6275 | no new best | 0 | |
| s4 | 0 | - | stop | | |

Cross-branch code traffic that the parent links don't show:
- b01-a02 ported b02-a01's reader and dropped its own parent's code (the CFGSHIFTMASK MOP and the RE-HW counters). In practice it was a child of b02-a01.
- b04-a02 built its activated run-ahead on b02-a01's reader, not on its parent b04-a01's reader.
- b03-a02 compiled out its parent's probe (`RE_PROBE`). Its code is the root plus the down-phase stream decoupling.

## What worked

1. **Down look-ahead with a spin re-check** (b02-a01, +1.07%; the round's only gain outside noise).
   - It answers r03's open question about r03-b04-a04. On the 4-expert shapes and dsv4-flash, the one-shot `pages_reservable` check after the down multicast usually failed, because the sender's compute was still on block kb-1.
   - The slot frees during the activated-block wait. Re-checking during that spin starts the read there.
   - Mean-core DN fell on all 9 shapes: -14 dsv3, -16 k2.7, -30 k3, -7 glm, -21 dsv4-pro, -5 to -11 on the 4-expert shapes and dsv4-flash.
   - **The gate/up half was null.** Mean RE-GU was +1..+5 µs on the DRAM-bound shapes, about 0 on four shapes, and -5 µs only on gpt-oss.
   - No fire counter was added. The reflection can't tell whether the gate/up look-ahead only moves DRAM share or rarely fires.
2. **The matmul is not the limiter** (b03-a01, measurement node).
   - It ran an in-situ gate/up matmul probe with no possible data wait: 21.0 cycles per tile-mult (W = 6 shapes) and 25.1-25.4 (N = 9 shapes).
   - In that probe, FPU (16.0) and unpack (17.2-17.5) overlap.
   - This closes the ~31-cycle question from r03: the RE-GU/RE-DN windows are the 21-25-cycle matmul plus waiting.
   - It also refutes b01-a01's "SrcA behaves single-banked" reading from the same step.
   - UNPACK-thread wait tallies:

     | phase | stream | wait share |
     |---|---|---|
     | gate/up | x (row stream) | 25-42% |
     | gate/up | weights | 3-15% |
     | down | activated (row stream) | 30-58% |
     | down | down weights | 3-11% |

   - These shares steered all of step 2. Later nodes showed they partly mislabel weight lateness (see below).
3. **Activated-stream decoupling in the down phase is real on latency-bound down chains** (b02-a02, b03-a02, b04-a02).
   - kimi-k3, the shape with many small down blocks, gained in all three:
     - mean-core DN -70 µs (b02-a02, -14%) and -42 µs (b04-a02, -8.6%);
     - UNPACK DN window 461 → 368 µs (b03-a02, -20%);
     - chip -1.0% / -1.2% / -3.7%.
   - b03-a02 also cut DN on dsv4-flash by 15% and on mistral by 13%. b02-a02 cut mistral's crit DN from 61 to 46 µs.
   - About half to two-thirds of kimi-k3's DN saving reappeared as GU on the next expert (+22 to +67 µs). This is the conserved GU+DN signature of a DRAM-bound shape, seen before in r02-b02-a03, r03-b04-a03 and r04-b04-a01.
4. **Idle-row tie layout** (b03-a03, neutral overall).
   - When the best uniform layout leaves groups idle, the planner takes a 3+3+2+2 layout with the same modelled makespan.
   - minimax-m3 -4.6% chip (368.8 → 352.0, mean core -17 µs). All four groups now end at 329-333 µs.
   - mistral is flat: its slow group moved to rows y8-9.
   - The size repair cancelled the gain. The planner at -Os on UNPACK delays the first GU by 2.3-3.1 µs on every shape, and by 7.5 µs on kimi-k3.
5. **Removing the per-tile unpack config stall** (b01-a01): UNPACK-thread stall share 17-19% → 1.4-2.8%. It is correct and bit-identical, but not on the critical path (see the table below).

## What failed and why

| node | score | outcome | reason (from its reflection) |
|---|---|---|---|
| b01-a01 | 1.6011 | neutral (-0.6% vs root) | It replaced the per-tile RDCFG / ADDDMAREG / STALLWAIT / WRCFG chain with an op-local unpack MOP using one CFGSHIFTMASK per tile. The UNPACK-thread stall vanished, but cycles per mult, UNP0 busy and FPU busy were unchanged within ±1 cycle: the chain overlapped the previous tile's UNPACR. Its single-bank SrcA explanation was refuted by b03-a01 in the next step. |
| b01-a02 | 1.5915 | neutral to negative (-2.2% vs b02-a01, whose reader it ported) | An activated stream one block ahead with parity valid flags. DN went back to its parent's level on 7 of 9 shapes, losing b02-a01's DN gain, and nothing was gained back. Its reading: the activated wait was weight lateness seen through the row stream, which is waited first. It also switched off b02-a01's gate/up look-ahead to make room. Run 1 overflowed on kimi-k2.7 by 160 B. |
| b02-a02 | 1.6093 | -1.1% vs parent | A non-blocking activated pump with a running-count `act_valid`. It won on kimi-k3 (DN -14%) and mistral (-2.8% chip). It lost on the h7168 shapes: dsv4-pro DN +7% (+2.6% chip) and kimi-k2.7 +3.2%. The inferred cause is that the act send still ends in a blocking `async_write_barrier` and now lands inside the weight-critical waits, delaying the column weight multicast. A second likely factor is d_slots = 2 on h7168, which leaves little run-ahead. The reflection classifies this as a repairable failure. |
| b03-a01 | 1.5945 | neutral (measurement) | The probe at kernel start delays the first GU by 1-2 µs, about 0.3-1% on the short shapes. The rest is noise. |
| b03-a02 | 1.6119 | neutral (+0.1% vs probe-free root) | It decoupled both down streams in one polling loop. Acking down weights independently of the activated stream let the column senders' down reads run ahead into the DRAM share of groups still in gate/up: kimi-k2.7 +2.1%, dsv4-pro +1.6%, dsv3 +1.2%. kimi-k3 -3.7% and dsv4-flash -2.4% offset that. This repeats r02-b04-a02's free-running down-stream loss. |
| b03-a03 | 1.6066 | neutral (-0.3% vs parent) | The tie pass adds about 0.6 KB per planning RISC. Run 1 overflowed on 6 shapes (worst kimi-k2.7, +992 B): the parent was only about 0.8 KB under the limit, not the estimated 4.5 KB. The repair built the planner at -Os on the writer and UNPACK. The slower UNPACK plan delays the first GU about 3 µs on every shape, cancelling minimax-m3's -4.6%. |
| b04-a01 | 1.6002 | neutral (-0.6% vs root) | It decoupled gate/up x and weight acks on the receivers. The early ack fired often: early x ack in 19-34% of blocks, early w ack in 7-21%. GU still did not move on 8 of 9 shapes. The next block's sender was not ready anyway: the x owner waits for its tilize or for the slowest row core, and the weight writer serially reads then multicasts. The early owner x multicast (run 1 only) fired in ≤ 1.8% of blocks. Run 1 overflowed on kimi-k2.7 by 240 B. |
| b04-a02 | 1.6037 | neutral (+0.2% vs parent, -1.5% vs b02-a01) | An activated run-ahead with per-slot valid flags and an activated ring of up to 4 slots (host change). kimi-k3 DN -8.6%, chip -1.2%. DN was flat on the other 8 shapes, and the rest of the chip-time loss is max-core lottery. Its conclusion matches b01-a02: on 8 of 9 shapes the activated wait was a proxy for the slowest column's down weights. Run 1 overflowed on dsv3 and kimi-k2.7 (112 / 368 B). The repair was a capture-by-struct `ActState` for the pump lambda, -0.8 KB on NCRISC. |

Newly established negatives (in addition to the r01-r03 lists):
- **DM handshake / ack timing in either phase.**
  - gate/up: b04-a01, plus b02-a01's gate/up half.
  - down / activated: b01-a02 and b04-a02 everywhere except kimi-k3.
  - Earlier: r03-b02-a02, r03-b04-a02.
  - On 8 of 9 shapes, both phases' block period is set by the group's DRAM share.
- **Matmul LLK work** of any kind:
  - the unpack address chain (b01-a01);
  - the r03 operand-swap / reuse / subblock nodes.

  Without data waits the pipeline already runs at 21-25 cycles per mult (b03-a01).
- **Letting down-weight reads run free of the activated stream** (b03-a02). This is the second time, after r02-b04-a02.
- **Planner at -Os on UNPACK** (b03-a03). The UNPACK planner is on the first-GU critical path. The writer at -Os is free (b01-a02).

**Disagreement between reflections on the activated stream:**
- **Repairable:**
  - b02-a02 blames its own blocking act send and proposes making it non-blocking.
  - b03-a02 proposes decoupling only the activated stream, keeping the down-weight ack behind the activated arrival (`w_ack == w_rx && act_rx > w_rx`), then porting it onto b02-a01.
- **Flawed premise:** b01-a02 and b04-a02 say the activated wait was weight lateness except on kimi-k3, so there is nothing left to take.
- Neither repair was tried: all four branches were closed by the end of step 3. The question is open, and the stake is mostly kimi-k3 (about +0.13-0.4% geomean).

Open levers named by the reflections:
- **Kernel-config headroom on the best lineage.**
  - The h7168 programs sit about 0.3-0.8 KB under the 70656 B limit. Every reader or planner change this round needed a size cut first.
  - Three reflections name r03-b01-a02's plan-once-per-core (about 12 KB headroom, and no UNPACK planning time) as the real fix: b01-a02, b03-a03 and b04-a01.
  - It is not on the b02-a01 lineage.
- **Port b03-a03's tie pass onto b02-a01** together with plan-once-per-core, or with the UNPACK planner back at -O2 and a hard-coded 3+3+2+2 candidate. Expect minimax-m3 about -4%.
- **kimi-k3-only activated run-ahead.**
  - Option 1: port b04-a02's 4-slot ring alone onto b02-a01.
  - Option 2: gate b02-a02's pump per shape (many small down blocks).
- **Does the gate/up look-ahead fire?** (b02-a01 next #1.) If it rarely fires, delete it and recover about 0.9 KB of reader.
- **Group tail (max − mean):**
  - gpt-oss 26-29 µs, mistral 21-28, mm3 40, kimi-k3 97-101, mostly in DN.
  - mistral: one 2-row group runs 10-15 µs slower than another with the same work. The candidates are x-owner position and DRAM-bank position (b03-a03 next #3).
- **dsv3 / glm phase-B 4-row group:** it is crit on its GU (dsv3 crit GU 354 vs mean 328). Try a {3,3,4}-ordered phase B, or the largest leftover on a 3-row group (b02-a01 next #3).
- **First-block cold start:** r03-b04-a03 next #3. Carried over and still untried.

## Policy observations (input for dreaming)

- **v1's frontier prune ran the round with 9 attempts instead of 16.**
  - Step 3 had one worker, so three worker slots sat idle for 13 min. Step 4 never ran.
  - This is the case v1's notes flagged to watch ("how often step 4 runs with only one or two workers").
  - Cost fell from $118.48 to $71.93 and wall time from 1 h 48 min to 1 h 20 min.
- **The prune closed the branch that holds the round best.**
  - b02 was closed because its head, a02 (1.6093), was 1.1% below the round best. That best, 1.6275, is b02's own a01.
  - The rule compares heads only. A branch whose child regresses is closed even when its earlier node is the leader, and no branch is left refining the best node.
  - b02-a02's reflection classified itself as a repairable failure with a specific one-file fix (a non-blocking act send). The fix was never tried.
  - Proposal input:
    - compare the branch best, not the head, against the round best;
    - or keep the branch that holds the round best open, re-parenting its next attempt onto that node, which is the `recover` role's revert target.
- **The prune threshold sits at the noise floor.**
  - 1.1x noise (1.1%) is about the ±1% band, and the leader itself was only +1.07% over the root.
  - The closed heads were 1.12% (b02), 1.28% (b03), 1.46% (b04) and 2.21% (b01) below the leader.
  - The 4-expert shapes swing ±5% between runs. For example, b04-a02's chip losses on mistral / mm3 / dsv4-flash / glm came with ≤ 2 µs of mean-core movement.
  - A node can therefore be closed for a lottery draw. Re-measuring the leader and the closed heads, or using a mean-core secondary metric, would show which closes are real.
- **Possibly premature closes** (unverifiable without the untried attempts):
  - b02: the repair above.
  - b03: its a02 next #1/#2 (act-only decoupling, then port onto b02-a01) and a03 next #2 (tie pass onto b02-a01 at about -4% on mm3) were concrete, cheap ports onto the leader. b03 closed after a03 at 1.6066.
  - b04: its a02 next #2 (k3-only ring on b02-a01) was not tried.
  - All of these are small (≤ 0.5% geomean each), but together they are the round's remaining candidates.
- **The `recover` role never appeared.** decisions.jsonl tags every step-2/3 item `exploit`, including the refinement of b03-a02 and the regressed heads that a `recover` tag would have covered. As v1's notes say, run_campaign.sh forwards only `node,parent` to workers, so the role has no effect either way.
- **Duplicated ideas. This round's waste was again duplication.**
  - **Step 2:** all four attempts implemented the same hypothesis, decoupling the activated row stream from the down-weight stream in the down phase: b01-a02, b02-a02, b03-a02, b04-a02.
    - Sources: b02-a01's next #2 and b03-a01's next #1, both visible through the shared history.
    - Scores: 1.5915, 1.6093, 1.6119, 1.6037.
    - All four found the same kimi-k3 gain and losses or no change elsewhere. Two were enough to show this.
    - Cost: $28.51 of $71.93.
    - This repeats r03 step 4, where 3 of 4 attempts ran the same SrcB-streaming hypothesis. Two rounds in a row, a step's direction came from one node's "next #1" and every free worker took it.
  - **Step 1:** b02-a01 (gate/up look-ahead) and b04-a01 (gate/up ack decoupling) both targeted the gate/up handshake. Both found GU unmoved.
  - Proposal input: dedupe by mechanism tags or proposal text at batch time, e.g. at most one attempt per (phase, stream, mechanism-tag) per step, or pass sibling proposals to each worker.
- **A measurement node again steered a step into a misread hypothesis.**
  - b03-a01's UNPACK wait tallies ("x 25-42%, activated 30-58%") were taken at face value by three of the four step-2 workers.
  - Two of those workers (b01-a02, b04-a02) concluded that the tallies counted weight lateness as row-stream wait, because the row stream is waited first. b03-a01 had noted the caveat but discounted it.
  - In r03 it was b02-a03's counters and the SrcB hypothesis. Each time, one cheap discriminating measurement would have been enough: here, tallying both waits from block start, which b01-a02 next #3 suggests.
- **Collapse close did not fire.** The largest within-branch drop was b01-a02 at -0.6% vs its parent.
- **Round root was a measurement node,** as r03's summary warned. It cost nothing here: its counters are profiler-only. Two step-2 nodes did remove or gate them to get kernel-config space (b01-a02 removed them; b02-a02 made `RE_HWPROF` opt-in).
- **Kernel-config-buffer size is the binding constraint on this lineage.**
  - 4 of 9 attempts overflowed on their first device run: b01-a02, b03-a03, b04-a01, b04-a02.
  - Each repair cost a re-run, and two repairs cost latency or mechanism:
    - b03-a03's -Os UNPACK planner cancelled its gain;
    - b04-a01 dropped its early owner multicast and counters.
  - Workers' local size estimates were off by 0.5-1 KB, because local -O2 objects don't track the JIT LTO build.
  - r03's proposal of a campaign-level size check (JIT ELF sizes before the scored run) was not implemented. Six more worker-local size tools appeared in `/tmp`: `/tmp/r04b01/sz`, `/tmp/r04b01a02/sz`, `/tmp/r04b01a02/build_o2.sh`, `/tmp/r04b03a01/sz`, `/tmp/r04b04/cc`, `/tmp/r04b04a02/cc`, plus `/tmp/w_r04b03a03c/sim/sz.cpp`.
- **Yield:**
  - 9 of 9 valid.
  - 1 new best (b02-a01), just outside noise.
  - 7 of 9 below the round root (1.6103). Only b02-a01 and b03-a02 (1.6119) are above it. Of those below, only b01-a02 is outside noise, at -1.2%.
  - 6 of 9 below their parent: b01-a01, b01-a02, b02-a02, b03-a01, b03-a03 and b04-a01. b04-a02 is +0.2% over its parent but -1.5% under b02-a01, whose reader it built on.
- **Diminishing returns.**

| round | gain | attempts |
|---|---|---|
| r01 | +50% over the baseline | 16 |
| r02 | +4.6% | 16 |
| r03 | +2.6% by score (+1.2% outside noise) | 16 |
| r04 | +1.07% | 9 |

  The reflections agree that on 8 of 9 shapes both phases are paced by the group's DRAM share. What is left is in three places:
  - the group tail (max − mean);
  - per-shape layout cases (mm3 tie pass, dsv3/glm phase-B group, kimi-k3 activated run-ahead);
  - the size headroom needed to land them.

## Ops notes

- **Pre-round:**
  - r03 completed at 01:49:48, and dreaming activated `v1` at 01:55:01.
  - The root was built (`build_r04_root_1.log`, 01:55:13), and the drift check passed (geomean -0.07%, worst gpt-oss -0.76%).
- **Step durations.** The barrier is synchronous, so the slowest worker sets each step's length. Times are commit times (+02:00).

| step | duration | slowest | fastest finished |
|---|---|---|---|
| 1 | 30.5 min | b03-a01 (02:26:20) | b02-a01 at 02:07:41, then about 19 min idle (b01-a01 02:08:47, b04-a01 02:17:55) |
| 2 | 36.0 min | b01-a02 (03:00:25) | b03-a02 at 02:36:18, then about 24 min idle (b04-a02 02:45:15, b02-a02 02:46:03) |
| 3 | 13.0 min | b03-a03 (03:13:26), the only worker | 3 slots idle for the whole step |

- **Worker exits:** 5 of 9 workers exited with rc=143: b01-a02, b02-a01, b02-a02, b03-a03 and b04-a02. Each had committed before the driver stopped it. Tags were present for all 9, and no attempts were lost. The other 4 exited rc=0.
- **verify.log:** 9/9 ok, no issues.
- **Device runs needing a repeat:** four first runs failed at program creation with `Program size (...) too large for kernel config buffer (70656)`. None hung, and every other shape in those runs was bit-identical. All four were repaired and re-run before scoring.

| node | shapes that overflowed | bytes over |
|---|---|---|
| b01-a02 | kimi-k2.7 | 160 B |
| b03-a03 | 6 shapes | up to 992 B |
| b04-a01 | kimi-k2.7 | 240 B |
| b04-a02 | dsv3, kimi-k2.7 | 112 / 368 B |

  No JIT errors and no hangs this round.
- **Builds.**
  - Host builds ran for 3 attempts: b01-a02, b02-a01 and b04-a02, plus the root build.
  - Only b04-a02 changed host code: the program factory's activated ring of up to 4 slots and its flags CB.
  - b01-a02 and b02-a01 are kernel-only. Their rebuilds were most likely triggered by the shared eval checkout changing, as with r03-b04-a04.
  - The other 6 were kernel-only with no host build.
- **Eval / tooling notes from workers:**
  - Local -O2 / -Os object sizes underestimate the JIT program by up to about 0.5-1 KB. The JIT ELFs under `jitcache/tt-metal-cache*/kernels/<kernel>/<hash>/` give the real per-RISC sizes (b04-a02).
  - b03-a01's UNPACK-thread (TRISC0) RE-UW tallies and RE-HW probe records arrived for all 117 launches.
  - The probe must stay compiled out (`RE_PROBE`) in score-bearing nodes: it adds 1-2 µs to the first GU.
  - Worker analysis scripts cited by later nodes again live only in `/tmp`: `/tmp/r04b04/zones2.py`, `/tmp/r04b03a01/probe.py` and `ends.py`, `/tmp/r04b01a02/rows.py`, `/tmp/w_r04b03a03c/*`. They are not part of any node commit.
- **Reported worker cost** totals $71.93: b01 $19.05, b02 $10.09, b03 $22.36, b04 $20.43. That is down from $118.48 in r03, with 9 attempts instead of 16.
  - As before, the turns/minutes on the `[cost ...]` lines often don't match wall time. b04-a01 reports 1 turn and 0 min for $11.13 against about 22 min of wall time; b03-a01 reports 5 turns and 0 min for $13.85 against about 30 min.
- **Timestamps:** `decisions.jsonl` and `manifest.json` are in UTC (23:55:56Z). The driver and worker logs use local time (+02:00).
