# Round r06 summary: routed-expert-prefill

- **Policy:** `v2` (one-step rounds: W=4, R=1), beta 0.6. This is the second round on v2. Dreaming after r05 produced no `v3` (driver log 04:19:13, "keeping v2").
- **W / R:** W=4 branches, R=1 attempt per branch (max_steps 30). Step 1 ran 4 workers from the root. Step 2 returned an empty batch with `stop: true`.
- **Round root:** `dream/routed-expert-prefill/n/r05-b02-a01` @ `5d961ca906c`, score **1.6667**.
  - This is r05's best node: the ~60/40 NoC-0/NoC-1 split of the weight DRAM reads, host+kernel and bit-identical.
  - The pre-round drift check measured geomean -0.02%, worst shape gpt-oss -0.22% (noise band 1.0%).
- **Wall time:** 04:20:08 → 05:39:06 (+02:00), 1 h 19 min (r05: 45 min). Three workers were done by 04:42:56. One worker sat idle for ~58 min after its eval had finished (see Ops notes).
- **Attempts:** 4 committed, 4 valid, 0 runtime_error, 0 accuracy_fail, 0 hang, 0 lost.
  - Every node scored on its first and only device run. No kernel-config-buffer overflows.
  - All 4 outputs are bit-identical to the parent and the baseline (PCC / max_abs / mean_abs on every printed digit).
- **Branches closed:** none. With R=1 every branch ends after its one attempt.
- **Round gain:** 1.6667 → 1.6931 (+1.6%), outside the ±1% noise band. One node beat the root (b04-a01). Two lost outside noise (b02-a01 -1.6%, b03-a01 -5.7%), and one was neutral (b01-a01 -0.3%).

## Best node: `r06-b04-a01`, score 1.6931

Mechanism: wrap-free down-weight multicasts.
- Weight senders sit only on their group's top or bottom row, alternating by column. For 2-row groups the placement is unchanged (`base + c % 2`). Groups of 3+ rows (phase B 4+3+3 / 5+5) now use corner senders only.
- A bottom-row sender's down multicast runs on its writer's NoC 1 (-y) instead of the reader's NoC 0 (+y). So no down multicast wraps around the column torus through the other groups' rows.
- Gate/up is untouched. Top-row senders still multicast gate/up on the writer's NoC 1, which wraps.
- Host+kernel (5 files: program factory, reader, writer, `adaptive_chunk.hpp`, `fused_swiglu.cpp`), about +0.4 KB net. The writer's planner went to -Os to pay for the reader's growth.
- It is the down-only repair, step (b), of r05-b04-a01's hung torus-aware sender rows. That hang's measured premise is now a measured gain.

Lineage: r05-b02-a01 (root) → b04-a01.

| shape | baseline µs | round root µs | best µs | speedup vs baseline | Δ vs round root |
|---|---|---|---|---|---|
| deepseek-v3-h7168-i2048 | 801.27 | 504.86 | 494.38 | 1.621 | -2.1% |
| kimi-k2-7-h7168-i2048 | 1161.99 | 701.38 | 695.40 | 1.671 | -0.9% |
| kimi-k3-latent-h3584-i3072 | 2570.42 | 1243.64 | 1233.96 | 2.083 | -0.8% |
| glm-5-3-h6144-i2048 | 704.06 | 436.68 | 441.99 | 1.593 | +1.2% |
| mistral-small-4-h4096-i2048 | 261.79 | 190.11 | 180.19 | 1.453 | -5.2% |
| deepseek-v4-flash-h4096-i2048 | 525.15 | 298.41 | 294.73 | 1.782 | -1.2% |
| deepseek-v4-pro-h7168-i3072 | 1604.12 | 993.18 | 985.23 | 1.628 | -0.8% |
| minimax-m3-h6144-i3072 | 477.77 | 361.90 | 347.78 | 1.374 | -3.9% |
| gpt-oss-120b-h2880-i2880 | 572.05 | 260.97 | 260.30 | 2.198 | -0.3% |
| **geomean** | | | | **1.693** | |

- Mean-core DN fell 7-14% on 8 of 9 shapes: gpt-oss -14%, dsv4-flash -9%, dsv4-pro -8.4%, dsv3 / mistral -8%, k2.7 / k3 / mm3 -7%. glm DN was flat (107 → 109).
- **mistral and minimax-m3 isolate the NoC change.** On this lineage they run 5x2 groups, so their sender placement did not change. The only change is that odd-column senders multicast down on NoC 1. That alone gave DN -7..-8% and chip -5.2 / -3.9%. These shapes are near compute-bound, so the gain did not leak into GU.
- **On the DRAM-bound shapes part of the saving is conserved.** Mean-core GU grew +11..+16 µs on dsv3, k2.7, k3, glm and dsv4-pro while DN fell 10-32 µs. Faster down blocks take DRAM/NoC share from the groups still in gate/up. This is the same GU+DN signature as r02-b02-a03, r03-b04-a03 and r04-b03-a02.
- glm is the only chip loss (+1.2%, mean core end +15 µs, within its spread).

Per-shape bests from other nodes this round:

| shape | best µs | node | vs round root |
|---|---|---|---|
| glm | 438.28 | b01-a01 | +0.4% |

- The best node holds the round's per-shape bests on the other 8 shapes. No node beat the root on glm.
- Campaign-wide, r05-b01-a01 (not on this lineage) still holds the top-10 leaderboard's best mistral (176.18 µs) and gpt-oss (254.00 µs).

## Score grid (branch x attempt)

| branch | a01 | branch best | Δ vs root | outcome |
|---|---|---|---|---|
| b01 | 1.6619 | 1.6619 (a01) | -0.3% | neutral, premise refuted |
| b02 | 1.6400 | 1.6400 (a01) | -1.6% | loss, outside noise |
| b03 | 1.5712 | 1.5712 (a01) | -5.7% | loss, outside noise |
| b04 | **1.6931** | 1.6931 (a01) | +1.6% | win, round best |

Bold marks the round best. There was one step:

| step | workers | running best | set by | step-best Δ | outside noise? |
|---|---|---|---|---|---|
| s1 | 4 | 1.6931 | b04-a01 | +0.0264 | yes (+1.6%) |
| s2 | 0 | - | stop | | |

Cross-branch code traffic that the parent links don't show:
- b04-a01 repairs r05-b04-a01 (a sibling of the root, not an ancestor), following that node's staged repair plan, step (b).
- b01-a01 implements the row-tilted NoC split. r05-b02-a01's reflection had raised row speed as next #1, and the r05 summary listed the first-block row gradient as an open lever.
- b02-a01 implements the "stagger the bottom groups' start" branch of r05-b03-a01's next #1. It skipped that reflection's proposed first measurement (zones that separate the NoC-0 response path from DRAM arbitration).
- b03-a01 builds on r05-b02-a01's NoC link model (`/tmp/r05b02a01/noc*.py`).
- b04-a01 reuses r05-b02-a01's link model and r04-b01-a02's size scripts.

## What worked

1. **Wrap-free down multicasts** (b04-a01, +1.6%).
   - **Causal confirmation of the torus-wrap penalty.** r05-b04-a01 inferred it from wait tallies (2-11x longer down waits on wrapped receivers). Here, removing the wrap cut DN 7-14% per mean core.
   - **Moving the down multicast to the writer is safe.** Every new wait depends only on strictly earlier blocks. It also adds overlap: a bottom sender's reader head read of block kb+1 now runs during the writer's multicast of kb.
   - **3+-row groups:** the link model predicted down column links 75 → 35 GB/s on 5+5. dsv4-pro DN fell 8.4% and k2.7 7%. This run can't separate placement from the NoC choice on those shapes.
   - DN is lock-step along a row (activated stream). So the wrap cost showed up as a longer DN for the whole row, not as a per-column difference (`/tmp/r06b04/cols.py`).
2. **NoC-1 DRAM endpoint remap works and helps the down phase** (b03-a01; the node lost overall).
   - **Reading a BH DRAM bank on NoC 1 through any of its 3 NoC tiles returns the same data** on this FW-19.7 machine. Outputs were bit-identical.
   - DN fell on the h7168 / i3072 shapes even though the down split moved back to 50/50: dsv4-pro -8% (top rows y2-3 298 → 246 µs), k2.7 -7%, mm3 -8%. By r05-b02-a01's measurement, that split change alone should have cost DN +2..+13%.
   - Coordinates and guard are recorded in the reflection: translated `(17 + bank/4, 12 + 3*(bank%4) + port)`, NoC-1 endpoint = port 1, `DS_NO_DRAM_EP_REMAP=1` to disable.
3. **Two measurements that change the row-gradient picture** (b01-a01 and b02-a01).
   - **A top-row group's speed does not follow its own NoC split** (b01-a01). Top rows went back to ~50-53% NoC 0, their r04 split, but their first-expert GU stayed at the r05 value: dsv3 262, k2.7 328, dsv4-pro 370, against r04's 226 / 219 / 293. Bottom rows at 66-69% NoC 0 got faster still (k2.7 y10 108 → 94, dsv4-pro y10-11 214 → 160). The top rows are slowed by the other rows' NoC-0 traffic, not by their own.
   - **The gradient is a self-reinforcing first-mover lead, not a fixed positional penalty** (b02-a01). Holding lower groups 5 µs moved minimax-m3's top group GU 259 → 210 µs, and the next group became the loser with a 26 µs gap. The lead set in the synchronized start-up burst persists through the gate/up phase.
   - **It is a first-expert effect** (b01-a01 next #3). Later experts (k3 GU1..5, k2.7 GU1) run at nearly equal speed on all rows. Any fix only has to cover the first ~100-300 µs.

## What failed and why

| node | score | outcome | reason (from its reflection) |
|---|---|---|---|
| b01-a01 | 1.6619 | neutral (-0.3%), flawed idea as signed | Tilted the gate/up NoC split by grid row: 0 head rows on NoC 0 for the top row up to 2x the host count for the bottom row, same ~0.6 mean. It worked as built but sped up the bottom rows, not the top. The top groups set the chip time and ended at the same time (dsv3 y2-5 503, k2.7 695, dsv4-pro 992, glm 436). The faster bottom groups only lengthened the idle tail. The premise that a row's own NoC-0 share sets its speed is refuted. |
| b02-a01 | 1.6400 | loss (-1.6%), flawed idea | Plan-aware start-up stagger: lower row groups held their first DRAM read 2.5 µs per grid row. The top group gained as intended (mm3 y2-3 end 359 → 319, mistral 177 → 166). But the group just below became the new slowest, and the bottom groups lost 25-50 µs against 15-20 µs delays. Mean core end got worse on every staggered shape (dsv3 459 → 479, dsv4-flash 270 → 294). dsv4-flash, with equal 3+3+2+2 loads, lost 5.8%. A static stagger only rotates the loser and lowers aggregate rate. |
| b03-a01 | 1.5712 | loss (-5.7%), repairable | NoC-1 weight reads (writer up rows + down tail tiles) went to each DRAM bank's NoC tile nearest above the reading row, and the split moved back toward 50/50 (`up_head_rows` 3 → 1 of 16, `d_head` ceil(N/2)). DN improved, but GU got 10-17% slower on every DRAM-bound shape, worst on the top rows (dsv4-pro y2-6 GU 663-685 → 808-854 µs, +22%). Two candidate causes the run can't separate: (a) the GU split change, which by r05 would cost only +2..+8%; (b) port concentration: "nearest tile" sends bank 0/4's NoC-1 reads to the NoC-0 endpoint tile from every row, so one AXI port carries both the gate (NoC 0) and up (NoC 1) streams while the bank's other two ports idle. The link model has no per-port capacity term. |

Newly established negatives (in addition to the r01-r05 lists):
- **Static start-up stagger or pacing in any form** (b02-a01; r01-b01-a03 was the in-loop pacing version).
- **Tilting the NoC-0 share away from the top rows** (b01-a01). The top rows' speed did not move.
- **NoC-1 weight reads through the NoC-0 endpoint tile in gate/up** (b03-a01, suspected cause (b), not yet isolated).
- **Moving sender rows per phase** (r05-b04-a01 hang; b04-a01 lists it as do-not-retry).

Open levers named by the reflections:
- **Wrap-free gate/up multicasts** (b04-a01 next #1). Top-row senders still wrap on the writer's NoC 1: half of every 2-row layout's columns. r05-b04-a01 measured gate/up wrap waits at 1.4-4x.
  - Moving them to the reader's NoC 0 serializes the reader's 60% gate+up-head read behind its own multicast flush.
  - Options: (a) try it as-is and measure; (b) for top senders only, shift the read split toward the writer.
  - Watch size: the reader's multicast path is compiled out today.
- **Combine with r05-b01-a01** (plan once per core + idle-row tie pass). b04-a01 next #2 names it, as the r05 summary did.
  - It frees several KB of kernel-config space. b04-a01 says only a few hundred bytes are left on h7168 without it.
  - Its 3+3+2+2 layout for mistral / mm3 uses 3-row groups, which b04's corner placement already covers.
  - The port conflicts only in `adaptive_chunk.hpp`'s x_owner region and pragma block.
- **Planner rebalance for row speed** (r05-b02-a01 next #1; repeated by b01-a01 next #2, b02-a01 next #3, b04-a01 next #3). It deals the extra or bigger work to the faster bottom groups:
  - kimi-k3: 5/5/6/6/6 instead of 6/6/6/5/5;
  - dsv3 / glm: the phase-B 4-row group at the bottom;
  - mm3 / mistral: put the 4 experts on y4-11 and leave y2-3 idle.

  b02-a01's finding argues for it: it uses the lead instead of fighting it. Four reflections across two rounds have named it, and no worker has tried it.
- **Down-only DRAM endpoint remap** (b03-a01 next #1). Keep the parent's gate/up exactly and remap only the DOWN_SPLIT tail read at 50/50. Expected +1..+2%. It touches the same down phase as b04-a01, so it is untested whether the two DN gains stack.
- **Remap without the NoC-0 endpoint tile** (b03-a01 next #2): restrict NoC-1 reads to {default port 1, free port}, with `up_head_rows` held at 3/16. That isolates cause (b).
- **The opposite tilt** (b01-a01 next #1): bottom rows lighter on NoC 0, top rows heavier. It is a one-line change in `up_head_rows_for_row`, (G - 1 - gy). b01's own reasoning predicts it moves the top rows toward their r04 speeds.
- **Cap in-flight weight reads per sender** (b02-a01 next #2): `gu_lookahead` off, and check whether the per-row GU spread shrinks.

## Policy observations (input for dreaming)

- **v2's second round paid, less than the first.**

  | round | attempts | wall time | worker cost | gain |
  |---|---|---|---|---|
  | r06 | 4 | 1 h 19 min (≈23 min of work) | $30.37 | +1.6% |
  | r05 | 4 | 45 min | $49.45 | +2.4% |
  | r04 | 9 | 1 h 20 min | $71.93 | +1.07% |
  | r03 | 16 | 1 h 48 min | $118.48 | +1.2% outside noise |

  - 1 of 4 attempts beat the root, against 2 of 4 in r05.
  - Wall time doubled for an ops reason (one idle worker), not because of the policy.
- **A failed attempt was retried from history and became the round's win.** This is the test r05's summary asked for. r05-b04-a01 hung with a staged repair plan. b04-a01 took step (b) (down-only), fixed the hang and set the campaign best. R=1 lost no repair value here: the cross-round handoff through the reflection worked.
- **Re-rooting dropped r05's second win, and r06 did not recover it.**
  - No r06 worker ported r05-b01-a01 (plan-once + tie pass). The r05 summary proposed seeding a "combine X and Y" slot. The policy did not change (no v3), so no slot was seeded.
  - Cost: the h7168 programs are back to a few hundred bytes of headroom. b04-a01 had to put the writer's planner at -Os to fit. The top-10 leaderboard's best mistral / gpt-oss values remain on r05-b01-a01.
  - b04-a01 next #2 names the combination again. Without a policy change, the same open-lever-but-nobody-picks-it pattern can repeat in r07.
  - Proposal input (stronger than r05's): when a node beats the root outside noise and is not on the next root's lineage, reserve one step-1 slot to port it onto the new root.
- **Duplicated ideas: three of four workers attacked the same effect.**
  - b01-a01 (NoC tilt by row), b02-a01 (start stagger) and b03-a01 (NoC-1 endpoint + split) all targeted the top-rows-are-slow gradient or the NoC-0/NoC-1 balance behind it. r05's summary predicted this ("the r06 risk is ... two or more workers implement ... the row-speed plan rebalance"). It happened, but with different mechanisms instead of the same one.
  - b02-a01's next #2 proposes the row tilt "top groups use less NoC 0". That is exactly what sibling b01-a01 ran in the same step, and it was neutral. Siblings do not see each other's work in a one-step round.
  - The lever all of them name as the alternative, the planner rebalance, was taken by nobody. It has been a top-3 item in four reflections over two rounds (r05-b02-a01 #1, b01-a01 #2, b02-a01 #3, b04-a01 #3).
  - Proposal input: dedupe by mechanism tags at batch time (`noc-balance` / `row-fairness` / `row-gradient` here). Or give each worker the list of levers the other slots are taking.
- **Wasted or weakly informative attempts.**
  - **b02-a01** retried a pacing idea from a class already negative (r01-b01-a03). It skipped the cheap discriminating measurement that r05-b03-a01 asked for first. Its finding (first-mover lead) is useful, but it cost 1.6%.
  - **b03-a01** changed two knobs at once: the endpoint remap and the split back toward 50/50. So its GU loss can't be attributed. Its own reflection now asks for "one knob per node". The DN result is informative.
  - **b01-a01** was a clean single-knob test with a refuted premise and a useful measurement. That is not waste.
- **No closes, so no premature closes.** R=1 removes the close decision.
- **Chip vs mean, again.**
  - b01-a01: chip neutral, but the bottom rows got measurably faster. The chip metric only sees the slowest (top) group.
  - b04-a01 on glm: chip +1.2% (inside its spread), with DN flat and GU +14 µs.
  - mistral / mm3 / gpt-oss remain the max-core lottery shapes. b04-a01's -5.2% / -3.9% on mistral / mm3 has a matching DN mechanism, unlike r05's lottery moves.
- **Conserved GU+DN on DRAM-bound shapes is now the recurring limit.** Down-phase speedups on dsv3 / k2.7 / k3 / glm / dsv4-pro give back 30-100% of their DN gain as GU time in other groups. This is the fourth node with that signature.
- **Support collapse:** as in r05, R=1 records no depth. r06 adds four depth-1 nodes.
- **Yield:**
  - 4 of 4 valid (r05: 3 of 4), the first round with no runtime, accuracy or hang failures.
  - 1 new best over the root, outside noise: b04-a01 (+1.6%).
  - 1 neutral: b01-a01 (-0.3%).
  - 2 losses outside noise: b02-a01 (-1.6%), b03-a01 (-5.7%).
- **Diminishing returns:**

  | round | gain | attempts |
  |---|---|---|
  | r01 | +50% over the baseline | 16 |
  | r02 | +4.6% | 16 |
  | r03 | +2.6% by score (+1.2% outside noise) | 16 |
  | r04 | +1.07% | 9 |
  | r05 | +2.4% | 4 |
  | r06 | +1.6% | 4 |

  r05 and r06 both gained from NoC-geometry fixes: the NoC-1 response-link ceiling, then the down-multicast torus wrap. The reflections now point at the gate/up half of the wrap, the r05-b01-a01 combination, and the planner-side use of the row gradient.

## Ops notes

- **Pre-round:**
  - r05 completed at 04:14:22. Dreaming ran and produced no `v3` (04:19:13).
  - The root was built (`build_r06_root_1.log`, 04:19:25) and measured 04:19:26-04:20:07. The drift check passed (geomean -0.02%, worst gpt-oss -0.22%).
  - Overhead between r05 completion and r06 planning was 5.8 min.
- **Step duration.** The barrier is synchronous, so the slowest worker sets the step's length. Times are local (+02:00).

| worker | eval window | commit | exit | note |
|---|---|---|---|---|
| b03-a01 | 04:31:39-04:32:24 | 04:33:53 | 04:35:53, rc=143 | "committed but the session is still running; stopping leftover processes" |
| b04-a01 | 04:38:02-04:38:48 | 05:38:58 | 05:39:05, rc=0 | ~60 min between eval end and commit, see below |
| b01-a01 | 04:40:23-04:41:07 | 04:42:12 | 04:42:20, rc=0 | |
| b02-a01 | 04:41:10-04:41:57 | 04:42:51 | 04:42:56, rc=0 | |

  - Step 1 ran 1 h 19 min (04:20:08 → 05:39:06). Without b04-a01's stall it would have ended at ~04:43 (≈23 min).
  - The driver logged all four results at 05:39:05-06.
- **b04-a01 idle stall (~58 min).**
  - The worker launched the eval and waited for it with a background loop: `until ! pgrep -f "eval_attempt.sh --campaign routed-expert-prefill --node r06-b04-a01"; do sleep 15; done`, with a 3600 s timeout.
  - `pgrep -f` matches the watcher's own shell command line, which contains the same string. So the loop never exited when the eval finished at 04:38:48.
  - The session ended its turn at ~04:37:50 (first result: 1053 s, 80 turns) and was only re-invoked when the watcher hit its 3600 s limit. It then read the results, wrote the reflection and committed within ~75 s.
  - The other three workers and the driver waited at the step barrier for the whole hour. No device time was used.
  - Fix candidates:
    - the eval wrapper blocks or writes a done-file that workers poll;
    - worker instructions forbid `pgrep -f` self-matching patterns;
    - the driver adds an idle watchdog that nudges a worker whose eval report is complete but which has not committed.
- **Device:** no hangs, resets or JIT errors. Each eval took ~45 s. They ran serially on the device lock with no visible queueing beyond a few seconds.
- **Builds:** b01-a01, b03-a01 and b04-a01 have a build log; b02-a01 has none.
  - Host code changed in b03-a01 (program factory: DRAM port remap args, split formulas) and b04-a01 (factory: sender roles).
  - b01-a01 is kernel-only. Its rebuild at 04:40:22 most likely came from the shared eval checkout switching back from b04-a01's host code. b02-a01 (kernel-only) ran right after on the same host build.
- **verify.log:** 4/4 ok, no issues.
- **Worker exits:** b03-a01 rc=143 after committing (the driver stopped leftover processes). The others exited rc=0. Tags were present for all 4, and no attempts were lost.
- **Eval / tooling notes from workers:**
  - Analysis scripts again live only in `/tmp`. None of the four r06 nodes committed them; r05-b04-a01 is still the only node that has.
    - `/tmp/r06b01a01/`: `zones.py`, `percore.py`, `ends.py`, `pt.cpp`, `counts.py`, `tilt.py`, `fair.py`;
    - `/tmp/r06b02/`: `gu.py`, `ends.py`, `x0.py`, `x0grid.py`, `plan/drv2.cpp`;
    - `/tmp/r06b03a01w/m/ep*.py` (link model with DRAM endpoints);
    - `/tmp/r06b04/`: `nocw2.py`, `chk2.py`, `agg.py`, `cols.py`, `sz/kc.sh`, `sz/kcb.sh`.
  - Older `/tmp` scripts are still in use: `/tmp/r04b01a02/rows.py` (b03, b04), `/tmp/r04b01a02/sz/` (b04), `/tmp/r05b02a01/noc4.py` and `b.sh` (b04).
  - b04-a01 measured kernel sizes with the eval cache's sfpi toolchain (`riscv-tt-elf-nm`) on base vs new objects, as r05 advised.
- **Reported worker cost** totals $30.37: b01 $6.81, b02 $8.74, b03 $6.58, b04 $8.24 (r05: $49.45).
  - As before, the turns / minutes on the `[cost ...]` lines don't match wall time. b04-a01 reports 7 turns and 1 min; its log has 80 turns and 1053 s in the first session.
- **`logs/summary_r06.jsonl`** is this summary-writing session's own transcript, not a worker result log.
- **Timestamps:** `decisions.jsonl` and `manifest.json` are in UTC (02:20:08Z). The driver and worker logs use local time (+02:00).
