# Round r05 summary: routed-expert-prefill

- **Policy:** `v2` (one-step rounds after the first: W=4, R=1), beta 0.6. This is the first round on v2. Dreaming after r04 promoted `v1-cand4` as `v2` (activated 03:25:32). Its r01-r04 replay gave V 1.5703 vs v1's 1.5511.
- **W / R:** W=4 branches, R=1 attempt per branch (max_steps 30). Step 1 ran 4 workers from the root. Step 2 returned an empty batch with `stop: true`.
- **Round root:** `dream/routed-expert-prefill/n/r04-b02-a01` @ `23a0c81920c`, score **1.6275**.
  - This is r04's best node: the weight-sender look-ahead in both phases, kernel-only and bit-identical.
  - The pre-round drift check measured geomean -0.06%, worst shape gpt-oss -0.31% (noise band 1.0%).
- **Wall time:** 03:26:28 → 04:11:25 (+02:00), 45 min (r04: 1 h 20 min, r03: 1 h 48 min).
- **Attempts:** 4 committed, 3 valid, 0 runtime_error, 0 accuracy_fail, 1 hang, 0 lost.
  - No kernel-config-buffer overflows this round (r04: 4 of 9 first runs).
  - Every valid node scored on its first and only device run.
- **Branches closed:** none. With R=1 every branch ends after its one attempt.
- **Round gain:** 1.6275 → 1.6667 (+2.4%), outside the ±1% noise band. Two nodes beat the root outside noise: b02-a01 (+2.4%) and b01-a01 (+1.1%).

## Best node: `r05-b02-a01`, score 1.6667

Mechanism: rebalance the weight DRAM reads about 60/40 toward NoC 0.
- **Gate/up:** the reader also reads the first ~0.2w up K-rows of each gate/up block (`up_head_rows`), using the gate-layout accessor.
- **Down:** the reader reads the leading ~60% of tiles of every down K-row (`d_head_tiles`). This is a tile split instead of the previous row split.
- No byte, order or handshake changes. The hot NoC-1 response links in the DRAM columns carry ~40% of the weight bytes instead of 50%.
- Host+kernel (program factory, reader +240 B, writer +4 B), bit-identical to the baseline.

Lineage: r04-b02-a01 (root) → b02-a01.

| shape | baseline µs | round root µs | best µs | speedup vs baseline | Δ vs round root |
|---|---|---|---|---|---|
| deepseek-v3-h7168-i2048 | 801.27 | 512.82 | 504.86 | 1.587 | -1.6% |
| kimi-k2-7-h7168-i2048 | 1161.99 | 717.28 | 701.38 | 1.657 | -2.2% |
| kimi-k3-latent-h3584-i3072 | 2570.42 | 1342.79 | 1243.64 | 2.067 | -7.4% |
| glm-5-3-h6144-i2048 | 704.06 | 449.24 | 436.68 | 1.612 | -2.8% |
| mistral-small-4-h4096-i2048 | 261.79 | 183.76 | 190.11 | 1.377 | +3.5% |
| deepseek-v4-flash-h4096-i2048 | 525.15 | 308.69 | 298.41 | 1.760 | -3.3% |
| deepseek-v4-pro-h7168-i3072 | 1604.12 | 1056.41 | 993.18 | 1.615 | -6.0% |
| minimax-m3-h6144-i3072 | 477.77 | 372.86 | 361.90 | 1.320 | -2.9% |
| gpt-oss-120b-h2880-i2880 | 572.05 | 255.75 | 260.97 | 2.192 | +2.0% |
| **geomean** | | | | **1.667** | |

- The mean row end fell on every DRAM-bound shape, by more than the chip time:

  | shape | mean row end |
  |---|---|
  | kimi-k3 | -12% |
  | dsv4-pro | -8.3% |
  | minimax-m3 | -6.9% |
  | dsv4-flash | -6.3% |
  | glm | -5.9% |
  | dsv3 | -5.2% |
  | kimi-k2.7 | -4.7% |

- The chip time (max core) took less of the gain because the row groups no longer run at equal speed. The lower groups (y7-y11) gained far more than the upper groups (y2-y6).
- mistral +3.5% and gpt-oss +2.0% are their usual max-core lottery. Their mean row ends moved 164 → 160 and 229 → 232 µs, and mistral's mean DN fell 11%.

Per-shape bests from other nodes this round:

| shape | best µs | node |
|---|---|---|
| mistral | 176.18 | b01-a01 |
| minimax-m3 | 350.99 | b01-a01 |
| gpt-oss | 254.00 | b01-a01 |

- The best node holds the round's per-shape bests on the other 6 shapes.
- minimax-m3 (-5.9%) and mistral (-4.1%) on b01-a01 come from a mechanism with a measured effect on those shapes: the 3+3+2+2 layout.
- gpt-oss on b01-a01 is within noise. Its plan did not change.

## Score grid (branch x attempt)

| branch | a01 | branch best | Δ vs root | outcome |
|---|---|---|---|---|
| b01 | 1.6457 | 1.6457 (a01) | +1.1% | win, just outside noise |
| b02 | **1.6667** | 1.6667 (a01) | +2.4% | win, round best |
| b03 | 1.6177 | 1.6177 (a01) | -0.6% | neutral |
| b04 | 0.0000 | - | - | hang |

Bold marks the round best. There was one step:

| step | workers | running best | set by | step-best Δ | outside noise? |
|---|---|---|---|---|---|
| s1 | 4 | 1.6667 | b02-a01 | +0.0392 | yes (+2.4%) |
| s2 | 0 | - | stop | | |

Cross-branch code traffic that the parent links don't show:
- b01-a01 ports two r04 nodes onto the root:
  - r03-b01-a02's plan-once-per-core;
  - r04-b03-a03's idle-row tie pass.

  Both were open levers in r04's summary.
- b04-a01's analysis ran on r04-b03-a01's UNPACK wait tallies (a sibling of the root, not an ancestor).

## What worked

1. **Weight stream NoC rebalance** (b02-a01, +2.4%; the round's largest gain and the largest since r02).
   - **The "DRAM ceiling" was the NoC-1 response links.** Earlier nodes assumed the ~420-450 GB/s ceiling was DRAM. Moving ~10% of the weight bytes from NoC 1 to NoC 0 shortened both phases on every DRAM-bound shape.
   - **Gate/up:** -3 to -14% (kimi-k3 -14%, dsv4-flash -9%, dsv4-pro -6.5%).
   - **Down:** -2 to -13% (kimi-k2.7 -13%, minimax-m3 -13%, dsv4-pro -12%, dsv3 -10%).
   - **Bandwidth:** whole-op weight bandwidth went 422 → 449 GB/s on dsv4-pro and 388 → 418 GB/s on kimi-k3. The mean-core rate is ~475 GB/s.
   - **Link model** (`/tmp/r05b02a01/noc4.py`): NoC-1 responses run north along the DRAM column first, so the four banks of a column share those links.
     - At a 50/50 split the hottest NoC-1 link carried 65-79 GB/s, against 45 GB/s on NoC 0.
     - At ~60/40 both sit at ~52-56 GB/s.
   - This is the first node that raises the weight delivery ceiling. Earlier nodes reordered who waits.
2. **Plan once per core + idle-row tie pass** (b01-a01, +1.1%).
   - The whole gain is minimax-m3 (-5.9%, mean core end 343 → 319 µs) and mistral (-4.1%), the two shapes whose plan changed to 3+3+2+2.
   - The tie pass reproduces r04-b03-a03's minimax-m3 gain without that node's -Os UNPACK planner penalty. mistral, flat in r04-b03-a03, also gained this time.
   - **Plan sharing is latency-neutral.** First-GU-start moved:
     - +0.2..+0.5 µs on the 7 unchanged shapes;
     - +1.5 µs on kimi-k3 (28-expert plan), because compute now waits for the reader's plan page.
   - The other 7 shapes have identical plans and code paths. Their geomean is about 0.
   - Plan sharing frees the writer's and UNPACK's planner copies (~4-5 KB each at -O2). The h7168 programs now have several KB of kernel-config headroom, where the root had a few hundred bytes. This removes the binding constraint r04 reported, **on the b01 lineage only**.
3. **Two independent measurements of a row-position gradient** (b02-a01 and b03-a01).
   - **b02-a01, steady state:** lower row groups now finish well ahead of upper ones.
     - kimi-k3: rows y8-11 end at 936-950 µs; y2-7 end at 1171-1216 µs.
     - dsv3: the phase-B 4-row group on y2-5 is now critical by 50-90 µs (503 vs 450 / 409).
     - kimi-k2.7 and dsv4-pro: y7-11 end 40-60 µs before y2-6.
   - **b03-a01, first x block (new `RE-X0` zone):** x(0) reaches the bottom row at ~15-20 µs and the top row 25-60 µs later. It is monotonic in grid row on every shape.
     - The prologue is not staggered: the earliest core of each row reaches its kb = 0 wait at the same 5-12 µs.
     - The lag builds up in the first block's DRAM reads (x owner stick reads, gate sender reads) issued from the top rows.
   - Neither node resolved the cause. Candidates are the NoC-0 response path, DRAM arbitration, or the first block's multicast/ack chain.

## What failed and why

| node | score | outcome | reason (from its reflection) |
|---|---|---|---|
| b03-a01 | 1.6177 | neutral (-0.6% vs root), flawed idea | It staggered the row-shared x owner rotation by M-row, so a gate/up K-block's x reads, tilize and multicast spread over ~10 columns instead of 1-2. The owner-column hotspot is real geometrically (5 → 1 owners per column per block on 5x2), but it does not set the pace. Per-row GU gaps are unchanged (mistral y2-3 121 vs y8-9 96 µs; root 120 vs 93). The mean core end moved +2..+7 µs on 8 of 9 shapes, inside spread but the wrong way. The reflection recommends reverting the offset (`x_owner_offset` → 0) and keeping the profiler-only `RE-X0` zone. |
| b04-a01 | 0 | hang, repairable | Torus-aware weight-sender rows: gate/up sender = the group's bottom row, down sender = the top row, so neither column multicast wraps around the torus. The first shape (dsv3) hung, and the eval timed out at 900 s and reset the device. No profile was dumped. The static re-trace found no semaphore cycle. The suspects are role combinations no valid node has run: (1) x owner == gate/up sender together with r04-b02-a01's gate/up look-ahead (most likely); (2) different cores as a column's gate/up and down senders; (3) every top-row core being both down sender and activated sender. |

b04-a01's measured premise still stands. It was computed from r04-b03-a01's tallies with `analysis/wrap.py`. Receivers whose column weight multicast wraps the torus wait much longer:

| shape | gate/up wait, wrap / no wrap (µs) | down wait, wrap / no wrap (µs) |
|---|---|---|
| kimi-k3 | 156.5 / 108.7 | 130.7 / 46.1 |
| dsv4-flash | 44.2 / 21.6 | 18.3 / 9.0 |
| minimax-m3 | 31.1 / 23.7 | 15.6 / 4.1 |
| mistral | 14.9 / 9.3 | 6.7 / 0.6 |
| gpt-oss | 10.4 / 2.4 | 4.1 / 1.1 |

The mechanism's value is unmeasured. The proposed repair has three parts:
- (b) Fix only the down multicast (down sender = top row), keeping the gate/up sender and owners as on the root.
- (c) For gate/up, switch the multicasting RISC per core (reader on NoC 0 for top-row senders) instead of moving the sender row.
- Before any device run, a host-side handshake simulation of one 2-row group to catch deadlocks.

Newly established negatives (in addition to the r01-r04 lists):
- **x owner placement across columns** (b03-a01). Owner placement within the row was already negative (r03-b02-a02).
- **Reverting to a 50/50 NoC split, or moving a whole weight stream onto one NoC** (b02-a01's reading; r03-b03-a02 and r02-b04-a02).
- **The weight streams are not DRAM-bound at ~420-450 GB/s.** The r02-r04 reflections that called dsv3/glm/k2.7 "DRAM-bound, nothing left on the schedule" were reading a NoC-link ceiling (b02-a01).

Open levers named by the reflections:
- **Combine b01-a01 and b02-a01.** The mechanisms are orthogonal: b01 changes the plan and its hand-off, b02 the reader/writer split of weight reads.
  - Per-shape bests are split between them: b01 on mistral and minimax-m3, b02 on the other six shapes.
  - Both edit the reader, writer and program factory, so this is a merge, not a cherry-pick.
  - b02's reader +240 B fits on b01's headroom.
- **Rebalance the plan for the new row speeds** (b02-a01 next #1; planner-only, `adaptive_chunk.hpp`).
  - kimi-k3: deal the extra experts to the lower, faster groups instead of the first ones. The lower groups now end ~270 µs early, more than one expert (~190 µs). The expected result is a max of ~1216 → ~1130 µs (-6%).
  - dsv3 / glm: phase B as (3,3,4), with the 4-row group at the bottom. r02-b02-a04 next #1 proposed the same {3,3,4} table entry.
  - This touches the same planner file as b01-a01's tie pass.
- **Sweep the split once per phase** (b02-a01 next #2): GU `up_head_rows` 4 of 16, DN `d_head_tiles` ~0.65. One knob per node. The model puts the GU optimum at ~0.6 and DN at 0.55-0.62 for 2-row groups.
- **Non-wrapping column multicasts:**
  - b02-a01 next #3: send each multicast on the NoC that reaches the group in 1-4 hops.
  - b04-a01 repair (c): the same thing from the other side.
  - These are the same idea. See "Duplicated ideas" below.
- **First-block row gradient** (b03-a01 next #1). Add profiler-only zones around the owner's up-front x read barrier, the sender's kb = 0 gate read barrier and the writer's kb = 0 `up` barrier, to tell the NoC-0 response path from DRAM arbitration. Then:
  - **If NoC 0:** move the top rows' block-0/1 x reads to the owner's idle BRISC.
  - **If DRAM arbitration:** stagger the bottom groups' start.
- **Spend b01-a01's headroom** (b01-a01 next #1): r04-b04-a02's 4-slot activated ring gated to kimi-k3-like shapes, or r04-b02-a02's pump with a non-blocking act send.
- **The 2-row pcM-3 group on y10-11** (mistral / minimax-m3, b01-a01 layout) is the tail in ~1/3 of calls: mistral 198-205 vs ~160 µs, minimax-m3 385 / 419 vs ~330 µs.
  - The rest of that lottery is per group and per call, not positional (b01-a01 next #3).
  - It persists even when the group runs alone at the end, so it is not DRAM share.

## Policy observations (input for dreaming)

- **v2 ran as designed and paid.**

  | round | attempts | wall time | worker cost | gain |
  |---|---|---|---|---|
  | r05 | 4 | 45 min | $49.45 | +2.4% |
  | r04 | 9 | 1 h 20 min | $71.93 | +1.07% |
  | r03 | 16 | 1 h 48 min | $118.48 | +1.2% outside noise |

  - The round gain is the largest since r02, and 2 of 4 attempts beat the root outside noise.
  - b01-a01 is the kind of "rebase onto the leader" that v2 bet re-rooting would deliver at full width: a step-1 port of two r04 levers (plan-once, tie pass) onto r04's best.
  - The bet held this round. The port came from the r04 summary's open-lever list, not from in-round steps.
- **Re-rooting on one node drops the second win.**
  - r06 re-roots on b02-a01. b01-a01's +1.1% (minimax-m3 -5.9%, mistral -4.1%, and the kernel-config headroom) is not on that lineage.
  - Under R=1 there is no in-round step that could have combined the two. The combination has to come from a step-1 worker in r06 choosing it from history.
  - b02-a01's own top lever (plan rebalance) edits the same planner file. Without b01's headroom, every planner or reader change on the b02 lineage again starts at a few hundred bytes of margin. That was the r04 constraint that b01 removed.
  - Proposal input: when two step-1 nodes both beat the root outside noise with orthogonal tags, either seed one r06 slot with an explicit "combine X and Y" task, or root on the merge.
- **Failed first attempt (v2's "what to watch").**
  - b04-a01 hung with a concrete, staged repair plan: (b) down-only, then (c) RISC switch, plus a host-side deadlock sim.
  - Under R=1 the repair was not tried in-round. Whether r06 retries it from history is the test v2's notes asked for.
  - Its premise is measured (2-11x down-wait ratio on wrapped receivers), so it is a live candidate.
- **No duplicated hypotheses in step 1.** The four mechanisms were distinct:
  - planner hand-off + layout;
  - NoC split of weight reads;
  - x owner placement;
  - weight-sender row placement.

  But three of the four (b02, b03, b04) are NoC-geometry / placement ideas, and two of their "next" lists converge:
  - b02-a01 next #3 (multicast on the NoC that reaches the group without wrapping) and b04-a01 repair (c) (per-core choice of multicasting RISC) are the same change.
  - b02-a01 next #1 (rebalance plan for row speed) and b03-a01 next #1 (first-block row gradient) both target the top-rows-are-slow effect.

  In r03 and r04, the duplicated step came from one node's "next #1". The r06 risk is the same: two or more workers implement the non-wrapping multicast or the row-speed plan rebalance.
  - Proposal input: dedupe by mechanism tags at batch time (`multicast-geometry` / `noc-topology` here), or pass sibling proposals to each worker.
- **A hang costs about a third of the step.**
  - b04-a01's eval held the shared device lock 03:48:47 → 04:03:51 (900 s timeout, then a 4-device reset).
  - b03-a01 reported queueing on the device lock, and its eval ran right after the reset (04:04-04:05). It committed at 04:07:28, ~21 min after b01/b02.
  - The step ended at 04:11:25, when b04 committed its hang reflection.
  - The baseline dsv3 call is ~0.8 ms and a full valid eval took about 1-2 min this round, so a 900 s timeout is ~10x what a valid run needs.
  - Proposal input: a shorter per-shape timeout, or a first-call watchdog.
- **No closes, so no premature closes.** R=1 removes the prune and collapse-close questions from r04.
- **Neutral-by-chip, positive-by-mean.** b02-a01 again shows the chip metric (max core) under-reporting a mechanism. Mean row end fell 4.7-12% on 7 shapes; chip fell 1.6-7.4%.
  - On mistral / gpt-oss, chip moved +2..+3.5% against flat or better means.
  - The ±3-5% lottery on 1-expert-per-group shapes is as large as most single-node gains. It is still scored at face value.
- **Support collapse:** as v2's notes warn, R=1 rounds record no depth. r05 adds four depth-1 nodes and no parent→child Δ beyond the root.
- **Yield:**
  - 3 of 4 valid.
  - 2 new bests over the root, both outside noise: b02-a01 (+2.4%) and b01-a01 (+1.1%).
  - 1 neutral: b03-a01 (-0.6%, inside noise).
  - 1 hang: b04-a01.
- **Diminishing returns reversed for one round.**

  | round | gain | attempts |
  |---|---|---|
  | r01 | +50% over the baseline | 16 |
  | r02 | +4.6% | 16 |
  | r03 | +2.6% by score (+1.2% outside noise) | 16 |
  | r04 | +1.07% | 9 |
  | r05 | +2.4% | 4 |

  r05's gain came from a resource ceiling the earlier reflections had misattributed: NoC-1 links, not DRAM. It did not come from finer scheduling. The reflections now point at three things:
  - placement and geometry (row-speed-aware plans, non-wrapping multicasts, split ratio);
  - the first-block row gradient;
  - the group-tail lottery.

## Ops notes

- **Pre-round:**
  - r04 completed at 03:19:01, and dreaming activated `v2` at 03:25:32.
  - The root was built (`build_r05_root_1.log`, 03:25), and the drift check passed (geomean -0.06%, worst gpt-oss -0.31%).
  - Overhead between r04 completion and r05 planning was 7.5 min.
- **Step duration.** The barrier is synchronous, so the slowest worker sets the step's length. Times are commit times (+02:00).

| worker | commit | eval window | note |
|---|---|---|---|
| b02-a01 | 03:46:11 | ~03:44-03:45 | first finished, then ~25 min idle |
| b01-a01 | 03:46:58 | ~03:45-03:46 | ~24 min idle |
| b03-a01 | 04:07:28 | 04:04-04:05 | queued on the device lock behind b04's hang |
| b04-a01 | 04:11:17 | 03:48:47-04:03:51 | hang, 900 s timeout, then device reset |

  Step 1 ran 45 min (03:26:28 → 04:11:25). The driver logged all four results at 04:11:25.
- **Worker exits:** b01-a01 and b02-a01 exited with rc=143 after committing (03:49:01, 03:48:12). b03-a01 and b04-a01 exited rc=0. Tags were present for all 4, and no attempts were lost.
- **verify.log:** 4/4 ok, no issues.
- **Device:**
  - b04-a01's eval hung on its first shape (deepseek-v3). The eval timed out at 900 s and reset PCI devices 0-3. All four reset successfully, and b03-a01's eval right after it was valid.
  - There was no profile dump for b04-a01; `eval/error.txt` is committed.
  - No kernel-config-buffer overflows and no JIT errors this round.
- **Builds:** all 4 attempts plus the root have a build log.
  - Host code changed in b01-a01 (program factory: plan-page CB), b02-a01 (factory: `up_head_rows`, `d_head_tiles`) and b04-a01 (factory: sender roles).
  - b03-a01 is kernel-only. Its build was most likely triggered by the shared eval checkout changing from b04-a01's host code.
- **Eval / tooling notes from workers:**
  - Again, worker analysis scripts cited for reuse live only in `/tmp`:
    - `/tmp/r05b02a01/noc*.py` (NoC link-load model), `b.sh`, `b2.sh`;
    - `/tmp/r05b01a01/rows.py`, `outl.py`, `cores.py`;
    - `/tmp/r05b03-a01-nikola/ana/` (`rx0.py`, `rx0min.py`, `owners.py`, `uwrow.py`, `calls.py`), plus `build.sh`, `kc_base`, `kc_new`;
    - the r04 scripts `/tmp/r04b04/zones2.py` and `/tmp/r04b01a02/rows.py`, still used by three of four workers.
  - **Exception:** b04-a01 committed its analysis scripts in the node (`attempts/r05-b04-a01/analysis/`: `wrap.py`, `strag.py`, `grp2.py`, `slowuw.py`, `rows.py`). This is the first node to do so.
  - b01-a01 repeats r04's advice to check sizes with the JIT ELFs in the eval cache, not local -O2 builds.
  - b03-a01's `RE-X0` zone is profiler-only (~0.3 KB NCRISC in profiler builds).
- **Reported worker cost** totals $49.45: b01 $9.66, b02 $10.60, b03 $13.38, b04 $15.81 (r04: $71.93 for 9 attempts).
  - The turns/minutes on the `[cost ...]` lines again don't match wall time. b01-a01 reports 4 turns and 0 min against ~20 min of wall time; b03-a01 reports 8 turns and 2 min against ~41 min.
- **Timestamps:** `decisions.jsonl` and `manifest.json` are in UTC (01:26:28Z). The driver and worker logs use local time (+02:00).
