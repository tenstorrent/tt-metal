# Round r01 summary: routed-expert-prefill

- **Policy:** `v0` (fixed campaign defaults, "parallel refine"), beta 0.6
- **W / R:** W=4 branches, R=4 attempts per branch (max_steps 30, 5 decision steps used)
- **Round root:** `dream/routed-expert-prefill/root` @ `c54a00a7697`, score **1.0000** (it is the baseline; the pre-round drift check measured geomean +0.29%)
- **Wall time:** 20:40:12 → 22:22:27 (+02:00), 1 h 42 min
- **Attempts:** 16 committed, 15 valid, 1 accuracy_fail (`r01-b04-a02`), 0 lost. Branches closed: none.

## Best node: `r01-b04-a04`, score 1.5011

Mechanism: use all 10 Blackhole worker rows, with an 11x10 grid and expert row groups of 10/5/2 rows (previously 11x8 with 8/4/2). Numerics are bit-identical to the baseline. Lineage: root → b04-a01 → b04-a02 → b04-a03 (port of b01-a02) → b04-a04.

| shape | baseline µs | best µs | speedup |
|---|---|---|---|
| deepseek-v3-h7168-i2048 | 801.27 | 547.46 | 1.464 |
| kimi-k2-7-h7168-i2048 | 1161.99 | 764.28 | 1.520 |
| kimi-k3-latent-h3584-i3072 | 2570.42 | 1378.84 | 1.864 |
| glm-5-3-h6144-i2048 | 704.06 | 487.63 | 1.444 |
| mistral-small-4-h4096-i2048 | 261.79 | 193.00 | 1.356 |
| deepseek-v4-flash-h4096-i2048 | 525.15 | 331.22 | 1.586 |
| deepseek-v4-pro-h7168-i3072 | 1604.12 | 1128.00 | 1.422 |
| minimax-m3-h6144-i3072 | 477.77 | 381.96 | 1.251 |
| gpt-oss-120b-h2880-i2880 | 572.05 | 338.76 | 1.689 |
| **geomean** | | | **1.501** |

Runner-up: `r01-b03-a04` scored 1.4856 with the same 10-row idea, but there the picker takes a 10-row layout only when it strictly beats 8 rows in the cost model. b04-a04 always uses 10 rows. That gave it an extra 2-3% on dsv3, dsv4-pro and dsv4-flash, because the light groups finish early and the heavy groups then have DRAM to themselves.

## Score grid (branch x attempt)

| branch | a01 | a02 | a03 | a04 | branch best |
|---|---|---|---|---|---|
| b01 | 1.4376 | **1.4716** | 1.0363 | 1.4576 | 1.4716 (a02) |
| b02 | 1.4017 | 1.4460 | 1.4665 | 1.4705 | 1.4705 (a04) |
| b03 | 0.9416 | 0.9626 | 1.4574 | 1.4856 | 1.4856 (a04) |
| b04 | 0.9330 | 0.0000 (accuracy_fail) | 1.4622 | **1.5011** | 1.5011 (a04) |

Running best after each step: 1.4376 (s1, b01-a01) → 1.4716 (s2, b01-a02) → 1.4716 (s3, no new best) → 1.5011 (s4, b04-a04).

## What worked

1. **Expert-parallel row groups** (b01-a01 1.4376, b02-a01 1.4017). These were found independently in two branches. The 8 M-rows are split into 1/2/4-row groups that run different experts concurrently. That removes phantom M-rows and gives 4-8x more weight-issuing cores. Weight bandwidth rose from about 200-265 GB/s to about 330-390 GB/s. Every other gain this round sits on top of this.
2. **Collision-free DM placement for row groups** (b02-a02 +3.2% over its parent). A fixed, staggered x sender keeps x reads off the weight-sender RISCs, groups have at least 2 rows, and cores no longer read bias for other groups' experts. gpt-oss went from 477 to 383 µs. This closed the gap between the b01 and b02 schedules.
3. **Row-shared x tilize** (b01-a02 +2.4%, b02-a03 +1.4% when ported). One owner core per M-row tilizes each row-major x K-block and multicasts bf8 tiles, replacing 11 redundant tilizes. This is worth about 5% on compute-bound shapes (gpt-oss, mistral) and is neutral on DRAM-bound shapes.
4. **Using the 2 spare BH worker rows** (b03-a04 +1.9%, b04-a04 +2.7% over their parents). On kimi-k3 the critical group drops from 7 to 6 experts (-7%), and gpt-oss uses 5-row groups (-8 to -10%). This is the first work-partition change after the row groups.

## What failed and why

| node | score | failure | reason (from its reflection) |
|---|---|---|---|
| b03-a01 | 0.9416 | regression -6% | It issues the next weight block's DRAM read before the current multicast. Read requests and responses contend with the critical multicast on the sender NIU/RISC, so the read-to-multicast chain was never the bottleneck. |
| b04-a01 | 0.9330 | regression -7% | Same idea as b03-a01, same result and same explanation. |
| b03-a02 | 0.9626 | regression -4% vs root | It batches x over 4 K-blocks. The saved per-block inits are worth nothing measurable. x arrives in bursts with the same lead time, so its latency is exposed through the lock-step grid. |
| b04-a02 | 0 (accuracy_fail) | mean_abs +2-3.5% on all 9 shapes | Changing `in0_block_w_gu` from 16 to 32 doubles the K-span accumulated in bf16 dst (fp32_dest_acc_en=false), so it is a precision knob. Timing alone was about 1.004 (neutral). |
| b01-a03 | 1.0363 | regression -30% vs parent | Group pacing via a grid-multicast progress board. The spread between groups collapsed to about 1%, but every group slowed by 29-55%. The non-posted 87-destination publish sits on a weight-sender/multicast core, and later write barriers wait on its acks. |
| b01-a04 | 1.4576 | neutral | Cross-unit x look-ahead. The x(0) bubble at the expert boundary is not on the critical path. |
| b03-a03 | 1.4574 | neutral | The weight-sender feeds the next unit's first block during the drain. The weight-side boundary bubble is not on the critical path either. |
| b04-a03 | 1.4622 | neutral | 3-slot input rings. The median core was not throttled at 2 slots, and the slow group stays slow for the whole call (a persistent bandwidth share, not jitter). |
| b02-a04 | 1.4705 | neutral | Ragged 1x5+1x4 subblocks instead of 3x 1x3. Per-call LLK unpack-context cost is not significant; the per-tile-mult cost is the srcA bf4 unpack. Subblock geometry is exhausted (taller subblocks do not cut unpacks on the BH LLK). |

Findings now established as negative results: read-ahead before the multicast, wider gate/up K-blocks (accuracy), x batching, deeper rings, expert-boundary feeds on both the weight and x paths, group pacing, and matmul subblock shape (height or width).

Open levers named by several reflections:
- **Group-tail imbalance.** At 5 groups, expert counts rarely divide evenly: kimi-k2.7 is 3/3/2/2/2, kimi-k3 6/6/6/5/5, dsv3/glm 2/2/2/1/1. Mistral/minimax leave 2 rows idle. Candidate fixes are an M-split or N-split of the last-wave experts across idle groups.
- **Random per-call "slow group".** Groups doing equal work finish 3-14% apart. Proposed probes are NoC VC separation for weight read requests and a row→group placement study.
- **Serial SFPU activation phase** (estimated 7-17% of group time, unmeasured). Overlapping it on the PACK thread is the largest untried compute lever.
- **gpt-oss per-expert fixed cost**: the phantom 11th down K-block, single-buffered bias CBs, and SwiGLU-OAI.
- The large shapes (dsv3, kimi-k2.7, glm, dsv4-pro, now kimi-k3) are at about 406-434 GB/s total DRAM traffic. That leaves an estimated 5-10% headroom.

## Policy observations (input for dreaming)

- **v0 always refines the latest child, even after a regression.** b01-a04 was parented on b01-a03 (1.0363), and its worker had to revert the pacing code by hand. b03-a02 was parented on b03-a01 (0.9416) and b04-a02 on b04-a01 (0.9330). Both spent part of the attempt reverting, and neither improved on the root. Parent selection should fall back to the branch best (or the global best) when the latest child scored below its parent.
- **Workers rebased by hand because the policy could not.** By step 3, three of four lineages had manually ported or rebased onto b01-a02: b02-a03 (port), b03-a03 (rebased onto b01-a02 code), and b04-a03 (port + ring depth). Each of those attempts carries its delta on top of a port, so the scores mix two effects. The b03-a03 and b04-a03 scores are mostly b01-a02's score. A policy-level "rebase onto global best" action would make these attempts cheaper and their deltas cleaner.
- **Duplicated ideas within the same step** (no cross-branch visibility at step time):
  - Step 1: b03-a01 and b04-a01 both tried weight read-ahead before the multicast. Both regressed, so about 2 attempts were wasted on one idea.
  - Step 1: b01-a01 and b02-a01 both built row groups. Both won; this is a useful replication, but the work was redundant.
  - Step 3: three ports of b01-a02.
  - Step 4: b03-a04 and b04-a04 both added the 10-row grid. b04-a04's unconditional variant was better.
- **No premature closes.** `closed` stayed empty all round. The opposite problem appeared: b03 and b04 stayed below 1.0 for two steps each and were kept open on the regressed child.
- **Wasted or low-yield attempts.** 5 of 16 scored below the root (b03-a01, b04-a01, b03-a02, b04-a02, b01-a03). 4 more were neutral within noise (b01-a04, b02-a04, b03-a03, b04-a03). Only 4 set a new running best (b01-a01, b01-a02, b03-a04, b04-a04).
- **The measurement node was recommended repeatedly but never run.** At least 6 reflections ask for DeviceZoneScopedN phase zones (gate/up, activation, down, sender wait) before more speculative changes. The fixed exploit-every-branch policy has no slot for a measurement-only attempt.
- **Noise on the one-expert-per-group shapes is about ±5% per run** (mistral 186-200 µs and minimax across nodes with identical code), well above the campaign's ±1% band. Several reflections judged by per-group mean TRISC1 end times instead. Scoring or acceptance may want a per-shape noise estimate.
- **Diminishing returns by step.** Step-best deltas were +0.4376, +0.0340, 0, +0.0295. Step 3 produced no new best, and the gain at step 4 came from a new work-partition idea, not from refinement.

## Ops notes

- **First driver start aborted on drift** (20:36). The single-shape check flagged gpt-oss at +1.40% vs a 1.0% noise band. The drift check was changed to use the geomean plus 3x-noise outliers, confirmed by a second measurement (commit `8889b1b6097`). On restart (20:39) it passed with geomean +0.29% and worst shape mistral +2.74%.
- **Step durations** (synchronous barrier; the slowest worker sets the step):

| step | duration | slowest | fastest finished |
|---|---|---|---|
| 1 | 17.4 min | b03-a01 | b02-a01 at 20:52 |
| 2 | 26.5 min | b01-a02 | b02-a02 at 21:07 |
| 3 | 23.4 min | b03-a03 | b02-a03 at 21:28, then about 19 min idle |
| 4 | 34.9 min | b01-a04 | b04-a04 at 21:57, then about 25 min idle |

- **7/16 workers exited rc=143.** In each case the worker had committed while the session was still running, and the driver killed the leftover processes. Tags were present for all 16. No attempts were lost.
- **verify.log:** 16/16 ok, no issues.
- **Host build** was skipped (kernel-only change) for b02-a01 and b03-a01. All other attempts have a build log.
- **Reported worker cost** totals $92.93 (b01 $34.57, b02 $15.54, b03 $26.12, b04 $16.70). The turns/minutes fields on the `[cost ...]` lines (3-5 turns, 0-1 min) do not match the 4-35 min wall times, except b03-a04 (82 turns, 14 min). The accounting appears to capture only the final session segment, so the cost figures may also be undercounts.
- Timestamps in `decisions.jsonl` / `manifest.json` are UTC (18:40Z); the driver and worker logs use local time (+02:00).
