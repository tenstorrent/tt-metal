# Round r02 summary: routed-expert-prefill

- **Policy:** `v0` (fixed campaign defaults, "parallel refine"), beta 0.6. Dreaming after r01 produced five candidates (`v0-cand1`..`v0-cand5`), and none was promoted, so this round ran `v0` unchanged.
- **W / R:** W=4 branches, R=4 attempts per branch (max_steps 30, 5 decision steps used)
- **Round root:** `dream/routed-expert-prefill/n/r01-b04-a04` @ `ff0bcd951ef`, score **1.5011**. The pre-round drift check measured geomean -0.05%, worst shape gpt-oss -0.43% (noise band 1.0%).
- **Wall time:** 22:28:36 → 23:50:13 (+02:00), 1 h 22 min
- **Attempts:** 16 committed, 16 valid, 0 accuracy_fail, 0 lost. Branches closed: none.
- **Round gain:** 1.5011 → 1.5697 (+4.6%)

## Best node: `r02-b02-a04`, score 1.5697

Mechanism: on r02-b03-a03's code, phase B of the two-phase schedule may use a mixed-height layout (4+3+3 etc.), with one leftover expert per group, largest first. dsv3 and glm each have 3 leftover experts; these now run at per_core_M 2 on the 4 rows that used to idle, instead of at per_core_M 3 on 2-row groups. The change is planner-only (one header) and numerics are bit-identical to the baseline. Lineage: r01-b04-a04 → b02-a01 → b02-a02 → b02-a03, then rebased onto b03-a03's code (r01-b04-a04 → b03-a01 → b03-a02 → b03-a03) → b02-a04.

| shape | baseline µs | round root µs | best µs | speedup vs baseline | Δ vs round root |
|---|---|---|---|---|---|
| deepseek-v3-h7168-i2048 | 801.27 | 547.46 | 529.07 | 1.514 | -3.4% |
| kimi-k2-7-h7168-i2048 | 1161.99 | 764.28 | 736.47 | 1.578 | -3.6% |
| kimi-k3-latent-h3584-i3072 | 2570.42 | 1378.84 | 1405.94 | 1.828 | +2.0% |
| glm-5-3-h6144-i2048 | 704.06 | 487.63 | 453.49 | 1.553 | -7.0% |
| mistral-small-4-h4096-i2048 | 261.79 | 193.00 | 195.63 | 1.338 | +1.4% |
| deepseek-v4-flash-h4096-i2048 | 525.15 | 331.22 | 327.93 | 1.601 | -1.0% |
| deepseek-v4-pro-h7168-i3072 | 1604.12 | 1128.00 | 1080.73 | 1.484 | -4.2% |
| minimax-m3-h6144-i3072 | 477.77 | 381.96 | 381.63 | 1.252 | -0.1% |
| gpt-oss-120b-h2880-i2880 | 572.05 | 338.76 | 267.00 | 2.143 | -21.2% |
| **geomean** | | | | **1.570** | |

The top three nodes cannot be separated, because they lie within 0.43% of each other, inside the ±1% noise band:

| node | score |
|---|---|
| b02-a04 | 1.5697 |
| b01-a04 | 1.5657 |
| b03-a04 | 1.5630 |

- **b01-a04 and b03-a04** carry the PACK-thread pipelined activation on top of b03-a03. b02-a04 does not have it.
- **b02-a04** carries the mixed phase B. b01-a04 and b03-a04 do not have it.
- **The two changes are orthogonal** (different files, planner vs compute K-loop). Combining them has not been tried yet.

Per-shape bests from other nodes this round:

| shape | best µs | node |
|---|---|---|
| kimi-k3 | 1353.55 | b01-a03 |
| mistral | 184.54 | b01-a03 |
| minimax-m3 | 368.25 | b01-a04 |
| gpt-oss | 261.90 | b03-a04 |
| dsv4-pro | 1068.83 | b01-a04 |
| dsv4-flash | 322.37 | b04-a01 |

## Score grid (branch x attempt)

| branch | a01 | a02 | a03 | a04 | branch best |
|---|---|---|---|---|---|
| b01 | 1.4854 | 1.4889 | 1.5275 | 1.5657 | 1.5657 (a04) |
| b02 | 1.5279 | 1.5191 | 1.5059 | **1.5697** | 1.5697 (a04) |
| b03 | 1.4973 | 1.5278 | **1.5435** | 1.5630 | 1.5630 (a04) |
| b04 | **1.5314** | 1.3612 | 1.5234 | 1.5427 | 1.5427 (a04) |

Bold marks the round's running best when it was set. The running best after each step was:

| step | running best | set by | step-best Δ |
|---|---|---|---|
| s1 | 1.5314 | b04-a01 | +0.0303 |
| s2 | 1.5314 | no new best | 0 |
| s3 | 1.5435 | b03-a03 | +0.0121 |
| s4 | 1.5697 | b02-a04 | +0.0262 |

## What worked

1. **Mixed-height row-group layouts** (b02-a01 +1.8%, b04-a01 +2.0%). Two branches found this independently in step 1.
   - Layouts such as 3+3+2+2 sit beside the uniform 10 / 5+5 / 2x5 layouts, with a height-aware largest-first expert deal.
   - gpt-oss improved by 18-21% because each group now runs one expert instead of two, which removes one per-expert fixed cost per group.
   - That fixed cost was measured at about 70 µs on b02-a01 (one-expert-per-group comparison) and fitted at about 150 µs on b04-a01 (F + U·pcM fit). Either way it is several per_core_M units, not 1.
   - dsv4-flash improved 2.7% on b04-a01. Every other shape kept its schedule exactly.
2. **Two-phase schedule** (b03-a01 → b03-a02 → b02-a04).
   - The plan runs full waves on the small row groups, then runs the leftover experts on taller groups.
   - The first version (b03-a01) was neutral: kimi-k2.7 -2.6% and dsv4-pro -3.9%, but dsv3 +3.2% because its leftover needed two phase-B rounds.
   - Restricting the plan to one phase-B round (b03-a02) kept the wins and removed the dsv3 loss.
   - A mixed-height phase B (b02-a04) extended it to dsv3 (-6.4%) and glm (-7.9%). Their rows now end within about 35 µs of each other, and both shapes run at about 440 GB/s.
3. **Op-local short SFPU bodies with the same math** (b03-a03 +1.0%).
   - SwiGLU-OAI went from 54 to 31 instructions per row and is replayed. Clamped SiLU-GLU went from 44 to 35, and SiTU-GLU from 77 to 65.
   - gpt-oss improved 4.8% and minimax-m3 4.2%. Across nodes, b04-a04 estimates minimax's real gain at only 1-2%.
   - kimi-k3 and dsv4 stayed flat because they are DRAM-bound.
   - b04-a04 gave plain SiLU the same treatment (34 → 25 instructions, about 4 µs per expert). That only shows on compute-bound SiLU shapes.
4. **PACK-thread activation pipelined into the last gate/up K-block** (b01-a03 +2.6%).
   - Each subblock's SFPU runs on PACK while the next subblock's last-block matmul runs on MATH.
   - RE-ACT dropped to about 0 µs on every binary-activation shape.
   - The gain stacks with the short bodies: b03-a04 is +1.3% over b03-a03, and b01-a04 is +2.5% over b01-a03.
5. **First per-phase measurement in the campaign** (MATH zones RE-GU / RE-ACT / RE-DN, in b01-a02 and b04-a02).
   - The serial activation cost about 15-55 µs per expert on the binary-activation shapes (gpt-oss ~21% of core busy time, kimi-k3 ~15-18%).
   - The down phase takes about 1/3 of the time, which matches its MAC and weight-byte share.
   - The kernel prologue under the old picker was about 1 µs.
   - These numbers directed the step-3 and step-4 compute wins (items 3 and 4).
6. **Single-row expert groups** (b04-a03): kimi-k3 -2.3%, about +0.25% geomean. It is small because kimi-k3 is DRAM-bound at about 408 GB/s with either layout.

## What failed and why

| node | score | outcome | reason (from its reflection) |
|---|---|---|---|
| b01-a01 | 1.4854 | -1.0% vs root | It used a picker that treats DRAM as one shared fluid ceiling, so few-tile shapes ran 2x5-row groups at pcM 1. Aggregate DRAM bandwidth actually grows with the number of concurrent weight streams: about 250 GB/s for 1 group, 340-390 for 2 and 440 for 5. Fewer, larger groups therefore lose. dsv4-flash was +5.2%. |
| b03-a01 | 1.4973 | neutral | Two-phase with the same uncapped model. Each phase-B expert ran at 2.3-2.7 c instead of the modelled 2.0, because a 5-row group's 11 column senders deliver only about 250 GB/s. dsv3 and glm needed two phase-B rounds and lost. This node was repaired in b03-a02. |
| b01-a02 | 1.4889 | neutral | It folded the per-core `start` page read into the counts multicast. The prologue was only about 1 µs, so there was nothing to win. The node's value is the zones (see What worked 5). |
| b02-a02 | 1.5191 | neutral (gpt-oss +4.1%) | Per-sender NoC request VCs for weight reads. The group-tail scatter was unchanged. The contended resources are read responses and DRAM banks, not request arbitration. Some VC/route combinations appear to be served worse. |
| b02-a03 | 1.5059 | -1.4% vs b02-a01 | Down-weight look-ahead across the activation gap, using a deeper down ring. The mean core end did not move: other groups already absorb the bandwidth a group leaves idle during its activation. The look-ahead only widened the group-tail spread (mistral +6.3%, gpt-oss +8.5%). |
| b04-a02 | 1.3612 | -11% vs parent | A writer-owned down-weight stream on NoC 1. This moved all down reads off NoC 0 and serialized read → barrier → multicast on one RISC. DRAM-bound shapes lost 10-22%. The premise was wrong: the down phase is about 1/3 of the time, proportional to its MACs, so it is not a bottleneck. The first device run also failed to JIT (`-Werror=unused-local-typedefs`). |
| b04-a03 | 1.5234 | neutral vs b04-a01 | 10 x 1-row groups. kimi-k3 improved only 2.3%, against a predicted 6-9%. The critical work fell by 17%, but each tile-row got about 16% slower, because DRAM stays at about 408 GB/s. |

Newly established negatives (in addition to the r01 list):
- moving the tail onto fewer, larger groups (tried twice);
- weight-read request VCs;
- any use of the activation gap for DRAM look-ahead;
- decoupling the down stream, or moving weight reads off the reader's NoC 0;
- prologue folding;
- cutting SFPU instructions further without a precision change. The bodies are now the bare op sequences: SiLU 25, SwiGLU-OAI 31, clamped SiLU-GLU about 30.

Where the shapes stand:
- **At the DRAM floor (about 405-455 GB/s):** dsv3, glm (since b02-a04), kimi-k2.7, dsv4-pro and kimi-k3. dsv4-flash is near it.
- **Not DRAM-bound:** gpt-oss, minimax-m3 and mistral.

Open levers named by the reflections:
- **Combine b01-a04's PACK-thread pipeline with b02-a04's mixed phase B.** The two siblings both sit on b03-a03, and the changes are orthogonal.
- **Planner prologue.** b01-a04 measured kernel start → first RE-GU at 29.2 µs on kimi-k3 and 10-11 µs on dsv3/glm/k2.7/dsv4, against 3-6 µs with the old picker. Five reflections (b03-a02, b02-a04, b03-a04, b04-a04, b01-a04) name the fix: reject two-phase before any modelling, cache `tiles_of` in a local array, and reuse the uniform pick. Nobody has tried it.
- **Sub-expert split (N-split of a last-wave expert across two groups).** This is the only lever named for the remaining max − mean gap and for mistral/minimax-m3, which leave 2 rows idle with 4 experts on 5 groups (b02-a03, b02-a04, b01-a04).
- **Per-group weight-stream rate**, e.g. two senders per column in tall groups (b01-a01, b03-a01).
- **Group-tail lottery probes:**
  - x stick reads on the weight-sender reader RISC (b04-a03);
  - DRAM bank-phase rotation across groups (b02-a02);
  - sender-side DM zones.
- **Exposed last-subblock SFPU at the GU → DN boundary**, about 2-5 µs per expert (b01-a03, b03-a04).

## Policy observations (input for dreaming)

- **r01's observations recurred unchanged** because no v1 was promoted. In particular, refining the latest child after a regression or neutral result happened again, three times:
  - b01-a02 was parented on b01-a01 (-1.0%);
  - b02-a03 on b02-a02 (neutral);
  - b04-a03 on b04-a02 (-11%).

  Each worker spent part of its attempt reverting the parent by hand. b02-a04 then abandoned its own lineage entirely and rebased onto b03-a03.
- **Workers combine and rebase by hand. By step 4, all four attempts ran b03-a03's code, fully or partly:**
  - b03-a02 ported b04-a01's layouts;
  - b01-a04 merged onto b03-a03's lineage;
  - b02-a04 rebased onto b03-a03;
  - b04-a04 ported b03-a03's SFPU bodies.

  As in r01, those deltas mix a port with a new idea. Cross-branch visibility between steps exists (workers read the regenerated history). No policy-level combine or rebase action decides who merges what.
- **Duplicated ideas within the same step:**
  - Step 1: b02-a01 and b04-a01 both built mixed-height layouts (1.5279 / 1.5314). Both won, but the work was redundant.
  - Step 1: b01-a01 and b03-a01 both moved tail work onto fewer, larger groups under a shared-DRAM-ceiling model. Both failed for the same reason, so about 2 attempts went to one flawed premise.
  - Step 2: b01-a02 and b04-a02 both added the same MATH phase zones, bundled with different speculative mechanisms. Both mechanisms failed, and the measurement was taken twice.
  - Step 4: b01-a04 and b03-a04 built the same combination (b01-a03's PACK pipeline plus b03-a03's bodies on the two-phase/mixed schedule), scoring 1.5657 vs 1.5630. One attempt was wasted.
- **Low-yield attempts.**
  - Four of 16 scored below the round root: b01-a01, b01-a02, b03-a01 and b04-a02.
  - Five scored below their parent: b01-a01, b02-a02, b02-a03, b03-a01 and b04-a02.
  - Six beat the best known at the start of their step: b02-a01, b04-a01, b03-a03, b01-a04, b02-a04 and b03-a04.
  - Step 2 produced no new best.
- **No premature closes.** `closed` stayed empty. b02 was kept open through two non-improving children (a02, a03) and then produced the round best, but through a rebase onto another branch, not through its own lineage.
- **Measurement is still not a policy action.** r01 asked for phase zones, and this round got them, but only as add-ons to speculative attempts (b01-a02, b04-a02). b03-a03 could not tell stall-bound from DRAM-bound for SiTU-GLU because its lineage had no zones.
- **Gains now come from combining, not refining.** Steps 3-4 gained +0.0383 in total, mostly from cross-branch combinations (b03-a03 lineage + b01-a03 pipeline + b02-a04 planner). Combination attempts were also cheap: b01-a04 took 5 min and b02-a04 8 min, against 15-25 min for new mechanisms.
- **Noise.**
  - The top three nodes are within 0.43%, so the "best" node is a noise draw among them.
  - Several reflections re-attribute earlier deltas to the ±3-5% group-tail lottery on mistral, minimax-m3, glm and gpt-oss. For example, b04-a04 judged b03-a03's minimax -4.2% partly a high parent draw.
  - b03-a02 fell 0.2% short of the best only because shapes with unchanged plans came in 1-3% slower.
  - Reflections increasingly judge by median-core TRISC1 end rather than op time. Per-shape noise estimates in scoring would help.
- **Diminishing returns:** r01 gained +50% over the baseline and r02 gained +4.6% over r01's best. Six of nine shapes are now at or near the DRAM floor, so most remaining headroom is on gpt-oss, minimax-m3 and mistral.

## Ops notes

- **Pre-round:** dreaming after r01 finished at 22:27:41 and kept v0. The root was built (`build_r02_root_1.log`), and the drift check passed (geomean -0.05%).
- **Step durations.** The barrier is synchronous, so the slowest worker sets each step's length. b04 was the slowest branch in every step.

| step | duration | slowest | fastest finished |
|---|---|---|---|
| 1 | 25.3 min | b04-a01 | b01-a01 at 22:38, then about 15 min idle |
| 2 | 21.9 min | b04-a02 | b03-a02 at 23:00, then about 15 min idle |
| 3 | 22.6 min | b04-a03 | b02-a03 at 23:31, then about 8 min idle |
| 4 | 11.9 min | b04-a04 | b01-a04 at 23:43, then about 7 min idle |

- **Worker exits:** 5 of 16 workers exited with rc=143: b03-a01, b03-a03, b03-a04, b04-a02 and b04-a04. Each had committed while its session was still running, and the driver stopped the leftover processes. Tags were present for all 16, and no attempts were lost.
- **verify.log:** 16/16 ok, no issues.
- **Builds:** the host build was skipped only for b03-a04 (kernel-only). b02-a02 and b04-a04 report that the eval rebuilt the host only because the shared eval checkout had changed. b04-a02's first device run failed to JIT on an unused typedef and was re-run once.
- **Reported worker cost** totals $91.27: b01 $18.27, b02 $21.78, b03 $20.76, b04 $30.46. As in r01, the turns/minutes on the `[cost ...]` lines often do not match wall time. For example, b02-a01 reports 9 turns and 0 min against 17 min of wall time, and b04-a03 reports 6 turns and 0 min against 22 min. The figures likely capture only the final session segment.
- **Timestamps:** `decisions.jsonl` and `manifest.json` are in UTC (20:28Z). The driver and worker logs use local time (+02:00).
