# Round r03 summary: routed-expert-prefill

- **Policy:** `v0` (fixed campaign defaults, "parallel refine"), beta 0.6. Dreaming after r02 produced five new candidates (`v0-cand1`..`v0-cand5`, overwriting the r01 set). None was promoted, so this round ran `v0` unchanged again. The best candidates beat v0 by 0.0037-0.0042 in replay, less than one attempt's cost (0.005), so they counted as ties. The recommended candidate was `v0-cand2`.
- **W / R:** W=4 branches, R=4 attempts per branch (max_steps 30, 5 decision steps used)
- **Round root:** `dream/routed-expert-prefill/n/r02-b02-a04` @ `cb2514ebcd5`, score **1.5697**. The pre-round drift check measured geomean -0.04%, worst shape gpt-oss -0.25% (noise band 1.0%).
- **Wall time:** 23:57:40 → 01:45:56 (+02:00), 1 h 48 min
- **Attempts:** 16 committed, 14 valid, 1 runtime_error (b01-a01), 1 accuracy_fail (b03-a03), 0 lost. Branches closed: none.
- **Round gain:** 1.5697 → 1.6103 (+2.6%). Only the step-1 gain (+1.2%, b02-a01) is outside the ±1% noise band; see the note on the best node below.

## Best node: `r03-b02-a03`, score 1.6103

Mechanism: a measurement node. It adds profiler-only Tensix hardware perf counters on the MATH thread (FPU/SFPU busy, unpacker busy, SrcA/SrcB writes, per-thread stalls) and samples the FW matmul throttle, accumulated per gate/up and down window. Outside profiler builds it emits no code, so it is functionally identical to its parent r03-b02-a02 (1.5952). Its reflection classifies the +0.9% over the parent as noise, not a win.

Lineage: r02-b02-a04 (root) → b02-a01 (r02-b02-a04's mixed phase-B planner + r02-b01-a04's PACK-thread pipelined activation + a planner prologue trim) → b02-a02 (x owner moved off the weight-sender cores, plus a TRISC planner at -O2 to fit the kernel config buffer) → b02-a03 (counters). Numerics are bit-identical to the baseline.

| shape | baseline µs | round root µs | best µs | speedup vs baseline | Δ vs round root |
|---|---|---|---|---|---|
| deepseek-v3-h7168-i2048 | 801.27 | 529.07 | 513.98 | 1.559 | -2.9% |
| kimi-k2-7-h7168-i2048 | 1161.99 | 736.47 | 729.53 | 1.593 | -0.9% |
| kimi-k3-latent-h3584-i3072 | 2570.42 | 1405.94 | 1365.81 | 1.882 | -2.9% |
| glm-5-3-h6144-i2048 | 704.06 | 453.49 | 451.23 | 1.560 | -0.5% |
| mistral-small-4-h4096-i2048 | 261.79 | 195.63 | 185.90 | 1.408 | -5.0% |
| deepseek-v4-flash-h4096-i2048 | 525.15 | 327.93 | 316.95 | 1.657 | -3.3% |
| deepseek-v4-pro-h7168-i3072 | 1604.12 | 1080.73 | 1070.94 | 1.498 | -0.9% |
| minimax-m3-h6144-i3072 | 477.77 | 381.63 | 365.46 | 1.307 | -4.2% |
| gpt-oss-120b-h2880-i2880 | 572.05 | 267.00 | 261.82 | 2.185 | -1.9% |
| **geomean** | | | | **1.610** | |

**The functional best is r03-b04-a04 (1.6078).** It is 0.16% below the best by score, inside the noise. Its code is r03-b02-a02 plus two mechanisms: the two-block activation walk from b04-a03, which was neutral, and the early down-weight DRAM read from b04-a04, which gives a measured down-phase saving. It lacks only b02-a03's profiler-only counters. Against the best node it is faster on kimi-k3 (-1.7%), gpt-oss (-1.3%), dsv3 (-0.9%), dsv4-pro (-0.6%) and kimi-k2.7 (-0.3%). It is slower on minimax-m3 (+2.3%), dsv4-flash (+1.9%), glm (+1.1%) and mistral (+1.1%). All of those slower shapes are group-tail-lottery shapes, and its mean-core end moved down on each of them.

Per-shape bests from other nodes this round:

| shape | best µs | node |
|---|---|---|
| deepseek-v3 | 509.47 | b04-a04 |
| kimi-k2.7 | 727.12 | b04-a04 |
| kimi-k3 | 1342.05 | b04-a04 |
| glm | 450.72 | b04-a03 |
| deepseek-v4-flash | 313.98 | b01-a04 |
| deepseek-v4-pro | 1064.33 | b04-a04 |
| gpt-oss | 258.41 | b04-a04 |

The best node itself holds the round's per-shape bests for mistral (185.90) and minimax-m3 (365.46).

## Score grid (branch x attempt)

| branch | a01 | a02 | a03 | a04 | branch best |
|---|---|---|---|---|---|
| b01 | 0 (runtime_error) | 1.5882 | 1.5917 | 1.5952 | 1.5952 (a04) |
| b02 | **1.5887** | **1.5952** | **1.6103** | 1.5717 | 1.6103 (a03) |
| b03 | 1.5751 | 1.4983 | 0 (accuracy_fail) | 1.5554 | 1.5751 (a01) |
| b04 | 1.5813 | 1.5761 | 1.6010 | 1.6078 | 1.6078 (a04) |

Bold marks the round's running best when it was set. The running best after each step was:

| step | running best | set by | step-best Δ | outside noise? |
|---|---|---|---|---|
| s1 | 1.5887 | b02-a01 | +0.0190 | yes (+1.2%) |
| s2 | 1.5952 | b02-a02 | +0.0065 | no (+0.4%) |
| s3 | 1.6103 | b02-a03 | +0.0151 | no (+0.9%, code unchanged) |
| s4 | 1.6103 | no new best | 0 | |

Several nodes ran another node's code rather than their parent's, so the grid's parent links understate the cross-branch traffic:
- b03-a03 rebased onto b02-a02's code.
- b04-a03 rebased onto b02-a02's code.
- b03-a04 rebased onto b02-a03's code.
- b01-a03 ported b04-a01's planner trim.

## What worked

1. **Combining r02's orthogonal top nodes** (b02-a01, +1.2%; the round's only gain outside noise).
   - It combined r02-b02-a04's mixed phase-B planner with r02-b01-a04's PACK-thread pipeline and added a planner prologue trim.
   - 7 of 9 shapes got faster. The exception was glm (+1.9%): the last subblock's SiLU leaks into DN on glm's 4-row phase-B group, which was already critical.
   - b01-a01 built the same combination from r02-b03-a04's compute code. It overflowed the kernel config buffer, and b01-a02 repaired it.
2. **The planner prologue is solved** (b03-a01, b04-a01, b02-a01's trim, b01-a03).
   - RE-PLAN on NCRISC was measured directly for the first time:
     - root planner: about 40-49 µs on kimi-k3 (modelled/simulated);
     - trimmed at -O2: 17.7-20.4 µs;
     - trimmed and built in an -O3 region: 14.2 µs.
   - The other shapes now plan in 2.4-5.7 µs.
   - kimi-k3 gained 1.4-1.9% and dsv4-flash up to 2.0%. Everything else moved inside noise: the DRAM-bound shapes only start competing for the same bytes sooner, and the 4-expert shapes are lottery-bound.
   - b01-a03 calibrated an RV32 instruction simulator against two device measurements, to within 2%.
   - Reflections agree that what remains is worth less than 0.4% geomean.
3. **Plan once per core** (b01-a02).
   - The reader plans and publishes the plan through an L1 scratch CB.
   - It is latency-neutral by design. It fixed b01-a01's overflow and freed about 12-14 KB of kernel-config-buffer headroom on the h7168 shapes (TRISC0 24.3 → 13.4 KB).
4. **Early down-weight DRAM read** (b04-a04, +0.4% by score; mechanistically real).
   - The down sender issues block kb+1's read right after it multicasts kb, instead of after all receivers ack kb+1.
   - RE-DN fell on the many-expert shapes: dsv3 -8%, glm -7%, kimi-k3 -6% (-31 µs per core), kimi-k2.7 -4%.
   - The mean core end fell 7-18 µs on five shapes, and chip time fell 1.0-1.3% on dsv3, kimi-k2.7, kimi-k3 and dsv4-pro.
   - This contradicts r02's "mean core end = weight bytes / DRAM BW" reading, at least for the down phase.
   - The 4-expert shapes and dsv4-flash did not move. There is no counter yet showing whether the look-ahead fired on them.
5. **Measurement nodes and counters**, which set most of the round's conclusions:
   - **b02-a03 (MATH HW counters):**
     - The FW matmul throttle never engaged (about 500k samples).
     - The FPU runs at exactly 16.1 cycles per tile-mult whenever it is busy.
     - Compute-bound shapes cost 29-33 cycles per mult; DRAM-bound shapes 44-61.
   - **b03-a02 (per-site DM wait counters, `re_prof.hpp`):** receivers spend most of gate/up waiting for x and weights (dsv3: 202 + 100 of 348 µs), and the multicast is only 1.2-2.9 µs per block.
   - **b04-a02 (RE-GW / RE-XW zones):**
     - The x read is never late.
     - A sender's gate read is about 2 µs (p50) of a 9-16 µs block period.
     - This is the first direct gate/up cadence measurement.
   - **b04-a03:**
     - Startup fill: column gx0's first gate read starts at 15-24 µs, against 5.6 µs elsewhere, because of the x(0) owner's 96 one-KB stick reads.
     - The group tail is random per call and sits mostly in DN.
6. **Two-block activation walk** (b04-a03, neutral, +0.36% vs its code base).
   - RE-ACT fell 3-4.6 µs on the SiLU shapes, and gpt-oss GU fell 3 µs.
   - Elsewhere GU grew by about what DN and ACT shrank. GU+ACT+DN per core is conserved within 1-2%, the DRAM-share signature.

## What failed and why

| node | score | outcome | reason (from its reflection) |
|---|---|---|---|
| b01-a01 | 0 | runtime_error | The program was too large for the kernel config buffer on the three h7168 shapes: 71168 / 72752 / 71440 B against a 70656 B limit. The planner was compiled into three RISCs, and the PACK pipeline added a per-subblock K-loop to all TRISCs. The mechanism was never measured. b01-a02 repaired it. |
| b03-a02 | 1.4983 | -4.9% vs parent | The reader read both gate and up, so the writer only multicast. That put all gate/up weight read responses on NoC 0 and cut aggregate weight bandwidth 8-10% (dsv3 373 → 334 GB/s, +11.5%). This is the same signature as r02-b04-a02. The writer's read/multicast serialization it targeted is real but only 1.2-2.9 µs per block. |
| b03-a03 | 0 | accuracy_fail | Plain SiLU became a fused SiLU-GLU on PACK, which removed the FPU multiply pass. PCC improved by about 1.6e-4, but mean_abs rose 1.38-1.52% on all four SiLU shapes, over the +1% gate. The exact fp32 product removes the LoFi multiply's downward bias, which had been partly cancelling the bf4 error. The relative gate effectively pins the baseline's LoFi numerics. Even if valid, the gain was about +0.5%: on DRAM-bound SiLU shapes the saved time moves into DN's down-weight wait. |
| b02-a04 | 1.5717 | -2.4% vs parent | Column streaming (ct < rt): the weight tile is held in SrcA and x is streamed through SrcB. SrcA writes per mult fell about 5x, but cycles per mult did not fall, because the SrcB unpack serialises with the FPU just as the SrcA one did. The extra context rounds per column made minimax-m3 +7.5%, gpt-oss +5.0% and mistral +4.7%. |
| b03-a04 | 1.5554 | -3.4% vs its code base (b02-a03) | Reuse-B (rt = per_core_M, ct = 2) through one out-of-line helper. It halved total unpack work per mult (20.7 → 11 cycles), yet cycles per mult rose 1-3 and mistral got 9.8% slower. This refutes b02-a03's "FPU + serialized SrcA unpack" reading: the roughly 15 idle FPU cycles per mult are not unpack bandwidth. The out-of-line helper also costs about 1-2% on the pcM ≤ 2 path. |
| b01-a04 | 1.5952 | neutral | It streamed x through SrcB with ct = 1. Measured cost: 31.1 cycles per mult on mistral, the same as before, so per-call TRISC overhead is not the limiter either. |
| b02-a02 | 1.5952 | neutral | It moved the x owner off the weight-sender cores. The collision it targeted happens on every block, but the 2-slot rings hide it (GU within ±3%). The first run also overflowed the kernel config buffer and was repaired before timing (TRISC planner at -O2). |
| b04-a02 | 1.5761 | neutral | It gave x reads their own NoC transaction id, off the gate barrier. The x read was never late (RE-XW about 0 on 8 of 9 shapes). |
| b03-a01 | 1.5751 | neutral (+0.3%) | A planner trim, correct and exact. It cut the planner by about a third rather than the modelled half or more: divides were cheap, and the cost is plain loop work at about 1.9 cycles per -O0 line. kimi-k3 -1.4%. |
| b04-a01 | 1.5813 | neutral (+0.74%) | A planner trim that did shrink as modelled: kimi-k3's median core moved -22 µs, exactly as predicted. glm and kimi-k2.7 were flat because they are DRAM-bound. |

Newly established negatives (in addition to the r01/r02 lists):
- **Any operand-swap, reuse-mode or subblock-geometry change to the matmul** (b01-a04, b02-a04, b03-a04, plus r01-b02-a04). Whichever operand streams, a tile-mult costs about 31 cycles.
- **Putting a whole weight stream on one NoC.** This has now failed twice (b03-a02 for gate/up, r02-b04-a02 for down). Both NoCs' response paths are needed to reach about 440 GB/s.
- **x-owner placement and x-read scheduling** (b02-a02, b04-a02, plus r01-b03-a02 and r01-b01-a04).
- **Compute-side activation hiding on DRAM-bound shapes** (b04-a03, b03-a03). Saved compute reappears as DRAM wait in the next phase.
- **A more precise SiLU·up product.** The relative mean_abs gate rejects it.
- **Further planner work.** At most about 14 µs remains on kimi-k3 and 2-6 µs elsewhere.

**Disagreement between reflections on what limits the 4-expert shapes:**
- **Compute-limited readings:**
  - b02-a03 / b03-a04: about 31 cycles per tile-mult against the 16 LoFi ideal, so per-mult compute is about 2x ideal.
  - b04-a01: a fit of T = F + U·pcM puts U at the LoFi ideal and F at about 100 µs per expert. On that reading, per-expert fixed cost, not matmul rate, is the waste.
- **DRAM-limited reading:**
  - b04-a03: mistral's gate/up already runs at about 450 GB/s aggregate.
  - b03-a03, however, measured mistral at only 296 GB/s of whole-op weight bandwidth.
- No node has reconciled these.

Open levers named by the reflections:
- **The roughly 15 idle FPU cycles per tile-mult** (b03-a04, b02-a04, b01-a04). Candidates, cheapest probe first:
  - MATH-thread issue rate (`set_dst_write_addr` + MOP + throttle-word read per call);
  - Src-bank handshake latency (counter sels 27/28 not yet read);
  - unpacker L1 port contention with NoC writes;
  - an op-local unpack MOP that alternates config contexts per tile.

  Measure before building anything.
- **Group tail (max − mean):** 20 µs on mistral, 33 on gpt-oss, 85 on kimi-k3, 18-19 on dsv3/mm3/dsv4-flash. It is mostly in DN and is the largest remaining waste. Suggested steps:
  - measure the slow group's DN blocks against a fast group's in the same call;
  - decouple the activated-multicast ack chain the same way b04-a04 decoupled the down read;
  - ack x(kb+1) as soon as its slot frees (b04-a04 next #2).
- **Did b04-a04's look-ahead fire on the 4-expert shapes?** One profiler-only counter would tell.
- **Start-up fill** on the x(0) owner column, about 10-15 µs on the one-expert-per-group shapes. Options: fewer, larger x(0) reads, splitting the sticks across the row, or delaying block-0 weight reads slightly.
- **Sub-expert N-split of a last-wave expert.** This is carried over from r02 and is still untried. mistral uses 8 of 10 rows.
- **SiLU op-local body on PACK** (`re_act::silu_act`). Untried on this lineage, and expected inside noise.
- **NoC response-path efficiency for DRAM-bound shapes** (b03-a02 next #4): measure per-NoC link use first.

## Policy observations (input for dreaming)

- **No v1 again, so r02's issues recurred.**
  - **Hand recoveries:** every attempt that `v0-cand2` would tag `recover` was recovered by its worker by hand, matching cand2's revert/port targets:
    - b01-a01 (runtime_error) → b01-a02 repaired it;
    - b03-a02 (below root) → b03-a03 rebased onto the best node, b02-a02;
    - b03-a03 (accuracy_fail) → b03-a04 rebased onto the best node, b02-a03.

    b04-a02 (-0.3%, within noise) → b04-a03 also rebased onto b02-a02.
  - **cand1's collapse close (more than about 8% below branch best)** would not have fired in r03. The worst drop was b03-a02 at -4.9%.
- **Duplicated ideas. This round's waste was mostly duplication, not bad ideas:**
  - **Step 4:** three of four attempts (b01-a04, b02-a04, b03-a04) implemented the same hypothesis: stream x / activated through SrcB and hold the weight in SrcA. All three followed b02-a03's step-3 "next #1" through the shared history. They scored 1.5952 (neutral), 1.5717 and 1.5554. One attempt would have refuted it; the other two cost about $16 and two device slots. This is the most expensive duplication in the campaign so far.
  - **Step 1, combination:** b01-a01 and b02-a01 both built the r02-flagged combination (mixed phase-B planner + PACK pipeline). One overflowed and the other won.
  - **Step 1, planner trim:** b03-a01 and b04-a01 both trimmed the planner, and b02-a01 included a third trim. Step 3's b01-a03 then ported b04-a01's trim again. Four attempts touched the planner for about 2% on one shape (kimi-k3).
  - **Step 2:** b03-a02 and b04-a02 both instrumented the gate/up DM chain (DM wait counters vs RE-GW/RE-XW zones), each bundled with a speculative change. Both changes were negative or neutral.
- **Measurement nodes are now chosen by workers.**
  - b02-a03 was a pure measurement node and set the round's direction for step 4. Its interpretation was wrong, and three attempts were needed to show it.
  - It also became the round best by noise (code unchanged, +0.9%). With score-only best selection, the next round root would be a measurement node, while b04-a04 holds the round's only new mechanism with a measured DN saving.
  - Proposal input: do not credit profiler-only or within-noise nodes as a new best over their parent, or break near-ties toward nodes with a measured mechanism.
- **Low-yield attempts.**
  - 2 of 16 were invalid: b01-a01 and b03-a03.
  - 2 valid ones scored below the round root: b03-a02 and b03-a04.
  - 3 scored below their parent: b03-a02, b04-a02 and b02-a04. b03-a04's parent was invalid; it was 3.4% below its code base.
  - 3 set a new running best (b02-a01, b02-a02, b02-a03), but only b02-a01 did so beyond noise.
  - After step 1, nothing beat the running best by more than the noise band.
- **Plateau stop data point (for the rejected `v0-cand5`).**
  - In r03, steps 2 and 3 produced no new best beyond noise, so cand5 would have stopped before step 4.
  - That would have saved 4 attempts (about $25: b01-a04 $4.21, b02-a04 $8.84, b03-a04 $7.15, b04-a04 $5.10) and about 19 min.
  - It would have lost b04-a04, which is not the best by score but holds the only new real mechanism of step 4.
  - In r02 the same rule cost the round best. The evidence is now split one round each.
- **Stall close data point (for the rejected `v0-cand3`).** b02 had two results in a row within 1.1% of their parent (a02 +0.4%, a03 +0.9%). cand3 would have closed it before a04, which regressed, so it would have saved one attempt with no loss. b01 also reaches the condition, but only at its last attempt.
- **No premature closes.** `closed` stayed empty.
  - b03's own lineage ended at a02 (regression). Its a03 and a04 both rebased onto b02's code, so b03 functioned as a spare worker slot rather than a branch.
  - b04 did the same at a03.
  - By step 3, three of four branches ran b02-a02's code.
- **Kernel-config-buffer size is now a binding constraint.**
  - Two runs hit the 70656 B limit (b01-a01, b02-a02's first run). The repairs (plan-once, -O2 TRISC planner region, out-of-line helpers) cost attempts and some latency.
  - At least four workers built their own local size-check harness in `/tmp`: `/tmp/r03b02a03/sz`, `/tmp/w_r03b03a04/sz`, `/tmp/routed-r03b04a03`, `/tmp/r03b01a03/sim`.
  - A campaign-level size check before the device run would save the failed runs and the duplicated tooling.
- **Noise vs effect size.** Most per-attempt effects this round were 0.2-0.9%, inside the ±1% band.
  - Reflections increasingly judge by mean-core vs crit-core TRISC1 end (from MATH zones) rather than by chip time.
  - The group-tail lottery is ±5% on mistral, minimax-m3 and gpt-oss.
  - Repeated measurement, or a mean-core secondary metric, would separate real mechanisms (b04-a01, b04-a04) from noise draws (b02-a03).
- **Diminishing returns.**

| round | gain |
|---|---|
| r01 | +50% over the baseline |
| r02 | +4.6% over r01's best |
| r03 | +2.6% by score (+1.2% outside noise) |

  Eight of nine shapes are at or near their DRAM share. gpt-oss is the only clearly compute-bound shape. The remaining named wastes are the group tail, start-up fill and the unexplained idle FPU cycles per mult.

## Ops notes

- **Pre-round:**
  - Dreaming after r02 finished at 23:56:44 and kept v0. Its candidates were overwritten in place; the r01 set is in ledger commit `386239e5445`.
  - The root was built (`build_r03_root_1.log`), and the drift check passed (geomean -0.04%).
- **Step durations.** The barrier is synchronous, so the slowest worker sets each step's length. Times are commit times (+02:00).

| step | duration | slowest | fastest finished |
|---|---|---|---|
| 1 | 25.0 min | b04-a01 (00:22:33) | b01-a01 at 00:01:48, then about 21 min idle (b02-a01 at 00:02:41) |
| 2 | 29.0 min | b03-a02 (00:51:37) | b01-a02 at 00:29:45, then about 22 min idle |
| 3 | 34.9 min | b04-a03 (01:24:30) | b01-a03 at 01:00:38, then about 26 min idle |
| 4 | 19.3 min | b02-a04 (01:45:51) | b01-a04 at 01:37:11, then about 9 min idle |

  The two cheap step-1 combinations took 4-5 min each. Measurement-heavy or new-mechanism attempts took 20-40 min.
- **Worker exits:** 4 of 16 workers exited with rc=143: b01-a02, b01-a03, b01-a04 and b04-a03. Each had committed while its session was still running, and the driver stopped the leftover processes. Tags were present for all 16, and no attempts were lost.
- **verify.log:** 16/16 ok, no issues.
- **Device runs needing a repeat:** b02-a02's first run hit the kernel-config-buffer overflow on the three h7168 shapes and was repaired and re-run before timing. b01-a01's overflow was committed as its result. No hangs and no JIT errors this round.
- **Builds.**
  - Host builds ran for 8 of 16 attempts: b01-a01..a04, b02-a02, b03-a02, b03-a03 and b04-a04.
  - The real host changes were b01-a02 (new CB + named CT args) and b03-a02 (factory `up_mode`). b03-a03 reverted that factory change.
  - b04-a04 is kernel-only and rebuilt only because the shared eval checkout had changed (20 s).
  - The rest were kernel-only with no host build.
- **Eval tooling issues seen by workers:**
  - When some shapes fail, `eval/summary.md`'s µs columns are misaligned, because group-by-order assignment shifts (b01-a01).
  - Custom zones on TRISC_0 (UNPACK) do not record. Only NCRISC and TRISC_1 zones appear (b03-a01, b04-a01).
  - Another eval cleaned the per-run JIT cache before b01-a01 could capture reader/writer ELF sizes.
  - Worker analysis scripts live in `/tmp/...` paths that later nodes cite, but `/tmp` is not part of the node commit.
- **Reported worker cost** totals $118.48: b01 $12.63, b02 $34.67, b03 $34.19, b04 $36.99. That is up from $91.27 in r02. As before, the turns/minutes on the `[cost ...]` lines often do not match wall time. For example, b02-a02 reports 1 turn and 0 min for $10.16 against about 18 min of wall time, and b04-a03 reports 6 turns and 1 min against about 33 min. The figures likely capture only the final session segment.
- **Timestamps:** `decisions.jsonl` and `manifest.json` are in UTC (21:57:40Z). The driver and worker logs use local time (+02:00).
